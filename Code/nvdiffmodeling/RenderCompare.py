# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
import time
import argparse
import json
import csv

import numpy as np
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr

import src.renderutils as ru
from src import obj
from src import util
from src import mesh
from src import texture
from src import render
from src import regularizer
from src.mesh import Mesh

from PIL import Image
import math
import cv2
from skimage.metrics import structural_similarity

import OpenEXR

RADIUS = 3

def psnr(error_map):
    error_map = np.array(error_map)
    mse = np.mean( error_map ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# Enable to debug back-prop anomalies
# torch.autograd.set_detect_anomaly(True)

###############################################################################
# Utility mesh loader
###############################################################################

def load_mesh(filename, mtl_override=None):
    name, ext = os.path.splitext(filename)
    if ext == ".obj":
        return obj.load_obj(filename, clear_ks=True, mtl_override=mtl_override)
    assert False, "Invalid mesh file extension"


###############################################################################
# Loss setup
###############################################################################

def createLoss(FLAGS):
    if FLAGS.loss == "smape":
        return lambda img, ref: ru.image_loss(img, ref, loss='smape', tonemapper='none')
    elif FLAGS.loss == "mse":
        return lambda img, ref: ru.image_loss(img, ref, loss='mse', tonemapper='none')
    elif FLAGS.loss == "logl1":
        return lambda img, ref: ru.image_loss(img, ref, loss='l1', tonemapper='log_srgb')
    elif FLAGS.loss == "logl2":
        return lambda img, ref: ru.image_loss(img, ref, loss='mse', tonemapper='log_srgb')
    elif FLAGS.loss == "relativel2":
        return lambda img, ref: ru.image_loss(img, ref, loss='relmse', tonemapper='none')
    else:
        assert False


###############################################################################
# Main shape fitter function / optimization loop
###############################################################################

def render_compare(
        FLAGS,
        out_dir,
        log_interval=10,
        mesh_scale=2.0
):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "mesh"), exist_ok=True)

    # Projection matrix
    proj_mtx = util.projection(x=0.4, f=1000.0)

    # Reference mesh
    ref_mesh = load_mesh(FLAGS.ref_mesh, FLAGS.mtl_override)
    print("Ref mesh has %d triangles and %d vertices." % (ref_mesh.t_pos_idx.shape[0], ref_mesh.v_pos.shape[0]))

    # Check if the training texture resolution is acceptable
    ref_texture_res = np.maximum(ref_mesh.material['kd'].getRes(), ref_mesh.material['ks'].getRes())
    if 'normal' in ref_mesh.material:
        ref_texture_res = np.maximum(ref_texture_res, ref_mesh.material['normal'].getRes())
    if FLAGS.texture_res[0] < ref_texture_res[0] or FLAGS.texture_res[1] < ref_texture_res[1]:
        print("---> WARNING: Picked a texture resolution lower than the reference mesh [%d, %d] < [%d, %d]" % (
        FLAGS.texture_res[0], FLAGS.texture_res[1], ref_texture_res[0], ref_texture_res[1]))

    # Base mesh
    base_mesh = load_mesh(FLAGS.base_mesh)
    print("Base mesh has %d triangles and %d vertices." % (base_mesh.t_pos_idx.shape[0], base_mesh.v_pos.shape[0]))

    # Create normalized size versions of the base and reference meshes. Normalized base_mesh is important as it makes it easier to configure learning rate.
    normalized_base_mesh = mesh.unit_size(base_mesh)

    assert not FLAGS.random_train_res or FLAGS.custom_mip, "Random training resolution requires custom mip."

    # ==============================================================================================
    #  Initialize weights / variables for trainable mesh
    # ==============================================================================================
    trainable_list = []

    v_pos_opt = normalized_base_mesh.v_pos.clone().detach().requires_grad_(True)

    # Trainable normal map, initialize to (0,0,1) & make sure normals are always in positive hemisphere
    if FLAGS.random_textures:
        normal_map_opt = texture.create_trainable(np.array([0, 0, 1]), FLAGS.texture_res, not FLAGS.custom_mip)
    else:
        if 'normal' not in ref_mesh.material:
            normal_map_opt = texture.create_trainable(np.array([0, 0, 1]), FLAGS.texture_res, not FLAGS.custom_mip)
        else:
            normal_map_opt = texture.create_trainable(ref_mesh.material['normal'], FLAGS.texture_res,
                                                      not FLAGS.custom_mip)

    # Setup Kd, Ks albedo and specular textures
    if FLAGS.random_textures:
        if FLAGS.layers > 1:
            kd_map_opt = texture.create_trainable(np.random.uniform(size=FLAGS.texture_res + [4], low=0.0, high=1.0),
                                                  FLAGS.texture_res, not FLAGS.custom_mip)
        else:
            kd_map_opt = texture.create_trainable(np.random.uniform(size=FLAGS.texture_res + [3], low=0.0, high=1.0),
                                                  FLAGS.texture_res, not FLAGS.custom_mip)

        ksR = np.random.uniform(size=FLAGS.texture_res + [1], low=0.0, high=0.01)
        ksG = np.random.uniform(size=FLAGS.texture_res + [1], low=FLAGS.min_roughness, high=1.0)
        ksB = np.random.uniform(size=FLAGS.texture_res + [1], low=0.0, high=1.0)
        ks_map_opt = texture.create_trainable(np.concatenate((ksR, ksG, ksB), axis=2), FLAGS.texture_res,
                                              not FLAGS.custom_mip)
    else:
        kd_map_opt = texture.create_trainable(base_mesh.material['kd'], FLAGS.texture_res, not FLAGS.custom_mip)
        ks_map_opt = texture.create_trainable(ref_mesh.material['ks'], FLAGS.texture_res, not FLAGS.custom_mip)

    # Trainable displacement map
    displacement_map_var = None
    if FLAGS.subdivision > 0:
        displacement_map_var = torch.tensor(np.zeros(FLAGS.texture_res + [1], dtype=np.float32), dtype=torch.float32,
                                            device='cuda', requires_grad=True)

    # Add trainable arguments according to config
    if not 'position' in FLAGS.skip_train:
        trainable_list += [v_pos_opt]
    if not 'normal' in FLAGS.skip_train:
        trainable_list += normal_map_opt.getMips()
    if not 'kd' in FLAGS.skip_train:
        trainable_list += kd_map_opt.getMips()
    if not 'ks' in FLAGS.skip_train:
        trainable_list += ks_map_opt.getMips()
    if not 'displacement' in FLAGS.skip_train and displacement_map_var is not None:
        trainable_list += [displacement_map_var]

    # ==============================================================================================
    #  Setup material for optimized mesh
    # ==============================================================================================

    opt_material = {
        'bsdf': ref_mesh.material['bsdf'],
        'kd': kd_map_opt,
        'ks': ks_map_opt,
        'normal': normal_map_opt
    }

    # ==============================================================================================
    #  Setup reference mesh. Compute tangentspace and animate with skinning
    # ==============================================================================================

    render_ref_mesh = mesh.compute_tangents(ref_mesh)

    # Compute AABB of reference mesh. Used for centering during rendering TODO: Use pre frame AABB?
    ref_mesh_aabb = mesh.aabb(render_ref_mesh.eval())

    # ==============================================================================================
    #  Setup base mesh operation graph, precomputes topology etc.
    # ==============================================================================================

    # Create optimized mesh with trainable positions 
    opt_base_mesh = Mesh(v_pos_opt, normalized_base_mesh.t_pos_idx, material=opt_material, base=normalized_base_mesh)

    # Scale from [-1, 1] local coordinate space to match extents of the reference mesh
    opt_base_mesh = mesh.align_with_reference(opt_base_mesh, ref_mesh)

    # Compute smooth vertex normals
    opt_base_mesh = mesh.auto_normals(opt_base_mesh)

    # Set up tangent space
    opt_base_mesh = mesh.compute_tangents(opt_base_mesh)

    # Subdivide if we're doing displacement mapping
    if FLAGS.subdivision > 0:
        # Subdivide & displace optimized mesh
        subdiv_opt_mesh = mesh.subdivide(opt_base_mesh, steps=FLAGS.subdivision)
        opt_detail_mesh = mesh.displace(subdiv_opt_mesh, displacement_map_var, FLAGS.displacement,
                                        keep_connectivity=True)
    else:
        opt_detail_mesh = opt_base_mesh

    # ==============================================================================================
    #  Setup torch optimizer
    # ==============================================================================================

    optimizer = torch.optim.Adam(trainable_list, lr=FLAGS.learning_rate)

    # Background color
    if FLAGS.background == 'checker':
        background = torch.tensor(util.checkerboard(FLAGS.display_res, 8), dtype=torch.float32, device='cuda')
    elif FLAGS.background == 'white':
        background = torch.ones((1, FLAGS.display_res, FLAGS.display_res, 3), dtype=torch.float32, device='cuda')
    else:
        background = None

    # ==============================================================================================
    #  Training loop
    # ==============================================================================================
    img_cnt = 0
    ang = 0.0
    img_loss_vec = []
    iter_dur_vec = []
    glctx = dr.RasterizeGLContext()
    avg_psnr_loss = 0.0
    avg_ssim_loss = 0.0

    tri_list = []

    tmpRes = open("./out/compare/tmpRes.txt", "w")
    res = open("./out/res.txt", "a")

    for it in range(FLAGS.iter):
        # ==============================================================================================
        #  Display / save outputs. Do it before training so we get initial meshes
        # ==============================================================================================

        # Show/save image before training step (want to get correct rendering of input)
        if 1:

            mvp = np.zeros((FLAGS.batch, 4, 4), dtype=np.float32)
            campos = np.zeros((FLAGS.batch, 3), dtype=np.float32)
            lightpos = np.zeros((FLAGS.batch, 3), dtype=np.float32)

            for b in range(FLAGS.batch):

                # sample camera pos
                with open('E:/Projects/VS/ModelCompress/sample.csv', 'r') as file:
                    reader = csv.reader(file)
                    rows = list(reader)

                PI = 3.14159265358979323846
                theta = 2 * PI * float(rows[it*2][0])
                phi = math.acos(1 - 2 * float(rows[it*2 + 1][0]))
                x = math.sin(phi) * math.cos(theta) * RADIUS
                y = math.sin(phi) * math.sin(theta) * RADIUS
                z = math.cos(phi) * RADIUS
                eye = np.array([x, y, z])
                normal = eye / RADIUS
                tangent = np.array([eye[1], -eye[0], 0])
                direction = np.cross(eye, tangent)
                tangent = direction / np.linalg.norm(direction)
                up = tangent
                at = np.array([0, 0, 0])
                r_mv = util.lookAt(eye, at, up)

                # Random rotation/translation matrix for optimization.
                mvp[b] = np.matmul(proj_mtx, r_mv).astype(np.float32)
                campos[b] = np.linalg.inv(r_mv)[:3, 3]
                lightpos[b] = campos[b]

            params = {'mvp': mvp, 'lightpos': lightpos, 'campos': campos,
                      'resolution': [FLAGS.display_res, FLAGS.display_res],
                      'time': 0}

            # Render images, don't need to track any gradients
            with torch.no_grad():
                # Center meshes
                _opt_detail = mesh.center_by_reference(opt_detail_mesh.eval(params), ref_mesh_aabb, mesh_scale)
                _opt_ref = mesh.center_by_reference(render_ref_mesh.eval(params), ref_mesh_aabb, mesh_scale)

                # Render
                if FLAGS.subdivision > 0:
                    _opt_base = mesh.center_by_reference(opt_base_mesh.eval(params), ref_mesh_aabb, mesh_scale)
                    img_base = render.render_mesh(glctx, _opt_base, mvp, campos, lightpos, FLAGS.light_power,
                                                  FLAGS.display_res,
                                                  num_layers=FLAGS.layers, background=background,
                                                  min_roughness=FLAGS.min_roughness)
                    img_base = util.scale_img_nhwc(img_base, [FLAGS.display_res, FLAGS.display_res])

                img_opt = render.render_mesh(glctx, _opt_detail, mvp, campos, lightpos, FLAGS.light_power,
                                             FLAGS.display_res,
                                             num_layers=FLAGS.layers, background=background,
                                             min_roughness=FLAGS.min_roughness)
                img_ref = render.render_mesh(glctx, _opt_ref, mvp, campos, lightpos, FLAGS.light_power,
                                             FLAGS.display_res,
                                             num_layers=1, spp=FLAGS.spp, background=background,
                                             min_roughness=FLAGS.min_roughness)

                # Rescale
                img_opt = util.scale_img_nhwc(img_opt, [FLAGS.display_res, FLAGS.display_res])
                img_ref = util.scale_img_nhwc(img_ref, [FLAGS.display_res, FLAGS.display_res])

                if FLAGS.subdivision > 0:
                    img_disp = torch.clamp(torch.abs(displacement_map_var[None, ...]), min=0.0, max=1.0).repeat(1, 1, 1,
                                                                                                                3)
                    img_disp = util.scale_img_nhwc(img_disp, [FLAGS.display_res, FLAGS.display_res])
                    result_image = torch.cat([img_base, img_opt, img_ref], axis=2)
                else:
                    result_image = torch.cat([img_opt, img_ref], axis=2)

            result_image[0] = util.tonemap_srgb(result_image[0])
            np_result_image = result_image[0].detach().cpu().numpy()

            img_opt[0] = util.tonemap_srgb(img_opt[0])
            np_img_opt = img_opt[0].detach().cpu().numpy()
            util.save_image(out_dir + '/' + ('img_opt_%06d.png' % img_cnt), np_img_opt)

            img_ref[0] = util.tonemap_srgb(img_ref[0])
            np_img_ref = img_ref[0].detach().cpu().numpy()
            util.save_image(out_dir + '/' + ('img_ref_%06d.png' % img_cnt), np_img_ref)

            rast_res = render.render_mesh_rast(glctx, _opt_detail, mvp, campos, lightpos, FLAGS.light_power,
                                            FLAGS.display_res,
                                            num_layers=FLAGS.layers, background=background,
                                            min_roughness=FLAGS.min_roughness)
            img = rast_res.cpu().numpy()[0, :, :, :] 
            
            img = img.astype(np.float32)
            r_channel = img[:, :, 3].ravel()
            g_channel = img[:, :, 3].ravel()
            b_channel = img[:, :, 3].ravel()
            a_channel = img[:, :, 3].ravel()

            exr = OpenEXR.OutputFile(out_dir + '/' + ('img_rast_%06d.exr' % img_cnt), OpenEXR.Header(512, 512))
            exr.writePixels({'R': r_channel.tobytes(), 'G': g_channel.tobytes(), 'B': b_channel.tobytes(), 'A': a_channel.tobytes()})
            exr.close()

            img = img.astype(np.int64)

            img1 = cv2.imread(out_dir + '/' + ('img_opt_%06d.png' % img_cnt))
            img2 = cv2.imread(out_dir + '/' + ('img_ref_%06d.png' % img_cnt))

            error_map = np.array(img1) - np.array(img2)
            error_map_length = np.linalg.norm(img1.astype(np.float32) - img2.astype(np.float32), axis=2)

            for index in tri_list:
                triangle_pixels = np.where(img[:, :, 3] == index)
                for i, j in zip(triangle_pixels[0], triangle_pixels[1]):
                    error_map[i, j] = 0
                    error_map_length[i, j] = 0

            psnr_loss = psnr(error_map)

            if FLAGS.PSNR_thred != None:
                iter = 0
                print('iter: %d, psnr_loss: %f' % (iter, psnr_loss))
                while psnr_loss < FLAGS.PSNR_thred:
                    max_error_index = np.unravel_index(np.argsort(error_map_length.ravel())[-1:], error_map_length.shape)
                    max_error_index = img[max_error_index[0], max_error_index[1], 3]
                    # print('max_error_tri_index: %d' % max_error_index)
                    triangle_pixels = np.where(img[:, :, 3] == max_error_index)
                    tri_list = np.unique(np.concatenate((tri_list, max_error_index)))
                    for i, j in zip(triangle_pixels[0], triangle_pixels[1]):
                        error_map[i, j] = [0, 0, 0]
                        error_map_length[i, j] = 0

                    psnr_loss = psnr(error_map)
                    iter += 1
                    # print('iter: %d, psnr_loss: %f' % (iter, psnr_loss))

            diff_arr = np.linalg.norm(img1.astype(np.float32) - img2.astype(np.float32), axis=2)
            diff_arr = cv2.normalize(diff_arr, None, 0, 255, cv2.NORM_MINMAX)
            diff_img = cv2.applyColorMap(diff_arr.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
            cv2.imwrite(out_dir + '/' + ('img_diff_%06d.png' % img_cnt), diff_img)

            ssim_loss = structural_similarity(img1, img2, channel_axis = -1)

            avg_psnr_loss += psnr_loss
            avg_ssim_loss += ssim_loss

            print("iter=%5d, psnr_loss=%.6f, ssim_loss=%.6f" %(it, psnr_loss, ssim_loss))
            tmpRes.write("iter=%5d, psnr_loss=%.6f, ssim_loss=%.6f\n" %(it, psnr_loss, ssim_loss))

            util.save_image(out_dir + '/' + ('img_%06d.png' % img_cnt), np_result_image)
            img_cnt = img_cnt + 1

    np.savetxt(out_dir + '/' + 'tri_list.txt', tri_list, fmt='%d')

    avg_psnr_loss = avg_psnr_loss/FLAGS.iter
    print("avg_psnr_loss=%.6f" % (avg_psnr_loss))
    res.write("avg_psnr_loss=%.6f\n" % (avg_psnr_loss))
    avg_ssim_loss = avg_ssim_loss/FLAGS.iter

    with open("E:/Projects/VS/ModelCompress/data.csv", "a") as csvfile:
        csvfile.write(str(avg_psnr_loss) +"," + str(avg_ssim_loss) + "\n")
        csvfile.close()

    print("avg_ssim_loss=%.6f" % (avg_ssim_loss))
    res.write("avg_ssim_loss=%.6f\n" % (avg_ssim_loss))
    tmpRes.close()
    res.close()


# ----------------------------------------------------------------------------
# Main function.
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='diffmodeling')
    parser.add_argument('-i', '--iter', type=int, default=5000)
    parser.add_argument('-b', '--batch', type=int, default=1)
    parser.add_argument('-s', '--spp', type=int, default=1)
    parser.add_argument('-l', '--layers', type=int, default=1)
    parser.add_argument('-r', '--train-res', type=int, default=512)
    parser.add_argument('-rtr', '--random-train-res', action='store_true', default=False)
    parser.add_argument('-dr', '--display-res', type=int, default=None)
    parser.add_argument('-tr', '--texture-res', nargs=2, type=int, default=[1024, 1024])
    parser.add_argument('-di', '--display-interval', type=int, default=0)
    parser.add_argument('-si', '--save-interval', type=int, default=1000)
    parser.add_argument('-lr', '--learning-rate', type=float, default=None)
    parser.add_argument('-lp', '--light-power', type=float, default=5.0)
    parser.add_argument('-mr', '--min-roughness', type=float, default=0.08)
    parser.add_argument('-sd', '--subdivision', type=int, default=0)
    parser.add_argument('-mip', '--custom-mip', action='store_true', default=False)
    parser.add_argument('-rt', '--random-textures', action='store_true', default=False)
    parser.add_argument('-lf', '--laplacian-factor', type=float, default=None)
    parser.add_argument('-rl', '--relative-laplacian', type=bool, default=False)
    parser.add_argument('-bg', '--background', default='white', choices=['black', 'white', 'checker'])
    parser.add_argument('--loss', default='mse', choices=['logl1', 'logl2', 'mse', 'smape', 'relativel2'])
    parser.add_argument('-o', '--out-dir', type=str, default=None)
    parser.add_argument('--config', type=str, default=None, help='Config file')
    parser.add_argument('-rm', '--ref_mesh', type=str)
    parser.add_argument('-bm', '--base-mesh', type=str)
    parser.add_argument('--PSNR_thred',type = float, default = None)


    FLAGS = parser.parse_args()

    print('FLAGS:', FLAGS)

    FLAGS.camera_eye = [0.0, 0.0, RADIUS]
    FLAGS.camera_up = [0.0, 1.0, 0.0]
    FLAGS.skip_train = ['position', 'ks', 'normal']
    FLAGS.displacement = 0.15
    FLAGS.mtl_override = None

    if FLAGS.config is not None:
        with open(FLAGS.config) as f:
            data = json.load(f)
            for key in data:
                print(key, data[key])
                FLAGS.__dict__[key] = data[key]

    if FLAGS.display_res is None:
        FLAGS.display_res = FLAGS.train_res
    if FLAGS.out_dir is None:
        out_dir = 'out/cube_%d' % (FLAGS.train_res)
    else:
        out_dir = 'out/' + FLAGS.out_dir

    render_compare(FLAGS, out_dir)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------

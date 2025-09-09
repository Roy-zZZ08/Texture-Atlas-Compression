import os
import shutil
import cv2
import time
import argparse

# Data Preparation
def clean_mesh(obj_path, output_path):
    vertices = []
    uvs = []
    faces = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertices.append(list(map(float, line.strip().split()[1:])))
            elif line.startswith('vt '):
                uvs.append(list(map(float, line.strip().split()[1:])))
            elif line.startswith('f '):
                faces.append(line.strip().split()[1:])

    epsilon = 1e-6
    vertex_map = {}
    new_vertices = []
    new_uvs = []
    for i, v in enumerate(vertices):
        duplicate = False
        for j, nv in enumerate(new_vertices):
            if all(abs(v[k] - nv[k]) < epsilon for k in range(3)):
                vertex_map[i] = j
                duplicate = True
                break
        if not duplicate:
            vertex_map[i] = len(new_vertices)
            new_vertices.append(v)

    uv_map = {}
    for i, uv in enumerate(uvs):
        duplicate = False
        for j, nuv in enumerate(new_uvs):
            if all(abs(uv[k] - nuv[k]) < epsilon for k in range(2)):
                uv_map[i] = j
                duplicate = True
                break
        if not duplicate:
            uv_map[i] = len(new_uvs)
            new_uvs.append(uv)

    valid_faces = []
    for face in faces:
        if len(face) != 3:
            continue  
        vertex_indices = []
        for part in face:
            components = part.split('/')
            v = components[0]
            vertex_indices.append(int(v))
        if len(set(vertex_indices)) < 3:
            continue 
        valid_faces.append(face)

    with open(output_path, 'w') as f:
        for v in new_vertices:
            f.write(f"v {' '.join(map(str, v))}\n")
        for uv in new_uvs:
            f.write(f"vt {' '.join(map(str, uv))}\n")
        for face in valid_faces:
            vertex_indices = []
            for part in face:
                components = part.split('/')
                v = components[0]
                vertex_indices.append(vertex_map[int(v)-1]+1)
            if len(set(vertex_indices)) < 3:
                continue  
            f.write("f ")
            for part in face:
                components = part.split('/')
                if len(components) == 2:
                    v, vt = components
                    f.write(f"{vertex_map[int(v)-1]+1}/{uv_map[int(vt)-1]+1 if vt else ''} ")
                elif len(components) == 3:
                    v, vt, vn = components
                    f.write(f"{vertex_map[int(v)-1]+1}/{uv_map[int(vt)-1]+1 if vt else ''}/{vn if vn else ''} ")
            f.write("\n")



def data_preparation(data_Path):
    if os.path.exists('./tmp') == True:
        shutil.rmtree('./tmp')   
    else:
        os.mkdir('./tmp') 
    if os.path.exists('./out') == True:
        shutil.rmtree('./out')   
    else:
        os.mkdir('./out') 

    # read model and texture
    image = cv2.imread(data_Path + '/texture.png')
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    cv2.imwrite('./tmp/texture.jpg', image)
    clean_mesh(data_Path + '/model.obj', './tmp/model.obj')

# Feature Extraction
def feature_extraction():
    start_time = time.time()    
    cmd = 'python MaskGenerator.py --checkpoint sam_vit_l_0b3195.pth --input tmp/texture.jpg  --output tmp/SAMout'
    os.system(cmd)
    end_time = time.time()
    duration = end_time - start_time
    with open("./tmp/Time.csv", "a") as f:
        f.write("Feature Extraction, {:.3f} \n".format(duration))
        f.close()

# Nvdiffmodeling Optimize
def optimize(iter):
    # data preparation
    with open('./tmp/model_nomat.obj') as tmp_model1:
        contents = tmp_model1.read() 
    with open('./nvdiffmodeling/data/optimize/model.obj','w') as tmp_model2: 
        tmp_model2.write("mtllib model.mtl\nusemtl blinn1SG\n")
        tmp_model2.write(contents)

    with open('./tmp/repack.obj') as tmp_model1: 
        contents = tmp_model1.read() 
    with open('./nvdiffmodeling/data/optimize/repack.obj','w') as tmp_model2: 
        tmp_model2.write("mtllib repack.mtl\nusemtl blinn1SG\n")
        tmp_model2.write(contents)

    shutil.copy('./tmp/texture.jpg', './nvdiffmodeling/data/optimize/texture.jpg')
    shutil.copy('./tmp/repackTex.png', './nvdiffmodeling/data/optimize/repackTex.png')

    print("start optimize")
    start_time = time.time()
    cmd = 'python ./nvdiffmodeling/train.py --config ./nvdiffmodeling/configs/optimize_iter{}.json'.format(iter)
    os.system(cmd)
    end_time = time.time()
    duration = end_time - start_time

    if(iter == 0):
        with open("./tmp/Time.csv", "a") as f:
            f.write("Texture Baking_iter0, {:.3f}\n".format(duration))
        f.close()

    src = cv2.imread('./out/optimize/mesh/texture_kd.png')
    ref = cv2.imread('./tmp/imgSize.png')
    dst = cv2.resize(src, (ref.shape[1], ref.shape[0]))
    cv2.imwrite('./out/texture_compressed.png', dst)

# Render Compare
def compare(iter, PSNR_thred):
    print("start render compare")
    shutil.copy('./out/texture_compressed.png', './nvdiffmodeling/data/compare/repackTex.png')
    shutil.copy('./tmp/texture.jpg', './nvdiffmodeling/data/compare/texture.jpg')

    with open('./tmp/model_nomat.obj') as tmp_model1: 
        contents = tmp_model1.read() 
    with open('./nvdiffmodeling/data/compare/model.obj','w') as tmp_model2:
        tmp_model2.write("mtllib model.mtl\nusemtl blinn1SG\n")
        tmp_model2.write(contents)
    with open('./tmp/repack.obj') as tmp_model1: 
        contents = tmp_model1.read() 
    with open('./nvdiffmodeling/data/compare/repack.obj','w') as tmp_model2: 
        tmp_model2.write("mtllib repack.mtl\nusemtl blinn1SG\n")
        tmp_model2.write(contents)

    if(iter == 0):
        start_time = time.time()
        cmd = 'python ./nvdiffmodeling/RenderCompare.py --config ./nvdiffmodeling/configs/compare.json --PSNR_thred {}'.format(PSNR_thred)
        os.system(cmd)
        end_time = time.time()
        duration = end_time - start_time
        with open("./tmp/Time.csv", "a") as f:
            f.write("Render Compare, {:.3f}\n".format(duration))
        f.close()
    else:
        cmd = 'python ./nvdiffmodeling/RenderCompare.py --config ./nvdiffmodeling/configs/compare.json'
        os.system(cmd)

# Pack Result
def pack_result(iter):
    with open('./tmp/model_nomat.obj') as tmp_model1: 
        contents = tmp_model1.read() 
    with open('./out/model.obj','w') as tmp_model2:
        tmp_model2.write("mtllib model.mtl\nusemtl blinn1SG\n")
        tmp_model2.write(contents)
    with open('./tmp/repack.obj') as tmp_model1: 
        contents = tmp_model1.read() 
    with open('./out/model_compressed.obj','w') as tmp_model2: 
        tmp_model2.write("mtllib model_compressed.mtl\nusemtl blinn1SG\n")
        tmp_model2.write(contents)

    img = cv2.imread('./tmp/texture.jpg')
    cv2.imwrite('./out/texture.png', img)

    if(iter == 0):
        shutil.move('./tmp/Time.csv', './out/Time.csv')

    shutil.copy('./model.mtl', './out/model.mtl')
    shutil.copy('./model_compressed.mtl', './out/model_compressed.mtl')

    shutil.move('./avgHSV.csv', './tmp/avgHSV.csv')
    shutil.move('./out.csv', './tmp/out.csv')
    shutil.move('./tmp', './out')
    if os.path.exists('./out_{}'.format(iter)) == True:
        shutil.rmtree('./out_{}'.format(iter))   
    shutil.move('./out', './out_{}'.format(iter))

def main(args):
    PSNR_thred = args.PSNR_thred
    error_theta = args.error_theta
    data_Path = args.data_Path
    output_Path = args.output_Path

    # ================== iter 0 ==============================
    # construct folder
    if os.path.exists(output_Path):
        shutil.rmtree(output_Path)

    data_preparation(data_Path)

    # record time
    with open("./tmp/Time.csv", "w") as f:
            f.write("Step,Time(s)\n")
            f.close()

    feature_extraction()

    # texture compression
    print("start compress")
    log = open("./tmp/tri_list.txt", "w")
    log.close()
    main = 'UVcompress.exe {}'.format(error_theta)
    r_v = os.system(main) 
    print (r_v)

    optimize(0)
    compare(0, PSNR_thred)
    pack_result(0)

    # ================== iter 1 ==============================

    data_preparation(data_Path)
    shutil.move('./out_0/tmp/SAMout', './tmp/SAMout')

    print("start compress")
    shutil.copy('./out_0/compare/tri_list.txt', './tmp/tri_list.txt')
    main = 'UVcompress.exe {}'.format(error_theta)
    r_v = os.system(main) 
    print (r_v)

    optimize(1)
    compare(1, PSNR_thred)
    pack_result(1)
    shutil.copy('./out_0/Time.csv', './out_1/Time.csv')
    shutil.move('./out_1', output_Path)
    shutil.move('./out_0', output_Path + '/Out_iter0')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AtlasCompress")
    parser.add_argument("--PSNR_thred", type=float, default=38.0, help="PSNR threshold")
    parser.add_argument("--error_theta", type=float, default=10.0, help="Error theta")
    parser.add_argument("--data_Path", type=str, help="Data path")
    parser.add_argument("--output_Path", type=str, help="Output path")

    args = parser.parse_args()
    main(args)
# AtlasCompression

Texture Atlas Compression Based on Repeated Content Removal

# Installation

Requirements:
 - [Microsoft Visual Studio](https://visualstudio.microsoft.com/) 2019+ with Microsoft Visual C++ installed
 - [Cuda](https://developer.nvidia.com/cuda-toolkit) 10.2+
 - [Pytorch](https://pytorch.org/) 1.6+
 - [OpenCV](https://opencv.org/)

Tested in Anaconda3 with Python 3.8 and PyTorch 1.12 + Cuda 11.3

## Setup (Windows)

1. Install [Microsoft Visual Studio](https://visualstudio.microsoft.com/) 2019+ with Microsoft Visual C++. 
2. Place the [opencv_world455.dll](https://drive.google.com/file/d/1HR7hME28Qw3O54GeV-CXuHKTqBOsQSXZ/view?usp=drive_link) file in the `Code` folder directory.
3. Install [Cuda](https://developer.nvidia.com/cuda-toolkit) 10.2 or above. **Note:** Install CUDA toolkit from https://developer.nvidia.com/cuda-toolkit (not through anaconda)
4. Install the appropriate version of PyTorch compatible with the installed Cuda toolkit.
5. Install [nvdiffrast](https://github.com/NVlabs/nvdiffrast) in conda env. Follow the [installation instructions](https://nvlabs.github.io/nvdiffrast/#windows).
6. download the [ViT-L SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth), put it in the `Code` folder like `Code\sam_vit_l_0b3195.pth`

Examples
========

[Additional test models can be downloaded here](https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research)
Unzip and place the model and texture file in the project `input_Path` folder:

e.g., `input_Path\model.obj`，`input_Path\texture.png`
(Texture size is currently supported only for 1024x1024 dimensions.)

Change into the `AtlasCompress\Code` directory and run:

```python
python AtlasCompress.py --PSNR_thred 38.0 --error_theta 10.0 --data_Path "input_path" --output_Path "output_path"
```
The results will be stored in the `output_Path` folder.


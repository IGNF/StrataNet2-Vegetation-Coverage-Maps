# PointNet-based model for the prediction of vegetation coverage using 3D LiDAR point clouds

PyTorch implementation of a weakly supervised algorithm for the prediction of vegetation coverage of different stratum. The algorithm is based on (PointNet++)[https://arxiv.org/abs/1706.02413] for 3D data classification and segmentation.

The model takes raw, unordered set of LiDAR points and computes for each point the pointwise probability of membership to one of four following class:
- bare soil
- low vegetation
- medium vegetation
- high vegetation

Alongside, the model computes a fifth value in the 0-1 range, which is interpreted as a density. This density is then multiplied with membership probabilities to yield pointwise coverage predictions for all three vegetation strata. Note: the bare soil probability is ignored, but is important to have a proper definition of membership probabilities. 

Finally, pointwise coverages values are max-projected on each stratum, which yields four 2D rasters of vegetation coverage values. The average value of each raster then gives the coverage value for the area of interest.

The model is applied to circular, 10m radius plots. With pointwise classification and coverage map generation, one can expain predictions.  

![](exemples_images/3_stratum.png)

### Example usage

## Installation

### Requirements
The project requires an environment with PyTorch installed (only tested with version 1.7.0).
Module [torch_scatter](https://github.com/rusty1s/pytorch_scatter) is also required.
The installation of torch_scatter can be challenging, please, check your CUDA and PyTorch version and carefully follow the instructions indicated below.

The project requires GDAL library to save the results to a GeoTIFF file. If you have difficulties installing GDAL, just delete `create_tiff()` function from `utils/create_final_images.py`. The result is equally saved to a .png image.



### Install 
We suppose that you already have pytorch installed. Please, use requirements.txt to install other packages by launching `python -m pip install -r requirements.txt`.

# Install torch_scatter
Launch this code to check your TORCH and CUDA versions if you don't know them.
```python
import torch

def format_pytorch_version(version):
  return version.split('+')[0]

TORCH_version = torch.__version__
TORCH = format_pytorch_version(TORCH_version)

def format_cuda_version(version):
  return 'cu' + version.replace('.', '')

CUDA_version = torch.version.cuda
CUDA = format_cuda_version(CUDA_version)

print("TORCH")
print(TORCH)
print("CUDA")
print(CUDA)
```

Then replace {TORCH} and {CUDA} by the obtained values to install the corresponding packages:


`pip install torch-scatter     -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html`

`pip install torch-sparse      -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html`

`pip install torch-cluster     -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html`

`pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html`

`pip install torch-geometric`



## Code 
The code is launched from `main.py`.
It can be launched from IDE, as well as from console with precising the arguments. Please, check the code for different arguments or do `python main.py --help`.

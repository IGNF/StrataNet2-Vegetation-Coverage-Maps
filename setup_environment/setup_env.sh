#!/bin/bash
set -e
source $(conda info --base)/etc/profile.d/conda.sh
conda env create -f setup_environment/requirements_full.yml
conda activate covernet
pip install torch==1.8.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install torchnet==0.0.4
pip install -r setup_environment/torch_extensions.txt --find-links https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
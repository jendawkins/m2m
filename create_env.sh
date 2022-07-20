#!/bin/bash

# run "chmod u+x create_env.sh" if permission denied error

conda create --name m2m python=3.6 -y

# May not need this line and/or may need to change to miniconda path
source ~/miniconda3/etc/profile.d/conda.sh

conda activate m2m

conda install -c etetoolkit ete3 ete_toolchain -y

ete3 build check

pip install torch==1.4.0 torchvision==0.5.0

conda install -c conda-forge rdkit -y

pip install scikit-learn
pip install matplotlib
conda install xlrd -y
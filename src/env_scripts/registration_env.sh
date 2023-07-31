#!/bin/bash
# General, Common Requirements
conda create --name py310 -c conda-forge python=3.10 -y
conda run -n py310 pip install git+https://github.com/google-research/sofima
conda run -n py310 pip install numpy pandas tensorstore boto3 PyYAML
conda run -n py310 pip install google-cloud-storage  # Needed until Tensorstore has S3 support

# CPU versions of SOFIMA dependencies
conda run -n py310 pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
conda run -n py310 pip install tensorflow 

# Verify Coarse Registration
conda run -n py310 pip install aind-ng-link

# ***CuDNN library installation may be required here***
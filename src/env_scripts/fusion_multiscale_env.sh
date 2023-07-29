#!/bin/bash
# General, Common Requirements
conda create --name py310 -c conda-forge python=3.10 -y
conda run -n py311 pip install git+https://github.com/google-research/sofima
conda run -n py310 pip install numpy pandas tensorstore boto3 yaml
conda run -n py310 pip install google-cloud-storage  # Needed until Tensorstore has S3 support

# CPU versions of SOFIMA dependencies
conda run -n py310 pip install --upgrade "jax[cpu]"
conda run -n py310 pip install tensorflow-cpu

# Multiscale requirements
conda run -n py310 pip install aind-data-transfer ome-zarr xarray_multiscale hdf5plugin kerchunk ujson
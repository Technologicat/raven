#!/bin/bash
#
# Set up CUDA paths, required for some Python libraries such as spaCy in GPU mode.
# To use:
#
#     $(pdm venv activate)
#     source env.sh

# Get the directory of this script
# https://stackoverflow.com/a/246128
SOURCE=${BASH_SOURCE[0]}
while [ -L "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
    DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
    SOURCE=$(readlink "$SOURCE")
    [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )

# --------------------------------------------------------------------------------
# Config

# Location of Raven's venv, installed by PDM when you "pdm install".
VENV_BASE="$DIR/.venv"

# The version of Python installed in that venv.
PYTHON_VERSION="3.10"

# Where that Python's packages reside.
SITE_PACKAGES="${VENV_BASE}/lib/python${PYTHON_VERSION}/site-packages"

# --------------------------------------------------------------------------------
# Set up the environment variables

# Add `ptxas` executable to path
export PATH=$PATH:${SITE_PACKAGES}/nvidia/cuda_nvcc/bin

# The CUDA directory that contains `nvvm/libdevice/libdevice.10.bc`
export XLA_FLAGS=--xla_gpu_cuda_data_dir=${SITE_PACKAGES}/nvidia/cuda_nvcc

# Where to find `libcuda.so.1`
# This is the system `libcuda` from OS package `libnvidia-compute-xxx`, where xxx is the major version number of your NVIDIA drivers (e.g. 545).
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${SITE_PACKAGES}/nvidia/cuda_runtime/lib/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${SITE_PACKAGES}/nvidia/cuda_nvrtc/lib/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${SITE_PACKAGES}/nvidia/cublas/lib/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${SITE_PACKAGES}/nvidia/cudnn/lib/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${SITE_PACKAGES}/nvidia/cufft/lib/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${SITE_PACKAGES}/nvidia/curand/lib/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${SITE_PACKAGES}/nvidia/cusolver/lib/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${SITE_PACKAGES}/nvidia/cusparse/lib/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${SITE_PACKAGES}/tensorrt/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${SITE_PACKAGES}/tensorrt_libs/

CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn; print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=${CUDNN_PATH}/lib:${LD_LIBRARY_PATH}

# Test if we set up everything correctly
python -c "import cupy; import cupyx"
if [ $? -ne 0 ]; then
    echo -ne "Error setting up CUDA; please check the contents of `env.sh`.\n"
else
    echo -ne "CUDA environment setup complete. You can now start Raven.\n"
fi

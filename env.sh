#!/bin/bash
#
# Set up CUDA paths, required for some Python libraries such as spaCy in GPU mode.

# source ~/Documents/JAMK/fenics-stuff/extrafeathers/demo/vae/env.sh

# We need some pip packages from NVIDIA:
# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
#   nvidia-tensorrt  (should pull in cuda_runtime, cuda_nvrtc, cublas, cudnn)
#      nvidia_cuda_nvrtc_cu11  (in case it doesn't, this is the package name)
#      nvidia_cuda_runtime_cu11
#      nvidia_cublas_cu11
#      nvidia_cudnn_cu11
#   nvidia_cufft_cu11  (with appropriate cuda version; check ~/.local/lib/python3.10/site-packages/nvidia*)
#   nvidia_curand_cu11
#   nvidia_cusolver_cu11
#   nvidia_cusparse_cu11
#   nvidia_cuda_nvcc_cu11
#
# or ..._cu12, or whatever the current version is.
#
# May need to install specific versions, e.g.
#   pip install nvidia_cudnn_cu11==8.6.0.163
# (see error messages, if any, produced when running TensorFlow; should say if there is a version mismatch)
#
# Here is a semi-recent all-in-one install command:
#     pip install nvidia_cuda_nvrtc_cu12 nvidia_cuda_runtime_cu12 nvidia_cublas_cu12 nvidia_cudnn_cu12 nvidia_cufft_cu12 nvidia_curand_cu12 nvidia_cusolver_cu12 nvidia_cusparse_cu12 nvidia_cuda_nvcc_cu12

# ptxas
export PATH=$PATH:~/.local/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin
# cuda directory containing nvvm/libdevice/libdevice.10.bc
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/jje/.local/lib/python3.10/site-packages/nvidia/cuda_nvcc

# libcuda.so.1
# This is the system libcuda from libnvidia-compute-xxx, where xxx is the version number (e.g. 525).
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/lib/python3.10/site-packages/nvidia/cublas/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/lib/python3.10/site-packages/nvidia/cudnn/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/lib/python3.10/site-packages/nvidia/cufft/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/lib/python3.10/site-packages/nvidia/curand/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/lib/python3.10/site-packages/nvidia/cusolver/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/lib/python3.10/site-packages/nvidia/cusparse/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/lib/python3.10/site-packages/tensorrt/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/lib/python3.10/site-packages/tensorrt_libs/

CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$LD_LIBRARY_PATH


# Test if we set up everything correctly
python -c "import cupy; import cupyx"
if [ $? -ne 0 ]; then
    echo -ne "Error setting up CUDA\n"
fi

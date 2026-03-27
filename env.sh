#!/bin/bash
#
# Set up CUDA paths for pip-installed NVIDIA packages.
#
# pip-installed CUDA libs live inside site-packages/nvidia/,
# where the system linker can't find them. This script adds
# the relevant directories to LD_LIBRARY_PATH and PATH.
#
# Discovers paths dynamically — works with CUDA 12, 13, or both.
#
# To use:
#
#     source .venv/bin/activate
#     source env.sh

# --------------------------------------------------------------------------------
# Locate the venv's nvidia site-packages

# Get the directory of this script
# https://stackoverflow.com/a/246128
SOURCE=${BASH_SOURCE[0]}
while [ -L "$SOURCE" ]; do
    DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
    SOURCE=$(readlink "$SOURCE")
    [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE
done
DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )

if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "env.sh: no active venv — activate it first." >&2
    return 1 2>/dev/null || exit 1
fi

NVIDIA_BASE="${VIRTUAL_ENV}/lib/python$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages/nvidia"

if [ ! -d "$NVIDIA_BASE" ]; then
    echo "env.sh: ${NVIDIA_BASE} not found — no pip-installed NVIDIA packages?" >&2
    return 1 2>/dev/null || exit 1
fi

# --------------------------------------------------------------------------------
# LD_LIBRARY_PATH — add every nvidia lib/ directory that contains .so files

_RAVEN_CUDA_PATHS=""
while IFS= read -r libdir; do
    _RAVEN_CUDA_PATHS="${_RAVEN_CUDA_PATHS:+${_RAVEN_CUDA_PATHS}:}${libdir}"
done < <(find "$NVIDIA_BASE" -maxdepth 3 -name "lib" -type d | while read -r d; do
    ls "$d"/*.so* &>/dev/null && echo "$d"
done | sort)

# TensorRT keeps .so files in its package root, not a lib/ subdir
for _trt_dir in "${VIRTUAL_ENV}/lib/python"*/site-packages/tensorrt_libs; do
    if [ -d "$_trt_dir" ] && ls "$_trt_dir"/*.so* &>/dev/null; then
        _RAVEN_CUDA_PATHS="${_RAVEN_CUDA_PATHS:+${_RAVEN_CUDA_PATHS}:}${_trt_dir}"
    fi
done

if [ -n "$_RAVEN_CUDA_PATHS" ]; then
    export LD_LIBRARY_PATH="${_RAVEN_CUDA_PATHS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
unset _RAVEN_CUDA_PATHS _trt_dir

# --------------------------------------------------------------------------------
# PATH — add ptxas (needed by Triton / torch.compile)

_NVCC_BIN="${NVIDIA_BASE}/cuda_nvcc/bin"
if [ -d "$_NVCC_BIN" ]; then
    export PATH="${_NVCC_BIN}${PATH:+:${PATH}}"
fi
unset _NVCC_BIN

# --------------------------------------------------------------------------------

unset NVIDIA_BASE
echo "env.sh: CUDA environment ready."

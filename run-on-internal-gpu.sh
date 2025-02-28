#!/bin/bash
#
# Detect eGPU availability.
if [ $(nvidia-smi -L | wc -l) -ge 2 ]; then
    # eGPU is connected. Internal is #1.
    GPU_ID=1
else
    # Only one GPU (internal) available. Internal is #0.
    GPU_ID=0
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID

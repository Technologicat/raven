#!/bin/bash
#
# Hide the eGPU, so that CUDA code runs on the internal dGPU.
# Useful for benchmarking against the weaker card.
#
#     source run-on-internal-gpu.sh
#
# Two GPU numberings are in play here, and they do not agree:
#
#   - nvidia-smi enumerates in PCI bus order, which puts the internal dGPU first.
#   - CUDA (hence torch) defaults to CUDA_DEVICE_ORDER=FASTEST_FIRST, which puts
#     the beefier eGPU first. CUDA_VISIBLE_DEVICES is interpreted in *this*
#     ordering.
#
# So with the eGPU attached, the internal dGPU is GPU 0 to nvidia-smi but device
# 1 to CUDA — the two are exactly inverted. The indices below are CUDA's.
#
# This is why nvidia-smi is used here only to *count* the GPUs, never to index
# them: a count is the one thing both orderings agree on. Do not "fix" this by
# reading an index out of `nvidia-smi -L` — that reports the internal dGPU as
# GPU 0 even when it is CUDA device 1, and the script would then select the eGPU,
# i.e. exactly what it exists to avoid.
#
# (Setting CUDA_DEVICE_ORDER=PCI_BUS_ID would make the two agree, at the cost of
# reordering devices for every CUDA process in the shell. Not worth it just to
# make one script read more obviously.)

if [ $(nvidia-smi -L | wc -l) -ge 2 ]; then
    # eGPU is attached, so CUDA sees it first. The internal dGPU is device 1.
    GPU_ID=1
else
    # Only the internal dGPU is present, so it is the only device CUDA sees.
    GPU_ID=0
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID

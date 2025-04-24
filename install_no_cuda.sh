#!/bin/bash
#
# Run this script to install Raven, *without* CUDA support (CPU mode).
# This is useful for machines *without* an NVIDIA GPU.

pdm python install --min
pdm install

echo -ne "Raven is installed! You can now `source activate_venv.sh` and then start with `raven-visualizer`.\n"

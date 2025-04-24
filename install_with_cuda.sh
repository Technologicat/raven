#!/bin/bash
#
# Run this script to install Raven, with CUDA support (for machines with an NVIDIA GPU).

pdm python install --min
pdm install --prod -G cuda

echo -ne "Raven is installed! You can now `source activate_venv.sh` and then start with `raven-visualizer`.\n"
echo -ne "CUDA available. After activating the venv, you can check that it works by `raven-check-cuda`.\n"

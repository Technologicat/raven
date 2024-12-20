#!/bin/bash
#
# Set up CUDA paths, required for some Python libraries such as spaCy in GPU mode.

source ~/Documents/JAMK/fenics-stuff/extrafeathers/demo/vae/env.sh
python -c "import cupy; import cupyx"
if [ $? -ne 0 ]; then
    echo -ne "Error setting up CUDA\n"
fi

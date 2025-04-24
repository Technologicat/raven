#!/bin/bash
#
# This file must be used with "source activate_venv.sh" *from bash*
# you cannot run it directly
#
# When this script has been sourced, Raven commands such as `raven-visualizer`
# are available no matter which directory your terminal is in.

if [ "${BASH_SOURCE-}" = "$0" ]; then
    echo "You must source this script: \$ source $0" >&2
    exit 33
fi

$(pdm venv activate)

#!/bin/bash
#
# Avoids hammering HuggingFace servers on raven-server startup.

export HF_HUB_OFFLINE=1

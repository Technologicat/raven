"""Low-VRAM (8GB) alternative config that runs all Raven modules on CPU, leaving all VRAM available for the LLM server.

To use this, start the server with:

  raven-server --config raven.server.config_lowvram
"""

import torch

from .config import *  # Use the default config as a base.  # noqa: F401, F403

# We override just this, disabling the avatar and running all other modules on CPU.
#
# All modules that require heavy compute will be horribly slow - but the point is, a 7B/8B LLM needs
# about 8GB VRAM by itself, so when running with low VRAM, we can't afford to put anything else on the GPU.
enabled_modules = {
    # # no avatar when running with low VRAM
    # "avatar": {"device_string": "cuda:0",
    #            "dtype": torch.float16},
    "classify": {"device_string": "cpu",
                 "dtype": torch.float32},
    "embeddings": {"device_string": "cpu",
                   "dtype": torch.float32},
    "imagefx": {"device_string": "cpu",
                "dtype": torch.float32},
    "tts": {"device_string": "cpu"},
    "websearch": {},
}

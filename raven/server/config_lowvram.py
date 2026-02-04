"""Low-VRAM (8GB) alternative config that runs most Raven modules on CPU, leaving most VRAM available for the LLM server.

To use this, start the server with:

  raven-server --config raven.server.config_lowvram
"""

import torch

from .config import *  # Use the default config as a base.  # noqa: F401, F403

# We override just this, running everything except the avatar on CPU.
#
# All modules that require heavy compute will be horribly slow - but the point is, even Qwen3 2507 4B
# at 4bit and decent context size needs ~6GB VRAM by itself, so when running with low VRAM, we can't afford
# to put anything else on the GPU, except possibly the avatar.
enabled_modules = {
    "avatar": {"device_string": "cuda:0",
               "dtype": torch.float16},
    "classify": {"device_string": "cpu",
                 "dtype": torch.float32},
    "embeddings": {"device_string": "cpu",
                   "dtype": torch.float32},
    "imagefx": {"device_string": "cpu",
                "dtype": torch.float32},
    "natlang": {"device_string": "cpu"},  # this module has no dtype setting
    "sanitize": {"device_string": "cpu"},  # this module has no dtype setting
    "stt": {"device_string": "cpu",
            "dtype": torch.float32},
    "translate": {"device_string": "cpu",
                  "dtype": torch.float32},
    "tts": {"device_string": "cpu"},
    "websearch": {},
}

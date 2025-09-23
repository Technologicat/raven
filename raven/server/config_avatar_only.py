"""Avatar settings editor only alternative config. Necessary modules only.

For running just `raven-avatar-settings-editor`.

To use this, start the server with:

  raven-server --config raven.server.config_avatar_only
"""

import torch

from .config import *  # Use the default config as a base.  # noqa: F401, F403

# We override just this.
enabled_modules = {
    "avatar": {"device_string": "cuda:0",
               "dtype": torch.float16},
    "classify": {"device_string": "cuda:0",
                 "dtype": torch.float16},
    "imagefx": {"device_string": "cuda:0",
                "dtype": torch.float16},
    "natlang": {"device_string": "cuda:0"},  # this module has no dtype setting
    "tts": {"device_string": "cuda:0"},
}

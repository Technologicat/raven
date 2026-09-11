"""Avatar settings editor only alternative config. Necessary modules only.

For running just `raven-avatar-settings-editor`.

To use this, start the server with:

  raven-server --config raven.server.config_avatar_only
"""

import torch

from .. import configoverrides
from .config import *  # Use the default config as a base.  # noqa: F401, F403

# We override just this.
enabled_modules = {
    "avatar": {"device_string": "gpu",
               "dtype": torch.float16},
    "classify": {"device_string": "gpu",
                 "dtype": torch.float16},
    "imagefx": {"device_string": "gpu",
                "dtype": torch.float16},
    "natlang": {"device_string": "gpu"},  # this module has no dtype setting
    "tts": {"device_string": "gpu"},
}


# Machine-local overrides (`~/.config/raven/overrides.json`); applied last, so they can name anything above.
#
# Two keys reach this module. `raven.server.config`'s were applied before the star-import above and came
# through it; this call then reads `raven.server.config_avatar_only`, for anything meant to hold only when
# the server is started on this variant.
configoverrides.apply(__name__, globals())

"""Global configuration for the Raven constellation.

Some components also have their own configurations, which see:

  - client.config
  - server.config
  - librarian.config
  - visualizer.config
"""

import pathlib

# Used for various things. E.g. the web API keys go here.
toplevel_userdata_dir = "~/.config/raven/"

# Convert to an absolute path, just once here.
toplevel_userdata_dir = pathlib.Path(toplevel_userdata_dir).expanduser().resolve()

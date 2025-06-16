"""Client-side-specific config for Raven-avatar."""

import pathlib

from ..server import config as server_config

# TODO: Assumption: The `userdata_dir` is local anyway so we can just as well use the server app's.
# If you run on a different machine, set it here for the client side.
# The client currently only uses `userdata_dir` to load web API keys from.
userdata_dir = server_config.userdata_dir

# Raven-server
raven_server_url = "http://localhost:5100"
raven_api_key_file = userdata_dir / "api_key.txt"

# AI speech synthesizer server
#
# # https://github.com/remsky/Kokoro-FastAPI
# tts_url = "http://localhost:8880"
# tts_server_type = "kokoro"
# tts_api_key_file = userdata_dir / "tts_api_key.txt"

# Raven-server also provides a speech endpoint using a local Kokoro-82M.
# This has the advantage of more robust lipsyncing for the AI avatar, as well as
# slightly lower CPU usage while the TTS is idle.
tts_url = "http://localhost:5100"
tts_server_type = "raven"
tts_api_key_file = raven_api_key_file

# --------------------------------------------------------------------------------

# Convert to an absolute path, just once here.
userdata_dir = pathlib.Path(userdata_dir).expanduser().resolve()

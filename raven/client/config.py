"""Client-side-specific config for Raven-avatar."""

from ..server import config as server_config  # NOTE: default config location (can be overridden on the command line when starting the server)

# TODO: Assumption: The `userdata_dir` of the client/server pair is local anyway, so we can just as well use the server app's.
#
# If you run on a different machine, set it here for the client side.
#
# NOTE: If you change this, this must be a `pathlib.Path`. It is recommended to `.expanduser().resolve()` it, too, to make an absolute path.
#
# The client currently only uses `userdata_dir` to load web API keys from.
client_userdata_dir = server_config.server_userdata_dir

# --------------------------------------------------------------------------------

# Where to reach Raven-server
raven_server_url = "http://localhost:5100"
raven_api_key_file = client_userdata_dir / "api_key.txt"

# Where to reach the AI speech synthesizer server

# Raven-server provides a speech endpoint using a local Kokoro-82M.
# This has the advantage of rather robust lipsyncing for the AI avatar, as well as
# slightly lower CPU usage (compared to Kokoro-FastAPI) while the TTS is idle.
tts_server_type = "raven"
tts_url = "http://localhost:5100"
tts_api_key_file = raven_api_key_file

# # We also support Kokoro-FastAPI.
# # Note lipsynced speech may occasionally fail to speak anything with this option.
# # https://github.com/remsky/Kokoro-FastAPI
# tts_server_type = "kokoro"
# tts_url = "http://localhost:8880"
# tts_api_key_file = client_userdata_dir / "tts_api_key.txt"

# # Use these settings if you want to disable TTS
# tts_server_type = None
# tts_url = None
# tts_api_key_file = None

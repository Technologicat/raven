"""Client-side-specific config for Raven-avatar."""

import pathlib

from ..common import config as common_config

config_dir = pathlib.Path(common_config.config_base_dir).expanduser().resolve()

# Raven-avatar server
avatar_url = "http://localhost:5100"
avatar_api_key_file = config_dir / "api_key.txt"

# AI speech synthesizer server
#
# # https://github.com/remsky/Kokoro-FastAPI
# tts_url = "http://localhost:8880"
# tts_server_type = "kokoro"
# tts_api_key_file = config_dir / "tts_api_key.txt"

# The Raven-avatar server also provides a speech endpoint using a local Kokoro-82M.
# This has the advantage of more robust lip syncing for the AI avatar, as well as
# slightly lower CPU usage while the TTS is idle.
tts_url = "http://localhost:5100"
tts_server_type = "raven"
tts_api_key_file = avatar_api_key_file

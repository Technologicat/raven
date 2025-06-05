"""Client-side-specific config for Raven-avatar."""

import pathlib

from ..common import config as common_config

avatar_url = "http://localhost:5100"  # Raven-avatar server
tts_url = "http://localhost:8880"  # AI speech synthesizer server, https://github.com/remsky/Kokoro-FastAPI

config_dir = pathlib.Path(common_config.config_base_dir).expanduser().resolve()

avatar_api_key_file = config_dir / "api_key.txt"
tts_api_key_file = config_dir / "tts_api_key.txt"

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

# Which audio device to use for TTS speech.
#
# This is the device name as a string.
# For available devices on your system, run `raven-check-audio-devices`.
#
# The special value `None` uses the system's default device.
#
tts_playback_audio_device = None
# tts_playback_audio_device = "Built-in Audio Analog Stereo"
# tts_playback_audio_device = "M Audio Duo Analog Stereo"

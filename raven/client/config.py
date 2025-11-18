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

# Which audio playback device to use for TTS (text to speech, speech synthesizer).
#
# This is the device name as a string.
# For available devices on your system, run `raven-check-audio-devices`.
#
# The special value `None` uses the first available device
# (first in the order listed by `raven-check-audio-devices`).
#
# The special value "system-default" uses the system's default device
# (i.e. the same one that other apps use).
#
tts_playback_audio_device = "system-default"  # OS's default, i.e. the same one other apps use
# tts_playback_audio_device = None  # first available as listed by `raven-check-audio-devices`
# tts_playback_audio_device = "Built-in Audio Analog Stereo"
# tts_playback_audio_device = "M Audio Duo Analog Stereo"

# Which audio capture device to use for STT (speech to text, speech recognition).
#
# This is the device name as a string.
# For available devices on your system, run `raven-check-audio-devices`.
#
# The special value `None` uses the first NON-monitoring audio capture device
# (first in the order listed by `raven-check-audio-devices`).
#
# (A monitoring capture device is a capture device that records the audio
#  that is going to a playback device.)

# This has NO system-default setting, as our recording backend doesn't support that.
#
stt_capture_audio_device = None
# stt_capture_audio_device = "Built-in Audio Analog Stereo"

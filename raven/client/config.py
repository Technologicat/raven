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

# --------------------------------------------------------------------------------
# Device settings for local-mode fallback of `MaybeRemote.*` services.
#
# When the corresponding `tts` / `stt` / ... module on Raven-server is reachable,
# `MaybeRemote` services go through it (no local model is loaded). The records here
# parameterize the in-process fallback's compute device, used when `<svc>_allow_local`
# is `True` AND the server isn't reachable.
#
# Same shape as `raven.server.config.enabled_modules` and as the `devices` dict in
# `raven.{librarian,visualizer}.config`. Validated by `raven.common.deviceinfo.validate`
# during `raven.client.api.initialize` (CUDA → CPU fallback, `device_name` injection).
devices = {
    "tts": {"device_string": "cpu"},  # Local TTS on CPU is workable for chat-paced speech, slower than server-mode GPU.
}

# TTS local-mode fallback settings.
#
# Most apps leave `tts_allow_local = False`: client apps are typically paired with a
# server (the avatar especially requires it), so falling back to local Kokoro pays
# costs the user wasn't expecting — extra RAM for the model, plus a multi-hundred-
# megabyte download the first time if the server is on another machine (on localhost
# the HuggingFace cache is shared, so download cost is zero, but RAM still doubles).
# Same reason `raven.librarian` passes `local_model_loader_fallback=False` to its
# `HybridIR` for the embedder + spaCy. Apps that want standalone capability
# (e.g. a future no-avatar Librarian) flip this on.
tts_allow_local = False

# HuggingFace repo id for the local-mode Kokoro TTS model. Defaults to the same model
# the server uses, so client-local and server-side synthesis match.
tts_model_name = server_config.kokoro_models

# Phonemizer language code for Kokoro. "a" is American English; "b" is British English.
# Word-level metadata (needed for avatar lipsync) currently only supports English.
# See `raven.common.audio.speech.tts.load_tts_pipeline` for the full list.
tts_lang_code = "a"

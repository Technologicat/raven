"""Client-side-specific config for Raven-avatar."""

from typing import NamedTuple

import torch

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

# Network timeouts for talking to Raven-server. `requests` accepts a `(connect, read)` tuple for its
# `timeout=` argument; `Timeout` is such a tuple, but with named fields so the call sites and the values
# below read clearly. (A plain 2-tuple would work too, but "which number is which?" is exactly what we
# want to avoid.) A `read` of `None` means unbounded (used for streaming endpoints).
class Timeout(NamedTuple):
    connect: float
    read: float | None

# The connect timeout bounds the "server unreachable" case — Raven-server currently runs on the same
# machine, but it may be configured to run on another machine (e.g. a lab setup), and if that host is
# down a request would otherwise hang on the OS-level connection attempt. With a connect timeout it
# fails fast instead.
#
# The read timeout is `requests`' *between-bytes* timeout, not a total-request budget; it must
# comfortably exceed the server's worst-case time-to-first-byte for the heavy endpoints (embeddings,
# imagefx, stt, translate, natlang), hence the generous default.
network_timeout = Timeout(connect=10.0, read=300.0)

# Streaming endpoints (avatar `result_feed`, TTS `speak`) legitimately stay open for a long time, so we
# bound only the connect; a read timeout would abort a healthy long-lived stream mid-flight.
network_timeout_streaming = Timeout(connect=10.0, read=None)

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

# How the recorder decides that you have stopped speaking, and how its VU meter behaves.
#
# These are starting values. Raven-librarian offers all three as live controls, and remembers
# what you set them to — so once you have tuned them there, the app state file is what it starts
# from, and these apply only to a fresh state file and to the panel's reset button.
#
# `stt_silence_threshold`: dBFS below which the input counts as silence. 0 is full scale and
#                          -90 is the quietest 16-bit audio can be, so a lower number means a
#                          more sensitive microphone or a noisier room.
#
#                          `None` measures the room at the start of each recording instead,
#                          which cannot work if your input has a noise gate in front of it.
#
#                          Worth knowing when choosing one: a single audio frame above the
#                          threshold restarts the autostop clock, so in a noisy room the value
#                          has to sit above the occasional bang, not above the average level.
#
# `stt_autostop_timeout`: seconds of continuous silence after which the recording stops itself
#                         and is sent for transcription. `None` disables it, leaving the
#                         microphone button as the only way to stop — which is what to reach for
#                         in a room too loud for any threshold to separate speech from noise.
#
# `stt_vu_peak_hold`: seconds the VU meter holds a peak before letting it fall. This is also how
#                     far back the meter's peak line lets you see, which is what makes it useful
#                     for choosing a threshold.
#
stt_silence_threshold = -40.0  # dBFS
# stt_silence_threshold = None  # measure the room at the start of each recording
stt_autostop_timeout = 1.5  # seconds
# stt_autostop_timeout = None  # never stop by itself
stt_vu_peak_hold = 1.0  # seconds

# --------------------------------------------------------------------------------
# Device settings for local-mode fallback of `MaybeRemote.*` services.
#
# When the corresponding `tts` / `stt` / ... module on Raven-server is reachable,
# `MaybeRemote` services go through it (no local model is loaded). The records here
# parameterize the in-process fallback's compute device, used when `<svc>_allow_local`
# is `True` AND the server isn't reachable.
#
# Same shape as `raven.server.config.enabled_modules` and as `raven.librarian.config.devices`,
# which stays separate because Librarian's RAG backend may legitimately want a different model
# and device from an importer's. Validated by `raven.common.deviceinfo.validate` during
# `raven.client.api.initialize` (CUDA → CPU fallback, `device_name` injection).
# See also `run-on-internal-gpu.sh` for another way to select the GPU when starting an app, without
# modifying any files.
devices = {
    "tts": {"device_string": "cpu"},  # Local TTS on CPU is workable for chat-paced speech, slower than server-mode GPU.
    "sanitize": {"device_string": "gpu"},  # dehyphenation; no configurable dtype
    "embeddings": {"device_string": "gpu",
                   "dtype": torch.float16},
    "nlp": {"device_string": "gpu"},  # no configurable dtype
}

# Which dehyphenation model `MaybeRemote.Dehyphenator` loads in local mode. Character-level contextual
# embeddings by Flair-NLP, used to repair text broken across lines — as extracted from a PDF, typically.
#
# Here rather than in one app's config because three of them want it and only one of those is that app:
# `visualizer.importer`, `papers.pdf2bib` and `tools.dehyphenate` all reach `MaybeRemote.Dehyphenator`,
# and `pdf2bib` carried a standing note that a tool should not be loading the Visualizer's config to get
# at it.
#
# NOTE: Raven uses dehyphenation models in two places, and they do not have to be the same.
#  - Client-side local fallback: this setting.
#  - Raven-server: served by the `sanitize` module; see `raven.server.config.dehyphenation_model`.
#
# This is NOT a HuggingFace model name, but is auto-downloaded (by Flair-NLP) on first use, into
# `~/.flair/embeddings/`. Loaded by the `dehyphen` package; omit the "-forward" or "-backward" part of the
# name, which is added automatically. Try "multi" first — it should support 300+ languages; if that does
# not perform adequately, look at the docs.
#
# For available models, see:
#     https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/FLAIR_EMBEDDINGS.md
#     https://github.com/flairNLP/flair/blob/master/flair/embeddings/token.py
dehyphenation_model = "multi"

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

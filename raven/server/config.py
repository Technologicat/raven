"""Raven-server configuration.

This module is licensed under the 2-clause BSD license.
"""

import torch

from .. import config as global_config

# Where to store files. Currently only used for the server API key, and websearch's debug functionality.
server_userdata_dir = global_config.toplevel_userdata_dir / "server"

# When `raven.server` is running in "--secure" mode:
#   - Require the client to provide the API key that is in this file.
#   - When the server starts, if the API key file doesn't exist, create it.
#   - When the server starts, print the API key to the server's console.
server_api_key_file = server_userdata_dir / "api_key.txt"

# This can be used to enable only those modules you need, to save CPU/GPU/RAM/VRAM resources.
#
# To switch a module off, comment out its line here.
#
# NOTE: This configures the server-side devices.
#
enabled_modules = {
    "avatar": {"device_string": "cuda:0",
               "dtype": torch.float16},
    "classify": {"device_string": "cuda:0",
                 "dtype": torch.float16},
    "embeddings": {"device_string": "cuda:0",
                   "dtype": torch.float16},
    "imagefx": {"device_string": "cuda:0",
                "dtype": torch.float16},
    "natlang": {"device_string": "cuda:0"},  # this module has no dtype setting
    "sanitize": {"device_string": "cuda:0"},  # this module has no dtype setting
    "stt": {"device_string": "cuda:0",
            "dtype": torch.float16},
    "summarize": {"device_string": "cuda:0",  # device settings used for the simple summarizer
                  "dtype": torch.float16},
    "translate": {"device_string": "cuda:0",
                  "dtype": torch.float16},
    "tts": {"device_string": "cuda:0"},
    "websearch": {},  # websearch doesn't use any heavy compute; this is here only to provide the option to turn the module off.
}

# The port Raven-server listens to. Can be overridden on the command line.
#
default_port = 5100

# --------------------------------------------------------------------------------
# Miscellaneous AI model config

# Unless otherwise explained for a particular setting, these are HuggingFace model names.
# Each model is auto-downloaded on first use.
#
# Unless otherwise explained, the model install location is the default used by the
# `huggingface_hub` package, namely `~/.cache/huggingface/hub`.

# Text classification model for emotion detection.
#
# Used for dynamically auto-updating the emotion shown by the AI's avatar.
#
# See:
#     https://huggingface.co/tasks/text-classification
#
classification_model = "joeddav/distilbert-base-uncased-go-emotions-student"
# classification_model = "nateraw/bert-base-uncased-emotion"

# Character-level contextual embeddings by Flair-NLP. Used for dehyphenation of broken text (e.g. as extracted from a PDF file).
#
# NOTE: Raven uses dehyphenation models in two places, and they don't have to be the same.
#  - Raven-visualizer: processing of abstracts during BibTeX import
#  - Raven-server: served by the `sanitize` module (this setting)
#
# This is NOT a HuggingFace model name, but is auto-downloaded (by Flair-NLP) on first use.
#
# This is installed into `~/.flair/embeddings/`.
#
# For available models, see:
#     https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/FLAIR_EMBEDDINGS.md
#     https://github.com/flairNLP/flair/blob/master/flair/embeddings/token.py
#
# This model is loaded by the `dehyphen` package; omit the "-forward" or "-backward" part
# of the model name, those are added automatically.
#
# At first, try "multi", it should support 300+ languages. If that doesn't perform adequately, then look at the docs.
#
dehyphenation_model = "multi"

# AI models that produce the high-dimensional semantic vectors, served by the `embeddings` module.
# This is a mapping of role -> model name.
#
# NOTE: Raven uses embedding models in three places, and they don't have to be the same.
#  - Raven-librarian: RAG backend
#  - Raven-visualizer: producing the semantic map
#  - Raven-server: served by the `embeddings` module (this setting)
#
# See:
#     https://sbert.net/docs/sentence_transformer/pretrained_models.html
#     https://huggingface.co/tasks/sentence-similarity
#
# Some general-use models:
#     "Snowflake/snowflake-arctic-embed-l"  ~1.3 GB
#     "Snowflake/snowflake-arctic-embed-m"  ~440 MB
#     "sentence-transformers/all-mpnet-base-v2"  ~440 MB
#
# Several roles can use the same model. Duplicates are handled automatically; only one copy of each unique model is loaded.
#
# Upon loading, each model will be served under two names: its role name, and the model name (if these are different).
# This is sometimes convenient, e.g. when `raven.librarian.hybridir.HybridIR` loads a database.
#
# Keep the number of unique models small. At server startup, *all* unique models are loaded onto the device specified
# for "embeddings" in `enabled_modules`. So this can eat a lot of VRAM.
#
embedding_models = {
    "default": "Snowflake/snowflake-arctic-embed-l",  # general-use model
    "qa": "sentence-transformers/multi-qa-mpnet-base-cos-v1",  # maps questions and related answers near each other
}

# Models for the Kokoro speech synthesizer (text to speech, TTS).
#
# The newer, smaller and faster KittenTTS is currently not supported, because it does not output
# per-word timestamps and per-word phonemes, which are needed for avatar lipsync. This is tracked here:
#   https://github.com/KittenML/KittenTTS/issues/14
#
kokoro_models = "hexgrad/Kokoro-82M"  # ~360 MB

# NLP model for spaCy, used for breaking text into sentences in the `summarize` module.
#
# NOTE: Raven uses spaCy models in three places, and they don't have to be the same.
#  - Raven-visualizer: keyword extraction
#  - Raven-librarian: tokenization for keyword search
#  - Raven-server: breaking text into sentences in the `summarize` module (this setting)
#                  and served by the `nlp` module
#
# This is NOT a HuggingFace model name, but is auto-downloaded (by spaCy) on first use.
# For available models, see:
#     https://spacy.io/models
#
spacy_model = "en_core_web_sm"  # Small pipeline; fast, runs fine on CPU, but can also benefit from GPU acceleration.
# spacy_model = "en_core_web_trf"  # Transformer-based pipeline; more accurate, slower, requires GPU, takes lots of VRAM.

# Speech recognition (speech to text, STT) model.
#
# Served by the `stt` module. Used by Raven-librarian for speech input in the AI chat.
# Supports "multiple languages" (which, the model card doesn't say), but currently we only use English.
#
# https://huggingface.co/openai/whisper-large-v3-turbo
#
speech_recognition_model = "openai/whisper-large-v3-turbo"  # TODO: also this model is ~1.6 GB; look into quantized variants to save VRAM (may need to change the backend to the vLLM library to run those)

# AI model used by the `summarize` module, for abstractive summarization.
#
# This is a small AI model specialized to the task of summarization ONLY, not a general-purpose LLM.
#
# NOTE: Raven uses a summarizer model in two places, and they don't have to be the same.
#  - Raven-visualizer: tldr AI summarization of abstracts in importer
#  - Raven-server: `summarize` module (this setting)
#
# `summarization_prefix`: Some summarization models need input to be formatted like
#     "summarize: Actual text goes here...". This sets the prefix, which in this example is "summarize: ".
#     For whether you need this and what the value should be, see the model card for your particular model.
#
# summarization_model = "ArtifactAI/led_base_16384_arxiv_summarization"  # ~650 MB
# summarization_model = "ArtifactAI/led_large_16384_arxiv_summarization"  # ~1.8 GB
# summarization_model = "Falconsai/text_summarization"  # ~250 MB
summarization_model = "philschmid/flan-t5-base-samsum"  # ~1 GB, performs well
summarization_prefix = ""  # for all of the summarizers listed above

# summarization_model = "KipperDev/bart_summarizer_model"
# summarization_prefix = "summarize: "

# Machine translation AI models.
#
# These are used for translating one natural language to another, e.g. English to Finnish.
#
# See:
#   https://huggingface.co/tasks/translation
#
# As with `embedding_models`, the models are de-duplicated, and *all* unique models are loaded upon server startup.
# So keep the number of unique models small to save your sanity and VRAM.
#
# The format is:
#
#   {target_langcode: {source_langcode: model_name,
#                      ...},
#    ...
#   }
#
translation_models = {
    "fi": {"en": "Helsinki-NLP/opus-mt-tc-big-en-fi"},  # to fi, from en
    # "en": {"fi": "Helsinki-NLP/opus-mt-tc-big-fi-en"},  # to en, from fi
}

# --------------------------------------------------------------------------------
# AI avatar

# THA3 animator models. There are currently no alternative models, this is just for specifying where to download from.
#
# HuggingFace model name, auto-downloaded on first use.
#
# Unlike the other HuggingFace models, this is installed into `raven/vendor/tha3/models`.
#
talkinghead_models = "OktayAlpk/talking-head-anime-3"  # ~900 MB

# Default configuration for the pixel-space postprocessor, to make the AI's avatar
# look more cyberpunk via pixel-space glitch artistry.
#
# This documents the correct ordering of the filters.
# Feel free to improvise, but make sure to understand why your filter chain makes sense.
#
# For details, see `postprocessor.py`.
#
postprocessor_defaults = [
    # physical input signal
    ("bloom", {}),

    # video camera
    ("chromatic_aberration", {}),
    ("vignetting", {}),

    # scifi hologram output
    ("translucency", {}),
    # ("noise", {"strength": 0.1, "sigma": 0.0, "channel": "A"}),

    # # lo-fi analog video
    # ("analog_lowres", {}),
    # ("noise", {"strength": 0.2, "sigma": 2.0, "channel": "A"}),
    # ("analog_rippling_hsync", {}),
    # # ("analog_vhsglitches", {}),
    # ("analog_vhstracking", {}),

    # CRT TV output (for a touch of retrofuturism)
    ("banding", {}),
    ("scanlines", {})
]

# Default configuration for the animator, loaded when raven-avatar is launched.
#
# This doubles as the authoritative documentation of the animator settings, beside the animation driver docstrings and the actual source code.
#
# MUST have ALL settings defined. Whenever animator settings are loaded, this is used for validating which settings exist and what their default values are.
#
# For details, see `animator.py`.
#
animator_defaults = {
    # Attempt to render this many output frames per second. This only affects smoothness of the output (depending on the speed of the hardware).
    #
    # The speed at which the animation evolves is always based on wall time. Snapshots are rendered at the target FPS,
    # or if the hardware is too slow to reach the target FPS, then as often as hardware allows.
    #
    # To obtain smooth animation, make the target FPS lower than what your hardware could produce, so that some compute remains untapped,
    # available to smooth over the occasional hiccup due to other running programs.
    "target_fps": 25,

    # The video stream is sent as a multipart-x-mixed-replace of frame images.
    #
    # The images are encoded on the CPU.
    # Available image formats are "QOI" (Quite OK Image, lossless, fast) and RGBA formats supported by Pillow: "PNG", "TGA" (RLE compressed), "IM" (IFUNC/LabEye).
    # For Pillow, see:
    #     https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html
    #
    # To optimize encoding speed when displaying the avatar in a Python client (such as `raven-avatar-settings-editor`), we recommend QOI.
    # For maximum compatibility, you can try PNG, but be aware that its encoder can be 30x slower than QOI, so especially when upscaling to 2.0, it may be too slow for realtime use.
    #
    "format": "PNG",

    # Upscaler settings.
    "upscale": 1.0,  # 1.0 = send as-is (512x512); e.g. 2.0 = upscale 2x -> 1024x1024 using anime4k before sending
    "upscale_preset": "C",  # only used if upscale != 1.0; "A", "B" or "C"; these roughly correspond to the presets of Anime4K  https://github.com/bloc97/Anime4K/blob/master/md/GLSL_Instructions_Advanced.md
    "upscale_quality": "low",  # only used if upscale != 1.0; quality "low": fast, acceptable image quality; "high": slow, good image quality

    # Performance profiling settings.
    "metrics_enabled": False,  # Detailed performance logging for the renderer; slows the renderer down, but shows where the rendering time goes. Average FPS calculation is always on, and doesn't slow down anything.

    # Canvas cropping settings.
    #
    # If the avatar does not occupy the whole 512x512 canvas, it is possible to cut away the empty space from the edges,
    # which makes postprocessing faster (since fewer pixels).
    #
    # Applied after upscaling, but before postprocessing.
    #
    # Note this means that the client receiving the video stream will need to read the image size from the stream (from each frame separately!).
    "crop_left": 0.0,  # cut how much inward from left edge, in units where the image width is 2.0
    "crop_right": 0.0,  # cut how much inward from right edge, in units where the image width is 2.0
    "crop_top": 0.0,  # cut how much inward from top edge, in units where the image height is 2.0
    "crop_bottom": 0.0,  # cut how much inward from bottom edge, in units where the image height is 2.0

    # Animation speed settings.
    "pose_interpolator_step": 0.3,  # 0 < this <= 1; relative change toward target at each frame at a reference of 25 FPS; FPS-corrected automatically. For details, see `interpolate` in `raven.server.modules.avatar`.

    # Eye-blinking settings.
    "blink_interval_min": 2.0,  # seconds, lower limit for random minimum time until next blink is allowed.
    "blink_interval_max": 5.0,  # seconds, upper limit for random minimum time until next blink is allowed.
    "blink_probability": 0.03,  # At each frame at a reference of 25 FPS; FPS-corrected automatically.
    "blink_confusion_duration": 10.0,  # seconds, upon entering "confusion" emotion, during which blinking quickly in succession is allowed.

    # Talking animation settings.
    "talking_fps": 12,  # How often to re-randomize mouth during the generic mouth randomizer talking animation. Not used when lipsyncing.
                        # Early 2000s anime used ~12 FPS as the fastest actual framerate of new cels (not counting camera panning effects and such).
    "talking_morph": "mouth_aaa_index",  # which mouth-open morph to use for the generic talking animation; for available values, see `posedict_keys` in `raven.server.modules.avatarutil`.

    # Sway (idle pose variation) settings.
    "sway_morphs": ["head_x_index", "head_y_index", "neck_z_index", "body_y_index", "body_z_index"],  # which morphs to sway; see `posedict_keys`
    "sway_interval_min": 5.0,  # seconds, lower limit for random time interval until randomizing new sway pose.
    "sway_interval_max": 10.0,  # seconds, upper limit for random time interval until randomizing new sway pose.
    "sway_macro_strength": 0.6,  # [0, 1], in sway pose, max abs deviation from emotion pose target morph value for each sway morph,
    # but also max deviation from center. The emotion pose itself may use higher values; in such cases,
    # sway will only occur toward the center. See `compute_sway_target_pose` for details.
    "sway_micro_strength": 0.02,  # [0, 1], max abs random noise added each frame. No limiting other than a clamp of final pose to [-1, 1].

    # Breathing animation settings.
    "breathing_cycle_duration": 4.0,  # seconds, for a full breathing cycle.

    # Scifi "data eyes" (LLM tool access indicator) animation settings.
    "data_eyes_fps": 12.0,  # cel animation framerate. Special value 0.0 disables the animation.

    # "Intense emotion" eye-waver animation settings.
    "eye_waver_fps": 12.0,  # cel animation framerate. Special value 0.0 disables the animation.

    # Backdrop image. Applied at the client side.
    #
    # Path can be absolute or relative path. Relative means relative to CWD when you start a client app. The avatar settings editor always saves an absolute path; but feel free to edit the JSON file.
    "backdrop_path": None,  # path to backdrop image (`str`), or `None` for no backdrop.
    "backdrop_blur": True,  # whether to blur the backdrop image. The blur is applied once, when the backdrop is loaded, so it doesn't affect rendering performance.

    # animefx: anime-style emotional reaction effects that hover *around* the character.
    #
    # All of these effects are optional. Effects are rendered in the order listed below. The ordering matters only if an emotion has multiple effects assigned to it.
    #
    # - When animefx are enabled, all characters use generic fx cels ("fx_*.png") by default, so that the effects can be enabled for any character without extra effort.
    #
    # - If you want to customize the look of the effects for a specific character, supply those cels for that character ("mycharacter_fx_*.png").
    #   Character-specific cels automatically override the generic ones.
    #
    # - To disable a specific animefx for all characters, but keep its settings, use its "enabled" setting.
    #
    # - To disable all animefx for all characters, use the "animefx_enabled" setting below.
    #
    # Full list of the 28 distilbert emotions, for reference:
    #   admiration
    #   amusement
    #   anger
    #   annoyance
    #   approval
    #   caring
    #   confusion
    #   curiosity
    #   desire
    #   disappointment
    #   disapproval
    #   disgust
    #   embarrassment
    #   excitement
    #   fear
    #   gratitude
    #   grief
    #   joy
    #   love
    #   nervousness
    #   neutral
    #   optimism
    #   pride
    #   realization
    #   relief
    #   remorse
    #   sadness
    #   surprise
    #
    "animefx_enabled": True,  # on/off switch for all animefx (in one animator instance; note each avatar session has its own instance)

    # format is [[effect_name0, config_dict0], ...]
    "animefx": [
        ["angervein", {"enabled": True,
                       "emotions": ["anger"],  # trigger emotion(s); entering any emotion listed here triggers the effect (anew each time).
                       "type": "cycle_with_fadeout",  # animation type: one of "cycle" (loop indefinitely), "sequence" (play once), "cycle_with_fadeout", "sequence_with_fadeout"
                       "fps": 6.0,  # for "cycle" or "cycle_with_fadeout": frames per second for the cel cycling
                       "duration": 1.0,  # seconds, total duration of animation (in this case the fadeout)
                       "cels": ["fx_angervein1", "fx_angervein2"]}],  # list of one or more cels that the animation consists of

        ["sweatdrop", {"enabled": True,
                       "emotions": ["embarrassment"],
                       "type": "sequence_with_fadeout",
                       "duration": 0.3,
                       "cels": ["fx_sweatdrop1", "fx_sweatdrop2", "fx_sweatdrop3"]}],

        ["smallsweatdrop", {"enabled": True,
                            "emotions": ["nervousness"],
                            "type": "sequence_with_fadeout",
                            "duration": 0.3,
                            "cels": ["fx_smallsweatdrop1", "fx_smallsweatdrop2", "fx_smallsweatdrop3"]}],

        ["heart", {"enabled": True,
                   "emotions": ["desire", "love"],
                   "type": "sequence_with_fadeout",
                   "duration": 0.3,
                   "cels": ["fx_heart1", "fx_heart2", "fx_heart3"]}],

        ["blackcloud", {"enabled": True,
                        "emotions": ["annoyance"],
                        "type": "cycle_with_fadeout",
                        "fps": 6.0,
                        "duration": 1.0,
                        "cels": ["fx_blackcloud1", "fx_blackcloud2"]}],

        ["flowers", {"enabled": True,
                     "emotions": ["joy"],
                     "type": "cycle_with_fadeout",
                     "fps": 6.0,
                     "duration": 1.0,
                     "cels": ["fx_flowers1", "fx_flowers2"]}],

        ["shock", {"enabled": True,
                   "emotions": ["disgust", "fear"],
                   "type": "sequence_with_fadeout",
                   "duration": 2.0,
                   "cels": ["fx_shock1"]}],  # this "sequence" has just one cel; that's fine

        ["notice", {"enabled": True,
                    "emotions": ["surprise"],
                    "type": "sequence",
                    "duration": 0.25,
                    "cels": ["fx_notice1", "fx_notice2", "fx_notice1", "fx_notice2"]}],  # the same cels can also repeat

        ["beaming", {"enabled": True,
                     "emotions": ["admiration", "amusement", "excitement", "pride"],  # TODO: approval, gratitude?
                     "type": "sequence",
                     "duration": 0.25,
                     "cels": ["fx_beaming1", "fx_beaming2"]}],

        ["question", {"enabled": True,
                      "emotions": ["confusion"],
                      "type": "sequence",
                      "duration": 0.25,
                      "cels": ["fx_question1", "fx_question2", "fx_question3"]}],

        ["exclaim", {"enabled": True,
                     "emotions": ["realization"],
                     "type": "sequence",
                     "duration": 0.25,
                     "cels": ["fx_exclaim1", "fx_exclaim2", "fx_exclaim3"]}],
    ],

    # postprocessor
    "postprocessor_chain": postprocessor_defaults
}

"""Avatar configuration.

This module is licensed under the 2-clause BSD license.
"""

import torch

# This can be used to enable only those modules you need, to save CPU/GPU/RAM/VRAM resources.
# To switch a module off, comment out its line here.
#
SERVER_ENABLED_MODULES = {
    "avatar": {"device_string": "cuda:0", "dtype": torch.float16},
    "classify": {"device_string": "cuda:0", "dtype": torch.float16},
    "embeddings": {"device_string": "cuda:0", "dtype": torch.float16},
    "imagefx": {"device_string": "cuda:0", "dtype": torch.float16},
    "tts": {"device_string": "cuda:0"},
    "websearch": {},  # websearch doesn't use any heavy compute; this is here to provide the option to turn the module off.
}

# Port the raven-avatar app is hosted on. Can be overridden on the command line.
#
DEFAULT_PORT = 5100

# AI model that produces the high-dimensional semantic vectors, for visualization in `raven-visualizer`.
# Available on HuggingFace. Auto-downloaded on first use.
#
EMBEDDING_MODEL = "Snowflake/snowflake-arctic-embed-l"
# EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Text classification model for emotion detection.
#
# Used for dynamically auto-updating the emotion shown by the AI's avatar.
#
# Huggingface model name, auto-downloaded on first use.
#
CLASSIFICATION_MODEL = "joeddav/distilbert-base-uncased-go-emotions-student"
# CLASSIFICATION_MODEL = "nateraw/bert-base-uncased-emotion"

# THA3 avatar animator models. There are currently no alternative models, this is just for specifying where to download from.
#
# Huggingface model name, auto-downloaded on first use.
#
TALKINGHEAD_MODELS = "OktayAlpk/talking-head-anime-3"

# Models for the Kokoro speech synthesizer (text to speech, TTS).
#
# Huggingface model name, auto-downloaded on first use.
#
KOKORO_MODELS = "hexgrad/Kokoro-82M"

# Where to store files. Currently only used for websearch's debug functionality.
config_base_dir = "~/.config/raven/avatar/"

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
    # Available output formats for a frame are "QOI" (Quite OK Image, lossless, fast) and RGBA formats supported by Pillow: "TGA" (RLE compressed), "PNG", "IM" (IFUNC/LabEye).
    # For Pillow, see:
    #     https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html
    #
    # SillyTavern expects PNG. To optimize encoding speed with a custom Python client, we recommend QOI.
    #
    "format": "PNG",
    "upscale": 1.0,  # 1.0 = send as-is (512x512); e.g. 2.0 = upscale 2x -> 1024x1024 using anime4k before sending
    "upscale_preset": "C",  # only used if upscale != 1.0; "A", "B" or "C"; these roughly correspond to the presets of Anime4K  https://github.com/bloc97/Anime4K/blob/master/md/GLSL_Instructions_Advanced.md
    "upscale_quality": "low",  # only used if upscale != 1.0; quality "low": fast, acceptable image quality; "high": slow, good image quality

    "metrics_enabled": False,  # Detailed performance logging; slows the renderer down, but shows how much time each step takes. Average FPS calculation is always on and doesn't slow anything.

    # If the avatar does not occupy the whole 512x512 canvas, it is possible to cut away the empty space from the edges,
    # which makes postprocessing faster (since fewer pixels).
    "crop_left": 0.0,  # cut how much inward from left edge, in units where the image width is 2.0
    "crop_right": 0.0,  # cut how much inward from right edge, in units where the image width is 2.0
    "crop_top": 0.0,  # cut how much inward from top edge, in units where the image height is 2.0
    "crop_bottom": 0.0,  # cut how much inward from bottom edge, in units where the image height is 2.0

    "pose_interpolator_step": 0.3,  # 0 < this <= 1; at each frame at a reference of 25 FPS; FPS-corrected automatically; see `interpolate_pose`.

    "blink_interval_min": 2.0,  # seconds, lower limit for random minimum time until next blink is allowed.
    "blink_interval_max": 5.0,  # seconds, upper limit for random minimum time until next blink is allowed.
    "blink_probability": 0.03,  # At each frame at a reference of 25 FPS; FPS-corrected automatically.
    "blink_confusion_duration": 10.0,  # seconds, upon entering "confusion" emotion, during which blinking quickly in succession is allowed.

    "talking_fps": 12,  # How often to re-randomize mouth during the simple mouth randomizer talking animation. Not used when lipsyncing.
                        # Early 2000s anime used ~12 FPS as the fastest actual framerate of new cels (not counting camera panning effects and such).
    "talking_morph": "mouth_aaa_index",  # which mouth-open morph to use for talking; for available values, see `util.posedict_keys`

    "sway_morphs": ["head_x_index", "head_y_index", "neck_z_index", "body_y_index", "body_z_index"],  # which morphs to sway; see `posedict_keys`
    "sway_interval_min": 5.0,  # seconds, lower limit for random time interval until randomizing new sway pose.
    "sway_interval_max": 10.0,  # seconds, upper limit for random time interval until randomizing new sway pose.
    "sway_macro_strength": 0.6,  # [0, 1], in sway pose, max abs deviation from emotion pose target morph value for each sway morph,
    # but also max deviation from center. The emotion pose itself may use higher values; in such cases,
    # sway will only occur toward the center. See `compute_sway_target_pose` for details.
    "sway_micro_strength": 0.02,  # [0, 1], max abs random noise added each frame. No limiting other than a clamp of final pose to [-1, 1].

    "breathing_cycle_duration": 4.0,  # seconds, for a full breathing cycle.

    "postprocessor_chain": postprocessor_defaults
}

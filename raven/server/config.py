"""Raven-server configuration.

This module is licensed under the 2-clause BSD license.
"""

import torch

from .. import config as global_config

# Where to store files. Currently only used for websearch's debug functionality.
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
    "tts": {"device_string": "cuda:0"},
    "websearch": {},  # websearch doesn't use any heavy compute; this is here only to provide the option to turn the module off.
}

# The port Raven-server listens to. Can be overridden on the command line.
#
default_port = 5100

# --------------------------------------------------------------------------------
# Miscellaneous AI model config

# Each is a Huggingface model name, auto-downloaded on first use.

# Text classification model for emotion detection.
#
# Used for dynamically auto-updating the emotion shown by the AI's avatar.
#
classification_model = "joeddav/distilbert-base-uncased-go-emotions-student"
# classification_model = "nateraw/bert-base-uncased-emotion"

# AI model that produces the high-dimensional semantic vectors, for visualization in `raven-visualizer`.
#
embedding_model = "Snowflake/snowflake-arctic-embed-l"
# embedding_model = "sentence-transformers/all-mpnet-base-v2"

# Models for the Kokoro speech synthesizer (text to speech, TTS).
#
kokoro_models = "hexgrad/Kokoro-82M"

# --------------------------------------------------------------------------------
# AI avatar

# THA3 animator models. There are currently no alternative models, this is just for specifying where to download from.
#
# Huggingface model name, auto-downloaded on first use.
#
talkinghead_models = "OktayAlpk/talking-head-anime-3"

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

    # "Intense emotion" eye-waver animation settings.
    "eye_waver_fps": 12.0,  # cel animation framerate. Special value 0.0 disables the animation.

    # animefx: anime-style effects that hover *around* the character.
    #
    # For the names of the cels for these to work, see `supported_cels` in `raven.server.modules.avatarutil`.
    # All of these effects are optional.
    #
    # - When animefx are enabled, all characters use generic fx cels ("fx_*.png") by default, so that the effects can be enabled for any character without extra effort.
    # - If you want to customize the look of the effects for a specific character, supply those cels for that character ("mycharacter_fx_*.png").
    #   Character-specific cels automatically override the generic ones.
    # - To disable a specific animefx for all characters, you can set its duration to the special value 0.0.
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

    # anger vein, cycling between two cels and fading out
    "fx_angervein_emotions": ["anger"],  # trigger emotion(s); entering any emotion listed here triggers the effect (anew each time).
    "fx_angervein_duration": 1.0,  # seconds, for fadeout (special value 0.0 = fadeout disabled)
    "fx_angervein_fps": 6.0,  # cel animation framerate

    # anime sweatdrop (large)
    "fx_sweatdrop_emotions": ["embarrassment"],
    "fx_sweatdrop_duration": 0.3,

    # anime sweatdrop(s) (small)
    "fx_smallsweatdrop_emotions": ["nervousness"],
    "fx_smallsweatdrop_duration": 0.3,

    # heart(s)
    "fx_heart_emotions": ["desire"],
    "fx_heart_duration": 0.3,

    # black cloud, frustration etc.
    "fx_blackcloud_emotions": ["annoyance", "disapproval"],
    "fx_blackcloud_duration": 1.0,
    "fx_blackcloud_fps": 6.0,

    # flowers, love etc.
    "fx_flowers_emotions": ["love"],
    "fx_flowers_duration": 1.0,
    "fx_flowers_fps": 6.0,

    # shock lines
    "fx_shock_emotions": ["disgust", "fear"],
    "fx_shock_duration": 2.0,

    # notice lines (or surprise lines)
    "fx_notice_emotions": ["surprise"],
    "fx_notice_duration": 0.25,

    # "beaming" lines (joy etc.)
    "fx_beaming_emotions": ["admiration", "amusement", "excitement", "joy"],  # TODO: approval, gratitude, pride?
    "fx_beaming_duration": 0.25,

    # question mark(s)
    "fx_question_emotions": ["confusion"],
    "fx_question_duration": 0.25,

    # exclamation mark(s)
    "fx_exclaim_emotions": ["realization"],
    "fx_exclaim_duration": 0.25,

    # postprocessor
    "postprocessor_chain": postprocessor_defaults
}

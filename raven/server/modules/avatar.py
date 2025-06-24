"""AI avatar animator.

This is the animation engine, running on top of the THA3 posing engine.

This module implements the live animation backend, and its server-side API.
For how to use that API, see `raven.server.app`.

If you want to edit THA3 expressions in a standalone GUI app, see `raven.avatar.pose_editor.app`.

If you want to test your AI character and edit postprocessor settings in a GUI app, see `raven.avatar.settings_editor.app`.
"""

__all__ = ["init_module", "is_available",
           "load", "reload", "unload",
           "load_emotion_templates",
           "load_animator_settings",
           "start", "stop",
           "start_talking", "stop_talking",
           "set_emotion",
           "set_overrides",
           "result_feed"]

import atexit
import copy
import functools
import io
import json
import logging
import math
import os
import pathlib
import random
import time
import numpy as np
import sys
import threading
import traceback
from typing import Any, Dict, List, Optional, Tuple
import uuid

from colorama import Fore, Style

from unpythonic import timer

import qoi
import PIL

import torch

from flask import Response

from ...common.hfutil import maybe_install_models
from ...common.running_average import RunningAverage

from ...common.video.postprocessor import Postprocessor
from ...common.video.upscaler import Upscaler

from ...vendor.tha3.poser.modes.load_poser import load_poser
from ...vendor.tha3.poser.poser import Poser
from ...vendor.tha3.util import torch_linear_to_srgb

from .. import config as server_config  # hf repo name for downloading THA3 models if needed

from . import avatarutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------
# Global variables

talkinghead_path = pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "..", "vendor")).expanduser().resolve()  # THA3 install location containing the "tha3" folder
emotions_path = pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "..", "avatar", "assets", "emotions")).expanduser().resolve()  # location containing the emotion template JSON files
animator_settings_path = pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "..", "avatar", "assets", "settings")).expanduser().resolve()  # location containing the default "animator.json"

module_initialized = False  # call `init_module` to initialize

# These will be set up in `init_module`
_device = None
_model = None
_poser = None  # THA3 engine instance (as returned by `load_poser`)

_avatar_instances = {}  # {instance_id0: {"animator": <Animator object>, "encoder": <Encoder object>}, ...}

# --------------------------------------------------------------------------------
# Module startup, status check, and auto-cleanup for server shutdown time.

def init_module(device: str, model: str) -> None:
    """Launch the avatar (live mode), served over HTTP.

    device: "cpu" or "cuda"
    model: one of the folder names inside "raven/vendor/tha3/models/"

           Determines the posing and postprocessing dtype.

    If something goes horribly wrong, raise `RuntimeError`.
    """
    global module_initialized
    global _device
    global _model
    global _poser

    if module_initialized:
        logger.warning("init_module: already initialized. Ignoring.")
        return

    print(f"Initializing {Fore.GREEN}{Style.BRIGHT}avatar{Style.RESET_ALL} on device '{Fore.GREEN}{Style.BRIGHT}{device}{Style.RESET_ALL}' with model '{Fore.GREEN}{Style.BRIGHT}{model}{Style.RESET_ALL}'...")

    sys.path.append(str(talkinghead_path))  # The vendored code from THA3 expects to find the `tha3` module at the top level of the module hierarchy
    print(f"THA3 is installed at '{str(talkinghead_path)}'")

    # Install the THA3 models if needed
    tha3_models_path = str(talkinghead_path / "tha3" / "models")
    maybe_install_models(hf_reponame=server_config.talkinghead_models, modelsdir=tha3_models_path)

    try:
        logger.info("init_module: loading the Talking Head Anime 3 (THA3) posing engine")
        modelsdir = str(talkinghead_path / "tha3" / "models")
        _poser = load_poser(model, device, modelsdir=modelsdir)
        _device = device
        _model = model
        module_initialized = True

    except RuntimeError as exc:
        print(f"{Fore.RED}{Style.BRIGHT}Internal server error during init of module 'avatar'.{Style.RESET_ALL} Details follow.")
        traceback.print_exc()
        logger.error(f"init_module: failed: {type(exc)}: {exc}")

        _poser = None
        _device = None
        _model = None

def is_available() -> bool:
    """Return whether this module is up and running."""
    return _poser is not None

def shutdown() -> None:
    remaining_instances = list(_avatar_instances.keys())
    for instance in remaining_instances:
        unload(instance)
atexit.register(shutdown)

# --------------------------------------------------------------------------------
# Implementations for API endpoints served by `server.py`

# TODO: the `stream` is a flask.request.file.stream; what's the type of that?
def load(stream, cel_streams: Dict) -> str:
    """Create a new avatar instance, loading a character image (512x512 RGBA PNG) from `stream`.

    The `stream` is a `flask.request.file.stream` containing the character's base image.

    `cel_streams` is a dict `{celname: flask.request.file.stream, ...}` containing the add-on cels, if any.
    If there are no add-on cels, you can pass an empty dict.

    Returns the instance ID (important; needed by all other API functions to operate on that specific instance).
    """
    if not module_initialized:
        raise RuntimeError("load: Module not initialized. Please call `init_module` before using the API.")

    try:
        instance_id = str(uuid.uuid4())
        while instance_id in _avatar_instances:  # guarantee no conflict even if UUID generation fails (very low chance)
            instance_id = str(uuid.uuid4())
        assert instance_id is not None
    except Exception as exc:
        traceback.print_exc()
        logger.error(f"load: failed: {type(exc)}: {exc}")
        raise

    encoder = None
    animator = None
    try:
        encoder = Encoder(instance_id)  # create encoder first; its output format will be set by `Animator.load_animator_settings`, which is called by `Animator.__init__`
        _avatar_instances[instance_id] = {"encoder": encoder}  # ugh, half-created
        animator = Animator(instance_id, _poser, _device)
        _avatar_instances[instance_id]["animator"] = animator  # ...there, much better.

        animator.start()
        encoder.start()

        reload(instance_id, stream, cel_streams)  # delegate; actually load the image(s)
    except Exception as exc:
        traceback.print_exc()
        logger.error(f"load: failed: {type(exc)}: {exc}")

        # Tear down anything that started before the error occurred.
        if encoder:
            try:
                encoder.exit()
            except Exception:
                pass
        if animator:
            try:
                animator.exit()
            except Exception:
                pass
        try:
            _avatar_instances.pop(instance_id)
        except KeyError:
            pass

        raise

    plural_s = "s" if len(_avatar_instances) != 1 else ""
    logger.info(f"load: created avatar instance '{instance_id}' (now have {len(_avatar_instances)} instance{plural_s})")

    return instance_id

def reload(instance_id: str, stream, cel_streams: Dict) -> None:
    """Send a new input image to an existing instance.

    The `stream` is a `flask.request.file.stream` containing the character's base image.

    `cel_streams` is a dict `{celname: flask.request.file.stream, ...}` containing the add-on cels, if any.
    If there are no add-on cels, you can pass an empty dict.
    """
    if not module_initialized:
        raise RuntimeError("reload: Module not initialized. Please call `init_module` before using the API.")

    if instance_id not in _avatar_instances:
        logger.error(f"reload: no such avatar instance '{instance_id}'")
        raise ValueError(f"reload: no such avatar instance '{instance_id}'")

    # Just in case, make a seekable in-memory copy of each `stream`.
    def _stream_to_buffer(stream):
        buffer = io.BytesIO()
        buffer.write(stream.read())
        buffer.seek(0)
        return buffer

    logger.info("reload: loading new input image from stream")
    source_image_buffer = _stream_to_buffer(stream)  # base image
    cel_buffers = {celname: _stream_to_buffer(cel_stream) for celname, cel_stream in cel_streams.items()}  # add-on cels, if any

    animator = _avatar_instances[instance_id]["animator"]
    animator.load_image(source_image_buffer, cel_buffers)

def unload(instance_id: str) -> None:
    """Unload the given instance.

    This will delete the corresponding animator and encoder instances, and cause the result feed (if any is running) to shut down.
    """
    if not module_initialized:
        raise RuntimeError("unload: Module not initialized. Please call `init_module` before using the API.")

    if instance_id not in _avatar_instances:
        logger.error(f"unload: no such avatar instance '{instance_id}'")
        raise ValueError(f"unload: no such avatar instance '{instance_id}'")

    try:
        encoder = _avatar_instances[instance_id]["encoder"]
        encoder.exit()
    except Exception:
        pass
    try:
        animator = _avatar_instances[instance_id]["animator"]
        animator.exit()
    except Exception:
        pass

    _avatar_instances.pop(instance_id)

    plural_s = "s" if len(_avatar_instances) != 1 else ""
    logger.info(f"unload: deleted avatar instance '{instance_id}' (now have {len(_avatar_instances)} instance{plural_s})")

def load_emotion_templates(instance_id: str, emotions: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None) -> None:
    """Load emotion templates. This is the API function that dispatches to the specified animator instance.

    `emotions`: `{emotion0: {"pose": {morph0: value0, ...}, "cels": {cel0: value0, ...}}, ...}`
                Optional dict of custom emotion templates.

                If not given, this loads the templates from the emotion JSON files
                in `raven/avatar/assets/emotions/`.

                If given:
                  - Each emotion NOT supplied is populated from the defaults.
                  - In each emotion that IS supplied, each morph that is NOT mentioned
                    is implicitly set to zero (due to how `apply_emotion_to_pose` works).

                For an example JSON file containing a suitable dictionary, see `raven/avatar/assets/emotions/_defaults.json`.

                For available morph names, see `posedict_keys` in `raven.server.modules.avatarutil`.
                For available cel names, see `supported_cels` in the same module.

                For some more detail, see `raven/vendor/tha3/poser/modes/pose_parameters.py`.
                "Arity 2" means `posedict_keys` has separate left/right morphs.

                If still in doubt, see the GUI panel implementations in `raven.avatar.pose_editor.app`.
    """
    if not module_initialized:
        raise RuntimeError("set_overrides: Module not initialized. Please call `init_module` before using the API.")

    if instance_id not in _avatar_instances:
        logger.error(f"set_overrides: no such avatar instance '{instance_id}'")
        raise ValueError(f"set_overrides: no such avatar instance '{instance_id}'")

    if not emotions:
        emotions = {}  # sending a blank dictionary will load server defaults

    animator = _avatar_instances[instance_id]["animator"]
    animator.load_emotion_templates(emotions)

def load_animator_settings(instance_id: str, settings: Optional[Dict[str, Any]] = None) -> None:
    """Load animator settings. This is the API function that dispatches to the specified animator instance.

    `settings`: `{setting0: value0, ...}`
                Optional dict of settings. The type and semantics of each value depends on each
                particular setting.

    For available settings, see `animator_defaults` in `raven/server/config.py`.

    Particularly for the setting `"postprocessor_chain"` (pixel-space glitch artistry),
    see `postprocessor.py`.
    """
    if not module_initialized:
        raise RuntimeError("set_overrides: Module not initialized. Please call `init_module` before using the API.")

    if instance_id not in _avatar_instances:
        logger.error(f"set_overrides: no such avatar instance '{instance_id}'")
        raise ValueError(f"set_overrides: no such avatar instance '{instance_id}'")

    if not settings:
        settings = {}  # sending a blank dictionary will load server defaults

    animator = _avatar_instances[instance_id]["animator"]
    animator.load_animator_settings(settings)

def start(instance_id: str) -> str:
    """Start/resume animation."""
    if not module_initialized:
        raise RuntimeError("start: Module not initialized. Please call `init_module` before using the API.")

    if instance_id not in _avatar_instances:
        logger.error(f"start: no such avatar instance '{instance_id}'")
        raise ValueError(f"start: no such avatar instance '{instance_id}'")

    animator = _avatar_instances[instance_id]["animator"]
    animator.animation_running = True

    logger.info("start: done")

def stop(instance_id: str) -> str:
    """Stop animation, but keep the avatar loaded for resuming later (to do that, use `start`)."""
    if not module_initialized:
        raise RuntimeError("stop: Module not initialized. Please call `init_module` before using the API.")

    if instance_id not in _avatar_instances:
        logger.error(f"stop: such avatar instance '{instance_id}'")
        raise ValueError(f"stop: no such avatar instance '{instance_id}'")

    animator = _avatar_instances[instance_id]["animator"]
    animator.animation_running = False

    logger.info("stop: done")

def start_talking(instance_id: str) -> str:
    """Start talking animation (generic, non-lipsync)."""
    if not module_initialized:
        raise RuntimeError("start_talking: Module not initialized. Please call `init_module` before using the API.")

    if instance_id not in _avatar_instances:
        logger.error(f"start_talking: no such avatar instance '{instance_id}'")
        raise ValueError(f"start_talking: no such avatar instance '{instance_id}'")

    animator = _avatar_instances[instance_id]["animator"]
    animator.is_talking = True

    logger.debug("start_talking: done")

def stop_talking(instance_id: str) -> str:
    """Stop talking animation (generic, non-lipsync)."""
    if not module_initialized:
        raise RuntimeError("stop_talking: Module not initialized. Please call `init_module` before using the API.")

    if instance_id not in _avatar_instances:
        logger.error(f"stop_talking: no such avatar instance '{instance_id}'")
        raise ValueError(f"stop_talking: no such avatar instance '{instance_id}'")

    animator = _avatar_instances[instance_id]["animator"]
    animator.is_talking = False

    logger.debug("stop_talking: done")

def set_emotion(instance_id: str, emotion: str) -> str:
    """Set the current emotion of the character."""
    if not module_initialized:
        raise RuntimeError("set_emotion: Module not initialized. Please call `init_module` before using the API.")

    if instance_id not in _avatar_instances:
        logger.error(f"set_emotion: no such avatar instance '{instance_id}'")
        raise ValueError(f"set_emotion: no such avatar instance '{instance_id}'")

    animator = _avatar_instances[instance_id]["animator"]

    if emotion not in animator.emotions:  # should exist on the animator it's being applied to
        logger.error(f"set_emotion: specified emotion '{emotion}' does not exist")
        raise ValueError(f"set_emotion: specified emotion '{emotion}' does not exist")

    logger.info(f"set_emotion: applying emotion {emotion}")
    animator.emotion = emotion

def set_overrides(instance_id: str, overrides: Dict[str, Any]) -> str:
    """Set manual overrides for morphs and/or cel blends.

    All previous overrides are replaced by the new ones.

    For a full list of animation keys you can override here, see `posedict_key_to_index`
    and `supported_cels`, both in `raven.server.modules.avatarutil`.

    The overrides remain in effect until replaced by new overrides.

    To unset all overrides, use `overrides = {}`.

    Overrides, as the name suggests, override morph and/or cel blend values
    that were computed by the animator. This is particularly useful for lipsyncing,
    but can also be used to disable certain animation features (e.g. eye wink morphs
    for characters with opaque glasses).
    """
    if not module_initialized:
        raise RuntimeError("set_overrides: Module not initialized. Please call `init_module` before using the API.")

    if instance_id not in _avatar_instances:
        logger.error(f"set_overrides: no such avatar instance '{instance_id}'")
        raise ValueError(f"set_overrides: no such avatar instance '{instance_id}'")

    logger.debug("set_overrides: applying overrides")  # too spammy as info (when lipsyncing)
    animator = _avatar_instances[instance_id]["animator"]
    animator.set_overrides(overrides)

# There are three tasks we must do each frame:
#
#   1) Render an animation frame
#   2) Encode the new animation frame for network transport
#   3) Send the animation frame over the network
#
# Instead of running serially:
#
#   [render1][encode1][send1] [render2][encode2][send2]
# ------------------------------------------------------> time
#
# we get better throughput by parallelizing and interleaving:
#
#   [render1] [render2] [render3] [render4] [render5]
#             [encode1] [encode2] [encode3] [encode4]
#                       [send1]   [send2]   [send3]
# ----------------------------------------------------> time
#
# Despite the global interpreter lock, this increases throughput, as well as improves the timing of the network send
# since the network thread only needs to care about getting the send timing right.
#
# Either there's enough waiting for I/O for the split between render and encode to make a difference, or it's the fact
# that much of the compute-heavy work in both of those is performed inside C libraries that release the GIL (Torch,
# and the PNG encoder in Pillow, respectively).
#
# This is a simplified picture. Some important details:
#
#   - At startup:
#     - The animator renders the first frame on its own.
#     - The encoder waits for the animator to publish a frame, and then starts normal operation.
#     - The network thread waits for the encoder to publish a frame, and then starts normal operation.
#   - In normal operation (after startup):
#     - The animator waits until the encoder has consumed the previous published frame. Then it proceeds to render and publish a new frame.
#       - This communication is handled through the flag `animator.new_frame_available`.
#     - The network thread does its own thing on a regular schedule, based on the desired target FPS.
#       - However, the network thread publishes metadata on which frame is the latest that has been sent over the network at least once.
#         This is stored as an `id` (i.e. memory address) in `encoder.latest_frame_sent` on the avatar instance's encoder.
#       - If the target FPS is too high for the animator and/or encoder to keep up with, the network thread re-sends
#         the latest frame published by the encoder as many times as necessary, to keep the network output at the target FPS
#         regardless of render/encode speed. This handles the case of hardware slower than the target FPS.
#       - On localhost, the network send is very fast, under 0.15 ms.
#     - The encoder uses the metadata to wait until the latest encoded frame has been sent at least once before publishing a new frame.
#       This ensures that no more frames are generated than are actually sent, and syncs also the animator (because the animator is
#       rate-limited by the encoder consuming its frames). This handles the case of hardware faster than the target FPS.
#     - When the animator and encoder are fast enough to keep up with the target FPS, generally when frame N is being sent,
#       frame N+1 is being encoded (or is already encoded, and waiting for frame N to be sent), and frame N+2 is being rendered.
#
def result_feed(instance_id: str) -> Response:
    """Return a Flask `Response` that repeatedly yields the current image as an image file in the configured format."""
    if not module_initialized:
        raise RuntimeError("result_feed: Module not initialized. Please call `init_module` before using the API.")

    if instance_id not in _avatar_instances:
        logger.error(f"result_feed: no such avatar instance '{instance_id}'")
        raise ValueError(f"result_feed: no such avatar instance '{instance_id}'")

    def generate():
        last_frame_send_complete_time = None
        last_report_time = None
        send_duration_sec = 0.0
        send_duration_statistics = RunningAverage()

        while True:
            if instance_id not in _avatar_instances:
                logger.info(f"result_feed.generate (avatar instance '{instance_id}'): Instance has been deleted, shutting down the result feed.")
                return  # Instance has been deleted (by a call to `unload`), so we're done. Shut down the stream.
            animator = _avatar_instances[instance_id]["animator"]
            encoder = _avatar_instances[instance_id]["encoder"]

            # Send the latest available animation frame.
            # Important: grab reference to `current_frame` only once, since it will be atomically updated without a lock.
            current_frame = encoder.current_frame
            if current_frame is not None:
                # How often should we send?
                #  - Excessive spamming can DoS the SillyTavern GUI, so there needs to be a rate limit.
                #  - OTOH, we must constantly send something, or the GUI will lock up waiting.
                # Therefore, send at a target FPS that yields a nice-looking animation.
                frame_duration_target_sec = 1 / animator.target_fps
                if last_frame_send_complete_time is not None:
                    time_now = time.time_ns()
                    this_frame_elapsed_sec = (time_now - last_frame_send_complete_time) / 10**9
                    # The 2* is a fudge factor. It doesn't matter if the frame is a bit too early, but we don't want it to be late.
                    time_until_frame_deadline = frame_duration_target_sec - this_frame_elapsed_sec - 2 * send_duration_sec
                else:
                    time_until_frame_deadline = 0.0  # nothing rendered yet

                if time_until_frame_deadline <= 0.0:
                    time_now = time.time_ns()
                    image_format, image_data = current_frame
                    content_type_header = f"Content-Type: image/{image_format.lower()}\r\n".encode()
                    content_length_header = f"Content-Length: {len(image_data)}\r\n".encode()
                    yield (b"--frame\r\n" +
                           content_type_header +
                           content_length_header +
                           b'Pragma-directive: no-cache\r\n'
                           b'Cache-directive: no-cache\r\n'
                           b'Cache-control: no-cache\r\n'
                           b'Pragma: no-cache\r\n'
                           b'Expires: 0\r\n'
                           b"\r\n" +  # A second successive CRLF sequence signals the end of the headers for this frame.
                           image_data +
                           b"\r\n")
                    encoder.latest_frame_sent = id(current_frame)  # atomic update, no need for lock
                    send_duration_sec = (time.time_ns() - time_now) / 10**9  # about 0.12 ms on localhost (compress_level=1 or 6, doesn't matter)
                    # print(f"send {send_duration_sec:0.6g}s")  # DEBUG

                    # Update the FPS counter, measuring the time between network sends.
                    time_now = time.time_ns()
                    if last_frame_send_complete_time is not None:
                        this_frame_elapsed_sec = (time_now - last_frame_send_complete_time) / 10**9
                        send_duration_statistics.add_datapoint(this_frame_elapsed_sec)
                    last_frame_send_complete_time = time_now
                else:
                    time.sleep(time_until_frame_deadline)

                # Log the FPS counter in 5-second intervals.
                time_now = time.time_ns()
                if animator.animation_running and (last_report_time is None or time_now - last_report_time > 5e9):
                    avg_send_sec = send_duration_statistics.average()
                    msec = round(1000 * avg_send_sec, 1)
                    target_msec = round(1000 * frame_duration_target_sec, 1)
                    fps = round(1 / avg_send_sec, 1) if avg_send_sec > 0.0 else 0.0
                    logger.info(f"output: {msec:.1f}ms [{fps:.1f} FPS]; target {target_msec:.1f}ms [{animator.target_fps:.1f} FPS]")
                    last_report_time = time_now

            else:  # first frame not yet available
                time.sleep(0.1)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

# --------------------------------------------------------------------------------
# Internal stuff

class Animator:
    """uWu Waifu"""

    def __init__(self, instance_id: str, poser: Poser, device: torch.device):
        self.instance_id = instance_id
        self.poser = poser
        self.device = device

        self.target_size = poser.get_image_size()
        self.upscaler = None
        self.upscale_factor = None  # Nice to know; but also so that we can re-instantiate the upscaler only when the settings actually change.
        self.upscale_preset = None
        self.upscale_quality = None
        self.postprocessor = Postprocessor(device,
                                           dtype=self.poser.dtype)  # dtype must match `output_image` in `Animator.render_animation_frame`
        self.render_duration_statistics = RunningAverage()  # used for FPS compensation in animation routines
        self.animator_thread = None

        self.source_image: Optional[torch.tensor] = None
        self.result_image: Optional[np.array] = None
        self.torch_cels: Dict[str, torch.tensor] = {}  # loaded in `load_image`
        self.new_frame_available = False
        self.last_report_time = None
        self.output_lock = threading.Lock()  # protect from concurrent access to `result_image` and the `new_frame_available` flag.

        self.animation_running = False  # used in initial bootup state, while loading a new image, and while explicitly paused

        self.reset_animation_state()
        self.load_emotion_templates()
        self.load_animator_settings()

    def reset_animation_state(self):
        """Reset character state trackers for all animation drivers."""
        self.current_pose = None
        self.current_celstack = None

        self.emotion = "neutral"
        self.last_emotion = None
        self.last_emotion_change_timestamp = None

        self.last_sway_target_timestamp = None
        self.last_sway_target_pose = None
        self.last_microsway_timestamp = None
        self.sway_interval = None

        self.last_blink_timestamp = None
        self.blink_interval = None

        self.is_talking = False  # generic non-lipsync talking animation (random mouth)
        self.was_talking = False  # state at previous frame
        self.last_talking_timestamp = None
        self.last_talking_target_value = None

        t0 = time.time_ns()
        self.breathing_epoch = t0
        self.waver_epoch = t0

        self.morph_and_cel_overrides = {}

    # --------------------------------------------------------------------------------
    # Management

    def start(self) -> None:
        """Start the animation thread."""
        logger.info(f"Animator.start (avatar instance '{self.instance_id}'): Animator is starting.")
        self._terminated = False
        def animator_update():
            while not self._terminated:
                try:
                    self.render_animation_frame()
                except Exception as exc:
                    logger.error(exc)
                    traceback.print_exc()
                    raise  # let the animator stop so we won't spam the log
                time.sleep(0.01)  # rate-limit the renderer to 100 FPS maximum (this could be adjusted later)
        self.animator_thread = threading.Thread(target=animator_update, daemon=True)
        self.animator_thread.start()
        logger.info(f"Animator.start (avatar instance '{self.instance_id}'): Animator startup complete.")

    def exit(self) -> None:
        """Terminate the animation thread.

        Called automatically when the process exits.
        """
        logger.info(f"Animator.exit (avatar instance '{self.instance_id}'): Animator is shutting down.")
        self._terminated = True
        if self.animator_thread is not None:
            self.animator_thread.join()
        self.animator_thread = None
        logger.info(f"Animator.exit (avatar instance '{self.instance_id}'): Animator shutdown complete.")

    def load_emotion_templates(self, emotions: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None) -> None:
        """Load emotion templates.

        `emotions`: `{emotion0: {"pose": {morph0: value0, ...}, "cels": {cel0: value0, ...}}, ...}`
                    Optional dict of custom emotion templates.

                    If not given, this loads the templates from the emotion JSON files
                    in `raven/avatar/assets/emotions/`.

                    If given:
                      - Each emotion NOT supplied is populated from the defaults.
                      - In each emotion that IS supplied, each morph that is NOT mentioned
                        is implicitly set to zero (due to how `apply_emotion_to_pose` works).

                    For an example JSON file containing a suitable dictionary, see `raven/avatar/assets/emotions/_defaults.json`.

                    For available morph names, see `posedict_keys` in `raven.server.modules.avatarutil`.
                    For available cel names, see `supported_cels` in the same module.

                    For some more detail, see `raven/vendor/tha3/poser/modes/pose_parameters.py`.
                    "Arity 2" means `posedict_keys` has separate left/right morphs.

                    If still in doubt, see the GUI panel implementations in `editor.py`.
        """
        # Load defaults as a base
        self.emotions, self.emotion_names = avatarutil.load_emotion_presets(emotions_path)

        # Then override defaults, and add any new custom emotions
        if emotions is not None:
            logger.info(f"load_emotion_templates: loading user-specified templates for emotions {list(sorted(emotions.keys()))}")

            self.emotions.update(emotions)

            emotion_names = set(self.emotion_names)
            emotion_names.update(emotions.keys())
            self.emotion_names = list(sorted(emotion_names))
        else:
            logger.info("load_emotion_templates: loaded default emotion templates")

    def load_animator_settings(self, settings: Optional[Dict[str, Any]] = None) -> None:
        """Load animator settings.

        `settings`: `{setting0: value0, ...}`
                    Optional dict of settings. The type and semantics of each value depends on each
                    particular setting.

        For available settings, see `animator_defaults` in `raven/server/config.py`.

        Particularly for the setting `"postprocessor_chain"` (pixel-space glitch artistry),
        see `postprocessor.py`.
        """
        if settings is None:
            settings = {}

        logger.info(f"load_animator_settings: user settings: {settings}")

        # Load server-side settings
        try:
            with open(animator_settings_path / "animator.json", "r") as json_file:
                server_settings = json.load(json_file)
        except Exception as exc:
            logger.info(f"load_animator_settings: skipping server settings, reason: {exc}")
            server_settings = {}

        # Let's define some helpers:
        def drop_unrecognized(settings: Dict[str, Any], context: str) -> None:  # DANGER: MUTATING FUNCTION
            unknown_fields = [field for field in settings if field not in server_config.animator_defaults]
            if unknown_fields:
                logger.warning(f"load_animator_settings: in {context}: this server did not recognize the following settings, ignoring them: {unknown_fields}")
            for field in unknown_fields:
                settings.pop(field)
            assert all(field in server_config.animator_defaults for field in settings)  # contract: only known settings remaining

        def typecheck(settings: Dict[str, Any], context: str) -> None:  # DANGER: MUTATING FUNCTION
            for field, default_value in server_config.animator_defaults.items():
                type_match = (int, float) if isinstance(default_value, (int, float)) else type(default_value)
                if field in settings and not isinstance(settings[field], type_match):
                    logger.warning(f"load_animator_settings: in {context}: incorrect type for '{field}': got {type(settings[field])} with value '{settings[field]}', expected {type_match}")
                    settings.pop(field)  # (safe; this is not the collection we are iterating over)

        def aggregate(settings: Dict[str, Any], fallback_settings: Dict[str, Any], fallback_context: str) -> None:  # DANGER: MUTATING FUNCTION
            for field, default_value in fallback_settings.items():
                if field not in settings:
                    logger.info(f"load_animator_settings: filling in '{field}' from {fallback_context}")
                    settings[field] = default_value

        # Now our settings loading strategy is as simple as:
        settings = dict(settings)  # copy to avoid modifying the original, since we'll pop some stuff.
        if settings:
            drop_unrecognized(settings, context="user settings")
            typecheck(settings, context="user settings")
        if server_settings:
            drop_unrecognized(server_settings, context="server settings")
            typecheck(server_settings, context="server settings")
        # both `settings` and `server_settings` are fully valid at this point
        aggregate(settings, fallback_settings=server_settings, fallback_context="server settings")  # first fill in from server-side settings
        aggregate(settings, fallback_settings=server_config.animator_defaults, fallback_context="built-in defaults")  # then fill in from hardcoded defaults

        logger.info(f"load_animator_settings: final settings (filled in as necessary): {settings}")

        # Some settings must be applied explicitly.
        logger.debug(f"load_animator_settings: Setting new target FPS = {settings['target_fps']}")
        self.target_fps = settings.pop("target_fps")  # controls the network send rate.

        logger.debug(f"load_animator_settings: Setting output format = {settings['format']}")
        encoder = _avatar_instances[self.instance_id]["encoder"]
        encoder.output_format = settings.pop("format")

        logger.debug("load_animator_settings: Sending new effect chain to postprocessor")
        self.postprocessor.chain = settings.pop("postprocessor_chain")  # ...and that's where the postprocessor reads its filter settings from.

        if settings["upscale"] != 1.0:
            # Avoid unnecessary hiccup (if the settings are changed while the animator is running) by re-instantiating the upscaler only when we have to.
            if settings["upscale"] == self.upscale_factor and settings["upscale_preset"] == self.upscale_preset and settings["upscale_quality"] == self.upscale_quality:
                logger.debug(f"load_animator_settings: Upscale factor {settings['upscale']}x, preset {settings['upscale_preset']}, quality {settings['upscale_quality']}; reusing existing upscaler.")
                # Can only happen when this is the second or later settings load in the same server session, so `self.target_size` has already been initialized.
            else:
                logger.debug(f"load_animator_settings: Upscale factor {settings['upscale']}x, preset {settings['upscale_preset']}, quality {settings['upscale_quality']}; configuring upscaler.")
                self.target_size = int(settings["upscale"] * self.poser.get_image_size())
                self.upscaler = Upscaler(device=self.device,
                                         dtype=self.poser.dtype,
                                         upscaled_width=self.target_size,
                                         upscaled_height=self.target_size,
                                         preset=settings["upscale_preset"],
                                         quality=settings["upscale_quality"])
            self.upscale_factor = settings["upscale"]
            self.upscale_preset = settings["upscale_preset"]
            self.upscale_quality = settings["upscale_quality"]
        else:
            logger.debug(f"load_animator_settings: Upscale factor {settings['upscale']}x; switching upscaler off.")
            self.target_size = self.poser.get_image_size()
            self.upscaler = None
            self.upscale_factor = None
            self.upscale_preset = None
            self.upscale_quality = None

        # The rest of the settings we can just store in an attribute, and let the animation drivers read them from there.
        self._settings = settings

    def load_image(self, filelike, cel_filelikes: Dict) -> None:
        """Load the image file `filelike`, and replace the current character with it.

        `filelike`: str or pathlib.Path to read a file; or a binary stream such as BytesIO to read that.
        """
        _load = functools.partial(avatarutil.torch_load_rgba_image,
                                  target_w=self.poser.get_image_size(),
                                  target_h=self.poser.get_image_size(),
                                  device=self.device,
                                  dtype=self.poser.dtype)  # load to GPU in linear RGB
        old_animation_running = self.animation_running  # TODO: ugh, would be better to use a lock or something. But one animator instance should only be used from one thread anyway.
        self.animation_running = False
        try:
            plural_s = "s" if len(cel_filelikes) != 1 else ""
            cels_str = f" Received cels: {list(cel_filelikes.keys())}." if len(cel_filelikes) else ""
            logger.info(f"load_image: Loading character image with {len(cel_filelikes)} add-on cel{plural_s}.{cels_str}")
            self.source_image = _load(filelike)  # base image
            self.torch_cels = {celname: _load(cel_filelike) for celname, cel_filelike in cel_filelikes.items()}  # add-on cels, if any
        except Exception as exc:
            self.source_image = None
            self.torch_cels = {}
            print(f"{Fore.RED}{Style.BRIGHT}ERROR{Style.RESET_ALL} (details below)")
            traceback.print_exc()
            logger.error(f"load_image: {type(exc)}: {exc}")
        finally:
            self.animation_running = old_animation_running

    # --------------------------------------------------------------------------------
    # Animation drivers

    def apply_emotion_to_pose(self, emotion_posedict: Dict[str, float], pose: List[float]) -> List[float]:
        """Copy all morphs except breathing from `emotion_posedict` to `pose`.

        If a morph does not exist in `emotion_posedict`, its value is copied from the original `pose`.

        Return the modified pose.
        """
        new_pose = list(pose)  # copy
        for idx, key in enumerate(avatarutil.posedict_keys):
            if key in emotion_posedict and key != "breathing_index":
                new_pose[idx] = emotion_posedict[key]
        return new_pose

    def animate_blinking(self, pose: List[float]) -> List[float]:
        """Eye blinking animation driver.

        Relevant `self._settings` keys:

        `"blink_interval_min"`: float, seconds, lower limit for random minimum time until next blink is allowed.
        `"blink_interval_max"`: float, seconds, upper limit for random minimum time until next blink is allowed.
        `"blink_probability"`: float, at each frame at a reference of 25 FPS. FPS-corrected automatically.
        `"blink_confusion_duration"`: float, seconds, upon entering "confusion" emotion, during which blinking
                                      quickly in succession is allowed.

        Return the modified pose.
        """
        # Compute FPS-corrected blink probability
        CALIBRATION_FPS = 25
        p_orig = self._settings["blink_probability"]  # blink probability per frame at CALIBRATION_FPS
        avg_render_sec = self.render_duration_statistics.average()
        if avg_render_sec > 0:
            avg_render_fps = 1 / avg_render_sec
            # Even if render completes faster, the avatar output is rate-limited to `self.target_fps` at most.
            avg_render_fps = min(avg_render_fps, self.target_fps)
        else:  # No statistics available yet; let's assume we're running at `self.target_fps`.
            avg_render_fps = self.target_fps
        # We give an independent trial for each of `n` "normalized frames" elapsed at `CALIBRATION_FPS` during one actual frame at `avg_render_fps`.
        # Note direction: rendering faster (higher FPS) means less likely to blink per frame, to obtain the same blink density per unit of wall time.
        n = CALIBRATION_FPS / avg_render_fps
        # If at least one of the normalized frames wants to blink, then the actual frame should blink.
        # Doesn't matter that `n` isn't an integer, since the power function over the reals is continuous and we just want a reasonable scaling here.
        p_scaled = 1.0 - (1.0 - p_orig)**n
        should_blink = (random.random() <= p_scaled)

        debug_fps = round(avg_render_fps, 1)
        logger.debug(f"animate_blinking: p @ {CALIBRATION_FPS} FPS = {p_orig}, scaled p @ {debug_fps:.1f} FPS = {p_scaled:0.6g}")

        # Prevent blinking too fast in succession.
        time_now = time.time_ns()
        if self.blink_interval is not None:
            # ...except when the "confusion" emotion has been entered recently.
            seconds_since_last_emotion_change = (time_now - self.last_emotion_change_timestamp) / 10**9
            if self.emotion == "confusion" and seconds_since_last_emotion_change < self._settings["blink_confusion_duration"]:
                pass
            else:
                seconds_since_last_blink = (time_now - self.last_blink_timestamp) / 10**9
                if seconds_since_last_blink < self.blink_interval:
                    should_blink = False

        if not should_blink:
            return pose

        # If there should be a blink, set the wink morphs to 1.
        new_pose = list(pose)  # copy
        for morph_name in ["eye_wink_left_index", "eye_wink_right_index"]:
            idx = avatarutil.posedict_key_to_index[morph_name]
            new_pose[idx] = 1.0

        # Typical for humans is 12...20 times per minute, i.e. 5...3 seconds interval.
        self.last_blink_timestamp = time_now
        self.blink_interval = random.uniform(self._settings["blink_interval_min"],
                                             self._settings["blink_interval_max"])  # seconds; duration of this blink before the next one can begin

        return new_pose

    def animate_talking(self, pose: List[float], target_pose: List[float]) -> List[float]:
        """Talking animation driver.

        Relevant `self._settings` keys:

        `"talking_fps"`: float, how often to re-randomize mouth during talking animation.
                         Early 2000s anime used ~12 FPS as the fastest actual framerate of
                         new cels (not counting camera panning effects and such).
        `"talking_morph"`: str, see `posedict_keys` for available values.
                           Which morph to use for opening and closing the mouth during talking.
                           Any other morphs in the mouth-open group are set to zero while
                           talking is in progress.

        Works by randomizing the mouth-open state in regular intervals.

        When talking ends, the mouth immediately snaps to its position in the target pose
        (to avoid a slow, unnatural closing, since most expressions have the mouth closed).

        Return the modified pose.
        """
        MOUTH_OPEN_MORPHS = ["mouth_aaa_index", "mouth_iii_index", "mouth_uuu_index", "mouth_eee_index", "mouth_ooo_index", "mouth_delta"]
        talking_morph = self._settings["talking_morph"]

        if not self.is_talking:
            try:
                if self.was_talking:  # when talking ends, snap mouth to target immediately
                    new_pose = list(pose)  # copy
                    for key in MOUTH_OPEN_MORPHS:
                        idx = avatarutil.posedict_key_to_index[key]
                        new_pose[idx] = target_pose[idx]
                    return new_pose
                return pose  # most common case: do nothing (not talking, and wasn't talking during previous frame)
            finally:  # reset state *after* processing
                self.last_talking_target_value = None
                self.last_talking_timestamp = None
                self.was_talking = False
        assert self.is_talking

        # With 25 FPS (or faster) output, randomizing the mouth every frame looks too fast.
        # Determine whether enough wall time has passed to randomize a new mouth position.
        TARGET_SEC = 1 / self._settings["talking_fps"]  # rate of "actual new cels" in talking animation
        time_now = time.time_ns()
        update_mouth = False
        if self.last_talking_timestamp is None:
            update_mouth = True
        else:
            time_elapsed_sec = (time_now - self.last_talking_timestamp) / 10**9
            if time_elapsed_sec >= TARGET_SEC:
                update_mouth = True

        # Apply the mouth open morph
        new_pose = list(pose)  # copy
        idx = avatarutil.posedict_key_to_index[talking_morph]
        if self.last_talking_target_value is None or update_mouth:
            # Randomize new mouth position
            x = pose[idx]
            x = abs(1.0 - x) + random.uniform(-2.0, 2.0)
            x = max(0.0, min(x, 1.0))  # clamp (not the manga studio)
            self.last_talking_target_value = x
            self.last_talking_timestamp = time_now
        else:
            # Keep the mouth at its latest randomized position (this overrides the interpolator that would pull the mouth toward the target emotion pose)
            x = self.last_talking_target_value
        new_pose[idx] = x

        # Zero out other morphs that affect mouth open/closed state.
        for key in MOUTH_OPEN_MORPHS:
            if key == talking_morph:
                continue
            idx = avatarutil.posedict_key_to_index[key]
            new_pose[idx] = 0.0

        self.was_talking = True
        return new_pose

    def set_overrides(self, overrides: Dict[str, float]) -> None:
        """Set manual overrides for morphs and/or cel blends.

        All previous overrides are replaced by the new ones.

        For a full list of animation keys you can override here, see `posedict_key_to_index`
        and `supported_cels`, both in `raven.server.modules.avatarutil`.

        The overrides remain in effect until replaced by new overrides.

        To unset all overrides, use `data = {}`.

        Overrides, as the name suggests, override morph and/or cel blend values
        that were computed by the animator. This is particularly useful for lipsyncing,
        but can also be used to disable certain animation features (e.g. eye wink morphs
        for characters with opaque glasses).
        """
        logger.debug(f"set_overrides: got data {overrides}")  # too spammy as info (when lipsyncing)
        # Validate
        for key in overrides:
            if key not in avatarutil.posedict_key_to_index and key not in avatarutil.supported_cels:
                logger.error(f"set_overrides: unknown animation key '{key}', rejecting overrides")
                raise ValueError(f"Unknown animation key '{key}'; see `raven.server.modules.avatarutil` for available morph and cel blend keys.")
        # Save
        logger.debug("set_overrides: data is valid, applying.")
        self.morph_and_cel_overrides = overrides  # atomic replace

    def apply_overrides(self, pose: List[float], celstack: List[Tuple[str, float]]) -> (List[float], List[Tuple[str, float]]):
        """Apply any manual overrides currently in effect.

        This is actual the animation driver, called by `render_animation_frame`.

        Returns `(new_pose, new_celstack)`, without modifying the originals.
        """
        new_pose = list(pose)  # copy
        new_celstack = copy.copy(celstack)
        overrides = self.morph_and_cel_overrides  # get ref so it doesn't matter if it's replaced while we're rendering
        for key, value in overrides.items():
            if key in avatarutil.posedict_key_to_index:
                idx = avatarutil.posedict_key_to_index[key]
                new_pose[idx] = value
            else:  # key in avatarutil.supported_cels:
                idx = avatarutil.get_cel_index_in_stack(key, new_celstack)
                new_celstack[idx] = (key, value)
        return new_pose, new_celstack

    def compute_sway_target_pose(self, original_target_pose: List[float]) -> List[float]:
        """History-free sway animation driver.

        `original_target_pose`: emotion pose to modify with a randomized sway target

        Relevant `self._settings` keys:

        `"sway_morphs"`: List[str], which morphs can sway. By default, this is all geometric transformations,
                         but disabling some can be useful for some characters (such as robots).
                         For available values, see `posedict_keys`.
        `"sway_interval_min"`: float, seconds, lower limit for random time interval until randomizing new sway pose.
        `"sway_interval_max"`: float, seconds, upper limit for random time interval until randomizing new sway pose.
                               Note the limits are ignored when `original_target_pose` changes (then immediately refreshing
                               the sway pose), because an emotion pose may affect the geometric transformations, too.
        `"sway_macro_strength"`: float, [0, 1]. In sway pose, max abs deviation from emotion pose target morph value
                                 for each sway morph, but also max deviation from center. The `original_target_pose`
                                 itself may use higher values; in such cases, sway will only occur toward the center.
                                 See the source code of this function for the exact details.
        `"sway_micro_strength"`: float, [0, 1]. Max abs random noise to sway target pose, added each frame, to make
                                 the animation look less robotic. No limiting other than a clamp of final pose to [-1, 1].

        The sway target pose is randomized again when necessary; this takes care of caching internally.

        Return the modified pose.
        """
        # We just modify the target pose, and let the ODE integrator (`interpolate`) do the actual animation.
        # - This way we don't need to track start state, progress, etc.
        # - This also makes the animation nonlinear automatically: a saturating exponential trajectory toward the target.
        #   - If we want a smooth start toward a target pose/morph, we can e.g. save the timestamp when the animation began, and then ramp the rate of change,
        #     beginning at zero and (some time later, as measured from the timestamp) ending at the original, non-ramped value. The ODE itself takes care of
        #     slowing down when we approach the target state.

        # As documented in the original THA tech reports, on the pose axes, zero is centered, and 1.0 = 15 degrees.
        random_max = self._settings["sway_macro_strength"]  # max sway magnitude from center position of each morph
        noise_max = self._settings["sway_micro_strength"]  # amount of dynamic noise (re-generated every frame), added on top of the sway target, no clamping except to [-1, 1]
        SWAYPARTS = self._settings["sway_morphs"]  # some characters might not sway on all axes (e.g. a robot)

        def macrosway() -> List[float]:  # this handles caching and everything
            time_now = time.time_ns()
            should_pick_new_sway_target = True
            if self.emotion == self.last_emotion:
                if self.sway_interval is not None:  # have we created a swayed pose at least once?
                    seconds_since_last_sway_target = (time_now - self.last_sway_target_timestamp) / 10**9
                    if seconds_since_last_sway_target < self.sway_interval:
                        should_pick_new_sway_target = False
            # else, emotion has changed, invalidating the old sway target, because it is based on the old emotion (since emotions may affect the pose too).

            if not should_pick_new_sway_target:
                if self.last_sway_target_pose is not None:  # When keeping the same sway target, return the cached sway pose if we have one.
                    return self.last_sway_target_pose
                else:  # Should not happen, but let's be robust.
                    return original_target_pose

            new_target_pose = list(original_target_pose)  # copy
            for key in SWAYPARTS:
                idx = avatarutil.posedict_key_to_index[key]
                target_value = original_target_pose[idx]

                # Determine the random range so that the swayed target always stays within `[-random_max, random_max]`, regardless of `target_value`.
                # TODO: This is a simple zeroth-order solution that just cuts the random range.
                #       Would be nicer to *gradually* decrease the available random range on the "outside" as the target value gets further from the origin.
                random_upper = max(0, random_max - target_value)  # e.g. if target_value = 0.2, then random_upper = 0.4  => max possible = 0.6 = random_max
                random_lower = min(0, -random_max - target_value)  # e.g. if target_value = -0.2, then random_lower = -0.4  => min possible = -0.6 = -random_max
                random_value = random.uniform(random_lower, random_upper)

                new_target_pose[idx] = target_value + random_value

            self.last_sway_target_pose = new_target_pose
            self.last_sway_target_timestamp = time_now
            self.sway_interval = random.uniform(self._settings["sway_interval_min"],
                                                self._settings["sway_interval_max"])  # seconds; duration of this sway target before randomizing new one
            return new_target_pose

        # Add dynamic noise (re-generated at 25 FPS) to the target to make the animation look less robotic, especially once we are near the target pose.
        def add_microsway() -> None:  # DANGER: MUTATING FUNCTION
            CALIBRATION_FPS = 25  # FPS at which randomizing a new microsway target looks good
            time_now = time.time_ns()
            should_microsway = True
            if self.last_microsway_timestamp is not None:
                seconds_since_last_microsway = (time_now - self.last_microsway_timestamp) / 10**9
                if seconds_since_last_microsway < 1 / CALIBRATION_FPS:
                    should_microsway = False

            if should_microsway:
                for key in SWAYPARTS:
                    idx = avatarutil.posedict_key_to_index[key]
                    x = new_target_pose[idx] + random.uniform(-noise_max, noise_max)
                    x = max(-1.0, min(x, 1.0))
                    new_target_pose[idx] = x
                self.last_microsway_timestamp = time_now

        new_target_pose = macrosway()
        add_microsway()
        return new_target_pose

    def animate_breathing(self, pose: List[float]) -> List[float]:
        """Breathing animation driver.

        Relevant `self._settings` keys:

        `"breathing_cycle_duration"`: seconds. Duration of one full breathing cycle.

        Returns the modified pose.
        """
        breathing_cycle_duration = self._settings["breathing_cycle_duration"]  # seconds

        time_now = time.time_ns()
        t = (time_now - self.breathing_epoch) / 10**9  # seconds since breathing-epoch
        cycle_pos = t / breathing_cycle_duration  # number of cycles since breathing-epoch
        if cycle_pos > 1.0:  # prevent loss of accuracy in long sessions
            self.breathing_epoch = time_now  # TODO: be more accurate here, should sync to a whole cycle
        cycle_pos = cycle_pos - float(int(cycle_pos))  # fractional part

        new_pose = list(pose)  # copy
        idx = avatarutil.posedict_key_to_index["breathing_index"]
        new_pose[idx] = math.sin(cycle_pos * math.pi)**2  # 0 ... 1 ... 0, smoothly, with slow start and end, fast middle
        return new_pose

    def animate_eye_waver(self, strength: float, celstack: List[Tuple[str, float]]) -> Tuple[str, float]:
        """Eye-waver (anime style "intense emotion") animation driver.

        Returns the modified celstack.
        """
        WAVER_FPS = 6  # as seen in every anime ever
        waver_cycle_duration = 1 / WAVER_FPS

        time_now = time.time_ns()
        t = (time_now - self.waver_epoch) / 10**9
        cycle_pos = t / waver_cycle_duration
        if cycle_pos > 1.0:
            self.waver_epoch = time_now
        cycle_pos = cycle_pos - float(int(cycle_pos))

        if cycle_pos < 0.5:
            active_morph = "waver1"
            inactive_morph = "waver2"
        else:
            active_morph = "waver2"
            inactive_morph = "waver1"

        active_idx = avatarutil.get_cel_index_in_stack(active_morph, celstack)
        inactive_idx = avatarutil.get_cel_index_in_stack(inactive_morph, celstack)

        new_celstack = copy.copy(celstack)
        if active_idx != -1:  # found?
            new_celstack[active_idx] = (active_morph, strength)
        if inactive_idx != -1:
            new_celstack[inactive_idx] = (inactive_morph, 0.0)

        # print(active_morph, strength, active_idx, inactive_idx)  # DEBUG

        return new_celstack

    def interpolate(self, current: List[float], target: List[float]) -> List[float]:
        """Interpolate a list of floats from `current` toward `target`.

        Relevant `self._settings` keys:

        `"pose_interpolator_step"`: [0, 1]; how far toward `target` to interpolate in one frame,
                                            assuming a reference of 25 FPS. This is FPS-corrected automatically.
                                            0 means just keep `current`, 1 means immediately replace with `target`.

        This is a history-free rate-based formulation, which needs only the current and target vectors, and
        the step size; there is no need to keep track of e.g. the initial values or the progress along the trajectory.

        Note that looping back the output as `current`, while keeping `target` constant, causes the vector
        to approach `target` on a saturating trajectory. This is because `step` is the fraction of the *current*
        difference between `current` and `target`, which obviously becomes smaller after each repeat.

        This is a feature, not a bug!
        """
        # The `step` parameter is calibrated against animation at 25 FPS, so we must scale it appropriately, taking
        # into account the actual FPS.
        #
        # How to do this requires some explanation. Numericist hat on. Let's do a quick back-of-the-envelope calculation.
        # This interpolator is essentially a solver for the first-order ODE:
        #
        #   u' = f(u, t)
        #
        # Consider the most common case, where the target remains constant over several animation frames.
        # Furthermore, consider just one element of the vector (they all behave similarly and independently).
        # Then our ODE is Newton's law of cooling:
        #
        #   u' = -β [u - u∞]
        #
        # where `u = u(t)` is the temperature, `u∞` is the constant temperature of the external environment,
        # and `β > 0` is a material-dependent cooling coefficient.
        #
        # But instead of numerical simulation at a constant timestep size, as would be typical in computational science,
        # we instead read off points off the analytical solution curve. The `step` parameter is *not* the timestep size;
        # instead, it controls the relative distance along the *u* axis that should be covered in one simulation step,
        # so it is actually related to the cooling coefficient β.
        #
        # (How exactly: write the left-hand side as `[unew - uold] / Δt + O([Δt]²)`, drop the error term, and decide
        #  whether to use `uold` (forward Euler) or `unew` (backward Euler) as `u` on the right-hand side. Then compare
        #  to our update formula. But those details don't matter here.)
        #
        # To match the notation in the rest of this code, let us denote the temperature (actually current value) as `x`
        # (instead of `u`). And to keep notation shorter, let `β := step` (although it's not exactly the `β` of the
        # continuous-in-time case above).
        #
        # To scale the animation speed linearly with regard to FPS, we must invert the relation between simulation step
        # number `n` and the solution value `x`. For an initial value `x0`, a constant target value `x∞`, and constant
        # step `β ∈ (0, 1]`, the current interpolator produces the sequence:
        #
        #   x1 = x0 + β [x∞ - x0] = [1 - β] x0 + β x∞
        #   x2 = x1 + β [x∞ - x1] = [1 - β] x1 + β x∞
        #   x3 = x2 + β [x∞ - x2] = [1 - β] x2 + β x∞
        #   ...
        #
        # Note that with exact arithmetic, if `β < 1`, the final value is only reached in the limit `n → ∞`.
        # For floating point, this is not the case. Eventually the increment becomes small enough that when
        # it is added, nothing happens. After sufficiently many steps, in practice `x` will stop just slightly
        # short of `x∞` (on the side it approached the target from).
        #
        # (For performance reasons, when approaching zero, one may need to beware of denormals, because those
        #  are usually implemented in (slow!) software on modern CPUs. So especially if the target is zero,
        #  it is useful to have some very small cutoff (inside the normal floating-point range) after which
        #  we make `x` instantly jump to the target value.)
        #
        # Inserting the definition of `x1` to the formula for `x2`, we can express `x2` in terms of `x0` and `x∞`:
        #
        #   x2 = [1 - β] ([1 - β] x0 + β x∞) + β x∞
        #      = [1 - β]² x0 + [1 - β] β x∞ + β x∞
        #      = [1 - β]² x0 + [[1 - β] + 1] β x∞
        #
        # Then inserting this to the formula for `x3`:
        #
        #   x3 = [1 - β] ([1 - β]² x0 + [[1 - β] + 1] β x∞) + β x∞
        #      = [1 - β]³ x0 + [1 - β]² β x∞ + [1 - β] β x∞ + β x∞
        #
        # To simplify notation, define:
        #
        #   α := 1 - β
        #
        # We have:
        #
        #   x1 = α  x0 + [1 - α] x∞
        #   x2 = α² x0 + [1 - α] [1 + α] x∞
        #      = α² x0 + [1 - α²] x∞
        #   x3 = α³ x0 + [1 - α] [1 + α + α²] x∞
        #      = α³ x0 + [1 - α³] x∞
        #
        # This suggests that the general pattern is (as can be proven by induction on `n`):
        #
        #   xn = α**n x0 + [1 - α**n] x∞
        #
        # This allows us to determine `x` as a function of simulation step number `n`. Now the scaling question becomes:
        # if we want to reach a given value `xn` by some given step `n_scaled` (instead of the original step `n`),
        # how must we change the step size `β` (or equivalently, the parameter `α`)?
        #
        # To simplify further, observe:
        #
        #   x1 = α x0 + [1 - α] [[x∞ - x0] + x0]
        #      = [α + [1 - α]] x0 + [1 - α] [x∞ - x0]
        #      = x0 + [1 - α] [x∞ - x0]
        #
        # Rearranging yields:
        #
        #   [x1 - x0] / [x∞ - x0] = 1 - α
        #
        # which gives us the relative distance from `x0` to `x∞` that is covered in one step. This isn't yet much
        # to write home about (it's essentially just a rearrangement of the definition of `x1`), but next, let's
        # treat `x2` the same way:
        #
        #   x2 = α² x0 + [1 - α] [1 + α] [[x∞ - x0] + x0]
        #      = [α² x0 + [1 - α²] x0] + [1 - α²] [x∞ - x0]
        #      = [α² + 1 - α²] x0 + [1 - α²] [x∞ - x0]
        #      = x0 + [1 - α²] [x∞ - x0]
        #
        # We obtain
        #
        #   [x2 - x0] / [x∞ - x0] = 1 - α²
        #
        # which is the relative distance, from the original `x0` toward the final `x∞`, that is covered in two steps
        # using the original step size `β = 1 - α`. Next up, `x3`:
        #
        #   x3 = α³ x0 + [1 - α³] [[x∞ - x0] + x0]
        #      = α³ x0 + [1 - α³] [x∞ - x0] + [1 - α³] x0
        #      = x0 + [1 - α³] [x∞ - x0]
        #
        # Rearranging,
        #
        #   [x3 - x0] / [x∞ - x0] = 1 - α³
        #
        # which is the relative distance covered in three steps. Hence, we have:
        #
        #   xrel := [xn - x0] / [x∞ - x0] = 1 - α**n
        #
        # so that
        #
        #   α**n = 1 - xrel              (**)
        #
        # and (taking the natural logarithm of both sides)
        #
        #   n log α = log [1 - xrel]
        #
        # Finally,
        #
        #   n = [log [1 - xrel]] / [log α]
        #
        # Given `α`, this gives the `n` where the interpolator has covered the fraction `xrel` of the original distance.
        # On the other hand, we can also solve (**) for `α`:
        #
        #   α = (1 - xrel)**(1 / n)
        #
        # which, given desired `n`, gives us the `α` that makes the interpolator cover the fraction `xrel` of the original distance in `n` steps.
        #
        CALIBRATION_FPS = 25  # FPS for which the default value `step` was calibrated
        xrel = 0.5  # just some convenient value
        step = self._settings["pose_interpolator_step"]
        alpha_orig = 1.0 - step
        if 0 < alpha_orig < 1:
            avg_render_sec = self.render_duration_statistics.average()
            if avg_render_sec > 0:
                avg_render_fps = 1 / avg_render_sec
                # Even if render completes faster, the avatar output is rate-limited to `self.target_fps` at most.
                avg_render_fps = min(avg_render_fps, self.target_fps)
            else:  # No statistics available yet; let's assume we're running at `target_fps`.
                avg_render_fps = self.target_fps

            # For a constant target and original `α`, compute the number of animation frames to cover `xrel` of distance from current value to target.
            n_orig = math.log(1.0 - xrel) / math.log(alpha_orig)
            # Compute the scaled `n`. Note the direction: we need a smaller `n` (fewer animation frames) if the render runs slower than the calibration FPS.
            n_scaled = (avg_render_fps / CALIBRATION_FPS) * n_orig
            # Then compute the `α` that reaches `xrel` distance in `n_scaled` animation frames.
            alpha_scaled = (1.0 - xrel)**(1 / n_scaled)
        else:  # avoid some divisions by zero at the extremes
            alpha_scaled = alpha_orig
        step_scaled = 1.0 - alpha_scaled

        debug_fps = round(avg_render_fps, 1)
        logger.debug(f"interpolate: step @ {CALIBRATION_FPS} FPS = {step}, scaled step @ {debug_fps:.1f} FPS = {step_scaled:0.6g}")

        # NOTE: When interpolation is applied to a pose, this overwrites blinking, talking, and breathing, but that doesn't matter,
        # because we apply this interpolator first. The other animation drivers may then partially overwrite our result.
        EPSILON = 1e-8
        new = list(current)  # copy
        for idx in range(len(current)):
            delta = target[idx] - current[idx]
            new[idx] = current[idx] + step_scaled * delta

            # Prevent denormal floats (which are really slow); important when running on CPU and approaching zero.
            # Our ϵ is really big compared to denormals; but there's no point in continuing to compute ever smaller
            # differences in the animated value when it has already almost (and visually, completely) reached the target.
            if abs(new[idx] - target[idx]) < EPSILON:
                new[idx] = target[idx]
        return new

    # --------------------------------------------------------------------------------
    # Animation logic

    def render_animation_frame(self) -> None:
        """Render an animation frame.

        If the previous rendered frame has not been retrieved yet, do nothing.
        """
        if self.source_image is None:  # if no input image, do nothing.
            return
        if not self.animation_running:  # if paused, do nothing.
            return
        if self.new_frame_available:  # if no one has retrieved the latest rendered frame yet, do nothing.
            return

        do_crop = any(self._settings[key] != 0 for key in ("crop_left", "crop_right", "crop_top", "crop_bottom"))

        metrics_enabled = self._settings["metrics_enabled"]
        def maybe_sync_cuda():
            if metrics_enabled:
                torch.cuda.synchronize()

        maybe_sync_cuda()
        time_render_start = time.time_ns()

        emotion = self.emotions[self.emotion]
        if self.current_pose is None:  # initialize character pose at startup
            # `current_pose` and `current_celstack` hold the character's instantaneous state.
            self.current_pose = avatarutil.posedict_to_pose(emotion["pose"])
            self.current_celstack = emotion["cels"]

        if self.emotion != self.last_emotion:  # some animation drivers need to know when the emotion last changed
            self.last_emotion_change_timestamp = time_render_start

        # Compute target pose and celstack (which we interpolate toward)
        target_posedict = emotion["pose"]
        target_celstack = emotion["cels"]
        target_pose = self.apply_emotion_to_pose(target_posedict, self.current_pose)
        target_pose = self.compute_sway_target_pose(target_pose)

        # Apply manual overrides. Doing this to *target* pose (not directly to current pose) makes the lipsync overrides take effect smoothly.
        # This looks especially good at pose interpolator step = 0.3.
        target_pose, target_celstack = self.apply_overrides(target_pose, target_celstack)

        # Animate pose
        self.current_pose = self.interpolate(self.current_pose, target_pose)
        self.current_pose = self.animate_blinking(self.current_pose)
        self.current_pose = self.animate_talking(self.current_pose, target_pose)
        self.current_pose = self.animate_breathing(self.current_pose)

        # Animate celstack
        target_cel_strengths = [v for k, v in target_celstack]
        current_cel_strengths = [v for k, v in self.current_celstack]  # TODO: fix unnecessary data conversion back and forth
        current_cel_strengths = self.interpolate(current_cel_strengths, target_cel_strengths)

        self.current_celstack = [(k, v) for (k, _), v in zip(self.current_celstack, current_cel_strengths)]

        waver1_idx = avatarutil.get_cel_index_in_stack("waver1", target_celstack)  # "waver1" in the emotion controls the eye-waver effect strength
        if waver1_idx != -1:  # found?
            _, strength = target_celstack[waver1_idx]
            self.current_celstack = self.animate_eye_waver(strength, self.current_celstack)

        # Update the last-emotion state last, so that animation drivers have access to the old emotion, too.
        self.last_emotion = self.emotion

        with torch.no_grad():
            # Detailed performance measurement protocol: sync CUDA (i.e. finish pending async CUDA operations), start timer, do desired CUDA operation(s), sync CUDA again, stop timer.
            with timer() as tim_celblend:
                blended_source_image = avatarutil.render_celstack(self.source_image, self.current_celstack, self.torch_cels)
                # data range [0, 1] -> [-1, 1], for poser
                blended_source_image.mul_(2.0)
                blended_source_image.sub_(1.0)
                maybe_sync_cuda()

            # - [0]: model's output index for the full result image
            # - model's data range is [-1, +1], linear intensity ("gamma encoded")
            with timer() as tim_pose:
                pose = torch.tensor(self.current_pose, device=self.device, dtype=self.poser.get_dtype())
                output_image = self.poser.pose(blended_source_image, pose)[0]
                maybe_sync_cuda()

            # [-1, 1] -> [0, 1]
            # output_image = (output_image + 1.0) / 2.0
            with timer() as tim_normalize:
                output_image.add_(1.0)
                output_image.mul_(0.5)
                maybe_sync_cuda()

            with timer() as tim_upscale:
                if self.upscaler is not None:
                    output_image = self.upscaler.upscale(output_image)
                    maybe_sync_cuda()

            # A simple crop filter, for removing empty space around character.
            # Apply this now so that if we're cropping, the postprocessor has fewer pixels to process.
            with timer() as tim_crop:
                if do_crop:
                    c, h, w = output_image.shape
                    x1 = int((self._settings["crop_left"] / 2.0) * w)
                    x2 = int((1 - (self._settings["crop_right"] / 2.0)) * w)
                    y1 = int((self._settings["crop_top"] / 2.0) * h)
                    y2 = int((1 - (self._settings["crop_bottom"] / 2.0)) * h)
                    output_image = output_image[:, y1:y2, x1:x2]
                    maybe_sync_cuda()

            with timer() as tim_postproc:
                self.postprocessor.render_into(output_image)  # apply pixel-space glitch artistry
                maybe_sync_cuda()

            with timer() as tim_gamma:
                output_image[:3, :, :] = torch_linear_to_srgb(output_image[:3, :, :])  # apply gamma correction
                maybe_sync_cuda()

            # convert [c, h, w] float -> [h, w, c] uint8
            with timer() as tim_dataformat:
                c, h, w = output_image.shape
                output_image = torch.transpose(output_image.reshape(c, h * w), 0, 1).reshape(h, w, c)
                output_image = (255.0 * output_image).byte()
                maybe_sync_cuda()

            with timer() as tim_sendtocpu:
                output_image_numpy = output_image.detach().cpu().numpy()
                maybe_sync_cuda()

        # Update FPS counter, measuring the complete render process.
        #
        # This is used for FPS compensation by the animation routines, and works regardless of whether metrics are enabled.
        # (CUDA will perform all its pending async operations at the latest when we send the final tensor to the CPU.)
        #
        # This says how fast the renderer *can* run on the current hardware; note we never render more frames than the client consumes.
        time_now = time.time_ns()
        if self.source_image is not None:
            render_elapsed_sec = (time_now - time_render_start) / 10**9
            self.render_duration_statistics.add_datapoint(render_elapsed_sec)

        if metrics_enabled:
            logger.info(f"total {1000 * render_elapsed_sec:0.1f} ms; cel blending {1000 * tim_celblend.dt:0.1f} ms, pose {1000 * tim_pose.dt:0.1f} ms, norm {1000 * tim_normalize.dt:0.1f} ms, upscale {1000 * tim_upscale.dt:0.1f} ms, crop {1000 * tim_crop.dt:0.1f} ms, post {1000 * tim_postproc.dt:0.1f} ms, gamma {1000 * tim_gamma.dt:0.1f} ms, chw->hwc {1000 * tim_dataformat.dt:0.1f} ms, to CPU {1000 * tim_sendtocpu.dt:0.1f} ms")

        # Set the new rendered frame as the output image, and mark the frame as ready for consumption.
        with self.output_lock:
            self.result_image = output_image_numpy  # atomic replace
            self.new_frame_available = True

        # Log the FPS counter in 5-second intervals. Note we only reach this when running (not paused).
        if self.last_report_time is None or time_now - self.last_report_time > 5e9:
            avg_render_sec = self.render_duration_statistics.average()
            msec = round(1000 * avg_render_sec, 1)
            fps = round(1 / avg_render_sec, 1) if avg_render_sec > 0.0 else 0.0
            logger.info(f"render {msec:.1f}ms [{fps} FPS available]")
            self.last_report_time = time_now

# --------------------------------------------------------------------------------

class Encoder:
    """Network transport encoder.

    We read each frame from the animator as it becomes ready, and keep it available in `self.current_frame`
    until the next frame arrives. The `self.current_frame` reference is replaced atomically, so this needs no lock
    (you always get the latest available frame at the time you access `current_frame`).
    """

    def __init__(self, instance_id: str) -> None:
        self.current_frame = None
        self.encoder_thread = None
        self.output_format = server_config.animator_defaults["format"]  # default until animator settings are loaded; note `output_format` is writable from other threads!
        self.instance_id = instance_id
        self.latest_frame_sent = None  # for co-operation with `result_feed` (NOTE: only one feed allowed per instance!) (TODO: relax this assumption? A bit difficult to do.)

    def start(self) -> None:
        """Start the output encoder thread."""
        logger.info(f"Encoder.start (avatar instance '{self.instance_id}'): Encoder is starting.")
        self._terminated = False
        def encoder_update():
            last_report_time = None
            encode_duration_statistics = RunningAverage()
            wait_duration_statistics = RunningAverage()

            while not self._terminated:
                time_encode_start = time.time_ns()

                # Retrieve a new frame from the animator if available.
                animator = _avatar_instances[self.instance_id]["animator"]
                have_new_frame = False
                with animator.output_lock:
                    if animator.new_frame_available:
                        image_rgba = animator.result_image  # atomic get
                        animator.new_frame_available = False  # animation frame consumed; start rendering the next one
                        have_new_frame = True  # This flag is needed so we can release the animator lock as early as possible.

                # If a new frame arrived, pack it for sending (only once for each new frame).
                if have_new_frame:
                    try:
                        # Important: grab reference to `output_format` just once per frame; may be changed (atomic replace) by another thread while we're encoding.
                        output_format = self.output_format

                        # time_now = time.time_ns()
                        if output_format.upper() == "QOI":  # Quite OK Image format - like PNG, but fast
                            # Ugh, we must copy because the data isn't C-contiguous... but this is still faster than the other formats.
                            current_frame = (output_format.upper(), qoi.encode(image_rgba.copy(order="C")))  # input: uint8 array of shape (h, w, c)
                        else:  # use PIL
                            pil_image = PIL.Image.fromarray(np.uint8(image_rgba[:, :, :3]))
                            if image_rgba.shape[2] == 4:
                                alpha_channel = image_rgba[:, :, 3]
                                pil_image.putalpha(PIL.Image.fromarray(np.uint8(alpha_channel)))

                            buffer = io.BytesIO()
                            if output_format == "PNG":
                                kwargs = {"compress_level": 1}
                            elif output_format == "TGA":
                                kwargs = {"compression": "tga_rle"}
                            else:
                                kwargs = {}
                            pil_image.save(buffer,
                                           format=output_format.upper(),
                                           **kwargs)
                            current_frame = (output_format, buffer.getvalue())
                        # pack_duration_sec = (time.time_ns() - time_now) / 10**9  # DEBUG / benchmarking

                        # We now have a new encoded frame; but first, sync with network send.
                        # This prevents from rendering/encoding more frames than are actually sent.
                        previous_frame = self.current_frame
                        if previous_frame is not None:
                            time_wait_start = time.time_ns()
                            # Wait in 1ms increments until the previous encoded frame has been sent
                            while self.latest_frame_sent != id(previous_frame) and not self._terminated:
                                time.sleep(0.001)
                            time_now = time.time_ns()
                            wait_elapsed_sec = (time_now - time_wait_start) / 10**9
                        else:
                            wait_elapsed_sec = 0.0

                        self.current_frame = current_frame  # atomic replace so no need for a lock
                    except Exception as exc:
                        logger.error(exc)
                        traceback.print_exc()
                        raise  # let the encoder shut down so we won't spam the log

                    # Update FPS counter.
                    time_now = time.time_ns()
                    walltime_elapsed_sec = (time_now - time_encode_start) / 10**9
                    encode_elapsed_sec = walltime_elapsed_sec - wait_elapsed_sec
                    encode_duration_statistics.add_datapoint(encode_elapsed_sec)
                    wait_duration_statistics.add_datapoint(wait_elapsed_sec)

                # Log the FPS counter in 5-second intervals.
                time_now = time.time_ns()
                if animator.animation_running and (last_report_time is None or time_now - last_report_time > 5e9):
                    avg_encode_sec = encode_duration_statistics.average()
                    msec = round(1000 * avg_encode_sec, 1)
                    avg_wait_sec = wait_duration_statistics.average()
                    wait_msec = round(1000 * avg_wait_sec, 1)
                    fps = round(1 / avg_encode_sec, 1) if avg_encode_sec > 0.0 else 0.0
                    logger.info(f"encode: {msec:.1f}ms [{fps} FPS available]; send sync wait {wait_msec:.1f}ms")
                    last_report_time = time_now

                time.sleep(0.01)  # rate-limit the encoder to 100 FPS maximum (this could be adjusted later)
        self.encoder_thread = threading.Thread(target=encoder_update, daemon=True)
        self.encoder_thread.start()
        logger.info(f"Encoder.start (avatar instance '{self.instance_id}'): Encoder startup complete.")

    def exit(self) -> None:
        """Terminate the output encoder thread.

        Called automatically when the process exits.
        """
        logger.info(f"Encoder.exit (avatar instance '{self.instance_id}'): Encoder is shutting down.")
        self._terminated = True
        if self.encoder_thread is not None:
            self.encoder_thread.join()
        self.encoder_thread = None
        self.current_frame = None
        logger.info(f"Encoder.exit (avatar instance '{self.instance_id}'): Encoder shutdown complete.")

"""Example Python client for Talkinghead, showing how to use its web API.

Convenient for testing.

Unlike the rest of `raven-avatar`, this example program is licensed under the 2-clause BSD license,
to facilitate integrating Talkinghead regardless of the license of your own software.
"""

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import concurrent.futures
import copy
import io
import json
import os
import pathlib
import re
import requests
import time
from typing import Callable, Dict, Generator, Iterator, List, Optional, Union

import PIL.Image

from unpythonic.env import env as envcls
from unpythonic.net.util import ReceiveBuffer

import pygame  # for audio (text to speech) support

import numpy as np

import dearpygui.dearpygui as dpg

from ..vendor.file_dialog.fdialog import FileDialog  # https://github.com/totallynotdrait/file_dialog, but with custom modifications
from .. import animation  # Raven's GUI animation system, nothing to do with the AI avatar.
from .. import bgtask  # TODO: read the result_feed in a bgtask; see Raven's preprocessor.py for how to use bgtask for things like this
from .. import utils as raven_utils

from .vendor.anime4k import anime4k

from . import client_util
from . import config
from .util import RunningAverage

# ----------------------------------------
# Module bootup

avatar_url = "http://localhost:5100"
config_dir = pathlib.Path(config.config_base_dir).expanduser().resolve()
api_key_file = config_dir / "api_key.txt"

default_headers = {
}

if os.path.exists(api_key_file):  # TODO: test this (I have no idea what I'm doing; check against `avatar/server.py`)
    with open(api_key_file, "r", encoding="utf-8") as f:
        api_key = f.read()
    default_headers["Authorization"] = api_key.strip()

# AI speech synthesizer - OpenAI compatible
# https://github.com/remsky/Kokoro-FastAPI
# TODO: add API key support for this too (Authorization: Bearer xxxx)
tts_url = "http://localhost:8880"
tts_default_headers = {
}

pygame.mixer.init()

# --------------------------------------------------------------------------------
# Utilities

def yell_on_error(response: requests.Response) -> None:
    if response.status_code != 200:
        logger.error(f"Avatar server returned error: {response.status_code} {response.reason}. Content of error response follows.")
        logger.error(response.text)
        raise RuntimeError(f"While calling avatar server: HTTP {response.status_code} {response.reason}")

# TODO: split to a new `netutil.py` or something
def multipart_x_mixed_replace_payload_extractor(source: Iterator[bytes],
                                                boundary_prefix: str,
                                                expected_mimetype: str) -> Generator[bytes, None, None]:
    """Generator: yield payloads from `source`, which is reading from a "multipart/x-mixed-replace" stream.

    The server MUST send the Content-Type and Content-Length headers.

    Content-Type must match `expected_mimetype`, e.g. "image/png".

    Loosely based on `unpythonic.net.msg.decodemsg`.
    """
    stream_iterator = iter(source)
    boundary_prefix = boundary_prefix.encode()  # str -> bytes
    payload_buffer = ReceiveBuffer()

    def read_more_input() -> None:
        try:
            data = next(stream_iterator)
        except StopIteration:
            raise EOFError
        payload_buffer.append(data)

    def synchronize() -> None:
        """Synchronize `payload_buffer` to the start of the next payload boundary marker (e.g. "--frame")."""
        while True:
            val = payload_buffer.getvalue()
            idx = val.rfind(boundary_prefix)
            if idx != -1:
                junk, start_of_payload = val[:idx], val[idx:]  # noqa: F841
                payload_buffer.set(start_of_payload)
                return
            # Clear the receive buffer after each chunk that didn't have a sync
            # marker in it. This prevents a malicious sender from crashing the
            # receiver by flooding it with nothing but junk.
            payload_buffer.set(b"")
            read_more_input()

    def read_headers() -> int:
        """Read and validate headers for one payload. Return the length of the payload body, in bytes."""
        while True:
            val = payload_buffer.getvalue()
            end_of_headers_idx = val.find(b"\r\n\r\n")
            if end_of_headers_idx != -1:  # headers completely streamed? (have a blank line at the end)
                break
        headers, start_of_body = val[:end_of_headers_idx], val[end_of_headers_idx + 4:]
        headers = headers.split(b"\r\n")
        if headers[0] != boundary_prefix:  # after sync, we should always have the payload boundary marker at the start of the buffer
            assert False
        body_length_bytes = None
        for field in headers[1:]:
            field = field.decode("utf-8")
            field_name, field_value = [text.strip().lower() for text in field.split(":")]
            if field_name == "content-type":
                if field_value != expected_mimetype:  # wrong type of data?
                    raise ValueError
            if field_name == "content-length":
                body_length_bytes = int(field_value)  # and let it raise if the value is invalid
        if body_length_bytes is None:
            raise ValueError("read_headers: payload is missing the 'Content-Length' header (mandatory for this client)")
        payload_buffer.set(start_of_body)
        return body_length_bytes

    def read_body(body_length_bytes: int) -> bytes:
        """Read the payload body and return it as a `bytes` object."""
        while True:
            val = payload_buffer.getvalue()
            if len(val) >= body_length_bytes:
                break
            read_more_input()
        body, leftovers = val[:body_length_bytes], val[body_length_bytes:]
        payload_buffer.set(leftovers)
        return body

    while True:
        synchronize()
        body_length_bytes = read_headers()
        payload = read_body(body_length_bytes)
        yield payload

# --------------------------------------------------------------------------------
# Python client for Talkinghead web API

# TODO: move into a new module

def classify_labels() -> List[str]:
    """Get list of emotion names from server."""
    headers = copy.copy(default_headers)
    response = requests.get(f"{avatar_url}/api/classify/labels", headers=headers)
    yell_on_error(response)
    output_data = response.json()  # -> {"labels": [emotion0, ...]}
    return list(sorted(output_data["labels"]))

def classify(text: str) -> Dict[str, float]:  # TODO: feature orthogonality
    """Classify the emotion of `text` and auto-update the avatar's emotion from that."""
    headers = copy.copy(default_headers)
    headers["Content-Type"] = "application/json"
    input_data = {"text": text}
    response = requests.post(f"{avatar_url}/api/classify", headers=headers, json=input_data)
    yell_on_error(response)
    output_data = response.json()  # -> ["classification": [{"label": "curiosity", "score": 0.5329479575157166}, ...]]

    sorted_records = output_data["classification"]  # sorted already
    return {record["label"]: record["score"] for record in sorted_records}

def talkinghead_load(filename: Union[pathlib.Path, str]) -> None:
    """Send a character (512x512 RGBA PNG image) to the animator.

    Then, if the animator is not running, start it automatically.
    """
    headers = copy.copy(default_headers)
    # Flask expects the file as multipart/form-data. `requests` sets this automatically when we send files, if we don't set a 'Content-Type' header.
    with open(filename, "rb") as image_file:
        files = {"file": image_file}
        response = requests.post(f"{avatar_url}/api/talkinghead/load", headers=headers, files=files)
    yell_on_error(response)

def talkinghead_unload() -> None:
    """Actually just pause the animator, don't unload anything."""
    headers = copy.copy(default_headers)
    response = requests.get(f"{avatar_url}/api/talkinghead/unload", headers=headers)
    yell_on_error(response)

def talkinghead_reload() -> None:
    """Resume the animator after it was paused via `talkinghead_unload`, without sending a new character."""
    headers = copy.copy(default_headers)
    response = requests.get(f"{avatar_url}/api/talkinghead/reload", headers=headers)
    yell_on_error(response)

def talkinghead_load_emotion_templates(emotions: Dict) -> None:
    headers = copy.copy(default_headers)
    headers["Content-Type"] = "application/json"
    response = requests.post(f"{avatar_url}/api/talkinghead/load_emotion_templates", json=emotions, headers=headers)
    yell_on_error(response)

def talkinghead_load_emotion_templates_from_file(filename: Union[pathlib.Path, str]) -> None:
    with open(filename, "r", encoding="utf-8") as json_file:
        emotions = json.load(json_file)
    talkinghead_load_emotion_templates(emotions)

def talkinghead_load_animator_settings(animator_settings: Dict) -> None:
    headers = copy.copy(default_headers)
    headers["Content-Type"] = "application/json"
    response = requests.post(f"{avatar_url}/api/talkinghead/load_animator_settings", json=animator_settings, headers=headers)
    yell_on_error(response)

def talkinghead_load_animator_settings_from_file(filename: Union[pathlib.Path, str]) -> None:
    with open(filename, "r", encoding="utf-8") as json_file:
        animator_settings = json.load(json_file)
    talkinghead_load_animator_settings(animator_settings)

def talkinghead_start_talking() -> None:
    headers = copy.copy(default_headers)
    response = requests.get(f"{avatar_url}/api/talkinghead/start_talking", headers=headers)
    yell_on_error(response)

def talkinghead_stop_talking() -> None:
    headers = copy.copy(default_headers)
    response = requests.get(f"{avatar_url}/api/talkinghead/stop_talking", headers=headers)
    yell_on_error(response)

def talkinghead_set_emotion(emotion_name: str) -> None:
    headers = copy.copy(default_headers)
    headers["Content-Type"] = "application/json"
    data = {"emotion_name": emotion_name}
    response = requests.post(f"{avatar_url}/api/talkinghead/set_emotion", headers=headers, json=data)
    yell_on_error(response)

def talkinghead_result_feed(chunk_size: int = 4096) -> Generator[bytes, None, None]:
    """Return a generator that yields `bytes` objects, one per video frame, as PNG.

    Due to the server's framerate control, the result feed attempts to feed data to the client at TARGET_FPS (default 25).
    New frames are not generated until the previous one has been consumed. Thus, while the animator is in the running state,
    it is recommended to continuously read the stream in a background thread.

    To close the connection (so that the server stops sending), call the `.close()` method of the generator.
    The connection also auto-closes when the generator is garbage-collected.
    """
    headers = copy.copy(default_headers)
    headers["Accept"] = "multipart/x-mixed-replace"
    stream_response = requests.get(f"{avatar_url}/api/talkinghead/result_feed", headers=headers, stream=True)
    yell_on_error(stream_response)

    stream_iterator = stream_response.iter_content(chunk_size=chunk_size)
    boundary = re.search(r"boundary=(\S+)", stream_response.headers["Content-Type"]).group(1)
    boundary_prefix = f"--{boundary}"  # e.g., '--frame'
    gen = multipart_x_mixed_replace_payload_extractor(source=stream_iterator,
                                                      boundary_prefix=boundary_prefix,
                                                      expected_mimetype="image/png")
    return gen

# # DEBUG/TEST - exercise each of the API endpoints
# print(classify_labels())  # get available emotion names from server
# talkinghead_load("example.png")  # send the avatar - mandatory
# talkinghead_load_animator_settings(os.path.join(os.path.dirname(__file__), "animator.json"))  # send animator config - optional, server defaults used if not sent
# talkinghead_load_emotion_templates(os.path.join(os.path.dirname(__file__), "emotions", "_defaults.json"))  # send the morph parameters for emotions - optional, server defaults used if not sent
# gen = talkinghead_result_feed()
# talkinghead_start_talking()  # start "talking right now" animation
# print(classify("What is the airspeed velocity of an unladen swallow?"))  # classify some text, auto-update emotion from result
# talkinghead_set_emotion("surprise")  # manually update emotion
# for _ in range(5):
#     image_data = next(gen)  # next-gen lol
#     image_file = io.BytesIO(image_data)
#     image = PIL.Image.open(image_file)
# talkinghead_stop_talking()  # stop "talking right now" animation
# talkinghead_unload()  # pause animating the talkinghead
# talkinghead_reload()  # resume animating the talkinghead
# import sys
# sys.exit(0)

# --------------------------------------------------------------------------------
# Python client for AI speech synthesizer

def tts_available() -> bool:
    """Return whether the speech synthesizer is available."""
    headers = copy.copy(tts_default_headers)
    response = requests.get(f"{tts_url}/health", headers=headers)
    if response.status_code != 200:
        return False
    return True

def tts_voices() -> None:
    """Return a list of voice names supported by the TTS endpoint."""
    headers = copy.copy(tts_default_headers)
    response = requests.get(f"{tts_url}/v1/audio/voices", headers=headers)
    yell_on_error(response)
    output_data = response.json()
    return output_data["voices"]

def tts_speak(voice: str,
              text: str,
              start_callback: Optional[Callable],
              stop_callback: Optional[Callable]) -> None:
    """Using the speech synthesizer, speak `text` using `voice`.

    If `start_callback` is provided, call it when the TTS starts speaking.
    If `stop_callback` is provided, call it when the TTS has stopped speaking.
    """
    headers = copy.copy(tts_default_headers)
    headers["Content-Type"] = "application/json"
    data = {"model": "kokoro",
            "voice": voice,
            "input": text,
            "response_format": "mp3",  # flac would be nice (faster to encode), but seems currently broken in kokoro (the audio may repeat twice)
            "speed": 1,
            "stream": True,
            "return_download_link": False}
    stream_response = requests.post(f"{tts_url}/v1/audio/speech", headers=headers, json=data, stream=True)
    yell_on_error(stream_response)

    # We run this in the background to
    def speak(task_env) -> None:
        it = stream_response.iter_content(chunk_size=4096)
        audio_buffer = io.BytesIO()
        try:
            while True:
                if task_env.cancelled:
                    return
                chunk = next(it)
                audio_buffer.write(chunk)
        except StopIteration:
            pass

        # # DEBUG - dump response to audio file
        # audio_buffer.seek(0)
        # with open("temp.mp3", "wb") as audio_file:
        #     audio_file.write(audio_buffer.getvalue())

        # play audio
        audio_buffer.seek(0)
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
        pygame.mixer.music.load(audio_buffer)
        # pygame.mixer.music.load(stream_response.raw)  # can't do this at least with mp3 since the raw stream doesn't support seeking.

        if start_callback is not None:
            start_callback()
        pygame.mixer.music.play()

        if stop_callback is not None:
            while pygame.mixer.music.get_busy():
                time.sleep(0.01)
            stop_callback()
    task_manager.submit(speak, envcls())

# --------------------------------------------------------------------------------
# DPG init

dpg.create_context()

# Initialize fonts. Must be done after `dpg.create_context`, or the app will just segfault at startup.
# https://dearpygui.readthedocs.io/en/latest/documentation/fonts.html
with dpg.font_registry() as the_font_registry:
    # Change the default font to something that looks clean and has good on-screen readability.
    # https://fonts.google.com/specimen/Open+Sans
    font_size = 20
    with dpg.font(os.path.join(os.path.dirname(__file__), "..", "fonts", "OpenSans-Regular.ttf"),  # load font from Raven's main assets
                  font_size) as default_font:
        pass
        # utils.setup_font_ranges()
    dpg.bind_font(default_font)

# Modify global theme
with dpg.theme() as global_theme:
    with dpg.theme_component(dpg.mvAll):
        # dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (53, 168, 84))  # same color as Linux Mint default selection color in the green theme
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6, category=dpg.mvThemeCat_Core)
        dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 8, category=dpg.mvThemeCat_Core)
        dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 8, category=dpg.mvThemeCat_Core)
dpg.bind_theme(global_theme)  # set this theme as the default

# FIX disabled controls not showing as disabled.
# DPG does not provide a default disabled-item theme, so we provide our own.
# Everything else is automatically inherited from DPG's global theme.
#     https://github.com/hoffstadt/DearPyGui/issues/2068
# TODO: Figure out how to get colors from a theme. Might not always be `(45, 45, 48)`.
#   - Maybe see how DPG's built-in theme editor does it - unless it's implemented at the C++ level.
#   - See also the theme color editor in https://github.com/hoffstadt/DearPyGui/wiki/Tools-and-Widgets
disabled_color = (0.50 * 255, 0.50 * 255, 0.50 * 255, 1.00 * 255)
disabled_button_color = (45, 45, 48)
disabled_button_hover_color = (45, 45, 48)
disabled_button_active_color = (45, 45, 48)
with dpg.theme(tag="disablable_button_theme"):
    # We customize just this. Everything else is inherited from the global theme.
    with dpg.theme_component(dpg.mvButton, enabled_state=False):
        dpg.add_theme_color(dpg.mvThemeCol_Text, disabled_color, category=dpg.mvThemeCat_Core)
        dpg.add_theme_color(dpg.mvThemeCol_Button, disabled_button_color, category=dpg.mvThemeCat_Core)
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, disabled_button_hover_color, category=dpg.mvThemeCat_Core)
        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, disabled_button_active_color, category=dpg.mvThemeCat_Core)

viewport_width = 1600
viewport_height = 700
dpg.create_viewport(title="Talkinghead",
                    width=viewport_width,
                    height=viewport_height)  # OS window (DPG "viewport")
dpg.setup_dearpygui()

# --------------------------------------------------------------------------------
# File dialog init

gui_instance = None  # initialized later, when the app starts

filedialog_open_image = None
filedialog_open_json = None
filedialog_open_animator_settings = None

def initialize_filedialogs(default_path):  # called at app startup, once we parse the default path from cmdline args (or set a default if not specified).
    """Create the file dialogs."""
    global filedialog_open_image
    global filedialog_open_json
    global filedialog_open_animator_settings
    filedialog_open_image = FileDialog(title="Open input image",
                                       tag="open_image_dialog",
                                       callback=_open_image_callback,
                                       modal=True,
                                       filter_list=[".png"],
                                       file_filter=".png",
                                       multi_selection=False,
                                       allow_drag=False,
                                       default_path=default_path)
    filedialog_open_json = FileDialog(title="Open emotion JSON file",
                                       tag="open_json_dialog",
                                       callback=_open_json_callback,
                                       modal=True,
                                       filter_list=[".json"],
                                       file_filter=".json",
                                       multi_selection=False,
                                       allow_drag=False,
                                       default_path=default_path)
    filedialog_open_animator_settings = FileDialog(title="Open animator settings JSON file",
                                                   tag="open_animator_settings_dialog",
                                                   callback=_open_animator_settings_callback,
                                                   modal=True,
                                                   filter_list=[".json"],
                                                   file_filter=".json",
                                                   multi_selection=False,
                                                   allow_drag=False,
                                                   default_path=default_path)

# --------------------------------------------------------------------------------
# "Open image" dialog

def show_open_image_dialog():
    """Button callback. Show the open file dialog, for the user to pick an image to open.

    If you need to close it programmatically, call `filedialog_open_image.cancel()` so it'll trigger the callback.
    """
    if gui_instance is None:
        return
    logger.debug("show_open_image_dialog: Showing open file dialog.")
    filedialog_open_image.show_file_dialog()
    logger.debug("show_open_image_dialog: Done.")

def _open_image_callback(selected_files):
    """Callback that fires when the open image dialog closes."""
    logger.debug("_open_image_callback: Open file dialog callback triggered.")
    if len(selected_files) > 1:  # Should not happen, since we set `multi_selection=False`.
        raise ValueError(f"Expected at most one selected file, got {len(selected_files)}.")
    if selected_files:
        selected_file = selected_files[0]
        logger.debug(f"_open_image_callback: User selected the file '{selected_file}'.")
        gui_instance.load_image(selected_file)
    else:  # empty selection -> cancelled
        logger.debug("_open_image_callback: Cancelled.")

def is_open_image_dialog_visible():
    """Return whether the open image dialog is open.

    We have this abstraction (not just `dpg.is_item_visible`) because the window might not exist yet.
    """
    if filedialog_open_image is None:
        return False
    return dpg.is_item_visible("open_image_dialog")  # tag

# --------------------------------------------------------------------------------
# "Open JSON" dialog (emotion templates)

def show_open_json_dialog():
    """Button callback. Show the open JSON dialog, for the user to pick an emotion templates JSON file to open.

    If you need to close it programmatically, call `filedialog_open_json.cancel()` so it'll trigger the callback.
    """
    if gui_instance is None:
        return
    logger.debug("show_open_json_dialog: Showing open file dialog.")
    filedialog_open_json.show_file_dialog()
    logger.debug("show_open_json_dialog: Done.")

def _open_json_callback(selected_files):
    """Callback that fires when the open JSON dialog closes."""
    logger.debug("_open_json_callback: Open file dialog callback triggered.")
    if len(selected_files) > 1:  # Should not happen, since we set `multi_selection=False`.
        raise ValueError(f"Expected at most one selected file, got {len(selected_files)}.")
    if selected_files:
        selected_file = selected_files[0]
        logger.debug(f"_open_json_callback: User selected the file '{selected_file}'.")
        gui_instance.load_json(selected_file)
    else:  # empty selection -> cancelled
        logger.debug("_open_json_callback: Cancelled.")

def is_open_json_dialog_visible():
    """Return whether the open JSON dialog is open.

    We have this abstraction (not just `dpg.is_item_visible`) because the window might not exist yet.
    """
    if filedialog_open_json is None:
        return False
    return dpg.is_item_visible("open_json_dialog")  # tag

# --------------------------------------------------------------------------------
# "Open JSON" dialog (animator settings)

def show_open_animator_settings_dialog():
    """Button callback. Show the open JSON dialog, for the user to pick an animator settings JSON file to open.

    If you need to close it programmatically, call `filedialog_open_animator_settings.cancel()` so it'll trigger the callback.
    """
    if gui_instance is None:
        return
    logger.debug("show_open_animator_settings_dialog: Showing open file dialog.")
    filedialog_open_animator_settings.show_file_dialog()
    logger.debug("show_open_animator_settings_dialog: Done.")

def _open_animator_settings_callback(selected_files):
    """Callback that fires when the open JSON dialog closes."""
    logger.debug("_open_animator_settings_callback: Open file dialog callback triggered.")
    if len(selected_files) > 1:  # Should not happen, since we set `multi_selection=False`.
        raise ValueError(f"Expected at most one selected file, got {len(selected_files)}.")
    if selected_files:
        selected_file = selected_files[0]
        logger.debug(f"_open_animator_settings_callback: User selected the file '{selected_file}'.")
        gui_instance.load_animator_settings(selected_file)
    else:  # empty selection -> cancelled
        logger.debug("_open_animator_settings_callback: Cancelled.")

def is_animator_settings_dialog_visible():
    """Return whether the open JSON dialog is open.

    We have this abstraction (not just `dpg.is_item_visible`) because the window might not exist yet.
    """
    if filedialog_open_animator_settings is None:
        return False
    return dpg.is_item_visible("open_animator_settings_dialog")  # tag

# --------------------------------------------------------------------------------
# GUI controls

def is_any_modal_window_visible():
    """Return whether *some* modal window is open.

    Currently these are file dialogs.
    """
    return (is_open_image_dialog_visible() or is_open_json_dialog_visible() or is_animator_settings_dialog_visible())

class TalkingheadExampleGUI:
    def __init__(self):
        self.image_size = 512
        self.button_width = 300

        self.talking = False
        self.animator_running = True

        # TODO: Investigate whether we could super-resolution in realtime with Anime4K-PyTorch. For 1920x1080 (2MP) that takes ~60ms per frame, but our image is only 0.25 MP, so it should be fast enough, if it works for this res.
        # TODO: Also investigate whether it's better to scale at the server side (one less GPU/CPU roundtrip, but more data to send) or at the client side.
        # https://github.com/bloc97/Anime4K
        # https://colab.research.google.com/drive/11xAn4fyAUJPZOjrxwnL2ipl_1DGGegkB#scrollTo=prK7pqyim1Uo

        with dpg.texture_registry(tag="talkinghead_example_textures"):
            self.blank_texture = np.zeros([self.image_size,  # height
                                           self.image_size,  # width
                                           4],  # RGBA
                                          dtype=np.float32).ravel()
            self.live_texture = dpg.add_raw_texture(width=self.image_size,
                                                    height=self.image_size,
                                                    default_value=self.blank_texture,
                                                    format=dpg.mvFormat_Float_rgba,
                                                    tag="live_texture")

        dpg.set_viewport_title(f"Talkinghead [{avatar_url}]")

        with dpg.window(tag="talkinghead_main_window",
                        label="Talkinghead main window") as self.window:  # label not actually shown, since this window is maximized to the whole viewport
            with dpg.group(horizontal=True):
                with dpg.group(tag="live_texture_group"):
                    dpg.add_spacer(width=self.image_size, height=0)  # keep the group at the image's width even when the image is hidden
                    dpg.add_image("live_texture", pos=(0, viewport_height - self.image_size - 8), tag="live_image")  # TODO: should render flush with bottom edge without causing a scrollbar to appear
                    dpg.add_text("FPS counter will appear here", color=(0, 255, 0), pos=(8, 0), tag="fps_text")
                    self.fps_statistics = RunningAverage()

                def position_please_standby_text():
                    # x0, y0 = raven_utils.get_widget_relative_pos("live_image", reference="main_window")
                    x0, y0 = raven_utils.get_widget_pos("live_image")
                    dpg.add_text("[No image loaded]", pos=(x0 + self.image_size / 2 - 60,
                                                           y0 + self.image_size / 2 - (font_size / 2)),
                                 tag="please_standby_text",
                                 parent="live_texture_group",
                                 show=False)
                dpg.set_frame_callback(10, position_please_standby_text)

                # TODO: reposition talkinghead on window resize
                # TODO: robustness: don't crash if the server is/goes down
                # TODO: animator/postproc settings editor
                # TODO: zooming (client-side, based on image data)
                with dpg.group(horizontal=False):
                    dpg.add_text("Load")
                    dpg.add_button(label="Load image [Ctrl+O]", width=self.button_width, callback=show_open_image_dialog, tag="open_image_button")
                    dpg.add_button(label="Load emotion templates [Ctrl+Shift+E]", width=self.button_width, callback=show_open_json_dialog, tag="open_json_button")
                    dpg.add_button(label="Load animator settings [Ctrl+Shift+A]", width=self.button_width, callback=show_open_animator_settings_dialog, tag="open_animator_settings_button")
                    dpg.add_spacer(height=8)

                    dpg.add_text("Emotion [Ctrl+E]")
                    self.emotion_names = classify_labels()
                    if "neutral" in self.emotion_names:
                        self.emotion_names.remove("neutral")
                        self.emotion_names = ["neutral"] + self.emotion_names
                    self.emotion_choice = dpg.add_combo(items=self.emotion_names,
                                                        default_value=self.emotion_names[0],
                                                        width=self.button_width,
                                                        callback=self.on_send_emotion)
                    talkinghead_set_emotion(self.emotion_names[0])  # initial emotion upon app startup; should be "neutral"
                    dpg.add_spacer(height=8)

                    dpg.add_text("Toggles")
                    dpg.add_button(label="Start talking [Ctrl+T]", width=self.button_width, callback=self.toggle_talking, tag="start_stop_talking_button")
                    dpg.add_button(label="Pause animator [Ctrl+P]", width=self.button_width, callback=self.toggle_animator_paused, tag="pause_resume_button")
                    dpg.add_spacer(height=8)

                    # AI speech synthesizer
                    tts_alive = tts_available()
                    if tts_alive:
                        heading_label = f"Voice [Ctrl+V] [{tts_url}]"
                        self.voice_names = tts_voices()
                    else:
                        heading_label = "Voice [Ctrl+V] [not connected]"
                        self.voice_names = ["[TTS server not available]"]
                    dpg.add_text(heading_label)
                    self.voice_choice = dpg.add_combo(items=self.voice_names,
                                                      default_value=self.voice_names[0],
                                                      width=self.button_width)
                    dpg.add_button(label="Speak [Ctrl+S]", width=self.button_width, callback=self.on_speak, enabled=tts_alive, tag="speak_button")
                    dpg.bind_item_theme("speak_button", "disablable_button_theme")

    def on_send_emotion(self, sender, app_data):  # GUI event handler
        # On clicking a choice in the combobox, `app_data` is that choice, but on arrow key, `app_data` is the keycode.
        logger.info(f"TalkingheadExampleGUI.on_send_emotion: sender = {sender}, app_data = {app_data}")
        current_emotion_name = dpg.get_value(self.emotion_choice)
        logger.info(f"TalkingheadExampleGUI.on_send_emotion: sending emotion '{current_emotion_name}'")
        talkinghead_set_emotion(current_emotion_name)

    def load_image(self, filename: Union[pathlib.Path, str]) -> None:
        try:
            talkinghead_load(filename)
        except Exception as exc:
            logger.error(f"TalkingheadExampleGUI.load_image: {type(exc)}: {exc}")
            client_util.modal_dialog(window_title="Error",
                                     message=f"Could not load image '{filename}', reason {type(exc)}: {exc}",
                                     buttons=["Close"],
                                     ok_button="Close",
                                     cancel_button="Close",
                                     centering_reference_window=self.window)

    def load_json(self, filename: Union[pathlib.Path, str]) -> None:
        try:
            talkinghead_load_emotion_templates_from_file(filename)
        except Exception as exc:
            logger.error(f"TalkingheadExampleGUI.load_json: {type(exc)}: {exc}")
            client_util.modal_dialog(window_title="Error",
                                     message=f"Could not load emotion templates JSON '{filename}', reason {type(exc)}: {exc}",
                                     buttons=["Close"],
                                     ok_button="Close",
                                     cancel_button="Close",
                                     centering_reference_window=self.window)

    def load_animator_settings(self, filename: Union[pathlib.Path, str]) -> None:
        try:
            talkinghead_load_animator_settings_from_file(filename)
        except Exception as exc:
            logger.error(f"TalkingheadExampleGUI.load_animator_settings: {type(exc)}: {exc}")
            client_util.modal_dialog(window_title="Error",
                                     message=f"Could not load animator settings JSON '{filename}', reason {type(exc)}: {exc}",
                                     buttons=["Close"],
                                     ok_button="Close",
                                     cancel_button="Close",
                                     centering_reference_window=self.window)

    def toggle_talking(self) -> None:
        """Toggle the talkinghead's talking state."""
        if not self.talking:
            talkinghead_start_talking()
            dpg.set_item_label("start_stop_talking_button", "Stop talking [Ctrl+T]")
        else:
            talkinghead_stop_talking()
            dpg.set_item_label("start_stop_talking_button", "Start talking [Ctrl+T]")
        self.talking = not self.talking

    def toggle_animator_paused(self) -> None:
        """Pause or resume the animation. Pausing when the talkinghead won't be visible (e.g. minimized window) saves resources as new frames are not computed."""
        if self.animator_running:
            talkinghead_unload()
            dpg.set_value("please_standby_text", "[Animator is paused]")
            dpg.show_item("please_standby_text")
            dpg.hide_item("live_image")
            dpg.set_item_label("pause_resume_button", "Resume animator [Ctrl+P]")
        else:
            talkinghead_reload()
            dpg.hide_item("please_standby_text")
            dpg.show_item("live_image")
            dpg.set_item_label("pause_resume_button", "Pause animator [Ctrl+P]")
        self.animator_running = not self.animator_running

    def on_speak(self, sender, app_data) -> None:
        current_voice = dpg.get_value(self.voice_choice)
        tts_speak(voice=current_voice,
                  text="Testing the AI speech synthesizer.",  # TODO
                  start_callback=talkinghead_start_talking,
                  stop_callback=talkinghead_stop_talking)

# Hotkey support
choice_map = None
def talkinghead_example_hotkeys_callback(sender, app_data):
    if gui_instance is None:
        return
    # Hotkeys while an "open file" or "save as" dialog is shown - fdialog handles its own hotkeys
    if is_any_modal_window_visible():
        return

    key = app_data
    shift_pressed = dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)
    ctrl_pressed = dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)

    # Ctrl+Shift+...
    if ctrl_pressed and shift_pressed:
        if key == dpg.mvKey_E:  # emotions
            show_open_json_dialog()
        elif key == dpg.mvKey_A:  # animator settings
            show_open_animator_settings_dialog()

    # Ctrl+...
    elif ctrl_pressed:
        if key == dpg.mvKey_O:
            show_open_image_dialog()
        elif key == dpg.mvKey_T:
            gui_instance.toggle_talking()
        elif key == dpg.mvKey_P:
            gui_instance.toggle_animator_paused()
        elif key == dpg.mvKey_E:
            dpg.focus_item(gui_instance.emotion_choice)
        elif key == dpg.mvKey_V:
            dpg.focus_item(gui_instance.voice_choice)
        elif key == dpg.mvKey_S:
            gui_instance.on_speak(sender, app_data)

    # Bare key
    #
    # NOTE: These are global across the whole app (when no modal window is open) - be very careful here!
    else:
        # {widget_tag_or_id: list_of_choices}
        global choice_map
        if choice_map is None:  # build on first use
            choice_map = {gui_instance.emotion_choice: (gui_instance.emotion_names, gui_instance.on_send_emotion),
                          gui_instance.voice_choice: (gui_instance.voice_names, None)}
        def browse(choice_widget, data):
            choices, callback = data
            index = choices.index(dpg.get_value(choice_widget))
            if key == dpg.mvKey_Down:
                new_index = min(index + 1, len(choices) - 1)
            elif key == dpg.mvKey_Up:
                new_index = max(index - 1, 0)
            elif key == dpg.mvKey_Home:
                new_index = 0
            elif key == dpg.mvKey_End:
                new_index = len(choices) - 1
            else:
                new_index = None
            if new_index is not None:
                dpg.set_value(choice_widget, choices[new_index])
                if callback is not None:
                    callback(sender, app_data)  # the callback doesn't trigger automatically if we programmatically set the combobox value
        focused_item = dpg.get_focused_item()
        if focused_item in choice_map.keys():
            browse(focused_item, choice_map[focused_item])
with dpg.handler_registry(tag="talkinghead_example_handler_registry"):  # global (whole viewport)
    dpg.add_key_press_handler(tag="talkinghead_example_hotkeys_handler", callback=talkinghead_example_hotkeys_callback)

# --------------------------------------------------------------------------------
# Animation client task

class ResultFeedReader:
    def __init__(self):
        self.gen = None

    def start(self):
        self.gen = talkinghead_result_feed()

    def is_running(self):
        return self.gen is not None

    def get_frame(self):
        return next(self.gen)  # next-gen lol

    def stop(self):
        self.gen.close()
        self.gen = None

# Implementation of preset A (HQ)
device = "cuda:0"  # TODO
upscaler_pipeline = anime4k.Anime4KPipeline(
    anime4k.ClampHighlight(),
    anime4k.create_model("Upscale_Denoise_VL"),
    anime4k.AutoDownscalePre(4),
    anime4k.create_model("Upscale_M"),
    screen_width=1024, screen_height=1024,
    final_stage_upscale_mode="bilinear"
).to(device).half()

# We must continuously retrieve new frames as they become ready, so this runs in the background.
def update_live_texture(task_env) -> None:
    assert task_env is not None
    reader = ResultFeedReader()
    reader.start()
    while not task_env.cancelled:
        frame_start_time = time.time_ns()

        if gui_instance:
            if not gui_instance.animator_running and reader.is_running():
                reader.stop()
                dpg.set_value("fps_text", "RX (avg) -- FPS")
            elif gui_instance.animator_running and not reader.is_running():
                reader.start()

        if reader.is_running():
            image_data = reader.get_frame()
        if gui_instance is None or not reader.is_running():
            time.sleep(0.01)
            continue

        image_file = io.BytesIO(image_data)

        # # EXPERIMENT: upscale 2x
        # #  - Would this be faster to do this on the server side? More data to send, but the image is still on the GPU.
        # original_pil_image = PIL.Image.open(image_file)  # input, 512x512
        # image_tensor = anime4k.to_tensor(original_pil_image.convert("RGB")).unsqueeze(0).to(device).half()
        # upscaled_image_tensor = upscaler_pipeline(image_tensor)
        # upscaled_pil_image = anime4k.to_pil(upscaled_image_tensor[0])
        # arr = np.asarray(upscaled_pil_image.convert("RGBA"))
        # arr = np.array(arr, dtype=np.float32) / 255
        # raw_data = arr.ravel()  # shape [h, w, c] -> linearly indexed
        # dpg.set_value(gui_instance.live_texture, raw_data)  # to GUI

        # at original resolution
        pil_image = PIL.Image.open(image_file)
        arr = np.asarray(pil_image.convert("RGBA"))
        arr = np.array(arr, dtype=np.float32) / 255
        raw_data = arr.ravel()  # shape [h, w, c] -> linearly indexed
        dpg.set_value(gui_instance.live_texture, raw_data)  # to GUI

        # Update FPS counter.
        # NOTE: Since we wait on the server to send a frame, the refresh is capped to the rate that data actually arrives at, i.e. the server's TARGET_FPS.
        #       If the machine could render faster, this just means less than 100% CPU/GPU usage.
        elapsed_time = time.time_ns() - frame_start_time
        fps = 1.0 / (elapsed_time / 10**9)
        gui_instance.fps_statistics.add_datapoint(fps)
        dpg.set_value("fps_text", f"RX (avg) {gui_instance.fps_statistics.average():0.2f} FPS")

# --------------------------------------------------------------------------------
# Main program

if __name__ == "__main__":
    gui_instance = TalkingheadExampleGUI()

    bg = concurrent.futures.ThreadPoolExecutor()
    task_manager = bgtask.TaskManager(name="talkinghead_example_client",
                                      mode="concurrent",
                                      executor=bg)
    talkinghead_load("example.png")  # this will also start the animator if it was paused
    talkinghead_load_emotion_templates({})  # send empty dict -> reset to server defaults
    talkinghead_load_animator_settings({})  # send empty dict -> reset to server defaults
    talkinghead_set_emotion("curiosity")
    def shutdown() -> None:
        task_manager.clear(wait=True)
        animation.animator.clear()
        global gui_instance
        gui_instance = None
    dpg.set_exit_callback(shutdown)
    task_manager.submit(update_live_texture, envcls())

    dpg.set_primary_window(gui_instance.window, True)  # Make this DPG "window" occupy the whole OS window (DPG "viewport").
    dpg.set_viewport_vsync(True)
    dpg.show_viewport()

    _default_path = os.getcwd()
    initialize_filedialogs(_default_path)

    # last_tts_check_time = 0
    # tts_check_interval = 5.0  # seconds
    def update_animations():
        animation.animator.render_frame()  # Our customized fdialog needs this for its overwrite confirm button flash.

        # # Enable/disable speech synthesizer controls depending on whether the TTS server is available
        # global last_tts_check_time
        # t0 = time.monotonic()
        # if t0 - last_tts_check_time >= tts_check_interval:
        #     last_tts_check_time = t0
        #     if tts_available():
        #         dpg.enable_item(gui_instance.voice_choice)
        #         dpg.enable_item("speak_button")
        #     else:
        #         dpg.disable_item(gui_instance.voice_choice)
        #         dpg.disable_item("speak_button")

    # We control the render loop manually to have a convenient place to update our GUI animations just before rendering each frame.
    while dpg.is_dearpygui_running():
        update_animations()
        dpg.render_dearpygui_frame()
    # dpg.start_dearpygui()  # automatic render loop

    dpg.destroy_context()

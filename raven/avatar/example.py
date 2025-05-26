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
from typing import Dict, Generator, Iterator, List, Union

import PIL.Image

from unpythonic.env import env as envcls
from unpythonic.net.util import ReceiveBuffer

import numpy as np

import dearpygui.dearpygui as dpg

from ..vendor.file_dialog.fdialog import FileDialog  # https://github.com/totallynotdrait/file_dialog, but with custom modifications
from .. import animation  # Raven's GUI animation system, nothing to do with the AI avatar.
from .. import bgtask  # TODO: read the result_feed in a bgtask; see Raven's preprocessor.py for how to use bgtask for things like this
# from .. import utils as raven_utils

from . import config

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

# --------------------------------------------------------------------------------
# Utilities

def yell_on_error(response: requests.Response) -> None:
    if response.status_code != 200:
        logger.error(f"Avatar server returned error: {response.status_code} {response.reason}. Content of error response follows.")
        logger.error(response.text)
        raise RuntimeError(f"While calling avatar server: HTTP {response.status_code} {response.reason}")

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

def talkinghead_load_emotion_templates(filename: Union[pathlib.Path, str]) -> None:
    headers = copy.copy(default_headers)
    headers["Content-Type"] = "application/json"
    with open(filename, "r", encoding="utf-8") as json_file:
        emotions = json.load(json_file)
    response = requests.post(f"{avatar_url}/api/talkinghead/load_emotion_templates", json=emotions, headers=headers)
    yell_on_error(response)

def talkinghead_load_animator_settings(filename: Union[pathlib.Path, str]) -> None:
    headers = copy.copy(default_headers)
    headers["Content-Type"] = "application/json"
    with open(filename, "r", encoding="utf-8") as json_file:
        emotions = json.load(json_file)
    response = requests.post(f"{avatar_url}/api/talkinghead/load_animator_settings", json=emotions, headers=headers)
    yell_on_error(response)

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
    """
    headers = copy.copy(default_headers)
    headers["Accept"] = "multipart/x-mixed-replace"
    response = requests.get(f"{avatar_url}/api/talkinghead/result_feed", headers=headers, stream=True)
    yell_on_error(response)

    stream_iterator = response.iter_content(chunk_size=chunk_size)
    boundary = re.search(r"boundary=(\S+)", response.headers["Content-Type"]).group(1)
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

dpg.create_viewport(title="Talkinghead example client",
                    width=1600,
                    height=1000)  # OS window (DPG "viewport")
dpg.setup_dearpygui()

# --------------------------------------------------------------------------------
# File dialog init

gui_instance = None  # initialized later, when the app starts

filedialog_open_image = None
filedialog_save_image = None
filedialog_open_json = None
filedialog_save_all_emotions = None

def initialize_filedialogs(default_path):  # called at app startup, once we parse the default path from cmdline args (or set a default if not specified).
    """Create the file dialogs."""
    global filedialog_open_image
    global filedialog_save_image
    global filedialog_open_json
    global filedialog_save_all_emotions
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
# "Open JSON" dialog

def show_open_json_dialog():
    """Button callback. Show the open JSON dialog, for the user to pick an emotion JSON file to open.

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
# GUI controls

def is_any_modal_window_visible():
    """Return whether *some* modal window is open.

    Currently these are file dialogs.
    """
    return (is_open_image_dialog_visible() or is_open_json_dialog_visible())

class TalkingheadExampleGUI:
    def __init__(self):
        self.image_size = 512
        self.button_width = 200

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

        with dpg.window(tag="talkinghead_example_main_window",
                        label="Talkinghead example client") as self.window:  # label not actually shown, since this window is maximized to the whole viewport
            with dpg.group(horizontal=True):
                dpg.add_image("live_texture", tag="live_image")

                # x0, y0 = raven_utils.get_widget_relative_pos("live_image", reference="main_window")
                # x0, y0 = raven_utils.get_widget_pos("live_image")
                # dpg.add_text("[No image loaded]", pos=(x0 + self.image_size / 2 - 60,
                #                                        y0 + self.image_size / 2 - (font_size / 2)),
                #              tag="no_image_loaded_text")

                # TODO: FPS counter
                # TODO: The rest of the controls
                with dpg.group(horizontal=False):
                    pass

                with dpg.group(horizontal=False):
                    dpg.add_button(label="Start talking", width=self.button_width)

# --------------------------------------------------------------------------------
# Animation client task

def update_live_texture(task_env):
    assert task_env is not None
    gen = talkinghead_result_feed()
    while not task_env.cancelled:
        image_data = next(gen)  # next-gen lol
        image_file = io.BytesIO(image_data)
        pil_image = PIL.Image.open(image_file)
        arr = np.asarray(pil_image.convert("RGBA"))
        arr = np.array(arr, dtype=np.float32) / 255
        raw_data = arr.ravel()  # shape [h, w, c] -> linearly indexed
        if gui_instance is not None:  # app is not shutting down
            dpg.set_value(gui_instance.live_texture, raw_data)  # to GUI

# --------------------------------------------------------------------------------
# Main program

if __name__ == "__main__":
    gui_instance = TalkingheadExampleGUI()

    bg = concurrent.futures.ThreadPoolExecutor()
    task_manager = bgtask.TaskManager(name="avatar_updater",
                                      mode="concurrent",
                                      executor=bg)
    talkinghead_load("example.png")
    def shutdown():
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

    def update_animations():
        animation.animator.render_frame()  # Our customized fdialog needs this for its overwrite confirm button flash.

    # We control the render loop manually to have a convenient place to update our GUI animations just before rendering each frame.
    while dpg.is_dearpygui_running():
        update_animations()
        dpg.render_dearpygui_frame()
    # dpg.start_dearpygui()  # automatic render loop

    dpg.destroy_context()

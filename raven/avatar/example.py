"""Postprocessor settings editor app for Talkinghead. Doubles as a tech demo.

!!! Start `server.py` first before running this app! !!!

To edit the emotion templates, see the separate app `editor.py`.

This module is licensed under the 2-clause BSD license, to facilitate Talkinghead integration anywhere.
"""

# TODO: save dialog for saving animator settings from GUI
# TODO: crash with an *informative* message when server is offline
# TODO: fit everything into window
# TODO: implement upscale slider
# TODO: reposition talkinghead on window resize

# TODO: implement color picker when a postprocessor parameter's GUI hint is "!RGB"
# TODO: add text entry for speech synthesizer testing
# TODO: robustness: don't crash if the server is/goes down
# TODO: editor for main animator config too (target FPS, talking speed, ...)
# TODO: zooming (add a zoom filter on the server - before postproc? Should be able to use crop + Anime4K for zooming.)
# TODO: support loading a background image (aligned to bottom left?)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import concurrent.futures
import io
import json
import os
import pathlib
import time
import traceback
from typing import Union

import qoi
import PIL.Image

from unpythonic.env import env as envcls

import numpy as np

import dearpygui.dearpygui as dpg

from ..vendor.file_dialog.fdialog import FileDialog  # https://github.com/totallynotdrait/file_dialog, but with custom modifications
from .. import animation  # Raven's GUI animation system, nothing to do with the AI avatar.
from .. import bgtask
from .. import utils as raven_utils

from . import client_api  # convenient Python functions that abstract away the web API
from . import client_util  # DPG GUI utilities
from . import config
from . import postprocessor
from .util import RunningAverage

# ----------------------------------------
# Module bootup

avatar_url = "http://localhost:5100"  # Avatar server
tts_url = "http://localhost:8880"  # AI speech synthesizer server, https://github.com/remsky/Kokoro-FastAPI

config_dir = pathlib.Path(config.config_base_dir).expanduser().resolve()
avatar_api_key_file = config_dir / "api_key.txt"
tts_api_key_file = config_dir / "tts_api_key.txt"

bg = concurrent.futures.ThreadPoolExecutor()
task_manager = bgtask.TaskManager(name="talkinghead_example_client",
                                  mode="concurrent",
                                  executor=bg)
client_api.init_module(avatar_url=avatar_url,
                       avatar_api_key_file=avatar_api_key_file,
                       tts_url=tts_url,
                       tts_api_key_file=tts_api_key_file,
                       executor=bg)  # reuse our executor so the TTS audio player goes in the same thread pool

# --------------------------------------------------------------------------------
# Utilities

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

viewport_width = 1850
viewport_height = 1040
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
        self.source_image_size = 512  # THA3 uses 512x512 images, can't be changed...

        self.upscale = 2.0  # ...but the animator has a realtime super-resolution filter (anime4k). E.g. upscale=1.5 -> 768x768; upscale=2.0 -> 1024x1024.
        self.upscale_preset = "C"  # "A", "B" or "C"; these roughly correspond to the presets of Anime4K  https://github.com/bloc97/Anime4K/blob/master/md/GLSL_Instructions_Advanced.md
        self.upscale_quality = "high"  # "low": fast, acceptable quality; "high": slow, good quality
        self.image_size = int(self.upscale * self.source_image_size)  # final size in GUI (for pixel-perfect texture)

        self.target_fps = 25  # default 25; maybe better to lower this when upscaling (see the server's terminal output for available FPS)
        self.comm_format = "QOI"  # Frame format for video stream

        self.button_width = 300

        self.talking = False
        self.animator_running = True
        self.animator_settings = None  # not loaded yet

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
                    self.frame_size_statistics = RunningAverage()

                def position_please_standby_text():
                    # x0, y0 = raven_utils.get_widget_relative_pos("live_image", reference="main_window")
                    x0, y0 = raven_utils.get_widget_pos("live_image")
                    dpg.add_text("[No image loaded]", pos=(x0 + self.image_size / 2 - 60,
                                                           y0 + self.image_size / 2 - (font_size / 2)),
                                 tag="please_standby_text",
                                 parent="live_texture_group",
                                 show=False)
                dpg.set_frame_callback(10, position_please_standby_text)

                with dpg.group(horizontal=True):
                    with dpg.group(horizontal=False):
                        dpg.add_text("Load / save")
                        dpg.add_button(label="Load image [Ctrl+O]", width=self.button_width, callback=show_open_image_dialog, tag="open_image_button")
                        dpg.add_button(label="Load emotion templates [Ctrl+Shift+E]", width=self.button_width, callback=show_open_json_dialog, tag="open_json_button")
                        dpg.add_text("[Use raven.avatar.editor to edit templates.]", color=(140, 140, 140))
                        dpg.add_spacer(height=4)
                        dpg.add_button(label="Load animator settings [Ctrl+Shift+A]", width=self.button_width, callback=show_open_animator_settings_dialog, tag="open_animator_settings_button")
                        dpg.add_button(label="Save animator settings", width=self.button_width, tag="save_animator_settings_button")  # TODO: implement
                        dpg.add_spacer(height=8)

                        dpg.add_text("Emotion [Ctrl+E]")
                        self.emotion_names = client_api.classify_labels()
                        if "neutral" in self.emotion_names:
                            self.emotion_names.remove("neutral")
                            self.emotion_names = ["neutral"] + self.emotion_names
                        self.emotion_choice = dpg.add_combo(items=self.emotion_names,
                                                            default_value=self.emotion_names[0],
                                                            width=self.button_width,
                                                            callback=self.on_send_emotion)
                        self.on_send_emotion(sender=self.emotion_choice, app_data=self.emotion_names[0])  # initial emotion upon app startup; should be "neutral"
                        dpg.add_spacer(height=8)

                        dpg.add_text("Toggles")
                        dpg.add_button(label="Start talking [Ctrl+T]", width=self.button_width, callback=self.toggle_talking, tag="start_stop_talking_button")
                        dpg.add_button(label="Pause animator [Ctrl+P]", width=self.button_width, callback=self.toggle_animator_paused, tag="pause_resume_button")
                        dpg.add_spacer(height=8)

                        # AI speech synthesizer
                        tts_alive = client_api.tts_available()
                        if tts_alive:
                            heading_label = f"Voice [Ctrl+V] [{tts_url}]"
                            self.voice_names = client_api.tts_voices()
                        else:
                            heading_label = "Voice [Ctrl+V] [not connected]"
                            self.voice_names = ["[TTS server not available]"]
                        dpg.add_text(heading_label)
                        self.voice_choice = dpg.add_combo(items=self.voice_names,
                                                          default_value=self.voice_names[0],
                                                          width=self.button_width)
                        dpg.add_button(label="Speak [Ctrl+S]", width=self.button_width, callback=self.on_speak, enabled=tts_alive, tag="speak_button")
                        dpg.bind_item_theme("speak_button", "disablable_button_theme")
                        dpg.add_spacer(height=8)

                        # Upscaler settings editor
                        # TODO: implement the upscale slider (must re-generate GUI texture, and skip rendering in the background thread until the new "canvas" is ready)
                        dpg.add_text("Upscaler [Ctrl+click to set a numeric value]")
                        dpg.add_slider_int(label="x 0.1x", default_value=20, min_value=10, max_value=40, clamped=True, width=self.button_width, tag="upscale_slider")
                        self.upscale_presets = ["A", "B", "C"]
                        with dpg.group(horizontal=True):
                            dpg.add_combo(items=self.upscale_presets,
                                          default_value=self.upscale_preset,
                                          width=self.button_width,
                                          callback=self.on_upscaler_settings_change,
                                          tag="upscale_preset_choice")
                            dpg.add_text("Preset")
                        with dpg.group(horizontal=True):
                            self.upscale_qualities = ["low", "high"]
                            dpg.add_combo(items=self.upscale_qualities,
                                          default_value=self.upscale_quality,
                                          width=self.button_width,
                                          callback=self.on_upscaler_settings_change,
                                          tag="upscale_quality_choice")
                            dpg.add_text("Quality")
                        dpg.add_spacer(height=8)

                    dpg.add_spacer(width=8)

                    # Postprocessor settings editor
                    #
                    # NOTE: Defaults and ranges for postprocessor parameters are set in `postprocessor.py`.
                    #
                    def build_postprocessor_gui():
                        def make_reset_filter_callback(filter_name):  # freeze by closure
                            def reset_filter():
                                logger.warning(f"reset_filter: resetting '{filter_name}' to defaults.")
                                all_filters = dict(postprocessor.Postprocessor.get_filters())
                                defaults = all_filters[filter_name]["defaults"]  # all parameters, with their default values
                                ranges = all_filters[filter_name]["ranges"]  # for GUI hints
                                for param_name in defaults:
                                    param_range = ranges[param_name]
                                    if len(param_range) == 1 and param_range[0].startswith("!"):
                                        gui_hint = param_range[0]
                                        if gui_hint == "!ignore":
                                            continue
                                        elif gui_hint == "!RGB":
                                            pass  # no need for special processing
                                        else:
                                            logger.warning(f"reset_filter: '{filter_name}.{param_name}': unrecognized GUI hint '{gui_hint}', ignoring this parameter.")
                                            continue
                                    widget = self.filter_param_to_gui_widget(filter_name, param_name, defaults[param_name])
                                    if widget is None:
                                        logger.warning(f"reset_filter: '{filter_name}.{param_name}': unknown parameter type {type(default_value)}, skipping.")
                                    dpg.set_value(widget, defaults[param_name])
                                self.on_postprocessor_settings_change(None, None)
                            return reset_filter

                        def make_reset_param_callback(filter_name, param_name, default_value):  # freeze by closure
                            widget = self.filter_param_to_gui_widget(filter_name, param_name, default_value)
                            if widget is None:
                                logger.warning(f"make_reset_param_callback: '{filter_name}.{param_name}': Unknown parameter type {type(default_value)}, returning no-op callback.")
                                return lambda: None
                            def reset_param():
                                logger.warning(f"reset_param: resetting '{filter_name}.{param_name}' to default.")
                                dpg.set_value(widget, default_value)
                                self.on_postprocessor_settings_change(None, None)
                            return reset_param

                        def prettify(name):
                            pretty = name.replace("_", " ")
                            pretty = pretty[0].upper() + pretty[1:]
                            return pretty

                        for filter_name, param_info in postprocessor.Postprocessor.get_filters():
                            with dpg.group(horizontal=True):
                                dpg.add_button(label="Reset", tag=f"{filter_name}_reset_button", callback=make_reset_filter_callback(filter_name))
                                dpg.add_checkbox(label=prettify(filter_name), default_value=False,
                                                 tag=f"{filter_name}_checkbox", callback=self.on_postprocessor_settings_change)
                            with dpg.group(horizontal=True):
                                dpg.add_spacer(width=4)
                                with dpg.group(horizontal=False):
                                    for param_name, default_value in param_info["defaults"].items():
                                        param_range = param_info["ranges"][param_name]
                                        if len(param_range) == 1 and param_range[0].startswith("!"):  # GUI hint?
                                            gui_hint = param_range[0]
                                            if gui_hint == "!ignore":
                                                continue
                                            logger.warning(f"build_postprocessor_gui: {filter_name}.{param_name}': unrecognized GUI hint '{gui_hint}', ignoring this parameter.")
                                            continue  # TODO: implement color picker when the hint is "!RGB"

                                        # Create GUI control depending on parameter's type
                                        if isinstance(default_value, bool):
                                            with dpg.group(horizontal=True):
                                                dpg.add_button(label="X", tag=f"{filter_name}_{param_name}_reset_button", callback=make_reset_param_callback(filter_name, param_name, default_value))
                                                dpg.add_checkbox(label=prettify(param_name), default_value=default_value,
                                                                 tag=f"{filter_name}_{param_name}_checkbox", callback=self.on_postprocessor_settings_change)
                                        elif isinstance(default_value, float):
                                            assert len(param_range) == 2  # param_range = [min, max]
                                            with dpg.group(horizontal=True):
                                                dpg.add_button(label="X", tag=f"{filter_name}_{param_name}_reset_button", callback=make_reset_param_callback(filter_name, param_name, default_value))
                                                dpg.add_slider_float(label=prettify(param_name), default_value=default_value, min_value=param_range[0], max_value=param_range[1], clamped=True, width=self.button_width,
                                                                     tag=f"{filter_name}_{param_name}_slider", callback=self.on_postprocessor_settings_change)
                                        elif isinstance(default_value, int):
                                            assert len(param_range) == 2  # param_range = [min, max]
                                            with dpg.group(horizontal=True):
                                                dpg.add_button(label="X", tag=f"{filter_name}_{param_name}_reset_button", callback=make_reset_param_callback(filter_name, param_name, default_value))
                                                dpg.add_slider_int(label=prettify(param_name), default_value=default_value, min_value=param_range[0], max_value=param_range[1], clamped=True, width=self.button_width,
                                                                   tag=f"{filter_name}_{param_name}_slider", callback=self.on_postprocessor_settings_change)
                                        elif isinstance(default_value, str):
                                            # param_range = list of choices
                                            with dpg.group(horizontal=True):
                                                dpg.add_button(label="X", tag=f"{filter_name}_{param_name}_reset_button", callback=make_reset_param_callback(filter_name, param_name, default_value))
                                                dpg.add_combo(items=param_range,
                                                              default_value=param_range[0],
                                                              width=self.button_width,
                                                              tag=f"{filter_name}_{param_name}_choice", callback=self.on_postprocessor_settings_change)
                                                dpg.add_text(prettify(param_name))
                                        else:
                                            assert False, f"{filter_name}.{param_name}: Unknown parameter type {type(default_value)}"
                            dpg.add_spacer(height=4)

                    with dpg.group(horizontal=False):
                        dpg.add_text("Postprocessor [Ctrl+click to set a numeric value]")
                        # dpg.add_text("[For advanced setup, edit animator.json.]", color=(140, 140, 140))
                        build_postprocessor_gui()

    def filter_param_to_gui_widget(self, filter_name, param_name, example_value):
        """Given a postprocessor filter name, a name for one of its parameters, and an example value for that parameter, return the DPG tag for the corresponding GUI widget.

        If the type of `example_value` is not supported, return `None`.
        """
        if isinstance(example_value, bool):
            return f"{filter_name}_{param_name}_checkbox"
        elif isinstance(example_value, (float, int)):
            return f"{filter_name}_{param_name}_slider"
        elif isinstance(example_value, str):
            return f"{filter_name}_{param_name}_choice"
        return None
        raise ValueError(f"filter_param_to_gui_widget: Unknown value type {type(example_value)}.")

    def strip_postprocessor_chain_for_gui(self, postprocessor_chain):
        """Strip to what we can currently set up in the GUI. Fixed render order, with at most one copy of each filter."""
        input_dict = dict(postprocessor_chain)  # [(filter0, params0), ...] -> {filter0: params0, ...}, keep last copy of each
        gui_postprocessor_chain = []
        for filter_name, param_info in postprocessor.Postprocessor.get_filters():  # this performs the reordering
            if filter_name in input_dict:
                gui_postprocessor_chain.append((filter_name, input_dict[filter_name]))
        return gui_postprocessor_chain

    def canonize_postprocessor_parameters_for_gui(self, postprocessor_chain):
        """Auto-populate missing fields to their default values.

        Be sure to feed your postprocessor chain through `strip_postprocessor_chain_for_gui` first.
        """
        all_filters = dict(postprocessor.Postprocessor.get_filters())

        validated_postprocessor_chain = []
        for filter_name, filter_settings in postprocessor_chain:
            if filter_name not in all_filters:
                logger.warning(f"canonize_postprocessor_parameters_for_gui: Unknown filter '{filter_name}', ignoring.")
                continue
            defaults = all_filters[filter_name]["defaults"]  # all parameters, with their default values
            validated_settings = {}
            for param_name in defaults:
                validated_settings[param_name] = filter_settings.get("param_name", defaults[param_name])
            validated_postprocessor_chain.append((filter_name, validated_settings))
        return validated_postprocessor_chain

    def populate_gui_from_canonized_postprocessor_chain(self, postprocessor_chain):
        """Ordering: strip -> canonize -> populate GUI"""
        all_filters = dict(postprocessor.Postprocessor.get_filters())
        input_dict = dict(postprocessor_chain)  # [(filter0, params0), ...] -> {filter0: params0, ...}, keep last copy of each
        for filter_name in all_filters:
            if filter_name not in input_dict:
                dpg.set_value(f"{filter_name}_checkbox", False)
                continue  # parameter values in GUI don't matter if the filter is disabled
            dpg.set_value(f"{filter_name}_checkbox", True)
            for param_name, param_value in input_dict[filter_name].items():
                widget = self.filter_param_to_gui_widget(filter_name, param_name, param_value)
                if widget is None:
                    logger.warning(f"Unknown parameter type {type(param_value)}, ignoring this parameter.")
                    continue
                dpg.set_value(widget, param_value)

    def generate_postprocessor_chain_from_gui(self):
        """Return a postprocessor_chain representing the postprocessor settings currently in the GUI. For saving."""
        all_filters = dict(postprocessor.Postprocessor.get_filters())
        postprocessor_chain = []
        for filter_name in all_filters:
            if dpg.get_value(f"{filter_name}_checkbox") is False:
                continue
            defaults = all_filters[filter_name]["defaults"]  # all parameters, with their default values
            settings = {}
            for param_name in defaults:
                widget = self.filter_param_to_gui_widget(filter_name, param_name, defaults[param_name])
                if widget is None:
                    logger.warning(f"Unknown parameter type {type(defaults[param_name])}, ignoring this parameter.")
                    continue
                settings[param_name] = dpg.get_value(widget)
            postprocessor_chain.append((filter_name, settings))
        return postprocessor_chain

    def on_send_emotion(self, sender, app_data):  # GUI event handler
        # On clicking a choice in the combobox, `app_data` is that choice, but on arrow key, `app_data` is the keycode.
        logger.info(f"TalkingheadExampleGUI.on_send_emotion: sender = {sender}, app_data = {app_data}")
        self.current_emotion = dpg.get_value(self.emotion_choice)
        logger.info(f"TalkingheadExampleGUI.on_send_emotion: sending emotion '{self.current_emotion}'")
        client_api.talkinghead_set_emotion(self.current_emotion)

    def load_image(self, filename: Union[pathlib.Path, str]) -> None:
        try:
            client_api.talkinghead_load(filename)
        except Exception as exc:
            logger.error(f"TalkingheadExampleGUI.load_image: {type(exc)}: {exc}")
            traceback.print_exc()
            client_util.modal_dialog(window_title="Error",
                                     message=f"Could not load image '{filename}', reason {type(exc)}: {exc}",
                                     buttons=["Close"],
                                     ok_button="Close",
                                     cancel_button="Close",
                                     centering_reference_window=self.window)

    def load_json(self, filename: Union[pathlib.Path, str]) -> None:
        try:
            client_api.talkinghead_load_emotion_templates_from_file(filename)
        except Exception as exc:
            logger.error(f"TalkingheadExampleGUI.load_json: {type(exc)}: {exc}")
            traceback.print_exc()
            client_util.modal_dialog(window_title="Error",
                                     message=f"Could not load emotion templates JSON '{filename}', reason {type(exc)}: {exc}",
                                     buttons=["Close"],
                                     ok_button="Close",
                                     cancel_button="Close",
                                     centering_reference_window=self.window)

    def on_upscaler_settings_change(self, sender, app_data):
        """Update the upscaler status and send changes to server."""
        self.upscale = dpg.get_value("upscale_slider") / 10
        self.upscale_preset = dpg.get_value("upscale_preset_choice")
        self.upscale_quality = dpg.get_value("upscale_quality_choice")
        self.on_postprocessor_settings_change(sender, app_data)

    def on_postprocessor_settings_change(self, sender, app_data):
        """Send new postprocessor and upscaler settings to server whenever a value changes in the GUI.

        Requires a settings file to be loaded.
        """
        try:
            ppc = self.generate_postprocessor_chain_from_gui()
            self.animator_settings["postprocessor_chain"] = ppc

            ups = {"upscale": self.upscale,
                   "upscale_preset": self.upscale_preset,
                   "upscale_quality": self.upscale_quality}
            self.animator_settings.update(ups)

            client_api.talkinghead_load_animator_settings(self.animator_settings)
        except Exception as exc:
            logger.error(f"TalkingheadExampleGUI.on_postprocessor_settings_change: {type(exc)}: {exc}")
            traceback.print_exc()

    def load_animator_settings(self, filename: Union[pathlib.Path, str]) -> None:
        try:
            with open(filename, "r", encoding="utf-8") as json_file:
                animator_settings = json.load(json_file)

            ppc = animator_settings["postprocessor_chain"] if "postprocessor_chain" in animator_settings else {}
            ppc = self.strip_postprocessor_chain_for_gui(ppc)
            ppc = self.canonize_postprocessor_parameters_for_gui(ppc)
            self.populate_gui_from_canonized_postprocessor_chain(ppc)
            animator_settings["postprocessor_chain"] = ppc

            if "upscale" in animator_settings:
                self.upscale = animator_settings["upscale"]
                dpg.set_value("upscale_slider", int(self.upscale * 10))
            if "upscale_preset" in animator_settings:
                self.upscale_preset = animator_settings["upscale_preset"]
                dpg.set_value("upscale_preset", self.upscale_preset)
            if "upscale_quality" in animator_settings:
                self.upscale_quality = animator_settings["upscale_quality"]
                dpg.set_value("upscale_quality", self.upscale_quality)

            custom_animator_settings = {"format": self.comm_format,
                                        "target_fps": self.target_fps,
                                        "upscale": self.upscale,
                                        "upscale_preset": self.upscale_preset,
                                        "upscale_quality": self.upscale_quality}
            animator_settings.update(custom_animator_settings)  # setup overrides

            # Any missing keys are auto-populated from server defaults.
            client_api.talkinghead_load_animator_settings(animator_settings)
            self.animator_settings = animator_settings
        except Exception as exc:
            logger.error(f"TalkingheadExampleGUI.load_animator_settings: {type(exc)}: {exc}")
            traceback.print_exc()
            client_util.modal_dialog(window_title="Error",
                                     message=f"Could not load animator settings JSON '{filename}', reason {type(exc)}: {exc}",
                                     buttons=["Close"],
                                     ok_button="Close",
                                     cancel_button="Close",
                                     centering_reference_window=self.window)

    def toggle_talking(self) -> None:
        """Toggle the talkinghead's talking state."""
        if not self.talking:
            client_api.talkinghead_start_talking()
            dpg.set_item_label("start_stop_talking_button", "Stop talking [Ctrl+T]")
        else:
            client_api.talkinghead_stop_talking()
            dpg.set_item_label("start_stop_talking_button", "Start talking [Ctrl+T]")
        self.talking = not self.talking

    def toggle_animator_paused(self) -> None:
        """Pause or resume the animation. Pausing when the talkinghead won't be visible (e.g. minimized window) saves resources as new frames are not computed."""
        if self.animator_running:
            client_api.talkinghead_unload()
            dpg.set_value("please_standby_text", "[Animator is paused]")
            dpg.show_item("please_standby_text")
            dpg.hide_item("live_image")
            dpg.set_item_label("pause_resume_button", "Resume animator [Ctrl+P]")
        else:
            client_api.talkinghead_reload()
            dpg.hide_item("please_standby_text")
            dpg.show_item("live_image")
            dpg.set_item_label("pause_resume_button", "Pause animator [Ctrl+P]")
        self.animator_running = not self.animator_running

    def on_speak(self, sender, app_data) -> None:
        current_voice = dpg.get_value(self.voice_choice)
        client_api.tts_speak(voice=current_voice,
                             text="Testing the AI speech synthesizer.",  # TODO: GUI control to enter text to speak
                             start_callback=client_api.talkinghead_start_talking,
                             stop_callback=client_api.talkinghead_stop_talking)

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
        self.gen = client_api.talkinghead_result_feed(expected_format=gui_instance.comm_format)

    def is_running(self):
        return self.gen is not None

    def get_frame(self):
        return next(self.gen)  # next-gen lol

    def stop(self):
        self.gen.close()
        self.gen = None

def si_prefix(number):
    """Convert a number to SI format (1000 -> 1K).

    https://medium.com/@ryan_forrester_/getting-file-sizes-in-python-a-complete-guide-01293aaa68ef
    """
    if number < 1000:
        return f"{number:.2f}"
    for unit in ['', 'K', 'M', 'G', 'T', 'P']:
        if number < 1000:
            return f"{number:.2f} {unit}"
        number /= 1000
    return f"{number:.2f} E"

# We must continuously retrieve new frames as they become ready, so this runs in the background.
def update_live_texture(task_env) -> None:
    assert task_env is not None
    def describe_performance(gui, video_height, video_width):  # actual received video height/width of the frame being described
        if gui is None:
            return "RX (avg) -- B/s @ -- FPS; avg -- B per frame (--x--, -- px)"

        avg_fps = gui.fps_statistics.average()
        avg_bytes = int(gui.frame_size_statistics.average())
        pixels = video_height * video_width

        if gui.upscale != 1.0:
            upscale_str = f"up {gui.upscale_preset} {gui.upscale_quality[0].upper()}Q @{gui.upscale}x -> "
        else:
            upscale_str = ""

        return f"RX (avg) {si_prefix(avg_fps * avg_bytes)}B/s @ {avg_fps:0.2f} FPS; avg {si_prefix(avg_bytes)}B per frame ({upscale_str}{video_width}x{video_height}, {si_prefix(pixels)}px)"

    reader = ResultFeedReader()
    reader.start()
    try:
        while not task_env.cancelled:
            frame_start_time = time.time_ns()

            if gui_instance:
                if not gui_instance.animator_running and reader.is_running():
                    reader.stop()
                    dpg.set_value("fps_text", describe_performance(None, None, None))
                elif gui_instance.animator_running and not reader.is_running():
                    reader.start()

            if reader.is_running():
                image_data = reader.get_frame()
                gui_instance.frame_size_statistics.add_datapoint(len(image_data))
            if gui_instance is None or not reader.is_running():
                time.sleep(0.04)   # 1/25 s
                continue

            if gui_instance.comm_format == "QOI":
                image_rgba = qoi.decode(image_data)  # -> uint8 array of shape (h, w, c)
                # Don't crash if we get frames at a different size from what is expected. But log a warning, as software rescaling is slow.
                h, w = image_rgba.shape[:2]
                if w != gui_instance.image_size or h != gui_instance.image_size:
                    logger.warning(f"update_live_texture: Got frame at wrong (old?) size {w}x{h}; slow CPU resizing to {gui_instance.image_size}x{gui_instance.image_size}")
                    pil_image = PIL.Image.fromarray(np.uint8(image_rgba[:, :, :3]))
                    if image_rgba.shape[2] == 4:
                        alpha_channel = image_rgba[:, :, 3]
                        pil_image.putalpha(PIL.Image.fromarray(np.uint8(alpha_channel)))
                    pil_image = pil_image.resize((gui_instance.image_size, gui_instance.image_size),
                                                 resample=PIL.Image.LANCZOS)
                    image_rgba = np.asarray(pil_image.convert("RGBA"))
            else:  # use PIL
                image_file = io.BytesIO(image_data)
                pil_image = PIL.Image.open(image_file)
                # Don't crash if we get frames at a different size from what is expected. But log a warning, as software rescaling is slow.
                w, h = pil_image.size
                if w != gui_instance.image_size or h != gui_instance.image_size:
                    logger.warning(f"update_live_texture: Got frame at wrong (old?) size {w}x{h}; slow CPU resizing to {gui_instance.image_size}x{gui_instance.image_size}")
                    pil_image = pil_image.resize((gui_instance.image_size, gui_instance.image_size),
                                                 resample=PIL.Image.LANCZOS)
                image_rgba = np.asarray(pil_image.convert("RGBA"))
            image_rgba = np.array(image_rgba, dtype=np.float32) / 255
            raw_data = image_rgba.ravel()  # shape [h, w, c] -> linearly indexed
            dpg.set_value(gui_instance.live_texture, raw_data)  # to GUI

            # Update FPS counter.
            # NOTE: Since we wait on the server to send a frame, the refresh is capped to the rate that data actually arrives at, i.e. the server's TARGET_FPS.
            #       If the machine could render faster, this just means less than 100% CPU/GPU usage.
            elapsed_time = time.time_ns() - frame_start_time
            fps = 1.0 / (elapsed_time / 10**9)
            gui_instance.fps_statistics.add_datapoint(fps)

            dpg.set_value("fps_text", describe_performance(gui_instance, h, w))
    except Exception as exc:
        logger.error(f"TalkingheadExampleGUI.update_live_texture: {type(exc)}: {exc}")

        # TODO: recovery if the server comes back online
        if gui_instance is not None:
            gui_instance.animator_running = False
            dpg.set_value("please_standby_text", "[Connection lost]")
            dpg.show_item("please_standby_text")
            dpg.hide_item("live_image")
            dpg.set_value("fps_text", describe_performance(None, None, None))


# --------------------------------------------------------------------------------
# Main program

if __name__ == "__main__":
    client_api.talkinghead_load_emotion_templates({})  # send empty dict -> reset to server defaults

    gui_instance = TalkingheadExampleGUI()  # will load animator settings

    client_api.talkinghead_load("example.png")  # this will also start the animator if it was paused

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

    # Load default animator settings from disk.
    #
    # We must defer loading the animator settings until after the GUI has been rendered at least once,
    # so that the GUI controls for the postprocessor are available, and so that if there are any issues
    # during loading, we can open a modal dialog.
    def _load_initial_animator_settings():
        animator_json_path = os.path.join(os.path.dirname(__file__), "animator.json")
        gui_instance.load_animator_settings(animator_json_path)
    dpg.set_frame_callback(2, _load_initial_animator_settings)

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

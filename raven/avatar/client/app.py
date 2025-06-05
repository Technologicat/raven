"""Avatar client.

!!! Start `raven.avatar.server.app` first before running this app! !!!

This GUI app is an editor for the avatar's postprocessor settings, and a live test environment for your characters.
To edit the emotion templates, see the separate app `raven.avatar.pose_editor.app`.

This module is licensed under the 2-clause BSD license.
"""

# nice to have (maybe later):
#
# TODO: robustness: don't crash if the server suddenly goes down
# TODO: support non-square avatar video stream (after server-side crop filter); should get image width/height from video stream

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import concurrent.futures
import copy
import io
import json
import os
import pathlib
import platform
import sys
import threading
import time
import traceback
from typing import Optional, Tuple, Union

import qoi
import PIL.Image

from colorama import Fore, Style, init as colorama_init
from unpythonic.env import env as envcls

import numpy as np

colorama_init()

# WORKAROUND: Deleting a texture or image widget causes DPG to segfault on Nvidia/Linux.
# https://github.com/hoffstadt/DearPyGui/issues/554
if platform.system().upper() == "LINUX":
    os.environ["__GLVND_DISALLOW_PATCHING"] = "1"

import dearpygui.dearpygui as dpg

from ...vendor.file_dialog.fdialog import FileDialog  # https://github.com/totallynotdrait/file_dialog, but with custom modifications
from ...common import animation  # Raven's GUI animation system, nothing to do with the AI avatar.
from ...common import bgtask
from ...common import guiutils

from ..common import config as common_config
from ..common.postprocessor import Postprocessor  # so we can query the filters (TODO: add a web API to get them from the actual running server?)
from ..common.running_average import RunningAverage

from . import config as client_config
from . import api  # convenient Python functions that abstract away the web API

# ----------------------------------------
# Module bootup

bg = concurrent.futures.ThreadPoolExecutor()
task_manager = bgtask.TaskManager(name="avatar_client",
                                  mode="concurrent",
                                  executor=bg)
api.init_module(avatar_url=client_config.avatar_url,
                avatar_api_key_file=client_config.avatar_api_key_file,
                tts_url=client_config.tts_url,
                tts_api_key_file=client_config.tts_api_key_file,
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
    with dpg.font(os.path.join(os.path.dirname(__file__), "..", "..", "fonts", "OpenSans-Regular.ttf"),  # load font from Raven's main assets
                  font_size) as default_font:
        guiutils.setup_font_ranges()
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

viewport_width = 1900
viewport_height = 980
dpg.create_viewport(title="Raven-avatar",
                    width=viewport_width,
                    height=viewport_height)  # OS window (DPG "viewport")
dpg.setup_dearpygui()

# --------------------------------------------------------------------------------
# File dialog init

gui_instance = None  # initialized later, when the app starts

filedialog_open_input_image = None
filedialog_open_backdrop_image = None
filedialog_open_json = None
filedialog_open_animator_settings = None
filedialog_save_animator_settings = None

def initialize_filedialogs():  # called at app startup
    """Create the file dialogs."""
    global filedialog_open_input_image
    global filedialog_open_backdrop_image
    global filedialog_open_json
    global filedialog_open_animator_settings
    global filedialog_save_animator_settings
    cwd = os.getcwd()  # might change during filedialog init
    filedialog_open_input_image = FileDialog(title="Open input image",
                                             tag="open_input_image_dialog",
                                             callback=_open_input_image_callback,
                                             modal=True,
                                             filter_list=[".png"],
                                             file_filter=".png",
                                             multi_selection=False,
                                             allow_drag=False,
                                             default_path=os.path.join(os.path.dirname(__file__), "..", "images"))
    filedialog_open_backdrop_image = FileDialog(title="Open backdrop image",
                                                tag="open_backdrop_image_dialog",
                                                callback=_open_backdrop_image_callback,
                                                modal=True,
                                                filter_list=[".png", ".jpg"],
                                                file_filter=".png",
                                                multi_selection=False,
                                                allow_drag=False,
                                                default_path=os.path.join(os.path.dirname(__file__), "..", "backdrops"))
    filedialog_open_json = FileDialog(title="Open emotion JSON file",
                                       tag="open_json_dialog",
                                       callback=_open_json_callback,
                                       modal=True,
                                       filter_list=[".json"],
                                       file_filter=".json",
                                       multi_selection=False,
                                       allow_drag=False,
                                       default_path=os.path.join(os.path.dirname(__file__), "..", "emotions"))
    filedialog_open_animator_settings = FileDialog(title="Open animator settings JSON file",
                                                   tag="open_animator_settings_dialog",
                                                   callback=_open_animator_settings_callback,
                                                   modal=True,
                                                   filter_list=[".json"],
                                                   file_filter=".json",
                                                   multi_selection=False,
                                                   allow_drag=False,
                                                   default_path=os.path.join(os.path.dirname(__file__), ".."))
    filedialog_save_animator_settings = FileDialog(title="Save animator settings JSON file",
                                                   tag="save_animator_settings_dialog",
                                                   callback=_save_animator_settings_callback,
                                                   modal=True,
                                                   filter_list=[".json"],
                                                   file_filter=".json",
                                                   save_mode=True,
                                                   default_file_extension=".json",  # used if the user does not provide a file extension when naming the save-as
                                                   allow_drag=False,
                                                   default_path=cwd)

# --------------------------------------------------------------------------------
# "Open input image" dialog

def show_open_input_image_dialog():
    """Button callback. Show the open file dialog, for the user to pick an input image (avatar) to open.

    If you need to close it programmatically, call `filedialog_open_input_image.cancel()` so it'll trigger the callback.
    """
    if gui_instance is None:
        return
    logger.debug("show_open_input_image_dialog: Showing open file dialog.")
    filedialog_open_input_image.show_file_dialog()
    logger.debug("show_open_input_image_dialog: Done.")

def _open_input_image_callback(selected_files):
    """Callback that fires when the open input image dialog closes."""
    logger.debug("_open_input_image_callback: Open file dialog callback triggered.")
    if len(selected_files) > 1:  # Should not happen, since we set `multi_selection=False`.
        raise ValueError(f"Expected at most one selected file, got {len(selected_files)}.")
    if selected_files:
        selected_file = selected_files[0]
        logger.debug(f"_open_input_image_callback: User selected the file '{selected_file}'.")
        gui_instance.load_input_image(selected_file)
    else:  # empty selection -> cancelled
        logger.debug("_open_input_image_callback: Cancelled.")

def is_open_input_image_dialog_visible():
    """Return whether the open input image dialog is open.

    We have this abstraction (not just `dpg.is_item_visible`) because the window might not exist yet.
    """
    if filedialog_open_input_image is None:
        return False
    return dpg.is_item_visible("open_input_image_dialog")  # tag

# --------------------------------------------------------------------------------
# "Open backdrop image" dialog

def show_open_backdrop_image_dialog():
    """Button callback. Show the open file dialog, for the user to pick a backdrop image to open.

    If you need to close it programmatically, call `filedialog_open_backdrop_image.cancel()` so it'll trigger the callback.
    """
    if gui_instance is None:
        return
    logger.debug("show_open_backdrop_image_dialog: Showing open file dialog.")
    filedialog_open_backdrop_image.show_file_dialog()
    logger.debug("show_open_backdrop_image_dialog: Done.")

def _open_backdrop_image_callback(selected_files):
    """Callback that fires when the open backdrop image dialog closes."""
    logger.debug("_open_backdrop_image_callback: Open file dialog callback triggered.")
    if len(selected_files) > 1:  # Should not happen, since we set `multi_selection=False`.
        raise ValueError(f"Expected at most one selected file, got {len(selected_files)}.")
    if selected_files:
        selected_file = selected_files[0]
        logger.debug(f"_open_backdrop_image_callback: User selected the file '{selected_file}'.")
        gui_instance.load_backdrop_image(selected_file)
    else:  # empty selection -> cancelled
        logger.debug("_open_backdrop_image_callback: Cancelled.")

def is_open_backdrop_image_dialog_visible():
    """Return whether the open backdrop image dialog is open.

    We have this abstraction (not just `dpg.is_item_visible`) because the window might not exist yet.
    """
    if filedialog_open_backdrop_image is None:
        return False
    return dpg.is_item_visible("open_backdrop_image_dialog")  # tag

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
# "Save JSON" dialog (animator settings)

def show_save_animator_settings_dialog():
    """Button callback. Show the save file dialog, for the user to pick a filename to save as.

    If you need to close it programmatically, call `filedialog_save_animator_settings.cancel()` so it'll trigger the callback.
    """
    if gui_instance is None:
        return
    logger.debug("show_save_animator_settings_dialog: Showing save file dialog.")
    filedialog_save_animator_settings.show_file_dialog()
    logger.debug("show_save_animator_settings_dialog: Done.")

def _save_animator_settings_callback(selected_files):
    """Callback that fires when the save image dialog closes."""
    logger.debug("_save_animator_settings_callback: Save file dialog callback triggered.")
    if len(selected_files) > 1:  # Should not happen, since we set `multi_selection=False`.
        raise ValueError(f"Expected at most one selected file, got {len(selected_files)}.")
    if selected_files:
        selected_file = selected_files[0]
        logger.debug(f"_save_animator_settings_callback: User selected the file '{selected_file}'.")
        gui_instance.save_animator_settings(selected_file)
    else:  # empty selection -> cancelled
        logger.debug("_save_animator_settings_callback: Cancelled.")

def is_save_animator_settings_dialog_visible():
    """Return whether the open image dialog is open.

    We have this abstraction (not just `dpg.is_item_visible`) because the window might not exist yet.
    """
    if filedialog_save_animator_settings is None:
        return False
    return dpg.is_item_visible("save_animator_settings_dialog")  # tag

# --------------------------------------------------------------------------------
# GUI controls

def is_any_modal_window_visible():
    """Return whether *some* modal window is open.

    Currently these are file dialogs.
    """
    return (is_open_input_image_dialog_visible() or is_open_json_dialog_visible() or
            is_animator_settings_dialog_visible() or is_save_animator_settings_dialog_visible())

class PostprocessorSettingsEditorGUI:
    """Main app window for the postprocessor settings editor for `raven.avatar`."""
    def __init__(self):
        self.source_image_size = 512  # THA3 uses 512x512 images, can't be changed...

        self.postprocessor_enabled = True

        self.upscale = 2.0  # ...but the animator has a realtime super-resolution filter (anime4k). E.g. upscale=1.5 -> 768x768; upscale=2.0 -> 1024x1024.
        self.upscale_preset = "C"  # "A", "B" or "C"; these roughly correspond to the presets of Anime4K  https://github.com/bloc97/Anime4K/blob/master/md/GLSL_Instructions_Advanced.md
        self.upscale_quality = "low"  # "low": fast, acceptable quality; "high": slow, good quality
        self.image_size = int(self.upscale * self.source_image_size)  # final size in GUI (for pixel-perfect texture)

        self.target_fps = 25  # default 25; maybe better to lower this when upscaling (see the server's terminal output for available FPS)
        self.comm_format = "QOI"  # Frame format for video stream

        self.button_width = 300

        self.upscale_change_lock = threading.Lock()
        self.live_texture = None  # The raw texture object
        self.live_texture_id_counter = 0  # For creating unique DPG IDs when the size changes on the fly, since the delete might not take immediately.
        self.live_image_widget = None  # GUI widget the texture renders to
        self.last_image_rgba = None  # For rescaling last received frame on upscaler size change before we get new data

        self.backdrop_texture = None  # The raw texture object
        self.backdrop_texture_id_counter = 0
        self.backdrop_image = None  # PIL image
        self.last_backdrop_image = None
        self.last_window_size = (None, None)

        self.talking_animation_running = False  # simple mouth randomizing animation
        self.speaking = False  # TTS
        self.animator_running = True
        self.animator_settings = None  # not loaded yet

        dpg.add_texture_registry(tag="talkinghead_example_textures")  # the DPG live texture and the window backdrop texture will be stored here
        dpg.set_viewport_title(f"Raven-avatar [{client_config.avatar_url}]")

        with dpg.window(tag="talkinghead_main_window",
                        label="Raven-avatar main window") as self.window:  # label not actually shown, since this window is maximized to the whole viewport
            with dpg.group(horizontal=True):
                # We can use a borderless child window as a fixed-size canvas that crops anything outside it (instead of automatically showing a scrollbar).
                # DPG adds its theme's margins, which in our case is 8 pixels of padding per side, hence the -16 to exactly cover the viewport's actually available height.
                with dpg.child_window(width=1024, height=viewport_height - 16,
                                      border=False, no_scrollbar=True, no_scroll_with_mouse=True,
                                      tag="avatar_child_window"):
                    dpg.add_drawlist(tag="backdrop_drawlist", width=1024, height=1024, pos=(0, 0))  # for backdrop image
                    # dpg.add_spacer(width=1024, height=0)  # keep the group at the image's width even when the image is hidden
                    self.init_live_texture(self.image_size)
                    dpg.add_text("FPS counter will appear here", color=(0, 255, 0), pos=(8, 0), tag="fps_text")
                    self.fps_statistics = RunningAverage()
                    self.frame_size_statistics = RunningAverage()

                def position_please_standby_text():
                    # x0, y0 = guiutils.get_widget_relative_pos(f"live_image_{self.live_texture_id_counter}", reference="main_window")
                    x0, y0 = guiutils.get_widget_pos(f"live_image_{self.live_texture_id_counter}")
                    dpg.add_text("[No image loaded]", pos=(x0 + self.image_size / 2 - 60,
                                                           y0 + self.image_size / 2 - (font_size / 2)),
                                 tag="please_standby_text",
                                 parent="avatar_child_window",
                                 show=False)
                dpg.set_frame_callback(10, position_please_standby_text)

                with dpg.child_window(width=self.button_width + 16, autosize_y=True):
                    dpg.add_button(label="Fullscreen/windowed [F11]", width=self.button_width, callback=toggle_fullscreen, tag="fullscreen_button")
                    dpg.add_spacer(height=8)

                    dpg.add_button(label="Load image [Ctrl+O]", width=self.button_width, callback=show_open_input_image_dialog, tag="open_image_button")
                    with dpg.group(horizontal=True):
                        def reset_backdrop():
                            self.load_backdrop_image(None)
                        dpg.add_button(label="X", callback=reset_backdrop, tag="backdrop_reset_button")
                        dpg.add_button(label="Load backdrop [Ctrl+B]", width=self.button_width - 25, callback=show_open_backdrop_image_dialog, tag="open_backdrop_button")
                    dpg.add_button(label="Load emotion templates [Ctrl+Shift+E]", width=self.button_width, callback=show_open_json_dialog, tag="open_json_button")
                    dpg.add_text("[Use raven.avatar.editor to edit templates.]", color=(140, 140, 140))

                    # Main animator settings
                    dpg.add_text("Animator [Ctrl+click to set a numeric value]")
                    with dpg.group(horizontal=True):
                        def reset_target_fps():
                            dpg.set_value("target_fps_slider", 25)
                            self.on_gui_settings_change(None, None)
                        dpg.add_button(label="X", callback=reset_target_fps, tag="target_fps_reset_button")
                        dpg.add_slider_int(label="FPS", default_value=25, min_value=10, max_value=60, clamped=True, width=self.button_width - 80,
                                           callback=self.on_gui_settings_change, tag="target_fps_slider")
                    with dpg.group(horizontal=True):
                        def reset_pose_interpolator_step():
                            dpg.set_value("pose_interpolator_step_slider", 3)
                            self.on_gui_settings_change(None, None)
                        dpg.add_button(label="X", callback=reset_pose_interpolator_step, tag="pose_interpolator_step_reset_button")
                        dpg.add_slider_int(label="Speed", default_value=3, min_value=1, max_value=9, clamped=True, width=self.button_width - 80,
                                           callback=self.on_gui_settings_change, tag="pose_interpolator_step_slider")
                    dpg.add_button(label="Pause [Ctrl+P]", width=self.button_width, callback=self.toggle_animator_paused, tag="pause_resume_button")
                    dpg.add_button(label="Load settings [Ctrl+Shift+A]", width=self.button_width, callback=show_open_animator_settings_dialog, tag="open_animator_settings_button")
                    dpg.add_button(label="Save settings [Ctrl+Shift+S]", width=self.button_width, callback=show_save_animator_settings_dialog, tag="save_animator_settings_button")
                    dpg.add_spacer(height=8)

                    # Upscaler settings
                    dpg.add_text("Upscaler [Ctrl+click to set a numeric value]")
                    dpg.add_slider_int(label="x 0.1x", default_value=int(10 * self.upscale), min_value=10, max_value=20, clamped=True, width=self.button_width - 64,
                                       callback=self.on_upscaler_settings_change, tag="upscale_slider")
                    self.upscale_presets = ["A", "B", "C"]
                    with dpg.group(horizontal=True):
                        dpg.add_combo(items=self.upscale_presets,
                                      default_value=self.upscale_preset,
                                      width=self.button_width - 64,
                                      callback=self.on_upscaler_settings_change,
                                      tag="upscale_preset_choice")
                        dpg.add_text("Preset")
                    with dpg.group(horizontal=True):
                        self.upscale_qualities = ["low", "high"]
                        dpg.add_combo(items=self.upscale_qualities,
                                      default_value=self.upscale_quality,
                                      width=self.button_width - 64,
                                      callback=self.on_upscaler_settings_change,
                                      tag="upscale_quality_choice")
                        dpg.add_text("Quality")
                    dpg.add_text("[Presets as in Anime4K.]", color=(140, 140, 140))

                    # Separator for section with interactive demo controls
                    with dpg.drawlist(width=self.button_width, height=1):
                        dpg.draw_line((0, 0),
                                      (self.button_width, 0),
                                      color=(140, 140, 140, 255),
                                      thickness=1)
                    dpg.add_text("Test your character!")

                    # Interactive demo controls
                    dpg.add_text("Emotion [Ctrl+E]")
                    self.emotion_names = api.classify_labels()
                    if "neutral" in self.emotion_names:
                        self.emotion_names.remove("neutral")
                        self.emotion_names = ["neutral"] + self.emotion_names
                    self.emotion_choice = dpg.add_combo(items=self.emotion_names,
                                                        default_value=self.emotion_names[0],
                                                        width=self.button_width,
                                                        callback=self.on_send_emotion)
                    self.on_send_emotion(sender=self.emotion_choice, app_data=self.emotion_names[0])  # initial emotion upon app startup; should be "neutral"
                    dpg.add_spacer(height=8)

                    dpg.add_text("Talking animation (generic, non-lipsync)")
                    dpg.add_button(label="Start [Ctrl+T]", width=self.button_width, callback=self.toggle_talking, tag="start_stop_talking_button")
                    with dpg.group(horizontal=True):
                        def reset_talking_fps():
                            dpg.set_value("talking_fps_slider", 12)
                            self.on_gui_settings_change(None, None)
                        dpg.add_button(label="X", callback=reset_talking_fps, tag="talking_fps_reset_button")
                        dpg.add_slider_int(label="Talk FPS", default_value=12, min_value=6, max_value=24, clamped=True, width=self.button_width - 86,
                                           callback=self.on_gui_settings_change, tag="talking_fps_slider")
                    dpg.add_spacer(height=8)

                    # AI speech synthesizer
                    tts_alive = api.tts_available()
                    if tts_alive:
                        print(f"{Fore.GREEN}{Style.BRIGHT}Connected to TTS server at {client_config.tts_url}.{Style.RESET_ALL}")
                        print(f"{Fore.GREEN}{Style.BRIGHT}Speech synthesis is available.{Style.RESET_ALL}")
                        heading_label = f"Voice [Ctrl+V] [{client_config.tts_url}]"
                        self.voice_names = api.tts_voices()
                    else:
                        print(f"{Fore.YELLOW}{Style.BRIGHT}WARNING: Cannot connect to TTS server at {client_config.tts_url}.{Style.RESET_ALL} Is the TTS server running?")
                        print(f"{Fore.YELLOW}{Style.BRIGHT}Speech synthesis is NOT available.{Style.RESET_ALL}")
                        heading_label = "Voice [Ctrl+V] [not connected]"
                        self.voice_names = ["[TTS server not available]"]
                    dpg.add_text(heading_label)
                    self.voice_choice = dpg.add_combo(items=self.voice_names,
                                                      default_value=self.voice_names[0],
                                                      width=self.button_width)
                    with dpg.group(horizontal=True):
                        dpg.add_text("Speed")
                        dpg.add_button(label="X", tag="speak_speed_reset_button", callback=lambda: dpg.set_value("speak_speed_slider", 10))
                        dpg.add_slider_int(label="x 0.1x", default_value=10, min_value=5, max_value=20, clamped=True, width=self.button_width - 122,
                                           tag="speak_speed_slider")
                    dpg.add_spacer(height=4)
                    dpg.add_checkbox(label="Lip sync [adjust video timing below]", default_value=True, tag="speak_lipsync_checkbox")
                    dpg.add_slider_int(label="x 0.1 s", default_value=-8, min_value=-20, max_value=20, clamped=True, width=self.button_width - 64, tag="speak_video_offset")
                    dpg.add_spacer(height=4)
                    dpg.add_input_text(default_value="",
                                       hint="[Enter text to speak]",
                                       width=self.button_width,
                                       tag="speak_input_text")
                    dpg.add_button(label="Speak [Ctrl+S]", width=self.button_width, callback=self.on_start_speaking, enabled=tts_alive, tag="speak_button")
                    dpg.bind_item_theme("speak_button", "disablable_button_theme")
                    dpg.add_spacer(height=8)

                # Postprocessor settings editor
                #
                # NOTE: Defaults and ranges for postprocessor parameters are set in `postprocessor.py`.
                #
                def build_postprocessor_gui():
                    def make_reset_filter_callback(filter_name):  # freeze by closure
                        def reset_filter():
                            logger.info(f"reset_filter: resetting '{filter_name}' to defaults.")
                            all_filters = dict(Postprocessor.get_filters())
                            defaults = all_filters[filter_name]["defaults"]  # all parameters, with their default values
                            ranges = all_filters[filter_name]["ranges"]  # for GUI hints
                            for param_name in defaults:
                                param_range = ranges[param_name]
                                default_value = defaults[param_name]
                                if len(param_range) == 1 and param_range[0].startswith("!"):
                                    gui_hint = param_range[0]
                                    if gui_hint == "!ignore":
                                        continue
                                    elif gui_hint == "!RGB":
                                        default_value = [int(255 * x) for x in default_value]  # float -> uint8 for DPG color picker
                                    else:
                                        logger.warning(f"reset_filter: '{filter_name}.{param_name}': unrecognized GUI hint '{gui_hint}', ignoring this parameter.")
                                        continue
                                widget = self.filter_param_to_gui_widget(filter_name, param_name, default_value)
                                if widget is None:
                                    logger.warning(f"reset_filter: '{filter_name}.{param_name}': unknown parameter type {type(default_value)}, skipping.")
                                dpg.set_value(widget, default_value)
                            self.on_gui_settings_change(None, None)
                        return reset_filter

                    def make_reset_param_callback(filter_name, param_name, default_value):  # freeze by closure
                        widget = self.filter_param_to_gui_widget(filter_name, param_name, default_value)
                        if widget is None:
                            logger.warning(f"make_reset_param_callback: '{filter_name}.{param_name}': Unknown parameter type {type(default_value)}, returning no-op callback.")
                            return lambda: None
                        if self._iscolor(default_value):
                            default_value = [int(255 * x) for x in default_value]  # float -> uint8 for DPG color picker
                        def reset_param():
                            logger.info(f"reset_param: resetting '{filter_name}.{param_name}' to default.")
                            dpg.set_value(widget, default_value)
                            self.on_gui_settings_change(None, None)
                        return reset_param

                    def prettify(name):
                        pretty = name.replace("_", " ")
                        pretty = pretty[0].upper() + pretty[1:]
                        return pretty

                    for filter_name, param_info in Postprocessor.get_filters():
                        with dpg.group(horizontal=True):
                            dpg.add_button(label="Reset", tag=f"{filter_name}_reset_button", callback=make_reset_filter_callback(filter_name))
                            dpg.add_checkbox(label=prettify(filter_name), default_value=False,
                                             tag=f"{filter_name}_checkbox", callback=self.on_gui_settings_change)
                        with dpg.group(horizontal=True):
                            dpg.add_spacer(width=4)
                            with dpg.group(horizontal=False):
                                for param_name, default_value in param_info["defaults"].items():
                                    param_range = param_info["ranges"][param_name]
                                    if len(param_range) == 1 and param_range[0].startswith("!"):  # GUI hint?
                                        gui_hint = param_range[0]
                                        if gui_hint == "!ignore":  # don't add a GUI for this parameter, ok (e.g. filter name, since in the GUI, we have only one instance of each filter type)
                                            continue
                                        elif gui_hint == "!RGB":  # requested a color picker, ok
                                            pass
                                        else:
                                            logger.warning(f"build_postprocessor_gui: {filter_name}.{param_name}': unrecognized GUI hint '{gui_hint}', ignoring this parameter.")
                                            continue

                                    # Create GUI control depending on parameter's type
                                    if isinstance(default_value, bool):
                                        with dpg.group(horizontal=True):
                                            dpg.add_button(label="X", tag=f"{filter_name}_{param_name}_reset_button", callback=make_reset_param_callback(filter_name, param_name, default_value))
                                            dpg.add_checkbox(label=prettify(param_name), default_value=default_value,
                                                             tag=f"{filter_name}_{param_name}_checkbox", callback=self.on_gui_settings_change)
                                    elif isinstance(default_value, float):
                                        assert len(param_range) == 2  # param_range = [min, max]
                                        with dpg.group(horizontal=True):
                                            dpg.add_button(label="X", tag=f"{filter_name}_{param_name}_reset_button", callback=make_reset_param_callback(filter_name, param_name, default_value))
                                            dpg.add_slider_float(label=prettify(param_name), default_value=default_value, min_value=param_range[0], max_value=param_range[1], clamped=True, width=self.button_width,
                                                                 tag=f"{filter_name}_{param_name}_slider", callback=self.on_gui_settings_change)
                                    elif isinstance(default_value, int):
                                        assert len(param_range) == 2  # param_range = [min, max]
                                        with dpg.group(horizontal=True):
                                            dpg.add_button(label="X", tag=f"{filter_name}_{param_name}_reset_button", callback=make_reset_param_callback(filter_name, param_name, default_value))
                                            dpg.add_slider_int(label=prettify(param_name), default_value=default_value, min_value=param_range[0], max_value=param_range[1], clamped=True, width=self.button_width,
                                                               tag=f"{filter_name}_{param_name}_slider", callback=self.on_gui_settings_change)
                                    elif isinstance(default_value, str):
                                        # param_range = list of choices
                                        with dpg.group(horizontal=True):
                                            dpg.add_button(label="X", tag=f"{filter_name}_{param_name}_reset_button", callback=make_reset_param_callback(filter_name, param_name, default_value))
                                            dpg.add_combo(items=param_range,
                                                          default_value=param_range[0],
                                                          width=self.button_width,
                                                          tag=f"{filter_name}_{param_name}_choice", callback=self.on_gui_settings_change)
                                            dpg.add_text(prettify(param_name))
                                    elif self._iscolor(default_value):  # RGB or RGBA color
                                        # no param_range (it was used for the GUI hint)
                                        with dpg.group(horizontal=True):
                                            dpg.add_button(label="X", tag=f"{filter_name}_{param_name}_reset_button", callback=make_reset_param_callback(filter_name, param_name, default_value))
                                            dpg.add_text(prettify(param_name))
                                        dpg.add_color_picker(default_value=[int(x * 255) for x in default_value],  # float -> uint8 for DPG color picker
                                                             width=self.button_width + 54,
                                                             display_type=dpg.mvColorEdit_uint8, no_alpha=True, alpha_bar=False, no_side_preview=True,
                                                             tag=f"{filter_name}_{param_name}_color_picker", callback=self.on_gui_settings_change)

                                    else:
                                        assert False, f"{filter_name}.{param_name}: Unknown parameter type {type(default_value)}"
                        dpg.add_spacer(height=4)

                with dpg.child_window(autosize_x=True, autosize_y=True):
                    dpg.add_checkbox(label="Postprocessor [Ctrl+click to set a numeric value]", default_value=True, callback=self.on_toggle_postprocessor)
                    # dpg.add_text("[For advanced setup, edit animator.json.]", color=(140, 140, 140))
                    build_postprocessor_gui()

    def init_live_texture(self, new_image_size: int) -> None:
        """Initialize (or re-initialize) the texture and image widgets for rendering the video stream of the live AI avatar.

        The image has square aspect ratio;, `new_image_size` is the length of a side, in pixels.
        """
        with self.upscale_change_lock:
            old_texture_id = self.live_texture_id_counter
            new_texture_id = old_texture_id + 1

            logger.info(f"init_live_texture: Creating new GUI item live_texture_{new_texture_id} for new size {new_image_size}x{new_image_size}")
            self.blank_texture = np.zeros([new_image_size,  # height
                                           new_image_size,  # width
                                           4],  # RGBA
                                          dtype=np.float32).ravel()
            if self.last_image_rgba is not None:
                # To reduce flicker when the texture is replaced: take the last frame we have,
                # rescale it, and use that as the initial content of the new texture.
                image_rgba = self.last_image_rgba  # from the background thread
                pil_image = PIL.Image.fromarray(np.uint8(image_rgba[:, :, :3]))
                if image_rgba.shape[2] == 4:
                    alpha_channel = image_rgba[:, :, 3]
                    pil_image.putalpha(PIL.Image.fromarray(np.uint8(alpha_channel)))
                pil_image = pil_image.resize((new_image_size, new_image_size),
                                             resample=PIL.Image.LANCZOS)
                image_rgba = pil_image.convert("RGBA")
                image_rgba = np.asarray(image_rgba, dtype=np.float32) / 255
                default_image = image_rgba.ravel()
            else:
                default_image = self.blank_texture
            self.live_texture = dpg.add_raw_texture(width=new_image_size,
                                                    height=new_image_size,
                                                    default_value=default_image,
                                                    format=dpg.mvFormat_Float_rgba,
                                                    tag=f"live_texture_{new_texture_id}",
                                                    parent="talkinghead_example_textures")
            self.live_texture_id_counter += 1  # now the new texture exists so it's safe to write to (in the background thread)
            self.image_size = new_image_size

            first_time = (self.live_image_widget is None)
            logger.info(f"init_live_texture: Creating new GUI item live_image_{new_texture_id}")
            self.live_image_widget = dpg.add_image(f"live_texture_{new_texture_id}",
                                                   show=self.animator_running,  # if paused, leave it hidden
                                                   tag=f"live_image_{new_texture_id}",
                                                   parent="avatar_child_window",
                                                   before="fps_text")
            if first_time:  # first frame; window size not initialized yet, so we can't rely on `self._resize_gui`
                dpg.set_item_pos(self.live_image_widget, (512 - self.image_size // 2, viewport_height - self.image_size))
            else:
                self._resize_gui()

            try:
                dpg.hide_item(f"live_image_{old_texture_id}")
            except SystemError:  # does not exist
                pass
            else:
                dpg.split_frame()  # Only safe after startup, once the GUI render loop is running. At startup, the old image widget doesn't exist, so we detect the situation from that.
            # Now the old image widget is guaranteed to be hidden, so we can delete it without breaking GUI render
            guiutils.maybe_delete_item(f"live_image_{old_texture_id}")
            guiutils.maybe_delete_item(f"live_texture_{old_texture_id}")

            logger.info("init_live_texture: done!")

    def load_backdrop_image(self, filename: Optional[Union[pathlib.Path, str]]) -> None:
        """Load a backdrop image. To clear the background, use `filename=None`."""
        if filename is not None:
            self.backdrop_image = PIL.Image.open(filename)
        else:
            self.backdrop_image = None
        self._resize_gui()  # render the new backdrop

    def _resize_gui(self) -> None:
        """Window resize handler."""
        try:
            w, h = guiutils.get_widget_size(self.window)
        except SystemError:  # main window does not exist
            return
        if w == 0 or h == 0:  # no meaningful main window size yet?
            return

        try:
            if self.live_image_widget is not None:
                dpg.set_item_pos(self.live_image_widget, (512 - self.image_size // 2, h - self.image_size))
        except SystemError:  # main window or live image widget does not exist
            pass

        try:
            dpg.set_item_height("avatar_child_window", h - 16)
        except SystemError:  # main window or live image widget does not exist
            pass

        old_w, old_h = self.last_window_size
        old_texture_id = self.backdrop_texture_id_counter
        if self.backdrop_image is not None and (self.backdrop_image != self.last_backdrop_image or w != old_w or h != old_h):
            new_texture_id = old_texture_id + 1

            image_w, image_h = self.backdrop_image.size

            # TODO: Consider changing the backdrop region so that if we have enough space in the window for all the controls, it could take more than 1024 pixels of width.
            # TODO: If the backdrop image is small and/or has a wild aspect ratio, would be more efficient to cut first, then scale.
            #
            # Scale image, preserving aspect ratio, to cover the whole backdrop region (1024 x h)
            # https://stackoverflow.com/questions/1373035/how-do-i-scale-one-rectangle-to-the-maximum-size-possible-within-another-rectang
            scale = max(1024 / image_w, h / image_h)  # max(dst.w / src.w, dst.h / src.h)
            pil_image = self.backdrop_image.resize((int(scale * image_w), int(scale * image_h)),
                                                   resample=PIL.Image.LANCZOS)
            # Then cut the part we need
            pil_image = pil_image.crop(box=(0, 0, 1024, h))  # (left, upper, right, lower), in pixels

            image_rgba = pil_image.convert("RGBA")
            image_rgba = np.asarray(image_rgba, dtype=np.float32) / 255
            raw_data = image_rgba.ravel()

            logger.info(f"_resize_gui: Creating new GUI item backdrop_texture_{new_texture_id}")
            self.backdrop_texture = dpg.add_raw_texture(width=1024,
                                                        height=h,
                                                        default_value=raw_data,
                                                        format=dpg.mvFormat_Float_rgba,
                                                        tag=f"backdrop_texture_{new_texture_id}",
                                                        parent="talkinghead_example_textures")
            self.backdrop_texture_id_counter += 1
            dpg.delete_item("backdrop_drawlist", children_only=True)  # delete old draw items
            dpg.configure_item("backdrop_drawlist", width=1024, height=h)
            dpg.draw_image(f"backdrop_texture_{new_texture_id}", (0, 0), (1024, h), uv_min=(0, 0), uv_max=(1, 1), parent="backdrop_drawlist")
            guiutils.maybe_delete_item(f"backdrop_texture_{old_texture_id}")
        elif self.backdrop_image is None:
            dpg.delete_item("backdrop_drawlist", children_only=True)  # delete old draw items
            dpg.configure_item("backdrop_drawlist", width=1024, height=h)
            guiutils.maybe_delete_item(f"backdrop_texture_{old_texture_id}")

        self.last_backdrop_image = self.backdrop_image
        self.last_window_size = (w, h)

    def _iscolor(self, value) -> bool:
        """Return whether `value` is likely an RGB or RGBA color in float or uint8 format."""
        return isinstance(value, list) and 3 <= len(value) <= 4 and all(isinstance(x, (float, int)) for x in value)

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
        elif self._iscolor(example_value):
            return f"{filter_name}_{param_name}_color_picker"
        logger.warning(f"filter_param_to_gui_widget: Unknown value type {type(example_value)}.")
        return None

    def strip_postprocessor_chain_for_gui(self, postprocessor_chain):
        """Strip to what we can currently set up in the GUI. Fixed render order, with at most one copy of each filter."""
        input_dict = dict(postprocessor_chain)  # [(filter0, params0), ...] -> {filter0: params0, ...}, keep last copy of each
        gui_postprocessor_chain = []
        for filter_name, param_info in Postprocessor.get_filters():  # this performs the reordering
            if filter_name in input_dict:
                gui_postprocessor_chain.append((filter_name, input_dict[filter_name]))
        return gui_postprocessor_chain

    def canonize_postprocessor_parameters_for_gui(self, postprocessor_chain):
        """Auto-populate missing fields to their default values.

        Be sure to feed your postprocessor chain through `strip_postprocessor_chain_for_gui` first.
        """
        all_filters = dict(Postprocessor.get_filters())

        validated_postprocessor_chain = []
        for filter_name, filter_settings in postprocessor_chain:
            if filter_name not in all_filters:
                logger.warning(f"canonize_postprocessor_parameters_for_gui: Unknown filter '{filter_name}', ignoring.")
                continue
            defaults = all_filters[filter_name]["defaults"]  # all parameters, with their default values
            validated_settings = {}
            for param_name in defaults:
                validated_settings[param_name] = filter_settings.get(param_name, defaults[param_name])
            validated_postprocessor_chain.append((filter_name, validated_settings))
        return validated_postprocessor_chain

    def populate_gui_from_canonized_postprocessor_chain(self, postprocessor_chain):
        """Ordering: strip -> canonize -> populate GUI"""
        all_filters = dict(Postprocessor.get_filters())
        input_dict = dict(postprocessor_chain)  # [(filter0, params0), ...] -> {filter0: params0, ...}, keep last copy of each
        for filter_name in all_filters:
            if filter_name not in input_dict:
                dpg.set_value(f"{filter_name}_checkbox", False)
                continue  # parameter values in GUI don't matter if the filter is disabled
            dpg.set_value(f"{filter_name}_checkbox", True)
            for param_name, param_value in input_dict[filter_name].items():
                widget = self.filter_param_to_gui_widget(filter_name, param_name, param_value)
                if widget is None:
                    logger.warning(f"populate_gui_from_canonized_postprocessor_chain: Unknown parameter type {type(param_value)}, ignoring this parameter.")
                    continue
                if self._iscolor(param_value):
                    param_value = [int(x * 255) for x in param_value]  # float -> uint8 for DPG color picker
                dpg.set_value(widget, param_value)

    def generate_postprocessor_chain_from_gui(self):
        """Return a postprocessor_chain representing the postprocessor settings currently in the GUI. For saving."""
        all_filters = dict(Postprocessor.get_filters())
        postprocessor_chain = []
        for filter_name in all_filters:
            if dpg.get_value(f"{filter_name}_checkbox") is False:
                continue
            defaults = all_filters[filter_name]["defaults"]  # all parameters, with their default values
            settings = {}
            for param_name in defaults:
                widget = self.filter_param_to_gui_widget(filter_name, param_name, defaults[param_name])
                if widget is None:
                    logger.warning(f"generate_postprocessor_chain_from_gui: Unknown parameter type {type(defaults[param_name])}, ignoring this parameter.")
                    continue
                param_value = dpg.get_value(widget)
                if self._iscolor(param_value):
                    param_value = [x / 255.0 for x in param_value]  # uint8 -> float from DPG color picker
                    param_value = param_value[:3]  # currently the postprocessor settings have RGB colors only (no RGBA)
                settings[param_name] = param_value
            postprocessor_chain.append((filter_name, settings))
        return postprocessor_chain

    def on_send_emotion(self, sender, app_data):  # GUI event handler
        # On clicking a choice in the combobox, `app_data` is that choice, but on arrow key, `app_data` is the keycode.
        logger.info(f"PostprocessorSettingsEditorGUI.on_send_emotion: sender = {sender}, app_data = {app_data}")
        self.current_emotion = dpg.get_value(self.emotion_choice)
        logger.info(f"PostprocessorSettingsEditorGUI.on_send_emotion: sending emotion '{self.current_emotion}'")
        api.talkinghead_set_emotion(self.current_emotion)

    def load_input_image(self, filename: Union[pathlib.Path, str]) -> None:
        try:
            logger.info(f"PostprocessorSettingsEditorGUI.load_input_image: loading avatar image '{filename}'")
            api.talkinghead_load(filename)
        except Exception as exc:
            logger.error(f"PostprocessorSettingsEditorGUI.load_input_image: {type(exc)}: {exc}")
            traceback.print_exc()
            guiutils.modal_dialog(window_title="Error",
                                  message=f"Could not load image '{filename}', reason {type(exc)}: {exc}",
                                  buttons=["Close"],
                                  ok_button="Close",
                                  cancel_button="Close",
                                  centering_reference_window=self.window)

    def load_json(self, filename: Union[pathlib.Path, str]) -> None:
        try:
            logger.info(f"PostprocessorSettingsEditorGUI.load_json: loading emotion templates '{filename}'")
            api.talkinghead_load_emotion_templates_from_file(filename)
        except Exception as exc:
            logger.error(f"PostprocessorSettingsEditorGUI.load_json: {type(exc)}: {exc}")
            traceback.print_exc()
            guiutils.modal_dialog(window_title="Error",
                                  message=f"Could not load emotion templates JSON '{filename}', reason {type(exc)}: {exc}",
                                  buttons=["Close"],
                                  ok_button="Close",
                                  cancel_button="Close",
                                  centering_reference_window=self.window)

    def on_toggle_postprocessor(self, sender, app_data):
        self.postprocessor_enabled = not self.postprocessor_enabled
        self.on_gui_settings_change(sender, app_data)

    def on_upscaler_settings_change(self, sender, app_data):
        """Update the upscaler status and send changes to server."""
        old_image_size = self.image_size
        new_upscale = dpg.get_value("upscale_slider") / 10
        new_image_size = int(new_upscale * self.source_image_size)
        if new_image_size != old_image_size:
            self.init_live_texture(new_image_size)

        self.upscale = new_upscale
        self.upscale_preset = dpg.get_value("upscale_preset_choice")
        self.upscale_quality = dpg.get_value("upscale_quality_choice")
        self.on_gui_settings_change(sender, app_data)

    def on_gui_settings_change(self, sender, app_data):
        """Send new animator/upscaler/postprocessor settings to the avatar server whenever a value changes in the GUI.

        A settings file must have been loaded before calling this.
        """
        try:
            if self.animator_settings is None:
                raise RuntimeError("PostprocessorSettingsEditorGUI.on_gui_settings_change: no animator settings loaded, no base for update")
            # self.animator_settings is valid

            # Update the stuff that can be edited in the GUI:
            #
            # Postprocessor settings
            if self.postprocessor_enabled:
                ppc = self.generate_postprocessor_chain_from_gui()
            else:
                ppc = []
            self.animator_settings["postprocessor_chain"] = ppc

            # Upscaler settings, plus anything tracked by `PostprocessorSettingsEditorGUI`
            custom_animator_settings = {"format": self.comm_format,
                                        "target_fps": dpg.get_value("target_fps_slider"),
                                        "talking_fps": dpg.get_value("talking_fps_slider"),
                                        "pose_interpolator_step": dpg.get_value("pose_interpolator_step_slider") / 10,
                                        "upscale": self.upscale,
                                        "upscale_preset": self.upscale_preset,
                                        "upscale_quality": self.upscale_quality}
            self.animator_settings.update(custom_animator_settings)

            # Send to server
            api.talkinghead_load_animator_settings(self.animator_settings)
        except Exception as exc:
            logger.error(f"PostprocessorSettingsEditorGUI.on_gui_settings_change: {type(exc)}: {exc}")
            traceback.print_exc()

    def load_animator_settings(self, filename: Union[pathlib.Path, str]) -> None:
        """Load an animator settings JSON file and send the settings both to the GUI and to the avatar server."""
        try:
            logger.info(f"PostprocessorSettingsEditorGUI.load_animator_settings: loading '{filename}'")
            with open(filename, "r", encoding="utf-8") as json_file:
                animator_settings = json.load(json_file)

            ppc = animator_settings["postprocessor_chain"] if "postprocessor_chain" in animator_settings else {}
            ppc = self.strip_postprocessor_chain_for_gui(ppc)
            ppc = self.canonize_postprocessor_parameters_for_gui(ppc)
            self.populate_gui_from_canonized_postprocessor_chain(ppc)
            animator_settings["postprocessor_chain"] = ppc

            if "target_fps" in animator_settings:
                dpg.set_value("target_fps_slider", animator_settings["target_fps"])
            if "talking_fps" in animator_settings:
                dpg.set_value("talking_fps_slider", animator_settings["talking_fps"])
            if "pose_interpolator_step" in animator_settings:
                dpg.set_value("pose_interpolator_step_slider", int(animator_settings["pose_interpolator_step"] * 10))

            if "upscale" in animator_settings:
                self.upscale = animator_settings["upscale"]
                dpg.set_value("upscale_slider", int(self.upscale * 10))
            if "upscale_preset" in animator_settings:
                self.upscale_preset = animator_settings["upscale_preset"]
                dpg.set_value("upscale_preset_choice", self.upscale_preset)
            if "upscale_quality" in animator_settings:
                self.upscale_quality = animator_settings["upscale_quality"]
                dpg.set_value("upscale_quality_choice", self.upscale_quality)

            # Make sure these fields exist (in case they didn't yet).
            # They're not mandatory (any missing keys are always auto-populated from server defaults),
            # but they're something `PostprocessorSettingsEditorGUI` tracks, so we should sync our state to the server.
            custom_animator_settings = {"format": self.comm_format,
                                        "target_fps": dpg.get_value("target_fps_slider"),
                                        "talking_fps": dpg.get_value("talking_fps_slider"),
                                        "pose_interpolator_step": dpg.get_value("pose_interpolator_step_slider") / 10,
                                        "upscale": self.upscale,
                                        "upscale_preset": self.upscale_preset,
                                        "upscale_quality": self.upscale_quality}
            animator_settings.update(custom_animator_settings)

            # Send to server
            api.talkinghead_load_animator_settings(animator_settings)

            # ...and only if that is successful, remember the settings.
            self.animator_settings = animator_settings
        except Exception as exc:
            logger.error(f"PostprocessorSettingsEditorGUI.load_animator_settings: {type(exc)}: {exc}")
            traceback.print_exc()
            guiutils.modal_dialog(window_title="Error",
                                  message=f"Could not load animator settings JSON '{filename}', reason {type(exc)}: {exc}",
                                  buttons=["Close"],
                                  ok_button="Close",
                                  cancel_button="Close",
                                  centering_reference_window=self.window)

    def save_animator_settings(self, filename: Union[pathlib.Path, str]) -> None:
        """Save the current settings from the GUI into an animator settings JSON file."""
        # We have done any necessary preparations (sync GUI state to `self.animator_settings`)
        # when the settings were last sent to the server, via one of:
        #    `on_gui_settings_change`
        #    `load_animator_settings`
        #
        # Hence we can just save the JSON file.
        try:
            logger.info(f"PostprocessorSettingsEditorGUI.save_animator_settings: saving as '{filename}'")
            if self.animator_settings is None:
                raise RuntimeError("save_animator_settings: no animator settings loaded, nothing to save")
            with open(filename, "w", encoding="utf-8") as json_file:
                json.dump(self.animator_settings, json_file, indent=4)
        except Exception as exc:
            logger.error(f"PostprocessorSettingsEditorGUI.save_animator_settings: {type(exc)}: {exc}")
            traceback.print_exc()
            guiutils.modal_dialog(window_title="Error",
                                  message=f"Could not save animator settings JSON '{filename}', reason {type(exc)}: {exc}",
                                  buttons=["Close"],
                                  ok_button="Close",
                                  cancel_button="Close",
                                  centering_reference_window=self.window)

    def toggle_talking(self) -> None:
        """Toggle the talkinghead's talking state (simple randomized mouth animation)."""
        if not self.talking_animation_running:
            api.talkinghead_start_talking()
            dpg.set_item_label("start_stop_talking_button", "Stop [Ctrl+T]")
        else:
            api.talkinghead_stop_talking()
            dpg.set_item_label("start_stop_talking_button", "Start [Ctrl+T]")
        self.talking_animation_running = not self.talking_animation_running

    def toggle_animator_paused(self) -> None:
        """Pause or resume the animation. Pausing when the talkinghead won't be visible (e.g. minimized window) saves resources as new frames are not computed."""
        if self.animator_running:
            api.talkinghead_unload()
            dpg.set_value("please_standby_text", "[Animator is paused]")
            dpg.show_item("please_standby_text")
            dpg.hide_item(f"live_image_{self.live_texture_id_counter}")
            dpg.set_item_label("pause_resume_button", "Resume [Ctrl+P]")
        else:
            api.talkinghead_reload()
            dpg.hide_item("please_standby_text")
            dpg.show_item(f"live_image_{self.live_texture_id_counter}")
            dpg.set_item_label("pause_resume_button", "Pause [Ctrl+P]")
        self.animator_running = not self.animator_running

    def on_stop_speaking(self, sender, app_data) -> None:
        api.tts_stop()
        dpg.set_item_label("speak_button", "Start speaking [Ctrl+S]")
        dpg.set_item_callback("speak_button", self.on_start_speaking)
        self.speaking = False

    def on_start_speaking(self, sender, app_data) -> None:
        self.speaking = True
        dpg.set_item_label("speak_button", "Stop speaking [Ctrl+S]")
        dpg.set_item_callback("speak_button", self.on_stop_speaking)

        selected_voice = dpg.get_value(self.voice_choice)
        text = dpg.get_value("speak_input_text")
        if text == "":
            # text = "Testing the AI speech synthesizer."
            # text = '"Wait", I said, but the cat said "meow".'  # includes quotes
            # text = "INFO:raven.avatar.client.api:tts_speak_lipsynced.speak: starting"  # log message
            # text = 'close mouth only if the pause is at least half a second, else act like "!keep".'  # code comment
            # text = "Sharon Apple is a computer-generated virtual idol and a central character in the Macross Plus franchise, created by Shoji Kawamori."
            text = "Sharon Apple. Before Hatsune Miku, before VTubers, there was Sharon Apple. The digital diva of Macross Plus hailed from the in-universe mind of Myung Fang Lone, and sings tunes by legendary composer Yoko Kanno. Sharon wasn't entirely artificially intelligent, though: the unfinished program required Myung to patch in emotions during her concerts."
        if dpg.get_value("speak_lipsync_checkbox"):
            def stop_lipsync_speaking():
                self.on_stop_speaking(None, None)  # stop the TTS and update the GUI
            api.tts_speak_lipsynced(voice=selected_voice,
                                    text=text,
                                    speed=dpg.get_value("speak_speed_slider") / 10,
                                    video_offset=dpg.get_value("speak_video_offset") / 10,
                                    start_callback=None,  # no start callback needed, the TTS client will start lipsyncing once the audio starts.
                                    stop_callback=stop_lipsync_speaking)
        else:
            def start_nonlipsync_speaking():
                api.talkinghead_start_talking()
            def stop_nonlipsync_speaking():
                api.talkinghead_stop_talking()
                self.on_stop_speaking(None, None)  # stop the TTS and update the GUI
            api.tts_speak(voice=selected_voice,
                          text=text,
                          speed=dpg.get_value("speak_speed_slider") / 10,
                          start_callback=start_nonlipsync_speaking,
                          stop_callback=stop_nonlipsync_speaking)

def toggle_fullscreen():
    dpg.toggle_viewport_fullscreen()
    resize_gui()  # see below

def resize_gui():
    """Wait for the viewport size to actually change, then resize dynamically sized GUI elements.

    This is handy for toggling fullscreen, because the size changes at the next frame at the earliest.
    For the viewport resize callback, that one fires (*almost* always?) after the size has already changed.
    """
    logger.debug("resize_gui: Entered. Waiting for viewport size change.")
    if guiutils.wait_for_resize(gui_instance.window):
        _resize_gui()
    logger.debug("resize_gui: Done.")

def _resize_gui():
    if gui_instance is None:
        return
    gui_instance._resize_gui()
dpg.set_viewport_resize_callback(_resize_gui)

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
        elif key == dpg.mvKey_A:  # load animator settings
            show_open_animator_settings_dialog()
        elif key == dpg.mvKey_S:  # save animator settings
            show_save_animator_settings_dialog()

    # Ctrl+...
    elif ctrl_pressed:
        if key == dpg.mvKey_O:
            show_open_input_image_dialog()
        if key == dpg.mvKey_B:
            show_open_backdrop_image_dialog()
        elif key == dpg.mvKey_T:
            gui_instance.toggle_talking()
        elif key == dpg.mvKey_P:
            gui_instance.toggle_animator_paused()
        elif key == dpg.mvKey_E:
            dpg.focus_item(gui_instance.emotion_choice)
        elif key == dpg.mvKey_V:
            dpg.focus_item(gui_instance.voice_choice)
        elif key == dpg.mvKey_S:
            if not gui_instance.speaking:
                gui_instance.on_start_speaking(sender, app_data)
            else:
                gui_instance.on_stop_speaking(sender, app_data)

    # Bare key
    #
    # NOTE: These are global across the whole app (when no modal window is open) - be very careful here!
    else:
        if key == dpg.mvKey_F11:
            toggle_fullscreen()
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

    def start(self) -> None:
        self.gen = api.talkinghead_result_feed()

    def is_running(self) -> bool:
        return self.gen is not None

    def get_frame(self) -> Tuple[Optional[str], bytes]:
        """-> (received_mimetype, payload)"""
        return next(self.gen)  # next-gen lol

    def stop(self) -> None:
        self.gen.close()
        self.gen = None

def si_prefix(number: Union[int, float]) -> str:
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
    def describe_performance(gui: PostprocessorSettingsEditorGUI, video_format: str, video_height: int, video_width: int):  # actual received video height/width of the frame being described
        if gui is None:
            return "RX (avg) -- B/s @ -- FPS; avg -- B per frame (--x--, -- px, --)"

        avg_fps = gui.fps_statistics.average()
        avg_bytes = int(gui.frame_size_statistics.average())
        pixels = video_height * video_width

        if gui.upscale != 1.0:
            upscale_str = f"up {gui.upscale_preset} {gui.upscale_quality[0].upper()}Q @{gui.upscale}x -> "
        else:
            upscale_str = ""

        return f"RX (avg) {si_prefix(avg_fps * avg_bytes)}B/s @ {avg_fps:0.2f} FPS; avg {si_prefix(avg_bytes)}B per frame ({upscale_str}{video_width}x{video_height}, {si_prefix(pixels)}px, {video_format})"

    reader = ResultFeedReader()
    reader.start()
    try:
        while not task_env.cancelled:
            frame_start_time = time.time_ns()

            if gui_instance:
                if not gui_instance.animator_running and reader.is_running():
                    reader.stop()
                    try:
                        dpg.set_value("fps_text", describe_performance(None, None, None, None))
                    except SystemError:  # does not exist (can happen at app shutdown)
                        pass
                elif gui_instance.animator_running and not reader.is_running():
                    reader.start()

            if reader.is_running():
                mimetype, image_data = reader.get_frame()
                gui_instance.frame_size_statistics.add_datapoint(len(image_data))
            if gui_instance is None or not reader.is_running():
                time.sleep(0.04)   # 1/25 s
                continue

            try:  # EAFP to avoid TOCTTOU
                # Before blitting, make sure the texture is of the expected size. When an upscale change is underway, it will be temporarily of the wrong size.
                tex = gui_instance.live_texture  # Get the reference only once, since it could change at any time if the user changes the upscaler settings.
                config = dpg.get_item_configuration(tex)
                expected_w = config["width"]
                expected_h = config["height"]
            except SystemError:  # does not exist
                time.sleep(0.04)   # 1/25 s
                continue  # can't do anything without a texture to blit to, so discard this frame

            if mimetype == "image/qoi":
                image_rgba = qoi.decode(image_data)  # -> uint8 array of shape (h, w, c)
                # Don't crash if we get frames at a different size from what is expected. But log a warning, as software rescaling is slow.
                h, w = image_rgba.shape[:2]
                if w != expected_w or h != expected_h:
                    logger.warning(f"update_live_texture: Got frame at wrong (old?) size {w}x{h}; slow CPU resizing to {expected_w}x{expected_h}")
                    pil_image = PIL.Image.fromarray(np.uint8(image_rgba[:, :, :3]))
                    if image_rgba.shape[2] == 4:
                        alpha_channel = image_rgba[:, :, 3]
                        pil_image.putalpha(PIL.Image.fromarray(np.uint8(alpha_channel)))
                    pil_image = pil_image.resize((expected_w, expected_h),
                                                 resample=PIL.Image.LANCZOS)
                    image_rgba = np.asarray(pil_image.convert("RGBA"))
            else:  # use PIL
                image_file = io.BytesIO(image_data)
                pil_image = PIL.Image.open(image_file)
                # Don't crash if we get frames at a different size from what is expected. But log a warning, as software rescaling is slow.
                w, h = pil_image.size
                if w != expected_w or h != expected_h:
                    logger.warning(f"update_live_texture: Got frame at wrong (old?) size {w}x{h}; slow CPU resizing to {expected_w}x{expected_h}")
                    pil_image = pil_image.resize((expected_w, expected_h),
                                                 resample=PIL.Image.LANCZOS)
                image_rgba = np.asarray(pil_image.convert("RGBA"))
            gui_instance.last_image_rgba = image_rgba  # for reducing flicker when upscaler settings change
            image_rgba = np.array(image_rgba, dtype=np.float32) / 255
            raw_data = image_rgba.ravel()  # shape [h, w, c] -> linearly indexed
            try:  # EAFP to avoid TOCTTOU
                dpg.set_value(tex, raw_data)  # to GUI
            except SystemError:  # does not exist (might have gone bye-bye while we were decoding)
                continue  # can't do anything without a texture to blit to, so discard this frame

            # Update FPS counter.
            # NOTE: Since we wait on the server to send a frame, the refresh is capped to the rate that data actually arrives at, i.e. the server's TARGET_FPS.
            #       If the machine could render faster, this just means less than 100% CPU/GPU usage.
            elapsed_time = time.time_ns() - frame_start_time
            fps = 1.0 / (elapsed_time / 10**9)
            gui_instance.fps_statistics.add_datapoint(fps)

            try:
                dpg.set_value("fps_text", describe_performance(gui_instance, mimetype, h, w))
            except SystemError:  # does not exist (can happen at app shutdown)
                pass
    except Exception as exc:
        logger.error(f"PostprocessorSettingsEditorGUI.update_live_texture: {type(exc)}: {exc}")

        # TODO: recovery if the server comes back online
        if gui_instance is not None:
            gui_instance.animator_running = False
            dpg.set_value("please_standby_text", "[Connection lost]")
            dpg.show_item("please_standby_text")
            dpg.hide_item(f"live_image_{gui_instance.live_texture_id_counter}")
            dpg.set_value("fps_text", describe_performance(None, None, None, None))


# --------------------------------------------------------------------------------
# Main program

if api.avatar_available():
    print(f"{Fore.GREEN}{Style.BRIGHT}Connected to avatar server at {client_config.avatar_url}.{Style.RESET_ALL}")
else:
    print(f"{Fore.RED}{Style.BRIGHT}ERROR: Cannot connect to avatar server at {client_config.avatar_url}.{Style.RESET_ALL} Is the avatar server running?")
    sys.exit(255)

gui_instance = PostprocessorSettingsEditorGUI()  # will load animator settings

api.talkinghead_load_emotion_templates({})  # send empty dict -> reset emotion templates to server defaults
api.talkinghead_load(os.path.join(os.path.dirname(__file__), "..", "images", "example.png"))  # this will also start the animator if it was paused (TODO: feature orthogonality)

def shutdown() -> None:
    api.tts_stop()  # Stop the TTS speaking so that the speech background thread (if any) exits.
    task_manager.clear(wait=True)
    animation.animator.clear()
    global gui_instance
    gui_instance = None
dpg.set_exit_callback(shutdown)
task_manager.submit(update_live_texture, envcls())

dpg.set_primary_window(gui_instance.window, True)  # Make this DPG "window" occupy the whole OS window (DPG "viewport").
dpg.set_viewport_vsync(True)
dpg.show_viewport()

initialize_filedialogs()

# Load default animator settings from disk.
#
# We must defer loading the animator settings until after the GUI has been rendered at least once,
# so that the GUI controls for the postprocessor are available, and so that if there are any issues
# during loading, we can open a modal dialog.
def _load_initial_animator_settings():
    animator_json_path = os.path.join(os.path.dirname(__file__), "..", "animator.json")

    if not os.path.exists(animator_json_path):
        logger.info(f"_load_initial_animator_settings: Default animator settings file '{animator_json_path}' missing, writing a default config.")
        try:
            animator_settings = copy.copy(common_config.animator_defaults)
            custom_animator_settings = {"format": gui_instance.comm_format,
                                        "target_fps": gui_instance.target_fps,
                                        "upscale": gui_instance.upscale,
                                        "upscale_preset": gui_instance.upscale_preset,
                                        "upscale_quality": gui_instance.upscale_quality}
            animator_settings.update(custom_animator_settings)
            with open(animator_json_path, "w", encoding="utf-8") as json_file:
                json.dump(animator_settings, json_file, indent=4)
        except Exception as exc:
            logger.error(f"_load_initial_animator_settings: Failed to write default config, bailing out: {type(exc)}: {exc}")
            traceback.print_exc()
            raise

    gui_instance.load_animator_settings(animator_json_path)

    # gui_instance.load_backdrop_image(os.path.join(os.path.dirname(__file__), "..", "backdrops", "anime-plains.png"))  # DEBUG

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

def main():  # TODO: we don't really need this; it's just for console_scripts.
    pass

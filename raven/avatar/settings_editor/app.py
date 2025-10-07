"""Raven-avatar settings editor.

!!! Start `raven.server.app` first before running this app! !!!

This GUI app provides a standalone renderer for the AI avatar character, where you can edit
the avatar's postprocessor settings, and test your characters.

To edit the emotion templates, see the separate app `raven.avatar.pose_editor.app`.

This module is licensed under the 2-clause BSD license.
"""

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from ... import __version__

logger.info(f"Raven-avatar-settings-editor version {__version__} starting.")

logger.info("Loading libraries...")
from unpythonic import timer
with timer() as tim:
    import atexit
    import concurrent.futures
    import copy
    import json
    import os
    import pathlib
    import platform
    import requests
    import sys
    import threading
    import traceback
    from typing import Any, Dict, Optional, Union

    from colorama import Fore, Style, init as colorama_init

    colorama_init()

    # WORKAROUND: Deleting a texture or image widget causes DPG to segfault on Nvidia/Linux.
    # https://github.com/hoffstadt/DearPyGui/issues/554
    if platform.system().upper() == "LINUX":
        os.environ["__GLVND_DISALLOW_PATCHING"] = "1"

    import dearpygui.dearpygui as dpg

    from ...vendor.IconsFontAwesome6 import IconsFontAwesome6 as fa  # https://github.com/juliettef/IconFontCppHeaders
    from ...vendor.file_dialog.fdialog import FileDialog  # https://github.com/totallynotdrait/file_dialog, but with custom modifications

    from ...common import bgtask
    from ...common.gui import animation as gui_animation  # Raven's GUI animation system, nothing to do with the AI avatar.
    from ...common.gui import messagebox
    from ...common.gui import utils as guiutils
    from ...common import utils as common_utils

    from ...client import api  # convenient Python functions that abstract away the web API
    from ...client import config as client_config
    from ...client.avatar_controller import DPGAvatarController
    from ...client.avatar_renderer import DPGAvatarRenderer

    from ...server import config as server_config  # NOTE: default config (can be overridden on the command line when starting the server)
logger.info(f"Libraries loaded in {tim.dt:0.6g}s.")

# ----------------------------------------
# Module bootup

bg = concurrent.futures.ThreadPoolExecutor()
task_manager = bgtask.TaskManager(name="avatar_settings_editor",
                                  mode="concurrent",
                                  executor=bg)
api.initialize(raven_server_url=client_config.raven_server_url,
               raven_api_key_file=client_config.raven_api_key_file,
               tts_playback_audio_device=client_config.tts_playback_audio_device,
               executor=bg)  # reuse our executor so the TTS audio player goes in the same thread pool

# These are initialized later, when the app starts
gui_instance = None
avatar_instance_id = None

# --------------------------------------------------------------------------------
# DPG init

dpg.create_context()

themes_and_fonts = guiutils.bootup(font_size=20)

# animation for REC indicator (cyclic, runs in the background)
with dpg.theme(tag="my_pulsating_red_text_theme"):
    with dpg.theme_component(dpg.mvAll):
        pulsating_red_color = dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 96, 96))  # color-matching the rec button, "disablable_red_button_theme"
    pulsating_red_text_glow = gui_animation.PulsatingColor(cycle_duration=2.0,
                                                           theme_color_widget=pulsating_red_color)
    gui_animation.animator.add(pulsating_red_text_glow)

if platform.system().upper() == "WINDOWS":
    icon_ext = "ico"
else:
    icon_ext = "png"

viewport_width = 1900
viewport_height = 980
dpg.create_viewport(title="Raven-avatar settings editor",
                    small_icon=str(pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "..", "icons", f"app_128_notext.{icon_ext}")).expanduser().resolve()),
                    large_icon=str(pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "..", "icons", f"app_256.{icon_ext}")).expanduser().resolve()),
                    width=viewport_width,
                    height=viewport_height)  # OS window (DPG "viewport")
dpg.setup_dearpygui()

# --------------------------------------------------------------------------------
# File dialog init

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
    filedialog_open_input_image = FileDialog(title="Open character image",
                                             tag="open_input_image_dialog",
                                             callback=_open_input_image_callback,
                                             modal=True,
                                             filter_list=[".png"],
                                             file_filter=".png",
                                             multi_selection=False,
                                             allow_drag=False,
                                             default_path=pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "assets", "characters")).expanduser().resolve())
    filedialog_open_backdrop_image = FileDialog(title="Open backdrop image",
                                                tag="open_backdrop_image_dialog",
                                                callback=_open_backdrop_image_callback,
                                                modal=True,
                                                filter_list=[".png", ".jpg"],
                                                file_filter=".png",
                                                multi_selection=False,
                                                allow_drag=False,
                                                default_path=pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "assets", "backdrops")).expanduser().resolve())
    filedialog_open_json = FileDialog(title="Open emotion templates",
                                       tag="open_json_dialog",
                                       callback=_open_json_callback,
                                       modal=True,
                                       filter_list=[".json"],
                                       file_filter=".json",
                                       multi_selection=False,
                                       allow_drag=False,
                                       default_path=pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "assets", "emotions")).expanduser().resolve())
    filedialog_open_animator_settings = FileDialog(title="Open animator settings",
                                                   tag="open_animator_settings_dialog",
                                                   callback=_open_animator_settings_callback,
                                                   modal=True,
                                                   filter_list=[".json"],
                                                   file_filter=".json",
                                                   multi_selection=False,
                                                   allow_drag=False,
                                                   default_path=pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "assets", "settings")).expanduser().resolve())
    filedialog_save_animator_settings = FileDialog(title="Save animator settings",
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
# Avatar video recording

recording_output_dir = pathlib.Path("rec").expanduser().resolve()  # path relative to CWD

class AvatarVideoRecorder:
    """A small helper for dumping avatar video frames to disk as image files, in whatever format sent by server.

    The audio file is recorded separately, see the `on_audio_ready` event of the TTS.
    """
    def __init__(self):
        self.lock = threading.RLock()
        self.output_dir = recording_output_dir
        self.basename = "frame"
        self._reset()

    def _reset(self):
        with self.lock:
            self.frame_no = 0
            self.recording = False

    def start(self):
        """Start recording video frames to disk. Frames will be numbered starting from 00000."""
        logger.info(f"AvatarVideoRecorder.start: starting recording video frames to directory '{str(self.output_dir)}'.")
        common_utils.create_directory(self.output_dir)
        with self.lock:
            self._reset()
            self.recording = True

    def stop(self):
        """Stop recording video frames to disk."""
        with self.lock:
            if self.recording:
                logger.info(f"AvatarVideoRecorder.stop: stopping recording ({self.frame_no} video frames recorded).")
            self._reset()

    # This runs for each received frame (attached in `dpg_avatar_renderer.start`), but no-ops when recording is not active.
    def _on_frame_received(self, timestamp: int, mimetype: str, image_data: bytes) -> None:
        with self.lock:
            if not self.recording:
                return
            _, ext = mimetype.split("/")  # e.g. "image/qoi" -> ["image", "qoi"]
            filename = os.path.join(str(self.output_dir), f"{self.basename}_{self.frame_no:05d}.{ext}")
            logger.info(f"AvatarVideoRecorder._on_frame_received: recording video frame to '{filename}': time {timestamp}, mimetype {mimetype}.")
            with open(filename, "wb") as image_file:
                image_file.write(image_data)
            self.frame_no += 1
video_recorder = AvatarVideoRecorder()

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
        self.source_image_size = 512  # THA3 always uses 512x512 inputs...

        self.upscale = 2.0  # ...but the animator has a realtime super-resolution filter (anime4k). E.g. upscale=1.5 -> 768x768; upscale=2.0 -> 1024x1024.
        self.upscale_preset = "C"  # "A", "B" or "C"; these roughly correspond to the presets of Anime4K  https://github.com/bloc97/Anime4K/blob/master/md/GLSL_Instructions_Advanced.md
        self.upscale_quality = "low"  # "low": fast, acceptable quality; "high": slow, good quality

        self.postprocessor_enabled = True

        self.target_fps = 25  # default 25; maybe better to lower this when upscaling (see the server's terminal output for available FPS)
        self.comm_format = "QOI"  # Frame format for video stream

        self.button_width = 300

        self.current_input_image_path = None  # for the Refresh (reload current character) feature

        self.talking_animation_running = False  # simple mouth randomizing animation
        self.speaking = False  # TTS
        self.animator_settings = None  # not loaded yet

        dpg.add_texture_registry(tag="avatar_settings_editor_textures")  # the DPG live texture and the window backdrop texture will be stored here
        dpg.set_viewport_title(f"Raven-avatar settings editor [{client_config.raven_server_url}]")

        with dpg.window(tag="avatar_settings_editor_main_window",
                        label="Raven-avatar settings editor main window") as self.window:  # label not actually shown, since this window is maximized to the whole viewport
            with dpg.group(horizontal=True):
                # We can use a borderless child window as a fixed-size canvas that crops anything outside it (instead of automatically showing a scrollbar).
                # DPG adds its theme's margins, which in our case is 8 pixels of padding per side, hence the -16 to exactly cover the viewport's actually available height.
                with dpg.child_window(width=1024, height=viewport_height - 16,
                                      border=False, no_scrollbar=True, no_scroll_with_mouse=True,
                                      tag="avatar_child_window"):
                    self.dpg_avatar_renderer = DPGAvatarRenderer(gui_parent="avatar_child_window",
                                                                 avatar_x_center=512,
                                                                 avatar_y_bottom=viewport_height - 16,
                                                                 paused_text="[Animator is paused]",
                                                                 task_manager=task_manager)
                    image_size = int(self.upscale * self.source_image_size)
                    self.dpg_avatar_renderer.configure_live_texture(image_size)
                    self.dpg_avatar_renderer.configure_fps_counter(show=True)

                    with dpg.group(pos=(8, 32), show=False, horizontal=True) as self.recording_indicator_group:
                        dpg.add_text(fa.ICON_CIRCLE, tag="recording_symbol")
                        dpg.bind_item_font("recording_symbol", themes_and_fonts.icon_font_solid)  # tag
                        dpg.bind_item_theme("recording_symbol", "my_pulsating_red_text_theme")  # tag
                        dpg.add_text("REC", tag="recording_text")

                with dpg.child_window(width=self.button_width + 16, autosize_y=True):
                    dpg.add_button(label="Fullscreen/windowed [F11]", width=self.button_width, callback=toggle_fullscreen, tag="fullscreen_button")
                    dpg.add_spacer(height=8)

                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Load character [Ctrl+O]", width=self.button_width - 67, callback=show_open_input_image_dialog, tag="open_image_button")
                        dpg.add_tooltip("open_image_button", tag="open_image_tooltip")  # tag
                        dpg.add_text("Load a character image (512x512 RGBA PNG)", parent="open_image_tooltip")  # tag

                        dpg.add_button(label="Refresh [Ctrl+R]", width=59, callback=self.on_reload_input_image, tag="reload_image_button")
                        dpg.add_tooltip("reload_image_button", tag="reload_image_tooltip")  # tag
                        dpg.add_text("Refresh the current character image from disk", parent="reload_image_tooltip")  # tag
                    with dpg.group(horizontal=True):
                        def reset_backdrop():
                            self.load_backdrop_image(None)
                        dpg.add_button(label="X", callback=reset_backdrop, tag="backdrop_reset_button")
                        dpg.add_tooltip("backdrop_reset_button", tag="backdrop_reset_tooltip")  # tag
                        dpg.add_text("Clear the backdrop image (use empty background)", parent="backdrop_reset_tooltip")  # tag

                        dpg.add_button(label="Load backdrop [Ctrl+B]", width=self.button_width - 92, callback=show_open_backdrop_image_dialog, tag="open_backdrop_button")
                        dpg.add_tooltip("open_backdrop_button", tag="open_backdrop_tooltip")  # tag
                        dpg.add_text("Load an image to use as the backdrop", parent="open_backdrop_tooltip")  # tag

                        dpg.add_checkbox(label="Blur", default_value=True, callback=self._resize_gui, tag="backdrop_blur_checkbox")
                        dpg.add_tooltip("backdrop_blur_checkbox", tag="backdrop_blur_tooltip")  # tag
                        dpg.add_text("Blur the backdrop image", parent="backdrop_blur_tooltip")  # tag
                    dpg.add_button(label="Load emotion templates [Ctrl+Shift+E]", width=self.button_width, callback=show_open_json_dialog, tag="open_json_button")
                    dpg.add_tooltip("open_json_button", tag="open_json_tooltip")  # tag
                    dpg.add_text("Load emotion templates JSON file\n(format as in raven/avatar/assets/emotions/_defaults.json)", parent="open_json_tooltip")  # tag
                    dpg.add_text("[Use raven.avatar.editor to edit templates.]", color=(140, 140, 140))

                    # Main animator settings
                    dpg.add_text("Animator [Ctrl+click to set a numeric value]")
                    with dpg.group(horizontal=True):
                        def reset_target_fps():
                            dpg.set_value("target_fps_slider", 25)
                            self.on_gui_settings_change(None, None)
                        dpg.add_button(label="X", callback=reset_target_fps, tag="target_fps_reset_button")
                        dpg.add_tooltip("target_fps_reset_button", tag="target_fps_reset_tooltip")  # tag
                        dpg.add_text("Reset the animator's target FPS to default", parent="target_fps_reset_tooltip")  # tag

                        dpg.add_slider_int(label="FPS", default_value=25, min_value=10, max_value=60, clamped=True, width=self.button_width - 80,
                                           callback=self.on_gui_settings_change, tag="target_fps_slider")
                        dpg.add_tooltip("target_fps_slider", tag="target_fps_tooltip")  # tag
                        dpg.add_text("Set the animator's target FPS\n(will attempt to render and send at this rate)", parent="target_fps_tooltip")  # tag
                    with dpg.group(horizontal=True):
                        def reset_pose_interpolator_step():
                            dpg.set_value("pose_interpolator_step_slider", 3)
                            self.on_gui_settings_change(None, None)
                        dpg.add_button(label="X", callback=reset_pose_interpolator_step, tag="pose_interpolator_step_reset_button")
                        dpg.add_tooltip("pose_interpolator_step_reset_button", tag="pose_interpolator_step_reset_tooltip")  # tag
                        dpg.add_text("Reset the animator's pose interpolator step to default", parent="pose_interpolator_step_reset_tooltip")  # tag

                        dpg.add_slider_int(label="Speed", default_value=3, min_value=1, max_value=9, clamped=True, width=self.button_width - 80,
                                           callback=self.on_gui_settings_change, tag="pose_interpolator_step_slider")
                        dpg.add_tooltip("pose_interpolator_step_slider", tag="pose_interpolator_step_tooltip")  # tag
                        dpg.add_text("Set the animator's pose interpolator step (larger = faster)", parent="pose_interpolator_step_tooltip")  # tag
                    dpg.add_button(label="Pause [Ctrl+P]", width=self.button_width, callback=self.toggle_animator_paused, tag="pause_resume_button")
                    dpg.add_tooltip("pause_resume_button", tag="pause_resume_tooltip")  # tag
                    dpg.add_text("Pause or resume the avatar\n(no render resources used while paused)", parent="pause_resume_tooltip")  # tag

                    dpg.add_button(label="Load settings [Ctrl+Shift+A]", width=self.button_width, callback=show_open_animator_settings_dialog, tag="open_animator_settings_button")
                    dpg.add_tooltip("open_animator_settings_button", tag="open_animator_settings_tooltip")  # tag
                    dpg.add_text("Load an animator settings file\n(the file contains everything that is configured in this app)", parent="open_animator_settings_tooltip")  # tag

                    dpg.add_button(label="Save settings [Ctrl+Shift+S]", width=self.button_width, callback=show_save_animator_settings_dialog, tag="save_animator_settings_button")
                    dpg.add_tooltip("save_animator_settings_button", tag="save_animator_settings_tooltip")  # tag
                    dpg.add_text("Save your work\n(the file will contain everything that is configured in this app)", parent="save_animator_settings_tooltip")  # tag
                    dpg.add_spacer(height=8)

                    # Upscaler settings
                    dpg.add_text("Upscaler [Ctrl+click to set a numeric value]")
                    dpg.add_slider_int(label="x 0.1x", default_value=int(10 * self.upscale), min_value=10, max_value=20, clamped=True, width=self.button_width - 64,
                                       callback=self.on_upscaler_settings_change, tag="upscale_slider")
                    dpg.add_tooltip("upscale_slider", tag="upscale_tooltip")  # tag
                    dpg.add_text("Set upscale factor for avatar video stream", parent="upscale_tooltip")  # tag
                    self.upscale_presets = ["A", "B", "C"]
                    with dpg.group(horizontal=True):
                        dpg.add_combo(items=self.upscale_presets,
                                      default_value=self.upscale_preset,
                                      width=self.button_width - 64,
                                      callback=self.on_upscaler_settings_change,
                                      tag="upscale_preset_choice")
                        dpg.add_tooltip("upscale_preset_choice", tag="upscale_preset_tooltip")  # tag
                        dpg.add_text("Choose Anime4K preset\n    A = optimized to remove blur, resampling artifacts, smearing\n    B = optimized to remove ringing/aliasing\n    C = optimized for images with no degradation", parent="upscale_preset_tooltip")  # tag
                        dpg.add_text("Preset")
                    with dpg.group(horizontal=True):
                        self.upscale_qualities = ["low", "high"]
                        dpg.add_combo(items=self.upscale_qualities,
                                      default_value=self.upscale_quality,
                                      width=self.button_width - 64,
                                      callback=self.on_upscaler_settings_change,
                                      tag="upscale_quality_choice")
                        dpg.add_tooltip("upscale_quality_choice", tag="upscale_quality_tooltip")  # tag
                        dpg.add_text("Choose upscale quality/speed tradeoff", parent="upscale_quality_tooltip")  # tag
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
                                                        callback=self.on_send_emotion,
                                                        tag="emotion_choice")
                    dpg.add_tooltip("emotion_choice", tag="emotion_tooltip")  # tag
                    dpg.add_text("Set the character's emotion\n(Ctrl+E; then Up, Down, Home, End to jump)", parent="emotion_tooltip")  # tag
                    self.on_send_emotion(sender=self.emotion_choice, app_data=self.emotion_names[0])  # initial emotion upon app startup; should be "neutral"

                    with dpg.group(horizontal=True):
                        dpg.add_checkbox(label="Animefx", default_value=True, callback=self.on_gui_settings_change, tag="animefx_checkbox")  # animefx: emotion change triggered anime effects
                        dpg.add_tooltip("animefx_checkbox", tag="animefx_tooltip")  # tag
                        dpg.add_text("Enable/disable hovering anime effects such as sweatdrops and exclamation marks", parent="animefx_tooltip")  # tag

                        self._data_eyes_state = False
                        def toggle_data_eyes():
                            self._data_eyes_state = not self._data_eyes_state
                            if self._data_eyes_state:
                                avatar_controller.start_data_eyes(config=avatar_record)
                                dpg.set_item_label("data_eyes_button", "Stop data eyes")  # tag  # TODO: DRY GUI labels
                            else:
                                avatar_controller.stop_data_eyes(config=avatar_record)
                                dpg.set_item_label("data_eyes_button", "Start data eyes")  # tag  # TODO: DRY GUI labels

                        dpg.add_button(label="Start data eyes", width=205, callback=toggle_data_eyes, tag="data_eyes_button")  # per-character "data eyes" effect
                        dpg.add_tooltip("data_eyes_button", tag="data_eyes_tooltip")  # tag
                        dpg.add_text("Enable/disable the character's 'data eyes' state", parent="data_eyes_tooltip")  # tag
                    dpg.add_spacer(height=2)

                    dpg.add_text("Talking animation (generic, non-lipsync)")
                    dpg.add_button(label="Start [Ctrl+T]", width=self.button_width, callback=self.toggle_talking, tag="start_stop_talking_button")
                    dpg.add_tooltip("start_stop_talking_button", tag="start_stop_talking_tooltip")  # tag
                    dpg.add_text("Start/stop generic no-audio talking animation (randomized mouth)", parent="start_stop_talking_tooltip")  # tag
                    with dpg.group(horizontal=True):
                        def reset_talking_fps():
                            dpg.set_value("talking_fps_slider", 12)
                            self.on_gui_settings_change(None, None)
                        dpg.add_button(label="X", callback=reset_talking_fps, tag="talking_fps_reset_button")
                        dpg.add_tooltip("talking_fps_reset_button", tag="talking_fps_reset_tooltip")  # tag
                        dpg.add_text("Reset the animator's FPS for the generic no-audio talking animation to default", parent="talking_fps_reset_tooltip")  # tag

                        dpg.add_slider_int(label="Talk FPS", default_value=12, min_value=6, max_value=24, clamped=True, width=self.button_width - 86,
                                           callback=self.on_gui_settings_change, tag="talking_fps_slider")
                    dpg.add_tooltip("talking_fps_slider", tag="talking_fps_tooltip")  # tag
                    dpg.add_text("Set the animator's FPS for the generic no-audio talking animation (how often to re-randomize mouth)", parent="talking_fps_tooltip")  # tag
                    dpg.add_spacer(height=2)

                    # AI speech synthesizer
                    tts_alive = api.tts_server_available()
                    if tts_alive:
                        print(f"{Fore.GREEN}{Style.BRIGHT}Connected to TTS at {client_config.raven_server_url}.{Style.RESET_ALL}")
                        print(f"{Fore.GREEN}{Style.BRIGHT}Speech synthesis is available.{Style.RESET_ALL}")
                        heading_label = f"Voice [Ctrl+V] [{client_config.raven_server_url}]"
                        self.voice_names = api.tts_list_voices()
                    else:
                        print(f"{Fore.YELLOW}{Style.BRIGHT}WARNING: Cannot connect to TTS at {client_config.raven_server_url}.{Style.RESET_ALL} Is the 'tts' module loaded?")
                        print(f"{Fore.YELLOW}{Style.BRIGHT}Speech synthesis is NOT available.{Style.RESET_ALL}")
                        heading_label = "Voice [Ctrl+V] [not connected]"
                        self.voice_names = ["[TTS not available]"]
                    dpg.add_text(heading_label)
                    self.voice_choice = dpg.add_combo(items=self.voice_names,
                                                      default_value=self.voice_names[0],
                                                      width=self.button_width,
                                                      tag="voice_choice")
                    dpg.add_tooltip("voice_choice", tag="voice_tooltip")  # tag
                    dpg.add_text("Choose the TTS voice\n(Ctrl+V; then Up, Down, Home, End to jump)", parent="voice_tooltip")  # tag
                    with dpg.group(horizontal=True):
                        dpg.add_text("Speed")
                        dpg.add_button(label="X", tag="speak_speed_reset_button", callback=lambda: dpg.set_value("speak_speed_slider", 10))
                        dpg.add_tooltip("speak_speed_reset_button", tag="speak_speed_reset_tooltip")  # tag
                        dpg.add_text("Reset the TTS audio speed to default", parent="speak_speed_reset_tooltip")  # tag

                        dpg.add_slider_int(label="x 0.1x", default_value=10, min_value=5, max_value=20, clamped=True, width=self.button_width - 122,
                                           tag="speak_speed_slider")
                        dpg.add_tooltip("speak_speed_slider", tag="speak_speed_tooltip")  # tag
                        dpg.add_text("Set the TTS audio speed\n(too high may cause skipped words)", parent="speak_speed_tooltip")  # tag
                    dpg.add_text("Lipsynced TTS [adjust video timing below]", tag="speak_lipsync_text")

                    dpg.add_slider_int(label="x 0.1 s", default_value=-8, min_value=-20, max_value=20, clamped=True, width=self.button_width - 64, tag="speak_video_offset")
                    dpg.add_tooltip("speak_video_offset", tag="speak_video_tooltip")  # tag
                    dpg.add_text("Adjust AV offset for lipsync playback\n(positive value = shift video later w.r.t the audio)", parent="speak_video_tooltip")  # tag

                    dpg.add_spacer(height=4)
                    dpg.add_input_text(default_value="",
                                       hint="[Enter text to speak]",
                                       width=self.button_width,
                                       tag="speak_input_text")
                    with dpg.group(horizontal=True):
                        # We use DPG's `user_data` feature to pass the information which speak button was clicked.
                        # Fixed button width is important here because the dynamic start/stop labels have different text extents.
                        dpg.add_button(label="Speak [Ctrl+S]", width=self.button_width - 36, callback=self.on_start_speaking, user_data="speak", enabled=tts_alive, tag="speak_button")  # TODO: DRY the GUI labels
                        dpg.bind_item_theme("speak_button", "disablable_button_theme")  # tag
                        dpg.add_tooltip("speak_button", tag="speak_tooltip")  # tag
                        self.speak_tooltip_text = dpg.add_text("Speak the entered text", parent="speak_tooltip")  # tag  # TODO: DRY the GUI labels

                        dpg.add_button(label=fa.ICON_CIRCLE, width=28, callback=self.on_start_speaking, user_data="speak_and_record", enabled=tts_alive, tag="speak_and_record_button")
                        dpg.bind_item_font("speak_and_record_button", themes_and_fonts.icon_font_solid)  # tag
                        dpg.bind_item_theme("speak_and_record_button", "disablable_red_button_theme")  # tag
                        dpg.add_tooltip("speak_and_record_button", tag="speak_and_record_tooltip")  # tag
                        self.speak_and_record_tooltip_text = dpg.add_text(f"Speak and record the entered text (.mp3 + .{self.comm_format.lower()} sequence)",  # TODO: DRY the GUI labels
                                                                          parent="speak_and_record_tooltip")  # tag

                # Postprocessor settings editor
                #
                # NOTE: Defaults and ranges for postprocessor parameters are set in `postprocessor.py`,
                # in the actual source code of the filters. This API endpoint dynamically gets the metadata
                # from the server.
                #
                self.all_postprocessor_filters = dict(api.avatar_get_available_filters())

                def build_postprocessor_gui():
                    def make_reset_filter_callback(filter_name):  # freeze by closure
                        def reset_filter():
                            logger.info(f"reset_filter: resetting '{filter_name}' to defaults.")
                            defaults = self.all_postprocessor_filters[filter_name]["defaults"]  # all parameters, with their default values
                            ranges = self.all_postprocessor_filters[filter_name]["ranges"]  # for GUI hints
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

                    for filter_name, param_info in self.all_postprocessor_filters.items():
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

    def load_backdrop_image(self, filename: Optional[Union[pathlib.Path, str]]) -> None:
        """Load a backdrop image. To clear the background, use `filename=None`."""
        self.dpg_avatar_renderer.load_backdrop_image(filename=filename)
        self.animator_settings["backdrop_path"] = str(filename) if filename is not None else None  # update the path in the animator settings
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
            dpg.set_item_height("avatar_child_window", h - 16)
        except SystemError:  # main window or live image widget does not exist
            pass

        self.dpg_avatar_renderer.reposition(new_y_bottom=h)
        self.dpg_avatar_renderer.configure_backdrop(new_width=1024,
                                                    new_height=h,
                                                    new_blur_state=dpg.get_value("backdrop_blur_checkbox"))

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
        for filter_name, param_info in self.all_postprocessor_filters.items():  # this performs the reordering
            if filter_name in input_dict:
                gui_postprocessor_chain.append((filter_name, input_dict[filter_name]))
        return gui_postprocessor_chain

    def canonize_postprocessor_parameters_for_gui(self, postprocessor_chain):
        """Auto-populate missing fields to their default values.

        Be sure to feed your postprocessor chain through `strip_postprocessor_chain_for_gui` first.
        """
        validated_postprocessor_chain = []
        for filter_name, filter_settings in postprocessor_chain:
            if filter_name not in self.all_postprocessor_filters:
                logger.warning(f"canonize_postprocessor_parameters_for_gui: Unknown filter '{filter_name}', ignoring.")
                continue
            defaults = self.all_postprocessor_filters[filter_name]["defaults"]  # all parameters, with their default values
            validated_settings = {}
            for param_name in defaults:
                validated_settings[param_name] = filter_settings.get(param_name, defaults[param_name])
            validated_postprocessor_chain.append((filter_name, validated_settings))
        return validated_postprocessor_chain

    def populate_gui_from_canonized_postprocessor_chain(self, postprocessor_chain):
        """Ordering: strip -> canonize -> populate GUI"""
        input_dict = dict(postprocessor_chain)  # [(filter0, params0), ...] -> {filter0: params0, ...}, keep last copy of each
        for filter_name in self.all_postprocessor_filters:
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
        postprocessor_chain = []
        for filter_name in self.all_postprocessor_filters:
            if dpg.get_value(f"{filter_name}_checkbox") is False:
                continue
            defaults = self.all_postprocessor_filters[filter_name]["defaults"]  # all parameters, with their default values
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
        api.avatar_set_emotion(avatar_instance_id, self.current_emotion)

    def load_input_image(self, filename: Union[pathlib.Path, str]) -> None:
        try:
            logger.info(f"PostprocessorSettingsEditorGUI.load_input_image: loading image '{filename}'")
            api.avatar_reload(avatar_instance_id, filename)
            self.current_input_image_path = filename
        except Exception as exc:
            logger.error(f"PostprocessorSettingsEditorGUI.load_input_image: {type(exc)}: {exc}")
            traceback.print_exc()
            messagebox.modal_dialog(window_title="Error",
                                    message=f"Could not load image '{filename}', reason {type(exc)}: {exc}",
                                    buttons=["Close"],
                                    ok_button="Close",
                                    cancel_button="Close",
                                    centering_reference_window=self.window)

    def on_reload_input_image(self, sender, app_data):
        if self.current_input_image_path is not None:
            logger.info("PostprocessorSettingsEditorGUI.on_reload_input_image: Refreshing current character from disk")
            self.load_input_image(self.current_input_image_path)
        else:
            logger.info("PostprocessorSettingsEditorGUI.on_reload_input_image: `self.current_input_image_path` not set, nothing to reload.")

    def load_json(self, filename: Union[pathlib.Path, str]) -> None:
        try:
            logger.info(f"PostprocessorSettingsEditorGUI.load_json: loading emotion templates '{filename}'")
            api.avatar_load_emotion_templates_from_file(avatar_instance_id, filename)
        except Exception as exc:
            logger.error(f"PostprocessorSettingsEditorGUI.load_json: {type(exc)}: {exc}")
            traceback.print_exc()
            messagebox.modal_dialog(window_title="Error",
                                    message=f"Could not load emotion templates '{filename}', reason {type(exc)}: {exc}",
                                    buttons=["Close"],
                                    ok_button="Close",
                                    cancel_button="Close",
                                    centering_reference_window=self.window)

    def on_toggle_postprocessor(self, sender, app_data):
        self.postprocessor_enabled = not self.postprocessor_enabled
        self.on_gui_settings_change(sender, app_data)

    def on_upscaler_settings_change(self, sender, app_data):
        """Update the upscaler status and send changes to server."""
        old_image_size = self.dpg_avatar_renderer.image_size
        new_upscale = dpg.get_value("upscale_slider") / 10
        new_image_size = int(new_upscale * self.source_image_size)
        if new_image_size != old_image_size:
            self.dpg_avatar_renderer.configure_live_texture(new_image_size)

        self.upscale = new_upscale
        self.upscale_preset = dpg.get_value("upscale_preset_choice")
        self.upscale_quality = dpg.get_value("upscale_quality_choice")
        self.on_gui_settings_change(sender, app_data)

    def on_gui_settings_change(self, sender, app_data):
        """Send new animator/upscaler/postprocessor settings to Raven-server whenever a value changes in the GUI.

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
                                        "upscale_quality": self.upscale_quality,
                                        "animefx_enabled": dpg.get_value("animefx_checkbox")}
            self.animator_settings.update(custom_animator_settings)

            # Send to server
            api.avatar_load_animator_settings(avatar_instance_id, self.animator_settings)
        except Exception as exc:
            logger.error(f"PostprocessorSettingsEditorGUI.on_gui_settings_change: {type(exc)}: {exc}")
            traceback.print_exc()

    def load_animator_settings(self, filename: Union[pathlib.Path, str]) -> None:
        """Load an animator settings JSON file and send the settings both to the GUI and to Raven-server."""
        try:
            logger.info(f"PostprocessorSettingsEditorGUI.load_animator_settings: loading '{str(filename)}'")
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

            if "animefx_enabled" in animator_settings:
                dpg.set_value("animefx_checkbox", animator_settings["animefx_enabled"])

            backdrop_path = animator_settings.get("backdrop_path", None)  # Default to no backdrop image if the file doesn't have this key.
            self.dpg_avatar_renderer.load_backdrop_image(backdrop_path)
            if "backdrop_blur" in animator_settings:
                dpg.set_value("backdrop_blur_checkbox", animator_settings["backdrop_blur"])

            # Make sure these fields exist (in case they didn't yet).
            # They're not mandatory (any missing keys are always auto-populated from server defaults),
            # but they're something `PostprocessorSettingsEditorGUI` tracks, so we should sync our state to the server.
            # IMPORTANT: Take the default values from the same place where they sent to above.
            custom_animator_settings = {"format": self.comm_format,
                                        "target_fps": dpg.get_value("target_fps_slider"),
                                        "talking_fps": dpg.get_value("talking_fps_slider"),
                                        "pose_interpolator_step": dpg.get_value("pose_interpolator_step_slider") / 10,
                                        "upscale": self.upscale,
                                        "upscale_preset": self.upscale_preset,
                                        "upscale_quality": self.upscale_quality,
                                        "animefx_enabled": dpg.get_value("animefx_checkbox"),
                                        "backdrop_path": backdrop_path,
                                        "backdrop_blur": dpg.get_value("backdrop_blur_checkbox")}
            animator_settings.update(custom_animator_settings)

            # Send to server
            api.avatar_load_animator_settings(avatar_instance_id, animator_settings)

            # Make sure the possibly updated backdrop applies
            self._resize_gui()

            # ...and only if that is successful, remember the settings.
            self.animator_settings = animator_settings
        except Exception as exc:
            logger.error(f"PostprocessorSettingsEditorGUI.load_animator_settings: {type(exc)}: {exc}")
            traceback.print_exc()
            messagebox.modal_dialog(window_title="Error",
                                    message=f"Could not load animator settings '{str(filename)}', reason {type(exc)}: {exc}",
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
            logger.info(f"PostprocessorSettingsEditorGUI.save_animator_settings: saving as '{str(filename)}'")
            if self.animator_settings is None:
                raise RuntimeError("save_animator_settings: no animator settings loaded, nothing to save")
            with open(filename, "w", encoding="utf-8") as json_file:
                json.dump(self.animator_settings, json_file, indent=4)
        except Exception as exc:
            logger.error(f"PostprocessorSettingsEditorGUI.save_animator_settings: {type(exc)}: {exc}")
            traceback.print_exc()
            messagebox.modal_dialog(window_title="Error",
                                    message=f"Could not save animator settings '{str(filename)}', reason {type(exc)}: {exc}",
                                    buttons=["Close"],
                                    ok_button="Close",
                                    cancel_button="Close",
                                    centering_reference_window=self.window)

    def toggle_talking(self) -> None:
        """Toggle the avatar's talking state (simple randomized mouth animation)."""
        if not self.talking_animation_running:
            api.avatar_start_talking(avatar_instance_id)
            dpg.set_item_label("start_stop_talking_button", "Stop [Ctrl+T]")  # tag
        else:
            api.avatar_stop_talking(avatar_instance_id)
            dpg.set_item_label("start_stop_talking_button", "Start [Ctrl+T]")  # tag
        self.talking_animation_running = not self.talking_animation_running

    def toggle_animator_paused(self) -> None:
        """Pause or resume the animation. Pausing when the avatar won't be visible (e.g. minimized window) saves resources as new frames are not computed."""
        if self.dpg_avatar_renderer.animator_running:
            self.dpg_avatar_renderer.pause(action="pause")
            dpg.set_item_label("pause_resume_button", "Resume [Ctrl+P]")  # tag
        else:
            self.dpg_avatar_renderer.pause(action="resume")
            dpg.set_item_label("pause_resume_button", "Pause [Ctrl+P]")  # tag

    def on_stop_speaking(self, sender, app_data, user_data) -> None:
        """DPG GUI event handler: stop speaking (and recording, if active).

        `user_data`: One of "speak" or "speak_and_record", identifies which button was clicked.
                     Set the appropriate value as the `user_data` of your button when you create it.
        """
        # mode = user_data

        avatar_controller.stop_tts()
        dpg.hide_item(self.recording_indicator_group)  # If not recording, this is a no-op.

        dpg.set_item_label("speak_button", "Speak [Ctrl+S]")  # TODO: DRY the GUI labels  # tag
        dpg.set_value(self.speak_tooltip_text, "Speak the entered text")  # TODO: DRY the GUI labels
        dpg.set_item_callback("speak_button", self.on_start_speaking)  # tag

        dpg.set_item_label("speak_and_record_button", fa.ICON_CIRCLE)  # tag
        dpg.set_value(self.speak_and_record_tooltip_text, f"Speak and record the entered text (.mp3 + .{self.comm_format.lower()} sequence)")  # TODO: DRY the GUI labels
        dpg.set_item_callback("speak_and_record_button", self.on_start_speaking)  # tag  # TODO: DRY the GUI labels
        dpg.enable_item("speak_and_record_button")  # tag

        self.speaking = False

    def on_start_speaking(self, sender, app_data, user_data) -> None:
        """DPG GUI event handler: start speaking (and optionally recording).

        `user_data`: One of "speak" or "speak_and_record", identifies which button was clicked.
                     Set the appropriate value as the `user_data` of your button when you create it.
        """
        mode = user_data

        self.speaking = True

        dpg.set_item_label("speak_button", "Stop speaking [Ctrl+S]")  # tag  # TODO: DRY the GUI labels
        dpg.set_value(self.speak_tooltip_text, "Stop the speech synthesizer and speech animation")  # TODO: DRY the GUI labels
        dpg.set_item_callback("speak_button", self.on_stop_speaking)  # tag

        dpg.set_item_label("speak_and_record_button", fa.ICON_SQUARE)  # tag
        dpg.set_value(self.speak_and_record_tooltip_text, "Stop recording")  # TODO: DRY the GUI labels
        dpg.set_item_callback("speak_and_record_button", self.on_stop_speaking)  # tag  # TODO: DRY the GUI labels
        if mode == "speak":  # When just speaking, disable the stop-recording button for UX clarity (does not make sense to stop recording, since we're not recording)
            dpg.disable_item("speak_and_record_button")  # tag

        if mode == "speak_and_record":
            pulsating_red_text_glow.reset()  # start new pulsation cycle
            dpg.show_item(self.recording_indicator_group)

            def on_audio_ready(output_record: Dict[str, Any], audio_data: bytes) -> None:
                """Save the TTS speech audio file (for each sentence) to disk."""
                common_utils.create_directory(recording_output_dir)
                filename = os.path.join(str(recording_output_dir), f"line_{output_record['line_number']:05d}_sentence_{output_record['sentence_number_on_line']:05d}.mp3")
                logger.info(f"PostprocessorSettingsEditorGUI.on_start_speaking.on_audio_ready: Saving TTS speech audio to '{filename}' (sentence output_record['sentence_uuid']: '{output_record['sentence']}')")
                with open(filename, "wb") as audio_file:
                    audio_file.write(audio_data)
                logger.info(f"PostprocessorSettingsEditorGUI.on_start_speaking.on_audio_ready: TTS speech audio saved to '{filename}'")
        else:
            on_audio_ready = None

        selected_voice = dpg.get_value(self.voice_choice)

        # Get text entered by user, and if none, use a default text.
        text = dpg.get_value("speak_input_text")
        if text == "":
            # text = "Testing the AI speech synthesizer."
            # text = '"Wait", I said, but the cat said "meow".'  # includes quotes
            # text = "INFO:raven.client.api:tts_speak_lipsynced.speak: starting"  # log message
            # text = 'close mouth only if the pause is at least half a second, else act like "!keep".'  # code comment
            # text = "Sharon Apple is a computer-generated virtual idol and a central character in the Macross Plus franchise, created by Shoji Kawamori."
            # text = "Sharon Apple. Before Hatsune Miku, before VTubers, there was Sharon Apple. The digital diva of Macross Plus hailed from the in-universe mind of Myung Fang Lone, and sings tunes by legendary composer Yoko Kanno. Sharon wasn't entirely artificially intelligent, though: the unfinished program required Myung to patch in emotions during her concerts."
            # text = "From approximately 10,000 BCE, the Neolithic Revolution initiated humanity’s shift from nomadic hunter-gatherer societies to settled agricultural communities. This was followed by the Bronze Age, spanning from roughly 3,300 to 1,200 BCE, which fostered the emergence of early cities and empires such as Sumer and Ancient Egypt. The Iron Age began around 1,200 BCE, driven by advancements in metallurgy and extending until approximately 500 BCE, enabling the rise of powerful civilizations like Persia and the Roman Republic. The Classical Era, from about 800 BCE to 500 CE, represented the zenith of Greek philosophy, Roman law, and widespread religious diffusion. The Medieval Period, lasting from 500 to 1,500 CE, witnessed the development of feudal systems in Europe alongside the Islamic Golden Age. The Early Modern Era, from 1,500 to 1,800 CE, brought the Age of Exploration, Enlightenment ideas, and the birth of modern nation-states. The Industrial Revolution commenced in the late 18th century, triggering mechanized manufacturing and urbanization. Finally, the Modern Era, starting in the early 19th century, continues to define today’s interconnected, digitized global society."
            text = 'The failure of any experiment to detect motion through the aether led Hendrik Lorentz, starting in eighteen ninety two, to develop a theory of electrodynamics based on an immobile luminiferous aether, physical length contraction, and a "local time" in which Maxwell\'s equations retain their form in all inertial frames of reference.'

        # Queue up the TTS, also setting up event handlers.
        #
        def on_start_batch(output_record: Dict[str, Any]) -> None:
            logger.info("on_start_batch: starting to speak")
            if mode == "speak_and_record":
                logger.info("PostprocessorSettingsEditorGUI.on_start_speaking.on_start_batch: mode is 'speak_and_record', starting video recorder")
                video_recorder.start()
        def on_stop_batch(output_record: Dict[str, Any]) -> None:
            if video_recorder.recording:
                video_recorder.stop()

                # Save audio timing report
                common_utils.create_directory(recording_output_dir)
                filename = os.path.join(str(recording_output_dir), "audio_timing.txt")
                logger.info(f"PostprocessorSettingsEditorGUI.on_start_speaking.on_stop_batch: Saving audio timing report to '{filename}'")
                with open(filename, "w") as timings_file:
                    timings_file.write("Audio timing report\n")
                    timings_file.write("===================\n\n")
                    timings_file.write("<start video frame> - <end video frame> (<start time> - <end time>) (duration <duration>): line <line number>, sentence <sentence number>: 'spoken sentence'\n")
                    timings_file.write(f"Times are in seconds, computed at {self.target_fps} FPS.\n\n")
                    for (lineno, sentenceno, sentence), start_ts, end_ts in zip(sentences, batch_audio_start_timestamps, batch_audio_end_timestamps):
                        timings_file.write(f"{start_ts:05d} - {end_ts:05d} ({start_ts / self.target_fps:0.2f}s - {end_ts / self.target_fps:0.2f}s) (duration {(end_ts - start_ts) / self.target_fps}s): line {lineno:05d}, sentence {sentenceno:05d}: '{sentence}'\n")

            self.on_stop_speaking(None, None, user_data)  # update the GUI

        # Sentence-level handlers to get TTS speech audio timing info
        sentences = []
        batch_audio_start_timestamps = []
        batch_audio_end_timestamps = []
        def on_start_sentence(output_record: Dict[str, Any]) -> None:
            if video_recorder.recording:
                frame_no = video_recorder.frame_no
                logger.info(f"PostprocessorSettingsEditorGUI.on_start_speaking.on_start_sentence: video frame {frame_no}: start of sentence {output_record['sentence_uuid']}: '{output_record['sentence']}'")
                sentences.append((output_record['line_number'],
                                  output_record['sentence_number_on_line'],
                                  output_record['sentence']))
                batch_audio_start_timestamps.append(frame_no)
        def on_stop_sentence(output_record: Dict[str, Any]) -> None:
            if video_recorder.recording:
                frame_no = video_recorder.frame_no
                logger.info(f"PostprocessorSettingsEditorGUI.on_start_speaking.on_start_sentence: video frame {frame_no}: end of sentence {output_record['sentence_uuid']}: '{output_record['sentence']}'")
                batch_audio_end_timestamps.append(frame_no)

        avatar_controller.send_text_to_tts(config=avatar_record,
                                           text=text,
                                           voice=selected_voice,
                                           voice_speed=dpg.get_value("speak_speed_slider") / 10,
                                           video_offset=dpg.get_value("speak_video_offset") / 10,
                                           on_audio_ready=on_audio_ready,
                                           on_start_speaking=on_start_batch,
                                           on_stop_speaking=on_stop_batch,
                                           on_start_sentence=on_start_sentence,
                                           on_stop_sentence=on_stop_sentence)

# --------------------------------------------------------------------------------
# App window (viewport) resizing

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

# --------------------------------------------------------------------------------
# Hotkey support

combobox_choice_map = None   # DPG tag or ID -> (choice_strings, callback)
def avatar_settings_editor_hotkeys_callback(sender, app_data):
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

        # Some hidden debug features. Mnemonic: "Mr. T Lite" (Ctrl + Shift + M, R, T, L)
        if key == dpg.mvKey_M:
            dpg.show_metrics()
        elif key == dpg.mvKey_R:
            dpg.show_item_registry()
        elif key == dpg.mvKey_T:
            dpg.show_font_manager()
        elif key == dpg.mvKey_L:
            dpg.show_style_editor()

    # Ctrl+...
    elif ctrl_pressed:
        if key == dpg.mvKey_O:
            show_open_input_image_dialog()
        elif key == dpg.mvKey_R:
            gui_instance.on_reload_input_image(sender, app_data)
        elif key == dpg.mvKey_B:
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
                gui_instance.on_start_speaking(sender, app_data, "speak")  # emulate clicking the "Speak / Stop speaking" button (as opposed to the record/stop button)
            else:
                gui_instance.on_stop_speaking(sender, app_data, "speak")  # emulate clicking the "Speak / Stop speaking" button (as opposed to the record/stop button)

    # Bare key
    #
    # NOTE: These are global across the whole app (when no modal window is open) - be very careful here!
    else:
        if key == dpg.mvKey_F11:
            toggle_fullscreen()
        else:
            # {widget_tag_or_id: list_of_choices}
            global combobox_choice_map
            if combobox_choice_map is None:  # build on first use (now that `gui_instance` is available)
                combobox_choice_map = {gui_instance.emotion_choice: (gui_instance.emotion_names, gui_instance.on_send_emotion),
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
            focused_item = dpg.get_item_alias(focused_item)
            if focused_item in combobox_choice_map.keys():
                browse(focused_item, combobox_choice_map[focused_item])
with dpg.handler_registry(tag="avatar_settings_editor_handler_registry"):  # global (whole viewport)
    dpg.add_key_press_handler(tag="avatar_settings_editor_hotkeys_handler", callback=avatar_settings_editor_hotkeys_callback)

# --------------------------------------------------------------------------------
# Start the app

logger.info("App bootup...")

if api.raven_server_available():
    print(f"{Fore.GREEN}{Style.BRIGHT}Connected to Raven-server at {client_config.raven_server_url}.{Style.RESET_ALL}")
else:
    print(f"{Fore.RED}{Style.BRIGHT}ERROR: Cannot connect to Raven-server at {client_config.raven_server_url}.{Style.RESET_ALL} Is Raven-server running?")
    sys.exit(255)

# IMPORTANT: `avatar_load` first before we start the GUI, to create the avatar instance.
_startup_input_image_path = pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "assets", "characters", "other", "aria1.png")).expanduser().resolve()
avatar_instance_id = api.avatar_load(_startup_input_image_path)
api.avatar_load_emotion_templates(avatar_instance_id, {})  # send empty dict -> reset emotion templates to server defaults
gui_instance = PostprocessorSettingsEditorGUI()  # will load animator settings into the GUI, as well as send them to the avatar instance.
gui_instance.current_input_image_path = _startup_input_image_path  # so that the Refresh button works
api.avatar_start(avatar_instance_id)  # start the avatar rendering on the server...
gui_instance.dpg_avatar_renderer.start(avatar_instance_id,  # ...and start displaying the live video on the client
                                       on_frame_received=video_recorder._on_frame_received)

# We don't use most of the features of the controller here (particularly the autotranslator and subtitler), but we want the sentence-splitting and precomputing TTS,
# which gives much better lipsync and better latency than TTS'ing a long text in one go. The cost is producing a separate audio file for each sentence.
# If there is just one sentence (as judged by the server's `natlang` module), it works as before.
avatar_controller = DPGAvatarController(stop_tts_button_gui_widget=None,  # We have no dedicated stop button, but two play/stop toggles (one with recording). We manage the state ourselves.
                                        on_tts_idle=None,
                                        tts_idle_check_interval=None,
                                        subtitles_enabled=False,
                                        subtitle_text_gui_widget=None,
                                        subtitle_left_x0=0,
                                        subtitle_bottom_y0=0,
                                        translator_source_lang=None,
                                        translator_target_lang=None,
                                        main_window_w=0,
                                        main_window_h=0,
                                        executor=bg)  # use the same thread pool as our main task manager
avatar_record = avatar_controller.register_avatar_instance(avatar_instance_id=avatar_instance_id,
                                                           emotion_autoreset_interval=None,
                                                           emotion_blacklist=(),  # only used for `avatar_controller.update_emotion_from_text`
                                                           data_eyes_fadeout_duration=0.75)

def gui_shutdown() -> None:
    """App exit: gracefully shut down parts that access DPG."""
    logger.info("gui_shutdown: entered")
    avatar_controller.stop_tts()  # Stop the TTS speaking so that the speech background thread (if any) exits.
    task_manager.clear(wait=True)  # Wait until background tasks actually exit.
    avatar_controller.shutdown()
    gui_animation.animator.clear()
    global gui_instance
    gui_instance = None
    logger.info("gui_shutdown: done")
dpg.set_exit_callback(gui_shutdown)

def app_shutdown() -> None:
    """App exit: gracefully shut down parts that don't need DPG.

    This is guaranteed to run even if DPG shutdown never completes gracefully, as long as it doesn't hang the main thread, or segfault the process.

    Currently, we release server-side resources here.
    """
    logger.info("app_shutdown: entered")
    if avatar_instance_id is not None:
        try:
            api.avatar_unload(avatar_instance_id)  # delete the instance so the server can release the resources
        except requests.exceptions.ConnectionError:  # server has gone bye-bye
            pass
    logger.info("app_shutdown: done")
atexit.register(app_shutdown)

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
    animator_json_path = pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "assets", "settings", "animator.json")).expanduser().resolve()

    if not os.path.exists(animator_json_path):
        logger.info(f"_load_initial_animator_settings: Default animator settings file '{str(animator_json_path)}' missing, writing a default config.")
        try:
            animator_settings = copy.copy(server_config.animator_defaults)
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
    # gui_instance.load_backdrop_image(pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "assets", "backdrops", "anime-plains.png")).expanduser().resolve())  # DEBUG

dpg.set_frame_callback(2, _load_initial_animator_settings)

# last_tts_check_time = 0
# tts_check_interval = 5.0  # seconds
def update_animations():
    gui_animation.animator.render_frame()  # Our customized fdialog needs this for its overwrite confirm button flash.

    # # Enable/disable speech synthesizer controls depending on whether the TTS server is available
    # global last_tts_check_time
    # t0 = time.monotonic()
    # if t0 - last_tts_check_time >= tts_check_interval:
    #     last_tts_check_time = t0
    #     if tts_server_available():
    #         dpg.enable_item(gui_instance.voice_choice)
    #         dpg.enable_item("speak_button")
    #     else:
    #         dpg.disable_item(gui_instance.voice_choice)
    #         dpg.disable_item("speak_button")

try:
    # We control the render loop manually to have a convenient place to update our GUI animations just before rendering each frame.
    while dpg.is_dearpygui_running():
        update_animations()
        dpg.render_dearpygui_frame()
    # dpg.start_dearpygui()  # automatic render loop
except KeyboardInterrupt:
    task_manager.clear(wait=False)  # signal background tasks to exit

dpg.destroy_context()

def main():  # TODO: we don't really need this; it's just for console_scripts.
    pass

"""Raven-avatar pose editor, based on the original THA3 pose editor GUI app.

Pose an anime character, based on a suitable 512×512 static input image and THA3, an AI poser model.

This GUI app is an editor for the emotion templates. To edit postprocessor settings in a GUI and to
test your characters, see the separate app `raven.avatar.settings_editor.app`.

Beside being the pose editor, this app also allows the automatic generation of the 28 emotional expression sprites
for your AI character from one static input image, for use with distilbert classification in SillyTavern
(Character Expressions extension). This is much faster than inpainting all 28 expressions manually in
Stable Diffusion, enables agile experimentation on the look of your character, since you only need to produce
one new image to change the look. The resulting quality is not perfect, but often acceptable.

This module uses THA3 directly, as does `raven.server.modules.avatar`.


**Who**:

Original code written and neural networks designed and trained by Pramook Khungurn (@pkhungurn):
    https://github.com/pkhungurn/talking-head-anime-3-demo
    https://arxiv.org/abs/2311.17409

This fork was originally maintained by the SillyTavern-extras project.

At this point, the pose editor app was improved and documented by Juha Jeronen (@Technologicat).

After SillyTavern-extras was discontinued, the Talkinghead code was moved to the Raven project by Juha Jeronen (@Technologicat),
and it is now known as Raven-avatar (part of a constellation of AI-related software).


This module was part of the old Talkinghead, and is licensed under the GNU AGPL, see `LICENSE`.
"""

# Note that the AGPL license means that any BSD-licensed module must not import anything from this module. The other way is allowed,
# i.e. this module may import names from BSD-licensed modules; it's then using those modules under the BSD license.
#
# The parts of GUI code that look common to this and `raven.avatar.settings_editor.app` are actually originally adapted from Raven-visualizer,
# which is BSD-licensed.

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

import argparse
import functools
import json
import logging
import os
import pathlib
import sys
import threading
import time
import traceback
from typing import Dict, List, Tuple, Optional, Union

import PIL.Image

import numpy as np

import torch

import dearpygui.dearpygui as dpg

from ...vendor.file_dialog.fdialog import FileDialog  # https://github.com/totallynotdrait/file_dialog, but with custom modifications

from ...vendor.tha3.poser.modes.load_poser import load_poser
from ...vendor.tha3.poser.poser import Poser, PoseParameterCategory, PoseParameterGroup
from ...vendor.tha3.util import torch_linear_to_srgb

from ...common.gui import animation as gui_animation  # Raven's GUI animation system, nothing to do with the AI avatar.
from ...common.gui import fontsetup
from ...common.gui import messagebox
from ...common.gui import utils as guiutils
from ...common.hfutil import maybe_install_models
from ...common.running_average import RunningAverage
from ...common.video import compositor

from ...server import config as server_config  # NOTE: default config (can be overridden on the command line when starting the server)
from ...server.modules import avatarutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# The vendored code from THA3 expects to find the `tha3` module at the top level of the module hierarchy
talkinghead_path = pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "..", "vendor")).expanduser().resolve()  # THA3 install location containing the "tha3" folder
print(f"THA3 is installed at '{str(talkinghead_path)}'")
sys.path.append(str(talkinghead_path))

# See also `supported_cels` in `raven.server.modules.avatarutil`.
gui_cel_blending_layout = ["blush1",
                           "blush2",
                           "blush3",
                           None,  # GUI spacer
                           "shadow1",
                           None,
                           "sweat1",
                           "sweat2",
                           "sweat3",
                           None,
                           "tears1",
                           "tears2",
                           "tears3",
                           None,
                           "waver1"]  # Animation strength control; the live animator auto-cycles cels between waver1/waver2.

# Detect image file formats supported by the installed Pillow, and format a list for the file open/save dialogs.
# TODO: This is not very useful unless we can filter these to get only formats that support an alpha channel.
# TODO: And there's not much point formatting them for a wxPython file dialog in a DearPyGui app.
#
# https://docs.wxpython.org/wx.FileDialog.html
# https://stackoverflow.com/questions/71112986/retrieve-a-list-of-supported-read-file-extensions-formats
#
# exts = PIL.Image.registered_extensions()
# PIL_supported_input_formats = {ex[1:].lower() for ex, f in exts.items() if f in PIL.Image.OPEN}  # {".png", ".jpg", ...} -> {"png", "jpg", ...}
# PIL_supported_output_formats = {ex[1:].lower() for ex, f in exts.items() if f in PIL.Image.SAVE}
# def format_fileformat_list(supported_formats):
#     return ["All files (*)|*"] + [f"{fmt.upper()} images (*.{fmt})|*.{fmt}" for fmt in sorted(supported_formats)]
# input_index_to_ext = [""] + sorted(PIL_supported_input_formats)  # list index -> file extension
# input_ext_to_index = {ext: idx for idx, ext in enumerate(input_index_to_ext)}  # file extension -> list index
# output_index_to_ext = [""] + sorted(PIL_supported_output_formats)
# output_ext_to_index = {ext: idx for idx, ext in enumerate(output_index_to_ext)}
# input_exts_and_descs_str = "|".join(format_fileformat_list(PIL_supported_input_formats))  # filter-spec accepted by `wx.FileDialog`
# output_exts_and_descs_str = "|".join(format_fileformat_list(PIL_supported_output_formats))

# --------------------------------------------------------------------------------
# DPG init

dpg.create_context()

# Initialize fonts. Must be done after `dpg.create_context`, or the app will just segfault at startup.
# https://dearpygui.readthedocs.io/en/latest/documentation/fonts.html
with dpg.font_registry() as the_font_registry:
    # Change the default font to something that looks clean and has good on-screen readability.
    # https://fonts.google.com/specimen/Open+Sans
    font_size = 20
    with dpg.font(pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "..", "fonts", "OpenSans-Regular.ttf")).expanduser().resolve(),  # load font from Raven's main assets
                  font_size) as default_font:
        fontsetup.setup_font_ranges()
    dpg.bind_font(default_font)

# Modify global theme
with dpg.theme() as global_theme:
    with dpg.theme_component(dpg.mvAll):
        # dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (53, 168, 84))  # same color as Linux Mint default selection color in the green theme
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6, category=dpg.mvThemeCat_Core)
        dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 8, category=dpg.mvThemeCat_Core)
        dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 8, category=dpg.mvThemeCat_Core)
dpg.bind_theme(global_theme)  # set this theme as the default

# Viewport size depends on the image size, so it needs to be tuned after `PoseEditorGUI` initializes
dpg.create_viewport(title="Raven-avatar pose editor",
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

def initialize_filedialogs():  # called at app startup
    """Create the file dialogs."""
    global filedialog_open_image
    global filedialog_save_image
    global filedialog_open_json
    global filedialog_save_all_emotions
    cwd = os.getcwd()  # might change during filedialog init
    filedialog_open_image = FileDialog(title="Open character image",
                                       tag="open_image_dialog",
                                       callback=_open_image_callback,
                                       modal=True,
                                       filter_list=[".png"],
                                       file_filter=".png",
                                       multi_selection=False,
                                       allow_drag=False,
                                       default_path=pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "assets", "characters")).expanduser().resolve())
    filedialog_save_image = FileDialog(title="Save posed image",
                                       tag="save_image_dialog",
                                       callback=_save_image_callback,
                                       modal=True,
                                       filter_list=[".png"],
                                       file_filter=".png",
                                       save_mode=True,
                                       default_file_extension=".png",  # used if the user does not provide a file extension when naming the save-as
                                       allow_drag=False,
                                       default_path=cwd)
    filedialog_open_json = FileDialog(title="Open emotion temmplates",
                                       tag="open_json_dialog",
                                       callback=_open_json_callback,
                                       modal=True,
                                       filter_list=[".json"],
                                       file_filter=".json",
                                       multi_selection=False,
                                       allow_drag=False,
                                       default_path=pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "assets", "emotions")).expanduser().resolve())
    filedialog_save_all_emotions = FileDialog(title="Save all emotion templates",
                                              tag="save_all_emotions_dialog",
                                              callback=_save_all_emotions_callback,
                                              modal=True,
                                              filter_list=[""],
                                              file_filter="",
                                              dirs_only=True,  # *directory* picker mode
                                              save_mode=True,
                                              default_file_extension="",  # used if the user does not provide a file extension when naming the save-as
                                              allow_drag=False,
                                              default_path=cwd)

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
# "Save image" dialog

def show_save_image_dialog():
    """Button callback. Show the save file dialog, for the user to pick a filename to save as.

    If you need to close it programmatically, call `filedialog_save_image.cancel()` so it'll trigger the callback.
    """
    if gui_instance is None:
        return
    if gui_instance.torch_base_image is None:  # character not loaded?
        return
    logger.debug("show_save_image_dialog: Showing save file dialog.")
    filedialog_save_image.show_file_dialog()
    logger.debug("show_save_image_dialog: Done.")

def _save_image_callback(selected_files):
    """Callback that fires when the save image dialog closes."""
    logger.debug("_save_image_callback: Save file dialog callback triggered.")
    if len(selected_files) > 1:  # Should not happen, since we set `multi_selection=False`.
        raise ValueError(f"Expected at most one selected file, got {len(selected_files)}.")
    if selected_files:
        selected_file = selected_files[0]
        logger.debug(f"_save_image_callback: User selected the file '{selected_file}'.")
        gui_instance.save_image(selected_file)
    else:  # empty selection -> cancelled
        logger.debug("_save_image_callback: Cancelled.")

def is_save_image_dialog_visible():
    """Return whether the open image dialog is open.

    We have this abstraction (not just `dpg.is_item_visible`) because the window might not exist yet.
    """
    if filedialog_save_image is None:
        return False
    return dpg.is_item_visible("save_image_dialog")  # tag

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
# "Save all emotions" dialog

def show_save_all_emotions_dialog():
    """Button callback. Show the save file dialog, for the user to pick a filename to save as.

    If you need to close it programmatically, call `filedialog_save_all_emotions.cancel()` so it'll trigger the callback.
    """
    if gui_instance is None:
        return
    if gui_instance.torch_base_image is None:  # character not loaded?
        return
    logger.debug("show_save_all_emotions_dialog: Showing save file dialog.")
    filedialog_save_all_emotions.show_file_dialog()
    logger.debug("show_save_all_emotions_dialog: Done.")

def _save_all_emotions_callback(selected_files):
    """Callback that fires when the save all emotions dialog closes."""
    logger.debug("_save_all_emotions_callback: Save file dialog callback triggered.")
    if len(selected_files) > 1:  # Should not happen, since we set `multi_selection=False`.
        raise ValueError(f"Expected at most one selected file, got {len(selected_files)}.")
    if selected_files:
        selected_file = selected_files[0]
        logger.debug(f"_save_all_emotions_callback: User selected the file '{selected_file}'.")
        gui_instance.save_all_emotions(selected_file)
    else:  # empty selection -> cancelled
        logger.debug("_save_all_emotions_callback: Cancelled.")

def is_save_all_emotions_dialog_visible():
    """Return whether the save all emotions dialog is open.

    We have this abstraction (not just `dpg.is_item_visible`) because the window might not exist yet.
    """
    if filedialog_save_all_emotions is None:
        return False
    return dpg.is_item_visible("save_all_emotions_dialog")  # tag

# --------------------------------------------------------------------------------
# GUI controls

def is_any_modal_window_visible():
    """Return whether *some* modal window is open.

    Currently these are file dialogs.
    """
    return (is_open_image_dialog_visible() or is_save_image_dialog_visible() or
            is_open_json_dialog_visible() or is_save_all_emotions_dialog_visible())

def get_slider_range(slider):
    slider_config = dpg.get_item_configuration(slider)
    min_value = slider_config["min_value"]
    max_value = slider_config["max_value"]
    return min_value, max_value

def slider_value_to_relpos(slider):
    """Return the relative position [0, 1] of a slider within its value range."""
    min_value, max_value = get_slider_range(slider)
    value = dpg.get_value(slider)
    relpos = (value - min_value) / (max_value - min_value)
    return relpos

def relpos_to_slider_value(slider, relpos):
    min_value, max_value = get_slider_range(slider)
    value = int(min_value + relpos * (max_value - min_value))
    return value


class CelBlendControlPanel:
    """A very simple control panel for controlling cel blending (i.e. source image edits by blending in other RGBA images).

    Cel blending is useful for blushing, the "intense emotion" eye-wavering effect, and similar.
    The actual blending is done elsewhere; this is just the GUI slider to allow the user to control the value.

    The `label` is shown in the GUI. It tells the user which cel this control applies to.

    The value range is always [0, 1]. The value 0 means "don't blend in this image at all" and 1 means "blend at full strength".

    `parent` is the DPG ID or tag of the parent GUI widget. If not given, DPG's stack state is used (i.e. placed in whatever GUI
    section you're building when you instantiate this object).
    """

    def __init__(self,
                 label: str,
                 parent: Optional[Union[str, int]] = None):
        self.label = label
        self.slider = dpg.add_slider_int(label=label, default_value=0, min_value=0, max_value=1000, parent=parent)

    def write_to_celstack(self, celstack: List[Tuple[str, float]]) -> None:
        """Update a cel stack in-place by the current value set in this control panel.

        If `self.label` is missing from `celstack` (it doesn't have the cel this panel is controlling),
        then do nothing.
        """
        idx = compositor.get_cel_index_in_stack(self.label, celstack)
        if idx != -1:  # found?
            celstack[idx] = (self.label, slider_value_to_relpos(self.slider))
        # else:
        #     logger.warning(f"This control panel's label '{self.label}' not in celstack, ignoring.")  # DEBUG (spammy)

    def read_from_celstack(self, celstack: List[Tuple[str, float]]) -> None:
        """Overwrite the current value in this control panel by that taken from `celstack`.

        If `self.label` is missing from `celstack` (it doesn't have the cel this panel is controlling),
        then set this control panel to the value 0.
        """
        idx = compositor.get_cel_index_in_stack(self.label, celstack)
        if idx != -1:  # found?
            _, strength = celstack[idx]
        else:
            strength = 0.0
            # logger.warning(f"This control panel's label '{self.label}' not in celstack, setting control panel to default value.")  # DEBUG (spammy)
        dpg.set_value(self.slider, relpos_to_slider_value(self.slider, strength))


class SimpleParamGroupsControlPanel:
    """A simple control panel for groups of arity-1 continuous parameters (i.e. float value, and no separate left/right controls).

    The panel represents a *category*, such as "body rotation".

    A category may have several *parameter groups*, all of which are active simultaneously. Here "parameter group" is a misnomer,
    since in all use sites for this panel, each group has only one parameter. For example, "body rotation" has the groups ["body_y", "body_z"].
    """

    def __init__(self,
                 pose_param_category: PoseParameterCategory,
                 param_groups: List[PoseParameterGroup]):

        self.param_groups = [group for group in param_groups if group.get_category().value == pose_param_category.value]
        for param_group in self.param_groups:
            assert not param_group.is_discrete()
            assert param_group.get_arity() == 1

        self.sliders = []
        with dpg.group():
            for param_group in self.param_groups:
                # HACK: iris_rotation_*, head_*, body_* have range [-1, 1], but breathing has range [0, 1],
                #       and all of them should default to the *value* 0.
                param_range = param_group.get_range()
                min_slider_value = int(param_range[0] * 1000)
                max_slider_value = int(param_range[1] * 1000)
                slider = dpg.add_slider_int(label=param_group.get_group_name(),
                                            default_value=0,
                                            min_value=min_slider_value,
                                            max_value=max_slider_value,
                                            clamped=True)
                self.sliders.append(slider)

    def write_to_pose(self, pose: List[float]) -> None:
        """Update `pose` (in-place) by the current value(s) set in this control panel."""
        for param_group, slider in zip(self.param_groups, self.sliders):
            param_index = param_group.get_parameter_index()
            param_range = param_group.get_range()
            param_value = param_range[0] + (param_range[1] - param_range[0]) * slider_value_to_relpos(slider)
            pose[param_index] = param_value

    def read_from_pose(self, pose: List[float]) -> None:
        """Overwrite the current value(s) in this control panel by those taken from `pose`."""
        for param_group, slider in zip(self.param_groups, self.sliders):
            param_index = param_group.get_parameter_index()
            param_range = param_group.get_range()
            param_value = pose[param_index]  # cherry-pick only relevant values from `pose`
            relpos = (param_value - param_range[0]) / (param_range[1] - param_range[0])
            dpg.set_value(slider, relpos_to_slider_value(slider, relpos))  # TODO: do we need to trigger the callback manually?


class MorphCategoryControlPanel:
    """A more complex control panel with grouping semantics.

    The panel represents a *category*, such as "eyebrow".

    A category may have several *parameter groups*, only one of which can be active at any given time.

    For example, the "eyebrow" category has the parameter groups ["eyebrow_troubled", "eyebrow_angry", ...].

    Each parameter group can be:
      - Continuous with arity 1 (one slider),
      - Continuous with arity 2 (two sliders, for separate left/right control), or
      - Discrete (on/off).

    The panel allows the user to select a parameter group within the category, and enables/disables its
    UI controls appropriately. The user can then use the controls to set the values for the selected
    parameter group within the category represented by the panel.
    """
    def __init__(self,
                 category_title: str,
                 pose_param_category: PoseParameterCategory,
                 param_groups: List[PoseParameterGroup]):
        self.category_title = category_title
        self.pose_param_category = pose_param_category

        with dpg.group():
            dpg.add_text(category_title)

            self.param_groups = [group for group in param_groups if group.get_category().value == pose_param_category.value]
            if not self.param_groups:
                assert False  # TODO: should not happen
            self.param_group_names = [group.get_group_name() for group in self.param_groups]
            self.choice = dpg.add_combo(items=self.param_group_names,
                                        default_value=self.param_group_names[0])
            dpg.set_item_callback(self.choice, self.on_choice_updated)

            self.left_slider = dpg.add_slider_int(default_value=-1000,
                                                  min_value=-1000,
                                                  max_value=1000,
                                                  clamped=True)
            self.right_slider = dpg.add_slider_int(default_value=-1000,
                                                   min_value=-1000,
                                                   max_value=1000,
                                                   clamped=True)

            self.checkbox = dpg.add_checkbox(label="Show", default_value=True)

            self.update_ui()

    def update_ui(self) -> None:
        """Enable/disable UI controls based on the currently active parameter group."""
        param_group_name = dpg.get_value(self.choice)
        param_group_index = self.param_group_names.index(param_group_name)
        param_group = self.param_groups[param_group_index]
        if param_group.is_discrete():
            dpg.hide_item(self.left_slider)
            dpg.hide_item(self.right_slider)
            dpg.show_item(self.checkbox)
        elif param_group.get_arity() == 1:
            dpg.show_item(self.left_slider)
            dpg.hide_item(self.right_slider)
            dpg.hide_item(self.checkbox)
        else:
            dpg.show_item(self.left_slider)
            dpg.show_item(self.right_slider)
            dpg.hide_item(self.checkbox)

    def on_choice_updated(self, sender, app_data) -> None:
        """Automatically optimize usability for the new arity and discrete/continuous state."""
        selected_morph_index = self.param_group_names.index(dpg.get_value(self.choice))
        param_group = self.param_groups[selected_morph_index]
        # logger.debug(f"on_choice_updated: sender = {sender}, app_data = {app_data}, new morph index = {selected_morph_index}, new param_group = {param_group}")
        if param_group.is_discrete():
            dpg.set_value(self.checkbox, True)  # discrete parameter group: set to "on" when switched into

            for slider in (self.left_slider, self.right_slider):
                min_value, ignored_max_value = get_slider_range(slider)
                dpg.set_value(slider, min_value)
        else:
            new_arity = param_group.get_arity()
            if dpg.is_item_visible(self.right_slider):
                old_arity = 2
            elif dpg.is_item_visible(self.left_slider):
                old_arity = 1
            else:
                old_arity = 0  # discrete

            if new_arity == 2 and old_arity == 1:  # copy value left -> right
                dpg.set_value(self.right_slider, dpg.get_value(self.left_slider))
            elif new_arity == 1:  # arity 1, right slider not in use, so zero it out visually.
                min_value, ignored_max_value = get_slider_range(self.right_slider)
                dpg.set_value(self.right_slider, min_value)
        self.update_ui()

        # chain the main GUI's callback
        if gui_instance is not None:
            gui_instance.on_pose_edited(sender, app_data)

    def write_to_pose(self, pose: List[float]) -> None:
        """Update `pose` (in-place) by the current value(s) set in this control panel.

        Only the currently chosen parameter group is applied.
        """
        if len(self.param_groups) == 0:
            return
        selected_morph_index = self.param_group_names.index(dpg.get_value(self.choice))
        param_group = self.param_groups[selected_morph_index]
        param_index = param_group.get_parameter_index()
        if param_group.is_discrete():
            if dpg.get_value(self.checkbox):
                for i in range(param_group.get_arity()):
                    pose[param_index + i] = 1.0
        else:
            param_range = param_group.get_range()
            pose[param_index] = param_range[0] + (param_range[1] - param_range[0]) * slider_value_to_relpos(self.left_slider)
            if param_group.get_arity() == 2:
                pose[param_index + 1] = param_range[0] + (param_range[1] - param_range[0]) * slider_value_to_relpos(self.right_slider)

    def read_from_pose(self, pose: List[float]) -> None:
        """Overwrite the current value(s) in this control panel by those taken from `pose`.

        All parameter groups in this panel are scanned to find a nonzero value in `pose`.
        The parameter group that first finds a nonzero value wins, selects its morph for this panel,
        and applies the values to the sliders in the panel.

        If nothing matches, the first available morph is selected, and the sliders are set to zero.
        """
        # Find which morph (param group) is active in our category in `pose`.
        for morph_index, param_group in enumerate(self.param_groups):
            param_index = param_group.get_parameter_index()
            param_value = pose[param_index]
            if param_value != 0.0:
                break
            # An arity-2 param group is active also when just the right slider is nonzero.
            if param_group.get_arity() == 2:
                param_value = pose[param_index + 1]
                if param_value != 0.0:
                    break
        else:  # No param group in this panel's category had a nonzero value in `pose`.
            if len(self.param_groups) > 0:
                logger.debug(f"category {self.category_title}: no nonzero values, chose default morph {self.param_group_names[0]}")
                dpg.set_value(self.choice, self.param_group_names[0])  # choose the first param group
                for slider in (self.left_slider, self.right_slider):
                    min_value, ignored_max_value = get_slider_range(slider)
                    dpg.set_value(slider, min_value)
                dpg.set_value(self.checkbox, False)
                self.update_ui()
                return
        logger.debug(f"category {self.category_title}: found nonzero values, chose morph {self.param_group_names[morph_index]}")
        dpg.set_value(self.choice, self.param_group_names[morph_index])
        if param_group.is_discrete():
            for slider in (self.left_slider, self.right_slider):
                min_value, ignored_max_value = get_slider_range(slider)
                dpg.set_value(slider, min_value)
            if pose[param_index]:
                dpg.set_value(self.checkbox, True)
            else:
                dpg.set_value(self.checkbox, False)
        else:
            dpg.set_value(self.checkbox, False)
            param_range = param_group.get_range()
            param_value = pose[param_index]
            relpos = (param_value - param_range[0]) / (param_range[1] - param_range[0])
            dpg.set_value(self.left_slider, relpos_to_slider_value(self.left_slider, relpos))
            if param_group.get_arity() == 2:
                param_value = pose[param_index + 1]
                relpos = (param_value - param_range[0]) / (param_range[1] - param_range[0])
                dpg.set_value(self.right_slider, relpos_to_slider_value(self.right_slider, relpos))
            else:  # arity 1, right slider not in use, so zero it out visually.
                min_value, ignored_max_value = get_slider_range(self.right_slider)
                dpg.set_value(self.right_slider, min_value)
        self.update_ui()


class PoseEditorGUI:
    """Main app window for pose editor."""
    def __init__(self, poser: Poser, device: torch.device, model: str):
        self.poser = poser
        self.dtype = self.poser.get_dtype()
        self.device = device
        self.image_size = self.poser.get_image_size()
        self.gui_extra_height = 510

        with dpg.texture_registry(tag="pose_editor_textures"):
            self.blank_texture = np.zeros([self.image_size,  # height
                                           self.image_size,  # width
                                           4],  # RGBA
                                          dtype=np.float32).ravel()
            self.source_image_texture = dpg.add_raw_texture(width=self.image_size,
                                                            height=self.image_size,
                                                            default_value=self.blank_texture,
                                                            format=dpg.mvFormat_Float_rgba,
                                                            tag="source_image_texture")
            self.result_image_texture = dpg.add_raw_texture(width=self.image_size,
                                                            height=self.image_size,
                                                            default_value=self.blank_texture,
                                                            format=dpg.mvFormat_Float_rgba,
                                                            tag="result_image_texture")

        if args.device.startswith("cuda") and torch.cuda.is_available():
            disp_device = torch.cuda.get_device_name(args.device)
        else:
            disp_device = "CPU"
        dpg.set_viewport_title(f"Raven-avatar pose editor [{disp_device}] [THA3:{model}]")

        with dpg.window(tag="pose_editor_window",
                        label="Raven-avatar poser editor main window") as self.window:  # label not actually shown, since this window is maximized to the whole viewport
            with dpg.group(horizontal=True):
                self.init_left_panel()
                self.init_morphs_control_panel()
                self.init_pose_and_cels_control_panel()
                self.init_right_panel()

        self.fps_statistics = RunningAverage()

        self.last_pose = None
        self.last_base_image = None
        self.last_celstack = None
        self.last_emotion_name = None
        self.last_output_index = dpg.get_value(self.output_index_choice)
        self.last_output_numpy_image = None
        self.render_needed = True
        self.torch_base_image = None  # The loaded character (base image).
        self.torch_cels = {}  # Additional cels that are available for cel blending. {name0: torch_tensor0, ...}
        self.celstack = []  # In order, from bottommost to topmost. Empty stack = use only base image. [(name0, strength0), ...] where strength is in [0, 1].
        self.lock = threading.RLock()

    def init_left_panel(self) -> None:
        """Initialize the input image and emotion preset GUI panel."""
        with dpg.child_window(tag="left_panel",
                              width=self.image_size,
                              height=self.image_size + self.gui_extra_height,
                              no_scrollbar=True,
                              no_scroll_with_mouse=True):
            dpg.add_image("source_image_texture", tag="source_image_image")

            x0, y0 = guiutils.get_widget_relative_pos("source_image_image", reference="left_panel")
            dpg.add_text("[No image loaded]", pos=(x0 + self.image_size / 2 - 60,
                                                   y0 + self.image_size / 2 - (font_size / 2)),
                         tag="source_no_image_loaded_text")

            # Emotion picker.
            emotions_dir = pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "assets", "emotions")).expanduser().resolve()
            self.emotions, self.emotion_names = avatarutil.load_emotion_presets(emotions_dir)

            with dpg.group():
                dpg.add_text("Emotion preset [Ctrl+P]")
                self.emotion_choice = dpg.add_combo(items=self.emotion_names,
                                                    default_value=self.emotion_names[0],
                                                    width=self.image_size - 16,
                                                    callback=self.update_output)

            with dpg.group():
                self.load_image_button = dpg.add_button(label="Load character [Ctrl+O]",
                                                        width=self.image_size - 16,
                                                        callback=show_open_image_dialog)
                self.load_image_button = dpg.add_button(label="Load single emotion template [Ctrl+Shift+O]",
                                                        width=self.image_size - 16,
                                                        callback=show_open_json_dialog)

    def init_morphs_control_panel(self) -> None:
        """Initialize the morphs editor GUI panel."""
        with dpg.child_window(tag="morphs_control_panel",
                              width=0.8 * self.image_size,
                              height=self.image_size + self.gui_extra_height,
                              no_scrollbar=True,
                              no_scroll_with_mouse=True):
            dpg.add_text("Morphs [Ctrl+click to set a numeric value]")
            dpg.add_spacer(height=4)

            morph_categories = [PoseParameterCategory.EYEBROW,
                                PoseParameterCategory.EYE,
                                PoseParameterCategory.MOUTH,
                                PoseParameterCategory.IRIS_MORPH]
            morph_category_titles = {PoseParameterCategory.EYEBROW: "Eyebrow",
                                     PoseParameterCategory.EYE: "Eye",
                                     PoseParameterCategory.MOUTH: "Mouth",
                                     PoseParameterCategory.IRIS_MORPH: "Iris"}
            self.morph_control_panels = {}
            for category in morph_categories:
                param_groups = self.poser.get_pose_parameter_groups()
                filtered_param_groups = [group for group in param_groups if group.get_category().value == category.value]
                if len(filtered_param_groups) == 0:
                    continue
                control_panel = MorphCategoryControlPanel(
                    morph_category_titles[category],
                    category,
                    self.poser.get_pose_parameter_groups())
                # Trigger the choice of the "[custom]" emotion preset (and a recompute) when the pose is edited in this panel.
                # dpg.set_item_callback(control_panel.choice, self.on_pose_edited)  # TODO: this already has a callback, and we need both callbacks, so we chain from that manually.
                dpg.set_item_callback(control_panel.left_slider, self.on_pose_edited)
                dpg.set_item_callback(control_panel.right_slider, self.on_pose_edited)
                dpg.set_item_callback(control_panel.checkbox, self.on_pose_edited)
                self.morph_control_panels[category] = control_panel
                dpg.add_spacer(height=4)

    def init_pose_and_cels_control_panel(self):
        """Initialize the pose editor and cel blending GUI panel."""
        # New to Raven-avatar: cel blending (for things like blush or "intense emotion" eye-wavering effect)
        with dpg.child_window(tag="pose_and_cels_control_panel",
                              width=0.8 * self.image_size,
                              height=self.image_size + self.gui_extra_height,
                              no_scrollbar=True,
                              no_scroll_with_mouse=True):
            dpg.add_text("Pose [Ctrl+click to set a numeric value]")
            dpg.add_spacer(height=4)

            self.non_morph_control_panels = {}
            non_morph_categories = [PoseParameterCategory.IRIS_ROTATION,
                                    PoseParameterCategory.FACE_ROTATION,
                                    PoseParameterCategory.BODY_ROTATION,
                                    PoseParameterCategory.BREATHING]
            for category in non_morph_categories:
                param_groups = self.poser.get_pose_parameter_groups()
                filtered_param_groups = [group for group in param_groups if group.get_category().value == category.value]
                if len(filtered_param_groups) == 0:
                    continue
                control_panel = SimpleParamGroupsControlPanel(category,
                                                              self.poser.get_pose_parameter_groups())
                # Trigger the choice of the "[custom]" emotion preset (and a recompute) when the pose is edited in this panel.
                for slider in control_panel.sliders:
                    dpg.set_item_callback(slider, self.on_pose_edited)
                self.non_morph_control_panels[category] = control_panel
                dpg.add_spacer(height=4)
            dpg.add_spacer(height=8)

            dpg.add_text("Cels [Ctrl+click to set a numeric value]")
            dpg.add_spacer(height=4)

            with dpg.group(tag="cel_blending_controls_group"):
                self.cel_control_panels = {}
                for label in gui_cel_blending_layout:
                    if label is None:
                        dpg.add_spacer(height=4)
                        continue
                    control_panel = CelBlendControlPanel(label=label, parent="cel_blending_controls_group")
                    dpg.set_item_callback(control_panel.slider, self.on_pose_edited)
                    self.cel_control_panels[label] = control_panel
                dpg.add_spacer(height=4)

    def init_right_panel(self) -> None:
        """Initialize the output image and output controls GUI panel."""
        with dpg.child_window(tag="right_panel",
                              width=self.image_size,
                              height=self.image_size + self.gui_extra_height,
                              no_scrollbar=True,
                              no_scroll_with_mouse=True):
            dpg.add_image("result_image_texture", tag="result_image_image")
            x0, y0 = guiutils.get_widget_relative_pos("result_image_image", reference="right_panel")
            dpg.add_text("[No image loaded]", pos=(x0 + self.image_size / 2 - 60,
                                                   y0 + self.image_size / 2 - (font_size / 2)),
                         tag="result_no_image_loaded_text")

            with dpg.group():
                dpg.add_text("Output index [Ctrl+I] [meaning depends on the model]")
                self.output_index_choice_items = [str(i) for i in range(self.poser.get_output_length())]
                self.output_index_choice = dpg.add_combo(items=self.output_index_choice_items,
                                                         default_value=self.output_index_choice_items[0],
                                                         width=self.image_size - 16,
                                                         callback=self.update_output)

            with dpg.group():
                self.save_image_button = dpg.add_button(label="Save posed image and emotion [Ctrl+S]",
                                                        width=self.image_size - 16,
                                                        callback=show_save_image_dialog)
                self.save_all_emotions_button = dpg.add_button(label="Batch save image and emotion from all presets [Ctrl+Shift+S]",
                                                               width=self.image_size - 16,
                                                               callback=show_save_all_emotions_dialog)

            self.fps_text = dpg.add_text("FPS counter will appear here", color=(0, 255, 0))

    def focus_presets(self) -> None:
        dpg.focus_item(self.emotion_choice)

    # TODO: Add hotkeys for each morph control group, and for the non-morph control groups.
    def focus_editor(self) -> None:
        if not self.morph_control_panels:
            return
        first_morph_control_panel = list(self.morph_control_panels.values())[0]
        dpg.focus_item(first_morph_control_panel.choice)

    def focus_output_index(self) -> None:
        dpg.focus_item(self.output_index_choice)

    def on_pose_edited(self, sender, app_data) -> None:
        """Automatically choose the '[custom]' emotion preset (to indicate edited state) when the pose is manually edited."""
        # logger.debug(f"on_pose_edited: sender = {sender}, app_data = {app_data}")
        dpg.set_value(self.emotion_choice, self.emotion_names[0])
        self.last_emotion_name = self.emotion_names[0]
        self.update_output()

    def load_image(self, image_file_name: str) -> None:
        """Load an input image and its associated add-on cels, if any.

        The cels should have the same basename, followed by an underscore and then the cel name.
        E.g. "example.png" may have a cel "example_blush.png".

        We keep all cels in memory as linear RGB, with alpha channel, in Torch layout, on GPU.

        The texture (float32 CPU, ravel) and poser (range [-1, 1]) conversions are done as the last step.
        """
        _load = functools.partial(avatarutil.torch_load_rgba_image,
                                  target_w=self.image_size,
                                  target_h=self.image_size,
                                  device=self.device,
                                  dtype=self.dtype)  # load to GPU in linear RGB
        try:
            logger.info(f"PoseEditorGUI.load_image: Loading '{image_file_name}'.")
            self.torch_base_image = _load(image_file_name)

            # Send the original unmodified base image (before any cel blending) to the GUI source image panel
            numpy_source_image = avatarutil.torch_image_to_numpy(self.torch_base_image)
            raw_data = numpy_source_image.ravel()  # [h, w, c] -> linearly indexed
            dpg.set_value(self.source_image_texture, raw_data)  # send the cel-bldended final source image to the GUI

            # Scan for and load add-on cels for cel blending.
            # This sets up which cels are actually available for this character.
            # The compositor skips any effects for which the current character has no cel, so we don't need to worry about that.
            # Here in the pose editor, we also skip all animefx cels by only loading `supported_cels` (cels for the semi-realistic cel blend effects).
            cels_filenames = avatarutil.scan_addon_cels(image_file_name)
            self.torch_cels = {celname: _load(filename) for celname, filename in cels_filenames.items() if celname in avatarutil.supported_cels}

            # Load all supported cels into the stack, but set the blend strengths to zero.
            # The live animator needs all cels to be present in the emotion template, even "waver2",
            # which isn't directly settable in the GUI. It is used as part of the eye-waver animation,
            # with the animation strength controlled by the "waver1" slider in the GUI.
            self.celstack.clear()
            for celname in avatarutil.supported_cels:
                self.celstack.append((celname, 0.0))

            self.render_needed = True

            logger.info(f"PoseEditorGUI.load_image: Loaded image '{image_file_name}'.")
        except Exception as exc:
            traceback.print_exc()
            logger.error(f"PoseEditorGUI.load_image: Could not load image '{image_file_name}', reason: {type(exc)}: {exc}")
            self.torch_base_image = None
            self.render_needed = True
            messagebox.modal_dialog(window_title="Error",
                                    message=f"Could not load image '{image_file_name}', reason {type(exc)}: {exc}",
                                    buttons=["Close"],
                                    ok_button="Close",
                                    cancel_button="Close",
                                    centering_reference_window=self.window)
        self.update_output()

    def load_json(self, json_file_name: str) -> None:
        """Load (apply) a custom emotion JSON file.

        Note this is for loading and applying a single emotion only.
        """
        try:
            # Load the emotion JSON file
            logger.info(f"PoseEditorGUI.load_json: Loading '{json_file_name}'.")
            with open(json_file_name, "r") as json_file:
                emotions_from_json = json.load(json_file)
            # TODO: Here we just take the first emotion from the file.
            if not emotions_from_json:
                logger.warning(f"PoseEditorGUI.load_json: No emotions defined in file '{json_file_name}'.")
                return
            first_emotion_name = list(emotions_from_json.keys())[0]  # first in insertion order, i.e. topmost in file
            if len(emotions_from_json) > 1:
                logger.warning(f"PoseEditorGUI.load_json: File '{json_file_name}' contains multiple emotions, loading the first one '{first_emotion_name}'.")
            emotion = emotions_from_json[first_emotion_name]
            posedict = emotion["pose"]
            pose = avatarutil.posedict_to_pose(posedict)
            celstack = list(emotion["cels"].items())

            # Auto-select "[custom]"
            dpg.set_value(self.emotion_choice, self.emotion_names[0])

            # Apply the loaded emotion
            self.set_current_pose(pose, celstack)
        except Exception as exc:
            traceback.print_exc()
            logger.error(f"PoseEditorGUI.load_json: Could not load emotion file '{json_file_name}', reason: {type(exc)}: {exc}")
            messagebox.modal_dialog(window_title="Error",
                                    message=f"Could not load emotion file '{json_file_name}', reason {type(exc)}: {exc}",
                                    buttons=["Close"],
                                    ok_button="Close",
                                    cancel_button="Close",
                                    centering_reference_window=self.window)
        else:
            logger.info(f"PoseEditorGUI.load_json: Loaded emotion file '{json_file_name}'.")
            self.update_output()

    def get_current_pose(self) -> (List[float], List[Tuple[str, float]]):
        """Get the current pose of the character as a list of morph values (in the order the models expect them).

        Also get current cel blend values.

        We do this by reading the values from the UI elements in the control panel.
        """
        current_pose = [0.0 for i in range(self.poser.get_num_parameters())]
        for panel in self.morph_control_panels.values():
            panel.write_to_pose(current_pose)
        for panel in self.non_morph_control_panels.values():
            panel.write_to_pose(current_pose)
        current_celstack = [(celname, 0.0) for celname in avatarutil.supported_cels]
        for cel_control_panel in self.cel_control_panels.values():
            cel_control_panel.write_to_celstack(current_celstack)
        return current_pose, current_celstack

    def set_current_pose(self, pose: List[float], celstack: List[Tuple[str, float]]) -> None:
        """Write `pose` and `celstack` to the UI controls in the editor panel."""
        # `update_output` calls us; but if it is not already running (i.e. if we are called by something else),
        # we should not let it run until the pose update is complete.
        with self.lock:
            for panel in self.morph_control_panels.values():
                panel.read_from_pose(pose)
            for panel in self.non_morph_control_panels.values():
                panel.read_from_pose(pose)
            for panel in self.cel_control_panels.values():
                panel.read_from_celstack(celstack)

    def _render(self, celstack: Dict[str, float], pose: List[float], output_index: int):
        """Render `celstack` and `pose` on active device. Return the image data as a float NumPy array of shape [h, w, c], on CPU."""
        assert self.torch_base_image is not None

        posetensor = torch.tensor(pose, device=self.device, dtype=self.dtype)
        with torch.no_grad():
            # Cel blending: apply cels to base image.
            torch_source_image = compositor.render_celstack(self.torch_base_image, celstack, self.torch_cels)

            # # DEBUG
            # numpy_source_image = avatarutil.torch_image_to_numpy(torch_source_image)  # display-ready copy to CPU, shape [h, w, c], range [0, 1], SRGB, 4 channels (RGBA), for GUI
            # raw_data = numpy_source_image.ravel()  # [h, w, c] -> linearly indexed
            # dpg.set_value(self.source_image_texture, raw_data)  # send the cel-blended final source image to the GUI

            # data range [0, 1] -> [-1, 1], for poser
            torch_source_image.mul_(2.0)
            torch_source_image.sub_(1.0)

            # - model's data range is [-1, +1], linear intensity ("gamma encoded")
            output_image = self.poser.pose(torch_source_image, posetensor, output_index)[0].float()

            # [-1, 1] -> [0, 1]
            output_image.add_(1.0)
            output_image.mul_(0.5)
            output_image[:3, :, :] = torch_linear_to_srgb(output_image[:3, :, :])  # apply gamma correction

            # reshape [c, h, w] -> [h, w, c]
            c, h, w = output_image.shape
            output_image = torch.transpose(output_image.reshape(c, h * w), 0, 1).reshape(h, w, c)

            arr = output_image.detach().cpu().numpy()
        return arr

    def update_output(self) -> None:
        """Render the output image, and update the "no image loaded" widget status."""
        with self.lock:
            if self.torch_base_image is None:  # no character loaded -> nothing to do
                return

            # Apply the currently selected emotion, unless "[custom]" is selected, in which case skip this.
            # Note this may modify the current pose, hence we do this first.
            current_emotion_name = dpg.get_value(self.emotion_choice)
            if current_emotion_name != self.emotion_names[0] and current_emotion_name != self.last_emotion_name:  # changed, and not "[custom]"
                self.last_emotion_name = current_emotion_name
                logger.info(f"Loading emotion preset {current_emotion_name}")
                emotion = self.emotions[current_emotion_name]
                celstack = emotion["cels"]
                posedict = emotion["pose"]
                pose = avatarutil.posedict_to_pose(posedict)
                self.set_current_pose(pose, celstack)
                current_pose, current_celstack = pose, celstack
            else:
                current_pose, current_celstack = self.get_current_pose()

            output_index = int(dpg.get_value(self.output_index_choice))
            if not self.render_needed:  # render can always be forced by setting `render_needed` (e.g. when loading a character)
                if (self.last_pose is not None and
                        self.last_pose == current_pose and
                        self.last_output_index == output_index and
                        self.last_celstack is not None and
                        self.last_celstack == current_celstack and
                        self.last_base_image is self.torch_base_image):
                    return
            try:
                if self.torch_base_image is None:  # anything to render?
                    dpg.set_value(self.source_image_texture, self.blank_texture)
                    dpg.set_value(self.result_image_texture, self.blank_texture)
                    dpg.show_item("source_no_image_loaded_text")
                    dpg.show_item("result_no_image_loaded_text")
                    return
                dpg.hide_item("source_no_image_loaded_text")
                dpg.hide_item("result_no_image_loaded_text")

                render_start_time = time.time_ns()

                # Cel-blend and apply the poser.
                numpy_output_image = self._render(current_celstack, current_pose, output_index)  # apply the poser
                raw_data = numpy_output_image.ravel()  # shape [h, w, c] -> linearly indexed
                dpg.set_value(self.result_image_texture, raw_data)  # send the final posed image to the GUI

                self.last_output_numpy_image = numpy_output_image  # for file saving
            except Exception as exc:
                traceback.print_exc()
                dpg.set_value(self.result_image_texture, self.blank_texture)
                dpg.show_item("result_no_image_loaded_text")
                logger.error(f"Could not render, reason: {type(exc)}: {exc}")
                messagebox.modal_dialog(window_title="Error",
                                        message=f"Could not render, reason {type(exc)}: {exc}",
                                        buttons=["Close"],
                                        ok_button="Close",
                                        cancel_button="Close",
                                        centering_reference_window=self.window)
                return
            else:
                # Update FPS counter, measuring the render speed only.
                elapsed_time = time.time_ns() - render_start_time
                fps = 1.0 / (elapsed_time / 10**9)
                if self.torch_base_image is not None:
                    self.fps_statistics.add_datapoint(fps)
                dpg.set_value(self.fps_text, f"Render (avg): {self.fps_statistics.average():0.2f} FPS")
            finally:
                self.render_needed = False
                self.last_pose = current_pose
                self.last_output_index = output_index
                self.last_celstack = current_celstack
                self.last_base_image = self.torch_base_image

    def save_image(self, image_file_name: str) -> None:
        """Save the output image.

        The pose is automatically saved into the same directory as the output image, with
        file name determined from the image file name (e.g. "my_emotion.png" -> "my_emotion.json").
        """
        if self.torch_base_image is None:  # no character loaded -> do nothing
            return

        current_pose, current_celstack = self.get_current_pose()
        current_posedict = avatarutil.pose_to_posedict(current_pose)
        self.save_numpy_image(self.last_output_numpy_image,
                              current_celstack,
                              current_posedict,
                              image_file_name)

        # Since it is possible to save the image and emotion template JSON to "raven/avatar/assets/emotions", refresh the emotion presets list.

        current_emotion_name = dpg.get_value(self.emotion_choice)

        emotions_dir = pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "assets", "emotions")).expanduser().resolve()
        self.emotions, self.emotion_names = avatarutil.load_emotion_presets(emotions_dir)

        dpg.configure_item(self.emotion_choice, items=self.emotion_names)
        if current_emotion_name in self.emotion_names:  # still exists after update?
            dpg.set_value(self.emotion_choice, current_emotion_name)
        else:
            dpg.set_value(self.emotion_choice, self.emotion_names[0])

    def save_all_emotions(self, dir_name: str) -> None:
        """Batch save an output image using each of the emotion presets.

        Does not affect the output image displayed in the GUI.
        """
        if self.torch_base_image is None:  # no character loaded -> do nothing
            return

        logger.info(f"Batch saving output based on all emotion presets to directory {dir_name}...")

        if not os.path.exists(dir_name):
            p = pathlib.Path(dir_name).expanduser().resolve()
            pathlib.Path.mkdir(p, parents=True, exist_ok=True)

        for emotion_name, emotion in self.emotions.items():
            if emotion_name.startswith("[") and emotion_name.endswith("]"):
                continue  # skip "[custom]" and "[reset]"
            image_file_name = os.path.join(dir_name, f"{emotion_name}.png")
            try:
                celstack = emotion["cels"]
                posedict = emotion["pose"]
                pose = avatarutil.posedict_to_pose(posedict)
                output_index = int(dpg.get_value(self.output_index_choice))
                arr = self._render(celstack, pose, output_index)

                self.save_numpy_image(arr,
                                      celstack,
                                      posedict,
                                      image_file_name)
            except Exception as exc:
                logger.error(f"Could not save '{image_file_name}', reason: {type(exc)}: {exc}")

        # Save `_emotions.json`, for use as customized emotion templates.
        #
        # There are three possibilities what we could do here:
        #
        #   - Trim away any morphs that have a zero value, because zero is the default,
        #     optimizing for file size. But this is just a small amount of text anyway.
        #   - Add any zero morphs that are missing. Because `self.emotions` came from files,
        #     it might not have all keys. This yields an easily editable file that explicitly
        #     lists what is possible.
        #   - Just dump the data from `self.emotions` as-is. This way the content for each
        #     emotion  matches the emotion templates in `talkinghead/emotions/*.json`.
        #     This approach is the most transparent.
        #
        # At least for now, we opt for transparency. It is also the simplest to implement.
        #
        # Note that what we produce here is not a copy of `_defaults.json`, but instead, the result
        # of the loading logic with fallback. That is, the content of the individual emotion files
        # overrides the factory presets as far as `self.emotions` is concerned.
        #
        # We just trim away the [custom] and [reset] "emotions", which have no meaning outside the manual poser.
        # The result will be stored in alphabetically sorted order automatically, because `dict` preserves
        # insertion order, and `self.emotions` itself is stored alphabetically.
        logger.info(f"Saving {dir_name}/_emotions.json...")
        trimmed_emotions = {k: v for k, v in self.emotions.items() if not (k.startswith("[") and k.endswith("]"))}
        emotions_json_file_name = os.path.join(dir_name, "_emotions.json")
        with open(emotions_json_file_name, "w") as file:
            json.dump(trimmed_emotions, file, indent=4)

        logger.info("Batch save finished.")

    def save_numpy_image(self,
                         numpy_image: np.array,
                         celstack: List[Tuple[str, float]],
                         posedict: Dict[str, float],
                         image_file_name: str) -> None:
        """Save the output image.

        `numpy_image` is an RGBA float array with range [0, 1].
        `celstack` is the stack of cels that corresponds to the `numpy_image`, for saving "cels" to the emotion JSON.
        `posedict` is the pose dictionary that corresponds to the `numpy_image`, for saving "pose" to the emotion JSON.

        Output format is determined by file extension (which must be supported by the installed `Pillow`).
        Automatically save also the corresponding settings as JSON.

        The settings are saved into the same directory as the output image, with file name determined
        from the image file name (e.g. "my_emotion.png" -> "my_emotion.json").
        """
        numpy_image = np.uint8(255.0 * numpy_image)

        try:
            pil_image = PIL.Image.fromarray(numpy_image, mode="RGBA")
            os.makedirs(os.path.dirname(image_file_name), exist_ok=True)
            pil_image.save(image_file_name)
        except Exception as exc:
            logger.error(f"Could not save '{image_file_name}', reason: {type(exc)}: {exc}")
        else:
            logger.info(f"Saved image '{image_file_name}'")

        json_file_path = os.path.splitext(image_file_name)[0] + ".json"

        filename_without_extension = os.path.splitext(os.path.basename(image_file_name))[0]
        # JSON structure: {emotion_name0: {"pose": posedict0, "cels": celdict0}, ...}
        celdict = dict(celstack)
        data_dict_with_filename = {filename_without_extension: {"pose": posedict, "cels": celdict}}

        try:
            with open(json_file_path, "w") as file:
                json.dump(data_dict_with_filename, file, indent=4)
        except Exception as exc:
            logger.error(f"Could not save '{json_file_path}', reason: {type(exc)}: {exc}")
        else:
            logger.info(f"Saved emotion file '{json_file_path}'")

# Hotkey support
choice_map = None
def pose_editor_hotkeys_callback(sender, app_data):
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
        if key == dpg.mvKey_O:
            show_open_json_dialog()
        elif key == dpg.mvKey_S:
            show_save_all_emotions_dialog()

    # Ctrl+...
    elif ctrl_pressed:
        if key == dpg.mvKey_O:
            show_open_image_dialog()
        elif key == dpg.mvKey_S:
            show_save_image_dialog()
        elif key == dpg.mvKey_P:
            gui_instance.focus_presets()
        elif key == dpg.mvKey_I:
            gui_instance.focus_output_index()

    # Bare key
    #
    # NOTE: These are global across the whole app (when no modal window is open) - be very careful here!
    else:
        # {widget_tag_or_id: list_of_choices}
        global choice_map
        if choice_map is None:  # build on first use
            choice_map = {gui_instance.emotion_choice: (gui_instance.emotion_names, lambda sender, app_data: gui_instance.update_output()),
                          gui_instance.output_index_choice: (gui_instance.output_index_choice_items, lambda sender, app_data: gui_instance.update_output())}
            for panel in gui_instance.morph_control_panels.values():
                choice_map[panel.choice] = (panel.param_group_names, panel.on_choice_updated)
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

with dpg.handler_registry(tag="pose_editor_handler_registry"):  # global (whole viewport)
    dpg.add_key_press_handler(tag="pose_editor_hotkeys_handler", callback=pose_editor_hotkeys_callback)

# --------------------------------------------------------------------------------
# Start the app

parser = argparse.ArgumentParser(description="THA 3 Manual Poser. Pose a character image manually. Useful for generating static expression images and for editing the emotion templates.")
parser.add_argument("--device",
                    type=str,
                    required=False,
                    default="cuda",
                    choices=["cpu", "cuda"],
                    help='The device to use for PyTorch ("cuda" for GPU, "cpu" for CPU).')
parser.add_argument("--model",
                    type=str,
                    required=False,
                    default="separable_float",
                    choices=["standard_float", "separable_float", "standard_half", "separable_half"],
                    help="The model to use. 'float' means fp32, 'half' means fp16.")
parser.add_argument("--factory-reset",
                    metavar="EMOTION",
                    type=str,
                    help="Overwrite the emotion preset EMOTION with its factory default, and exit. This CANNOT be undone!",
                    default="")
parser.add_argument("--factory-reset-all",
                    action="store_true",
                    help="Overwrite ALL emotion presets with their factory defaults, and exit. This CANNOT be undone!")
args = parser.parse_args()

# Blunder recovery options
if args.factory_reset_all:
    print("Factory-resetting all emotion templates...")
    with open(pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "assets", "emotions", "_defaults.json")).expanduser().resolve(), "r") as json_file:
        factory_default_emotions = json.load(json_file)
    factory_default_emotions.pop("zero")  # not an actual emotion
    for key in factory_default_emotions:
        with open(pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "assets", "emotions", f"{key}.json")).expanduser().resolve(), "w") as file:
            json.dump({key: factory_default_emotions[key]}, file, indent=4)
    print("Done.")
    sys.exit(0)
if args.factory_reset:
    key = args.factory_reset
    print(f"Factory-resetting emotion template '{key}'...")
    with open(pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "assets", "emotions", "_defaults.json")).expanduser().resolve(), "r") as json_file:
        factory_default_emotions = json.load(json_file)
    factory_default_emotions.pop("zero")  # not an actual emotion
    if key not in factory_default_emotions:
        print(f"No such factory-defined emotion: '{key}'. Valid values: {sorted(list(factory_default_emotions.keys()))}")
        sys.exit(1)
    with open(pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "assets", "emotions", f"{key}.json")).expanduser().resolve(), "w") as file:
        json.dump({key: factory_default_emotions[key]}, file, indent=4)
    print("Done.")
    sys.exit(0)

# Install the THA3 models if needed
modelsdir = str(talkinghead_path / "tha3" / "models")
maybe_install_models(hf_reponame=server_config.talkinghead_models, modelsdir=modelsdir)

try:
    device = torch.device(args.device)
    poser = load_poser(args.model, device, modelsdir=modelsdir)
except RuntimeError as e:
    logger.error(e)
    sys.exit(255)

# Create the "output" directory under the CWD, if it doesn't exist. This is our default save location.  # TODO: Maybe needs a better default save location.
p = pathlib.Path("output").expanduser().resolve()
pathlib.Path.mkdir(p, parents=True, exist_ok=True)

gui_instance = PoseEditorGUI(poser, device, args.model)

def shutdown():
    gui_animation.animator.clear()
dpg.set_exit_callback(shutdown)

dpg.set_primary_window(gui_instance.window, True)  # Make this DPG "window" occupy the whole OS window (DPG "viewport").
dpg.set_viewport_vsync(True)
dpg.show_viewport()

initialize_filedialogs()

def tune_viewport():
    dpg.set_viewport_width(3.6 * gui_instance.image_size + 40)
    dpg.set_viewport_height(gui_instance.image_size + gui_instance.gui_extra_height + 16)
    dpg.split_frame()
    dpg.set_viewport_resizable(False)
dpg.set_frame_callback(10, tune_viewport)

def update_animations():
    gui_animation.animator.render_frame()  # Our customized fdialog needs this for its overwrite confirm button flash.

# We control the render loop manually to have a convenient place to update our GUI animations just before rendering each frame.
while dpg.is_dearpygui_running():
    update_animations()
    dpg.render_dearpygui_frame()
# dpg.start_dearpygui()  # automatic render loop

dpg.destroy_context()

def main():  # TODO: we don't really need this; it's just for console_scripts so that we can provide a command-line entrypoint.
    pass

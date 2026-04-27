#!/usr/bin/env python
"""Visualize BibTeX data. This can put an entire field of science into one picture."""

# As any GUI app, this visualizer has lots of state. The clearest presentation here is as a script interleaving function definitions
# and GUI creation, with the state stored in module-level globals.
#
# Hence, we are extra careful: all module-level globals are actually needed somewhere. To avoid polluting the module-level namespace
# with temporaries, we use unpythonic's `@call` to limit the scope of any temporary variables into a temporary function (which is
# really just a code block that gets run immediately).
#
# Any line with at least one string-literal reference to any DPG GUI widget tag is commented with "tag" (no quotes), to facilitate searching.
# To find all, search for both "# tag" (the comment) and "tag=" (widget definitions).

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .. import __version__

logger.info(f"Raven-visualizer version {__version__} starting.")

logger.info("Loading libraries...")
from unpythonic import timer, UnionFilter
with timer() as tim:
    import argparse
    import concurrent.futures
    from copy import deepcopy
    import math
    import os
    import pathlib
    import platform
    import threading
    from typing import Union

    import numpy as np

    from unpythonic.env import env
    envcls = env  # for functions that need an `env` parameter due to `@dlet`, so that they can also instantiate env objects (oops)
    from unpythonic import call, box, unbox, sym

    import dearpygui.dearpygui as dpg

    # Vendored libraries
    from ..vendor.IconsFontAwesome6 import IconsFontAwesome6 as fa  # https://github.com/juliettef/IconFontCppHeaders
    from ..vendor import DearPyGui_Markdown as dpg_markdown  # https://github.com/IvanNazaruk/DearPyGui-Markdown
    from ..vendor.file_dialog.fdialog import FileDialog  # https://github.com/totallynotdrait/file_dialog, but with custom modifications

    from ..client import api
    from ..client import config as client_config

    from ..common import bgtask
    from ..common import numutils
    from ..common import utils as common_utils

    from ..common.gui import animation as gui_animation
    from ..common.gui import helpcard
    from ..common.gui import utils as guiutils

    from .app_state import app_state  # Visualizer-wide shared state namespace (see `app_state.py`)
    from . import annotation
    from . import config as visualizer_config
    from . import importer  # BibTeX importer
    from . import info_panel
    from . import plotter
    from . import selection
    from . import word_cloud

    gui_config = visualizer_config.gui_config  # shorthand, this is used a lot

    # Emit further log messages only from a few select modules (our own plus some vendored)
    for handler in logging.root.handlers:
        handler.addFilter(UnionFilter(logging.Filter(__name__),
                                      logging.Filter("raven.client.mayberemote"),
                                      logging.Filter("raven.client.api"),
                                      logging.Filter("raven.client.util"),
                                      logging.Filter("raven.common.bgtask"),
                                      logging.Filter("raven.common.deviceinfo"),
                                      logging.Filter("raven.common.gui.animation"),
                                      logging.Filter("raven.common.gui.fontsetup"),
                                      logging.Filter("raven.common.gui.utils"),
                                      logging.Filter("raven.common.gui.widgetfinder"),
                                      logging.Filter("raven.common.utils"),
                                      logging.Filter("raven.librarian.llmclient"),
                                      logging.Filter("raven.visualizer.importer"),
                                      logging.Filter("raven.vendor.file_dialog.fdialog")))
logger.info(f"Libraries loaded in {tim.dt:0.6g}s.")

# --------------------------------------------------------------------------------
# Selection management subsystem wire-up
selection.reset_undo_history(_update_gui=False)  # GUI not initialized yet. This is the only time the flag should be set to `False`!


# --------------------------------------------------------------------------------
# Modal window related utilities

def enter_modal_mode():
    """Prepare the GUI for showing a modal window: hide annotation, disable current item button glow, ...

    Call this AFTER showing your modal so that the window detects as being shown in any functionality that checks that.
    This automatically waits for one frame for the window to actually render.
    """
    logger.debug("enter_modal_mode: App entering modal mode.")
    dpg.split_frame()
    app_state.update_mouse_hover(force=True, wait=False)  # hide annotation (just in case it's there)
    info_panel.scroll_position_changed(reset=True)  # force update of current item in `update_current_search_result_status`, so `CurrentItemControlsGlow` disables its highlight

def exit_modal_mode():
    """Restore the GUI to main window mode (when a modal is closed): show annotation if relevant, enable current item button glow, ...

    Call this AFTER hiding your modal so that the window detects as being hidden in any functionality that checks that.
    This automatically waits for one frame for the window to actually render.
    """
    logger.debug("exit_modal_mode: App returning to main window mode.")
    dpg.split_frame()
    info_panel.scroll_position_changed(reset=True)  # force update of current item in `update_current_search_result_status`, so `CurrentItemControlsGlow` enables its highlight
    app_state.update_mouse_hover(force=True, wait=False)  # show annotation if relevant

# Register the modal-mode helpers on `app_state` so submodules can reach them.
app_state.enter_modal_mode = enter_modal_mode
app_state.exit_modal_mode = exit_modal_mode

def is_any_modal_window_visible():
    """Return whether *some* modal window is open.

    Currently these are the help card, the "open file" dialog, and the "save word cloud" dialog.
    """
    return (is_open_file_dialog_visible() or word_cloud.is_save_dialog_visible() or
            is_open_import_dialog_visible() or is_save_import_dialog_visible() or
            help_window.is_visible())

# Register on `app_state` so submodules (e.g. `annotation`) can call it.
app_state.is_any_modal_window_visible = is_any_modal_window_visible

# --------------------------------------------------------------------------------
# Set up DPG - basic startup, load fonts, set up global theme

# We do this as early as possible, because before the startup is complete, trying to `dpg.add_xxx` or `with dpg.xxx:` anything will segfault the app.

logger.info("DPG bootup...")
with timer() as tim:
    dpg.create_context()

    app_state.themes_and_fonts = guiutils.bootup(font_size=gui_config.font_size)

    # https://dearpygui.readthedocs.io/en/latest/documentation/themes.html#plot-colors
    with dpg.theme(tag="my_plotter_theme"):
        with dpg.theme_component(dpg.mvPlot):
            dpg.add_theme_color(dpg.mvPlotCol_AxisGrid, gui_config.plotter_grid_color, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_PlotBg, gui_config.plotter_background_color, category=dpg.mvThemeCat_Plots)
            # Disable the axis mouseover highlight, matching the axis colors to the plotter frame color in the default theme (measured using GIMP).
            dpg.add_theme_color(dpg.mvPlotCol_AxisBg, (51, 51, 51), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_AxisBgActive, (51, 51, 51), category=dpg.mvThemeCat_Plots)  # TODO: what is this?
            dpg.add_theme_color(dpg.mvPlotCol_AxisBgHovered, (51, 51, 51), category=dpg.mvThemeCat_Plots)

    # Initialize textures.
    with dpg.texture_registry(tag="app_textures"):
        dpg.add_raw_texture(width=gui_config.word_cloud_w,  # TODO: once we add a settings dialog, we may need to change the texture size while the app is running.
                            height=gui_config.word_cloud_h,
                            default_value=np.ones([gui_config.word_cloud_h, gui_config.word_cloud_w, 4], dtype=np.float64),
                            format=dpg.mvFormat_Float_rgba,
                            tag="word_cloud_texture")

    if platform.system().upper() == "WINDOWS":
        icon_ext = "ico"
    else:
        icon_ext = "png"

    dpg.create_viewport(title=f"Raven-visualizer {__version__}",
                        small_icon=str(pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "icons", f"app_128_notext.{icon_ext}")).expanduser().resolve()),
                        large_icon=str(pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "icons", f"app_256.{icon_ext}")).expanduser().resolve()),
                        width=gui_config.main_window_w,
                        height=gui_config.main_window_h)  # OS window (DPG "viewport")
    dpg.setup_dearpygui()
logger.info(f"    Done in {tim.dt:0.6g}s.")

# --------------------------------------------------------------------------------
# Dataset loading

app_state.dataset = None  # currently loaded dataset (as an `unpythonic.env.env`)


def clear_background_tasks(wait: bool):
    """Stop (cancel) and delete all background tasks."""
    info_panel.clear_tasks(wait=wait)
    annotation.clear_tasks(wait=wait)
    word_cloud.clear_tasks(wait=wait)

def reset_app_state(_update_gui=True):
    """Reset everything, to prepare for loading new data to the GUI.

    `_update_gui`: internal, used during app exit.
                   Everywhere else, should be the default `True`.
    """
    reason = "for loading new data to the GUI" if _update_gui else "(app exiting)"
    logger.info(f"Resetting app state {reason}.")

    # Stop old background tasks (and wait until they actually exit)
    clear_background_tasks(wait=True)

    # Stop GUI animations
    gui_animation.animator.clear()

    # Only update the GUI elements if not exiting, because when exiting, the GUI is already being deleted.
    if _update_gui:
        # Re-add the background animations that should always be present in the animator.
        # These monitor the app state and live-update at every frame.
        gui_animation.animator.add(PlotterPulsatingGlow(cycle_duration=gui_config.glow_cycle_duration))
        gui_animation.animator.add(CurrentItemControlsGlow(cycle_duration=gui_config.glow_cycle_duration))

        # Clear undo history and selection
        selection.reset_undo_history()
        selection.update(common_utils.make_blank_index_array(), mode="replace", force=True, wait=False, update_selection_undo_history=False)

        # Clear the search
        dpg.set_value("search_field", "")  # tag
        update_search(wait=False)

        # Remove old data series, if any
        dpg.delete_item("axis1", children_only=True)  # tag

        # But restore the highlights for the next dataset
        plotter.create_highlight_series()

        # Delete old cluster-color-coding scatterplot themes
        plotter.clear_cluster_color_themes()

        dpg.set_item_label("plot", "Semantic map [no dataset loaded]")  # tag  # TODO: DRY duplicate definitions for labels


def open_file(filename):
    """Load new data into the GUI. Public API."""
    logger.info(f"open_file: Opening file '{filename}'.")
    reset_app_state()
    app_state.dataset = plotter.parse_dataset_file(filename)
    plotter.load_dataset(app_state.dataset)

# --------------------------------------------------------------------------------
# File dialog init

filedialog_open = None
app_state.filedialog_save = None
filedialog_open_import = None
filedialog_save_import = None

def initialize_filedialogs(default_path):  # called at app startup, once we parse the default path from cmdline args (or set a default if not specified).
    """Create the file dialogs."""
    global filedialog_open
    global filedialog_open_import
    global filedialog_save_import
    filedialog_open = FileDialog(title="Open dataset",
                                 tag="open_file_dialog",
                                 callback=_open_file_callback,
                                 modal=True,
                                 filter_list=[".pickle"],
                                 file_filter=".pickle",
                                 multi_selection=False,
                                 allow_drag=False,
                                 default_path=default_path)
    app_state.filedialog_save = FileDialog(title="Save word cloud as PNG",
                                           tag="save_word_cloud_dialog",
                                           callback=word_cloud.save_callback,
                                           modal=True,
                                           filter_list=[".png"],
                                           file_filter=".png",
                                           save_mode=True,
                                           default_file_extension=".png",  # used if the user does not provide a file extension when naming the save-as
                                           allow_drag=False,
                                           default_path=default_path)
    filedialog_open_import = FileDialog(title="Choose BibTeX file(s) to import [Ctrl+click to multi-select]",
                                        tag="open_import_dialog",
                                        callback=_open_import_callback,
                                        modal=True,
                                        filter_list=[".bib"],
                                        file_filter=".bib",
                                        multi_selection=True,
                                        allow_drag=False,
                                        default_path=default_path)
    filedialog_save_import = FileDialog(title="Save imported dataset as",
                                        tag="save_import_dialog",
                                        callback=_save_import_callback,
                                        modal=True,
                                        filter_list=[".pickle"],
                                        file_filter=".pickle",
                                        save_mode=True,
                                        default_file_extension=".pickle",  # used if the user does not provide a file extension when naming the save-as
                                        allow_drag=False,
                                        default_path=default_path)

# --------------------------------------------------------------------------------
# "Open file" dialog

def show_open_file_dialog():
    """Button callback. Show the open file dialog, for the user to pick a dataset to open.

    (And prepare the GUI for it: hide annotation, disable current item button glow, ...)
    If you need to close it programmatically, call `filedialog_open.cancel()` so it'll trigger the callback (necessary to restore the GUI back into main window mode).
    """
    logger.debug("show_open_file_dialog: Showing open file dialog.")
    filedialog_open.show_file_dialog()
    app_state.enter_modal_mode()
    logger.debug("show_open_file_dialog: Done.")

def _open_file_callback(selected_files):
    """Callback that fires when the open file dialog closes."""
    logger.debug("_open_file_callback: Open file dialog callback triggered.")
    app_state.exit_modal_mode()
    if len(selected_files) > 1:  # Should not happen, since we set `multi_selection=False`.
        raise ValueError(f"Expected at most one selected file, got {len(selected_files)}.")
    if selected_files:
        selected_file = selected_files[0]
        logger.debug(f"_open_file_callback: User selected the file '{selected_file}'.")
        open_file(selected_file)
    else:  # empty selection -> cancelled
        logger.debug("_open_file_callback: Cancelled.")

def is_open_file_dialog_visible():
    """Return whether the open file dialog is open.

    We have this abstraction (not just `dpg.is_item_visible`) because the window might not exist yet.
    """
    if filedialog_open is None:
        return False
    return dpg.is_item_visible("open_file_dialog")  # tag

# --------------------------------------------------------------------------------
# BibTeX importer integration

importer_input_files_box = box([])
importer_output_file_box = box("")

importer_action_start = sym("start")
importer_action_stop = sym("stop")

def toggle_importer_window():
    """Show/hide the BibTeX importer window."""
    if dpg.is_item_visible("importer_window"):
        dpg.hide_item("importer_window")
    else:
        dpg.show_item("importer_window")
        guiutils.recenter_window("importer_window", reference_window=main_window)

def show_open_import_dialog():
    """Button callback. Show the open import file dialog, for the user to pick which BibTeX files to import."""
    logger.debug("show_open_import_dialog: Showing open import dialog.")
    filedialog_open_import.show_file_dialog()
    app_state.enter_modal_mode()
    logger.debug("show_open_import_dialog: Done.")

def _open_import_callback(selected_files):
    """Callback that fires when the open import file dialog closes."""
    logger.debug("_open_import_callback: Open import dialog callback triggered.")
    app_state.exit_modal_mode()
    if selected_files:
        logger.debug(f"_open_import_callback: User selected the file(s) {selected_files}.")
        importer_input_files_box << deepcopy(selected_files)  # Make a copy of the filename list, so that the GUI dialog can clear its own list without affecting ours.
        update_open_import_gui_table()
    else:  # empty selection -> cancelled
        logger.debug("_open_import_callback: Cancelled.")

def is_open_import_dialog_visible():
    """Return whether the open import file dialog is open.

    We have this abstraction (not just `dpg.is_item_visible`) because the window might not exist yet.
    """
    if filedialog_open_import is None:
        return False
    return dpg.is_item_visible("open_import_dialog")  # tag

def show_save_import_dialog():
    """Button callback. Show the save import file dialog, to ask the user for a filename to save the imported dataset as."""
    logger.debug("show_save_import_dialog: Showing save import dialog.")
    filedialog_save_import.show_file_dialog()
    app_state.enter_modal_mode()
    logger.debug("show_save_import_dialog: Done.")

def _save_import_callback(selected_files):
    """Callback that fires when the save import file dialog closes."""
    logger.debug("_save_import_callback: Save import dialog callback triggered.")
    app_state.exit_modal_mode()
    if len(selected_files) > 1:  # Should not happen, since we set `multi_selection=False`.
        raise ValueError(f"Expected at most one selected file, got {len(selected_files)}.")
    if selected_files:
        selected_file = selected_files[0]
        logger.debug(f"_save_import_callback: User selected the file '{selected_file}'.")
        importer_output_file_box << selected_file
        update_save_import_gui_table()
    else:  # empty selection -> cancelled
        logger.debug("_save_import_callback: Cancelled.")

def is_save_import_dialog_visible():
    """Return whether the save import file dialog is open.

    We have this abstraction (not just `dpg.is_item_visible`) because the window might not exist yet.
    """
    if filedialog_save_import is None:
        return False
    return dpg.is_item_visible("save_import_dialog")  # tag

def update_importer_status():
    """Update the BibTeX importer status in the GUI.

    This is called automatically every frame while the importer task is running.

    This is also called one more time when the importer task exits, via the `done_callback` mechanism.
    """
    # The importer generates the GUI messages. We only need to get them from there.
    dpg.set_value("importer_status_text", unbox(importer.status_box))

    # Update the importer progress bar.
    if importer.progress is not None:
        progress_value = importer.progress.value
    else:
        progress_value = 0.0
    percentage = int(100 * progress_value)
    dpg.set_value("importer_progress_bar", progress_value)
    dpg.configure_item("importer_progress_bar", overlay=f"{percentage}%")
    # dpg.set_item_label("importer_window", f"BibTeX import [running, {percentage}%]")  # TODO: would be nice to see status while minimized, but prevents dragging the window for some reason.

def importer_started_callback(task_env):
    """Callback that fires when the BibTeX importer task actually starts.

    We use this to update the GUI state.
    """
    dpg.set_item_label("importer_startstop_button", fa.ICON_STOP)
    dpg.set_value("importer_startstop_tooltip_text", "Cancel BibTeX import [Ctrl+Enter]")  # TODO: DRY duplicate definitions for labels
    dpg.enable_item("importer_startstop_button")

    dpg.set_item_label("importer_startstop_heading_text_button", "Running; click to cancel")  # TODO: DRY duplicate definitions for labels
    dpg.set_value("importer_startstop_heading_text_tooltip_text", "Cancel BibTeX import [Ctrl+Enter]")  # TODO: DRY duplicate definitions for labels
    dpg.enable_item("importer_startstop_heading_text_button")

def importer_done_callback(task_env):
    """Callback that fires when the BibTeX importer task actually exits, via the `done_callback` mechanism.

    The callback fires regardless of whether the task completed successfully, errored out, or was cancelled.
    See `start_task` for details how to use the `task_env.cancelled`, `task_env.result_code` and `task_env.exc` attributes.

    We use this to update the GUI state.
    """
    update_importer_status()
    dpg.configure_item("importer_progress_bar", overlay="")
    dpg.hide_item("importer_progress_bar")

    dpg.set_item_label("importer_startstop_button", fa.ICON_PLAY)
    dpg.set_value("importer_startstop_tooltip_text", "Start BibTeX import [Ctrl+Enter]")  # TODO: DRY duplicate definitions for labels
    dpg.enable_item("importer_startstop_button")

    dpg.set_item_label("importer_startstop_heading_text_button", "Start")  # TODO: DRY duplicate definitions for labels
    dpg.set_value("importer_startstop_heading_text_tooltip_text", "Start BibTeX import [Ctrl+Enter]")  # TODO: DRY duplicate definitions for labels
    dpg.enable_item("importer_startstop_heading_text_button")

    # dpg.set_item_label("importer_window", "BibTeX import")  # TODO: DRY duplicate definitions for labels

def start_importer(output_file, *input_files):
    """Start the BibTeX importer to import `input_files` (.bib) into `output_file` (Raven-visualizer dataset format, currently .pickle)."""
    if importer.has_task():
        return
    dpg.show_item("importer_progress_bar")
    dpg.disable_item("importer_startstop_button")  # Prevent multiple clicks: wait until the task actually starts before allowing the user to tell it to stop. The button will be re-enabled by the `started_callback`.
    dpg.disable_item("importer_startstop_heading_text_button")
    importer.start_task(importer_started_callback, importer_done_callback, output_file, *input_files)

def stop_importer():
    """Stop (cancel) the BibTeX importer task, if any is running."""
    if not importer.has_task():
        return
    dpg.disable_item("importer_startstop_button")  # We must wait until the previous task actually exits before we can start a new one. The button will be re-enabled by the `done_callback`.
    dpg.disable_item("importer_startstop_heading_text_button")
    dpg.set_item_label("importer_startstop_heading_text_button", "Canceling...")  # TODO: DRY duplicate definitions for labels
    importer.cancel_task()

def start_or_stop_importer():
    """The actual GUI button callback. Start or stop the BibTeX importer task, using the input/output filenames currently selected in the GUI."""
    logger.info("start_or_stop_importer: called.")
    if importer.has_task():
        logger.info("start_or_stop_importer: importer task is running, so we will stop it.")
        action = importer_action_stop
    else:
        logger.info("start_or_stop_importer: no importer task running, so we will start one.")
        action = importer_action_start

    if action is importer_action_start:
        output_file = unbox(importer_output_file_box)
        input_files = unbox(importer_input_files_box)
        logger.info(f"start_or_stop_importer: output file is '{output_file}', input files are '{input_files}'.")
        if output_file and input_files:  # filenames specified?
            logger.info("start_or_stop_importer: filenames have been specified. Invoking importer.")
            start_importer(output_file, *input_files)
        else:
            logger.info("start_or_stop_importer: input, output or both filenames missing. Cannot start importer.")
    else:
        stop_importer()

# --------------------------------------------------------------------------------
# Animations, live updates

search_string_box = box("")
search_result_data_idxs_box = box(common_utils.make_blank_index_array())

# Publish on `app_state` so the annotation tooltip (and later the info panel / search module) can read them.
app_state.search_string_box = search_string_box
app_state.search_result_data_idxs_box = search_result_data_idxs_box

def update_search(wait=True):
    """Perform search and update the search results.

    This gets called automatically when the content of the search field changes via keyboard input.
    This is also explicitly called by a few other use sites, which modify the search field content.

    `wait`: Whether to wait for more keyboard input before starting to render the info panel and
            tooltip annotation updates.

            Passed to `update_info_panel` and to `update_mouse_hover`, which see.
    """
    search_string = dpg.get_value("search_field")  # tag
    if not search_string:
        search_result_data_idxs = common_utils.make_blank_index_array()
    else:
        # Simple O(n) scan for exact matches, ANDed across all fragments. No stopwording, lemmatization or anything fancy.
        # TODO: Search also in document authors (full author list). For this, need to update the GUI wherever we show author names - e.g. searching for "Virtanen" in a paper "Aaltonen et al." that has 200 authors.
        # TODO: With `raven.librarian.hybridir.HybridIR`, we could integrate also a semi-intelligent (keyword + semantic) fulltext search here. Think about the GUI, as the classic mode is useful too.
        case_sensitive_fragments, case_insensitive_fragments = common_utils.search_string_to_fragments(search_string, sort=False)  # minor speedup: don't need to sort, since all must match
        search_result_data_idxs = []
        for data_idx, entry in enumerate(app_state.dataset.sorted_entries):  # `data_idx`: index to `sorted_xxx`
            text = entry.normalized_title
            text_lowercase = text.lower()
            if (all(term in text_lowercase for term in case_insensitive_fragments) and
               all(term in text for term in case_sensitive_fragments)):
                search_result_data_idxs.append(data_idx)
        search_result_data_idxs = np.array(search_result_data_idxs)

    # Send the new data into the boxes
    search_string_box << search_string
    search_result_data_idxs_box << search_result_data_idxs

    if len(search_result_data_idxs):
        # Highlight the search result data points (by plotting them as another series on top).
        dpg.set_value("my_search_results_scatter_series", [list(app_state.dataset.sorted_lowdim_data[search_result_data_idxs, 0]),  # tag
                                                           list(app_state.dataset.sorted_lowdim_data[search_result_data_idxs, 1])])
        # Re-use the "Search" header to show the number of matches.
        plural_s = "es" if len(search_result_data_idxs) != 1 else ""
        dpg.set_value(search_header_text, f"[{len(search_result_data_idxs)} match{plural_s}]")
    else:
        dpg.set_value("my_search_results_scatter_series", [[], []])  # tag
        if not search_string:  # Search not active, restore the "Search" header
            dpg.set_value(search_header_text, "Search")  # TODO: DRY duplicate definitions for labels
        else:  # Search active, but no matches
            dpg.set_value(search_header_text, "[no matches]")

    # Update tooltip and info panel to update the highlight status.
    # TODO: Currently, this may cause the set of data points considered to be under the mouse cursor to change
    #       the first time this happens at a given mouse position (upon a click in the plot area). Debug this.
    #       If the plot mouse position is one frame out of date (update order?), that would explain it.
    app_state.update_info_panel(wait=wait)
    app_state.update_mouse_hover(force=True, wait=wait)

# Register on `app_state` so submodules (e.g. `info_panel`) can call it.
app_state.update_search = update_search


class PlotterPulsatingGlow(gui_animation.Animation):  # this animation is set up by `reset_app_state`
    def __init__(self, cycle_duration):
        """Cyclic animation to pulsate the glow highlight for search result datapoints and selected datapoints."""
        super().__init__()
        self.cycle_duration = cycle_duration

    @classmethod
    def _compute_alpha(cls, x, n_data, n_many):
        """Compute translucency for plotter highlight, accounting for the effect of data mass on perceived translucency.

        High alpha per datapoint when very few datapoints; low alpha per datapoint when many datapoints.

        `x`: float, [0, 1]. The animation control channel. More means brighter.
        `n_data`: int, how many datapoints there are in the set being highlighted.
        `m_many`: int, how many datapoints are so many that the minimum per-datapoint alpha should be used.

        Returns the `alpha` value (int, [0, 255]).
        """
        # Coefficients for `alpha = a0 + a1 * x`, in the maximally bright and maximally dim cases.
        # These have been manually calibrated (via a coarse eyeball estimate) to give the same
        # perceived brightness for the highlighted set of datapoints regardless of the amount of data.
        a0_bright = 64
        a1_bright = 255 - a0_bright
        a0_dim = 32
        a1_dim = 64
        # Interpolate the coefficients from bright to dim, smoothly, depending on relative data mass.
        relative_data_mass = numutils.clamp(n_data / n_many)  # 0 ... 1, linear clamp
        r = numutils.nonanalytic_smooth_transition(relative_data_mass, m=2.0)  # 0 ... 1, smoothed
        a0 = a0_bright * (1.0 - r) + a0_dim * r
        a1 = a1_bright * (1.0 - r) + a1_dim * r
        # Compute the final alpha using the interpolated coefficients.
        alpha = a0 + int(a1 * x)
        # logger.debug(f"compute_data_highlight_alpha: relative data mass = {relative_data_mass}, smooth parameter = {r}, animation control value = {x}; result alpha = {alpha}")
        return alpha

    def render_frame(self, t):
        dt = (t - self.t0) / 10**9  # seconds since t0
        cycle_pos = dt / self.cycle_duration  # number of cycles since t0
        if cycle_pos > 1.0:  # prevent loss of accuracy in long sessions
            self.reset()
        cycle_pos = cycle_pos - float(int(cycle_pos))  # fractional part; raw position in animation cycle

        # We pulsate the search results and selected items at opposite phases to make both easy
        # to see when they overlap. We use colors that make the highlights stand out from the
        # default Viridis colormap used for plotting the data.
        #
        # For how to do this in DPG, see e.g. https://github.com/hoffstadt/DearPyGui/issues/1512
        # Basically, bind a custom theme to the GUI widgets that need to have their color animated,
        # and then edit the theme's colors per-frame (just before render).
        #
        # Convert animation cycle position to animation control channel value.
        # Same approach as in the AI avatar code, see `raven.server.modules.avatar.animate_breathing`.
        animation_pos = math.sin(cycle_pos * math.pi)**2  # 0 ... 1 ... 0, smoothly, with slow start and end, fast middle
        alpha_search = self._compute_alpha(animation_pos,
                                           len(unbox(search_result_data_idxs_box)),
                                           gui_config.n_many_searchresults)
        alpha_selection = self._compute_alpha(1.0 - animation_pos,
                                              len(unbox(app_state.selection_data_idxs_box)),
                                              gui_config.n_many_selection)
        dpg.set_value(search_results_highlight_color, (*gui_config.plotter_search_results_highlight_color, alpha_search))
        dpg.set_value(selection_highlight_color, (*gui_config.plotter_selection_highlight_color, alpha_selection))

        return gui_animation.action_continue


class CurrentItemControlsGlow(gui_animation.Animation):  # this animation is set up by `reset_app_state`
    def __init__(self, cycle_duration):
        """Cyclic animation to pulsate the current item controls.

        Very specific; needs support from outside to update `current_item`, and the highlight
        is hardcoded for the current GUI controls layout (2x2 buttons per info panel item).
        """
        super().__init__()
        self.cycle_duration = cycle_duration

    def render_frame(self, t):
        """Update the highlight on the current item GUI controls.

        `cycle_pos`: float, [0, 1]. Position in the highlight pulsation animation cycle (to sync this with the plotter glow).

        This runs every frame, so the implementation is as minimal as possible, and exits as early as possible.

        This animation is controlled by `info_panel.current_item_info`; see `info_panel.update_current_item_info`.
        """
        if not info_panel.current_item_info_lock.acquire(blocking=False):
            # If we didn't get the lock, it means `current_item_info` is being updated. Never mind, we can try again next frame.
            return gui_animation.action_continue
        try:  # ok, got the lock
            have_current_item = False
            if info_panel.current_item_info.item is not None:
                have_current_item = True
                x0 = info_panel.current_item_info.x0
                y0 = info_panel.current_item_info.y0
                # w = info_panel.current_item_info.w
                # h = info_panel.current_item_info.h
        finally:
            info_panel.current_item_info_lock.release()

        if have_current_item:
            dt = (t - self.t0) / 10**9  # seconds since t0
            cycle_pos = dt / self.cycle_duration  # number of cycles since t0
            if cycle_pos > 1.0:  # prevent loss of accuracy in long sessions
                self.reset()
            cycle_pos = cycle_pos - float(int(cycle_pos))  # fractional part; raw position in animation cycle

            animation_pos = math.sin(cycle_pos * math.pi)**2  # 0 ... 1 ... 0, smoothly, with slow start and end, fast middle
            alpha_min = 16
            alpha_max = 48
            alpha = (1.0 - animation_pos) * alpha_min + animation_pos * alpha_max
            highlight_color = (196, 196, 255, alpha)  # same as `scroll_ends_here_color`, except alpha

            dpg.delete_item("viewport_drawlist", children_only=True)  # tag  # delete old draw items

            # Highlight the button group
            dpg.draw_rectangle((x0 - 4, y0 - 4),
                               (x0 + 2 * gui_config.toolbutton_w + 4, y0 + 2 * gui_config.toolbutton_w + 1),  # kluge constants chosen manually to make this look good
                               color=highlight_color,
                               fill=highlight_color,
                               rounding=8,
                               parent="viewport_drawlist")  # tag
            # # Variant, where the current item is the title text
            # dpg.draw_rectangle((x0 - 4, y0),
            #                    (x0 + gui_config.title_wrap_w + 4, y0 + h + 4),
            #                    color=highlight_color,
            #                    fill=highlight_color,
            #                    rounding=8,
            #                    parent="viewport_drawlist")  # tag
        else:
            dpg.delete_item("viewport_drawlist", children_only=True)  # tag  # delete old draw items

        return gui_animation.action_continue


def update_animations():
    # # Resize the search field dynamically. We don't need this with the current layout; keeping for documentation only.
    # # Note that in DPG, text widgets have no `width` (always zero), but they have a rect_size.
    # w_header, h_header = dpg.get_item_rect_size(search_header_text)
    # w_plotarea = dpg.get_item_width(theplot)
    # # x0, y0 = dpg.get_item_rect_min(search_header_text)
    # # print(w_header, w_plotarea, x0)
    # dpg.set_item_width("search_field", w_plotarea - w_header)

    # HACK: force correct info panel height.
    # At app startup, the main window thinks it has height=100, which is wrong.
    # The scroll end flasher needs the correct height for "item_information_panel"  # tag
    # to know the viewport coordinates for its bottom overlay.
    info_panel.update_height()

    # ----------------------------------------
    # Show loading spinner when info panel is refreshing

    info_panel_render_status_box = bgtask.ManagedTask.get_status_box("raven_visualizer_info_panel_render")
    if unbox(info_panel_render_status_box) is bgtask.status_pending:
        dpg.hide_item("info_panel_rendering_spinner")  # tag
        dpg.show_item("info_panel_pending_spinner")  # tag
    elif unbox(info_panel_render_status_box) is bgtask.status_running:
        dpg.hide_item("info_panel_pending_spinner")  # tag
        dpg.show_item("info_panel_rendering_spinner")  # tag
    else:  # bgtask.status_stopped
        dpg.hide_item("info_panel_pending_spinner")  # tag
        dpg.hide_item("info_panel_rendering_spinner")  # tag

    # ----------------------------------------
    # Update search-related GUI elements

    # Color the search field
    search_string = unbox(search_string_box)
    search_result_data_idxs = unbox(search_result_data_idxs_box)
    if not search_string:
        dpg.set_value(search_field_text_color, (255, 255, 255))  # no search active
    else:
        if len(search_result_data_idxs):
            dpg.set_value(search_field_text_color, (180, 255, 180))  # found, green
        else:
            dpg.set_value(search_field_text_color, (255, 128, 128))  # not found, red

    info_panel.update_current_search_result_status()  # The "[x/x]" topmost currently visible search result indicator (also updates `current_item_info` for `CurrentItemControlsGlow`)

    # ----------------------------------------
    # Update various other things that need per-frame updates

    info_panel.update_navigation_controls()  # Info panel top/bottom/pageup/pagedown buttons

    if importer.has_task():
        update_importer_status()

    # ----------------------------------------
    # Render all currently running animations

    gui_animation.animator.render_frame()


# --------------------------------------------------------------------------------
# Set up the main window

logger.info("Initial GUI setup...")
with timer() as tim:
    with dpg.window(tag="main_window", label="Raven-visualizer main window") as main_window:  # DPG "window" inside the app OS window ("viewport"), container for the whole GUI
        dpg.add_viewport_drawlist(front=True, tag="viewport_drawlist")  # for current item highlight in info panel
        with dpg.group(tag="main_container",
                       horizontal=True):  # Container to make a horizontal top-level layout

            # Info panel
            with dpg.group(tag="info_and_help"):
                # Title
                with dpg.child_window(tag="item_information_header",
                                        width=gui_config.info_panel_w,
                                        height=gui_config.info_panel_header_h,
                                        no_scrollbar=True,  # we want to hide the "hello"
                                        no_scroll_with_mouse=True):
                    with dpg.group(horizontal=True, tag="item_information_header_group"):
                        # Copy report to clipboard button
                        # The callback function is bound in `info_panel.build_window()`.
                        dpg.add_button(tag="copy_report_to_clipboard_button",
                                       label=fa.ICON_COPY,
                                       enabled=False)
                        dpg.bind_item_font("copy_report_to_clipboard_button", app_state.themes_and_fonts.icon_font_solid)  # tag
                        dpg.bind_item_theme("copy_report_to_clipboard_button", "disablable_widget_theme")  # tag
                        with dpg.tooltip("copy_report_to_clipboard_button", tag="copy_report_tooltip"):  # tag
                            dpg.add_text("Copy report to clipboard [F8]\n    no modifier: as plain text\n    with Shift: as Markdown", tag="copy_report_tooltip_text")  # TODO: DRY duplicate definitions for labels

                        # Static header text
                        dpg.add_text("Item information", color=(255, 255, 255, 255), tag="item_information_title")

                        # Dynamic header text, this will be replaced by the item count statistics when something is shown in the info panel.
                        item_information_text = dpg.add_text("[nothing selected]", color=(140, 140, 140, 255), tag="item_information_selection_item_count")  # TODO: DRY duplicate definitions for labels
                        total_count_text = dpg.add_text("[x items shown]", color=(140, 140, 140, 255), tag="item_information_total_count", show=False)

                        # Spinners to indicate that the item info panel is refreshing. The color shows the state (update pending, or updating).
                        # At most one spinner is shown at a time.
                        dpg.add_loading_indicator(style=0,
                                                  radius=1.0,
                                                  color=(255, 96, 96, 255),  # orange
                                                  secondary_color=(128, 32, 32, 255),
                                                  show=False,
                                                  tag="info_panel_pending_spinner")
                        dpg.add_loading_indicator(style=0,
                                                  radius=1.0,
                                                  color=(96, 96, 255, 255),  # blue
                                                  secondary_color=(32, 32, 128, 255),
                                                  show=False,
                                                  tag="info_panel_rendering_spinner")

                    # FIX: Trigger Markdown renderer to load all font families at startup, so it won't bite us with a race condition later when we populate the info panel.
                    #
                    # The render seems to be asynchronous, so if you populate other stuff into the same child window while `dpg_markdown` is loading its fonts,
                    # some place in the rendering engine may forget where it was going. Some of your content will be omitted, and the rest abruptly injected
                    # into the middle of the Markdown render that was in progress. Triggering the font loading now (while we're NOT populating the info panel
                    # in a tight loop) seems to avoid this issue.
                    #
                    # This could also have something to do with the DPG container stack, which seems to be global. Just to be safe, we have already changed
                    # both background renderers (the info panel and the annotation) not to use the container stack, but to parent each GUI widget explicitly.
                    #
                    # Note also that if we call `dpg_markdown.add_text` twice or more before the first frame renders, it segfaults DPG (at least 1.11).
                    # So this is the only Markdown render in the initial main window setup. Any other Markdown text is rendered later.
                    with dpg.group(tag="markdown_font_loader_trigger_dummy"):
                        dpg_markdown.add_text("hello, *hello*, **hello**, ***hello***")  # regular, bold, italic, bold italic

                with dpg.child_window(tag="item_information_navigation_controls",
                                      width=gui_config.info_panel_w,
                                      height=gui_config.info_panel_header_h,
                                      no_scrollbar=True,
                                      no_scroll_with_mouse=True):
                    with dpg.group(horizontal=True, tag="item_information_navigation_controls_group"):
                        # The callback functions for all buttons in this group are defined (and bound) later when we define the info panel.
                        go_to_top_button = dpg.add_button(tag="go_to_top_button",
                                                          label=fa.ICON_ANGLES_UP,
                                                          width=gui_config.info_panel_button_w,
                                                          enabled=False)
                        dpg.bind_item_font("go_to_top_button", app_state.themes_and_fonts.icon_font_solid)  # tag
                        dpg.bind_item_theme("go_to_top_button", "disablable_widget_theme")  # tag
                        with dpg.tooltip("go_to_top_button"):  # tag
                            dpg.add_text("To top [Home, when search field not focused]")

                        page_up_button = dpg.add_button(tag="page_up_button",
                                                        label=fa.ICON_ANGLE_UP,
                                                        width=gui_config.info_panel_button_w,
                                                        enabled=False)
                        dpg.bind_item_font("page_up_button", app_state.themes_and_fonts.icon_font_solid)  # tag
                        dpg.bind_item_theme("page_up_button", "disablable_widget_theme")  # tag
                        with dpg.tooltip("page_up_button"):  # tag
                            dpg.add_text("Page up [Page Up, when search field not focused]")

                        page_down_button = dpg.add_button(tag="page_down_button",
                                                          label=fa.ICON_ANGLE_DOWN,
                                                          width=gui_config.info_panel_button_w,
                                                          enabled=False)
                        dpg.bind_item_font("page_down_button", app_state.themes_and_fonts.icon_font_solid)  # tag
                        dpg.bind_item_theme("page_down_button", "disablable_widget_theme")  # tag
                        with dpg.tooltip("page_down_button"):  # tag
                            dpg.add_text("Page down [Page Down, when search field not focused]")

                        go_to_bottom_button = dpg.add_button(tag="go_to_bottom_button",
                                                             label=fa.ICON_ANGLES_DOWN,
                                                             width=gui_config.info_panel_button_w,
                                                             enabled=False)
                        dpg.bind_item_font("go_to_bottom_button", app_state.themes_and_fonts.icon_font_solid)  # tag
                        dpg.bind_item_theme("go_to_bottom_button", "disablable_widget_theme")  # tag
                        with dpg.tooltip("go_to_bottom_button"):  # tag
                            dpg.add_text("To bottom [End, when search field not focused]")

                        dpg.add_spacer(width=6)

                        # Scroll between search matches buttons.
                        prev_search_match_button = dpg.add_button(tag="prev_search_match_button",
                                                                  # arrow=True,
                                                                  # direction=dpg.mvDir_Up,  # The standard arrow looks too confusing, being close to other arrow buttons (in info panel content) but with different meaning.
                                                                  label=fa.ICON_CIRCLE_UP,
                                                                  width=gui_config.info_panel_button_w,
                                                                  enabled=False)
                        dpg.bind_item_font("prev_search_match_button", app_state.themes_and_fonts.icon_font_solid)  # tag
                        dpg.bind_item_theme("prev_search_match_button", "disablable_widget_theme")  # tag
                        with dpg.tooltip("prev_search_match_button"):  # tag
                            dpg.add_text("Previous search match [Shift+F3]")
                        next_search_match_button = dpg.add_button(tag="next_search_match_button",
                                                                  # arrow=True,
                                                                  # direction=dpg.mvDir_Down,
                                                                  label=fa.ICON_CIRCLE_DOWN,
                                                                  width=gui_config.info_panel_button_w,
                                                                  enabled=False)
                        dpg.bind_item_font("next_search_match_button", app_state.themes_and_fonts.icon_font_solid)  # tag
                        dpg.bind_item_theme("next_search_match_button", "disablable_widget_theme")  # tag
                        with dpg.tooltip("next_search_match_button"):  # tag
                            dpg.add_text("Next search match [F3]")

                        dpg.add_text("[no search active]", color=(140, 140, 140, 255), tag="item_information_search_controls_item_count")  # TODO: DRY duplicate definitions for labels
                        dpg.add_text("[x/x]", color=(140, 140, 140, 255), tag="item_information_search_controls_current_item", show=False)

                # Item information content.
                # The content group itself (alias "info_panel_content_group") is created by `info_panel.build_window()`;
                # it must be the *first* child, before the end spacer — the info panel worker inserts new builds using
                # `before="info_panel_content_end_spacer"`.
                with dpg.child_window(tag="item_information_panel",
                                      width=gui_config.info_panel_w,
                                      height=gui_config.main_window_h - gui_config.info_panel_reserved_h):
                    dpg.add_spacer(width=gui_config.info_panel_w - 20, height=0, tag="info_panel_content_end_spacer")

                # Plotter help
                with dpg.child_window(tag="plotter_help_panel",
                                      width=gui_config.info_panel_w,
                                      autosize_y=True,
                                      no_scrollbar=True,
                                      no_scroll_with_mouse=True):
                    @call  # avoid polluting top-level namespace
                    def _():
                        help_heading_color = (255, 255, 255, 255)
                        help_text_color = (180, 180, 180, 255)
                        dpg.add_text("Plotter help", color=help_heading_color, tag="plotter_help_header_text")
                        with dpg.drawlist(width=gui_config.info_panel_w - 20, height=1):
                            dpg.draw_line((0, 0), (gui_config.info_panel_w - 21, 0), color=(140, 140, 140, 255), thickness=1)
                        with dpg.table(header_row=False):
                            dpg.add_table_column()
                            dpg.add_table_column()
                            dpg.add_table_column()
                            dpg.add_table_column()
                            dpg.add_table_column()
                            dpg.add_table_column()
                            with dpg.table_row():
                                # dpg.add_text("Zoom", color=help_heading_color)
                                # dpg.add_text("Wheel", color=help_dim_color)
                                dpg.add_text("Pan", color=help_heading_color)
                                dpg.add_text("Middle-drag", color=help_text_color)
                                dpg.add_text("Zoom region", color=help_heading_color)
                                dpg.add_text("Right-drag", color=help_text_color)
                                dpg.add_text("Reset zoom", color=help_heading_color)
                                dpg.add_text("Double-click", color=help_text_color)
                            with dpg.table_row():
                                dpg.add_text("Select", color=help_heading_color)
                                dpg.add_text("LMB (hold)", color=help_text_color)
                                dpg.add_text("Select more", color=help_heading_color)
                                dpg.add_text("Shift+LMB", color=help_text_color)
                                dpg.add_text("Select less", color=help_heading_color)
                                dpg.add_text("Ctrl+LMB", color=help_text_color)

            # Toolbar
            with dpg.group(horizontal=False, tag="toolbar_group"):
                def add_separator(*, height=None, line=True, line_offset=None):
                    if height is None:
                        height = gui_config.toolbar_separator_h
                    guiutils.add_toolbar_separator(horizontal=False,
                                                   toolbar_extent=gui_config.toolbar_inner_w,
                                                   size=height, line=line,
                                                   line_offset=line_offset)
                if gui_config.toolbutton_indent is None:
                    gui_config.toolbutton_indent = (gui_config.toolbar_inner_w - gui_config.toolbutton_w) // 2  # pixels, to center the buttons

                dpg.add_text("Tools", tag="toolbar_header_text")
                add_separator(height=gui_config.toolbar_separator_h // 2, line_offset=0)

                # File controls

                dpg.add_button(label=fa.ICON_FOLDER,
                               tag="open_file_button",
                               callback=show_open_file_dialog,
                               indent=gui_config.toolbutton_indent,
                               width=gui_config.toolbutton_w)
                dpg.bind_item_font("open_file_button", app_state.themes_and_fonts.icon_font_solid)  # tag
                with dpg.tooltip("open_file_button", tag="open_file_tooltip"):  # tag
                    dpg.add_text("Open dataset [Ctrl+O]", tag="open_file_tooltip_text")

                dpg.add_button(label=fa.ICON_DOWNLOAD,
                               tag="open_importer_window_button",
                               callback=toggle_importer_window,
                               indent=gui_config.toolbutton_indent,
                               width=gui_config.toolbutton_w)
                dpg.bind_item_font("open_importer_window_button", app_state.themes_and_fonts.icon_font_solid)  # tag
                with dpg.tooltip("open_importer_window_button", tag="open_importer_window_tooltip"):  # tag
                    dpg.add_text("Import BibTeX files [Ctrl+I]", tag="open_importer_window_tooltip_text")

                add_separator()

                # Zoom controls

                dpg.add_button(label=fa.ICON_HOUSE,
                               tag="zoom_reset_button",
                               callback=plotter.reset_zoom,
                               indent=gui_config.toolbutton_indent,
                               width=gui_config.toolbutton_w)
                dpg.bind_item_font("zoom_reset_button", app_state.themes_and_fonts.icon_font_solid)  # tag
                with dpg.tooltip("zoom_reset_button", tag="zoom_reset_tooltip"):  # tag
                    dpg.add_text("Reset zoom [Ctrl+Home]", tag="zoom_reset_tooltip_text")

                # # TODO: Does not work currently (tested: DPG 1.11, 2.0.0), sets constraints too so zoom/pan no longer works. Wait for new DPG version?
                # def zoom_to_selection():
                #     selection_data_idxs = unbox(selection_data_idxs_box)  # item indices into `sorted_xxx`
                #     if not len(selection_data_idxs):
                #         return
                #     xmin = np.min(dataset.sorted_lowdim_data[selection_data_idxs][:, 0])
                #     xmax = np.max(dataset.sorted_lowdim_data[selection_data_idxs][:, 0])
                #     ymin = np.min(dataset.sorted_lowdim_data[selection_data_idxs][:, 1])
                #     ymax = np.max(dataset.sorted_lowdim_data[selection_data_idxs][:, 1])
                #     dpg.set_axis_limits("axis0", xmin, xmax)
                #     dpg.set_axis_limits("axis1", ymin, ymax)
                # dpg.add_button(label="Zoom selection", tag="zoom_to_selection_button",
                #                callback=zoom_to_selection)
                # with dpg.tooltip("zoom_to_selection_button"):  # tag
                #     dpg.add_text("Zoom to currently selected items")

                add_separator()

                # Selection controls

                dpg.add_button(label=fa.ICON_ARROW_ROTATE_LEFT,
                               tag="selection_undo_button",
                               callback=selection.undo,
                               indent=gui_config.toolbutton_indent,
                               width=gui_config.toolbutton_w,
                               enabled=False)
                dpg.bind_item_font("selection_undo_button", app_state.themes_and_fonts.icon_font_solid)  # tag
                dpg.bind_item_theme("selection_undo_button", "disablable_widget_theme")  # tag
                with dpg.tooltip("selection_undo_button", tag="selection_undo_tooltip"):  # tag
                    dpg.add_text("Undo selection change [Ctrl+Shift+Z]",
                                 tag="selection_undo_tooltip_text")

                dpg.add_button(label=fa.ICON_ARROW_ROTATE_RIGHT,
                               tag="selection_redo_button",
                               callback=selection.redo,
                               indent=gui_config.toolbutton_indent,
                               width=gui_config.toolbutton_w,
                               enabled=False)
                dpg.bind_item_font("selection_redo_button", app_state.themes_and_fonts.icon_font_solid)  # tag
                dpg.bind_item_theme("selection_redo_button", "disablable_widget_theme")  # tag
                with dpg.tooltip("selection_redo_button", tag="selection_redo_tooltip"):  # tag
                    dpg.add_text("Redo selection change [Ctrl+Shift+Y]",
                                 tag="selection_redo_tooltip_text")

                def select_search_results():
                    """Select all datapoints matching the current search."""
                    selection.update(unbox(search_result_data_idxs_box),
                                     selection.keyboard_state_to_mode())
                dpg.add_button(label=fa.ICON_MAGNIFYING_GLASS,
                               tag="select_search_results_button",
                               callback=select_search_results,
                               indent=gui_config.toolbutton_indent,
                               width=gui_config.toolbutton_w)
                dpg.bind_item_font("select_search_results_button", app_state.themes_and_fonts.icon_font_solid)  # tag
                with dpg.tooltip("select_search_results_button", tag="select_search_results_tooltip"):  # tag
                    dpg.add_text("Select items matched by current search [Enter, while the search field has focus]\n    with Shift: add\n    with Ctrl: subtract\n    with Ctrl+Shift: intersect",
                                 tag="select_search_results_tooltip_text")

                def select_visible_all():
                    """Select those datapoints that are currently visible in the plotter view."""
                    selection.update(plotter.get_visible_datapoints(),
                                     selection.keyboard_state_to_mode())
                dpg.add_button(label=fa.ICON_SQUARE,
                               tag="select_visible_all_button",
                               callback=select_visible_all,
                               indent=gui_config.toolbutton_indent,
                               width=gui_config.toolbutton_w)
                dpg.bind_item_font("select_visible_all_button", app_state.themes_and_fonts.icon_font_regular)  # tag
                with dpg.tooltip("select_visible_all_button", tag="select_visible_all_tooltip"):  # tag
                    dpg.add_text("Select items currently on-screen in the plotter [F9]\n    with Shift: add\n    with Ctrl: subtract\n    with Ctrl+Shift: intersect",
                                 tag="select_visible_all_tooltip_text")

                dpg.add_button(label=fa.ICON_CLOUD,
                               tag="word_cloud_button",
                               callback=word_cloud.toggle_window,
                               indent=gui_config.toolbutton_indent,
                               width=gui_config.toolbutton_w)
                dpg.bind_item_font("word_cloud_button", app_state.themes_and_fonts.icon_font_solid)  # tag
                with dpg.tooltip("word_cloud_button", tag="word_cloud_tooltip"):  # tag
                    dpg.add_text("Toggle word cloud window [F10]",
                                 tag="word_cloud_button_tooltip_text")

                # Miscellaneous controls

                add_separator()
                def toggle_fullscreen():
                    dpg.toggle_viewport_fullscreen()
                    resize_gui()  # see below
                dpg.add_button(label=fa.ICON_EXPAND,
                               tag="fullscreen_button",
                               callback=toggle_fullscreen,
                               indent=gui_config.toolbutton_indent,
                               width=gui_config.toolbutton_w)
                dpg.bind_item_font("fullscreen_button", app_state.themes_and_fonts.icon_font_solid)  # tag
                with dpg.tooltip("fullscreen_button", tag="fullscreen_tooltip"):  # tag
                    dpg.add_text("Toggle fullscreen [F11]",
                                 tag="fullscreen_tooltip_text")

                add_separator()

                # We'll define and bind the callback later, when we set up the help window.
                dpg.add_button(label=fa.ICON_CIRCLE_QUESTION,
                               tag="help_button",
                               indent=gui_config.toolbutton_indent,
                               width=gui_config.toolbutton_w)
                dpg.bind_item_font("help_button", app_state.themes_and_fonts.icon_font_regular)  # tag
                with dpg.tooltip("help_button", tag="help_tooltip"):  # tag
                    dpg.add_text("Open the Help card [F1]",
                                 tag="help_tooltip_text")

            # Search and plotter
            with dpg.child_window(tag="search_and_plotter_panel",
                                  autosize_x=True,
                                  autosize_y=True):
                # Search bar
                #
                # The plotter can't take height=-1 if it's the first item, so for now, put the search at the top.
                with dpg.group(tag="search_group",
                               horizontal=True):
                    search_header_text = dpg.add_text("Search", color=(140, 140, 140), tag="search_header_text")  # TODO: DRY duplicate definitions for labels

                    def clear_search():
                        dpg.set_value("search_field", "")  # tag
                        update_search()  # we should wait, because this button may get hammered.
                        dpg.focus_item("search_field")  # tag
                    dpg.add_button(label=fa.ICON_X, callback=clear_search, tag="clear_search_button")
                    dpg.bind_item_font("clear_search_button", app_state.themes_and_fonts.icon_font_solid)  # tag
                    with dpg.tooltip("clear_search_button", tag="clear_search_tooltip"):  # tag
                        dpg.add_text("Clear the search",
                                     tag="clear_search_tooltip_text")
                    with dpg.theme(tag="clear_search_theme"):  # tag
                        with dpg.theme_component(dpg.mvAll):
                            dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 96, 96))  # make the icon on the "clear search" button red
                    dpg.bind_item_theme("clear_search_button", "clear_search_theme")  # tag

                    dpg.add_input_text(tag="search_field",
                                       default_value="",
                                       hint="[Ctrl+F] [incremental fragment search in document titles; 'cat photo' matches 'photocatalytic'; lowercase = case-insensitive]",
                                       width=-1,
                                       callback=update_search)

                    with dpg.theme(tag="search_field_theme"):
                        with dpg.theme_component(dpg.mvInputText):
                            search_field_text_color = dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255))
                            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, (140, 140, 140))
                    dpg.bind_item_theme("search_field", "search_field_theme")  # tag

                # Plotter.
                # Configure explicitly so we are future-proof against possible defaults changes in DPG.
                with dpg.plot(tag="plot",
                              label="Semantic map [no dataset loaded]",  # TODO: DRY duplicate definitions for labels
                              width=-1,
                              height=-1,
                              no_menus=True,
                              delay_search=True,  # possible optimization, we don't need to access the actual scatter series often.
                              fit_button=dpg.mvMouseButton_Left,
                              pan_button=dpg.mvMouseButton_Middle,
                              box_select_button=dpg.mvMouseButton_Right,
                              box_select_cancel_button=dpg.mvMouseButton_Left) as theplot:  # The plot itself, with title. -1 = use whole parent container area (for widgets that support that feature).
                    # A DPG plot must have exactly one x axis, and one to three y axes. An y axis owns the data widget ("series").
                    dpg.add_plot_axis(dpg.mvXAxis, label="x", tag="axis0")
                    dpg.add_plot_axis(dpg.mvYAxis, label="y", tag="axis1")

                    # Create themes for highlighting datapoints.
                    search_results_highlight_color = None
                    selection_highlight_color = None
                    @call
                    def _():
                        # Data items hovered over by the mouse cursor.
                        color = (255, 255, 255, 40)  # actual color (this one does not need to be dynamic)
                        with dpg.theme(tag="my_selection_theme"):
                            with dpg.theme_component(dpg.mvScatterSeries):
                                dpg.add_theme_color(dpg.mvPlotCol_Line, color, category=dpg.mvThemeCat_Plots)
                                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 6, category=dpg.mvThemeCat_Plots)

                        # Data items currently selected. Data points to be filled in by selection handler.
                        color = (180, 255, 255, 40)  # dummy color; animated by `update_animations`
                        with dpg.theme(tag="my_selection_datapoints_theme"):
                            with dpg.theme_component(dpg.mvScatterSeries):
                                global selection_highlight_color
                                selection_highlight_color = dpg.add_theme_color(dpg.mvPlotCol_Line, color, category=dpg.mvThemeCat_Plots)
                                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 6, category=dpg.mvThemeCat_Plots)

                        # Data items matching the current search. Data points to be filled in by search handler.
                        color = (255, 255, 255, 32)  # dummy color; animated by `update_animations`
                        with dpg.theme(tag="my_search_results_theme"):
                            with dpg.theme_component(dpg.mvScatterSeries):
                                global search_results_highlight_color
                                search_results_highlight_color = dpg.add_theme_color(dpg.mvPlotCol_Line, color, category=dpg.mvThemeCat_Plots)
                                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 6, category=dpg.mvThemeCat_Plots)

                        plotter.create_highlight_series()  # some utilities may access the highlight series before the app has completely booted up
                dpg.bind_item_theme("plot", "my_plotter_theme")

    # Word cloud display.
    with dpg.window(show=False, modal=False, no_title_bar=False, tag="word_cloud_window",
                    label="Word cloud",
                    no_scrollbar=True, autosize=True):
        dpg.add_image("word_cloud_texture", tag="word_cloud_image")
        with dpg.group(horizontal=True, tag="word_cloud_toolbar"):
            dpg.add_button(label=fa.ICON_HARD_DRIVE,
                           tag="word_cloud_save_button",
                           callback=word_cloud.show_save_dialog,
                           indent=gui_config.toolbutton_indent,
                           width=gui_config.toolbutton_w)
            dpg.bind_item_font("word_cloud_save_button", app_state.themes_and_fonts.icon_font_solid)  # tag
            with dpg.tooltip("word_cloud_save_button", tag="word_cloud_save_tooltip"):  # tag
                dpg.add_text("Save word cloud as PNG [Ctrl+S]", tag="word_cloud_save_tooltip_text")

    # BibTeX importer integration. This allows invoking the BibTeX importer from the Raven-visualizer GUI.
    with dpg.window(show=False, modal=False, no_title_bar=False, tag="importer_window",
                    label="BibTeX import",
                    no_scrollbar=True, autosize=True) as importer_window:
        with dpg.group(horizontal=False):
            def importer_separator():
                """Add a horizontal line with a good-looking amount of vertical space around it. Used in the BibTeX importer window."""
                dpg.add_spacer(width=gui_config.importer_w, height=2)  # leave some vertical space
                with dpg.drawlist(width=gui_config.importer_w, height=1):
                    dpg.draw_line((0, 0), (gui_config.importer_w - 1, 0), color=(140, 140, 140, 255), thickness=1)
                dpg.add_spacer(width=gui_config.importer_w, height=1)  # leave some vertical space

            # dpg.add_text("[To start, select files, and then click the play button.]", color=(140, 140, 140, 255))
            dpg.add_spacer(width=gui_config.importer_w)  # ensure window width

            def update_save_import_gui_table():
                """In the BibTeX importer window, update the output filename in the GUI.

                Called by `_save_import_callback` when the save import file dialog closes.
                """
                for child in dpg.get_item_children("save_import_table", slot=1):  # This won't affect table columns, because they live in a different slot.
                    dpg.delete_item(child)

                importer_output_file = unbox(importer_output_file_box)
                with dpg.table_row(parent="save_import_table"):
                    if importer_output_file:
                        dpg.add_text(os.path.basename(importer_output_file), color=(140, 140, 140, 255))
                    else:
                        dpg.add_text("[not selected]", color=(140, 140, 140, 255))

            def update_open_import_gui_table():
                """In the BibTeX importer window, update the input filenames in the GUI.

                Called by `_open_import_callback` when the open import file dialog closes.
                """
                for child in dpg.get_item_children("open_import_table", slot=1):  # This won't affect table columns, because they live in a different slot.
                    dpg.delete_item(child)

                importer_input_files = unbox(importer_input_files_box)
                if importer_input_files:
                    for importer_input_file in importer_input_files:
                        with dpg.table_row(parent="open_import_table"):
                            dpg.add_text(os.path.basename(importer_input_file), color=(140, 140, 140, 255))
                else:
                    with dpg.table_row(parent="open_import_table"):
                        dpg.add_text("[not selected]", color=(140, 140, 140, 255))

            with dpg.group(horizontal=True):
                dpg.add_button(label=fa.ICON_HARD_DRIVE,
                               tag="importer_save_button",
                               width=gui_config.toolbutton_w,
                               callback=show_save_import_dialog)
                dpg.bind_item_font("importer_save_button", app_state.themes_and_fonts.icon_font_solid)  # tag
                with dpg.tooltip("importer_save_button", tag="importer_save_tooltip"):  # tag
                    dpg.add_text("Select output dataset file to save as [Ctrl+S]", tag="importer_save_tooltip_text")  # TODO: DRY duplicate definitions for labels

                # We use a separate button widget instead of a header row.
                #
                # The header row would look clickable, but it isn't. It only supports a sort callback when `sortable=True`,
                # and abusing that as a button click callback is nontrivial. It gets called also when the table is rendered
                # (i.e. when the import window is opened), which also leads to an incorrect window size for the file-open dialog.
                with dpg.group():
                    dpg.add_button(label="Output dataset file",
                                   tag="importer_save_heading_text_button",
                                   width=gui_config.importer_w - gui_config.toolbutton_w - 11,
                                   callback=show_save_import_dialog)
                    with dpg.tooltip("importer_save_heading_text_button", tag="importer_save_heading_text_tooltip"):  # tag
                        dpg.add_text("Select output dataset file to save as [Ctrl+S]", tag="importer_save_heading_text_tooltip_text")  # TODO: DRY duplicate definitions for labels
                    with dpg.table(header_row=False,
                                   sortable=False,
                                   width=gui_config.importer_w - gui_config.toolbutton_w - 11,
                                   tag="save_import_table"):
                        dpg.add_table_column(label="Output dataset file")
                    update_save_import_gui_table()

            with dpg.group(horizontal=True):
                dpg.add_button(label=fa.ICON_FOLDER,
                               tag="importer_select_input_files_button",
                               width=gui_config.toolbutton_w,
                               callback=show_open_import_dialog)
                dpg.bind_item_font("importer_select_input_files_button", app_state.themes_and_fonts.icon_font_solid)  # tag
                with dpg.tooltip("importer_select_input_files_button", tag="importer_select_input_files_tooltip"):  # tag
                    dpg.add_text("Select input BibTeX files [Ctrl+O]", tag="importer_select_input_files_tooltip_text")  # TODO: DRY duplicate definitions for labels

                with dpg.group():
                    dpg.add_button(label="Input BibTeX files",
                                   tag="importer_select_input_files_heading_text_button",
                                   width=gui_config.importer_w - gui_config.toolbutton_w - 11,
                                   callback=show_open_import_dialog)
                    with dpg.tooltip("importer_select_input_files_heading_text_button", tag="importer_select_input_files_heading_text_tooltip"):  # tag
                        dpg.add_text("Select input BibTeX files [Ctrl+O]", tag="importer_select_input_files_heading_text_tooltip_text")  # TODO: DRY duplicate definitions for labels
                    with dpg.table(header_row=False,
                                   sortable=False,
                                   width=gui_config.importer_w - gui_config.toolbutton_w - 11,
                                   tag="open_import_table"):
                        dpg.add_table_column(label="Input BibTeX files")
                    update_open_import_gui_table()

            dpg.add_spacer(width=gui_config.importer_w, height=2)  # leave some vertical space

            with dpg.group(horizontal=True):
                dpg.add_button(label=fa.ICON_PLAY,
                               tag="importer_startstop_button",
                               width=gui_config.toolbutton_w,
                               callback=start_or_stop_importer,
                               enabled=True)
                dpg.bind_item_font("importer_startstop_button", app_state.themes_and_fonts.icon_font_solid)  # tag
                dpg.bind_item_theme("importer_startstop_button", "disablable_widget_theme")  # tag
                with dpg.tooltip("importer_startstop_button", tag="importer_startstop_tooltip"):  # tag
                    dpg.add_text("Start BibTeX import [Ctrl+Enter]", tag="importer_startstop_tooltip_text")  # TODO: DRY duplicate definitions for labels

                dpg.add_button(label="Start",
                               tag="importer_startstop_heading_text_button",
                               width=gui_config.importer_w - gui_config.toolbutton_w - 11,
                               callback=start_or_stop_importer)
                dpg.bind_item_theme("importer_startstop_heading_text_button", "disablable_widget_theme")  # tag
                with dpg.tooltip("importer_startstop_heading_text_button", tag="importer_startstop_heading_text_tooltip"):
                    dpg.add_text("Start BibTeX import [Ctrl+Enter]", tag="importer_startstop_heading_text_tooltip_text")

            importer_separator()

            dpg.add_progress_bar(default_value=0, width=-1, show=False, tag="importer_progress_bar")
            dpg.add_text("[To start, select files, and then click the play button.]", wrap=gui_config.importer_w, color=(140, 140, 140, 255), tag="importer_status_text")

logger.info(f"    Done in {tim.dt:0.6g}s.")

# --------------------------------------------------------------------------------
# Annotation tooltip subsystem wire-up
annotation.build_window()
app_state.update_mouse_hover = annotation.update  # Published here (not inside the module) so cross-module callers can reach it via `app_state`.

# --------------------------------------------------------------------------------
# Item information panel subsystem wire-up
info_panel.build_window()
app_state.update_info_panel = info_panel.update  # Published here so cross-module callers (selection, update_search) can reach it via `app_state`.


# --------------------------------------------------------------------------------
# Built-in help window

hotkey_info = (env(key_indent=0, key="Ctrl+O", action_indent=0, action="Open a dataset", notes=""),
               env(key_indent=0, key="Ctrl+I", action_indent=0, action="Import BibTeX files", notes="Use this to create a dataset"),
               env(key_indent=0, key="Ctrl+F", action_indent=0, action="Focus search field", notes=""),
               env(key_indent=1, key="Enter", action_indent=0, action="Select search matches, and unfocus", notes="When search field focused"),
               env(key_indent=2, key="Shift+Enter", action_indent=1, action="Same, but add to selection", notes="When search field focused"),
               env(key_indent=2, key="Ctrl+Enter", action_indent=1, action="Same, but subtract from selection", notes="When search field focused"),
               env(key_indent=2, key="Ctrl+Shift+Enter", action_indent=1, action="Same, but intersect with selection", notes="When search field focused"),
               env(key_indent=1, key="Esc", action_indent=0, action="Cancel search term edit, and unfocus", notes="When search field focused"),
               env(key_indent=0, key="F3", action_indent=0, action="Scroll to next search match", notes="When matches shown in info panel"),
               env(key_indent=0, key="Shift+F3", action_indent=0, action="Scroll to previous search match", notes="When matches shown in info panel"),
               helpcard.hotkey_blank_entry,
               env(key_indent=0, key="Ctrl+U", action_indent=0, action="Scroll to start of current cluster", notes='"up"'),
               env(key_indent=1, key="Ctrl+N", action_indent=0, action="Scroll to next cluster", notes=""),
               env(key_indent=1, key="Ctrl+P", action_indent=0, action="Scroll to previous cluster", notes=""),
               env(key_indent=0, key="Home", action_indent=0, action="Scroll to top", notes="When search field NOT focused"),
               env(key_indent=1, key="End", action_indent=0, action="Scroll to bottom", notes="When search field NOT focused"),
               env(key_indent=1, key="Page Up", action_indent=0, action="Scroll up", notes="When search field NOT focused"),
               env(key_indent=1, key="Page Down", action_indent=0, action="Scroll down", notes="When search field NOT focused"),
               env(key_indent=1, key="Up arrow", action_indent=0, action="Scroll up slightly", notes="When search field NOT focused"),
               env(key_indent=1, key="Down arrow", action_indent=0, action="Scroll down slightly", notes="When search field NOT focused"),

               helpcard.hotkey_new_column,
               env(key_indent=0, key="F6", action_indent=0, action="Search/unsearch current item", notes="Searching highlights it in the plotter"),
               env(key_indent=1, key="Shift+F6", action_indent=0, action="Set selection to current item only", notes=""),
               env(key_indent=1, key="Ctrl+F6", action_indent=0, action="Remove current item from selection", notes=""),
               env(key_indent=0, key="F7", action_indent=0, action="Select current cluster", notes=""),
               env(key_indent=1, key="Shift+F7", action_indent=1, action="Same, but add to selection", notes=""),
               env(key_indent=1, key="Ctrl+F7", action_indent=1, action="Same, but subtract from selection", notes=""),
               env(key_indent=1, key="Ctrl+Shift+F7", action_indent=1, action="Same, but intersect with selection", notes=""),
               env(key_indent=0, key="F8", action_indent=0, action="Copy report to clipboard", notes="As plain text, .txt"),
               env(key_indent=1, key="Shift+F8", action_indent=0, action="Copy report to clipboard", notes="As Markdown, .md"),
               env(key_indent=0, key="F9", action_indent=0, action="Select all data currently visible in plotter", notes=""),
               env(key_indent=1, key="Shift+F9", action_indent=1, action="Same, but add to selection", notes=""),
               env(key_indent=1, key="Ctrl+F9", action_indent=1, action="Same, but subtract from selection", notes=""),
               env(key_indent=1, key="Ctrl+Shift+F9", action_indent=1, action="Same, but intersect with selection", notes=""),
               env(key_indent=0, key="F10", action_indent=0, action="Toggle word cloud window", notes="From keywords of selected items"),
               env(key_indent=0, key="Ctrl+Shift+C", action_indent=0, action="Copy current item to clipboard", notes="As plain text, for web search"),
               env(key_indent=0, key="Ctrl+Shift+Z", action_indent=0, action="Undo last selection change", notes=""),
               env(key_indent=0, key="Ctrl+Shift+Y", action_indent=0, action="Redo last selection change", notes=""),
               env(key_indent=0, key="Ctrl+Home", action_indent=0, action="Reset plotter zoom", notes=""),
               env(key_indent=0, key="F11", action_indent=0, action="Toggle fullscreen mode", notes=""),
               env(key_indent=0, key="F1", action_indent=0, action="Open this Help card", notes=""),
               )
def render_help_extras(self: helpcard.HelpWindow,
                       gui_parent: Union[str, int]) -> None:
    """Render app-specific extra information into the help card.

    Called by `HelpWindow` when the help card is first rendered.
    """
    c_search = f'<font color="{gui_config.plotter_search_results_highlight_color}">'
    c_selection = f'<font color="{gui_config.plotter_selection_highlight_color}">'

    # Legend for table
    dpg_markdown.add_text(f"{self.c_hed}**Terminology**{self.c_end}", parent=gui_parent)
    g = dpg.add_group(horizontal=True, parent=gui_parent)
    g1 = dpg.add_group(horizontal=False, parent=g)
    dpg_markdown.add_text(f"- {self.c_txt}**Current item**: The topmost item **fully** visible in the info panel. The controls of the current item glow slightly.{self.c_end}",
                          parent=g1)
    dpg_markdown.add_text(f"- {self.c_txt}**Current cluster**: The cluster the current item belongs to. Clusters are auto-detected by a linguistic analysis.{self.c_end}",
                          parent=g1)
    g2 = dpg.add_group(horizontal=False, parent=g)
    dpg_markdown.add_text(f"- {self.c_txt}**Selection set**: The selected items, {self.c_end}{c_selection}**glowing**{self.c_end}{self.c_txt} in the plotter. As many are loaded into the info panel as reasonably fit.{self.c_end}",
                          parent=g2)
    dpg_markdown.add_text(f"- {self.c_txt}**Search result set**: The items matching the current search, {self.c_end}{c_search}**glowing**{self.c_end}{self.c_txt} in the plotter.{self.c_end}",
                          parent=g2)
    dpg.add_spacer(width=1, height=app_state.themes_and_fonts.font_size, parent=g)

    # Additional general help
    dpg_markdown.add_text(f"{self.c_hed}**How search works**{self.c_end}",
                          parent=gui_parent)
    dpg_markdown.add_text(f"{self.c_txt}Each space-separated search term is a **fragment**. For a data point to match, **all** fragments must match. Ordering of fragments does **not** matter. The {self.c_end}{c_search}search result{self.c_end}{self.c_txt} and {self.c_end}{c_selection}selection{self.c_end}{self.c_txt} sets are **independent**. {self.c_end}{c_search}Search results{self.c_end}{self.c_txt} live-update as you type.{self.c_end}",
                          parent=gui_parent)
    dpg_markdown.add_text(f'- {self.c_txt}A **lowercase** fragment matches **that fragment {self.c_end}{self.c_hig}case-insensitively{self.c_end}{self.c_txt}**. E.g. *"hydrogen"* matches also *"Hydrogen"*.{self.c_end}',
                          parent=gui_parent)
    dpg_markdown.add_text(f'- {self.c_txt}A fragment with **at least one uppercase** letter matches **that fragment {self.c_end}{self.c_hig}case-sensitively{self.c_end}{self.c_txt}**. E.g. *"TiO"* matches only titanium oxide, not *"bastion"*.{self.c_end}',
                          parent=gui_parent)
    dpg_markdown.add_text(f'- {self.c_txt}You can use regular numbers in place of subscript/superscript numbers. E.g. *"h2so4"* matches also *"H₂SO₄"*, and *"x2"* matches also *"x²"*. {self.c_end}',
                          parent=gui_parent)
    dpg_markdown.add_text(f"{self.c_txt}When the search field is focused, the usual text editing keys are available (*Enter, Esc, Home, End, Shift-select, Ctrl+Left, Ctrl+Right, Ctrl+A, Ctrl+Z, Ctrl+Y*).{self.c_end}",
                          parent=gui_parent)
help_window = helpcard.HelpWindow(hotkey_info=hotkey_info,
                                  width=gui_config.help_window_w,
                                  height=gui_config.help_window_h,
                                  reference_window=main_window,
                                  themes_and_fonts=app_state.themes_and_fonts,
                                  on_render_extras=render_help_extras,
                                  on_show=app_state.enter_modal_mode,
                                  on_hide=app_state.exit_modal_mode)
dpg.set_item_callback("help_button", help_window.show)  # tag

# --------------------------------------------------------------------------------
# GUI resizing handler

def resize_gui():
    """Wait for the viewport size to actually change, then resize dynamically sized GUI elements.

    This is handy for toggling fullscreen, because the size changes at the next frame at the earliest.
    For the viewport resize callback, that one fires (*almost* always?) after the size has already changed.
    """
    logger.debug("resize_gui: Entered. Waiting for viewport size change.")
    if guiutils.wait_for_resize(main_window):
        _resize_gui()
    logger.debug("resize_gui: Done.")

def _resize_gui():
    """Resize dynamically sized GUI elements, RIGHT NOW."""
    logger.debug("_resize_gui: Entered.")
    logger.debug("_resize_gui: Updating info panel height.")
    info_panel.update_height()
    logger.debug("_resize_gui: Updating info panel current item on-screen coordinates.")
    info_panel.update_current_item_info()
    logger.debug("_resize_gui: Recentering help window.")
    help_window.reposition()
    logger.debug("_resize_gui: Updating annotation tooltip.")
    app_state.update_mouse_hover(force=True, wait=False)
    logger.debug("_resize_gui: Rebuilding dimmer overlay.")
    info_panel.rebuild_dimmer_overlay()
    logger.debug("_resize_gui: Done.")


# Old versions of DPG have a bug where they don't always call the viewport resize callback, but it seems to work in 1.11 and later.
# https://github.com/hoffstadt/DearPyGui/issues/1896
dpg.set_viewport_resize_callback(_resize_gui)

# --------------------------------------------------------------------------------
# Mouse events

select_radius_update_lock = threading.RLock()
select_radius_draw_item = None
select_radius_last_pos = None
select_radius_last_scale_x = None
select_radius_last_scale_y = None

unit_manygon = np.array([(np.cos(t), np.sin(t)) for t in np.linspace(0, 2 * np.pi, 65)])  # discrete approximation of the unit apeirogon :P

def clear_select_radius_indicator():
    global select_radius_draw_item
    global select_radius_last_pos
    global select_radius_last_scale_x
    global select_radius_last_scale_y
    with select_radius_update_lock:
        if select_radius_draw_item is not None:
            dpg.delete_item(select_radius_draw_item)
        select_radius_draw_item = None
        select_radius_last_pos = None
        select_radius_last_scale_x = None
        select_radius_last_scale_y = None

def draw_select_radius_indicator():
    global select_radius_draw_item
    global select_radius_last_pos
    global select_radius_last_scale_x
    global select_radius_last_scale_y

    # Avoid unnecessary clear/redraw to prevent flickering
    p = dpg.get_plot_mouse_pos()
    pixels_per_data_unit_x, pixels_per_data_unit_y = guiutils.get_pixels_per_plotter_data_unit("plot", "axis0", "axis1")  # tag
    if pixels_per_data_unit_x == 0.0 or pixels_per_data_unit_y == 0.0:  # no dataset open?
        clear_select_radius_indicator()
        return
    same_pos = (select_radius_last_pos is not None and select_radius_last_pos == p)
    same_scale_x = (select_radius_last_scale_x is not None and select_radius_last_scale_x == pixels_per_data_unit_x)
    same_scale_y = (select_radius_last_scale_y is not None and select_radius_last_scale_y == pixels_per_data_unit_y)
    same_zoom = (same_scale_x and same_scale_y)

    with select_radius_update_lock:
        select_radius_last_pos = p
        select_radius_last_scale_x = pixels_per_data_unit_x
        select_radius_last_scale_y = pixels_per_data_unit_y

        if not (same_pos and same_zoom):
            clear_select_radius_indicator()  # remove old indicator if any

        # NOTE: To avoid race conditions, we can touch `select_radius_draw_item` only inside the critical section.
        if (select_radius_draw_item is not None) and (same_pos and same_zoom):
            return
        brush_radius_data_x = gui_config.selection_brush_radius_pixels / pixels_per_data_unit_x  # TODO: what if division by zero?
        brush_radius_data_y = gui_config.selection_brush_radius_pixels / pixels_per_data_unit_y  # TODO: what if division by zero?
        deltas = np.copy(unit_manygon)  # unit circle
        # Convert a circle with a radius of the selection brush size, from pixel space to data space (where x and y axes may have different scalings).
        deltas[:, 0] *= brush_radius_data_x
        deltas[:, 1] *= brush_radius_data_y
        points = (np.array(p) + deltas).tolist()
        select_radius_draw_item = dpg.draw_polygon(points,  # in data space
                                                   color=(255, 255, 255, 255),
                                                   fill=(0, 0, 0, 0),
                                                   parent="plot")  # tag

def mouse_inside_plot_widget():
    """Return whether the mouse cursor is inside the plot widget."""
    return guiutils.is_mouse_inside_widget("plot")  # tag
app_state.mouse_inside_plot_widget = mouse_inside_plot_widget  # so submodules (e.g. `annotation`) can reach it

def mouse_inside_info_panel():
    """Return whether the mouse cursor is inside the info panel."""
    return guiutils.is_mouse_inside_widget("item_information_panel")  # tag

def mouse_wheel_callback(sender, app_data):
    """Update the plotter data tooltip when the user zooms with the mouse wheel.

    Also, if scrolling the info panel, flash the end when reached.
    """
    # If we reach the end of the info panel, flash it.
    if mouse_inside_info_panel():
        # direction = app_data  # -1 = down, +1 = up  # for documentation only
        current_y_scroll = dpg.get_y_scroll("item_information_panel")  # tag
        info_panel.flash_scroll_end_by_position(current_y_scroll)

    # Zooming in the plotter may change which data points are under the cursor within the tooltip-trigger pixel distance.
    if mouse_inside_plot_widget():
        app_state.update_mouse_hover(force=True, wait=True)

lmb_pressed_inside_plot = False  # for tracking whether a drag started inside the plot (to prevent losing selection while scrolling info panel using the scrollbar, with the mouse then entering the plot area while LMB is down)
def mouse_click_callback(sender, app_data):
    """Handle the case where the user selects items by clicking, without moving the mouse."""
    # print(dpg.get_item_type(sender), sender, app_data)  # dpg.get_item_alias(sender), but just printing `sender` shows the alias if it has one, and otherwise the raw numeric ID.

    global lmb_pressed_inside_plot

    if not mouse_inside_plot_widget():  # should not happen; we're an item handler for the plot widget
        lmb_pressed_inside_plot = False
        return

    # `sender` is always the handler registry; `app_data` contains the tag/ID of the actual GUI widget that triggered the event.
    mouse_button, real_sender_ = app_data  # for documentation of `app_data` only

    # Left-click to select
    if mouse_button == dpg.mvMouseButton_Left:
        lmb_pressed_inside_plot = True
        draw_select_radius_indicator()
        selection.update(plotter.get_data_idxs_at_mouse(),
                         selection.keyboard_state_to_mode(),
                         wait=False,
                         update_selection_undo_history=False)  # `mouse_release_callback` will commit regardless of if this event is actually a click or a starting mouse-draw

    # Right-click to scroll to item at mouse cursor (if it is shown in the info panel)
    elif mouse_button == dpg.mvMouseButton_Right:
        data_idxs_at_mouse = plotter.get_data_idxs_at_mouse()  # item indices into `sorted_xxx`
        if not len(data_idxs_at_mouse):
            return

        # Find items under the mouse cursor that is included in the info panel.
        #   - Consider only items listed in the mouse-hover annotation tooltip. These are stored in `annotation.data_idxs`.
        #   - If a search is active, the item should also match the current search.
        with annotation.content_lock:
            annotation_data_idxs_set = set(annotation.data_idxs)  # performance - better to amortize this here, or O(n) lookup for each `in` test?
            search_string = unbox(search_string_box)
            with info_panel.content_lock:  # we need to access `info_panel.entry_title_widgets`
                if not search_string:  # no search active
                    jumpable_data_idxs = {data_idx for data_idx in data_idxs_at_mouse
                                          if (data_idx in annotation_data_idxs_set) and (data_idx in info_panel.entry_title_widgets)}
                else:
                    search_result_data_idxs_set = set(unbox(search_result_data_idxs_box))
                    jumpable_data_idxs = {data_idx for data_idx in data_idxs_at_mouse
                                          if (data_idx in annotation_data_idxs_set) and (data_idx in search_result_data_idxs_set) and (data_idx in info_panel.entry_title_widgets)}
                if not jumpable_data_idxs:
                    return

                # Then find the item that is listed first in the annotation tooltip, to keep the behavior easily predictable for the user.
                # We can use `annotation.data_idxs`, which has them in that order.
                jump_target_data_idx = next(filter(lambda data_idx: data_idx in jumpable_data_idxs,
                                                   annotation.data_idxs),
                                            None)
                if jump_target_data_idx is None:
                    return

                info_panel.scroll_to_item(info_panel.entry_title_widgets[jump_target_data_idx])

def keydown_callback(sender, app_data):
    """Enable selection brush indicator when the mouse is in the plot area and Shift/Ctrl is held down.

    This gives immediate visual feedback that the "select more" or "select less" mode is active.
    """
    key, time_since_press_ = app_data  # for documentation only

    if not mouse_inside_plot_widget():
        return
    if key in (dpg.mvKey_LControl, dpg.mvKey_RControl, dpg.mvKey_LShift, dpg.mvKey_RShift):
        draw_select_radius_indicator()

def keyup_callback(sender, app_data):
    """Disable selection brush indicator when Shift/Ctrl is released (and the mouse button is not down)."""
    key = app_data  # for documentation only

    if key in (dpg.mvKey_LControl, dpg.mvKey_RControl, dpg.mvKey_LShift, dpg.mvKey_RShift):
        if not dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
            clear_select_radius_indicator()

def mouse_move_callback():
    """Update the relevant GUI elements when the mouse moves.

    Currently these are:
        - Plotter data tooltip.
        - Select radius indicator for mouse-draw select.
    """
    clear_select_radius_indicator()

    if not mouse_inside_plot_widget():
        annotation.clear_mouse_hover()
        return
    # We are inside the plot widget.

    # We do the following in likely-fastest-to-likely-slowest order, to refresh each relevant GUI element as quickly as possible.

    # mouse-draw select (but only when drag began inside the plot)
    if lmb_pressed_inside_plot and dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
        draw_select_radius_indicator()
        selection.update(plotter.get_data_idxs_at_mouse(),
                         selection.keyboard_state_to_mode(),
                         wait=True,
                         update_selection_undo_history=False)  # mouse release will commit later.

    # plotter data tooltip
    app_state.update_mouse_hover(force=False, wait=True)

def mouse_release_callback(sender, app_data):
    """Finalize a mouse-click select or mouse-draw select."""
    global lmb_pressed_inside_plot
    lmb_pressed_inside_plot = False  # finalize the drag

    if not mouse_inside_plot_widget():
        return

    mouse_button = app_data  # for documentation of `app_data` only

    # commit new selection to undo history when mouse-draw select ends
    if mouse_button == dpg.mvMouseButton_Left:
        clear_select_radius_indicator()
        selection.commit_change_to_undo_history()

def hotkeys_callback(sender, app_data):
    """Handle hotkeys."""
    key = app_data  # for documentation only
    ctrl_pressed = dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)
    shift_pressed = dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)

    # NOTE: If you update this, to make the hotkeys discoverable, update also:
    #  - The tooltips wherever the GUI elements are created or updated (search for e.g. "[F9]", may appear in multiple places)
    #  - The help window

    # Hotkeys that are always available, regardless of any dialogs (even if modal)
    if key == dpg.mvKey_F11:  # de facto standard hotkey for toggle fullscreen
        toggle_fullscreen()

    # Hotkeys while the Help card is shown - helpcard handles its own hotkeys
    elif help_window.is_visible():
        return

    # Hotkeys while an "open file" or "save as" dialog is shown - fdialog handles its own hotkeys
    elif (is_open_file_dialog_visible() or word_cloud.is_save_dialog_visible() or
          is_open_import_dialog_visible() or is_save_import_dialog_visible()):
        return

    # Hotkeys while the word cloud viewer is shown
    elif dpg.is_item_visible("word_cloud_window"):
        if ctrl_pressed and key == dpg.mvKey_S:
            word_cloud.show_save_dialog()
            return

    # Hotkeys while the BibTeX importer window is shown
    elif dpg.is_item_visible("importer_window"):  # tag
        if ctrl_pressed:
            if key == dpg.mvKey_O:
                show_open_import_dialog()
                return
            elif key == dpg.mvKey_S:
                show_save_import_dialog()
                return
            elif key == dpg.mvKey_Return:
                start_or_stop_importer()
                return

    # Hotkeys for main window, while no modal window is shown
    if dpg.is_item_focused("search_field") and key == dpg.mvKey_Return:  # tag  # regardless of modifier state, to allow Shift+Enter and Ctrl+Enter.
        select_search_results()
        dpg.focus_item("item_information_panel")  # tag
    elif dpg.is_item_focused("search_field") and key == dpg.mvKey_Escape:  # tag  # cancel current search edit (handled by the text input internally, by sending a change event; but we need to handle the keyboard focus)
        dpg.focus_item("item_information_panel")  # tag
    elif key == dpg.mvKey_F1:  # de facto standard hotkey for help
        help_window.show()
    elif key == dpg.mvKey_F3:  # some old MS-DOS software in the 1990s used F3 for next/prev search match, I think?
        if (dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)):
            if dpg.is_item_enabled("prev_search_match_button"):  # tag
                info_panel.scroll_to_prev_search_match()
        else:
            if dpg.is_item_enabled("next_search_match_button"):  # tag
                info_panel.scroll_to_next_search_match()
    elif key == dpg.mvKey_F6:  # Use an F-key, because this too has Shift/Ctrl modes.
        info_panel.search_or_select_current_entry()
    elif key == dpg.mvKey_F7:  # Use an F-key, because this too needs selection mode modifiers.
        info_panel.select_current_cluster()
    elif key == dpg.mvKey_F8 and dpg.is_item_enabled("copy_report_to_clipboard_button"):  # tag  # NOTE: Shift is a modifier here too
        info_panel.copy_report_to_clipboard()
    elif key == dpg.mvKey_F9:  # Use an F-key, because this too needs selection mode modifiers.
        select_visible_all()
    elif key == dpg.mvKey_F10:
        word_cloud.toggle_window()
    # Ctrl+Shift+...
    elif ctrl_pressed and shift_pressed:
        if key == dpg.mvKey_Z and dpg.is_item_enabled("selection_undo_button"):  # tag
            selection.undo()
        elif key == dpg.mvKey_Y and dpg.is_item_enabled("selection_redo_button"):  # tag
            selection.redo()
        elif key == dpg.mvKey_C:
            info_panel.copy_current_entry_to_clipboard()
        # Some hidden debug features. Mnemonic: "Mr. T Lite" (Ctrl + Shift + M, R, T, L)
        elif key == dpg.mvKey_M:
            dpg.show_metrics()
        elif key == dpg.mvKey_R:
            dpg.show_item_registry()
        elif key == dpg.mvKey_T:
            dpg.show_font_manager()
        elif key == dpg.mvKey_L:
            dpg.show_style_editor()
    # Ctrl+...
    elif ctrl_pressed:
        if key == dpg.mvKey_F:
            dpg.focus_item("search_field")  # tag
        elif key == dpg.mvKey_O:
            show_open_file_dialog()
        elif key == dpg.mvKey_I:
            toggle_importer_window()
        elif key == dpg.mvKey_Home:
            plotter.reset_zoom()
        elif key == dpg.mvKey_N:
            info_panel.scroll_to_next_cluster()
        elif key == dpg.mvKey_P:
            info_panel.scroll_to_prev_cluster()
        elif key == dpg.mvKey_U:
            info_panel.scroll_to_top_of_current_cluster()
    # Bare key
    #
    # NOTE: These are global across the whole app (when no modal window is open) - be very careful here!
    elif not dpg.is_item_focused("search_field"):  # tag
        if key == dpg.mvKey_Home:
            info_panel.go_to_top()
        elif key == dpg.mvKey_End:
            info_panel.go_to_bottom()
        elif key == dpg.mvKey_Next or key == 518:  # page down  # TODO: fix: in DPG 2.0.0, Page Down is no longer "Next" but a mysterious 518 - what is the new name?
            info_panel.page_down()
        elif key == dpg.mvKey_Prior or key == 517:  # page up  # TODO: fix: in DPG 2.0.0, Page Up is no longer "Prior" but a mysterious 517 - what is the new name?
            info_panel.page_up()
        elif key == dpg.mvKey_Down:  # arrow down
            @call
            def _():
                current_y_scroll = dpg.get_y_scroll("item_information_panel")  # tag
                w_info, h_info = dpg.get_item_rect_size("item_information_panel")  # tag
                new_y_scroll = current_y_scroll + 0.1 * h_info
                info_panel.scroll_to_position(new_y_scroll)
        elif key == dpg.mvKey_Up:  # arrow up
            @call
            def _():
                current_y_scroll = dpg.get_y_scroll("item_information_panel")  # tag
                w_info, h_info = dpg.get_item_rect_size("item_information_panel")  # tag
                new_y_scroll = current_y_scroll - 0.1 * h_info
                info_panel.scroll_to_position(new_y_scroll)

# Set up global mouse and keyboard handlers
with dpg.handler_registry(tag="global_handler_registry"):  # global (whole viewport)
    dpg.add_mouse_move_handler(tag="mouse_move_handler", callback=mouse_move_callback)
    dpg.add_mouse_release_handler(tag="mouse_release_handler", callback=mouse_release_callback)
    dpg.add_mouse_wheel_handler(tag="mouse_wheel_handler", callback=mouse_wheel_callback)
    # dpg.add_mouse_click_handler(tag="mouse_click_handler", callback=mouse_click_callback)
    dpg.add_key_press_handler(tag="hotkeys_handler", callback=hotkeys_callback)
    dpg.add_key_down_handler(tag="keydown_handler", callback=keydown_callback)
    dpg.add_key_release_handler(tag="keyup_handler", callback=keyup_callback)

# Inside the plot widget only (but also incorrectly, outside the actual plot area, which doesn't seem to have its on-screen coordinates stored anywhere accessible).
# But see also `dpg.get_item_rect_min`, `dpg.get_mouse_pos(local=False)` https://github.com/hoffstadt/DearPyGui/issues/2311
with dpg.item_handler_registry(tag="plot_handler_registry") as registry:
    # dpg.add_item_hover_handler(callback=mouse_move_callback)
    dpg.add_item_clicked_handler(tag="plot_mouse_click_handler", callback=mouse_click_callback)  # button=dpg.mvMouseButton_Left
dpg.bind_item_handler_registry("plot", registry)  # tag

# --------------------------------------------------------------------------------
# Set up app exit cleanup

# NOTE: In DPG 2.0.0, this works correctly.
# NOTE: In DPG 1.x, if the info panel is updating while the app shuts down, DPG's exit callback doesn't actually trigger, and DPG segfaults.
#   - At least it's not `update_animations`, the same happens also even if we disable that.
#   - Maybe it's because `_update_info_panel` renders GUI stuff from a background thread? Trying to create GUI items while the app shuts down?
def gui_shutdown():
    logger.info("gui_shutdown: entered")
    reset_app_state(_update_gui=False)  # Exiting, GUI might no longer exist when this is called.
    logger.info("gui_shutdown: done")
dpg.set_exit_callback(gui_shutdown)

# --------------------------------------------------------------------------------
# Start the app

logger.info("App bootup...")

parser = argparse.ArgumentParser(description="""Visualize BibTeX data.""",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument('-v', '--version', action='version', version=('%(prog)s ' + __version__))
parser.add_argument(dest='filename', nargs='?', default=None, type=str, metavar='file',
                    help='dataset to open at startup (optional)')
opts = parser.parse_args()

# `raven.client.api` must be initialized before any mayberemote call. The BibTeX importer uses
# mayberemote for NLP during the import pipeline, so it needs this. No server connection is
# made here — that happens lazily on the first HTTP call.
api.initialize(raven_server_url=client_config.raven_server_url,
               raven_api_key_file=client_config.raven_api_key_file)

app_state.bg = concurrent.futures.ThreadPoolExecutor()  # for info panel and tooltip annotation updates
# Subsystem task managers (annotation, info panel, word cloud) are created lazily inside their own modules on first use.
importer.init(executor=app_state.bg)  # BibTeX importer

# import sys
# print(dir(sys.modules["__main__"]))  # DEBUG: Check this occasionally to make sure we don't accidentally store any temporary variables in the module-level namespace.

dpg.set_primary_window(main_window, True)  # Make this DPG "window" occupy the whole OS window (DPG "viewport").
dpg.set_viewport_vsync(True)
dpg.show_viewport()

# Load the file optionally provided on the command line
if opts.filename:
    _default_path = os.path.dirname(common_utils.absolutize_filename(opts.filename))
    open_file(opts.filename)
else:
    _default_path = os.getcwd()
    reset_app_state()  # effectively, open a blank dataset
initialize_filedialogs(_default_path)

# HACK: Create the dimmer as soon as possible (some time after the first frame so that other GUI elements initialize their sizes).
# The window for the "scroll ends here" animation is also created at frame 10, but via another mechanism (trying to create it each frame, but the implementation blocks it until frame 10).
dpg.set_frame_callback(10, info_panel.create_dimmer_overlay)

logger.info("App render loop starting.")

try:
    # We control the render loop manually to have a convenient place to update our GUI animations just before rendering each frame.
    while dpg.is_dearpygui_running():
        update_animations()
        dpg.render_dearpygui_frame()
    # dpg.start_dearpygui()  # automatic render loop
except KeyboardInterrupt:
    clear_background_tasks(wait=False)  # signal background tasks to exit

logger.info("App render loop exited.")

dpg.destroy_context()


def main():  # TODO: we don't really need this; it's just for console_scripts.
    pass

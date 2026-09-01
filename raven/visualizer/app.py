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

import argparse

from .. import __version__

parser = argparse.ArgumentParser(description="""Visualize BibTeX data.""",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('-v', '--version', action='version', version=('%(prog)s ' + __version__))
parser.add_argument(dest='filename', nargs='?', default=None, type=str, metavar='file',
                    help='dataset to open at startup (optional)')
parser.add_argument('--log', metavar='PATH', default=None,
                    help='mirror stderr log to this file (overwritten each run)')
parser.add_argument('--log-level', default='INFO',
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                    help='root logger level (default: INFO)')
parser.add_argument('--server-url', metavar='URL', default=None,
                    help='Raven server to talk to, overriding the configured one; e.g. http://localhost:5100. '
                         'Optional here — the importer loads NLP and embedding models locally when no server '
                         'answers — so pointing this at nothing is how to exercise that fallback.')
parser.add_argument('--qr', action='store_true',
                    help='show a "Get Raven" QR code in a corner of the window, for demoing at an exhibit')
opts = parser.parse_args()

import logging
from ..common import logsetup
# TODO: apply this list's *shape* to the other Raven apps. It is hand-curated — everything the Visualizer
# emits, plus the shared-layer modules whose output is worth reading here, and nothing third-party — and it
# is the standard the constellation is aiming at rather than a local quirk. Every other app currently
# passes no `allow` at all, so each one emits its dependencies' logging too; normalizing them means either
# curating a list per app or settling for a blanket `"raven"`, which is a decision nobody has made yet.
#
# Recorded because the asymmetry looks like a defect from either end: the next reader may well "fix" this
# one to match the others, which would be the wrong direction.
logsetup.configure(level=getattr(logging, opts.log_level),
                   logfile=opts.log,
                   allow=[__name__,  # this module, whatever it is called (`__main__` under `python -m`)...
                          __package__,  # ...and every other Visualizer module, including ones not written yet
                          "raven.client.mayberemote",
                          "raven.client.api",
                          "raven.client.util",
                          "raven.common.bgtask",
                          "raven.common.deviceinfo",
                          "raven.common.gui.animation",
                          "raven.common.gui.filedrop",
                          "raven.common.gui.fontsetup",
                          "raven.common.gui.utils",
                          "raven.common.gui.widgetfinder",
                          "raven.common.utils",
                          "raven.librarian.llmclient",
                          "raven.vendor.file_dialog.fdialog"])
logger = logging.getLogger(__name__)

logger.info(f"Raven-visualizer version {__version__} starting.")

logger.info("Loading libraries...")
from unpythonic import timer
with timer() as tim:
    import concurrent.futures
    import math
    import os
    import pathlib
    import platform
    from typing import Union

    import numpy as np

    from unpythonic.env import env
    envcls = env  # for functions that need an `env` parameter due to `@dlet`, so that they can also instantiate env objects (oops)
    from unpythonic import call, unbox

    import dearpygui.dearpygui as dpg

    # Vendored libraries
    from ..vendor.IconsFontAwesome6 import IconsFontAwesome6 as fa  # https://github.com/juliettef/IconFontCppHeaders
    from ..vendor import DearPyGui_Markdown as dpg_markdown  # https://github.com/IvanNazaruk/DearPyGui-Markdown
    from ..vendor.file_dialog.fdialog import FileDialog  # https://github.com/totallynotdrait/file_dialog, but with custom modifications

    from ..client import api
    from ..client import config as client_config

    from ..common import bgtask
    from ..common import utils as common_utils

    from ..common.gui import animation as gui_animation
    from ..common.gui import helpcard
    from ..common.gui import messagebox
    from ..common.gui import filedrop
    from ..common.gui import qroverlay
    from ..common.gui import tooltip as gui_tooltip
    from ..common.gui import utils as guiutils

    from .app_state import app_state  # Visualizer-wide shared state namespace (see `app_state.py`)
    from . import annotation
    from . import config as visualizer_config
    from . import importer  # BibTeX importer
    from . import importer_gui  # ...and its GUI
    from . import info_panel
    from . import plotter
    from . import search
    from . import selection
    from . import word_cloud

    gui_config = visualizer_config.gui_config  # shorthand, this is used a lot
logger.info(f"Libraries loaded in {tim.dt:0.6g}s.")

# --------------------------------------------------------------------------------
# Selection management subsystem wire-up
selection.reset_undo_history(_update_gui=False)  # GUI not initialized yet. This is the only time the flag should be set to `False`!


# --------------------------------------------------------------------------------
# Modal window related utilities

def enter_modal_mode():
    """Prepare the GUI for showing a modal window: hide annotation, take the current item's keyboard mark off, ...

    Call this AFTER showing your modal so that the window detects as being shown in any functionality that checks that.
    This automatically waits for one frame for the window to actually render.
    """
    logger.debug("enter_modal_mode: App entering modal mode.")
    dpg.split_frame()
    app_state.update_mouse_hover(force=True, wait=False)  # hide annotation (just in case it's there)
    info_panel.scroll_position_changed(reset=True)  # force update of current item in `update_current_search_result_status`, so the keyboard mark comes off

def exit_modal_mode():
    """Restore the GUI to main window mode (when a modal is closed): show annotation if relevant, put the current item's keyboard mark back, ...

    Call this AFTER hiding your modal so that the window detects as being hidden in any functionality that checks that.
    This automatically waits for one frame for the window to actually render.
    """
    logger.debug("exit_modal_mode: App returning to main window mode.")
    dpg.split_frame()
    info_panel.scroll_position_changed(reset=True)  # force update of current item in `update_current_search_result_status`, so the keyboard mark goes back on
    app_state.update_mouse_hover(force=True, wait=False)  # show annotation if relevant

# Register the modal-mode helpers on `app_state` so submodules can reach them.
app_state.enter_modal_mode = enter_modal_mode
app_state.exit_modal_mode = exit_modal_mode

def is_any_modal_window_visible():
    """Return whether *some* modal window is open.

    Currently these are the help card, the "open file" dialog, the "save word cloud" dialog, and the
    BibTeX importer's own two file dialogs.

    The messagebox term is here ahead of any caller: this app shows no messagebox today, and the `messagebox`
    import exists for this check alone. It is deliberate rather than speculative — the failure it forecloses
    is the one *Raven-librarian* actually hit, where the guard was written before the app had modals and
    nobody revisited it when the first one arrived. A modal blocks the mouse but not the keyboard, so an
    unguarded hotkey keeps firing behind whatever is on top, and the symptom (Enter both dismissing a dialog
    and doing something else) does not look like a missing line in this function.
    """
    return (is_open_file_dialog_visible() or word_cloud.is_save_dialog_visible() or
            importer_gui.is_any_dialog_visible() or
            help_window.is_visible() or
            messagebox.is_visible())

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

    # Stop GUI animations. Loading new data spares the ambient ones: those belong to the app rather than to
    # the dataset being replaced, and are installed once, before the render loop, so there is nowhere to put
    # them back. Exiting takes everything, so that nothing is still calling DPG as the GUI goes away.
    gui_animation.animator.clear(include_ambient=not _update_gui)

    # Only update the GUI elements if not exiting, because when exiting, the GUI is already being deleted.
    if _update_gui:
        # Clear undo history and selection
        selection.reset_undo_history()
        selection.update(common_utils.make_blank_index_array(), mode="replace", force=True, wait=False, update_selection_undo_history=False)

        # Clear the search
        dpg.set_value("search_field", "")  # tag
        search.update(wait=False)

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

def initialize_filedialogs(default_path):  # called at app startup, once we parse the default path from cmdline args (or set a default if not specified).
    """Create the file dialogs."""
    global filedialog_open
    filedialog_open = FileDialog(title="Open dataset",
                                 tag="open_file_dialog",
                                 callback=_open_file_callback,
                                 filter_list=[".pickle"],
                                 default_path=default_path)
    app_state.filedialog_save = FileDialog(title="Save word cloud as PNG",
                                           tag="save_word_cloud_dialog",
                                           callback=word_cloud.save_callback,
                                           filter_list=[".png"],
                                           save_mode=True,
                                           default_path=default_path)
    importer_gui.initialize_filedialogs(default_path)

# --------------------------------------------------------------------------------
# "Open file" dialog

def show_open_file_dialog():
    """Button callback. Show the open file dialog, for the user to pick a dataset to open.

    (And prepare the GUI for it: hide annotation, take the current item's keyboard mark off, ...)
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
    return filedialog_open.is_visible()

# --------------------------------------------------------------------------------
# Animations, live updates

class PlotterPulsatingGlow(gui_animation.Animation):  # this animation is installed once, at app startup
    def __init__(self, cycle_duration):
        """Cyclic animation to pulsate the glow highlight for search result datapoints and selected datapoints."""
        # Ambient: it pulsates for as long as the plotter exists, so an idle-framerate throttle that read it
        # as activity would never let the app idle. It holds nothing belonging to the dataset either — the
        # highlight themes and the index boxes it drives are app-lifetime — which is what lets it sit
        # through a reset for new data instead of being torn down and rebuilt per dataset.
        super().__init__(ambient=True)
        self.cycle_duration = cycle_duration

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
        alpha_search = plotter.compute_highlight_alpha(animation_pos,
                                                       len(unbox(search.search_result_data_idxs_box)),
                                                       gui_config.n_many_searchresults)
        alpha_selection = plotter.compute_highlight_alpha(1.0 - animation_pos,
                                                          len(unbox(app_state.selection_data_idxs_box)),
                                                          gui_config.n_many_selection)
        dpg.set_value(search_results_highlight_color, (*gui_config.plotter_search_results_highlight_color, alpha_search))
        dpg.set_value(selection_highlight_color, (*gui_config.plotter_selection_highlight_color, alpha_selection))

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

    search.update_field_color()
    info_panel.update_current_search_result_status()  # The "[x/x]" topmost currently visible search result indicator (also moves the keyboard mark onto the current item)

    # ----------------------------------------
    # Update various other things that need per-frame updates

    info_panel.update_navigation_controls()  # Info panel top/bottom/pageup/pagedown buttons

    if importer.has_task():
        importer_gui.update_status()

    # ----------------------------------------
    # Render all currently running animations

    gui_animation.animator.render_frame()


# --------------------------------------------------------------------------------
# Set up the main window

logger.info("Initial GUI setup...")
with timer() as tim:
    with dpg.window(tag="main_window", label="Raven-visualizer main window") as main_window:  # DPG "window" inside the app OS window ("viewport"), container for the whole GUI
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
                        # Self-sizing: the copy acknowledgment replaces this three-line caption with one line.
                        app_state.copy_report_tooltip = gui_tooltip.Tooltip("copy_report_to_clipboard_button",  # tag
                                                                            "Copy report to clipboard [F8]\n    no modifier: as plain text\n    with Shift: as Markdown")  # TODO: DRY duplicate definitions for labels

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
                               callback=importer_gui.toggle_window,
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
                    selection.update(unbox(search.search_result_data_idxs_box),
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
                # Self-sizing: `word_cloud` swaps in a "please wait" caption while a cloud is rendering.
                app_state.word_cloud_tooltip = gui_tooltip.Tooltip("word_cloud_button",  # tag
                                                                    "Toggle word cloud window [F10]")

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
                    dpg.add_text("Search", color=(140, 140, 140), tag="search_header_text")  # tag  # TODO: DRY duplicate definitions for labels

                    def clear_search():
                        dpg.set_value("search_field", "")  # tag
                        search.update()  # we should wait, because this button may get hammered.
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
                                       callback=search.search_field_callback)

                    with dpg.theme(tag="search_field_theme"):
                        with dpg.theme_component(dpg.mvInputText):
                            dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255), tag="search_field_text_color")
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
            # Self-sizing: the save acknowledgment names the file, which is longer than the caption.
            app_state.word_cloud_save_tooltip = gui_tooltip.Tooltip("word_cloud_save_button",  # tag
                                                                     "Save word cloud as PNG [Ctrl+S]")

    # BibTeX importer integration. This allows invoking the BibTeX importer from the Raven-visualizer GUI.
    importer_gui.build_window()

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
    dpg_markdown.add_text(f"{self.c_hed}**Terminology**{self.c_end}", parent=gui_parent, wrap=self.content_width)
    # The terminology section is two columns side by side, so its text wraps at half the card rather than
    # at all of it. Halved before the spacer between them is subtracted, which errs narrow - and narrow is
    # the safe direction here, wrapping a word early where the other way runs text under the next column.
    column_width = self.content_width // 2
    g = dpg.add_group(horizontal=True, parent=gui_parent)
    g1 = dpg.add_group(horizontal=False, parent=g)
    dpg_markdown.add_text(f"- {self.c_txt}**Current item**: The topmost item **fully** visible in the info panel. A pulsating blue dot marks it.{self.c_end}",
                          parent=g1, wrap=column_width)
    dpg_markdown.add_text(f"- {self.c_txt}**Current cluster**: The cluster the current item belongs to. Clusters are auto-detected by a linguistic analysis.{self.c_end}",
                          parent=g1, wrap=column_width)
    g2 = dpg.add_group(horizontal=False, parent=g)
    dpg_markdown.add_text(f"- {self.c_txt}**Selection set**: The selected items, {self.c_end}{c_selection}**glowing**{self.c_end}{self.c_txt} in the plotter. As many are loaded into the info panel as reasonably fit.{self.c_end}",
                          parent=g2, wrap=column_width)
    dpg_markdown.add_text(f"- {self.c_txt}**Search result set**: The items matching the current search, {self.c_end}{c_search}**glowing**{self.c_end}{self.c_txt} in the plotter.{self.c_end}",
                          parent=g2, wrap=column_width)
    dpg.add_spacer(width=1, height=app_state.themes_and_fonts.font_size, parent=g)

    # Additional general help
    dpg_markdown.add_text(f"{self.c_hed}**How search works**{self.c_end}",
                          parent=gui_parent, wrap=self.content_width)
    dpg_markdown.add_text(f"{self.c_txt}Each space-separated search term is a **fragment**. For a data point to match, **all** fragments must match. Ordering of fragments does **not** matter. The {self.c_end}{c_search}search result{self.c_end}{self.c_txt} and {self.c_end}{c_selection}selection{self.c_end}{self.c_txt} sets are **independent**. {self.c_end}{c_search}Search results{self.c_end}{self.c_txt} live-update as you type.{self.c_end}",
                          parent=gui_parent, wrap=self.content_width)
    dpg_markdown.add_text(f'- {self.c_txt}A **lowercase** fragment matches **that fragment {self.c_end}{self.c_hig}case-insensitively{self.c_end}{self.c_txt}**. E.g. *"hydrogen"* matches also *"Hydrogen"*.{self.c_end}',
                          parent=gui_parent, wrap=self.content_width)
    dpg_markdown.add_text(f'- {self.c_txt}A fragment with **at least one uppercase** letter matches **that fragment {self.c_end}{self.c_hig}case-sensitively{self.c_end}{self.c_txt}**. E.g. *"TiO"* matches only titanium oxide, not *"bastion"*.{self.c_end}',
                          parent=gui_parent, wrap=self.content_width)
    dpg_markdown.add_text(f'- {self.c_txt}You can use regular numbers in place of subscript/superscript numbers. E.g. *"h2so4"* matches also *"H₂SO₄"*, and *"x2"* matches also *"x²"*. {self.c_end}',
                          parent=gui_parent, wrap=self.content_width)
    dpg_markdown.add_text(f"{self.c_txt}When the search field is focused, the usual text editing keys are available (*Enter, Esc, Home, End, Shift-select, Ctrl+Left, Ctrl+Right, Ctrl+A, Ctrl+Z, Ctrl+Y*).{self.c_end}",
                          parent=gui_parent, wrap=self.content_width)
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
        info_panel.note_wheel_scroll()

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
        plotter.draw_select_radius_indicator()
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
            search_string = unbox(search.search_string_box)
            with info_panel.content_lock:  # we need to access `info_panel.entry_title_widgets`
                if not search_string:  # no search active
                    jumpable_data_idxs = {data_idx for data_idx in data_idxs_at_mouse
                                          if (data_idx in annotation_data_idxs_set) and (data_idx in info_panel.entry_title_widgets)}
                else:
                    search_result_data_idxs_set = set(unbox(search.search_result_data_idxs_box))
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
        plotter.draw_select_radius_indicator()

def keyup_callback(sender, app_data):
    """Disable selection brush indicator when Shift/Ctrl is released (and the mouse button is not down)."""
    key = app_data  # for documentation only

    if key in (dpg.mvKey_LControl, dpg.mvKey_RControl, dpg.mvKey_LShift, dpg.mvKey_RShift):
        if not dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
            plotter.clear_select_radius_indicator()

def mouse_move_callback():
    """Update the relevant GUI elements when the mouse moves.

    Currently these are:
        - Plotter data tooltip.
        - Select radius indicator for mouse-draw select.
    """
    plotter.clear_select_radius_indicator()

    if not mouse_inside_plot_widget():
        annotation.clear_mouse_hover()
        return
    # We are inside the plot widget.

    # We do the following in likely-fastest-to-likely-slowest order, to refresh each relevant GUI element as quickly as possible.

    # mouse-draw select (but only when drag began inside the plot)
    if lmb_pressed_inside_plot and dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
        plotter.draw_select_radius_indicator()
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
        plotter.clear_select_radius_indicator()
        selection.commit_change_to_undo_history()

def hotkeys_callback(sender, app_data):
    """Handle hotkeys."""
    key = app_data  # for documentation only
    ctrl_pressed = dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)
    shift_pressed = dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)

    # No shared keymap — bindings live here, and the surfaces that make them
    # discoverable mirror them by hand (KISS; hotkeys change rarely). If you add,
    # remove, or rebind a key, update those surfaces too:
    #   - the help card (search "HelpWindow")
    #   - any tooltip naming the key (search its bracketed hint, e.g. "[Ctrl+O]")

    # Hotkeys that are always available, regardless of any dialogs (even if modal)
    if key == dpg.mvKey_F11:  # de facto standard hotkey for toggle fullscreen
        toggle_fullscreen()

    # Hotkeys while the Help card is shown - helpcard handles its own hotkeys
    elif help_window.is_visible():
        return

    # Hotkeys while an "open file" or "save as" dialog is shown - fdialog handles its own hotkeys
    elif (is_open_file_dialog_visible() or word_cloud.is_save_dialog_visible() or
          importer_gui.is_any_dialog_visible()):
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
                importer_gui.show_open_dialog()
                return
            elif key == dpg.mvKey_S:
                importer_gui.show_save_dialog()
                return
            elif key == dpg.mvKey_Return:
                importer_gui.start_or_stop()
                return

    # Hotkeys for main window, while no modal window is shown
    #
    # The two branches below deliberately ask *different* questions of the same field, because Enter and the
    # navigation keys arrive in different states.
    #
    # **Enter must ask `is_item_focused`.** A *single-line* `InputText` deactivates itself on Enter — the key
    # commits the edit — so by the time this handler runs, `is_item_active` is already `False` and a gate on
    # it can never fire. (Measured: focused stays `True`, active goes `False`. A *multiline* field differs,
    # since there Enter inserts a newline and the field stays active — which is why Librarian's composer,
    # which is multiline, gates its own Enter on `is_item_active`. The predicate follows the field's kind.)
    #
    # **The bare-key branch further down must ask `is_item_active`.** ImGui gives nav focus to the first
    # navigable item of a newly focused window by itself, so this field can report focused with nobody having
    # typed in it, and a bare-key branch gated on `is_item_focused` goes dead from app start for no visible
    # reason.
    #
    # See `dpg-notes.md`, "Keyboard input".
    if dpg.is_item_focused("search_field") and key == dpg.mvKey_Return:  # tag  # regardless of modifier state, to allow Shift+Enter and Ctrl+Enter.
        select_search_results()
        # Take the caret out of the search field, so the navigation keys below reach the info panel — having
        # accepted a search, the reader is done typing and about to read.
        #
        # The button, rather than `item_information_panel` as this did before: `dpg.focus_item` cannot focus a
        # child window, and asked to, it puts focus on the enclosing window's first navigable item and
        # *activates* it — so aiming at the panel was liable to hand the caret straight back to a text field.
        # A focused button is inert here (DPG leaves ImGui's keyboard-nav activation off, so it ignores Space
        # and Enter), which is what makes it a safe place to park.
        dpg.focus_item("clear_search_button")  # tag
    # Escape needs no branch of its own: ImGui's `InputText` cancels the edit *and* deactivates itself, and
    # deactivated is exactly what the bare-key branch below tests for. The handler that used to be here
    # existed only to repair the keyboard focus afterwards, which was both unnecessary and — aimed at a child
    # window — the one call able to put the caret back where it had just left.
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
            importer_gui.toggle_window()
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
    elif not dpg.is_item_active("search_field"):  # tag  # *active*, not *focused* — see the note above the Enter branch
        if key == dpg.mvKey_Home:
            info_panel.go_to_top()
        elif key == dpg.mvKey_End:
            info_panel.go_to_bottom()
        elif key == dpg.mvKey_Next or key == 518:  # page down — DPG 2.0+ delivers 518; mvKey_Next (267) is a stale 1.x value that no longer arrives. See dpg-notes.md "Keyboard input".
            info_panel.page_down()
        elif key == dpg.mvKey_Prior or key == 517:  # page up — DPG 2.0+ delivers 517; mvKey_Prior (266) is a stale 1.x value that no longer arrives. See dpg-notes.md "Keyboard input".
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

# `raven.client.api` must be initialized before any mayberemote call. The BibTeX importer uses
# mayberemote for NLP during the import pipeline, so it needs this. No server connection is
# made here — that happens lazily on the first HTTP call.
raven_server_url = opts.server_url if opts.server_url is not None else client_config.raven_server_url
if opts.server_url is not None:
    logger.info(f"Using Raven server '{raven_server_url}' from --server-url, overriding the configured '{client_config.raven_server_url}'.")
api.initialize(raven_server_url=raven_server_url,
               raven_api_key_file=client_config.raven_api_key_file)

# Probe the server once at startup, so its presence or absence is explicit in the log rather than
# only surfacing later when the importer first reaches for it. For the Visualizer the server is
# optional — the importer falls back to loading NLP/embedding models locally — so both outcomes are
# informational, not errors.
if api.raven_server_available():
    logger.info(f"Raven-server is available at '{raven_server_url}'; server-side acceleration will be used where applicable.")
else:
    logger.info(f"Raven-server is not available at '{raven_server_url}'; running standalone, models will be loaded locally as needed.")

app_state.bg = concurrent.futures.ThreadPoolExecutor()  # for info panel and tooltip annotation updates
# Subsystem task managers (annotation, info panel, word cloud) are created lazily inside their own modules on first use.
importer.init(executor=app_state.bg)  # BibTeX importer

# import sys
# print(dir(sys.modules["__main__"]))  # DEBUG: Check this occasionally to make sure we don't accidentally store any temporary variables in the module-level namespace.

dpg.set_primary_window(main_window, True)  # Make this DPG "window" occupy the whole OS window (DPG "viewport").
dpg.set_viewport_vsync(True)
dpg.show_viewport()

# The glow that pulsates the search-result and selection highlights. It monitors the app state and updates
# at every frame, for the whole life of the app, so it goes in once here rather than per dataset.
gui_animation.animator.add(PlotterPulsatingGlow(cycle_duration=gui_config.glow_cycle_duration))

if opts.qr:
    qroverlay.install()

# Accept datasets and BibTeX files dragged in from the file manager. Installed right after `show_viewport`
# because that call is what makes DPG's window reachable through GLFW on this thread.
#
# The two kinds land in different places by their nature: a dataset is something to *open*, and only one can
# be open, so several at once is an error rather than a choice. BibTeX is input to the importer, which takes
# any number, so a dropped set opens the importer window with them already filled in.
filedrop.install(filedrop.make_router([filedrop.DropRule(matches=filedrop.by_extension(".pickle"),
                                                         handler=lambda paths: open_file(paths[0]),
                                                         label="a dataset (.pickle)",
                                                         multiple=False),
                                       filedrop.DropRule(matches=filedrop.by_extension(".bib"),
                                                         handler=importer_gui.import_bibtex_files,
                                                         label="BibTeX files to import (.bib)")],
                                      reference_window=main_window,
                                      what="Raven-visualizer",
                                      blocked=is_any_modal_window_visible))

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

exitcode = 0
try:
    # We control the render loop manually to have a convenient place to update our GUI animations just before rendering each frame.
    while dpg.is_dearpygui_running():
        update_animations()
        dpg.render_dearpygui_frame()
    # dpg.start_dearpygui()  # automatic render loop
except Exception:
    exitcode = 1
    logger.exception("Unhandled exception in render loop")
except KeyboardInterrupt:
    pass
finally:
    logger.info("App render loop exited.")

    clear_background_tasks(wait=False)  # signal background tasks to exit

    # Join each file dialog's tick thread before the context goes. A dialog that has been opened runs one,
    # and it calls DPG — after `destroy_context` that is a call into freed memory, so the failure is a
    # segfault rather than an exception. Here rather than in the exit callback, because joining is waiting
    # and the exit callback runs inside `render_dearpygui_frame`, where waiting deadlocks anything parked
    # in `split_frame`.
    for filedialog in (filedialog_open, app_state.filedialog_save):
        if filedialog is not None:
            filedialog.destroy()
    importer_gui.destroy_filedialogs()

    try:
        dpg.destroy_context()
    except BaseException:
        logger.exception("dpg.destroy_context() failed")
    common_utils.bail(exitcode)

def main():  # TODO: we don't really need this; it's just for console_scripts.
    pass

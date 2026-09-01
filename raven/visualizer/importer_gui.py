"""The BibTeX importer's GUI, for invoking `raven.visualizer.importer` from the Visualizer.

A second application UI inside the Visualizer: its own window, its own two file dialogs (pick input
`.bib` files, pick the output dataset), a per-frame status poller, and the start/stop lifecycle that
drives `importer`'s background task.

The module is the component: there is exactly one importer window, so its widgets, its selected
filenames and its file dialogs live here as module-level state rather than in an instance.

Cross-module dependencies are `app_state.{themes_and_fonts, enter_modal_mode, exit_modal_mode}` and
the `importer` module itself, which is reached through `_importer` rather than imported at the top —
everything else here is cheap to import, and that is worth keeping.

`build_window` must run inside the GUI build (it creates the window), and `initialize_filedialogs`
once the default path for the dialogs is known. `destroy_filedialogs` belongs in the app's teardown,
before `dpg.destroy_context`.
"""

__all__ = ["show_window",
           "toggle_window",

           "initialize_filedialogs",
           "destroy_filedialogs",
           "is_any_dialog_visible",
           "show_open_dialog",
           "show_save_dialog",

           "import_bibtex_files",

           "update_status",
           "start_or_stop",

           "build_window"]

import logging
logger = logging.getLogger(__name__)

import os

import dearpygui.dearpygui as dpg

from unpythonic import box, sym, unbox

from ..vendor.IconsFontAwesome6 import IconsFontAwesome6 as fa
from ..vendor.file_dialog.fdialog import FileDialog

from ..common.gui import utils as guiutils

from . import config as visualizer_config
from .app_state import app_state

gui_config = visualizer_config.gui_config


def _importer():
    """Return the BibTeX import pipeline, `raven.visualizer.importer`, importing it on first use.

    Callers bind it to a local named `importer` and then use it normally.

    That module pulls in sklearn, torch and spaCy at import time. This one is a window and its callbacks —
    nothing else it touches costs more than `dearpygui` — so importing the pipeline at the top would make a
    GUI module as expensive as the pipeline, and would take its tests out of CI, where that stack is
    deliberately absent and the tests would report a skip that reads as a pass.

    The app imports the pipeline on its own account anyway, so by the time any callback here fires this
    costs a `sys.modules` lookup.
    """
    from . import importer
    return importer

# --------------------------------------------------------------------------------
# Module-local state

_input_files_box = box([])  # the `.bib` files the user has picked as input
_output_file_box = box("")  # the dataset file the user has picked to save as

_filedialog_open = None  # FileDialog for picking the input files, created by `initialize_filedialogs`
_filedialog_save = None  # FileDialog for picking the output file, likewise

_action_start = sym("start")
_action_stop = sym("stop")


# --------------------------------------------------------------------------------
# The two filename displays
#
# These are module-level rather than nested in `build_window` (beside the tables they fill) because the
# file dialog callbacks call them too, whenever the user picks a new set of files.

def _update_output_file_table():
    """In the importer window, update the output filename display."""
    for child in dpg.get_item_children("save_import_table", slot=1):  # This won't affect table columns, because they live in a different slot.  # tag
        dpg.delete_item(child)

    output_file = unbox(_output_file_box)
    with dpg.table_row(parent="save_import_table"):  # tag
        if output_file:
            dpg.add_text(os.path.basename(output_file), color=(140, 140, 140, 255))
        else:
            dpg.add_text("[not selected]", color=(140, 140, 140, 255))

def _update_input_files_table():
    """In the importer window, update the input filename display."""
    for child in dpg.get_item_children("open_import_table", slot=1):  # This won't affect table columns, because they live in a different slot.  # tag
        dpg.delete_item(child)

    input_files = unbox(_input_files_box)
    if input_files:
        for input_file in input_files:
            with dpg.table_row(parent="open_import_table"):  # tag
                dpg.add_text(os.path.basename(input_file), color=(140, 140, 140, 255))
    else:
        with dpg.table_row(parent="open_import_table"):  # tag
            dpg.add_text("[not selected]", color=(140, 140, 140, 255))


# --------------------------------------------------------------------------------
# The window itself

def show_window():
    """Show the BibTeX importer window, centered on the main window.

    Already open: leave it alone, in the position the user gave it. Centering is for the way in, when the
    window has no position anyone chose.
    """
    if dpg.is_item_visible("importer_window"):  # tag
        return
    dpg.show_item("importer_window")  # tag
    guiutils.recenter_window("importer_window", reference_window="main_window")  # tag

def toggle_window():
    """Show/hide the BibTeX importer window."""
    if dpg.is_item_visible("importer_window"):  # tag
        dpg.hide_item("importer_window")  # tag
    else:
        show_window()


# --------------------------------------------------------------------------------
# File dialogs

def initialize_filedialogs(default_path):
    """Create the importer's two file dialogs, both starting at `default_path`.

    Called at app startup, once the default path is known (from the command line, or the working directory).
    """
    global _filedialog_open
    global _filedialog_save
    _filedialog_open = FileDialog(title="Choose BibTeX file(s) to import [Ctrl+click to multi-select]",
                                  tag="open_import_dialog",
                                  callback=_open_dialog_callback,
                                  filter_list=[".bib"],
                                  multi_selection=True,
                                  default_path=default_path)
    _filedialog_save = FileDialog(title="Save imported dataset as",
                                  tag="save_import_dialog",
                                  callback=_save_dialog_callback,
                                  filter_list=[".pickle"],
                                  save_mode=True,
                                  default_path=default_path)

def destroy_filedialogs():
    """Join the tick threads of the importer's file dialogs, so the DPG context can be destroyed safely."""
    for filedialog in (_filedialog_open, _filedialog_save):
        if filedialog is not None:
            filedialog.destroy()

def is_any_dialog_visible():
    """Return whether either of the importer's file dialogs is open.

    We have this abstraction (not just `dpg.is_item_visible`) because the dialogs might not exist yet.
    """
    return ((_filedialog_open is not None and _filedialog_open.is_visible()) or
            (_filedialog_save is not None and _filedialog_save.is_visible()))

def show_open_dialog():
    """Button callback. Show the open import file dialog, for the user to pick which BibTeX files to import.

    Does nothing if the dialog does not exist yet.
    """
    logger.debug("show_open_dialog: Showing open import dialog.")
    # The dialogs are created later in the app's bootup than the window that opens them, so the same
    # "might not exist yet" that `is_any_dialog_visible` guards against applies here.
    if _filedialog_open is None:
        logger.warning("show_open_dialog: the open import dialog does not exist yet, ignoring.")
        return
    _filedialog_open.show_file_dialog()
    app_state.enter_modal_mode()
    logger.debug("show_open_dialog: Done.")

def _open_dialog_callback(selected_files):
    """Callback that fires when the open import file dialog closes."""
    logger.debug("_open_dialog_callback: Open import dialog callback triggered.")
    app_state.exit_modal_mode()
    if selected_files:
        logger.debug(f"_open_dialog_callback: User selected the file(s) {selected_files}.")
        _input_files_box << selected_files  # the dialog hands over a list of its own, so this one is ours to keep
        _update_input_files_table()
    else:  # empty selection -> cancelled
        logger.debug("_open_dialog_callback: Cancelled.")

def show_save_dialog():
    """Button callback. Show the save import file dialog, to ask the user for a filename to save the imported dataset as.

    Does nothing if the dialog does not exist yet.
    """
    logger.debug("show_save_dialog: Showing save import dialog.")
    if _filedialog_save is None:  # see `show_open_dialog`
        logger.warning("show_save_dialog: the save import dialog does not exist yet, ignoring.")
        return
    _filedialog_save.show_file_dialog()
    app_state.enter_modal_mode()
    logger.debug("show_save_dialog: Done.")

def _save_dialog_callback(selected_files):
    """Callback that fires when the save import file dialog closes."""
    logger.debug("_save_dialog_callback: Save import dialog callback triggered.")
    app_state.exit_modal_mode()
    if len(selected_files) > 1:  # Should not happen, since we set `multi_selection=False`.
        raise ValueError(f"Expected at most one selected file, got {len(selected_files)}.")
    if selected_files:
        selected_file = selected_files[0]
        logger.debug(f"_save_dialog_callback: User selected the file '{selected_file}'.")
        _output_file_box << selected_file
        _update_output_file_table()
    else:  # empty selection -> cancelled
        logger.debug("_save_dialog_callback: Cancelled.")


# --------------------------------------------------------------------------------
# Entry point for other parts of the app

def import_bibtex_files(filenames: list[str]) -> None:
    """Open the BibTeX importer window, with `filenames` already filled in as the input files.

    This is where a drag-and-drop of BibTeX files lands. It stops short of starting the import: the importer
    also needs an output dataset to save as, and picking that is the user's next step, so the window opens
    ready rather than running.
    """
    logger.debug(f"import_bibtex_files: {len(filenames)} file(s).")
    _input_files_box << list(filenames)  # our own copy — the box outlives this call
    _update_input_files_table()
    show_window()


# --------------------------------------------------------------------------------
# Status display

def update_status():
    """Update the BibTeX importer status in the GUI.

    The app calls this every frame while the importer task is running. It is also called one more time
    when the task exits, via the `done_callback` mechanism.
    """
    importer = _importer()

    # The importer generates the GUI messages. We only need to get them from there.
    dpg.set_value("importer_status_text", unbox(importer.status_box))  # tag

    # Update the importer progress bar.
    if importer.progress is not None:
        progress_value = importer.progress.value
    else:
        progress_value = 0.0
    percentage = int(100 * progress_value)
    dpg.set_value("importer_progress_bar", progress_value)  # tag
    dpg.configure_item("importer_progress_bar", overlay=f"{percentage}%")  # tag
    # dpg.set_item_label("importer_window", f"BibTeX import [running, {percentage}%]")  # tag  # TODO: would be nice to see status while minimized, but prevents dragging the window for some reason.

def _started_callback(task_env):
    """Callback that fires when the BibTeX importer task actually starts.

    We use this to update the GUI state.
    """
    dpg.set_item_label("importer_startstop_button", fa.ICON_STOP)  # tag
    dpg.set_value("importer_startstop_tooltip_text", "Cancel BibTeX import [Ctrl+Enter]")  # tag  # TODO: DRY duplicate definitions for labels
    dpg.enable_item("importer_startstop_button")  # tag

    dpg.set_item_label("importer_startstop_heading_text_button", "Running; click to cancel")  # tag  # TODO: DRY duplicate definitions for labels
    dpg.set_value("importer_startstop_heading_text_tooltip_text", "Cancel BibTeX import [Ctrl+Enter]")  # tag  # TODO: DRY duplicate definitions for labels
    dpg.enable_item("importer_startstop_heading_text_button")  # tag

def _done_callback(task_env):
    """Callback that fires when the BibTeX importer task actually exits, via the `done_callback` mechanism.

    The callback fires regardless of whether the task completed successfully, errored out, or was cancelled.
    See `importer.start_task` for details how to use the `task_env.cancelled`, `task_env.result_code` and
    `task_env.exc` attributes.

    We use this to update the GUI state.
    """
    update_status()
    dpg.configure_item("importer_progress_bar", overlay="")  # tag
    dpg.hide_item("importer_progress_bar")  # tag

    dpg.set_item_label("importer_startstop_button", fa.ICON_PLAY)  # tag
    dpg.set_value("importer_startstop_tooltip_text", "Start BibTeX import [Ctrl+Enter]")  # tag  # TODO: DRY duplicate definitions for labels
    dpg.enable_item("importer_startstop_button")  # tag

    dpg.set_item_label("importer_startstop_heading_text_button", "Start")  # tag  # TODO: DRY duplicate definitions for labels
    dpg.set_value("importer_startstop_heading_text_tooltip_text", "Start BibTeX import [Ctrl+Enter]")  # tag  # TODO: DRY duplicate definitions for labels
    dpg.enable_item("importer_startstop_heading_text_button")  # tag

    # dpg.set_item_label("importer_window", "BibTeX import")  # tag  # TODO: DRY duplicate definitions for labels


# --------------------------------------------------------------------------------
# Start/stop lifecycle

def _start(output_file, *input_files):
    """Start the BibTeX importer to import `input_files` (.bib) into `output_file` (Raven-visualizer dataset format, currently .pickle)."""
    importer = _importer()
    if importer.has_task():
        return
    dpg.show_item("importer_progress_bar")  # tag
    dpg.disable_item("importer_startstop_button")  # tag  # Prevent multiple clicks: wait until the task actually starts before allowing the user to tell it to stop. The button will be re-enabled by the `_started_callback`.
    dpg.disable_item("importer_startstop_heading_text_button")  # tag
    importer.start_task(_started_callback, _done_callback, output_file, *input_files)

def _stop():
    """Stop (cancel) the BibTeX importer task, if any is running."""
    importer = _importer()
    if not importer.has_task():
        return
    dpg.disable_item("importer_startstop_button")  # tag  # We must wait until the previous task actually exits before we can start a new one. The button will be re-enabled by the `_done_callback`.
    dpg.disable_item("importer_startstop_heading_text_button")  # tag
    dpg.set_item_label("importer_startstop_heading_text_button", "Canceling...")  # tag  # TODO: DRY duplicate definitions for labels
    importer.cancel_task()

def start_or_stop():
    """The actual GUI button callback. Start or stop the BibTeX importer task, using the input/output filenames currently selected in the GUI."""
    logger.info("start_or_stop: called.")
    importer = _importer()
    if importer.has_task():
        logger.info("start_or_stop: importer task is running, so we will stop it.")
        action = _action_stop
    else:
        logger.info("start_or_stop: no importer task running, so we will start one.")
        action = _action_start

    if action is _action_start:
        output_file = unbox(_output_file_box)
        input_files = unbox(_input_files_box)
        logger.info(f"start_or_stop: output file is '{output_file}', input files are '{input_files}'.")
        if output_file and input_files:  # filenames specified?
            logger.info("start_or_stop: filenames have been specified. Invoking importer.")
            _start(output_file, *input_files)
        else:
            logger.info("start_or_stop: input, output or both filenames missing. Cannot start importer.")
    else:
        _stop()


# --------------------------------------------------------------------------------
# Layout

def build_window():
    """Create the BibTeX importer window. Call this once, during the app's GUI build."""
    with dpg.window(show=False, modal=False, no_title_bar=False, tag="importer_window",
                    label="BibTeX import",
                    no_scrollbar=True, autosize=True):
        with dpg.group(horizontal=False):
            def separator():
                """Add a horizontal line with a good-looking amount of vertical space around it."""
                dpg.add_spacer(width=gui_config.importer_w, height=2)  # leave some vertical space
                with dpg.drawlist(width=gui_config.importer_w, height=1):
                    dpg.draw_line((0, 0), (gui_config.importer_w - 1, 0), color=(140, 140, 140, 255), thickness=1)
                dpg.add_spacer(width=gui_config.importer_w, height=1)  # leave some vertical space

            dpg.add_spacer(width=gui_config.importer_w)  # ensure window width

            with dpg.group(horizontal=True):
                dpg.add_button(label=fa.ICON_HARD_DRIVE,
                               tag="importer_save_button",
                               width=gui_config.toolbutton_w,
                               callback=show_save_dialog)
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
                                   callback=show_save_dialog)
                    with dpg.tooltip("importer_save_heading_text_button", tag="importer_save_heading_text_tooltip"):  # tag
                        dpg.add_text("Select output dataset file to save as [Ctrl+S]", tag="importer_save_heading_text_tooltip_text")  # TODO: DRY duplicate definitions for labels
                    with dpg.table(header_row=False,
                                   sortable=False,
                                   width=gui_config.importer_w - gui_config.toolbutton_w - 11,
                                   tag="save_import_table"):
                        dpg.add_table_column(label="Output dataset file")
                    _update_output_file_table()

            with dpg.group(horizontal=True):
                dpg.add_button(label=fa.ICON_FOLDER,
                               tag="importer_select_input_files_button",
                               width=gui_config.toolbutton_w,
                               callback=show_open_dialog)
                dpg.bind_item_font("importer_select_input_files_button", app_state.themes_and_fonts.icon_font_solid)  # tag
                with dpg.tooltip("importer_select_input_files_button", tag="importer_select_input_files_tooltip"):  # tag
                    dpg.add_text("Select input BibTeX files [Ctrl+O]", tag="importer_select_input_files_tooltip_text")  # TODO: DRY duplicate definitions for labels

                with dpg.group():
                    dpg.add_button(label="Input BibTeX files",
                                   tag="importer_select_input_files_heading_text_button",
                                   width=gui_config.importer_w - gui_config.toolbutton_w - 11,
                                   callback=show_open_dialog)
                    with dpg.tooltip("importer_select_input_files_heading_text_button", tag="importer_select_input_files_heading_text_tooltip"):  # tag
                        dpg.add_text("Select input BibTeX files [Ctrl+O]", tag="importer_select_input_files_heading_text_tooltip_text")  # TODO: DRY duplicate definitions for labels
                    with dpg.table(header_row=False,
                                   sortable=False,
                                   width=gui_config.importer_w - gui_config.toolbutton_w - 11,
                                   tag="open_import_table"):
                        dpg.add_table_column(label="Input BibTeX files")
                    _update_input_files_table()

            dpg.add_spacer(width=gui_config.importer_w, height=2)  # leave some vertical space

            with dpg.group(horizontal=True):
                dpg.add_button(label=fa.ICON_PLAY,
                               tag="importer_startstop_button",
                               width=gui_config.toolbutton_w,
                               callback=start_or_stop,
                               enabled=True)
                dpg.bind_item_font("importer_startstop_button", app_state.themes_and_fonts.icon_font_solid)  # tag
                dpg.bind_item_theme("importer_startstop_button", "disablable_widget_theme")  # tag
                with dpg.tooltip("importer_startstop_button", tag="importer_startstop_tooltip"):  # tag
                    dpg.add_text("Start BibTeX import [Ctrl+Enter]", tag="importer_startstop_tooltip_text")  # TODO: DRY duplicate definitions for labels

                dpg.add_button(label="Start",
                               tag="importer_startstop_heading_text_button",
                               width=gui_config.importer_w - gui_config.toolbutton_w - 11,
                               callback=start_or_stop)
                dpg.bind_item_theme("importer_startstop_heading_text_button", "disablable_widget_theme")  # tag
                with dpg.tooltip("importer_startstop_heading_text_button", tag="importer_startstop_heading_text_tooltip"):
                    dpg.add_text("Start BibTeX import [Ctrl+Enter]", tag="importer_startstop_heading_text_tooltip_text")

            separator()

            dpg.add_progress_bar(default_value=0, width=-1, show=False, tag="importer_progress_bar")
            dpg.add_text("[To start, select files, and then click the play button.]", wrap=gui_config.importer_w, color=(140, 140, 140, 255), tag="importer_status_text")

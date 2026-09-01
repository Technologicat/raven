"""Unit tests for raven.visualizer.importer_gui.

Three kinds of decision live here:

  - **what the two filename tables say.** They are the only place the user sees what an import is about to
    run on, and they are refreshed from three directions -- the layout build, a file dialog closing, and a
    drag-and-drop -- so "one row per file, basenames only, and `[not selected]` when there are none" is a
    rule worth pinning rather than re-reading.
  - **whether an import may start.** The start button is also the stop button, and it refuses to start
    without both an output file and at least one input file, so its callback is a small decision table.
  - **the button-disabling protocol.** Start and stop both disable the button and hand re-enabling to the
    importer's `started`/`done` callbacks, which is what stops a double-click from queueing a second run of
    something that takes an hour.

This drives a real DPG context with an unmapped viewport rather than a recording stand-in, because the
layout is half of what moved out of `app.py` and the widgets it creates are what the rest of the module
addresses by tag; see `dpg-notes.md`, "Testing DPG code". The importer pipeline itself is a stand-in --
starting a real one is the expensive thing the guards exist to avoid.

These run in CI, which is why the stand-in is installed without ever touching the real pipeline: `importer`
reaches sklearn, torch and spaCy, none of which CI installs, so one incautious import would turn the whole
file into a skip that reads as a pass.
"""

import ast
import pathlib

import pytest

importer_gui = pytest.importorskip("raven.visualizer.importer_gui")

import dearpygui.dearpygui as dpg  # noqa: E402 -- after importorskip by design

from unpythonic import box, unbox  # noqa: E402 -- ditto
from unpythonic.env import env  # noqa: E402 -- ditto

from raven.vendor.IconsFontAwesome6 import IconsFontAwesome6 as fa  # noqa: E402 -- ditto
from raven.visualizer.app_state import app_state  # noqa: E402 -- ditto

WINDOW = "importer_window"  # tag
INPUT_TABLE = "open_import_table"  # tag
OUTPUT_TABLE = "save_import_table"  # tag
STARTSTOP = "importer_startstop_button"  # tag
STARTSTOP_HEADING = "importer_startstop_heading_text_button"  # tag
PROGRESS_BAR = "importer_progress_bar"  # tag
STATUS_TEXT = "importer_status_text"  # tag


class FakeImporter:
    """Stands in for `raven.visualizer.importer`, recording what the GUI asked the pipeline to do.

    Only the task interface is modelled: the GUI never reaches into the pipeline itself.
    """
    def __init__(self, running=False):
        self.running = running
        self.started_with = None  # (output_filename, input_filenames), as `start_task` received them
        self.cancelled = False
        self.started_callback = None
        self.done_callback = None
        self.status_box = box("")
        self.progress = None

    def has_task(self):
        return self.running

    def start_task(self, started_callback, done_callback, output_filename, *input_filenames):
        self.started_with = (output_filename, input_filenames)
        self.started_callback = started_callback
        self.done_callback = done_callback
        self.running = True

    def cancel_task(self):
        self.cancelled = True
        self.running = False


def table_texts(table_tag):
    """Return the text of every cell in `table_tag`, top to bottom."""
    return [dpg.get_value(cell)
            for row in dpg.get_item_children(table_tag, slot=1)
            for cell in dpg.get_item_children(row, slot=1)]


def is_enabled(tag):
    return dpg.get_item_configuration(tag)["enabled"]


@pytest.fixture
def gui(monkeypatch):
    """The importer window, freshly built into its own DPG context, with a stand-in pipeline behind it.

    Yields the `FakeImporter`, which is what a test asserts against on the pipeline side.
    """
    dpg.create_context()
    dpg.create_viewport(width=100, height=100)  # never shown: tests must not steal focus
    dpg.setup_dearpygui()

    # `build_window` binds these; the app makes them during its own bootup.
    with dpg.theme(tag="disablable_widget_theme"):  # tag
        pass
    monkeypatch.setattr(app_state, "themes_and_fonts", env(icon_font_solid=0), raising=False)

    # The selected filenames are module-local and outlive a test, so every test starts with none chosen.
    monkeypatch.setattr(importer_gui, "_input_files_box", box([]))
    monkeypatch.setattr(importer_gui, "_output_file_box", box(""))
    monkeypatch.setattr(importer_gui, "_filedialog_open", None)
    monkeypatch.setattr(importer_gui, "_filedialog_save", None)

    # Standing in for the accessor rather than for a module attribute, so the real pipeline -- and the ML
    # stack behind it -- is never imported. That is what keeps this file in CI; see `_importer`.
    fake_importer = FakeImporter()
    monkeypatch.setattr(importer_gui, "_importer", lambda: fake_importer)

    # Recentering wants a rendered frame to measure against, and nothing here renders.
    monkeypatch.setattr(importer_gui.guiutils, "recenter_window", lambda *args, **kwargs: None)

    importer_gui.build_window()
    yield fake_importer
    dpg.destroy_context()


# --------------------------------------------------------------------------------
# What this module is allowed to cost

def test_the_pipeline_is_not_imported_at_module_level():
    """Asserted against the source, because by the time this runs another test may have imported it anyway.

    A top-level `from . import importer` here would put every test in this file into the `ml` group, and the
    only visible consequence would be that CI stopped running them -- no failure, just a skip. So the guard
    has to be structural: the pipeline is reached through `_importer`, and nothing else.
    """
    tree = ast.parse(pathlib.Path(importer_gui.__file__).read_text(encoding="utf-8"))
    offenders = [node.lineno for node in tree.body
                 if isinstance(node, ast.ImportFrom) and any(alias.name == "importer" for alias in node.names)]
    assert not offenders, (f"importer_gui.py imports the pipeline at module level (line(s) {offenders}); "
                           f"that takes this whole test file out of CI")


# --------------------------------------------------------------------------------
# Layout

def test_the_window_carries_every_tag_the_module_addresses_it_by(gui):
    """The layout and the callbacks were two distant regions of `app.py`, joined only by these names."""
    for tag in (WINDOW, INPUT_TABLE, OUTPUT_TABLE, STARTSTOP, STARTSTOP_HEADING, PROGRESS_BAR, STATUS_TEXT,
                "importer_save_button",  # tag
                "importer_select_input_files_button",  # tag
                "importer_startstop_tooltip_text",  # tag
                "importer_startstop_heading_text_tooltip_text"):  # tag
        assert dpg.does_item_exist(tag), f"{tag} is missing from the built window"


def test_the_window_starts_hidden_and_the_progress_bar_with_it(gui):
    """Ctrl+I is the way in, so the window must not appear on its own; the bar belongs to a running import."""
    assert dpg.get_item_configuration(WINDOW)["show"] is False
    assert dpg.get_item_configuration(PROGRESS_BAR)["show"] is False


# --------------------------------------------------------------------------------
# The two filename tables

def test_both_tables_start_out_saying_nothing_is_selected(gui):
    assert table_texts(INPUT_TABLE) == ["[not selected]"]
    assert table_texts(OUTPUT_TABLE) == ["[not selected]"]


def test_the_input_table_lists_one_row_per_file_by_basename(gui):
    importer_gui._input_files_box << ["/home/someone/papers/first.bib", "/elsewhere/second.bib"]
    importer_gui._update_input_files_table()
    assert table_texts(INPUT_TABLE) == ["first.bib", "second.bib"]


def test_the_input_table_goes_back_to_not_selected_when_the_files_are_cleared(gui):
    importer_gui._input_files_box << ["/home/someone/papers/first.bib"]
    importer_gui._update_input_files_table()
    assert table_texts(INPUT_TABLE) == ["first.bib"], ("nothing was ever listed, so this fixture cannot "
                                                       "tell a cleared table from one that never filled")

    importer_gui._input_files_box << []
    importer_gui._update_input_files_table()
    assert table_texts(INPUT_TABLE) == ["[not selected]"]


def test_the_output_table_shows_the_basename_of_the_chosen_file(gui):
    importer_gui._output_file_box << "/home/someone/datasets/mydata.pickle"
    importer_gui._update_output_file_table()
    assert table_texts(OUTPUT_TABLE) == ["mydata.pickle"]


def test_refreshing_a_table_replaces_its_rows_rather_than_appending(gui):
    """The refresh deletes the old rows first, so picking a second, shorter set must not leave the first."""
    importer_gui._input_files_box << ["/a/one.bib", "/a/two.bib", "/a/three.bib"]
    importer_gui._update_input_files_table()
    importer_gui._input_files_box << ["/a/four.bib"]
    importer_gui._update_input_files_table()
    assert table_texts(INPUT_TABLE) == ["four.bib"]


# --------------------------------------------------------------------------------
# Drag-and-drop

def test_dropped_files_fill_the_input_table_and_open_the_window(gui):
    importer_gui.import_bibtex_files(["/tmp/dropped.bib"])
    assert table_texts(INPUT_TABLE) == ["dropped.bib"]
    assert dpg.get_item_configuration(WINDOW)["show"] is True


def test_a_drop_stops_short_of_starting_the_import(gui):
    """The importer also needs an output dataset, which is the user's next step."""
    importer_gui.import_bibtex_files(["/tmp/dropped.bib"])
    assert gui.started_with is None


def test_the_dropped_list_is_copied_rather_than_kept(gui):
    """The box outlives the call, and `filedrop` is free to reuse the list it handed over."""
    dropped = ["/tmp/dropped.bib"]
    importer_gui.import_bibtex_files(dropped)
    dropped.append("/tmp/an_afterthought.bib")
    assert unbox(importer_gui._input_files_box) == ["/tmp/dropped.bib"]


# --------------------------------------------------------------------------------
# Showing and hiding the window

def test_showing_an_already_open_window_leaves_the_position_the_user_gave_it(gui, monkeypatch):
    recentered = []
    monkeypatch.setattr(importer_gui.guiutils, "recenter_window",
                        lambda *args, **kwargs: recentered.append(args))

    importer_gui.show_window()
    assert recentered, ("the window was not recentered even on the way in, so this fixture cannot tell "
                        "a skipped recentering from one that never happens")

    monkeypatch.setattr(dpg, "is_item_visible", lambda tag: True)  # nothing renders here, so say so directly
    importer_gui.show_window()
    assert len(recentered) == 1


def test_toggle_hides_a_window_that_is_open(gui, monkeypatch):
    monkeypatch.setattr(dpg, "is_item_visible", lambda tag: True)
    importer_gui.toggle_window()
    assert dpg.get_item_configuration(WINDOW)["show"] is False


# --------------------------------------------------------------------------------
# Starting and stopping

def test_an_import_starts_once_both_filenames_are_known(gui):
    importer_gui._output_file_box << "/out/dataset.pickle"
    importer_gui._input_files_box << ["/in/one.bib", "/in/two.bib"]
    importer_gui.start_or_stop()
    assert gui.started_with == ("/out/dataset.pickle", ("/in/one.bib", "/in/two.bib"))


@pytest.mark.parametrize("output_file, input_files",
                         [("", ["/in/one.bib"]),
                          ("/out/dataset.pickle", []),
                          ("", [])],
                         ids=["no output file", "no input files", "neither"])
def test_an_import_is_refused_while_a_filename_is_missing(gui, output_file, input_files):
    importer_gui._output_file_box << output_file
    importer_gui._input_files_box << input_files
    importer_gui.start_or_stop()
    assert gui.started_with is None
    assert is_enabled(STARTSTOP), "the button was disabled for an import that never started"


def test_starting_disables_the_button_until_the_task_says_it_is_running(gui):
    """Prevents a second click queueing a second import while the first is still getting going."""
    importer_gui._output_file_box << "/out/dataset.pickle"
    importer_gui._input_files_box << ["/in/one.bib"]
    importer_gui.start_or_stop()
    assert not is_enabled(STARTSTOP)
    assert not is_enabled(STARTSTOP_HEADING)
    assert dpg.get_item_configuration(PROGRESS_BAR)["show"] is True

    gui.started_callback(None)
    assert is_enabled(STARTSTOP)
    assert dpg.get_item_label(STARTSTOP) == fa.ICON_STOP
    assert dpg.get_value("importer_startstop_tooltip_text") == "Cancel BibTeX import [Ctrl+Enter]"  # tag


def test_the_same_button_cancels_a_running_import(gui):
    gui.running = True
    importer_gui.start_or_stop()
    assert gui.cancelled
    assert not is_enabled(STARTSTOP), "the button must stay disabled until the task actually exits"
    assert dpg.get_item_label(STARTSTOP_HEADING) == "Canceling..."


def test_a_finished_import_hands_the_button_back_saying_start(gui):
    importer_gui._output_file_box << "/out/dataset.pickle"
    importer_gui._input_files_box << ["/in/one.bib"]
    importer_gui.start_or_stop()
    gui.started_callback(None)
    assert dpg.get_item_label(STARTSTOP) == fa.ICON_STOP, ("the button never changed, so this fixture "
                                                           "cannot tell a reset from an untouched button")

    gui.running = False
    gui.done_callback(None)
    assert dpg.get_item_label(STARTSTOP) == fa.ICON_PLAY
    assert dpg.get_item_label(STARTSTOP_HEADING) == "Start"
    assert is_enabled(STARTSTOP)
    assert dpg.get_item_configuration(PROGRESS_BAR)["show"] is False


def test_a_second_start_is_ignored_while_a_task_exists(gui):
    """`start_or_stop` routes a running task to cancel, but `_start` guards the direct path too."""
    gui.running = True
    importer_gui._start("/out/dataset.pickle", "/in/one.bib")
    assert gui.started_with is None


# --------------------------------------------------------------------------------
# Status display

def test_the_status_line_and_progress_bar_report_what_the_pipeline_published(gui):
    gui.status_box << "Clustering (3/7)"
    gui.progress = env(value=0.42)
    importer_gui.update_status()
    assert dpg.get_value(STATUS_TEXT) == "Clustering (3/7)"
    assert dpg.get_value(PROGRESS_BAR) == pytest.approx(0.42)
    assert dpg.get_item_configuration(PROGRESS_BAR)["overlay"] == "42%"


def test_the_progress_bar_reads_zero_before_the_pipeline_has_any_progress_to_report(gui):
    """`importer.progress` is `None` between runs, and the bar is updated every frame regardless."""
    gui.progress = None
    importer_gui.update_status()
    assert dpg.get_value(PROGRESS_BAR) == pytest.approx(0.0)
    assert dpg.get_item_configuration(PROGRESS_BAR)["overlay"] == "0%"


# --------------------------------------------------------------------------------
# File dialogs

def test_no_dialog_is_visible_before_the_dialogs_exist(gui):
    """They are created late, once the default path is known, and the modal check runs from the first frame."""
    assert importer_gui.is_any_dialog_visible() is False


def test_either_dialog_being_open_counts_as_a_modal(gui, monkeypatch):
    monkeypatch.setattr(importer_gui, "_filedialog_open", env(is_visible=lambda: False))
    monkeypatch.setattr(importer_gui, "_filedialog_save", env(is_visible=lambda: False))
    assert importer_gui.is_any_dialog_visible() is False

    monkeypatch.setattr(importer_gui, "_filedialog_save", env(is_visible=lambda: True))
    assert importer_gui.is_any_dialog_visible() is True


def test_destroying_the_dialogs_is_safe_before_they_exist(gui):
    """Teardown runs whatever killed the app, including a crash during bootup."""
    importer_gui.destroy_filedialogs()


def test_destroying_the_dialogs_joins_both_tick_threads(gui, monkeypatch):
    destroyed = []
    monkeypatch.setattr(importer_gui, "_filedialog_open", env(destroy=lambda: destroyed.append("open")))
    monkeypatch.setattr(importer_gui, "_filedialog_save", env(destroy=lambda: destroyed.append("save")))
    importer_gui.destroy_filedialogs()
    assert destroyed == ["open", "save"]


class RecordingDialog:
    """Stands in for a `FileDialog`, recording whether it was asked to open."""
    def __init__(self, visible=False):
        self.shown = 0
        self.visible = visible

    def show_file_dialog(self):
        self.shown += 1

    def is_visible(self):
        return self.visible


@pytest.fixture
def modal_mode(monkeypatch):
    """Record the modal-mode transitions, which are what stop hotkeys firing behind an open dialog."""
    entered, exited = [], []
    monkeypatch.setattr(app_state, "enter_modal_mode", lambda: entered.append(True), raising=False)
    monkeypatch.setattr(app_state, "exit_modal_mode", lambda: exited.append(True), raising=False)
    return entered, exited


@pytest.mark.parametrize("show, dialog_attribute",
                         [(lambda: importer_gui.show_open_dialog(), "_filedialog_open"),
                          (lambda: importer_gui.show_save_dialog(), "_filedialog_save")],
                         ids=["open", "save"])
def test_opening_a_dialog_also_enters_modal_mode(gui, monkeypatch, modal_mode, show, dialog_attribute):
    """The two halves are separable and only one of them is visible on screen.

    Showing the dialog without entering modal mode leaves every hotkey live behind it, which is the failure
    `is_any_modal_window_visible` exists to prevent and the one Librarian actually hit. Nothing about the
    dialog's appearance says whether the second half happened, so it is asserted here.
    """
    entered, _ = modal_mode
    dialog = RecordingDialog()
    monkeypatch.setattr(importer_gui, dialog_attribute, dialog)

    show()
    assert dialog.shown == 1
    assert entered == [True], "the dialog was shown without entering modal mode"


@pytest.mark.parametrize("show", [importer_gui.show_open_dialog, importer_gui.show_save_dialog],
                         ids=["open", "save"])
def test_opening_a_dialog_that_does_not_exist_yet_is_ignored(gui, modal_mode, show, caplog):
    """The dialogs are created later in the app's bootup than the window whose buttons open them.

    The fixture leaves both at `None`, which is the state between those two points.
    """
    entered, _ = modal_mode
    with caplog.at_level("WARNING"):
        show()
    assert entered == [], "modal mode was entered with no dialog to be modal about"
    assert any("does not exist yet" in record.message for record in caplog.records), \
        f"the refusal should say why; got {[r.message for r in caplog.records]}"


def test_the_dialogs_are_built_to_match_what_each_one_is_picking(gui, tmp_path):
    """Filters and modes, which fail quietly: a wrong filter shows the user an empty folder.

    Input is any number of `.bib` files to read; output is one dataset file to write, so the save dialog
    additionally wants the overwrite confirmation that `save_mode` brings.
    """
    importer_gui.initialize_filedialogs(str(tmp_path))

    assert importer_gui._filedialog_open is not None
    assert importer_gui._filedialog_save is not None
    assert importer_gui.is_any_dialog_visible() is False, "creating a dialog must not show it"

    assert ".bib" in importer_gui._filedialog_open.filter_list
    assert importer_gui._filedialog_open.multi_selection is True
    assert importer_gui._filedialog_open.save_mode is False

    assert ".pickle" in importer_gui._filedialog_save.filter_list
    assert importer_gui._filedialog_save.multi_selection is False
    assert importer_gui._filedialog_save.save_mode is True


def test_a_closing_open_dialog_fills_in_the_files_it_returned(gui, monkeypatch):
    monkeypatch.setattr(app_state, "exit_modal_mode", lambda: None, raising=False)
    importer_gui._open_dialog_callback(["/in/one.bib", "/in/two.bib"])
    assert table_texts(INPUT_TABLE) == ["one.bib", "two.bib"]


def test_a_cancelled_open_dialog_leaves_the_previous_choice_alone(gui, monkeypatch):
    monkeypatch.setattr(app_state, "exit_modal_mode", lambda: None, raising=False)
    importer_gui._open_dialog_callback(["/in/one.bib"])
    importer_gui._open_dialog_callback([])
    assert table_texts(INPUT_TABLE) == ["one.bib"]


def test_a_closing_save_dialog_fills_in_the_file_it_returned(gui, monkeypatch):
    monkeypatch.setattr(app_state, "exit_modal_mode", lambda: None, raising=False)
    importer_gui._save_dialog_callback(["/out/dataset.pickle"])
    assert table_texts(OUTPUT_TABLE) == ["dataset.pickle"]


def test_the_save_dialog_returning_two_files_is_an_error(gui, monkeypatch):
    """It is created with `multi_selection=False`, so this cannot happen without the dialog being wrong."""
    monkeypatch.setattr(app_state, "exit_modal_mode", lambda: None, raising=False)
    with pytest.raises(ValueError):
        importer_gui._save_dialog_callback(["/out/one.pickle", "/out/two.pickle"])

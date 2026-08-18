"""Tests for `FileDialog`'s file type filter and Find field.

Two layers. `_normalize_filter` is a module-level pure function needing no DPG context — which is why it
was hoisted out of the constructor's closure. The rest drives a real dialog against a real directory,
because what the filter *means* only shows up in which rows survive into `shown_items`.

One test maps a window and so carries the `gui` marker — the one that measures the sort row, widths being
a thing DPG has no answer for until it has rendered. Everything else runs against an unmapped viewport and
takes no keyboard focus.

The Find field's matching rule is not re-tested here: it is `raven.common.utils.make_search_matcher`,
shared with Visualizer's search and the xdot widget's, and tested with those. What *is* tested here is
that the dialog routes the Find field through it.
"""

import os
import pathlib
import threading
from unittest import mock

import pytest

dpg = pytest.importorskip("dearpygui.dearpygui", reason="dearpygui not installed")

from raven.common import filelisting  # noqa: E402 -- after importorskip by design
from raven.common.gui import utils as guiutils  # noqa: E402 -- after importorskip by design
from raven.vendor.file_dialog.fdialog import FileDialog, _PLACES, _complete_from, _normalize_filter  # noqa: E402 -- after importorskip by design


def test_normalize_bare_string_is_its_own_label():
    assert _normalize_filter(".py") == (".py", (".py",))


def test_normalize_catch_all_matches_everything():
    """`None` in place of an extension tuple is the ".*" catch-all, distinct from an empty tuple."""
    assert _normalize_filter(".*") == (".*", None)


def test_normalize_pair_separates_label_from_extensions():
    label, extensions = _normalize_filter(("Images", [".png", ".jpg"]))
    assert label == "Images"
    assert set(extensions) == {".png", ".jpg"}


def test_normalize_lowercases_and_deduplicates_extensions():
    _label, extensions = _normalize_filter(("Images", [".PNG", ".png", ".JPG"]))
    assert extensions == (".jpg", ".png")


def test_normalize_accepts_any_iterable_of_extensions():
    """Callers pass whatever the source of truth hands them — `docextract` a tuple, `codec` a frozenset."""
    _label, extensions = _normalize_filter(("Images", frozenset({".png", ".jpg"})))
    assert extensions == (".jpg", ".png")


def test_normalize_extensions_are_sorted():
    """Sorted so the tooltip listing them is stable across runs, whatever the source container's order."""
    _label, extensions = _normalize_filter(("Docs", [".pdf", ".md", ".txt"]))
    assert extensions == (".md", ".pdf", ".txt")


def test_normalize_reads_every_pair_as_label_plus_extensions():
    """A 2-tuple is always `(label, extensions)`; there is no "tuple of extensions" form.

    Pins the ambiguity rather than leaving it to be rediscovered: `(".png", ".jpg")` is a filter labelled
    ".png" matching the suffixes of the *string* ".jpg" — one character at a time, since a `str` is an
    iterable of `str`. Deterministic nonsense, and a caller who meant a set should have written a list.
    """
    label, extensions = _normalize_filter((".png", ".jpg"))
    assert label == ".png"
    assert extensions == (".", "g", "j", "p")


# --------------------------------------------------------------------------------
# The filter and the Find field, against a real directory

DIRECTORY_CONTENTS = ["photo.png", "scan.JPG", "notes.md", "paper.pdf", "archive.tar.gz", "README"]


@pytest.fixture(scope="module")
def dpg_context():
    """One DPG context for the whole module, with an unmapped viewport.

    Module-scoped because a context is not cheap and DPG keeps global state, which is the house pattern for
    every DPG test here.
    """
    dpg.create_context()
    dpg.create_viewport(width=100, height=100)  # never shown: tests must not steal focus
    dpg.setup_dearpygui()
    yield
    dpg.destroy_context()


@pytest.fixture
def dialog(dpg_context, tmp_path, request):
    """A `FileDialog` over a populated temporary directory.

    Two pieces of cleanup that are easy to miss. The tag is per-test because the shared context outlives
    each dialog, and a duplicate DPG widget ID takes the process down rather than raising. And the dialog
    `chdir`s the *whole process* on construction and on every navigation, so the original working directory
    is restored on the way out — otherwise one test relocates the rest of the suite.
    """
    for name in DIRECTORY_CONTENTS:
        pathlib.Path(tmp_path, name).touch()

    old_cwd = os.getcwd()
    yield FileDialog(tag=f"test_file_dialog_{request.node.name}",
                     default_path=str(tmp_path),
                     filter_list=[".*",
                                  ("Images", [".png", ".jpg", ".webp"]),
                                  ("Documents", [".md", ".pdf"]),
                                  ".tar.gz"],
                     file_filter=".*")
    os.chdir(old_cwd)


@pytest.fixture
def make_dialog(dpg_context, tmp_path, request):
    """Build a `FileDialog` over an empty temporary directory, with arbitrary keyword arguments.

    For the tests that are about what the *constructor* decides rather than about what gets listed. Same
    per-test tag and working-directory care as `dialog`; the counter distinguishes several dialogs built
    within one test.
    """
    old_cwd = os.getcwd()
    built = 0

    def build(**kwargs):
        nonlocal built
        built += 1
        return FileDialog(tag=f"test_file_dialog_{request.node.name}_{built}",
                          default_path=str(tmp_path),
                          **kwargs)
    yield build
    os.chdir(old_cwd)


def shown(dialog):
    """The basenames the dialog is currently listing."""
    return sorted(os.path.basename(path) for path in dialog.shown_items)


# --------------------------------------------------------------------------------
# What a directory picker promises to return

def test_a_lone_subfolder_is_not_picked_for_you(make_dialog, tmp_path):
    """Browsing a folder that happens to hold one subfolder must still offer *this* folder.

    The unique-match shortcut is for narrowing — type until one folder survives, then accept it. Applied
    with nothing typed it fires on any directory containing a single subdirectory, promising the child
    while the cursor rests on `..` meaning the parent. `~/Pictures` with one album in it is enough.
    """
    (tmp_path / "the_only_album").mkdir()
    d = make_dialog(pick="dir")
    d.reset_dir()
    assert d._effective_target() == os.getcwd()


def test_narrowing_to_one_folder_does_pick_it(make_dialog, tmp_path):
    """The other half: with something typed, one surviving folder is the answer the typing asked for."""
    (tmp_path / "alpha").mkdir()
    (tmp_path / "beta").mkdir()
    d = make_dialog(pick="dir")
    d.reset_dir(file_name_filter="alph")
    assert os.path.basename(d._effective_target()) == "alpha"


def test_a_file_picker_promises_nothing(make_dialog, tmp_path):
    """`_effective_target` is a directory-picker notion; in a file picker OK with no selection is a
    question rather than an answer, and the line that reports it is not shown at all."""
    (tmp_path / "only.txt").write_text("x")
    d = make_dialog()
    d.reset_dir()
    assert d._effective_target() is None


def test_file_filter_defaults_to_the_first_offered_item(make_dialog):
    """Every call site passed the first item's own label, so the constructor may as well say it."""
    assert make_dialog(filter_list=[".xdot", ".dot", ".gv"]).file_filter == ".xdot"
    assert make_dialog(filter_list=[("Images", [".png"]), ".*"]).file_filter == "Images"


def test_an_explicit_file_filter_still_wins(make_dialog):
    assert make_dialog(filter_list=[".xdot", ".dot", ".gv"], file_filter=".gv").file_filter == ".gv"


def test_save_extension_is_derived_from_a_single_extension_filter(make_dialog):
    """The usual shape of a save dialog: one filter, one extension, previously written a third time."""
    assert make_dialog(filter_list=[".png"], save_mode=True).default_file_extension == ".png"


def test_save_extension_is_not_derived_from_a_multi_extension_filter(make_dialog):
    """There is no principled choice among several, so nothing is added rather than an unpredictable one."""
    assert make_dialog(filter_list=[("Images", [".png", ".jpg"])], save_mode=True).default_file_extension is None


def test_save_extension_is_not_derived_from_the_catch_all(make_dialog):
    assert make_dialog(filter_list=[".*"], save_mode=True).default_file_extension is None


def test_an_explicit_save_extension_still_wins(make_dialog):
    """Including the empty string, which is how a caller says "add nothing" against a single-extension filter."""
    assert make_dialog(filter_list=[".png"], save_mode=True, default_file_extension=".jpg").default_file_extension == ".jpg"
    assert make_dialog(filter_list=[".png"], save_mode=True, default_file_extension="").default_file_extension == ""


def test_selection_and_drag_are_off_unless_asked_for(make_dialog):
    """Both defaults were the other way round, and all sixteen call sites had to say so."""
    picker = make_dialog(filter_list=[".png"])
    assert picker.multi_selection is False
    assert picker.allow_drag is False
    assert make_dialog(filter_list=[".png"], multi_selection=True).multi_selection is True


def test_catch_all_filter_shows_everything(dialog):
    assert shown(dialog) == sorted(DIRECTORY_CONTENTS)


def test_multi_extension_filter_selects_the_whole_set(dialog):
    """The point of the pair form: one labelled item covering several extensions."""
    dialog.set_type_filter("Documents")
    assert shown(dialog) == ["notes.md", "paper.pdf"]


def test_extension_matching_ignores_case(dialog):
    """`scan.JPG` is an image. Matching used to be case-sensitive, so it was not."""
    dialog.set_type_filter("Images")
    assert shown(dialog) == ["photo.png", "scan.JPG"]


def test_a_filter_may_name_a_multi_part_suffix(dialog):
    """Matching is by suffix rather than by `splitext`, so ".tar.gz" is expressible."""
    dialog.set_type_filter(".tar.gz")
    assert shown(dialog) == ["archive.tar.gz"]


def test_an_extension_absent_from_the_directory_shows_nothing(dialog):
    """Guards against a filter that silently degrades to the catch-all when nothing matches."""
    dialog.set_type_filter("Images")
    dialog.reset_dir(file_name_filter="notes")
    assert shown(dialog) == []


def test_find_field_is_case_insensitive_for_a_lowercase_query(dialog):
    dialog.reset_dir(file_name_filter="jpg")
    assert shown(dialog) == ["scan.JPG"]


def test_find_field_is_case_sensitive_for_a_query_with_uppercase(dialog):
    dialog.reset_dir(file_name_filter="PHOTO")
    assert shown(dialog) == []


def test_find_field_ands_fragments_in_any_order(dialog):
    """The Find field is wired to the shared incremental fragment search, not a plain substring test."""
    dialog.reset_dir(file_name_filter="pdf pa")
    assert shown(dialog) == ["paper.pdf"]


def test_replacing_the_filter_list_re_filters_an_open_listing(dialog):
    """For an app whose acceptable types depend on state that changes while it runs.

    `dpg.show_item` rather than `show_file_dialog`, which calls `dpg.split_frame` and would hang a test
    suite that has no render loop to satisfy the wait.
    """
    dpg.show_item(dialog.tag)
    dialog.set_filter_list([("Text-ish", [".md", ".txt"]), ".*"])
    assert shown(dialog) == ["notes.md"]
    assert dialog.file_filter == "Text-ish"


def test_replacing_the_filter_list_leaves_a_closed_listing_alone(dialog):
    """Opening rebuilds anyway, and on a large directory that rebuild is seconds — so don't do it twice."""
    before = shown(dialog)
    dialog.set_filter_list([("Text-ish", [".md", ".txt"]), ".*"])
    assert dialog.file_filter == "Text-ish"  # the selection is applied...
    assert shown(dialog) == before  # ...but the stale listing is left for the next open to replace


def test_replacing_the_filter_list_updates_the_offered_labels(dialog):
    """The combo has to forget the old items, or the user can still pick a filter that no longer exists."""
    dialog.set_filter_list([("Documents", [".md", ".pdf"]), ".*"])
    assert dpg.get_item_configuration(dialog.combo_file_filter)["items"] == ["Documents", ".*"]
    assert dpg.get_value(dialog.combo_file_filter) == "Documents"


def test_replacing_the_filter_list_can_select_a_later_item(dialog):
    """Shown first, so the listing assertion tests the new filter rather than agreeing with the old one."""
    dpg.show_item(dialog.tag)
    dialog.set_filter_list([("Documents", [".md", ".pdf"]), ".png"], file_filter=".png")
    assert dialog.file_filter == ".png"
    assert shown(dialog) == ["photo.png"]


def test_replacing_the_filter_list_re_derives_the_save_extension(make_dialog):
    """A derived extension belongs to the filter that produced it, so it cannot outlive that filter."""
    picker = make_dialog(filter_list=[".png"], save_mode=True)
    assert picker.default_file_extension == ".png"
    picker.set_filter_list([".json"])
    assert picker.default_file_extension == ".json"
    picker.set_filter_list([("Images", [".png", ".jpg"])])
    assert picker.default_file_extension is None  # no principled choice among several


def test_replacing_the_filter_list_leaves_an_explicit_save_extension_alone(make_dialog):
    """What the caller stated is the caller's, and no filter change may quietly overwrite it."""
    picker = make_dialog(filter_list=[".png"], save_mode=True, default_file_extension=".bak")
    picker.set_filter_list([".json"])
    assert picker.default_file_extension == ".bak"


def test_find_field_and_type_filter_compose(dialog):
    """Two independent concerns; a row has to survive both."""
    dialog.set_type_filter("Images")
    dialog.reset_dir(file_name_filter="o")
    assert shown(dialog) == ["photo.png"]  # "notes.md" also contains "o", but is not an image


# --------------------------------------------------------------------------------
# Closing the dialog

def count_rebuilds(dialog, monkeypatch):
    """Count `reset_dir` calls from here on, without changing what it does."""
    calls = []
    original = dialog.reset_dir

    def counting_reset_dir(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)
    monkeypatch.setattr(dialog, "reset_dir", counting_reset_dir)
    return calls


def test_cancelling_does_not_rebuild_the_listing(dialog, monkeypatch):
    """A rebuild on the way out is pure cost: the rows are hidden, and the next open rebuilds anyway.

    Timing cannot pin this — the cost only shows on a directory far too large for a test — so what is
    asserted is the property that produced it. On a 2520-entry directory a rebuild measured ~0.19 s, and
    because DPG runs callbacks one at a time, that delay landed on whichever callback came next: the
    opener button, appearing dead for as long as the close took.
    """
    dpg.show_item(dialog.tag)
    calls = count_rebuilds(dialog, monkeypatch)
    dialog.cancel()
    assert calls == []


def test_accepting_does_not_rebuild_the_listing(dialog, monkeypatch):
    """The `ok` path used to rebuild twice, so it cost twice what cancelling did."""
    dpg.show_item(dialog.tag)
    dialog.reset_dir(file_name_filter="photo")  # narrow to one, which `ok` accepts
    calls = count_rebuilds(dialog, monkeypatch)
    dialog.ok()
    assert calls == []


def test_closing_forgets_the_selection(dialog):
    """Whatever the close skips, it may not leave stale state for the next `ok` to act on."""
    dpg.show_item(dialog.tag)
    dialog.reset_dir(file_name_filter="photo")
    dialog.ok()
    assert dialog.selected_files == []
    assert dialog.shown_items == []


# --------------------------------------------------------------------------------
# Sorting, which both views share

def test_clicking_the_same_criterion_reverses_it(dialog):
    """The table header's own semantics, moved to a button: click to sort, click again to reverse.

    A listing opens sorted by name ascending, so the *first* click on Name reverses it — which is what a
    file manager does with its already-active column, and is why the test does not start there.
    """
    dialog.sort_by(filelisting.SortKey.SIZE)
    assert (dialog._sort_key, dialog._sort_descending) == (filelisting.SortKey.SIZE, False)
    dialog.sort_by(filelisting.SortKey.SIZE)
    assert dialog._sort_descending is True


def test_a_different_criterion_starts_ascending(dialog):
    dialog.sort_by(filelisting.SortKey.NAME)
    dialog.sort_by(filelisting.SortKey.NAME)  # now descending
    dialog.sort_by(filelisting.SortKey.SIZE)
    assert (dialog._sort_key, dialog._sort_descending) == (filelisting.SortKey.SIZE, False)


def test_the_sort_order_survives_a_view_switch(dialog):
    """Stated as a requirement: switching views must not change anything else."""
    dialog.sort_by(filelisting.SortKey.DATE)
    dialog.sort_by(filelisting.SortKey.DATE)  # descending
    dialog.set_grid_mode(True)
    assert (dialog._sort_key, dialog._sort_descending) == (filelisting.SortKey.DATE, True)
    dialog.set_grid_mode(False)
    assert (dialog._sort_key, dialog._sort_descending) == (filelisting.SortKey.DATE, True)


# --------------------------------------------------------------------------------
# Which view comes up

def test_an_image_typed_filter_brings_up_the_grid(dialog):
    """Picking an image by name is close to useless — generated images have hashes for filenames."""
    dialog.set_type_filter("Images")
    assert dialog._grid_mode is True


def test_the_catch_all_filter_does_not(dialog):
    """".*" selects images *among* everything; a directory of source code as thumbnails is a wall of icons."""
    dialog.set_type_filter("Images")
    dialog.set_type_filter(".*")
    assert dialog._grid_mode is False


def test_a_hand_set_view_overrides_the_automatic_one(dialog):
    """In either direction, and across filter changes within the same opening.

    Having said "not this time", the user should not have to say it again for every filter they try.
    """
    dialog.set_grid_mode(False)
    dialog.set_type_filter("Images")
    assert dialog._grid_mode is False
    dialog.set_grid_mode(True)
    dialog.set_type_filter("Documents")
    assert dialog._grid_mode is True


def test_a_hand_set_view_does_not_outlive_the_opening(dialog):
    """It cannot, because the dialog does not close: one instance serves the whole app run.

    An override that survived an opening would survive the session — tick the box once and the automatic
    switching is gone until the app restarts, which is a one-way door rather than an override.
    """
    dialog.set_type_filter("Images")
    assert dialog._grid_mode is True  # automatic: an image-typed filter
    dialog.set_grid_mode(False)  # ...overridden by hand
    dialog.cancel()

    dialog.show_file_dialog()
    assert dialog._grid_mode is True  # the automatic rule decides again
    dialog.cancel()


def test_an_explicit_view_is_what_each_opening_resets_to(make_dialog):
    """An app that asked for a view asked for it every time, not only the first."""
    d = make_dialog(filter_list=[("Images", [".png", ".jpg"]), ".*"], show_thumbnails=False)
    assert d._grid_mode is False  # not the grid, despite an image-typed filter
    d.set_grid_mode(True)
    d.cancel()

    d.show_file_dialog()
    assert d._grid_mode is False  # back to what the caller asked for, not to the automatic rule
    d.cancel()


def test_a_directory_picker_has_no_grid_view(make_dialog):
    """With no files listed every tile would be the same folder icon, so `pick="dir"` refuses the grid."""
    d = make_dialog(pick="dir", filter_list=[("Images", [".png", ".jpg"])], show_thumbnails=True)
    assert d._grid_mode is False
    d.set_grid_mode(True)
    assert d._grid_mode is False


def test_dir_with_contents_has_a_grid_view(make_dialog):
    """The mode exists to be looked at, so it gets the grid the plain directory picker refuses."""
    d = make_dialog(pick="dir-with-contents", filter_list=[("Images", [".png", ".jpg"])], show_thumbnails=True)
    assert d._grid_mode is True


def test_pick_splits_what_is_returned_from_what_is_listed(make_dialog):
    """The two axes `pick` separates, which used to be one flag."""
    assert (make_dialog(pick="file").returns_dir, make_dialog(pick="file").lists_files) == (False, True)
    assert (make_dialog(pick="dir").returns_dir, make_dialog(pick="dir").lists_files) == (True, False)
    d = make_dialog(pick="dir-with-contents")
    assert (d.returns_dir, d.lists_files) == (True, True)


def test_unknown_pick_mode_is_rejected(make_dialog):
    """A typo in a mode string is otherwise a dialog that silently behaves as a file picker."""
    with pytest.raises(ValueError):
        make_dialog(pick="directory")


def test_both_views_list_the_same_entries(dialog):
    """The unique-match shortcut in `ok` reads `shown_items`, which must not depend on the view."""
    dialog.set_type_filter(".*")
    as_rows = shown(dialog)
    dialog.set_grid_mode(True)
    assert shown(dialog) == as_rows


def test_selecting_in_the_grid_reaches_the_dialog(make_dialog, tmp_path):
    """Without this the dialog knew only about the cursor, so OK returned one file however many were marked.

    Librarian's attach dialog is `multi_selection=True` and says so in its own title, which is exactly where
    marking five images and getting one would land.
    """
    for name in DIRECTORY_CONTENTS:
        pathlib.Path(tmp_path, name).touch()
    d = make_dialog(filter_list=[("Images", [".png", ".jpg"])], multi_selection=True, show_thumbnails=True)
    grid = d._grid
    assert grid is not None

    images = [idx for idx, entry in enumerate(grid.entries) if not entry.is_dir]
    assert len(images) >= 2
    for idx in images:
        grid.toggle_select(idx)

    assert sorted(d.selected_files) == sorted(grid.entries[idx].path for idx in images)


def test_a_single_selection_dialog_does_not_offer_multi_select_in_the_grid(make_dialog, tmp_path):
    """Letting the user mark five and then honouring one is worse than not letting them mark five."""
    for name in DIRECTORY_CONTENTS:
        pathlib.Path(tmp_path, name).touch()
    d = make_dialog(filter_list=[("Images", [".png", ".jpg"])], multi_selection=False, show_thumbnails=True)
    assert d._grid._allow_multi_select is False


def test_the_grid_does_not_offer_dot_dot_as_a_selection(make_dialog, tmp_path):
    """It is the way out of the directory, not a thing in it."""
    for name in DIRECTORY_CONTENTS:
        pathlib.Path(tmp_path, name).touch()
    d = make_dialog(filter_list=[("Images", [".png", ".jpg"])], multi_selection=True, show_thumbnails=True)
    grid = d._grid
    parent = [idx for idx, entry in enumerate(grid.entries) if entry.is_parent]
    assert parent  # it is listed, and navigable...
    grid.toggle_select(parent[0])
    assert d.selected_files == []  # ...but not choosable
    assert grid.selected == set()  # ...and not even *shown* selected, which would read as a bug


def test_selecting_everything_in_the_grid_skips_what_cannot_be_returned(make_dialog, tmp_path):
    """`..` and the directories are listed so they can be navigated, not so they can be picked."""
    for name in DIRECTORY_CONTENTS:
        pathlib.Path(tmp_path, name).touch()
    pathlib.Path(tmp_path, "subdir").mkdir()
    d = make_dialog(filter_list=[".*"], multi_selection=True, show_thumbnails=True)
    grid = d._grid

    grid.select_all()

    chosen = {entry.name for entry in grid.selected_entries}
    assert ".." not in chosen
    assert "subdir" not in chosen
    assert "photo.png" in chosen


def test_the_selection_survives_a_view_switch(dialog):
    """Stated as a requirement: switching views must not change anything.

    The cursor was re-anchored by path and the selection was not, so a file chosen in the grid came back
    from the toggle still chosen but no longer *shown* as chosen.
    """
    dialog.set_type_filter(".*")
    dialog.set_grid_mode(True)
    chosen = next(entry for entry in dialog._grid.entries if not entry.is_dir)
    dialog._grid.set_selected_paths([chosen.path])
    assert dialog.selected_files == [chosen.path]

    dialog.set_grid_mode(False)
    assert dialog.selected_files == [chosen.path]

    dialog.set_grid_mode(True)
    assert dialog.selected_files == [chosen.path]


def test_the_selection_survives_a_find_field_keystroke(dialog):
    """A rebuild per keystroke must not quietly unselect what the user picked a moment ago."""
    dialog.set_type_filter(".*")
    dialog.reset_dir()
    chosen = os.path.join(os.getcwd(), "photo.png")
    dialog.selected_files.append(chosen)

    dialog.reset_dir(file_name_filter="photo")  # still matches

    assert dialog.selected_files == [chosen]


def test_a_selection_filtered_out_of_the_listing_is_dropped(dialog):
    """What is selected is what you can see selected — no state hiding behind the find field.

    The alternative, remembering it until the filter widens again, would let OK return files the user can
    no longer see and may have forgotten choosing.
    """
    dialog.set_type_filter(".*")
    dialog.reset_dir()
    dialog.selected_files.append(os.path.join(os.getcwd(), "photo.png"))

    dialog.reset_dir(file_name_filter="notes")  # no longer matches

    assert dialog.selected_files == []


def test_the_hidden_view_is_left_empty(dialog):
    """A stale listing behind the shown one is both memory and, on a switch back, the wrong answer."""
    dialog.set_grid_mode(True)
    assert dpg.get_item_children(f"explorer_{dialog.instance_tag}", 1) == []  # tag


def test_closing_from_the_tick_thread_does_not_try_to_join_it(dialog):
    """Choosing a file in grid view closes the dialog *from* the tick thread, which cannot join itself.

    A double-click is dispatched from the grid's own `update`, so `ok` runs on that thread. Joining it there
    raises `RuntimeError`, and the exception landed mid-`ok` — after the file had been handed to the app,
    before the selection was cleared — leaving state behind for the next `ok` to act on. The failure was
    caught in a log rather than by anything stopping, which is why it wants a test.
    """
    dpg.show_item(dialog.tag)
    dialog._ticker = threading.current_thread()  # stand in for being called from the ticker
    dialog.reset_dir(file_name_filter="photo")
    dialog.ok()  # must not raise
    assert dialog.selected_files == []
    assert dialog.shown_items == []


# --------------------------------------------------------------------------------
# Tab, and the keys it frees

def test_the_caret_starts_in_the_find_field(dialog):
    """Whatever the previous dialog was doing, a fresh one is ready to be typed into."""
    dialog.show_file_dialog()
    assert dialog._caret_in_listing is False


def test_tab_moves_the_caret_to_the_listing_and_back(dialog):
    dialog.reset_dir()

    dialog._handle_key(dpg.mvKey_Tab)
    assert dialog._caret_in_listing is True

    dialog._handle_key(dpg.mvKey_Tab)
    assert dialog._caret_in_listing is False


def test_left_and_right_belong_to_the_text_caret_until_tab(dialog):
    """The find field is a single-line entry, so it spends Left and Right on the text.

    They are not unwanted in the listing, they are occupied — which is the whole reason Tab exists, and
    why the grid could not be fully navigated before it.
    """
    dialog.reset_dir()
    dialog._table_cursor.set_current(2)

    dialog._handle_key(dpg.mvKey_Right)
    dialog._handle_key(dpg.mvKey_Left)

    assert dialog._table_cursor.current == 2


def test_tab_frees_left_and_right_for_the_listing(dialog):
    dialog.reset_dir()
    dialog._table_cursor.set_current(2)
    dialog._handle_key(dpg.mvKey_Tab)

    dialog._handle_key(dpg.mvKey_Right)
    assert dialog._table_cursor.current == 3

    dialog._handle_key(dpg.mvKey_Left)
    assert dialog._table_cursor.current == 2


def test_up_and_down_work_from_either_home(dialog):
    """Only the horizontal pair is contested — a single-line field leaves Up and Down alone throughout."""
    dialog.reset_dir()

    dialog._table_cursor.set_current(2)
    dialog._handle_key(dpg.mvKey_Down)
    assert dialog._table_cursor.current == 3

    dialog._handle_key(dpg.mvKey_Tab)
    dialog._handle_key(dpg.mvKey_Down)
    assert dialog._table_cursor.current == 4


def test_focusing_the_find_field_brings_the_caret_with_it(dialog):
    """Ctrl+F is a way back to the field, so it has to agree with Tab about where the caret now is.

    The flag and the DPG focus are two records of one fact; a key that moved focus without updating the
    flag would leave the arrow keys bound to the listing while the user typed into the field.
    """
    dialog.reset_dir()
    dialog._handle_key(dpg.mvKey_Tab)
    assert dialog._caret_in_listing is True

    with mock.patch.object(dpg, "is_key_down", lambda key: key in (dpg.mvKey_LControl,)):
        dialog._handle_key(dpg.mvKey_F)

    assert dialog._caret_in_listing is False


# --------------------------------------------------------------------------------
# The places panel

def test_the_places_are_ordered_predictably():
    """Home first, then the rest alphabetically, case-insensitively.

    Upstream's order was neither, and nothing enforced one — so this exists to catch an addition landing
    wherever it was typed rather than where a reader will look for it.
    """
    labels = [label for label, _icon in _PLACES]
    assert labels[0] == "Home"
    assert labels[1:] == sorted(labels[1:], key=str.casefold)


def test_every_place_names_an_icon_the_dialog_has(dialog):
    """The icon is reached by `getattr`, so a typo would be an `AttributeError` while building the panel."""
    for label, icon in _PLACES:
        assert hasattr(dialog, icon), f"place '{label}' names a missing icon attribute '{icon}'"


# --------------------------------------------------------------------------------
# Enter: "go as deep as this entry allows"
#
# The governing rule of the whole keyboard design, and until now checked only by hand. Each test names the
# case of the rule it covers, so a failure says which half of the sentence broke.

def cursor_onto(dialog, basename):
    """Put the keyboard cursor on the row for `basename`, and return its index."""
    for idx, entry in enumerate(dialog._row_entries):
        if os.path.basename(entry.path) == basename or entry.name == basename:
            dialog._table_cursor.set_current(idx)
            return idx
    raise AssertionError(f"no row named '{basename}' in {[e.name for e in dialog._row_entries]}")


def test_enter_on_a_directory_descends_into_it(make_dialog, tmp_path):
    # Navigation is `chdir`; the constructor has already put us in `tmp_path`. Calling `reset_dir` with a
    # different directory would list that one without moving there, which is not a state the app reaches.
    (tmp_path / "album").mkdir()
    dialog = make_dialog(pick="file")

    cursor_onto(dialog, "album")
    dialog._handle_key(dpg.mvKey_Return)

    assert os.path.realpath(os.getcwd()) == os.path.realpath(tmp_path / "album")


def test_enter_on_the_parent_entry_goes_up(make_dialog, tmp_path):
    (tmp_path / "album").mkdir()
    dialog = make_dialog(pick="file")
    dialog.chdir(str(tmp_path / "album"))
    assert os.path.realpath(os.getcwd()) == os.path.realpath(tmp_path / "album"), "precondition: we are inside"

    cursor_onto(dialog, "..")
    dialog._handle_key(dpg.mvKey_Return)

    assert os.path.realpath(os.getcwd()) == os.path.realpath(tmp_path)


def test_enter_on_a_choosable_file_accepts_it(dialog):
    """A file in a file picker is the bottom, so accepting it *is* the deepest available move."""
    chosen = []
    dialog.change_callback(lambda paths: chosen.append(paths))
    dialog.reset_dir()

    cursor_onto(dialog, "photo.png")
    dialog._handle_key(dpg.mvKey_Return)

    assert chosen and [os.path.basename(p) for p in chosen[0]] == ["photo.png"]


def test_enter_on_scenery_declines(make_dialog, tmp_path):
    """A file in a folder picker is shown so the folder can be judged by it, and Enter does nothing to it.

    The dialog must stay open and return nothing — the failure this guards against is a folder picker that
    hands back a file because the cursor happened to rest on one.
    """
    (tmp_path / "photo.png").touch()
    chosen = []
    dialog = make_dialog(pick="dir-with-contents", callback=lambda paths: chosen.append(paths))
    dialog.reset_dir()

    cursor_onto(dialog, "photo.png")
    dialog._handle_key(dpg.mvKey_Return)

    assert chosen == []
    assert dialog.selected_files == []


def test_ctrl_enter_declines_to_descend(make_dialog, tmp_path):
    """The counterpart rule: Ctrl+Enter commits here, even with a directory under the cursor."""
    (tmp_path / "album").mkdir()
    chosen = []
    dialog = make_dialog(pick="dir", callback=lambda paths: chosen.append(paths))
    dialog.reset_dir()

    cursor_onto(dialog, "album")
    with mock.patch.object(dpg, "is_key_down", lambda key: key in (dpg.mvKey_LControl,)):
        dialog._handle_key(dpg.mvKey_Return)

    assert os.path.realpath(os.getcwd()) == os.path.realpath(tmp_path), "Ctrl+Enter must not descend"
    assert chosen and os.path.realpath(chosen[0][0]) == os.path.realpath(tmp_path / "album")


def test_enter_with_no_cursor_falls_back_to_the_ok_button(make_dialog, tmp_path):
    """With nothing under the cursor, Enter means whatever OK would have meant.

    A defensive branch rather than a routine one: every real listing contains `..`, and `set_current`
    refuses an out-of-range index, so the cursor cannot be moved off the rows once they exist. The state
    is therefore staged here by emptying the rows — which is what a dialog looks like before its first
    listing is built. The branch is worth keeping and worth pinning; a bare `ok()` is the right answer to
    "act on the cursor" when there is no cursor, and the alternative is an `IndexError` on a rare path.
    """
    chosen = []
    dialog = make_dialog(pick="dir", callback=lambda paths: chosen.append(paths))
    dialog.reset_dir()
    dialog._row_entries.clear()

    dialog._handle_key(dpg.mvKey_Return)

    assert chosen and os.path.realpath(chosen[0][0]) == os.path.realpath(tmp_path)


# --------------------------------------------------------------------------------
# Tab completion
#
# The rule: extend the field to the longest common prefix of the entries on screen.
#
# `candidates` is always what the *listing* is showing, so every set below is one the fragment search can
# actually produce for the query beside it. Feeding a set it could not — say `headers.h` against `re`,
# which does not contain `re` — tests a situation the dialog never reaches, and hides real behaviour.

def test_completion_extends_to_the_common_prefix():
    """Typing `re` leaves only the readmes on screen; what they share is `readme.`."""
    assert _complete_from("re", ["readme.txt", "readme.md"]) == "readme."


def test_completion_works_from_a_fragment_that_prefixes_nothing():
    """`eadm` starts neither name, and it does not need to: the listing already holds only the matches."""
    assert _complete_from("eadm", ["readme.txt", "readme.md"]) == "readme."


def test_completion_declines_when_the_shown_entries_share_nothing():
    """`ead` matches `headers.h` too, and `readme`/`headers` have no common prefix. Better to do nothing."""
    assert _complete_from("ead", ["readme.txt", "readme.md", "headers.h"]) is None


def test_completion_never_discards_an_entry_the_user_can_see():
    """All three contain `data`, so all three are legitimate and none may be filtered away by completing.

    Preferring the candidates that *start with* the query would answer `datasets` here — the only one that
    does — and applying that to the field would drop `rawdata` off the screen. There is nothing the three
    share, so the honest answer is to complete nothing.
    """
    assert _complete_from("data", ["rawdata", "datasets", "tempdatasets"]) is None


def test_completion_completes_a_unique_match_fully():
    assert _complete_from("re", ["readme.txt"]) == "readme.txt"


def test_completion_declines_when_there_is_nothing_left_to_add(dialog):
    assert _complete_from("readme.", ["readme.txt", "readme.md"]) is None


def test_completion_stays_case_insensitive_when_the_query_was():
    """The casing follows the same rule as everything else here: never narrow what is on screen.

    A lowercase query compares case-insensitively, so writing an entry's own spelling back into the field
    would make it case-*sensitive* and drop whatever differs in case. The folded form matches everything
    that matched before, which is why `read` against these two answers `readme` and not `README`.
    """
    assert _complete_from("read", ["README", "readme.txt"]) == "readme"
    # With one candidate there is nothing to drop, and the same folded answer still matches it.
    assert _complete_from("read", ["README"]) == "readme"


def test_an_uppercase_query_matches_exactly():
    # An uppercase query is case-sensitive, so `readme.txt` is not on screen to be completed against.
    assert _complete_from("READ", ["README"]) == "README"
    # ...whereas a lowercase one shows both, and what they share stops at the case difference.
    assert _complete_from("read", ["README", "readme.txt"]) == "readme"  # folded, so both stay shown


def test_completion_of_an_empty_query_is_what_the_whole_listing_shares(dialog):
    assert _complete_from("", ["alpha", "album"]) == "al"


def test_completion_declines_on_an_empty_listing():
    assert _complete_from("x", []) is None


def test_tab_completes_the_find_field_and_hands_over_the_arrow_keys(make_dialog, tmp_path):
    """The two halves of Tab, in the order that makes the write possible.

    Leaving the field is what lets it be written at all, so the completion is applied on the way out —
    and the caret ends up in the listing either way.
    """
    for name in ("readme.txt", "readme.md", "headers.h"):
        (tmp_path / name).touch()
    dialog = make_dialog(pick="file", filter_list=[".*"], file_filter=".*")
    dialog.reset_dir(file_name_filter="re")

    dialog._handle_key(dpg.mvKey_Tab)

    assert dpg.get_value(dialog.search_field) == "readme."
    assert dialog._caret_in_listing is True


def test_tab_back_fills_the_field_from_the_cursor(make_dialog, tmp_path):
    """Coming back from the listing means "give me the one I navigated to".

    The same in both modes, deliberately: in save mode it is how an existing name becomes the template for
    a variant, and in open mode it collapses the listing to the entry picked. A rule that differed between
    them would be one more thing to hold in mind for no gain.
    """
    for name in ("rawdata", "datasets", "tempdatasets"):
        (tmp_path / name).touch()
    dialog = make_dialog(pick="file", filter_list=[".*"], file_filter=".*")
    dialog.reset_dir(file_name_filter="data")
    dialog._handle_key(dpg.mvKey_Tab)                 # into the listing
    dialog._handle_key(dpg.mvKey_Down)                # arrow to a different match
    picked = os.path.basename(dialog._table_cursor.current_key)

    dialog._handle_key(dpg.mvKey_Tab)                 # ...and back

    assert dpg.get_value(dialog.search_field) == picked
    assert dialog._caret_in_listing is False
    assert shown(dialog) == [picked], "the listing narrows to what was picked"


def test_tab_back_from_the_parent_entry_leaves_the_field_alone(make_dialog, tmp_path):
    """`..` is a way out of the directory rather than a name, so there is nothing to take from it."""
    # Two names sharing no prefix, so the *outbound* Tab completes nothing and cannot be mistaken for the
    # inbound one. With a single entry it would fill the field with that entry's whole name on the way out.
    (tmp_path / "album").mkdir()
    (tmp_path / "briefs").mkdir()
    dialog = make_dialog(pick="file", filter_list=[".*"], file_filter=".*")
    dialog.reset_dir()
    assert dialog._row_entries[dialog._table_cursor.current].is_parent, "precondition: cursor on `..`"

    dialog._handle_key(dpg.mvKey_Tab)
    assert dpg.get_value(dialog.search_field) == "", "precondition: nothing was completed on the way out"

    dialog._handle_key(dpg.mvKey_Tab)

    assert dpg.get_value(dialog.search_field) == ""


def test_tab_leaves_the_field_alone_when_there_is_nothing_to_complete(make_dialog, tmp_path):
    for name in ("readme.txt", "headers.h"):
        (tmp_path / name).touch()
    dialog = make_dialog(pick="file", filter_list=[".*"], file_filter=".*")
    dialog.reset_dir(file_name_filter="ead")

    dialog._handle_key(dpg.mvKey_Tab)

    assert dpg.get_value(dialog.search_field) == ""  # the filter came from `reset_dir`, not from typing
    assert dialog._caret_in_listing is True, "the caret still moves; only the completion declined"


# --------------------------------------------------------------------------------
# A search belongs to the directory it was typed in

def test_navigating_away_clears_the_search(make_dialog, tmp_path):
    """Going somewhere else must not leave a stale query in the field.

    The listing is rebuilt unfiltered on arrival, so a query left behind describes nothing that is on
    screen: the field claims to be narrowing and the listing shows everything. Worse where the query
    matched nothing, since then the only row to act on is `..` — so the very act of escaping a failed
    search was what left the field lying.
    """
    (tmp_path / "album").mkdir()
    dialog = make_dialog(pick="file")
    dialog.chdir(str(tmp_path / "album"))

    dpg.set_value(dialog.search_field, "zzz")  # matches nothing here
    dialog._update_search()
    assert shown(dialog) == [], "precondition: the search matches nothing"

    cursor_onto(dialog, "..")
    dialog._handle_key(dpg.mvKey_Return)

    assert dpg.get_value(dialog.search_field) == ""
    assert "album" in shown(dialog), "the new directory is listed in full"


def test_the_mouse_and_the_keyboard_agree_about_this(make_dialog, tmp_path):
    """`chdir` is the one place that navigates, so every route through it clears the field alike."""
    (tmp_path / "album").mkdir()
    dialog = make_dialog(pick="file")
    dpg.set_value(dialog.search_field, "alb")
    dialog._update_search()

    dialog.chdir(str(tmp_path / "album"))  # what a double-click ends up calling

    assert dpg.get_value(dialog.search_field) == ""


# --------------------------------------------------------------------------------
# Where a search leaves the cursor

def test_a_search_puts_the_cursor_on_its_first_hit(dialog):
    """Otherwise "type a few characters, press Enter" leaves the directory instead of opening the match.

    The cursor rests on `..` while nothing is typed, and `..` is what Enter acts on — so a filter that
    narrows to exactly what you were after, with the cursor still parked above it, sends you up a level.
    """
    dialog.reset_dir(file_name_filter="photo")
    assert dialog._row_entries[0].is_parent, "precondition: `..` is row 0"
    assert dialog._table_cursor.current == 1


def test_the_parent_entry_answers_a_search_like_any_other_name(make_dialog, tmp_path):
    """Typing `..` puts the cursor on `..`, so "go up" is reachable by search and not only by a key.

    The listing keeps `..` whatever is typed — it is the way out, and must stay available when a query
    matches nothing — but that is a separate thing from whether it can be *searched for*.
    """
    # `a.txt` matters: the query must match something *besides* `..`, or the listing collapses to one row
    # and any rule at all leaves the cursor on it. This is the case that tells "first match" apart from
    # "first match after the parent".
    (tmp_path / "a.txt").touch()
    dialog = make_dialog(pick="file", filter_list=[".*"], file_filter=".*")

    dialog.reset_dir(file_name_filter=".")  # matches `..` and `a.txt` alike

    assert len(dialog._row_entries) == 2, "precondition: both rows are showing"
    assert dialog._table_cursor.current == 0
    assert dialog._row_entries[0].is_parent


def test_a_search_matching_nothing_leaves_the_cursor_on_the_way_out(make_dialog, tmp_path):
    """The one row left to act on, so Enter still escapes a query that found nothing."""
    (tmp_path / "album").mkdir()
    dialog = make_dialog(pick="file", filter_list=[".*"], file_filter=".*")

    dialog.reset_dir(file_name_filter="zzqqxx")

    assert shown(dialog) == []
    assert dialog._table_cursor.current == 0
    assert dialog._row_entries[0].is_parent


def test_the_grid_moves_its_cursor_onto_the_first_hit_too(dialog):
    """Both views, or the rule is a property of the table rather than of the dialog."""
    dialog.set_grid_mode(True)
    dialog.reset_dir(file_name_filter="photo")
    assert dialog._grid.current == 1


def test_landing_on_a_match_is_not_a_choice_the_user_made(dialog):
    """The user picked a *query*, not an entry, so the landing must not become the cursor's anchor.

    The anchor is where the cursor tries to return when the listing changes again. Anchoring an entry
    nobody chose means erasing the query returns the cursor to whatever happened to match first, rather
    than to the resting place it started from.
    """
    dialog.reset_dir(file_name_filter="photo")
    assert dialog._table_cursor.current == 1
    landed_on = dialog._row_entries[1].path

    dialog.reset_dir()  # query erased; same directory, so the cursor re-anchors rather than starting over

    assert dialog._table_cursor.current_key != landed_on, \
        "the cursor followed an entry the user never chose"


def test_erasing_the_query_returns_the_cursor_to_its_resting_place(dialog):
    """One rule, read forwards: a cursor nobody moved goes back to where it was before the search.

    With a query typed that is the first hit; with none it is `..`. Holding the *index* instead would
    leave the cursor on whichever entry happened to occupy that row in the wider listing — a position
    with no meaning to anyone.
    """
    dialog.reset_dir(file_name_filter="photo")
    assert dialog._table_cursor.current == 1

    dialog.reset_dir()

    assert dialog._table_cursor.current == 0
    assert dialog._row_entries[0].is_parent, "which is `..`"


def test_a_search_shows_its_first_hit_even_after_arrowing_somewhere(make_dialog, tmp_path):
    """Typing a query is a fresh intent, so it overrides wherever the cursor had got to.

    Without this, arrowing anywhere switched the search off: the cursor stayed with the entry it had been
    moved to, and a query typed afterwards narrowed the listing while leaving the cursor behind.

    The three names matter. All match the query, so the entry arrowed to survives the filter — which is
    the case that tells the two rules apart. Where the arrowed entry is filtered *out*, the fallback
    happens to clamp onto the first hit anyway, and a test built on that passes either way.
    """
    for name in ("aaa.txt", "aab.txt", "aac.txt"):
        (tmp_path / name).touch()
    dialog = make_dialog(pick="file", filter_list=[".*"], file_filter=".*")
    dialog.reset_dir()
    dialog._table_cursor.set_current(3)  # `aac.txt`, as an arrow key would — which anchors
    assert os.path.basename(dialog._table_cursor.current_key) == "aac.txt"

    dialog.reset_dir(file_name_filter="aa")  # matches all three, the arrowed one included

    assert dialog._table_cursor.current == 1
    assert os.path.basename(dialog._table_cursor.current_key) == "aaa.txt"


def test_the_arrowed_entry_is_still_where_erasing_the_query_returns_you(make_dialog, tmp_path):
    """The other half: the search moved the cursor, but it did not take the choice away."""
    for name in ("aaa.txt", "aab.txt", "aac.txt"):
        (tmp_path / name).touch()
    dialog = make_dialog(pick="file", filter_list=[".*"], file_filter=".*")
    dialog.reset_dir()
    dialog._table_cursor.set_current(3)
    chosen = dialog._table_cursor.current_key

    dialog.reset_dir(file_name_filter="aa")  # cursor jumps to the first hit
    dialog.reset_dir()                       # query erased

    assert dialog._table_cursor.current_key == chosen


def test_a_cursor_the_user_moved_is_left_where_it_belongs(dialog):
    """The other half, and why this is one rule rather than a special case for the empty query.

    Arrowing somewhere is a choice, so it anchors — and the cursor then returns to *that entry* when the
    listing changes, rather than being sent back to `..` along with the unchosen ones.
    """
    dialog.reset_dir()
    dialog._table_cursor.set_current(3)  # as an arrow key would
    chosen = dialog._table_cursor.current_key

    dialog.reset_dir(file_name_filter="")  # a rebuild of the same directory

    assert dialog._table_cursor.current_key == chosen


# --------------------------------------------------------------------------------
# Up one level

def held(*keys):
    """Pretend the given modifier keys are held down for the duration of the block."""
    return mock.patch.object(dpg, "is_key_down", lambda key: key in keys)


@pytest.mark.parametrize("modifier", [dpg.mvKey_LAlt, dpg.mvKey_RAlt,
                                      dpg.mvKey_LControl, dpg.mvKey_RControl])
def test_a_modified_up_leaves_the_directory(make_dialog, tmp_path, modifier):
    """Alt+Up is the standard chord; Ctrl+Up is the one-handed alias, and the two are interchangeable.

    Both sides of each modifier, because the alias exists precisely *because* the two sides of Alt are
    not the same key on a Nordic layout — so which side a chord answers to is the point of it.
    """
    (tmp_path / "album").mkdir()
    dialog = make_dialog(pick="file")
    dialog.chdir(str(tmp_path / "album"))

    with held(modifier):
        dialog._handle_key(dpg.mvKey_Up)

    assert os.path.realpath(os.getcwd()) == os.path.realpath(tmp_path)


def test_an_unmodified_up_still_moves_the_cursor(make_dialog, tmp_path):
    """The half that would break silently: a chord claiming a bare key it was meant to share.

    Going up would leave the cursor at `..` in the parent, which reads exactly like a cursor that moved
    one row — so the directory is what this asserts, not the cursor.
    """
    (tmp_path / "album").mkdir()
    dialog = make_dialog(pick="file")
    dialog.chdir(str(tmp_path / "album"))
    dialog._table_cursor.set_current(0)

    with held():  # nothing held
        dialog._handle_key(dpg.mvKey_Up)

    assert os.path.realpath(os.getcwd()) == os.path.realpath(tmp_path / "album")


# --------------------------------------------------------------------------------
# Hidden files

def test_hidden_entries_are_out_of_the_listing_until_asked_for(make_dialog, tmp_path):
    """Both directions, since the toggle is the only way back once it has been used."""
    (tmp_path / ".config").mkdir()
    (tmp_path / "notes.txt").touch()
    (tmp_path / ".secret.txt").touch()
    dialog = make_dialog(pick="file")

    assert shown(dialog) == ["notes.txt"]

    dialog.set_show_hidden_files(True)
    assert shown(dialog) == [".config", ".secret.txt", "notes.txt"]

    dialog.set_show_hidden_files(False)
    assert shown(dialog) == ["notes.txt"]


def test_ctrl_h_toggles_them(make_dialog, tmp_path):
    (tmp_path / ".secret.txt").touch()
    dialog = make_dialog(pick="file")

    with held(dpg.mvKey_LControl):
        dialog._handle_key(dpg.mvKey_H)
    assert shown(dialog) == [".secret.txt"]

    with held(dpg.mvKey_LControl):
        dialog._handle_key(dpg.mvKey_H)
    assert shown(dialog) == []


def test_the_checkbox_and_the_hotkey_are_one_control(make_dialog, tmp_path):
    """Whichever route is taken, the other one's widget has to agree — a checkbox left unticked after
    Ctrl+H would offer to "show" what is already shown."""
    dialog = make_dialog(pick="file")

    with held(dpg.mvKey_LControl):
        dialog._handle_key(dpg.mvKey_H)

    assert dpg.get_value(dialog.checkbox_hidden_files) is True


def test_a_hidden_directory_is_reachable_in_a_folder_picker(make_dialog, tmp_path):
    """The case the toggle is offered unconditionally for: a folder picker lists no files to make
    thumbnails of, so its Thumbnails box is hidden — but hidden *folders* are exactly what a user
    reaching for a config directory came here to find."""
    (tmp_path / ".config").mkdir()
    dialog = make_dialog(pick="dir")

    with held(dpg.mvKey_LControl):
        dialog._handle_key(dpg.mvKey_H)

    assert shown(dialog) == [".config"]


def test_the_query_survives_the_toggle(make_dialog, tmp_path):
    """Showing hidden files re-lists rather than navigates, so the find field is left alone — unlike
    `chdir`, which clears it."""
    (tmp_path / ".notes.txt").touch()
    (tmp_path / "notes.md").touch()
    (tmp_path / "photo.png").touch()
    dialog = make_dialog(pick="file")
    dpg.set_value(dialog.search_field, "notes")
    dialog._update_search()

    with held(dpg.mvKey_LControl):
        dialog._handle_key(dpg.mvKey_H)

    assert dpg.get_value(dialog.search_field) == "notes"
    assert shown(dialog) == [".notes.txt", "notes.md"]


# --------------------------------------------------------------------------------
# Ctrl+Space: what Ctrl+click does, without the mouse

def test_ctrl_space_marks_and_unmarks_the_cursor_entry(make_dialog, tmp_path):
    (tmp_path / "one.txt").touch()
    (tmp_path / "two.txt").touch()
    dialog = make_dialog(pick="file", multi_selection=True)

    cursor_onto(dialog, "one.txt")
    with held(dpg.mvKey_LControl):
        dialog._handle_key(dpg.mvKey_Spacebar)
    assert [os.path.basename(p) for p in dialog.selected_files] == ["one.txt"]

    cursor_onto(dialog, "two.txt")
    with held(dpg.mvKey_LControl):
        dialog._handle_key(dpg.mvKey_Spacebar)
    assert sorted(os.path.basename(p) for p in dialog.selected_files) == ["one.txt", "two.txt"]

    cursor_onto(dialog, "one.txt")
    with held(dpg.mvKey_LControl):
        dialog._handle_key(dpg.mvKey_Spacebar)
    assert [os.path.basename(p) for p in dialog.selected_files] == ["two.txt"]


def test_ctrl_space_does_nothing_in_a_single_selection_dialog(make_dialog, tmp_path):
    """There is no Ctrl+click to mirror there, and a key that half-works is worse than one that does not:
    marking two files in a dialog that returns one would be a promise it cannot keep."""
    (tmp_path / "one.txt").touch()
    dialog = make_dialog(pick="file")  # `multi_selection` defaults off

    cursor_onto(dialog, "one.txt")
    with held(dpg.mvKey_LControl):
        dialog._handle_key(dpg.mvKey_Spacebar)

    assert dialog.selected_files == []


def test_ctrl_space_declines_on_what_the_dialog_cannot_return(make_dialog, tmp_path):
    """`..` and scenery are shown but are not answers, and the cursor rests on `..` by default."""
    (tmp_path / "album").mkdir()
    (tmp_path / "readme.txt").touch()
    dialog = make_dialog(pick="dir-with-contents", multi_selection=True)

    for name in ("..", "readme.txt"):  # the way out, and a file in a folder picker
        cursor_onto(dialog, name)
        with held(dpg.mvKey_LControl):
            dialog._handle_key(dpg.mvKey_Spacebar)
        assert dialog.selected_files == [], f"'{name}' is not something this dialog returns"


def test_a_marked_row_shows_as_marked(make_dialog, tmp_path):
    """The bookkeeping and the widget are two records of one fact, and the widget is the one the user
    reads — a selection the listing does not show is a selection nobody knows they made."""
    (tmp_path / "one.txt").touch()
    dialog = make_dialog(pick="file", multi_selection=True)

    idx = cursor_onto(dialog, "one.txt")
    with held(dpg.mvKey_LControl):
        dialog._handle_key(dpg.mvKey_Spacebar)

    name_cell = dialog._row_themes[idx][0][0]
    assert dpg.get_value(name_cell) is True


def test_unmarking_a_folder_updates_what_ok_promises(make_dialog, tmp_path):
    """An explicit selection outranks the cursor in a directory picker, so the promised-target line has
    to follow one. It did not follow a Ctrl+click either, which is why both routes now share the update.

    *Un*marking is what can tell the two apart. Marking the folder under the cursor promises the folder
    the cursor was already promising, so a line that never refreshed would look right the whole way
    through — the first version of this test asserted exactly that and passed without the refresh.
    Here the first-marked folder outranks the cursor, and dropping it hands the promise to the second.
    """
    (tmp_path / "a_album").mkdir()
    (tmp_path / "b_album").mkdir()
    dialog = make_dialog(pick="dir", multi_selection=True)

    for name in ("a_album", "b_album", "a_album"):  # mark, mark, unmark
        cursor_onto(dialog, name)
        with held(dpg.mvKey_LControl):
            dialog._handle_key(dpg.mvKey_Spacebar)

    assert [os.path.basename(p) for p in dialog.selected_files] == ["b_album"]
    assert dpg.get_value(dialog.text_target) == f"Will pick: {tmp_path / 'b_album'}"


@pytest.mark.gui
def test_the_sort_row_fits_the_minimum_width(make_dialog):
    """The floor in `min_size` is a measurement, and this is what re-takes it when the row grows.

    The sort buttons are fixed-width and the row does not reflow, so a control added to its right end
    pushes the rightmost one off the edge at widths that used to be fine — silently, since a clipped
    checkbox looks like a checkbox that was never there. `width=-1` on the find field makes it exactly
    as wide as the content area, which is the space the row is competing for.

    Carries the `gui` marker: a widget has no size until frames are rendered, and DPG aborts the process
    if asked to render without a mapped viewport. Measured at `font_size=20`, which every app in the
    constellation uses — the labels are what the row's width is made of, so at DPG's own default font it
    comes out 33 px narrower and the floor this justifies would be 33 px too low.
    """
    guiutils.setup_default_font(20)
    dialog = make_dialog()
    dpg.show_viewport()
    try:
        dialog.show_file_dialog()
        dpg.set_item_width(dialog.tag, dialog.min_size[0])  # tag
        for _ in range(6):
            dpg.render_dearpygui_frame()

        rightmost = dialog.checkbox_hidden_files
        needed = dpg.get_item_pos(rightmost)[0] + dpg.get_item_rect_size(rightmost)[0]
        available = dpg.get_item_rect_size(dialog.search_field)[0]
        assert available >= needed, (f"the sort row needs {needed} px but min_size leaves {available}; "
                                     f"raise min_size to about {dialog.min_size[0] + needed - available}")
    finally:
        dpg.hide_item(dialog.tag)  # tag
        dpg.bind_font(0)  # the module's other tests are not measuring, but leave them as they were found

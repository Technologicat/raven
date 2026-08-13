"""Tests for `FileDialog`'s file type filter and Find field.

Two layers. `_normalize_filter` is a module-level pure function needing no DPG context — which is why it
was hoisted out of the constructor's closure. The rest drives a real dialog against a real directory,
because what the filter *means* only shows up in which rows survive into `shown_items`.

No window is ever mapped, so nothing here takes keyboard focus and none of it carries the `gui` marker.

The Find field's matching rule is not re-tested here: it is `raven.common.utils.make_search_matcher`,
shared with Visualizer's search and the xdot widget's, and tested with those. What *is* tested here is
that the dialog routes the Find field through it.
"""

import os
import pathlib

import pytest

dpg = pytest.importorskip("dearpygui.dearpygui", reason="dearpygui not installed")

from raven.vendor.file_dialog.fdialog import FileDialog, _normalize_filter  # noqa: E402 -- after importorskip by design


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
    dialog.reset_dir(file_name_filter="notes", default_path=os.getcwd())
    assert shown(dialog) == []


def test_find_field_is_case_insensitive_for_a_lowercase_query(dialog):
    dialog.reset_dir(file_name_filter="jpg", default_path=os.getcwd())
    assert shown(dialog) == ["scan.JPG"]


def test_find_field_is_case_sensitive_for_a_query_with_uppercase(dialog):
    dialog.reset_dir(file_name_filter="PHOTO", default_path=os.getcwd())
    assert shown(dialog) == []


def test_find_field_ands_fragments_in_any_order(dialog):
    """The Find field is wired to the shared incremental fragment search, not a plain substring test."""
    dialog.reset_dir(file_name_filter="pdf pa", default_path=os.getcwd())
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
    dialog.reset_dir(file_name_filter="o", default_path=os.getcwd())
    assert shown(dialog) == ["photo.png"]  # "notes.md" also contains "o", but is not an image

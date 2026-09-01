"""Unit tests for raven.visualizer.info_panel.

The largest module in the package, and the one whose testable parts are the least contiguous. Three
kinds of code live in it, and only two are reachable without a running app:

  - **decisions**, which is what is covered here: what a hotkey does to the selection or the search
    field, what goes on the clipboard, where cluster navigation stops, and how a widget says what kind
    of thing it is.
  - **geometry** -- scroll positions, "which item is at the top of the content area", the anchoring that
    survives a rebuild. These read back positions that exist only once frames have rendered, so they need
    a running app rather than a context, and they are not tested here.
  - **the content build**, `_update_info_panel`, which is the other half of that and is not tested either.

Where a decision sits behind a geometry query, the query is where the tests cut: `_get_current_item`
answers "which item is at the top of the panel", and the interesting part is what the hotkeys then *do*
with it. Stubbing exactly there is what makes the rest reachable.
"""

import ast
import pathlib

import pytest

info_panel = pytest.importorskip("raven.visualizer.info_panel")

import dearpygui.dearpygui as dpg  # noqa: E402 -- after importorskip by design

from unpythonic import box  # noqa: E402 -- ditto
from unpythonic.env import env  # noqa: E402 -- ditto

from raven.visualizer.app_state import app_state  # noqa: E402 -- ditto

SEARCH_FIELD = "search_field"  # tag


STOPWORDS = {"the", "of", "a", "and"}  # stands in for spaCy's list; see the `gui` fixture


def test_spacy_is_not_imported_at_module_level():
    """Asserted against the source, because another test may have imported spaCy by the time this runs.

    The module needs one static thing from spaCy -- the English stopword set, for one search-field
    convenience -- and reaching it costs about 2.4 s and three thousand modules. Paying that at import
    time also takes this whole file out of CI, where spaCy is deliberately absent, and the only visible
    consequence would be a skip that reads as a pass. `_get_stopwords` is the way in.
    """
    tree = ast.parse(pathlib.Path(info_panel.__file__).read_text(encoding="utf-8"))
    offenders = [node.lineno for node in tree.body
                 if (isinstance(node, ast.ImportFrom) and (node.module or "").startswith("spacy"))
                 or (isinstance(node, ast.Import) and any(a.name.startswith("spacy") for a in node.names))]
    assert not offenders, (f"info_panel.py imports spaCy at module level (line(s) {offenders}); "
                           f"that takes this whole test file out of CI")


def test_the_stopword_set_is_the_english_one_and_is_cached():
    """The one test here that wants the real list, so the only one that needs spaCy.

    Deliberately separate from the tests of what the panel *does* with stopwords: those stand a fixed set
    in, so they run wherever the suite runs and cannot be broken by upstream editing its word list.
    """
    pytest.importorskip("spacy", reason="spaCy not installed (the stopword list's only source)")
    first = info_panel._get_stopwords()
    assert "the" in first and "of" in first, "this does not look like an English stopword set"
    assert "titanium" not in first, "an ordinary content word must not be filtered out of a search"
    assert info_panel._get_stopwords() is first, "the set is rebuilt on every call, which is the cost the "\
                                                 "lazy load exists to pay only once"


# --------------------------------------------------------------------------------
# The header and the navigation bar
#
# These moved out of `app.py`, where they could not be built in a test at all: `app.py` created the
# widgets and `info_panel` reached back for them by tag, with `build_window`'s docstring carrying the
# ordering requirement as prose that nothing enforced. Built here into a real context, so the wiring is
# checked rather than described.


@pytest.fixture
def dpg_context():
    """A DPG context with an unmapped viewport, plus the two things the builders bind to."""
    dpg.create_context()
    dpg.create_viewport(width=100, height=100)  # never shown: tests must not steal focus
    dpg.setup_dearpygui()
    with dpg.theme(tag="disablable_widget_theme"):  # tag
        pass
    yield
    dpg.destroy_context()


@pytest.fixture
def built_chrome(dpg_context, monkeypatch):
    """The header and the navigation bar, built as the app builds them.

    Inside a window, because both create *child* windows and DPG has nowhere to put one otherwise -- the
    app calls them from inside its own layout tree, which is the whole point of their shape.
    """
    monkeypatch.setattr(app_state, "themes_and_fonts", env(icon_font_solid=0), raising=False)
    monkeypatch.setattr(info_panel, "_copy_report_tooltip", None)
    with dpg.window():
        info_panel.build_header()
        info_panel.build_navigation_controls()


NAVIGATION_BUTTONS = {"go_to_top_button": "go_to_top",
                      "page_up_button": "page_up",
                      "page_down_button": "page_down",
                      "go_to_bottom_button": "go_to_bottom",
                      "prev_search_match_button": "scroll_to_prev_search_match",
                      "next_search_match_button": "scroll_to_next_search_match"}


def test_the_header_carries_the_widgets_the_module_writes_to(built_chrome):
    for tag in ("item_information_header", "item_information_title",
                "item_information_selection_item_count", "item_information_total_count",
                "info_panel_pending_spinner", "info_panel_rendering_spinner",
                "copy_report_to_clipboard_button"):
        assert dpg.does_item_exist(tag), f"{tag} is missing from the built header"
    assert info_panel._copy_report_tooltip is not None, "the copy button's self-sizing caption was not made"


def test_the_status_readouts_start_hidden_and_the_spinners_with_them(built_chrome):
    # Nothing is loaded yet, so there are no counts to show and no refresh in flight. `update` and the
    # per-frame poller are what reveal these.
    for tag in ("item_information_total_count", "info_panel_pending_spinner", "info_panel_rendering_spinner"):
        assert dpg.get_item_configuration(tag)["show"] is False, f"{tag} should start hidden"
    assert dpg.get_value("item_information_selection_item_count") == "[nothing selected]"


@pytest.mark.parametrize("tag, function_name", sorted(NAVIGATION_BUTTONS.items()))
def test_every_navigation_button_is_wired_to_its_own_function(built_chrome, tag, function_name):
    """The part that used to be seven `set_item_callback` calls against widgets another module had made.

    A wrong or missing binding is silent: the button draws, enables itself with the others, and does
    nothing when clicked.
    """
    assert dpg.does_item_exist(tag)
    assert dpg.get_item_callback(tag) is getattr(info_panel, function_name)


def test_the_copy_button_is_wired_too(built_chrome):
    assert dpg.get_item_callback("copy_report_to_clipboard_button") is info_panel.copy_report_to_clipboard


def test_every_navigation_button_starts_disabled(built_chrome):
    # There is nothing to navigate until a dataset is loaded and something is in the panel;
    # `update_navigation_controls` is what enables them.
    for tag in NAVIGATION_BUTTONS:
        assert dpg.get_item_configuration(tag)["enabled"] is False, f"{tag} should start disabled"
    assert dpg.get_item_configuration("copy_report_to_clipboard_button")["enabled"] is False


def test_the_search_match_readouts_start_at_no_search(built_chrome):
    assert dpg.get_value("item_information_search_controls_item_count") == "[no search active]"
    assert dpg.get_item_configuration("item_information_search_controls_current_item")["show"] is False


class RecordingDPG:
    """Stands in for `dearpygui.dearpygui`, holding the widget state `info_panel` reads back.

    `user_data` is the interesting one: Raven stores `(kind, data)` on widgets and finds them again by
    asking what kind they are, so a test that wants a widget of some kind registers it here.
    """
    def __init__(self, real_dpg):
        self._real_dpg = real_dpg
        self.user_data = {}  # item -> (kind, data)
        self.values = {}  # tag -> value
        self.keys_down = set()
        self.clipboard = None

    def __getattr__(self, name):
        return getattr(self._real_dpg, name)

    def get_item_configuration(self, item):
        if item in self.user_data:
            return {"user_data": self.user_data[item]}
        return {}  # a widget with nothing filed under `user_data` at all

    def get_value(self, tag):
        return self.values.get(tag, "")

    def set_value(self, tag, value):
        self.values[tag] = value

    def is_key_down(self, key):
        return key in self.keys_down

    def set_clipboard_text(self, text):
        self.clipboard = text


def make_entry(title, cluster_id, data_idx, **fields):
    return env(title=title, cluster_id=cluster_id, data_idx=data_idx,
               author="Author, An", year="2024", **fields)


@pytest.fixture
def gui(monkeypatch):
    """An info panel with three entries in two clusters, and everything outside it recorded."""
    fake_dpg = RecordingDPG(info_panel.dpg)
    monkeypatch.setattr(info_panel, "dpg", fake_dpg)
    # `select_cluster_by_id` reaches `selection.keyboard_state_to_mode`, which reads *that* module's own
    # `dpg` binding -- so patching only this module's leaves a live call into a toolkit with no context,
    # which segfaults rather than raising. Standing in for both keeps the real modifier mapping running
    # against the keys a test presses, which is the point: the alternative, stubbing the mapping itself,
    # would leave the test asserting against its own stub.
    monkeypatch.setattr(info_panel.selection, "dpg", fake_dpg)
    monkeypatch.setattr(info_panel.gui_animation, "flash_button", lambda **kwargs: None)
    # A fixed stopword set rather than spaCy's. What this module decides is to drop whatever the loader
    # hands it; *which* words those are is spaCy's business, and is asserted in the one test that says so.
    # Standing in here keeps these tests off a multi-second import CI does not have, and off a word list
    # that upstream is free to change under them.
    monkeypatch.setattr(info_panel, "_get_stopwords", lambda: STOPWORDS)

    entries = [make_entry("The synthesis of a novel catalyst", cluster_id=0, data_idx=0),
               make_entry("A study of methanol", cluster_id=0, data_idx=1),
               make_entry("Something else entirely", cluster_id=1, data_idx=2)]
    dataset = env(sorted_entries=entries)
    monkeypatch.setattr(app_state, "dataset", dataset, raising=False)

    # Read as an argument to the acknowledgment flash, so it has to exist even though the flash is stubbed.
    # Normally created by `build_header`, which these tests do not run.
    monkeypatch.setattr(info_panel, "_copy_report_tooltip", env(text=""))

    searched = []
    monkeypatch.setattr(app_state, "update_search", lambda **kwargs: searched.append(kwargs), raising=False)

    selections = []
    monkeypatch.setattr(info_panel.selection, "update",
                        lambda idxs, mode="replace", **kwargs: selections.append((list(idxs), mode, kwargs)))

    # The public content maps, which `_update_info_panel` normally swaps in wholesale. Widget IDs are
    # arbitrary integers; 0 is among them deliberately, since it is a valid DPG ID and a falsy one.
    monkeypatch.setattr(info_panel, "widget_to_data_idx", {0: 0, 11: 1, 12: 2})
    monkeypatch.setattr(info_panel, "entry_title_widgets", {0: 0, 1: 11, 2: 12})
    monkeypatch.setattr(info_panel, "cluster_ids_in_selection", [0, 1, -1])
    monkeypatch.setattr(info_panel, "cluster_id_to_display_idx", {0: 0, 1: 1, -1: 2})
    monkeypatch.setattr(info_panel, "report_plaintext", box("the report, as plain text"))
    monkeypatch.setattr(info_panel, "report_markdown", box("# the report, as Markdown"))

    scrolled = []
    monkeypatch.setattr(info_panel, "scroll_to_item", lambda item: scrolled.append(item))

    return env(dpg=fake_dpg, dataset=dataset, entries=entries,
               searched=searched, selections=selections, scrolled=scrolled)


def be_current(monkeypatch, item):
    """Make `item` the current item -- the topmost fully visible one.

    That answer comes from widget geometry, which needs rendered frames; the decisions the hotkeys make
    about it do not. This is the seam between the two.
    """
    monkeypatch.setattr(info_panel, "_get_current_item", lambda: item)


# --------------------------------------------------------------------------------
# How a widget says what kind of thing it is

def test_a_widget_is_recognized_by_the_kind_filed_on_it(gui):
    gui.dpg.user_data[11] = ("entry_title_container", None)
    assert info_panel._is_entry_title_container_group(11) == 11


def test_a_widget_of_another_kind_is_not_recognized(gui):
    # Negative control: the predicates all share one implementation, so this is what says the `kind`
    # comparison happens at all.
    gui.dpg.user_data[11] = ("cluster_title", None)
    assert info_panel._is_entry_title_container_group(11) is None


def test_a_matching_widget_whose_id_is_zero_is_still_returned(gui):
    # DPG item 0 is a valid ID and a falsy value, which is why these predicates answer with the item and
    # `None` rather than with a bool. A caller writing `if predicate(item):` would drop this widget, so
    # what has to hold is that the match is distinguishable from the miss.
    gui.dpg.user_data[0] = ("entry_title_container", None)
    assert info_panel._is_entry_title_container_group(0) == 0
    assert info_panel._is_entry_title_container_group(0) is not None


def test_a_widget_with_no_user_data_at_all_is_not_recognized(gui):
    # Most widgets in the panel carry none: spacers, separators, the text inside a group.
    assert info_panel._get_user_data(11) is None
    assert info_panel._is_entry_title_container_group(11) is None


def test_asking_about_no_widget_is_answered_rather_than_raising(gui):
    # The finders return `None` when they find nothing, and their answer is fed straight back in.
    assert info_panel._get_user_data(None) is None
    assert info_panel._is_entry_title_container_group(None) is None


# --------------------------------------------------------------------------------
# The clipboard

def test_the_report_is_copied_as_plain_text_by_default(gui):
    info_panel.copy_report_to_clipboard()
    assert gui.dpg.clipboard == "the report, as plain text"


def test_holding_shift_copies_the_report_as_markdown(gui):
    gui.dpg.keys_down = {gui.dpg.mvKey_LShift}
    info_panel.copy_report_to_clipboard()
    assert gui.dpg.clipboard == "# the report, as Markdown"


def test_either_shift_key_selects_markdown(gui):
    # Negative control for the test above, which presses one particular key: the rule is about the
    # modifier, not about which hand reached it.
    gui.dpg.keys_down = {gui.dpg.mvKey_RShift}
    info_panel.copy_report_to_clipboard()
    assert gui.dpg.clipboard == "# the report, as Markdown"


def test_an_unknown_report_format_is_refused(gui):
    # Two formats are built during every panel rebuild, and picking one by string is how the hotkey and
    # the button both reach them. A typo should not quietly copy the wrong one, or an empty box.
    with pytest.raises(ValueError):
        info_panel._copy_report_to_clipboard(report_format="pdf")


def test_an_entry_is_cited_as_authors_year_and_title(gui):
    # The citation shape someone pastes into notes or an email, so the format is the feature. It takes
    # the entry rather than the widget showing it, which is what lets it be read at all.
    assert info_panel.format_entry_citation(gui.entries[1]) == "Author, An (2024): A study of methanol"


def test_copying_one_entry_puts_its_citation_on_the_clipboard(gui, monkeypatch):
    gui.dpg.user_data[99] = ("copy_entry_to_clipboard_button", (object(), "Copy this entry"))
    monkeypatch.setattr(info_panel.widgetfinder, "find_widget_depth_first", lambda item, accept: 99)
    info_panel._copy_entry_to_clipboard(11)
    assert gui.dpg.clipboard == info_panel.format_entry_citation(gui.entries[1])


def test_copying_the_current_entry_copies_the_one_at_the_top_of_the_panel(gui, monkeypatch):
    gui.dpg.user_data[99] = ("copy_entry_to_clipboard_button", (object(), "Copy this entry"))
    monkeypatch.setattr(info_panel.widgetfinder, "find_widget_depth_first", lambda item, accept: 99)
    be_current(monkeypatch, 12)
    info_panel.copy_current_entry_to_clipboard()
    assert gui.dpg.clipboard == "Author, An (2024): Something else entirely"


def test_copying_with_an_empty_panel_does_nothing(gui, monkeypatch):
    # Negative control for the test above, and the case that would otherwise raise: no items, so no
    # current item, and the hotkey still fires because a hotkey does not know the panel is empty.
    be_current(monkeypatch, None)
    info_panel.copy_current_entry_to_clipboard()
    assert gui.dpg.clipboard is None


# --------------------------------------------------------------------------------
# Search-or-select

def test_the_default_action_searches_the_plotter_for_the_current_entry(gui, monkeypatch):
    be_current(monkeypatch, 11)
    info_panel.search_or_select_current_entry()
    assert gui.dpg.values[SEARCH_FIELD] == "study methanol"
    assert gui.searched, "the search has to be re-run, or the field says one thing and the plot shows another"


def test_the_search_drops_stopwords_from_the_title(gui, monkeypatch):
    # A title makes a poor query unstripped: the search matches by substring, so its short common words
    # match almost everywhere. Asserted against the stubbed set, since what belongs to this module is
    # dropping them -- `test_search.test_a_lowercase_stopword_matches_inside_longer_words` is why.
    be_current(monkeypatch, 0)
    info_panel.search_or_select_current_entry()
    assert gui.dpg.values[SEARCH_FIELD] == "synthesis novel catalyst", \
        "'The', 'of' and 'a' are in the stubbed stopword set; the rest of the title is the query"


def test_asking_again_for_the_entry_already_searched_clears_the_search(gui, monkeypatch):
    # The same key both searches and un-searches, so a reader can look an item up and put the plot back
    # without reaching for the search field.
    be_current(monkeypatch, 11)
    info_panel.search_or_select_current_entry()
    info_panel.search_or_select_current_entry()
    assert gui.dpg.values[SEARCH_FIELD] == ""


def test_holding_shift_selects_the_current_entry_alone(gui, monkeypatch):
    be_current(monkeypatch, 11)
    gui.dpg.keys_down = {gui.dpg.mvKey_LShift}
    info_panel.search_or_select_current_entry()
    assert gui.selections == [([1], "replace", {"wait": False})]
    assert SEARCH_FIELD not in gui.dpg.values, "the selection actions leave the search alone"


def test_an_unknown_search_or_select_action_is_refused(gui):
    # The action arrives as a string from whichever handler read the gesture, so a typo should not
    # quietly fall through to the search branch, which is the one with no modifier attached to it.
    with pytest.raises(ValueError):
        info_panel._search_or_select_entry(gui.entries[1], "seach")


def test_holding_ctrl_removes_the_current_entry_from_the_selection(gui, monkeypatch):
    be_current(monkeypatch, 11)
    gui.dpg.keys_down = {gui.dpg.mvKey_LControl}
    info_panel.search_or_select_current_entry()
    assert gui.selections == [([1], "subtract", {"wait": False})]


def test_search_or_select_with_an_empty_panel_does_nothing(gui, monkeypatch):
    be_current(monkeypatch, None)
    info_panel.search_or_select_current_entry()
    assert gui.selections == []
    assert SEARCH_FIELD not in gui.dpg.values


# --------------------------------------------------------------------------------
# Cluster navigation

def in_cluster(monkeypatch, gui, display_idx):
    """Put the current item in the cluster shown `display_idx`-th in the panel."""
    cluster_id = info_panel.cluster_ids_in_selection[display_idx]
    entry = make_entry("Whatever", cluster_id=cluster_id, data_idx=0)
    monkeypatch.setattr(app_state, "dataset", env(sorted_entries=[entry]), raising=False)
    monkeypatch.setattr(info_panel, "widget_to_data_idx", {11: 0})
    be_current(monkeypatch, 11)


def test_scrolling_to_the_next_cluster_goes_to_the_one_after_the_current_item_s(gui, monkeypatch):
    in_cluster(monkeypatch, gui, 0)
    info_panel.scroll_to_next_cluster()
    assert gui.scrolled == [f"cluster_1_title_build{info_panel.build_number}"]


def test_scrolling_to_the_previous_cluster_goes_to_the_one_before(gui, monkeypatch):
    in_cluster(monkeypatch, gui, 1)
    info_panel.scroll_to_prev_cluster()
    assert gui.scrolled == [f"cluster_0_title_build{info_panel.build_number}"]


def test_the_next_cluster_from_the_last_one_stays_put(gui, monkeypatch):
    # No wraparound: the panel is a list a reader is walking down, and arriving back at the top would
    # read as having lost your place rather than as having reached the end.
    in_cluster(monkeypatch, gui, 2)
    info_panel.scroll_to_next_cluster()
    assert gui.scrolled == []


def test_the_previous_cluster_from_the_first_one_stays_put(gui, monkeypatch):
    in_cluster(monkeypatch, gui, 0)
    info_panel.scroll_to_prev_cluster()
    assert gui.scrolled == []


def test_scrolling_to_the_top_of_the_current_cluster_stays_in_that_cluster(gui, monkeypatch):
    # Negative control for the two above: they assert that nothing happened, which is also what a broken
    # `_scroll_to_cluster_by_id` would produce. This one goes somewhere.
    in_cluster(monkeypatch, gui, 1)
    info_panel.scroll_to_top_of_current_cluster()
    assert gui.scrolled == [f"cluster_1_title_build{info_panel.build_number}"]


def test_navigating_clusters_with_an_empty_panel_does_nothing(gui, monkeypatch):
    be_current(monkeypatch, None)
    info_panel.scroll_to_next_cluster()
    info_panel.scroll_to_prev_cluster()
    assert gui.scrolled == []


def test_a_cluster_not_currently_shown_offers_nowhere_to_navigate(gui, monkeypatch):
    # The panel shows the clusters of the current selection; the current item's cluster is normally one
    # of them, but a rebuild in flight can leave the old content showing an item whose cluster is not.
    entry = make_entry("Whatever", cluster_id=42, data_idx=0)
    monkeypatch.setattr(app_state, "dataset", env(sorted_entries=[entry]), raising=False)
    monkeypatch.setattr(info_panel, "widget_to_data_idx", {11: 0})
    be_current(monkeypatch, 11)
    info_panel.scroll_to_next_cluster()
    assert gui.scrolled == []


# --------------------------------------------------------------------------------
# Selecting a whole cluster

def test_selecting_a_cluster_selects_every_entry_in_it(gui):
    info_panel.select_cluster_by_id(0, "replace")
    data_idxs, mode, _ = gui.selections[0]
    assert sorted(data_idxs) == [0, 1], "both entries of cluster 0, and neither of cluster 1"
    assert mode == "replace"


def test_the_combine_mode_is_passed_through_rather_than_decided_here(gui):
    # Negative control for the mode in the test above: it is the caller's, so a cluster can be added to
    # or subtracted from a selection being built up.
    info_panel.select_cluster_by_id(1, "add")
    _, mode, _ = gui.selections[0]
    assert mode == "add"


def test_selecting_the_current_cluster_reads_the_modifiers_at_the_gesture(gui, monkeypatch):
    # The hotkey handler is where the keyboard is read, because that is where the gesture is. The
    # operation below it takes the answer, so it cannot read a modifier the user has since let go of.
    be_current(monkeypatch, 12)
    gui.dpg.keys_down = {gui.dpg.mvKey_LShift}
    info_panel.select_current_cluster()
    _, mode, _ = gui.selections[0]
    assert mode == "add"


def test_selecting_the_current_cluster_selects_the_current_item_s_cluster(gui, monkeypatch):
    be_current(monkeypatch, 12)  # the entry in cluster 1
    info_panel.select_current_cluster()
    data_idxs, _, _ = gui.selections[0]
    assert data_idxs == [2]


def test_selecting_the_current_cluster_with_an_empty_panel_does_nothing(gui, monkeypatch):
    be_current(monkeypatch, None)
    info_panel.select_current_cluster()
    assert gui.selections == []


# --------------------------------------------------------------------------------
# Lifecycle

def test_clearing_tasks_before_anything_has_been_rendered_is_harmless(monkeypatch):
    monkeypatch.setattr(info_panel, "_task_manager", None)
    info_panel.clear_tasks()


def test_clearing_tasks_reaches_the_task_manager_once_there_is_one(monkeypatch):
    # Negative control for the test above: the guard skips a missing manager rather than skipping always.
    cleared = []

    class FakeTaskManager:  # not an `env`: `clear` is one of its reserved names
        def clear(self, wait):
            cleared.append(wait)

    monkeypatch.setattr(info_panel, "_task_manager", FakeTaskManager())
    info_panel.clear_tasks(wait=True)
    assert cleared == [True]

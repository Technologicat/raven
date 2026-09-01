"""Unit tests for raven.visualizer.search.

`find_matches` is the whole reason this module exists apart from `app.py`: what the Visualizer counts as
a match, stated without a GUI. The rest of the module reports the answer -- the plotter's highlight
series, the header that doubles as a match counter, the colour of the field -- and is exercised here
against a recording stand-in, since those are commands rather than measurements.

The matching *rule* is `common_utils.make_search_matcher`, and is tested where it lives. What is tested
here is what this package does with it: which field it looks in, what it hands back, and that a query
matching nothing is distinguishable from no query at all.
"""

import numpy as np

import pytest

search = pytest.importorskip("raven.visualizer.search")

from unpythonic import unbox  # noqa: E402 -- after importorskip by design
from unpythonic.env import env  # noqa: E402 -- ditto

from raven.common import utils as common_utils  # noqa: E402 -- ditto
from raven.visualizer.app_state import app_state  # noqa: E402 -- ditto

SEARCH_FIELD = "search_field"  # tag
HEADER = "search_header_text"  # tag
RESULTS_SERIES = "my_search_results_scatter_series"  # tag
FIELD_COLOR = "search_field_text_color"  # tag


class RecordingDPG:
    """Stands in for `dearpygui.dearpygui`: holds the search field's text, records everything set."""
    def __init__(self, real_dpg):
        self._real_dpg = real_dpg
        self.values = {}

    def __getattr__(self, name):
        return getattr(self._real_dpg, name)

    def get_value(self, tag):
        return self.values.get(tag, "")

    def set_value(self, tag, value):
        self.values[tag] = value


def make_entry(title):
    """One entry, with the normalized title the scan actually looks at."""
    return env(title=title, normalized_title=common_utils.normalize_search_string(title.strip()))


TITLES = ["Photocatalytic degradation of methanol",
          "Laser ablation of steel",
          "A study of CO₂ capture",
          "Catalysis without photons"]


@pytest.fixture
def dataset():
    return env(sorted_entries=[make_entry(title) for title in TITLES],
               sorted_lowdim_data=np.array([[float(i), float(10 * i)] for i in range(len(TITLES))]))


@pytest.fixture
def gui(monkeypatch, dataset):
    fake_dpg = RecordingDPG(search.dpg)
    monkeypatch.setattr(search, "dpg", fake_dpg)
    monkeypatch.setattr(app_state, "dataset", dataset, raising=False)
    monkeypatch.setattr(app_state, "update_info_panel", lambda **kwargs: None, raising=False)
    monkeypatch.setattr(app_state, "update_mouse_hover", lambda **kwargs: None, raising=False)
    return env(dpg=fake_dpg, dataset=dataset)


def titles_matching(dataset, query):
    return [dataset.sorted_entries[i].title for i in search.find_matches(dataset, query)]


# --------------------------------------------------------------------------------
# What counts as a match
#
# The help card's "How search works" section (`app.render_help_extras`) states these same rules to users,
# in prose, with worked examples. If a test here has to change, that text is now wrong and nothing will
# say so -- these tests are what makes a rule change visible, so this is where the reminder belongs.

def test_a_fragment_matches_anywhere_in_the_title(dataset):
    # Incremental fragment search: "cat" finds "photocatalytic", which is the point of not tokenizing.
    assert titles_matching(dataset, "cat") == ["Photocatalytic degradation of methanol",
                                               "Catalysis without photons"]


def test_every_fragment_has_to_match(dataset):
    # Fragments are ANDed, in any order -- so adding one narrows rather than widens.
    assert titles_matching(dataset, "cat photo") == ["Photocatalytic degradation of methanol",
                                                     "Catalysis without photons"]
    assert titles_matching(dataset, "cat methanol") == ["Photocatalytic degradation of methanol"]


def test_a_lowercase_stopword_matches_inside_longer_words(dataset):
    """Why the info panel's arrow button strips stopwords before sending a title here as a query.

    Matching is by substring rather than by word, so a short common word matches wherever its letters
    happen to fall — which for a title-shaped query is nearly everywhere, and the highlighter then paints
    those fragments across the panel. Stripping them is the caller's policy; *this* is the property that
    makes it necessary, and it belongs here because it is `find_matches` that could stop being true.

    If this test ever has to change, look at `info_panel._search_or_select_entry`: the stripping there
    exists only for this, and may have become unnecessary or wrong.
    """
    # "an" is inside meth·an·ol, and inside abl·a·tion? No -- but inside "degradation" it is not either.
    # It is the one in "methanol" that matters, and one hit is enough to make the point.
    assert titles_matching(dataset, "an") == ["Photocatalytic degradation of methanol"]

    # An uppercase letter makes a fragment case-sensitive, so the same word typed the way someone
    # deliberately searching for it would type it does not do this. That is the escape hatch, and it is
    # also the control here: without it, this test would pass against a matcher that ignored case
    # entirely and the case rule could be lost without anything failing.
    assert titles_matching(dataset, "An") == []


def test_the_titles_are_matched_in_their_normalized_form(dataset):
    # The query is normalized too, so a title written "CO₂" has to be findable by typing "CO2" -- which
    # is what someone reading the subscript on screen will type.
    assert titles_matching(dataset, "CO2") == ["A study of CO₂ capture"]


def test_a_query_that_matches_nothing_returns_nothing(dataset):
    assert titles_matching(dataset, "superconductivity") == []


def test_an_empty_query_matches_nothing_rather_than_everything(dataset):
    # The distinction is load-bearing: "no search is running" and "every item matched" would look the
    # same in a result set, and the GUI reads that set to decide whether to dim the non-matches.
    assert titles_matching(dataset, "") == []


def test_the_result_is_an_index_array_even_when_nothing_matched(dataset):
    # These are used to slice `sorted_lowdim_data` with, and NumPy types an empty list as float64,
    # which raises when indexed with.
    for query in ("", "superconductivity"):
        matched = search.find_matches(dataset, query)
        assert dataset.sorted_lowdim_data[matched, 0].shape == (0,), \
            f"the result of '{query}' must still be usable as the index array it is"


# --------------------------------------------------------------------------------
# Reporting the answer

def run_search(gui, query):
    gui.dpg.values[SEARCH_FIELD] = query
    search.update()


def test_running_a_search_publishes_the_query_and_its_results(gui):
    # The tooltip and the info panel read these boxes rather than re-running the search.
    run_search(gui, "cat")
    assert unbox(search.search_string_box) == "cat"
    assert len(unbox(search.search_result_data_idxs_box)) == 2


def test_the_matched_datapoints_are_highlighted_on_the_plot(gui):
    run_search(gui, "methanol")
    xs, ys = gui.dpg.values[RESULTS_SERIES]
    assert xs == [0.0] and ys == [0.0], "the one match is the first entry"


def test_the_highlight_is_cleared_when_nothing_matches(gui):
    run_search(gui, "methanol")
    run_search(gui, "superconductivity")
    assert gui.dpg.values[RESULTS_SERIES] == [[], []]


def test_the_header_counts_the_matches(gui):
    run_search(gui, "cat")
    assert gui.dpg.values[HEADER] == "[2 matches]"


def test_the_header_says_match_in_the_singular_for_one(gui):
    # Negative control for the plural: a counter that always said "matches" would satisfy the test above.
    run_search(gui, "methanol")
    assert gui.dpg.values[HEADER] == "[1 match]"


def test_the_header_says_so_when_a_search_found_nothing(gui):
    run_search(gui, "superconductivity")
    assert gui.dpg.values[HEADER] == "[no matches]"


def test_the_header_goes_back_to_its_label_when_the_search_is_cleared(gui):
    # "no matches" and "not searching" are different things to say, and the header is the only place
    # that says either.
    run_search(gui, "superconductivity")
    run_search(gui, "")
    assert gui.dpg.values[HEADER] == "Search"


# --------------------------------------------------------------------------------
# The colour of the search field

def test_the_field_is_plain_while_no_search_is_running(gui):
    run_search(gui, "")
    search.update_field_color()
    assert gui.dpg.values[FIELD_COLOR] == search._COLOR_NO_SEARCH


def test_the_field_goes_green_when_the_query_finds_something(gui):
    run_search(gui, "cat")
    search.update_field_color()
    assert gui.dpg.values[FIELD_COLOR] == search._COLOR_FOUND


def test_the_field_goes_red_when_the_query_finds_nothing(gui):
    # The distinction that makes the colour worth having: a query you are partway through typing looks
    # exactly like one that is wrong, until the field tells you which.
    run_search(gui, "superconductivity")
    search.update_field_color()
    assert gui.dpg.values[FIELD_COLOR] == search._COLOR_NOT_FOUND


# --------------------------------------------------------------------------------
# Who gets told

def test_a_search_asks_the_info_panel_and_the_tooltip_to_refresh(gui, monkeypatch):
    # Both draw per-item match markers, so they are stale the moment the result set changes.
    told = []
    monkeypatch.setattr(app_state, "update_info_panel", lambda **kwargs: told.append(("panel", kwargs)), raising=False)
    monkeypatch.setattr(app_state, "update_mouse_hover", lambda **kwargs: told.append(("tooltip", kwargs)), raising=False)
    run_search(gui, "cat")
    assert [what for what, _ in told] == ["panel", "tooltip"]


def test_the_wait_flag_reaches_both_of_them(gui, monkeypatch):
    # Typing is a burst, and the two renders are the expensive part; waiting is what collapses a burst
    # into one refresh.
    told = []
    monkeypatch.setattr(app_state, "update_info_panel", lambda **kwargs: told.append(kwargs), raising=False)
    monkeypatch.setattr(app_state, "update_mouse_hover", lambda **kwargs: told.append(kwargs), raising=False)
    gui.dpg.values[SEARCH_FIELD] = "cat"
    search.update(wait=False)
    assert all(kwargs["wait"] is False for kwargs in told)
    told.clear()
    search.update(wait=True)
    assert all(kwargs["wait"] is True for kwargs in told)


def test_typing_in_the_field_runs_a_search_that_waits(gui):
    # DPG hands a callback its own three arguments, which is why the field has a callback of its own
    # rather than being wired straight to `update` -- `wait` would otherwise be the sender's widget ID.
    gui.dpg.values[SEARCH_FIELD] = "cat"
    search.search_field_callback(SEARCH_FIELD, "cat", None)
    assert unbox(search.search_string_box) == "cat"

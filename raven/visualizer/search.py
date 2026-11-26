"""Search over the currently loaded dataset, for the Visualizer.

Owns the search state — the query and the indices it matched — the scan that produces them, and the
three GUI elements that report them: the plotter's search-result highlight series, the "Search" header
that doubles as a match counter, and the colour of the search field itself.

A match among records a display filter hides is still a match: it is counted, as hidden, and not drawn. A
grayed one is on screen, so it is drawn; selecting the matches takes only what can be interacted with
(`interactable_matches`).

`find_matches` is the scan, and it touches no GUI: what the Visualizer counts as a match can be stated,
and changed, without one. The rest of this module is what shows the answer.

Public state, published on `app_state` at import so the annotation tooltip and the info panel can read
it without importing this module:

  - `search_string_box`: the current query, empty when no search is active.
  - `search_result_data_idxs_box`: indices into `sorted_xxx` of the entries it matched.

`app_state.update_search` is published the same way, since several places re-run the search after
writing to the search field.
"""

__all__ = ["search_string_box",
           "search_result_data_idxs_box",
           "find_matches",
           "update",
           "interactable_matches",
           "format_match_count",
           "search_field_callback",
           "update_field_color"]

import logging
import math
logger = logging.getLogger(__name__)

import re

import numpy as np

import dearpygui.dearpygui as dpg

from unpythonic import box, unbox

from ..common import utils as common_utils

from .app_state import app_state
from . import displaystate

# --------------------------------------------------------------------------------
# Public state

search_string_box = box("")
search_result_data_idxs_box = box(common_utils.make_blank_index_array())

app_state.search_string_box = search_string_box
app_state.search_result_data_idxs_box = search_result_data_idxs_box

_COLOR_NO_SEARCH = (255, 255, 255)
_COLOR_FOUND = (180, 255, 180)
_COLOR_NOT_FOUND = (255, 128, 128)


# --------------------------------------------------------------------------------
# The scan

def find_matches(dataset, search_string):
    """Return the indices (into `sorted_xxx`) of the entries matching `search_string`.

    `dataset`: the dataset to search.
    `search_string`: the query. Empty means no search, which matches nothing rather than everything —
                     "no search is running" and "every item matched" look the same in a result set, and
                     the GUI needs to tell them apart to know whether to dim anything.

    A query of `#` and a number, or `cluster:` and one, matches that cluster's entries: `"#7"`, `"cluster:7"`. A query
    of `~` and a year, or `year:` and one, matches the entries of that year: `"~2022"`, `"year:2022"`. Either takes a
    range, with a dash, and an open end leaves out its number: `"#7-9"`, `"~2019-2022"`, `"~2019-"`, `"~-2015"`.
    `"#misc"` and `"cluster:misc"` match Misc, which has no number.

    Otherwise, incremental fragment search over item titles: every whitespace-separated fragment must appear
    somewhere in the title, in any order, and a fragment carrying an uppercase letter matches
    case-sensitively. So "cat photo" finds "photocatalytic". See `common_utils.make_search_matcher`.

    The titles are matched in their normalized form (`entry.normalized_title`), which is what the query
    is normalized to as well — otherwise a search for "O2" would fail on a title spelling it "O₂".
    """
    # These rules are also stated to users, in prose, by the help card's "How search works" section
    # (`app.render_help_extras`) — including two worked examples, that "hydrogen" matches "Hydrogen" and
    # that "TiO" matches titanium oxide but not "bastion". Changing what counts as a match means editing
    # that text too. The tests below will notice a changed rule; nothing notices stale prose, which is
    # why the pointer is here rather than only in the help card.
    if not search_string:
        return common_utils.make_blank_index_array()

    # A field takes a sigil to type and a prefix to read, both meaning one thing, and a number or a range of them.
    #
    # "#7" names cluster 7, as its header does: its papers, by the importer's cluster, so that a cluster the purity
    # cutoff shows as Misc is still found by its number.
    def in_range(lead: str, value_of):
        maybe_match = re.fullmatch(rf"\s*(?:{lead})\s*(\d+)?\s*(-)?\s*(\d+)?\s*", search_string, flags=re.IGNORECASE)
        if maybe_match is None:
            return None
        first, dash, last = maybe_match.groups()
        if first is None and last is None:  # the lead alone, while typing, is nothing to look for yet
            return None
        low = int(first) if first is not None else -math.inf
        high = int(last) if last is not None else (math.inf if dash else low)
        return np.array([data_idx for data_idx, entry in enumerate(dataset.sorted_entries)
                         if (value := value_of(entry)) is not None and low <= value <= high],
                        dtype=np.int64)

    # Misc (-1) is not one of the clusters' numbers, so no number or range finds it; its name does.
    if re.fullmatch(r"\s*(?:#|cluster:)\s*misc\s*", search_string, flags=re.IGNORECASE):
        return np.array([data_idx for data_idx, entry in enumerate(dataset.sorted_entries) if entry.cluster_id == -1],
                        dtype=np.int64)
    for lead, value_of in ((r"#|cluster:", lambda entry: entry.cluster_id if entry.cluster_id >= 0 else None),
                           (r"~|year:", displaystate.year_of)):
        maybe_matches = in_range(lead, value_of)
        if maybe_matches is not None:
            return maybe_matches

    # A plain O(n) scan. No stopwording, lemmatization or anything fancy.
    # TODO: Search also in document authors (full author list). For this, need to update the GUI wherever we show author names - e.g. searching for "Virtanen" in a paper "Aaltonen et al." that has 200 authors.
    # TODO: With `raven.librarian.hybridir.HybridIR`, we could integrate also a semi-intelligent (keyword + semantic) fulltext search here. Think about the GUI, as the classic mode is useful too.
    matches_search = common_utils.make_search_matcher(search_string)
    data_idxs = [data_idx for data_idx, entry in enumerate(dataset.sorted_entries)  # `data_idx`: index to `sorted_xxx`
                 if matches_search(entry.normalized_title)]
    # The dtype is explicit because these are used to slice with, and NumPy types an empty list as
    # float64 — which a search that matched nothing produces, and which raises when indexed with.
    return np.array(data_idxs, dtype=np.int64)


# --------------------------------------------------------------------------------
# Running a search, and reporting it

def update(wait=True):
    """Run the search for whatever the search field currently holds, and update the GUI.

    Called automatically when the search field changes, and explicitly by the places that write to the
    field themselves (the info panel's search-for-this-item action, and the hotkey that clears it).

    `wait`: whether to wait for more keyboard input before starting the info panel and tooltip renders,
            which are the expensive part. Passed on to both.
    """
    search_string = dpg.get_value("search_field")  # tag
    search_result_data_idxs = find_matches(app_state.dataset, search_string)

    # Send the new data into the boxes
    search_string_box << search_string
    search_result_data_idxs_box << search_result_data_idxs

    if len(search_result_data_idxs):
        # Highlight the search result data points (by plotting them as another series on top): those on screen.
        drawn = displaystate.on_screen(app_state.dataset.sorted_entries, search_result_data_idxs)
        dpg.set_value("my_search_results_scatter_series", [list(app_state.dataset.sorted_lowdim_data[drawn, 0]),  # tag
                                                           list(app_state.dataset.sorted_lowdim_data[drawn, 1])])
        # Re-use the "Search" header to show the number of matches.
        dpg.set_value("search_header_text", format_match_count(len(search_result_data_idxs),  # tag
                                                               len(search_result_data_idxs) - len(drawn)))
    else:
        dpg.set_value("my_search_results_scatter_series", [[], []])  # tag
        if not search_string:  # Search not active, restore the "Search" header
            dpg.set_value("search_header_text", "Search")  # tag  # TODO: DRY duplicate definitions for labels
        else:  # Search active, but no matches
            dpg.set_value("search_header_text", format_match_count(0, 0))  # tag

    # Update tooltip and info panel to update the highlight status.
    # TODO: Currently, this may cause the set of data points considered to be under the mouse cursor to change
    #       the first time this happens at a given mouse position (upon a click in the plot area). Debug this.
    #       If the plot mouse position is one frame out of date (update order?), that would explain it.
    app_state.update_info_panel(wait=wait)
    app_state.update_mouse_hover(force=True, wait=wait)


def interactable_matches() -> np.ndarray:
    """The current matches that can be hovered and selected (`displaystate.interactable`): what selecting the
    search results selects."""
    matches = unbox(search_result_data_idxs_box)
    if app_state.dataset is None:
        return common_utils.make_blank_index_array()
    return displaystate.interactable(app_state.dataset.sorted_entries, matches,
                                     grayed_selectable=app_state.grayed_selectable)


def format_match_count(n_matches: int, n_hidden: int) -> str:
    """The "Search" header's text for a search with `n_matches` matches, `n_hidden` of them hidden by a display
    filter: `"[7 matches, 3 hidden]"`, or `"[no matches]"`."""
    if not n_matches:
        return "[no matches]"
    text = f"[{n_matches} {'match' if n_matches == 1 else 'matches'}"  # plural
    return text + (f", {n_hidden} hidden]" if n_hidden else "]")


def search_field_callback(sender, app_data, user_data):
    """DPG callback for the search field. Runs the search as the user types."""
    update(wait=True)  # more keystrokes are the likely next event, so let the expensive renders wait for them


def update_field_color():
    """Colour the search field by whether the current query found anything. Called per frame."""
    search_string = unbox(search_string_box)
    search_result_data_idxs = unbox(search_result_data_idxs_box)
    if not search_string:
        color = _COLOR_NO_SEARCH  # no search active
    elif len(search_result_data_idxs):
        color = _COLOR_FOUND  # found, green
    else:
        color = _COLOR_NOT_FOUND  # not found, red
    dpg.set_value("search_field_text_color", color)  # tag


# Register on `app_state` so submodules (e.g. `info_panel`) can re-run the search without importing us.
app_state.update_search = update

"""Tests for the search functionality."""

import pytest

from ..search import SearchState
from ..parser import parse_xdot


# Test graph with searchable text.
# Chosen so that "cat" matches two nodes (substring match),
# and "cat photo" matches exactly one (AND of fragments).
SEARCH_TEST_XDOT = """
digraph G {
    graph [bb="0,0,300,200"];
    photocatalytic [pos="50,150", width="1.5", height="0.5",
       _draw_="c 7 -#000000 e 50 150 54 18 ",
       _ldraw_="F 14 11 -Times-Roman c 7 -#000000 T 50 150 0 100 13 -photocatalytic "];
    reaction [pos="150,150", width="1.0", height="0.5",
       _draw_="c 7 -#000000 e 150 150 36 18 ",
       _ldraw_="F 14 11 -Times-Roman c 7 -#000000 T 150 150 0 60 8 -reaction "];
    catalyst [pos="250,150", width="1.0", height="0.5",
       _draw_="c 7 -#000000 e 250 150 36 18 ",
       _ldraw_="F 14 11 -Times-Roman c 7 -#000000 T 250 150 0 60 8 -catalyst "];
    other [pos="100,50", width="0.75", height="0.5",
       _draw_="c 7 -#000000 e 100 50 27 18 ",
       _ldraw_="F 14 11 -Times-Roman c 7 -#000000 T 100 50 0 40 5 -other "];
}
"""


class TestSearchState:
    """Test the SearchState class."""

    @pytest.fixture
    def search_state(self):
        """Create a SearchState with a test graph."""
        graph = parse_xdot(SEARCH_TEST_XDOT)
        state = SearchState()
        state.set_graph(graph)
        return state

    # -------------------------------------------------------------------
    # Basic search
    # -------------------------------------------------------------------

    def test_empty_search(self, search_state):
        """Empty search returns no results."""
        results = search_state.search("")
        assert len(results) == 0

    def test_single_fragment(self, search_state):
        """Single fragment matches nodes containing it."""
        results = search_state.search("catalyst")  # noqa: F841: this updates the search state
        ids = search_state.get_result_ids()
        assert "catalyst" in ids

    def test_search_returns_results(self, search_state):
        """search() returns the result list directly."""
        results = search_state.search("catalyst")
        assert len(results) >= 1
        # Return value should be the same as get_results()
        assert len(results) == len(search_state.get_results())

    def test_fragment_matches_substring(self, search_state):
        """Fragment search matches substrings: 'cat' matches both
        'catalyst' and 'photocatalytic'."""
        search_state.search("cat")
        ids = search_state.get_result_ids()
        assert "catalyst" in ids
        assert "photocatalytic" in ids

    def test_multi_fragment_and(self, search_state):
        """Multiple fragments are ANDed: 'cat photo' matches only
        'photocatalytic' (contains both), not 'catalyst' (missing 'photo').

        Fragment order doesn't matter — 'cat photo' and 'photo cat'
        are equivalent.
        """
        search_state.search("cat photo")
        ids = search_state.get_result_ids()
        assert "photocatalytic" in ids
        assert "catalyst" not in ids

    def test_multi_fragment_order_independent(self, search_state):
        """Fragment order doesn't affect results."""
        search_state.search("cat photo")
        ids_1 = set(search_state.get_result_ids())
        search_state.search("photo cat")
        ids_2 = set(search_state.get_result_ids())
        assert ids_1 == ids_2

    def test_multi_fragment_all_present(self, search_state):
        """Every result contains all search fragments."""
        search_state.search("cat photo")
        for result in search_state.get_results():
            texts = " ".join(result.get_texts()).lower()
            assert "cat" in texts
            assert "photo" in texts

    def test_no_match(self, search_state):
        """Search for nonexistent text returns no results."""
        results = search_state.search("xyznonexistent")
        assert len(results) == 0

    # -------------------------------------------------------------------
    # Smart-case (Emacs HELM style)
    #
    # Lowercase fragment -> case-insensitive.
    # Fragment with any uppercase -> case-sensitive.
    # Each fragment is checked independently.
    # -------------------------------------------------------------------

    def test_case_insensitive_lowercase_query(self, search_state):
        """Lowercase query matches case-insensitively."""
        results = search_state.search("catalyst")  # noqa: F841: this updates the search state
        ids = search_state.get_result_ids()
        assert "catalyst" in ids

    def test_case_sensitive_uppercase_query(self, search_state):
        """Query with uppercase letter is case-sensitive.

        All node labels in SEARCH_TEST_XDOT are lowercase, so an
        uppercase query should not match.
        """
        results = search_state.search("Catalyst")
        assert len(results) == 0

    def test_case_sensitive_mixed_case(self, search_state):
        """Mixed-case query is case-sensitive."""
        results = search_state.search("CaTaLySt")
        assert len(results) == 0

    def test_smart_case_per_fragment(self, search_state):
        """Smart-case is applied per fragment independently.

        'cat Photo' -> 'cat' is case-insensitive (lowercase),
        'Photo' is case-sensitive (has uppercase).
        Since the label is 'photocatalytic' (all lowercase),
        'Photo' won't match, so no results.
        """
        results = search_state.search("cat Photo")
        assert len(results) == 0

    # -------------------------------------------------------------------
    # Navigation
    # -------------------------------------------------------------------

    def test_next_match_returns_element(self, search_state):
        """next_match returns an Element."""
        search_state.search("cat")
        element = search_state.next_match()
        assert element is not None

    def test_next_wraparound(self, search_state):
        """next_match wraps around to the beginning after the last result."""
        search_state.search("cat")
        count = search_state.get_result_count()
        assert count >= 2

        # Navigate past the end
        for _ in range(count):
            search_state.next_match()
        # Now at last result; one more should wrap to index 0
        search_state.next_match()
        assert search_state.get_current_index() == 0

    def test_prev_wraparound(self, search_state):
        """prev_match wraps to the last result from the beginning."""
        search_state.search("cat")
        count = search_state.get_result_count()

        # prev from initial state (index -1) should wrap to last
        search_state.prev_match()
        assert search_state.get_current_index() == count - 1

    def test_next_no_results(self, search_state):
        """next_match returns None when there are no results."""
        search_state.search("xyznonexistent")
        assert search_state.next_match() is None

    def test_prev_no_results(self, search_state):
        """prev_match returns None when there are no results."""
        search_state.search("xyznonexistent")
        assert search_state.prev_match() is None

    def test_navigation_single_result(self, search_state):
        """Wraparound works with exactly one match."""
        search_state.search("reaction")
        assert search_state.get_result_count() == 1

        e1 = search_state.next_match()
        e2 = search_state.next_match()
        # Should wrap back to the same (only) element
        assert e1 is e2

    # -------------------------------------------------------------------
    # State management
    # -------------------------------------------------------------------

    def test_clear(self, search_state):
        """clear() resets results and query."""
        search_state.search("cat")
        assert search_state.has_results()

        search_state.clear()
        assert not search_state.has_results()
        assert search_state.get_query() == ""

    def test_get_result_ids(self, search_state):
        """get_result_ids returns string node IDs."""
        search_state.search("catalyst")
        ids = search_state.get_result_ids()
        assert len(ids) >= 1
        assert all(isinstance(id_, str) for id_ in ids)

    def test_set_new_graph_clears_results(self, search_state):
        """Setting a new graph clears results but retains the query."""
        search_state.search("cat")
        assert search_state.has_results()
        query = search_state.get_query()

        new_graph = parse_xdot('digraph G { graph [bb="0,0,10,10"]; }')
        search_state.set_graph(new_graph)

        assert not search_state.has_results()
        assert search_state.get_query() == query

    def test_set_new_graph_reruns_search(self, search_state):
        """Setting a new graph with an active query re-runs the search."""
        search_state.search("other")
        assert search_state.has_results()

        # Set a graph that has no "other" node
        new_graph = parse_xdot('digraph G { graph [bb="0,0,10,10"]; }')
        search_state.set_graph(new_graph)
        assert not search_state.has_results()

        # Set back the original graph — search should re-run and find "other"
        original_graph = parse_xdot(SEARCH_TEST_XDOT)
        search_state.set_graph(original_graph)
        assert search_state.has_results()
        assert "other" in search_state.get_result_ids()

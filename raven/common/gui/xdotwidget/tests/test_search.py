"""Tests for the search functionality."""

import pytest

from ..search import SearchState
from ..parser import parse_xdot


# Test graph with searchable text
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

    def test_empty_search(self, search_state):
        """Test that empty search returns no results."""
        results = search_state.search("")
        assert len(results) == 0

    def test_single_fragment_search(self, search_state):
        """Test searching with a single fragment."""
        results = search_state.search("catalyst")
        assert len(results) >= 1

    def test_fragment_search_matches_substring(self, search_state):
        """Test that fragment search matches substrings."""
        # "cat" should match "catalyst" and "photocatalytic"
        results = search_state.search("cat")
        assert len(results) >= 2

    def test_multi_fragment_search(self, search_state):
        """Test searching with multiple fragments (AND)."""
        # "photo cat" should match "photocatalytic" (contains both)
        results = search_state.search("photo cat")
        assert len(results) >= 1

        # All results should contain both fragments
        for result in results:
            texts = " ".join(result.get_texts()).lower()
            assert "photo" in texts
            assert "cat" in texts

    def test_case_insensitive(self, search_state):
        """Test that search is case-insensitive."""
        results_lower = search_state.search("catalyst")
        results_upper = search_state.search("CATALYST")
        results_mixed = search_state.search("CaTaLySt")

        assert len(results_lower) == len(results_upper)
        assert len(results_lower) == len(results_mixed)

    def test_no_match(self, search_state):
        """Test search with no matches."""
        results = search_state.search("xyznonexistent")
        assert len(results) == 0

    def test_navigation_wraparound(self, search_state):
        """Test that next/prev wrap around."""
        search_state.search("cat")  # Should have multiple results

        # Navigate to end
        count = search_state.get_result_count()
        for _ in range(count + 1):
            search_state.next_match()

        # Should have wrapped to beginning
        assert search_state.get_current_index() == 0

    def test_prev_navigation(self, search_state):
        """Test previous navigation."""
        search_state.search("cat")

        # Going prev from start should wrap to end
        search_state.prev_match()
        assert search_state.get_current_index() == search_state.get_result_count() - 1

    def test_clear(self, search_state):
        """Test clearing search state."""
        search_state.search("cat")
        assert search_state.has_results()

        search_state.clear()
        assert not search_state.has_results()
        assert search_state.get_query() == ""

    def test_get_result_ids(self, search_state):
        """Test getting node IDs from results."""
        search_state.search("catalyst")
        ids = search_state.get_result_ids()

        # Should have node IDs
        assert len(ids) >= 1
        assert all(isinstance(id_, str) for id_ in ids)

    def test_set_new_graph(self, search_state):
        """Test setting a new graph clears results."""
        search_state.search("cat")
        assert search_state.has_results()

        # Set a new graph
        new_graph = parse_xdot('digraph G { graph [bb="0,0,10,10"]; }')
        search_state.set_graph(new_graph)

        # Results should be cleared but query retained
        assert not search_state.has_results()

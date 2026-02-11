"""Search functionality for graph elements.

This module provides fragment-based search through graph nodes and edges,
similar to Emacs HELM's swoop search.
"""

__all__ = ["SearchState"]

from typing import List, Optional

from .graph import Graph, Element, Node


class SearchState:
    """Manages search state for a graph.

    Supports fragment search where "cat photo" matches "photocatalytic"
    (all fragments must be present, in any order).
    """

    def __init__(self):
        self._query: str = ""
        self._results: List[Element] = []
        self._current_index: int = -1
        self._graph: Optional[Graph] = None

    def set_graph(self, graph: Graph) -> None:
        """Set the graph to search in."""
        self._graph = graph
        self._results = []
        self._current_index = -1
        if self._query:
            self._do_search()

    def search(self, query: str) -> List[Element]:
        """Perform a search and return matching elements.

        `query`: Search string. Multiple space-separated words are treated
                 as fragments that must all be present (AND search).
                 Search is case-insensitive.

        Returns a list of matching Elements (Nodes and Edges).
        """
        self._query = query.strip()
        self._current_index = -1
        self._do_search()
        return self._results

    def _do_search(self) -> None:
        """Execute the current search query."""
        if not self._query or self._graph is None:
            self._results = []
            return

        self._results = self._graph.filter_items_by_text(self._query)

    def get_results(self) -> List[Element]:
        """Return the current search results."""
        return list(self._results)

    def get_result_count(self) -> int:
        """Return the number of search results."""
        return len(self._results)

    def get_current_index(self) -> int:
        """Return the current result index (-1 if no navigation yet)."""
        return self._current_index

    def get_current(self) -> Optional[Element]:
        """Return the currently selected search result, or None."""
        if not self._results or self._current_index < 0:
            return None
        if self._current_index >= len(self._results):
            return None
        return self._results[self._current_index]

    def next_match(self) -> Optional[Element]:
        """Navigate to the next match, with wraparound.

        Returns the next Element, or None if no results.
        """
        if not self._results:
            return None

        self._current_index += 1
        if self._current_index >= len(self._results):
            self._current_index = 0

        return self._results[self._current_index]

    def prev_match(self) -> Optional[Element]:
        """Navigate to the previous match, with wraparound.

        Returns the previous Element, or None if no results.
        """
        if not self._results:
            return None

        self._current_index -= 1
        if self._current_index < 0:
            self._current_index = len(self._results) - 1

        return self._results[self._current_index]

    def clear(self) -> None:
        """Clear the search state."""
        self._query = ""
        self._results = []
        self._current_index = -1

    def get_query(self) -> str:
        """Return the current search query."""
        return self._query

    def has_results(self) -> bool:
        """Return True if there are search results."""
        return len(self._results) > 0

    def get_result_ids(self) -> List[str]:
        """Return a list of node IDs from the search results.

        Only includes Nodes that have an internal_name.
        """
        return [
            e.internal_name
            for e in self._results
            if isinstance(e, Node) and e.internal_name
        ]

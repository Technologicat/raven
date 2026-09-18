"""Finding text in a chat: which messages of a branch match a search, and where in them.

A search is the constellation's incremental fragment search (`raven.common.utils.make_search_matcher`): every
whitespace-separated fragment must occur, in any order, and a fragment carrying an uppercase letter matches
case-sensitively. The unit is the message — a message matches when all the fragments occur in one of its
searchable texts — while the highlighting marks every occurrence of every fragment.

Pure: no DearPyGui, so what counts as a match is testable without a GUI.
"""

__all__ = ["SearchQuery", "MatchCounts", "make_query", "find_matches"]

import logging
logger = logging.getLogger(__name__)

from dataclasses import dataclass
from typing import Callable

from ..common import utils as common_utils

from . import chattree
from . import chatutil


@dataclass(frozen=True)
class SearchQuery:
    """A compiled search. Make one with `make_query`.

    `search_string`: as typed.
    `include_thinking`: whether thinking traces are searched.
    `include_tools`: whether tool messages — the results a tool call returned — are searched.
    `highlight`: the `(maybe_regex_case_sensitive, maybe_regex_case_insensitive)` pair for the renderer's
                 `highlight=`; see `raven.common.utils.compile_search_highlight_regexes`.
    """
    search_string: str
    include_thinking: bool
    include_tools: bool
    highlight: tuple
    matches: Callable[[str], bool]


@dataclass(frozen=True)
class MatchCounts:
    """How many times a query's fragments occur in one matching message, by which of its texts they are in.

    Occurrences rather than a yes-or-no, because a box in the chat graph reports how much is in it — where
    the search row's `[x/y]` counts messages, those being what stepping through the matches moves between.

    `content`: occurrences in the message text, or zero if the query did not match there.
    `thinking`: the same for its thinking trace. Zero whenever traces are not being searched.

    Counted only in a text the query actually *matched*, which is what makes `content` and `thinking`
    answer "did it match here" as well as "how many". A single fragment of a multi-fragment query can
    occur in a text that does not match — the renderer colours it, since it highlights fragment by
    fragment — and counting that as a hit would say a message has more in it than the reader can find.
    """
    content: int
    thinking: int

    def _get_total(self) -> int:
        """Return the number of occurrences anywhere in the message."""
        return self.content + self.thinking

    total = property(fget=_get_total,
                     doc="How many occurrences the message holds in all, which is what a box shows.")


def make_query(search_string: str, *, include_thinking: bool = False, include_tools: bool = True) -> SearchQuery | None:
    """Compile `search_string`. Returns `None` for an empty or blank one, which is no search running.

    The defaults are the search row's: thinking traces left out, tool messages searched.
    """
    if not search_string.strip():
        return None
    return SearchQuery(search_string=search_string,
                       include_thinking=include_thinking,
                       include_tools=include_tools,
                       highlight=common_utils.compile_search_highlight_regexes(search_string),
                       matches=common_utils.make_search_matcher(search_string))


def find_matches(datastore: chattree.Forest, node_ids: list[str], query: SearchQuery | None) -> list[tuple[str, MatchCounts]]:
    """Return the messages among `node_ids` that match `query`, in the order given, as `(node_id, counts)`.

    `counts` is a `MatchCounts`, which says both where the message matched and how many occurrences are
    there. A caller showing the match needs the *where*, a trace being collapsed by default and wanting
    opening to show one; a caller summarizing a part of the tree needs the *how many*.

    `query=None` (no search running) matches nothing, so "no search" and "nothing matched" look the same here;
    the caller knows which it asked.
    """
    if query is None:
        return []
    result = []
    for node_id in node_ids:
        maybe_counts = _counts_for(datastore.get_payload(node_id), query)
        if maybe_counts is not None:
            result.append((node_id, maybe_counts))
    return result


def _counts_for(payload: dict, query: SearchQuery) -> MatchCounts | None:
    """Return one message's `MatchCounts`, or `None` if it does not match at all."""
    message = payload["message"]
    if message["role"] == "tool" and not query.include_tools:
        return None
    # Matched in the normalized form, which is what the query is normalized to — otherwise "O2" would not
    # find "O₂" — and counted in that same form, so a count cannot disagree with the match beside it.
    content = common_utils.normalize_search_string(chatutil.content_to_text(message["content"]))
    n_content = _occurrences(content, query) if query.matches(content) else 0

    n_thinking = 0
    if query.include_thinking:
        reasoning = message.get("reasoning_content") or ""
        if reasoning:
            reasoning = common_utils.normalize_search_string(reasoning)
            if query.matches(reasoning):
                n_thinking = _occurrences(reasoning, query)

    if not (n_content or n_thinking):
        return None
    return MatchCounts(content=n_content, thinking=n_thinking)


def _occurrences(text: str, query: SearchQuery) -> int:
    """Return how many fragment occurrences `query` marks in `text` — the hits a renderer would paint."""
    # The two regexes hold disjoint fragments, the case rule putting each fragment in one or the other, so
    # a fragment cannot be counted twice. Two *different* fragments overlapping on one span can each count,
    # which is also what the renderer does with them.
    return sum(len(regex.findall(text)) for regex in query.highlight if regex is not None)

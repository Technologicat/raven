"""Finding text in a chat: which messages of a branch match a search, and where in them.

A search is the constellation's incremental fragment search (`raven.common.utils.make_search_matcher`): every
whitespace-separated fragment must occur, in any order, and a fragment carrying an uppercase letter matches
case-sensitively. The unit is the message — a message matches when all the fragments occur in one of its
searchable texts — while the highlighting marks every occurrence of every fragment.

Pure: no DearPyGui, so what counts as a match is testable without a GUI.
"""

__all__ = ["SearchQuery", "make_query", "find_matches"]

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


def make_query(search_string: str, *, include_thinking: bool = True, include_tools: bool = True) -> SearchQuery | None:
    """Compile `search_string`. Returns `None` for an empty or blank one, which is no search running."""
    if not search_string.strip():
        return None
    return SearchQuery(search_string=search_string,
                       include_thinking=include_thinking,
                       include_tools=include_tools,
                       highlight=common_utils.compile_search_highlight_regexes(search_string),
                       matches=common_utils.make_search_matcher(search_string))


def find_matches(datastore: chattree.Forest, node_ids: list[str], query: SearchQuery | None) -> list[tuple[str, str]]:
    """Return the messages among `node_ids` that match `query`, in the order given, as `(node_id, where)`.

    `where` is `"content"` when the message text matches, else `"thinking"` when only its thinking trace does —
    which a caller showing the match needs to know, a trace being collapsed by default.

    `query=None` (no search running) matches nothing, so "no search" and "nothing matched" look the same here;
    the caller knows which it asked.
    """
    if query is None:
        return []
    result = []
    for node_id in node_ids:
        maybe_where = _where_it_matches(datastore.get_payload(node_id), query)
        if maybe_where is not None:
            result.append((node_id, maybe_where))
    return result


def _where_it_matches(payload: dict, query: SearchQuery) -> str | None:
    message = payload["message"]
    if message["role"] == "tool" and not query.include_tools:
        return None
    # Matched in the normalized form, which is what the query is normalized to — otherwise "O2" would not find "O₂".
    if query.matches(common_utils.normalize_search_string(chatutil.content_to_text(message["content"]))):
        return "content"
    if query.include_thinking:
        reasoning = message.get("reasoning_content") or ""
        if reasoning and query.matches(common_utils.normalize_search_string(reasoning)):
            return "thinking"
    return None

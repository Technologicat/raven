"""Shared rendering vocabulary for the annotation tooltip and the info panel.

Extracted from `app.py` (2026-04-27) as the sixth and final step of the
refactoring plan in `briefs/visualizer-refactoring.md`. The annotation tooltip
(`annotation.py`) and the item information panel (`info_panel.py`) share the
data-gathering and search-highlighting layer; this module collects what they
have in common.

The two consumers diverge on *what* they render per item — the tooltip shows
compact icon-decorated titles, the info panel shows full per-item button rows
with abstracts and produces a clipboard report — so this module deliberately
stops at the data layer. It does not own DPG widgets.

Public API:

  - `get_entries_for_selection(data_idxs, *, sort_field, max_n)` — gather items
    by cluster, sorted alphabetically within each cluster. Returns
    `(entries_by_cluster, format_cluster_annotation)` where the formatter is a
    closure that yields `(cluster_title, cluster_keywords, cluster_content,
    more)` for a given cluster ID.
  - `order_cluster_ids(cluster_ids)` — sort ascending with the misc group
    (cluster `-1`) moved to the end.
  - `compile_search_highlight_regexes(search_string)` — split the search string
    into case-sensitive and case-insensitive fragment groups, compile each
    group into a single alternation regex. Returns
    `(maybe_regex_case_sensitive, maybe_regex_case_insensitive)`: each entry
    is either a compiled `re.Pattern` (truthy) ready for `re.sub`, or
    `None` (falsy) if there are no fragments of that kind. The empty/no-search
    case returns `(None, None)`. Caller can check with `if maybe_regex_case_xxx:`
    — "do we have a highlighter for this case?" — this is equivalent to an
    "are there fragments?" test, since the regex is built iff fragments exist.

Cross-module state read via `app_state`:
  `dataset` (for `sorted_entries`, `file_content.keywords_available`, and
  `file_content.vis_keywords_by_cluster`).
"""

__all__ = ["get_entries_for_selection",
           "order_cluster_ids",
           "compile_search_highlight_regexes"]

import collections
import math
import re

from ..common import utils as common_utils

from .app_state import app_state


def get_entries_for_selection(data_idxs, *, sort_field="title", max_n=None):
    """Gather item data for visualization, sorting by cluster.

    `data_idxs`: `list`, the selection of items to include in the report. Item indices into `sorted_xxx`.
    `sort_field`: `str`, the field to sort by within each cluster. The name of one of the attributes of an entry in `sorted_entries`.
    `max_n`: `int`, how many entries can be displayed reasonably. Default `None` means no limit.

    Return value is... complicated, see `annotation._render_worker` and `info_panel._update_info_panel` for usage examples.
    """

    # Gather the relevant entries from the vis data.
    entries_by_cluster = collections.defaultdict(lambda: list())
    for data_idx in data_idxs:  # item indices into `sorted_xxx`
        entry = app_state.dataset.sorted_entries[data_idx]
        entries_by_cluster[entry.cluster_id].append((data_idx, entry))

    # Alphabetize by `sort_field` (e.g. `title`) within each cluster, much faster to glance at.
    for entries_in_this_cluster in entries_by_cluster.values():
        entries_in_this_cluster.sort(key=lambda e: getattr(e[1], sort_field).strip().lower())  # e: `(data_idx, entry)`

    # If `max_n` is enabled, determine how many entries we can display from each cluster to approximately match the total count.
    # But display at least one entry from each cluster.
    if max_n is not None:
        n_clusters_in_selection = len(entries_by_cluster)
        if n_clusters_in_selection > 0:
            max_entries_per_cluster = math.ceil(max_n / n_clusters_in_selection)
        else:
            max_n = None

    def format_cluster_annotation(cluster_id):
        # The metadata for the cluster.
        if cluster_id != -1:  # the outlier set doesn't have a set of common keywords computed
            if app_state.dataset.file_content.keywords_available:
                cluster_title = f"#{cluster_id}"
                cluster_keywords = f"[{', '.join(app_state.dataset.file_content.vis_keywords_by_cluster[cluster_id])}]\n"
            else:
                cluster_title = f"#{cluster_id}"
                cluster_keywords = ""
        else:
            cluster_title = "Misc"
            cluster_keywords = ""

        # The entries themselves. Leave only the first few if there are too many to display.
        entries = entries_by_cluster[cluster_id]
        if max_n is not None:
            # TODO: How to compact this in the worst case? Many clusters, with 3 data points in each -> will render 3 * n_clusters entries.
            n_extra_entries = len(entries) - max_entries_per_cluster
            more = ""
            if n_extra_entries > 0:
                if n_extra_entries < 3:  # less pedantic to avoid cutting if there are just 1 or 2 more entries than the limit would allow
                    pass
                else:  # >=3 extra entries, cut at the original limit
                    entries = entries[:max_entries_per_cluster]
                    more = f"[...{n_extra_entries} more entries in {cluster_title}...]"
        cluster_content = entries

        return cluster_title, cluster_keywords, cluster_content, more

    return entries_by_cluster, format_cluster_annotation


def order_cluster_ids(cluster_ids):
    """Sort `cluster_ids` ascending, with the misc group (`-1`) moved to the end.

    Returns a new list. Duplicate IDs are removed.
    """
    out = list(sorted(set(cluster_ids)))
    if out and out[0] == -1:  # move the misc group (if any) to the end
        out = out[1:] + [-1]
    return out


def compile_search_highlight_regexes(search_string):
    """Compile alternation regexes for highlighting search-match fragments inside item titles.

    Same approach as SillyTavern-Timelines: sort fragments so the longest matches first
    (prefers longest match when fragments share substrings, e.g. "laser las").

    All fragments must match simultaneously, to avoid e.g. "col" matching the "<font color=...>"
    inserted by the highlighter when it first highlights "col".

    Returns `(maybe_regex_case_sensitive, maybe_regex_case_insensitive)`. Each entry is either a compiled
    `re.Pattern` (truthy) ready to feed into `re.sub`, or `None` (falsy) if no fragments of
    that kind exist — including when `search_string` is empty, in which case both are `None`.
    """
    if not search_string:
        return None, None

    case_sensitive_fragments, case_insensitive_fragments = common_utils.search_string_to_fragments(search_string, sort=True)

    case_sensitive_fragments = [common_utils.search_fragment_to_highlight_regex_fragment(x) for x in case_sensitive_fragments]
    case_insensitive_fragments = [common_utils.search_fragment_to_highlight_regex_fragment(x) for x in case_insensitive_fragments]

    maybe_regex_case_sensitive = None
    maybe_regex_case_insensitive = None
    if case_sensitive_fragments:
        maybe_regex_case_sensitive = re.compile(f"({'|'.join(case_sensitive_fragments)})")
    if case_insensitive_fragments:
        maybe_regex_case_insensitive = re.compile(f"({'|'.join(case_insensitive_fragments)})", re.IGNORECASE)

    return maybe_regex_case_sensitive, maybe_regex_case_insensitive

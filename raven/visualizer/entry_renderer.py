"""Shared rendering vocabulary for the annotation tooltip and the info panel.

The annotation tooltip (`annotation.py`) and the item information panel
(`info_panel.py`) share the data-gathering layer; this module collects what
they have in common. The search highlighting they also share is not
Visualizer-specific, and lives in `raven.common.utils`.

The two consumers diverge on *what* they render per item — the tooltip shows
compact icon-decorated titles, the info panel shows full per-item button rows
with abstracts and produces a clipboard report — so this module deliberately
stops at the data layer. It does not own DPG widgets.

Public API:

  - `get_entries_for_selection(data_idxs, *, sort_key, max_n)` — gather items
    by cluster, sorted within each cluster (`reference_order` by default). Returns
    `(entries_by_cluster, format_cluster_annotation)` where the formatter is a
    closure that yields `(cluster_title, cluster_keywords, cluster_content,
    more)` for a given cluster ID, `cluster_keywords` as `format_cluster_keywords`'s
    lines (`[]` where there are none); `themes_only=True` passes through to it.
  - `reference_order(entry)` — the sort key of a reference list: authors (or
    the title, when there are none), then year, then title.
  - `order_cluster_ids(cluster_ids)` — sort ascending with the misc group
    (cluster `-1`) moved to the end.
  - `format_dataset_purity(file_content)`, `format_cluster_purity(file_content,
    cluster_id)` — the importer's cluster purity, in words.
  - `format_cluster_keywords(file_content, cluster_id)` — a cluster's keywords as
    shown, as `(caption, text)` lines: its themes and terms where the dataset has both,
    else its label; `format_keyword_lines` puts them on one line of text.

Cross-module state read via `app_state`:
  `dataset` (for `sorted_entries`, `file_content.keywords_available`, and
  `file_content.vis_keywords_by_cluster`).
"""

__all__ = ["reference_order", "get_entries_for_selection",
           "format_selection_heading", "order_cluster_ids",
           "PURITY_EXPLANATION", "format_dataset_purity", "format_cluster_purity",
           "format_cluster_keywords", "format_keyword_lines"]

import collections
import math

from ..common import keyphrases
from ..common import utils as common_utils

from . import config as visualizer_config

from .app_state import app_state


def reference_order(entry) -> tuple:
    """The sort key of a reference list for an entry of `sorted_entries`: authors, then year, then title.

    A record with no authors sorts by its title in their place, as a reference list puts it, and by its
    abstract where it has no title either. Within each criterion a blank sorts after every non-blank value,
    so an undated work follows the dated ones by the same authors. Ties fall to the title, then the
    abstract, and last to `data_idx`, so the order is fully determined however much of a record is blank.

    Ignores case, diacritics and surrounding whitespace, so `Sá` sorts before `Sipos`. A field the entry
    lacks counts as blank.
    """
    def text(field):
        return common_utils.strip_diacritics(str(getattr(entry, field, ""))).strip().casefold()

    def blank_last(value):
        return (value == "", value)

    author, year, title, abstract = text("author"), text("year"), text("title"), text("abstract")
    sort_as = author or title or abstract
    return (blank_last(sort_as), blank_last(year), blank_last(title), blank_last(abstract),
            getattr(entry, "data_idx", 0))


def get_entries_for_selection(data_idxs, *, sort_key=reference_order, max_n=None, dataset=None):
    """Gather item data for visualization, sorting by cluster.

    `data_idxs`: `list`, the selection of items to include in the report. Item indices into `sorted_xxx`.
    `sort_key`: the order within each cluster, as a key function on an entry of `sorted_entries`.
    `max_n`: `int`, how many entries can be displayed reasonably. Default `None` means no limit.
    `dataset`: the dataset to read entries from. Defaults to the live `app_state.dataset`. Background workers pass an
               explicitly captured snapshot, so that a concurrent `open_file` swap can't make `data_idxs` (computed
               against one dataset) index into a different dataset's `sorted_entries`.

    Return value is... complicated, see `annotation._render_worker` and `info_panel._update_info_panel` for usage examples.
    """
    dataset = dataset if dataset is not None else app_state.dataset

    # Gather the relevant entries from the vis data.
    entries_by_cluster = collections.defaultdict(lambda: list())
    for data_idx in data_idxs:  # item indices into `sorted_xxx`
        entry = dataset.sorted_entries[data_idx]
        entries_by_cluster[entry.current_label_id].append((data_idx, entry))

    # Order within each cluster: by default, as a reference list is ordered.
    for entries_in_this_cluster in entries_by_cluster.values():
        entries_in_this_cluster.sort(key=lambda e: sort_key(e[1]))  # e: `(data_idx, entry)`

    # If `max_n` is enabled, determine how many entries we can display from each cluster to approximately match the total count.
    # But display at least one entry from each cluster.
    if max_n is not None:
        n_clusters_in_selection = len(entries_by_cluster)
        if n_clusters_in_selection > 0:
            max_entries_per_cluster = math.ceil(max_n / n_clusters_in_selection)
        else:
            max_n = None

    def format_cluster_annotation(cluster_id, *, themes_only=False):
        # The metadata for the cluster.
        if cluster_id != -1:  # the outlier set doesn't have a set of common keywords computed
            if dataset.file_content.keywords_available:
                cluster_title = f"#{cluster_id}"
                cluster_keywords = format_cluster_keywords(dataset.file_content, cluster_id, themes_only=themes_only)
            else:
                cluster_title = f"#{cluster_id}"
                cluster_keywords = []
        else:
            cluster_title = "Misc"
            cluster_keywords = []

        # The entries themselves. Leave only the first few if there are too many to display.
        entries = entries_by_cluster[cluster_id]
        more = ""  # nothing was cut, until something is; also the answer when there is no limit at all
        if max_n is not None:
            # TODO: How to compact this in the worst case? Many clusters, with 3 data points in each -> will render 3 * n_clusters entries.
            n_extra_entries = len(entries) - max_entries_per_cluster
            if n_extra_entries > 0:
                if n_extra_entries < 3:  # less pedantic to avoid cutting if there are just 1 or 2 more entries than the limit would allow
                    pass
                else:  # >=3 extra entries, cut at the original limit
                    entries = entries[:max_entries_per_cluster]
                    more = f"[...{n_extra_entries} more entries in {cluster_title}...]"
        cluster_content = entries

        return cluster_title, cluster_keywords, cluster_content, more

    return entries_by_cluster, format_cluster_annotation


def format_selection_heading(n_items: int, cluster_ids) -> str:
    """The info panel's heading for a selection: `"[1274 items total in 68 clusters + Misc]"`.

    `n_items`: how many items are selected.
    `cluster_ids`: the clusters they are in; `-1` is Misc, which is named rather than counted as a cluster, since
                   the clusters are numbered and Misc is not one of the numbers.
    """
    cluster_ids = set(cluster_ids)
    has_misc = -1 in cluster_ids
    n_clusters = len(cluster_ids - {-1})
    parts = []
    if n_clusters:
        parts.append(f"{n_clusters} cluster{'s' if n_clusters != 1 else ''}")
    if has_misc:
        parts.append("Misc")
    return f"[{n_items} item{'s' if n_items != 1 else ''} total in {' + '.join(parts)}]"


def order_cluster_ids(cluster_ids):
    """Sort `cluster_ids` ascending, with the misc group (`-1`) moved to the end.

    Returns a new list. Duplicate IDs are removed.
    """
    out = list(sorted(set(cluster_ids)))
    if out and out[0] == -1:  # move the misc group (if any) to the end
        out = out[1:] + [-1]
    return out


#: What cluster purity means, and what the purity cutoff does; the tooltip of the readouts and the slider.
PURITY_EXPLANATION = ("Cluster purity: of a record's closest neighbours in meaning, the share in its own cluster.\n"
                      "The line gives the average over every record in a cluster; each cluster's header gives its own.\n"
                      "\n"
                      "Clusters with purity below the cutoff are shown as Misc, on the map and in the info panel.\n"
                      "At 0%, every cluster is shown as itself. Ctrl+click the slider to type a value.")


def format_dataset_purity(file_content) -> str | None:
    """The dataset's cluster purity as a short readout, or `None` for a dataset that does not record it.

    `PURITY_EXPLANATION` is what it means, for a tooltip.

    `file_content`: a loaded dataset file (`dataset.file_content`). Datasets imported before purity was measured
                    lack it, as does one where no record is in a cluster.
    """
    maybe_purity = getattr(file_content, "dataset_purity", None)
    if maybe_purity is None:
        return None
    return f"Cluster purity: {maybe_purity:.0%} on average"


def format_cluster_purity(file_content, cluster_id: int) -> str | None:
    """Cluster `cluster_id`'s purity, as `"purity 62%"`, or `None` where there is none to show.

    `None` for the Misc group (-1), for a dataset that does not record purity, and for a cluster whose purity is
    undefined (a cluster of one). `format_dataset_purity` is the sentence that says what the number means.
    """
    purities = getattr(file_content, "cluster_purities", None)
    if cluster_id < 0 or purities is None or cluster_id >= len(purities) or purities[cluster_id] is None:
        return None
    return f"purity {purities[cluster_id]:.0%}"


def format_cluster_keywords(file_content, cluster_id: int, *, n: int = 6,
                            themes_only: bool = False) -> list[tuple[str | None, str]]:
    """Cluster `cluster_id`'s keywords as the panel and the tooltip show them, as lines of `(caption, text)`.

    `file_content`: a loaded dataset file (`dataset.file_content`).
    `n`: how many keywords a line shows.
    `themes_only`: show the themes line alone, where there is one; for a tooltip crowded with clusters.

    Where the dataset has both the LLM's keywords and the phrases (`"hybrid"`, or `"llm"` with the evidence on),
    `[("Themes", ...), ("Quoted", ...)]`: the LLM's keywords, which name what the cluster's papers share, without
    those it gave too many clusters to tell this one apart (`keyphrases.without_umbrella_terms`); and the
    phrases, which quote what the papers say, without a phrase that repeats a theme, and left out where none is left.
    Where the LLM found no theme in the cluster, `[("Quoted", ...)]` alone. Otherwise `[(None, "[keyword0, ...]")]`,
    the cluster's label, which every dataset has.
    """
    def at(maybe_lists):
        return list(maybe_lists[cluster_id]) if maybe_lists is not None and cluster_id < len(maybe_lists) else []

    themes = at(_distinctive_llm_keywords(file_content))
    phrases = at(getattr(file_content, "cluster_keyword_lists", None))
    if themes and themes != ["<unknown topic>"] and phrases:
        themes = themes[:n]
        if themes_only:
            return [("Themes", ", ".join(themes))]
        display = getattr(file_content, "keyphrase_display", None) or {}
        terms = keyphrases.lead_with(themes, phrases, len(themes) + n, display=display, nested=False)[len(themes):]
        # A small cluster may have no phrase left that a theme does not already say.
        return [("Themes", ", ".join(themes))] + ([("Quoted", ", ".join(terms))] if terms else [])
    # The LLM found no theme; the phrases still say something. Read off the LLM's own list, as the corpus-wide filter
    # would remove the marker from a corpus where many clusters have it.
    if at(getattr(file_content, "cluster_llm_keywords", None)) == ["<unknown topic>"] and phrases:
        return [("Quoted", ", ".join(term for term, _share in phrases[:n]))]
    return [(None, f"[{', '.join(file_content.vis_keywords_by_cluster[cluster_id])}]")]


def format_keyword_lines(lines: list[tuple[str | None, str]]) -> str:
    """`format_cluster_keywords`'s lines as one line of text, as a copied report shows them: "Themes: ...; Quoted: ..."."""
    return "; ".join(f"{caption}: {text}" if caption else text for caption, text in lines)


# One dataset's LLM keywords without the corpus-wide ones, kept for the dataset loaded last: every tooltip and panel
# build asks, and the answer needs every cluster's list. Keyed by identity, so a newly loaded dataset is computed
# afresh; holding the object keeps its identity from being reused.
_distinctive_cache: list = [None, None]  # [file_content, its lists]

def _distinctive_llm_keywords(file_content):
    """The dataset's `cluster_llm_keywords` without the terms it gave too many clusters, the terms other clusters
    share too moved after the cluster's own but for the first (`keyphrases.shared_last`), or `None`."""
    if _distinctive_cache[0] is not file_content:
        lists = getattr(file_content, "cluster_llm_keywords", None)
        _distinctive_cache[1] = (keyphrases.shared_last(keyphrases.without_umbrella_terms(
                                     lists, threshold_fraction=visualizer_config.clusters_keyword_umbrella_fraction))
                                 if lists is not None else None)
        _distinctive_cache[0] = file_content
    return _distinctive_cache[1]

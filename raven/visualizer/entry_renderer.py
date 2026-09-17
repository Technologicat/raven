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

  - `get_entries_for_selection(data_idxs, *, sort_field, max_n)` — gather items
    by cluster, sorted alphabetically within each cluster. Returns
    `(entries_by_cluster, format_cluster_annotation)` where the formatter is a
    closure that yields `(cluster_title, cluster_keywords, cluster_content,
    more)` for a given cluster ID.
  - `order_cluster_ids(cluster_ids)` — sort ascending with the misc group
    (cluster `-1`) moved to the end.

Cross-module state read via `app_state`:
  `dataset` (for `sorted_entries`, `file_content.keywords_available`, and
  `file_content.vis_keywords_by_cluster`).
"""

__all__ = ["get_entries_for_selection",
           "order_cluster_ids"]

import collections
import math

from .app_state import app_state


def get_entries_for_selection(data_idxs, *, sort_field="title", max_n=None, dataset=None):
    """Gather item data for visualization, sorting by cluster.

    `data_idxs`: `list`, the selection of items to include in the report. Item indices into `sorted_xxx`.
    `sort_field`: `str`, the field to sort by within each cluster. The name of one of the attributes of an entry in `sorted_entries`.
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
            if dataset.file_content.keywords_available:
                cluster_title = f"#{cluster_id}"
                cluster_keywords = f"[{', '.join(dataset.file_content.vis_keywords_by_cluster[cluster_id])}]\n"
            else:
                cluster_title = f"#{cluster_id}"
                cluster_keywords = ""
        else:
            cluster_title = "Misc"
            cluster_keywords = ""

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


def order_cluster_ids(cluster_ids):
    """Sort `cluster_ids` ascending, with the misc group (`-1`) moved to the end.

    Returns a new list. Duplicate IDs are removed.
    """
    out = list(sorted(set(cluster_ids)))
    if out and out[0] == -1:  # move the misc group (if any) to the end
        out = out[1:] + [-1]
    return out

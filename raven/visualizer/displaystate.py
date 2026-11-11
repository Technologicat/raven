"""What the Visualizer currently shows for each record, as distinct from what the importer found.

Each entry of `dataset.sorted_entries` keeps the importer's `cluster_id` and carries `current_label_id`, the label
it is shown with: coloured by on the map, and grouped under in the info panel and the tooltip. They start equal,
and the display filters change only `current_label_id`.

Record indices never move. A `data_idx` indexes the `sorted_*` arrays, sorted by the importer's label once when the
dataset loads, and the selection, the hover lookup, search and the info panel all hold one; so a filter regroups
records by changing labels, and the plot is refilled from those groups (`points_by_label`), never re-sorted.

No DPG here: `raven.visualizer.plotter.refresh_series` is what puts the groups on screen.
"""

__all__ = ["init", "relabel_by_purity", "points_by_label"]

import collections


def init(entries) -> None:
    """Show every entry under the label the importer gave it."""
    for entry in entries:
        entry.current_label_id = entry.cluster_id


def relabel_by_purity(entries, maybe_cluster_purities, cutoff: float) -> bool:
    """Show the records of every cluster whose purity is below `cutoff` as Misc (-1), and every other record under
    its own cluster.

    `maybe_cluster_purities`: the dataset's `cluster_purities`, indexed by cluster ID, or `None` for a dataset that
                              does not record them, in which case nothing is relabelled. A cluster whose purity is
                              `None` (undefined) keeps its label.
    `cutoff`: in [0, 1]. At 0 every cluster is shown as itself.

    Returns whether any entry's label changed, so that a caller can skip redrawing.
    """
    changed = False
    for entry in entries:
        cluster_id = entry.cluster_id
        demote = (maybe_cluster_purities is not None
                  and 0 <= cluster_id < len(maybe_cluster_purities)
                  and maybe_cluster_purities[cluster_id] is not None
                  and maybe_cluster_purities[cluster_id] < cutoff)
        label = -1 if demote else cluster_id
        if entry.current_label_id != label:
            entry.current_label_id = label
            changed = True
    return changed


def points_by_label(entries) -> dict[int, list[int]]:
    """The `data_idx` of each entry, grouped by its `current_label_id`, in index order within each group.

    `entries`: `dataset.sorted_entries`, whose positions are the `data_idx` values.
    """
    groups: dict[int, list[int]] = collections.defaultdict(list)
    for data_idx, entry in enumerate(entries):
        groups[entry.current_label_id].append(data_idx)
    return dict(groups)

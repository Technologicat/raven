"""What the Visualizer currently shows for each record, as distinct from what the importer found.

Each entry of `dataset.sorted_entries` keeps the importer's `cluster_id` and carries two things the display filters
change:

  - `current_label_id`, the label it is shown with: coloured by on the map, and grouped under in the info panel and
    the tooltip. It starts equal to `cluster_id`.
  - `visibility`, how it is shown: `SHOWN` in its colour, `GRAYED` (a soft hide, drawn in gray beneath everything
    else), or `HIDDEN` (not drawn). It starts `SHOWN`.

Record indices never move. A `data_idx` indexes the `sorted_*` arrays, sorted by the importer's label once when the
dataset loads, and the selection, the hover lookup, search and the info panel all hold one; so a filter regroups
records by changing labels, and the plot is refilled from those groups (`points_by_label`), never re-sorted.

No DPG here: `raven.visualizer.plotter.refresh_series` is what puts the groups on screen.
"""

__all__ = ["SHOWN", "GRAYED", "HIDDEN", "MISC_MODES",
           "init", "relabel_by_purity", "apply_misc_mode",
           "points_by_label", "points_with_visibility", "visibility_counts", "summary"]

import collections

SHOWN = "shown"
GRAYED = "grayed"
HIDDEN = "hidden"
#: The ways Misc can be shown, in the order a control cycles through them.
MISC_MODES = (SHOWN, GRAYED, HIDDEN)


def init(entries) -> None:
    """Show every entry, in full colour, under the label the importer gave it."""
    for entry in entries:
        entry.current_label_id = entry.cluster_id
        entry.visibility = SHOWN


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


def apply_misc_mode(entries, mode: str) -> bool:
    """Show every record currently labelled Misc (-1) as `mode`, one of `MISC_MODES`, and every other record `SHOWN`.

    Reads `current_label_id`, so a cluster `relabel_by_purity` has moved to Misc goes with it: call this after that.

    Returns whether any entry's visibility changed.
    """
    if mode not in MISC_MODES:
        raise ValueError(f"apply_misc_mode: unknown mode {mode!r}; known: {MISC_MODES}")
    changed = False
    for entry in entries:
        visibility = mode if entry.current_label_id == -1 else SHOWN
        if entry.visibility != visibility:
            entry.visibility = visibility
            changed = True
    return changed


def points_by_label(entries) -> dict[int, list[int]]:
    """The `data_idx` of each `SHOWN` entry, grouped by its `current_label_id`, in index order within each group.

    `entries`: `dataset.sorted_entries`, whose positions are the `data_idx` values. Grayed and hidden entries are left
    out; `points_with_visibility` finds the grayed ones.
    """
    groups: dict[int, list[int]] = collections.defaultdict(list)
    for data_idx, entry in enumerate(entries):
        if entry.visibility == SHOWN:
            groups[entry.current_label_id].append(data_idx)
    return dict(groups)


def points_with_visibility(entries, visibility: str) -> list[int]:
    """The `data_idx` of every entry shown as `visibility`, in index order."""
    return [data_idx for data_idx, entry in enumerate(entries) if entry.visibility == visibility]


def visibility_counts(entries) -> dict[str, int]:
    """How many entries are shown as each of `SHOWN`, `GRAYED` and `HIDDEN`."""
    counts = collections.Counter(entry.visibility for entry in entries)
    return {visibility: counts.get(visibility, 0) for visibility in MISC_MODES}


def summary(entries) -> str | None:
    """What a reader must be told about records not in full colour, or `None` when every record is.

    `"1,422 of 2,032 in focus"` when some are grayed and none hidden (the grayed ones are still on screen), and
    `"Showing 1,610 of 2,032"` when some are hidden, adding `", 1,422 in focus"` when some are also grayed.
    """
    counts = visibility_counts(entries)
    total = sum(counts.values())
    if counts[HIDDEN]:
        text = f"Showing {total - counts[HIDDEN]:,} of {total:,}"
        return text + (f", {counts[SHOWN]:,} in focus" if counts[GRAYED] else "")
    if counts[GRAYED]:
        return f"{counts[SHOWN]:,} of {total:,} in focus"
    return None

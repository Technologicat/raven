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

Each filter that sets visibility gives its own verdict per record (`misc_verdict`, `year_verdict`), and
`apply_filters` combines them: a record is shown only if every filter shows it, hidden if any hides it, and grayed
otherwise (`combine`).

Only records in full colour can be interacted with — hovered, selected, listed in the info panel — plus the grayed
ones when the user asks for them (`interactable`). Hidden records are not on screen at all (`on_screen`).

No DPG here: `raven.visualizer.plotter.refresh_series` is what puts the groups on screen.
"""

__all__ = ["SHOWN", "GRAYED", "HIDDEN", "MISC_MODES", "YEAR_MODES",
           "init", "relabel_by_purity",
           "misc_verdict", "year_of", "year_bounds", "YEAR_RANGE_ENDS", "step_year_range", "year_verdict",
           "combine", "apply_filters",
           "points_by_label", "points_with_visibility", "visibility_counts", "summary",
           "interactable", "on_screen"]

import collections

import numpy as np

SHOWN = "shown"
GRAYED = "grayed"
HIDDEN = "hidden"
#: The ways Misc can be shown, in the order a control cycles through them.
MISC_MODES = (SHOWN, GRAYED, HIDDEN)
#: The ways the year filter can show the records outside its range. Shown is not one: covering every year is what
#: switches the filter off.
YEAR_MODES = (GRAYED, HIDDEN)
#: How restrictive each visibility is, for `combine`.
_RESTRICTIVENESS = {SHOWN: 0, GRAYED: 1, HIDDEN: 2}


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


def misc_verdict(entry, mode: str) -> str:
    """How the Misc filter would show `entry`: as `mode` if it is labelled Misc (-1), else `SHOWN`.

    Reads `current_label_id`, so a cluster `relabel_by_purity` has moved to Misc goes with it.
    """
    return mode if entry.current_label_id == -1 else SHOWN


def year_of(entry) -> int | None:
    """`entry`'s year as a number, or `None` when it has none that reads as one.

    The importer's placeholder for a missing year is not a number, so it is `None` here like any other value that
    does not read as a year.
    """
    try:
        return int(str(getattr(entry, "year", "")).strip())
    except ValueError:
        return None


def year_bounds(entries) -> tuple[int, int] | None:
    """The earliest and the latest year among `entries`, or `None` when none of them has a year."""
    years = [year for year in (year_of(entry) for entry in entries) if year is not None]
    return (min(years), max(years)) if years else None


#: Which end of a year range `step_year_range` moves.
YEAR_RANGE_ENDS = ("both", "first", "last")


def step_year_range(year_range: tuple[int, int], *, delta: int, bounds: tuple[int, int],
                    ends: str = "both") -> tuple[int, int]:
    """`year_range` moved by `delta` years, within `bounds`, both `(first, last)` and inclusive.

    `ends`: one of `YEAR_RANGE_ENDS`. `"both"` slides the range, keeping its width; a slide that would cross a bound
            does not move it. `"first"` and `"last"` move that end alone, stopping at the bound and at the other end.

    Raises `ValueError` on an unknown `ends`.
    """
    first, last = year_range
    lo, hi = bounds
    if ends == "both":
        if first + delta < lo or last + delta > hi:
            return year_range
        return first + delta, last + delta
    if ends == "first":
        return max(lo, min(last, first + delta)), last
    if ends == "last":
        return first, max(first, min(hi, last + delta))
    raise ValueError(f"step_year_range: unknown ends {ends!r}; known: {YEAR_RANGE_ENDS}")


def year_verdict(entry, maybe_year_range: tuple[int, int] | None, mode: str) -> str:
    """How the year filter would show `entry`: `SHOWN` if its year is within `maybe_year_range`, inclusive at both
    ends, and as `mode` otherwise, a record with no year included.

    `maybe_year_range`: `(first, last)`, or `None` when the filter is off, which shows every record.
    """
    if maybe_year_range is None:
        return SHOWN
    first, last = maybe_year_range
    year = year_of(entry)
    return SHOWN if year is not None and first <= year <= last else mode


def combine(*verdicts: str) -> str:
    """The visibility several filters' verdicts add up to: the most restrictive of them.

    Shown only if every filter shows it, hidden if any hides it, and grayed otherwise. No verdicts is `SHOWN`.
    """
    return max(verdicts, key=_RESTRICTIVENESS.__getitem__, default=SHOWN)


def apply_filters(entries, *, misc_mode: str = SHOWN,
                  maybe_year_range: tuple[int, int] | None = None, year_mode: str = GRAYED) -> bool:
    """Set every entry's visibility from all the display filters at once (`combine`).

    `misc_mode`: how Misc is shown, one of `MISC_MODES` (`misc_verdict`). Call this after `relabel_by_purity`,
                 which decides what Misc is.
    `maybe_year_range`, `year_mode`: the year filter (`year_verdict`); `year_mode` is one of `YEAR_MODES`.

    Returns whether any entry's visibility changed.
    """
    if misc_mode not in MISC_MODES:
        raise ValueError(f"apply_filters: unknown Misc mode {misc_mode!r}; known: {MISC_MODES}")
    if year_mode not in YEAR_MODES:
        raise ValueError(f"apply_filters: unknown year mode {year_mode!r}; known: {YEAR_MODES}")
    changed = False
    for entry in entries:
        visibility = combine(misc_verdict(entry, misc_mode), year_verdict(entry, maybe_year_range, year_mode))
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


def interactable(entries, data_idxs, *, grayed_selectable: bool) -> np.ndarray:
    """Those of `data_idxs` that can be hovered and selected: the `SHOWN` ones, and the `GRAYED` ones too if
    `grayed_selectable`. Never the `HIDDEN` ones.

    `entries`: `dataset.sorted_entries`, whose positions are the `data_idx` values.
    `data_idxs`: indices into `entries`, in any iterable; their order is kept.

    Returns an `np.ndarray` of indices, of the dtype NumPy slices with even when empty.
    """
    allowed = (SHOWN, GRAYED) if grayed_selectable else (SHOWN,)
    return np.array([data_idx for data_idx in data_idxs if entries[data_idx].visibility in allowed], dtype=np.int64)


def on_screen(entries, data_idxs) -> np.ndarray:
    """Those of `data_idxs` that are drawn at all, in colour or grayed: every one but the `HIDDEN` ones.

    Same arguments and return value as `interactable`.
    """
    return interactable(entries, data_idxs, grayed_selectable=True)

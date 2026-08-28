"""Pure navigation arithmetic for a cursor over a list: where does a keypress move it, and where does it
land when the list is rebuilt underneath it?

Separated from the DPG-bound widgets so the algorithm can be tested by stating the invariant directly — no
context to create, no frames to render, no widget to drive. See `resolve_nav_target`.

Used by `raven.common.gui.thumbnailgrid`, and through it by every grid in the constellation; `reanchor_cursor`
additionally serves the file dialog's table views, which show the same listing the grid does and must agree
with it about where the cursor goes.

The module name is narrower than its contents: none of this is grid-specific — `resolve_nav_target` has
always operated on a flat list of visible indices, with two-dimensional movement expressed by the caller as
a delta of one row's width. Worth renaming when something else here is being touched anyway.
"""

__all__ = ["resolve_nav_target", "reanchor_cursor", "resolve_undo_nav_target"]

import bisect
from typing import Any, List, Optional, Sequence, Set


def resolve_nav_target(visible: List[int], current: int, delta: int) -> Optional[int]:
    """Resolve relative grid navigation to a target image index.

    `visible` is the list of image indices shown under the current filter, in
    ascending order (always true: it's built from `range(n_images)`). `current`
    is the global index of the current image, which may or may not be in
    `visible`. `delta` is the signed step (+1 next, -1 prev, ±n_cols rows, …).

    Returns the global index to navigate to, clamped to the ends of `visible`,
    or None if `visible` is empty.

    When `current` is hidden by the filter — e.g. just tagged out of a
    neutral-only view — it no longer occupies a slot in `visible`; it sits in
    the *gap* at its insertion point `ins`. A forward step must land on the
    item *after* the gap (`visible[ins]`, the one that took its place), a
    backward step on the item *before* it (`visible[ins - 1]`). Snapping to the
    nearest surviving item and then adding the full delta would skip one — the
    bug this gap arithmetic fixes.
    """
    if not visible:
        return None
    if current in visible:
        new_pos = visible.index(current) + delta
    else:
        ins = bisect.bisect_left(visible, current)
        new_pos = (ins - 1 + delta) if delta > 0 else (ins + delta)
    new_pos = max(0, min(len(visible) - 1, new_pos))
    return visible[new_pos]


def reanchor_cursor(keys: Sequence[Any],
                    previous_key: Any,
                    previous_index: Optional[int]) -> Optional[int]:
    """Where a cursor should land after the list it points into has been rebuilt.

    `keys` is the new list, in display order, of whatever identifies an entry across a rebuild — for a file
    listing, the paths. `previous_key` and `previous_index` are where the cursor was; either may be `None`
    if there was no cursor. Returns the new index, or `None` when there is nothing to point at.

    Two rules, and the order matters:

    1. **The same entry, wherever it moved to.** A re-sort or a re-filter that keeps the entry keeps the
       cursor on it, which is the answer in the overwhelming majority of rebuilds.
    2. **Otherwise the same position, clamped.** When the entry is gone, relative position is what the user
       still has: they narrowed a search and the file under the cursor stopped matching, and they were
       looking at the middle of the list. Falling to the top instead would throw them to the top of a
       directory on a keystroke, in a dialog where every keystroke rebuilds the listing.

    A re-sort never reaches rule 2, which is worth knowing before worrying about it: re-sorting keeps every
    entry, so rule 1 catches them all. Rule 2 is reached only when an entry genuinely left — a re-filter, a
    deletion, a refresh of a directory that changed underneath.

    **For a genuinely different list — a new directory — pass `None` for both**, and the cursor starts at the
    top. Clamping a position across a `chdir` would carry a number from one directory into another, where it
    means nothing; this function cannot tell the two cases apart, so the caller says which it is by whether
    it offers a previous position at all.
    """
    if not keys:
        return None
    if previous_key is not None:
        try:
            return list(keys).index(previous_key)
        except ValueError:
            pass
    if previous_index is None:
        return 0
    return max(0, min(len(keys) - 1, previous_index))


def resolve_undo_nav_target(affected: List[int], current: int,
                            visible: Set[int]) -> Optional[int]:
    """Where to move the view after an undo/redo — or None to stay put.

    `affected` is the list of grid indices touched by the undone/redone action;
    `current` the current image's index; `visible` the set of indices shown
    under the active filter.

    The guiding principle is *minimal movement*: keep the user on a changed
    image, but move the highlight as little as possible.

    - If the current image is itself one of the changed images, stay on it — the
      main view shows it regardless of whether its grid tile is filtered out.
      (Reverting a winner+losers set leaves the *winner* current, not the first
      loser; redoing a batch out of a filtered view leaves you on the image you
      were on, not jumping to the first of the now-hidden set.) The one exception
      is when the current image has been filtered out *and* another changed image
      is still visible — then hop to that visible one so you're not stranded on a
      hidden tile while a change sits in plain sight.
    - If the current image is unaffected, jump to the first changed image by grid
      position, preferring one that's visible.
    """
    if not affected:
        return None
    visible_affected = [i for i in affected if i in visible]
    if current in affected:
        # Stay on current unless it's hidden while a changed image is visible.
        if current in visible or not visible_affected:
            return None
        return min(visible_affected)
    return min(visible_affected) if visible_affected else min(affected)

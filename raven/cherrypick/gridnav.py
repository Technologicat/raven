"""Pure navigation arithmetic for raven-cherrypick's thumbnail grid.

Separated from the DPG-bound `grid` widget so the algorithm layer is testable
without the GUI stack (the test CI installs a minimal, dearpygui-free dependency
subset). See `resolve_nav_target`.
"""

__all__ = ["resolve_nav_target", "resolve_undo_nav_target"]

import bisect
from typing import List, Optional, Set


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

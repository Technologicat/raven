"""Pure navigation arithmetic for raven-cherrypick's thumbnail grid.

Separated from the DPG-bound `grid` widget so the algorithm layer is testable
without the GUI stack (the test CI installs a minimal, dearpygui-free dependency
subset). See `resolve_nav_target`.
"""

__all__ = ["resolve_nav_target"]

import bisect
from typing import List, Optional


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

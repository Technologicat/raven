"""Tests for raven.common.gui.gridnav navigation arithmetic.

Covers `resolve_nav_target`, the pure core of relative navigation, and
`reanchor_cursor`, which decides where a cursor lands when the list underneath
it is rebuilt. The focus for the first is the filtered-view case: when the
current image has been tagged out of the visible set, a step must land on the
image that took its place — not skip past it. For the second it is the policy
itself, which two views of one listing have to share.

The arithmetic lives apart from the widgets so these can state the invariant
directly — no context to create, no frames to render, no widget to drive.
"""

import pytest

from raven.common.gui.gridnav import reanchor_cursor, resolve_nav_target, resolve_undo_nav_target


# ---------------------------------------------------------------------------
# Current image visible (no filter, or filter still includes it)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("current, delta, expected", [
    (2, +1, 3),    # next
    (2, -1, 1),    # prev
    (0, -1, 0),    # prev clamps at first
    (4, +1, 4),    # next clamps at last
    (2, +10, 4),   # large forward step clamps
    (2, -10, 0),   # large backward step clamps
    (1, +3, 4),    # row-style step
])
def test_current_visible(current, delta, expected):
    visible = [0, 1, 2, 3, 4]
    assert resolve_nav_target(visible, current, delta) == expected


def test_single_visible_item_clamps_either_way():
    assert resolve_nav_target([5], 5, +1) == 5
    assert resolve_nav_target([5], 5, -1) == 5


# ---------------------------------------------------------------------------
# Empty visible set
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("delta", [+1, -1, +3, -3])
def test_empty_visible_returns_none(delta):
    assert resolve_nav_target([], 0, delta) is None


# ---------------------------------------------------------------------------
# Current image hidden by the filter (the regression)
# ---------------------------------------------------------------------------

def test_tagged_out_first_image_then_next_lands_on_replacement():
    """The bug: tag the first neutral image, press Right, land on the new first.

    Neutral-only view shows globals [0,1,2,3,4]; current is 0. Tagging 0 drops
    it from the view -> visible becomes [1,2,3,4], current still 0 (now hidden).
    Right must land on global 1 (the new first), not skip to 2.
    """
    assert resolve_nav_target([1, 2, 3, 4], 0, +1) == 1


def test_hidden_current_in_middle():
    # current 2 hidden; gap sits between 1 and 3.
    visible = [0, 1, 3, 4]
    assert resolve_nav_target(visible, 2, +1) == 3   # item after the gap
    assert resolve_nav_target(visible, 2, -1) == 1   # item before the gap


def test_hidden_current_multi_step():
    # current 2 hidden; gap between 1 and 4.
    visible = [0, 1, 4, 5]
    assert resolve_nav_target(visible, 2, +2) == 5
    assert resolve_nav_target(visible, 2, -2) == 0


def test_hidden_current_before_all_visible():
    # current 0 hidden, everything visible is after it.
    visible = [2, 3, 4]
    assert resolve_nav_target(visible, 0, +1) == 2   # first item after the gap
    assert resolve_nav_target(visible, 0, -1) == 2   # nothing before -> clamp to first


def test_hidden_current_after_all_visible():
    # current 5 hidden, everything visible is before it.
    visible = [0, 1, 2]
    assert resolve_nav_target(visible, 5, +1) == 2   # nothing after -> clamp to last
    assert resolve_nav_target(visible, 5, -1) == 2   # last item before the gap


# ---------------------------------------------------------------------------
# resolve_undo_nav_target: where the view lands after an undo/redo
# ---------------------------------------------------------------------------

def test_undo_nav_stays_when_current_is_affected_and_visible():
    # The winner+losers case: affected = losers (3, 4) + winner (7); current is
    # the winner, still visible. Stay on the winner — don't hop to a loser.
    assert resolve_undo_nav_target([3, 4, 7], current=7, visible={3, 4, 7, 9}) is None


def test_undo_nav_jumps_when_current_not_affected():
    # Current is elsewhere; show the change at the first affected (by position).
    assert resolve_undo_nav_target([3, 4, 7], current=9, visible={3, 4, 7, 9}) == 3


def test_undo_nav_prefers_visible_affected():
    # First-by-position (2) is hidden; land on the first *visible* affected (5).
    assert resolve_undo_nav_target([2, 5], current=9, visible={5, 9}) == 5


def test_undo_nav_falls_back_to_first_when_none_visible():
    # Nothing affected is visible (all filtered out) — go to the first anyway.
    assert resolve_undo_nav_target([2, 5], current=9, visible={9}) == 2


def test_undo_nav_jumps_when_current_affected_but_hidden_and_another_visible():
    # Current was affected and got filtered out, but another affected image (5)
    # is still visible — hop to it rather than stranding on the hidden tile.
    assert resolve_undo_nav_target([2, 5], current=2, visible={5}) == 5


def test_undo_nav_stays_when_current_affected_and_none_visible():
    # The redo-out-of-filter case: a batch was marked out of the neutral filter
    # with the last one still current; redoing pushes them all out again, so
    # current AND every other affected image go hidden. Stay on current (the main
    # view shows it) — jumping to another equally-hidden tile gains nothing.
    assert resolve_undo_nav_target([10, 11, 12], current=12, visible={3, 4, 5}) is None


def test_undo_nav_empty_affected_stays():
    assert resolve_undo_nav_target([], current=3, visible={3}) is None


# ---------------------------------------------------------------------------
# reanchor_cursor
# ---------------------------------------------------------------------------

def test_reanchor_follows_the_entry_when_it_survives():
    # A re-sort moves everything; the cursor rides along with the file it was on.
    assert reanchor_cursor(["c", "b", "a"], previous_key="b", previous_index=1) == 1
    assert reanchor_cursor(["c", "a", "b"], previous_key="b", previous_index=1) == 2


def test_reanchor_keeps_position_when_the_entry_is_gone():
    # Typing one more character filtered the cursor's file out. Relative position is what is left.
    assert reanchor_cursor(["a", "c", "d"], previous_key="b", previous_index=2) == 2


def test_reanchor_clamps_a_position_past_the_end():
    # The list got shorter than where the cursor was.
    assert reanchor_cursor(["a", "b"], previous_key="z", previous_index=7) == 1


def test_reanchor_does_not_fall_to_the_first_entry():
    """The policy this function exists to pin: a vanished entry keeps its place, it does not go to the top.

    The grid used to jump to entry 0 here, which throws the user to the top of the directory on a
    keystroke — in a dialog where every keystroke rebuilds the listing.
    """
    assert reanchor_cursor(["a", "b", "c", "d"], previous_key="gone", previous_index=3) == 3


def test_reanchor_empty_list_has_nowhere_to_point():
    assert reanchor_cursor([], previous_key="a", previous_index=0) is None


def test_reanchor_with_no_previous_cursor_starts_at_the_top():
    assert reanchor_cursor(["a", "b"], previous_key=None, previous_index=None) == 0

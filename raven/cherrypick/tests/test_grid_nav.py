"""Tests for raven.cherrypick.gridnav navigation arithmetic.

Covers `resolve_nav_target`, the pure core of relative grid navigation. The
focus is the filtered-view case: when the current image has been tagged out of
the visible set, a step must land on the image that took its place — not skip
past it.

Lives in its own dearpygui-free module so this test runs under the CI's minimal
dependency subset (the `grid` widget itself imports dearpygui).
"""

import pytest

from raven.cherrypick.gridnav import resolve_nav_target, resolve_undo_nav_target


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

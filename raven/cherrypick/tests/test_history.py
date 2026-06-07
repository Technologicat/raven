"""Tests for raven.cherrypick.history — undo/redo stack semantics.

Pure (dpg-free): exercises the diff bookkeeping and stack shuffling, not the
file moves or GUI refresh the app layers on top.
"""

from raven.cherrypick.history import TriageHistory
from raven.cherrypick.triage import TriageState

N = TriageState.NEUTRAL
C = TriageState.CHERRY
L = TriageState.LEMON


def test_empty_history():
    h = TriageHistory()
    assert not h.can_undo()
    assert not h.can_redo()
    assert h.undo() is None
    assert h.redo() is None


def test_record_empty_diff_is_noop():
    h = TriageHistory()
    h.record([])
    assert not h.can_undo()


def test_single_action_undo_returns_old_states():
    h = TriageHistory()
    h.record([("a.png", N, C)])           # marked a.png cherry
    assert h.can_undo() and not h.can_redo()
    assert h.undo() == [("a.png", N)]     # restore to neutral
    assert not h.can_undo() and h.can_redo()


def test_undo_then_redo_round_trip():
    h = TriageHistory()
    h.record([("a.png", N, C)])
    assert h.undo() == [("a.png", N)]
    assert h.redo() == [("a.png", C)]     # re-apply cherry
    assert h.can_undo() and not h.can_redo()


def test_recording_clears_redo():
    h = TriageHistory()
    h.record([("a.png", N, C)])
    h.undo()
    assert h.can_redo()
    h.record([("b.png", N, L)])           # new action off the undone branch
    assert not h.can_redo()               # redo branch discarded


def test_lifo_order_across_actions():
    h = TriageHistory()
    h.record([("a.png", N, C)])
    h.record([("b.png", N, L)])
    assert h.undo() == [("b.png", N)]     # most recent first
    assert h.undo() == [("a.png", N)]
    assert h.undo() is None


def test_batch_action_preserves_order_and_round_trips():
    # A winner-mark: losers -> lemon, then winner -> cherry (one action).
    batch_diff = [("loser1.png", N, L), ("loser2.png", C, L), ("winner.png", N, C)]
    h = TriageHistory()
    h.record(batch_diff)
    assert h.undo() == [("loser1.png", N), ("loser2.png", C), ("winner.png", N)]
    assert h.redo() == [("loser1.png", L), ("loser2.png", L), ("winner.png", C)]


def test_clear_empties_both_stacks():
    h = TriageHistory()
    h.record([("a.png", N, C)])
    h.undo()
    assert h.can_redo()
    h.clear()
    assert not h.can_undo()
    assert not h.can_redo()


def test_record_snapshots_the_diff():
    # The history must not alias the caller's list.
    h = TriageHistory()
    diff = [("a.png", N, C)]
    h.record(diff)
    diff.append(("b.png", N, L))          # mutate caller's list afterward
    assert h.undo() == [("a.png", N)]     # unaffected

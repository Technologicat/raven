"""Undo/redo history for raven-cherrypick triage moves.

A triage "move" is a pure state transition (neutral / cherry / lemon) and the
grid position is stable across it, so undo and redo reduce to replaying target
states — no position bookkeeping. This class holds the two stacks and computes
the target-state batch to apply for each direction; it performs no file or GUI
operations. The app applies the returned batch (moving files, refreshing the
view) and resolves filenames to current indices.

Records key on *filename*, not grid index, so they survive any rescan that
would renumber the list.
"""

__all__ = ["TriageHistory"]

from typing import List, Optional, Tuple

from .triage import TriageState

# One recorded user action: the per-image diffs (filename, old_state, new_state)
# that actually changed. A multi-image mark (selection, or winner) is one action.
Diff = List[Tuple[str, TriageState, TriageState]]
# A batch to apply: (filename, target_state) per image.
Batch = List[Tuple[str, TriageState]]


class TriageHistory:
    """Two-stack (undo / redo) linear history over triage diffs."""

    def __init__(self) -> None:
        self._undo: List[Diff] = []
        self._redo: List[Diff] = []

    def clear(self) -> None:
        """Drop all history (e.g. when opening a different folder)."""
        self._undo.clear()
        self._redo.clear()

    def record(self, diff: Diff) -> None:
        """Record a freshly applied user action; no-op for an empty diff.

        Recording a new action invalidates the redo stack — the standard linear
        undo semantics (you can't redo down a branch you've stepped off).
        """
        if not diff:
            return
        self._undo.append(list(diff))
        self._redo.clear()

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> Optional[Batch]:
        """Pop the last action; return the batch that restores the prior states.

        Returns None if there is nothing to undo. The popped action moves to the
        redo stack.
        """
        if not self._undo:
            return None
        diff = self._undo.pop()
        self._redo.append(diff)
        return [(filename, old_state) for filename, old_state, _new in diff]

    def redo(self) -> Optional[Batch]:
        """Pop the last undone action; return the batch that re-applies it.

        Returns None if there is nothing to redo. The popped action moves back to
        the undo stack.
        """
        if not self._redo:
            return None
        diff = self._redo.pop()
        self._undo.append(diff)
        return [(filename, new_state) for filename, _old, new_state in diff]

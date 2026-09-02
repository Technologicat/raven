"""Where a view has been, and the way back: a back/forward history over opaque states.

A list, a cursor, and two injected policies. The list holds whatever the caller wants to return to — a
selection, a directory, a description of what a graph view is showing — and this module never looks inside
one except through the policies it is given.

The two policies are the whole of what a history cannot know for itself:

- **Equality**, because committing the state you are already on should do nothing, and what counts as "the
  same" is domain knowledge. A selection is the same when its *set* of indices matches, whatever order the
  array happens to be in.
- **Validity**, because a state can go bad between being pushed and being popped, and only the caller can
  say so. A directory is deleted from another window; a chat node is removed by a cleanup pass. Going back
  to one is a request that cannot be satisfied, so the traversal steps over it rather than stopping there —
  a Back that appears to do nothing is worse than a Back that goes one step further.

Validity is re-tested on every traversal rather than pruned once, so a state that becomes reachable again —
a remount, a recreated directory — comes back with it.
"""

__all__ = ["NavigationHistory"]

import logging
from typing import Any, Callable, List, Optional, Sequence

logger = logging.getLogger(__name__)

# What a history uses when the caller names no policy: ordinary equality, and everything is valid.
#
# The second default is what makes this usable by a caller whose states cannot go bad -- Visualizer's
# selection history resets whenever the dataset it indexes is replaced, so its stack never outlives what
# its states refer to and there is nothing for a validity predicate to catch.
_DEFAULT_SAME: Callable[[Any, Any], bool] = lambda a, b: a == b  # noqa: E731 -- a name for a default, not a def
_DEFAULT_IS_VALID: Callable[[Any], bool] = lambda state: True  # noqa: E731 -- likewise


class NavigationHistory:
    """A back/forward history over states this class knows nothing about.

    Committing truncates whatever lay ahead, which is what makes the history a line rather than a tree:
    going back and then somewhere new abandons the branch you left, as every browser and file manager does.
    """

    def __init__(self, initial: Any = None,
                 same: Optional[Callable[[Any, Any], bool]] = None,
                 is_valid: Optional[Callable[[Any], bool]] = None):
        """Create a history, optionally holding `initial` as its first state.

        `initial`: Where the view is now. A history with no initial state accepts its first `commit` as
                   the starting point, which is what a caller that has nothing to show yet wants.
        `same`: `(a, b) -> bool`, asked whether a commit would change anything. Defaults to `==`.
        `is_valid`: `(state) -> bool`, asked whether a state can still be returned to. Defaults to always.
                    Called on every traversal rather than once, since the answer changes on its own.
        """
        self._same = same if same is not None else _DEFAULT_SAME
        self._is_valid = is_valid if is_valid is not None else _DEFAULT_IS_VALID
        self._states: List[Any] = [] if initial is None else [initial]
        self._cursor = 0

    def __len__(self) -> int:
        """Return how many states are held, valid or not."""
        return len(self._states)

    def _get_current(self) -> Any:
        """Return the state the cursor is on, or `None` for an empty history."""
        if not self._states:
            return None
        return self._states[self._cursor]

    current = property(fget=_get_current,
                       doc="The state the cursor is on, or `None` if the history is empty.")

    def _get_states(self) -> Sequence[Any]:
        """Return every state held, oldest first, as a snapshot."""
        return tuple(self._states)

    states = property(fget=_get_states,
                      doc="Every state held, oldest first, as a tuple. For inspection and testing.")

    def reset(self, state: Any = None) -> None:
        """Forget everything and start again, optionally at `state`."""
        self._states = [] if state is None else [state]
        self._cursor = 0

    def commit(self, state: Any) -> bool:
        """Record `state` as where the view is now. Returns whether anything was recorded.

        A commit equal to the current state is not recorded — a view that redraws without moving should
        not fill the history with places the reader never went.

        Anything ahead of the cursor is discarded, so going back and then somewhere new abandons the
        branch that was left.
        """
        if self._states and self._same(self._states[self._cursor], state):
            return False
        del self._states[self._cursor + 1:]
        self._states.append(state)
        self._cursor = len(self._states) - 1
        return True

    def _next_valid(self, step: int) -> Optional[int]:
        """Return the nearest index in direction `step` whose state is still valid, or `None`."""
        index = self._cursor + step
        while 0 <= index < len(self._states):
            if self._is_valid(self._states[index]):
                return index
            index += step
        return None

    def _get_can_go_back(self) -> bool:
        """Return whether there is a state behind the cursor that can still be returned to."""
        return self._next_valid(-1) is not None

    can_go_back = property(fget=_get_can_go_back,
                           doc="Whether a state behind the cursor can still be returned to. This is what "
                               "a Back button's enabled state should follow — not whether the cursor is "
                               "at the start, which stops being the same question once dead states are "
                               "skipped. It can go stale on its own, so re-read it whenever the caller "
                               "re-checks the world, and after every step.")

    def _get_can_go_forward(self) -> bool:
        """Return whether there is a state ahead of the cursor that can still be returned to."""
        return self._next_valid(1) is not None

    can_go_forward = property(fget=_get_can_go_forward,
                              doc="Whether a state ahead of the cursor can still be returned to. See "
                                  "`can_go_back`.")

    def back(self, apply: Optional[Callable[[Any], bool]] = None) -> bool:
        """Step to the nearest valid state behind the cursor. Returns whether the cursor moved.

        `apply`: `(state) -> bool`, called with the state before the cursor commits to it, and given the
                 last word. Returning `False` leaves the cursor where it was.

                 That veto is not decoration. A state can pass `is_valid` and still fail to be *entered* —
                 a directory that exists but cannot be read is the case that named this — and a cursor
                 that moved anyway would disagree with where the view actually is, which makes every later
                 step wrong rather than just this one.

                 Omit it where arriving cannot fail, and the step is taken unconditionally.
        """
        return self._step(-1, apply)

    def forward(self, apply: Optional[Callable[[Any], bool]] = None) -> bool:
        """Step to the nearest valid state ahead of the cursor. Returns whether the cursor moved.

        `apply`: as for `back`.

        Skips exactly the states `back` skipped, which is what keeps the pair inverse: both directions ask
        the same predicate, so a reader who pressed Back can always press Forward and arrive where they
        started. It falls out of using one predicate rather than being arranged.
        """
        return self._step(1, apply)

    def _step(self, direction: int, apply: Optional[Callable[[Any], bool]]) -> bool:
        """Move the cursor to the nearest valid state in `direction`, if the caller accepts it."""
        index = self._next_valid(direction)
        if index is None:
            return False
        if apply is not None and not apply(self._states[index]):
            return False
        self._cursor = index
        return True

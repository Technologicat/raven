"""A keyboard cursor for a DPG table, with the navigation interface `ThumbnailGrid` already has.

The file dialog shows one listing through two widgets — a table and a thumbnail grid — and a keypress has
to mean the same thing in both. The grid has had a cursor since it was written; this gives the table one,
under the same method names, so the key handler picks a navigator once and stops caring which view is up.

What this class owns is the *state*: which entry the cursor is on, where it goes on a keypress, and where
it lands when the listing is rebuilt underneath it. What it does not own is how a row *looks* or where it
*sits* — both arrive as callbacks, because both already depend on things this class has no business
knowing: whether a row is `..`, whether it is an unselectable file, whether it is selected, how tall the
header above it is.

The result is that nothing here imports DPG, so the rules can be tested by stating them.

The rebuild policy is `gridnav.reanchor_cursor`, shared with the grid so two views of one listing cannot
disagree about where the cursor went.
"""

__all__ = ["TableCursor"]

import logging
import threading
from collections.abc import Callable, Sequence
from typing import Any, Optional

from .gridnav import reanchor_cursor

logger = logging.getLogger(__name__)


class TableCursor:
    """Cursor state for a table of rows, one row per entry.

    Create once per table. Call `set_listing` whenever the rows are rebuilt, and the `navigate_*` methods
    from a key handler. The cursor is an index into the entries most recently given to `set_listing`.
    """

    def __init__(self, *,
                 on_paint: Callable[[int, bool], None],
                 on_scroll_into_view: Optional[Callable[[int], None]] = None,
                 page_size: Optional[Callable[[], int]] = None,
                 on_current_changed: Optional[Callable[[Optional[int]], None]] = None):
        """
        *on_paint*: ``f(idx, is_cursor)`` — draw entry `idx` as the cursor row, or as an ordinary one.
            Called for the row the cursor left and the row it arrived at, so an owner that paints by
            binding a theme has only to bind the cursor's theme or the row's usual one.

        *on_scroll_into_view*: ``f(idx)`` — bring row `idx` into view. `None` for a list short enough to be
            wholly visible, which wants no scrolling at all.

        *page_size*: ``f() -> int`` — how many rows a page key moves. `None` moves one row, which at least
            still moves.

        *on_current_changed*: ``f(idx)`` — the cursor moved, by any route. `None` when the cursor arrives
            nowhere (an empty listing).

        Geometry arrives as callbacks for the same reason painting does: it is the owner's, and guessing at
        it from here goes wrong quietly. A DPG table reports no `rect_size` at all and answers
        `get_widget_size` with its *configured* `(-1, -1)`, so arithmetic against "the view's height" reads
        -1 and scrolls on the third keypress; and a row's pitch is not the height the cells were created
        with — asking for 16 produced 18-pixel cells at a 22-pixel pitch, below a header that is itself an
        offset. Every one of those is measurable by the owner and a guess from here.
        """
        self._lock = threading.RLock()
        self._on_paint = on_paint
        self._on_current_changed = on_current_changed
        self._on_scroll_into_view = on_scroll_into_view
        self._page_size = page_size

        self._keys: list = []
        self._current: int = -1
        self._listing_key: Any = None
        # What the *user* last put the cursor on, as opposed to what the cursor is showing. A rebuild that
        # drops the cursor's entry leaves a different one under it, and the user did not choose that one;
        # keeping the two apart is what lets the cursor return when its entry comes back. Same distinction
        # as `FileGrid.set_current`.
        self._anchor_key: Any = None

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def set_listing(self, keys: Sequence[Any], *, listing_key: Any = None) -> bool:
        """Point the cursor at a freshly built set of rows, `keys` identifying them in display order.

        Call after the rows exist, since the cursor paints itself onto one of them.

        *listing_key*: what this is a listing *of* — for a directory browser, the directory path. A change
            means this is a different listing rather than a rebuild of the same one, and the cursor starts
            at the top rather than carrying a position across.

        Returns whether this was a different listing — i.e. whether the cursor started over rather than
        finding its way back. Handed back because the answer has just been worked out here and an owner
        with a policy for a fresh listing (a first row worth skipping, say) would otherwise have to
        reconstruct it from the same inputs and risk disagreeing.
        """
        with self._lock:
            same_listing = (listing_key == self._listing_key)
            self._listing_key = listing_key

            previous_anchor = self._anchor_key if same_listing else None
            previous_index = self._current if (same_listing and self._current >= 0) else None
            if not same_listing:
                self._anchor_key = None

            self._keys = list(keys)
            target = reanchor_cursor(self._keys,
                                     previous_key=previous_anchor,
                                     previous_index=previous_index)
            self._current = -1 if target is None else target
            if self._current >= 0:
                self._paint(self._current, True)
            self._notify()
            self._scroll_to_current()
            return not same_listing

    def _get_current(self) -> int:
        with self._lock:
            return self._current
    current = property(fget=_get_current, doc="Index of the row the cursor is on, or -1 when there is none.")

    def _get_current_key(self) -> Any:
        with self._lock:
            if 0 <= self._current < len(self._keys):
                return self._keys[self._current]
            return None
    current_key = property(fget=_get_current_key, doc="The key the cursor is on, or `None`.")

    def _get_is_anchored(self) -> bool:
        """Whether the cursor is where it is because someone put it there.

        True once the cursor has been *moved* — by an arrow key, or by a caller passing `anchor=True` —
        and False while it merely sits where a rebuild placed it. The distinction is what lets an owner
        tell "the entry I chose" from "the entry that happened to be under the cursor", which are the same
        position and want opposite treatment when the listing changes again.
        """
        with self._lock:
            return self._anchor_key is not None
    is_anchored = property(fget=_get_is_anchored,
                           doc="Whether the cursor's position was chosen rather than placed.")

    def set_current(self, idx: int, *, anchor: bool = True) -> None:
        """Move the cursor to row `idx`.

        *anchor*: `False` when the cursor is being *placed* rather than *moved* — by a rebuild, not by the
            user. A placed cursor does not become the entry the cursor tries to return to later.
        """
        with self._lock:
            if not (0 <= idx < len(self._keys)) or idx == self._current:
                return
            old = self._current
            self._current = idx
            if old >= 0:
                self._paint(old, False)
            self._paint(idx, True)
            if anchor:
                self._anchor_key = self._keys[idx]
            self._notify()
            self._scroll_to_current()

    # ------------------------------------------------------------------
    # Navigation — the names `ThumbnailGrid` uses, so a key handler can drive either
    # ------------------------------------------------------------------

    def navigate_next(self) -> Optional[int]:
        return self._navigate_by(1)

    def navigate_prev(self) -> Optional[int]:
        return self._navigate_by(-1)

    def navigate_row_down(self) -> Optional[int]:
        # A table row *is* an entry, so a row step and an entry step are the same move. They are separate
        # methods anyway, because the grid distinguishes them and this class exists to be interchangeable
        # with it.
        return self._navigate_by(1)

    def navigate_row_up(self) -> Optional[int]:
        return self._navigate_by(-1)

    def navigate_page_down(self) -> Optional[int]:
        return self._navigate_by(self.rows_per_page())

    def navigate_page_up(self) -> Optional[int]:
        return self._navigate_by(-self.rows_per_page())

    def navigate_first(self) -> Optional[int]:
        with self._lock:
            if not self._keys:
                return None
            self.set_current(0)
            return 0

    def navigate_last(self) -> Optional[int]:
        with self._lock:
            if not self._keys:
                return None
            last = len(self._keys) - 1
            self.set_current(last)
            return last

    def rows_per_page(self) -> int:
        """How far a page key moves. One row when the owner supplies no answer, which at least still moves."""
        if self._page_size is None:
            return 1
        try:
            return max(1, int(self._page_size()))
        except Exception as exc:  # noqa: BLE001 -- an unmeasurable view must not make the key dead
            logger.error(f"TableCursor.rows_per_page: instance 0x{id(self):x}: {type(exc)}: {exc}")
            return 1

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _navigate_by(self, delta: int) -> Optional[int]:
        with self._lock:
            if not self._keys:
                return None
            target = max(0, min(len(self._keys) - 1, self._current + delta))
            self.set_current(target)
            return target

    def _paint(self, idx: int, is_cursor: bool) -> None:
        try:
            self._on_paint(idx, is_cursor)
        except Exception as exc:  # noqa: BLE001 -- a row that cannot be painted must not stop the cursor
            logger.error(f"TableCursor._paint: instance 0x{id(self):x}: row {idx}: {type(exc)}: {exc}")

    def _notify(self) -> None:
        if self._on_current_changed is None:
            return
        try:
            self._on_current_changed(self._current if self._current >= 0 else None)
        except Exception as exc:  # noqa: BLE001 -- as above; the owner's reaction is not this class's risk
            logger.error(f"TableCursor._notify: instance 0x{id(self):x}: {type(exc)}: {exc}")

    def _scroll_to_current(self) -> None:
        if self._on_scroll_into_view is None or self._current < 0:
            return
        try:
            self._on_scroll_into_view(self._current)
        except Exception as exc:  # noqa: BLE001 -- a view that will not scroll must not strand the cursor
            logger.error(f"TableCursor._scroll_to_current: instance 0x{id(self):x}: {type(exc)}: {exc}")

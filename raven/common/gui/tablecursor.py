"""A keyboard cursor for a DPG table, with the navigation interface `ThumbnailGrid` already has.

The file dialog shows one listing through two widgets — a table and a thumbnail grid — and a keypress has
to mean the same thing in both. The grid has had a cursor since it was written; this gives the table one,
under the same method names, so the key handler picks a navigator once and stops caring which view is up.

What this class owns is the *state*: which entry the cursor is on, where it goes on a keypress, and where
it lands when the listing is rebuilt underneath it. What it deliberately does not own is how a row *looks* —
that arrives as a paint callback, because a row's appearance already depends on things this class has no
business knowing (a `..` entry, an unselectable file, a selection).

The rebuild policy is `gridnav.reanchor_cursor`, shared with the grid so two views of one listing cannot
disagree about where the cursor went.
"""

__all__ = ["TableCursor"]

import logging
import threading
from collections.abc import Callable, Sequence
from typing import Any, Optional, Union

import dearpygui.dearpygui as dpg

from . import utils as guiutils
from .gridnav import reanchor_cursor

logger = logging.getLogger(__name__)


class TableCursor:
    """Cursor state for a table of rows, one row per entry.

    Create once per table. Call `set_listing` whenever the rows are rebuilt, and the `navigate_*` methods
    from a key handler. The cursor is an index into the entries most recently given to `set_listing`.
    """

    def __init__(self, *,
                 on_paint: Callable[[int, bool], None],
                 scroll_target: Optional[Union[str, int]] = None,
                 row_height: int = 0,
                 on_current_changed: Optional[Callable[[Optional[int]], None]] = None):
        """
        *on_paint*: ``f(idx, is_cursor)`` — draw entry `idx` as the cursor row, or as an ordinary one.
            Called for the row the cursor left and the row it arrived at, so an owner that paints by
            binding a theme has only to bind the cursor's theme or the row's usual one.

        *scroll_target*: the scrolling container holding the rows, if the view should follow the cursor.
            `None` leaves scrolling alone, which is what a list short enough to be wholly visible wants.

        *row_height*: pixels per row, for turning a cursor index into a scroll offset and for sizing a page.
            Required for scrolling; ignored without a *scroll_target*.

        *on_current_changed*: ``f(idx)`` — the cursor moved, by any route. `None` when the cursor arrives
            nowhere (an empty listing).
        """
        self._lock = threading.RLock()
        self._on_paint = on_paint
        self._on_current_changed = on_current_changed
        self._scroll_target = scroll_target
        self._row_height = row_height

        self._keys: list = []
        self._current: int = -1
        self._listing_key: Any = None
        # What the *user* last put the cursor on, as opposed to what the cursor is showing. A rebuild that
        # drops the cursor's entry leaves a different one under it, and the user did not choose that one;
        # keeping the two apart is what lets the cursor return when its entry comes back. Same distinction
        # as `FileGrid.set_current`.
        self._anchor_key: Any = None
        # The scroll offset last written, not the one last read: DPG does not report a commanded scroll
        # back for a frame or more, so reading the position to decide the next one compounds the lag.
        self._commanded_y_scroll: float = 0.0

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def set_listing(self, keys: Sequence[Any], *, listing_key: Any = None) -> None:
        """Point the cursor at a freshly built set of rows, `keys` identifying them in display order.

        Call after the rows exist, since the cursor paints itself onto one of them.

        *listing_key*: what this is a listing *of* — for a directory browser, the directory path. A change
            means this is a different listing rather than a rebuild of the same one, and the cursor starts
            at the top rather than carrying a position across.
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

    def get_current(self) -> int:
        with self._lock:
            return self._current
    current = property(fget=get_current, doc="Index of the row the cursor is on, or -1 when there is none.")

    def get_current_key(self) -> Any:
        with self._lock:
            if 0 <= self._current < len(self._keys):
                return self._keys[self._current]
            return None
    current_key = property(fget=get_current_key, doc="The key the cursor is on, or `None`.")

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
        """How far a page key moves: most of a screenful, keeping a row of context.

        One row short of a full page deliberately, so paging leaves something recognisable on screen to
        read the new position against — the same rule the Librarian chat log and the Visualizer info panel
        follow. Falls back to one row when the view cannot be measured, which at least still moves.
        """
        with self._lock:
            if self._scroll_target is None or self._row_height <= 0:
                return 1
            with guiutils.nonexistent_ok():
                _, height = guiutils.get_widget_size(self._scroll_target)
                return max(1, int(height / self._row_height) - 1)
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
        """Bring the cursor row into view, moving the least that does so.

        Scrolling only when the row is actually outside the visible band is what keeps arrow navigation
        from yanking the view on every keypress: within the band, the cursor moves and the listing holds
        still, which is what a reader expects.
        """
        if self._scroll_target is None or self._row_height <= 0 or self._current < 0:
            return
        with guiutils.nonexistent_ok():
            _, height = guiutils.get_widget_size(self._scroll_target)
            if not height:
                return
            row_top = self._current * self._row_height
            row_bottom = row_top + self._row_height

            # Against the offset last commanded rather than the one DPG reports: a scroll issued on the
            # previous keypress is not visible in `get_y_scroll` yet, so reading it back would compute this
            # move from a stale position and fall a row behind on every repeat.
            view_top = self._commanded_y_scroll
            if row_top < view_top:
                new_top = row_top
            elif row_bottom > view_top + height:
                new_top = row_bottom - height
            else:
                return

            new_top = max(0.0, float(new_top))
            self._commanded_y_scroll = new_top
            dpg.set_y_scroll(self._scroll_target, new_top)

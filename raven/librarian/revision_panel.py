"""The revision history of one chat message: every stored version of it, with a way to show or delete each.

Opened by clicking a message's revision number, or with Ctrl+Shift+E for the message the keyboard mark is
on. Not modal — it takes the keys only while the focus is on one of its own rows, the pane pattern the audio
input panel uses, and passes on anything it does not claim.

The operations are the controller's; this module takes them as callables, so it needs no controller to be
built or tested.
"""

__all__ = ["DPGRevisionPanel"]

import logging
logger = logging.getLogger(__name__)

import threading
import time
from typing import Callable, Optional, Union

import dearpygui.dearpygui as dpg

from ..common.gui import animation as gui_animation
from ..common.gui import keyboardmark
from ..common.gui import tooltip as gui_tooltip
from ..common.gui import utils as guiutils
from ..common.gui.tablecursor import TableCursor

from ..vendor.IconsFontAwesome6 import IconsFontAwesome6 as fa

from . import chattree
from . import chatutil
from . import config as librarian_config

gui_config = librarian_config.gui_config

DIM_TEXT = (140, 140, 140)


class DPGRevisionPanel:
    """The "Revisions" panel for one chat message at a time.

    `datastore`: the chat datastore the revisions are read from.
    `themes_and_fonts`: the app's `guiutils.bootup` result, for the icon font.
    `show_revision`: `f(node_id, revision_id) -> str | None`, making that revision the one the chat shows.
                     Returns `None` when done, or a short reason when refused.
    `delete_revision`: `f(node_id, revision_id) -> str | None`, deleting that revision; same contract.
    `on_close`: optional zero-argument callable, run when the panel closes — to hand the keyboard back.
    `centering_reference_window`: DPG tag or ID to center on the first time the panel opens; the main
                                  window, normally. Later opens leave the panel where the user put it.
    """

    def __init__(self,
                 datastore: chattree.Forest,
                 themes_and_fonts,
                 *,
                 show_revision: Callable[[str, int], Optional[str]],
                 delete_revision: Callable[[str, int], Optional[str]],
                 on_close: Optional[Callable[[], None]] = None,
                 centering_reference_window: Optional[Union[int, str]] = None):
        self.datastore = datastore
        self.themes_and_fonts = themes_and_fonts
        self._show_revision = show_revision
        self._delete_revision = delete_revision
        self._on_close = on_close
        self.centering_reference_window = centering_reference_window

        self.is_open = False
        self.node_id = None  # the message whose revisions are listed
        self.window_id = None
        self._has_been_positioned = False
        self._build_count = 0  # for the rows' tags; DPG frees deleted items lazily
        self._seen_generation = None  # the datastore's `generation` when the list was last built
        self._rows_lock = threading.RLock()  # see `_rebuild_rows`
        self._focus_request = None  # the `give_focus` from the last open, while it may still be asking
        self._rows_take_focus = True  # see `_rebuild_rows`
        self._descriptions = None  # the revisions the rows were last built from, for `poll` to compare
        self._cursor_theme = None  # populated by `_build_window`, with the colour it pulses
        self._cursor_color = None

        # One entry per listed revision, in display order: the revision ID, the row's selectable, and its
        # delete button with that button's tooltip.
        self._rows = []
        self._cursor = TableCursor(on_paint=self._paint_row,
                                   on_current_changed=self._focus_row)
        # The delete that is waiting for its confirming second press: `(revision_id, time.monotonic())`.
        self._armed_delete = None

    # ------------------------------------------------------------------------------
    # Opening and closing

    def open(self, node_id: str) -> None:
        """Show the revisions of the message at `node_id`, with the keyboard on the one the chat shows.

        Opening it for another message while it is open switches it to that message.
        """
        if self.window_id is None:
            self._build_window()
        self.node_id = node_id
        self._armed_delete = None
        self._set_status(None)
        self.is_open = True
        self._rebuild_rows(keep_revision=self.datastore.get_revision(node_id))
        if self.centering_reference_window is not None and not self._has_been_positioned:
            dpg.split_frame()  # let anything that is closing finish first, or ours may not appear
            guiutils.recenter_window(self.window_id, reference_window=self.centering_reference_window)  # this shows it
            self._has_been_positioned = True
        else:
            dpg.show_item(self.window_id)
        # Asked for until it lands, rather than once: opened by a click on a revision number, a single request
        # did not land, where the same request from the hotkey did. Why is not established; see `dpg-notes.md`.
        if 0 <= self._cursor.current < len(self._rows):
            self._cancel_focus_request()
            self._focus_request = gui_animation.give_focus(self._rows[self._cursor.current][1])

    def _on_window_close(self) -> None:
        """The window's own close button. Wired as the window's `on_close`."""
        self.close()

    def close(self) -> None:
        """Hide the panel, and hand the keyboard back."""
        if not self.is_open:
            return
        self.is_open = False
        self._armed_delete = None
        self._cancel_focus_request()  # or it would go on pulling the focus toward a hidden row
        with guiutils.nonexistent_ok():
            dpg.hide_item(self.window_id)
        if self._on_close is not None:
            self._on_close()

    def destroy(self) -> None:
        """Tear the panel down. Reverse of the order it was built in.

        An app never needs this — its one panel lives as long as it does — but a caller that builds a
        panel and lets it go does: the cursor colour's place in the shared keyboard-mark pulse is held by
        the process-wide animator, which outlives the panel and every widget in it.
        """
        self.is_open = False
        self._cancel_focus_request()
        for _revision_id, _selectable, _delete_button, tooltip in self._rows:
            tooltip.destroy()
        self._rows = []
        if self._cursor_color is not None:
            keyboardmark.leave_pulse(self._cursor_color)
            self._cursor_color = None
        with guiutils.nonexistent_ok():
            if self.window_id is not None:
                dpg.delete_item(self.window_id)
            if self._cursor_theme is not None:
                dpg.delete_item(self._cursor_theme)
        self.window_id = None
        self._cursor_theme = None

    def toggle(self, node_id: str) -> None:
        """Close the panel if it is showing `node_id`, and show `node_id` otherwise. What the hotkey calls."""
        if self.is_open and self.node_id == node_id:
            self.close()
        else:
            self.open(node_id)

    def poll(self) -> None:
        """Refresh if the datastore has changed since the list was built. Call once per frame.

        Polled as the chat graph polls, `chattree.Forest.generation` being a counter only a mutation
        advances: an edit, a Continue or a delete can change this message's revisions from elsewhere, and
        none of them knows this panel exists.
        """
        # Only tried, never waited for. This runs on the render thread every frame, and the rows are rebuilt
        # from the callback thread too — a click shows a revision, which also advances the counter. A rebuild
        # in progress is about to leave the list current anyway, and if it does not, the next frame asks again.
        if not self._rows_lock.acquire(blocking=False):
            return
        try:
            if not self.is_open or self.datastore.generation == self._seen_generation:
                return
            # The counter moves on any change to the tree — a streaming reply moves it chunk by chunk — and
            # nearly all of them leave this message's revisions as they were. Rebuilding on each was a rebuild
            # per frame while the AI wrote, each one taking the focus.
            self._seen_generation = self.datastore.generation
            if (self.node_id in self.datastore.nodes and
                    chatutil.describe_revisions(self.datastore, self.node_id) == self._descriptions):
                return
            self.refresh()
        finally:
            self._rows_lock.release()

    def refresh(self) -> None:
        """Re-read the listed message's revisions, as after something else changed them. No-op when closed.

        Closes the panel if the message itself is gone. Takes the focus only if the panel already had it.
        """
        if not self.is_open:
            return
        if self.node_id not in self.datastore.nodes:
            self.close()
            return
        keep = self._rows[self._cursor.current][0] if 0 <= self._cursor.current < len(self._rows) else None
        self._rebuild_rows(keep_revision=keep, take_focus=self.has_keyboard())

    # ------------------------------------------------------------------------------
    # Keyboard

    def has_keyboard(self) -> bool:
        """Whether the keyboard is currently in this panel, and its keys should apply."""
        if not self.is_open:
            return False
        # The window itself counts, as well as its rows: a click on the panel's background focuses the window
        # and no row, and the keys should come back to the list with it.
        if dpg.is_item_focused(self.window_id):
            return True
        focused = dpg.get_focused_item()
        return any(focused in guiutils.item_identifiers(widget)
                   for _revision_id, selectable, delete_button, _tooltip in self._rows
                   for widget in (selectable, delete_button))

    def handle_key(self, key: int, ctrl: bool = False, shift: bool = False, alt: bool = False) -> bool:
        """Act on `key` if it is one of ours. Return whether it was taken.

        Only ever called while `has_keyboard`. Every key here is a bare one, so a modified press is declined,
        which is what lets the app's own chords go on working from inside the panel.
        """
        if ctrl or shift or alt:
            return False
        if key == dpg.mvKey_Escape:
            self.close()
        elif key == dpg.mvKey_Up:
            self._cursor.navigate_row_up()
        elif key == dpg.mvKey_Down:
            self._cursor.navigate_row_down()
        elif key == dpg.mvKey_Home:
            self._cursor.navigate_first()
        elif key == dpg.mvKey_End:
            self._cursor.navigate_last()
        elif key == dpg.mvKey_Return:
            if 0 <= self._cursor.current < len(self._rows):
                self._show(self._rows[self._cursor.current][0])
        elif key == dpg.mvKey_Delete:
            if 0 <= self._cursor.current < len(self._rows):
                self._delete(self._rows[self._cursor.current][0])
        else:
            return False
        return True

    # ------------------------------------------------------------------------------
    # Actions

    def _row_for(self, revision_id: int) -> Optional[tuple]:
        return next((row for row in self._rows if row[0] == revision_id), None)

    def _show(self, revision_id: int) -> None:
        """Make `revision_id` the one the chat shows; if it already is, close the panel.

        So Enter on the revision already shown closes, and so does a double-click, whose first click shows it.
        """
        if self.datastore.get_revision(self.node_id) == revision_id:
            self.close()
            return
        maybe_refusal = self._show_revision(self.node_id, revision_id)
        self._set_status(maybe_refusal)
        if maybe_refusal is None:
            self._rebuild_rows(keep_revision=revision_id)

    def _set_status(self, maybe_text: Optional[str]) -> None:
        """Say why the last action was refused, below the list; `None` clears it."""
        with guiutils.nonexistent_ok():
            dpg.set_value(self._status_text, maybe_text or "")
            dpg.configure_item(self._status_text, show=maybe_text is not None)

    def _delete(self, revision_id: int) -> None:
        """Delete `revision_id` on the second press within the confirmation window; ask for it on the first."""
        maybe_row = self._row_for(revision_id)
        if maybe_row is None:
            return
        _revision_id, _selectable, delete_button, delete_tooltip = maybe_row
        armed = (self._armed_delete is not None and
                 self._armed_delete[0] == revision_id and
                 time.monotonic() - self._armed_delete[1] <= gui_config.delete_confirm_duration)
        if not armed:
            self._armed_delete = (revision_id, time.monotonic())
            gui_animation.flash_delete_confirmation(button=delete_button, tooltip=delete_tooltip,
                                                    duration=gui_config.delete_confirm_duration)
            return
        self._armed_delete = None
        maybe_refusal = self._delete_revision(self.node_id, revision_id)
        self._set_status(maybe_refusal)
        if maybe_refusal is not None:
            gui_animation.flash_button(button=delete_button, tooltip=delete_tooltip, ok=False,
                                       message=maybe_refusal, duration=gui_config.acknowledgment_duration)
            return
        # The cursor stays at the same place in the list, which now holds the neighbour.
        index = self._cursor.current
        self._rebuild_rows(keep_revision=None)
        if self._rows:
            self._cursor.set_current(min(index, len(self._rows) - 1))

    # ------------------------------------------------------------------------------
    # Building

    def _build_window(self) -> None:
        """Build the panel, hidden. Built once and reused; the rows are rebuilt per message."""
        self.window_id = dpg.add_window(label="Revisions",
                                        modal=False,
                                        show=False,
                                        no_collapse=True,
                                        autosize=True,
                                        min_size=[1, 1],
                                        on_close=self._on_window_close)
        dpg.add_text("Each revision is a saved version of this message. The one the chat shows is marked.\n"
                     "Enter or a click shows a revision, and again closes this list; Delete, pressed twice,\n"
                     "deletes it; Esc closes.",
                     color=DIM_TEXT, parent=self.window_id)
        self._table = dpg.add_table(header_row=True, policy=dpg.mvTable_SizingFixedFit,
                                    borders_innerH=False, borders_outerH=False,
                                    borders_innerV=False, borders_outerV=False,
                                    parent=self.window_id)
        for label in ("", "Revision", "Written", "Opening", ""):
            dpg.add_table_column(label=label, parent=self._table)
        self._status_text = dpg.add_text("", color=(255, 140, 120), show=False, parent=self.window_id)

        # The cursor row's colour is the keyboard mark's, breathing with every other mark on screen.
        with dpg.theme() as self._cursor_theme:
            with dpg.theme_component(dpg.mvAll):
                self._cursor_color = dpg.add_theme_color(dpg.mvThemeCol_Text, keyboardmark.COLOR,
                                                         category=dpg.mvThemeCat_Core)
        keyboardmark.join_pulse(self._cursor_color)

    def _rebuild_rows(self, keep_revision: Optional[int], take_focus: bool = True) -> None:
        """Rebuild the table from the datastore, with the cursor on `keep_revision` if it is still there.

        `take_focus`: whether to put the focus on the cursor row afterwards. `False` for a rebuild nobody in
                      the panel asked for, which must not take the keyboard from wherever it is.
        """
        # Two threads rebuild this — the render thread through `poll`, the callback thread after a click or
        # a key — and two rebuilds interleaved delete each other's new rows mid-build.
        with self._rows_lock:
            self._rows_take_focus = take_focus  # read by `_focus_row`, which the cursor calls as it lands
            try:
                self._rebuild_rows_locked(keep_revision)
            finally:
                self._rows_take_focus = True

    def _rebuild_rows_locked(self, keep_revision: Optional[int]) -> None:
        for _revision_id, _selectable, _delete_button, tooltip in self._rows:
            tooltip.destroy()
        with guiutils.nonexistent_ok():
            dpg.delete_item(self._table, children_only=True, slot=1)  # the rows; slot 0 holds the columns
        self._rows = []
        self._build_count += 1
        self._seen_generation = self.datastore.generation

        descriptions = chatutil.describe_revisions(self.datastore, self.node_id)
        self._descriptions = descriptions  # what `poll` compares against
        for description in descriptions:
            revision_id = description["revision"]
            row = dpg.add_table_row(parent=self._table)
            mark = dpg.add_text(fa.ICON_EYE if description["active"] else "", parent=row)
            dpg.bind_item_font(mark, self.themes_and_fonts.icon_font_solid)
            # Not spanning the row: a selectable that did would lie over the delete button and take its clicks.
            selectable = dpg.add_selectable(label=f"R{revision_id}",
                                            callback=lambda sender, app_data, user_data: self._show(user_data),
                                            user_data=revision_id,
                                            tag=f"revision_panel_row_{revision_id}_build{self._build_count}",
                                            parent=row)
            dpg.add_text(description["datetime"], color=DIM_TEXT, parent=row)
            dpg.add_text(description["opening"] or "(no text)", parent=row)
            delete_button = dpg.add_button(label=fa.ICON_TRASH_CAN,
                                           width=gui_config.toolbutton_w,
                                           callback=lambda sender, app_data, user_data: self._delete(user_data),
                                           user_data=revision_id,
                                           parent=row)
            dpg.bind_item_font(delete_button, self.themes_and_fonts.icon_font_solid)
            delete_tooltip = gui_tooltip.Tooltip(delete_button, "Delete this revision. Press twice to confirm. It cannot be undone")
            self._rows.append((revision_id, selectable, delete_button, delete_tooltip))

        keys = [revision_id for revision_id, *_ in self._rows]
        self._cursor.set_listing(keys)
        if keep_revision is not None and keep_revision in keys:
            self._cursor.set_current_key(keep_revision)
        self._focus_row(self._cursor.current)

    def _paint_row(self, idx: int, is_cursor: bool) -> None:
        """Draw row `idx` as the cursor row, or as an ordinary one."""
        if not (0 <= idx < len(self._rows)):
            return
        with guiutils.nonexistent_ok():
            dpg.bind_item_theme(self._rows[idx][1], self._cursor_theme if is_cursor else 0)

    def _focus_row(self, maybe_idx: Optional[int]) -> None:
        """Put DPG's focus on row `maybe_idx`, so the panel keeps the keyboard as the cursor moves."""
        if not self.is_open or not self._rows_take_focus or maybe_idx is None or not (0 <= maybe_idx < len(self._rows)):
            return
        self._cancel_focus_request()  # one still asking for the row the cursor left would pull the focus back
        with guiutils.nonexistent_ok():
            guiutils.focus_item(self._rows[maybe_idx][1])

    def _cancel_focus_request(self) -> None:
        if self._focus_request is not None:
            gui_animation.animator.cancel(self._focus_request)
            self._focus_request = None

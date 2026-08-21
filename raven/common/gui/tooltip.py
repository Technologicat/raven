"""A tooltip that resizes without ever rendering a frame at the wrong size.

This module is licensed under the 2-clause BSD license, to facilitate integration anywhere.
"""

__all__ = ["Tooltip"]

import logging
logger = logging.getLogger(__name__)

import threading
from typing import Optional, Union

import dearpygui.dearpygui as dpg

from unpythonic import sym

from . import animation as gui_animation
from . import utils as guiutils

# Every tooltip that is currently on screen — membership means exactly `tooltip._shown`. The sweeper walks
# this rather than every tooltip in the app, because DPG has a hover handler but no un-hover handler:
# something has to notice that the mouse left, and the only tooltips that can need hiding are the ones
# already up. So a busy app pays for the one tooltip under the cursor, not for the two hundred it defines.
#
# `_on_hover` enrols and `_hide` removes, and `_hide` is the only way out: the sweeper calls it when the
# mouse leaves — which covers a *deleted* target too, since a missing item reports as not hovered — and
# `destroy` calls it for a tooltip being dismantled while the app keeps running.
_visible = set()

# Tooltips with a text change in flight. Separate from `_visible` because a change can be made to a tooltip
# that is *not* on screen, and it must still be settled — otherwise the next hover shows the stale size,
# which is the same glitch by a slower route.
_pending = set()

_sweep_lock = threading.Lock()  # guards both sets

# How many frames a text change is carried offscreen before the window is placed. Two, and both are needed
# for different reasons: the first is where autosize fits the window to its new content, out of sight, and
# the second is where `get_item_rect_size` catches up and starts reporting that new size.
#
# Placing on the first frame is what a reading of the autosize lag alone suggests, and it is wrong in a way
# that only shows near a viewport edge: the placement is computed from the size the tooltip *used* to be, so
# a caption that has just grown is put where the short one fitted, drawn once overflowing, and moved on the
# frame after. Measured 2026-08-20 on DPG 2.3.1 — `investigations/dpg-autosize/probe_settle_size.py` prints
# the reported size frame by frame.
_SETTLE_FRAMES = 2

class _Sweeper(gui_animation.Animation):
    def __init__(self):
        """Carries each tooltip's pending text change forward, keeps the shown ones under the cursor, and
        hides the ones the mouse has left.

        One of these serves every tooltip in the app, and it is ambient: a tooltip appearing is not the GUI
        *doing* something, so an idle-framerate throttle should not be held open by it.

        All three jobs live here because each needs a frame to have passed, and because the render loop
        already ticks the animator in every Raven app — so a tooltip needs nothing wired into the app that
        hosts it.
        """
        super().__init__(ambient=True)

    def render_frame(self, t: int) -> sym:
        with _sweep_lock:
            advancing = list(_pending)
            visible = list(_visible)
        for tooltip in advancing:
            tooltip._advance()
        for tooltip in visible:
            if tooltip.should_be_visible:
                tooltip._follow()
            else:
                tooltip._hide()
        return gui_animation.action_continue

    def finish(self) -> None:
        # `Animator.clear` finalizes everything it holds, which happens at app teardown and between tests.
        # Saying so lets the next tooltip that appears register a fresh sweeper instead of waiting forever
        # for one that is no longer running.
        global _sweeper
        _sweeper = None

_sweeper = None  # the one running instance, or `None` if nothing has needed it yet

def _ensure_sweeper() -> None:
    """Register the shared sweeper, unless it is already running."""
    global _sweeper
    if _sweeper is None:
        _sweeper = _Sweeper()
        gui_animation.animator.add(_sweeper)

class Tooltip:
    def __init__(self,
                 target: Union[str, int],
                 text: str = "",
                 *,
                 tag: Optional[str] = None,
                 wrap: int = -1,
                 offset: int | tuple[int, int] = (25, 10),
                 x_algorithm: str = "snap",
                 y_algorithm: str = "snap"):
        """A tooltip for `target`, whose text can change without a mis-sized frame.

        Use this instead of `dpg.tooltip` wherever the text changes while the tooltip may be on screen — a
        button whose tooltip caption is replaced by an acknowledgment, a status readout, anything that
        reports. A `dpg.tooltip` renders one frame at its previous size when its content changes, which
        reads as a glitch; this does not, and cannot, because it never lets that frame be seen.

        For a caption that is written once and never changes, `dpg.tooltip` is simpler and equivalent.

        **Not usable from inside a modal**, which cannot spawn a window — and this is a window, which is
        the whole reason it can be placed. A modal dialog's tooltips stay `dpg.tooltip` and keep the
        resize glitch; there is currently nothing to be done about that.

        `target`: DPG tag or ID. The widget this tooltip belongs to: it appears while the mouse is over
                  that widget, and goes away when the mouse leaves.

        `text`: the initial text. Change it later by assigning to `text`.

        `tag`: DPG tag for the tooltip window, if you want one. IDs are used internally either way.

        `wrap`: wrap width in pixels for the text, in `dpg.add_text`'s spelling: pixels from the start of
                the text until wrapping starts, or `-1` for no wrapping (the default). Note the window
                fits itself to the text, so an unwrapped tooltip is as wide as its longest line.

                `0` is not "no wrapping" — it wraps at zero pixels, which is one character per line.

        `offset`: how far from the mouse cursor to place the tooltip, in pixels: `(x, y)`, or one number
                  for both axes.

                  The default is where DPG puts its own tooltips, measured off the screen because DPG
                  reports a tooltip window's position through no API at all. Matching it is what lets a
                  tooltip migrated to this class sit exactly where the plain `dpg.tooltip` beside it does.
                  It is deliberately not square: DPG offsets further horizontally than vertically.

                  **Not decorative.** A tooltip is a separate window, so one placed under the cursor takes
                  the hover away from the widget beneath it — which is the very hover keeping the tooltip
                  open. Passing 0 makes a tooltip that flickers.

        `x_algorithm`, `y_algorithm`: how the tooltip is placed along each axis when the cursor is near an
                                      edge of the viewport. See `guiutils.compute_tooltip_position_scalar`
                                      for what each one does.

                                      Both default to `"snap"`, which places the tooltip at `offset` from
                                      the cursor whenever it fits there — the same rule on both axes, and
                                      the one DPG's own tooltips follow. What that buys beyond matching
                                      them is that the position does not depend on the tooltip's *size*,
                                      so a caption replaced by a shorter one stays where it was instead of
                                      sliding as it shrinks. `"smooth"` is the better choice where the
                                      caption is fixed and the cursor roams a large area — the plotter
                                      annotation in Raven-visualizer is the case it was written for.

        Two DPG widgets are public, for a caller that needs to paint the tooltip itself: `window` is the
        tooltip's window, and `caption` the text item inside it. `animation.flash_button` accepts a
        `Tooltip` in its `tooltip=` parameter and finds them itself, so reach for them directly only when
        building a `WidgetFlash` by hand.
        """
        self.target = target
        self.offset = (offset, offset) if isinstance(offset, int) else tuple(offset)
        self.x_algorithm = x_algorithm
        self.y_algorithm = y_algorithm
        self._shown = False
        self._pending_text = None  # queued by `text`, applied by `_advance` on the render thread
        self._settle_countdown = 0  # ...and placed once this many further frames have gone by offscreen
        self._text_lock = threading.Lock()

        with dpg.window(show=False,
                        modal=False,
                        no_title_bar=True,
                        no_collapse=True,
                        no_scrollbar=True,
                        no_resize=True,
                        no_move=True,
                        no_focus_on_appearing=True,
                        autosize=True,
                        # Without this the window is at least ~100x100 whatever it holds, and autosize will
                        # not shrink past it. A short tooltip then carries a skirt of empty window — which
                        # is worse than it looks, because a DPG window takes the mouse across its whole
                        # rect, so the skirt becomes a dead zone over whatever is beneath.
                        min_size=[1, 1],
                        **({"tag": tag} if tag is not None else {})) as self.window:
            self.caption = dpg.add_text(text, wrap=wrap)

        # DPG has `add_item_hover_handler` and no un-hover counterpart, so appearing is event-driven and
        # disappearing is swept. Both halves are needed: a handler alone never learns that the mouse left.
        with dpg.item_handler_registry() as self.handler_registry:
            dpg.add_item_hover_handler(callback=self._on_hover)
        with guiutils.nonexistent_ok():
            dpg.bind_item_handler_registry(target, self.handler_registry)

    def _get_should_be_visible(self) -> bool:
        """Whether the mouse is over `target` right now, and there is therefore a tooltip to show."""
        with guiutils.nonexistent_ok():
            return dpg.is_item_hovered(self.target)
        return False  # the target is gone, e.g. a chat view rebuilt under a hovered button

    should_be_visible = property(fget=_get_should_be_visible,
                                 doc="Whether the mouse is over `target` right now. Read-only.")

    def _get_text(self) -> str:
        """The text currently in the tooltip."""
        with guiutils.nonexistent_ok():
            return dpg.get_value(self.caption)
        return ""

    def _set_text(self, text: str) -> None:
        """Set the text, keeping the window's size correct on every frame that reaches the screen.

        Callable from any thread, including the render loop. The change is queued and applied over the next
        two frames by the sweeper, so this returns before the new text is on screen.
        """
        with self._text_lock:
            self._pending_text = text
        with _sweep_lock:
            _pending.add(self)
        _ensure_sweeper()

    text = property(fget=_get_text, fset=_set_text,
                    doc="""The tooltip's text. Assigning to it resizes the window without a visible glitch.

                        Assignable from any thread. The assignment returns before the new text is on
                        screen: making the window the right size for it, and being able to read that size
                        back, takes a frame each, so the change is applied over the next three.""")

    def _advance(self) -> None:
        """Carry a queued text change one frame further. Called by the sweeper, on the render thread.

        Autosize fits a window to the content it measured on the *previous* frame, so a window whose text
        just changed is drawn once at the old size — clipped when the text grew, skirted when it shrank.
        The way around it is to let those frames happen where nobody is looking: park the window offscreen
        and show it *there* (a hidden item is not laid out at all, so hiding it merely postpones the bad
        frame), let `_SETTLE_FRAMES` pass, and only then bring it to the cursor.

        Hence a small state machine rather than a wait. Waiting is the natural way to write this —
        `guiutils.wait_for_resize` exists for it — but the wait cannot be performed by the thread that
        renders the frames, and handing it to a worker leaves that worker inside DPG when the app tears
        down. See `investigations/dpg-autosize/`.
        """
        with self._text_lock:
            pending, self._pending_text = self._pending_text, None

        if pending is not None:
            applied = False
            with guiutils.nonexistent_ok():
                if dpg.get_value(self.caption) != pending:
                    guiutils.park_offscreen(self.window)  # offscreen, but drawn
                    dpg.set_value(self.caption, pending)
                    dpg.show_item(self.window)
                    applied = True
            if applied:
                with self._text_lock:
                    self._settle_countdown = _SETTLE_FRAMES
                return
            # Identical text, or a window that has gone away: nothing moves, so nothing needs to settle.
            # Any countdown already running is left to finish — it belongs to an earlier change that does.

        with self._text_lock:
            remaining, self._settle_countdown = self._settle_countdown, max(0, self._settle_countdown - 1)

        if remaining > 1:  # more settling to come, so renew the park: it lasts exactly one frame
            # Without this, every settling frame but the first is drawn back inside the viewport — ImGui
            # clamps a window whose position did not come through the API that frame, so the tooltip
            # appears as a 19 px corner of itself in the bottom right. See `guiutils.park_offscreen`.
            with guiutils.nonexistent_ok():
                guiutils.park_offscreen(self.window)

        if remaining == 1:  # the settle frames have been drawn; the reported size is now the new one
            with guiutils.nonexistent_ok():
                if self._shown and self.should_be_visible:
                    self._place()
                else:
                    dpg.hide_item(self.window)

        with self._text_lock:
            at_rest = self._pending_text is None and not self._settle_countdown
        if at_rest:
            with _sweep_lock:
                _pending.discard(self)

    def _follow(self) -> None:
        """Keep the tooltip at its offset from the cursor as the mouse moves, the way DPG's own do.

        Called by the sweeper on every frame a tooltip is on screen.

        A tooltip with a text change in flight is left alone: it is parked offscreen on purpose until the
        frames that resize it have been drawn, and bringing it to the cursor before then is exactly the
        mis-sized frame this class exists to avoid.
        """
        with self._text_lock:
            mid_change = self._pending_text is not None or self._settle_countdown
        if mid_change:
            return
        with guiutils.nonexistent_ok():
            self._place()

    def _place(self) -> None:
        """Position the tooltip near the cursor. The window must already be correctly sized."""
        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
        tooltip_w, tooltip_h = dpg.get_item_rect_size(self.window)
        dpg.set_item_pos(self.window,
                         [guiutils.compute_tooltip_position_scalar(algorithm=self.x_algorithm,
                                                                   cursor_pos=mouse_x,
                                                                   tooltip_size=tooltip_w,
                                                                   viewport_size=dpg.get_viewport_client_width(),
                                                                   offset=self.offset[0]),
                          guiutils.compute_tooltip_position_scalar(algorithm=self.y_algorithm,
                                                                   cursor_pos=mouse_y,
                                                                   tooltip_size=tooltip_h,
                                                                   viewport_size=dpg.get_viewport_client_height(),
                                                                   offset=self.offset[1])])

    def _on_hover(self, sender, app_data, user_data) -> None:
        """DPG GUI event handler: the mouse is over `target`. Fires every frame it stays there."""
        if self._shown:
            return
        with guiutils.nonexistent_ok():
            self._place()
            dpg.show_item(self.window)
            self._shown = True
        if not self._shown:  # the target went away between the handler firing and the placement
            return
        with _sweep_lock:
            _visible.add(self)
        _ensure_sweeper()

    def _hide(self) -> None:
        """Take the tooltip off screen. Called by the sweeper once the mouse has left `target`."""
        with guiutils.nonexistent_ok():
            dpg.hide_item(self.window)
        self._shown = False
        with _sweep_lock:
            _visible.discard(self)

    def destroy(self) -> None:
        """Release the tooltip's window and handler registry.

        Not needed when the whole GUI is going away; this is for a tooltip that outlives its usefulness
        while the app keeps running.
        """
        self._hide()
        with self._text_lock:  # a change still in flight has nowhere to land now
            self._pending_text = None
            self._settle_countdown = 0
        with _sweep_lock:
            _pending.discard(self)
        with guiutils.nonexistent_ok():
            dpg.bind_item_handler_registry(self.target, 0)
        guiutils.maybe_delete_item(self.handler_registry)
        guiutils.maybe_delete_item(self.window)

"""A tooltip that resizes without ever rendering a frame at the wrong size."""

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
_visible_lock = threading.Lock()

class _HoverSweeper(gui_animation.Animation):
    def __init__(self):
        """Hides each shown tooltip once the mouse leaves the widget it belongs to.

        One of these serves every tooltip in the app. It is ambient — a tooltip appearing is not the GUI
        *doing* something, so an idle-framerate throttle should not be held open by it.
        """
        super().__init__(ambient=True)

    def render_frame(self, t: int) -> sym:
        with _visible_lock:
            leaving = [tooltip for tooltip in _visible if not tooltip.should_be_visible]
        for tooltip in leaving:
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
        _sweeper = _HoverSweeper()
        gui_animation.animator.add(_sweeper)

class Tooltip:
    def __init__(self,
                 target: Union[str, int],
                 text: str = "",
                 *,
                 tag: Optional[str] = None,
                 wrap: int = 0,
                 offset: int = 20,
                 x_algorithm: str = "snap",
                 y_algorithm: str = "smooth"):
        """A tooltip for `target`, whose text can change without a mis-sized frame.

        Use this instead of `dpg.tooltip` wherever the text changes while the tooltip may be on screen — a
        button whose tooltip caption is replaced by an acknowledgment, a status readout, anything that
        reports. A `dpg.tooltip` renders one frame at its previous size when its content changes, which
        reads as a glitch; this does not, and cannot, because it never lets that frame be seen.

        For a caption that is written once and never changes, `dpg.tooltip` is simpler and equivalent.

        `target`: DPG tag or ID. The widget this tooltip belongs to: it appears while the mouse is over
                  that widget, and goes away when the mouse leaves.

        `text`: the initial text. Change it later by assigning to `text`.

        `tag`: DPG tag for the tooltip window, if you want one. IDs are used internally either way.

        `wrap`: wrap width in pixels for the text, or 0 for no wrapping (the default). Note the window
                fits itself to the text, so an unwrapped tooltip is as wide as its longest line.

        `offset`: how far from the mouse cursor to place the tooltip, in pixels.

                  **Not decorative.** A tooltip is a separate window, so one placed under the cursor takes
                  the hover away from the widget beneath it — which is the very hover keeping the tooltip
                  open. Passing 0 makes a tooltip that flickers.

        `x_algorithm`, `y_algorithm`: how the tooltip is placed along each axis when the cursor is near an
                                      edge of the viewport. See `guiutils.compute_tooltip_position_scalar`
                                      for what each one does; the defaults are the pairing it recommends.
        """
        self.target = target
        self.offset = offset
        self.x_algorithm = x_algorithm
        self.y_algorithm = y_algorithm
        self._shown = False
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

        Callable from any thread. Settling the new size means waiting for a frame, which cannot be done
        *by* the thread that renders them — so from the render loop this returns immediately and the
        settle happens on a worker. The text lands either way; only the moment differs.
        """
        if guiutils.is_render_thread():
            # A flash restores its message from `finish`, which the animator calls on the render thread, so
            # this is an ordinary path rather than a misuse to be reported. Handing off costs a thread per
            # change, which is affordable because the text of a tooltip changes when a person does
            # something — never per frame.
            threading.Thread(target=self._settle_text, args=(text,), daemon=True).start()
            return
        self._settle_text(text)

    def _settle_text(self, text: str) -> None:
        """Apply `text` and let the window resize to it out of sight. Never call from the render thread.

        The window is parked offscreen and shown *there* while it resizes: autosize fits a window to the
        content it measured on the previous frame, so the frame in between shows the new text at the old
        size. Hiding it across that frame does not help — a hidden item is not laid out, so the stale frame
        simply moves to whenever it is shown again. It has to be drawn somewhere, and offscreen is where
        nobody is looking. See `investigations/dpg-autosize/`.
        """
        with self._text_lock:  # two changes in flight would park, resize and place over each other
            with guiutils.nonexistent_ok():
                if dpg.get_value(self.caption) == text:
                    return  # nothing moves, so nothing needs to settle
                was_shown = self._shown
                viewport_w = dpg.get_viewport_client_width()
                viewport_h = dpg.get_viewport_client_height()
                dpg.set_item_pos(self.window, [viewport_w, viewport_h])  # offscreen, but drawn, so it resizes
                dpg.set_value(self.caption, text)
                dpg.show_item(self.window)
                guiutils.wait_for_resize(self.window)
                if was_shown and self.should_be_visible:
                    self._place()
                else:
                    dpg.hide_item(self.window)

    text = property(fget=_get_text, fset=_set_text,
                    doc="""The tooltip's text. Assigning to it resizes the window without a visible glitch.

                        Assignable from any thread. From the render loop the assignment returns before the
                        new text is on screen, since settling the size means waiting for a frame and that
                        thread is the one that would have to draw it.""")

    def _place(self) -> None:
        """Position the tooltip near the cursor and show it. The window must already be correctly sized."""
        mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
        tooltip_w, tooltip_h = dpg.get_item_rect_size(self.window)
        dpg.set_item_pos(self.window,
                         [guiutils.compute_tooltip_position_scalar(algorithm=self.x_algorithm,
                                                                   cursor_pos=mouse_x,
                                                                   tooltip_size=tooltip_w,
                                                                   viewport_size=dpg.get_viewport_client_width(),
                                                                   offset=self.offset),
                          guiutils.compute_tooltip_position_scalar(algorithm=self.y_algorithm,
                                                                   cursor_pos=mouse_y,
                                                                   tooltip_size=tooltip_h,
                                                                   viewport_size=dpg.get_viewport_client_height(),
                                                                   offset=self.offset)])
        dpg.show_item(self.window)

    def _on_hover(self, sender, app_data, user_data) -> None:
        """DPG GUI event handler: the mouse is over `target`. Fires every frame it stays there."""
        if self._shown:
            return
        with guiutils.nonexistent_ok():
            self._place()
            self._shown = True
        if not self._shown:  # the target went away between the handler firing and the placement
            return
        with _visible_lock:
            _visible.add(self)
        _ensure_sweeper()

    def _hide(self) -> None:
        """Take the tooltip off screen. Called by the sweeper once the mouse has left `target`."""
        with guiutils.nonexistent_ok():
            dpg.hide_item(self.window)
        self._shown = False
        with _visible_lock:
            _visible.discard(self)

    def destroy(self) -> None:
        """Release the tooltip's window and handler registry.

        Not needed when the whole GUI is going away; this is for a tooltip that outlives its usefulness
        while the app keeps running.
        """
        self._hide()
        with guiutils.nonexistent_ok():
            dpg.bind_item_handler_registry(self.target, 0)
        guiutils.maybe_delete_item(self.handler_registry)
        guiutils.maybe_delete_item(self.window)

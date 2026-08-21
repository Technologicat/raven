"""The blue that says where the keyboard is, shared by every widget that wears it.

Two different facts wear this colour, and they are deliberately not distinguished by it:

- **Where the cursor is** — the entry Enter would act on. Drawn as text colour by a table
  (`tablecursor`), as an inner border by a grid (`thumbnailgrid`).
- **Where the caret is parked** — which control the arrow keys are driving, in a GUI that routes them by
  hand because DPG draws nothing on a focused combo, a listing, or a panel of its own. That is what `Mark`
  is for.

One hue and one rhythm across both, because they are one idea seen from two sides: *the keyboard is here*.
A reader glancing at any Raven app should recognize it without being taught, which is also why this lives in
`raven.common.gui` rather than in whichever widget happened to need it first — the values were `thumbnailgrid`'s
until the file dialog became the second caller and a combo mark was about to be the third, at which point a
grid would have been dictating the colour of a combo's border in an app with no grid in it.

`Mark` is the second of those, and it needs no drawing at all: it recolours the widget's own edge through a
theme, so it clips, scrolls and stacks exactly as the widget does. The alternative it replaced — a rectangle
on the viewport drawlist — is unconditionally on top, which draws it over dimmers and modals that are meant
to be covering it.

Widgets that draw their own mark into a canvas they already own (`thumbnailgrid`) tick `pulsating_alpha`
against `PULSE_SECONDS` instead. Either way the rhythm is the shared one: `join_pulse` puts a theme colour
into the single animation this module runs, so a mark appearing while another is on screen falls into step
with it rather than breathing against it.

This module is licensed under the 2-clause BSD license, to facilitate integration anywhere.
"""

__all__ = ["COLOR", "PULSE_SECONDS",  # the vocabulary
           "join_pulse", "leave_pulse", "pulse_is_running",  # the one rhythm, for a widget that paints itself
           "MarkKind", "Mark", "install_focus_follower"]  # the mark as a component

import logging
logger = logging.getLogger(__name__)

import enum
import threading
from typing import Sequence, Union

from unpythonic import sym

import dearpygui.dearpygui as dpg

from . import animation as gui_animation
from . import utils as guiutils

# Names are bare because the module supplies the namespace: `keyboardmark.COLOR` at a call site, rather
# than a `KEYBOARD_MARK_` prefix repeated inside the module that already says it.
#
# **Which makes the import style load-bearing here rather than merely preferred.** Import the module —
# `from ..gui import keyboardmark` — and never the names inside it: `from ..gui.keyboardmark import COLOR`
# leaves a bare `COLOR` at every use site, saying nothing about *which* colour it is or why that widget
# should be wearing it. The bare names are readable only while the namespace is still attached to them, so
# a module whose public names are deliberately short is the one place that rule cannot be relaxed.
COLOR = (80, 160, 255, 255)

# How long one breath takes, in seconds. Shared for the same reason the colour is: two marks pulsating at
# different rates read as two things blinking at each other rather than as one mark meaning one thing.
PULSE_SECONDS = 2.0

# What an unlit mark is set to. Transparent rather than absent: a mark goes dark by having its colour
# written, which is the same operation the pulse performs every frame, so there is no second mechanism —
# and no bind/unbind path — that could leave a widget in a state neither lit nor cleanly off.
_INVISIBLE = (*COLOR[:3], 0)

# --------------------------------------------------------------------------------
# The one rhythm

_pulse_lock = threading.Lock()
_pulse = None  # gui_animation.PulsatingColor, while at least one mark is lit


def join_pulse(theme_color_widget: Union[str, int]) -> None:
    """Have `theme_color_widget` breathe with every other keyboard mark on screen.

    Starts the shared animation if this is the first widget to join it.
    """
    global _pulse
    with _pulse_lock:
        if _pulse is None:
            # Created with this widget rather than empty, because `PulsatingColor` reads the colour it
            # breathes off the first widget it is given.
            dpg.set_value(theme_color_widget, COLOR)
            _pulse = gui_animation.animator.add(
                gui_animation.PulsatingColor(cycle_duration=PULSE_SECONDS,
                                             theme_color_widget=theme_color_widget))
        else:
            _pulse.attach(theme_color_widget)


def leave_pulse(theme_color_widget: Union[str, int]) -> None:
    """Stop `theme_color_widget` breathing, and make it invisible.

    Stops the shared animation once the last widget has left it, so an app with nothing marked is not
    writing a colour nobody can see once a frame.
    """
    global _pulse
    with _pulse_lock:
        if _pulse is not None:
            _pulse.detach(theme_color_widget)
            if not _pulse.theme_color_widgets:
                gui_animation.animator.cancel(_pulse)
                _pulse = None
    # Written after leaving, not before: the animation runs on the render thread, so a colour written while
    # this widget is still attached is one the very next frame can overwrite.
    with guiutils.nonexistent_ok():
        dpg.set_value(theme_color_widget, _INVISIBLE)


def pulse_is_running() -> bool:
    """Whether the shared pulse animation is currently registered with the animator."""
    with _pulse_lock:
        return _pulse is not None


# --------------------------------------------------------------------------------
# The mark as a component

class MarkKind(enum.Enum):
    """Which shape a `Mark` takes.

    **Try them in this order — recolour, then dot** — cheapest in screen estate first. Recolouring an edge
    the widget already has costs no space at all; a dot needs a side and a gap, so a layout has to be
    willing to give it one.

    The choice is otherwise perceptual, and the quantity is *perimeter*: a pulsating outline's claim on the
    eye scales with how long it is, so framing a ten-button row puts far more motion on screen than framing
    a combo, for a mark that means exactly the same thing. A dot's claim is constant whatever it marks. So
    a long row takes `DOT` and everything else takes an edge.
    """
    FRAME = "frame"  # a framed widget's own edge: a button, a combo, an input field — or every one of them inside a group
    PANEL = "panel"  # a child window's own edge
    DOT = "dot"  # a glyph the call site placed in its own layout, coloured rather than outlined


# Which style variable gives the target an edge. They are separate variables rather than one because the
# distinction is load-bearing: a theme carrying `FrameBorderSize` bound to a panel would border every button
# inside it, and one carrying `ChildBorderSize` bound to a row of buttons would do nothing at all.
_BORDER_SIZE_STYLE = {MarkKind.FRAME: dpg.mvStyleVar_FrameBorderSize,
                      MarkKind.PANEL: dpg.mvStyleVar_ChildBorderSize}


class Mark:
    def __init__(self,
                 target: Union[str, int],
                 kind: MarkKind = MarkKind.FRAME,
                 thickness: int = 2):
        """The blue pulse that says *the keyboard is here*, on one widget, switched by `lit`.

        `target`: DPG tag or ID of the widget to mark.

                  **For `FRAME`, prefer the group around the widget over the widget itself.** DPG binds one
                  theme per item, so marking a widget that has a theme of its own would displace it; themes
                  compose down the parent chain, so marking the enclosing group supplies the border and
                  leaves that theme in place. Most call sites already have such a group.

                  Marking a container marks *every* matching descendant — every framed widget under a
                  `FRAME`, every child window under a `PANEL`. For a row of buttons that is the intent.

        `kind`: See `MarkKind`. Which of the two border styles is set, or `DOT` for a glyph that is coloured
                instead of outlined.

        `thickness`: Border width in pixels, for the two edge forms. Thin by default: the mark frames a
                     control the reader is looking at, rather than competing with it.

        A mark starts unlit, so a call site can build one alongside its widget and switch it later::

            self._mark = keyboardmark.Mark(self.places_panel, kind=keyboardmark.MarkKind.PANEL)
            ...
            self._mark.lit = (self._caret_home is CaretHome.PLACES)

        Call `detach` when the marked widget goes away; a mark holds a DPG theme and, while lit, a place in
        the shared pulse.
        """
        self.target = target
        self.kind = kind
        self._lit_now = False
        self._lock = threading.Lock()

        # Captured so `detach` can put back whatever was there. A target that already has one is a call
        # site that should have marked its group instead — say so, since the loss is otherwise silent and
        # shows up as a widget that quietly stops being styled.
        self._previous_theme = dpg.get_item_theme(target)
        if self._previous_theme is not None:
            logger.warning(f"Mark.__init__: target '{target}' already has a theme, which this mark displaces until detached. Mark the enclosing group instead, so the two compose.")

        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvAll):
                if kind is MarkKind.DOT:
                    self._color_widget = dpg.add_theme_color(dpg.mvThemeCol_Text, _INVISIBLE)
                else:
                    self._color_widget = dpg.add_theme_color(dpg.mvThemeCol_Border, _INVISIBLE)
                    dpg.add_theme_style(_BORDER_SIZE_STYLE[kind], thickness)
        self._theme = theme
        dpg.bind_item_theme(target, theme)

    def _get_lit(self) -> bool:
        """Whether this mark is currently showing."""
        return self._lit_now

    def _set_lit(self, value: bool) -> None:
        """Show or hide this mark, joining or leaving the shared pulse."""
        with self._lock:
            if value == self._lit_now:
                return
            self._lit_now = value
            if value:
                join_pulse(self._color_widget)
            else:
                leave_pulse(self._color_widget)

    lit = property(fget=_get_lit, fset=_set_lit,
                   doc="Whether this mark is showing. Setting it joins or leaves the shared pulse.")

    def detach(self) -> None:
        """Take the mark off its widget and delete the theme behind it.

        Call this when the marked widget goes away. Safe to call more than once, and safe when the widget
        is already gone.
        """
        self.lit = False
        with guiutils.nonexistent_ok():
            dpg.bind_item_theme(self.target, self._previous_theme)
        guiutils.maybe_delete_item(self._theme)


def install_focus_follower(widgets: Sequence[Union[str, int]],
                           kind: MarkKind = MarkKind.FRAME,
                           thickness: int = 2) -> gui_animation.Animation:
    """Mark whichever of `widgets` currently holds DPG's focus. One call per app.

    For the keyboard-browsable combos: an app that routes the arrow keys by asking `dpg.get_focused_item`
    already knows which control has them, and DPG draws nothing on a focused combo of its own — so the
    marking rule is the routing rule, and neither needs restating at the call site.

    `widgets`: DPG tags or IDs. A widget carrying a theme of its own wants a `Mark` on its enclosing group
               instead — see `Mark` — which this cannot express, since the widget that takes the focus and
               the one that wears the mark are then two different items.

    Returns the animation driving it, for `gui_animation.animator.cancel` if it should ever stop. The marks
    it holds are detached when it is cancelled.
    """
    # Keyed by ID, because `dpg.get_focused_item` answers with one: a tag compared against it never matches,
    # and never matching looks exactly like "nothing is focused".
    marks = {(dpg.get_alias_id(widget) if isinstance(widget, str) else widget): Mark(widget, kind=kind, thickness=thickness)
             for widget in widgets}

    class _FocusFollower(gui_animation.Animation):
        def __init__(self):
            # Ambient: it says where the keyboard is, not that anything is happening, so it must not hold
            # the frame rate up for as long as the app has a combo in it.
            super().__init__(ambient=True)

        def render_frame(self, t: int) -> sym:
            focused = dpg.get_focused_item()
            for item_id, mark in marks.items():
                mark.lit = (item_id == focused)
            return gui_animation.action_continue

        def finish(self) -> None:
            for mark in marks.values():
                mark.detach()

    return gui_animation.animator.add(_FocusFollower())

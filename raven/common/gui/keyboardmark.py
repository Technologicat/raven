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
on the viewport drawlist — sits unconditionally above every window, so anything meant to cover it has to be
answered by *hiding the mark by hand*, once per thing that can cover it. A theme needs none of that, which
is the whole reason this ended up simpler than it was specified as.

Widgets that draw their own mark into a canvas they already own (`thumbnailgrid`) tick `pulsating_alpha`
against `PULSE_SECONDS` instead. Either way the rhythm is the shared one: `join_pulse` puts a theme colour
into the single animation this module runs, so a mark appearing while another is on screen falls into step
with it rather than breathing against it.

This module is licensed under the 2-clause BSD license, to facilitate integration anywhere.
"""

__all__ = ["COLOR", "PULSE_SECONDS",  # the vocabulary
           "join_pulse", "leave_pulse", "pulse_is_running",  # the one rhythm, for a widget that paints itself
           "DOT_GLYPH", "DOT_SLOT_W", "add_dot",  # the glyph a DOT mark lights
           "MarkKind", "Mark", "install_focus_follower"]  # the mark as a component

import logging
logger = logging.getLogger(__name__)

import enum
import threading
from collections.abc import Sequence

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

# **Nothing here may call the animator while holding `_pulse_lock`**, and the reason is a lock-order
# inversion that is invisible in a single-threaded reading of either side.
#
# The animator holds its own lock across `render_frame`, so an animation that lights or darkens a mark —
# which the focus follower does on every frame — reaches this module *already holding it*: animator lock,
# then `_pulse_lock`. A mark switched from anywhere else (a key handler, a scroll poll, a background task)
# arrives the other way round: `_pulse_lock`, then the animator's lock via `add` or `cancel`. Two threads
# doing those at once is a deadlock, and one with no traceback and no log line — the GUI simply stops.
#
# So the lock guards the decision, and the animator is called after it is released.


def join_pulse(theme_color_widget: str | int) -> None:
    """Have `theme_color_widget` breathe with every other keyboard mark on screen.

    Starts the shared animation if this is the first widget to join it.
    """
    global _pulse
    with _pulse_lock:
        if _pulse is None:
            # Created with this widget rather than empty, because `PulsatingColor` reads the colour it
            # breathes off the first widget it is given.
            dpg.set_value(theme_color_widget, COLOR)
            _pulse = gui_animation.PulsatingColor(cycle_duration=PULSE_SECONDS,
                                                  theme_color_widget=theme_color_widget)
            to_register = _pulse
        else:
            _pulse.attach(theme_color_widget)
            to_register = None
    if to_register is not None:
        gui_animation.animator.add(to_register)


def leave_pulse(theme_color_widget: str | int) -> None:
    """Stop `theme_color_widget` breathing, and make it invisible.

    Stops the shared animation once the last widget has left it, so an app with nothing marked is not
    writing a colour nobody can see once a frame.
    """
    global _pulse
    to_cancel = None
    with _pulse_lock:
        if _pulse is not None:
            _pulse.detach(theme_color_widget)
            if not _pulse.theme_color_widgets:
                to_cancel = _pulse
                _pulse = None
    if to_cancel is not None:
        gui_animation.animator.cancel(to_cancel)
    # Written after leaving, not before: the animation runs on the render thread, so a colour written while
    # this widget is still attached is one the very next frame can overwrite.
    with guiutils.nonexistent_ok():
        dpg.set_value(theme_color_widget, _INVISIBLE)


def pulse_is_running() -> bool:
    """Whether the shared pulse animation is currently registered with the animator."""
    with _pulse_lock:
        return _pulse is not None


# --------------------------------------------------------------------------------
# The glyph a DOT mark lights

# A bullet in the ordinary text font rather than an icon: measured 2026-08-21 at the font size every app in
# the constellation uses, it is 6 px wide against 20 px for FontAwesome's filled circle, which read as a
# blob beside a row of 28 px buttons. It also costs no font atlas space, where a second icon font at a
# smaller size would.
#
# The other small glyphs are not options: `●` U+25CF, `▪` U+25AA and `∙` U+2219 all came back as the
# missing-glyph box, so they are outside the ranges Raven's font loads. `·` U+00B7 does render, at 4 px.
DOT_GLYPH = "•"  # U+2022 BULLET

# How much room the dot takes in a row: the glyph's 6 px plus DPG's 8 px of item spacing. A layout that
# right-aligns its buttons takes this off the aligning spacer, so adding a dot moves nothing.
DOT_SLOT_W = 14

_unlit_dot_theme = None  # created on first use by `_get_unlit_dot_theme`


def _get_unlit_dot_theme() -> str | int:
    """The theme every dot wears while its widget is not the marked one.

    One theme shared by every dot in the process, and the thing a `DOT` `Mark` displaces on whichever dot is
    current and gives back when it moves on.
    """
    global _unlit_dot_theme
    if _unlit_dot_theme is None:
        # Explicit parents rather than `with`, because this is built on *first use* and its callers build
        # widgets on background threads. The container stack is global, so a `with` here would splice this
        # theme into whatever container that thread happened to be filling.
        _unlit_dot_theme = dpg.add_theme()
        component = dpg.add_theme_component(dpg.mvAll, parent=_unlit_dot_theme)
        dpg.add_theme_color(dpg.mvThemeCol_Text, _INVISIBLE, parent=component)
    return _unlit_dot_theme


def add_dot(*,
            parent: str | int,
            tag: str | int | None = None) -> str | int:
    """Add the glyph a `DOT` `Mark` lights, unlit, at the current end of `parent`.

    `parent`: the container to add it to. Explicit rather than taken from the container stack, because the
              callers build their rows on background threads.

    `tag`: a DPG tag for the dot, or `None` (default) to let DPG assign an id.

    Returns the dot's widget id, which is what to assign to a `Mark`'s `target`.

    **Present on every candidate and invisible on all but one**, which is why this returns a widget rather
    than adding one only where the mark currently is: hiding a dot would take its width out of the row and
    repack the row's contents every time the mark moved.
    """
    kwargs = {"tag": tag} if tag is not None else {}
    dot = dpg.add_text(DOT_GLYPH, parent=parent, **kwargs)
    dpg.bind_item_theme(dot, _get_unlit_dot_theme())
    return dot


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
                 target: str | int,
                 kind: MarkKind = MarkKind.FRAME,
                 item_type: int = dpg.mvAll,
                 thickness: int = 2,
                 padding: tuple[int, int] | None = None,
                 tooltip: str | None = None):
        """The blue pulse that says *the keyboard is here*, on one widget, switched by `lit`.

        `target`: DPG tag or ID of the widget to mark.

                  **For `FRAME`, prefer the group around the widget over the widget itself.** DPG binds one
                  theme per item, so marking a widget that has a theme of its own would displace it; themes
                  compose down the parent chain, so marking the enclosing group supplies the border and
                  leaves that theme in place. Most call sites already have such a group.

                  Marking a container marks *every* matching descendant — every framed widget under a
                  `FRAME`, every child window under a `PANEL`. For a row of buttons that is the intent;
                  where it is not, narrow it with `item_type`.

        `kind`: See `MarkKind`. Which of the two border styles is set, or `DOT` for a glyph that is coloured
                instead of outlined.

        `item_type`: An `mvInputText`-style DPG item type constant, narrowing the mark to descendants of
                     that type. Defaults to `dpg.mvAll`, which is everything.

                     This is what lets a widget be marked in place, inside whatever row it already lives
                     in: the file dialog's path field shares a row with two image buttons, and
                     `item_type=dpg.mvInputText` bounds the mark to the field without a group having to be
                     invented around it. Scoping a component this way still reaches descendants of the
                     bound container — measured, since a type filter could plausibly have meant "the bound
                     item, if it is of this type".

        `thickness`: Border width in pixels, for the two edge forms. Thin by default: the mark frames a
                     control the reader is looking at, rather than competing with it.

        `padding`: `mvStyleVar_WindowPadding` for the target, or `None` (default) to leave it alone.

                   Only `PANEL` has a use for this, and it is the one place adopting a mark is not free.
                   A child window has to be created `border=True` to have an edge to recolour at all, and
                   ImGui's border flag also switches *on* `WindowPadding` — measured 8 px on every side,
                   against exactly 0 for a borderless child. So a panel converted for the sake of a mark
                   silently loses 16 px of content area in each direction unless it passes `(0, 0)` here,
                   which restores the borderless layout to the pixel.

                   Not the default, because a panel that always had a border has that padding by design and
                   would be reflowed by taking it away.

        `tooltip`: What hovering the marked widget says, or `None` (default) for no tooltip.

                   Per call site, because what the mark *means* is per call site: on a combo it says which
                   control the arrow keys are driving, and on a dot beside a row of buttons it says which
                   of many similar things the hotkeys will act on. One wording could not serve both.

                   It appears only while the mark is lit, and on whichever widget currently wears it — so a
                   mark that moves takes its tooltip along, and a widget it has left says nothing. Both
                   matter: a tooltip promising that the keys are here, on a control they are not currently
                   pointed at, is worse than none.

                   Where the marked widget has a tooltip already, this is a *second* one and they will
                   fight; give it to a call site whose widget has none. The dot exists for the mark, so it
                   is the natural place.

        A mark starts unlit, so a call site can build one alongside its widget and switch it later::

            self._mark = keyboardmark.Mark(self.places_panel, kind=keyboardmark.MarkKind.PANEL)
            ...
            self._mark.lit = (self._caret_home is CaretHome.PLACES)

        `target` may also be `None`, for a mark that **moves** — one that says which of many similar things
        the keys act on, where the many are rebuilt as the view changes and a theme apiece would be waste.
        Assign to `target` to move it; see the property.

        Call `detach` when the marked widget goes away; a mark holds a DPG theme and, while lit, a place in
        the shared pulse.
        """
        self.kind = kind
        self._target = None
        self._previous_theme = None
        self._lit_now = False
        self._tooltip_text = tooltip
        self._tooltip = None  # the DPG item, which belongs to whichever widget currently wears the mark
        self._lock = threading.RLock()  # `target` and `lit` reach each other

        # Explicit parents throughout, no `with`. A `Mark` is built whenever one is needed — mid-session,
        # and from whichever thread is building the widget it will sit on — and DPG's container stack is
        # one process-wide global shared by themes and widgets alike, so a `with` here can capture what
        # another thread is adding. See `dpg-notes.md`, "DPG parent management".
        theme = dpg.add_theme()
        component = dpg.add_theme_component(item_type, parent=theme)
        if kind is MarkKind.DOT:
            self._color_widget = dpg.add_theme_color(dpg.mvThemeCol_Text, _INVISIBLE, parent=component)
        else:
            self._color_widget = dpg.add_theme_color(dpg.mvThemeCol_Border, _INVISIBLE, parent=component)
            dpg.add_theme_style(_BORDER_SIZE_STYLE[kind], thickness, parent=component)
        # Set once and left alone, unlike the colour: it is layout, and a layout that changed with the
        # caret would make the panel's contents jump every time the keys moved.
        if padding is not None:
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, *padding, parent=component)
        self._theme = theme

        if target is not None:
            self.target = target
            # Only at construction. A widget that already has a theme is usually a call site that meant to
            # mark the group around it — themes compose down the parent chain, so that keeps both — and the
            # loss is otherwise silent, showing up as a widget that quietly stops being styled.
            #
            # A *move* says nothing, because a mark that moves between many similar widgets is the case
            # where giving them all a common theme and displacing it in turn is the intended design, and a
            # warning per move would be a warning per keystroke.
            if self._previous_theme is not None:
                logger.warning(f"Mark.__init__: target '{target}' already has a theme, which this mark displaces until detached. Mark the enclosing group instead, so the two compose.")

    def _get_target(self) -> str | int | None:
        """Which widget currently wears this mark, or `None`."""
        return self._target

    def _set_target(self, target: str | int | None) -> None:
        """Move the mark to another widget, giving the old one back the theme it had."""
        with self._lock:
            if target == self._target:
                return
            # Darkened first, so that a mark being moved is never briefly lit on the widget it is leaving.
            # Assigning `None` is how a caller says *nothing is current*, and a lit mark bound to nothing
            # would be a pulse animating a colour no widget is wearing.
            if target is None:
                self.lit = False
            if self._target is not None:
                with guiutils.nonexistent_ok():
                    dpg.bind_item_theme(self._target, self._previous_theme)
            # Rebuilt at the new target rather than moved, since which item a tooltip belongs to is fixed
            # when it is created and is not readable afterwards.
            #
            # **The mark has to delete it, because nothing else will.** A DPG tooltip is not a child of the
            # item it describes — measured 2026-08-21: it lands as a *sibling* in that item's window — so
            # deleting the target leaves the tooltip behind, and a caller tearing down a subtree with a
            # children-only delete does not reach it either. Same story as the `Tooltip` windows Librarian
            # tracks by hand in `owned_tooltips`.
            self._destroy_tooltip()
            self._target = target
            self._previous_theme = None
            if target is not None:
                self._previous_theme = dpg.get_item_theme(target)
                dpg.bind_item_theme(target, self._theme)
                self._build_tooltip()

    target = property(fget=_get_target, fset=_set_target,
                      doc="""Which widget wears this mark. Assign to move it; `None` takes it off and darkens it.

                          The widget it leaves gets back whatever theme it had, so the common shape for a
                          moving mark is to give every candidate one theme saying *not me* and let the mark
                          displace it in turn.""")

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
            # A tooltip that outlived the mark would promise the keyboard is somewhere it is not, which is
            # worse than saying nothing — so it comes and goes with the mark rather than with the widget.
            if self._tooltip is not None:
                with guiutils.nonexistent_ok():
                    dpg.configure_item(self._tooltip, show=value)

    lit = property(fget=_get_lit, fset=_set_lit,
                   doc="Whether this mark is showing. Setting it joins or leaves the shared pulse.")

    def _build_tooltip(self) -> None:
        """Give the current target the mark's tooltip, if it was asked for one."""
        if self._tooltip_text is None or self._target is None:
            return
        with guiutils.nonexistent_ok():
            # Hidden unless the mark is currently lit — `_set_lit` is what shows it, and a mark moved while
            # lit needs the new tooltip to arrive already showing.
            self._tooltip = dpg.add_tooltip(self._target, show=self._lit_now)
            dpg.add_text(self._tooltip_text, parent=self._tooltip)

    def _destroy_tooltip(self) -> None:
        """Take the tooltip off whichever widget has it."""
        if self._tooltip is None:
            return
        guiutils.maybe_delete_item(self._tooltip)
        self._tooltip = None

    def detach(self) -> None:
        """Take the mark off its widget and delete the theme behind it.

        Call this when the marked widget goes away. Safe to call more than once, and safe when the widget
        is already gone.
        """
        self.target = None  # which darkens it and gives the widget back its theme
        guiutils.maybe_delete_item(self._theme)


def install_focus_follower(widgets: Sequence[str | int],
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
    # Compared against *both* names DPG may answer with, since `get_focused_item` gives a tagged widget's
    # alias and an untagged one's ID. See `guiutils.item_identifiers`.
    marks = [(guiutils.item_identifiers(widget), Mark(widget, kind=kind, thickness=thickness)) for widget in widgets]

    class _FocusFollower(gui_animation.Animation):
        def __init__(self):
            # Ambient: it says where the keyboard is, not that anything is happening, so it must not hold
            # the frame rate up for as long as the app has a combo in it.
            super().__init__(ambient=True)

        def render_frame(self, t: int) -> sym:
            focused = dpg.get_focused_item()
            for identifiers, mark in marks:
                mark.lit = (focused in identifiers)
            return gui_animation.action_continue

        def finish(self) -> None:
            for _identifiers_, mark in marks:
                mark.detach()

    return gui_animation.animator.add(_FocusFollower())

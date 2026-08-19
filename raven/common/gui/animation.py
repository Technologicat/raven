"""General framework for DPG GUI animations."""

__all__ = ["Animator", "animator",  # controller and its global instance (need only one per app)
           "Animation", "Overlay",  # base classes
           "Dimmer",  # overlays
           "WidgetFlash", "flash_button", "highlight_widget", "set_text_under_flash",  # the flash animation, its two conveniences, and writing to a widget one has borrowed
           "SmoothScrolling", "PulsatingColor",  # animations
           "ScrollEndFlasher", "WHEEL_SETTLE_FRAMES",  # animated overlay, and the wheel-settle delay it uses
           "pulsation_envelope", "pulsating_alpha",  # utilities: the curve pulsating animations follow, and the alpha it yields
           "action_continue", "action_finish", "action_cancel"]  # return values for `render_frame`

import logging
logger = logging.getLogger(__name__)

import math
import threading
import time
from typing import Callable, Optional, Tuple, Union

from unpythonic import box, sym

from ..smoothvalue import SmoothInt, CALIBRATION_FPS

import dearpygui.dearpygui as dpg

from .. import numutils

from . import utils as guiutils

# --------------------------------------------------------------------------------
# Animation mechanism

action_continue = sym("continue")  # keep rendering
action_finish = sym("finish")  # end animation, call the `finish` method
action_cancel = sym("cancel")  # end animation without calling the `finish` method

class Animator:
    def __init__(self):
        """A simple animation manager.

        Raven's customized render loop calls our `render_frame` once per frame,
        to update any animations that are running.

        Use the global `animator` instance from this module; don't create your own.
        """
        self._animations = []
        self._lock = threading.RLock()

    def add(self, animation: "Animation") -> "Animation":
        """Register a new `Animation` instance, so that our `render_frame` will call its `render_frame` method.

        Its start time `animation.t0` is set automatically to the current time as returned by `time.monotonic_ns()`.

        For convenience, returns `animation`.
        """
        with self._lock:
            animation.reset()  # set the animation start time
            self._animations.append(animation)
        return animation

    def cancel(self, animation: "Animation", finalize: bool = True) -> "Animation":
        """Terminate a running `Animation` instance.

        `animation`: One of the animations registered using `add`.

        `finalize`: If `True` (default), call the `finish` method of the animation before removing it.

                    In some special cases, it can be useful to set this to `False` to reduce flicker,
                    if the old animation is immediately replaced by a new one of the same type,
                    targeting the same GUI element (so no need to hide/re-show).

        Note that when an animation finishes normally, it is automatically removed. This is meant for
        immediately stopping and removing an animation that has not finished yet.

        For convenience, returns `animation`.
        """
        with self._lock:
            if finalize:
                animation.finish()
            try:
                self._animations.remove(animation)  # uses default comparison by `id()` since `Animation` has no `__eq__` operator
            except ValueError:  # not in list
                logger.debug(f"Animator.cancel: specified {animation} is not in the animation registry (maybe already finished?), skipping removal.")
        return animation

    def render_frame(self) -> None:
        """Render one frame of each registered animation, in the order they were registered.

        Each animation whose `render_frame` returns `action_finish` is considered finished.
        Each finished animation gets its `finish` method called automatically.

        After all registered animations have had a frame rendered, the animation registry is updated
        to remove any animations that are no longer running.

        This should be called in your DPG render loop::

            while dpg.is_dearpygui_running():
                animator.render_frame()
                dpg.render_dearpygui_frame()
        """
        with self._lock:
            time_now = time.monotonic_ns()
            running_animations = []
            for animation in self._animations:
                action = animation.render_frame(t=time_now)
                if action is action_continue:
                    running_animations.append(animation)
                elif action is action_finish:
                    animation.finish()
                elif action is action_cancel:
                    pass  # when cancelled, do nothing, just remove the animation
                else:
                    raise ValueError(f"Animator.render_frame: animation {animation} returned unknown action {action}, expected one of the `raven.common.gui.animation.action_X` constants (where X is 'continue', 'finish', or 'cancel').")
            self._animations.clear()
            self._animations.extend(running_animations)

    @property
    def active_count(self) -> int:
        """Number of currently registered (running) animations."""
        with self._lock:
            return len(self._animations)

    @property
    def transient_count(self) -> int:
        """Number of running animations that report the GUI is *doing* something. See `Animation.ambient`.

        This is what an app's idle-framerate throttle should ask. `active_count` counts a pulsating
        indicator too, so an app that watched it would run at full frame rate for as long as one was on
        screen — which, for an indicator that pulsates whenever its widget exists, is for ever.

        Throttling does not stop an ambient animation, it coarsens it: the render loop keeps drawing at its
        idle rate, so a cycle measured in seconds still reads as a pulsation at a dozen frames a second.
        """
        with self._lock:
            return sum(1 for animation in self._animations if not animation.ambient)

    def clear(self) -> None:
        """Terminate all registered animations and clear the list of registered animations.

        To terminate a specific animation (by object instance), see `cancel`.
        """
        with self._lock:
            for animation in self._animations:
                animation.finish()
            self._animations.clear()
animator = Animator()

class Animation:
    def __init__(self, ambient: bool = False):
        """Base class for Raven's GUI animations.

        An `Animation` can be added to an `Animator`.

        `ambient`: Whether this animation belongs to the GUI's resting state instead of reporting that
                   something is happening. Read by `Animator.transient_count`, which is what an app's
                   idle-framerate throttle should ask.

                   Ambience is a property of the use, not of the class: the same pulsation is ambient as an
                   indicator that cycles for as long as its widget exists, and transient when it is used to
                   say "look here, this just changed".
        """
        super().__init__()
        # Keep this simple to avoid ravioli code.
        # `t0` and `ambient` should be pretty much the only attributes defined in the base class.
        self.ambient = ambient
        self.reset()

    def reset(self) -> None:
        """Semantically: (re-)start the animation from the beginning.

        Technically, in this base class: Set the animation start time `self.t0` to the current time,
        as given by `time.monotonic_ns()`.
        """
        self.t0 = time.monotonic_ns()

    def render_frame(self, t: int) -> sym:
        """Override this in a derived class to render one frame of your animation.

        `t`: time at start of current frame as returned by `time.monotonic_ns()`.

        The animation start time is available in `self.t0`.

        It is also allowed to write to `self.t0`, e.g. for a cyclic animation
        so as not to lose float accuracy in long sessions.

        Return value must be one of:
            `action_continue` if the animation should continue,
            `action_finish` if the animation should end, automatically calling its `finish` method.
            `action_cancel` if the animation should end, *without* calling its `finish` method
                                      (useful if the animation determined it didn't need to start,
                                       e.g. if another copy was already running on the same GUI element).

        The animator automatically removes (from its animation registry) any animations that return
        anything other than `action_continue`.
        """
        return action_finish

    def finish(self) -> None:
        """Override this in a derived class, if you need to clean up any state for your animation when it finishes normally."""

# --------------------------------------------------------------------------------
# Overlay window support

class Overlay:
    def __init__(self, target: Union[str, int], tag: str):
        """Base class for Raven's overlay windows (currently the dimmer, and the scroll end animation).

        `target`: DPG ID or tag. The child window for which to build the overlay.
        `tag`: DPG tag, for naming the overlay.
        """
        super().__init__()
        # Keep this simple to avoid ravioli code.
        # `target`, `tag` and `overlay_update_lock` should be pretty much the only attributes defined in the base class.
        self.target = target
        self.tag = tag
        self.overlay_update_lock = threading.Lock()

# --------------------------------------------------------------------------------
# Overlays

class Dimmer(Overlay):
    def __init__(self, target: Union[str, int], tag: str, color: Tuple = (0, 0, 0, 128), rounding: int = 8):
        """Dimmer for a child window. Can be used e.g. to indicate that the window is updating.

        `target`: DPG ID or tag. The child window for which to build the overlay.
        `tag`: DPG tag, for naming the overlay.
        `color`: tuple, RGB or RGBA, in any format accepted by DPG.
        `rounding`: window rounding radius, in pixels (match this to your theme).
        """
        super().__init__(target, tag)
        self.window = None
        self.drawlist = None
        self.color = color
        self.rounding = rounding

    def build(self, rebuild: bool = False) -> None:
        # Ensure stuff we depend on is initialized before we try to create this
        if dpg.get_frame_count() < 10:
            return

        with self.overlay_update_lock:  # This prevents a crash upon hammering F11 (toggle fullscreen) while the info panel is updating (causing lots of rebuilds)
            if not rebuild and (self.window is not None):  # Avoid unnecessary rebuilding
                return

            # We want rounding on each side (like window rounding),
            # so we must make the overlay window `2 * rounding` pixels larger in each direction.
            config = dpg.get_item_configuration(self.target)
            w = config["width"]
            h = config["height"]
            w += 2 * self.rounding
            h += 2 * self.rounding

            # Child windows don't have a `rect_min`; instead, they have `pos`.
            pos = dpg.get_item_pos(self.target)
            # Center the overlay on the target. Now this window covers the target child window.
            pos = [pos[0] - self.rounding, pos[1] - self.rounding]

            if self.window is None:  # create only once ("rebuild" here actually means "reconfigure")
                logger.debug(f"Dimmer.build: frame {dpg.get_frame_count()}: instance '{self.tag}' creating overlay")
                with dpg.window(show=False, modal=False, no_title_bar=True, tag=self.tag,
                                pos=pos,
                                width=w, height=h,
                                min_size=[1, 1],  # DPG's ~[100, 100] floor clamps explicit sizes too, and would overflow a small target
                                no_collapse=True,
                                no_focus_on_appearing=True,
                                # no_bring_to_front_on_focus=True,  # for some reason, prevents displaying the window at all
                                no_resize=True,
                                no_move=True,
                                no_background=True,
                                no_scrollbar=True,
                                no_scroll_with_mouse=True) as self.window:
                    self.drawlist = dpg.add_drawlist(width=w, height=h)
                rebuild = True

            if rebuild:
                logger.debug(f"Dimmer.build: frame {dpg.get_frame_count()}: instance '{self.tag}' updating drawlist")
                dpg.delete_item(self.drawlist, children_only=True)
                dpg.configure_item(self.window, width=w, height=h)
                dpg.configure_item(self.drawlist, width=w, height=h)
                dpg.draw_rectangle((0, 0), (w - 2 * self.rounding, h - 2 * self.rounding),
                                   color=self.color,
                                   fill=self.color,
                                   rounding=self.rounding,
                                   parent=self.drawlist)

    def show(self) -> None:
        """Dim the target window (e.g. to show that it is updating)."""
        self.build()
        with guiutils.nonexistent_ok():
            dpg.show_item(self.window)

    def hide(self) -> None:
        """Un-dim the target window."""
        with guiutils.nonexistent_ok():
            dpg.hide_item(self.window)

# --------------------------------------------------------------------------------
# Animations

class WidgetFlash(Animation):
    # For some animation types, such as this one, for any given GUI element, at most one instance
    # of the animation should be active at a time.
    #
    # Thus we need some instance management. We handle this as follows.
    #
    # An instance of a given animation type only becomes *reified* if it's the only one on that
    # GUI element (at the point in time when the new instance is being created).
    #
    # Only a reified instance actually starts animating.
    #
    # If the instance cannot be reified (i.e. there is already a previous instance on the same GUI element),
    # it enters *ghost mode*, where it only updates the existing instance (in some way appropriate for the
    # specific animation type), and then exits at the next frame.
    class_lock = threading.RLock()
    id_counter = 0  # for generating unique DPG IDs
    instances = {}  # DPG tag or ID (of `target`) -> animation instance

    # TODO: We could also customize `__new__` to return the existing instance, see `unpythpnic.symbol.sym`.
    def __init__(self,
                 message: str,
                 target: Union[str, int],
                 target_tooltip: Union[str, int],
                 target_text: Union[str, int],
                 duration: float,
                 message_duration: Optional[float] = None,
                 flash_color: Tuple = (96, 128, 96),
                 text_color: Tuple = (180, 255, 180)):
        """Animation to flash a GUI widget (and its tooltip, if visible) to draw the user's attention.

        Two uses, sharing one transient fade-out shape. As an *acknowledgment*, it tells the user that
        pressing a button actually took, when the action has no other immediately visible effect. As a
        *highlight*, it draws the eye to the widget a navigation jump just landed on.

        Which visual channel is animated depends on what `target` is, because the two kinds of widget have
        nothing in common to fade:

          - A **text widget** has no background, so its own text color is faded from `text_color` back to
            whatever it was. Set per-widget (`configure_item`), which is where an `add_text(color=...)` keeps
            its color — a theme's text color would not override that.
          - **Anything else** is treated as a button-like widget: an animated theme fades its background
            (and its tooltip's popup background) from `flash_color` back to the default.

        Each GUI element (determined by `target`) can only have one `WidgetFlash` animation running at a
        time. If an instance already exists, trying to create the animation will restart the existing
        instance instead (and update its message to `message`).

        `message`: str, text to show in the `target_text` widget while the animation is running.
                   Original content will be restored automatically when the animation finishes normally.
                   Can be `None` for "don't change", or also when `target_text is None`.

        `target`: DPG tag or ID, the widget to animate. A text widget flashes its text color; anything else
                  flashes its background. Whatever theme it had is restored when the flash ends.

        `target_tooltip`: DPG tag or ID, the tooltip to animate (by flashing its background).
                          Can be `None`.

        `target_text`: DPG tag or ID, the text widget to animate (by changing the text content,
                       and the text color, for the duration of the animation). Can be `None`.

                       The text can be inside the tooltip (when `target_tooltip is not None`),
                       but is really completely independent of `target` and `target_tooltip`.

                       It may also *be* `target`, which is how a status line says something in the color
                       that says it: the line's own text color fades while the message stands in it.

        `duration`: float, how long the flash itself takes, in seconds — the fade from `flash_color` or
                    `text_color` back to what the widget had.

        `message_duration`: float, how long `message` stays, in seconds. `None` (the default) is "as long
                            as the flash", which is what an acknowledgement wants: the fade and the word
                            are one gesture.

                            A *report* wants them apart. The fade is what catches the eye, so it has to be
                            quick; the message has to be readable, so it has to stay. Give the fade its
                            second and the message its three, and the flash holds the faded color until
                            the message has had its dwell.

        `flash_color`: tuple `(R, G, B)`, each component in [0, 255]. Default is light green.
                       The background color a button-like `target` starts from. Unused for a text `target`.

        `text_color`: tuple `(R, G, B)`, each component in [0, 255]. Default is light green.
                      For a button-like `target`, the (constant) text color during the flash. For a text
                      `target`, the color its text starts from before fading back to its own.

                      **A text `target` needs a color of its own to fade back to** — give it one at
                      `add_text(color=...)`. DPG reports an unset color as the sentinel `r = -1` rather
                      than as the theme's color, which it does not expose, so a widget that never
                      declared one has no destination and the fade runs toward negative, i.e. to black.
        """
        super().__init__()
        self.instance_lock = threading.Lock()

        self.message = message
        self.target = target
        self.target_tooltip = target_tooltip
        self.target_text = target_text
        self.duration = duration
        self.message_duration = message_duration
        self.flash_color = flash_color
        self.text_color = text_color

        # These are used during animation
        self.theme = None
        self.original_message = None
        self.target_is_text = False  # set in `start`; selects which visual channel is animated
        self.original_target_color = None  # for a text target: its own color, to fade back to
        # Whatever was bound before we bound ours. One snapshot per widget, because these are three independent
        # widgets that each may or may not have had a theme: restoring a single shared snapshot to all of them
        # hands two of them a theme belonging to the third.
        self.original_target_theme = None
        self.original_tooltip_theme = None
        self.original_text_theme = None
        self.reified = False  # `True`: running; `False`: ghost mode, update other instance and exit.

        self.start()

    def _total_duration(self) -> float:
        """How long this flash runs: the fade, or the message's dwell where that is asked to be longer."""
        if self.message_duration is None:
            return self.duration
        return max(self.duration, self.message_duration)

    def render_frame(self, t: int) -> sym:
        if not self.reified:  # ghost mode
            return action_cancel

        dt = (t - self.t0) / 10**9  # seconds since t0

        # The flash ends when the fade does, unless a message is asked to outstay it — then the fade
        # completes and holds while the message finishes its dwell. Clamped rather than run past 1, so the
        # color sits at the widget's own for the remainder instead of overshooting it.
        if dt >= self._total_duration():
            return action_finish

        r = numutils.clamp(dt / self.duration)
        r = numutils.nonanalytic_smooth_transition(r)

        if self.target_is_text:
            # The target may be deleted mid-flash (a chat view rebuild does exactly this), and then there is
            # nothing left to fade. Ending here rather than pressing on lets `finish` release the resources.
            if not dpg.does_item_exist(self.target):
                return action_finish
            R0, G0, B0 = self.text_color
            R1, G1, B1, A1 = self.original_target_color  # the item's own color; no guessing needed here
            R = R0 * (1.0 - r) + R1 * r
            G = G0 * (1.0 - r) + G1 * r
            B = B0 * (1.0 - r) + B1 * r
            with guiutils.nonexistent_ok():
                dpg.configure_item(self.target, color=(R, G, B, A1))
            return action_continue

        R0, G0, B0 = self.flash_color
        R1, G1, B1 = 45, 45, 48  # default button background color  TODO: read from global theme
        R = R0 * (1.0 - r) + R1 * r
        G = G0 * (1.0 - r) + G1 * r
        B = B0 * (1.0 - r) + B1 * r
        dpg.set_value(self.highlight_button_color, (R, G, B))
        dpg.set_value(self.highlight_hovered_color, (R, G, B))
        dpg.set_value(self.highlight_active_color, (R, G, B))
        dpg.set_value(self.highlight_popupbg_color, (R, G, B))
        dpg.set_value(self.highlight_disabled_button_color, (R, G, B))
        dpg.set_value(self.highlight_disabled_hovered_color, (R, G, B))
        dpg.set_value(self.highlight_disabled_active_color, (R, G, B))

        return action_continue

    def start(self) -> None:
        """Internal method, called automatically by constructor.

        Manages de-duplication (when added to the same GUI element as an existing animation of this type)
        as well as resource allocation. The resources are released by `finish` (called by `Animator`
        when the animation ends).
        """
        with self.instance_lock:
            if self.reified:  # already running (avoid double resource allocation and registration)
                self.reset()
                return

            with type(self).class_lock:
                # If an instance is already running on this GUI element, just restart it (and update its message).
                if self.target in type(self).instances:
                    other = type(self).instances[self.target]
                    # `message=None` is "don't change", so a flash joining one already running leaves the
                    # words where they are and takes only the restart.
                    if self.message is not None:
                        other.message = self.message
                        if other.target_text is not None:
                            with guiutils.nonexistent_ok():
                                # The instance being joined may have started with no message of its own, in
                                # which case it captured nothing and restores nothing — and ours would stand
                                # on that line for good.
                                if other.original_message is None:
                                    other.original_message = dpg.get_value(other.target_text)
                                dpg.set_value(other.target_text, other.message)
                    other.reset()
                    return

                # Which visual channel to animate. A text item has no background to flash, so the text color
                # is all there is; anything else is treated as button-like. Read from DPG rather than declared
                # by the caller, so that existing call sites need no new argument.
                self.target_is_text = False
                with guiutils.nonexistent_ok():
                    self.target_is_text = dpg.get_item_type(self.target).endswith("mvText")

                if self.target_is_text:
                    # `get_item_configuration` reports color as normalized floats while `configure_item` takes
                    # 0-255, so scale on the way in; the round trip is then exact, including the `r = -1`
                    # sentinel DPG uses for "no explicit color" (which restores to unset, as it should).
                    with guiutils.nonexistent_ok():
                        self.original_target_color = [255.0 * c for c in dpg.get_item_configuration(self.target)["color"]]
                    if self.original_target_color is None:  # target vanished between construction and here
                        return
                    # The message is independent of which visual channel fades, so a text target carries one
                    # too. Only the text is taken: the color here is this branch's own business, and binding
                    # the flash theme to `target_text` as well would fight the fade where the two widgets
                    # are the same one — which is the case this exists for, a status line saying something
                    # went wrong in the color that says so.
                    if self.target_text is not None and self.message is not None:
                        with guiutils.nonexistent_ok():
                            self.original_message = dpg.get_value(self.target_text)
                            dpg.set_value(self.target_text, self.message)
                    type(self).instances[self.target] = self
                    self.reified = True
                    return

                with dpg.theme(tag=f"acknowledgement_highlight_theme_{type(self).id_counter}") as self.theme:  # create unique DPG ID each time
                    with dpg.theme_component(dpg.mvAll):
                        # common
                        dpg.add_theme_color(dpg.mvThemeCol_Text, self.text_color)
                        # button
                        self.highlight_button_color = dpg.add_theme_color(dpg.mvThemeCol_Button, self.flash_color)
                        self.highlight_hovered_color = dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, self.flash_color)
                        self.highlight_active_color = dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, self.flash_color)
                        # tooltip
                        self.highlight_popupbg_color = dpg.add_theme_color(dpg.mvThemeCol_PopupBg, self.flash_color)
                    # Button in disabled state (see also "disablable_widget_theme" in `raven.common.gui.utils`)
                    disabled_color = (0.50 * 255, 0.50 * 255, 0.50 * 255, 1.00 * 255)
                    with dpg.theme_component(dpg.mvButton, enabled_state=False):
                        dpg.add_theme_color(dpg.mvThemeCol_Text, disabled_color, category=dpg.mvThemeCat_Core)
                        self.highlight_disabled_button_color = dpg.add_theme_color(dpg.mvThemeCol_Button, self.flash_color, category=dpg.mvThemeCat_Core)
                        self.highlight_disabled_hovered_color = dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, self.flash_color, category=dpg.mvThemeCat_Core)
                        self.highlight_disabled_active_color = dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, self.flash_color, category=dpg.mvThemeCat_Core)
                type(self).id_counter += 1

                # Capture what was bound before we take the widget over, so `finish` puts back exactly that.
                # `get_item_theme` returns `None` for an unbound widget and `bind_item_theme` accepts `None`
                # to unbind, so the round trip is symmetric and needs no special case for "had no theme".
                with guiutils.nonexistent_ok():
                    self.original_target_theme = dpg.get_item_theme(self.target)
                    dpg.bind_item_theme(self.target, self.theme)
                if self.target_tooltip is not None:
                    with guiutils.nonexistent_ok():
                        self.original_tooltip_theme = dpg.get_item_theme(self.target_tooltip)
                        dpg.bind_item_theme(self.target_tooltip, self.theme)
                if self.target_text is not None:
                    with guiutils.nonexistent_ok():
                        self.original_message = dpg.get_value(self.target_text)
                        self.original_text_theme = dpg.get_item_theme(self.target_text)
                        if self.message is not None:  # `None`: flash the text's color, leave its words alone
                            dpg.set_value(self.target_text, self.message)
                        dpg.bind_item_theme(self.target_text, self.theme)

                type(self).instances[self.target] = self
                self.reified = True  # This is the instance that animates `self.target`.

    def finish(self) -> None:
        """Clean up resources upon the end of the animation."""
        with self.instance_lock:
            # A ghost instance allocated nothing and registered nothing, so it has nothing to release — and
            # must not act as though it had. The normal route never brings one here (`render_frame` returns
            # `action_cancel`, which the animator handles without calling `finish`), but `Animator.clear` and
            # `Animator.cancel(finalize=True)` finalize every registered animation indiscriminately. Without
            # this guard a ghost would rebind the theme of a widget its reified twin is still animating, and
            # pop that twin out of `instances` — after which the next flash on the same widget would reify a
            # second instance and bind over the first.
            if not self.reified:
                return

            if self.target_is_text:
                with guiutils.nonexistent_ok():
                    dpg.configure_item(self.target, color=self.original_target_color)
                if self.original_message is not None:  # `None`: there was no message to put back
                    with guiutils.nonexistent_ok():
                        dpg.set_value(self.target_text, self.original_message)
                    self.original_message = None
                self.original_target_color = None
                self.reified = False
                with type(self).class_lock:
                    type(self).instances.pop(self.target, None)
                return

            # Restore whatever the widget had, rather than assuming it was the standard button theme. Binding
            # a fixed theme here would silently *give* one to a widget that had none, so a flash would leave a
            # visible mark on the thing it was only supposed to draw attention to.
            with guiutils.nonexistent_ok():
                dpg.bind_item_theme(self.target, self.original_target_theme)

            if self.target_tooltip is not None:
                with guiutils.nonexistent_ok():
                    dpg.bind_item_theme(self.target_tooltip, self.original_tooltip_theme)

            if self.target_text is not None:
                with guiutils.nonexistent_ok():
                    if self.original_message is not None:  # `None`: there was no message to put back
                        dpg.set_value(self.target_text, self.original_message)
                    dpg.bind_item_theme(self.target_text, self.original_text_theme)

            with guiutils.nonexistent_ok():
                dpg.delete_item(self.theme)

            self.theme = None
            self.reified = False

            with type(self).class_lock:
                type(self).instances.pop(self.target, None)

def flash_button(*,
                 button: Union[str, int],
                 message: str,
                 duration: float,
                 tooltip: Union[str, int, None] = None,
                 text: Union[str, int, None] = None,
                 ok: bool = True) -> None:
    """Flash a button as a non-intrusive acknowledgment of an action — green for success, red for failure.

    Convenience wrapper over `WidgetFlash` and the shared `animator`: it picks the success/failure colors from
    `ok`, so call sites don't repeat that. This is the standard way to confirm a button press whose effect
    isn't otherwise immediately visible (a copy, a folder opened elsewhere), and to report that such an action
    failed without a modal dialog.

    `button`: the button to flash (DPG tag or ID).
    `message`: text shown in `text` for the flash duration, then restored (`None` leaves the text unchanged).
    `duration`: flash duration in seconds.
    `tooltip`: the button's tooltip to flash along with it, if any (`None` to flash the button alone).
    `text`: the text widget whose content becomes `message` during the flash — typically the text inside
            `tooltip`, but independent of it.
    `ok`: `True` (default) flashes green (success); `False` flashes red (failure). The green matches
          `WidgetFlash`'s own default colors, so a plain success acknowledgment need not think about color.
    """
    animator.add(WidgetFlash(message=message,
                             target=button,
                             target_tooltip=tooltip,
                             target_text=text,
                             duration=duration,
                             flash_color=((96, 128, 96) if ok else (150, 96, 96)),
                             text_color=((180, 255, 180) if ok else (255, 180, 180))))

def highlight_widget(*,
                     widget: Union[str, int],
                     duration: float,
                     color: Tuple = (255, 255, 255)) -> None:
    """Flash `widget` to show the user where a navigation jump landed.

    The sibling of `flash_button`: same transient fade, different job. `flash_button` answers "your click
    took"; this answers "here is the thing you asked to see", for a widget the user did not press and which
    would otherwise be indistinguishable from its neighbours after the view scrolls to it.

    `widget`: the widget to flash (DPG tag or ID). A text widget flashes its text color back to its own;
              anything else flashes its background.
    `duration`: flash duration in seconds.
    `color`: the color to flash from, fading back to the widget's own. White by default — a navigation
             landing is not a success or failure report, so it deliberately avoids the green/red that
             `flash_button` uses to mean exactly that.
    """
    animator.add(WidgetFlash(message=None,
                             target=widget,
                             target_tooltip=None,
                             target_text=None,
                             duration=duration,
                             flash_color=color,
                             text_color=color))

def set_text_under_flash(widget: Union[str, int], text: str) -> None:
    """Set `widget`'s text, whoever currently owns it.

    Where a `WidgetFlash` is showing a message over `widget`, `text` becomes what it restores when it ends,
    and the message on screen is left to finish saying what it says. Otherwise this is `dpg.set_value`.

    For a status line whose text is *derived* from app state and rewritten whenever that state moves. A
    plain write during a flash would wipe the message in the moment it is meant to be read, and be undone
    anyway a second later — the flash putting back the value it captured, which by then names something
    that has since moved on.
    """
    # Ghosts need no handling: only a reified instance is ever registered, so this can only return the
    # animation actually on the widget. One arriving *later* is harmless too — it restarts the running
    # instance and rewrites its message, leaving `original_message` alone, which is the value written here.
    flash = WidgetFlash.instances.get(widget)
    if flash is not None:
        with flash.instance_lock:
            # `reified` is a *timing* guard, and a live one: `finish` runs on the render thread while this
            # runs on whichever thread the GUI event came in on, so the flash can end — and unregister
            # itself — between the lookup above and the lock here. It has then already restored, and
            # writing through would drop the value on the floor.
            #
            # `original_message is None` means this flash is not holding the widget's text at all — it may
            # be flashing only a color — so there is nothing to write through to, and the widget is ours.
            if flash.reified and flash.original_message is not None:
                flash.original_message = text
                return
    dpg.set_value(widget, text)

# --------------------------------------------------------------------------------

class SmoothScrolling(Animation):
    class_lock = threading.RLock()
    instances = {}  # DPG tag or ID (of `target_child_window`) -> animation instance

    def __init__(self,
                 target_child_window: Union[str, int],
                 target_y_scroll: int,
                 smooth: bool = True,
                 smooth_step: float = 0.8,
                 flasher: Optional["ScrollEndFlasher"] = None,
                 finish_callback: Optional[Callable] = None,
                 commanded_y_scroll: Optional[box] = None):
        """Scroll a child window, optionally smoothly.

        **Prefer the `scroll` classmethod**, which is the supported way to start one of these. Constructing
        an instance is inert - it packages a request and does nothing with it. `scroll` then takes the class
        lock once, hands the request to whichever instance will run it, and registers that one with the
        animator.

        Each GUI element (determined by `target_child_window`) can only have one `SmoothScrolling`
        animation running at a time. If one is already running on it, `start` **retargets** that instance
        rather than starting a second animation, and this instance becomes an inert ghost.
        Retargeting adopts the new request *whole* - see `start` for exactly what that means and why.

        `target_child_window`: DPG tag or ID, the child window to scroll.
        `target_y_scroll`: int, target scroll position in scrollbar coordinates.
        `smooth`: bool.
                  If `True`, will animate a smooth scroll.
                  If `False`, will jump to target position instantly (the point is to offer the same API).
        `smooth_step`: float, a nondimensional rate in the half-open interval (0, 1].
                       Independent of the render FPS.
        `flasher`: `ScrollEndFlasher` instance, optional.
                   Automatically activated when the top/bottom is reached.

                   Pass `None` for scrolls the *program* initiates, such as following a streaming reply's
                   tail. The flasher asserts "you tried to go further and could not", which is a statement
                   about a user's thwarted intent; automatic tail-following has no thwarted intent, since
                   reaching the end is the whole point. Passing one anyway strobes the overlay once per
                   arriving chunk.
        `finish_callback`: 0-argument callable. Run some custom code when the animation finishes normally.
                           Keep it minimal; trying to instantiate a new scroll animation will block while
                           the callback is running (because a new instance might target the same GUI element,
                           and we guarantee the teardown to be atomic).

                           It says "the animation object has ended", not "my scroll request completed" -
                           those differ, because a retarget re-aims the running instance rather than
                           replacing it. So a caller holding a reference to the instance still holds a live
                           one after someone else retargets it, and this fires only when the object really
                           goes away. Retargeting *chains* callbacks rather than replacing them; see `start`.

                           A callback that raises is logged and skipped, and does not stop the others or
                           the deregistration.
        `commanded_y_scroll`: `unpythonic.box`, optional. Receives every scroll position this animation
                              writes, in the same breath as the `dpg.set_y_scroll` that writes it.

                              For callers that need to distinguish "the view moved because we moved it"
                              from "the user scrolled". That test compares the panel's reported position
                              against the last value *written*, and an animation writes a new one every
                              frame - so a caller holding only its own pre-animation value would read the
                              entire scroll as a user scroll.

                              The caller owns the box because the animation does not outlive its own scroll:
                              `finish` pops the instance, and the check is needed precisely in the gaps when
                              no animation exists. There is still exactly one writer at a time, so the value
                              cannot drift.

                              Compare against the last written value, **not** the target: those come apart
                              exactly while an animation runs, which is the case in question.

        Note that mouse wheel and scrollbar dragging do not invoke the scroll animation; for those,
        the scroll position is handled internally by DPG. Hence those don't cause a flash here.
        If you want, you can handle the mouse wheel case separately in a global mouse wheel callback.

        This is pretty sophisticated, to make the movement smooth, but also keep things working as expected
        when the target position changes on the fly.

        The interpolation is delegated to `SmoothInt` from `raven.common.smoothvalue`,
        which handles FPS-corrected exponential decay with subpixel tracking.
        """
        super().__init__()
        self.instance_lock = threading.Lock()

        self.target_child_window = target_child_window
        self.target_y_scroll = target_y_scroll
        self.smooth = smooth
        self.smooth_step = smooth_step
        self.flasher = flasher
        self.commanded_y_scroll = commanded_y_scroll

        # A list rather than one callable, because retargeting an existing animation chains the new
        # request's callback onto the surviving instance instead of replacing it - see `start`.
        self.finish_callbacks = [finish_callback] if finish_callback is not None else []

        # How far the most recent frame moved the view, in pixels. Published because a caller comparing the
        # panel's reported position against what it commanded needs to know how much of any difference this
        # animation could have caused: the report lags the last written value by one frame, so a legitimate
        # gap of exactly one step appears while a scroll is in flight. Early in an exponential decay one step
        # is large - hundreds of pixels - so a fixed tolerance sized for a human's small scroll will read a
        # fast animation as user input.
        self.last_step = 0.0

        self.prev_frame_new_y_scroll = None  # target position of last frame, for monitoring of stuck animation
        self.update_pending_frames = 0
        self._sv = SmoothInt(value=0, rate=smooth_step)
        self.reified = False  # `True`: running; `False`: ghost mode, update other instance and exit.

    @classmethod
    def scroll(cls,
               target_child_window: Union[str, int],
               target_y_scroll: int,
               smooth: bool = True,
               smooth_step: float = 0.8,
               flasher: Optional["ScrollEndFlasher"] = None,
               finish_callback: Optional[Callable] = None,
               commanded_y_scroll: Optional[box] = None) -> "SmoothScrolling":
        """Scroll `target_child_window` to `target_y_scroll`. Returns the animation that will do it.

        This is how a scroll is started. Arguments are the constructor's, which see.

        Safe to call while a scroll is already running on the same window: the running animation adopts the
        new request whole and keeps its subpixel position, so the movement bends toward the new target
        instead of jumping.

        The returned instance is always the one actually animating, so a caller may keep it in order to
        stop that scroll later. It may be an instance an *earlier* caller created, because a retarget
        re-aims the running animation rather than replacing it - which is why `finish_callback`, and not
        the return value, is what tells a holder its reference has died.
        """
        with cls.class_lock:
            request = cls(target_child_window=target_child_window,
                          target_y_scroll=target_y_scroll,
                          smooth=smooth,
                          smooth_step=smooth_step,
                          flasher=flasher,
                          finish_callback=finish_callback,
                          commanded_y_scroll=commanded_y_scroll)
            request.start()
            if request.reified:
                return animator.add(request)
            # A ghost: `start` gave the request to the instance already running on this window, and that one
            # is registered with the animator already. The lookup cannot miss - `finish` takes `class_lock`
            # to deregister, and we are holding it.
            return cls.instances[target_child_window]

    @classmethod
    def stop(cls, target_child_window: Union[str, int]) -> None:
        """Abandon the scroll running on `target_child_window`, if any. Idempotent.

        **The view stays where the glide had got to**; the remaining distance is not travelled. To instead
        complete the movement at once, retarget with `smooth=False` - a retarget adopts the new request
        wholesale, so it jumps to the target and ends. Keeping those separate is the point: "jump there
        now" is expressible either way, and "give up here" would not be.

        Finish callbacks run, so a caller holding a reference is told the object died.
        `commanded_y_scroll` needs no fixing up - it carries every position written, so it already names
        where the view was left.

        Use this rather than calling `finish` on an instance. `finish` performs *this class's* teardown -
        it deregisters from `instances` and runs the finish callbacks - but leaves the animation registered
        with the animator, which goes on calling its `render_frame`, which goes on scrolling. Only
        `Animator.cancel` removes it from that list. Measured 2026-08-14; both spellings look like "stop"
        and only one is.

        A caller holding the instance may equivalently call `animator.cancel(instance)` itself; this exists
        for callers that have only the window.
        """
        with cls.class_lock:
            instance = cls.instances.get(target_child_window)
            if instance is not None:
                animator.cancel(instance)

    def _set_y_scroll(self, new_y_scroll: int) -> None:
        """Move the scrollbar, and record where we put it.

        The pair belongs in one place. `commanded_y_scroll` exists so a caller can tell our writes from the
        user's, and a bare `dpg.set_y_scroll` anywhere else in this class would be invisible to that caller
        and therefore read as a user scroll.

        The box is written *before* DPG, deliberately. Either order leaves a brief window for a reader on
        another thread, but they are not equally bad: box-first means the caller may see a position DPG has
        not applied yet, which is the ordinary lagging-by-a-frame state the comparison already tolerates,
        whereas DPG-first means it may see a movement it has no record of commanding - which is exactly the
        signature of a user scroll, and the one reading we must never produce by accident.
        """
        if self.commanded_y_scroll is not None:
            self.commanded_y_scroll << new_y_scroll
        dpg.set_y_scroll(self.target_child_window, new_y_scroll)

    def render_frame(self, t: int) -> sym:
        if not self.reified:  # ghost mode
            return action_cancel

        update_pending_threshold = 4  # Frames. Smaller threshold looks better, but may fire prematurely if a GUI update takes too many frames.
        action = action_continue

        with self.instance_lock:
            current_y_scroll = dpg.get_y_scroll(self.target_child_window)
            if not self.smooth:
                if current_y_scroll == self.target_y_scroll:
                    action = action_finish
                    # If we reach the start or the end of the scrollable, flash it.
                    if self.flasher is not None:
                        self.flasher.show_by_position(self.target_y_scroll)
                # First frame in this scroll? -> do it
                elif self.prev_frame_new_y_scroll is None:
                    new_y_scroll = self.target_y_scroll
                    self.prev_frame_new_y_scroll = new_y_scroll  # No longer first frame (in non-smooth mode, doesn't matter what value we store here as long as it's not `None`).
                    self._set_y_scroll(new_y_scroll)
                # Waited for a short timeout? -> time to check for end of scrollbar (but this shouldn't happen now that `scroll_info_panel_to_position` clamps the value to the max allowed by the scrollbar)
                elif self.update_pending_frames >= update_pending_threshold:
                    action = action_finish
                    if current_y_scroll != self.target_y_scroll:
                        logger.debug(f"SmoothScrolling.render_frame: frame {dpg.get_frame_count()}: instance for '{self.target_child_window}': did not reach target position (target position past end of scrollbar?)")
                        if self.flasher is not None:
                            self.flasher.show(where="bottom")
                # Waiting for the timeout?
                else:
                    self.update_pending_frames = self.update_pending_frames + 1
            else:
                # Only proceed if DPG has actually applied our previous update, or if this is the first update since this scroll animation was started.
                # This prevents stuttering, as well as keeps our subpixel calculations correct.
                if self.prev_frame_new_y_scroll is None or current_y_scroll == self.prev_frame_new_y_scroll:
                    self.update_pending_frames = 0

                    # First frame: sync SmoothInt from DPG's current scroll position.
                    if self.prev_frame_new_y_scroll is None:
                        self._sv.set_immediate(current_y_scroll)
                        self._sv.target = self.target_y_scroll

                    # Advance the interpolation by one frame.
                    # Use DPG's averaged frame rate for dt (more stable than wall-clock delta).
                    fps = dpg.get_frame_rate()
                    dt = 1.0 / fps if fps > 0 else 1.0 / CALIBRATION_FPS  # `get_frame_rate` reads 0 until its average has samples, i.e. at startup
                    still_animating = self._sv.update(dt=dt)
                    new_y_scroll = self._sv.current
                    self.last_step = abs(new_y_scroll - current_y_scroll)  # how far this frame moved the view; see the attribute's note in `__init__`

                    logger.debug(f"SmoothScrolling.render_frame: frame {dpg.get_frame_count()}: instance for '{self.target_child_window}': old raw = {current_y_scroll}, subpixel = {self._sv.current_exact}, new int = {new_y_scroll}, target = {self.target_y_scroll}")

                    if not still_animating:
                        new_y_scroll = self._sv.target
                        action = action_finish
                        logger.debug(f"SmoothScrolling.render_frame: frame {dpg.get_frame_count()}: instance for '{self.target_child_window}': scrolling completed")

                        # If we reach the start or the end of the scrollable, flash it.
                        if self.flasher is not None:
                            self.flasher.show_by_position(self.target_y_scroll)

                    if action is action_continue:
                        self.prev_frame_new_y_scroll = new_y_scroll

                    self._set_y_scroll(new_y_scroll)

                # Timeout waiting for DPG to update the position? -> probably end of scrollbar (but shouldn't happen now that `scroll_info_panel_to_position` clamps the value to the max allowed by the scrollbar)
                elif self.update_pending_frames >= update_pending_threshold:
                    action = action_finish
                    logger.info(f"SmoothScrolling.render_frame: frame {dpg.get_frame_count()}: instance for '{self.target_child_window}': timeout waiting for the scrollbar to reach the position we wrote ({self.prev_frame_new_y_scroll}); it is at {current_y_scroll}, target was {self.target_y_scroll}. DPG clamps a write to the scrollable range as it stands, so this is what a request past the end looks like. Giving up and recording where it actually is.")
                    # Record where the panel *is*, not where we asked it to go. Leaving the record on an
                    # unreachable value is not cosmetic: a caller comparing the two - `should_follow_tail`
                    # does - reads the difference as the user having scrolled, and there is nothing left
                    # running to correct it afterwards, because this branch is the animation giving up.
                    # Observed rather than predicted, deliberately: `get_y_scroll_max` cannot be trusted at
                    # the moment of a write (it reads 0 before layout), so the position DPG reports here is
                    # the only reliable statement of what the write achieved.
                    # See `investigations/follow-tail-drift/`.
                    if self.commanded_y_scroll is not None:
                        self.commanded_y_scroll << current_y_scroll
                    if self.flasher is not None:
                        self.flasher.show(where="bottom")

                # Waiting for DPG to update the position?
                else:
                    self.update_pending_frames = self.update_pending_frames + 1

        return action

    def start(self) -> None:
        """Hand this instance's request to whichever animation will run it. Called by `scroll`.

        Manages de-duplication (when added to the same GUI element as an existing animation of this type).

        When an animation is already running on this GUI element, the running instance **adopts this
        request whole** and this instance becomes an inert ghost. Adopting the destination alone is not
        enough: the surviving instance is only a vehicle, whereas `smooth`, `smooth_step`, `flasher` and
        `commanded_y_scroll` are properties of *this* scroll request and describe how it should look and
        who asked for it.

        Getting that wrong is not subtle in practice. One user-initiated scroll creates an instance
        carrying a flasher; if every subsequent tail-follow merely retargets it, the flasher stays attached
        and the overlay strobes once per streamed chunk for the length of the reply. The mirror case loses
        a wanted flash instead: a follow instance in flight when the user clicks jump-to-latest would
        swallow the confirmation. "Latest wins, wholesale" is also simply an easier rule to hold than one
        with a carve-out in it.

        `finish_callback` is the exception: it **chains** rather than being replaced. The reason is that it
        does not describe a request at all - it means "this animation object has ended", and a retarget
        ends nothing. The running instance persists and is merely re-aimed, so a caller holding a reference
        to it still holds a live one. Visualizer depends on exactly that: it keeps `_scroll_animation`
        pointing at the reified instance in order to stop *that* animation before swapping the info panel's
        content, and its callback exists to null the reference once the object dies. Firing it at handover
        would null a reference that is still live, and the stop would then silently stop nothing.

        Replacing it is equally wrong in the mirror direction: the outgoing caller would never be told, and
        Visualizer's reference would dangle at a finished instance - whose `finish` pops from `instances`
        without a default, so the next stop attempt raises `KeyError`.

        So every caller that asked to be notified is notified, once, when the instance actually finishes.
        Callbacks are deduplicated by identity, which bounds the chain by the number of *distinct* callers
        rather than by the number of retargets - the difference matters because a streaming reply retargets
        once per arriving chunk. That dedup is not merely a guard: a callback meaning "it ended" should fire
        once even if its caller retargeted ten times.
        """
        with self.instance_lock:
            if self.reified:  # already running (avoid double resource allocation and registration)
                return

            with type(self).class_lock:
                # If an instance is already running on this GUI element, retarget it rather than starting a
                # second animation. The seamless transition is the point: the surviving instance keeps its
                # subpixel position, so the movement bends toward the new target instead of jumping.
                if self.target_child_window in type(self).instances:
                    other = type(self).instances[self.target_child_window]
                    with other.instance_lock:
                        other.target_y_scroll = self.target_y_scroll
                        other._sv.target = self.target_y_scroll

                        other.smooth = self.smooth
                        other.smooth_step = self.smooth_step
                        other._sv.rate = self.smooth_step  # the field and the interpolator's copy move together
                        other.flasher = self.flasher
                        other.commanded_y_scroll = self.commanded_y_scroll

                        for callback in self.finish_callbacks:
                            if callback not in other.finish_callbacks:
                                other.finish_callbacks.append(callback)
                    return

                type(self).instances[self.target_child_window] = self
                self.reified = True  # This is the instance that animates `self.target_child_window`.

    def finish(self) -> None:
        """Call any registered finish callbacks, and clean up resources upon the end of the animation.

        There may be several: retargeting chains each new request's callback onto the surviving instance
        rather than replacing it, so everyone who asked to be told is told here, in registration order.

        **A ghost finishing is a no-op.** A ghost owns nothing - its request was handed to the running
        instance and its own callbacks were chained there - so acting as though it owned something would
        evict the instance that is actually animating, and run those callbacks a second time. This is
        reachable: `Animator.clear` finalizes every registered animation, and a caller that writes
        `animator.add(SmoothScrolling(...))` registers whichever of the two it got.

        **Deregistration happens before the callbacks, not after.** A callback is explicitly allowed to
        start a new scroll animation; if this instance were still registered at that moment, the new request
        would retarget a corpse and then be discarded by the very `pop` that follows. Popping first means it
        reifies properly instead. The atomicity the class promises is unaffected - `class_lock` is held
        throughout, so another thread still blocks until teardown completes.

        A callback that raises does not take the others with it. Teardown has already completed by then,
        which is the ordering that makes this safe rather than merely tidy.
        """
        with type(self).class_lock:
            if type(self).instances.get(self.target_child_window) is not self:  # ghost, or already finished
                return
            type(self).instances.pop(self.target_child_window)

            for callback in self.finish_callbacks:
                try:
                    callback()
                except Exception as exc:
                    logger.warning(f"SmoothScrolling.finish: instance for '{self.target_child_window}': finish callback {callback} raised, continuing teardown: {type(exc)}: {exc}")

# The FPS-corrected exponential decay math is now in `raven.common.smoothvalue` (which see).
# For the full derivation, see `raven.server.modules.avatar.interpolate`.

# --------------------------------------------------------------------------------
# Pulsation envelope

def pulsating_alpha(t0: int, now: int, cycle_duration: float) -> int:
    """The alpha a pulsation is at right now. For pulsating something that has no theme color.

    `t0`, `now`: as `time.monotonic_ns()`. `t0` is when the cycle last started.
    `cycle_duration`: seconds for one complete cycle, which starts and ends fully opaque.

    Returns the alpha, an int in [64, 255].
    """
    # Shared with `PulsatingColor` so that two marks meaning the same thing breathe together. A drawn shape
    # is out of that animation's reach — DPG draw items take a colour, not a theme — so the alternative is
    # a second copy of this arithmetic, drifting a shade off it.
    cycle_pos = ((now - t0) / 10**9) / cycle_duration
    cycle_pos = cycle_pos - float(int(cycle_pos))  # fractional part; raw position in the cycle
    a_base = 64
    return a_base + int((255 - a_base) * pulsation_envelope(cycle_pos))


def pulsation_envelope(t: float) -> float:
    """Cosine-squared envelope: 1 at *t*=0, 0 at *t*=0.5, 1 at *t*=1.

    Used by `PulsatingColor` and available for manual alpha calculations
    (e.g. compare mode tile fade). *t* is the normalized cycle position
    [0, 1], where 0 and 1 are cycle boundaries.

    The curve provides smooth acceleration and deceleration (slow at
    the extremes, fast in the middle).
    """
    return math.cos(t * math.pi) ** 2

# --------------------------------------------------------------------------------

class PulsatingColor(Animation):
    def __init__(self,
                 cycle_duration: float,
                 theme_color_widget: Union[str, int],
                 ambient: bool = True):
        """A simple cyclic animation to pulsate a color by varying its alpha.

        `cycle_duration`: seconds, for one complete cycle.
                          A cycle starts and ends with full alpha (fully opaque).

        `ambient`: See `Animation`. Defaults to `True` here, unlike the base class: this animation never
                   ends on its own, and every use of it in Raven is an indicator that cycles for as long as
                   its widget is on screen. Pass `False` for a use that is not one — the frame rate is then
                   held up while it plays. The transient "look here, this just changed" role belongs to
                   `WidgetFlash`.

        `theme_color_widget`: DPG tag or ID of the theme color to pulsate.
                              This is parameterized so you can create your own and
                              attach it to any theme component you want.

                              May also be a list or tuple of them, which then pulsate as one — same color,
                              same phase, one animation. That is what a mark drawn by more than one theme
                              needs: a table cursor whose cells wear different alignments needs a theme
                              variant per alignment, and they are one cursor.

                              The color is shared, not per widget: `self.rgb` holds it, and assigning to
                              that attribute recolors every widget this animation drives (which is how
                              `raven-conference-timer` changes its pause glow while it runs).

        Example::

            with dpg.theme(tag="my_pulsating_red_text_theme"):
                with dpg.theme_component(dpg.mvAll):
                    pulsating_red_color = dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 96, 96))

            pulsating_red_animation = PulsatingColor(cycle_duration=2.0,
                                                     theme_color_widget=pulsating_red_color)
            animator.add(pulsating_red_animation)

        Then bind the theme "my_pulsating_red_text_theme" to the GUI widget(s) that you want
        to have the pulsating red text color.

        This animation, once created, runs continuously in the background.

        To reset the pulsation sequence, call the `reset` method. This is useful e.g. when an
        indicator icon bound to the theme appears, so that the pulsation animation always starts
        at the same animation frame.
        """
        super().__init__(ambient=ambient)
        self.cycle_duration = cycle_duration
        # A DPG widget is a tag or an ID, so anything that is neither a `str` nor an `int` is a collection
        # of them. Normalized here so that the render loop has one shape to write to.
        self.theme_color_widgets = ([theme_color_widget] if isinstance(theme_color_widget, (str, int))
                                    else list(theme_color_widget))
        self.rgb = dpg.get_value(self.theme_color_widgets[0])[:3]  # get the initial RGB color

    def render_frame(self, t: int) -> sym:
        if (t - self.t0) / 10**9 > self.cycle_duration:  # prevent loss of accuracy in long sessions
            self.reset()

        alpha = pulsating_alpha(self.t0, t, self.cycle_duration)
        color = (*self.rgb, alpha)
        for theme_color_widget in self.theme_color_widgets:
            dpg.set_value(theme_color_widget, color)

        return action_continue

# --------------------------------------------------------------------------------
# Animated overlay

# Frames to wait after a mouse-wheel event before asking where the scroll ended up. DPG applies the wheel
# scroll internally, during the frame, so a wheel handler runs too early to see the result.
WHEEL_SETTLE_FRAMES = 1


class _WheelSettleCheck(Animation):
    """One-shot: a frame after the wheel moved a scrollable, ask its flasher where the scroll landed.

    Registered with the animator rather than driven by a per-app tick, so a view gets this from the
    animator its app already runs, with nothing added to any render loop. See
    `ScrollEndFlasher.note_wheel_scroll`.
    """

    def __init__(self, flasher: "ScrollEndFlasher"):
        super().__init__()
        self.flasher = flasher
        self.remaining = WHEEL_SETTLE_FRAMES

    def render_frame(self, t: int) -> sym:
        self.remaining -= 1
        if self.remaining > 0:
            return action_continue
        self.flasher.show_by_position(dpg.get_y_scroll(self.flasher.target))  # tag
        return action_finish


# Inherit from `Overlay` first, so that `super().__init__(...)` passes its arguments where we want it to.
# Then the `super().__init__()` call inside `Overlay.__init__` will initialize the `Animation` part.
class ScrollEndFlasher(Overlay, Animation):
    def __init__(self, *,
                 target: Union[str, int],
                 tag: str,
                 duration: float,
                 font: Union[str, int],
                 text_top: str,
                 text_bottom: str,
                 custom_finish_pred: Optional[Callable] = None):
        """Flasher to indicate when the end of a scrollable area has been reached.

        **It has two jobs, and both are deliberate.** It announces *arrival* at an end, and it marks a
        *refused* request to go further. Firing on arrival is not noise to be trimmed: half the point is to
        show the wall as you reach it, so nobody has to walk into it once to discover it is there. A design
        that flashed only on refusal was tried in the thumbnail grid on 2026-08-14 and rejected on sight -
        it also made that widget inconsistent with the other two.

        **Budget one trigger per way the user can move the view, and check that list rather than assuming
        the scroll animation covers it.** `SmoothScrolling` only knows about movements it performs, so it
        covers exactly the paths that go through it:

        - *Keys and programmatic scrolls* go through it, so pass it a `flasher` and both jobs are done: it
          flashes when a scroll lands on an end, and a refused scroll is a scroll landing on the end it is
          already at.
        - *The mouse wheel does not.* DPG scrolls a child window internally, so no animation exists and
          nothing fires. **This is the one that gets missed**, because the view visibly scrolls and looks
          handled. Hook the wheel and call `show_by_position` with the window's current scroll position -
          Visualizer's info panel and `raven.common.gui.thumbnailgrid` both do. Note the position reads
          *before* DPG applies the event, so this catches a wheel turned while already at an end.
        - *A cursor whose scrolling merely follows it* does not either, on the refusal side: a cursor
          clamped at the last row requests no scroll for the animation to see. Arrival still comes from the
          scroll. `thumbnailgrid` is the worked example.

        Librarian's chat log has only the first of the three, which is why it needs only the `flasher`.

        `target`: DPG ID or tag. The child window for which to build the overlay.
        `tag`: DPG tag, for naming the overlay.
        `duration`: float, fadeout animation duration in seconds.
        `font`: DPG id or tag. A font to use to render `text_top` and `text_bottom`.
        `text_top`: Text (or symbol) shown when cannot scroll further up.
        `text_bottom`: Text (or symbol) shown when cannot scroll further down.
        `custom_finish_pred`: optional 1-arg callable, must return `bool`.
                              The argument is the `ScrollEndFlasher` instance.
                              Called just before rendering a frame.
                              If the function returns `True`, the animation finishes
                              (and triggers cleanup), without rendering the current frame.
        """
        super().__init__(target, tag)

        self.duration = duration
        self.font = font
        self.text_top = text_top
        self.text_bottom = text_bottom
        self.custom_finish_pred = custom_finish_pred

        self.window_top = None
        self.drawlist_top = None
        self.window_bottom = None
        self.drawlist_bottom = None

        self.animation_running = False
        self.where = None  # the kind of the currently running animation: "top", "bottom", or "both"

    def show_by_position(self, target_y_scroll: int) -> Optional[str]:
        """Like `show`, but determine position automatically.

        `target_y_scroll`: int, target scroll position, in scrollbar coordinates of `self.target`.
                           Special value -1 means the end position.

        The scroll position is parameterized to allow animated scrolling to work;
        if you get it from the scrollbar (`dpg.get_y_scroll(some_child_window)`),
        the value may be out of date if it is being updated during the current frame.

        This allows dispatching the flashing animation immediately, without waiting
        for one frame (or sometimes several frames; see source code of `SmoothScrolling`)
        for the scrollbar position to update.

        Returns which end was flashed (one of "top", "bottom", "both"),
        or `None` if `target_y_scroll` was not at either end.

        **With less than a screenful the answer is "both", and that is a decision, not a shortcut.** It has
        now been arrived at twice, so here is the argument. A directional answer is not available in this
        case: by the time a flash is dispatched, all that is left of the request is a position. A page-down
        that clamped to 0 and a page-up that clamped to 0 are the same call, and a scroll that was already
        where it was asked to go has no direction at all. Supplying one would mean threading the original
        request's direction through `SmoothScrolling` to here, and still answering "both" wherever there
        was no direction to thread.

        It is also the better answer. Both arrows say *"nothing above, nothing below - you are seeing all
        of it"*, which is a different fact from "you have reached the bottom" and the one worth having in
        the case where a view has no hidden content. Callers that flash from something other than a
        position - a refused cursor move, say - should match this rule rather than report their own
        direction, or one widget's keyboard ends up disagreeing with its own mouse wheel.
        """
        max_y_scroll = dpg.get_y_scroll_max(self.target)  # tag

        if target_y_scroll == -1:  # end?
            target_y_scroll = max_y_scroll

        where = None
        if max_y_scroll > 0:
            if target_y_scroll == 0:
                where = "top"
            elif target_y_scroll == max_y_scroll:
                where = "bottom"
        else:  # less than a screenful of data -> reached both ends.
            if target_y_scroll == 0:
                where = "both"

        logger.debug(f"ScrollEndFlasher.show_by_position: target_y_scroll = {target_y_scroll}, max_y_scroll = {max_y_scroll}, where = {where}")

        if where is not None:
            self.show(where)

        return where

    def show(self, where: str) -> None:
        """Dispatch the animation.

        This indicates in the GUI that the target child window cannot scroll any further
        in the specified direction.

        `where`: str, the extremity that has been reached. One of "top", "bottom", "both".
                 Here "both" is useful if there is less than one screenful of data

                 Sometimes using "both" is also easier, when there is no meaningful delta
                 from which to compute the direction the user is scrolling to, such as
                 with an instant programmatic jump to the scroll target position.
        """
        if self.animation_running:  # only one simultaneous animation per instance; replace old animation if it exists (effectively restarting the animation).
            animator.cancel(self, finalize=False)  # no need to call `finish` since we'll start a new animation of the same type on the same GUI element right away.
        self.animation_running = True
        self.where = where
        animator.add(self)

    def hide(self) -> None:
        """Hide the overlay immediately. Called automatically by `finish` when the animation ends."""
        if self.window_top is not None:
            dpg.hide_item(self.window_top)
        if self.window_bottom is not None:
            dpg.hide_item(self.window_bottom)

    def render_frame(self, t: int) -> sym:
        """Called automatically by `Animator`."""
        if dpg.get_frame_count() < 10:
            return action_continue

        dt = (t - self.t0) / 10**9  # seconds since t0
        animation_pos = dt / self.duration

        if animation_pos >= 1.0:
            return action_finish
        if self.custom_finish_pred is not None and self.custom_finish_pred(self):
            return action_finish

        scroll_ends_here_color = [196, 196, 255, 64]

        r = numutils.clamp(animation_pos)
        r = numutils.nonanalytic_smooth_transition(r)
        alpha = (1.0 - r) * scroll_ends_here_color[3]
        scroll_ends_here_color[3] = alpha

        with self.overlay_update_lock:
            # We want 8 pixels of rounding on each side (like window rounding),
            # so we must make the overlay window 16 pixels larger in each direction.
            config = dpg.get_item_configuration(self.target)
            w = config["width"]
            h = config["height"]
            w += 16
            h += 16

            # Child windows don't have a `rect_min`; instead, they have `pos`.
            pos = dpg.get_item_pos(self.target)
            # Center the overlay on the target. Now this window covers the target child window.
            pos = [pos[0] - 8, pos[1] - 8]

            # Use two windows, one for each end, to avoid the overlay capturing mouse input (especially the wheel) while it is shown.
            # We create these just once.
            if self.window_top is None:
                logger.debug(f"ScrollEndFlasher.build: frame {dpg.get_frame_count()}: instance '{self.tag}' creating overlay (top)")
                with dpg.window(show=False, modal=False, no_title_bar=True, tag=f"{self.tag}_window_top",
                                pos=pos,
                                width=w, height=48,
                                min_size=[1, 1],  # or DPG silently makes this 100 tall; see note below
                                no_collapse=True,
                                no_focus_on_appearing=True,
                                no_resize=True,
                                no_move=True,
                                no_background=True,
                                no_scrollbar=True,
                                no_scroll_with_mouse=True) as self.window_top:
                    self.drawlist_top = dpg.add_drawlist(width=w, height=48)
            if self.window_bottom is None:
                logger.debug(f"ScrollEndFlasher.build: frame {dpg.get_frame_count()}: instance '{self.tag}' creating overlay (bottom)")
                # `min_size` matters more here than it looks. It defaults to about [100, 100] and clamps an
                # *explicit* size, not only an autosize one — so these 48 px bands were really 100 px, and a
                # DPG window swallows the mouse across its whole rect whether or not it draws a background.
                # The surplus was therefore an invisible dead zone for the wheel: 52 px over the panel from
                # the top band, and 52 px past its bottom edge from this one, for as long as a flash lasted.
                # Which is the very thing splitting this overlay into two windows was meant to avoid.
                # Measured 2026-08-03; see `investigations/dpg-overlays/`.
                with dpg.window(show=False, modal=False, no_title_bar=True, tag=f"{self.tag}_window_bottom",
                                pos=[pos[0], pos[1] + h - 48],
                                width=w, height=48,
                                min_size=[1, 1],
                                no_collapse=True,
                                no_focus_on_appearing=True,
                                no_resize=True,
                                no_move=True,
                                no_background=True,
                                no_scrollbar=True,
                                no_scroll_with_mouse=True) as self.window_bottom:
                    self.drawlist_bottom = dpg.add_drawlist(width=w, height=48)

            # logger.debug(f"Dimmer.build: frame {dpg.get_frame_count()}: instance '{self.tag}' updating geometry and drawing")  # too spammy

            dpg.delete_item(self.drawlist_top, children_only=True)
            dpg.configure_item(self.window_top, width=w, height=48)
            dpg.configure_item(self.drawlist_top, width=w, height=48)

            dpg.delete_item(self.drawlist_bottom, children_only=True)
            dpg.configure_item(self.window_bottom, width=w, height=48)
            dpg.configure_item(self.drawlist_bottom, width=w, height=48)
            dpg.set_item_pos(self.window_bottom, [pos[0], pos[1] + h - 48])

            icon_size = 24
            def draw_on(parent, icon_text):
                # TODO: Improve the visual look (a cap of a circle would look better than a rounded rectangle)
                dpg.draw_rectangle((0, 0), (w - 16, 32), color=(0, 0, 0, 0), fill=scroll_ends_here_color, rounding=8, parent=parent)
                # TODO: Get rid of the kluge offsets.
                icon_upper_left = ((w - icon_size) // 2 - 12 + 3, 3)  # make the icon exactly centered on the rounded rectangle (this was measured in GIMP)  # 3 px: inner padding?
                t = dpg.draw_text(icon_upper_left, icon_text, size=icon_size, color=scroll_ends_here_color, parent=parent)
                dpg.bind_item_font(t, self.font)
            if self.where in ("top", "both"):
                draw_on(self.drawlist_top, icon_text=self.text_top)
            if self.where in ("bottom", "both"):
                draw_on(self.drawlist_bottom, icon_text=self.text_bottom)

            # # Draw a "no" symbol (crossed-out circle). (See also `fa.ICON_BAN`.)
            # circle_center = (w / 2 - 8, 14)
            # circle_radius = 12
            # line_thickness = 4
            # offs_45deg = circle_radius * 0.5**0.5
            # dpg.draw_circle(circle_center, circle_radius,
            #                 thickness=line_thickness, color=(120, 180, 255, alpha),  # blue, with alpha
            #                 parent=self.drawlist_top)
            # dpg.draw_line((circle_center[0] - offs_45deg, circle_center[1] - offs_45deg),
            #               (circle_center[0] + offs_45deg, circle_center[1] + offs_45deg),
            #               thickness=line_thickness, color=(120, 180, 255, alpha),  # blue, with alpha
            #               parent=self.drawlist_top)

            dpg.show_item(self.window_top)
            dpg.show_item(self.window_bottom)

        return action_continue

    def finish(self) -> None:
        """Clean up upon the end of the animation."""
        self.animation_running = False
        self.where = None
        self.hide()

    def note_wheel_scroll(self) -> None:
        """Call this from a mouse-wheel handler over the scrollable this flasher watches.

        The wheel is the movement path nothing else here can see: DPG scrolls a child window internally,
        so no `SmoothScrolling` exists to carry the flasher. Every view with a scrollbar needs this, and it
        is easy to miss precisely because the view visibly scrolls and so looks handled.

        **Checks twice**, because a wheel handler runs before DPG has applied the event. The immediate
        check catches a wheel turned while already at an end; a one-shot animation re-checks once the
        scroll has landed, catching the turn that *arrives* at one. Without the second, a lone tick onto
        the end is silent — a burst hides that, since the next tick reports it, but a single tick does not.

        During a burst each event schedules its own re-check, which is what keeps the overlay lit for as
        long as the wheel keeps turning.
        """
        self.show_by_position(dpg.get_y_scroll(self.target))  # tag
        animator.add(_WheelSettleCheck(self))

    def destroy(self) -> None:
        """Stop this flasher and delete its overlay windows. For a flasher that outlives its usefulness.

        `finish` only *hides* the overlay, which is right for an animation that has ended and may run
        again. An app-lifetime flasher never needs more than that, which is why the two other users do not
        call this; a flasher belonging to a widget that gets rebuilt does, or its windows accumulate.
        """
        animator.cancel(self, finalize=True)
        for window in (self.window_top, self.window_bottom):
            if window is not None and dpg.does_item_exist(window):  # tag
                dpg.delete_item(window)
        self.window_top = None
        self.window_bottom = None
        self.drawlist_top = None
        self.drawlist_bottom = None

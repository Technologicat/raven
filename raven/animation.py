__all__ = ["Animator", "animator",  # controller and its global instance (need only one per app)
           "Animation", "Overlay",  # base classes
           "Dimmer",  # overlays
           "ButtonFlash", "SmoothScrolling",  # animations
           "ScrollEndFlasher",  # animated overlay
           "action_continue", "action_finish", "action_cancel"]  # return values for `render_frame`

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

import math
import threading
import time

from unpythonic import sym

import dearpygui.dearpygui as dpg

from . import utils

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
        """
        self.animations = []
        self.animation_list_lock = threading.RLock()

    def add(self, animation):
        """Register a new `Animation` instance, so that our `render_frame` will call its `render_frame` method.

        Its start time `animation.t0` is set automatically to the current time as returned by `time.time_ns()`.
        """
        with self.animation_list_lock:
            animation.reset()  # set the animation start time
            self.animations.append(animation)
        return animation

    def cancel(self, animation, finalize=True):
        """Terminate a running `Animation` instance.

        `animation`: One of the animations registered using `add`.

        `finalize`: If `True` (default), call the `finish` method of the animation before removing it.

                    In some special cases, it can be useful to set this to `False` to reduce flicker,
                    if the old animation is immediately replaced by a new one of the same type,
                    targeting the same GUI element (so no need to hide/re-show).

        Note that when an animation finishes normally, it is automatically removed. This is meant for
        immediately stopping and removing an animation that has not finished yet.
        """
        with self.animation_list_lock:
            if finalize:
                animation.finish()
            try:
                self.animations.remove(animation)  # uses default comparison by `id()` since `Animation` has no `__eq__` operator
            except ValueError:  # not in list
                logger.debug(f"Animator.cancel: specified {animation} is not in the animation registry (maybe already finished?), skipping removal.")
        return animation

    def render_frame(self):
        """Render one frame of each registered animation, in the order they were registered.

        Each animation whose `render_frame` returns `action_finish` is considered finished.
        Each finished animation gets its `finish` method called automatically.

        After all registered animations have had a frame rendered, the animation registry is updated
        to remove any animations that are no longer running.
        """
        with self.animation_list_lock:
            time_now = time.time_ns()
            running_animations = []
            for animation in self.animations:
                action = animation.render_frame(t=time_now)
                if action is action_continue:
                    running_animations.append(animation)
                elif action is action_finish:
                    animation.finish()
                elif action is action_cancel:
                    pass  # when cancelled, do nothing, just remove the animation
                else:
                    raise ValueError(f"Animator.render_frame: unknown action {action}, expected one of the `animation_action_X` constants (where X is 'continue', 'finish', or 'cancel').")
            self.animations.clear()
            self.animations.extend(running_animations)

    def clear(self):
        """Terminate all registered animations and clear the list of registered animations.

        To terminate a specific animation (by object instance), see `cancel`.
        """
        with self.animation_list_lock:
            for animation in self.animations:
                animation.finish()
            self.animations.clear()
animator = Animator()

class Animation:
    def __init__(self):
        """Base class for Raven's animations.

        An `Animation` can be added to an `Animator`.
        """
        super().__init__()
        # Keep this simple to avoid ravioli code.
        # `t0` should be pretty much the only attribute defined in the base class.
        self.reset()

    def reset(self):
        """Semantically: (re-)start the animation from the beginning.

        Technically, in this base class: Set the animation start time `self.t0` to the current time,
        as given by `time.time_ns()`.
        """
        self.t0 = time.time_ns()

    def render_frame(self, t):
        """Override this in a derived class to render one frame of your animation.

        `t`: int; time at start of current frame as returned by `time.time_ns()`.

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

    def finish(self):
        """Override this in a derived class to clean up any state for your animation when it finishes normally."""

# --------------------------------------------------------------------------------
# Overlay window support

class Overlay:
    def __init__(self, target, tag):
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
    def __init__(self, target, tag, color=(0, 0, 0, 128), rounding=8):
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

    def build(self, rebuild=False):
        # Ensure stuff we depend on is initialized before we try to create this
        if dpg.get_frame_count() < 10:
            return None

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

    def show(self):
        """Dim the target window (e.g. to show that it is updating)."""
        self.build()
        try:  # EAFP to avoid TOCTTOU
            dpg.show_item(self.window)
        except SystemError:  # does not exist
            pass

    def hide(self):
        """Un-dim the target window."""
        try:  # EAFP to avoid TOCTTOU
            dpg.hide_item(self.window)
        except SystemError:  # does not exist
            pass

# --------------------------------------------------------------------------------
# Animations

class ButtonFlash(Animation):
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
    instances = {}  # DPG tag or ID (of `target_button`) -> animation instance

    # TODO: We could also customize `__new__` to return the existing instance, see `unpythpnic.symbol.sym`.
    def __init__(self, message, target_button, target_tooltip, target_text, original_theme, duration, flash_color=(96, 128, 96), text_color=(180, 255, 180)):
        """Animation to flash a button (and its tooltip, if visible) to draw the user's attention.

        This is useful to let the user know that pressing the button actually took,
        when its action has no other immediately visible effects.

        Each GUI element (determined by `target_button`) can only have one `ButtonFlash`
        animation running at a time. If an instance already exists, trying to create the animation
        will restart the existing instance instead (and update its message to `message`).

        `message`: str, text to show in the `target_text` widget while the animation is running.
                   Original content will be restored automatically when the animation finishes normally.
                   Can be `None` for "don't change", or also when `target_text is None`.

        `target_button`: DPG tag or ID, the button to animate (by flashing its background).

        `target_tooltip`: DPG tag or ID, the tooltip to animate (by flashing its background).
                          Can be `None`.

        `target_text`: DPG tag or ID, the text widget to animate (by changing the text content,
                       and the text color, for the duration of the animation). Can be `None`.

                       The text can be inside the tooltip (when `target_tooltip is not None`),
                       but is really completely independent of `target_button` and `target_tooltip`.

        `original_theme`: DPG tag or ID, the theme to restore when the flashing ends.
                          Mandatory when `target_tooltip is not None`, and only used in that case.

        `duration`: float, animation duration in seconds.

        `flash_color`: tuple `(R, G, B)`, each component in [0, 255]. Default is light green.

        `text_color`: tuple `(R, G, B)`, each component in [0, 255]. Default is light green.
        """
        super().__init__()
        self.instance_lock = threading.Lock()

        self.message = message
        self.target_button = target_button
        self.target_tooltip = target_tooltip
        self.target_text = target_text
        self.original_theme = original_theme
        self.duration = duration
        self.flash_color = flash_color
        self.text_color = text_color

        # These are used during animation
        self.theme = None
        self.original_message = None
        self.reified = False  # `True`: running; `False`: ghost mode, update other instance and exit.

        self.start()

    def render_frame(self, t):
        if not self.reified:  # ghost mode
            return action_cancel

        dt = (t - self.t0) / 10**9  # seconds since t0
        animation_pos = dt / self.duration

        if animation_pos >= 1.0:
            return action_finish

        r = utils.clamp(animation_pos)
        r = utils.nonanalytic_smooth_transition(r)

        R0, G0, B0 = self.flash_color
        R1, G1, B1 = 45, 45, 48  # default button background color  TODO: read from global theme
        R = R0 * (1.0 - r) + R1 * r
        G = G0 * (1.0 - r) + G1 * r
        B = B0 * (1.0 - r) + B1 * r
        dpg.set_value(self.highlight_button_color, (R, G, B))
        dpg.set_value(self.highlight_hovered_color, (R, G, B))
        dpg.set_value(self.highlight_active_color, (R, G, B))
        dpg.set_value(self.highlight_popupbg_color, (R, G, B))

        return action_continue

    def start(self):
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
                if self.target_button in type(self).instances:
                    other = type(self).instances[self.target_button]
                    other.message = self.message
                    if other.target_text is not None:
                        dpg.set_value(other.target_text, other.message)
                    other.reset()
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
                type(self).id_counter += 1

                dpg.bind_item_theme(self.target_button, self.theme)
                if self.target_tooltip is not None:
                    dpg.bind_item_theme(self.target_tooltip, self.theme)
                if self.target_text is not None:
                    self.original_message = dpg.get_value(self.target_text)
                    dpg.set_value(self.target_text, self.message)
                    dpg.bind_item_theme(self.target_text, self.theme)

                type(self).instances[self.target_button] = self
                self.reified = True  # This is the instance that animates `self.target_button`.

    def finish(self):
        with self.instance_lock:
            dpg.bind_item_theme(self.target_button, "disablable_button_theme")  # tag
            if self.target_tooltip is not None:
                dpg.bind_item_theme(self.target_tooltip, self.original_theme)
            if self.target_text is not None:
                dpg.set_value(self.target_text, self.original_message)
                dpg.bind_item_theme(self.target_text, self.original_theme)
            dpg.delete_item(self.theme)
            self.theme = None
            self.reified = False

            with type(self).class_lock:
                type(self).instances.pop(self.target_button)

# --------------------------------------------------------------------------------

class SmoothScrolling(Animation):
    class_lock = threading.RLock()
    instances = {}  # DPG tag or ID (of `target_child_window`) -> animation instance

    def __init__(self, target_child_window, target_y_scroll, smooth=True, smooth_step=0.8, flasher=None, finish_callback=None):
        """Scroll a child window, optionally smoothly.

        Each GUI element (determined by `target_child_window`) can only have one `SmoothScrolling`
        animation running at a time. If an instance already exists, trying to create the animation
        will update the existing animation's `target_y_scroll` instead.

        `target_child_window`: DPG tag or ID, the child window to scroll.
        `target_y_scroll`: int, target scroll position in scrollbar coordinates.
        `smooth`: bool.
                  If `True`, will animate a smooth scroll.
                  If `False`, will jump to target position instantly (the point is to offer the same API).
        `smooth_step`: float, a nondimensional rate in the half-open interval (0, 1].
                       Independent of the render FPS.
        `flasher`: `ScrollEndFlasher` instance, optional.
                   Automatically activated when the top/bottom is reached.
        `finish_callback`: 0-argument callable. Run some custom code when the animation finishes normally.
                           Keep it minimal; trying to instantiate a new scroll animation will block while
                           the callback is running (because a new instance might target the same GUI element,
                           and we guarantee the teardown to be atomic).

        Note that mouse wheel and scrollbar dragging do not invoke the scroll animation; for those,
        the scroll position is handled internally by DPG. Hence those don't cause a flash here.
        If you want, you can handle the mouse wheel case separately in a global mouse wheel callback.

        This is pretty sophisticated, to make the movement smooth, but also keep things working as expected
        when the target position changes on the fly.

        The animation depends only on the current and target positions, and has a reference rate,
        no reference duration. Essentially, we use Newton's law of cooling (first-order ODE),
        and apply its analytical solution.

        For the math details, see the detailed comment below the source code of this class.
        """
        super().__init__()
        self.instance_lock = threading.Lock()

        self.target_child_window = target_child_window
        self.target_y_scroll = target_y_scroll
        self.smooth = smooth
        self.smooth_step = smooth_step
        self.flasher = flasher
        self.finish_callback = finish_callback

        self.prev_frame_new_y_scroll = None  # target position of last frame, for monitoring of stuck animation
        self.update_pending_frames = 0
        self.fracpart = 0.0  # fractional part of position, for subpixel correction
        self.reified = False  # `True`: running; `False`: ghost mode, update other instance and exit.

        self.start()

    def render_frame(self, t):
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
                    dpg.set_y_scroll(self.target_child_window, new_y_scroll)
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

                    # Magic section -->
                    # Framerate correction for rate-based animation, to reach a constant animation rate per unit of wall time, regardless of render FPS.
                    CALIBRATION_FPS = 25  # FPS for which `step` was calibrated
                    xrel = 0.5  # just some convenient value
                    step = self.smooth_step
                    alpha_orig = 1.0 - step
                    if 0 < alpha_orig < 1:
                        avg_render_fps = dpg.get_frame_rate()

                        # For a constant target position and original `α`, compute the number of animation frames to cover `xrel` of distance from initial position to final position.
                        # This is how many frames we need at `CALIBRATION_FPS`.
                        n_orig = math.log(1.0 - xrel) / math.log(alpha_orig)
                        # Compute the scaled `n`, to account for `avg_render_fps`. Note the direction: we need a smaller `n` (fewer animation frames) if the render runs slower than `CALIBRATION_FPS`.
                        n_scaled = (avg_render_fps / CALIBRATION_FPS) * n_orig
                        # Then compute the `α` that reaches `xrel` distance in `n_scaled` animation frames.
                        alpha_scaled = (1.0 - xrel)**(1 / n_scaled)
                    else:  # avoid some divisions by zero at the extremes
                        alpha_scaled = alpha_orig
                    step_scaled = 1.0 - alpha_scaled
                    # <-- End magic section

                    # Calculate old and new positions, with subpixel correction (IMPORTANT!).
                    subpixel_corrected_current_y_scroll = current_y_scroll + self.fracpart  # NOTE: float
                    remaining = self.target_y_scroll - subpixel_corrected_current_y_scroll  # remaining distance, float
                    delta = step_scaled * remaining  # distance to cover in this frame, float

                    subpixel_corrected_new_y_scroll = subpixel_corrected_current_y_scroll + delta
                    fracpart = subpixel_corrected_new_y_scroll - int(subpixel_corrected_new_y_scroll)
                    new_y_scroll = int(subpixel_corrected_new_y_scroll)  # NOTE: truncate, no rounding of any kind

                    logger.debug(f"SmoothScrolling.render_frame: frame {dpg.get_frame_count()}: instance for '{self.target_child_window}': old raw = {current_y_scroll}, old subpixel = {subpixel_corrected_current_y_scroll}, delta = {delta}, new subpixel = {subpixel_corrected_new_y_scroll}, target = {self.target_y_scroll}, start-of-frame remaining distance = {remaining}")

                    # Once we reach less than one pixel of distance from the final position, just snap there and end the animation.
                    # We jump at <= 1.0, not < 1.0, to avoid some roundoff trouble.
                    if abs(self.target_y_scroll - subpixel_corrected_current_y_scroll) <= 1.0:
                        new_y_scroll = self.target_y_scroll
                        action = action_finish
                        logger.debug(f"SmoothScrolling.render_frame: frame {dpg.get_frame_count()}: instance for '{self.target_child_window}': scrolling completed")

                        # If we reach the start or the end of the scrollable, flash it.
                        if self.flasher is not None:
                            self.flasher.show_by_position(self.target_y_scroll)

                    if action is action_continue:
                        self.prev_frame_new_y_scroll = new_y_scroll
                        self.fracpart = fracpart

                    dpg.set_y_scroll(self.target_child_window, new_y_scroll)

                # Timeout waiting for DPG to update the position? -> probably end of scrollbar (but shouldn't happen now that `scroll_info_panel_to_position` clamps the value to the max allowed by the scrollbar)
                elif self.update_pending_frames >= update_pending_threshold:
                    action = action_finish
                    logger.debug(f"SmoothScrolling.render_frame: frame {dpg.get_frame_count()}: instance for '{self.target_child_window}': timeout waiting for scrollbar to update its scroll position (target position past end of scrollbar?)")
                    if self.flasher is not None:
                        self.flasher.show(where="bottom")

                # Waiting for DPG to update the position?
                else:
                    self.update_pending_frames = self.update_pending_frames + 1

        return action

    def start(self):
        """Internal method, called automatically by constructor.

        Manages de-duplication (when added to the same GUI element as an existing animation of this type).
        """
        with self.instance_lock:
            if self.reified:  # already running (avoid double resource allocation and registration)
                return

            with type(self).class_lock:
                # If an instance is already running on this GUI element, just update its target scroll position.
                # This allows a seamless transition to the new scroll animation, retaining the subpixel position.
                if self.target_child_window in type(self).instances:
                    other = type(self).instances[self.target_child_window]
                    with other.instance_lock:
                        other.target_y_scroll = self.target_y_scroll
                    return

                type(self).instances[self.target_child_window] = self
                self.reified = True  # This is the instance that animates `self.target_child_window`.

    def finish(self):
        with type(self).class_lock:
            if self.finish_callback is not None:
                self.finish_callback()
            type(self).instances.pop(self.target_child_window)

# The math for the scroll animation comes from SillyTavern-extras, `talkinghead.tha3.app`, function `interpolate_pose`.
# This depends only on the current and target positions, and has a reference *rate*, no reference duration.
# This allows us to change the target position while the animation is running, and it'll adapt.
#
# Pasting this comment as-is. Here in Raven, the equivalent of "pose" is the scroll position.
#
# ---8<---8<---8<---
#
# The `step` parameter is calibrated against animation at 25 FPS, so we must scale it appropriately, taking
# into account the actual FPS.
#
# How to do this requires some explanation. Numericist hat on. Let's do a quick back-of-the-envelope calculation.
# This pose interpolator is essentially a solver for the first-order ODE:
#
#   u' = f(u, t)
#
# Consider the most common case, where the target pose remains constant over several animation frames.
# Furthermore, consider just one morph (they all behave similarly). Then our ODE is Newton's law of cooling:
#
#   u' = -β [u - u∞]
#
# where `u = u(t)` is the temperature, `u∞` is the constant temperature of the external environment,
# and `β > 0` is a material-dependent cooling coefficient.
#
# But instead of numerical simulation at a constant timestep size, as would be typical in computational science,
# we instead read off points off the analytical solution curve. The `step` parameter is *not* the timestep size;
# instead, it controls the relative distance along the *u* axis that should be covered in one simulation step,
# so it is actually related to the cooling coefficient β.
#
# (How exactly: write the left-hand side as `[unew - uold] / Δt + O([Δt]²)`, drop the error term, and decide
#  whether to use `uold` (forward Euler) or `unew` (backward Euler) as `u` on the right-hand side. Then compare
#  to our update formula. But those details don't matter here.)
#
# To match the notation in the rest of this code, let us denote the temperature (actually pose morph value) as `x`
# (instead of `u`). And to keep notation shorter, let `β := step` (although it's not exactly the `β` of the
# continuous-in-time case above).
#
# To scale the animation speed linearly with regard to FPS, we must invert the relation between simulation step
# number `n` and the solution value `x`. For an initial value `x0`, a constant target value `x∞`, and constant
# step `β ∈ (0, 1]`, the pose interpolator produces the sequence:
#
#   x1 = x0 + β [x∞ - x0] = [1 - β] x0 + β x∞
#   x2 = x1 + β [x∞ - x1] = [1 - β] x1 + β x∞
#   x3 = x2 + β [x∞ - x2] = [1 - β] x2 + β x∞
#   ...
#
# Note that with exact arithmetic, if `β < 1`, the final value is only reached in the limit `n → ∞`.
# For floating point, this is not the case. Eventually the increment becomes small enough that when
# it is added, nothing happens. After sufficiently many steps, in practice `x` will stop just slightly
# short of `x∞` (on the side it approached the target from).
#
# (For performance reasons, when approaching zero, one may need to beware of denormals, because those
#  are usually implemented in (slow!) software on modern CPUs. So especially if the target is zero,
#  it is useful to have some very small cutoff (inside the normal floating-point range) after which
#  we make `x` instantly jump to the target value.)
#
# Inserting the definition of `x1` to the formula for `x2`, we can express `x2` in terms of `x0` and `x∞`:
#
#   x2 = [1 - β] ([1 - β] x0 + β x∞) + β x∞
#      = [1 - β]² x0 + [1 - β] β x∞ + β x∞
#      = [1 - β]² x0 + [[1 - β] + 1] β x∞
#
# Then inserting this to the formula for `x3`:
#
#   x3 = [1 - β] ([1 - β]² x0 + [[1 - β] + 1] β x∞) + β x∞
#      = [1 - β]³ x0 + [1 - β]² β x∞ + [1 - β] β x∞ + β x∞
#
# To simplify notation, define:
#
#   α := 1 - β
#
# We have:
#
#   x1 = α  x0 + [1 - α] x∞
#   x2 = α² x0 + [1 - α] [1 + α] x∞
#      = α² x0 + [1 - α²] x∞
#   x3 = α³ x0 + [1 - α] [1 + α + α²] x∞
#      = α³ x0 + [1 - α³] x∞
#
# This suggests that the general pattern is (as can be proven by induction on `n`):
#
#   xn = α**n x0 + [1 - α**n] x∞
#
# This allows us to determine `x` as a function of simulation step number `n`. Now the scaling question becomes:
# if we want to reach a given value `xn` by some given step `n_scaled` (instead of the original step `n`),
# how must we change the step size `β` (or equivalently, the parameter `α`)?
#
# To simplify further, observe:
#
#   x1 = α x0 + [1 - α] [[x∞ - x0] + x0]
#      = [α + [1 - α]] x0 + [1 - α] [x∞ - x0]
#      = x0 + [1 - α] [x∞ - x0]
#
# Rearranging yields:
#
#   [x1 - x0] / [x∞ - x0] = 1 - α
#
# which gives us the relative distance from `x0` to `x∞` that is covered in one step. This isn't yet much
# to write home about (it's essentially just a rearrangement of the definition of `x1`), but next, let's
# treat `x2` the same way:
#
#   x2 = α² x0 + [1 - α] [1 + α] [[x∞ - x0] + x0]
#      = [α² x0 + [1 - α²] x0] + [1 - α²] [x∞ - x0]
#      = [α² + 1 - α²] x0 + [1 - α²] [x∞ - x0]
#      = x0 + [1 - α²] [x∞ - x0]
#
# We obtain
#
#   [x2 - x0] / [x∞ - x0] = 1 - α²
#
# which is the relative distance, from the original `x0` toward the final `x∞`, that is covered in two steps
# using the original step size `β = 1 - α`. Next up, `x3`:
#
#   x3 = α³ x0 + [1 - α³] [[x∞ - x0] + x0]
#      = α³ x0 + [1 - α³] [x∞ - x0] + [1 - α³] x0
#      = x0 + [1 - α³] [x∞ - x0]
#
# Rearranging,
#
#   [x3 - x0] / [x∞ - x0] = 1 - α³
#
# which is the relative distance covered in three steps. Hence, we have:
#
#   xrel := [xn - x0] / [x∞ - x0] = 1 - α**n
#
# so that
#
#   α**n = 1 - xrel              (**)
#
# and (taking the natural logarithm of both sides)
#
#   n log α = log [1 - xrel]
#
# Finally,
#
#   n = [log [1 - xrel]] / [log α]
#
# Given `α`, this gives the `n` where the interpolator has covered the fraction `xrel` of the original distance.
# On the other hand, we can also solve (**) for `α`:
#
#   α = (1 - xrel)**(1 / n)
#
# which, given desired `n`, gives us the `α` that makes the interpolator cover the fraction `xrel` of the original distance in `n` steps.
#
# ---8<---8<---8<---

# --------------------------------------------------------------------------------
# Animated overlay

# Inherit from `Overlay` first, so that `super().__init__(...)` passes its arguments where we want it to.
# Then the `super().__init__()` call inside `Overlay.__init__` will initialize the `Animation` part.
class ScrollEndFlasher(Overlay, Animation):
    def __init__(self, *, target, tag, duration, font, text_top, text_bottom, custom_finish_pred=None):
        """Flasher to indicate when the end of a scrollable area has been reached.

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

    def show_by_position(self, target_y_scroll):
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

    def show(self, where):
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

    def hide(self):
        """Hide the overlay immediately. Called automatically by `finish` when the animation ends."""
        if self.window_top is not None:
            dpg.hide_item(self.window_top)
        if self.window_bottom is not None:
            dpg.hide_item(self.window_bottom)

    def render_frame(self, t):
        """Called automatically by `Animator`."""
        if dpg.get_frame_count() < 10:
            return None

        dt = (t - self.t0) / 10**9  # seconds since t0
        animation_pos = dt / self.duration

        if animation_pos >= 1.0:
            return action_finish
        if self.custom_finish_pred is not None and self.custom_finish_pred(self):
            return action_finish

        scroll_ends_here_color = [196, 196, 255, 64]

        r = utils.clamp(animation_pos)
        r = utils.nonanalytic_smooth_transition(r)
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
                with dpg.window(show=False, modal=False, no_title_bar=True, tag=f"{self.tag}_window_bottom",
                                pos=[pos[0], pos[1] + h - 48],
                                width=w, height=48,
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

    def finish(self):
        """Animation finish callback for `Animator`.

        Called when the animation finishes normally.
        """
        self.animation_running = False
        self.where = None
        self.hide()

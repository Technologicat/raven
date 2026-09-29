"""Pure-math layout utilities — no DPG dependency.

Pan/zoom coordinate transforms, zoom-to-fit, tooltip positioning, and following the end of a growing log.
Shared by the xdot widget, image viewer, and other viewport-based UIs.

This module is licensed under the 2-clause BSD license, to facilitate integration anywhere.
"""

__all__ = [
    "screen_to_content", "content_to_screen",
    "zoom_keep_point", "compute_zoom_to_fit",
    "compute_tooltip_position_scalar",
    "TailFollowDecision", "decide_tail_follow",
]

import dataclasses
from typing import Tuple

from .. import numutils


# ---------------------------------------------------------------------------
# Viewport pan/zoom math (shared by xdot widget, image viewer, etc.)
# ---------------------------------------------------------------------------

def screen_to_content(sx: float, sy: float,
                      pan_cx: float, pan_cy: float,
                      zoom: float,
                      view_w: float, view_h: float) -> Tuple[float, float]:
    """Convert screen (drawlist) coordinates to content (image/graph) coordinates.

    Pan model: ``(pan_cx, pan_cy)`` is the content coordinate at the center of
    the view.  ``zoom`` is screen pixels per content unit.
    """
    if zoom == 0:
        zoom = 1.0
    gx = (sx - view_w / 2) / zoom + pan_cx
    gy = (sy - view_h / 2) / zoom + pan_cy
    return gx, gy

def content_to_screen(cx: float, cy: float,
                      pan_cx: float, pan_cy: float,
                      zoom: float,
                      view_w: float, view_h: float) -> Tuple[float, float]:
    """Convert content (image/graph) coordinates to screen (drawlist) coordinates.

    Inverse of `screen_to_content`.
    """
    sx = (cx - pan_cx) * zoom + view_w / 2
    sy = (cy - pan_cy) * zoom + view_h / 2
    return sx, sy

def zoom_keep_point(old_zoom: float, new_zoom: float,
                    sx: float, sy: float,
                    pan_cx: float, pan_cy: float,
                    view_w: float, view_h: float) -> Tuple[float, float]:
    """Compute new pan after a zoom change, keeping a screen point stationary.

    The point at screen position ``(sx, sy)`` maps to the same content
    coordinate before and after the zoom change.

    Returns ``(new_pan_cx, new_pan_cy)``.
    """
    # From `screen_to_content`:
    #   gx = (sx - w / 2) / zoom + pan_x
    #
    # So in screen coords:
    #   sx = (gx - pan_x) * zoom + w / 2
    #
    # After zoom change, we want the same gx to map to the same sx:
    #   sx = (gx - new_pan_x) * new_zoom + w / 2
    #
    # Solving for new_pan_x:
    #   new_pan_x = gx - (sx - w / 2) / new_zoom
    #
    # y component similarly.
    if old_zoom == 0:
        old_zoom = 1.0
    gx = (sx - view_w / 2) / old_zoom + pan_cx
    gy = (sy - view_h / 2) / old_zoom + pan_cy
    new_pan_cx = gx - (sx - view_w / 2) / new_zoom
    new_pan_cy = gy - (sy - view_h / 2) / new_zoom
    return new_pan_cx, new_pan_cy

def compute_zoom_to_fit(content_w: float, content_h: float,
                        view_w: float, view_h: float,
                        margin: int = 10) -> Tuple[float, float, float]:
    """Compute zoom and pan to fit content in the view, centered.

    Returns ``(zoom, pan_cx, pan_cy)`` where pan is the content coordinate
    at the center of the view.  Returns ``(1.0, 0.0, 0.0)`` if the view
    or content has zero size.
    """
    avail_w = view_w - 2 * margin
    avail_h = view_h - 2 * margin
    if avail_w <= 0 or avail_h <= 0 or content_w <= 0 or content_h <= 0:
        return 1.0, 0.0, 0.0
    zoom = min(avail_w / content_w, avail_h / content_h)
    pan_cx = content_w / 2
    pan_cy = content_h / 2
    return zoom, pan_cx, pan_cy


# ---------------------------------------------------------------------------
# Tooltip positioning
# ---------------------------------------------------------------------------

def compute_tooltip_position_scalar(*,
                                    algorithm: str,
                                    cursor_pos: int,
                                    tooltip_size: int,
                                    viewport_size: int,
                                    offset: int = 20) -> int:
    """Compute x or y position for a tooltip. (Either one of them; hence "scalar".)

    This positions the tooltip elegantly, trying to keep it completely within the DPG viewport area.
    This is mostly useful for tooltips triggered by custom code, such as for a scatterplot dataset in a plotter.

    `algorithm`: one of "snap", "snap_old", "smooth".
                 "snap": Right/bottom side if the tooltip fits there, else left/top side.
                 "snap_old": Right/bottom side when the cursor is at the left/top side of viewport, else left/top side.
                 "smooth": Cursor at left edge -> right/bottom side; cursor at right edge -> left/top side; in between,
                           smoothly varying as a function of the cursor position. For the perfectionists.

                 If unsure, try "snap" for the x coordinate, and "smooth" for the y coordinate; usually looks good.

    `cursor_pos`: mouse cursor position (x or y) depending on which axis you are computing, in viewport coordinates.
    `tooltip_size`: width or height (depending on axis) of the tooltip window, in pixels.
    `viewport_size`: width or height (depending on axis), size of the DPG viewport (or equivalently, primary window), in pixels.
    `offset`: int. This allows positioning the tooltip a bit off from `cursor_pos`, so that the mouse cursor won't
              immediately hover over it when the tooltip is shown.

              This is important, because in DPG a tooltip is a separate window, so this would prevent further
              mouse hover events of the actual window under the tooltip from being triggered (until the mouse
              exits the tooltip area).

    Usage::

        mouse_pos = dpg.get_mouse_pos(local=False)  # in viewport coordinates
        tooltip_size = dpg.get_item_rect_size(my_tooltip_window)  # after `dpg.split_frame()` if needed
        w, h = dpg.get_item_rect_size(my_primary_window)
        xpos = compute_tooltip_position_scalar(algorithm="snap",
                                               cursor_pos=mouse_pos[0],
                                               tooltip_size=tooltip_size[0],
                                               viewport_size=w)
        ypos = compute_tooltip_position_scalar(algorithm="smooth",
                                               cursor_pos=mouse_pos[1],
                                               tooltip_size=tooltip_size[1],
                                               viewport_size=h)
        dpg.set_item_pos(my_tooltip_window, [xpos, ypos])
    """
    if algorithm not in ("snap", "snap_old", "smooth"):
        raise ValueError(f"Unknown `algorithm` '{algorithm}'; supported: 'snap', 'snap_old', 'smooth'.")

    if algorithm == "snap":  # Right/bottom side if the tooltip fits there, else left/top side.
        if cursor_pos + offset + tooltip_size < viewport_size:  # does it fit?
            return cursor_pos + offset
        elif cursor_pos - offset - tooltip_size >= 0:  # does it fit?
            return cursor_pos - offset - tooltip_size
        else:  # as far as it can go to the right/below while the right/bottom edge remains inside the viewport
            return viewport_size - tooltip_size

    elif algorithm == "snap_old":  # Right/bottom side when the cursor is at the left/top side of viewport, else left/top side.
        if cursor_pos < viewport_size / 2:
            return cursor_pos + offset
        else:
            return cursor_pos - offset - tooltip_size

    elif algorithm == "smooth":  # Cursor at left edge -> right/bottom side; cursor at right edge -> left/top side; in between, smoothly varying as a function of the cursor position.
        # Candidate position to the right/below (preferable in the left/top half of the viewport)
        if cursor_pos + offset + tooltip_size < viewport_size:  # does it fit?
            pos1 = cursor_pos + offset
        else:  # as far as it can go to the right/below while the right/bottom edge remains inside the viewport
            pos1 = viewport_size - tooltip_size

        # Candidate position to the left/above (preferable in the right/bottom half of the viewport)
        if cursor_pos - offset - tooltip_size >= 0:  # does it fit?
            pos2 = cursor_pos - offset - tooltip_size
        else:  # as far as it can go to the left/above while the left/top edge remains inside the viewport
            pos2 = 0

        # Weighted average of the two candidates, with a smooth transition.
        # This makes the tooltip x position vary smoothly as a function of the data point location in the plot window.
        # Due to symmetry, this places the tooltip exactly at the middle when the mouse is at the midpoint of the viewport (not necessarily at an axis line; that depends on axis limits).
        r = numutils.clamp(cursor_pos / viewport_size)  # relative coordinate, [0, 1]
        s = numutils.nonanalytic_smooth_transition(r, m=2.0)
        pos = (1.0 - s) * pos1 + s * pos2

        return pos


# ---------------------------------------------------------------------------
# Following the tail of a growing scroll view
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class TailFollowDecision:
    """The answer of `decide_tail_follow`, with the intermediate figures that produced it, for logging.

    `follow`: whether new content should pull the view along with it.
    `at_end`: whether the view is, or our own scrolling is heading, within `tolerance` of the end.
    `undisturbed`: whether the position is still where we last put it, within `drift_tolerance`.
    `gap`: how far above the end the view reports being, in pixels.
    `settled_y_scroll`, `settled_gap`: where our own scrolling is heading, and how far that is above the end.
    `drift_tolerance`: the bound `undisturbed` was judged against.
    `maybe_expected_y_scroll`, `maybe_drift`: the commanded position clamped to the current maximum, and how
                                             far the view is from it; `None` when nothing was commanded.
    """
    follow: bool
    at_end: bool
    undisturbed: bool
    gap: float
    settled_y_scroll: float
    settled_gap: float
    drift_tolerance: float
    maybe_expected_y_scroll: float | None
    maybe_drift: float | None


def decide_tail_follow(*,
                       y_scroll: float,
                       max_y_scroll: float,
                       maybe_target_y_scroll: float | None,
                       last_step: float,
                       maybe_commanded_y_scroll: float | None,
                       commanded_to_end: bool,
                       tolerance: float) -> TailFollowDecision:
    """Decide whether a scroll view showing a growing log should keep following its end.

    Two things move in such a view: the reader moves the scroll *position*, and arriving content moves the
    *maximum*. Asking only "is the view at the bottom" cannot tell those apart, so it reads new content as
    the reader having scrolled away. This asks two questions instead, and follows if either says yes:

      - is the view at the end — judged by where our own scrolling is *heading*, not by where it has got to;
      - were we following, and is the position still where we last put it — which content arriving cannot
        change, and the reader scrolling does.

    `y_scroll`, `max_y_scroll`: the view's scroll position and maximum, as it reports them now.
    `maybe_target_y_scroll`: where a scroll animation of ours is heading, or `None` if none is running.
    `last_step`: how far that animation moved the position on its last frame; `0` if none is running.
    `maybe_commanded_y_scroll`: the position we last wrote, or `None` if we never have.
    `commanded_to_end`: whether that write was a scroll to the end, i.e. whether we were following.
    `tolerance`: how close to the end counts as at the end, and how far the position may sit from where we
                 put it before the reader is taken to have moved it. In pixels.

    Stores nothing: each call decides on the evidence it is given.
    """
    gap = max_y_scroll - y_scroll

    # "At the end" is asked of where our own scrolling is *going*, not of where the view has got to so far.
    # While a scroll of ours is in flight the reported position is somewhere along the way, so a scroll the
    # reader just asked for still reads as at-the-end until the animation has carried it clear of the
    # tolerance — and whether that has happened when the next sample is taken is a matter of timing. That
    # makes the arrow keys behave as if they had a threshold: during a reply a single Up is usually undone,
    # while holding Up eventually sticks, because repeats move the target faster than the content arrives.
    # Consulting the animation's target decides on the reader's request rather than on how far it has been
    # carried out, so one press is enough and the answer does not depend on when it was asked.
    settled_y_scroll = maybe_target_y_scroll if maybe_target_y_scroll is not None else y_scroll
    settled_gap = max_y_scroll - settled_y_scroll
    at_end = (settled_gap <= tolerance)

    # Has the position moved since we last set it? Content arriving cannot do that — it moves the maximum and
    # leaves the position alone — so a mismatch means the reader moved it. Compared against the *clamped*
    # command, since a view pulls the position down by itself when content shrinks, and that is our doing
    # rather than the reader's.
    #
    # With an animation running, the command is its *last written position*, not its target — those come
    # apart precisely while a scroll is in flight. The position tracks the last written value one frame
    # behind, and only the reader breaks that. Intent ("are we heading for the end?") is carried separately,
    # by `commanded_to_end`.
    #
    # The tolerance grows to cover one frame of our own animation while one is running. The report lags the
    # last written value by exactly one step, so that much of a gap is ours — and early in an exponential
    # decay a step is hundreds of pixels, far past a tolerance sized for a human nudging the wheel. Measured
    # in Raven-librarian on a live reply before this: 43 samples in 857 read as user scrolls at drift
    # 51–78 px against a 40 px tolerance. With nothing running `last_step` is `0`, so the sitting-still case —
    # where a real reader's scroll must be caught — keeps the tight bound.
    drift_tolerance = max(tolerance, last_step)
    if maybe_commanded_y_scroll is not None:
        maybe_expected_y_scroll = min(maybe_commanded_y_scroll, max_y_scroll)
        maybe_drift = abs(y_scroll - maybe_expected_y_scroll)
        undisturbed = (maybe_drift <= drift_tolerance)
    else:
        maybe_expected_y_scroll = None
        maybe_drift = None
        undisturbed = False

    # Following continues if we are at the end by position (however we got there — including the reader
    # scrolling back down, which is how this recovers), or if we were following the tail and the position
    # is still where we left it.
    follow = at_end or (commanded_to_end and undisturbed)
    return TailFollowDecision(follow=follow,
                              at_end=at_end,
                              undisturbed=undisturbed,
                              gap=gap,
                              settled_y_scroll=settled_y_scroll,
                              settled_gap=settled_gap,
                              drift_tolerance=drift_tolerance,
                              maybe_expected_y_scroll=maybe_expected_y_scroll,
                              maybe_drift=maybe_drift)

"""Pure-math layout utilities — no DPG dependency.

Pan/zoom coordinate transforms, zoom-to-fit, and tooltip positioning.
Shared by the xdot widget, image viewer, and other viewport-based UIs.

This module is licensed under the 2-clause BSD license, to facilitate integration anywhere.
"""

__all__ = [
    "screen_to_content", "content_to_screen",
    "zoom_keep_point", "compute_zoom_to_fit",
    "compute_tooltip_position_scalar",
]

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

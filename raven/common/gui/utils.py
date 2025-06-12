"""DPG GUI utilities.

This module is licensed under the 2-clause BSD license, to facilitate integration anywhere.
"""

__all__ = ["maybe_delete_item", "has_child_items",
           "get_widget_pos", "get_widget_size", "get_widget_relative_pos", "is_mouse_inside_widget",
           "recenter_window",
           "wait_for_resize",
           "compute_tooltip_position_scalar",
           "get_pixels_per_plotter_data_unit"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import Union

import dearpygui.dearpygui as dpg

from .. import numutils


def maybe_delete_item(item):
    """Delete `item` (DPG ID or tag), if it exists. If not, the error is ignored."""
    logger.info(f"maybe_delete_item: Deleting old GUI item '{item}', if it exists.")
    try:
        dpg.delete_item(item)
    except SystemError:  # does not exist
        pass

def has_child_items(widget):
    """Return whether `widget` (DPG tag or ID) has child items in any of its slots."""
    for slot in range(4):
        if len(dpg.get_item_children(widget, slot=slot)):
            return True
    return False

def get_widget_pos(widget):
    """Return `widget`'s (DPG tag or ID) position `(x0, y0)`, in viewport coordinates.

    This papers over the fact that most items support `dpg.get_item_rect_min`,
    but e.g. with child windows, one needs to use `dpg.get_item_pos` instead.
    """
    try:
        x0, y0 = dpg.get_item_rect_min(widget)
    except KeyError:  # some items don't have `rect_min` (e.g. child windows)
        x0, y0 = dpg.get_item_pos(widget)
    return x0, y0

def get_widget_size(widget):
    """Return `widget`'s (DPG tag or ID) on-screen size `(width, height)`, in pixels.

    This papers over the fact that most items support `dpg.get_item_rect_size`,
    but e.g. child windows store their size in the item configuration instead.
    """
    try:
        w, h = dpg.get_item_rect_size(widget)
    except KeyError:  # e.g. child window
        config = dpg.get_item_configuration(widget)
        w = config["width"]
        h = config["height"]
    return w, h

def get_widget_relative_pos(widget, reference):
    """Return `widget`'s (DPG tag or ID) position, measured relative to the `reference` widget (DPG tag or ID).

    This is handy when you need child window coordinates (use the child window as `reference`).
    """
    x0, y0 = get_widget_pos(widget)  # in viewport coordinates  # tag
    x0_c, y0_c = get_widget_pos(reference)  # in viewport coordinates
    x0_local = x0 - x0_c
    y0_local = y0 - y0_c
    return x0_local, y0_local

def is_mouse_inside_widget(widget):
    """Return whether the mouse cursor is inside `widget` (DPG ID or tag)."""
    x0, y0 = get_widget_pos(widget)
    w, h = get_widget_size(widget)
    m = dpg.get_mouse_pos(local=False)  # in viewport coordinates
    if m[0] < x0 or m[0] >= x0 + w or m[1] < y0 or m[1] >= y0 + h:
        return False
    return True

def wait_for_resize(widget, wait_frames_max=10):
    """Wait (calling `dpg.split_frame()`) until the on-screen size of `widget` (DPG tag or ID) changes.

    If `wait_frames_max` frames have elapsed without the size changing, return.

    Return `True` if the size changed, `False` otherwise.
    """
    waited = 0
    old_size = get_widget_size(widget)
    while waited < wait_frames_max:
        dpg.split_frame()  # let the autosize happen
        waited += 1

        new_size = get_widget_size(widget)
        if new_size != old_size:
            logger.debug(f"wait_for_resize: waited {waited} frame{'s' if waited != 1 else ''} for resize of DPG widget {widget}")
            return True
    else:
        logger.debug(f"wait_for_resize: timeout ({wait_frames_max} frames) when waiting for resize of DPG widget {widget}")
    return False

def recenter_window(thewindow: Union[str, int], *, reference_window: Union[str, int]):
    """Reposition `thewindow` (DPG ID or tag), if visible, so that it is centered on `reference_window`.

    To center on viewport, pass your maximized main window as `reference_window`.
    """
    if reference_window is None:
        return
    if thewindow is None:
        return
    # Sanity check. Just try to call *some* DPG function with `thewindow` to check that the handle is valid.
    try:
        dpg.get_item_alias(thewindow)
    except Exception:
        logger.debug(f"recenter_window: {thewindow} does not exist, skipping.")
        return

    reference_window_w, reference_window_h = get_widget_size(reference_window)
    logger.debug(f"recenter_window: Reference window (tag '{dpg.get_item_alias(reference_window)}', type {dpg.get_item_type(reference_window)}) size is {reference_window_w}x{reference_window_h}.")

    # Render offscreen so we get the final size. Only needed if the size can change.
    dpg.set_item_pos(thewindow,
                     (reference_window_w,
                      reference_window_h))
    dpg.show_item(thewindow)
    logger.debug(f"recenter_window: After show command: Window is visible? {dpg.is_item_visible(thewindow)}.")
    dpg.split_frame()  # wait for render
    logger.debug(f"recenter_window: After wait for render: Window is visible? {dpg.is_item_visible(thewindow)}.")

    w, h = get_widget_size(thewindow)
    logger.debug(f"recenter_window: Window {thewindow} (tag '{dpg.get_item_alias(thewindow)}', type {dpg.get_item_type(thewindow)}) size is {w}x{h}.")

    # Center the window in the viewport
    dpg.set_item_pos(thewindow,
                     (max(0, (reference_window_w - w) // 2),
                      max(0, (reference_window_h - h) // 2)))

def compute_tooltip_position_scalar(*, algorithm, cursor_pos, tooltip_size, viewport_size, offset=20):
    """Compute x or y position for a tooltip. (Either one of them; hence "scalar".)

    This positions the tooltip elegantly, trying to keep it completely within the DPG viewport area.
    This is mostly useful for tooltips triggered by custom code, such as for a scatterplot dataset in a plotter.

    `algorithm`: str, one of "snap", "snap_old", "smooth".
                 "snap": Right/bottom side if the tooltip fits there, else left/top side.
                 "snap_old": Right/bottom side when the cursor is at the left/top side of viewport, else left/top side.
                 "smooth": Cursor at left edge -> right/bottom side; cursor at right edge -> left/top side; in between,
                           smoothly varying as a function of the cursor position. For the perfectionists.

                 If unsure, try "snap" for the x coordinate, and "smooth" for the y coordinate; usually looks good.

    `cursor_pos`: tuple `(x, y)`, mouse cursor position, in viewport coordinates.
    `tooltip_size`: tuple `(width, height)`, size of the tooltip window, in pixels.
    `viewport_size`: tuple `(width, height)`, size of the DPG viewport (or equivalently, primary window), in pixels.
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

def get_pixels_per_plotter_data_unit(plot_widget, xaxis, yaxis):
    """Estimate pixels per DPG plotter data unit, for conversion between viewport space and data space.

    `plot_widget`: dpg tag or ID, the plotter widget (`dpg.plot`).
    `xaxis`: dpg tag or ID, the x axis widget of the plotter (`dpg.add_plot_axis(dpg.mvXAxis, ...)`).
    `yaxis`: dpg tag or ID, the y axis widget of the plotter (`dpg.add_plot_axis(dpg.mvYAxis, ...)`).

    This is subtly wrong, because the plot widget includes also the space for the axis labels and such.
    But there seems to be no way to get pixels per data unit from a plot in DPG (unless using a custom series,
    which we don't).

    Raven uses this for estimation of on-screen distances in data space. For that purpose this is good enough.

    Returns the tuple `(pixels_per_data_unit_x, pixels_per_data_unit_y)`.

    Note that if axes are not equal aspect, the x/y results may be different.
    """
    # x0, y0 = dpg.get_item_rect_min("plot")
    pixels_w, pixels_h = dpg.get_item_rect_size(plot_widget)
    xmin, xmax = dpg.get_axis_limits(xaxis)  # in data space
    ymin, ymax = dpg.get_axis_limits(yaxis)  # in data space
    data_w = xmax - xmin
    data_h = ymax - ymin
    if data_w == 0 or data_h == 0:  # no data in view (can happen e.g. if the plot hasn't been rendered yet)
        return [0.0, 0.0]
    pixels_per_data_unit_x = pixels_w / data_w
    pixels_per_data_unit_y = pixels_h / data_h
    return pixels_per_data_unit_x, pixels_per_data_unit_y

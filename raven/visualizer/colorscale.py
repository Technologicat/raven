"""The Visualizer's colour scale: where a cluster or a year takes its colour on the plot's colormap, and the bar that
shows the scale beside the *Colour by* selector, a sparkline of the records shown at each value.

The bar names the value under the mouse in a tooltip and outlines its column (`update_hover`); what a click on it does
is the app's (`hovered`).
"""

__all__ = ["series_color", "year_color_position", "colormap_position",
           "COLOR_BAR", "COLOR_BAR_SIZE", "add_bar", "build_tooltip_window", "bar_values", "draw",
           "hovered", "update_hover"]

import dearpygui.dearpygui as dpg

from unpythonic.env import env

from ..common import numutils
from ..common.gui import layout_math
from ..common.gui import utils as guiutils

from . import config as visualizer_config
from . import displaystate
from .app_state import app_state

gui_config = visualizer_config.gui_config


def series_color(position: float) -> list[int]:
    """The colour of a scatter series at `position` in [0, 1] on the configured colormap, as translucent RGBA.

    The position is taken within `gui_config.plotter_colormap_range`, the part of the colormap in use.
    """
    first, last = gui_config.plotter_colormap_range
    color = [int(255 * component)
             for component in dpg.sample_colormap(gui_config.plotter_colormap, t=first + (last - first) * position)]
    color[-1] = int(0.5 * color[-1])
    return color


def year_color_position(year: int, bounds: tuple[int, int]) -> float:
    """Where on the colormap `year` takes its colour when colouring by year, in (0, 1], given the dataset's year
    `bounds`. The years are spread as the clusters are (`colormap_position`), above Misc, which keeps 0."""
    first, last = bounds
    return colormap_position(year - first, last - first)


def colormap_position(label: int, max_label: int) -> float:
    """Where on the colormap cluster `label` takes its colour, in [0, 1], given the largest label `max_label`.

    The outliers (label -1) sit at 0, and the clusters are spread evenly above them. A dataset of outliers
    alone (`max_label` -1) puts everything at 0.
    """
    # Without the guard that last case is 0 / 0, which numpy integers answer with NaN rather than an error,
    # and ImPlot then indexes its colour table with the NaN and crashes the process.
    n_slots = max_label + 1
    return float((label + 1) / n_slots) if n_slots > 0 else 0.0


#: The colour bar beside the *Colour by* selector: "2012 ▬▬▬ 2026", or the first and last cluster, drawn into a drawlist
#: as tall as the combo beside it, so that its contents centre on the row. Each value's column grows with the number of
#: records shown in its colour, so the bar is a sparkline of where the data is: linearly where the counts fit the
#: column's pixels, and compressed data-adaptively where they span more (`numutils.histogram_adjusted_heights`).
COLOR_BAR = "color_bar_drawlist"  # tag
COLOR_BAR_SIZE = (200, 26)
# A drawlist answers no hover test, so the bar sits in a group, which does.
_COLOR_BAR_GROUP = "color_bar_group"  # tag
_TOOLTIP_WINDOW = "color_bar_tooltip_window"  # tag

_layout = None  # what the last `draw` drew, for finding the column under the mouse; `None` while no bar is drawn
_hovered = None  # the index of the column under the mouse, or `None`
_highlight = None  # DPG ID of the outline around that column, or `None`


def add_bar() -> None:
    """Add the bar, empty, at the current place in the GUI being built; `draw` fills it."""
    with dpg.group(tag=_COLOR_BAR_GROUP):
        dpg.add_drawlist(*COLOR_BAR_SIZE, tag=COLOR_BAR)


def build_tooltip_window() -> None:
    """Create the window that names the column under the mouse. A window rather than a DPG tooltip, so that it can sit
    off the cursor, which leaves the bar clickable."""
    with dpg.window(show=False, modal=False, no_title_bar=True, tag=_TOOLTIP_WINDOW,
                    no_collapse=True, no_scrollbar=True, no_focus_on_appearing=True, no_move=True, autosize=True,
                    min_size=[1, 1]):  # without it the window is at least 100x100, whatever it holds
        dpg.add_text("", tag="color_bar_tooltip_value_text")
        dpg.add_text("", color=(140, 140, 140), tag="color_bar_tooltip_help_text")
_DIGIT_WIDTH_EM = 1171 / 2048  # the advance of every digit in OpenSans Regular


def bar_values(ds, grouping: str) -> env | None:
    """What the colour bar of `ds` shows for `grouping`: the records shown in full colour per year, or per cluster,
    from the first to the last. Misc is not among them: in neither grouping does it sit on the scale.

    Returns `env(first, positions, counts)`: the first year or cluster, and per value from it on, its place on the
    colormap (`year_color_position`, `colormap_position`) and its count. `None` where there is nothing to show: no
    dataset, no years in a dataset grouped by year, or Misc alone.
    """
    if ds is None:
        return None
    if grouping == displaystate.YEAR:
        if ds.year_bounds is None:
            return None
        first, last = ds.year_bounds
        positions = [year_color_position(year, ds.year_bounds) for year in range(first, last + 1)]
    else:
        first, last = 0, int(ds.sorted_labels[-1])
        if last < first:  # Misc alone
            return None
        positions = [colormap_position(label, last) for label in range(first, last + 1)]
    groups = displaystate.points_by_label(ds.sorted_entries, grouping=grouping)
    counts = [len(groups.get(first + k, ())) for k in range(len(positions))]
    return env(first=first, positions=positions, counts=counts)


def draw(ds) -> None:
    """Draw the colour bar of `ds` for `app_state.grouping`: a column per year, or per cluster, between the first and the
    last, as tall as the number of records shown in full colour there. Misc is not on it: in neither grouping does it
    sit on the scale."""
    global _layout, _hovered, _highlight
    dpg.delete_item(COLOR_BAR, children_only=True)
    _layout, _hovered, _highlight = None, None, None
    dpg.hide_item(_TOOLTIP_WINDOW)
    grouping = app_state.grouping
    maybe_values = bar_values(ds, grouping)
    if maybe_values is None:
        return
    first, positions, counts = maybe_values.first, maybe_values.positions, maybe_values.counts
    last = first + len(positions) - 1
    w, h = COLOR_BAR_SIZE
    text_size = 16
    label_w = 40  # a four-digit year, at that size
    text_color = (140, 140, 140)
    text_y = (h - text_size) / 2
    # A drawlist has no text alignment, so the first label is right-aligned against the bar by computing its width.
    # Not by `dpg.get_text_size`: this also runs at startup, before the first frame, and there the call hung the app.
    # The labels are digits, all 1171/2048 em wide in OpenSans, the UI font, an advance that includes some blank
    # space after the glyph, which is why the gap to the bar is smaller than the last label's.
    first_w = len(str(first)) * _DIGIT_WIDTH_EM * text_size
    dpg.draw_text((label_w - 1 - first_w, text_y), str(first), size=text_size, color=text_color, parent=COLOR_BAR)
    dpg.draw_text((w - label_w + 4, text_y), str(last), size=text_size, color=text_color, parent=COLOR_BAR)
    x0, x1 = label_w, w - label_w
    bottom, top = h - 2, 2  # the columns' baseline, and how high the tallest reaches
    # A 1 px strip at the base for every value, so the scale reads where nothing is shown, and the column on it, at
    # least a pixel tall for one record.
    heights = numutils.histogram_adjusted_heights(counts, low=1, high=bottom - 1 - top)
    compressed = numutils.needs_histogram_adjustment(counts, low=1, high=bottom - 1 - top)
    for k, (position, height) in enumerate(zip(positions, heights)):  # the series' colours, without their translucency
        color = series_color(position)[:3] + [255]
        left, right = x0 + (x1 - x0) * k / len(positions), x0 + (x1 - x0) * (k + 1) / len(positions)
        column_top = bottom - 1 - height
        dpg.draw_rectangle((left, column_top), (right, bottom), color=color, fill=color, parent=COLOR_BAR)
    _layout = env(grouping=grouping, first=first, counts=counts, heights=heights, x0=x0, x1=x1, top=top,
                  bottom=bottom, compressed=compressed)


def _column_at_mouse() -> int | None:
    """The index of the bar's column under the mouse, or `None`."""
    if _layout is None or not guiutils.is_mouse_on_widget(_COLOR_BAR_GROUP):
        return None
    mouse_x, _mouse_y = dpg.get_mouse_pos(local=False)
    bar_x, _bar_y = guiutils.get_widget_pos(COLOR_BAR)
    x = mouse_x - bar_x
    if not _layout.x0 <= x < _layout.x1:
        return None
    return int((x - _layout.x0) / (_layout.x1 - _layout.x0) * len(_layout.counts))


def hovered() -> tuple[str, int] | None:
    """The column under the mouse as `(grouping, value)`, the value a cluster's number or a year, or `None`."""
    if _hovered is None:
        return None
    return _layout.grouping, _layout.first + _hovered


def update_hover() -> bool:
    """Outline the column under the mouse and name it in a tooltip, or clear both. Mouse-move handler.

    Returns whether the mouse is on a column.
    """
    global _hovered, _highlight
    column = _column_at_mouse()
    if column == _hovered:
        return column is not None
    _hovered = column
    if _highlight is not None:
        dpg.delete_item(_highlight)
        _highlight = None
    if column is None:
        dpg.hide_item(_TOOLTIP_WINDOW)
        return False

    n = len(_layout.counts)
    left = _layout.x0 + (_layout.x1 - _layout.x0) * column / n
    right = _layout.x0 + (_layout.x1 - _layout.x0) * (column + 1) / n
    # As tall as the column, from the top of its 1 px base strip up, so that a spike reads apart from its neighbour.
    column_top = _layout.bottom - 1 - _layout.heights[column]
    _highlight = dpg.draw_rectangle((left - 1, column_top - 1), (right + 1, _layout.bottom + 1),
                                    color=(255, 255, 255), thickness=1, parent=COLOR_BAR)

    value = _layout.first + column
    name = f"#{value}" if _layout.grouping == displaystate.CLUSTER else str(value)
    count = _layout.counts[column]
    dpg.set_value("color_bar_tooltip_value_text",  # tag
                  f"{name}: {count} record{'s' if count != 1 else ''} shown in colour")
    help_lines = ["Click to select; Shift adds, Ctrl subtracts, Ctrl+Shift intersects."]
    if _layout.compressed:
        help_lines.append("Heights are compressed, so that the small ones show.")
    dpg.set_value("color_bar_tooltip_help_text", "\n".join(help_lines))  # tag

    # Off the cursor, so that the mouse stays on the bar and a click reaches it, as the plot's tooltip does.
    w, h = dpg.get_item_rect_size("main_window")  # tag
    tooltip_w, tooltip_h = dpg.get_item_rect_size(_TOOLTIP_WINDOW)
    mouse_x, mouse_y = dpg.get_mouse_pos(local=False)
    x = layout_math.compute_tooltip_position_scalar(algorithm="snap", cursor_pos=mouse_x,
                                                    tooltip_size=max(tooltip_w, 300), viewport_size=w)
    y = layout_math.compute_tooltip_position_scalar(algorithm="snap", cursor_pos=mouse_y,
                                                    tooltip_size=max(tooltip_h, 50), viewport_size=h)
    dpg.set_item_pos(_TOOLTIP_WINDOW, [x, y])
    dpg.show_item(_TOOLTIP_WINDOW)
    return True

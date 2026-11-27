"""The histogram window: records per cluster or per year, as the colour bar shows them, drawn large enough to print.

The same counts and colours as the colour bar beside the *Colour by* selector (`colorscale.bar_values`), on one of
three height scales: linear, logarithmic, or adaptive, the histogram adjustment of Ward Larson, Rushmeier and Piatko
(1997), which gives the small bars room without flattening the large ones. Drawn with matplotlib, so a figure saves to
PNG, PDF or SVG.

The figure itself is built by `build_figure`, which needs no GUI; the rest is the window, its controls, its
background render, and saving.
"""

__all__ = ["SCALES", "build_figure",
           "get_scale", "set_scale", "cycle_scale", "set_labels", "get_labels", "add_controls",
           "update", "texture_size", "toggle_window",
           "handle_key",
           "show_save_dialog", "save_callback", "save_to_file", "is_save_dialog_visible",
           "clear_tasks"]

import array
import logging
logger = logging.getLogger(__name__)

import pathlib

import numpy as np

import dearpygui.dearpygui as dpg

import matplotlib.colors
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.scale import FuncScale
from matplotlib.ticker import FixedLocator, MaxNLocator, NullFormatter

from unpythonic.env import env as envcls

from ..common import bgtask
from ..common import color
from ..common import numutils
from ..common.gui import animation as gui_animation

from . import colorscale
from . import config as visualizer_config
from . import displaystate
from . import entry_renderer
from .app_state import app_state

gui_config = visualizer_config.gui_config

#: The height scales, in the order Shift+F12 cycles them.
SCALES = ("adaptive", "log", "linear")

_TICK_FONT_SIZE = 8.0  # points
_LABEL_GAP = 4.0  # points, between the axis and a label sitting at its tick
_LEADER_GAP = 14.0  # points, between the axis and a label moved off its tick, leaving room for the leader
_LINE = 1.25  # a label's height, in font sizes, the least distance between two labels
_MIN_CAPTION_FONT_SIZE = 4.0  # points; still legible in a printed vector figure


# --------------------------------------------------------------------------------
# The figure

def build_figure(counts, colors, *, first: int, grouping: str, scale: str = "adaptive",
                 tallest: int = 0, unreadable: bool = False, captions: list[str] | None = None,
                 title: str = "", background="white",
                 figsize: tuple[float, float] = (8.0, 3.5), dpi: float = 100.0) -> Figure:
    """A bar chart of `counts`, one bar per year or cluster from `first` on, for display or for saving.

    `counts`: records per bar. At least one must be positive.
    `colors`: each bar's colour, as matplotlib takes one (RGB in [0, 1], say).
    `grouping`: `displaystate.YEAR` or `displaystate.CLUSTER`, which names the horizontal axis.
    `scale`: one of `SCALES`.
    `tallest`: label this many of the tallest bars with their counts.
    `unreadable`: label the bars whose counts cannot be read off the axis, as many as fit (`numutils.data_labels`).
    `captions`: per bar, a caption to write under it in place of its number, such as a cluster's leading theme.
        The figure grows taller by the longest caption, so that they fit under the axes.
    `title`: written above the axes.
    `background`: the figure's background, as matplotlib takes a colour. The text, the axes and the ticks are black or
        white, whichever stands out more against it.
    `figsize`, `dpi`: the figure's size in inches, and its pixels per inch on screen or in a PNG.

    Labels that crowd each other move off their ticks, with a leader line back (`numutils.spread_labels`).
    """
    if scale not in SCALES:
        raise ValueError(f"build_figure: `scale` must be one of {SCALES}, got {scale!r}.")
    counts = np.asarray(counts, dtype=float)
    if not (counts > 0).any():
        raise ValueError("build_figure: no bar has a positive count, so there is nothing to scale.")
    xs = np.arange(first, first + len(counts))
    vmax = float(counts.max())
    data_values = sorted({float(v) for v in counts if v > 0})
    tallest_values = data_values[-tallest:] if tallest > 0 else []

    foreground = _foreground_for(background)
    fig = Figure(figsize=figsize, dpi=dpi, facecolor=background)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1, facecolor=background)
    ax.bar(xs, counts, width=0.8, color=colors)
    ax.set_xlim(first - 0.6, xs[-1] + 0.6)
    if captions is not None:
        # Small enough that neighbouring captions do not overlap: a line of text per bar, the axes taking roughly
        # the width of the figure. Down to a floor, past which they overlap rather than vanish.
        bar_pitch_pt = 0.85 * figsize[0] * 72.0 / len(counts)
        caption_size = max(_MIN_CAPTION_FONT_SIZE, min(_TICK_FONT_SIZE, bar_pitch_pt / _LINE))
        ax.set_xticks(xs, captions, rotation=90, fontsize=caption_size)
        fig.canvas.draw()
        caption_height = max(label.get_window_extent().height for label in ax.get_xticklabels()) / dpi
        figsize = (figsize[0], figsize[1] + caption_height)
        fig.set_size_inches(*figsize)
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(axis="x", labelsize=_TICK_FONT_SIZE)
    ax.set_xlabel("Year" if grouping == displaystate.YEAR else "Cluster", fontsize=_TICK_FONT_SIZE + 1)
    if title:
        ax.set_title(title, fontsize=_TICK_FONT_SIZE + 1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(foreground)
    ax.tick_params(colors=foreground, which="both")
    for text in (ax.xaxis.label, ax.title):
        text.set_color(foreground)

    def label_axis(axes_height_pt: float) -> None:
        """The height scale, its ticks and their labels, for axes `axes_height_pt` points tall."""
        for text in list(ax.texts):  # the previous pass's labels
            text.remove()
        # The scale's heights are in points, from one point, the shortest bar that still shows, to the axes' height.
        low, high = 1.0, axes_height_pt
        if scale == "linear":
            forward, inverse = numutils.linear_scale(counts, high=high)
        elif scale == "log":
            forward, inverse = numutils.log_scale(counts, low=low, high=high)
        else:
            forward, inverse = numutils.histogram_adjustment(counts, low=low, high=high)
        ax.set_yscale(FuncScale(ax, (forward, inverse)))
        ax.set_ylim(0, vmax)

        # The ticks: round numbers on a scale of its own, decades and their fractions on the others.
        spacing = _LINE * _TICK_FONT_SIZE / axes_height_pt

        def position(value):
            return float(forward(value)) / high
        if scale == "linear":
            ticks = [float(t) for t in MaxNLocator(nbins=6, integer=True).tick_values(0, vmax) if 0 <= t <= vmax]
            ticks = sorted(set(ticks) | {vmax} | set(tallest_values))
        else:
            ticks = numutils.decade_ticks(position, data_values[0], vmax, spacing=spacing,
                                          include=(0.0, vmax, *tallest_values), resolution=1.0)
        unreadable_values = (numutils.data_labels(position, data_values, ticks, spacing=spacing,
                                                  low=-spacing / 2, high=1 + spacing / 2) if unreadable else [])
        ticks = sorted(set(ticks) | set(unreadable_values))
        any_moved = _draw_tick_labels(ax, ticks, position, spacing, axes_height_pt,
                                      marked={vmax, *tallest_values, *unreadable_values})
        for text in ax.texts:
            text.set_color(foreground)
        widest = max(len(f"{t:.0f}") for t in ticks)
        ax.set_ylabel("Records", color=foreground, fontsize=_TICK_FONT_SIZE + 1,
                      labelpad=(_LEADER_GAP if any_moved else _LABEL_GAP) + 0.6 * _TICK_FONT_SIZE * widest)

    # The labels' spacing, as a fraction of the axis, depends on the axes' height, which `tight_layout` sets only once
    # it has seen the labels. So: label at the height the figure starts with, lay out, and label again at the height
    # the layout chose; the second layout then changes the height by a fraction of a point.
    for _pass in range(2):
        label_axis(ax.get_position().height * figsize[1] * 72.0)
        fig.tight_layout()
    return fig


def _foreground_for(background) -> str:
    """Black or white, whichever has the higher contrast against `background`, a colour as matplotlib takes one."""
    rgb = matplotlib.colors.to_rgb(background)
    return "black" if color.contrast_ratio(rgb, (0, 0, 0)) >= color.contrast_ratio(rgb, (1, 1, 1)) else "white"


def _draw_tick_labels(ax, ticks, position, spacing, axes_height_pt, *, marked) -> bool:
    """Tick marks at `ticks`, and their labels spread so that none crowds another, with a leader to a moved one.

    `marked`: tick values that are bars' counts rather than round numbers, written in bold.

    Returns whether any label was moved onto a leader.
    """
    any_moved = False
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.tick_params(axis="y", length=3)
    wanted = np.array([position(t) for t in ticks])
    placed = numutils.spread_labels(wanted, spacing=spacing, low=-spacing / 2, high=1 + spacing / 2)
    for value, y, label_y in zip(ticks, wanted, placed):
        moved = abs(label_y - y) > 0.3 * spacing  # a nudge of a few points reads as the tick's own label
        any_moved = any_moved or moved
        gap = _LEADER_GAP if moved else _LABEL_GAP
        ax.annotate(f"{value:.0f}", xy=(0, y), xycoords="axes fraction",
                    xytext=(-gap, (label_y - y) * axes_height_pt), textcoords="offset points",
                    ha="right", va="center", fontsize=_TICK_FONT_SIZE, annotation_clip=False,
                    fontweight="bold" if value in marked else "normal",
                    arrowprops=(dict(arrowstyle="-", color="0.45", lw=0.6, shrinkA=1, shrinkB=0) if moved else None))
    return any_moved


def _render_rgba(fig: Figure) -> np.ndarray:
    """`fig` as an RGBA array, `[h, w, 4]`, in [0, 1]."""
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba(), dtype=np.float32) / 255.0


# --------------------------------------------------------------------------------
# Settings

_scale = "adaptive"
_tallest = True  # label the tallest bars, `gui_config.histogram_tallest_labelled` of them
_unreadable = True  # label the bars that cannot be read off the axis
_themes = False  # write each cluster's leading theme under its bar


def get_scale() -> str:
    """The height scale, one of `SCALES`."""
    return _scale


def set_scale(scale: str) -> None:
    """Set the height scale, one of `SCALES`, and redraw if the window is open."""
    global _scale
    if scale not in SCALES:
        raise ValueError(f"set_scale: `scale` must be one of {SCALES}, got {scale!r}.")
    _scale = scale
    if dpg.does_item_exist("histogram_scale_radio"):  # tag  # not built yet, or in a test without the window
        dpg.set_value("histogram_scale_radio", scale.capitalize())  # tag
    update(only_if_visible=True)


def cycle_scale() -> None:
    """Switch to the next height scale."""
    set_scale(SCALES[(SCALES.index(_scale) + 1) % len(SCALES)])


def set_labels(*, tallest: bool | None = None, unreadable: bool | None = None, themes: bool | None = None) -> None:
    """Choose what gets labelled, and redraw if the window is open. An argument left out keeps its setting.

    `tallest`: the tallest bars' counts. `unreadable`: the counts that cannot be read off the axis. `themes`: each
    cluster's leading theme under its bar, beside its number, when grouping by cluster.
    """
    global _tallest, _unreadable, _themes
    if tallest is not None:
        _tallest = tallest
    if unreadable is not None:
        _unreadable = unreadable
    if themes is not None:
        _themes = themes
    for tag, value in (("histogram_tallest_checkbox", _tallest), ("histogram_unreadable_checkbox", _unreadable),  # tag
                       ("histogram_themes_checkbox", _themes)):  # tag
        if dpg.does_item_exist(tag):  # not built yet, or in a test without the window
            dpg.set_value(tag, value)
    update(only_if_visible=True)


def get_labels() -> dict[str, bool]:
    """What gets labelled, as `set_labels` takes it."""
    return {"tallest": _tallest, "unreadable": _unreadable, "themes": _themes}


def add_controls() -> None:
    """The window's controls: the scale, and what gets labelled. Call inside the window's toolbar."""
    dpg.add_radio_button(items=[s.capitalize() for s in SCALES], default_value=_scale.capitalize(), horizontal=True,
                         tag="histogram_scale_radio", callback=lambda sender, app_data: set_scale(app_data.lower()))
    dpg.add_checkbox(label="Tallest", default_value=_tallest, tag="histogram_tallest_checkbox",
                     callback=lambda sender, app_data: set_labels(tallest=app_data))
    dpg.add_checkbox(label="Unreadable", default_value=_unreadable, tag="histogram_unreadable_checkbox",
                     callback=lambda sender, app_data: set_labels(unreadable=app_data))
    dpg.add_checkbox(label="Themes", default_value=_themes, tag="histogram_themes_checkbox",
                     callback=lambda sender, app_data: set_labels(themes=app_data))
    for widget, caption in (
            ("histogram_scale_radio",  # tag
             "The height scale [Shift+F12]\n"
             "Adaptive gives the small bars room without flattening the large ones. Log puts the height linear in the\n"
             "logarithm of the count, and linear makes it proportional to the count."),
            ("histogram_tallest_checkbox",  # tag
             f"Write the counts of the {gui_config.histogram_tallest_labelled} tallest bars on the axis"),
            ("histogram_unreadable_checkbox",  # tag
             "Write the counts that cannot be read off the axis, as many as fit"),
            ("histogram_themes_checkbox",  # tag
             "Write each cluster's leading theme under its bar, when colouring by cluster")):
        with dpg.tooltip(widget):
            dpg.add_text(caption)


# --------------------------------------------------------------------------------
# Rendering

_task_manager = None  # bgtask.TaskManager, created on the first `update` (needs `app_state.bg`)


def _get_task_manager():
    global _task_manager
    if _task_manager is None:
        _task_manager = bgtask.TaskManager(name="histogram_update", mode="sequential", executor=app_state.bg)
    return _task_manager


def _cluster_caption(file_content, cluster_id: int) -> str:
    """Cluster `cluster_id`'s leading theme, or its leading quoted phrase where it has no theme."""
    lines = entry_renderer.format_cluster_keywords(file_content, cluster_id, n=1, themes_only=True)
    if lines and lines[0][0] is not None:
        return lines[0][1]
    keywords = file_content.vis_keywords_by_cluster[cluster_id] if cluster_id < len(file_content.vis_keywords_by_cluster) else []
    return keywords[0] if len(keywords) else ""


def _figure_for(ds, *, dpi: float):
    """The figure for `ds` as the app shows it now, or `None` where there is nothing to draw."""
    grouping = app_state.grouping
    maybe_values = colorscale.bar_values(ds, grouping)
    if maybe_values is None or not any(maybe_values.counts):
        return None
    colors = [[c / 255 for c in colorscale.series_color(p)[:3]] for p in maybe_values.positions]
    captions = None
    if _themes and grouping == displaystate.CLUSTER:
        captions = [f"{maybe_values.first + k} {_cluster_caption(ds.file_content, maybe_values.first + k)}".strip()
                    for k in range(len(maybe_values.counts))]
    title = pathlib.Path(ds.filename).stem if getattr(ds, "filename", None) else ""
    return build_figure(maybe_values.counts, colors, first=maybe_values.first, grouping=grouping, scale=_scale,
                        tallest=gui_config.histogram_tallest_labelled if _tallest else 0, unreadable=_unreadable,
                        captions=captions, title=title, background=gui_config.figure_background_color,
                        figsize=gui_config.histogram_figure_size, dpi=dpi)


def update(*, only_if_visible: bool = False) -> None:
    """Draw the histogram for what the app shows now, and show the window. Task submitter.

    `only_if_visible`: redraw only if the window is open; for following changes to the display.
    """
    if only_if_visible and not (dpg.does_item_exist("histogram_window") and dpg.is_item_visible("histogram_window")):  # tag
        return
    task = bgtask.ManagedTask(category="raven_visualizer_histogram_render", entrypoint=_render_worker,
                              running_poll_interval=0.1, pending_wait_duration=0.1)
    _get_task_manager().submit(task, envcls(wait=False))


def _render_worker(*, task_env) -> None:
    if task_env.cancelled:
        return
    ds = app_state.dataset  # captured once: a dataset opened meanwhile swaps it
    w, h = texture_size()
    image = np.ones([h, w, 4], dtype=np.float32)  # a blank page where there is nothing to draw
    maybe_fig = _figure_for(ds, dpi=gui_config.histogram_display_dpi) if ds is not None else None
    if task_env.cancelled:
        return
    if maybe_fig is not None:
        # A figure grown taller for its captions is drawn smaller, to fit the window's image, whose size is fixed.
        fig_w, fig_h = maybe_fig.get_size_inches()
        maybe_fig.set_dpi(min(gui_config.histogram_display_dpi, w / fig_w, h / fig_h))
        rgba = _render_rgba(maybe_fig)
        rh, rw = min(h, rgba.shape[0]), min(w, rgba.shape[1])
        image[:rh, :rw] = rgba[:rh, :rw]
    dpg.set_value("histogram_texture", array.array("f", image.ravel()))  # tag
    dpg.show_item("histogram_window")  # tag
    logger.debug(f"_render_worker: {task_env.task_name}: histogram drawn ({_scale}).")


def texture_size() -> tuple[int, int]:
    """The window's image, in pixels: the figure at the display resolution."""
    fw, fh = gui_config.histogram_figure_size
    return round(fw * gui_config.histogram_display_dpi), round(fh * gui_config.histogram_display_dpi)


def toggle_window() -> None:
    """Show or hide the histogram window, drawing it first when showing."""
    if dpg.is_item_visible("histogram_window"):  # tag
        dpg.hide_item("histogram_window")  # tag
    else:
        update()


# --------------------------------------------------------------------------------
# Keys and saving

def handle_key(key, ctrl_pressed: bool, shift_pressed: bool) -> bool:
    """Act on `key` if this window binds it, Ctrl+S to save. Returns whether it did."""
    if ctrl_pressed and key == dpg.mvKey_S:
        show_save_dialog()
        return True
    return False


def show_save_dialog() -> None:
    """Ask for a file to save the histogram to."""
    app_state.filedialog_save_histogram.show_file_dialog()
    app_state.enter_modal_mode()


def save_callback(selected_files) -> None:
    """When the save dialog closes: save to the file chosen, if any."""
    app_state.exit_modal_mode()
    if selected_files:
        save_to_file(selected_files[0])  # the dialog has already asked about overwriting


def save_to_file(filename) -> None:
    """Save the histogram as the app shows it now to `filename`, in the background: PNG, PDF or SVG, by the suffix.

    A PNG is written at `gui_config.histogram_save_dpi`; PDF and SVG are vector.
    """
    gui_animation.flash_button(button="histogram_save_button",  # tag
                               message=f"Saved to '{filename}'!",
                               duration=gui_config.acknowledgment_duration,
                               tooltip=app_state.histogram_save_tooltip)
    ds = app_state.dataset

    def write_task():
        maybe_fig = _figure_for(ds, dpi=gui_config.histogram_save_dpi) if ds is not None else None
        if maybe_fig is None:
            logger.warning(f"save_to_file.write_task: nothing to draw, so nothing saved to '{filename}'.")
            return
        maybe_fig.savefig(filename, dpi=gui_config.histogram_save_dpi)
        logger.info(f"save_to_file.write_task: histogram saved to '{filename}'.")
    app_state.bg.submit(write_task)


def is_save_dialog_visible() -> bool:
    """Whether the save dialog is open."""
    maybe_dialog = getattr(app_state, "filedialog_save_histogram", None)
    return maybe_dialog is not None and maybe_dialog.is_visible()


def clear_tasks(wait: bool = False) -> None:
    """Cancel a pending redraw. At app shutdown and when a dataset is opened."""
    if _task_manager is not None:
        _task_manager.clear(wait=wait)

"""Plotter: the semantic map / scatterplot at the core of the Visualizer.

Owns dataset loading (from a Raven-visualizer dataset), dataset parsing
and preprocessing (cluster sort, kd-tree, keyword normalization), rendering data
into the DPG scatter plot (one series per cluster, colored via the configured
colormap), plotter-space queries (visible datapoints, datapoints near the mouse
cursor), zoom reset, and lifecycle cleanup of the cluster-colour themes.

The three "highlight" scatter series (`my_mouse_hover_scatter_series`,
`my_selection_scatter_series`, `my_search_results_scatter_series`) are also
created here — they overlay the cluster series, and are consumed by other
subsystems (`annotation`/tooltip, `selection`, search). Only *creation* lives
here; the value updates stay with each consumer.

Module-local state (the list of dynamically-created cluster-colour themes) stays
encapsulated here; external cleanup goes through `clear_cluster_color_themes`.
"""

__all__ = ["get_visible_datapoints",
           "get_data_idxs_at_mouse",
           "reset_zoom",
           "parse_dataset_file",
           "load_dataset",
           "create_highlight_series",
           "clear_cluster_color_themes"]

import itertools
import logging
import os
import pickle
logger = logging.getLogger(__name__)

import numpy as np
import scipy.spatial

import dearpygui.dearpygui as dpg

from unpythonic import call, timer, window
from unpythonic.env import env

from ..common import utils as common_utils
from ..common.gui import utils as guiutils

from . import config as visualizer_config
from .app_state import app_state

gui_config = visualizer_config.gui_config

# --------------------------------------------------------------------------------
# Module-local state

_cluster_color_themes = []  # DPG theme IDs of the per-cluster scatter-series colour themes; populated by `load_dataset`, consumed by `clear_cluster_color_themes`


# --------------------------------------------------------------------------------
# Plotter-space queries

def get_visible_datapoints():
    """Return a list of all data points (indices to `sorted_xxx`) currently visible in the plotter."""
    if app_state.dataset is None:  # nothing plotted when no dataset loaded
        return common_utils.make_blank_index_array()

    xmin, xmax = dpg.get_axis_limits("axis0")  # in data space  # tag
    ymin, ymax = dpg.get_axis_limits("axis1")  # in data space  # tag

    # fix: accept also data points at the edges of the plotter bounding box
    x_range = xmax - xmin
    y_range = ymax - ymin
    eps = 1e-5

    filtxmin = app_state.dataset.sorted_lowdim_data[:, 0] >= xmin - eps * x_range
    filtxmax = app_state.dataset.sorted_lowdim_data[:, 0] <= xmax + eps * x_range
    filtx = filtxmin * filtxmax
    filtymin = app_state.dataset.sorted_lowdim_data[:, 1] >= ymin - eps * y_range
    filtymax = app_state.dataset.sorted_lowdim_data[:, 1] <= ymax + eps * y_range
    filty = filtymin * filtymax
    filt = filtx * filty
    return np.where(filt)[0]


def get_data_idxs_at_mouse():
    """Return a list of data points (indices to `sorted_xxx`) that are currently under the mouse cursor."""
    if app_state.dataset is None:  # nothing plotted when no dataset loaded
        return common_utils.make_blank_index_array()
    pixels_per_data_unit_x, pixels_per_data_unit_y = guiutils.get_pixels_per_plotter_data_unit("plot", "axis0", "axis1")  # tag
    if pixels_per_data_unit_x == 0.0 or pixels_per_data_unit_y == 0.0:
        return common_utils.make_blank_index_array()

    # FIXME: DPG BUG WORKAROUND: when not initialized yet, `get_plot_mouse_pos` returns `[0, 0]`.
    # This happens especially if the mouse cursor starts outside the plot area when the app starts.
    # For many t-SNE plots, there are likely some data points near the origin.
    p = np.array(dpg.get_plot_mouse_pos())
    first_time = (p == np.array([0.0, 0.0])).all()  # exactly zero - unlikely to happen otherwise (since we likely get asymmetric axis limits from t-SNE)
    if first_time:
        return common_utils.make_blank_index_array()

    # Find `k` data points nearest to the mouse cursor.
    # Since the plot aspect ratio is not necessarily square, we need x/y distances separately to judge the pixel distance.
    # Hence the data space distances the `kdtree` gives us are not meaningful for our purposes.
    data_space_distances_, data_idxs = app_state.dataset.kdtree.query(p, k=gui_config.datapoints_at_mouse_max_neighbors)  # `data_idxs`: item indices into `sorted_xxx`

    # Compute pixel distance, from mouse cursor, of each matched data point.
    deltas = app_state.dataset.sorted_lowdim_data[data_idxs, :] - p  # Distances from mouse cursor in data space, tensor of shape [k, 2].
    deltas[:, 0] *= pixels_per_data_unit_x  # pixel distance, x
    deltas[:, 1] *= pixels_per_data_unit_y  # pixel distance, y
    pixel_distance = (deltas[:, 0]**2 + deltas[:, 1]**2)**0.5

    # Filter for data points within the maximum allowed pixel distance (selection brush size).
    filt = (pixel_distance <= gui_config.selection_brush_radius_pixels)

    return data_idxs[filt]


def reset_zoom():
    """Reset the plotter's zoom level to show all data."""
    dpg.fit_axis_data("axis0")  # tag
    dpg.fit_axis_data("axis1")  # tag


# --------------------------------------------------------------------------------
# Dataset loading and rendering

def _read_dataset_file(filename):
    """Load a dataset file. Low-level helper."""
    with open(filename, "rb") as visdata_file:
        data = pickle.load(visdata_file)
    if data["version"] != 1:
        raise NotImplementedError(f"Dataset {filename} has version '{data['version']}', expected '1'. Don't know how to visualize this dataset.")
    return env(**data)


def parse_dataset_file(filename):
    """Parse a dataset file and process it for visualization. Public API.

    Returns a dataset: `unpythonic.env` with the datafile contents, and some preprocessed fields to facilitate visualization.
    """
    app_state.dataset = env()
    absolute_filename = common_utils.absolutize_filename(filename)
    app_state.dataset.filename = filename
    app_state.dataset.absolute_filename = absolute_filename

    logger.info(f"Reading Raven-visualizer dataset '{filename}' (resolved to '{absolute_filename}')...")
    with timer() as tim:
        app_state.dataset.file_content = _read_dataset_file(absolute_filename)
    logger.info(f"    Done in {tim.dt:0.6g}s.")

    # In DPG (as of this writing, DPG v2.0), one scatter series has only a single global color.
    #
    # To color the data by cluster ID, we create a separate scatter plot for each cluster.
    # Fortunately, DPG is fast enough that it can render hundreds of scatter plots in realtime.
    #
    # For this we need to sort the data by label (cluster ID).
    #
    # An easy way is to argsort the labels and make a copy of the data, so we get logically contiguous blocks
    # of data for each label. The O(n log(n)) sorting cost upon dataset loading is cheap enough.
    #
    logger.info("Sorting data by cluster...")
    with timer() as tim:  # set up `sorted_xxx`
        app_state.dataset.sorted_orig_data_idxs = np.argsort(app_state.dataset.file_content.labels)  # sort by label (cluster ID)
        app_state.dataset.sorted_labels = app_state.dataset.file_content.labels[app_state.dataset.sorted_orig_data_idxs]
        app_state.dataset.sorted_lowdim_data = app_state.dataset.file_content.lowdim_data[app_state.dataset.sorted_orig_data_idxs, :]  # [data_idx, axis], where axis is 0 (x) or 1 (y).
        app_state.dataset.sorted_entries = [app_state.dataset.file_content.vis_data[orig_data_idx] for orig_data_idx in app_state.dataset.sorted_orig_data_idxs]  # the actual data records, extracted from BibTeX (Python list)
        @call
        def _():
            # Compute normalized titles for searching, and insert a reverse lookup for the item's index in `sorted_xxx`.
            for data_idx, entry in enumerate(app_state.dataset.sorted_entries):
                entry.data_idx = data_idx  # index to `sorted_xxx`
                entry.normalized_title = common_utils.normalize_search_string(entry.title.strip())  # for searching

        # Find contiguous blocks with the same label (cluster ID).
        app_state.dataset.cluster_id_jump_data_idxs = np.where(np.diff(app_state.dataset.sorted_labels, prepend=np.nan))[0]  # https://stackoverflow.com/a/65657397
        app_state.dataset.cluster_id_jump_data_idxs = list(itertools.chain(list(app_state.dataset.cluster_id_jump_data_idxs), (None,)))  # -> [i0, i1, ..., iN, None] -> used for slices, `None` = end
    logger.info(f"    Done in {tim.dt:0.6g}s.")

    # For mouseover support. We need to manually detect the data points closest to the mouse cursor.
    logger.info("Indexing dataset for nearest-neighbors search...")
    with timer() as tim:
        app_state.dataset.kdtree = scipy.spatial.cKDTree(data=app_state.dataset.sorted_lowdim_data)
    logger.info(f"    Done in {tim.dt:0.6g}s.")
    return app_state.dataset


def load_dataset(ds):
    """Render `ds` (see `parse_dataset_file`) into the plotter's DPG scatter plot.

    Creates one scatter series per cluster, each with its own colour theme sampled from the
    configured colormap. The cluster-colour themes are tracked in a module-local list so that
    `clear_cluster_color_themes` can delete them when a new dataset is loaded.

    IMPORTANT: call `reset_app_state()` in `app.py` just before calling this (to clear old
    data, tasks, and animations, and to reset the selection).
    """
    logger.info(f"Plotting Raven-visualizer dataset '{ds.absolute_filename}'...")
    with timer() as tim:
        # Group data points by label
        datas = []
        for start, end in window(2, ds.cluster_id_jump_data_idxs):  # indices to `sorted_xxx`
            xs = list(ds.sorted_lowdim_data[start:end, 0])
            ys = list(ds.sorted_lowdim_data[start:end, 1])
            label = ds.sorted_labels[start]  # all `ds.sorted_labels[start:end]` are the same
            datas.append((label, xs, ys))

        max_label = ds.sorted_labels[-1]  # The labels have been sorted in ascending order so the largest one is last.
        for label, xs, ys in datas:
            series_tag = f"my_scatter_series_{label}"  # tag
            series_theme = f"my_plot_theme_{label}"  # tag
            colormap = gui_config.plotter_colormap

            # Render this data series, placing it before the first highlight series so that all highlights render on top.
            dpg.add_scatter_series(xs, ys, tag=series_tag, parent="axis1", before="my_mouse_hover_scatter_series")  # tag

            # Compute the color for this series, and create a theme for it.
            # See:
            #     https://dearpygui.readthedocs.io/en/latest/reference/dearpygui.html#dearpygui.dearpygui.sample_colormap
            #     https://dearpygui.readthedocs.io/en/latest/documentation/themes.html
            color = dpg.sample_colormap(colormap, t=(label + 1) / (max_label + 1))
            color = [int(255 * component) for component in color]  # RGBA
            color[-1] = int(0.5 * color[-1])  # A; make translucent
            with dpg.theme(tag=series_theme) as this_scatterplot_theme:
                with dpg.theme_component(dpg.mvScatterSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, color, category=dpg.mvThemeCat_Plots)
            dpg.bind_item_theme(series_tag, series_theme)
            _cluster_color_themes.append(this_scatterplot_theme)

        dpg.set_item_label("plot", f"Semantic map [{os.path.basename(ds.absolute_filename)}]")  # tag
        reset_zoom()
    logger.info(f"    Done in {tim.dt:0.6g}s.")


# --------------------------------------------------------------------------------
# Highlight scatter series (overlay layers, consumed by selection / annotation / search)

def create_highlight_series():
    """Create the special scatterplot data series used for highlighting datapoints in the plotter.

    Called at initial GUI setup and after clearing old data (in `reset_app_state`).
    """
    # Data items hovered over. Data points to be filled in by mouse move handler.
    series_tag = "my_mouse_hover_scatter_series"
    dpg.add_scatter_series([], [], tag=series_tag, parent="axis1")
    dpg.bind_item_theme(series_tag, "my_selection_theme")  # tag

    # Data items currently selected. Data points to be filled in by selection handler.
    series_tag = "my_selection_scatter_series"
    dpg.add_scatter_series([], [], tag=series_tag, parent="axis1")
    dpg.bind_item_theme(series_tag, "my_selection_datapoints_theme")  # tag

    # Data items matching the current search. Data points to be filled in by search handler.
    series_tag = "my_search_results_scatter_series"
    dpg.add_scatter_series([], [], tag=series_tag, parent="axis1")
    dpg.bind_item_theme(series_tag, "my_search_results_theme")  # tag


# --------------------------------------------------------------------------------
# Lifecycle helpers

def clear_cluster_color_themes():
    """Delete all DPG themes we created for per-cluster scatter-series colouring.

    Called by `app.reset_app_state` when loading a new dataset.
    """
    for theme in _cluster_color_themes:
        dpg.delete_item(theme)
    _cluster_color_themes.clear()

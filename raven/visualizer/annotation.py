"""Mouse-hover annotation tooltip for the plotter.

Extracted from `app.py` (2026-04-23) as the fourth step of the refactoring plan
in `briefs/visualizer-refactoring.md`. Owns the plotter tooltip: the window, the
background task that rebuilds its content on mouse movement, the double-buffered
content swap, and the mouse-hover scatter-series highlight on the plot.

The tooltip lists the items currently under the mouse cursor, grouped by
cluster, with per-item icons showing selection/search status, plus a small help
legend at the bottom explaining the icon glyphs. Content is rebuilt in a
background thread and swapped in atomically to avoid flicker.

Public API: `build_window` (called once at GUI setup), `update` (task submitter,
registered on `app_state` as `update_mouse_hover`), `clear_mouse_hover` (hide
tooltip + clear plot highlight), `clear_tasks` (cancel pending render tasks).
Public state: `content_lock` (guards `data_idxs`), `data_idxs` (list of data
indices currently shown in the tooltip — read by the right-click handler in
`app.py` to decide whether the click can scroll the info panel).

Cross-module state this module reads via `app_state`:
`{dataset, selection_data_idxs_box, themes_and_fonts, bg}` from the
shared-namespace fields; `{is_any_modal_window_visible, mouse_inside_plot_widget,
search_string_box, search_result_data_idxs_box, info_panel_content_lock,
info_panel_entry_title_widgets, get_entries_for_selection}` registered by
`app.py` during startup. The info-panel-owned fields will move into
`info_panel.py` when that extraction lands.
"""

__all__ = ["content_lock",
           "data_idxs",
           "build_window",
           "update",
           "clear_mouse_hover",
           "clear_tasks"]

import gc
import logging
import threading
logger = logging.getLogger(__name__)

import dearpygui.dearpygui as dpg

from unpythonic import dlet, sym, unbox
from unpythonic.env import env as envcls

from ..common import bgtask
from ..common.gui import utils as guiutils

from ..vendor.IconsFontAwesome6 import IconsFontAwesome6 as fa

from . import config as visualizer_config
from . import plotter
from .app_state import app_state

gui_config = visualizer_config.gui_config

# --------------------------------------------------------------------------------
# Public state (read by `app.py`'s right-click handler)

content_lock = threading.RLock()  # Content double buffering (swap). Reentrant so nested acquisitions in the same thread stay simple.
data_idxs = []  # Item indices (into `sorted_xxx`) currently listed in the tooltip while it is open.


# --------------------------------------------------------------------------------
# Module-local state

_build_number = 0  # Sequence number of the last completed tooltip build.

_current_group = None  # DPG widget ID of the group currently holding the tooltip content (set by `build_window`, reassigned on each successful swap in `_render_worker`).

_task_manager = None  # bgtask.TaskManager, lazily created on first `update` call (needs `app_state.bg`).


def _get_task_manager():
    """Lazy-create the annotation render task manager. Requires `app_state.bg` to be set."""
    global _task_manager
    if _task_manager is None:
        _task_manager = bgtask.TaskManager(name="annotation_update",
                                           mode="sequential",
                                           executor=app_state.bg)
    return _task_manager


# --------------------------------------------------------------------------------
# Window creation (called once during GUI setup)

def build_window():
    """Create the tooltip window and its initial (empty) content group.

    A tooltip is really just a window with no title bar.
      - `autosize` is important so the window updates its height when the content is rebuilt.
      - A DPG window (as of 1.x) doesn't have an option to autosize height only, but we can
        set the width by using a drawlist or spacer, and wrapping text to less than that width.
    """
    global _current_group
    with dpg.window(show=False, modal=False, no_title_bar=True, tag="annotation_tooltip_window",
                    no_collapse=True,
                    no_scrollbar=True,
                    no_focus_on_appearing=True,
                    autosize=True):
        with dpg.group() as _current_group:
            dpg.add_text("[no data]", wrap=0, color=(180, 180, 180))
        dpg.set_item_alias(_current_group, "annotation_group")  # tag  # Debug-registry name only — the hot path uses `_current_group` (the widget ID) directly.


# --------------------------------------------------------------------------------
# Task submitter, and plot highlighter.

@dlet(m_prev=None)
def update(*, force=False, wait=True, wait_duration=0.05, env=None):
    """Update the plotter tooltip annotation and the datapoint highlight under the mouse cursor. Public API.

    `force`: By default, we only update the annotation if the mouse has moved since the annotation was last rendered.
             To force an update (e.g. when the mouse wheel has been rotated), use `force=True`.
    `wait`: bool, whether to wait for a short cancellation period before actually starting the annotation update.
            This can be useful if the GUI feature you are triggering this from expects more input in a typical case
            (e.g. more mouse movement).
    `wait_duration`: float, seconds.

    `env`: Internal, handled by `@dlet`. Stores local state (last seen mouse position) between calls.

    For more notes, see `update_info_panel`, which works the same way.
    """
    # For a more responsive GUI, always update the plotter highlight right now.
    m = dpg.get_mouse_pos(local=False)
    if force or m != env.m_prev:
        # Highlight the mouse hover items (by plotting them as another series on top).
        at_mouse = plotter.get_data_idxs_at_mouse()  # item indices into `sorted_xxx`.
        if len(at_mouse):
            dpg.set_value("my_mouse_hover_scatter_series", [list(app_state.dataset.sorted_lowdim_data[at_mouse, 0]),  # tag
                                                            list(app_state.dataset.sorted_lowdim_data[at_mouse, 1])])
        else:
            dpg.set_value("my_mouse_hover_scatter_series", [[], []])  # tag
    if m != env.m_prev:  # Hide the annotation tooltip as soon as the mouse moves. This allows the user to move the mouse where the tooltip was, and get correct plot coordinates.
        dpg.hide_item("annotation_tooltip_window")  # tag
    env.m_prev = m

    render_task = bgtask.ManagedTask(category="raven_visualizer_annotation_render",
                                     entrypoint=_render_worker,
                                     running_poll_interval=0.01,
                                     pending_wait_duration=wait_duration)
    _get_task_manager().submit(render_task, envcls(wait=wait))


# --------------------------------------------------------------------------------
# Worker.

@dlet(internal_build_number=0)  # For making unique DPG tags. Incremented each time, regardless of whether completed or cancelled.
def _render_worker(*, task_env, env=None):
    """Update the plotter annotation tooltip for the items under the mouse cursor.

    `task_env`: Handled by `update`. Importantly, contains the `cancelled` flag for the task.
    """
    # TODO: This function is too spammy even for debug logging, needs a "detailed debug" log level.
    # logger.debug(f"_render_worker: {task_env.task_name}: Annotation update task running.")

    # For "double-buffering"
    global _build_number
    global _current_group
    annotation_target_group = None  # DPG widget for building new content, will be initialized later
    new_content_swapped_in = False

    # Under some conditions no annotation should be shown
    #  - Modal window open (so the rest of the GUI should be inactive)
    #  - The mouse moved outside the plot area while the update was waiting in the queue
    if app_state.is_any_modal_window_visible() or not app_state.mouse_inside_plot_widget():
        dpg.hide_item("annotation_tooltip_window")  # tag
        # logger.debug(f"_render_worker: {task_env.task_name}: Annotation update task completed. No items under mouse, so nothing to do.")
        return

    mouse_pos = dpg.get_mouse_pos(local=False)
    at_mouse = plotter.get_data_idxs_at_mouse()  # item indices into `sorted_xxx`.

    with content_lock:
        old_mouse_hover_data_idxs_set = set(data_idxs)  # For checking if we need to resize/reposition (reduces flickering). Ordering doesn't matter, because the tooltip is always populated in the same order.
        data_idxs.clear()
        if not len(at_mouse):  # No data point(s) under mouse cursor -> hide the annotation if any, and we're done.
            dpg.hide_item("annotation_tooltip_window")  # tag
            return

        # logger.debug(f"_render_worker: {task_env.task_name}: Annotation build {env.internal_build_number} starting.")
        # annotation_t0 = time.monotonic()

        # Start rebuilding the tooltip content.
        # `with dpg.group(...):` would look clearer, but it's better to not touch the DPG container stack from a background thread.
        gc.collect()
        dpg.split_frame()
        annotation_target_group = dpg.add_group(show=False, parent="annotation_tooltip_window")  # tag

        # After this point (content target group GUI widget created), if something goes wrong, we must clean up the partially built content.
        try:
            # for highlighting
            search_string = unbox(app_state.search_string_box)
            search_result_data_idxs = unbox(app_state.search_result_data_idxs_box)
            selection_data_idxs = unbox(app_state.selection_data_idxs_box)

            # Actual content
            entries_by_cluster, formatter = app_state.get_entries_for_selection(at_mouse, max_n=gui_config.max_titles_in_tooltip)
            clusters_at_mouse = list(sorted(set(entries_by_cluster.keys())))
            if clusters_at_mouse and clusters_at_mouse[0] == -1:  # move the misc group (if it's there) to the end
                clusters_at_mouse = clusters_at_mouse[1:] + [-1]

            have_jumpable_item = False  # for whether we should show the help for that
            item_ininfo = sym("ininfo")
            item_selected = sym("selected")
            item_notselected = sym("notselected")
            item_match = sym("match")
            item_nomatch = sym("nomatch")
            item_searchoff = sym("searchoff")

            with app_state.info_panel_content_lock:
                for cluster_id in clusters_at_mouse:
                    if task_env is not None and task_env.cancelled:
                        break

                    cluster_title, cluster_keywords, cluster_content, more = formatter(cluster_id)
                    cluster_title_group = dpg.add_group(horizontal=True, tag=f"cluster_{cluster_id}_annotation_title_group_build{env.internal_build_number}", parent=annotation_target_group)
                    dpg.add_text(cluster_title, color=(180, 180, 180), tag=f"cluster_{cluster_id}_annotation_title_text_build{env.internal_build_number}", parent=cluster_title_group)  # "#42"
                    dpg.add_text(cluster_keywords, wrap=0, color=(140, 140, 140), tag=f"cluster_{cluster_id}_annotation_keywords_text_build{env.internal_build_number}", parent=cluster_title_group)  # "[keyword0, ...]"
                    for data_idx, entry in cluster_content:  # `data_idx`: item index into `sorted_xxx`
                        if task_env is not None and task_env.cancelled:
                            break

                        # Highlight search results, but only when a search is active.
                        # When no search active, show all items at full brightness.
                        if not search_string or data_idx in search_result_data_idxs:
                            title_color = (255, 255, 255, 255)
                        else:  # Search active, non-matching item -> dim it.
                            title_color = (140, 140, 140, 255)

                        if data_idx in app_state.info_panel_entry_title_widgets:  # shown in the info panel
                            item_selection_status = item_ininfo
                            selection_mark_text = fa.ICON_CLIPBOARD_CHECK
                            selection_mark_font = app_state.themes_and_fonts.icon_font_solid
                            if data_idx in selection_data_idxs:  # Usually, all items in the info panel are in the selection...
                                selection_mark_color = (120, 180, 255)  # blue
                            else:  # ...but while the info panel is updating, the old content (shown until the update completes) may have some items that are no longer included in the new selection.
                                item_selection_status = item_ininfo
                                selection_mark_color = (80, 80, 80)  # very dark gray  # (255, 180, 120)  # orange
                        else:  # not shown in the info panel
                            selection_mark_text = fa.ICON_CLIPBOARD
                            selection_mark_font = app_state.themes_and_fonts.icon_font_regular
                            if data_idx in selection_data_idxs:  # selected
                                item_selection_status = item_selected
                                selection_mark_color = (120, 180, 255)  # blue
                            else:  # not selected
                                item_selection_status = item_notselected
                                selection_mark_color = (80, 80, 80)  # very dark gray  # (255, 180, 120)  # orange

                        item_group = dpg.add_group(horizontal=True, tag=f"cluster_{cluster_id}_item_{data_idx}_annotation_group_build{env.internal_build_number}", parent=annotation_target_group)
                        mark_widget = dpg.add_text(selection_mark_text, color=selection_mark_color, tag=f"cluster_{cluster_id}_item_{data_idx}_annotation_selection_mark_build{env.internal_build_number}", parent=item_group)
                        dpg.bind_item_font(mark_widget, selection_mark_font)

                        if search_string:
                            if data_idx in search_result_data_idxs:
                                item_search_status = item_match
                                search_mark_color = (180, 255, 180)  # "search green"
                            else:  # not in results
                                item_search_status = item_nomatch
                                search_mark_color = (80, 80, 80)  # very dark gray
                            mark_widget = dpg.add_text(fa.ICON_MAGNIFYING_GLASS, color=search_mark_color, tag=f"cluster_{cluster_id}_item_{data_idx}_annotation_search_mark_build{env.internal_build_number}", parent=item_group)
                            dpg.bind_item_font(mark_widget, app_state.themes_and_fonts.icon_font_solid)
                        else:  # no search active
                            item_search_status = item_searchoff

                        dpg.add_text(entry.title, color=title_color, wrap=0, tag=f"cluster_{cluster_id}_item_{data_idx}_annotation_title_build{env.internal_build_number}", parent=item_group)  # "A study of stuff..."

                        if item_selection_status is item_ininfo and (not search_string or item_search_status is item_match):
                            have_jumpable_item = True

                        data_idxs.append(data_idx)

                    if task_env is None or not task_env.cancelled:
                        if more:
                            dpg.add_text(more, wrap=0, color=(100, 100, 100), tag=f"cluster_{cluster_id}_annotation_more_build{env.internal_build_number}", parent=annotation_target_group)  # "[...N more entries...]"

            # Finalize (if not cancelled)
            if task_env is None or not task_env.cancelled:
                # ----------------------------------------
                # Help
                # TODO: performance: we could render the help statically just once (with elements that can be optionally shown/hidden)
                hint_color = (140, 140, 140)

                help_separator = dpg.add_drawlist(width=gui_config.annotation_tooltip_w, height=1, tag=f"annotation_help_separator_build{env.internal_build_number}", parent=annotation_target_group)
                dpg.draw_line((0, 0), (gui_config.annotation_tooltip_w - 1, 0), color=(140, 140, 140, 255), thickness=1, parent=help_separator)

                annotation_help_selection_legend_group = dpg.add_group(horizontal=True, tag=f"annotation_help_selection_legend_group_build{env.internal_build_number}", parent=annotation_target_group)

                selection_mark_widget = dpg.add_text(fa.ICON_CLIPBOARD_CHECK, color=(120, 180, 255), tag=f"annotation_help_legend_ininfo_icon_build{env.internal_build_number}", parent=annotation_help_selection_legend_group)  # blue
                dpg.bind_item_font(selection_mark_widget, app_state.themes_and_fonts.icon_font_solid)
                dpg.add_text(": selected, in info panel;", color=hint_color, tag=f"annotation_help_legend_ininfo_explanation_build{env.internal_build_number}", parent=annotation_help_selection_legend_group)

                selection_mark_widget = dpg.add_text(fa.ICON_CLIPBOARD, color=(120, 180, 255), tag=f"annotation_help_legend_selected_icon_build{env.internal_build_number}", parent=annotation_help_selection_legend_group)  # blue
                dpg.bind_item_font(selection_mark_widget, app_state.themes_and_fonts.icon_font_regular)
                dpg.add_text(": selected, not in info panel;", color=hint_color, tag=f"annotation_help_legend_selected_explanation_build{env.internal_build_number}", parent=annotation_help_selection_legend_group)

                selection_mark_widget = dpg.add_text(fa.ICON_CLIPBOARD, color=(80, 80, 80), tag=f"annotation_help_legend_notselected_icon_build{env.internal_build_number}", parent=annotation_help_selection_legend_group)  # very dark gray
                dpg.bind_item_font(selection_mark_widget, app_state.themes_and_fonts.icon_font_regular)
                dpg.add_text(": not selected", color=hint_color, tag=f"annotation_help_legend_notselected_explanation_build{env.internal_build_number}", parent=annotation_help_selection_legend_group)

                if search_string:
                    annotation_help_search_legend_group = dpg.add_group(horizontal=True, tag=f"annotation_help_search_legend_group_build{env.internal_build_number}", parent=annotation_target_group)

                    search_mark_widget = dpg.add_text(fa.ICON_MAGNIFYING_GLASS, color=(180, 255, 180), tag=f"annotation_help_legend_match_icon_build{env.internal_build_number}", parent=annotation_help_search_legend_group)
                    dpg.bind_item_font(search_mark_widget, app_state.themes_and_fonts.icon_font_solid)
                    dpg.add_text(": match;", color=hint_color, tag=f"annotation_help_legend_match_explanation_build{env.internal_build_number}", parent=annotation_help_search_legend_group)

                    search_mark_widget = dpg.add_text(fa.ICON_MAGNIFYING_GLASS, color=(80, 80, 80), tag=f"annotation_help_legend_nomatch_icon_build{env.internal_build_number}", parent=annotation_help_search_legend_group)
                    dpg.bind_item_font(search_mark_widget, app_state.themes_and_fonts.icon_font_solid)
                    dpg.add_text(": no match", color=hint_color, tag=f"annotation_help_legend_nomatch_explanation_build{env.internal_build_number}", parent=annotation_help_search_legend_group)

                if have_jumpable_item:
                    annotation_help_jumpable_group = dpg.add_group(horizontal=True, tag=f"annotation_help_jumpable_group_build{env.internal_build_number}", parent=annotation_target_group)

                    dpg.add_text("[Right-click to scroll info panel to topmost", color=hint_color, tag=f"annotation_help_jumpable_explanation_left_build{env.internal_build_number}", parent=annotation_help_jumpable_group)
                    selection_mark_widget = dpg.add_text(fa.ICON_CLIPBOARD_CHECK, color=(120, 180, 255), tag=f"annotation_help_jumpable_selection_icon_build{env.internal_build_number}", parent=annotation_help_jumpable_group)
                    dpg.bind_item_font(selection_mark_widget, app_state.themes_and_fonts.icon_font_solid)
                    if search_string:
                        search_mark_widget = dpg.add_text(fa.ICON_MAGNIFYING_GLASS, color=(180, 255, 180), tag=f"annotation_help_jumpable_search_icon_build{env.internal_build_number}", parent=annotation_help_jumpable_group)
                        dpg.bind_item_font(search_mark_widget, app_state.themes_and_fonts.icon_font_solid)
                    dpg.add_text("item]", color=hint_color, tag=f"annotation_help_jumpable_explanation_right_build{env.internal_build_number}", parent=annotation_help_jumpable_group)
                else:
                    annotation_help_notjumpable_group = dpg.add_group(horizontal=True, tag=f"annotation_help_notjumpable_group_build{env.internal_build_number}", parent=annotation_target_group)

                    dpg.add_text("[Right-click disabled, no", color=hint_color, tag=f"annotation_help_notjumpable_explanation_left_build{env.internal_build_number}", parent=annotation_help_notjumpable_group)
                    selection_mark_widget = dpg.add_text(fa.ICON_CLIPBOARD_CHECK, color=(120, 180, 255), tag=f"annotation_help_notjumpable_selection_icon_build{env.internal_build_number}", parent=annotation_help_notjumpable_group)
                    dpg.bind_item_font(selection_mark_widget, app_state.themes_and_fonts.icon_font_solid)
                    if search_string:
                        search_mark_widget = dpg.add_text(fa.ICON_MAGNIFYING_GLASS, color=(180, 255, 180), tag=f"annotation_help_notjumpable_search_icon_build{env.internal_build_number}", parent=annotation_help_notjumpable_group)
                        dpg.bind_item_font(search_mark_widget, app_state.themes_and_fonts.icon_font_solid)
                    dpg.add_text("item listed]", color=hint_color, tag=f"annotation_help_notjumpable_explanation_right_build{env.internal_build_number}", parent=annotation_help_notjumpable_group)

                # Swap the new content in ("double-buffering")
                # logger.debug(f"_render_worker: {task_env.task_name}: Swapping in new content (old GUI widget ID {_current_group}; new GUI widget ID {annotation_target_group}).")
                mouse_hover_set_changed = (set(data_idxs) != old_mouse_hover_data_idxs_set)
                if mouse_hover_set_changed:  # temporarily hide the window when the content changes (so that it doesn't flicker while being content-swapped and repositioned)
                    dpg.hide_item("annotation_tooltip_window")  # tag
                dpg.hide_item(_current_group)
                dpg.show_item(annotation_target_group)
                dpg.split_frame()  # wait for render
                dpg.delete_item(_current_group)
                _current_group = None
                dpg.set_item_alias(annotation_target_group, "annotation_group")  # tag  # Debug-registry name only.
                _current_group = annotation_target_group
                new_content_swapped_in = True

                # Resize/reposition the tooltip only when the set of shown items has actually changed.
                # This reduces flickering e.g. when clicking on a datapoint, only changing its selection status.
                if mouse_hover_set_changed:
                    w, h = dpg.get_item_rect_size("main_window")  # tag
                    dpg.set_item_pos("annotation_tooltip_window", [w, h])  # tag  # offscreen, but not hidden -> will be rendered -> triggers the DPG autosize mechanism
                    dpg.show_item("annotation_tooltip_window")  # tag

                    # Tooltip window dimensions after autosizing not available yet, so we need to wait until we can compute the final position the tooltip.
                    guiutils.wait_for_resize("annotation_tooltip_window")  # tag
                    tooltip_size = dpg.get_item_rect_size("annotation_tooltip_window")  # tag

                    # Position the tooltip elegantly, trying to keep the whole tooltip within the viewport area.
                    #
                    # IMPORTANT: This automatically positions the tooltip a bit off from the mouse cursor position so that the cursor won't hover over it.
                    # This keeps `get_plot_mouse_pos` working, as well as improves tooltip readability since the mouse cursor doesn't cover part of it.
                    xpos = guiutils.compute_tooltip_position_scalar(algorithm="snap",
                                                                    cursor_pos=mouse_pos[0],
                                                                    tooltip_size=tooltip_size[0],
                                                                    viewport_size=w)
                    ypos = guiutils.compute_tooltip_position_scalar(algorithm="smooth",
                                                                    cursor_pos=mouse_pos[1],
                                                                    tooltip_size=tooltip_size[1],
                                                                    viewport_size=h)

                    dpg.set_item_pos("annotation_tooltip_window", [xpos, ypos])  # tag
                dpg.show_item("annotation_tooltip_window")  # tag  # just in case it's hidden

                # # Try to bring the tooltip to the front so it isn't covered by the dimmer.
                # # This steals keyboard focus from the search field, because DPG thinks the focused item is actually "main_container" (the top-level layout). Doesn't help to record the focused item earlier, same result.
                # originally_focused_item = dpg.get_focused_item()
                # logger.debug(f"Originally focused: item {originally_focused_item} (tag '{dpg.get_item_alias(originally_focused_item)}', type {dpg.get_item_type(originally_focused_item)})")
                # dpg.focus_item("annotation_tooltip_window")  # tag
                # dpg.split_frame()
                # dpg.focus_item(originally_focused_item)

        except Exception:
            logger.debug(f"_render_worker: {task_env.task_name}: Annotation update task raised an exception; cancelling task.")
            task_env.cancelled = True
            if not new_content_swapped_in:  # clean up: swap back the old content (if it still exists) in case the exception occurred during finalizing
                if annotation_target_group is not None:
                    dpg.hide_item(annotation_target_group)
                if _current_group is not None:
                    dpg.show_item(_current_group)
            raise

        finally:
            if task_env is not None and task_env.cancelled:
                # logger.debug(f"_render_worker: {task_env.task_name}: Annotation update task cancelled.")

                # If the new content was built (partially or completely) but not swapped in, it's unused, so delete it.
                if (annotation_target_group is not None) and (not new_content_swapped_in):
                    # logger.debug(f"_render_worker: {task_env.task_name}: Deleting partially built content.")
                    dpg.delete_item(annotation_target_group)
            else:
                # logger.debug(f"_render_worker: {task_env.task_name}: Annotation update task completed.")

                # Publish the build ID we used while building
                _build_number = env.internal_build_number

            # dt = time.monotonic() - annotation_t0
            # logger.debug(f"_render_worker: {task_env.task_name}: Annotation build {env.internal_build_number} exiting. Rendered in {dt:0.2f}s.")
            env.internal_build_number += 1  # always increase internal build, even when cancelled, for unique IDs.


def clear_mouse_hover():
    """Hide the annotation tooltip, and clear the mouse highlight in the plotter."""
    dpg.hide_item("annotation_tooltip_window")  # tag
    dpg.set_value("my_mouse_hover_scatter_series", [[], []])  # tag


def clear_tasks(wait=False):
    """Cancel any pending annotation render tasks. Called at app shutdown and on dataset reload."""
    if _task_manager is not None:
        _task_manager.clear(wait=wait)

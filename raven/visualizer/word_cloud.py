"""Word cloud rendering, display, and PNG save for the Visualizer.

Extracted from `app.py` (2026-04-17) as the first step of the app-refactoring
plan in `briefs/visualizer-refactoring.md`. Fairly self-contained: the only
cross-module dependencies are `app_state.{dataset, bg, filedialog_save,
themes_and_fonts, selection_data_idxs_box, enter_modal_mode, exit_modal_mode}`.

Public API is registered in `__all__`. Internal state — the last-rendered
dataset id, the last-rendered selection, the texture array, the last
generated `WordCloud` object — stays module-local.
"""

__all__ = ["update",
           "toggle_window",
           "show_save_dialog",
           "save_callback",
           "save_to_file",
           "is_save_dialog_visible",
           "clear_tasks"]

import array
import collections
import logging
logger = logging.getLogger(__name__)

import numpy as np

import dearpygui.dearpygui as dpg

from unpythonic import box, unbox
from unpythonic.env import env as envcls

from wordcloud import WordCloud

from ..common import bgtask
from ..common.gui import animation as gui_animation

from ..vendor.IconsFontAwesome6 import IconsFontAwesome6 as fa

from . import config as visualizer_config
from .app_state import app_state

gui_config = visualizer_config.gui_config

# --------------------------------------------------------------------------------
# Module-local state

_task_manager = None  # bgtask.TaskManager, lazily created on first `update` call (needs `app_state.bg`)

_last_dataset_addr = None  # `id()` of the last dataset the word cloud was generated for (detect user opening a different file)
_last_data_idxs = set()  # last selection, to detect selection changes
_image_box = box(np.ones([gui_config.word_cloud_h, gui_config.word_cloud_w, 4],  # texture data; array is mutated in place
                         dtype=np.float64))
_data_box = box(None)  # the last generated `WordCloud` object, for `save_to_file`


def _get_task_manager():
    """Lazy-create the word cloud render task manager. Requires `app_state.bg` to be set."""
    global _task_manager
    if _task_manager is None:
        _task_manager = bgtask.TaskManager(name="word_cloud_update",
                                           mode="sequential",
                                           executor=app_state.bg)
    return _task_manager


# --------------------------------------------------------------------------------
# Rendering

def update(data_idxs, *, only_if_visible=False, wait=False):
    """Compute a word cloud for the given data points, updating the texture. Show the window when done.

    Task submitter.

    We only actually update the word cloud if the selection has changed or a new dataset has been loaded;
    otherwise we just (re-)show the existing word cloud.

    `data_idxs`: rank-1 np.array, indices into `sorted_xxx`.
    `only_if_visible`: bool. If `True`, only actually run the update if the word cloud window is already visible.
                       Used for live-updating when the selection changes (no point in updating if hidden).
    `wait`: bool, whether to wait for more keyboard/mouse input before starting the update.
    """
    doit = True
    if only_if_visible and not dpg.is_item_visible("word_cloud_window"):
        doit = False
    if not doit:
        return

    render_task = bgtask.ManagedTask(category="raven_visualizer_word_cloud_render",
                                     entrypoint=_render_worker,
                                     running_poll_interval=0.1,
                                     pending_wait_duration=0.1)
    _get_task_manager().submit(render_task, envcls(wait=wait,
                                                   data_idxs=data_idxs))


def _render_worker(*, task_env):
    """Compute a word cloud for the given data points, updating the texture. Show the window when done.

    Worker.

    This handles also updating the GUI, to indicate that the word cloud is being updated,
    as well as resetting those notifications when done.

    `task_env`: Handled by `update`. Importantly, contains the `cancelled` flag for the task.
                Also contains `data_idxs`, specifying which entries to render the word cloud for.
    """
    global _last_dataset_addr
    global _last_data_idxs

    logger.debug(f"_render_worker: {task_env.task_name}: Word cloud update task running.")
    try:
        assert task_env is not None
        if task_env.cancelled:
            logger.debug(f"_render_worker: {task_env.task_name}: Word cloud update task cancelled (before starting).")
            return

        data_idxs = task_env.data_idxs

        if app_state.dataset is None:
            logger.debug(f"_render_worker: {task_env.task_name}: No dataset loaded. Clearing texture.")
            arr = unbox(_image_box)
            arr[:, :, :3] = 0.0
        else:
            # No need to recompute -> just show the window.
            if id(app_state.dataset) == _last_dataset_addr and set(data_idxs) == _last_data_idxs:
                logger.debug(f"_render_worker: {task_env.task_name}: Same dataset and same selection as last time. Showing word cloud window. Task completed.")
                dpg.show_item("word_cloud_window")
                return

            arr = unbox(_image_box)
            if not len(data_idxs):  # no selected data points?
                logger.debug(f"_render_worker: {task_env.task_name}: No data points selected. Clearing texture.")
                arr[:, :, :3] = 0.0
            else:
                dpg.set_item_label("word_cloud_window", "Word cloud [updating]")
                dpg.set_item_label("word_cloud_button", fa.ICON_CLOUD_BOLT)
                dpg.set_value("word_cloud_button_tooltip_text", "Generating word cloud, just for you. Please wait. [F10]")
                gui_animation.animator.add(gui_animation.ButtonFlash(message=None,
                                                                     target_button="word_cloud_button",
                                                                     target_tooltip=None,  # we handle the tooltip manually
                                                                     target_text=None,
                                                                     original_theme=app_state.themes_and_fonts.global_theme,
                                                                     duration=gui_config.acknowledgment_duration))

                # Combine keyword counts of the specified items
                logger.debug(f"_render_worker: {task_env.task_name}: Collecting keywords for selected data points.")
                keywords = collections.defaultdict(lambda: 0)
                for data_idx in data_idxs:
                    if task_env.cancelled:
                        logger.debug(f"_render_worker: {task_env.task_name}: Word cloud update task cancelled (while collecting keywords).")
                        return
                    for kw, count in app_state.dataset.sorted_entries[data_idx].keywords.items():
                        keywords[kw] += count

                logger.debug(f"_render_worker: {task_env.task_name}: Invoking word cloud generator.")
                wc = WordCloud(width=gui_config.word_cloud_w,
                               height=gui_config.word_cloud_h,
                               background_color=gui_config.word_cloud_background_color,
                               colormap=gui_config.word_cloud_colormap,
                               max_words=1000)
                wc.generate_from_frequencies(keywords)  # -> RGB tensor of shape [h, w, 3]
                _data_box << wc

                logger.debug(f"_render_worker: {task_env.task_name}: Updating texture.")
                arr[:, :, :3] = wc.to_array() / 255  # RGB, range [0, 255] -> RGBA, range [0, 1]

        logger.debug(f"_render_worker: {task_env.task_name}: Sending updated texture to GUI. Showing word cloud window.")
        raw_data = array.array('f', arr.ravel())  # shape [h, w, c] -> linearly indexed
        dpg.set_value("word_cloud_texture", raw_data)
        dpg.show_item("word_cloud_window")

        _last_dataset_addr = id(app_state.dataset)  # Conserve RAM by not storing the actual dataset object, but only its memory address. If this changes, it means that the dataset has changed.
        _last_data_idxs = set(data_idxs)

        logger.debug(f"_render_worker: {task_env.task_name}: Word cloud update task completed.")

    finally:
        dpg.set_item_label("word_cloud_window", "Word cloud")  # TODO: DRY duplicate definitions for labels
        dpg.set_item_label("word_cloud_button", fa.ICON_CLOUD)
        dpg.set_value("word_cloud_button_tooltip_text", "Toggle word cloud window [F10]")  # TODO: DRY duplicate definitions for labels


def toggle_window():
    """Show/hide the word cloud window.

    Will update the word cloud first if necessary.
    """
    if dpg.is_item_visible("word_cloud_window"):
        dpg.hide_item("word_cloud_window")
    else:
        update(unbox(app_state.selection_data_idxs_box))  # will show the window when done


# --------------------------------------------------------------------------------
# Saving

def show_save_dialog():
    """Show the "save word cloud" file dialog, to ask the user for a filename to save the word cloud image as."""
    logger.debug("show_save_dialog: Showing save word cloud dialog.")
    app_state.filedialog_save.show_file_dialog()
    app_state.enter_modal_mode()
    logger.debug("show_save_dialog: Done.")


def save_callback(selected_files):
    """Callback that fires when the "save word cloud" dialog closes.

    Wired up when the save-word-cloud FileDialog is created (in `app.py`'s `initialize_filedialogs`).
    """
    logger.debug("save_callback: Save word cloud dialog callback triggered.")
    if len(selected_files) > 1:  # Should not happen, since `multi_selection=False`.
        raise ValueError(f"Expected at most one selected file, got {len(selected_files)}.")
    app_state.exit_modal_mode()
    if selected_files:
        selected_file = selected_files[0]
        logger.debug(f"save_callback: User selected the file '{selected_file}'.")
        save_to_file(selected_file)  # Overwrite confirmation is handled on the file dialog side; if we get here, the user has allowed the overwrite.
    else:
        logger.debug("save_callback: Cancelled.")


def save_to_file(filename):
    """Dispatch a background task to save the word cloud image to a file, and acknowledge the action in the GUI.

    This is called *after* the "save word cloud" dialog closes.
    """
    logger.debug(f"save_to_file: Dispatching a save to '{filename}', and acknowledging in GUI.")

    # The animation can run while we're saving.
    gui_animation.animator.add(gui_animation.ButtonFlash(message=f"Saved to '{filename}'!",
                                                         target_button="word_cloud_save_button",
                                                         target_tooltip="word_cloud_save_tooltip",
                                                         target_text="word_cloud_save_tooltip_text",
                                                         original_theme=dpg.get_item_theme("word_cloud_save_tooltip"),
                                                         duration=gui_config.acknowledgment_duration))

    def write_task():
        logger.debug(f"save_to_file.write_task: Saving word cloud image to '{filename}'.")
        wc = unbox(_data_box)
        wc.to_file(filename)
        logger.debug("save_to_file.write_task: Done.")
    app_state.bg.submit(write_task)  # just add it manually to the thread pool executor; we don't need any fancy management here.


def is_save_dialog_visible():
    """Return whether the "save word cloud" dialog is open.

    We have this abstraction (not just `dpg.is_item_visible`) because the dialog might not exist yet.
    """
    if app_state.filedialog_save is None:
        return False
    return dpg.is_item_visible("save_word_cloud_dialog")


def clear_tasks(wait=False):
    """Cancel any pending word cloud render tasks. Called at app shutdown and on dataset reload."""
    if _task_manager is not None:
        _task_manager.clear(wait=wait)

"""Word cloud rendering, display, and PNG save for the Visualizer.

Three modes: phrases (the default), summing the noun phrases each record carries; words, summing its
single-word counts; and themes, the LLM's keywords for each cluster, weighted by how many of the selected records
are in the clusters that have them. A dataset imported before records carried phrases has no phrases, and one
keyworded without the LLM has no themes; the mode control offers what the dataset has.

Fairly self-contained: the only cross-module dependencies are
`app_state.{dataset, bg, filedialog_save, themes_and_fonts,
selection_data_idxs_box, enter_modal_mode, exit_modal_mode}`.

Public API is registered in `__all__`. Internal state — the last-rendered
dataset id, the last-rendered selection, the texture array, the last
generated `WordCloud` object — stays module-local.
"""

__all__ = ["MODES", "themes_keep_corpus_wide",
           "get_mode", "set_mode", "set_themes_keep_corpus_wide", "cycle_mode", "available_modes", "add_mode_control",
           "update",
           "toggle_window",
           "handle_key",
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
from ..common import keyphrases
from ..common.gui import animation as gui_animation
from ..common.gui import tooltip as gui_tooltip

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

MODES = ("phrases", "words", "themes")
_mode_box = box("phrases")  # what the user chose; the app sets it from the saved state at startup
_last_render_key = None  # what the last cloud was rendered with (`_render_key`), part of what the render cache keys on

# Whether the themes cloud keeps the themes the LLM gave too many clusters, as the cluster labels do not. Off, the cloud
# shows what tells the selection's clusters apart; on, a theme most clusters share leads it. The window's checkbox sets
# it, for comparing the two on real selections; whether the checkbox stays depends on what that shows.
themes_keep_corpus_wide = False

_MODE_TOOLTIP = ("Build the word cloud from noun phrases, from single words, or from the clusters' themes [Shift+F10]\n"
                 "Themes are weighted by how many of the selected items are in the clusters that have them.")
_MODE_TOOLTIP_MISSING = {"phrases": "This dataset has no phrases. Import it again to get them.",
                         "themes": "This dataset has no themes. Import it with LLM keywords to get them."}


# --------------------------------------------------------------------------------
# The mode

def get_mode() -> str:
    """The mode the user chose, one of `MODES`, whether or not the current dataset can render it."""
    return unbox(_mode_box)


def set_mode(mode: str) -> None:
    """Choose what the word cloud is built from, one of `MODES`, and redraw it if it is open."""
    if mode not in MODES:
        raise ValueError(f"set_mode: unknown mode {mode!r}; expected one of {MODES}")
    _mode_box << mode
    if not dpg.does_item_exist("word_cloud_window"):  # tag  # at startup, restoring the saved mode before the GUI exists
        return
    _sync_mode_control(app_state.dataset)
    update(unbox(app_state.selection_data_idxs_box), only_if_visible=True)


def set_themes_keep_corpus_wide(keep: bool) -> None:
    """Keep the themes most clusters share in the themes cloud, or leave them out, and redraw it if it is open."""
    global themes_keep_corpus_wide
    themes_keep_corpus_wide = keep
    if not dpg.does_item_exist("word_cloud_window"):  # tag
        return
    if dpg.does_item_exist("word_cloud_shared_themes_checkbox"):  # tag
        dpg.set_value("word_cloud_shared_themes_checkbox", keep)  # tag
    update(unbox(app_state.selection_data_idxs_box), only_if_visible=True)


def cycle_mode() -> None:
    """Switch to the next of the modes the dataset has, in the order of `MODES`. Does nothing while it has only one."""
    modes = available_modes(app_state.dataset)
    if len(modes) > 1:
        current = _effective_mode(app_state.dataset)
        set_mode(modes[(modes.index(current) + 1) % len(modes)])


def available_modes(ds) -> tuple[str, ...]:
    """The modes dataset `ds` can render, in the order of `MODES`.

    Words always. Phrases where the records carry them, which they do not in a dataset imported by an older version.
    Themes where the dataset keeps the LLM's keywords for each cluster (`clusters_keyword_method` `"llm"` or
    `"hybrid"`).
    """
    file_content = getattr(ds, "file_content", None)
    have = {"words": True,
            "phrases": getattr(file_content, "keyphrase_display", None) is not None,
            "themes": getattr(file_content, "cluster_llm_keywords", None) is not None}
    return tuple(mode for mode in MODES if have[mode])


def _effective_mode(ds) -> str:
    """The chosen mode where the dataset has it, otherwise the first it has (phrases, else words)."""
    modes = available_modes(ds)
    return get_mode() if get_mode() in modes else modes[0]


def _render_key(ds) -> tuple:
    """What a cloud of `ds` is rendered with, beside the selection: the effective mode, and in themes mode the switch."""
    mode = _effective_mode(ds)
    return (mode, themes_keep_corpus_wide if mode == "themes" else None)


def _sync_mode_control(ds) -> None:
    """Offer the dataset's modes in the window's control, showing the effective one; disabled when there is only one."""
    if not dpg.does_item_exist("word_cloud_mode_radio"):  # tag  # not built yet, or in a test without the window
        return
    modes = available_modes(ds)
    dpg.configure_item("word_cloud_mode_radio", items=[mode.capitalize() for mode in modes], enabled=len(modes) > 1)  # tag
    dpg.set_value("word_cloud_mode_radio", _effective_mode(ds).capitalize())  # tag
    if dpg.does_item_exist("word_cloud_shared_themes_checkbox"):  # tag
        dpg.configure_item("word_cloud_shared_themes_checkbox", enabled=_effective_mode(ds) == "themes")  # tag
    maybe_tooltip = getattr(app_state, "word_cloud_mode_tooltip", None)
    if maybe_tooltip is not None:
        missing = [_MODE_TOOLTIP_MISSING[mode] for mode in MODES if mode not in modes]
        maybe_tooltip.text = "\n\n".join([_MODE_TOOLTIP] + missing)


def add_mode_control() -> None:
    """Add the mode control to the current DPG container, with its tooltip. Call once, when building the window."""
    def callback(sender, app_data, user_data):  # `app_data` is the label chosen, such as "Phrases"
        set_mode(app_data.lower())
    dpg.add_radio_button(items=[mode.capitalize() for mode in MODES], tag="word_cloud_mode_radio", horizontal=True,
                         default_value=get_mode().capitalize(), callback=callback)
    # Disabled rather than hidden outside themes mode, so that nothing beside it moves.
    dpg.add_checkbox(label="Shared themes", tag="word_cloud_shared_themes_checkbox", default_value=themes_keep_corpus_wide,
                     enabled=get_mode() == "themes",
                     callback=lambda sender, app_data: set_themes_keep_corpus_wide(app_data))
    dpg.bind_item_theme("word_cloud_shared_themes_checkbox", "disablable_widget_theme")  # tag
    gui_tooltip.Tooltip("word_cloud_shared_themes_checkbox",  # tag
                        "In themes mode, include the themes most clusters share.\n"
                        "Left out, the cloud shows what tells the selected clusters apart, as the cluster labels do.")
    # Self-sizing, as its text changes with what the loaded dataset can do.
    app_state.word_cloud_mode_tooltip = gui_tooltip.Tooltip("word_cloud_mode_radio", _MODE_TOOLTIP)  # tag


def _theme_weights(cluster_llm_keywords, cluster_sizes, *, keep_corpus_wide: bool = False) -> dict[str, int]:
    """Each theme's weight: how many of the selected records are in the clusters whose LLM keywords include it.

    `cluster_llm_keywords`: the dataset's, a list of keywords per cluster ID.
    `cluster_sizes`: `{cluster ID: how many of the selected records are in it}`; Misc (-1) has no themes.
    `keep_corpus_wide`: keep the themes the LLM gave too many clusters to tell any apart
                        (`keyphrases.without_umbrella_terms`), which are left out otherwise, as in the cluster labels.

    One theme spelled two ways by two clusters ("Large language models", "large language models") counts once, under
    the spelling met first.
    """
    if not keep_corpus_wide:
        cluster_llm_keywords = keyphrases.without_umbrella_terms(
            cluster_llm_keywords, threshold_fraction=visualizer_config.clusters_keyword_umbrella_fraction)
    weights: dict[str, int] = {}
    spelled: dict[str, str] = {}
    for cluster_id, n in cluster_sizes.items():
        if cluster_id < 0 or cluster_id >= len(cluster_llm_keywords):
            continue
        for theme in dict.fromkeys(cluster_llm_keywords[cluster_id]):  # once per cluster, in order
            if theme == "<unknown topic>":  # the importer's marker for a cluster the LLM found no theme in
                continue
            shown = spelled.setdefault(theme.casefold(), theme)
            weights[shown] = weights.get(shown, 0) + n
    return weights


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
    global _last_render_key

    logger.debug(f"_render_worker: {task_env.task_name}: Word cloud update task running.")
    try:
        assert task_env is not None
        if task_env.cancelled:
            logger.debug(f"_render_worker: {task_env.task_name}: Word cloud update task cancelled (before starting).")
            return

        data_idxs = task_env.data_idxs
        ds = app_state.dataset  # capture once; a concurrent `open_file` may atomically swap `app_state.dataset` mid-run — keep one consistent view

        if ds is None:
            logger.debug(f"_render_worker: {task_env.task_name}: No dataset loaded. Clearing texture.")
            arr = unbox(_image_box)
            arr[:, :, :3] = 0.0
        else:
            mode = _effective_mode(ds)
            _sync_mode_control(ds)
            # No need to recompute -> just show the window.
            if id(ds) == _last_dataset_addr and set(data_idxs) == _last_data_idxs and _render_key(ds) == _last_render_key:
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
                app_state.word_cloud_tooltip.text = "Generating word cloud, just for you. Please wait. [F10]"
                gui_animation.flash_button(button="word_cloud_button",  # tag
                                           duration=gui_config.acknowledgment_duration)  # the tooltip is handled manually, so it is not flashed along

                # Combine the keyword counts of the specified items: in phrase mode, how many of them hold each
                # phrase, nested phrases counting once as in the cluster labels; in word mode, each word's count;
                # in theme mode, how many of them are in the clusters having each theme.
                logger.debug(f"_render_worker: {task_env.task_name}: Collecting {mode} for selected data points.")
                keywords = collections.defaultdict(lambda: 0)
                record_terms = []
                cluster_sizes = collections.Counter()
                for data_idx in data_idxs:
                    if task_env.cancelled:
                        logger.debug(f"_render_worker: {task_env.task_name}: Word cloud update task cancelled (while collecting keywords).")
                        return
                    entry = ds.sorted_entries[data_idx]
                    if mode == "phrases":
                        record_terms.append(set(getattr(entry, "keyphrases", ())))
                    elif mode == "themes":
                        cluster_sizes[entry.cluster_id] += 1  # the importer's cluster, which the themes describe
                    else:
                        for kw, count in entry.keywords.items():
                            keywords[kw] += count
                if mode == "phrases":
                    display = ds.file_content.keyphrase_display
                    for term, count in keyphrases.top_terms(record_terms, gui_config.word_cloud_max_phrases):
                        keywords[display.get(term, term)] += count
                elif mode == "themes":
                    keywords.update(_theme_weights(ds.file_content.cluster_llm_keywords, cluster_sizes,
                                                   keep_corpus_wide=themes_keep_corpus_wide))

                if not keywords:  # `generate_from_frequencies` raises on nothing; a selection can hold no words at all
                    logger.debug(f"_render_worker: {task_env.task_name}: Nothing to draw for this selection. Clearing texture.")
                    arr[:, :, :3] = 0.0
                else:
                    logger.debug(f"_render_worker: {task_env.task_name}: Invoking word cloud generator.")
                    wc = WordCloud(width=gui_config.word_cloud_w,
                                   height=gui_config.word_cloud_h,
                                   background_color=gui_config.figure_background_color,
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

        _last_dataset_addr = id(ds)  # Conserve RAM by not storing the actual dataset object, but only its memory address. If this changes, it means that the dataset has changed.
        _last_data_idxs = set(data_idxs)
        _last_render_key = _render_key(ds) if ds is not None else None

        logger.debug(f"_render_worker: {task_env.task_name}: Word cloud update task completed.")

    finally:
        dpg.set_item_label("word_cloud_window", "Word cloud")  # TODO: DRY duplicate definitions for labels
        dpg.set_item_label("word_cloud_button", fa.ICON_CLOUD)
        app_state.word_cloud_tooltip.text = "Toggle word cloud window [F10]"  # TODO: DRY duplicate definitions for labels


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

def handle_key(key, ctrl_pressed: bool, shift_pressed: bool) -> bool:
    """Act on `key` if this window binds it. Returns whether it did.

    The bindings live here, beside the window they act on, and the app asks each of its non-modal windows
    in turn. Returning whether the key was taken is what lets a key this window does not bind carry on to
    whoever does — the app itself, or another window that is also open.
    """
    if ctrl_pressed and key == dpg.mvKey_S:
        show_save_dialog()
        return True
    return False


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
    gui_animation.flash_button(button="word_cloud_save_button",  # tag
                               message=f"Saved to '{filename}'!",
                               duration=gui_config.acknowledgment_duration,
                               tooltip=app_state.word_cloud_save_tooltip)

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
    return app_state.filedialog_save.is_visible()


def clear_tasks(wait=False):
    """Cancel any pending word cloud render tasks. Called at app shutdown and on dataset reload."""
    if _task_manager is not None:
        _task_manager.clear(wait=wait)

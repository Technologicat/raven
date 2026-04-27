"""Item information panel for the Visualizer.

Owns the info panel: the background task that rebuilds the panel's content on
selection/search changes, the double-buffered content swap, scroll anchoring,
clipboard/report generation, search-match navigation, cluster navigation, the
"current item" highlight, the dimmer overlay, and the scroll-end flasher.

The info panel is the largest single subsystem in the Visualizer (~1450 lines,
roughly half the program). It shares a rendering vocabulary with the annotation
tooltip (`annotation.py`); the data-gathering and search-fragment regex
compilation common to both live in `entry_renderer.py`.

Public API (all called from `app.py` after the GUI layout exists):

  - `build_window` — one-shot wire-up: store the initial content group ID,
    hook button callbacks, create the scroll-end flasher.
  - `update` — task submitter; registered on `app_state` as
    `update_info_panel` so `selection` and `app.update_search` can call it.
  - `clear_tasks(wait)` — cancel pending render tasks (shutdown/dataset reload).
  - Per-frame updaters (called from `app.update_animations`):
    `update_height`, `update_navigation_controls`,
    `update_current_search_result_status`, `update_current_item_info`.
  - Modal-mode handshake: `scroll_position_changed(reset=True)` to force a
    current-item refresh on modal open/close.
  - Scroll/navigation hotkey + button handlers: `scroll_to_position`,
    `scroll_to_item`, `scroll_to_next_search_match`, `scroll_to_prev_search_match`,
    `scroll_to_next_cluster`, `scroll_to_prev_cluster`,
    `scroll_to_top_of_current_cluster`, `go_to_top`, `go_to_bottom`,
    `page_up`, `page_down`.
  - Clipboard and search-or-select hotkey handlers: `copy_report_to_clipboard`,
    `copy_current_entry_to_clipboard`, `search_or_select_current_entry`,
    `select_cluster_by_id`, `select_current_cluster`.
  - Dimmer overlay: `create_dimmer_overlay`, `rebuild_dimmer_overlay`
    (only called from the resize handler; show/hide is internal).
  - Scroll-end flasher: `flash_scroll_end_by_position(y_scroll)` — tiny wrapper
    used by `app.mouse_wheel_callback`.

Public state (guarded by `content_lock`; swapped atomically by the worker):

  - `content_lock` (RLock), `build_number` (int).
  - `entry_title_widgets`, `widget_to_data_idx`, `widget_to_display_idx`.
  - `search_result_widgets`, `search_result_widget_to_display_idx`.
  - `cluster_ids_in_selection`, `cluster_id_to_display_idx`.
  - `report_plaintext`, `report_markdown` (unpythonic `box` objects).
  - `current_item_info` (+ `current_item_info_lock`) — consumed by the
    `CurrentItemControlsGlow` animation in `app.py`.

Cross-module state read via `app_state`:
  `{dataset, selection_data_idxs_box, selection_changed,
  selection_anchor_data_idxs_set, themes_and_fonts, bg,
  search_string_box, search_result_data_idxs_box, is_any_modal_window_visible,
  update_mouse_hover, update_search}`. Data gathering and search-highlight
  regex compilation go through `entry_renderer`.
"""

__all__ = ["content_lock",
           "entry_title_widgets",
           "widget_to_data_idx",
           "widget_to_display_idx",
           "search_result_widgets",
           "search_result_widget_to_display_idx",
           "cluster_ids_in_selection",
           "cluster_id_to_display_idx",
           "report_plaintext",
           "report_markdown",
           "build_number",
           "current_item_info",
           "current_item_info_lock",
           "build_window",
           "update",
           "clear_tasks",
           "update_height",
           "update_navigation_controls",
           "update_current_search_result_status",
           "update_next_prev_search_result_buttons",
           "update_current_item_info",
           "clear_current_item_info",
           "scroll_position_changed",
           "scroll_to_position",
           "scroll_to_item",
           "scroll_to_next_search_match",
           "scroll_to_prev_search_match",
           "scroll_to_next_cluster",
           "scroll_to_prev_cluster",
           "scroll_to_top_of_current_cluster",
           "go_to_top",
           "go_to_bottom",
           "page_up",
           "page_down",
           "copy_report_to_clipboard",
           "copy_current_entry_to_clipboard",
           "search_or_select_current_entry",
           "select_cluster_by_id",
           "select_current_cluster",
           "create_dimmer_overlay",
           "rebuild_dimmer_overlay",
           "flash_scroll_end_by_position"]

import functools
import gc
import logging
import re
import threading
import time
from io import StringIO

logger = logging.getLogger(__name__)

import dearpygui.dearpygui as dpg

from spacy.lang.en import English

from unpythonic import box, dlet, islice, unbox
from unpythonic.env import env
from unpythonic.env import env as envcls

from ..common import bgtask
from ..common import numutils

from ..common.gui import animation as gui_animation
from ..common.gui import utils as guiutils
from ..common.gui import widgetfinder

from ..vendor.IconsFontAwesome6 import IconsFontAwesome6 as fa
from ..vendor import DearPyGui_Markdown as dpg_markdown

from . import config as visualizer_config
from . import entry_renderer
from . import selection
from .app_state import app_state

gui_config = visualizer_config.gui_config

_nlp_en = English()
_stopwords = _nlp_en.Defaults.stop_words


# --------------------------------------------------------------------------------
# Public state (guarded by `content_lock`; swapped atomically in `_update_info_panel`)

content_lock = threading.RLock()  # Content double buffering (swap). Reentrant so nested acquisitions in the same thread stay simple.

build_number = 0  # Sequence number of last completed info panel build, so callbacks can refer to the GUI widgets of the entries currently in the info panel by their DPG tags.

entry_title_widgets = {}  # `data_idx` (index in `sorted_xxx`) -> DPG ID of GUI widget for the title container group of that entry in the info panel
widget_to_data_idx = {}  # reverse lookup: DPG ID of entry title GUI widget -> `data_idx`
widget_to_display_idx = {}  # reverse lookup: DPG ID of entry title GUI widget -> insertion order in `entry_title_widgets`. Used in scroll anchoring.

search_result_widgets = []  # DPG IDs of entry title container group widgets that match the current search, for the "scroll to next/previous match" buttons.
search_result_widget_to_display_idx = {}  # reverse lookup: DPG ID -> index in `search_result_widgets`

cluster_ids_in_selection = []  # cluster IDs of the clusters currently shown in the info panel. Info panel always shows at least one item per cluster, so this matches the whole selection.
cluster_id_to_display_idx = {}  # reverse lookup: cluster ID -> index in `cluster_ids_in_selection`

report_plaintext = box("")  # Full info panel content in plain text (.txt) format
report_markdown = box("")  # Full info panel content in Markdown (.md) format

current_item_info = env(item=None, x0=None, y0=None, w=None, h=None)  # `item`: GUI widget DPG tag or ID; `x0`, `y0`: screen space coordinates, in pixels; `w`, `h`: in pixels
current_item_info_lock = threading.Lock()


# --------------------------------------------------------------------------------
# Module-local state

_content_group = None  # DPG widget ID of the group currently holding the info panel content (resolved in `build_window`, reassigned on each successful swap in `_update_info_panel`).

_task_manager = None  # bgtask.TaskManager, lazily created on first `update` call (needs `app_state.bg`).

_scroll_end_flasher = None  # gui_animation.ScrollEndFlasher, created in `build_window`.

_dimmer_overlay = None  # gui_animation.Dimmer, created lazily by `create_dimmer_overlay`.

_scroll_animation = None  # reference to the current info panel scroll animation (if any), so we can stop only this animation.
_scroll_animation_lock = threading.RLock()


def _get_task_manager():
    """Lazy-create the info panel render task manager. Requires `app_state.bg` to be set."""
    global _task_manager
    if _task_manager is None:
        _task_manager = bgtask.TaskManager(name="info_panel_update",
                                           mode="sequential",
                                           executor=app_state.bg)
    return _task_manager


def clear_tasks(wait=False):
    """Cancel any pending info panel render tasks. Called at app shutdown and on dataset reload."""
    if _task_manager is not None:
        _task_manager.clear(wait=wait)


# --------------------------------------------------------------------------------
# Window / callback wiring (called once after app.py's GUI layout exists)

def build_window():
    """Create the initial content group, wire button callbacks, create the scroll-end flasher.

    Must be called after `app.py`'s GUI layout has created `item_information_panel` (with the
    `info_panel_content_end_spacer` as its only child) and the navigation/copy/search buttons.
    """
    global _content_group
    global _scroll_end_flasher

    # Create the initial content group as the first child of `item_information_panel`, before the
    # end spacer. Keeping `_content_group` as the int ID returned by `dpg.add_group` (rather than
    # looking it up via `dpg.get_alias_id` later) — int-ID handling is the DPG Pitfall #6 rule.
    _content_group = dpg.add_group(horizontal=False, parent="item_information_panel", before="info_panel_content_end_spacer")  # tag
    dpg.add_text("[Select item(s) to view information]", color=(140, 140, 140, 255), parent=_content_group)  # TODO: DRY duplicate definitions for labels
    dpg.set_item_alias(_content_group, "info_panel_content_group")  # tag  # debug-registry name; int ID is the hot path

    dpg.set_item_callback("go_to_top_button", go_to_top)  # tag
    dpg.set_item_callback("go_to_bottom_button", go_to_bottom)  # tag
    dpg.set_item_callback("page_up_button", page_up)  # tag
    dpg.set_item_callback("page_down_button", page_down)  # tag
    dpg.set_item_callback("copy_report_to_clipboard_button", copy_report_to_clipboard)  # tag
    dpg.set_item_callback("next_search_match_button", scroll_to_next_search_match)  # tag
    dpg.set_item_callback("prev_search_match_button", scroll_to_prev_search_match)  # tag

    _scroll_end_flasher = gui_animation.ScrollEndFlasher(target="item_information_panel",
                                                         tag="scroll_end_flasher",
                                                         duration=gui_config.scroll_ends_here_duration,
                                                         custom_finish_pred=lambda self: app_state.is_any_modal_window_visible(),  # end animation (and hide the flasher) immediately if any modal window becomes visible
                                                         font=app_state.themes_and_fonts.icon_font_solid,
                                                         text_top=fa.ICON_ARROWS_UP_TO_LINE,
                                                         text_bottom=fa.ICON_ARROWS_DOWN_TO_LINE)


def flash_scroll_end_by_position(y_scroll):
    """Show the scroll-end flasher at the given scroll position, if the flasher exists."""
    if _scroll_end_flasher is not None:
        _scroll_end_flasher.show_by_position(y_scroll)


# --------------------------------------------------------------------------------
# Height / content area queries

def update_height():
    """Resize the info panel content area RIGHT NOW, based on main window height.

    HACK: at app startup, the main window thinks it has height=100, which is wrong —
    the scroll-end flasher needs the correct height for its bottom overlay.
    """
    w, h = guiutils.get_widget_size("main_window")  # tag
    dpg.set_item_height("item_information_panel", h - gui_config.info_panel_reserved_h)  # tag


def _get_content_area_start_pos():
    """Return `(x0, y0)`, the upper left corner of the content area, in viewport coordinates."""
    x0, y0 = guiutils.get_widget_pos("item_information_panel")  # tag
    x0_content = x0 + 8 + 3  # 8px outer padding + 3px inner padding
    y0_content = y0 + 8 + 3
    return x0_content, y0_content


def _get_content_area_size():
    """Return `(width, height)`, the size of the content area, in pixels."""
    update_height()  # HACK: at app startup the main window reports height=100.
    return guiutils.get_widget_size("item_information_panel")  # tag


# --------------------------------------------------------------------------------
# User-data helpers and DPG widget filter predicates
#
# Raven stores user data on DPG widgets as `(kind, data)`, where `kind` is a str.
# Predicates return the item on match and `None` otherwise (because 0 is a valid DPG ID).

def _get_user_data(item):
    """Return a DPG widget's user data. Return `None` if not present."""
    if item is None:
        return None
    item_config = dpg.get_item_configuration(item)  # no `try`, so that bugs fail loudly
    try:
        return item_config["user_data"]
    except KeyError:
        pass
    return None


def _is_user_data_kind(value, item):
    """Return `item` if the user data's `kind == value`, else `None`."""
    if item is None:
        return None
    user_data = _get_user_data(item)
    if user_data is not None:
        kind, data = user_data
        if kind == value:
            return item
    return None


def _is_entry_title_container_group(item):  # The container has also the buttons in addition to the actual title text.
    return _is_user_data_kind("entry_title_container", item)


def _is_entry_title_text_item(item):  # The actual title text, actually a group widget containing text snippets, spacers, and such.
    return _is_user_data_kind("entry_title_text", item)


def _is_cluster_title(item):  # e.g. "#42"
    return _is_user_data_kind("cluster_title", item)


def _is_copy_entry_to_clipboard_button(item):
    return _is_user_data_kind("copy_entry_to_clipboard_button", item)


# --------------------------------------------------------------------------------
# Programmatic control of scroll position

def _find_next_or_prev_item(widgets, *, _next=True, kluge=True, extra_y_offset=0):
    """Find the next/previous GUI widget in `widgets`, relative to the top of the content area.

    `widgets` must contain only valid items (no confounders) — allows a classical binary search.
    Parameterized: either the full set of items in the info panel (`entry_title_widgets.values()`)
    or only the search matches (`search_result_widgets`).

    `_next`: if `True` find the next item, else the previous.
    `kluge`: if `True`, reject items within one text-line height of the top (so we look for the
             really next/previous one, not the one currently at the top).
    `extra_y_offset`: useful e.g. to check for the first item *out of view* below the bottom of
                      the content area (set offset to the content area height).
    """
    if not len(widgets):
        return None

    _, y0_content = _get_content_area_start_pos()  # the "current match" sits at the top of the content area
    if kluge:
        kluge = (+1 if _next else -1) * app_state.themes_and_fonts.font_size  # one line of text
    else:
        kluge = 0

    def is_completely_below_top_of_content_area(widget):
        if widgetfinder.is_completely_below_target_y(widget, target_y=y0_content + kluge + extra_y_offset) is not None:
            return widget
        return None

    return widgetfinder.binary_search_widget(widgets=widgets,
                                             accept=is_completely_below_top_of_content_area,
                                             consider=None,
                                             skip=None,
                                             direction=("right" if _next else "left"))


def scroll_to_position(target_y_scroll):
    """Scroll the info panel to the given position.

    Animated if `gui_config.smooth_scrolling` is `True`. Starts a new info panel scroll animation,
    replacing the existing one if any.

    `target_y_scroll`: int; pixels; final position in *info panel* coordinates (not viewport).
                       First possible position is 0. Use `None` to scroll to the end.
                       Value is clamped automatically.

    It is safe to call while a scroll animation is running — the animation adapts on the fly.

    Returns the final target position after `None` resolution and clamping.
    """
    min_y_scroll = 0
    max_y_scroll = dpg.get_y_scroll_max("item_information_panel")  # tag
    if target_y_scroll is None:
        target_y_scroll = max_y_scroll
    target_y_scroll = numutils.clamp(target_y_scroll, min_y_scroll, max_y_scroll)

    global _scroll_animation
    with _scroll_animation_lock:
        with gui_animation.SmoothScrolling.class_lock:
            gui_animation.animator.add(gui_animation.SmoothScrolling(target_child_window="item_information_panel",  # tag
                                                                     target_y_scroll=target_y_scroll,
                                                                     smooth=gui_config.smooth_scrolling,
                                                                     smooth_step=gui_config.smooth_scrolling_step_parameter,
                                                                     flasher=_scroll_end_flasher,
                                                                     finish_callback=_clear_scroll_animation_reference))
            _scroll_animation = gui_animation.SmoothScrolling.instances["item_information_panel"]  # reified instance

    return target_y_scroll


def _clear_scroll_animation_reference():
    """Clear the global scroll animation reference; used as finish callback for `SmoothScrolling`."""
    global _scroll_animation
    with _scroll_animation_lock:
        _scroll_animation = None


def scroll_to_item(item):
    """Scroll the info panel so that, at the new final position, `item` is at the top of the content area.

    Returns the target scroll position (which will be reached later when the animation completes).
    """
    y_scroll = dpg.get_y_scroll("item_information_panel")  # tag
    _, y0_content = _get_content_area_start_pos()
    x1, y1 = dpg.get_item_rect_min(item)  # TODO: handle error case where `item` does not exist
    new_y_scroll = max(0, y_scroll + (y1 - y0_content))
    return scroll_to_position(new_y_scroll)


# --------------------------------------------------------------------------------
# Navigation controls (top/bottom/pageup/pagedown)

def update_navigation_controls():
    """Enable/disable the info panel's top/bottom/pageup/pagedown buttons based on scroll position."""
    current_y_scroll = dpg.get_y_scroll("item_information_panel")  # tag
    max_y_scroll = dpg.get_y_scroll_max("item_information_panel")  # tag
    if max_y_scroll == 0:  # less than one screenful of data
        dpg.disable_item("go_to_top_button")  # tag
        dpg.disable_item("page_up_button")  # tag
        dpg.disable_item("go_to_bottom_button")  # tag
        dpg.disable_item("page_down_button")  # tag
    else:
        if current_y_scroll == 0:
            dpg.disable_item("go_to_top_button")  # tag
            dpg.disable_item("page_up_button")  # tag
        else:
            dpg.enable_item("go_to_top_button")  # tag
            dpg.enable_item("page_up_button")  # tag
        if current_y_scroll == max_y_scroll:
            dpg.disable_item("go_to_bottom_button")  # tag
            dpg.disable_item("page_down_button")  # tag
        else:
            dpg.enable_item("go_to_bottom_button")  # tag
            dpg.enable_item("page_down_button")  # tag


def go_to_top():
    """Scroll the info panel to the top."""
    scroll_to_position(0)


def go_to_bottom():
    """Scroll the info panel to the bottom."""
    scroll_to_position(None)


def page_up():
    """Scroll the info panel up by one page."""
    current_y_scroll = dpg.get_y_scroll("item_information_panel")  # tag
    w_info, h_info = dpg.get_item_rect_size("item_information_panel")  # tag
    new_y_scroll = current_y_scroll - 0.7 * h_info
    scroll_to_position(new_y_scroll)


def page_down():
    """Scroll the info panel down by one page."""
    current_y_scroll = dpg.get_y_scroll("item_information_panel")  # tag
    w_info, h_info = dpg.get_item_rect_size("item_information_panel")  # tag
    new_y_scroll = current_y_scroll + 0.7 * h_info
    scroll_to_position(new_y_scroll)


# --------------------------------------------------------------------------------
# Clipboard: full report and single entry

def copy_report_to_clipboard():
    """Copy all current content of info panel to OS clipboard.

    Default: plain text. With Shift held: Markdown.
    """
    shift_pressed = dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)
    _copy_report_to_clipboard(report_format=("md" if shift_pressed else "txt"))


def _copy_report_to_clipboard(*, report_format):
    """Implementation. `report_format`: one of 'txt', 'md'."""
    if report_format not in ("txt", "md"):
        raise ValueError(f"Unknown report format '{report_format}'; expected one of 'txt', 'md'.")

    if report_format == "txt":
        report_text = unbox(report_plaintext)
    else:
        report_text = unbox(report_markdown)

    dpg.set_clipboard_text(report_text)

    gui_animation.animator.add(gui_animation.ButtonFlash(message=f"Copied to clipboard! ({'plain text' if report_format == 'txt' else 'Markdown'})",
                                                         target_button="copy_report_to_clipboard_button",  # tag
                                                         target_tooltip="copy_report_tooltip",  # tag
                                                         target_text="copy_report_tooltip_text",  # tag
                                                         original_theme=dpg.get_item_theme("copy_report_tooltip"),  # tag
                                                         duration=gui_config.acknowledgment_duration))


def copy_current_entry_to_clipboard():
    """Copy the authors, year and title of the current item to the clipboard.

    The current item is the topmost item visible in the info panel. Hotkey handler.
    """
    with content_lock:  # lock here so we are guaranteed to process the same item throughout
        item = _get_current_item()
        if item is None:
            logger.debug("copy_current_entry_to_clipboard: No current item (info panel empty?)")
            return
        _copy_entry_to_clipboard(item)


def _copy_entry_to_clipboard(item):
    """Implementation. `item` is the DPG ID/tag of the entry title container group.

    We take the widget (not the raw entry) because we need access to the button to play the
    acknowledgment animation on it.
    """
    with content_lock:
        data_idx = widget_to_data_idx[item]
        entry = app_state.dataset.sorted_entries[data_idx]

        button = widgetfinder.find_widget_depth_first(item, accept=_is_copy_entry_to_clipboard_button)
        user_data = _get_user_data(button)
        kind_, data = user_data
        tooltip, tooltip_text = data

    dpg.set_clipboard_text(f"{entry.author} ({entry.year}): {entry.title}")

    gui_animation.animator.add(gui_animation.ButtonFlash(message="Copied to clipboard!",
                                                         target_button=button,
                                                         target_tooltip=tooltip,
                                                         target_text=tooltip_text,
                                                         original_theme=dpg.get_item_theme(tooltip),
                                                         duration=gui_config.acknowledgment_duration))


# --------------------------------------------------------------------------------
# Search-or-select: act on the current item

def search_or_select_current_entry():
    """Search for the current item in the plotter, or change the selection. Hotkey handler.

    The current item is the topmost item visible in the info panel.
    """
    with content_lock:  # faster to acquire just once instead of again inside `_get_current_item`
        item = _get_current_item()
        if item is None:
            logger.debug("search_or_select_current_entry: No current item (info panel empty?)")
            return
        data_idx = widget_to_data_idx[item]
    entry = app_state.dataset.sorted_entries[data_idx]
    _search_or_select_entry(entry)


def _search_or_select_entry(entry):
    """Search for `entry` in the plotter, or change the selection. Implementation.

    Alternative modes:
        no modifier: toggle-search for `entry` in the plotter
        Shift: set selection to `entry` only
        Ctrl:  remove `entry` from selection

    The selection-modifying modes trigger an info panel update.
    """
    shift_pressed = dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)
    ctrl_pressed = dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)

    if shift_pressed:
        selection.update([entry.data_idx], mode="replace", wait=False)
    elif ctrl_pressed:
        selection.update([entry.data_idx], mode="subtract", wait=False)
    else:
        # Exclude stopwords so the MD renderer has fewer short sequences to highlight;
        # also makes "Methanol" not match the "an" inside it (case-insensitive fragments match first).
        filtered_title = " ".join(word for word in entry.title.strip().split() if word.lower() not in _stopwords)
        if dpg.get_value("search_field") != filtered_title:  # tag
            dpg.set_value("search_field", filtered_title)  # tag
        else:  # already searching for this item -> clear the search
            dpg.set_value("search_field", "")  # tag
        app_state.update_search(wait=False)


# --------------------------------------------------------------------------------
# Search-match integration

@dlet(prev_y_scrolls={})
def scroll_position_changed(*, site_tag=None, reset=False, env):
    """Return whether the info panel scrollbar position has changed since the last call.

    `site_tag`: any hashable; each unique tag stores state independently.
    `reset`: if `True`, reset the tracking and store the current position (useful when content
             changes).

    HACK: polling, because there is no callback for child-window scroll position changes.
    Used to drive `update_current_search_result_status`, which is expensive enough to skip when
    not needed.
    """
    if reset or site_tag not in env.prev_y_scrolls:
        env.prev_y_scrolls[site_tag] = None
    if reset:
        return False
    y_scroll = dpg.get_y_scroll("item_information_panel")  # tag
    result = (env.prev_y_scrolls[site_tag] is None or y_scroll != env.prev_y_scrolls[site_tag])
    env.prev_y_scrolls[site_tag] = y_scroll
    return result


def update_current_search_result_status():
    """Update the [x/x] indicator in the info panel, and highlight the current item.

    Runs every frame, so keep it minimal and exit as early as possible.
    """
    if not scroll_position_changed():
        return

    # Avoid race condition: the render worker might swap the content at any moment. Non-blocking
    # acquire — missing a few frames is fine; we must not block the GUI thread.
    if not content_lock.acquire(blocking=False):
        # Technically we should reset scroll tracking here, but the worker already does it after
        # swapping in new content.
        return
    try:
        # This needs `content_lock` for the item search and should only run when the scroll
        # position has changed (like we do), so we do it here despite the search-result framing.
        # TODO: FIX BUG: sometimes the highlight doesn't take right after a click-to-select.
        update_current_item_info()

        if not len(search_result_widgets):
            dpg.hide_item("item_information_search_controls_current_item")  # tag
            return

        # Find the topmost search result below the top of the content area.
        search_result_item = _find_next_or_prev_item(widgets=search_result_widgets, kluge=False)
        if search_result_item is None:  # all matches are above the visible area
            dpg.hide_item("item_information_search_controls_current_item")  # tag
            dpg.enable_item("prev_search_match_button")  # tag
            return
        search_result_display_idx = search_result_widget_to_display_idx[search_result_item]

        # Update the next/prev buttons too — the scroll may have moved regardless of search.
        if search_result_display_idx == 0:
            dpg.disable_item("prev_search_match_button")  # tag
        else:
            dpg.enable_item("prev_search_match_button")  # tag

        if search_result_display_idx == len(search_result_widgets) - 1:
            dpg.disable_item("next_search_match_button")  # tag
        else:
            dpg.enable_item("next_search_match_button")  # tag

        # Is the search result on screen?
        x0_search_result_item, y0_search_result_item = dpg.get_item_rect_min(search_result_item)
        _, y0_content = _get_content_area_start_pos()
        _, h_content = _get_content_area_size()
        # 8px outer padding + 3px inner padding
        if y0_search_result_item >= y0_content + h_content - 8 - 3:  # below the visible area
            dpg.hide_item("item_information_search_controls_current_item")  # tag
            dpg.enable_item("next_search_match_button")  # tag  # unstick in case the above check disabled it (one-result case)
            return
        dpg.set_value("item_information_search_controls_current_item", f"[{1 + search_result_display_idx}/{len(search_result_widgets)}]")  # tag  # 1-based for humans
        dpg.show_item("item_information_search_controls_current_item")  # tag
    finally:
        content_lock.release()


def update_next_prev_search_result_buttons():
    """Enable/disable the next/previous search result buttons.

    Called at the end of an info panel update. `update_current_search_result_status` does this
    separately on scroll changes; here we handle the post-swap case (where the widget list is new).
    """
    with content_lock:  # public API — be careful, could be called from anywhere
        if not len(search_result_widgets):
            dpg.disable_item("next_search_match_button")  # tag
            dpg.disable_item("prev_search_match_button")  # tag
            return
        next_match = _find_next_or_prev_item(widgets=search_result_widgets)
        prev_match = _find_next_or_prev_item(widgets=search_result_widgets, _next=False)
        if next_match is not None:
            dpg.enable_item("next_search_match_button")  # tag
        else:
            dpg.disable_item("next_search_match_button")  # tag
        if prev_match is not None:
            dpg.enable_item("prev_search_match_button")  # tag
        else:
            dpg.disable_item("prev_search_match_button")  # tag


def scroll_to_next_search_match():
    """Scroll the info panel to the next item matching the current search."""
    # TODO: Fix race. Hammering this button can start the next update before the previous one
    # finishes rendering, causing the item search to raise `RuntimeError`. We have no way to know
    # when DPG has finished updating all viewport-coordinate item positions; one `split_frame`
    # does not always suffice. For now we silence — we just miss one click from the hammering.
    # `update_current_search_result_status` will update the nav buttons at the next frame.
    try:
        with content_lock:
            if (next_match := _find_next_or_prev_item(widgets=search_result_widgets)) is not None:
                scroll_to_item(next_match)
    except RuntimeError:
        pass


def scroll_to_prev_search_match():
    """Scroll the info panel to the previous item matching the current search."""
    try:
        with content_lock:
            if (prev_match := _find_next_or_prev_item(widgets=search_result_widgets, _next=False)) is not None:
                scroll_to_item(prev_match)
    except RuntimeError:
        pass


# --------------------------------------------------------------------------------
# Cluster navigation

def _get_current_item():
    """Return the DPG ID/tag of the current item (topmost fully-visible item in the info panel)."""
    with content_lock:
        # TODO: Performance: `update_current_search_result_status` may call us per-frame.
        # Consider storing the list too in `_update_info_panel` to avoid rebuilding it every call.
        return _find_next_or_prev_item(widgets=list(entry_title_widgets.values()), kluge=False)


def _get_cluster_of_current_item():
    """Return the cluster ID of the current item (topmost fully-visible item in the info panel)."""
    with content_lock:
        current_item = _get_current_item()
        if current_item is not None:
            data_idx = widget_to_data_idx[current_item]
            entry = app_state.dataset.sorted_entries[data_idx]
            return entry.cluster_id
        logger.debug("_get_cluster_of_current_item: No current item (info panel empty?)")
        return None


def _scroll_to_cluster_by_id(cluster_id):
    """Scroll to `cluster_id` (must be one of those currently shown; see `cluster_ids_in_selection`)."""
    if cluster_id is None:
        return
    scroll_to_item(f"cluster_{cluster_id}_title_build{build_number}")  # tag  # see `_update_info_panel`


def _get_cluster_display_idx_of_current_item():
    """Return the index in `cluster_ids_in_selection` of the cluster the current item belongs to.

    The index (0-based) tells how-manyth cluster visible in the info panel it is. Useful for
    getting the next/previous displayed cluster. Returns the index (int) or `None` on failure.
    """
    with content_lock:
        cluster_id = _get_cluster_of_current_item()
        try:
            display_idx = cluster_id_to_display_idx[cluster_id]
        except KeyError:
            logger.debug(f"_get_cluster_display_idx_of_current_item: Cluster #{cluster_id} not found (maybe this cluster is currently not shown in info panel?)")
            return None
    return display_idx


def _scroll_to_cluster_by_display_idx(display_idx):
    """Scroll to the cluster identified by its sequential index in the info panel.

    NOTE: sequential index, not cluster ID. See `cluster_ids_in_selection` for the corresponding
    cluster IDs.
    """
    with content_lock:
        if (display_idx is not None) and (display_idx >= 0) and (display_idx <= len(cluster_ids_in_selection) - 1):
            cluster_id = cluster_ids_in_selection[display_idx]
            _scroll_to_cluster_by_id(cluster_id)


def scroll_to_next_cluster():
    """Scroll the info panel to the next cluster, starting from the cluster of the current item."""
    with content_lock:
        display_idx = _get_cluster_display_idx_of_current_item()
        if display_idx is not None:
            _scroll_to_cluster_by_display_idx(display_idx + 1)


def scroll_to_prev_cluster():
    """Scroll the info panel to the previous cluster, starting from the cluster of the current item."""
    with content_lock:
        display_idx = _get_cluster_display_idx_of_current_item()
        if display_idx is not None:
            _scroll_to_cluster_by_display_idx(display_idx - 1)


def scroll_to_top_of_current_cluster():
    """Scroll the info panel to the top of the cluster of the current item."""
    with content_lock:
        cluster_id = _get_cluster_of_current_item()
        _scroll_to_cluster_by_id(cluster_id)


def select_cluster_by_id(cluster_id):
    """Select all data in cluster `cluster_id`.

    Shift/Ctrl/Ctrl+Shift modes available. Triggers an info panel update if the selection changes.
    """
    data_idxs = [data_idx for data_idx, entry in enumerate(app_state.dataset.sorted_entries) if entry.cluster_id == cluster_id]
    selection.update(data_idxs, selection.keyboard_state_to_mode(), wait=False)


def select_current_cluster():
    """Select all data in the same cluster as the current item. Hotkey handler."""
    with content_lock:  # faster to acquire just once instead of again inside `_get_current_item`
        item = _get_current_item()
        if item is None:
            logger.debug("select_current_cluster: No current item (info panel empty?)")
            return
        data_idx = widget_to_data_idx[item]
    entry = app_state.dataset.sorted_entries[data_idx]
    select_cluster_by_id(entry.cluster_id)


# --------------------------------------------------------------------------------
# Current-item tracking (drives the `CurrentItemControlsGlow` animation in `app.py`)

def update_current_item_info():
    """Update the on-screen position of the current item.

    Called per-frame by `update_current_search_result_status` (which already holds
    `content_lock`, avoiding unnecessary release/re-lock).

    When any modal window is visible, the info is cleared, disabling the highlight.
    """
    with content_lock:
        if app_state.is_any_modal_window_visible():
            current_item = None
        else:
            current_item = _get_current_item()
        if current_item is not None:
            current_item_x0, current_item_y0 = dpg.get_item_rect_min(current_item)
            current_item_w, current_item_h = dpg.get_item_rect_size(current_item)
        else:
            current_item_x0, current_item_y0 = None, None
            current_item_w, current_item_h = None, None
    with current_item_info_lock:
        current_item_info.item = current_item
        current_item_info.x0 = current_item_x0
        current_item_info.y0 = current_item_y0
        current_item_info.w = current_item_w
        current_item_info.h = current_item_h


def clear_current_item_info():
    """Clear the on-screen position of the current item (turns the highlight off)."""
    with current_item_info_lock:
        current_item_info.item = None
        current_item_info.x0 = None
        current_item_info.y0 = None
        current_item_info.w = None
        current_item_info.h = None


# --------------------------------------------------------------------------------
# Dimmer overlay (shown while the info panel is updating)

def create_dimmer_overlay():
    """Create the info panel dimmer. Idempotent — no-op if already created."""
    global _dimmer_overlay
    if _dimmer_overlay is None:
        _dimmer_overlay = gui_animation.Dimmer(target="item_information_panel",  # tag
                                               tag="dimmer_overlay_window",
                                               color=(37, 37, 38, 255))  # TODO: this is the info panel content background color in the default theme. Figure out how to read colors from a theme.
        _dimmer_overlay.build()


def _show_dimmer_overlay():
    create_dimmer_overlay()
    _dimmer_overlay.show()


def _hide_dimmer_overlay():
    create_dimmer_overlay()
    _dimmer_overlay.hide()


def rebuild_dimmer_overlay():
    """Rebuild the dimmer (called from the viewport-resize handler)."""
    if _dimmer_overlay is not None:
        _dimmer_overlay.build(rebuild=True)


# --------------------------------------------------------------------------------
# Info panel updater: task submitter

def update(*, wait=True, wait_duration=0.25):
    """Update the data displayed in the info panel. Public API.

    Markdown rendering may take a while, so we try to avoid triggering extra updates.

    `wait`: whether to wait a short cancellation period before starting the update (debounces
            rapid repeat triggers, e.g. keyboard input).
    `wait_duration`: float, seconds.

    **Implementation notes**

    A call is posted by the GUI event queue whenever the search field or selection changes. To
    keep the GUI responsive, we return quickly and run the update asynchronously.

    DPG supports GUI updates from arbitrary threads (as long as you are careful, as usual in
    concurrent programming: don't break state that another thread was accessing — e.g. don't
    clear a panel another thread is populating).

    Async tasks are managed so that:
      - Only one runs at a time (one thread mutates the panel at any given time).
      - Each can wait a cancellation period in "pending" before actually starting.
      - Pending tasks superseded by a new one are cancelled.

    The last two features let the user type more into the search field without each letter
    triggering a (possibly lengthy) update separately. *Running* tasks are never cancelled.
    """
    info_panel_render_task = bgtask.ManagedTask(category="raven_visualizer_info_panel_render",
                                                entrypoint=_update_info_panel,
                                                running_poll_interval=0.1,
                                                pending_wait_duration=wait_duration)
    _get_task_manager().submit(info_panel_render_task, envcls(wait=wait))


# --------------------------------------------------------------------------------
# Info panel updater: worker

@dlet(scroll_anchor_data={},  # stripped tag -> y_diff, where stripped tag is without the "_buildX" suffix
      internal_build_number=0)  # for making unique DPG tags; incremented on every call (completed or cancelled)
def _update_info_panel(*, task_env=None, env=None):
    """Perform the actual update.

    NOTE: in this function ONLY, we don't need to acquire `content_lock` to guard against sudden
    content swaps, because this function is the only one that does those swaps, and it is only
    ever entered with the info panel render lock held (managed in the `ManagedTask`).
    """
    # For "double-buffering"
    global _content_group
    global build_number
    global _scroll_animation

    info_panel_content_target = None  # DPG widget for building new content, initialized later
    new_content_swapped_in = False

    logger.debug(f"_update_info_panel: {task_env.task_name}: Info panel update task running.")
    logger.debug(f"_update_info_panel: {task_env.task_name}: Info panel build {env.internal_build_number} starting.")
    info_panel_t0 = time.monotonic()

    # --------------------------------------------------------------------------------
    # Prepare search result highlighting

    selection_data_idxs = unbox(app_state.selection_data_idxs_box)
    search_result_data_idxs = unbox(app_state.search_result_data_idxs_box)
    search_string = unbox(app_state.search_string_box)
    maybe_regex_case_sensitive, maybe_regex_case_insensitive = entry_renderer.compile_search_highlight_regexes(search_string)

    # --------------------------------------------------------------------------------
    # Preserve scroll position across the update when possible.
    #
    # Ship-of-Theseus: the info panel is completely repopulated every time, so "the same"
    # scroll position does not exist. We anchor instead: find an entry title container group at
    # least partially in view, look for the same entry after the rebuild, and compute the new
    # scroll position so its y-coordinate on screen stays the same.

    def get_scroll_anchor_item_data(item):  # DEBUG only
        try:
            item_config = dpg.get_item_configuration(item)
            user_data = item_config["user_data"]
            if user_data is not None:
                kind_, data_idx = user_data
                entry = app_state.dataset.sorted_entries[data_idx]
                return f"{entry.author} ({entry.year}): {entry.title}"
        except Exception:
            pass
        return None

    def strip_build_number_from_tag(tag):
        m = tag.rfind("_build")
        if m == -1:
            return tag
        return tag[:m]

    def compute_scroll_anchors():
        """Compute scroll anchors from the old, before-update data."""
        # If the selection is the same, or at least one item is common, we can try to anchor.
        # NOTE: We may find an anchor item that turns out to not be shown *after* the update —
        # this step succeeds but the anchor search (later, after the update) finds nothing.
        if (not app_state.selection_changed) or len(app_state.selection_anchor_data_idxs_set):
            logger.debug(f"_update_info_panel.compute_scroll_anchors: {task_env.task_name}: Old data: Start finding scroll anchor item...")

            current_y_scroll = dpg.get_y_scroll("item_information_panel")  # tag

            # Performance: consider only possible anchor items (common to old and new selection).
            # Fortunately we listed them during the previous build — they are the entry container
            # groups, so we can do a classical binary search (O(log n)).
            #
            # This avoids scanning the info panel children linearly: in the no-valid-anchor case
            # that would scan the whole list (>10 s for ~400 entries).
            #
            # NOTE: we scan the *old entries*, because we want the anchor from pre-update content.
            env.scroll_anchor_data.clear()
            if app_state.selection_changed:
                possible_anchors_only = [item for data_idx, item in entry_title_widgets.items() if data_idx in app_state.selection_anchor_data_idxs_set]
                logger.debug(f"_update_info_panel.compute_scroll_anchors: {task_env.task_name}: Selection changed; old info panel selection_data_idxs common with new selection: {list(sorted(widget_to_data_idx[x] for x in possible_anchors_only))}")
            else:
                logger.debug(f"_update_info_panel.compute_scroll_anchors: {task_env.task_name}: Selection not changed; can anchor on any info panel item.")
                possible_anchors_only = list(entry_title_widgets.values())
            is_partially_below_top_of_viewport = functools.partial(widgetfinder.is_partially_below_target_y, target_y=0)
            item = widgetfinder.binary_search_widget(widgets=possible_anchors_only,
                                                     accept=is_partially_below_top_of_viewport,
                                                     consider=None,
                                                     skip=None,
                                                     direction="right")

            # Multi-anchor: anchor on any visible item.
            # May help if the topmost item is not shown after the rebuild but another one is.
            # Not a complete solution — may still fail if none are shown after the rebuild.
            if item is not None:
                start_display_idx = widget_to_display_idx[item]
                _, info_panel_h = _get_content_area_size()
                is_partially_above_bottom_of_viewport = functools.partial(widgetfinder.is_partially_above_target_y, target_y=info_panel_h)
                visible_items = []
                for item_ in islice(entry_title_widgets.values())[start_display_idx:]:
                    if not is_partially_above_bottom_of_viewport(item_):
                        break
                    visible_items.append(item_)
                scroll_anchors_debug_str = "\n    ".join(f"{item_}, tag '{dpg.get_item_alias(item_)}', type {dpg.get_item_type(item_)}, data_idx {widget_to_data_idx[item_]}" for item_ in visible_items)
                plural_s = "s" if len(visible_items) != 1 else ""
                scroll_anchors_final_debug_str = f", list follows.\n    {scroll_anchors_debug_str}" if len(visible_items) else "."
                logger.debug(f"_update_info_panel.compute_scroll_anchors: {task_env.task_name}: Old data: Found {len(visible_items)} at least partially visible scroll-anchorable item{plural_s}{scroll_anchors_final_debug_str}")

                content_start_x0, content_start_y0 = dpg.get_item_rect_min(_content_group)
                for item in visible_items:
                    raw = str(item)
                    alias = dpg.get_item_alias(item)
                    item_str = f"{item}, tag '{dpg.get_item_alias(item)}'" if raw != alias else f"'{alias}'"
                    logger.debug(f"_update_info_panel.compute_scroll_anchors: {task_env.task_name}: Old data: Recording scroll anchor {item_str}, data_idx {widget_to_data_idx[item]}.")
                    try:
                        logger.debug(f"_update_info_panel.compute_scroll_anchors: {task_env.task_name}: Old data:     Item type is {dpg.get_item_type(item)}")
                        if (anchor_item_data := get_scroll_anchor_item_data(item)) is not None:
                            logger.debug(f"_update_info_panel.compute_scroll_anchors: {task_env.task_name}: Old data:     Item data is '{anchor_item_data}'")
                    except Exception:  # not found (race condition?)
                        pass

                    # NOTE: a DPG group must be rendered at least once to have a meaningful size.
                    x0, y0 = dpg.get_item_rect_min(item)
                    w, h = dpg.get_item_rect_size(item)
                    logger.debug(f"_update_info_panel.compute_scroll_anchors: {task_env.task_name}: Old data:     Item old position is y0 = {y0}, y_last = {y0 + h - 1}")

                    item_y_offset_from_content_start = y0 - content_start_y0
                    # Additive conversion: difference between scrollbar position and info-panel
                    # y-coordinate of the anchor item's start.
                    y_diff = current_y_scroll - item_y_offset_from_content_start

                    stripped_tag = strip_build_number_from_tag(dpg.get_item_alias(item))
                    env.scroll_anchor_data[stripped_tag] = y_diff

                    logger.debug(f"_update_info_panel.compute_scroll_anchors: {task_env.task_name}: Old data:     Content area start y = {content_start_y0}, item y = {y0}, Δy = {item_y_offset_from_content_start}, scroll position = {current_y_scroll}, diff = {-y_diff}")
                plural_s = "s" if len(env.scroll_anchor_data) != 1 else ""
                logger.debug(f"_update_info_panel.compute_scroll_anchors: {task_env.task_name}: Old data: Scroll anchors updated. Found {len(env.scroll_anchor_data)} possible anchor{plural_s}.")
            else:
                # Items common to old/new selection exist but none are on-screen (or included) before the update.
                logger.debug(f"_update_info_panel.compute_scroll_anchors: {task_env.task_name}: Old data: Selection has changed with no anchorable item in info panel. Resetting scroll anchors.")
                env.scroll_anchor_data.clear()
        else:
            # No items common between old and new -> info panel content changes completely.
            logger.debug(f"_update_info_panel.compute_scroll_anchors: {task_env.task_name}: Old data: Selection has changed with no items common with previous selection. Resetting scroll anchors.")
            env.scroll_anchor_data.clear()

    new_y_scroll = None  # for setting the scroll position when the render completes

    def compute_new_scroll_target_position(anchor_tag):
        """Compute new scroll position based on DPG GUI widget `anchor_tag` (in new data).

        Uses the anchor's recorded diff (scroll position vs. anchor-in-viewport coordinates)
        from the old data to apply the same diff to the new scroll position.
        """
        nonlocal new_y_scroll

        data_idx = widget_to_data_idx_new.get(anchor_tag, "unknown")

        if new_y_scroll is not None:
            logger.debug(f"_update_info_panel.compute_new_scroll_target_position: {task_env.task_name}: New data: Detected scroll anchor '{anchor_tag}', data_idx {data_idx}, but new scroll position already recorded. Skipping.")
            return

        stripped_tag = strip_build_number_from_tag(anchor_tag)
        if stripped_tag not in env.scroll_anchor_data:
            raise ValueError(f"_update_info_panel.compute_new_scroll_target_position: {task_env.task_name}: New data: ERROR: '{anchor_tag}', data_idx {data_idx} not in recorded scroll anchors.")

        logger.debug(f"_update_info_panel.compute_new_scroll_target_position: {task_env.task_name}: New data: Detected scroll anchor '{anchor_tag}', data_idx {data_idx}.")

        try:
            logger.debug(f"_update_info_panel.compute_new_scroll_target_position: {task_env.task_name}: New data:     Item type is {dpg.get_item_type(anchor_tag)}")
            if (anchor_item_data := get_scroll_anchor_item_data(anchor_tag)) is not None:
                logger.debug(f"_update_info_panel.compute_new_scroll_target_position: {task_env.task_name}: New data:     Item data is '{anchor_item_data}'")
        except Exception:  # not found (race condition?)
            pass

        new_x0, new_y0 = dpg.get_item_rect_min(anchor_tag)
        new_w, new_h = dpg.get_item_rect_size(anchor_tag)
        logger.debug(f"_update_info_panel.compute_new_scroll_target_position: {task_env.task_name}: New data:     Item new position is y0 = {new_y0}, y_last = {new_y0 + new_h - 1}")

        new_content_start_x0, new_content_start_y0 = dpg.get_item_rect_min(info_panel_content_target)
        new_item_y_offset_from_content_start = new_y0 - new_content_start_y0

        logger.debug(f"_update_info_panel.compute_new_scroll_target_position: {task_env.task_name}: New data:     Content area start y = {new_content_start_y0}, item y = {new_y0}, Δy = {new_item_y_offset_from_content_start}")

        new_y_scroll = max(0, new_item_y_offset_from_content_start + env.scroll_anchor_data[stripped_tag])

        logger.debug(f"_update_info_panel.compute_new_scroll_target_position: {task_env.task_name}: New data: New scroll position recorded: {new_y_scroll}.")

    # --------------------------------------------------------------------------------
    # Start rebuilding the info panel content.

    dpg.set_value("item_information_total_count", "[updating]")  # tag  # final value set when the update completes
    dpg.show_item("item_information_total_count")  # tag
    if search_string:
        dpg.set_value("item_information_search_controls_item_count", "[updating]")  # tag
    else:
        dpg.set_value("item_information_search_controls_item_count", "[no search active]")  # tag  # TODO: DRY duplicate definitions for labels

    # Make sure any partially built old content is really gone (including tags, which must be unique).
    # Hammering LMB over the plotter causes many selection changes -> many cancellations.
    gc.collect()
    dpg.split_frame()

    # `with dpg.group(...):` would look clearer, but don't touch the DPG container stack from a background thread.
    info_panel_content_target = dpg.add_group(horizontal=False, show=False, parent="item_information_panel", before="info_panel_content_end_spacer")  # tag

    # After this point the content target exists; cleanup is needed on failure.
    try:
        entry_title_widgets_new = {}
        widget_to_data_idx_new = {}
        widget_to_display_idx_new = {}
        search_result_widgets_new = []
        search_result_widget_to_display_idx_new = {}
        cluster_ids_in_selection_new = []
        cluster_id_to_display_idx_new = {}

        entries_by_cluster, formatter = entry_renderer.get_entries_for_selection(selection_data_idxs, max_n=gui_config.max_items_in_info_panel)

        cluster_ids_in_selection_new.clear()
        cluster_id_to_display_idx_new.clear()
        cluster_ids = entry_renderer.order_cluster_ids(entries_by_cluster.keys())
        cluster_ids_in_selection_new.extend(cluster_ids)
        cluster_id_to_display_idx_new.update({cluster_id: display_idx for display_idx, cluster_id in enumerate(cluster_ids_in_selection_new)})

        # Update the info panel title (talks about the whole selection, not the subset shown).
        # `selection_data_idxs` covers the whole selection; the info panel always shows at least
        # one entry per cluster, so the cluster count matches the whole selection. The item count
        # gets its final value when the update completes.
        if len(selection_data_idxs):
            item_plural_s = "s" if len(selection_data_idxs) != 1 else ""
            cluster_plural_s = "s" if len(cluster_ids_in_selection_new) != 1 else ""
            top_heading_text = f"[{len(selection_data_idxs)} item{item_plural_s} total in {len(cluster_ids_in_selection_new)} cluster{cluster_plural_s}]"
        else:
            top_heading_text = "[nothing selected]"  # TODO: DRY duplicate definitions for labels
            dpg.add_text("[Select item(s) to view information]", color=(140, 140, 140, 255), parent=info_panel_content_target)  # TODO: DRY duplicate definitions for labels
        dpg.set_value("item_information_selection_item_count", top_heading_text)  # tag

        # Start writing the report (exported to clipboard).
        report_text = StringIO()
        report_text.write(top_heading_text + "\n")
        report_text.write("=" * len(top_heading_text) + "\n\n")
        report_md = StringIO()
        report_md.write(f"# {top_heading_text}\n\n")

        # Per-entry button callback factories
        def make_scroll_to_cluster(display_idx):
            """Callback to scroll the info panel to the given cluster by sequential index."""
            def scroll():  # freeze `display_idx` by closure
                _scroll_to_cluster_by_display_idx(display_idx)
            return scroll

        def make_copy_entry_to_clipboard(title_container_group):
            """Callback to copy the authors/year/title of the given entry to the OS clipboard."""
            def copy_this_entry():
                _copy_entry_to_clipboard(title_container_group)  # needs access to the widget for the acknowledgment animation
            return copy_this_entry

        def make_search_or_select_entry(entry):
            """Callback to search for the given item in the plotter, or change the selection."""
            def search_or_select():
                _search_or_select_entry(entry)
            return search_or_select

        def make_select_cluster(cluster_id):
            """Callback to select all data in cluster `cluster_id`.

            Shift, Ctrl, Ctrl+Shift modes available. Triggers an info panel update.
            """
            def select_this_cluster():
                select_cluster_by_id(cluster_id)
            return select_this_cluster

        # Build info panel content and write report.
        total_entries_shown_in_info_panel = 0
        for display_idx, cluster_id in enumerate(cluster_ids_in_selection_new):
            dpg.set_value("item_information_total_count", f"[updating {display_idx + 1}/{len(cluster_ids_in_selection_new)}]")  # tag

            first_cluster = (display_idx == 0)
            last_cluster = (display_idx == len(cluster_ids_in_selection_new) - 1)

            if task_env is not None and task_env.cancelled:
                break

            cluster_title, cluster_keywords, cluster_content, more = formatter(cluster_id)
            total_entries_shown_in_info_panel += len(cluster_content)

            cluster_header_group = dpg.add_group(horizontal=True, parent=info_panel_content_target, tag=f"cluster_{cluster_id}_header_group_build{env.internal_build_number}")

            # Next/previous cluster buttons
            up_enabled = (not first_cluster)
            up_button = dpg.add_button(tag=f"cluster_{cluster_id}_up_button_build{env.internal_build_number}",
                                       arrow=True,
                                       direction=dpg.mvDir_Up,
                                       enabled=up_enabled,
                                       callback=(make_scroll_to_cluster(display_idx - 1) if up_enabled else lambda: None),
                                       parent=cluster_header_group)
            up_tooltip = dpg.add_tooltip(up_button)
            dpg.add_text("Previous cluster [Ctrl+P]", parent=up_tooltip)
            dpg.bind_item_theme(f"cluster_{cluster_id}_up_button_build{env.internal_build_number}", "disablable_widget_theme")  # tag

            down_enabled = (not last_cluster)
            down_button = dpg.add_button(tag=f"cluster_{cluster_id}_down_button_build{env.internal_build_number}",
                                         arrow=True,
                                         direction=dpg.mvDir_Down,
                                         enabled=down_enabled,
                                         callback=(make_scroll_to_cluster(display_idx + 1) if down_enabled else lambda: None),
                                         parent=cluster_header_group)
            down_tooltip = dpg.add_tooltip(down_button)
            dpg.add_text("Next cluster [Ctrl+N]", parent=down_tooltip)
            dpg.bind_item_theme(f"cluster_{cluster_id}_down_button_build{env.internal_build_number}", "disablable_widget_theme")  # tag

            # Cluster title and keywords
            cluster_title_widget = dpg.add_text(cluster_title, tag=f"cluster_{cluster_id}_title_build{env.internal_build_number}", color=(180, 180, 180), parent=cluster_header_group)
            dpg.set_item_user_data(cluster_title_widget, ("cluster_title", cluster_id))  # for `_is_cluster_title`
            plural_s = "s" if len(entries_by_cluster[cluster_id]) != 1 else ""
            entries_text = f"[{len(entries_by_cluster[cluster_id])} item{plural_s}]"
            dpg.add_text(entries_text, wrap=0, color=(140, 140, 140), tag=f"cluster_{cluster_id}_item_count_build{env.internal_build_number}", parent=cluster_header_group)
            dpg.add_text(cluster_keywords, wrap=0, color=(140, 140, 140), tag=f"cluster_{cluster_id}_keywords_build{env.internal_build_number}", parent=cluster_header_group)

            # Report: cluster heading
            report_cluster_heading_text = f"{cluster_title} {entries_text} {cluster_keywords}".strip()
            report_text.write(report_cluster_heading_text + "\n")
            report_text.write("-" * len(report_cluster_heading_text) + "\n\n")
            report_md.write(f"## {report_cluster_heading_text}\n\n")

            # Cluster title separator
            cluster_title_separator = dpg.add_drawlist(width=gui_config.info_panel_w - 20, height=1, parent=info_panel_content_target, tag=f"cluster_{cluster_id}_title_separator_build{env.internal_build_number}")
            dpg.draw_line((0, 0), (gui_config.info_panel_w - 21, 0), color=(140, 140, 140, 255), thickness=1, parent=cluster_title_separator)

            # Items in cluster
            for data_idx, entry in cluster_content:
                if task_env is not None and task_env.cancelled:
                    break

                # Highlight search results only when a search is active.
                if not search_string or data_idx in search_result_data_idxs:
                    use_bright_text = True
                    title_color = (255, 255, 255, 255)
                    abstract_color = (180, 180, 180, 255)
                else:
                    use_bright_text = False
                    title_color = (140, 140, 140, 255)
                    abstract_color = (110, 110, 110, 255)
                is_search_match = (search_string and use_bright_text)

                # Containers
                entry_container_group = dpg.add_group(parent=info_panel_content_target, tag=f"cluster_{cluster_id}_entry_{data_idx}_build{env.internal_build_number}")
                entry_title_container_group = dpg.add_group(horizontal=True, tag=f"cluster_{cluster_id}_entry_{data_idx}_header_group_build{env.internal_build_number}", parent=entry_container_group)

                # Per-item buttons, column 1
                entry_buttons_column_1_group = dpg.add_group(horizontal=False, tag=f"cluster_{cluster_id}_entry_{data_idx}_header_button_column_1_group_build{env.internal_build_number}", parent=entry_title_container_group)

                # Back to top of this cluster
                b = dpg.add_button(tag=f"cluster_{cluster_id}_entry_{data_idx}_back_to_cluster_top_button_build{env.internal_build_number}",
                                   arrow=True,
                                   direction=dpg.mvDir_Up,
                                   callback=make_scroll_to_cluster(display_idx),
                                   parent=entry_buttons_column_1_group)
                dpg.bind_item_font(b, app_state.themes_and_fonts.icon_font_solid)
                b_tooltip = dpg.add_tooltip(b)
                b_tooltip_text = dpg.add_text(f"Back to top of cluster #{cluster_id} [Ctrl+U]" if cluster_id != -1 else "Back to top of Misc [Ctrl+U]",
                                              parent=b_tooltip)

                # Copy this item to clipboard
                b = dpg.add_button(label=fa.ICON_COPY,
                                   tag=f"cluster_{cluster_id}_entry_{data_idx}_copy_to_clipboard_button_build{env.internal_build_number}",
                                   width=gui_config.info_panel_button_w,
                                   parent=entry_buttons_column_1_group)
                dpg.bind_item_font(b, app_state.themes_and_fonts.icon_font_solid)
                b_tooltip = dpg.add_tooltip(b)
                b_tooltip_text = dpg.add_text("Copy item authors, year and title to clipboard [Ctrl+Shift+C]",  # TODO: DRY duplicate definitions for labels
                                              parent=b_tooltip)
                dpg.set_item_callback(b, make_copy_entry_to_clipboard(entry_title_container_group))
                dpg.set_item_user_data(b, ("copy_entry_to_clipboard_button", (b_tooltip, b_tooltip_text)))  # for `copy_current_entry_to_clipboard` hotkey

                # Per-item buttons, column 2
                entry_buttons_column_2_group = dpg.add_group(horizontal=False, tag=f"cluster_{cluster_id}_entry_{data_idx}_header_button_column_2_group_build{env.internal_build_number}", parent=entry_title_container_group)

                # Search this item in plotter
                b = dpg.add_button(label=fa.ICON_ARROW_RIGHT,
                                   tag=f"cluster_{cluster_id}_entry_{data_idx}_search_in_plotter_button_build{env.internal_build_number}",
                                   width=gui_config.info_panel_button_w,
                                   parent=entry_buttons_column_2_group)
                dpg.bind_item_font(b, app_state.themes_and_fonts.icon_font_solid)
                b_tooltip = dpg.add_tooltip(b)
                dpg.add_text("Search for this item in the plotter [F6]\n(clear search if already searching for this item)\n    with Shift: set selection to this item only\n    with Ctrl: remove this item from selection",
                             parent=b_tooltip)
                dpg.set_item_callback(b, make_search_or_select_entry(entry))

                b = dpg.add_button(label=fa.ICON_WAND_MAGIC_SPARKLES,  # wand, by analogy with smart select in graphics programs
                                   tag=f"cluster_{cluster_id}_entry_{data_idx}_select_this_cluster_button_build{env.internal_build_number}",
                                   width=gui_config.info_panel_button_w,
                                   parent=entry_buttons_column_2_group)
                dpg.bind_item_font(b, app_state.themes_and_fonts.icon_font_solid)
                b_tooltip = dpg.add_tooltip(b)
                cluster_name_str = f"#{cluster_id}" if cluster_id != -1 else "Misc"
                dpg.add_text(f"Select all items in the same cluster ({cluster_name_str}) as this item [F7]\n    with Shift: add\n    with Ctrl: subtract\n    with Ctrl+Shift: intersect",
                             parent=b_tooltip)
                dpg.set_item_callback(b, make_select_cluster(cluster_id))

                # Item authors, year, title (with search result highlight, if any)
                entry_title_text = entry.title
                if search_string:
                    if maybe_regex_case_insensitive:  # case-insensitive first so a fragment like "col" won't match the "<font color=...>"
                        # The font tags don't stack in the MD renderer, so close the surrounding
                        # tag (for title color) when the highlight starts, and re-open it after.
                        entry_title_text = re.sub(maybe_regex_case_insensitive, f"</font>**<font color='#ff0000'>\\1</font>**<font color='{title_color}'>", entry_title_text)
                    if maybe_regex_case_sensitive:  # case-sensitive fragments contain at least one uppercase letter -> safe (won't match anything the case-insensitive pass added)
                        entry_title_text = re.sub(maybe_regex_case_sensitive, f"</font>**<font color='#ff0000'>\\1</font>**<font color='{title_color}'>", entry_title_text)
                if search_string and entry_title_text != entry.title:  # substitutions changed the text -> render as Markdown to enable highlighting
                    header = f"<font color='{title_color}'>{entry.author} ({entry.year}): {entry_title_text}</font>"
                    entry_title_group = dpg_markdown.add_text(header, wrap=gui_config.title_wrap_w, parent=entry_title_container_group, tag=f"cluster_{cluster_id}_entry_{data_idx}_title_build{env.internal_build_number}")  # MD renderer renders into its own group
                    if is_search_match:
                        search_result_widgets_new.append(entry_title_container_group)
                        search_result_widget_to_display_idx_new[entry_title_container_group] = len(search_result_widgets_new) - 1
                else:  # search not active, or no match in this title -> render as plain text (much faster)
                    header = f"{entry.author} ({entry.year}): {entry_title_text}"
                    # Match the MD renderer's line spacing for visual consistency. Before scroll
                    # anchoring was introduced, line-height differences caused big scroll jumps
                    # near the end of a large dataset when toggling match/non-match state.
                    #
                    # The plain and highlighted titles may still use different line counts (bold
                    # is slightly wider, italic slightly narrower, changing color is ~1px wider(?)).
                    # Preserving the original line division would need a slack-aware layout like
                    # the tooltip's entry count limiter. Instead, we anchor the scroll position on
                    # a text item whose content doesn't change across the update.
                    #
                    # 1) Wrap with the MD renderer (DPG lacks a text-wrap utility). A bit slow —
                    #    it's a Python-level low-level utility — but meh.
                    entity = dpg_markdown.text_entities.StrEntity(header)
                    entity = dpg_markdown.wrap_text_entity(entity, width=gui_config.title_wrap_w)  # -> iterable of lines
                    # 2) Render line by line, controlling vertical spacing explicitly with a spacer.
                    entry_title_group = dpg.add_group(horizontal=False, tag=f"cluster_{cluster_id}_entry_{data_idx}_title_build{env.internal_build_number}", parent=entry_title_container_group)
                    for lineno, line_content in enumerate(entity):
                        last_line = (lineno == len(entity) - 1)
                        dpg.add_text(line_content, color=title_color, parent=entry_title_group)
                        if not last_line:
                            # Align with the MD renderer line height. Spacer width doesn't matter.
                            # TODO: no idea where the two extra pixels of height in the MD renderer come from.
                            dpg.add_spacer(width=10, height=2, parent=entry_title_group)
                    dpg.bind_item_theme(entry_title_group, "my_no_spacing_theme")  # tag  # default spacing off, like in the MD renderer

                entry_title_widgets_new[data_idx] = entry_title_container_group
                widget_to_data_idx_new[entry_title_container_group] = data_idx
                widget_to_display_idx_new[entry_title_container_group] = len(entry_title_widgets_new) - 1
                dpg.set_item_user_data(entry_title_container_group, ("entry_title_container", data_idx))  # for `_is_entry_title_container_group`
                dpg.set_item_user_data(entry_title_group, ("entry_title_text", data_idx))  # for `_is_entry_title_text_item`

                # Item abstract (optional)
                if entry.abstract:
                    dpg.add_text(entry.abstract, color=abstract_color, wrap=gui_config.main_text_wrap_w, tag=f"cluster_{cluster_id}_entry_{data_idx}_abstract_build{env.internal_build_number}", parent=entry_container_group)
                dpg.add_text("", tag=f"cluster_{cluster_id}_entry_{data_idx}_end_blank_text_build{env.internal_build_number}", parent=entry_container_group)

                # Report: write item
                if entry.abstract:
                    report_text.write(f"{entry.author} ({entry.year}): {entry.title}\n\n{entry.abstract.strip()}\n\n")
                    report_md.write(f"### {entry.author} ({entry.year}): {entry.title}\n\n{entry.abstract.strip()}\n\n")
                else:
                    report_text.write(f"{entry.author} ({entry.year}): {entry.title}\n\n")  # TODO: tag as "[no abstract]"?
                    report_md.write(f"### {entry.author} ({entry.year}): {entry.title}\n\n")  # TODO: tag as "[no abstract]"?

            if task_env is None or not task_env.cancelled:
                if more:
                    dpg.add_text(more, wrap=0, color=(100, 100, 100), tag=f"cluster_{cluster_id}_more_build{env.internal_build_number}", parent=info_panel_content_target)
                    report_text.write(f"{more}\n\n")
                    report_md.write(f"{more}\n\n")

                # Cluster separator
                if not last_cluster:
                    cluster_end_separator_1 = dpg.add_drawlist(width=gui_config.info_panel_w - 20, height=1, parent=info_panel_content_target, tag=f"cluster_{cluster_id}_end_separator_1_build{env.internal_build_number}")
                    dpg.draw_line((0, 0), (gui_config.info_panel_w - 21, 0), color=(140, 140, 140, 255), thickness=1, parent=cluster_end_separator_1)
                    cluster_end_separator_2 = dpg.add_drawlist(width=gui_config.info_panel_w - 20, height=1, parent=info_panel_content_target, tag=f"cluster_{cluster_id}_end_separator_2_build{env.internal_build_number}")
                    dpg.draw_line((0, 0), (gui_config.info_panel_w - 21, 0), color=(140, 140, 140, 255), thickness=1, parent=cluster_end_separator_2)
                    dpg.add_text("", tag=f"cluster_{cluster_id}_end_blank_text_build{env.internal_build_number}", parent=info_panel_content_target)

                    report_text.write("-" * 80 + "\n")
                    report_text.write("-" * 80 + "\n\n\n")
                    report_md.write("-----\n\n")

        # Finalize (if not cancelled)
        if task_env is None or not task_env.cancelled:
            # About to swap the whole content — stop the scroll animation if running.
            with _scroll_animation_lock:
                if _scroll_animation is not None:
                    _scroll_animation.finish()
                    _scroll_animation = None

            # Anchor the scroll position from old data just before swapping in new content, so
            # we pick up the latest position in case the user scrolled while we were building.
            compute_scroll_anchors()

            # Render the new content so the items get real positions.
            #
            # Using `set_item_pos`/`reset_pos` to render off-screen produces unreliable positions
            # (the y coordinate depends on x for some reason — interaction between info panel
            # child window size and text wrap?). Even setting position to `rect_min` of the old
            # group (exactly on top) sometimes gets it wrong (at least near the end of long
            # content, when switching search on/off).
            #
            # Best to hide the old group, show the new one, and let DPG handle layout. With
            # exactly one container shown, the new one appears exactly where the old one was, and
            # positions measured in the new data work.
            clear_current_item_info()  # highlight off; `update_animations` will auto-update on the next frame
            dpg.hide_item(_content_group)
            dpg.show_item(info_panel_content_target)
            _show_dimmer_overlay()
            dpg.split_frame()  # wait for render, to have valid positions for widgets in the new data

            # Find the new items (if any) corresponding to recorded scroll anchors. Try each; it's
            # a no-op (with a log message) after the scroll position has been set successfully.
            scroll_anchor_stripped_tags = list(env.scroll_anchor_data.keys())
            scroll_anchor_new_tags = [tag for tag in entry_title_widgets_new.values()
                                      if strip_build_number_from_tag(tag) in scroll_anchor_stripped_tags]
            for tag in scroll_anchor_new_tags:
                compute_new_scroll_target_position(tag)

            with content_lock:
                # Swap the new content in ("double-buffering")
                logger.debug(f"_update_info_panel: {task_env.task_name}: Swapping in new content (old GUI widget ID {_content_group}; new GUI widget ID {info_panel_content_target}).")
                dpg.delete_item(_content_group)
                _content_group = None  # in case the next line raises
                dpg.set_item_alias(info_panel_content_target, "info_panel_content_group")  # tag
                _content_group = info_panel_content_target
                new_content_swapped_in = True

                logger.debug(f"_update_info_panel: {task_env.task_name}: Swapping in new navigation metadata.")
                cluster_ids_in_selection.clear()
                cluster_ids_in_selection.extend(cluster_ids_in_selection_new)
                cluster_id_to_display_idx.clear()
                cluster_id_to_display_idx.update(cluster_id_to_display_idx_new)
                entry_title_widgets.clear()
                entry_title_widgets.update(entry_title_widgets_new)
                widget_to_data_idx.clear()
                widget_to_data_idx.update(widget_to_data_idx_new)
                widget_to_display_idx.clear()
                widget_to_display_idx.update(widget_to_display_idx_new)
                search_result_widgets.clear()
                search_result_widgets.extend(search_result_widgets_new)
                search_result_widget_to_display_idx.clear()
                search_result_widget_to_display_idx.update(search_result_widget_to_display_idx_new)

                logger.debug(f"_update_info_panel: {task_env.task_name}: Content swapping complete.")

            # Finish the report
            report_plaintext << report_text.getvalue()
            report_markdown << report_md.getvalue()
            dpg.enable_item("copy_report_to_clipboard_button")  # tag

            # Final item count
            if total_entries_shown_in_info_panel > 0:
                dpg.set_value("item_information_total_count", f"[{total_entries_shown_in_info_panel} item{'s' if total_entries_shown_in_info_panel != 1 else ''} shown]")  # tag
                dpg.show_item("item_information_total_count")  # tag
            else:  # build finished and no items are shown -> hide the total count field
                dpg.hide_item("item_information_total_count")  # tag

            # Restore/reset scroll position
            dpg.split_frame()  # let the content swap take before proceeding
            if not len(env.scroll_anchor_data):
                logger.debug(f"_update_info_panel: {task_env.task_name}: New data: no anchorable items in old data, resetting scroll position.")
                dpg.set_y_scroll("item_information_panel", 0)  # tag
            elif new_y_scroll is not None:
                logger.debug(f"_update_info_panel: {task_env.task_name}: New data: scrolling to anchor, new_y_scroll = {new_y_scroll}")
                dpg.set_y_scroll("item_information_panel", new_y_scroll)  # tag
            else:  # anchorable items exist in old data but none are shown after update
                logger.debug(f"_update_info_panel: {task_env.task_name}: New data: at least one anchor exists, but none are shown after update. Resetting scroll position.")
                dpg.set_y_scroll("item_information_panel", 0)  # tag
            scroll_position_changed(reset=True)  # for `update_current_search_result_status` — *after* swapping in the new content
            app_state.selection_changed = False

            # Items shown may have changed. Re-render the annotation tooltip (if active) to update its "shown in info panel" status.
            app_state.update_mouse_hover(force=True, wait=False)

            # Update search controls last.
            if search_string:
                num = len(search_result_widgets)
                search_results_info_panel_str = f"[{num if num else 'no'} search result{'s' if num != 1 else ''} shown]"
            else:
                search_results_info_panel_str = "[no search active]"  # TODO: DRY duplicate definitions for labels
            dpg.set_value("item_information_search_controls_item_count", search_results_info_panel_str)  # tag
            dpg.split_frame()  # let the scrollbar position update before proceeding
            _hide_dimmer_overlay()
            update_next_prev_search_result_buttons()

    except Exception:
        logger.debug(f"_update_info_panel: {task_env.task_name}: Info panel update task raised an exception; cancelling task.")
        task_env.cancelled = True
        if not new_content_swapped_in:  # re-show the old content (if it still exists) on failure during finalizing
            if info_panel_content_target is not None:
                dpg.hide_item(info_panel_content_target)
            if _content_group is not None:
                dpg.show_item(_content_group)
        raise

    finally:
        if task_env is not None and task_env.cancelled:
            logger.debug(f"_update_info_panel: {task_env.task_name}: Info panel update task cancelled.")

            # Built (partially or fully) but not swapped in -> unused, delete it.
            if (info_panel_content_target is not None) and (not new_content_swapped_in):
                logger.debug(f"_update_info_panel: {task_env.task_name}: Deleting partially built content.")
                dpg.delete_item(info_panel_content_target)

            # These will be refreshed when the next update starts (we only cancel a running task
            # when superseded by a new one); meanwhile show up-to-date status.
            dpg.set_value("item_information_total_count", "[update cancelled]")  # tag
            dpg.set_value("item_information_search_controls_item_count", "[update cancelled]")  # tag
        else:
            logger.debug(f"_update_info_panel: {task_env.task_name}: Info panel update task completed.")

            # Publish the build ID so callbacks can find the content.
            build_number = env.internal_build_number

        dt = time.monotonic() - info_panel_t0
        plural_s = "ies" if total_entries_shown_in_info_panel != 1 else "y"
        logger.debug(f"_update_info_panel: {task_env.task_name}: Info panel build {env.internal_build_number} exiting. Rendered {total_entries_shown_in_info_panel} entr{plural_s} in {dt:0.2f}s.")
        env.internal_build_number += 1  # always increase, even when cancelled, for unique IDs

"""Selection management for the Visualizer's plotter.

Handles the undo/redo stack, the four selection combine modes (`replace`, `add`,
`subtract`, `intersect`), the highlight scatter series update, and the
keyboard-modifier → mode mapping.

Module-local state (the undo stack and the undo cursor) stays encapsulated here —
only `app_state.selection_data_idxs_box` (the current selection) and
`app_state.selection_changed` / `app_state.selection_anchor_data_idxs_set` (scroll-
anchoring handshake with the info panel) are cross-module visible.
"""

__all__ = ["reset_undo_history",
           "commit_change_to_undo_history",
           "undo", "redo",
           "update",
           "update_highlight",
           "keyboard_state_to_mode"]

import numpy as np

import dearpygui.dearpygui as dpg

from unpythonic import box, unbox

from ..common import navhistory
from ..common import utils as common_utils

from .app_state import app_state
from . import word_cloud

# --------------------------------------------------------------------------------
# Module-local state

# The states are `np.array`s of data indices, compared as *sets*: the same items in another order are the
# same selection, and committing one should not record a move nobody made.
#
# No validity predicate. `reset_undo_history` fires on dataset load, so the stack never outlives the
# dataset its indices point into, and there is no way for an entry to go stale while it sits here.
_history = navhistory.NavigationHistory(same=lambda a, b: set(a) == set(b))


# --------------------------------------------------------------------------------
# Undo history management

def reset_undo_history(_update_gui=True):
    """Reset the selection undo history. Used when loading a new dataset.

    `_update_gui`: internal, used during app initialization.
                   Everywhere else, should be the default `True`.
    """
    app_state.selection_data_idxs_box = box(common_utils.make_blank_index_array())
    _history.reset(unbox(app_state.selection_data_idxs_box))
    app_state.selection_changed = False  # ...after last completed info panel update (that was finalized); used for scroll anchoring
    app_state.selection_anchor_data_idxs_set = set()  # items common across previous and current selection; used for scroll anchoring  # indices to `sorted_xxx`
    if _update_gui:
        _update_undo_buttons()


def _update_undo_buttons():
    """Enable each of the undo/redo buttons exactly when there is somewhere to go in that direction."""
    (dpg.enable_item if _history.can_go_back else dpg.disable_item)("selection_undo_button")  # tag
    (dpg.enable_item if _history.can_go_forward else dpg.disable_item)("selection_redo_button")  # tag


def commit_change_to_undo_history():
    """Update the selection undo history, and update the state of the undo/redo GUI buttons.

    If the current selection is the same as that at the current position in the undo stack,
    then do nothing.

    Return whether a commit was actually needed.
    """
    if not _history.commit(unbox(app_state.selection_data_idxs_box)):
        return False
    _update_undo_buttons()
    return True


def _go(step):
    """Walk one step through the undo history and bring the selection with it. Returns whether it moved.

    The two directions differ only in which way they step, so they share this: the arrival is the same
    work either way, and it is the half that has to agree between them.
    """
    if not step():
        return False
    _update_undo_buttons()

    old_selection_data_idxs = unbox(app_state.selection_data_idxs_box)
    new_selection_data_idxs = _history.current

    app_state.selection_data_idxs_box << new_selection_data_idxs

    app_state.selection_anchor_data_idxs_set = set(new_selection_data_idxs).intersection(set(old_selection_data_idxs))
    app_state.selection_changed = True

    update_highlight()
    # Wait, because undo/redo may be clicked or hotkeyed several times in quick succession.
    app_state.update_info_panel(wait=True)
    app_state.update_mouse_hover(force=True, wait=True)
    word_cloud.update(new_selection_data_idxs, only_if_visible=True, wait=True)
    return True


def undo():
    """Walk one step back in the selection undo history, and update the state of the undo/redo GUI buttons.

    Do nothing if already at the beginning.

    No return value.
    """
    _go(_history.back)


def redo():
    """Walk one step forward in the selection undo history, and update the state of the undo/redo GUI buttons.

    Do nothing if already at the end.

    No return value.
    """
    _go(_history.forward)


# --------------------------------------------------------------------------------
# Selection update (with modes)

def update(new_selection_data_idxs, mode="replace", *, force=False, wait=False, update_selection_undo_history=True):
    """Update `app_state.selection_data_idxs_box`, updating also the selection undo stack (optionally) and the GUI.

    `new_selection_data_idxs`: `np.array` (or `list`) of indices to `sorted_xxx`
    `mode`: one of:
        "replace" (current selection with `new_selection_data_idxs`)
        "add" (`new_selection_data_idxs` to current selection)
        "subtract" (`new_selection_data_idxs` from current selection)
        "intersect" (`new_selection_data_idxs` with current selection)
    `force`: if `True`, don't care whether the selection set actually changes, but update the GUI regardless.
             Used when loading a new dataset.
    `wait`: bool, whether to wait for more keyboard/mouse input before starting long-running GUI updates.
            Used by mouse-draw select.
    `update_selection_undo_history`: as it says on the tin. Mouse-draw mode sets this to `False` so that
                                     every small movement of the mouse won't emit a separate undo entry.

                                     When this is `False`, you can later commit changes to undo history
                                     by calling `commit_change_to_undo_history`.

    Returns whether any changes were made.
      - "replace" mode will not make any changes, if the new selection is the same as the old one.
      - "add" mode will not make any changes if all datapoints in the new selection were already selected.
      - "subtract" mode will not make any changes if none of the datapoints in the new selection were selected.
      - "intersect" mode will not make any changes if the new selection covers the whole current selection.

    When no changes are made, this does nothing and exits early, without updating the GUI (unless `force=True`).
    """

    old_selection_data_idxs = unbox(app_state.selection_data_idxs_box)  # `np.array` of indices to `sorted_xxx`
    old_set = set(old_selection_data_idxs)
    new_set = set(new_selection_data_idxs)
    if mode == "add":
        if not force and not len(new_set.difference(old_set)):  # no new points?
            return False
        new_selection_data_idxs = np.array(list(old_set.union(new_set)))
    elif mode == "subtract":
        if not force and not len(new_set.intersection(old_set)):  # not removing any existing points?
            return False
        new_selection_data_idxs = np.array(list(old_set.difference(new_set)))
    elif mode == "intersect":
        common_selection_data_idxs_set = new_set.intersection(old_set)
        if not force and common_selection_data_idxs_set == old_set:
            return False
        new_selection_data_idxs = np.array(list(common_selection_data_idxs_set))
    else:  # mode == "replace":
        if not force and new_set == old_set:  # no changes?
            return False
        new_selection_data_idxs = np.array(new_selection_data_idxs)
    # The selection has changed (or `force=True`).
    #
    # Now `new_selection_data_idxs` contains the indices (to `sorted_xxx`) of datapoints
    # that comprise the final new selection, after accounting for `mode`.
    #
    # An emptied selection has to stay an *index* array: NumPy types `np.array([])` as float64, which
    # raises when used to slice with. Every non-empty path above already yields int64, so this is the
    # empty case only - and it can arrive through any mode except "add".
    if not len(new_selection_data_idxs):
        new_selection_data_idxs = common_utils.make_blank_index_array()
    final_new_set = set(new_selection_data_idxs)

    # Info panel scroll anchoring.
    # We must do this here (not in `commit_change_to_undo_history`) for two reasons:
    #   - Update order in mouse-draw select; it updates the selection continuously, but commits only when the mouse button is released.
    #   - This isn't really even related to the undo history, but to which items are currently shown, whether recorded in undo history or not.
    app_state.selection_anchor_data_idxs_set = final_new_set.intersection(old_set)  # Items common between the old and new selection are applicable as scroll anchors.
    app_state.selection_changed = True

    app_state.selection_data_idxs_box << new_selection_data_idxs  # Send the new data into the box.
    if update_selection_undo_history:
        commit_change_to_undo_history()

    # Update GUI elements.
    update_highlight()

    # Selection updates don't typically happen quickly in succession, so we can usually
    # tell the deferred updates to start immediately. The exception to the rule is the
    # mouse-draw select, which calls us with `wait=True`.
    app_state.update_info_panel(wait=wait)
    app_state.update_mouse_hover(force=True, wait=wait)
    word_cloud.update(new_selection_data_idxs, only_if_visible=True, wait=wait)

    return True  # the selection has changed (or `force=True`)


def update_highlight():
    """Update highlight for datapoints currently in selection.

    Low-level function. `update` calls this automatically.

    Generally, this only needs to be called if you manually send something
    into `app_state.selection_data_idxs_box` (like undo and redo do).
    """
    selection_data_idxs = unbox(app_state.selection_data_idxs_box)
    if len(selection_data_idxs):
        dpg.set_value("my_selection_scatter_series", [list(app_state.dataset.sorted_lowdim_data[selection_data_idxs, 0]),  # tag
                                                      list(app_state.dataset.sorted_lowdim_data[selection_data_idxs, 1])])
    else:
        dpg.set_value("my_selection_scatter_series", [[], []])  # tag


# --------------------------------------------------------------------------------
# Keyboard → mode mapping

def keyboard_state_to_mode():
    """Map current keyboard modifier state (Shift, Ctrl) to selection mode (replace, add, subtract, intersect).

    Helper for features that call `update`.
    """
    shift_pressed = dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)
    ctrl_pressed = dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)
    if shift_pressed and ctrl_pressed:
        return "intersect"
    elif shift_pressed:
        return "add"
    elif ctrl_pressed:
        return "subtract"
    return "replace"

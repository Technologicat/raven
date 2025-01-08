#!/usr/bin/env python
"""Visualize BibTeX data. This can put an entire field of science into one picture.

This GUI app performs visualization only. See `extract.py` to analyze your data and to generate the visualization dataset.
"""

# As any GUI app, this visualizer has lots of state. The clearest presentation here is as a script interleaving function definitions
# and GUI creation, with the state stored in module-level globals.
#
# Hence, we are extra careful: all module-level globals are actually needed somewhere. To avoid polluting the module-level namespace
# with temporaries, we use unpythonic's `@call` to limit the scope of any temporary variables into a temporary function (which is
# really just a code block that gets run immediately).
#
# Any line with at least one string-literal reference to any DPG GUI widget tag is commented with "tag" (no quotes), to facilitate searching.
# To find all, search for both "# tag" (the comment) and "tag=" (widget definitions).

__version__ = "0.1.0"

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Based on:
#    https://stackoverflow.com/questions/17275334/what-is-a-correct-way-to-filter-different-loggers-using-python-logging
#    https://docs.python.org/3/library/logging.html#logging.Filter
class UnionFilter(logging.Filter):  # Why isn't this thing in the stdlib?
    def __init__(self, *filters):
        """A `logging.Filter` that matches a record if at least one of the given `*filters` matches it."""
        self.filters = filters
    def filter(self, record):
        return any(f.filter(record) for f in self.filters)
for handler in logging.root.handlers:
    handler.addFilter(UnionFilter(logging.Filter(__name__),
                                  logging.Filter("file_dialog.fdialog")))

logger.info(f"Raven version {__version__} starting.")

logger.info("Loading libraries...")
from unpythonic import timer
with timer() as tim:
    import argparse
    import collections
    import concurrent.futures
    import functools
    import gc
    from io import StringIO
    import itertools
    import math
    import os
    import pathlib
    import pickle
    import re
    import threading
    import time
    import traceback
    import unicodedata
    import warnings

    import numpy as np

    import scipy.spatial.ckdtree

    from spacy.lang.en import English
    nlp_en = English()
    stopwords = nlp_en.Defaults.stop_words

    from unpythonic.env import env
    envcls = env  # for functions that need an `env` parameter due to `@dlet`, so that they can also instantiate env objects (oops)
    from unpythonic import window, dlet, call, box, unbox, sym, gensym, islice

    import dearpygui.dearpygui as dpg

    # Vendored libraries
    from .vendor.IconsFontAwesome6 import IconsFontAwesome6 as fa  # https://github.com/juliettef/IconFontCppHeaders
    from .vendor import DearPyGui_Markdown as dpg_markdown  # https://github.com/IvanNazaruk/DearPyGui-Markdown
    from .vendor.file_dialog.fdialog import FileDialog  # https://github.com/totallynotdrait/file_dialog, but with custom modifications
logger.info(f"    Done in {tim.dt:0.6g}s.")

from . import config

# TODO: Section this into subnamespaces?
gui_config = env(  # ----------------------------------------
                 # GUI element sizes, in pixels.
                 main_window_w=1920, main_window_h=1040,  # The default size just fits onto a 1080p screen in Linux Mint.
                 help_window_w=1700, help_window_h=1000,  # The help content is static, these values have been chosen to fit it.
                 info_panel_w=600,
                 info_panel_header_h=40,  # The title section and the navigation controls section both have this height.
                 title_wrap_w=500,  # Note there will be two columns of buttons to the left of each item title.
                 main_text_wrap_w=540,  # For wrapping the abstract.
                 info_panel_reserved_h=230,  # In the left part of the app window, how much vertical space to leave to GUI elements *other than* the item info content area.
                 toolbar_inner_w=36,  # Width of the content area of the "Tools" toolbar.
                 toolbar_separator_h=12,  # Height of a section separator spacer in the toolbar.
                 toolbutton_w=30,  # Width of a toolbutton in the "Tools" toolbar.
                 toolbutton_indent=None,  # The default `None` means "centered" (the value is then computed and stored while setting up the GUI).
                 info_panel_button_w=26,  # Width of the inline buttons in the info panel. Same width as a DPG arrow button so all buttons align properly.
                 annotation_tooltip_w=800,  # Just the width; height is automatic depending on content.
                 font_size=20,  # Also in pixels.
                 # ----------------------------------------
                 # Animations
                 n_many_searchresults=200,  # Number of data points to reach minimum per-datapoint glow highlight brightness for search results.
                 n_many_selection=30,  # Number of data points to reach minimum per-datapoint glow highlight brightness for selection.
                 glow_cycle_duration=2.0,  # seconds, for glow animations.
                 acknowledgment_duration=1.0,  # seconds, for button flashes upon clicking/hotkey.
                 scroll_ends_here_duration=0.5,  # seconds, for scrolling-past-end animation fadeout.
                 smooth_scrolling=True,  # whether to animate scrolling (all info panel scrolling, except scrollbar and mouse wheel, which are handled internally by DPG)
                 smooth_scrolling_step_parameter=0.8,  # Essentially, a nondimensional rate in the half-open interval (0, 1]; see math comment after `SmoothScrolling`.
                 # ----------------------------------------
                 # Mouse
                 selection_brush_radius_pixels=10,
                 datapoints_at_mouse_max_neighbors=100,  # affects performance
                 # ----------------------------------------
                 # Max numbers of dynamic stuff to put into GUI.
                 # Approximate; we always show at least one item per cluster.
                 max_titles_in_tooltip=10,
                 max_items_in_info_panel=100)

# --------------------------------------------------------------------------------
# General utilities

def clamp(x, ell=0.0, u=1.0):  # not the manga studio
    """Clamp value `x` between `ell` and `u`. Return clamped value."""
    return min(max(ell, x), u)

def nonanalytic_smooth_transition(x, m=1.0):  # extrafeathers.pdes.numutil
    """Non-analytic smooth transition from 0 to 1, on interval x ∈ [0, 1].

    The transition is reflection-symmetric through the point (1/2, 1/2).

    Outside the interval:
        s(x, m) = 0  for x < 0
        s(x, m) = 1  for x > 1

    The parameter `m` controls the steepness of the transition region.
    Larger `m` packs the transition closer to `x = 1/2`, making it
    more abrupt (although technically, still infinitely smooth).

    `m` is passed to `psi`, which see.
    """
    p = psi(x, m)
    return p / (p + psi(1.0 - x, m))

def psi(x, m=1.0):  # extrafeathers.pdes.numutil
    """Building block for non-analytic smooth functions.

        psi(x, m) := exp(-1 / x^m) χ(0, ∞)(x)

    where χ is the indicator function (1 if x is in the set, 0 otherwise).

    Suppresses divide by zero warnings and errors, so can be evaluated
    also at `x = 0`.

    This is the helper function used in the construction of the standard
    mollifier in PDE theory.
    """
    with warnings.catch_warnings():  # for NumPy arrays
        warnings.filterwarnings(action="ignore",
                                message="^divide by zero .*$",
                                category=RuntimeWarning,
                                module="__main__")
        try:
            return np.exp(-1.0 / x**m) * (x > 0.0)
        except ZeroDivisionError:  # for scalar x
            return 0.0

def make_blank_index_array():
    """Make a blank array of the same type as that used for slicing an array in NumPy."""
    return np.array([], dtype=np.int64)

def normalize_string(s):
    """Normalize a string for searching.

    This converts subscripts and superscripts into their regular equivalents.
    """
    s = " ".join(unicodedata.normalize("NFKC", s).strip().split())
    for k, v in config.subscript_to_regular.items():
        s = s.replace(k, v)
    for k, v in config.superscript_to_regular.items():
        s = s.replace(k, v)
    return s

def search_string_to_fragments(s):
    """Convert search string `s` into `(case_sensitive_fragments, case_insensitive_fragments)`.

    Incremental fragment search, like in Emacs HELM (or in Firefox address bar):
      - "cat photo" matches "photocatalytic".
      - Lowercase search term means case-insensitive for that term (handled in functions
        that perform search, such as `update_search` and `update_info_panel`).
    """
    search_terms = [normalize_string(x.strip()) for x in s.split()]
    is_case_sensitive = [x.lower() != x for x in search_terms]
    case_sensitive_fragments = [x for x, sens in zip(search_terms, is_case_sensitive) if sens]
    case_insensitive_fragments = [x for x, sens in zip(search_terms, is_case_sensitive) if not sens]
    return case_sensitive_fragments, case_insensitive_fragments

def get_main_window_size():
    """Return `(width, height), the size of the main window (and thus also the viewport), in pixels.`"""
    config = dpg.get_item_configuration(main_window)
    w = config["width"]
    h = config["height"]
    return w, h

def has_child_items(item):
    """Return whether DPG item `item` has child items in any of its slots."""
    for slot in range(4):
        if len(dpg.get_item_children(item, slot=slot)):
            return True
    return False

def get_visible_datapoints():  # for `sorted_xxx`, see further below
    """Return indices (in `sorted_xxx`) of all datapoints currently visible in the plotter."""
    xmin, xmax = dpg.get_axis_limits("axis0")  # in data space
    ymin, ymax = dpg.get_axis_limits("axis1")  # in data space
    filtxmin = dataset.sorted_lowdim_data[:, 0] >= xmin
    filtxmax = dataset.sorted_lowdim_data[:, 0] <= xmax
    filtx = filtxmin * filtxmax
    filtymin = dataset.sorted_lowdim_data[:, 1] >= ymin
    filtymax = dataset.sorted_lowdim_data[:, 1] <= ymax
    filty = filtymin * filtymax
    filt = filtx * filty
    return np.where(filt)[0]

# --------------------------------------------------------------------------------
# Selection management (related to datapoints in the plotter)

def reset_undo_history(_update_gui=True):
    """Reset the selection undo history. Used when loading a new dataset.

    `_update_gui`: internal, used during app initialization.
                   Everywhere else, should be the default `True`.
    """
    global selection_data_idxs_box  # This is boxed so we can easily replace immutable contents (`np.array`).
    global selection_undo_stack
    global selection_undo_pos
    global selection_changed
    global selection_anchor_data_idxs_set
    selection_data_idxs_box = box(make_blank_index_array())
    selection_undo_stack = [unbox(selection_data_idxs_box)]
    selection_undo_pos = 0
    selection_changed = False  # ...after last completed info panel update (that was finalized); used for scroll anchoring
    selection_anchor_data_idxs_set = set()  # items common across previous and current selection; used for scroll anchoring  # indices to `sorted_xxx`
    if _update_gui:
        dpg.disable_item("selection_undo_button")  # tag
        dpg.disable_item("selection_redo_button")  # tag
reset_undo_history(_update_gui=False)  # GUI not initialized yet. This is the only time the flag should be set to `False`!

def commit_selection_change_to_undo_history():
    """Update the selection undo history, and update the state of the undo/redo GUI buttons.

    If the current selection is the same as that at the current position in the undo stack,
    then do nothing.

    Return whether a commit was actually needed.
    """
    global selection_undo_stack
    global selection_undo_pos

    # Only proceed with the commit if the selection is actually different from what we have at the current undo position.
    old_selection_data_idxs = selection_undo_stack[selection_undo_pos]
    new_selection_data_idxs = unbox(selection_data_idxs_box)
    old_selection_data_idxs_set = set(old_selection_data_idxs)
    new_selection_data_idxs_set = set(new_selection_data_idxs)
    if new_selection_data_idxs_set == old_selection_data_idxs_set:
        return False

    selection_undo_stack = selection_undo_stack[:selection_undo_pos + 1]
    selection_undo_stack.append(new_selection_data_idxs)
    selection_undo_pos = len(selection_undo_stack) - 1

    dpg.enable_item("selection_undo_button")  # tag
    dpg.disable_item("selection_redo_button")  # tag

    return True

def selection_undo():
    """Walk one step back in the selection undo history, and update the state of the undo/redo GUI buttons.

    Do nothing if already at the beginning.

    No return value.
    """
    global selection_undo_pos
    global selection_changed
    global selection_anchor_data_idxs_set
    if selection_undo_pos == 0:
        return
    selection_undo_pos -= 1
    if selection_undo_pos == 0:
        dpg.disable_item("selection_undo_button")  # tag
    dpg.enable_item("selection_redo_button")  # tag

    # See also `commit_selection_change_to_undo_history` and `update_selection`; we must do some of the same things here.
    old_selection_data_idxs = unbox(selection_data_idxs_box)
    new_selection_data_idxs = selection_undo_stack[selection_undo_pos]

    selection_data_idxs_box << new_selection_data_idxs

    selection_anchor_data_idxs_set = set(new_selection_data_idxs).intersection(set(old_selection_data_idxs))
    selection_changed = True

    update_selection_highlight()
    update_info_panel(wait=True)  # wait, because undo may be clicked/hotkeyed several times quickly in succession
    update_mouse_hover(force=True, wait=True)

def selection_redo():
    """Walk one step forward in the selection undo history, and update the state of the undo/redo GUI buttons.

    Do nothing if already at the end.

    No return value.
    """
    global selection_undo_pos
    global selection_changed
    global selection_anchor_data_idxs_set
    if selection_undo_pos == len(selection_undo_stack) - 1:
        return
    selection_undo_pos += 1
    if selection_undo_pos == len(selection_undo_stack) - 1:
        dpg.disable_item("selection_redo_button")  # tag
    dpg.enable_item("selection_undo_button")  # tag

    # See also `commit_selection_change_to_undo_history` and `update_selection`; we must do some of the same things here.
    old_selection_data_idxs = unbox(selection_data_idxs_box)
    new_selection_data_idxs = selection_undo_stack[selection_undo_pos]

    selection_data_idxs_box << new_selection_data_idxs

    selection_anchor_data_idxs_set = set(new_selection_data_idxs).intersection(set(old_selection_data_idxs))
    selection_changed = True

    update_selection_highlight()
    update_info_panel(wait=True)  # wait, because redo may be clicked/hotkeyed several times quickly in succession
    update_mouse_hover(force=True, wait=True)

def update_selection(new_selection_data_idxs, mode="replace", *, force=False, wait=False, update_selection_undo_history=True):
    """Update `selection_data_idxs_box`, updating also the selection undo stack (optionally) and the GUI.

    `new_selection_data_idxs`: `np.array` (or `list`) of indices to `sorted_xxx`
    `mode`: one of:
        "replace" (current selection with `new_selection_data_idxs`)
        "add" (`new_selection_data_idxs` to current selection)
        "subtract" (`new_selection_data_idxs` from current selection)
        "intersect" (`new_selection_data_idxs` with current selection)
    `force`: if `True`, don't care whether the selection set actually changes, but update the GUI regardless.
             Used when loading a new dataset.
    `wait`: whether to wait for more keyboard/mouse input before starting the info panel update.
            Used by mouse-draw select.
    `update_selection_undo_history`: as it says on the tin. Mouse-draw mode sets this to `False` so that
                                     every small movement of the mouse won't emit a separate undo entry.

                                     When this is `False`, you can later commit changes to undo history
                                     by calling `commit_selection_change_to_undo_history`.

    Returns whether any changes were made.
      - "replace" mode will not make any changes, if the new selection is the same as the old one.
      - "add" mode will not make any changes if all datapoints in the new selection were already selected.
      - "subtract" mode will not make any changes if none of the datapoints in the new selection were selected.
      - "intersect" mode will not make any changes if the new selection covers the whole current selection.

    When no changes are made, this does nothing and exits early, without updating the GUI (unless `force=True`).
    """
    global selection_changed
    global selection_anchor_data_idxs_set

    old_selection_data_idxs = unbox(selection_data_idxs_box)  # `np.array` of indices to `sorted_xxx`
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
    final_new_set = set(new_selection_data_idxs)

    # Info panel scroll anchoring.
    # We must do this here (not in `commit_selection_change_to_undo_history`) for two reasons:
    #   - Update order in mouse-draw select; it updates the selection continuously, but commits only when the mouse button is released.
    #   - This isn't really even related to the undo history, but to which items are currently shown, whether recorded in undo history or not.
    selection_anchor_data_idxs_set = final_new_set.intersection(old_set)  # Items common between the old and new selection are applicable as scroll anchors.
    selection_changed = True
    # logger.debug(f"update_selection: new selection anchor set is {selection_anchor_data_idxs_set}")

    selection_data_idxs_box << new_selection_data_idxs  # Send the new data into the box.
    if update_selection_undo_history:
        commit_selection_change_to_undo_history()

    # Update GUI elements.
    update_selection_highlight()

    # Selection updates don't typically happen quickly in succession, so we can usually
    # tell the deferred updates to start immediately. The exception to the rule is the
    # mouse-draw select, which calls us with `wait=True`.
    update_info_panel(wait=wait)
    update_mouse_hover(force=True, wait=wait)

    return True  # the selection has changed (or `force=True`)

def update_selection_highlight():
    """Update highlight for datapoints currently in selection.

    Low-level function. `update_selection` calls this automatically.

    Generally, this only needs to be called if you manually send something
    into `selection_data_idxs_box` (like undo and redo do).
    """
    selection_data_idxs = unbox(selection_data_idxs_box)
    if len(selection_data_idxs):
        dpg.set_value("my_selection_scatter_series", [list(dataset.sorted_lowdim_data[selection_data_idxs, 0]),  # tag
                                                      list(dataset.sorted_lowdim_data[selection_data_idxs, 1])])
    else:
        dpg.set_value("my_selection_scatter_series", [[], []])  # tag

def keyboard_state_to_selection_mode():
    """Map current keyboard modifier state (Shift, Ctrl) to selection mode (replace, add, subtract).

    Helper for features that call `update_selection`.
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

# --------------------------------------------------------------------------------
# Modal window related utilities

def enter_modal_mode():
    """Prepare the GUI for showing a modal window: hide annotation, disable current item button glow, ...

    Call this AFTER showing your modal so that the window detects as being shown in any functionality that checks that.
    This automatically waits for one frame for the window to actually render.
    """
    logger.debug("enter_modal_mode: App entering modal mode.")
    dpg.split_frame()
    update_mouse_hover(force=True, wait=False)  # hide annotation (just in case it's there)
    _info_panel_scroll_position_changed(reset=True)  # force update of current item in `update_current_search_result_status`, so `CurrentItemControlsGlow` disables its highlight

def exit_modal_mode():
    """Restore the GUI to main window mode (when a modal is closed): show annotation if relevant, enable current item button glow, ...

    Call this AFTER hiding your modal so that the window detects as being hidden in any functionality that checks that.
    This automatically waits for one frame for the window to actually render.
    """
    logger.debug("exit_modal_mode: App returning to main window mode.")
    dpg.split_frame()
    _info_panel_scroll_position_changed(reset=True)  # force update of current item in `update_current_search_result_status`, so `CurrentItemControlsGlow` re-enables its highlight
    update_mouse_hover(force=True, wait=False)  # show annotation if relevant

def is_any_modal_window_visible():
    """Return whether *some* modal window is open.

    Currently these are the help card and the "open file" dialog.
    """
    return is_file_dialog_visible() or is_help_window_visible()

# --------------------------------------------------------------------------------
# Set up DPG - basic startup, load fonts, set up global theme

# We do this as early as possible, because before the startup is complete,
# trying to `dpg.add_xxx` or `with dpg.xxx:` anything will segfault the app.

def setup_font_ranges():
    """Set up special characters for a font.

    The price of GPU-accelerated rendering - font textures. In DPG, only Latin is enabled by default.
    We add anything that `extract.py` may introduce from its LaTeX and HTML conversions.
    """
    # # Maybe just this?
    # dpg.add_font_range(0x300, 0x2fff)
    # return

    dpg.add_font_range_hint(dpg.mvFontRangeHint_Default)
    # Greek for math
    dpg.add_font_range(0x370, 0x3ff)
    # subscripts
    dpg.add_font_range(0x2080, 0x2089)  # zero through nine
    dpg.add_font_range(0x1d62, 0x1d65)  # i, r, u, v
    dpg.add_font_range(0x2090, 0x209c)  # a, e, o, x, schwa, h, k, l, m, n, p, s, t
    dpg.add_font_range(0x1d66, 0x1d6a)  # β, γ, ρ, φ, χ
    dpg.add_font_range(0x208a, 0x208e)  # +, -, =, (, )
    dpg.add_font_chars([0x2c7c])  # j
    # superscripts
    dpg.add_font_chars([0x2070, 0x00b9, 0x00b2, 0x00b3, 0x2074, 0x2075, 0x2076, 0x2077, 0x2078, 0x2079])  # zero through nine
    dpg.add_font_chars([0x2071, 0x207f])  # i, n
    dpg.add_font_range(0x207a, 0x207e)  # +, -, =, (, )

def markdown_add_font_callback(file, size: int | float, parent=0, **kwargs) -> int:  # IMPORTANT: parameter names as in `dpg_markdown`, arguments are sent in by name.
    """dpg_markdown callback to load a font. Called whenever a new font size or family is needed."""
    if not isinstance(size, (int, float)):
        raise ValueError(f"markdown_add_font_callback: `size`: expected `int` or `float`, got `{type(size)}`")
    with dpg.font(file, size, parent=parent, **kwargs) as font:
        setup_font_ranges()
    return font

logger.info("DPG bootup...")
with timer() as tim:
    dpg.create_context()

    # Initialize fonts. Must be done after `dpg.create_context`.
    # https://dearpygui.readthedocs.io/en/latest/documentation/fonts.html
    with dpg.font_registry() as the_font_registry:
        # Change the default font to something that looks clean and has good on-screen readability.
        # https://fonts.google.com/specimen/Open+Sans
        with dpg.font(os.path.join(os.path.dirname(__file__), "fonts", "OpenSans-Regular.ttf"),
                      gui_config.font_size) as default_font:
            setup_font_ranges()
        dpg.bind_font(default_font)

        # FontAwesome 6 for symbols (toolbar button icons etc.).
        with dpg.font(os.path.join(os.path.dirname(__file__), "fonts", fa.FONT_ICON_FILE_NAME_FAR),
                      gui_config.font_size) as icon_font_regular:
            dpg.add_font_range(fa.ICON_MIN, fa.ICON_MAX_16)
        with dpg.font(os.path.join(os.path.dirname(__file__), "fonts", fa.FONT_ICON_FILE_NAME_FAS),
                      gui_config.font_size) as icon_font_solid:
            dpg.add_font_range(fa.ICON_MIN, fa.ICON_MAX_16)

    # Configure fonts for the Markdown renderer.
    # https://github.com/IvanNazaruk/DearPyGui-Markdown
    # https://fonts.google.com/specimen/Inter+Tight
    dpg_markdown.set_font_registry(the_font_registry)
    dpg_markdown.set_add_font_function(markdown_add_font_callback)
    # TODO: Find the best font for rendering scientific Unicode text.
    #   - InterTight renders subscript numbers as superscript numbers, which breaks the rendering of chemistry formulas.
    #   - OpenSans is missing the subscript-x glyph, which also breaks the rendering of chemistry formulas.
    # dpg_markdown.set_font(font_size=gui_config.font_size,
    #                       default=os.path.join(os.path.dirname(__file__), "fonts", "InterTight-Regular.ttf"),
    #                       bold=os.path.join(os.path.dirname(__file__), "fonts", "InterTight-Bold.ttf"),
    #                       italic=os.path.join(os.path.dirname(__file__), "fonts", "InterTight-Italic.ttf"),
    #                       italic_bold=os.path.join(os.path.dirname(__file__), "fonts", "InterTight-BoldItalic.ttf"))
    dpg_markdown.set_font(font_size=gui_config.font_size,
                          default=os.path.join(os.path.dirname(__file__), "fonts", "OpenSans-Regular.ttf"),
                          bold=os.path.join(os.path.dirname(__file__), "fonts", "OpenSans-Bold.ttf"),
                          italic=os.path.join(os.path.dirname(__file__), "fonts", "OpenSans-Italic.ttf"),
                          italic_bold=os.path.join(os.path.dirname(__file__), "fonts", "OpenSans-BoldItalic.ttf"))
    # USAGE: dpg_markdown.add_text(some_markdown_string)
    #
    # For font color/size, use these syntaxes:
    #     <font color="(255, 0, 0)">Test</font>
    #     <font color="#ff0000">Test</font>
    #     <font size="50">Test</font>
    #     <font size=50>Test</font>
    # color/size can be used in the same font tag.

    # Modify global theme
    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvAll):
            # dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (53, 168, 84))  # same color as Linux Mint default selection color in the green theme
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 8, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 8, category=dpg.mvThemeCat_Core)
    dpg.bind_theme(global_theme)  # set this theme as the default

    # Add a theme for tight text layout
    with dpg.theme(tag="my_no_spacing_theme"):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 0, category=dpg.mvThemeCat_Core)

    # FIX disabled controls not showing as disabled.
    # DPG does not provide a default disabled-item theme, so we provide our own.
    # Everything else is automatically inherited from DPG's global theme.
    #     https://github.com/hoffstadt/DearPyGui/issues/2068
    # TODO: Figure out how to get colors from a theme. Might not always be `(45, 45, 48)`.
    #   - Maybe see how DPG's built-in theme editor does it - unless it's implemented at the C++ level.
    #   - See also the theme color editor in https://github.com/hoffstadt/DearPyGui/wiki/Tools-and-Widgets
    disabled_color = (0.50 * 255, 0.50 * 255, 0.50 * 255, 1.00 * 255)
    disabled_button_color = (45, 45, 48)
    disabled_button_hover_color = (45, 45, 48)
    disabled_button_active_color = (45, 45, 48)
    with dpg.theme(tag="disablable_button_theme"):
        # We customize just this. Everything else is inherited from the global theme.
        with dpg.theme_component(dpg.mvButton, enabled_state=False):
            dpg.add_theme_color(dpg.mvThemeCol_Text, disabled_color, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Button, disabled_button_color, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, disabled_button_hover_color, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, disabled_button_active_color, category=dpg.mvThemeCat_Core)

    dpg.create_viewport(title=f"Raven {__version__}",
                        width=gui_config.main_window_w,
                        height=gui_config.main_window_h)  # OS window (DPG "viewport")
    dpg.setup_dearpygui()
logger.info(f"    Done in {tim.dt:0.6g}s.")

# --------------------------------------------------------------------------------
# Dataset loading

dataset = None  # currently loaded dataset (as an `unpythonic.env`)
dynamically_created_cluster_color_coding_themes = []  # for cleaning them up when another dataset is loaded

def _to_absolute(filename):
    """Convert `filename` to an absolute filename."""
    return str(pathlib.Path(filename).expanduser().resolve())

def _read_pickle(filename):
    """Load a visualization dataset. Low-level helper."""
    with open(filename, "rb") as visdata_file:
        data = pickle.load(visdata_file)
    if data["version"] != 1:
        raise NotImplementedError(f"Dataset {filename} has version '{data['version']}', expected '1'. Don't know how to visualize this dataset.")
    return env(**data)

def parse_dataset_file(filename):
    """Parse a dataset file and process it for visualization. Public API.

    Returns a dataset: `unpythonic.env` with the datafile contents, and some preprocessed fields to facilitate visualization.
    """
    dataset = env()
    absolute_filename = _to_absolute(filename)
    dataset.filename = filename
    dataset.absolute_filename = absolute_filename

    logger.info(f"Reading visualization dataset '{filename}' (resolved to '{absolute_filename}')...")
    with timer() as tim:
        dataset.file_content = _read_pickle(absolute_filename)
    logger.info(f"    Done in {tim.dt:0.6g}s.")

    # In DPG, one scatter series has only a single global color, which for our application is unfortunate.
    #
    # To color the data by cluster ID, we create a separate scatter plot for each cluster.
    # Fortunately, DPG is fast enough that it can render hundreds of scatter plots in realtime.
    #
    # For this we need to sort the data by label (cluster ID).
    #
    logger.info("Sorting data by cluster...")
    with timer() as tim:  # set up `sorted_xxx`
        dataset.sorted_orig_data_idxs = np.argsort(dataset.file_content.labels)  # sort by label (cluster ID)
        dataset.sorted_labels = dataset.file_content.labels[dataset.sorted_orig_data_idxs]
        dataset.sorted_lowdim_data = dataset.file_content.lowdim_data[dataset.sorted_orig_data_idxs, :]  # [data_idx, axis], where axis is 0 (x) or 1 (y).
        dataset.sorted_entries = [dataset.file_content.vis_data[orig_data_idx] for orig_data_idx in dataset.sorted_orig_data_idxs]  # the actual data records, extracted from BibTeX (Python list)
        @call
        def _():
            # Compute normalized titles for searching, and insert a reverse lookup for the item's index in `sorted_xxx`.
            for data_idx, entry in enumerate(dataset.sorted_entries):
                entry.data_idx = data_idx  # index to `sorted_xxx`
                entry.normalized_title = normalize_string(entry.title.strip())  # for searching

        # for k, v in dataset.sorted_entries[0].items():  # DEBUG: print one input data record (it's a dict)
        #     print(f"{k}: {v}")

        # Find contiguous blocks with the same label (cluster ID).
        dataset.cluster_id_jump_data_idxs = np.where(np.diff(dataset.sorted_labels, prepend=np.nan))[0]  # https://stackoverflow.com/a/65657397
        dataset.cluster_id_jump_data_idxs = list(itertools.chain(list(dataset.cluster_id_jump_data_idxs), (None,)))  # -> [i0, i1, ..., iN, None] -> used for slices, `None` = end
    logger.info(f"    Done in {tim.dt:0.6g}s.")

    # For mouseover support. We need to manually detect the data points closest to the mouse cursor.
    logger.info("Indexing dataset for nearest-neighbors search...")
    with timer() as tim:
        dataset.kdtree = scipy.spatial.cKDTree(data=dataset.sorted_lowdim_data)
    logger.info(f"    Done in {tim.dt:0.6g}s.")
    return dataset

def _create_highlight_scatter_series():
    """Create some special scatterplot data series, used for highlighting datapoints in the plotter."""
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

def clear_background_tasks():
    """Stop (cancel) and delete all background tasks."""
    info_panel_task_manager.clear(wait=True)
    annotation_task_manager.clear(wait=True)

def reset_app_state(_update_gui=True):
    """Reset everything, to prepare for loading new data to the GUI.

    `_update_gui`: internal, used during app exit.
                   Everywhere else, should be the default `True`.
    """
    reason = "for loading new data to the GUI" if _update_gui else "(app exiting)"
    logger.info(f"Resetting app state {reason}.")

    # Stop old background tasks
    clear_background_tasks()

    # Stop animations
    animator.clear()

    # These should always be present in the animator.
    # These monitor the app state and live-update at every frame.
    animator.add(PlotterPulsatingGlow(cycle_duration=gui_config.glow_cycle_duration))
    animator.add(CurrentItemControlsGlow(cycle_duration=gui_config.glow_cycle_duration))

    # Only update the GUI elements if not exiting, because when exiting, the GUI is already being deleted.
    if _update_gui:
        # Clear undo history and selection
        reset_undo_history()
        update_selection(make_blank_index_array(), mode="replace", force=True, wait=False, update_selection_undo_history=False)

        # Clear the search
        dpg.set_value("search_field", "")  # tag
        update_search(wait=False)

        # Remove old data series, if any
        dpg.delete_item("axis1", children_only=True)  # tag

        # But restore the highlights for the next dataset
        _create_highlight_scatter_series()

        # Delete old cluster-color-coding scatterplot themes
        for theme in dynamically_created_cluster_color_coding_themes:
            dpg.delete_item(theme)

        dpg.set_item_label("plot", "Semantic map [no dataset loaded]")  # tag  # TODO: DRY duplicate definitions for labels

def reset_plotter_zoom():
    """Reset the plotter's zoom level to show all data."""
    dpg.fit_axis_data("axis0")  # tag
    dpg.fit_axis_data("axis1")  # tag

def load_data_into_plotter(dataset):
    """Load `dataset` (see `parse_dataset_file`) to the plotter.

    IMPORTANT: call `reset_app_state()` just before calling this.
    """
    logger.info(f"Plotting visualization dataset '{dataset.absolute_filename}'...")
    with timer() as tim:
        # Group data points by label
        datas = []
        for start, end in window(2, dataset.cluster_id_jump_data_idxs):  # indices to `sorted_xxx`
            xs = list(dataset.sorted_lowdim_data[start:end, 0])
            ys = list(dataset.sorted_lowdim_data[start:end, 1])
            label = dataset.sorted_labels[start]  # all `dataset.sorted_labels[start:end]` are the same
            datas.append((label, xs, ys))

        max_label = dataset.sorted_labels[-1]  # The labels have been sorted in ascending order so the largest one is last.
        for label, xs, ys in datas:
            # Render the series, placing it before the first highlight series so that all highlights render on top.
            series_tag = f"my_scatter_series_{label}"
            dpg.add_scatter_series(xs, ys, tag=series_tag, parent="axis1", before="my_mouse_hover_scatter_series")  # tag

            series_theme = f"my_plot_theme_{label}"  # tag
            color = dpg.sample_colormap(dpg.mvPlotColormap_Viridis, t=(label + 1) / (max_label + 1))
            color = [int(255 * component) for component in color]  # RGBA
            color[-1] = int(0.5 * color[-1])  # A; make translucent

            # Create a separate theme for each scatter series to set its color.
            # https://dearpygui.readthedocs.io/en/1.x/documentation/themes.html
            with dpg.theme(tag=series_theme) as this_scatterplot_theme:
                with dpg.theme_component(dpg.mvScatterSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, color, category=dpg.mvThemeCat_Plots)
                    # # We could customize other stuff, too.
                    # dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle, category=dpg.mvThemeCat_Plots)
                    # dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 4, category=dpg.mvThemeCat_Plots)
            dpg.bind_item_theme(series_tag, series_theme)
            dynamically_created_cluster_color_coding_themes.append(this_scatterplot_theme)

        dpg.set_item_label("plot", f"Semantic map [{os.path.basename(dataset.absolute_filename)}]")  # tag
        reset_plotter_zoom()
    logger.info(f"    Done in {tim.dt:0.6g}s.")

    # Trigger an info panel update
    update_selection(make_blank_index_array(), mode="replace", force=True, wait=False, update_selection_undo_history=False)

def open_file(filename):
    """Load new data into the GUI. Public API."""
    logger.info(f"open_file: Opening file '{filename}'.")
    global dataset
    reset_app_state()
    dataset = parse_dataset_file(filename)
    load_data_into_plotter(dataset)

# --------------------------------------------------------------------------------
# "Open file" dialog

filedialog = None

def initialize_filedialog(default_path):
    """Create the "open file" dialog."""
    global filedialog
    filedialog = FileDialog(title="Open dataset",
                            tag="open_file_dialog",
                            callback=_open_file_callback,
                            modal=True,
                            filter_list=[".pickle"],
                            file_filter=".pickle",
                            multi_selection=False,
                            allow_drag=False,
                            default_path=default_path)

def show_open_file_dialog():
    """Show the "open file" dialog.

    (And prepare the GUI for it: hide annotation, disable current item button glow, ...)
    If you need to close it programmatically, call `filedialog.cancel()` so it'll trigger the callback (necessary to restore the GUI back into main window mode).
    """
    logger.debug("show_open_file_dialog: Showing open file dialog.")
    filedialog.show_file_dialog()
    enter_modal_mode()
    logger.debug("show_open_file_dialog: Done.")

def _open_file_callback(selected_files):
    """Callback that fires when the "open file" dialog closes."""
    logger.debug("_open_file_callback: Open file dialog callback triggered.")
    if len(selected_files) > 1:
        raise ValueError(f"Expected at most one selected file, got {len(selected_files)}.")
    exit_modal_mode()
    if selected_files:
        selected_file = selected_files[0]
        logger.debug(f"_open_file_callback: User selected the file '{selected_file}'.")
        open_file(selected_file)
    else:  # empty selection -> cancelled
        logger.debug("_open_file_callback: Cancelled.")

def is_file_dialog_visible():
    """Return whether the "open file" dialog is open.

    We have this abstraction (not just `dpg.is_item_visible`) because the window might not exist yet.
    """
    if filedialog is None:
        return False
    return dpg.is_item_visible("open_file_dialog")  # tag

# --------------------------------------------------------------------------------
# Animation mechanism

animation_action_continue = sym("continue")  # keep rendering
animation_action_finish = sym("finish")  # end animation, call the `finish` method
animation_action_cancel = sym("cancel")  # end animation without calling the `finish` method
class Animator:
    def __init__(self):
        """A simple animation manager.

        Raven's customized render loop calls our `render_frame` once per frame,
        to update any animations that are running.
        """
        self.animations = []
        self.animation_list_lock = threading.RLock()

    def add(self, animation):
        """Register a new `Animation` instance, so that our `render_frame` will call its `render_frame` method.

        Its start time `animation.t0` is set automatically to the current time as returned by `time.time_ns()`.
        """
        with self.animation_list_lock:
            animation.reset()  # set the animation start time
            self.animations.append(animation)
        return animation

    def cancel(self, animation, finalize=True):
        """Terminate a running `Animation` instance.

        `animation`: One of the animations registered using `add`.

        `finalize`: If `True` (default), call the `finish` method of the animation before removing it.

                    In some special cases, it can be useful to set this to `False` to reduce flicker,
                    if the old animation is immediately replaced by a new one of the same type,
                    targeting the same GUI element (so no need to hide/re-show).

        Note that when an animation finishes normally, it is automatically removed. This is meant for
        immediately stopping and removing an animation that has not finished yet.
        """
        with self.animation_list_lock:
            if finalize:
                animation.finish()
            try:
                self.animations.remove(animation)  # uses default comparison by `id()` since `Animation` has no `__eq__` operator
            except ValueError:  # not in list
                logger.debug(f"Animator.cancel: specified {animation} is not in the animation registry (maybe already finished?), skipping removal.")
        return animation

    def render_frame(self):
        """Render one frame of each registered animation, in the order they were registered.

        Each animation whose `render_frame` returns `animation_action_finish` is considered finished.
        Each finished animation gets its `finish` method called automatically.

        After all registered animations have had a frame rendered, the animation registry is updated
        to remove any animations that are no longer running.
        """
        with self.animation_list_lock:
            time_now = time.time_ns()
            running_animations = []
            for animation in self.animations:
                action = animation.render_frame(t=time_now)
                if action is animation_action_continue:
                    running_animations.append(animation)
                elif action is animation_action_finish:
                    animation.finish()
                elif action is animation_action_cancel:
                    pass  # when cancelled, do nothing, just remove the animation
                else:
                    raise ValueError(f"Animator.render_frame: unknown action {action}, expected one of the `animation_action_X` constants (where X is 'continue', 'finish', or 'cancel').")
            self.animations.clear()
            self.animations.extend(running_animations)

    def clear(self):
        """Terminate all registered animations and clear the list of registered animations.

        To terminate a specific animation (by object instance), see `cancel`.
        """
        with self.animation_list_lock:
            for animation in self.animations:
                animation.finish()
            self.animations.clear()
animator = Animator()


class Animation:
    def __init__(self):
        """Base class for Raven's animations.

        An `Animation` can be added to an `Animator`.
        """
        # Keep this simple to avoid ravioli code.
        # `t0` should be pretty much the only attribute defined in the base class.
        self.reset()

    def reset(self):
        """Semantically: (re-)start the animation from the beginning.

        Technically, in this base class: Set the animation start time `self.t0` to the current time,
        as given by `time.time_ns()`.
        """
        self.t0 = time.time_ns()

    def render_frame(self, t):
        """Override this in a derived class to render one frame of your animation.

        `t`: int; time at start of current frame as returned by `time.time_ns()`.

        The animation start time is available in `self.t0`.

        It is also allowed to write to `self.t0`, e.g. for a cyclic animation
        so as not to lose float accuracy in long sessions.

        Return value must be one of:
            `animation_action_continue` if the animation should continue,
            `animation_action_finish` if the animation should end, automatically calling its `finish` method.
            `animation_action_cancel` if the animation should end, *without* calling its `finish` method
                                      (useful if the animation determined it didn't need to start,
                                       e.g. if another copy was already running on the same GUI element).

        The animator automatically removes (from its animation registry) any animations that return
        anything other than `animation_action_continue`.
        """
        return animation_action_finish

    def finish(self):
        """Override this in a derived class to clean up any state for your animation when it finishes normally."""

# --------------------------------------------------------------------------------
# Overlay window support

# Currently used by the info panel dimmer and the "scroll ends here" animation.

class Overlay:
    def __init__(self, target, tag):
        """Base class for Raven's overlay windows (currently the dimmer, and the scroll end animation).

        `target`: DPG ID or tag. The child window for which to build the overlay.
        `tag`: DPG tag, for naming the overlay.
        """
        # Keep this simple to avoid ravioli code.
        # `target`, `tag` and `overlay_update_lock` should be pretty much the only attributes defined in the base class.
        self.target = target
        self.tag = tag
        self.overlay_update_lock = threading.Lock()

# --------------------------------------------------------------------------------
# Animations, live updates

search_string_box = box("")
search_result_data_idxs_box = box(make_blank_index_array())

def update_search(wait=True):
    """Perform search and update the search results.

    Note this search callback only gets called when the content of the search field actually changes.

    `wait`: Whether to wait for more keyboard input before starting to render the info panel and
            tooltip annotation updates.

            Passed to `update_info_panel` and to `update_mouse_hover`, which see.
    """
    search_string = dpg.get_value("search_field")  # tag
    if not search_string:
        search_result_data_idxs = make_blank_index_array()
    else:
        case_sensitive_fragments, case_insensitive_fragments = search_string_to_fragments(search_string)
        search_result_data_idxs = []
        for data_idx, entry in enumerate(dataset.sorted_entries):  # `data_idx`: index to `sorted_xxx`
            text = entry.normalized_title
            text_lowercase = text.lower()
            if (all(term in text_lowercase for term in case_insensitive_fragments) and
               all(term in text for term in case_sensitive_fragments)):
                search_result_data_idxs.append(data_idx)
        search_result_data_idxs = np.array(search_result_data_idxs)

    # Send the new data into the boxes
    search_string_box << search_string
    search_result_data_idxs_box << search_result_data_idxs

    if len(search_result_data_idxs):
        # Highlight the search result data points (by plotting them as another series on top).
        dpg.set_value("my_search_results_scatter_series", [list(dataset.sorted_lowdim_data[search_result_data_idxs, 0]),  # tag
                                                           list(dataset.sorted_lowdim_data[search_result_data_idxs, 1])])
        # Re-use the "Search" header to show the number of matches.
        plural_s = "es" if len(search_result_data_idxs) != 1 else ""
        dpg.set_value(search_header_text, f"[{len(search_result_data_idxs)} match{plural_s}]")
    else:
        dpg.set_value("my_search_results_scatter_series", [[], []])  # tag
        if not search_string:  # Search not active, restore the "Search" header
            dpg.set_value(search_header_text, "Search")  # TODO: DRY duplicate definitions for labels
        else:  # Search active, but no matches
            dpg.set_value(search_header_text, "[no matches]")

    # Update tooltip and info panel to update the highlight status.
    # TODO: Currently, this may cause the set of data points considered to be under the mouse cursor to change
    #       the first time this happens at a given mouse position (upon a click in the plot area). Debug this.
    #       If the plot mouse position is one frame out of date (update order?), that would explain it.
    update_info_panel(wait=wait)
    update_mouse_hover(force=True, wait=wait)


class ButtonFlash(Animation):
    # For some animation types, such as this one, for any given GUI element, at most one instance
    # of the animation should be active at a time.
    #
    # Thus we need some instance management. We handle this as follows.
    #
    # An instance of a given animation type only becomes *reified* if it's the only one on that
    # GUI element (at the point in time when the new instance is being created).
    #
    # Only a reified instance actually starts animating.
    #
    # If the instance cannot be reified (i.e. there is already a previous instance on the same GUI element),
    # it enters *ghost mode*, where it only updates the existing instance (in some way appropriate for the
    # specific animation type), and then exits at the next frame.
    class_lock = threading.RLock()
    id_counter = 0  # for generating unique DPG IDs
    instances = {}  # DPG tag or ID (of `target_button`) -> animation instance

    # TODO: We could also customize `__new__` to return the existing instance, see `unpythpnic.symbol.sym`.
    def __init__(self, message, target_button, target_tooltip, target_text, duration):
        """Animation to flash a button (and its tooltip, if visible) to draw the user's attention.

        This is useful to let the user know that pressing the button actually took,
        when its action has no other immediately visible effects.

        Each GUI element (determined by `target_button`) can only have one `ButtonFlash`
        animation running at a time. If an instance already exists, trying to create the animation
        will restart the existing instance instead (and update its message to `message`).

        `message`: str, text to show in the `target_text` widget while the animation is running.
                   Original content will be restored automatically when the animation finishes normally.
                   Can be `None` for "don't change", or also when `target_text is None`.
        `target_button`: DPG tag or ID, the button to animate.
        `target_tooltip`: DPG tag or ID, the tooltip to animate. Can be `None`.
        `target_text`: DPG tag or ID, the text widget inside the tooltip to animate. Can be `None`.
        """
        super().__init__()
        self.instance_lock = threading.Lock()

        self.message = message
        self.target_button = target_button
        self.target_tooltip = target_tooltip
        self.target_text = target_text
        self.duration = duration

        # These are used during animation
        self.theme = None
        self.original_message = None
        self.reified = False  # `True`: running; `False`: ghost mode, update other instance and exit.

        self.start()

    def render_frame(self, t):
        if not self.reified:  # ghost mode
            return animation_action_cancel

        dt = (t - self.t0) / 10**9  # seconds since t0
        animation_pos = dt / self.duration

        if animation_pos >= 1.0:
            return animation_action_finish

        r = clamp(animation_pos)
        r = nonanalytic_smooth_transition(r)

        R0, G0, B0 = 96, 128, 96  # light green
        R1, G1, B1 = 45, 45, 48  # default button background color  TODO: read from global theme
        R = R0 * (1.0 - r) + R1 * r
        G = G0 * (1.0 - r) + G1 * r
        B = B0 * (1.0 - r) + B1 * r
        dpg.set_value(self.highlight_button_color, (R, G, B))
        dpg.set_value(self.highlight_hovered_color, (R, G, B))
        dpg.set_value(self.highlight_active_color, (R, G, B))
        dpg.set_value(self.highlight_popupbg_color, (R, G, B))

        return animation_action_continue

    def start(self):
        """Internal method, called automatically by constructor.

        Manages de-duplication (when added to the same GUI element as an existing animation of this type)
        as well as resource allocation. The resources are released by `finish` (called by `Animator`
        when the animation ends).
        """
        with self.instance_lock:
            if self.reified:  # already running (avoid double resource allocation and registration)
                self.reset()
                return

            with type(self).class_lock:
                # If an instance is already running on this GUI element, just restart it (and update its message).
                if self.target_button in type(self).instances:
                    other = type(self).instances[self.target_button]
                    other.message = self.message
                    dpg.set_value(other.target_text, other.message)
                    other.reset()
                    return

                with dpg.theme(tag=f"acknowledgement_highlight_theme_{type(self).id_counter}") as self.theme:  # create unique DPG ID each time
                    with dpg.theme_component(dpg.mvAll):
                        # common
                        dpg.add_theme_color(dpg.mvThemeCol_Text, (180, 255, 180))
                        # button
                        self.highlight_button_color = dpg.add_theme_color(dpg.mvThemeCol_Button, (96, 128, 96))
                        self.highlight_hovered_color = dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (96, 128, 96))
                        self.highlight_active_color = dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (96, 128, 96))
                        # tooltip
                        self.highlight_popupbg_color = dpg.add_theme_color(dpg.mvThemeCol_PopupBg, (96, 128, 96))
                type(self).id_counter += 1

                dpg.bind_item_theme(self.target_button, self.theme)
                if self.target_tooltip is not None:
                    dpg.bind_item_theme(self.target_tooltip, self.theme)
                if self.target_text is not None:
                    self.original_message = dpg.get_value(self.target_text)
                    dpg.set_value(self.target_text, self.message)
                    # dpg.bind_item_theme(self.target_text, self.theme)

                type(self).instances[self.target_button] = self
                self.reified = True  # This is the instance that animates `self.target_button`.

    def finish(self):
        with self.instance_lock:
            dpg.bind_item_theme(self.target_button, "disablable_button_theme")  # tag
            if self.target_tooltip is not None:
                dpg.bind_item_theme(self.target_tooltip, global_theme)
            if self.target_text is not None:
                dpg.set_value(self.target_text, self.original_message)
                # dpg.bind_item_theme(self.target_text, global_theme)
            dpg.delete_item(self.theme)
            self.theme = None
            self.reified = False

            with type(self).class_lock:
                type(self).instances.pop(self.target_button)


class PlotterPulsatingGlow(Animation):
    def __init__(self, cycle_duration):
        """Cyclic animation to pulsate the glow highlight for search result datapoints and selected datapoints."""
        super().__init__()
        self.cycle_duration = cycle_duration

    @classmethod
    def _compute_alpha(cls, x, n_data, n_many):
        """Compute translucency for plotter highlight, accounting for the effect of data mass on perceived translucency.

        High alpha per datapoint when very few datapoints; low alpha per datapoint when many datapoints.

        `x`: float, [0, 1]. The animation control channel. More means brighter.
        `n_data`: int, how many datapoints there are in the set being highlighted.
        `m_many`: int, how many datapoints are so many that the minimum per-datapoint alpha should be used.

        Returns the `alpha` value (int, [0, 255]).
        """
        # Coefficients for `alpha = a0 + a1 * x`, in the maximally bright and maximally dim cases.
        # These have been manually calibrated (via a coarse eyeball estimate) to give the same
        # perceived brightness for the highlighted set of datapoints regardless of the amount of data.
        a0_bright = 64
        a1_bright = 255 - a0_bright
        a0_dim = 32
        a1_dim = 64
        # Interpolate the coefficients from bright to dim, smoothly, depending on relative data mass.
        relative_data_mass = clamp(n_data / n_many)  # 0 ... 1, linear clamp
        r = nonanalytic_smooth_transition(relative_data_mass, m=2.0)  # 0 ... 1, smoothed
        a0 = a0_bright * (1.0 - r) + a0_dim * r
        a1 = a1_bright * (1.0 - r) + a1_dim * r
        # Compute the final alpha using the interpolated coefficients.
        alpha = a0 + int(a1 * x)
        # logger.debug(f"compute_data_highlight_alpha: relative data mass = {relative_data_mass}, smooth parameter = {r}, animation control value = {x}; result alpha = {alpha}")
        return alpha

    def render_frame(self, t):
        dt = (t - self.t0) / 10**9  # seconds since t0
        cycle_pos = dt / self.cycle_duration  # number of cycles since t0
        if cycle_pos > 1.0:  # prevent loss of accuracy in long sessions
            self.reset()
        cycle_pos = cycle_pos - float(int(cycle_pos))  # fractional part; raw position in animation cycle

        # We pulsate the search results and selected items at opposite phases to make both easy
        # to see when they overlap. We use colors that make the highlights stand out from the
        # Viridis colormap used for plotting the data.
        #
        # For how to do this in DPG, see e.g. https://github.com/hoffstadt/DearPyGui/issues/1512
        # Basically, bind a custom theme to the GUI widgets that need to have their color animated,
        # and then edit the theme's colors per-frame (just before render).
        #
        # Convert animation cycle position to animation control channel value.
        # Approach taken from SillyTavern-extras, `talkinghead.tha3.app.animate_breathing`.
        animation_pos = math.sin(cycle_pos * math.pi)**2  # 0 ... 1 ... 0, smoothly, with slow start and end, fast middle
        alpha_search = self._compute_alpha(animation_pos,
                                           len(unbox(search_result_data_idxs_box)),
                                           gui_config.n_many_searchresults)
        alpha_selection = self._compute_alpha(1.0 - animation_pos,
                                              len(unbox(selection_data_idxs_box)),
                                              gui_config.n_many_selection)
        dpg.set_value(search_results_highlight_color, (255, 96, 96, alpha_search))  # red
        dpg.set_value(selection_highlight_color, (96, 255, 255, alpha_selection))  # cyan

        return animation_action_continue


class CurrentItemControlsGlow(Animation):
    def __init__(self, cycle_duration):
        """Cyclic animation to pulsate the current item controls.

        Very specific; needs support from outside to update `current_item`, and the highlight
        is hardcoded for the current GUI controls layout (2x2 buttons per info panel item).
        """
        super().__init__()
        self.cycle_duration = cycle_duration

    def render_frame(self, t):
        """Update the highlight on the current item GUI controls.

        `cycle_pos`: float, [0, 1]. Position in the highlight pulsation animation cycle (to sync this with the plotter glow).

        This runs every frame, so the implementation is as minimal as possible, and exits as early as possible.

        This animation is controlled by `current_item`; see `update_current_item_info`.
        """
        if not current_item_info_lock.acquire(blocking=False):
            # If we didn't get the lock, it means `current_item` is being updated. Never mind, we can try again next frame.
            return animation_action_continue
        try:  # ok, got the lock
            have_current_item = False
            if current_item_info.item is not None:
                have_current_item = True
                x0 = current_item_info.x0
                y0 = current_item_info.y0
                # w = current_item_info.w
                # h = current_item_info.h
        finally:
            current_item_info_lock.release()

        if have_current_item:
            dt = (t - self.t0) / 10**9  # seconds since t0
            cycle_pos = dt / self.cycle_duration  # number of cycles since t0
            if cycle_pos > 1.0:  # prevent loss of accuracy in long sessions
                self.reset()
            cycle_pos = cycle_pos - float(int(cycle_pos))  # fractional part; raw position in animation cycle

            animation_pos = math.sin(cycle_pos * math.pi)**2  # 0 ... 1 ... 0, smoothly, with slow start and end, fast middle
            alpha_min = 16
            alpha_max = 48
            alpha = (1.0 - animation_pos) * alpha_min + animation_pos * alpha_max
            highlight_color = (196, 196, 255, alpha)  # same as `scroll_ends_here_color`, except alpha

            dpg.delete_item("viewport_drawlist", children_only=True)  # tag  # delete old draw items

            # Highlight the button group
            dpg.draw_rectangle((x0 - 4, y0 - 4),
                               (x0 + 2 * gui_config.toolbutton_w + 4, y0 + 2 * gui_config.toolbutton_w + 1),  # kluge constants chosen manually to make this look good
                               color=highlight_color,
                               fill=highlight_color,
                               rounding=8,
                               parent="viewport_drawlist")  # tag
            # # Variant, where the current item is the title text
            # dpg.draw_rectangle((x0 - 4, y0),
            #                    (x0 + gui_config.title_wrap_w + 4, y0 + h + 4),
            #                    color=highlight_color,
            #                    fill=highlight_color,
            #                    rounding=8,
            #                    parent="viewport_drawlist")  # tag
        else:
            dpg.delete_item("viewport_drawlist", children_only=True)  # tag  # delete old draw items

        return animation_action_continue


class Dimmer(Overlay):
    def __init__(self, target, tag):
        """Dimmer for a child window. Can be used e.g. to indicate that the window is updating.

        `target`: DPG ID or tag. The child window for which to build the overlay.
        `tag`: DPG tag, for naming the overlay.
        """
        super().__init__(target, tag)
        self.window = None
        self.drawlist = None

    def build(self, rebuild=False):
        # Ensure stuff we depend on is initialized before we try to create this
        if dpg.get_frame_count() < 10:
            return None

        # dimmer_color = (0, 0, 0, 128)
        dimmer_color = (37, 37, 38, 255)  # TODO: This is the info panel content area background color in the default theme. Figure out how to get colors from a theme.

        with self.overlay_update_lock:  # This prevents a crash upon hammering F11 (toggle fullscreen) while the info panel is updating (causing lots of rebuilds)
            if not rebuild and (self.window is not None):  # Avoid unnecessary rebuilding
                return

            # We want 8 pixels of rounding on each side (like window rounding),
            # so we must make the overlay window 16 pixels larger in each direction.
            config = dpg.get_item_configuration(self.target)
            w = config["width"]
            h = config["height"]
            w += 16
            h += 16

            # Child windows don't have a `rect_min`; instead, they have `pos`.
            pos = dpg.get_item_pos(self.target)
            # Center the overlay on the target. Now this window covers the target child window.
            pos = [pos[0] - 8, pos[1] - 8]

            if self.window is None:  # create only once ("rebuild" here actually means "reconfigure")
                logger.debug(f"Dimmer.build: frame {dpg.get_frame_count()}: instance '{self.tag}' creating overlay")
                with dpg.window(show=False, modal=False, no_title_bar=True, tag=self.tag,
                                pos=pos,
                                width=w, height=h,
                                no_collapse=True,
                                no_focus_on_appearing=True,
                                # no_bring_to_front_on_focus=True,  # for some reason, prevents displaying the window at all
                                no_resize=True,
                                no_move=True,
                                no_background=True,
                                no_scrollbar=True,
                                no_scroll_with_mouse=True) as self.window:
                    self.drawlist = dpg.add_drawlist(width=w, height=h)
                rebuild = True

            if rebuild:
                logger.debug(f"Dimmer.build: frame {dpg.get_frame_count()}: instance '{self.tag}' updating drawlist")
                dpg.delete_item(self.drawlist, children_only=True)
                dpg.configure_item(self.window, width=w, height=h)
                dpg.configure_item(self.drawlist, width=w, height=h)
                dpg.draw_rectangle((0, 0), (w - 16, h - 16), color=dimmer_color, fill=dimmer_color, rounding=8, parent=self.drawlist)

    def show(self):
        """Dim the target window (e.g. to show that it is updating)."""
        self.build()
        try:  # EAFP to avoid TOCTTOU
            dpg.show_item(self.window)
        except SystemError:  # does not exist
            pass

    def hide(self):
        """Un-dim the target window."""
        try:  # EAFP to avoid TOCTTOU
            dpg.hide_item(self.window)
        except SystemError:  # does not exist
            pass

info_panel_dimmer_overlay = None
def create_info_panel_dimmer_overlay():
    """Create a dimmer for the info panel. Used for indicating that the info panel is updating."""
    global info_panel_dimmer_overlay
    if info_panel_dimmer_overlay is None:
        info_panel_dimmer_overlay = Dimmer(target="item_information_panel",
                                           tag="dimmer_overlay_window")
        info_panel_dimmer_overlay.build()
def show_info_panel_dimmer_overlay():
    """Dim the info panel."""
    create_info_panel_dimmer_overlay()
    info_panel_dimmer_overlay.show()
def hide_info_panel_dimmer_overlay():
    """Un-dim the info panel."""
    create_info_panel_dimmer_overlay()
    info_panel_dimmer_overlay.hide()


class ScrollEndFlasher(Overlay, Animation):
    def __init__(self, target, tag, duration):
        """Flasher to indicate when the end of a scrollable area has been reached.

        `target`: DPG ID or tag. The child window for which to build the overlay.
        `tag`: DPG tag, for naming the overlay.
        `duration`: float, fadeout animation duration in seconds.
        """
        super().__init__(target, tag)

        self.duration = duration

        self.window_top = None
        self.drawlist_top = None
        self.window_bottom = None
        self.drawlist_bottom = None

        self.animation_running = False
        self.where = None  # the kind of the currently running animation: "top", "bottom", or "both"

    def show_by_position(self, target_y_scroll):
        """Like `show`, but determine position automatically.

        `target_y_scroll`: int, target scroll position, in scrollbar coordinates of `self.target`.
                           Special value -1 means the end position.

        The scroll position is parameterized to allow animated scrolling to work;
        if you get it from the scrollbar (`dpg.get_y_scroll(some_child_window)`),
        the value may be out of date if it is being updated during the current frame.

        This allows dispatching the flashing animation immediately, without waiting
        for one frame (or sometimes several frames; see source code of `SmoothScrolling`)
        for the scrollbar position to update.

        Returns which end was flashed (one of "top", "bottom", "both"),
        or `None` if `target_y_scroll` was not at either end.
        """
        max_y_scroll = dpg.get_y_scroll_max(self.target)  # tag

        if target_y_scroll == -1:  # end?
            target_y_scroll = max_y_scroll

        where = None
        if max_y_scroll > 0:
            if target_y_scroll == 0:
                where = "top"
            elif target_y_scroll == max_y_scroll:
                where = "bottom"
        else:  # less than a screenful of data -> reached both ends.
            if target_y_scroll == 0:
                where = "both"

        logger.debug(f"ScrollEndFlasher.show_by_position: target_y_scroll = {target_y_scroll}, max_y_scroll = {max_y_scroll}, where = {where}")

        if where is not None:
            self.show(where)

        return where

    def show(self, where):
        """Dispatch the animation.

        This indicates in the GUI that the target child window cannot scroll any further
        in the specified direction.

        `where`: str, the extremity that has been reached. One of "top", "bottom", "both".
                 Here "both" is useful if there is less than one screenful of data

                 Sometimes using "both" is also easier, when there is no meaningful delta
                 from which to compute the direction the user is scrolling to, such as
                 with an instant programmatic jump to the scroll target position.
        """
        if self.animation_running:  # only one simultaneous animation per instance; replace old animation if it exists (effectively restarting the animation).
            animator.cancel(self, finalize=False)  # no need to call `finish` since we'll start a new animation of the same type on the same GUI element right away.
        self.animation_running = True
        self.where = where
        animator.add(self)

    def hide(self):
        """Hide the overlay immediately. Called automatically by `finish` when the animation ends."""
        dpg.hide_item(self.window_top)
        dpg.hide_item(self.window_bottom)

    def render_frame(self, t):
        """Called automatically by `Animator`."""
        if dpg.get_frame_count() < 10:
            return None

        dt = (t - self.t0) / 10**9  # seconds since t0
        animation_pos = dt / self.duration

        if animation_pos >= 1.0:
            return animation_action_finish
        if is_any_modal_window_visible():
            return animation_action_finish

        scroll_ends_here_color = [196, 196, 255, 64]

        r = clamp(animation_pos)
        r = nonanalytic_smooth_transition(r)
        alpha = (1.0 - r) * scroll_ends_here_color[3]
        scroll_ends_here_color[3] = alpha

        with self.overlay_update_lock:
            # We want 8 pixels of rounding on each side (like window rounding),
            # so we must make the overlay window 16 pixels larger in each direction.
            config = dpg.get_item_configuration(self.target)
            w = config["width"]
            h = config["height"]
            w += 16
            h += 16

            # Child windows don't have a `rect_min`; instead, they have `pos`.
            pos = dpg.get_item_pos(self.target)
            # Center the overlay on the target. Now this window covers the target child window.
            pos = [pos[0] - 8, pos[1] - 8]

            # Use two windows, one for each end, to avoid the overlay capturing mouse input (especially the wheel) while it is shown.
            # We create these just once.
            if self.window_top is None:
                logger.debug(f"ScrollEndFlasher.build: frame {dpg.get_frame_count()}: instance '{self.tag}' creating overlay (top)")
                with dpg.window(show=False, modal=False, no_title_bar=True, tag=f"{self.tag}_window_top",
                                pos=pos,
                                width=w, height=48,
                                no_collapse=True,
                                no_focus_on_appearing=True,
                                no_resize=True,
                                no_move=True,
                                no_background=True,
                                no_scrollbar=True,
                                no_scroll_with_mouse=True) as self.window_top:
                    self.drawlist_top = dpg.add_drawlist(width=w, height=48)
            if self.window_bottom is None:
                logger.debug(f"ScrollEndFlasher.build: frame {dpg.get_frame_count()}: instance '{self.tag}' creating overlay (bottom)")
                with dpg.window(show=False, modal=False, no_title_bar=True, tag=f"{self.tag}_window_bottom",
                                pos=[pos[0], pos[1] + h - 48],
                                width=w, height=48,
                                no_collapse=True,
                                no_focus_on_appearing=True,
                                no_resize=True,
                                no_move=True,
                                no_background=True,
                                no_scrollbar=True,
                                no_scroll_with_mouse=True) as self.window_bottom:
                    self.drawlist_bottom = dpg.add_drawlist(width=w, height=48)

            # logger.debug(f"Dimmer.build: frame {dpg.get_frame_count()}: instance '{self.tag}' updating geometry and drawing")  # too spammy

            dpg.delete_item(self.drawlist_top, children_only=True)
            dpg.configure_item(self.window_top, width=w, height=48)
            dpg.configure_item(self.drawlist_top, width=w, height=48)

            dpg.delete_item(self.drawlist_bottom, children_only=True)
            dpg.configure_item(self.window_bottom, width=w, height=48)
            dpg.configure_item(self.drawlist_bottom, width=w, height=48)
            dpg.set_item_pos(self.window_bottom, [pos[0], pos[1] + h - 48])

            icon_size = 24
            def draw_on(parent, icon_text):
                # TODO: Improve the visual look (a cap of a circle would look better than a rounded rectangle)
                dpg.draw_rectangle((0, 0), (w - 16, 32), color=(0, 0, 0, 0), fill=scroll_ends_here_color, rounding=8, parent=parent)
                # TODO: Get rid of the kluge offsets.
                icon_upper_left = ((w - icon_size) // 2 - 12 + 3, 3)  # make the icon exactly centered on the rounded rectangle (this was measured in GIMP)  # 3 px: inner padding?
                t = dpg.draw_text(icon_upper_left, icon_text, size=icon_size, color=scroll_ends_here_color, parent=parent)
                dpg.bind_item_font(t, icon_font_solid)
            if self.where in ("top", "both"):
                draw_on(self.drawlist_top, icon_text=fa.ICON_ARROWS_UP_TO_LINE)
            if self.where in ("bottom", "both"):
                draw_on(self.drawlist_bottom, icon_text=fa.ICON_ARROWS_DOWN_TO_LINE)

            # # Draw a "no" symbol (crossed-out circle). (See also `fa.ICON_BAN`.)
            # circle_center = (w / 2 - 8, 14)
            # circle_radius = 12
            # line_thickness = 4
            # offs_45deg = circle_radius * 0.5**0.5
            # dpg.draw_circle(circle_center, circle_radius,
            #                 thickness=line_thickness, color=(120, 180, 255, alpha),  # blue, with alpha
            #                 parent=self.drawlist_top)
            # dpg.draw_line((circle_center[0] - offs_45deg, circle_center[1] - offs_45deg),
            #               (circle_center[0] + offs_45deg, circle_center[1] + offs_45deg),
            #               thickness=line_thickness, color=(120, 180, 255, alpha),  # blue, with alpha
            #               parent=self.drawlist_top)

            dpg.show_item(self.window_top)
            dpg.show_item(self.window_bottom)

        return animation_action_continue

    def finish(self):
        """Animation finish callback for `Animator`.

        Called when the animation finishes normally.
        """
        self.animation_running = False
        self.where = None
        self.hide()

info_panel_scroll_end_flasher = ScrollEndFlasher(target="item_information_panel",
                                                 tag="scroll_end_flasher",
                                                 duration=gui_config.scroll_ends_here_duration)


class SmoothScrolling(Animation):
    class_lock = threading.RLock()
    instances = {}  # DPG tag or ID (of `target_child_window`) -> animation instance

    def __init__(self, target_child_window, target_y_scroll, smooth=True, smooth_step=0.8, flasher=None, finish_callback=None):
        """Scroll a child window smoothly.

        Each GUI element (determined by `target_child_window`) can only have one `SmoothScrolling`
        animation running at a time. If an instance already exists, trying to create the animation
        will update its `target_y_scroll` instead.

        `target_child_window`: DPG tag or ID, the child window to scroll.
        `target_y_scroll`: int, target scroll position in scrollbar coordinates.
        `smooth`: bool.
                  If `True`, will animate a smooth scroll.
                  If `False`, will jump to target position instantly (the point is to offer the same API).
        `smooth_step`: float, a nondimensional rate in the half-open interval (0, 1].
                       Independent of the render FPS.
        `flasher`: `ScrollEndFlasher` instance, optional.
                   Automatically activated when the top/bottom is reached.
        `finish_callback`: 0-argument callable. Run some custom code when the animation finishes normally.
                           Keep it minimal; trying to instantiate a new scroll animation will block while
                           the callback is running (because a new instance might target the same GUI element,
                           and we guarantee the teardown to be atomic).

        Note that mouse wheel and scrollbar dragging do not invoke the scroll animation; for those,
        the scroll position is handled internally by DPG. Hence those don't cause a flash here.
        If you want, you can handle the mouse wheel case separately in a global mouse wheel callback.

        This is pretty sophisticated, to make the movement smooth, but also keep things working as expected
        when the target position changes on the fly.

        The animation depends only on the current and target positions, and has a reference rate,
        no reference duration. Essentially, we use Newton's law of cooling (first-order ODE),
        and apply its analytical solution.

        For the math details, see the detailed comment below the source code of this class.
        """
        super().__init__()
        self.instance_lock = threading.Lock()

        self.target_child_window = target_child_window
        self.target_y_scroll = target_y_scroll
        self.smooth = smooth
        self.smooth_step = smooth_step
        self.flasher = flasher
        self.finish_callback = finish_callback

        self.prev_frame_new_y_scroll = None  # target position of last frame, for monitoring of stuck animation
        self.update_pending_frames = 0
        self.fracpart = 0.0  # fractional part of position, for subpixel correction
        self.reified = False  # `True`: running; `False`: ghost mode, update other instance and exit.

        self.start()

    def render_frame(self, t):
        if not self.reified:  # ghost mode
            return animation_action_cancel

        update_pending_threshold = 4  # Frames. Smaller threshold looks better, but may fire prematurely if a GUI update takes too many frames.
        action = animation_action_continue

        with self.instance_lock:
            current_y_scroll = dpg.get_y_scroll(self.target_child_window)
            if not self.smooth:
                if current_y_scroll == self.target_y_scroll:
                    action = animation_action_finish
                    # If we reach the start or the end of the scrollable, flash it.
                    if self.flasher is not None:
                        self.flasher.show_by_position(self.target_y_scroll)
                # First frame in this scroll? -> do it
                elif self.prev_frame_new_y_scroll is None:
                    new_y_scroll = self.target_y_scroll
                    self.prev_frame_new_y_scroll = new_y_scroll  # No longer first frame (in non-smooth mode, doesn't matter what value we store here as long as it's not `None`).
                    dpg.set_y_scroll(self.target_child_window, new_y_scroll)
                # Waited for a short timeout? -> time to check for end of scrollbar (but this shouldn't happen now that `scroll_info_panel_to_position` clamps the value to the max allowed by the scrollbar)
                elif self.update_pending_frames >= update_pending_threshold:
                    action = animation_action_finish
                    if current_y_scroll != self.target_y_scroll:
                        logger.debug(f"SmoothScrolling.render_frame: frame {dpg.get_frame_count()}: instance for '{self.target_child_window}': did not reach target position (target position past end of scrollbar?)")
                        if self.flasher is not None:
                            self.flasher.show(where="bottom")
                # Waiting for the timeout?
                else:
                    self.update_pending_frames = self.update_pending_frames + 1
            else:
                # Only proceed if DPG has actually applied our previous update, or if this is the first update since this scroll animation was started.
                # This prevents stuttering, as well as keeps our subpixel calculations correct.
                if self.prev_frame_new_y_scroll is None or current_y_scroll == self.prev_frame_new_y_scroll:
                    self.update_pending_frames = 0

                    # Magic section -->
                    # Framerate correction for rate-based animation, to reach a constant animation rate per unit of wall time, regardless of render FPS.
                    CALIBRATION_FPS = 25  # FPS for which `step` was calibrated
                    xrel = 0.5  # just some convenient value
                    step = self.smooth_step
                    alpha_orig = 1.0 - step
                    if 0 < alpha_orig < 1:
                        avg_render_fps = dpg.get_frame_rate()

                        # For a constant target position and original `α`, compute the number of animation frames to cover `xrel` of distance from initial position to final position.
                        # This is how many frames we need at `CALIBRATION_FPS`.
                        n_orig = math.log(1.0 - xrel) / math.log(alpha_orig)
                        # Compute the scaled `n`, to account for `avg_render_fps`. Note the direction: we need a smaller `n` (fewer animation frames) if the render runs slower than `CALIBRATION_FPS`.
                        n_scaled = (avg_render_fps / CALIBRATION_FPS) * n_orig
                        # Then compute the `α` that reaches `xrel` distance in `n_scaled` animation frames.
                        alpha_scaled = (1.0 - xrel)**(1 / n_scaled)
                    else:  # avoid some divisions by zero at the extremes
                        alpha_scaled = alpha_orig
                    step_scaled = 1.0 - alpha_scaled
                    # <-- End magic section

                    # Calculate old and new positions, with subpixel correction (IMPORTANT!).
                    subpixel_corrected_current_y_scroll = current_y_scroll + self.fracpart  # NOTE: float
                    remaining = self.target_y_scroll - subpixel_corrected_current_y_scroll  # remaining distance, float
                    delta = step_scaled * remaining  # distance to cover in this frame, float

                    subpixel_corrected_new_y_scroll = subpixel_corrected_current_y_scroll + delta
                    fracpart = subpixel_corrected_new_y_scroll - int(subpixel_corrected_new_y_scroll)
                    new_y_scroll = int(subpixel_corrected_new_y_scroll)  # NOTE: truncate, no rounding of any kind

                    logger.debug(f"SmoothScrolling.render_frame: frame {dpg.get_frame_count()}: instance for '{self.target_child_window}': old raw = {current_y_scroll}, old subpixel = {subpixel_corrected_current_y_scroll}, delta = {delta}, new subpixel = {subpixel_corrected_new_y_scroll}, target = {self.target_y_scroll}, start-of-frame remaining distance = {remaining}")

                    # Once we reach less than one pixel of distance from the final position, just snap there and end the animation.
                    # We jump at <= 1.0, not < 1.0, to avoid some roundoff trouble.
                    if abs(self.target_y_scroll - subpixel_corrected_current_y_scroll) <= 1.0:
                        new_y_scroll = self.target_y_scroll
                        action = animation_action_finish
                        logger.debug(f"SmoothScrolling.render_frame: frame {dpg.get_frame_count()}: instance for '{self.target_child_window}': scrolling completed")

                        # If we reach the start or the end of the scrollable, flash it.
                        if self.flasher is not None:
                            self.flasher.show_by_position(self.target_y_scroll)

                    if action is animation_action_continue:
                        self.prev_frame_new_y_scroll = new_y_scroll
                        self.fracpart = fracpart

                    dpg.set_y_scroll(self.target_child_window, new_y_scroll)

                # Timeout waiting for DPG to update the position? -> probably end of scrollbar (but shouldn't happen now that `scroll_info_panel_to_position` clamps the value to the max allowed by the scrollbar)
                elif self.update_pending_frames >= update_pending_threshold:
                    action = animation_action_finish
                    logger.debug(f"SmoothScrolling.render_frame: frame {dpg.get_frame_count()}: instance for '{self.target_child_window}': timeout waiting for scrollbar to update its scroll position (target position past end of scrollbar?)")
                    if self.flasher is not None:
                        self.flasher.show(where="bottom")

                # Waiting for DPG to update the position?
                else:
                    self.update_pending_frames = self.update_pending_frames + 1

        return action

    def start(self):
        """Internal method, called automatically by constructor.

        Manages de-duplication (when added to the same GUI element as an existing animation of this type).
        """
        with self.instance_lock:
            if self.reified:  # already running (avoid double resource allocation and registration)
                return

            with type(self).class_lock:
                # If an instance is already running on this GUI element, just update its target scroll position.
                # This allows a seamless transition to the new scroll animation, retaining the subpixel position.
                if self.target_child_window in type(self).instances:
                    other = type(self).instances[self.target_child_window]
                    with other.instance_lock:
                        other.target_y_scroll = self.target_y_scroll
                    return

                type(self).instances[self.target_child_window] = self
                self.reified = True  # This is the instance that animates `self.target_child_window`.

    def finish(self):
        with type(self).class_lock:
            if self.finish_callback is not None:
                self.finish_callback()
            type(self).instances.pop(self.target_child_window)

# The math for the scroll animation comes from SillyTavern-extras, `talkinghead.tha3.app`, function `interpolate_pose`.
# This depends only on the current and target positions, and has a reference *rate*, no reference duration.
# This allows us to change the target position while the animation is running, and it'll adapt.
#
# Pasting this comment as-is. Here in Raven, the equivalent of "pose" is the scroll position.
#
# ---8<---8<---8<---
#
# The `step` parameter is calibrated against animation at 25 FPS, so we must scale it appropriately, taking
# into account the actual FPS.
#
# How to do this requires some explanation. Numericist hat on. Let's do a quick back-of-the-envelope calculation.
# This pose interpolator is essentially a solver for the first-order ODE:
#
#   u' = f(u, t)
#
# Consider the most common case, where the target pose remains constant over several animation frames.
# Furthermore, consider just one morph (they all behave similarly). Then our ODE is Newton's law of cooling:
#
#   u' = -β [u - u∞]
#
# where `u = u(t)` is the temperature, `u∞` is the constant temperature of the external environment,
# and `β > 0` is a material-dependent cooling coefficient.
#
# But instead of numerical simulation at a constant timestep size, as would be typical in computational science,
# we instead read off points off the analytical solution curve. The `step` parameter is *not* the timestep size;
# instead, it controls the relative distance along the *u* axis that should be covered in one simulation step,
# so it is actually related to the cooling coefficient β.
#
# (How exactly: write the left-hand side as `[unew - uold] / Δt + O([Δt]²)`, drop the error term, and decide
#  whether to use `uold` (forward Euler) or `unew` (backward Euler) as `u` on the right-hand side. Then compare
#  to our update formula. But those details don't matter here.)
#
# To match the notation in the rest of this code, let us denote the temperature (actually pose morph value) as `x`
# (instead of `u`). And to keep notation shorter, let `β := step` (although it's not exactly the `β` of the
# continuous-in-time case above).
#
# To scale the animation speed linearly with regard to FPS, we must invert the relation between simulation step
# number `n` and the solution value `x`. For an initial value `x0`, a constant target value `x∞`, and constant
# step `β ∈ (0, 1]`, the pose interpolator produces the sequence:
#
#   x1 = x0 + β [x∞ - x0] = [1 - β] x0 + β x∞
#   x2 = x1 + β [x∞ - x1] = [1 - β] x1 + β x∞
#   x3 = x2 + β [x∞ - x2] = [1 - β] x2 + β x∞
#   ...
#
# Note that with exact arithmetic, if `β < 1`, the final value is only reached in the limit `n → ∞`.
# For floating point, this is not the case. Eventually the increment becomes small enough that when
# it is added, nothing happens. After sufficiently many steps, in practice `x` will stop just slightly
# short of `x∞` (on the side it approached the target from).
#
# (For performance reasons, when approaching zero, one may need to beware of denormals, because those
#  are usually implemented in (slow!) software on modern CPUs. So especially if the target is zero,
#  it is useful to have some very small cutoff (inside the normal floating-point range) after which
#  we make `x` instantly jump to the target value.)
#
# Inserting the definition of `x1` to the formula for `x2`, we can express `x2` in terms of `x0` and `x∞`:
#
#   x2 = [1 - β] ([1 - β] x0 + β x∞) + β x∞
#      = [1 - β]² x0 + [1 - β] β x∞ + β x∞
#      = [1 - β]² x0 + [[1 - β] + 1] β x∞
#
# Then inserting this to the formula for `x3`:
#
#   x3 = [1 - β] ([1 - β]² x0 + [[1 - β] + 1] β x∞) + β x∞
#      = [1 - β]³ x0 + [1 - β]² β x∞ + [1 - β] β x∞ + β x∞
#
# To simplify notation, define:
#
#   α := 1 - β
#
# We have:
#
#   x1 = α  x0 + [1 - α] x∞
#   x2 = α² x0 + [1 - α] [1 + α] x∞
#      = α² x0 + [1 - α²] x∞
#   x3 = α³ x0 + [1 - α] [1 + α + α²] x∞
#      = α³ x0 + [1 - α³] x∞
#
# This suggests that the general pattern is (as can be proven by induction on `n`):
#
#   xn = α**n x0 + [1 - α**n] x∞
#
# This allows us to determine `x` as a function of simulation step number `n`. Now the scaling question becomes:
# if we want to reach a given value `xn` by some given step `n_scaled` (instead of the original step `n`),
# how must we change the step size `β` (or equivalently, the parameter `α`)?
#
# To simplify further, observe:
#
#   x1 = α x0 + [1 - α] [[x∞ - x0] + x0]
#      = [α + [1 - α]] x0 + [1 - α] [x∞ - x0]
#      = x0 + [1 - α] [x∞ - x0]
#
# Rearranging yields:
#
#   [x1 - x0] / [x∞ - x0] = 1 - α
#
# which gives us the relative distance from `x0` to `x∞` that is covered in one step. This isn't yet much
# to write home about (it's essentially just a rearrangement of the definition of `x1`), but next, let's
# treat `x2` the same way:
#
#   x2 = α² x0 + [1 - α] [1 + α] [[x∞ - x0] + x0]
#      = [α² x0 + [1 - α²] x0] + [1 - α²] [x∞ - x0]
#      = [α² + 1 - α²] x0 + [1 - α²] [x∞ - x0]
#      = x0 + [1 - α²] [x∞ - x0]
#
# We obtain
#
#   [x2 - x0] / [x∞ - x0] = 1 - α²
#
# which is the relative distance, from the original `x0` toward the final `x∞`, that is covered in two steps
# using the original step size `β = 1 - α`. Next up, `x3`:
#
#   x3 = α³ x0 + [1 - α³] [[x∞ - x0] + x0]
#      = α³ x0 + [1 - α³] [x∞ - x0] + [1 - α³] x0
#      = x0 + [1 - α³] [x∞ - x0]
#
# Rearranging,
#
#   [x3 - x0] / [x∞ - x0] = 1 - α³
#
# which is the relative distance covered in three steps. Hence, we have:
#
#   xrel := [xn - x0] / [x∞ - x0] = 1 - α**n
#
# so that
#
#   α**n = 1 - xrel              (**)
#
# and (taking the natural logarithm of both sides)
#
#   n log α = log [1 - xrel]
#
# Finally,
#
#   n = [log [1 - xrel]] / [log α]
#
# Given `α`, this gives the `n` where the interpolator has covered the fraction `xrel` of the original distance.
# On the other hand, we can also solve (**) for `α`:
#
#   α = (1 - xrel)**(1 / n)
#
# which, given desired `n`, gives us the `α` that makes the interpolator cover the fraction `xrel` of the original distance in `n` steps.
#
# ---8<---8<---8<---

task_status_stopped = sym("stopped")
task_status_pending = sym("pending")
task_status_running = sym("running")

scroll_animation = None  # keep a reference so that we can stop the scroll animation, and only that animation
scroll_animation_lock = threading.RLock()
def clear_global_scroll_animation_reference():  # used as finish callback for `SmoothScrolling`
    global scroll_animation
    with scroll_animation_lock:
        scroll_animation = None

def update_animations():
    # # Resize the search field dynamically. We don't need this with the current layout; keeping for documentation only.
    # # Note that in DPG, text widgets have no `width` (always zero), but they have a rect_size.
    # w_header, h_header = dpg.get_item_rect_size(search_header_text)
    # w_plotarea = dpg.get_item_width(theplot)
    # # x0, y0 = dpg.get_item_rect_min(search_header_text)
    # # print(w_header, w_plotarea, x0)
    # dpg.set_item_width("search_field", w_plotarea - w_header)

    # HACK: force correct info panel height.
    # At app startup, the main window thinks it has height=100, which is wrong.
    # The scroll end flasher needs the correct height for "item_information_panel"  # tag
    # to know the viewport coordinates for its bottom overlay.
    _update_info_panel_height()

    # ----------------------------------------
    # Show loading spinner when info panel is refreshing

    if unbox(info_panel_render_status_box) is task_status_pending:
        dpg.hide_item("info_panel_rendering_spinner")  # tag
        dpg.show_item("info_panel_pending_spinner")  # tag
    elif unbox(info_panel_render_status_box) is task_status_running:
        dpg.hide_item("info_panel_pending_spinner")  # tag
        dpg.show_item("info_panel_rendering_spinner")  # tag
    else:  # task_status_stopped
        dpg.hide_item("info_panel_pending_spinner")  # tag
        dpg.hide_item("info_panel_rendering_spinner")  # tag

    # ----------------------------------------
    # Update search-related GUI elements

    # Color the search field
    search_string = unbox(search_string_box)
    search_result_data_idxs = unbox(search_result_data_idxs_box)
    if not search_string:
        dpg.set_value(search_field_text_color, (255, 255, 255))  # no search active
    else:
        if len(search_result_data_idxs):
            dpg.set_value(search_field_text_color, (180, 255, 180))  # found, green
        else:
            dpg.set_value(search_field_text_color, (255, 128, 128))  # not found, red

    update_current_search_result_status()  # The "[x/x]" topmost currently visible search result indicator (also updates `current_item` for `CurrentItemControlsGlow`)

    # ----------------------------------------
    # Update various other things that need per-frame updates

    update_info_panel_navigation_controls()  # Info panel top/bottom/pageup/pagedown buttons

    # ----------------------------------------
    # Render all currently running animations

    animator.render_frame()

current_item_info = env(item=None, x0=None, y0=None, w=None, h=None)  # `item`: GUI widget DPG tag or ID; `x0`, `y0`: screen space coordinates, in pixels; `w`, `h`: in pixels
current_item_info_lock = threading.Lock()

def update_current_item_info():  # Called per-frame by `update_current_search_result_status` (which already holds `info_panel_content_lock`, so as to avoid unnecessary release/re-lock).
    """Update the data used to track the on-screen position of the current item.

    `CurrentItemControlsGlow` uses this data to highlight the current item's per-item GUI controls
    (to show which item any hotkeys affecting a single item apply to).

    When any modal window is visible, the current item info is cleared, thus turning the highlight off
    while a modal window is open.
    """
    with info_panel_content_lock:
        if is_any_modal_window_visible():
            current_item = None
        else:
            current_item = _get_current_info_panel_item()
            # if current_item is not None:  # find the title text widget (old variant of highlighting)
            #     current_item = find_item_depth_first(current_item, accept=is_entry_title_text_item)
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
    """Clear the data used to track the on-screen position of the current item.

    `CurrentItemControlsGlow` uses this data to highlight the current item's per-item GUI controls
    (to show which item any hotkeys affecting a single item apply to). Clearing it turns the highlight off.
    """
    with current_item_info_lock:
        current_item_info.item = None
        current_item_info.x0 = None
        current_item_info.y0 = None
        current_item_info.w = None
        current_item_info.h = None

# --------------------------------------------------------------------------------
# Set up the main window

logger.info("Initial GUI setup...")
with timer() as tim:
    with dpg.window(tag="main_window", label="Raven main window") as main_window:  # DPG "window" inside the app OS window ("viewport"), container for the whole GUI
        dpg.add_viewport_drawlist(front=True, tag="viewport_drawlist")  # for current item highlight in info panel
        with dpg.group(tag="main_container",
                       horizontal=True):  # Container to make a horizontal top-level layout

            # Info panel
            with dpg.group(tag="info_and_help"):
                # Title
                with dpg.child_window(tag="item_information_header",
                                      width=gui_config.info_panel_w,
                                      height=gui_config.info_panel_header_h,
                                      no_scrollbar=True,  # we want to hide the "hello"
                                      no_scroll_with_mouse=True):
                    with dpg.group(horizontal=True, tag="item_information_header_group"):
                        # Copy report to clipboard button
                        # The callback function is defined (and bound) later when we define the info panel.
                        copy_report_button = dpg.add_button(tag="copy_report_to_clipboard_button",
                                                            label=fa.ICON_COPY,
                                                            enabled=False)
                        dpg.bind_item_font("copy_report_to_clipboard_button", icon_font_solid)  # tag
                        dpg.bind_item_theme("copy_report_to_clipboard_button", "disablable_button_theme")  # tag
                        with dpg.tooltip("copy_report_to_clipboard_button") as copy_report_tooltip:  # tag
                            copy_report_tooltip_text = dpg.add_text("Copy report to clipboard [F8]\n    no modifier: as plain text\n    with Shift: as Markdown")  # TODO: DRY duplicate definitions for labels

                        # Static header text
                        dpg.add_text("Item information", color=(255, 255, 255, 255), tag="item_information_title")

                        # Dynamic header text, this will be replaced by the item count statistics when something is shown in the info panel.
                        item_information_text = dpg.add_text("[nothing selected]", color=(140, 140, 140, 255), tag="item_information_selection_item_count")  # TODO: DRY duplicate definitions for labels
                        total_count_text = dpg.add_text("[x items shown]", color=(140, 140, 140, 255), tag="item_information_total_count", show=False)

                        # Spinners to indicate that the item info panel is refreshing. The color shows the state (update pending, or updating).
                        # At most one spinner is shown at a time.
                        dpg.add_loading_indicator(style=0,
                                                  radius=1.0,
                                                  color=(255, 96, 96, 255),  # orange
                                                  secondary_color=(128, 32, 32, 255),
                                                  show=False,
                                                  tag="info_panel_pending_spinner")
                        dpg.add_loading_indicator(style=0,
                                                  radius=1.0,
                                                  color=(96, 96, 255, 255),  # blue
                                                  secondary_color=(32, 32, 128, 255),
                                                  show=False,
                                                  tag="info_panel_rendering_spinner")

                    # FIX: Trigger Markdown renderer to load all font families at startup, so it won't bite us with a race condition later when we populate the info panel.
                    #
                    # The render seems to be asynchronous, so if you populate other stuff into the same child window while `dpg_markdown` is loading its fonts,
                    # some place in the rendering engine may forget where it was going. Some of your content will be omitted, and the rest abruptly injected
                    # into the middle of the Markdown render that was in progress. Triggering the font loading now (while we're NOT populating the info panel
                    # in a tight loop) seems to avoid this issue.
                    #
                    # This could also have something to do with the DPG container stack, which seems to be global. Just to be safe, we have already changed
                    # both background renderers (the info panel and the annotation) not to use the container stack, but to parent each GUI widget explicitly.
                    #
                    # Note also that if we call `dpg_markdown.add_text` twice or more before the first frame renders, it segfaults DPG (at least 1.11).
                    # So this is the only Markdown render in the initial main window setup. Any other Markdown text is rendered later.
                    with dpg.group(tag="markdown_font_loader_trigger_dummy"):
                        dpg_markdown.add_text("hello, *hello*, **hello**, ***hello***")  # regular, bold, italic, bold italic

                with dpg.child_window(tag="item_information_navigation_controls",
                                      width=gui_config.info_panel_w,
                                      height=gui_config.info_panel_header_h,
                                      no_scrollbar=True,
                                      no_scroll_with_mouse=True):
                    with dpg.group(horizontal=True, tag="item_information_navigation_controls_group"):
                        # The callback functions for all buttons in this group are defined (and bound) later when we define the info panel.
                        go_to_top_button = dpg.add_button(tag="go_to_top_button",
                                                          label=fa.ICON_ANGLES_UP,
                                                          width=gui_config.info_panel_button_w,
                                                          enabled=False)
                        dpg.bind_item_font("go_to_top_button", icon_font_solid)  # tag
                        dpg.bind_item_theme("go_to_top_button", "disablable_button_theme")  # tag
                        with dpg.tooltip("go_to_top_button"):  # tag
                            dpg.add_text("To top [Home, when search field not focused]")

                        page_up_button = dpg.add_button(tag="page_up_button",
                                                        label=fa.ICON_ANGLE_UP,
                                                        width=gui_config.info_panel_button_w,
                                                        enabled=False)
                        dpg.bind_item_font("page_up_button", icon_font_solid)  # tag
                        dpg.bind_item_theme("page_up_button", "disablable_button_theme")  # tag
                        with dpg.tooltip("page_up_button"):  # tag
                            dpg.add_text("Page up [Page Up, when search field not focused]")

                        page_down_button = dpg.add_button(tag="page_down_button",
                                                          label=fa.ICON_ANGLE_DOWN,
                                                          width=gui_config.info_panel_button_w,
                                                          enabled=False)
                        dpg.bind_item_font("page_down_button", icon_font_solid)  # tag
                        dpg.bind_item_theme("page_down_button", "disablable_button_theme")  # tag
                        with dpg.tooltip("page_down_button"):  # tag
                            dpg.add_text("Page down [Page Down, when search field not focused]")

                        go_to_bottom_button = dpg.add_button(tag="go_to_bottom_button",
                                                             label=fa.ICON_ANGLES_DOWN,
                                                             width=gui_config.info_panel_button_w,
                                                             enabled=False)
                        dpg.bind_item_font("go_to_bottom_button", icon_font_solid)  # tag
                        dpg.bind_item_theme("go_to_bottom_button", "disablable_button_theme")  # tag
                        with dpg.tooltip("go_to_bottom_button"):  # tag
                            dpg.add_text("To bottom [End, when search field not focused]")

                        dpg.add_spacer(width=6)

                        # Scroll between search matches buttons.
                        prev_search_match_button = dpg.add_button(tag="prev_search_match_button",
                                                                  # arrow=True,
                                                                  # direction=dpg.mvDir_Up,  # The standard arrow looks too confusing, being close to other arrow buttons (in info panel content) but with different meaning.
                                                                  label=fa.ICON_CIRCLE_UP,
                                                                  width=gui_config.info_panel_button_w,
                                                                  enabled=False)
                        dpg.bind_item_font("prev_search_match_button", icon_font_solid)  # tag
                        dpg.bind_item_theme("prev_search_match_button", "disablable_button_theme")  # tag
                        with dpg.tooltip("prev_search_match_button"):  # tag
                            dpg.add_text("Previous search match [Shift+F3]")
                        next_search_match_button = dpg.add_button(tag="next_search_match_button",
                                                                  # arrow=True,
                                                                  # direction=dpg.mvDir_Down,
                                                                  label=fa.ICON_CIRCLE_DOWN,
                                                                  width=gui_config.info_panel_button_w,
                                                                  enabled=False)
                        dpg.bind_item_font("next_search_match_button", icon_font_solid)  # tag
                        dpg.bind_item_theme("next_search_match_button", "disablable_button_theme")  # tag
                        with dpg.tooltip("next_search_match_button"):  # tag
                            dpg.add_text("Next search match [F3]")

                        dpg.add_text("[no search active]", color=(140, 140, 140, 255), tag="item_information_search_controls_item_count")  # TODO: DRY duplicate definitions for labels
                        dpg.add_text("[x/x]", color=(140, 140, 140, 255), tag="item_information_search_controls_current_item", show=False)

                # Item information content
                with dpg.child_window(tag="item_information_panel",
                                      width=gui_config.info_panel_w,
                                      height=gui_config.main_window_h - gui_config.info_panel_reserved_h):
                    # with dpg.drawlist(width=gui_config.info_panel_w - 20, height=1):
                    #     dpg.draw_line((0, 0), (gui_config.info_panel_w - 21, 0), color=(140, 140, 140, 255), thickness=1)
                    # dpg.add_text("")
                    with dpg.group(horizontal=False) as info_panel_content_group:  # info container, will be refilled by `update_info_panel`
                        dpg.add_text("[Select item(s) to view information]", color=(140, 140, 140, 255))  # TODO: DRY duplicate definitions for labels
                    dpg.set_item_alias(info_panel_content_group, "info_panel_content_group")  # tag  # Set the alias separately for unified handling with the instances created later (so they show similarly in the debug registry)
                    dpg.add_spacer(width=gui_config.info_panel_w - 20, height=0, tag="info_panel_content_end_spacer")

                # Plotter help
                with dpg.child_window(tag="plotter_help_panel",
                                      width=gui_config.info_panel_w,
                                      autosize_y=True,
                                      no_scrollbar=True,
                                      no_scroll_with_mouse=True):
                    @call  # avoid polluting top-level namespace
                    def _():
                        help_heading_color = (255, 255, 255, 255)
                        help_text_color = (180, 180, 180, 255)
                        dpg.add_text("Plotter help", color=help_heading_color, tag="plotter_help_header_text")
                        with dpg.drawlist(width=gui_config.info_panel_w - 20, height=1):
                            dpg.draw_line((0, 0), (gui_config.info_panel_w - 21, 0), color=(140, 140, 140, 255), thickness=1)
                        with dpg.table(header_row=False):
                            dpg.add_table_column()
                            dpg.add_table_column()
                            dpg.add_table_column()
                            dpg.add_table_column()
                            dpg.add_table_column()
                            dpg.add_table_column()
                            with dpg.table_row():
                                # dpg.add_text("Zoom", color=help_heading_color)
                                # dpg.add_text("Wheel", color=help_dim_color)
                                dpg.add_text("Pan", color=help_heading_color)
                                dpg.add_text("Middle-drag", color=help_text_color)
                                dpg.add_text("Zoom region", color=help_heading_color)
                                dpg.add_text("Right-drag", color=help_text_color)
                                dpg.add_text("Reset zoom", color=help_heading_color)
                                dpg.add_text("Double-click", color=help_text_color)
                            with dpg.table_row():
                                dpg.add_text("Select", color=help_heading_color)
                                dpg.add_text("LMB (hold)", color=help_text_color)
                                dpg.add_text("Select more", color=help_heading_color)
                                dpg.add_text("Shift+LMB", color=help_text_color)
                                dpg.add_text("Select less", color=help_heading_color)
                                dpg.add_text("Ctrl+LMB", color=help_text_color)

            # Toolbar
            with dpg.group(horizontal=False, tag="toolbar_group"):
                def toolbar_separator(*, height=None, line=True, line_offset_y=None):
                    if height is None:
                        height = gui_config.toolbar_separator_h
                    if line_offset_y is None:
                        line_offset_y = height // 2
                    if line:
                        with dpg.drawlist(width=gui_config.toolbar_inner_w, height=height):
                            dpg.draw_line((0, line_offset_y),
                                          (99, line_offset_y),
                                          color=(140, 140, 140, 255),
                                          thickness=1)
                    else:
                        dpg.add_spacer(width=gui_config.toolbar_inner_w, height=height)
                gui_config.toolbutton_indent = (gui_config.toolbar_inner_w - gui_config.toolbutton_w) // 2  # pixels, to center the buttons

                dpg.add_text("Tools", tag="toolbar_header_text")
                toolbar_separator(height=gui_config.toolbar_separator_h // 2, line_offset_y=0)

                # File controls

                dpg.add_button(label=fa.ICON_FOLDER,
                               tag="open_file_button",
                               callback=show_open_file_dialog,
                               indent=gui_config.toolbutton_indent,
                               width=gui_config.toolbutton_w)
                dpg.bind_item_font("open_file_button", icon_font_solid)  # tag
                with dpg.tooltip("open_file_button"):  # tag
                    dpg.add_text("Open dataset [Ctrl+O]")

                toolbar_separator()

                # Zoom controls

                dpg.add_button(label=fa.ICON_HOUSE,
                               tag="zoom_reset_button",
                               callback=reset_plotter_zoom,
                               indent=gui_config.toolbutton_indent,
                               width=gui_config.toolbutton_w)
                dpg.bind_item_font("zoom_reset_button", icon_font_solid)  # tag
                with dpg.tooltip("zoom_reset_button"):  # tag
                    dpg.add_text("Reset zoom [Ctrl+Home]")

                # # TODO: Does not work currently (tested: DPG 1.11, 2.0.0), sets constraints too so zoom/pan no longer works. Wait for new DPG version?
                # def zoom_to_selection():
                #     selection_data_idxs = unbox(selection_data_idxs_box)  # item indices into `sorted_xxx`
                #     if not len(selection_data_idxs):
                #         return
                #     xmin = np.min(dataset.sorted_lowdim_data[selection_data_idxs][:, 0])
                #     xmax = np.max(dataset.sorted_lowdim_data[selection_data_idxs][:, 0])
                #     ymin = np.min(dataset.sorted_lowdim_data[selection_data_idxs][:, 1])
                #     ymax = np.max(dataset.sorted_lowdim_data[selection_data_idxs][:, 1])
                #     dpg.set_axis_limits("axis0", xmin, xmax)
                #     dpg.set_axis_limits("axis1", ymin, ymax)
                # dpg.add_button(label="Zoom selection", tag="zoom_to_selection_button",
                #                callback=zoom_to_selection)
                # with dpg.tooltip("zoom_to_selection_button"):  # tag
                #     dpg.add_text("Zoom to currently selected items")

                toolbar_separator()

                # Selection controls

                dpg.add_button(label=fa.ICON_ARROW_ROTATE_LEFT,
                               tag="selection_undo_button",
                               callback=selection_undo,
                               indent=gui_config.toolbutton_indent,
                               width=gui_config.toolbutton_w,
                               enabled=False)
                dpg.bind_item_font("selection_undo_button", icon_font_solid)  # tag
                dpg.bind_item_theme("selection_undo_button", "disablable_button_theme")  # tag
                with dpg.tooltip("selection_undo_button"):  # tag
                    dpg.add_text("Undo selection change [Ctrl+Shift+Z]")

                dpg.add_button(label=fa.ICON_ARROW_ROTATE_RIGHT,
                               tag="selection_redo_button",
                               callback=selection_redo,
                               indent=gui_config.toolbutton_indent,
                               width=gui_config.toolbutton_w,
                               enabled=False)
                dpg.bind_item_font("selection_redo_button", icon_font_solid)  # tag
                dpg.bind_item_theme("selection_redo_button", "disablable_button_theme")  # tag
                with dpg.tooltip("selection_redo_button"):  # tag
                    dpg.add_text("Redo selection change [Ctrl+Shift+Y]")

                def select_search_results():
                    """Select all datapoints matching the current search."""
                    update_selection(unbox(search_result_data_idxs_box),
                                     keyboard_state_to_selection_mode())
                dpg.add_button(label=fa.ICON_MAGNIFYING_GLASS,
                               tag="select_search_results_button",
                               callback=select_search_results,
                               indent=gui_config.toolbutton_indent,
                               width=gui_config.toolbutton_w)
                dpg.bind_item_font("select_search_results_button", icon_font_solid)  # tag
                with dpg.tooltip("select_search_results_button"):  # tag
                    dpg.add_text("Select items matched by current search [Enter, while the search field has focus]\n    with Shift: add\n    with Ctrl: subtract\n    with Ctrl+Shift: intersect")

                def select_visible_all():
                    """Select those datapoints that are currently visible in the plotter view."""
                    update_selection(get_visible_datapoints(),
                                     keyboard_state_to_selection_mode())
                dpg.add_button(label=fa.ICON_SQUARE,
                               tag="select_visible_all_button",
                               callback=select_visible_all,
                               indent=gui_config.toolbutton_indent,
                               width=gui_config.toolbutton_w)
                dpg.bind_item_font("select_visible_all_button", icon_font_regular)  # tag
                with dpg.tooltip("select_visible_all_button"):  # tag
                    dpg.add_text("Select items currently on-screen in the plotter [F9]\n    with Shift: add\n    with Ctrl: subtract\n    with Ctrl+Shift: intersect")

                toolbar_separator()
                def toggle_fullscreen():
                    dpg.toggle_viewport_fullscreen()
                    resize_gui()  # see below
                dpg.add_button(label=fa.ICON_EXPAND,
                               tag="fullscreen_button",
                               callback=toggle_fullscreen,
                               indent=gui_config.toolbutton_indent,
                               width=gui_config.toolbutton_w)
                dpg.bind_item_font("fullscreen_button", icon_font_solid)  # tag
                with dpg.tooltip("fullscreen_button"):  # tag
                    dpg.add_text("Toggle fullscreen [F11]")

                toolbar_separator()
                # We'll define and bind the callback later, when we set up the help window.
                dpg.add_button(label=fa.ICON_QUESTION,
                               tag="help_button",
                               indent=gui_config.toolbutton_indent,
                               width=gui_config.toolbutton_w)
                dpg.bind_item_font("help_button", icon_font_solid)  # tag
                with dpg.tooltip("help_button"):  # tag
                    dpg.add_text("Open the Help card [F1]")

            # Search and plotter
            with dpg.child_window(tag="search_and_plotter_panel",
                                  autosize_x=True,
                                  autosize_y=True):
                # Search bar
                #
                # The plotter can't take height=-1 if it's the first item, so for now, put the search at the top.
                with dpg.group(tag="search_group",
                               horizontal=True):
                    search_header_text = dpg.add_text("Search", color=(140, 140, 140), tag="search_header_text")  # TODO: DRY duplicate definitions for labels

                    def clear_search():
                        dpg.set_value("search_field", "")  # tag
                        update_search()
                        dpg.focus_item("search_field")  # tag
                    dpg.add_button(label=fa.ICON_X, callback=clear_search, tag="clear_search_button")
                    dpg.bind_item_font("clear_search_button", icon_font_solid)  # tag
                    with dpg.tooltip("clear_search_button"):  # tag
                        dpg.add_text("Clear the search")
                    with dpg.theme(tag="clear_search_theme"):  # tag
                        with dpg.theme_component(dpg.mvAll):
                            dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 96, 96))  # make the icon on the "clear search" button red
                    dpg.bind_item_theme("clear_search_button", "clear_search_theme")  # tag

                    dpg.add_input_text(tag="search_field",
                                       default_value="",
                                       hint="[Ctrl+F] [incremental fragment search; 'cat photo' matches 'photocatalytic'; lowercase = case-insensitive]",
                                       width=-1,
                                       callback=update_search)

                    with dpg.theme(tag="search_field_theme"):
                        with dpg.theme_component(dpg.mvInputText):
                            search_field_text_color = dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255))
                            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, (140, 140, 140))
                    dpg.bind_item_theme("search_field", "search_field_theme")  # tag

                # Plotter.
                # Configure explicitly so we are future-proof against possible defaults changes in DPG.
                with dpg.plot(tag="plot",
                              label="Semantic map [no dataset loaded]",  # TODO: DRY duplicate definitions for labels
                              width=-1,
                              height=-1,
                              no_menus=True,
                              delay_search=True,  # possible optimization, we don't need to access the actual scatter series often.
                              fit_button=dpg.mvMouseButton_Left,
                              pan_button=dpg.mvMouseButton_Middle,
                              box_select_button=dpg.mvMouseButton_Right,
                              box_select_cancel_button=dpg.mvMouseButton_Left) as theplot:  # The plot itself, with title. -1 = use whole parent container area (for widgets that support that feature).
                    # A DPG plot must have exactly one x axis, and one to three y axes. An y axis owns the data widget ("series").
                    dpg.add_plot_axis(dpg.mvXAxis, label="x", tag="axis0")
                    dpg.add_plot_axis(dpg.mvYAxis, label="y", tag="axis1")

                    # Create themes for highlighting datapoints.
                    search_results_highlight_color = None
                    selection_highlight_color = None
                    @call
                    def _():
                        # Data items hovered over by the mouse cursor.
                        color = (255, 255, 255, 40)  # actual color (this one does not need to be dynamic)
                        with dpg.theme(tag="my_selection_theme"):
                            with dpg.theme_component(dpg.mvScatterSeries):
                                dpg.add_theme_color(dpg.mvPlotCol_Line, color, category=dpg.mvThemeCat_Plots)
                                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 6, category=dpg.mvThemeCat_Plots)

                        # Data items currently selected. Data points to be filled in by selection handler.
                        color = (180, 255, 255, 40)  # dummy color; animated by `update_animations`
                        with dpg.theme(tag="my_selection_datapoints_theme"):
                            with dpg.theme_component(dpg.mvScatterSeries):
                                global selection_highlight_color
                                selection_highlight_color = dpg.add_theme_color(dpg.mvPlotCol_Line, color, category=dpg.mvThemeCat_Plots)
                                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 6, category=dpg.mvThemeCat_Plots)

                        # Data items matching the current search. Data points to be filled in by search handler.
                        color = (255, 255, 255, 32)  # dummy color; animated by `update_animations`
                        with dpg.theme(tag="my_search_results_theme"):
                            with dpg.theme_component(dpg.mvScatterSeries):
                                global search_results_highlight_color
                                search_results_highlight_color = dpg.add_theme_color(dpg.mvPlotCol_Line, color, category=dpg.mvThemeCat_Plots)
                                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 6, category=dpg.mvThemeCat_Plots)

                        _create_highlight_scatter_series()  # some utilities may access the highlight series before the app has completely booted up

                        # https://dearpygui.readthedocs.io/en/1.x/reference/dearpygui.html?highlight=colormap#dearpygui.dearpygui.sample_colormap
                        # # From section "Constants" in: https://dearpygui.readthedocs.io/en/1.x/_modules/dearpygui/dearpygui.html?highlight=colormap#
                        # mvPlotColormap_Default=internal_dpg.mvPlotColormap_Default
                        # mvPlotColormap_Deep=internal_dpg.mvPlotColormap_Deep
                        # mvPlotColormap_Dark=internal_dpg.mvPlotColormap_Dark
                        # mvPlotColormap_Pastel=internal_dpg.mvPlotColormap_Pastel
                        # mvPlotColormap_Paired=internal_dpg.mvPlotColormap_Paired
                        # mvPlotColormap_Viridis=internal_dpg.mvPlotColormap_Viridis
                        # mvPlotColormap_Plasma=internal_dpg.mvPlotColormap_Plasma
                        # mvPlotColormap_Hot=internal_dpg.mvPlotColormap_Hot
                        # mvPlotColormap_Cool=internal_dpg.mvPlotColormap_Cool
                        # mvPlotColormap_Pink=internal_dpg.mvPlotColormap_Pink
                        # mvPlotColormap_Jet=internal_dpg.mvPlotColormap_Jet
                        # mvPlotColormap_Twilight=internal_dpg.mvPlotColormap_Twilight
                        # mvPlotColormap_RdBu=internal_dpg.mvPlotColormap_RdBu
                        # mvPlotColormap_BrBG=internal_dpg.mvPlotColormap_BrBG
                        # mvPlotColormap_PiYG=internal_dpg.mvPlotColormap_PiYG
                        # mvPlotColormap_Spectral=internal_dpg.mvPlotColormap_Spectral
                        # mvPlotColormap_Greys=internal_dpg.mvPlotColormap_Greys
logger.info(f"    Done in {tim.dt:0.6g}s.")

# --------------------------------------------------------------------------------
# Background task manager

class SequentialTaskManager:
    def __init__(self, name, executor):
        """Simple manager for there-can-be-only-one (of each kind) background tasks, for GUI updates.

        Tasks can be submitted concurrently, but when a new one is submitted, all previous tasks are automatically cancelled.
        If you have several kinds of background tasks, create one `SequentialTaskManager` instance for each kind.
        You can use the same `ThreadPoolExecutor` in all of them.

        This is useful when the underlying data changes so that completing the old, running GUI update task
        no longer makes sense. Raven uses this for handling annotation tooltip and info panel updates.

        In practice this is a rather thin wrapper over `ThreadPoolExecutor`, adding the auto-cancel mechanism.

        See also our friend function `make_managed_task`.

        `name`: Name for this task manager, lowercase with underscores recommended.
                This is included in the automatically generated names for the tasks,
                to easily distinguish tasks from different managers in log messages.
        `executor`: A `ThreadPoolExecutor` that actually manages the concurrent execution.
        """
        self.name = name  # used in generated task names
        self.executor = executor
        self.tasks = {}  # task name (unique) -> (future, env)
        self.lock = threading.RLock()

    def submit(self, function, env):
        """Submit a new task.

        `function`: callable, must take one positional argument.
        `env`: `unpythonic.env.env`, passed to `function` as the only argument.

               When `submit` returns, `env` will contain two new attributes:

                   `task_name`: str, unique name of the task, for use in log messages.

                   `cancelled`: bool. This flag signals task cancellation.

                                The `function` must monitor this flag, and terminate
                                as soon as conveniently possible if the flag ever
                                becomes `True`.

                                Note that a task may become cancelled before it even starts,
                                if there are not enough resources (threads in the pool) to
                                attempt to start them all simultaneously. This happens
                                especially often if `function` needs to wait internally
                                before it can actually start doing its job.

                                Then both the task that is actively waiting, and any tasks
                                added later to the queue (waiting for a free thread in the pool)
                                may have their `cancelled` flags set to `True`.

        Returns an `unpythonic.gsym` representing the task name. Task names are unique.
        """
        with self.lock:
            self.clear()
            env.task_name = gensym(f"{self.name}_task")
            env.cancelled = False
            future = self.executor.submit(function, env)
            self.tasks[env.task_name] = (future, env)  # store a reference to `env` so we have access to the `cancelled` flag
            future.add_done_callback(self._done_callback)  # autoremove the task when it exits (we don't need its return value)
            return env.task_name

    def _find_task_by_future(self, future):
        """Internal method. Find the `task_name` for a given `future`. Return `task_name`, or `None` if not found."""
        with self.lock:
            for task_name, (f, e) in self.tasks.items():
                if f is future:
                    return task_name
            return None

    def _done_callback(self, future):
        """Internal method. Remove a completed task, by a reference to its `future` (that we get from `ThreadPoolExecutor`)."""
        with self.lock:
            task_name = self._find_task_by_future(future)
            if task_name is not None:  # not removed already? (`cancel` might have removed it)
                self.tasks.pop(task_name)

    def cancel(self, task_name, pop=True):
        """Cancel a specific task, by name.

        Usually there is no need to call this manually.

        `task_name`: `unpythonic.gsym`, the return value from `submit`.
        `pop`: bool, whether to remove the task from `self.tasks`.
               Default is `True`, which is almost always the right thing to do.
               The option is provided mainly for internal use by `clear`.
        """
        with self.lock:
            if task_name not in self.tasks:
                raise ValueError(f"SequentialTaskManager.cancel_task: instance '{self.name}': no such task '{task_name}'")
            if pop:
                future, e = self.tasks.pop(task_name)
            else:
                future, e = self.tasks[task_name]
            future.cancel()  # in case it's still queued in the executor (don't start it)
            e.cancelled = True  # in case it's running (pythonic co-operative cancellation)

    def clear(self, wait=False):
        """Cancel all tasks.

        During normal operation, usually there is no need to call this manually, but can be useful during app shutdown.

        `wait`: Whether to wait for all tasks to exit before returning.
        """
        with self.lock:
            for task_name in list(self.tasks.keys()):
                self.cancel(task_name, pop=False)
            if wait:
                while not all(future.done() for future, e in self.tasks.values()):
                    time.sleep(0.01)
            self.tasks.clear()

def make_managed_task(*, status_box, lock, entrypoint, running_poll_interval, pending_wait_duration):
    """Make a background task that makes double-sure that only one instance is running at a time.

    This works together with `SequentialTaskManager`, adding on top of it mechanisms to track the status for
    the same kind of tasks (for displaying spinners in the GUI), as well as a lock to ensure that only one call
    to `entrypoint` can be running at a time.

    `status_box`: An `unpythonic.box` that stores the task status flag. At any one time, the flag itself is one of:
                  `task_status_stopped`: No task (of this kind) in progress.
                  `task_status_pending`: A task has entered the pending state (see below).
                  `task_status_running`: A task is currently running.
    `lock`: A `threading.Lock`, to control that at most one call to `entrypoint` is running at a time.
    `entrypoint`: callable, the task itself. It must accept a kwarg `task_env`, used for sending in the task environment
                  (`unpythonic.env.env`), see below.
    `running_poll_interval`: float, when the task starts, it polls for the task status flag until there is no previous task running.
                             This parameter sets the poll interval.
    `pending_wait_duration`: float, once the polling finishes, the task enters the pending state.

                             This parameter sets how long the task waits in the pending state before it starts running.

                             This mechanism is used by e.g. the search field to wait for more keyboard input before starting to update
                             the info panel. Each keystroke in the search field submits a new managed task to the relevant
                             `SequentialTaskManager`. If more input occurs during the pending state, the manager cancels any
                             previous tasks, so that only the most recently submitted one remains.

                             Only if the pending step completes successfully (task not cancelled during `pending_wait_duration`),
                             the manager proceeds to acquire `lock`, changes the task status to `running`, and calls the task's
                             `entrypoint`.

                             Once `entrypoint` exits:

                                 If the `cancelled` flag is not set in the task environment (i.e. the task ran to completion),
                                 the manager changes the task status to `stopped`.

                                 If the `cancelled` flag is set in the task environment, the manager changes the task status
                                 to pending (because in Raven, cancellation only occurs when replaced by a new task of the same kind).

    Returns a 1-argument function, which can be submitted to a `SequentialTaskManager`.
    The function's argument is an `unpythonic.env.env`, representing the task environment.

    The environment has one mandatory attribute, `wait` (bool), that MUST be filled in by the task submitter:
        If `wait=True`, the task uses the pending state mechanism as described above.
        If `wait=False`, the pending state is skipped, and the task starts running as soon as all previous tasks
                         of the same kind have exited.

    When `entrypoint` is entered, the task environment (sent in as the kwarg `task_env`) will contain two more attributes,
    filled in by `SequentialTaskManager`:
        `task_name`: `unpythonic.gsym`, unique task name. This is the return value from `SequentialTaskManager.submit`,
                      thereby made visible for the task itself, for use in log messages.
        `cancelled`: bool, co-operative cancellation flag. The task must monitor this flag, and if it ever becomes `True`,
                     exit as soon as reasonably possible.

    Currently, the task submitter is allowed to create and use any other attributes to pass custom data into the task.
    """
    # TODO: This function is too spammy even for debug logging, needs a "detailed debug" log level.
    def _managed_task(env):
        # logger.debug(f"_managed_task: {env.task_name}: setup")
        # The task might be cancelled before this function is even entered. This happens if there are many tasks
        # (more than processing threads) in the queue, since submitting a new task to a `SequentialTaskManager` cancels all previous ones.
        if env.cancelled:
            # logger.debug(f"_managed_task: {env.task_name}: cancelled (from task queue)")
            return
        while unbox(status_box) is task_status_running:  # wait for the previous task of the same kind to finish, if already running
            time.sleep(running_poll_interval)
        if env.cancelled:
            # logger.debug(f"_managed_task: {env.task_name}: cancelled (from initial wait state)")
            return
        if env.wait:  # The pending mechanism is optional, so it can be disabled in use cases where "wait for more input" doesn't make sense.
            # logger.debug(f"_managed_task: {env.task_name}: pending")
            status_box << task_status_pending
            time.sleep(pending_wait_duration)  # [s], cancellation period
            if env.cancelled:  # Note we only cancel a task when it has been obsoleted by a newer one. So the task status is still pending.
                # logger.debug(f"_managed_task: {env.task_name}: cancelled (from pending state)")
                return
        # logger.debug(f"_managed_task: {env.task_name}: acquiring lock for this kind of task")
        with lock:
            try:
                # logger.debug(f"_managed_task: {env.task_name}: starting")
                status_box << task_status_running
                entrypoint(task_env=env)
            except Exception as exc:  # VERY IMPORTANT, to not silently swallow uncaught exceptions from background tasks
                logger.warning(f"_managed_task: {env.task_name}: exited with exception {type(exc)}: {exc}")
                traceback.print_exc()  # DEBUG
                raise
            # else:
            #     logger.debug(f"_managed_task: {env.task_name}: exited with status {'OK' if not env.cancelled else 'CANCELLED (from running state)'}")
            finally:
                if not env.cancelled:
                    status_box << task_status_stopped
                else:
                    status_box << task_status_pending
        # logger.debug(f"_managed_task: {env.task_name}: all done")
    return _managed_task

# --------------------------------------------------------------------------------
# Helpers common for the annotation tooltip and the info panel

def get_pixels_per_plotter_data_unit():
    """Estimate pixels per plotter data unit, for conversion between viewport space and data space.

    This is subtly wrong, because the plot widget includes also the space for the axis labels and such.
    But there seems to be no way to get pixels per data unit from a plot in DPG (unless using a custom series,
    which we don't).

    We use this for estimation of on-screen distances in data space. For that purpose this is good enough.

    Returns the tuple `(pixels_per_data_unit_x, pixels_per_data_unit_y)`.

    Note that if axes are not equal aspect, the x/y results may be different.
    """
    # x0, y0 = dpg.get_item_rect_min("plot")
    pixels_w, pixels_h = dpg.get_item_rect_size("plot")  # tag
    xmin, xmax = dpg.get_axis_limits("axis0")  # tag  # in data space
    ymin, ymax = dpg.get_axis_limits("axis1")  # tag  # in data space
    data_w = xmax - xmin
    data_h = ymax - ymin
    if data_w == 0 or data_h == 0:  # no data in view (can happen e.g. if the plot hasn't been rendered yet)
        return [0.0, 0.0]
    pixels_per_data_unit_x = pixels_w / data_w
    pixels_per_data_unit_y = pixels_h / data_h
    return pixels_per_data_unit_x, pixels_per_data_unit_y

def get_data_idxs_at_mouse():
    """Return a list of data points (indices to `sorted_xxx`) that are currently under the mouse cursor.

    Helper for mouse hover and mouse click handlers.
    """
    global dataset  # only for documenting intent (we don't write to it)
    if dataset is None:  # nothing plotted when no dataset loaded
        return make_blank_index_array()

    # Find `k` data points nearest to the mouse cursor.
    p = np.array(dpg.get_plot_mouse_pos())

    # FIXME: DPG BUG WORKAROUND: when not initialized yet, `get_plot_mouse_pos` returns `[0, 0]`.
    # This happens especially if the mouse cursor starts outside the plot area when the app starts.
    # For many t-SNE plots, there are likely some data points near the origin.
    first_time = (p == np.array([0.0, 0.0])).all()  # exactly zero - unlikely to happen otherwise (since we likely get asymmetric axis limits from t-SNE)
    if first_time:
        return make_blank_index_array()

    # Since the plot aspect ratio is not necessarily square, we need x/y distances separately to judge the pixel distance.
    # Hence the data space distances from the `kdtree` are not meaningful for our purposes.
    data_space_distances_, data_idxs = dataset.kdtree.query(p, k=gui_config.datapoints_at_mouse_max_neighbors)  # `ks`: item indices into `sorted_xxx`
    deltas = dataset.sorted_lowdim_data[data_idxs, :] - p  # [#k, 2].

    pixels_per_data_unit_x, pixels_per_data_unit_y = get_pixels_per_plotter_data_unit()
    if pixels_per_data_unit_x == 0.0 or pixels_per_data_unit_y == 0.0:
        return make_blank_index_array()

    # Compute pixel distance, from mouse cursor, of each matched data point.
    deltas[:, 0] *= pixels_per_data_unit_x
    deltas[:, 1] *= pixels_per_data_unit_y
    pixel_distance = (deltas[:, 0]**2 + deltas[:, 1]**2)**0.5

    # Filter for data points within the maximum allowed pixel distance.
    filt = (pixel_distance <= gui_config.selection_brush_radius_pixels)

    # logger.debug(f"get_data_idxs_at_mouse: p = {p}, data_idxs[filt] = {data_idxs[filt]}, pixel_distance = {pixel_distance[filt]}")

    return data_idxs[filt]

def get_entries_for_selection(data_idxs, *, sort_field="title", max_n=None):
    """Gather item data for visualization, sorting by cluster.

    `data_idxs`: `list`, the selection of items to include in the report. Item indices into `sorted_xxx`.
    `sort_field`: `str`, the field to sort by within each cluster. The name of one of the attributes of an entry in `sorted_entries`.
    `max_n`: `int`, how many entries can be displayed reasonably. Default `None` means no limit.

    Return value is... complicated, see `_update_annotation` and `_update_info_panel` for usage examples.
    """

    # Gather the relevant entries from the vis data.
    entries_by_cluster = collections.defaultdict(lambda: list())
    for data_idx in data_idxs:  # item indices into `sorted_xxx`
        entry = dataset.sorted_entries[data_idx]
        entries_by_cluster[entry.cluster_id].append((data_idx, entry))

    # Alphabetize by `sort_field` (e.g. `title`) within each cluster, much faster to glance at.
    for entries_in_this_cluster in entries_by_cluster.values():
        entries_in_this_cluster.sort(key=lambda e: getattr(e[1], sort_field).strip().lower())  # e: `(data_idx, entry)`

    # If `max_n` is enabled, determine how many entries we can display from each cluster to approximately match the total count.
    # But display at least one entry from each cluster.
    if max_n is not None:
        n_clusters_in_selection = len(entries_by_cluster)
        if n_clusters_in_selection > 0:
            max_entries_per_cluster = math.ceil(max_n / n_clusters_in_selection)
        else:
            max_n = None

    def format_cluster_annotation(cluster_id):
        # The metadata for the cluster.
        if cluster_id != -1:  # the outlier set doesn't have a set of common keywords computed
            if dataset.file_content.keywords_available:
                cluster_title = f"#{cluster_id}"
                cluster_keywords = f"[{', '.join(dataset.file_content.vis_keywords_by_cluster[cluster_id])}]\n"
            else:
                cluster_title = f"#{cluster_id}"
                cluster_keywords = ""
        else:
            cluster_title = "Misc"
            cluster_keywords = ""

        # The entries themselves. Leave only the first few if there are too many to display.
        entries = entries_by_cluster[cluster_id]
        if max_n is not None:
            # TODO: How to compact this in the worst case? Many clusters, with 3 data points in each -> will render 3 * n_clusters entries.
            n_extra_entries = len(entries) - max_entries_per_cluster
            more = ""
            if n_extra_entries > 0:
                if n_extra_entries < 3:  # less pedantic to avoid cutting if there are just 1 or 2 more entries than the limit would allow
                    pass
                else:  # >=3 extra entries, cut at the original limit
                    entries = entries[:max_entries_per_cluster]
                    more = f"[...{n_extra_entries} more entries in {cluster_title}...]"
        cluster_content = entries

        return cluster_title, cluster_keywords, cluster_content, more

    return entries_by_cluster, format_cluster_annotation


# --------------------------------------------------------------------------------
# Annotation tooltip for mouse hover on the plotter

annotation_render_status_box = box(task_status_stopped)
annotation_render_lock = threading.Lock()  # Render access - only one copy of the renderer may be rebuilding the annotation tooltip at any given time.
annotation_content_lock = threading.RLock()  # Content double buffering (swap). Allowing the same thread to enter multiple nested critical sections makes some checks simpler here.

annotation_build_number = 0  # Sequence number of last completed annotation tooltip build.

annotation_data_idxs = []  # Which datapoints are currently listed in the annotation tooltip, while the tooltip is open; item indices into `sorted_xxx`.

# A tooltip is really just a window with no title bar.
#   - `autosize` is important to have this window update its height when the tooltip content is rebuilt.
#   - A DPG window (as of 1.x) doesn't have an option to autosize height only, but we can set the width by using a drawlist or spacer, and wrapping text to less than that width.
with dpg.window(show=False, modal=False, no_title_bar=True, tag="annotation_tooltip_window",
                no_collapse=True,
                no_scrollbar=True,
                no_focus_on_appearing=True,
                autosize=True) as annotation_window:
    with dpg.group() as annotation_group:
        dpg.add_text("[no data]", wrap=0, color=(180, 180, 180))
    dpg.set_item_alias(annotation_group, "annotation_group")  # tag  # Set the alias separately for unified handling with the instances created later (so they show similarly in the debug registry)

# Task submitter, and plot highlighter.
@dlet(m_prev=None)
def update_mouse_hover(*, force=False, wait=True, wait_duration=0.05, env=None):
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
        data_idxs = get_data_idxs_at_mouse()  # item indices into `sorted_xxx`.
        if len(data_idxs):
            dpg.set_value("my_mouse_hover_scatter_series", [list(dataset.sorted_lowdim_data[data_idxs, 0]),  # tag
                                                            list(dataset.sorted_lowdim_data[data_idxs, 1])])
        else:
            dpg.set_value("my_mouse_hover_scatter_series", [[], []])  # tag
    if m != env.m_prev:  # Hide the annotation tooltip as soon as the mouse moves. This allows the user to move the mouse where the tooltip was, and get correct plot coordinates.
        dpg.hide_item(annotation_window)
    env.m_prev = m

    annotation_render_task = make_managed_task(status_box=annotation_render_status_box,
                                               lock=annotation_render_lock,
                                               entrypoint=_update_annotation,
                                               running_poll_interval=0.01,
                                               pending_wait_duration=wait_duration)
    annotation_task_manager.submit(annotation_render_task, envcls(wait=wait))

# Worker.
@dlet(internal_build_number=0)  # For making unique DPG tags. Incremented each time, regardless of whether completed or cancelled.
def _update_annotation(*, task_env, env=None):
    """Update the plotter annotation tooltip for the items under the mouse cursor.

    `task_env`: Handled by `update_mouse_hover`. Importantly, contains the `cancelled` flag for the task.
    """
    # TODO: This function is too spammy even for debug logging, needs a "detailed debug" log level.
    # logger.debug(f"_update_annotation: {task_env.task_name}: Annotation update task running.")

    # For "double-buffering"
    global annotation_group
    global annotation_build_number
    annotation_target_group = None  # DPG widget for building new content, will be initialized later
    new_content_swapped_in = False

    # Under some conditions no annotation should be shown
    #  - Modal window open (so the rest of the GUI should be inactive)
    #  - The mouse moved outside the plot area while the update was waiting in the queue
    if is_any_modal_window_visible() or not mouse_inside_plot_widget():
        dpg.hide_item(annotation_window)
        # logger.debug(f"_update_annotation: {task_env.task_name}: Annotation update task completed. No items under mouse, so nothing to do.")
        return

    mouse_pos = dpg.get_mouse_pos(local=False)
    data_idxs = get_data_idxs_at_mouse()  # item indices into `sorted_xxx`.

    with annotation_content_lock:
        old_mouse_hover_data_idxs_set = set(annotation_data_idxs)  # For checking if we need to resize/reposition (reduces flickering). Ordering doesn't matter, because the tooltip is always populated in the same order.
        annotation_data_idxs.clear()
        if not len(data_idxs):  # No data point(s) under mouse cursor -> hide the annotation if any, and we're done.
            dpg.hide_item(annotation_window)
            return

        # logger.debug(f"_update_annotation: {task_env.task_name}: Annotation build {env.internal_build_number} starting.")
        # annotation_t0 = time.monotonic()

        # Start rebuilding the tooltip content.
        # `with dpg.group(...):` would look clearer, but it's better to not touch the DPG container stack from a background thread.
        gc.collect()
        dpg.split_frame()
        annotation_target_group = dpg.add_group(show=False, parent="annotation_tooltip_window")  # tag

        # After this point (content target group GUI widget created), if something goes wrong, we must clean up the partially built content.
        try:
            # for highlighting
            search_string = unbox(search_string_box)
            search_result_data_idxs = unbox(search_result_data_idxs_box)
            selection_data_idxs = unbox(selection_data_idxs_box)

            # Actual content
            entries_by_cluster, formatter = get_entries_for_selection(data_idxs, max_n=gui_config.max_titles_in_tooltip)
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

            with info_panel_content_lock:
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

                        if data_idx in info_panel_entry_title_widgets:  # shown in the info panel
                            item_selection_status = item_ininfo
                            selection_mark_text = fa.ICON_CLIPBOARD_CHECK
                            selection_mark_font = icon_font_solid
                            if data_idx in selection_data_idxs:  # Usually, all items in the info panel are in the selection...
                                selection_mark_color = (120, 180, 255)  # blue
                            else:  # ...but while the info panel is updating, the old content (shown until the update completes) may have some items that are no longer included in the new selection.
                                item_selection_status = item_ininfo
                                selection_mark_color = (80, 80, 80)  # very dark gray  # (255, 180, 120)  # orange
                        else:  # not shown in the info panel
                            selection_mark_text = fa.ICON_CLIPBOARD
                            selection_mark_font = icon_font_regular
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
                            dpg.bind_item_font(mark_widget, icon_font_solid)
                        else:  # no search active
                            item_search_status = item_searchoff

                        dpg.add_text(entry.title, color=title_color, wrap=0, tag=f"cluster_{cluster_id}_item_{data_idx}_annotation_title_build{env.internal_build_number}", parent=item_group)  # "A study of stuff..."

                        if item_selection_status is item_ininfo and (not search_string or item_search_status is item_match):
                            have_jumpable_item = True

                        annotation_data_idxs.append(data_idx)

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
                dpg.bind_item_font(selection_mark_widget, icon_font_solid)
                dpg.add_text(": selected, in info panel;", color=hint_color, tag=f"annotation_help_legend_ininfo_explanation_build{env.internal_build_number}", parent=annotation_help_selection_legend_group)

                selection_mark_widget = dpg.add_text(fa.ICON_CLIPBOARD, color=(120, 180, 255), tag=f"annotation_help_legend_selected_icon_build{env.internal_build_number}", parent=annotation_help_selection_legend_group)  # blue
                dpg.bind_item_font(selection_mark_widget, icon_font_regular)
                dpg.add_text(": selected, not in info panel;", color=hint_color, tag=f"annotation_help_legend_selected_explanation_build{env.internal_build_number}", parent=annotation_help_selection_legend_group)

                selection_mark_widget = dpg.add_text(fa.ICON_CLIPBOARD, color=(80, 80, 80), tag=f"annotation_help_legend_notselected_icon_build{env.internal_build_number}", parent=annotation_help_selection_legend_group)  # very dark gray
                dpg.bind_item_font(selection_mark_widget, icon_font_regular)
                dpg.add_text(": not selected", color=hint_color, tag=f"annotation_help_legend_notselected_explanation_build{env.internal_build_number}", parent=annotation_help_selection_legend_group)

                if search_string:
                    annotation_help_search_legend_group = dpg.add_group(horizontal=True, tag=f"annotation_help_search_legend_group_build{env.internal_build_number}", parent=annotation_target_group)

                    search_mark_widget = dpg.add_text(fa.ICON_MAGNIFYING_GLASS, color=(180, 255, 180), tag=f"annotation_help_legend_match_icon_build{env.internal_build_number}", parent=annotation_help_search_legend_group)
                    dpg.bind_item_font(search_mark_widget, icon_font_solid)
                    dpg.add_text(": match;", color=hint_color, tag=f"annotation_help_legend_match_explanation_build{env.internal_build_number}", parent=annotation_help_search_legend_group)

                    search_mark_widget = dpg.add_text(fa.ICON_MAGNIFYING_GLASS, color=(80, 80, 80), tag=f"annotation_help_legend_nomatch_icon_build{env.internal_build_number}", parent=annotation_help_search_legend_group)
                    dpg.bind_item_font(search_mark_widget, icon_font_solid)
                    dpg.add_text(": no match", color=hint_color, tag=f"annotation_help_legend_nomatch_explanation_build{env.internal_build_number}", parent=annotation_help_search_legend_group)

                if have_jumpable_item:
                    annotation_help_jumpable_group = dpg.add_group(horizontal=True, tag=f"annotation_help_jumpable_group_build{env.internal_build_number}", parent=annotation_target_group)

                    dpg.add_text("[Right-click to scroll info panel to topmost", color=hint_color, tag=f"annotation_help_jumpable_explanation_left_build{env.internal_build_number}", parent=annotation_help_jumpable_group)
                    selection_mark_widget = dpg.add_text(fa.ICON_CLIPBOARD_CHECK, color=(120, 180, 255), tag=f"annotation_help_jumpable_selection_icon_build{env.internal_build_number}", parent=annotation_help_jumpable_group)
                    dpg.bind_item_font(selection_mark_widget, icon_font_solid)
                    if search_string:
                        search_mark_widget = dpg.add_text(fa.ICON_MAGNIFYING_GLASS, color=(180, 255, 180), tag=f"annotation_help_jumpable_search_icon_build{env.internal_build_number}", parent=annotation_help_jumpable_group)
                        dpg.bind_item_font(search_mark_widget, icon_font_solid)
                    dpg.add_text("item]", color=hint_color, tag=f"annotation_help_jumpable_explanation_right_build{env.internal_build_number}", parent=annotation_help_jumpable_group)
                else:
                    annotation_help_notjumpable_group = dpg.add_group(horizontal=True, tag=f"annotation_help_notjumpable_group_build{env.internal_build_number}", parent=annotation_target_group)

                    dpg.add_text("[Right-click disabled, no", color=hint_color, tag=f"annotation_help_notjumpable_explanation_left_build{env.internal_build_number}", parent=annotation_help_notjumpable_group)
                    selection_mark_widget = dpg.add_text(fa.ICON_CLIPBOARD_CHECK, color=(120, 180, 255), tag=f"annotation_help_notjumpable_selection_icon_build{env.internal_build_number}", parent=annotation_help_notjumpable_group)
                    dpg.bind_item_font(selection_mark_widget, icon_font_solid)
                    if search_string:
                        search_mark_widget = dpg.add_text(fa.ICON_MAGNIFYING_GLASS, color=(180, 255, 180), tag=f"annotation_help_notjumpable_search_icon_build{env.internal_build_number}", parent=annotation_help_notjumpable_group)
                        dpg.bind_item_font(search_mark_widget, icon_font_solid)
                    dpg.add_text("item listed]", color=hint_color, tag=f"annotation_help_notjumpable_explanation_right_build{env.internal_build_number}", parent=annotation_help_notjumpable_group)

                # Position the tooltip elegantly, trying to keep the whole tooltip within the viewport area.
                #
                # IMPORTANT: Draw the tooltip a bit off from the mouse cursor position so that the cursor won't hover over it.
                # This keeps `get_plot_mouse_pos` working, as well as improves tooltip readability since the mouse cursor doesn't cover part of it.

                def compute_tooltip_position_scalar(*, algorithm, cursor_pos, tooltip_size, viewport_size, offset=20):
                    """Compute x or y position for the annotation tooltip."""
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
                        r = clamp(cursor_pos / viewport_size)  # relative coordinate, [0, 1]
                        s = nonanalytic_smooth_transition(r, m=2.0)
                        pos = (1.0 - s) * pos1 + s * pos2

                        return pos

                # Swap the new content in ("double-buffering")
                # logger.debug(f"_update_annotation: {task_env.task_name}: Swapping in new content (old GUI widget ID {annotation_group}; new GUI widget ID {annotation_target_group}).")
                mouse_hover_set_changed = (set(annotation_data_idxs) != old_mouse_hover_data_idxs_set)
                if mouse_hover_set_changed:  # temporarily hide the window when the content changes (so that it doesn't flicker while being content-swapped and repositioned)
                    dpg.hide_item(annotation_window)
                dpg.hide_item(annotation_group)
                dpg.show_item(annotation_target_group)
                dpg.split_frame()  # wait for render
                dpg.delete_item(annotation_group)
                annotation_group = None
                dpg.set_item_alias(annotation_target_group, "annotation_group")  # tag
                annotation_group = annotation_target_group
                new_content_swapped_in = True

                # Resize/reposition the tooltip only when the set of shown items has actually changed.
                # This reduces flickering e.g. when clicking on a datapoint, only changing its selection status.
                if mouse_hover_set_changed:
                    w, h = dpg.get_item_rect_size(main_window)
                    dpg.set_item_pos(annotation_window, [w, h])  # offscreen, but not hidden -> will be rendered -> triggers the DPG autosize mechanism
                    dpg.show_item(annotation_window)

                    # Tooltip window dimensions after autosizing not available yet, so we need to wait until we can compute the final position the tooltip.
                    # TODO: Do we need a general utility to wait until a given GUI element updates?
                    wait_frames_max = 10
                    waited = 0
                    old_tooltip_size = dpg.get_item_rect_size(annotation_window)
                    while waited < wait_frames_max:
                        dpg.split_frame()  # let the autosize happen
                        waited += 1

                        tooltip_size = dpg.get_item_rect_size(annotation_window)
                        if tooltip_size != old_tooltip_size:
                            # logger.debug(f"_update_annotation: {task_env.task_name}: waited {waited} frame{'s' if waited != 1 else ''} for annotation window autosizing")
                            break
                    # else:
                    #     logger.debug(f"_update_annotation: {task_env.task_name}: timeout ({wait_frames_max} frames) when waiting for annotation window autosizing")

                    xpos = compute_tooltip_position_scalar(algorithm="snap",
                                                           cursor_pos=mouse_pos[0],
                                                           tooltip_size=tooltip_size[0],
                                                           viewport_size=w)
                    ypos = compute_tooltip_position_scalar(algorithm="smooth",
                                                           cursor_pos=mouse_pos[1],
                                                           tooltip_size=tooltip_size[1],
                                                           viewport_size=h)

                    dpg.set_item_pos(annotation_window, [xpos, ypos])
                dpg.show_item(annotation_window)  # just in case it's hidden

                # # Try to bring the tooltip to the front so it isn't covered by the dimmer.
                # # This steals keyboard focus from the search field, because DPG thinks the focused item is actually "main_container" (the top-level layout). Doesn't help to record the focused item earlier, same result.
                # originally_focused_item = dpg.get_focused_item()
                # logger.debug(f"Originally focused: item {originally_focused_item} (tag '{dpg.get_item_alias(originally_focused_item)}', type {dpg.get_item_type(originally_focused_item)})")
                # dpg.focus_item(annotation_window)
                # dpg.split_frame()
                # dpg.focus_item(originally_focused_item)

        except Exception:
            logger.debug(f"_update_annotation: {task_env.task_name}: Annotation update task raised an exception; cancelling task.")
            task_env.cancelled = True
            if not new_content_swapped_in:  # clean up: swap back the old content (if it still exists) in case the exception occurred during finalizing
                if annotation_target_group is not None:
                    dpg.hide_item(annotation_target_group)
                if annotation_group is not None:
                    dpg.show_item(annotation_group)
            raise

        finally:
            if task_env is not None and task_env.cancelled:
                # logger.debug(f"_update_annotation: {task_env.task_name}: Annotation update task cancelled.")

                # If the new content was built (partially or completely) but not swapped in, it's unused, so delete it.
                if (annotation_target_group is not None) and (not new_content_swapped_in):
                    # logger.debug(f"_update_annotation: {task_env.task_name}: Deleting partially built content.")
                    dpg.delete_item(annotation_target_group)
            else:
                # logger.debug(f"_update_annotation: {task_env.task_name}: Annotation update task completed.")

                # Publish the build ID we used while building
                annotation_build_number = env.internal_build_number

            # dt = time.monotonic() - annotation_t0
            # logger.debug(f"_update_annotation: {task_env.task_name}: Annotation build {env.internal_build_number} exiting. Rendered in {dt:0.2f}s.")
            env.internal_build_number += 1  # always increase internal build, even when cancelled, for unique IDs.

def clear_mouse_hover():
    """Hide the annotation tooltip, and clear the mouse highlight in the plotter."""
    dpg.hide_item(annotation_window)
    dpg.set_value("my_mouse_hover_scatter_series", [[], []])  # tag


# --------------------------------------------------------------------------------
# Item information panel
#
# NOTE: Fasten your seatbelt, everything related to the info panel makes up ~half of the entire program.

info_panel_render_status_box = box(task_status_stopped)
info_panel_render_lock = threading.Lock()  # Render access - only one copy of the renderer may be rebuilding the info panel at any given time.
info_panel_content_lock = threading.RLock()  # Content double buffering (swap). Allowing the same thread to enter multiple nested critical sections makes some checks simpler here.

info_panel_build_number = 0  # Sequence number of last completed info panel build, so that callbacks can refer to the GUI widgets of the entries currently in the info panel by their DPG tags.

info_panel_entry_title_widgets = {}  # `data_idx` (index in `sorted_xxx`) -> DPG ID of GUI widget for the title container group of that entry in the info panel
info_panel_widget_to_data_idx = {}  # reverse lookup: DPG ID of entry title GUI widget -> `data_idx` (index in `sorted_xxx`)
info_panel_widget_to_display_idx = {}  # reverse lookup: DPG ID of entry title GUI widget -> insertion order in `info_panel_entry_title_widgets` (how-manyth item in the info panel a given item is). Used in scroll anchoring.

info_panel_search_result_widgets = []  # DPG IDs of entry title container group widgets that match the current search, for the "scroll to next/previous match" buttons.
info_panel_search_result_widget_to_display_idx = {}  # reverse lookup: DPG ID -> index in `info_panel_search_result_widgets` (to look up how-manyth search result a given item is)

cluster_ids_in_selection = []  # cluster IDs (as in the dataset) of the clusters currently shown in the info panel. Note we always show at least one item from each cluster, so these are the same as in the whole selection.
cluster_id_to_display_idx = {}  # reverse lookup: cluster ID -> index in `cluster_ids_in_selection` (for the cluster scroll hotkeys to look up next/previous cluster shown in the info panel)

report_plaintext = box("")  # Text report of full content of info panel, in plain text (.txt) format
report_markdown = box("")  # Text report of full content of info panel, in Markdown (.md) format

# ----------------------------------------
# Content area helpers

def get_info_panel_content_area_start_pos():
    """Return `(x0, y0)`, the upper left corner of the content area of the info panel, in viewport coordinates."""
    # Item info panel starts at, in viewport coordinates:
    x0, y0 = dpg.get_item_pos("item_information_panel")  # tag
    # Its content area starts at, in viewport coordinates:
    x0_content = x0 + 8 + 3  # 8px outer padding + 3px inner padding
    y0_content = y0 + 8 + 3  # 8px outer padding + 3px inner padding
    return x0_content, y0_content

def get_info_panel_content_area_size():
    """Return `(width, height), the size of the content area of the info panel, in pixels.`"""
    _update_info_panel_height()  # HACK: at app startup, the main window thinks it has height=100, which is wrong.
    # Child windows store width/height in their config.
    config = dpg.get_item_configuration("item_information_panel")  # tag
    w = config["width"]
    h = config["height"]
    return w, h

# ----------------------------------------
# Info panel updater: task submitter.

def update_info_panel(*, wait=True, wait_duration=0.25):
    """Update the data displayed in the info panel. Public API.

    Markdown rendering may take a while, so we try to avoid triggering extra updates.

    `wait`: bool, whether to wait for a short cancellation period before actually starting the update.
            This can be useful if the GUI feature you are triggering this from expects more input in a typical case.
    `wait_duration`: float, seconds.

    **Implementation notes**:

    A call to `update_info_panel` is posted by the GUI event queue whenever the search field or the selection changes.
    To keep the GUI responsive, we want to return quickly, and run the update asynchronously.

    The convenient way to do this is to use a separate background thread. Conveniently, DPG supports GUI updates
    from arbitrary threads (as long as you are careful, as usual in concurrent programming: don't break any state
    that another thread was accessing).

    (Example: don't clear a panel that another thread is populating. If you do so, the other thread may find that
     one of its GUI items has gone missing when it tries to bind a theme to it, thus crashing the app.)

    We manage asynchronous update tasks so that:

      - Only one task can be running at a time, so that at most one thread modifies the GUI panel contents at any given time.
      - Optionally, each task waits for a short cancellation period (in a "pending" state) before actually starting.
      - Any previous tasks still within their cancellation period are cancelled when a new one is queued.

    The last two features allow the user to type more keyboard input into the search field,
    without each individual typed letter triggering the (possibly lengthy) update separately.

    Note that *running* tasks are never cancelled, so if a previous update was already in progress, it will still
    complete, and then a new update will be triggered by the latest search input.
    """
    info_panel_render_task = make_managed_task(status_box=info_panel_render_status_box,
                                               lock=info_panel_render_lock,
                                               entrypoint=_update_info_panel,
                                               running_poll_interval=0.1,
                                               pending_wait_duration=wait_duration)
    info_panel_task_manager.submit(info_panel_render_task, envcls(wait=wait))

# ----------------------------------------
# Item detectors for GUI widget search utilities

# In this section, `item` is a GUI widget's DPG tag or ID.
#
# NOTE: zero is a valid DPG ID. Hence in our filter functions, we use `None` to mean "no match", and otherwise return the item (DPG tag or ID) as-is.

def get_user_data(item):
    """Return a DPG widget's user data. Return `None` if not present."""
    if item is None:
        return None
    config = dpg.get_item_configuration(item)  # no `try`, because we want this to fail-fast (and loud) in case of bugs in our code.
    try:
        return config["user_data"]
    except KeyError:
        pass
    return None

def is_user_data_kind(value, item):
    """Check a DPG widget's user data, and return the item if the user data `kind == value`. Else return `None`.

    We always store user data as `(kind, data)`, where `kind` is str, and `data` is arbitrary (depending on `kind`).
    """
    if item is None:
        return None
    user_data = get_user_data(item)
    if user_data is not None:
        kind, data = user_data
        if kind == value:
            return item
    return None

def is_entry_title_container_group(item):  # The container has also the buttons in addition to the actual title text.
    return is_user_data_kind("entry_title_container", item)
def is_entry_title_text_item(item):  # The actual title text. Note this is actually a group widget containing text snippets, spacers, and such.
    return is_user_data_kind("entry_title_text", item)
def is_cluster_title(item):  # e.g. "#42"
    return is_user_data_kind("cluster_title", item)
def is_copy_entry_to_clipboard_button(item):
    return is_user_data_kind("copy_entry_to_clipboard_button", item)

# def is_not_entry_title_container_group(item):
#     if is_entry_title_container_group(item) is None:
#         return item
#     return None

def is_non_blank_text_item(item):
    if item is None:
        return None
    if dpg.get_item_type(item) == "mvAppItemType::mvText":
        value = dpg.get_value(item)
        if value is not None and value != "":
            return item
    return None

# def is_text_item_with_given_text(item, *, text):
#     if item is None:
#         return None
#     if dpg.get_item_type(item) == "mvAppItemType::mvText" and dpg.get_value(item) == text:
#         return item
#     return None

def is_completely_below_target_y(item, *, target_y):  # in viewport coordinates
    if item is None:
        return None
    try:
        x0, y0 = dpg.get_item_rect_min(item)
    except KeyError:  # some items don't have `rect_min` (e.g. child windows)
        x0, y0 = dpg.get_item_pos(item)

    # logger.debug(f"is_completely_below_target_y: item {item} (tag '{dpg.get_item_alias(item)}', type {dpg.get_item_type(item)}): x0, y0 = {x0}, {y0}, target_y = {target_y}, match = {y0 >= target_y}")

    if y0 >= target_y:
        return item
    return None

def is_completely_above_target_y(item, *, target_y):  # in viewport coordinates
    if item is None:
        return None
    try:
        x0, y0 = dpg.get_item_rect_min(item)
    except KeyError:  # some items don't have `rect_min` (e.g. child windows)
        x0, y0 = dpg.get_item_pos(item)

    try:
        w, h = dpg.get_item_rect_size(item)
    except KeyError:  # e.g. child window
        config = dpg.get_item_configuration(item)
        # w = config["width"]
        h = config["height"]

    item_y_last = y0 + h - 1
    if item_y_last < target_y:
        return item
    return None

def is_partially_below_target_y(item, *, target_y):  # in viewport coordinates
    if item is None:
        return None
    try:
        x0, y0 = dpg.get_item_rect_min(item)
    except KeyError:  # some items don't have `rect_min` (e.g. child windows)
        x0, y0 = dpg.get_item_pos(item)

    try:
        w, h = dpg.get_item_rect_size(item)
    except KeyError:  # e.g. child window
        config = dpg.get_item_configuration(item)
        # w = config["width"]
        h = config["height"]

    item_y_last = y0 + h - 1
    if item_y_last >= target_y:
        return item
    return None

def is_partially_above_target_y(item, *, target_y):  # in viewport coordinates
    if item is None:
        return None
    try:
        x0, y0 = dpg.get_item_rect_min(item)
    except KeyError:  # some items don't have `rect_min` (e.g. child windows)
        x0, y0 = dpg.get_item_pos(item)

    # logger.debug(f"is_partially_above_target_y: item {item} (tag '{dpg.get_item_alias(item)}', type {dpg.get_item_type(item)}): x0, y0 = {x0}, {y0}, target_y = {target_y}, match = {y0 >= target_y}")

    if y0 < target_y:
        return item
    return None

# ----------------------------------------
# GUI widget search utilities

def find_item_depth_first(root, *, accept, skip=None, _indent=0):
    """Find a specific DPG GUI widget by depth-first recursion.

    `root`: Widget to start in. DPG tag or ID.
    `accept`: 1-arg callable. The argument is a DPG tag or ID.
              Should return its input if the item should be accepted (thus terminating the search), or `None` otherwise.
    `skip`: 1-arg callable, optional. The argument is a DPG tag or ID.
            Should return its input if the item should be skipped (thus moving on), or `None` otherwise.
            This is applied first, before testing `accept` or descending (if the item is a group).
    `_indent`: For debug messages. Managed internally. Currently unused.

    Skipping is especially useful to ignore non-relevant containers (group items) as early as possible,
    without descending into them. (To make them easily detectable, use the `user_data` feature of DPG
    to store something for your detector.)
    """
    # logger.debug(f"find_item_depth_first: {' ' * _indent}{root} (tag '{dpg.get_item_alias(root)}', type {dpg.get_item_type(root)}): should skip = {(skip(root) is not None) if skip is not None else 'n/a'}, acceptable = {accept(root) is not None}, #children = {len(dpg.get_item_children(root, slot=1))}")
    if root is None:  # Should not happen during recursion, but can happen if accidentally manually called with `root=None`.
        return None
    if (skip is not None) and (skip(root) is not None):
        return None
    if (result := accept(root)) is not None:
        # logger.debug(f"find_item_depth_first: {' ' * _indent}match found")
        return result
    if dpg.get_item_type(root) == "mvAppItemType::mvGroup":
        for item in dpg.get_item_children(root, slot=1):
            if (result := find_item_depth_first(item, accept=accept, skip=skip, _indent=_indent + 2)) is not None:
                return result
    return None

def binary_search_item(items, *, accept, consider, skip=None, direction="right"):
    """Binary-search a list of DPG GUI widgets to find the first one in the list that satisfies a monotonic criterion.

    The main use case is to quickly (O(log(n))) find the first item that is currently on-screen in a scrollable window
    that has lots (thousands or more) of widgets.

    `items`: List of GUI widgets to be searched (DPG tag or ID for each).

    `accept`: 1-arg callable. The argument is a DPG tag or ID.
              Should return its input if the item is acceptable as the search result, or `None` otherwise.

              When a visited item is a group, this is automatically recursively applied depth-first to find any acceptable item at any level.

    Additionally, `items` may contain *confounders*, which are uninteresting items we don't want to accept as search results.
    This typically happens when you `items = dpg.get_item_children(some_root_item, slot=1)` - beside the items you want,
    the GUI may have separators and such.

    The definition of a confounder is controlled by the following two parameters:

    `consider`: 1-arg callable, optional. The argument is a DPG tag or ID.
                Should return its input if the item merits looking into as a possible search result, or `None` otherwise.

                This is needed when confounders are present in the data. If you know there are no confounders, use `consider=None`
                to make the search faster. Then also the tree search will be disabled; we only look at the explicitly listed `items`.

                If `consider is None`, then `skip` must also be `None`.

                When confounders are present, `consider` is used to give the binary search useful intermediate candidates
                so that the search can proceed.

                When `consider is not None`, and a visited child of `root` is a group, this is automatically recursively
                applied depth-first to find any item that could be considered, at any level.

                Technically, this is the accept condition for the non-confounder scan (thus terminating the scan, accepting the item),
                which is tested *after* the `skip` condition below.

                For example, when looking for text, `consider` may accept any non-blank text item (that was not skipped by `skip`).
                This way any blank (useless) texts as well as e.g. drawlist widgets are not considered.

    `skip`: 1-arg callable, optional. The argument is a DPG tag or ID.
            Should return its input if the item should be skipped (thus moving on), or `None` otherwise.

            This is used to avoid giving the binary search intermediate candidates that we do not want as a search result.

            Technically, this is the skip condition for the non-confounder scan (confounders should be skipped).
            Skipping an item skips also its children.

            Items that were not skipped are then checked using the `consider` condition, descending into children depth-first.

            This can be used to skip confounder items, such as drawlists if interested in text items only.

    `direction`: str. Return the item on which side of the discontinuity, like taking a one-sided limit in mathematics. One of:
        "right": Return the first item that satisfies `accept(item)`.
        "left": Return the last item that does *not* satisfy `accept(item)`.

    **IMPORTANT**:

    For binary search to work, the acceptability criterion must be monotonic. Without loss of generality, we assume that
    `items[0]` is not acceptable, `items[-1]` is acceptable, and that there is exactly one jump from unacceptable to acceptable
    somewhere in the list `items`.

    Rather than looking for a specific item, this finds the minimal `j` such that none of `root.children[<j]` are acceptable,
    whereas one of the items contained (recursively) somewhere in `root.children[j]` is acceptable.

    Essentially, we look for a jump of the step function where the item kind changes from unacceptable to acceptable.
    But we allow the list of children to have confounders sprinkled amid the items we are actually interested in.
    """
    # search_kind = "first acceptable item" if direction == "right" else "last non-acceptable item"
    # logger.debug(f"binary_search_item: starting, direction = '{direction}'; searching for {search_kind}.")

    # sanity check
    if consider is None and skip is not None:
        raise ValueError(f"When `consider is None`, the `skip` parameter must also be `None`; got skip = {skip}")

    # Trivial case: no items
    if not len(items):
        # logger.debug("binary_search_item: `items` is empty; no match.")
        return None

    if consider is not None:  # confounders present in data?
        def scan_for_non_confounder(start, *, step, jend=None):  # `step`: the only useful values are +1 (toward right) or -1 (toward left)
            # logger.debug(f"binary_search_item: non-confounder scan, start = {start}, step = {step}, jend = {jend}")
            if jend is None:
                if step > 0:
                    jend = len(items)
                else:
                    jend = 0
            j = start
            while (result := find_item_depth_first(items[j],
                                                   accept=consider,  # NOTE the accept condition - looking for any non-confounder.
                                                   skip=skip)) is None:
                j += step
                if (step > 0 and j >= jend) or (step < 0 and j < jend) or (step == 0):
                    return None, None
            # logger.debug(f"binary_search_item: non-confounder scan, final j = {j}, result = {result}")
            return j, result
        scan_for_non_confounder_toward_right = functools.partial(scan_for_non_confounder, step=+1)
        scan_for_non_confounder_toward_left = functools.partial(scan_for_non_confounder, step=-1)
    else:
        def no_scan(start, *args, **kwargs):
            return start, items[start]
        scan_for_non_confounder_toward_right = no_scan
        scan_for_non_confounder_toward_left = no_scan

    if direction == "right":
        # Trivial case: the first non-confounder item is acceptable.
        j_left, item_left = scan_for_non_confounder_toward_right(start=0)
        if (item_left is not None) and (accept(item_left) is not None):  # NOTE: `item_left` may be `None` if no non-confounder was found.
            # logger.debug(f"binary_search_item: trivial case, direction = 'right': first item {item_left} is acceptable; returning that item.")
            return item_left

        # Trivial case: no acceptable item exists.
        j_right, item_right = scan_for_non_confounder_toward_left(start=len(items) - 1)
        if j_left == j_right:
            # logger.debug("binary_search_item: trivial case, direction = 'right': search collapsed, no acceptable item; no match.")
            return None
        if (item_right is None) or (accept(item_right) is None):
            # logger.debug("binary_search_item: trivial case, direction = 'right': no acceptable item exists; no match.")
            return None
    elif direction == "left":  # Mirror-symmetric to the previous case.
        # If the last non-confounder item is not acceptable, it is the last item in the dataset that is not acceptable, so we return it.
        j_right, item_right = scan_for_non_confounder_toward_left(start=len(items) - 1)
        if (item_right is not None) and (accept(item_right) is None):
            # logger.debug(f"binary_search_item: trivial case, direction = 'left': last item {item_right} is not acceptable; returning that item.")
            return item_right

        # If all non-confounder items are acceptable (or if there are non non-confounders), no "last unacceptable item" exists.
        j_left, item_left = scan_for_non_confounder_toward_right(start=0)
        if j_left == j_right:
            # logger.debug("binary_search_item: trivial case, direction = 'left': search collapsed, all items are acceptable; no match.")
            return None
        if (item_left is None) or (accept(item_left) is not None):
            # logger.debug("binary_search_item: trivial case, direction = 'left': no non-acceptable item exists; no match.")
            return None
    else:
        raise ValueError(f"Unknown value for `direction` parameter: {direction}. Valid values are 'right' (return first acceptable item) or 'left' (return last non-acceptable item).")

    # General case
    #
    # The idea is to have the right (as opposed to left) goalpost to land on the target item.

    # Preconditions for general case (as well as invariant during iteration)
    assert accept(item_left) is None, "binary_search_item: left item should not be acceptable"
    assert accept(item_right) is not None, "binary_search_item: right item should be acceptable"

    if consider is None:  # no confounders - use a classical binary search.
        iteration = 0  # DEBUG
        while True:
            iteration += 1  # DEBUG
            # logger.debug(f"binary_search_item: classical mode, iteration {iteration}, j_left = {j_left}, j_right = {j_right}")

            jmid = (j_left + j_right) // 2
            item_mid = items[jmid]
            mid_acceptable = (accept(item_mid) is not None)
            if mid_acceptable and jmid < j_right:
                # logger.debug("binary_search_item:        moving right goalpost to mid")
                j_right = jmid
                continue
            if (not mid_acceptable) and jmid > j_left:
                # logger.debug("binary_search_item:        moving left goalpost to mid")
                j_left = jmid
                continue

            break

    else:  # with confounders
        iteration = 0  # DEBUG
        while True:
            iteration += 1  # DEBUG
            # logger.debug(f"binary_search_item: confounder mode, iteration {iteration}, j_left = {j_left}, j_right = {j_right}")

            # The classical binary search would test just at this index:
            jmid = (j_left + j_right) // 2

            # But in this case, because we have confounders, we must scan a short distance linearly in both directions to find an item that is not a confounder.
            jmid_right, item_right = scan_for_non_confounder_toward_right(start=jmid)
            mid_right_acceptable = (accept(item_right) is not None)
            jmid_left, item_left = scan_for_non_confounder_toward_left(start=jmid)
            mid_left_acceptable = (accept(item_left) is not None)

            # NOTE: To be safe, the right goalpost only goes as far as the *right-side* mid-item, and the left goalpost as far as the *left-side* mid-item.
            # Once we have a (very small) range that cannot be safely collapsed further in this manner, we stop the binary search, and perform a final linear scan in that range.
            # logger.debug(f"binary_search_item:    jmid_left = {jmid_left} (item {item_left}, acceptable: {mid_left_acceptable})")
            # logger.debug(f"binary_search_item:    jmid_right = {jmid_right} (item {item_right}, acceptable: {mid_right_acceptable})")
            if mid_right_acceptable and jmid_right < j_right:
                # logger.debug("binary_search_item:        moving right goalpost to mid-right")
                j_right = jmid_right
                continue
            if (not mid_left_acceptable) and jmid_left > j_left:
                # logger.debug("binary_search_item:        moving left goalpost to mid-left")
                j_left = jmid_left
                continue

            break

    # Scan the final interval (a few items at most) linearly to find the first acceptable one (it's usually the one at `j_right`)
    # logger.debug(f"binary_search_item: final j_left = {j_left}, j_right = {j_right}")

    # I'm tempted to check the invariant (left item is not acceptable; right is acceptable),
    # but the matching item might actually be somewhere inside `items[j_right]`, and the whole tree
    # for `items[j_left]` should be non-acceptable. No point in an extra recursive search here.

    if consider is None:
        if direction == "right":
            # logger.debug(f"binary_search_item: found first acceptable item j = {j_right}, item {items[j_right]}")
            return items[j_right]
        elif direction == "left":
            # logger.debug(f"binary_search_item: found last non-acceptable item j = {j_left}, item {items[j_left]}")
            return items[j_left]
    else:
        def consider_and_accept(item):
            """Check whether `item` passes both `consider` and `accept` conditions."""
            if (result := consider(item)) is not None:
                # It is possible that `result is not item`, in case the actual non-confounder `result` was found somewhere inside the original `item`.
                return accept(result)
            return None

        # logger.debug("binary_search_item: final linear scan for result in the detected range")
        if direction == "right":
            for j in range(j_left, j_right + 1):
                item = items[j]
                # logger.debug(f"binary_search_item: final linear scan: index j = {j}, item {item}")
                if (result := find_item_depth_first(item,
                                                    accept=consider_and_accept,  # NOTE the accept condition - looking for an item satisfying the actual search criterion (but is also a non-confounder).
                                                    skip=skip)) is not None:
                    # logger.debug(f"binary_search_item: final result: found first acceptable item {result}, returning that item.")
                    return result
        elif direction == "left":
            for j in range(j_right, j_left - 1, -1):
                item = items[j]
                # logger.debug(f"binary_search_item: final linear scan: index j = {j}, item {item}")
                if find_item_depth_first(item,
                                         accept=consider_and_accept,  # NOTE the accept condition - looking for an item satisfying the actual search criterion (but is also a non-confounder).
                                         skip=skip) is None:
                    # logger.debug(f"binary_search_item: final result: found last non-acceptable item {item}, returning that item.")
                    return item
        # the "else" case was checked at the beginning

    raise RuntimeError("binary_search_item: did not find anything; this should not happen. Maybe preconditions not satisfied, or `accept` function not monotonic?")

# ----------------------------------------
# Programmatic control of scroll position

def info_panel_find_next_or_prev_item(items, *, _next=True, kluge=True, extra_y_offset=0):
    """Find the next/previous GUI widget in `items`, in relation to the current position of the top of the info panel content area.

    Note `items` must contain only valid items, no confounders. This allows us to use a faster classical binary search.

    `items` is parameterized so that this can be used both for the full set of items shown in the info panel
    (`items=list(info_panel_entry_title_widgets.values())`) as well as for search results only (`items=info_panel_search_result_widgets`).

    `_next`: bool, If `True`, find the next item, else find the previous item.
    `kluge`: bool. If `True`, reject any item too near the top of the content area, to look for the really next/previous one,
             not the one currently at the top. "Too near" is the height of one line of text.
    `extra_y_offset`: int. This is useful e.g. for checking for first item *out of view* below the bottom of the content area
                      (by setting this offset to the content area height).

    Returns the DPG ID or tag for the item, or `None` if no such item exists.
    """
    if not len(items):
        return None

    _, y0_content = get_info_panel_content_area_start_pos()  # The "current match" is positioned at the top of the content area.
    if kluge:  # In the position check, optionally reject the item too near the top of the content area (to look for the next/previous one, not the one currently at the top).
        kluge = (+1 if _next else -1) * gui_config.font_size  # Pixels. We arbitrarily use one line of text as a guideline.
    else:
        kluge = 0
    def is_completely_below_top_of_content_area(item):
        if is_completely_below_target_y(item, target_y=y0_content + kluge + extra_y_offset) is not None:
            return item
        return None
    # logger.debug(f"info_panel_find_next_or_prev_item: frame {dpg.get_frame_count()}: searching (_next = {_next}, kluge = {kluge}, extra_y_offset = {extra_y_offset}).")
    return binary_search_item(items=items,
                              accept=is_completely_below_top_of_content_area,
                              consider=None,
                              skip=None,
                              direction=("right" if _next else "left"))

def scroll_info_panel_to_position(target_y_scroll):
    """Scroll the info panel to given position.

    Animated if `gui_config.smooth_scrolling` is `True`. This starts a new info panel scroll animation,
    replacing the existing animation, if any.

    `target_y_scroll`: int; pixels; final position of the scroll animation, in coordinates of *the item info panel*
                       (not viewport coordinates).

                       The first possible position is 0. To scroll to the end, use the special value `None`.

                       The value is clamped to the valid range, which is determined automatically.

                       If you have the handle to an item near the end, see the source code of `scroll_info_panel_to_item`
                       for how to convert viewport coordinates into coordinates the scrollbar understands.

    It does not matter if a scroll animation is already running; the animation engine for the scroll animation
    adapts to target position changes on the fly.

    Returns the final target scroll position after conversion of possible `None` to an actual value, and then clamping.
    """
    # Convert `None` and clamp.
    min_y_scroll = 0
    max_y_scroll = dpg.get_y_scroll_max("item_information_panel")  # tag
    if target_y_scroll is None:
        target_y_scroll = max_y_scroll
    target_y_scroll = clamp(target_y_scroll, min_y_scroll, max_y_scroll)

    # Dispatch the animation.
    #
    # Note the scroll animation is rate-based and depends only on the *current* and final positions.
    #
    # This internally handles conflicts: if a scroll animation is already running on the same GUI element,
    # it is just updated instead of creating a new one.
    global scroll_animation
    with scroll_animation_lock:
        with SmoothScrolling.class_lock:
            animator.add(SmoothScrolling(target_child_window="item_information_panel",
                                         target_y_scroll=target_y_scroll,
                                         smooth=gui_config.smooth_scrolling,
                                         smooth_step=gui_config.smooth_scrolling_step_parameter,
                                         flasher=info_panel_scroll_end_flasher,
                                         finish_callback=clear_global_scroll_animation_reference))
            scroll_animation = SmoothScrolling.instances["item_information_panel"]  # get the reified instance

    return target_y_scroll

def scroll_info_panel_to_item(item):
    """Dispatch an info panel scroll animation such that, at the new final position, `item` is at the start of the visible content area.

    `item`: DPG ID or tag of the target GUI widget.

    Returns the new final scroll position.
    """
    # Old scroll position. The scroll position is in coordinates of *the item info panel* (not viewport coordinates).
    y_scroll = dpg.get_y_scroll("item_information_panel")  # tag
    # Start position of content area, in viewport coordinates:
    _, y0_content = get_info_panel_content_area_start_pos()
    # Position of desired item, in viewport coordinates, with the scrollbar at its current position.
    # Can be out of range or even negative, if the item is out of view!
    x1, y1 = dpg.get_item_rect_min(item)  # TODO: Handle the error case where `item` does not exist
    # The scroll position that brings y1 to the start of the content area, in item info panel coordinates, is:
    new_y_scroll = max(0, y_scroll + (y1 - y0_content))
    return scroll_info_panel_to_position(new_y_scroll)

# ----------------------------------------
# Info panel navigation controls (home/end/pageup/pagedown)

def update_info_panel_navigation_controls():
    """Enable/disable the info panel's top/bottom/pageup/pagedown buttons."""
    current_y_scroll = dpg.get_y_scroll("item_information_panel")  # tag
    max_y_scroll = dpg.get_y_scroll_max("item_information_panel")  # tag
    if max_y_scroll == 0:  # less than one screenful of data?
        dpg.disable_item(go_to_top_button)
        dpg.disable_item(page_up_button)
        dpg.disable_item(go_to_bottom_button)
        dpg.disable_item(page_down_button)
    else:
        if current_y_scroll == 0:
            dpg.disable_item(go_to_top_button)
            dpg.disable_item(page_up_button)
        else:
            dpg.enable_item(go_to_top_button)
            dpg.enable_item(page_up_button)
        if current_y_scroll == max_y_scroll:
            dpg.disable_item(go_to_bottom_button)
            dpg.disable_item(page_down_button)
        else:
            dpg.enable_item(go_to_bottom_button)
            dpg.enable_item(page_down_button)

def go_to_top():
    """Scroll the info panel to the top."""
    scroll_info_panel_to_position(0)

def go_to_bottom():
    """Scroll the info panel to the bottom."""
    scroll_info_panel_to_position(None)  # scroll to end

def go_page_up():
    """Scroll the info panel up by one page."""
    current_y_scroll = dpg.get_y_scroll("item_information_panel")  # tag
    w_info, h_info = dpg.get_item_rect_size("item_information_panel")  # tag
    new_y_scroll = current_y_scroll - 0.7 * h_info
    scroll_info_panel_to_position(new_y_scroll)

def go_page_down():
    """Scroll the info panel down by one page."""
    current_y_scroll = dpg.get_y_scroll("item_information_panel")  # tag
    w_info, h_info = dpg.get_item_rect_size("item_information_panel")  # tag
    new_y_scroll = current_y_scroll + 0.7 * h_info
    scroll_info_panel_to_position(new_y_scroll)

dpg.set_item_callback(go_to_top_button, go_to_top)
dpg.set_item_callback(go_to_bottom_button, go_to_bottom)
dpg.set_item_callback(page_up_button, go_page_up)
dpg.set_item_callback(page_down_button, go_page_down)

# ----------------------------------------
# Other global info panel GUI controls

def copy_report_to_clipboard():
    """Copy all current content of info panel to OS clipboard. Public API.

    By default, copy in plain text format.

    If the Shift key is held down when this is called, copy as Markdown instead.
    """
    shift_pressed = dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)
    _copy_report_to_clipboard(report_format=("md" if shift_pressed else "txt"))

dpg.set_item_callback(copy_report_button, copy_report_to_clipboard)

def _copy_report_to_clipboard(*, report_format):
    """Copy all current content of info panel to OS clipboard. Implementation.

    `report_format`: str, one of "txt", "md".
    """
    if report_format not in ("txt", "md"):
        raise ValueError(f"Unknown report format '{report_format}'; expected one of 'txt', 'md'.")

    if report_format == "txt":
        report_text = unbox(report_plaintext)
    else:
        report_text = unbox(report_markdown)

    dpg.set_clipboard_text(report_text)

    # Acknowledge the action in the GUI.
    animator.add(ButtonFlash(message=f"Copied to clipboard! ({'plain text' if report_format == 'txt' else 'Markdown'})",
                             target_button=copy_report_button,
                             target_tooltip=copy_report_tooltip,
                             target_text=copy_report_tooltip_text,
                             duration=gui_config.acknowledgment_duration))

def copy_current_entry_to_clipboard():
    """Copy the authors, year and title of the current item to the clipboard.

    The current item is the topmost item visible in the info panel.

    Hotkey handler.
    """
    with info_panel_content_lock:  # lock here so we are guaranteed to process the same item throughout
        item = _get_current_info_panel_item()
        if item is None:
            logger.debug("copy_current_entry_to_clipboard: No current item (info panel empty?)")
            return
        _copy_entry_to_clipboard(item)

def _copy_entry_to_clipboard(item):
    """Copy the authors, year and title of the given item to the clipboard.

    `item`: DPG ID or tag of the entry title container group.

            We take this instead of a raw `entry` because this needs access
            to the GUI widgets to show the acknowledgment animation.

    Implementation.
    """
    with info_panel_content_lock:
        data_idx = info_panel_widget_to_data_idx[item]
        entry = dataset.sorted_entries[data_idx]

        button = find_item_depth_first(item, accept=is_copy_entry_to_clipboard_button)
        user_data = get_user_data(button)
        kind_, data = user_data
        tooltip, tooltip_text = data

    dpg.set_clipboard_text(f"{entry.author} ({entry.year}): {entry.title}")

    # Acknowledge the action in the GUI.
    animator.add(ButtonFlash(message="Copied to clipboard!",
                             target_button=button,
                             target_tooltip=tooltip,
                             target_text=tooltip_text,
                             duration=gui_config.acknowledgment_duration))

def search_or_select_current_entry():
    """Search for the current item in the plotter, or change the selection.

    The current item is the topmost item visible in the info panel.

    Hotkey handler.
    """
    with info_panel_content_lock:  # probably faster to acquire just once, instead of separate here and inside `_get_current_info_panel_item`
        item = _get_current_info_panel_item()
        if item is None:
            logger.debug("search_or_select_current_entry: No current item (info panel empty?)")
            return
        data_idx = info_panel_widget_to_data_idx[item]
    entry = dataset.sorted_entries[data_idx]
    _search_or_select_entry(entry)

def _search_or_select_entry(entry):
    """Search for the given item in the plotter, or change the selection.

    Implementation.

    `entry`: One of the entries in `dataset.sorted_entries`.

    Alternative modes, instead of searching:
        no modifier key: search for `entry` in the plotter (or clear the search if already searching for `entry`)
        with Shift: set selection to `entry` only
        with Ctrl: remove `entry` from selection (to allow the user to easily clean up stray data points from the info panel)

    The alternative modes trigger an info panel update if the selection changes.
    """
    shift_pressed = dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)
    ctrl_pressed = dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)

    if shift_pressed:
        update_selection([entry.data_idx],  # index to `sorted_xxx`
                         mode="replace",
                         wait=False)
    elif ctrl_pressed:
        update_selection([entry.data_idx],  # index to `sorted_xxx`
                         mode="subtract",
                         wait=False)
    else:
        # Exclude stopwords from search ("a", "an", "the", "for", ...).
        #
        # This makes the job of the Markdown renderer easier since it doesn't have to highlight many one to three letter sequences.
        # This also matches "Methanol" instead of the "an" inside it (note we match case-insensitive fragments first).
        # As a bonus, this is pretty much what a human would type into the search field when looking for a specific title.
        filtered_title = " ".join(word for word in entry.title.strip().split() if word.lower() not in stopwords)
        # Toggle: if already searching for this item, clear the search.
        if dpg.get_value("search_field") != filtered_title:  # tag
            dpg.set_value("search_field", filtered_title)  # tag
        else:
            dpg.set_value("search_field", "")  # tag
        update_search(wait=False)

# ----------------------------------------
# Fragment search integration

@dlet(prev_y_scrolls={})
def _info_panel_scroll_position_changed(*, site_tag=None, reset=False, env):
    """Return whether the info panel scrollbar position has changed since the last time this function was called.

    `site_tag`: Optional, any hashable; each unique tag stores state changes separately (i.e. whether state has changed since last queried for that `site_tag`).

    `reset`: Optional; if `True`, reset the status and store the current position. Useful e.g. when the info panel content changes.

    We need this polling HACK because there is no way to wire a callback to a child window scroll position change.
    We use this information to determine whether the current search result indicator ("[x/x]") needs to be updated
    (it takes some computation, so we skip the update whenever not needed; see `update_current_search_result_status`).
    """
    if reset or site_tag not in env.prev_y_scrolls:
        env.prev_y_scrolls[site_tag] = None
    if reset:
        return False  # not changed since reset (that happened just now)
    y_scroll = dpg.get_y_scroll("item_information_panel")  # tag
    result = (env.prev_y_scrolls[site_tag] is None or y_scroll != env.prev_y_scrolls[site_tag])
    env.prev_y_scrolls[site_tag] = y_scroll
    return result

def update_current_search_result_status():
    """Update the [x/x] indicator in the info panel, for how-manyth search result we are looking at.

    Also highlight the current item, to show which item any global hotkeys affecting a single item apply to.

    This runs every frame, so the implementation is as minimal as possible, and exits as early as possible.
    """
    if not _info_panel_scroll_position_changed():
        return

    # Avoid race condition: The info panel renderer might swap the info panel content at any moment (if an ongoing render happens to complete).
    # Use the non-blocking mode (just try to acquire the lock once, don't wait), because this function runs once per frame.
    # It doesn't matter if this update misses a few frames while the content is being swapped, as long as it doesn't block the GUI thread while that happens.
    if not info_panel_content_lock.acquire(blocking=False):
        # Technically, we should reset the scroll position tracking here, so our next update won't be rejected.
        # But the info panel already does that after it has swapped in the new content.
        return
    try:  # ok, got the lock
        # ----------------------------------------
        # Update current item, for highlighting.

        # This needs `info_panel_content_lock` for the item search, and should update only when the scroll position has changed (like we do), so we do it here, although it has nothing to do with search results.  (TODO: refactor?)
        # TODO: FIX BUG: Sometimes the highlight doesn't take right after the info panel updates (click on data to select). Current item not updated immediately. Figure out the race condition.
        update_current_item_info()

        # ----------------------------------------
        # Update the search result indicator

        if not len(info_panel_search_result_widgets):
            dpg.hide_item("item_information_search_controls_current_item")  # tag
            return

        # logger.debug(f"update_current_search_result_status: frame {dpg.get_frame_count()}: updating")

        # Find the topmost search result below the top of the content area.
        search_result_item = info_panel_find_next_or_prev_item(items=info_panel_search_result_widgets, kluge=False)
        if search_result_item is None:  # all search results are above the visible area (scroll position is past the last result)
            dpg.hide_item("item_information_search_controls_current_item")  # tag
            dpg.enable_item("prev_search_match_button")  # tag
            return
        search_result_display_idx = info_panel_search_result_widget_to_display_idx[search_result_item]  # how-manyth search result it is (in 0-based indexing)

        # Update the buttons too while at it, so that they enable/disable appropriately also when the info panel is scrolled (no matter how)
        if search_result_display_idx == 0:  # first result
            dpg.disable_item("prev_search_match_button")  # tag
        else:
            dpg.enable_item("prev_search_match_button")  # tag

        if search_result_display_idx == len(info_panel_search_result_widgets) - 1:  # last result
            dpg.disable_item("next_search_match_button")  # tag
        else:
            dpg.enable_item("next_search_match_button")  # tag

        # Is the search result on screen?
        x0_search_result_item, y0_search_result_item = dpg.get_item_rect_min(search_result_item)
        _, y0_content = get_info_panel_content_area_start_pos()
        _, h_content = get_info_panel_content_area_size()
        # 8px outer padding + 3px inner padding
        if y0_search_result_item >= y0_content + h_content - 8 - 3:  # No, it is below the visible area (either scroll position is before the first result, or there is at least a screenful of non-matching items in between).
            dpg.hide_item("item_information_search_controls_current_item")  # tag
            dpg.enable_item("next_search_match_button")  # tag  # just in case the above button checks disabled it (there might be just one result)
            return
        # Yes, it is on screen.
        dpg.set_value("item_information_search_controls_current_item", f"[{1 + search_result_display_idx}/{len(info_panel_search_result_widgets)}]")  # tag  # Show human-readable 1-based index.
        dpg.show_item("item_information_search_controls_current_item")  # tag
    finally:
        info_panel_content_lock.release()

# `update_current_search_result_status` also does this when the scroll position changes, but slightly differently. It also needs the next/prev match info anyway.
def update_next_prev_search_result_buttons():
    """Enable/disable the next/previous search result buttons in the info panel.

    Called at the end of an info panel update.
    """
    with info_panel_content_lock:  # public API, could be called from anywhere, so let's be careful.
        if not len(info_panel_search_result_widgets):  # no items matching the search shown in the info panel?
            dpg.disable_item("next_search_match_button")  # tag
            dpg.disable_item("prev_search_match_button")  # tag
            return
        # we have at least one search result shown in the info panel
        next_match = info_panel_find_next_or_prev_item(items=info_panel_search_result_widgets)
        prev_match = info_panel_find_next_or_prev_item(items=info_panel_search_result_widgets, _next=False)
        if next_match is not None:
            dpg.enable_item("next_search_match_button")  # tag
        else:
            dpg.disable_item("next_search_match_button")  # tag
        if prev_match is not None:
            dpg.enable_item("prev_search_match_button")  # tag
        else:
            dpg.disable_item("prev_search_match_button")  # tag

def scroll_info_panel_to_next_search_match():
    """Scroll the info panel to the next item matching the current search."""
    # TODO: Fix the race condition. If this button is hammered, the next update may start before the previous one finishes rendering, causing the item search to error out with a `RuntimeError`.
    # The problem is, we have no way of knowing when DPG has finished updating all the item coordinates (they are in viewport coordinates,
    # so they change when the scrollbar position changes). Waiting for one frame (`dpg.split_frame()`) doesn't always help.
    #
    # To trigger the error:
    #   - Select all (~10k datapoints is fine).
    #   - Type something very common into the search field, to have lots of search matches.
    #   - Hammer the "next search result" button relentlessly until the error pops in the console.
    #
    # For now we just silence the error, because all it does is to miss that one button click from the relentless hammering.
    #
    # Note `update_current_search_result_status`, at the next frame rendered, will take care of updating the search navigation buttons,
    # so we don't need to do that here. It would have the same race condition issue - also the button updater calls the item search.
    try:
        with info_panel_content_lock:
            if (next_match := info_panel_find_next_or_prev_item(items=info_panel_search_result_widgets)) is not None:
                scroll_info_panel_to_item(next_match)
    except RuntimeError:
        pass

def scroll_info_panel_to_prev_search_match():
    """Scroll the info panel to the previous item matching the current search."""
    try:
        with info_panel_content_lock:
            if (prev_match := info_panel_find_next_or_prev_item(items=info_panel_search_result_widgets, _next=False)) is not None:
                scroll_info_panel_to_item(prev_match)
    except RuntimeError:
        pass

dpg.set_item_callback(next_search_match_button, scroll_info_panel_to_next_search_match)
dpg.set_item_callback(prev_search_match_button, scroll_info_panel_to_prev_search_match)

# ----------------------------------------
# Cluster-related controls

def _get_current_info_panel_item():
    """Return the DPG ID/tag of the current item.

    The current item is defined as the topmost item fully visible in the info panel.
    """
    with info_panel_content_lock:
        # TODO: Performance: `update_current_search_result_status` may need to call us per-frame while scrolling. Maybe store the list too (in `_update_info_panel`) so we don't need to reconstruct it every time.
        return info_panel_find_next_or_prev_item(items=list(info_panel_entry_title_widgets.values()), kluge=False)

def _get_cluster_of_current_info_panel_item():
    """Return the cluster ID of the current item.

    The current item is defined as the topmost item fully visible in the info panel.
    """
    with info_panel_content_lock:
        current_item = _get_current_info_panel_item()
        if current_item is not None:
            data_idx = info_panel_widget_to_data_idx[current_item]  # item index to `sorted_xxx`
            entry = dataset.sorted_entries[data_idx]
            cluster_id = entry.cluster_id
            return cluster_id
        logger.debug("_get_cluster_of_current_info_panel_item: No current item (info panel empty?)")
        return None

def _scroll_info_panel_to_cluster_by_id(cluster_id):
    """Scroll info panel to given cluster, by cluster ID (as in the dataset).

    The cluster must be one of those currently shown in the info panel (see `cluster_ids_in_selection`).
    """
    if cluster_id is None:
        return
    scroll_info_panel_to_item(f"cluster_{cluster_id}_title_build{info_panel_build_number}")  # tag  # see `_update_info_panel`

def _get_cluster_display_idx_of_current_info_panel_item():
    """Return the index in `cluster_ids_in_selection` of the cluster the current item belongs to.

    The current item is defined as the topmost item fully visible in the info panel.

    The index (0-based) tells how-manyth cluster visible in the info panel it is.
    This is useful for getting the next/previous displayed cluster.

    Returns the index (int), or `None` on failure.
    """
    with info_panel_content_lock:
        cluster_id = _get_cluster_of_current_info_panel_item()
        try:
            display_idx = cluster_id_to_display_idx[cluster_id]  # index in `cluster_ids_in_selection`
        except KeyError:
            logger.debug(f"_get_cluster_display_idx_of_current_info_panel_item: Cluster #{cluster_id} not found in `cluster_id_to_display_idx` (maybe this cluster is currently not shown in info panel?)")
            return None
    return display_idx

def _scroll_info_panel_to_cluster_by_display_idx(display_idx):
    """Scroll the info panel to given cluster, by sequential numbering in info panel.

    NOTE: sequential numbering in info panel, not cluster ID. For the corresponding cluster IDs,
    see `cluster_ids_in_selection`.

    Implementation.
    """
    with info_panel_content_lock:
        if (display_idx is not None) and (display_idx >= 0) and (display_idx <= len(cluster_ids_in_selection) - 1):
            cluster_id = cluster_ids_in_selection[display_idx]
            _scroll_info_panel_to_cluster_by_id(cluster_id)

def scroll_info_panel_to_next_cluster():
    """Scroll the info panel to the next cluster, starting from the cluster of the current item.

    The current item is defined as the topmost item fully visible in the info panel.

    Hotkey handler.
    """
    with info_panel_content_lock:
        display_idx = _get_cluster_display_idx_of_current_info_panel_item()
        if display_idx is not None:
            _scroll_info_panel_to_cluster_by_display_idx(display_idx + 1)

def scroll_info_panel_to_prev_cluster():
    """Scroll the info panel to the previous cluster, starting from the cluster of the current item.

    The current item is defined as the topmost item fully visible in the info panel.

    Hotkey handler.
    """
    with info_panel_content_lock:
        display_idx = _get_cluster_display_idx_of_current_info_panel_item()
        if display_idx is not None:
            _scroll_info_panel_to_cluster_by_display_idx(display_idx - 1)

def scroll_info_panel_to_top_of_current_cluster():
    """Scroll the info panel to the top of the cluster of the current item.

    The current item is defined as the topmost item fully visible in the info panel.

    Hotkey handler.
    """
    with info_panel_content_lock:
        cluster_id = _get_cluster_of_current_info_panel_item()
        _scroll_info_panel_to_cluster_by_id(cluster_id)

def select_cluster_by_id(cluster_id):
    """Select all data in cluster `cluster_id`.

    Shift (add), Ctrl (subtract), Ctrl+Shift (intersect) modes available.

    Triggers an info panel update if the selection changes.
    """
    data_idxs = [data_idx for data_idx, entry in enumerate(dataset.sorted_entries) if entry.cluster_id == cluster_id]  # indices to `sorted_xxx`
    update_selection(data_idxs,
                     keyboard_state_to_selection_mode(),
                     wait=False)

def select_current_cluster():
    """Select all data in the same cluster as the current item.

    The current item is defined as the topmost item fully visible in the info panel.

    Shift (add), Ctrl (subtract), Ctrl+Shift (intersect) modes available.

    Triggers an info panel update if the selection changes.

    Hotkey handler.
    """
    with info_panel_content_lock:  # probably faster to acquire just once, instead of separate here and inside `_get_current_info_panel_item`
        item = _get_current_info_panel_item()
        if item is None:
            logger.debug("search_current_cluster: No current item (info panel empty?)")
            return
        data_idx = info_panel_widget_to_data_idx[item]
    entry = dataset.sorted_entries[data_idx]
    select_cluster_by_id(entry.cluster_id)

# ----------------------------------------
# Info panel updater: worker.

@dlet(scroll_anchor_data={},  # stripped tag -> y_diff, where stripped tag is without the "_buildX" suffix.
      internal_build_number=0)  # For making unique DPG tags. Incremented each time, regardless of whether completed or cancelled.
def _update_info_panel(*, task_env=None, env=None):
    """Update the data displayed in the info panel. Internal worker.

    Performs the actual update.

    `task_env`: Handled by `update_info_panel`. Importantly, contains the `cancelled` flag for the task.
    `env`: Internal, handled by `@dlet`. Stores local state between calls (for scroll position management).
    """
    # NOTE: In this function ONLY, we don't need to acquire `info_panel_content_lock` to guard against sudden content swaps,
    # because this function is the only one that does those swaps, and this is only ever entered with `info_panel_render_lock` held.

    # For stopping the scroll animation, if running when we perform the content swap.
    global scroll_animation

    # For scroll anchoring.
    global selection_changed
    global selection_anchor_data_idxs_set

    # For "double-buffering"
    global info_panel_content_group
    global info_panel_build_number
    info_panel_content_target = None  # DPG widget for building new content, will be initialized later
    new_content_swapped_in = False

    logger.debug(f"_update_info_panel: {task_env.task_name}: Info panel update task running.")
    logger.debug(f"_update_info_panel: {task_env.task_name}: Info panel build {env.internal_build_number} starting.")
    info_panel_t0 = time.monotonic()

    # --------------------------------------------------------------------------------
    # Prepare search result highlighting.

    selection_data_idxs = unbox(selection_data_idxs_box)  # item indices into `sorted_xxx`
    search_result_data_idxs = unbox(search_result_data_idxs_box)  # for deciding item colors (when a search active, dim non-matching items)
    search_string = unbox(search_string_box)
    if search_string:
        # Use the same approach as SillyTavern-Timelines. See `highlightTextSearchMatches`
        # in https://github.com/SillyTavern/SillyTavern-Timelines/blob/master/index.js
        #
        # We match the fragments from longest to shortest. This prefers
        # the longest match when the fragments have common substrings.
        # For example, "laser las".
        #
        # TODO: Currently we sort here, although it should arguably be the job of `search_string_to_fragments`.
        # However, doing it this way is faster, because that runs whenever each individual letter is typed
        # into the search field, whereas this runs only when the info panel updates.
        #
        # Also we must match all fragments simultaneously, to avoid e.g. "col" matching
        # the "<font color=...>" inserted by this highlighter when it first highlights "col".
        case_sensitive_fragments, case_insensitive_fragments = search_string_to_fragments(search_string)
        case_sensitive_fragments = list(sorted(case_sensitive_fragments, key=lambda x: -len(x)))  # longest to shortest
        case_insensitive_fragments = list(sorted(case_insensitive_fragments, key=lambda x: -len(x)))  # longest to shortest

        def search_fragment_to_highlight_regex_fragment(s):
            """Make a search fragment usable in a regex for search highlighting."""
            # Escape regex special characters.  TODO: ^, $, others?
            s = s.replace("(", r"\(")
            s = s.replace(")", r"\)")
            s = s.replace("[", r"\[")
            s = s.replace("]", r"\]")
            s = s.replace("{", r"\{")
            s = s.replace("}", r"\}")
            s = s.replace(".", r"\.")
            # Look also for superscript and subscript variants of numbers.
            # We can't do this for letters, because there are simply too many letters in each item title. :)
            for digit in "0123456789":
                s = s.replace(digit, f"({digit}|{config.regular_to_subscript_numbers[digit]}|{config.regular_to_superscript_numbers[digit]})")
            return s
        case_sensitive_fragments = [search_fragment_to_highlight_regex_fragment(x) for x in case_sensitive_fragments]
        case_insensitive_fragments = [search_fragment_to_highlight_regex_fragment(x) for x in case_insensitive_fragments]

        if case_sensitive_fragments:
            regex_case_sensitive = re.compile(f"({'|'.join(case_sensitive_fragments)})")
        if case_insensitive_fragments:
            regex_case_insensitive = re.compile(f"({'|'.join(case_insensitive_fragments)})", re.IGNORECASE)

    # --------------------------------------------------------------------------------
    # Preserve scroll position when the search terms change or the selection changes,
    # but at least one of the items on-screen before the update remains in the info panel
    # after the update.
    #
    # There is a ship-of-Theseus problem here: the info panel is completely repopulated every time it is updated,
    # because this is the easiest way to update it in a bug-free manner. So "the same" scroll position does not even exist.
    #
    # To anchor the scroll position, we look for the first entry title container group at least partially in view.
    # We then look for the same entry in the reconstructed panel content, and compute the new scroll position from it,
    # so that its y coordinate on screen remains the same (if possible) across the update.

    def get_scroll_anchor_item_data(item):  # DEBUG only
        try:
            config = dpg.get_item_configuration(item)
            user_data = config["user_data"]
            if user_data is not None:
                kind_, data_idx = user_data  # `data_idx`: index to `sorted_xxx`
                entry = dataset.sorted_entries[data_idx]
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
        # If the selection is the same as before, or there is at least one item common across old and new selection, we can try to anchor the scroll position.
        #
        # NOTE: It can happen that we find a scroll anchor item, but it is no longer shown *after* the update, in which case this step goes fine,
        # but the scroll anchor search (later, after the update completes) finds nothing.
        #
        # logger.debug(f"_update_info_panel: {task_env.task_name}: Old data: selection_anchor_data_idxs_set is {selection_anchor_data_idxs_set}")
        if (not selection_changed) or len(selection_anchor_data_idxs_set):
            logger.debug(f"_update_info_panel.compute_scroll_anchors: {task_env.task_name}: Old data: Start finding scroll anchor item...")

            # Before we do anything else, store the current scrollbar position. We need this to compute the offset (in pixels) between the scrollbar value and viewport y coordinate.
            current_y_scroll = dpg.get_y_scroll("item_information_panel")  # tag

            # Important for performance: consider possible anchor items only (those common between old and new selection).
            #
            # Fortunately, we have listed them already, during the previous info panel build - they are the entry container groups.
            # This allows us to use a classical binary search, because we have a list with no confounders. This is O(log(n)), so pretty much instant.
            #
            # This avoids scanning the list of info panel children linearly to look for a starting point for the binary search.
            # In the case when there is no valid anchor, it would scan the whole list (over 10 seconds for a full info panel of ~400 entries).
            #
            # NOTE: We scan the *old entries*, because we want to get the anchor from the old content before the new update is applied.
            env.scroll_anchor_data.clear()
            if selection_changed:
                # logger.debug(f"_update_info_panel.compute_scroll_anchors: {task_env.task_name}: selected data_idxs in info panel: {list(sorted(info_panel_entry_title_widgets.keys()))}")
                # logger.debug(f"_update_info_panel.compute_scroll_anchors: {task_env.task_name}: selected data_idxs common between old and new selection: {list(sorted(selection_anchor_data_idxs_set))}")
                possible_anchors_only = [item for data_idx, item in info_panel_entry_title_widgets.items() if data_idx in selection_anchor_data_idxs_set]  # `data_idx`: index to `sorted_xxx`
                logger.debug(f"_update_info_panel.compute_scroll_anchors: {task_env.task_name}: Selection changed; old info panel selection_data_idxs common with new selection: {list(sorted(info_panel_widget_to_data_idx[x] for x in possible_anchors_only))}")
            else:
                logger.debug(f"_update_info_panel.compute_scroll_anchors: {task_env.task_name}: Selection not changed; can anchor on any info panel item.")
                possible_anchors_only = list(info_panel_entry_title_widgets.values())
            is_partially_below_top_of_viewport = functools.partial(is_partially_below_target_y, target_y=0)
            item = binary_search_item(items=possible_anchors_only,
                                      accept=is_partially_below_top_of_viewport,
                                      consider=None,
                                      skip=None,
                                      direction="right")

            # Multi-anchor: anchor using any item visible in viewport.
            #
            # This may sometimes help, if the topmost item is not shown after the rebuild, but at least one of the others happens to be.
            # Of course, this not a complete solution, as it may still happen that none of the anchors are shown after the rebuild.
            if item is not None:
                # Find all visible items, starting from the first one we found via the binary search.
                # There are only a few due to screen estate being limited (even at 4k resolution), so we can linearly scan them.
                start_display_idx = info_panel_widget_to_display_idx[item]  # how-manyth item in the info panel
                _, info_panel_h = get_info_panel_content_area_size()
                is_partially_above_bottom_of_viewport = functools.partial(is_partially_above_target_y, target_y=info_panel_h)
                visible_items = []
                for item_ in islice(info_panel_entry_title_widgets.values())[start_display_idx:]:
                    if not is_partially_above_bottom_of_viewport(item_):
                        break
                    visible_items.append(item_)
                scroll_anchors_debug_str = "\n    ".join(f"{item_}, tag '{dpg.get_item_alias(item_)}', type {dpg.get_item_type(item_)}, data_idx {info_panel_widget_to_data_idx[item_]}" for item_ in visible_items)
                plural_s = "s" if len(visible_items) != 1 else ""
                scroll_anchors_final_debug_str = f", list follows.\n    {scroll_anchors_debug_str}" if len(visible_items) else "."
                logger.debug(f"_update_info_panel.compute_scroll_anchors: {task_env.task_name}: Old data: Found {len(visible_items)} at least partially visible scroll-anchorable item{plural_s}{scroll_anchors_final_debug_str}")

                # Record `y_diff` for each visible item (possible anchor).
                content_start_x0, content_start_y0 = dpg.get_item_rect_min(info_panel_content_group)  # start of content, in viewport coordinates (may be out of view!)
                for item in visible_items:
                    raw = str(item)
                    alias = dpg.get_item_alias(item)
                    item_str = f"{item}, tag '{dpg.get_item_alias(item)}'" if raw != alias else f"'{alias}'"
                    logger.debug(f"_update_info_panel.compute_scroll_anchors: {task_env.task_name}: Old data: Recording scroll anchor {item_str}, data_idx {info_panel_widget_to_data_idx[item]}.")
                    try:
                        logger.debug(f"_update_info_panel.compute_scroll_anchors: {task_env.task_name}: Old data:     Item type is {dpg.get_item_type(item)}")
                        if (anchor_item_data := get_scroll_anchor_item_data(item)) is not None:
                            logger.debug(f"_update_info_panel.compute_scroll_anchors: {task_env.task_name}: Old data:     Item data is '{anchor_item_data}'")
                    except Exception:  # not found (race condition?)
                        pass

                    # NOTE: A DPG group needs to be rendered at least once to get a meaningful size.
                    x0, y0 = dpg.get_item_rect_min(item)
                    w, h = dpg.get_item_rect_size(item)
                    logger.debug(f"_update_info_panel.compute_scroll_anchors: {task_env.task_name}: Old data:     Item old position is y0 = {y0}, y_last = {y0 + h - 1}")

                    item_y_offset_from_content_start = y0 - content_start_y0  # start of original anchor item, in info panel coordinates

                    # Additive conversion term: difference between the scrollbar position and the info-panel y coordinate of the start of the anchor item.
                    y_diff = current_y_scroll - item_y_offset_from_content_start

                    stripped_tag = strip_build_number_from_tag(dpg.get_item_alias(item))
                    env.scroll_anchor_data[stripped_tag] = y_diff

                    logger.debug(f"_update_info_panel.compute_scroll_anchors: {task_env.task_name}: Old data:     Content area start y = {content_start_y0}, item y = {y0}, Δy = {item_y_offset_from_content_start}, scroll position = {current_y_scroll}, diff = {-y_diff}")
                plural_s = "s" if len(env.scroll_anchor_data) != 1 else ""
                logger.debug(f"_update_info_panel.compute_scroll_anchors: {task_env.task_name}: Old data: Scroll anchors updated. Found {len(env.scroll_anchor_data)} possible anchor{plural_s}.")
            else:
                # No anchorable item exists. This happens when there are items common across old and new selection, but before the update,
                # none of them are on-screen (or even included) in the info panel.
                logger.debug(f"_update_info_panel.compute_scroll_anchors: {task_env.task_name}: Old data: Selection has changed with no anchorable item in info panel. Resetting scroll anchors.")
                env.scroll_anchor_data.clear()
        else:
            # When there are no common items between old and new selection, the info panel content changes completely, so "the same" scroll position does not exist.
            logger.debug(f"_update_info_panel.compute_scroll_anchors: {task_env.task_name}: Old data: Selection has changed with no items common with previous selection. Resetting scroll anchors.")
            env.scroll_anchor_data.clear()

    new_y_scroll = None  # For setting the scroll position when the render completes; this is computed from a scroll anchor.
    def compute_new_scroll_target_position(anchor_tag):
        """Compute new scroll position based on DPG GUI widget `anchor` (in new data).

        This uses the anchor's recorded diff (scroll position, vs. anchor position in viewport coordinates)
        from the old data to apply the same diff to the new scroll position.

        The new scroll position is stored in `new_y_scroll`.
        """
        nonlocal new_y_scroll

        data_idx = info_panel_widget_to_data_idx_new.get(anchor_tag, "unknown")  # DEBUG logging

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

        # NOTE: A DPG group needs to be rendered at least once to get a meaningful size.
        new_x0, new_y0 = dpg.get_item_rect_min(anchor_tag)
        new_w, new_h = dpg.get_item_rect_size(anchor_tag)
        logger.debug(f"_update_info_panel.compute_new_scroll_target_position: {task_env.task_name}: New data:     Item new position is y0 = {new_y0}, y_last = {new_y0 + new_h - 1}")

        new_content_start_x0, new_content_start_y0 = dpg.get_item_rect_min(info_panel_content_target)  # start of new content, in viewport coordinates
        new_item_y_offset_from_content_start = new_y0 - new_content_start_y0  # start of new anchor item, in info panel coordinates

        logger.debug(f"_update_info_panel.compute_new_scroll_target_position: {task_env.task_name}: New data:     Content area start y = {new_content_start_y0}, item y = {new_y0}, Δy = {new_item_y_offset_from_content_start}")

        new_y_scroll = max(0, new_item_y_offset_from_content_start + env.scroll_anchor_data[stripped_tag])  # use the diff as measured from the old item, before update

        logger.debug(f"_update_info_panel.compute_new_scroll_target_position: {task_env.task_name}: New data: New scroll position recorded: {new_y_scroll}.")

    # --------------------------------------------------------------------------------
    # Start rebuilding the info panel content.

    # Update GUI indicators.
    dpg.set_value("item_information_total_count", "[updating]")  # we'll know the value once the update completes.
    dpg.show_item("item_information_total_count")
    if search_string:  # search active?
        dpg.set_value("item_information_search_controls_item_count", "[updating]")  # we'll know the value once the update completes.
    else:
        dpg.set_value("item_information_search_controls_item_count", "[no search active]")  # this we can set immediately  # TODO: DRY duplicate definitions for labels

    # Before starting the rebuild, make sure any partially built old content is really gone (including tags, because those must be unique).
    # When the left mouse button is hammered over the plotter, lots of selection changes occur, triggering many info panel cancellations.
    gc.collect()  # make sure the Python objects are gone
    dpg.split_frame()  # wait for DPG to update its registries

    # `with dpg.group(...):` would look clearer, but it's better to not touch the DPG container stack from a background thread.
    info_panel_content_target = dpg.add_group(horizontal=False, show=False, parent="item_information_panel", before="info_panel_content_end_spacer")  # tag

    # After this point (content target group GUI widget created), if something goes wrong, we must clean up the partially built content.
    try:
        info_panel_entry_title_widgets_new = {}
        info_panel_widget_to_data_idx_new = {}
        info_panel_widget_to_display_idx_new = {}
        info_panel_search_result_widgets_new = []
        info_panel_search_result_widget_to_display_idx_new = {}
        cluster_ids_in_selection_new = []
        cluster_id_to_display_idx_new = {}

        # Get item data grouped by cluster.
        entries_by_cluster, formatter = get_entries_for_selection(selection_data_idxs, max_n=gui_config.max_items_in_info_panel)

        # Build the list of clusters shown in the info panel. Also build the inverse mapping, to look up which cluster is the next/previous one shown in the info panel.
        cluster_ids_in_selection_new.clear()
        cluster_id_to_display_idx_new.clear()
        cluster_ids = list(sorted(set(entries_by_cluster.keys())))
        if cluster_ids and cluster_ids[0] == -1:  # move the misc group (if it's there) to the end
            cluster_ids = cluster_ids[1:] + [-1]
        cluster_ids_in_selection_new.extend(cluster_ids)
        cluster_id_to_display_idx_new.update({cluster_id: display_idx for display_idx, cluster_id in enumerate(cluster_ids_in_selection_new)})

        # Update the info panel title.
        # NOTE: This title talks about the *whole selection*, not about the subset that fits in the info panel.
        #   - `selection_data_idxs` contains the indices to `sorted_xxx` for the whole selection.
        #   - Although `get_entries_for_selection` returns info panel entries only, the info panel always shows
        #     at least one entry for each cluster, so the clusters are the same as in the whole selection.
        # The total count of items shown in the info panel gets its final update later, when the info panel update
        # itself completes.
        if len(selection_data_idxs):
            item_plural_s = "s" if len(selection_data_idxs) != 1 else ""
            cluster_plural_s = "s" if len(cluster_ids_in_selection_new) != 1 else ""
            top_heading_text = f"[{len(selection_data_idxs)} item{item_plural_s} total in {len(cluster_ids_in_selection_new)} cluster{cluster_plural_s}]"  # "[x items total in m clusters]"
        else:
            top_heading_text = "[nothing selected]"  # TODO: DRY duplicate definitions for labels
            dpg.add_text("[Select item(s) to view information]", color=(140, 140, 140, 255), parent=info_panel_content_target)  # TODO: DRY duplicate definitions for labels
        dpg.set_value(item_information_text, top_heading_text)

        # Start writing the report, for exporting to clipboard.
        #
        # Plain text (.txt)
        report_text = StringIO()
        report_text.write(top_heading_text + "\n")
        report_text.write("=" * len(top_heading_text) + "\n\n")
        # Markdown (.md)
        report_md = StringIO()
        report_md.write(f"# {top_heading_text}\n\n")

        # Per-entry button callback factories
        def make_scroll_info_panel_to_cluster(display_idx):
            """Make a callback to scroll the info panel to given cluster, by sequential numbering in info panel.

            NOTE: sequential numbering in info panel, not cluster ID.
            """
            def scroll_info_panel_to_the_cluster():  # freeze `display_idx` by closure
                _scroll_info_panel_to_cluster_by_display_idx(display_idx)
            return scroll_info_panel_to_the_cluster

        def make_copy_entry_to_clipboard(title_container_group):  # freeze input by closure
            """Make a callback to copy the authors/year/title of the given entry to the OS clipboard.

            These callbacks are bound to the per-item buttons.
            """
            def copy_this_entry_to_clipboard():
                _copy_entry_to_clipboard(title_container_group)  # This operation needs access to the GUI widgets to show the acknowledgment animation.
            return copy_this_entry_to_clipboard

        def make_search_or_select_entry(entry):  # freeze input by closure
            """Make a callback to search for the given item in the plotter, or change the selection (see `_search_or_select_entry`).

            These callbacks are bound to the per-item buttons.
            """
            def search_or_select_this_entry():
                _search_or_select_entry(entry)
            return search_or_select_this_entry

        def make_select_cluster(cluster_id):  # freeze input by closure
            """Make a callback to select all data in cluster `cluster_id`.

            Shift, Ctrl, Ctrl+Shift modes available.

            Triggers an info panel update if the selection changes.
            """
            def select_this_cluster():
                select_cluster_by_id(cluster_id)
            return select_this_cluster

        # Build info panel content and write report
        total_entries_shown_in_info_panel = 0
        for display_idx, cluster_id in enumerate(cluster_ids_in_selection_new):
            dpg.set_value("item_information_total_count", f"[updating {display_idx + 1}/{len(cluster_ids_in_selection_new)}]")  # tag

            first_cluster = (display_idx == 0)
            last_cluster = (display_idx == len(cluster_ids_in_selection_new) - 1)

            # Allow canceling an update in progress (if the search or selection GUI state changes and the report being generated is no longer relevant)
            if task_env is not None and task_env.cancelled:
                break

            cluster_title, cluster_keywords, cluster_content, more = formatter(cluster_id)
            total_entries_shown_in_info_panel += len(cluster_content)  # how many entries in this cluster

            cluster_header_group = dpg.add_group(horizontal=True, parent=info_panel_content_target, tag=f"cluster_{cluster_id}_header_group_build{env.internal_build_number}")

            # Next/previous cluster buttons
            up_enabled = (not first_cluster)
            up_button = dpg.add_button(tag=f"cluster_{cluster_id}_up_button_build{env.internal_build_number}",
                                       # label=fa.ICON_CARET_UP,
                                       arrow=True,
                                       direction=dpg.mvDir_Up,
                                       enabled=up_enabled,
                                       callback=(make_scroll_info_panel_to_cluster(display_idx - 1) if up_enabled else lambda: None),
                                       parent=cluster_header_group)
            # dpg.bind_item_font(f"cluster_{cluster_id}_up_button", icon_font_solid)
            up_tooltip = dpg.add_tooltip(up_button)
            dpg.add_text("Previous cluster [Ctrl+P]", parent=up_tooltip)
            dpg.bind_item_theme(f"cluster_{cluster_id}_up_button_build{env.internal_build_number}", "disablable_button_theme")  # tag
            # dpg.configure_item(f"cluster_{cluster_id}_up_button_build{env.internal_build_number}", enabled=up_enabled)  # tag

            down_enabled = (not last_cluster)
            down_button = dpg.add_button(tag=f"cluster_{cluster_id}_down_button_build{env.internal_build_number}",
                                         # label=fa.ICON_CARET_DOWN,
                                         arrow=True,
                                         direction=dpg.mvDir_Down,
                                         enabled=down_enabled,
                                         callback=(make_scroll_info_panel_to_cluster(display_idx + 1) if down_enabled else lambda: None),
                                         parent=cluster_header_group)
            # dpg.bind_item_font(f"cluster_{cluster_id}_down_button", icon_font_solid)
            down_tooltip = dpg.add_tooltip(down_button)
            dpg.add_text("Next cluster [Ctrl+N]", parent=down_tooltip)
            dpg.bind_item_theme(f"cluster_{cluster_id}_down_button_build{env.internal_build_number}", "disablable_button_theme")  # tag
            # dpg.configure_item(f"cluster_{cluster_id}_down_button_build{env.internal_build_number}", enabled=down_enabled)  # tag

            # Cluster title and keywords
            cluster_title_widget = dpg.add_text(cluster_title, tag=f"cluster_{cluster_id}_title_build{env.internal_build_number}", color=(180, 180, 180), parent=cluster_header_group)  # "#42"
            dpg.set_item_user_data(cluster_title_widget, ("cluster_title", cluster_id))  # for `is_cluster_title`
            plural_s = "s" if len(entries_by_cluster[cluster_id]) != 1 else ""
            entries_text = f"[{len(entries_by_cluster[cluster_id])} item{plural_s}]"
            dpg.add_text(entries_text, wrap=0, color=(140, 140, 140), tag=f"cluster_{cluster_id}_item_count_build{env.internal_build_number}", parent=cluster_header_group)  # tag  # "[x items]"
            dpg.add_text(cluster_keywords, wrap=0, color=(140, 140, 140), tag=f"cluster_{cluster_id}_keywords_build{env.internal_build_number}", parent=cluster_header_group)  # tag  # "[keyword0, ...]"

            # Report: cluster heading
            report_cluster_heading_text = f"{cluster_title} {entries_text} {cluster_keywords}".strip()
            report_text.write(report_cluster_heading_text + "\n")
            report_text.write("-" * len(report_cluster_heading_text) + "\n\n")

            report_md.write(f"## {report_cluster_heading_text}\n\n")

            # Cluster title separator
            cluster_title_separator = dpg.add_drawlist(width=gui_config.info_panel_w - 20, height=1, parent=info_panel_content_target, tag=f"cluster_{cluster_id}_title_separator_build{env.internal_build_number}")
            dpg.draw_line((0, 0), (gui_config.info_panel_w - 21, 0), color=(140, 140, 140, 255), thickness=1, parent=cluster_title_separator)

            # Items in cluster
            for data_idx, entry in cluster_content:  # `data_idx`: item index into `sorted_xxx`
                if task_env is not None and task_env.cancelled:
                    break

                # Highlight search results, but only when a search is active.
                # When no search active, show all items at full brightness.
                if not search_string or data_idx in search_result_data_idxs:
                    use_bright_text = True  # search match, or search not active
                    title_color = (255, 255, 255, 255)
                    abstract_color = (180, 180, 180, 255)
                else:  # Search active, non-matching item -> dim it.
                    use_bright_text = False
                    title_color = (140, 140, 140, 255)
                    abstract_color = (110, 110, 110, 255)
                is_search_match = (search_string and use_bright_text)

                # ----------------------------------------
                # Containers
                entry_container_group = dpg.add_group(parent=info_panel_content_target, tag=f"cluster_{cluster_id}_entry_{data_idx}_build{env.internal_build_number}")  # entry container: title + abstract

                entry_title_container_group = dpg.add_group(horizontal=True, tag=f"cluster_{cluster_id}_entry_{data_idx}_header_group_build{env.internal_build_number}", parent=entry_container_group)  # title container: buttons + the actual title text

                # ----------------------------------------
                # Per-item buttons, column 1
                entry_buttons_column_1_group = dpg.add_group(horizontal=False, tag=f"cluster_{cluster_id}_entry_{data_idx}_header_button_column_1_group_build{env.internal_build_number}", parent=entry_title_container_group)

                # Back to top of this cluster button
                b = dpg.add_button(tag=f"cluster_{cluster_id}_entry_{data_idx}_back_to_cluster_top_button_build{env.internal_build_number}",
                                   # label=fa.ICON_CHEVRON_UP,
                                   # width=gui_config.info_panel_button_w,  # width is not applicable when using `arrow=True`
                                   arrow=True,
                                   direction=dpg.mvDir_Up,
                                   callback=make_scroll_info_panel_to_cluster(display_idx),
                                   parent=entry_buttons_column_1_group)
                dpg.bind_item_font(b, icon_font_solid)
                b_tooltip = dpg.add_tooltip(b)
                b_tooltip_text = dpg.add_text(f"Back to top of cluster #{cluster_id} [Ctrl+U]" if cluster_id != -1 else "Back to top of Misc [Ctrl+U]",
                                              parent=b_tooltip)

                # Copy this item to clipboard button
                b = dpg.add_button(label=fa.ICON_COPY,
                                   tag=f"cluster_{cluster_id}_entry_{data_idx}_copy_to_clipboard_button_build{env.internal_build_number}",
                                   width=gui_config.info_panel_button_w,
                                   parent=entry_buttons_column_1_group)
                dpg.bind_item_font(b, icon_font_solid)
                b_tooltip = dpg.add_tooltip(b)
                b_tooltip_text = dpg.add_text("Copy item authors, year and title to clipboard [Ctrl+Shift+C]",  # TODO: DRY duplicate definitions for labels
                                              parent=b_tooltip)
                dpg.set_item_callback(b, make_copy_entry_to_clipboard(entry_title_container_group))
                dpg.set_item_user_data(b, ("copy_entry_to_clipboard_button", (b_tooltip, b_tooltip_text)))  # for the `copy_current_entry_to_clipboard` hotkey

                # ----------------------------------------
                # Per-item buttons, column 2

                entry_buttons_column_2_group = dpg.add_group(horizontal=False, tag=f"cluster_{cluster_id}_entry_{data_idx}_header_button_column_2_group_build{env.internal_build_number}", parent=entry_title_container_group)

                # Search this item in plotter button
                b = dpg.add_button(label=fa.ICON_ARROW_RIGHT,
                                   tag=f"cluster_{cluster_id}_entry_{data_idx}_search_in_plotter_button_build{env.internal_build_number}",
                                   width=gui_config.info_panel_button_w,
                                   parent=entry_buttons_column_2_group)
                dpg.bind_item_font(b, icon_font_solid)
                b_tooltip = dpg.add_tooltip(b)
                dpg.add_text("Search for this item in the plotter [F6]\n(clear search if already searching for this item)\n    with Shift: set selection to this item only\n    with Ctrl: remove this item from selection",
                             parent=b_tooltip)
                dpg.set_item_callback(b, make_search_or_select_entry(entry))

                b = dpg.add_button(label=fa.ICON_WAND_MAGIC_SPARKLES,  # wand, by analogy with smart select in graphics programs
                                   tag=f"cluster_{cluster_id}_entry_{data_idx}_select_this_cluster_button_build{env.internal_build_number}",
                                   width=gui_config.info_panel_button_w,
                                   parent=entry_buttons_column_2_group)
                dpg.bind_item_font(b, icon_font_solid)
                b_tooltip = dpg.add_tooltip(b)
                cluster_name_str = f"#{cluster_id}" if cluster_id != -1 else "Misc"
                dpg.add_text(f"Select all items in the same cluster ({cluster_name_str}) as this item [F7]\n    with Shift: add\n    with Ctrl: subtract\n    with Ctrl+Shift: intersect",
                             parent=b_tooltip)
                dpg.set_item_callback(b, make_select_cluster(cluster_id))

                # ----------------------------------------
                # item authors, year, title (with search result highlight, if any)

                entry_title_text = entry.title
                if search_string:  # search active?
                    if case_insensitive_fragments:  # case-insensitive first so that e.g. a fragment "col" won't match the "<font color=...>"
                        # The font tags don't stack in the MD renderer, so we must close the surrounding tag (for title color)
                        # when the highlight starts, and then re-open it after the highlight ends.
                        entry_title_text = re.sub(regex_case_insensitive, f"</font>**<font color='#ff0000'>\\1</font>**<font color='{title_color}'>", entry_title_text)
                    if case_sensitive_fragments:  # case-sensitive fragments contain at least one uppercase letter, so they're still safe (no match in any font tags added by the case-insensitive replace)
                        entry_title_text = re.sub(regex_case_sensitive, f"</font>**<font color='#ff0000'>\\1</font>**<font color='{title_color}'>", entry_title_text)
                if search_string and entry_title_text != entry.title:  # any changes made by the substitutions? -> render as Markdown to enable highlighting
                    header = f"<font color='{title_color}'>{entry.author} ({entry.year}): {entry_title_text}</font>"
                    entry_title_group = dpg_markdown.add_text(header, wrap=gui_config.title_wrap_w, parent=entry_title_container_group)  # The MD renderer internally renders the text items into a new group.
                    dpg.set_item_alias(entry_title_group, f"cluster_{cluster_id}_entry_{data_idx}_title_build{env.internal_build_number}")  # tag  # the MD renderer doesn't have a `tag` parameter, so we set the tag afterward.
                    if is_search_match:
                        info_panel_search_result_widgets_new.append(entry_title_container_group)
                        info_panel_search_result_widget_to_display_idx_new[entry_title_container_group] = len(info_panel_search_result_widgets_new) - 1  # reverse lookup to see in O(1) time how-manyth search result a given item is
                else:  # search not active, or no changes made (i.e. no match in this title). -> render as plain text (much faster)
                    header = f"{entry.author} ({entry.year}): {entry_title_text}"
                    # The Markdown renderer's line spacing differs from that of the plain text renderer. We fix this manually for consistency.
                    #
                    # Before scroll anchoring was introduced, the difference in line heights caused the scroll position to jump (near the end of a large dataset, a lot)
                    # when changing between matched/nonmatched state, because the plain/highlighted titles may take up a different amount of vertical space.
                    #
                    # For visual consistency, we make the output have as similar a line spacing as possible in both cases.
                    # The plain and highlighted titles may still use a different number of lines:
                    #   - Bold (which we currently use) is slightly wider.
                    #   - Italic is slightly narrower.
                    #   - Just changing the color is ever so subtly (one pixel?) wider (?!).
                    # If we wanted to preserve the original division of lines when possible, any static wrap width doesn't help. What we would need is a layout algorithm
                    # that gives a few pixels of slack, but only when needed, similarly to what we do in the entry count limiter in the annotation tooltip.
                    #
                    # What we currently do, instead, is anchor the scroll position - we use a text item whose content doesn't change in the update to compute
                    # the new value for "the same" scroll position, although the info panel content has been completely rebuilt ship-of-Theseus style.
                    #
                    # 1) Wrap the text with the MD renderer (DPG itself doesn't seem to have a utility for this).
                    #    This is a bit slow since it's a low-level utility implemented in Python, but meh.
                    entity = dpg_markdown.text_entities.StrEntity(header)
                    entity = dpg_markdown.wrap_text_entity(entity, width=gui_config.title_wrap_w)  # -> iterable of lines
                    # 2) Render the text line by line, controlling the vertical spacing explicitly with a spacer.
                    entry_title_group = dpg.add_group(horizontal=False, tag=f"cluster_{cluster_id}_entry_{data_idx}_title_build{env.internal_build_number}", parent=entry_title_container_group)
                    for lineno, line_content in enumerate(entity):
                        last_line = (lineno == len(entity) - 1)
                        dpg.add_text(line_content, color=title_color, parent=entry_title_group)
                        if not last_line:
                            # Align with the MD renderer line height. The width of the spacer doesn't matter here.
                            # TODO: I have no idea where the two pixels of extra height comes from in the MD renderer.
                            dpg.add_spacer(width=10, height=2, parent=entry_title_group)
                    dpg.bind_item_theme(entry_title_group, "my_no_spacing_theme")  # tag  # Default spacing off, like in the MD renderer.

                info_panel_entry_title_widgets_new[data_idx] = entry_title_container_group
                info_panel_widget_to_data_idx_new[entry_title_container_group] = data_idx  # reverse lookup item -> index in `sorted_xxx`
                info_panel_widget_to_display_idx_new[entry_title_container_group] = len(info_panel_entry_title_widgets_new) - 1  # reverse lookup item -> how manyth in info panel (index in `info_panel_entry_title_widgets_new`)
                dpg.set_item_user_data(entry_title_container_group, ("entry_title_container", data_idx))  # for `is_entry_title_container_group`  # `data_idx`: index to `sorted_xxx`
                dpg.set_item_user_data(entry_title_group, ("entry_title_text", data_idx))  # for `is_entry_title_text_item`  # `data_idx`: index to `sorted_xxx`

                # ----------------------------------------
                # item abstract

                dpg.add_text(entry.abstract, color=abstract_color, wrap=gui_config.main_text_wrap_w, tag=f"cluster_{cluster_id}_entry_{data_idx}_abstract_build{env.internal_build_number}", parent=entry_container_group)
                dpg.add_text("", tag=f"cluster_{cluster_id}_entry_{data_idx}_end_blank_text_build{env.internal_build_number}", parent=entry_container_group)

                # Report: write item
                report_text.write(f"{entry.author} ({entry.year}): {entry.title}\n\n{entry.abstract.strip()}\n\n")
                report_md.write(f"### {entry.author} ({entry.year}): {entry.title}\n\n{entry.abstract.strip()}\n\n")

            if task_env is None or not task_env.cancelled:
                if more:
                    dpg.add_text(more, wrap=0, color=(100, 100, 100), tag=f"cluster_{cluster_id}_more_build{env.internal_build_number}", parent=info_panel_content_target)  # "[...N more entries...]"
                    report_text.write(f"{more}\n\n")
                    report_md.write(f"{more}\n\n")

                # Cluster separator
                if not last_cluster:
                    cluster_end_separator_1 = dpg.add_drawlist(width=gui_config.info_panel_w - 20, height=1, parent=info_panel_content_target, tag=f"cluster_{cluster_id}_end_separator_1_build{env.internal_build_number}")
                    dpg.draw_line((0, 0), (gui_config.info_panel_w - 21, 0), color=(140, 140, 140, 255), thickness=1, parent=cluster_end_separator_1)
                    cluster_end_separator_2 = dpg.add_drawlist(width=gui_config.info_panel_w - 20, height=1, parent=info_panel_content_target, tag=f"cluster_{cluster_id}_end_separator_2_build{env.internal_build_number}")
                    dpg.draw_line((0, 0), (gui_config.info_panel_w - 21, 0), color=(140, 140, 140, 255), thickness=1, parent=cluster_end_separator_2)
                    dpg.add_text("", tag=f"cluster_{cluster_id}_end_blank_text_build{env.internal_build_number}", parent=info_panel_content_target)

                    # Report: cluster separator
                    report_text.write("-" * 80 + "\n")
                    report_text.write("-" * 80 + "\n\n\n")

                    report_md.write("-----\n\n")

        # Finalize (if not cancelled)
        if task_env is None or not task_env.cancelled:  # if the task was cancelled, the report is incomplete
            # We are about to swap the whole content of the info panel, so stop the scroll animation if it is running.
            with scroll_animation_lock:
                if scroll_animation is not None:
                    scroll_animation.finish()
                    scroll_animation = None

            # Anchor the scroll position from the old data just before we swap in the new content, so that
            # it takes the latest position, in case the user scrolled while the info panel was building.
            compute_scroll_anchors()

            # Now that we have the full new content, render it so that the items actually get positions.
            #
            # It is tempting to use `set_item_pos`/`reset_pos` to render offscreen, but that doesn't produce
            # reliable positions for some reason. The y position we get then depends on the *x* coordinate
            # (maybe interaction between info panel child window size and text wrap?).
            #
            # But even if we set the position to the `rect_min` of the old content group (so exactly on top),
            # it still sometimes gets it wrong (at least near the end of long content, when switching search on/off).
            #
            # So it's better to just hide the old group, show the new one, and let DPG handle the laydataset.
            # When exactly one of the containers is shown, we have arranged for the new one to appear exactly
            # where the old one was. Then we can measure positions in the new data and everything works.
            #
            clear_current_item_info()  # Turn the current item controls highlight off (after the new content has been swapped in, `update_animations` will auto-update it at the next frame)
            dpg.hide_item(info_panel_content_group)
            dpg.show_item(info_panel_content_target)
            show_info_panel_dimmer_overlay()
            dpg.split_frame()  # wait for render, to have valid positions for the widgets in the new data

            # Find the new items (if any) corresponding to the recorded scroll anchors. Try each anchor, just in case.
            # There are only a few, and it's a no-op (with a log message) after the scroll position has been set successfully.
            scroll_anchor_stripped_tags = list(env.scroll_anchor_data.keys())
            scroll_anchor_new_tags = [tag for tag in info_panel_entry_title_widgets_new.values()
                                      if strip_build_number_from_tag(tag) in scroll_anchor_stripped_tags]
            for tag in scroll_anchor_new_tags:
                compute_new_scroll_target_position(tag)

            with info_panel_content_lock:
                # Swap the new content in ("double-buffering")
                logger.debug(f"_update_info_panel: {task_env.task_name}: Swapping in new content (old GUI widget ID {info_panel_content_group}; new GUI widget ID {info_panel_content_target}).")
                dpg.delete_item(info_panel_content_group)
                # # Old DPG versions had a bug where the aliases needed to be deleted manually.
                # # https://github.com/hoffstadt/DearPyGui/issues/1350
                # def delete_aliases(root, *, children_only=False):
                #     if not children_only:
                #         tag = dpg.get_item_alias(root)
                #         if dpg.does_alias_exist(tag):
                #             dpg.remove_alias(tag)
                #     for slot in range(4):
                #         for item in dpg.get_item_children(root, slot=slot):
                #             delete_aliases(item, children_only=False)  # always False, since only the root item should be handled differently.
                # delete_aliases(info_panel_content_group, children_only=True)
                info_panel_content_group = None  # just in case the next line raises an exception
                # dpg.reset_pos(info_panel_content_target)  # move it into place
                dpg.set_item_alias(info_panel_content_target, "info_panel_content_group")  # tag
                info_panel_content_group = info_panel_content_target
                new_content_swapped_in = True

                logger.debug(f"_update_info_panel: {task_env.task_name}: Swapping in new navigation metadata.")
                cluster_ids_in_selection.clear()
                cluster_ids_in_selection.extend(cluster_ids_in_selection_new)
                cluster_id_to_display_idx.clear()
                cluster_id_to_display_idx.update(cluster_id_to_display_idx_new)
                info_panel_entry_title_widgets.clear()
                info_panel_entry_title_widgets.update(info_panel_entry_title_widgets_new)
                info_panel_widget_to_data_idx.clear()
                info_panel_widget_to_data_idx.update(info_panel_widget_to_data_idx_new)
                info_panel_widget_to_display_idx.clear()
                info_panel_widget_to_display_idx.update(info_panel_widget_to_display_idx_new)
                info_panel_search_result_widgets.clear()
                info_panel_search_result_widgets.extend(info_panel_search_result_widgets_new)
                info_panel_search_result_widget_to_display_idx.clear()
                info_panel_search_result_widget_to_display_idx.update(info_panel_search_result_widget_to_display_idx_new)

                logger.debug(f"_update_info_panel: {task_env.task_name}: Content swapping complete.")

            # Finish the report
            report_plaintext << report_text.getvalue()
            report_markdown << report_md.getvalue()
            dpg.enable_item(copy_report_button)

            # Update the final item count in the GUI
            if total_entries_shown_in_info_panel > 0:
                dpg.set_value("item_information_total_count", f"[{total_entries_shown_in_info_panel} item{'s' if total_entries_shown_in_info_panel != 1 else ''} shown]")  # tag
                dpg.show_item("item_information_total_count")  # tag
            else:  # info panel build finished, and no items are shown, so hide the total count field.
                dpg.hide_item("item_information_total_count")  # tag

            # Restore/reset scroll position
            dpg.split_frame()  # let the content swap take before proceeding
            if not len(env.scroll_anchor_data):  # no anchorable items in the old data
                logger.debug(f"_update_info_panel: {task_env.task_name}: New data: no anchorable items in old data, resetting scroll position.")
                dpg.set_y_scroll("item_information_panel", 0)  # tag
            # at least one anchor recorded
            elif new_y_scroll is not None:  # new scroll position was recorded
                logger.debug(f"_update_info_panel: {task_env.task_name}: New data: scrolling to anchor, new_y_scroll = {new_y_scroll}")
                dpg.set_y_scroll("item_information_panel", new_y_scroll)  # tag
            else:  # new scroll position not recorded -> anchorable items exist in old data, but none of them were included in the info panel in the new data
                logger.debug(f"_update_info_panel: {task_env.task_name}: New data: at least one anchor exists, but none are shown after update. Resetting scroll position.")
                dpg.set_y_scroll("item_information_panel", 0)  # tag
            _info_panel_scroll_position_changed(reset=True)  # for `update_current_search_result_status` (do this *after* swapping in the new content!)
            selection_changed = False

            # Which items are shown in info panel may have changed. Re-render the annotation tooltip in case it is active, to update the items' "shown in info panel" status there.
            update_mouse_hover(force=True, wait=False)

            # Update search controls. Do this last.
            if search_string:  # is a search active?
                num = len(info_panel_search_result_widgets)  # how many search results are actually shown in the info panel
                search_results_info_panel_str = f"[{'no' if not num else num} search result{'s' if num != 1 else ''} shown]"
            else:
                search_results_info_panel_str = "[no search active]"  # TODO: DRY duplicate definitions for labels
            dpg.set_value("item_information_search_controls_item_count", search_results_info_panel_str)  # tag
            dpg.split_frame()  # let the scrollbar position update before proceeding
            hide_info_panel_dimmer_overlay()
            update_next_prev_search_result_buttons()

    except Exception:
        logger.debug(f"_update_info_panel: {task_env.task_name}: Info panel update task raised an exception; cancelling task.")
        task_env.cancelled = True
        if not new_content_swapped_in:  # clean up: re-show the old content (if it still exists) in case the exception occurred during finalizing
            if info_panel_content_target is not None:
                dpg.hide_item(info_panel_content_target)
            if info_panel_content_group is not None:
                dpg.show_item(info_panel_content_group)
        raise

    finally:
        if task_env is not None and task_env.cancelled:
            logger.debug(f"_update_info_panel: {task_env.task_name}: Info panel update task cancelled.")

            # If the new content was built (partially or completely) but not swapped in, it's unused, so delete it.
            if (info_panel_content_target is not None) and (not new_content_swapped_in):
                logger.debug(f"_update_info_panel: {task_env.task_name}: Deleting partially built content.")
                dpg.delete_item(info_panel_content_target)

            # These will be soon refreshed again when the next update starts (since we only cancel a running update task when it is superseded by a new one),
            # but in the meantime we should show up-to-date status - which is, the update that was running has been cancelled.
            dpg.set_value("item_information_total_count", "[update cancelled]")  # tag
            dpg.set_value("item_information_search_controls_item_count", "[update cancelled]")  # tag
        else:
            logger.debug(f"_update_info_panel: {task_env.task_name}: Info panel update task completed.")

            # Publish the build ID we used while building, so that callbacks can find the content.
            info_panel_build_number = env.internal_build_number

        dt = time.monotonic() - info_panel_t0
        plural_s = "ies" if total_entries_shown_in_info_panel != 1 else "y"
        logger.debug(f"_update_info_panel: {task_env.task_name}: Info panel build {env.internal_build_number} exiting. Rendered {total_entries_shown_in_info_panel} entr{plural_s} in {dt:0.2f}s.")
        env.internal_build_number += 1  # always increase internal build, even when cancelled, for unique IDs.

# --------------------------------------------------------------------------------
# Built-in help window

# Human-readable list of hotkeys. The layout engine below auto-converts this into a GUI table of hotkeys.
#
# - The table entries are here listed in human reading order, column first.
# - To tell the layout engine to start a new column (other than the first one), use `hotkey_new_column`.
# - To leave an empty row in the current column, use `hotkey_blank_entry`. This is useful to visually separate groups of related hotkeys.
# - The columns don't have to be the same length.
#
hotkey_new_column = sym("next_column")
hotkey_blank_entry = env(key_indent=0, key="", action_indent=0, action="", notes="")
hotkey_help = (env(key_indent=0, key="Ctrl+O", action_indent=0, action="Open a dataset", notes=""),
               env(key_indent=0, key="Ctrl+F", action_indent=0, action="Focus search field", notes=""),
               env(key_indent=1, key="Enter", action_indent=0, action="Select search matches, and unfocus", notes="When search field focused"),
               env(key_indent=2, key="Shift+Enter", action_indent=1, action="Same, but add to selection", notes="When search field focused"),
               env(key_indent=2, key="Ctrl+Enter", action_indent=1, action="Same, but subtract from selection", notes="When search field focused"),
               env(key_indent=2, key="Ctrl+Shift+Enter", action_indent=1, action="Same, but intersect with selection", notes="When search field focused"),
               env(key_indent=1, key="Esc", action_indent=0, action="Cancel search term edit, and unfocus", notes="When search field focused"),
               env(key_indent=0, key="F3", action_indent=0, action="Scroll to next search match", notes="When matches shown in info panel"),
               env(key_indent=0, key="Shift+F3", action_indent=0, action="Scroll to previous search match", notes="When matches shown in info panel"),
               hotkey_blank_entry,
               env(key_indent=0, key="Ctrl+U", action_indent=0, action="Scroll to start of current cluster", notes='"up"'),
               env(key_indent=1, key="Ctrl+N", action_indent=0, action="Scroll to next cluster", notes=""),
               env(key_indent=1, key="Ctrl+P", action_indent=0, action="Scroll to previous cluster", notes=""),
               hotkey_blank_entry,
               env(key_indent=0, key="Home", action_indent=0, action="Scroll to top", notes="When search field NOT focused"),
               env(key_indent=1, key="End", action_indent=0, action="Scroll to bottom", notes="When search field NOT focused"),
               env(key_indent=1, key="Page Up", action_indent=0, action="Scroll up", notes="When search field NOT focused"),
               env(key_indent=1, key="Page Down", action_indent=0, action="Scroll down", notes="When search field NOT focused"),
               env(key_indent=1, key="Up arrow", action_indent=0, action="Scroll up slightly", notes="When search field NOT focused"),
               env(key_indent=1, key="Down arrow", action_indent=0, action="Scroll down slightly", notes="When search field NOT focused"),

               hotkey_new_column,
               env(key_indent=0, key="F6", action_indent=0, action="Search/unsearch current item", notes="Searching highlights it in the plotter"),
               env(key_indent=1, key="Shift+F6", action_indent=0, action="Set selection to current item only", notes=""),
               env(key_indent=1, key="Ctrl+F6", action_indent=0, action="Remove current item from selection", notes=""),
               env(key_indent=0, key="F7", action_indent=0, action="Select current cluster", notes=""),
               env(key_indent=1, key="Shift+F7", action_indent=1, action="Same, but add to selection", notes=""),
               env(key_indent=1, key="Ctrl+F7", action_indent=1, action="Same, but subtract from selection", notes=""),
               env(key_indent=1, key="Ctrl+Shift+F7", action_indent=1, action="Same, but intersect with selection", notes=""),
               env(key_indent=0, key="F8", action_indent=0, action="Copy report to clipboard", notes="As plain text, .txt"),
               env(key_indent=1, key="Shift+F8", action_indent=0, action="Copy report to clipboard", notes="As Markdown, .md"),
               env(key_indent=0, key="F9", action_indent=0, action="Select all data currently visible in plotter", notes=""),
               env(key_indent=1, key="Shift+F9", action_indent=1, action="Same, but add to selection", notes=""),
               env(key_indent=1, key="Ctrl+F9", action_indent=1, action="Same, but subtract from selection", notes=""),
               env(key_indent=1, key="Ctrl+Shift+F9", action_indent=1, action="Same, but intersect with selection", notes=""),
               env(key_indent=0, key="Ctrl+Shift+C", action_indent=0, action="Copy current item to clipboard", notes="As plain text, for web search"),
               env(key_indent=0, key="Ctrl+Shift+Z", action_indent=0, action="Undo last selection change", notes=""),
               env(key_indent=0, key="Ctrl+Shift+Y", action_indent=0, action="Redo last selection change", notes=""),
               hotkey_blank_entry,
               env(key_indent=0, key="Ctrl+Home", action_indent=0, action="Reset plotter zoom", notes=""),
               env(key_indent=0, key="F11", action_indent=0, action="Toggle fullscreen mode", notes=""),
               env(key_indent=0, key="F1", action_indent=0, action="Open this Help card", notes=""),
               )

# We create the window on demand, because we need some styling, but `dpg_markdown.add_text` hangs the app if used at startup.
# (Apparently, if called more than once before the first frame. Probably due to font loading?)
help_window = None
def make_help_window():
    """Create the built-in help card (if not created yet)."""
    global help_window
    if help_window is not None:
        return
    if dpg.get_frame_count() < 10:
        return

    with dpg.window(show=False, label="Help", tag="help_window",
                    modal=True,
                    on_close=hide_help_window,
                    no_collapse=True,
                    no_resize=True,
                    no_scrollbar=True,
                    no_scroll_with_mouse=True,
                    width=gui_config.help_window_w,
                    height=gui_config.help_window_h) as help_window:
        @call  # avoid polluting top-level namespace
        def _():
            help_heading_color = (255, 255, 255, 255)
            help_text_color = (180, 180, 180, 255)
            help_dim_color = (140, 140, 140, 255)
            help_indent_pixels = 20

            # Shorthand for color control sequences for MD renderer
            c_hed = f'<font color="{help_heading_color}">'
            c_txt = f'<font color="{help_text_color}">'
            c_dim = f'<font color="{help_dim_color}">'
            c_hig = '<font color="#ff0000">'  # help text highlight for very important parts
            c_search = '<font color="(255, 96, 96)">'  # same color as search highlight in plotter
            c_selection = '<font color="(96, 255, 255)">'  # same color as selection highlight in plotter
            c_end = '</font>'

            # Extract columns from the human-readable representation
            columns = []
            current_column = []
            for help_entry in hotkey_help:
                if help_entry is hotkey_new_column:
                    columns.append(current_column)
                    current_column = []
                else:
                    current_column.append(help_entry)
            if len(current_column):  # loop-and-a-half, kind of
                columns.append(current_column)
            ncols = len(columns)

            # Convert to rows (format actually used by DPG for constructing tables)
            rows = list(itertools.zip_longest(*columns, fillvalue=hotkey_blank_entry))

            with dpg.group(tag="hotkeys_help_group"):
                # Header
                dpg_markdown.add_text(f"{c_dim}[Press Esc to close. For a handy reference, screenshot this!]{c_end}")
                dpg.add_spacer(width=1, height=gui_config.font_size // 2)

                # Table of hotkeys. Render the specified layout.
                with dpg.table(header_row=True, borders_innerV=True, sortable=False):
                    for _ in range(ncols):
                        dpg.add_table_column(label="Key or combination")  # key
                        dpg.add_table_column(label="Action")  # action
                        dpg.add_table_column(label="Notes")  # notes

                    for row in rows:
                        with dpg.table_row():
                            for help_entry in row:
                                if help_entry.key_indent > 0:
                                    with dpg.group(horizontal=True):
                                        dpg.add_spacer(width=help_entry.key_indent * help_indent_pixels)
                                        dpg.add_text(help_entry.key, wrap=0, color=help_heading_color)
                                else:
                                    dpg.add_text(help_entry.key, wrap=0, color=help_heading_color)

                                if help_entry.action_indent > 0:
                                    with dpg.group(horizontal=True):
                                        dpg.add_spacer(width=help_entry.action_indent * help_indent_pixels)
                                        dpg.add_text(help_entry.action, wrap=0, color=help_dim_color)
                                else:
                                    dpg.add_text(help_entry.action, wrap=0, color=help_dim_color)

                                dpg.add_text(help_entry.notes, wrap=0, color=help_dim_color)
                dpg.add_spacer(width=1, height=gui_config.font_size)

                # Legend for table
                dpg_markdown.add_text(f"{c_hed}**Terminology**{c_end}")
                with dpg.group(horizontal=True):
                    with dpg.group(horizontal=False):
                        dpg_markdown.add_text(f"- {c_txt}**Current item**: The topmost item **fully** visible in the info panel. The controls of the current item glow slightly.{c_end}")
                        dpg_markdown.add_text(f"- {c_txt}**Current cluster**: The cluster the current item belongs to. Clusters are auto-detected by a linguistic analysis.{c_end}")
                    with dpg.group(horizontal=False):
                        dpg_markdown.add_text(f"- {c_txt}**Selection set**: The selected items, glowing {c_end}{c_selection}**cyan**{c_end}{c_txt} in the plotter. As many are loaded into the info panel as reasonably fit.{c_end}")
                        dpg_markdown.add_text(f"- {c_txt}**Search result set**: The items matching the current search, glowing {c_end}{c_search}**red**{c_end}{c_txt} in the plotter.{c_end}")
                dpg.add_spacer(width=1, height=gui_config.font_size)

                # Additional general help
                dpg_markdown.add_text(f"{c_hed}**How search works**{c_end}")
                dpg_markdown.add_text(f"{c_txt}Each space-separated search term is a **fragment**. For a data point to match, **all** fragments must match. Ordering of fragments does **not** matter. The {c_end}{c_search}search result{c_end}{c_txt} and {c_end}{c_selection}selection{c_end}{c_txt} sets are **independent**. {c_end}{c_search}Search results{c_end}{c_txt} live-update as you type.{c_end}")
                dpg_markdown.add_text(f'- {c_txt}A **lowercase** fragment matches **that fragment {c_end}{c_hig}case-insensitively{c_end}{c_txt}**. E.g. *"hydrogen"* matches also *"Hydrogen"*.{c_end}')
                dpg_markdown.add_text(f'- {c_txt}A fragment with **at least one uppercase** letter matches **that fragment {c_end}{c_hig}case-sensitively{c_end}{c_txt}**. E.g. *"TiO"* matches only titanium oxide, not *"bastion"*.{c_end}')
                dpg_markdown.add_text(f'- {c_txt}You can use regular numbers in place of subscript/superscript numbers. E.g. *"h2so4"* matches also *"H₂SO₄"*, and *"x2"* matches also *"x²"*. {c_end}')
                dpg_markdown.add_text(f"{c_txt}When the search field is focused, the usual text editing keys are available (*Enter, Esc, Home, End, Shift-select, Ctrl+Left, Ctrl+Right, Ctrl+A, Ctrl+Z, Ctrl+Y*).{c_end}")

def recenter_help_window():
    """Reposition the built-in help card (if visible) so that it is centered in the viewport."""

    if help_window is None:
        return

    main_window_w, main_window_h = get_main_window_size()
    logger.debug(f"recenter_help_window: Main window size is {main_window_w}x{main_window_h}.")

    # # Render offscreen so we get the final size. Only needed if the size can change.
    # dpg.set_item_pos(help_window,
    #                  (main_window_w,
    #                   main_window_h))
    # dpg.show_item(help_window)
    # logger.debug(f"recenter_help_window: After show command: Help window is visible? {dpg.is_item_visible(help_window)}.")
    # dpg.split_frame()  # wait for render
    #
    # logger.debug(f"recenter_help_window: After wait for render: Help window is visible? {dpg.is_item_visible(help_window)}.")

    config = dpg.get_item_configuration(help_window)
    w = config["width"]
    h = config["height"]
    logger.debug(f"recenter_help_window: Help window size is {w}x{h}.")

    # Center the help window in the viewport
    dpg.set_item_pos(help_window,
                     (max(0, (main_window_w - w) // 2),
                      max(0, (main_window_h - h) // 2)))

def show_help_window():
    """Show the built-in help card."""
    logger.debug("show_help_window: Ensuring help window exists.")
    make_help_window()
    logger.debug("show_help_window: Recentering help window.")
    recenter_help_window()
    logger.debug("show_help_window: Showing help window.")
    dpg.show_item(help_window)  # For some reason, we need to do this *after* `set_item_pos` in modal mode, or this works only every other time (1, 3, 5, ...). Maybe a modal must be inside the viewport to successfully show it?
    enter_modal_mode()
    logger.debug("show_help_window: Done.")
dpg.set_item_callback("help_button", show_help_window)  # tag

def hide_help_window():
    """Close the built-in help card, if it is open."""
    if help_window is None:
        logger.debug("hide_help_window: Help window does not exist. Nothing needs to be done.")
        return
    logger.debug("hide_help_window: Hiding help window.")
    dpg.hide_item(help_window)
    exit_modal_mode()
    logger.debug("hide_help_window: Done.")

def is_help_window_visible():
    """Return whether the built-in help card is open.

    We have this abstraction (not just `dpg.is_item_visible`) because the window might not exist, if it has not been opened yet.
    """
    if help_window is None:
        return False
    return dpg.is_item_visible(help_window)

# --------------------------------------------------------------------------------
# GUI resizing handler

def wait_for_viewport_size_change(wait_frames_max):
    """Wait for the viewport size to actually change.

    `wait_frames_max`: int or `None`.
       int: How many frames to wait at most. Return `True` if the viewport size changed, `False` otherwise.
       `None`: Wait indefinitely. Return `True` once the viewport size has changed.
    """
    waited = 0
    w0, h0 = get_main_window_size()
    while True:
        dpg.split_frame()
        waited += 1

        w, h = get_main_window_size()
        if w != w0 or h != h0:
            logger.debug(f"wait_for_viewport_size_change: viewport resized after waiting for {waited} frame{'s' if waited != 1 else ''}")
            return True

        if (wait_frames_max is not None) and (waited >= wait_frames_max):
            logger.debug(f"wait_for_viewport_size_change: timeout ({wait_frames_max} frame{'s' if wait_frames_max != 1 else ''}) while waiting for viewport to resize")
            return False

def resize_gui():
    """Wait for the viewport size to actually change, then resize dynamically sized GUI elements.

    This is handy for toggling fullscreen, because the size changes at the next frame at the earliest.
    For the viewport resize callback, that one fires (*almost* always?) after the size has already changed.
    """
    logger.debug("resize_gui: Entered. Waiting for viewport size change.")
    if wait_for_viewport_size_change(wait_frames_max=10):
        _resize_gui()
    logger.debug("resize_gui: Done.")

def _update_info_panel_height():
    """Resize the info panel content area RIGHT NOW, based on main window height."""
    config = dpg.get_item_configuration(main_window)
    h = config["height"]
    dpg.set_item_height("item_information_panel", h - gui_config.info_panel_reserved_h)  # tag

def _resize_gui():
    """Resize dynamically sized GUI elements, RIGHT NOW."""
    logger.debug("_resize_gui: Entered.")
    logger.debug("_resize_gui: Updating info panel height.")
    _update_info_panel_height()
    logger.debug("_resize_gui: Updating info panel current item on-screen coordinates.")
    update_current_item_info()
    logger.debug("_resize_gui: Recentering help window.")
    recenter_help_window()
    logger.debug("_resize_gui: Rebuilding dimmer overlay.")
    info_panel_dimmer_overlay.build(rebuild=True)
    logger.debug("_resize_gui: Done.")


# Old versions of DPG have a bug where they don't always call the viewport resize callback, but it seems to work in 1.11.
# https://github.com/hoffstadt/DearPyGui/issues/1896
dpg.set_viewport_resize_callback(_resize_gui)

# --------------------------------------------------------------------------------
# Mouse events

select_radius_update_lock = threading.RLock()
select_radius_draw_item = None
select_radius_last_pos = None
select_radius_last_scale_x = None
select_radius_last_scale_y = None

unit_manygon = np.array([(np.cos(t), np.sin(t)) for t in np.linspace(0, 2 * np.pi, 65)])  # discrete approximation of the unit apeirogon :P

def clear_select_radius_indicator():
    global select_radius_draw_item
    global select_radius_last_pos
    global select_radius_last_scale_x
    global select_radius_last_scale_y
    with select_radius_update_lock:
        if select_radius_draw_item is not None:
            dpg.delete_item(select_radius_draw_item)
        select_radius_draw_item = None
        select_radius_last_pos = None
        select_radius_last_scale_x = None
        select_radius_last_scale_y = None

def draw_select_radius_indicator():
    global select_radius_draw_item
    global select_radius_last_pos
    global select_radius_last_scale_x
    global select_radius_last_scale_y

    # Avoid unnecessary clear/redraw to prevent flickering
    p = dpg.get_plot_mouse_pos()
    pixels_per_data_unit_x, pixels_per_data_unit_y = get_pixels_per_plotter_data_unit()
    if pixels_per_data_unit_x == 0.0 or pixels_per_data_unit_y == 0.0:  # no dataset open?
        clear_select_radius_indicator()
        return
    same_pos = (select_radius_last_pos is not None and select_radius_last_pos == p)
    same_scale_x = (select_radius_last_scale_x is not None and select_radius_last_scale_x == pixels_per_data_unit_x)
    same_scale_y = (select_radius_last_scale_y is not None and select_radius_last_scale_y == pixels_per_data_unit_y)
    same_zoom = (same_scale_x and same_scale_y)

    with select_radius_update_lock:
        select_radius_last_pos = p
        select_radius_last_scale_x = pixels_per_data_unit_x
        select_radius_last_scale_y = pixels_per_data_unit_y

        if not (same_pos and same_zoom):
            clear_select_radius_indicator()  # remove old indicator if any

        # NOTE: To avoid race conditions, we can touch `select_radius_draw_item` only inside the critical section.
        if (select_radius_draw_item is not None) and (same_pos and same_zoom):
            return
        brush_radius_data_x = gui_config.selection_brush_radius_pixels / pixels_per_data_unit_x  # TODO: what if division by zero?
        brush_radius_data_y = gui_config.selection_brush_radius_pixels / pixels_per_data_unit_y  # TODO: what if division by zero?
        deltas = np.copy(unit_manygon)  # unit circle
        # Convert a circle with a radius of the selection brush size, from pixel space to data space (where x and y axes may have different scalings).
        deltas[:, 0] *= brush_radius_data_x
        deltas[:, 1] *= brush_radius_data_y
        points = (np.array(p) + deltas).tolist()
        select_radius_draw_item = dpg.draw_polygon(points,  # in data space
                                                   color=(255, 255, 255, 255),
                                                   fill=(0, 0, 0, 0),
                                                   parent="plot")  # tag

def _mouse_inside_widget(item):
    """Return whether the mouse cursor is inside the given widget (DPG ID or tag)."""
    try:
        x0, y0 = dpg.get_item_rect_min(item)
    except KeyError:  # some items don't have `rect_min` (e.g. child windows)
        x0, y0 = dpg.get_item_pos(item)
    try:
        w, h = dpg.get_item_rect_size(item)
    except KeyError:  # e.g. child window
        config = dpg.get_item_configuration(item)
        w = config["width"]
        h = config["height"]
    m = dpg.get_mouse_pos(local=False)  # in viewport coordinates
    if m[0] < x0 or m[0] >= x0 + w or m[1] < y0 or m[1] >= y0 + h:
        return False
    return True
def mouse_inside_plot_widget():
    """Return whether the mouse cursor is inside the plot widget."""
    return _mouse_inside_widget("plot")  # tag
def mouse_inside_info_panel():
    """Return whether the mouse cursor is inside the info panel."""
    return _mouse_inside_widget("item_information_panel")  # tag

def mouse_wheel_callback(sender, app_data):
    """Update the plotter data tooltip when the user zooms with the mouse wheel.

    Also, if scrolling the info panel, flash the end when reached.
    """
    # If we reach the end of the info panel, flash it.
    if mouse_inside_info_panel():
        # direction = app_data  # -1 = down, +1 = up  # for documentation only
        current_y_scroll = dpg.get_y_scroll("item_information_panel")  # tag
        info_panel_scroll_end_flasher.show_by_position(current_y_scroll)

    # Zooming in the plotter may change which data points are under the cursor within the tooltip-trigger pixel distance.
    if mouse_inside_plot_widget():
        update_mouse_hover(force=True, wait=True)

lmb_pressed_inside_plot = False  # for tracking whether a drag started inside the plot (to prevent losing selection while scrolling info panel using the scrollbar, with the mouse then entering the plot area while LMB is down)
def mouse_click_callback(sender, app_data):
    """Handle the case where the user selects items by clicking, without moving the mouse."""
    # print(dpg.get_item_type(sender), sender, app_data)  # dpg.get_item_alias(sender), but just printing `sender` shows the alias if it has one, and otherwise the raw numeric ID.

    global lmb_pressed_inside_plot

    if not mouse_inside_plot_widget():  # should not happen; we're an item handler for the plot widget
        lmb_pressed_inside_plot = False
        return

    # `sender` is always the handler registry; `app_data` contains the tag/ID of the actual GUI widget that triggered the event.
    mouse_button, real_sender_ = app_data  # for documentation of `app_data` only

    # Left-click to select
    if mouse_button == dpg.mvMouseButton_Left:
        lmb_pressed_inside_plot = True
        draw_select_radius_indicator()
        update_selection(get_data_idxs_at_mouse(),
                         keyboard_state_to_selection_mode(),
                         wait=False,
                         update_selection_undo_history=False)  # `mouse_release_callback` will commit regardless of if this event is actually a click or a starting mouse-draw

    # Right-click to scroll to item at mouse cursor (if it is shown in the info panel)
    elif mouse_button == dpg.mvMouseButton_Right:
        data_idxs_at_mouse = get_data_idxs_at_mouse()  # item indices into `sorted_xxx`
        if not len(data_idxs_at_mouse):
            return

        # Find items under the mouse cursor that is included in the info panel.
        #   - Consider only items listed in the mouse-hover annotation tooltip. These are stored in `annotation_data_idxs`.
        #   - If a search is active, the item should also match the current search.
        with annotation_content_lock:
            annotation_data_idxs_set = set(annotation_data_idxs)  # performance - better to amortize this here, or O(n) lookup for each `in` test?
            search_string = unbox(search_string_box)
            with info_panel_content_lock:  # we need to access `info_panel_entry_title_widgets`
                if not search_string:  # no search active
                    jumpable_data_idxs = {data_idx for data_idx in data_idxs_at_mouse
                                          if (data_idx in annotation_data_idxs_set) and (data_idx in info_panel_entry_title_widgets)}
                else:
                    search_result_data_idxs_set = set(unbox(search_result_data_idxs_box))
                    jumpable_data_idxs = {data_idx for data_idx in data_idxs_at_mouse
                                          if (data_idx in annotation_data_idxs_set) and (data_idx in search_result_data_idxs_set) and (data_idx in info_panel_entry_title_widgets)}
                if not jumpable_data_idxs:
                    return

                # Then find the item that is listed first in the annotation tooltip, to keep the behavior easily predictable for the user.
                # We can use `annotation_data_idxs`, which has them in that order.
                jump_target_data_idx = next(filter(lambda data_idx: data_idx in jumpable_data_idxs,
                                                   annotation_data_idxs),
                                            None)
                if jump_target_data_idx is None:
                    return

                scroll_info_panel_to_item(info_panel_entry_title_widgets[jump_target_data_idx])

def keydown_callback(sender, app_data):
    """Enable selection brush indicator when the mouse is in the plot area and Shift/Ctrl is held down.

    This gives immediate visual feedback that the "select more" or "select less" mode is active.
    """
    key, time_since_press_ = app_data  # for documentation only

    if not mouse_inside_plot_widget():
        return
    if key in (dpg.mvKey_LControl, dpg.mvKey_RControl, dpg.mvKey_LShift, dpg.mvKey_RShift):
        draw_select_radius_indicator()

def keyup_callback(sender, app_data):
    """Disable selection brush indicator when Shift/Ctrl is released (and the mouse button is not down)."""
    key = app_data  # for documentation only

    if key in (dpg.mvKey_LControl, dpg.mvKey_RControl, dpg.mvKey_LShift, dpg.mvKey_RShift):
        if not dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
            clear_select_radius_indicator()

def mouse_move_callback():
    """Update the relevant GUI elements when the mouse moves.

    Currently these are:
        - Plotter data tooltip.
        - Select radius indicator for mouse-draw select.
    """
    clear_select_radius_indicator()

    if not mouse_inside_plot_widget():
        clear_mouse_hover()
        return
    # We are inside the plot widget.

    # We do the following in likely-fastest-to-likely-slowest order, to refresh each relevant GUI element as quickly as possible.

    # mouse-draw select (but only when drag began inside the plot)
    if lmb_pressed_inside_plot and dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
        draw_select_radius_indicator()
        update_selection(get_data_idxs_at_mouse(),
                         keyboard_state_to_selection_mode(),
                         wait=True,
                         update_selection_undo_history=False)  # mouse release will commit later.

    # plotter data tooltip
    update_mouse_hover(force=False, wait=True)

def mouse_release_callback(sender, app_data):
    """Finalize a mouse-click select or mouse-draw select."""
    global lmb_pressed_inside_plot
    lmb_pressed_inside_plot = False  # finalize the drag

    if not mouse_inside_plot_widget():
        return

    mouse_button = app_data  # for documentation of `app_data` only

    # commit new selection to undo history when mouse-draw select ends
    if mouse_button == dpg.mvMouseButton_Left:
        clear_select_radius_indicator()
        commit_selection_change_to_undo_history()

def hotkeys_callback(sender, app_data):
    """Handle hotkeys."""
    key = app_data  # for documentation only

    # NOTE: If you update this, to make the hotkeys discoverable, update also:
    #  - The tooltips wherever the GUI elements are created or updated (search for e.g. "[F9]", may appear in multiple places)
    #  - The help window

    if key == dpg.mvKey_F11:  # de facto standard hotkey for toggle fullscreen. Should be available whether or not a modal window is open.
        toggle_fullscreen()

    # Hotkeys while the Help card is shown
    elif is_help_window_visible():
        if key == dpg.mvKey_Escape:
            hide_help_window()
        return

    # Hotkeys while the "open file" dialog is shown
    elif is_file_dialog_visible():
        # TODO: Add hotkeys to navigate up/down in the "open file" view, descend into folder, go one level up, go back to default path, ...
        if key == dpg.mvKey_Return:
            filedialog.ok()
        elif key == dpg.mvKey_Escape:
            filedialog.cancel()
        return

    # Hotkeys for main window, while no modal window is shown
    if dpg.is_item_focused("search_field") and key == dpg.mvKey_Return:  # tag  # regardless of modifier state, to allow Shift+Enter and Ctrl+Enter.
        select_search_results()
        dpg.focus_item("item_information_panel")  # tag
    elif dpg.is_item_focused("search_field") and key == dpg.mvKey_Escape:  # tag  # cancel current search edit (handled by the text input internally, by sending a change event; but we need to handle the keyboard focus)
        dpg.focus_item("item_information_panel")  # tag
    elif key == dpg.mvKey_F1:  # de facto standard hotkey for help
        show_help_window()
        dpg.focus_item(help_window)
    elif key == dpg.mvKey_F3:  # some old MS-DOS software in the 1990s used F3 for next/prev search match, I think?
        if (dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)):
            if dpg.is_item_enabled("prev_search_match_button"):  # tag
                scroll_info_panel_to_prev_search_match()
        else:
            if dpg.is_item_enabled("next_search_match_button"):  # tag
                scroll_info_panel_to_next_search_match()
    elif key == dpg.mvKey_F6:  # Use an F-key, because this too has Shift/Ctrl modes.
        search_or_select_current_entry()
    elif key == dpg.mvKey_F7:  # Use an F-key, because this too needs selection mode modifiers.
        select_current_cluster()
    elif key == dpg.mvKey_F8 and dpg.is_item_enabled("copy_report_to_clipboard_button"):  # tag  # NOTE: Shift is a modifier here too
        copy_report_to_clipboard()
    elif key == dpg.mvKey_F9:  # Use an F-key, because this too needs selection mode modifiers.
        select_visible_all()
    # Ctrl+Shift+...
    elif ((dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)) and
          (dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift))):
        if key == dpg.mvKey_Z and dpg.is_item_enabled("selection_undo_button"):  # tag
            selection_undo()
        elif key == dpg.mvKey_Y and dpg.is_item_enabled("selection_redo_button"):  # tag
            selection_redo()
        elif key == dpg.mvKey_C:
            copy_current_entry_to_clipboard()
    # Ctrl+...
    elif dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl):
        if key == dpg.mvKey_F:
            dpg.focus_item("search_field")  # tag
        elif key == dpg.mvKey_O:
            show_open_file_dialog()
        elif key == dpg.mvKey_Home:
            reset_plotter_zoom()
        elif key == dpg.mvKey_N:
            scroll_info_panel_to_next_cluster()
        elif key == dpg.mvKey_P:
            scroll_info_panel_to_prev_cluster()
        elif key == dpg.mvKey_U:
            scroll_info_panel_to_top_of_current_cluster()
        # Some hidden debug features. Mnemonic: "Mr. T Lite" (Ctrl + M, R, T, L)
        elif key == dpg.mvKey_M:
            dpg.show_metrics()
        elif key == dpg.mvKey_R:
            dpg.show_item_registry()
        elif key == dpg.mvKey_T:
            dpg.show_font_manager()
        elif key == dpg.mvKey_L:
            dpg.show_style_editor()
    # Bare key
    #
    # NOTE: These are global across the whole app (when no modal window is open) - be very careful here!
    elif not dpg.is_item_focused("search_field"):  # tag
        if key == dpg.mvKey_Home:
            go_to_top()
        elif key == dpg.mvKey_End:
            go_to_bottom()
        elif key == dpg.mvKey_Next or key == 518:  # page down  # TODO: fix: in DPG 2.0.0, Page Down is no longer "Next" but a mysterious 518 - what is the new name?
            go_page_down()
        elif key == dpg.mvKey_Prior or key == 517:  # page up  # TODO: fix: in DPG 2.0.0, Page Up is no longer "Prior" but a mysterious 517 - what is the new name?
            go_page_up()
        elif key == dpg.mvKey_Down:  # arrow down
            @call
            def _():
                current_y_scroll = dpg.get_y_scroll("item_information_panel")  # tag
                w_info, h_info = dpg.get_item_rect_size("item_information_panel")  # tag
                new_y_scroll = current_y_scroll + 0.1 * h_info
                scroll_info_panel_to_position(new_y_scroll)
        elif key == dpg.mvKey_Up:  # arrow up
            @call
            def _():
                current_y_scroll = dpg.get_y_scroll("item_information_panel")  # tag
                w_info, h_info = dpg.get_item_rect_size("item_information_panel")  # tag
                new_y_scroll = current_y_scroll - 0.1 * h_info
                scroll_info_panel_to_position(new_y_scroll)

# Set up global mouse and keyboard handlers
with dpg.handler_registry(tag="global_handler_registry"):  # global (whole viewport)
    dpg.add_mouse_move_handler(tag="mouse_move_handler", callback=mouse_move_callback)
    dpg.add_mouse_release_handler(tag="mouse_release_handler", callback=mouse_release_callback)
    dpg.add_mouse_wheel_handler(tag="mouse_wheel_handler", callback=mouse_wheel_callback)
    # dpg.add_mouse_click_handler(tag="mouse_click_handler", callback=mouse_click_callback)
    dpg.add_key_press_handler(tag="hotkeys_handler", callback=hotkeys_callback)
    dpg.add_key_down_handler(tag="keydown_handler", callback=keydown_callback)
    dpg.add_key_release_handler(tag="keyup_handler", callback=keyup_callback)

# Inside the plot widget only (but also incorrectly, outside the actual plot area, which doesn't seem to have its on-screen coordinates stored anywhere accessible).
# But see also `dpg.get_item_rect_min`, `dpg.get_mouse_pos(local=False)` https://github.com/hoffstadt/DearPyGui/issues/2311
with dpg.item_handler_registry(tag="plot_handler_registry") as registry:
    # dpg.add_item_hover_handler(callback=mouse_move_callback)
    dpg.add_item_clicked_handler(tag="plot_mouse_click_handler", callback=mouse_click_callback)  # button=dpg.mvMouseButton_Left
dpg.bind_item_handler_registry("plot", registry)  # tag

# --------------------------------------------------------------------------------
# Set up app exit cleanup

# NOTE: In DPG 2.0.0, this works correctly.
# NOTE: In DPG 1.x, if the info panel is updating while the app shuts down, DPG's exit callback doesn't actually trigger, and DPG segfaults.
#   - At least it's not `update_animations`, the same happens also even if we disable that.
#   - Maybe it's because `_update_info_panel` renders GUI stuff from a background thread? Trying to create GUI items while the app shuts down?
def clean_up_at_exit():
    logger.info("App exiting.")
    reset_app_state(_update_gui=False)  # Exiting, GUI might no longer exist when this is called.
dpg.set_exit_callback(clean_up_at_exit)

# --------------------------------------------------------------------------------
# Start the app

logger.info("App bootup...")

parser = argparse.ArgumentParser(description="""Visualize BibTeX data.""",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument('-v', '--version', action='version', version=('%(prog)s ' + __version__))
parser.add_argument(dest='filename', nargs='?', default=None, type=str, metavar='file',
                    help='dataset to open at startup (optional)')
opts = parser.parse_args()

bg = concurrent.futures.ThreadPoolExecutor()  # for info panel and tooltip annotation updates
annotation_task_manager = SequentialTaskManager(name="annotation_update",
                                                executor=bg)
info_panel_task_manager = SequentialTaskManager(name="info_panel_update",
                                                executor=bg)  # can re-use the same executor here to place both kinds of tasks in the same thread pool.

# import sys
# print(dir(sys.modules["__main__"]))  # DEBUG: Check this occasionally to make sure we don't accidentally store any temporary variables in the module-level namespace.

dpg.set_primary_window(main_window, True)  # Make this DPG "window" occupy the whole OS window (DPG "viewport").
dpg.set_viewport_vsync(True)
dpg.show_viewport()

# Load the file optionally provided on the command line
if opts.filename:
    _default_path = os.path.dirname(_to_absolute(opts.filename))
    open_file(opts.filename)
else:
    _default_path = os.getcwd()
    reset_app_state()  # effectively, open a blank dataset
initialize_filedialog(_default_path)

# HACK: Create the dimmer as soon as possible (some time after the first frame so that other GUI elements initialize their sizes).
# The window for the "scroll ends here" animation is also created at frame 10, but via another mechanism (trying to create it each frame, but the implementation blocks it until frame 10).
dpg.set_frame_callback(10, create_info_panel_dimmer_overlay)

logger.info("App render loop starting.")

# We control the render loop manually to have a convenient place to update our GUI animations just before rendering each frame.
while dpg.is_dearpygui_running():
    update_animations()
    dpg.render_dearpygui_frame()
# dpg.start_dearpygui()  # automatic render loop

logger.info("App render loop exited.")

dpg.destroy_context()


def main():  # TODO: we don't really need this; it's just for console_scripts.
    pass

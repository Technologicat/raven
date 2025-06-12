"""DPG utilities for finding a specific GUI widget.

Mostly useful with a scrollable child window that has lots of content.

See Raven-visualizer for real-world examples.
"""

__all__ = ["is_completely_below_target_y", "is_completely_above_target_y",
           "is_partially_below_target_y", "is_partially_above_target_y",
           "find_widget_depth_first", "binary_search_widget"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import functools

import dearpygui.dearpygui as dpg

from . import utils as guiutils

# NOTE: Zero is a valid DPG ID. Hence in our filter functions, we use `None` to mean "no match", and otherwise return the item (DPG tag or ID) as-is.

def is_completely_below_target_y(widget, *, target_y):
    """Return whether `widget` (DPG tag or ID) is completely at or below `target_y`, in viewport coordinates."""
    if widget is None:
        return None
    x0, y0 = guiutils.get_widget_pos(widget)

    # logger.debug(f"is_completely_below_target_y: widget {widget} (tag '{dpg.get_item_alias(widget)}', type {dpg.get_item_type(widget)}): x0, y0 = {x0}, {y0}, target_y = {target_y}, match = {y0 >= target_y}")

    if y0 >= target_y:
        return widget
    return None

def is_completely_above_target_y(widget, *, target_y):
    """Return whether `widget` (DPG tag or ID) is completely above `target_y`, in viewport coordinates."""
    if widget is None:
        return None
    x0, y0 = guiutils.get_widget_pos(widget)
    w, h = guiutils.get_widget_size(widget)

    widget_y_last = y0 + h - 1
    if widget_y_last < target_y:
        return widget
    return None

def is_partially_below_target_y(widget, *, target_y):
    """Return whether `widget` (DPG tag or ID) is at least partially at or below `target_y`, in viewport coordinates."""
    if widget is None:
        return None
    x0, y0 = guiutils.get_widget_pos(widget)
    w, h = guiutils.get_widget_size(widget)

    widget_y_last = y0 + h - 1
    if widget_y_last >= target_y:
        return widget
    return None

def is_partially_above_target_y(widget, *, target_y):
    """Return whether `widget` (DPG tag or ID) is at least partially above `target_y`, in viewport coordinates."""
    if widget is None:
        return None
    x0, y0 = guiutils.get_widget_pos(widget)

    # logger.debug(f"is_partially_above_target_y: widget {widget} (tag '{dpg.get_item_alias(widget)}', type {dpg.get_item_type(widget)}): x0, y0 = {x0}, {y0}, target_y = {target_y}, match = {y0 >= target_y}")

    if y0 < target_y:
        return widget
    return None

def find_widget_depth_first(root, *, accept, skip=None, _indent=0):
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
    # logger.debug(f"find_widget_depth_first: {' ' * _indent}{root} (tag '{dpg.get_item_alias(root)}', type {dpg.get_item_type(root)}): should skip = {(skip(root) is not None) if skip is not None else 'n/a'}, acceptable = {accept(root) is not None}, #children = {len(dpg.get_item_children(root, slot=1))}")
    if root is None:  # Should not happen during recursion, but can happen if accidentally manually called with `root=None`.
        return None
    if (skip is not None) and (skip(root) is not None):
        return None
    if (result := accept(root)) is not None:
        # logger.debug(f"find_widget_depth_first: {' ' * _indent}match found")
        return result
    if dpg.get_item_type(root) == "mvAppItemType::mvGroup":
        for widget in dpg.get_item_children(root, slot=1):
            if (result := find_widget_depth_first(widget, accept=accept, skip=skip, _indent=_indent + 2)) is not None:
                return result
    return None

# TODO: could we do this with the `bisect` stdlib module in the case where we don't need to consider confounders? OTOH, now we have a unified code for both cases.
def binary_search_widget(widgets, *, accept, consider, skip=None, direction="right"):
    """Binary-search a list of DPG GUI widgets to find the first one in the list that satisfies a monotonic criterion.

    The main use case is to quickly (O(log(n))) find the first widget that is currently on-screen in a scrollable window
    that has lots (thousands or more) of widgets.

    `widgets`: List of GUI widgets to be searched (DPG tag or ID for each).

    `accept`: 1-arg callable. The argument is a DPG tag or ID.
              Should return its input if the widget is acceptable as the search result, or `None` otherwise.

              When a visited widget is a group, this is automatically recursively applied depth-first to find any acceptable widget at any level.

    Additionally, `widgets` may contain *confounders*, which are uninteresting widgets we don't want to accept as search results.
    This typically happens when you `widgets = dpg.get_item_children(some_root_widget, slot=1)` - beside the widgets you want,
    the GUI may have separators and such.

    The definition of a confounder is controlled by the following two parameters:

    `consider`: 1-arg callable, optional. The argument is a DPG tag or ID.
                Should return its input if the widget merits looking into as a possible search result, or `None` otherwise.

                This is needed when confounders are present in the data. If you know there are no confounders, use `consider=None`
                to make the search faster. Then also the tree search will be disabled; we only look at the explicitly listed `widgets`.

                If `consider is None`, then `skip` must also be `None`.

                When confounders are present, `consider` is used to give the binary search useful intermediate candidates
                so that the search can proceed.

                When `consider is not None`, and a visited child of `root` is a group, this is automatically recursively
                applied depth-first to find any widget that could be considered, at any level.

                Technically, this is the accept condition for the non-confounder scan (thus terminating the scan, accepting the widget),
                which is tested *after* the `skip` condition below.

                For example, when looking for text, `consider` may accept any non-blank text widget (that was not skipped by `skip`).
                This way any blank (useless) texts as well as e.g. drawlist widgets are not considered.

    `skip`: 1-arg callable, optional. The argument is a DPG tag or ID.
            Should return its input if the widget should be skipped (thus moving on), or `None` otherwise.

            This is used to avoid giving the binary search intermediate candidates that we do not want as a search result.

            Technically, this is the skip condition for the non-confounder scan (confounders should be skipped).
            Skipping a widget skips also its children.

            Widgets that were not skipped are then checked using the `consider` condition, descending into children depth-first.

            This can be used to skip confounder widgets, such as drawlists if interested in text widgets only.

    `direction`: str. Return the widget on which side of the discontinuity, like taking a one-sided limit in mathematics. One of:
        "right": Return the first widget that satisfies `accept(widget)`.
        "left": Return the last widget that does *not* satisfy `accept(widget)`.

    **IMPORTANT**:

    For binary search to work, the acceptability criterion must be monotonic. Without loss of generality, we assume that
    `widgets[0]` is not acceptable, `widgets[-1]` is acceptable, and that there is exactly one jump from unacceptable to acceptable
    somewhere in the list `widgets`.

    Rather than looking for a specific widget, this finds the minimal `j` such that none of `root.children[<j]` are acceptable,
    whereas one of the widgets contained (recursively) somewhere in `root.children[j]` is acceptable.

    Essentially, we look for a jump of the step function where the widget kind changes from unacceptable to acceptable.
    But we allow the list of children to have confounders sprinkled amid the widgets we are actually interested in.
    """
    # search_kind = "first acceptable widget" if direction == "right" else "last non-acceptable widget"
    # logger.debug(f"binary_search_widget: starting, direction = '{direction}'; searching for {search_kind}.")

    # sanity check
    if consider is None and skip is not None:
        raise ValueError(f"When `consider is None`, the `skip` parameter must also be `None`; got skip = {skip}")

    # Trivial case: no widgets
    if not len(widgets):
        # logger.debug("binary_search_widget: `widgets` is empty; no match.")
        return None

    if consider is not None:  # confounders present in data?
        def scan_for_non_confounder(start, *, step, jend=None):  # `step`: the only useful values are +1 (toward right) or -1 (toward left)
            # logger.debug(f"binary_search_widget: non-confounder scan, start = {start}, step = {step}, jend = {jend}")
            if jend is None:
                if step > 0:
                    jend = len(widgets)
                else:
                    jend = 0
            j = start
            while (result := find_widget_depth_first(widgets[j],
                                                     accept=consider,  # NOTE the accept condition - looking for any non-confounder.
                                                     skip=skip)) is None:
                j += step
                if (step > 0 and j >= jend) or (step < 0 and j < jend) or (step == 0):
                    return None, None
            # logger.debug(f"binary_search_widget: non-confounder scan, final j = {j}, result = {result}")
            return j, result
        scan_for_non_confounder_toward_right = functools.partial(scan_for_non_confounder, step=+1)
        scan_for_non_confounder_toward_left = functools.partial(scan_for_non_confounder, step=-1)
    else:
        def no_scan(start, *args, **kwargs):
            return start, widgets[start]
        scan_for_non_confounder_toward_right = no_scan
        scan_for_non_confounder_toward_left = no_scan

    if direction == "right":
        # Trivial case: the first non-confounder widget is acceptable.
        j_left, widget_left = scan_for_non_confounder_toward_right(start=0)
        if (widget_left is not None) and (accept(widget_left) is not None):  # NOTE: `widget_left` may be `None` if no non-confounder was found.
            # logger.debug(f"binary_search_widget: trivial case, direction = 'right': first widget {widget_left} is acceptable; returning that widget.")
            return widget_left

        # Trivial case: no acceptable widget exists.
        j_right, widget_right = scan_for_non_confounder_toward_left(start=len(widgets) - 1)
        if j_left == j_right:
            # logger.debug("binary_search_widget: trivial case, direction = 'right': search collapsed, no acceptable widget; no match.")
            return None
        if (widget_right is None) or (accept(widget_right) is None):
            # logger.debug("binary_search_widget: trivial case, direction = 'right': no acceptable widget exists; no match.")
            return None
    elif direction == "left":  # Mirror-symmetric to the previous case.
        # If the last non-confounder widget is not acceptable, it is the last widget in the dataset that is not acceptable, so we return it.
        j_right, widget_right = scan_for_non_confounder_toward_left(start=len(widgets) - 1)
        if (widget_right is not None) and (accept(widget_right) is None):
            # logger.debug(f"binary_search_widget: trivial case, direction = 'left': last widget {widget_right} is not acceptable; returning that widget.")
            return widget_right

        # If all non-confounder widgets are acceptable (or if there are non non-confounders), no "last unacceptable widget" exists.
        j_left, widget_left = scan_for_non_confounder_toward_right(start=0)
        if j_left == j_right:
            # logger.debug("binary_search_widget: trivial case, direction = 'left': search collapsed, all widgets are acceptable; no match.")
            return None
        if (widget_left is None) or (accept(widget_left) is not None):
            # logger.debug("binary_search_widget: trivial case, direction = 'left': no non-acceptable widget exists; no match.")
            return None
    else:
        raise ValueError(f"Unknown value for `direction` parameter: {direction}. Valid values are 'right' (return first acceptable widget) or 'left' (return last non-acceptable widget).")

    # General case
    #
    # The idea is to have the right (as opposed to left) goalpost to land on the target widget.

    # Preconditions for general case (as well as invariant during iteration)
    assert accept(widget_left) is None, "binary_search_widget: left widget should not be acceptable"
    assert accept(widget_right) is not None, "binary_search_widget: right widget should be acceptable"

    if consider is None:  # no confounders - use a classical binary search.
        iteration = 0  # DEBUG
        while True:
            iteration += 1  # DEBUG
            # logger.debug(f"binary_search_widget: classical mode, iteration {iteration}, j_left = {j_left}, j_right = {j_right}")

            jmid = (j_left + j_right) // 2
            widget_mid = widgets[jmid]
            mid_acceptable = (accept(widget_mid) is not None)
            if mid_acceptable and jmid < j_right:
                # logger.debug("binary_search_widget:        moving right goalpost to mid")
                j_right = jmid
                continue
            if (not mid_acceptable) and jmid > j_left:
                # logger.debug("binary_search_widget:        moving left goalpost to mid")
                j_left = jmid
                continue

            break

    else:  # with confounders
        iteration = 0  # DEBUG
        while True:
            iteration += 1  # DEBUG
            # logger.debug(f"binary_search_widget: confounder mode, iteration {iteration}, j_left = {j_left}, j_right = {j_right}")

            # The classical binary search would test just at this index:
            jmid = (j_left + j_right) // 2

            # But in this case, because we have confounders, we must scan a short distance linearly in both directions to find a widget that is not a confounder.
            jmid_right, widget_right = scan_for_non_confounder_toward_right(start=jmid)
            mid_right_acceptable = (accept(widget_right) is not None)
            jmid_left, widget_left = scan_for_non_confounder_toward_left(start=jmid)
            mid_left_acceptable = (accept(widget_left) is not None)

            # NOTE: To be safe, the right goalpost only goes as far as the *right-side* mid-widget, and the left goalpost as far as the *left-side* mid-widget.
            # Once we have a (very small) range that cannot be safely collapsed further in this manner, we stop the binary search, and perform a final linear scan in that range.
            # logger.debug(f"binary_search_widget:    jmid_left = {jmid_left} (widget {widget_left}, acceptable: {mid_left_acceptable})")
            # logger.debug(f"binary_search_widget:    jmid_right = {jmid_right} (widget {widget_right}, acceptable: {mid_right_acceptable})")
            if mid_right_acceptable and jmid_right < j_right:
                # logger.debug("binary_search_widget:        moving right goalpost to mid-right")
                j_right = jmid_right
                continue
            if (not mid_left_acceptable) and jmid_left > j_left:
                # logger.debug("binary_search_widget:        moving left goalpost to mid-left")
                j_left = jmid_left
                continue

            break

    # Scan the final interval (a few widgets at most) linearly to find the first acceptable one (it's usually the one at `j_right`)
    # logger.debug(f"binary_search_widget: final j_left = {j_left}, j_right = {j_right}")

    # I'm tempted to check the invariant (left widget is not acceptable; right is acceptable),
    # but the matching widget might actually be somewhere inside `widgets[j_right]`, and the whole tree
    # for `widgets[j_left]` should be non-acceptable. No point in performing an extra recursive search here.

    if consider is None:
        if direction == "right":
            # logger.debug(f"binary_search_widget: found first acceptable widget j = {j_right}, widget {widgets[j_right]}")
            return widgets[j_right]
        elif direction == "left":
            # logger.debug(f"binary_search_widget: found last non-acceptable widget j = {j_left}, widget {widgets[j_left]}")
            return widgets[j_left]
    else:
        def consider_and_accept(widget):
            """Check whether `widget` passes both `consider` and `accept` conditions."""
            if (result := consider(widget)) is not None:
                # It is possible that `result is not widget`, in case the actual non-confounder `result` was found somewhere inside the original `widget`.
                return accept(result)
            return None

        # logger.debug("binary_search_widget: final linear scan for result in the detected range")
        if direction == "right":
            for j in range(j_left, j_right + 1):
                widget = widgets[j]
                # logger.debug(f"binary_search_widget: final linear scan: index j = {j}, widget {widget}")
                if (result := find_widget_depth_first(widget,
                                                      accept=consider_and_accept,  # NOTE the accept condition - looking for a widget satisfying the actual search criterion (but is also a non-confounder).
                                                      skip=skip)) is not None:
                    # logger.debug(f"binary_search_widget: final result: found first acceptable widget {result}, returning that widget.")
                    return result
        elif direction == "left":
            for j in range(j_right, j_left - 1, -1):
                widget = widgets[j]
                # logger.debug(f"binary_search_widget: final linear scan: index j = {j}, widget {widget}")
                if find_widget_depth_first(widget,
                                           accept=consider_and_accept,  # NOTE the accept condition - looking for a widget satisfying the actual search criterion (but is also a non-confounder).
                                           skip=skip) is None:
                    # logger.debug(f"binary_search_widget: final result: found last non-acceptable widget {widget}, returning that widget.")
                    return widget
        # the "else" case was checked at the beginning

    raise RuntimeError("binary_search_widget: did not find anything; this should not happen. Maybe preconditions not satisfied, or `accept` function not monotonic?")

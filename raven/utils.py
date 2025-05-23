__all__ = ["clamp", "nonanalytic_smooth_transition", "psi",
           "absolutize_filename", "strip_ext", "make_cache_filename", "validate_cache_mtime", "create_directory",
           "make_blank_index_array",
           "UnionFilter",
           "format_bibtex_author", "format_bibtex_authors", "unicodize_basic_markup",
           "normalize_search_string", "search_string_to_fragments", "search_fragment_to_highlight_regex_fragment",
           "setup_font_ranges", "markdown_add_font_callback",
           "has_child_items", "get_widget_pos", "get_widget_size", "get_widget_relative_pos", "is_mouse_inside_widget", "wait_for_resize",
           "compute_tooltip_position_scalar",
           "get_pixels_per_plotter_data_unit",
           "is_completely_below_target_y", "is_completely_above_target_y",
           "is_partially_below_target_y", "is_partially_above_target_y",
           "find_widget_depth_first", "binary_search_widget"]

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

import functools
import io
import os
import pathlib
import re
from typing import Union
import unicodedata
import warnings

import numpy as np

import dearpygui.dearpygui as dpg

from . import config

# --------------------------------------------------------------------------------
# Numerical utilities

def clamp(x, ell=0.0, u=1.0):  # not the manga studio
    """Clamp value `x` between `ell` and `u`. Return clamped value."""
    return min(max(ell, x), u)

def nonanalytic_smooth_transition(x, m=1.0):  # from `extrafeathers.pdes.numutil`
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

def psi(x, m=1.0):  # from `extrafeathers.pdes.numutil`
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

# --------------------------------------------------------------------------------
# File utilities

def absolutize_filename(filename: str) -> str:
    """Convert `filename` to an absolute filename."""
    return str(pathlib.Path(filename).expanduser().resolve())

def strip_ext(filename: str) -> str:
    """/foo/bar.bib -> /foo/bar"""
    return os.path.splitext(filename)[0]

def make_cache_filename(origfullpath: Union[str, pathlib.Path], suffix: str, ext: str) -> str:
    """foo/bar.bib -> foo/bar_<suffix>.<ext>

    Useful e.g. for naming a cache file based on the input filename.
    """
    origdirname = os.path.dirname(origfullpath)  # "foo/bar.bib" -> "foo"
    origfilename = strip_ext(os.path.basename(origfullpath))  # "foo/bar.bib" -> "bar"
    return os.path.join(origdirname, f"{origfilename}_{suffix}.{ext}")

def validate_cache_mtime(cachefullpath: Union[str, pathlib.Path], origfullpath: Union[str, pathlib.Path]) -> bool:
    """Return whether a cache file at `cachefullpath` is valid, by comparing its mtime to that of the original file at `origfullpath`."""
    stat_result_cache = os.stat(cachefullpath)
    stat_result_orig = os.stat(origfullpath)
    if stat_result_orig.st_mtime_ns <= stat_result_cache.st_mtime_ns:
        return True
    return False

# def delete_directory_recursively(path: str) -> None:
#     """Delete a directory recursively, like 'rm -rf' in the shell.
#
#     Ignores `FileNotFoundError`, but other errors raise. If an error occurs,
#     some files and directories may already have been deleted.
#     """
#     path = pathlib.Path(path).expanduser().resolve()
#
#     for root, dirs, files in os.walk(path, topdown=False, followlinks=False):
#         for x in files:
#             try:
#                 os.unlink(os.path.join(root, x))
#             except FileNotFoundError:
#                 pass
#
#         for x in dirs:
#             try:
#                 os.rmdir(os.path.join(root, x))
#             except FileNotFoundError:
#                 pass
#
#     try:
#         os.rmdir(path)
#     except FileNotFoundError:
#         pass

def create_directory(path: Union[str, pathlib.Path]) -> None:
    p = pathlib.Path(path).expanduser().resolve()
    pathlib.Path.mkdir(p, parents=True, exist_ok=True)

# def clear_and_create_directory(path: str) -> None:
#     delete_directory_recursively(path)
#     create_directory(path)

# --------------------------------------------------------------------------------
# Misc utilities

def make_blank_index_array() -> np.array:
    """Make a blank array of the same type as that used for slicing an array in NumPy."""
    return np.array([], dtype=np.int64)

class UnionFilter(logging.Filter):  # Why isn't this thing in the stdlib?  TODO: general utility, move to `unpythonic`
    def __init__(self, *filters):
        """A `logging.Filter` that matches a record if at least one of the given `*filters` matches it.

        Based on:
            https://stackoverflow.com/questions/17275334/what-is-a-correct-way-to-filter-different-loggers-using-python-logging
            https://docs.python.org/3/library/logging.html#logging.Filter

        For just the current module, one would::

            for handler in logging.root.handlers:
                handler.addFilter(logging.Filter(__name__))

        For more than one module, enter `UnionFilter`. For example::

            for handler in logging.root.handlers:
                handler.addFilter(UnionFilter(logging.Filter(__name__),
                                              logging.Filter("raven.animation"),
                                              logging.Filter("raven.bgtask"),
                                              logging.Filter("raven.preprocess"),
                                              logging.Filter("raven.utils"),
                                              logging.Filter("raven.vendor.file_dialog.fdialog")))
        """
        self.filters = filters
    def filter(self, record):
        return any(f.filter(record) for f in self.filters)

# --------------------------------------------------------------------------------
# String utilities

def format_bibtex_author(author):
    """Format an author name for use in a citation.

    `author`: output of `bibtexparser.middlewares.SplitNameParts`.

    Examples of `author` format, from `bibtexparser/middlewares/names.py`:

        >>> parse_single_name_into_parts("Donald E. Knuth")
        {'last': ['Knuth'], 'von': [], 'first': ['Donald', 'E.'], 'jr': []}

        >>> parse_single_name_into_parts("Brinch Hansen, Per")
        {'last': ['Brinch', 'Hansen'], 'von': [], 'first': ['Per'], 'jr': []}

        >>> parse_single_name_into_parts("Beeblebrox, IV, Zaphod")
        {'last': ['Beeblebrox'], 'von': [], 'first': ['Zaphod'], 'jr': ['IV']}

        >>> parse_single_name_into_parts("Ludwig van Beethoven")
        {'last': ['Beethoven'], 'von': ['van'], 'first': ['Ludwig'], 'jr': []}

    In these examples, we return:

        "Knuth"
        "Brinch Hansen"
        "Beeblebrox IV"
        "van Beethoven"
    """
    if not author.last:
        raise ValueError(f"missing last name in author {author}")
    von_part = f"{' '.join(author.von)} " if author.von else ""
    last_part = f"{' '.join(author.last)}"
    jr_part = f" {' '.join(author.jr)}" if author.jr else ""
    return f"{von_part}{last_part}{jr_part}"

def format_bibtex_authors(authors):
    """Format an author name for use in a citation.

    `author`: a list, where each element is an outputs of `bibtexparser.middlewares.SplitNameParts`.
              For details of that format, see the docstring of `format_bibtex_author`.

    Returns an `str` suitable for use in a citation:
        - One author: "Author"
        - Two authors: "Author and Other"
        - Three or more: "Author et al."

    The authors are kept in the same order as in the original list.
    """
    try:
        authors_list = [format_bibtex_author(author) for author in authors]
    except ValueError as exc:
        logger.warning(f"format_bibtex_authors: failed, reason: {str(exc)}")
        return ""
    if len(authors_list) >= 3:
        authors_str = f"{authors_list[0]} et al."
    elif len(authors_list) == 2:
        authors_str = f"{authors_list[0]} and {authors_list[1]}"
    elif len(authors_list) == 1:
        authors_str = authors_list[0]
    else:  # empty author list
        logger.warning("format_bibtex_authors: got an empty authors list")
        authors_str = ""
    return authors_str

# # https://stackoverflow.com/questions/46501292/normalize-whitespace-with-python
# def normalize_whitespace(s):
#     return " ".join(s.split())

def _substitute_chars(mapping, html_tag_name, match_obj):
    """Substitute characters in a regex match. Low-level function, used by `unicodize_basic_markup`.

    This can be used as a replacer in `re.sub`, e.g. for replacing HTML with Unicode
    in chemical formulas ("CO₂", "NOₓ") and math (e.g. "x²").

    `mapping`: e.g. `regular_to_subscript`; see `config.py`.
    `html_tag_name`: str or None. Name of HTML tag to strip (e.g. "sub").
                     If `None`, omit HTML processing.
    `match_obj`: provided by `re.sub`.

    Example::

        substitute_sub = functools.partial(_substitute_chars, config.regular_to_subscript, "sub")
        text = re.sub(r"<sub>(.*?)</sub>", substitute_sub, text, flags=re.IGNORECASE)
    """
    s = match_obj.group()

    # Strip HTML tag: "<sub>123</sub>" -> "123"
    if html_tag_name is not None:
        tag_start = f"<{html_tag_name}>"
        tag_end = f"</{html_tag_name}>"
        s = s[len(tag_start):-len(tag_end)]

    sio = io.StringIO()
    for c in s:
        sio.write(mapping.get(c, c))  # if `c` in `mapping`, use that, else use `c` itself.
    return sio.getvalue()

def unicodize_basic_markup(s):
    """Convert applicable parts of HTML and LaTeX in `s` to their Unicode equivalents."""
    s = " ".join(unicodedata.normalize("NFKC", s).strip().split())

    # Remove some common LaTeX encodings
    s = s.replace(r"\%", "%")
    s = s.replace(r"\$", "$")

    # Replace some HTML entities
    s = s.replace(r"&apos;", "'")
    s = s.replace(r"&quot;", '"')
    s = s.replace(r"&Auml;", "Ä")
    s = s.replace(r"&auml;", "ä")
    s = s.replace(r"&Ouml;", "Ö")
    s = s.replace(r"&ouml;", "ö")
    s = s.replace(r"&Aring;", "Å")
    s = s.replace(r"&aring;", "å")

    # Replace HTML with Unicode in chemical formulas (e.g. "CO₂", "NOₓ") and math (e.g. "x²")s
    substitute_sub = functools.partial(_substitute_chars, config.regular_to_subscript, "sub")
    substitute_sup = functools.partial(_substitute_chars, config.regular_to_superscript, "sup")
    s = re.sub(r"<sub>(.*?)</sub>", substitute_sub, s, flags=re.IGNORECASE)
    s = re.sub(r"<sup>(.*?)</sup>", substitute_sup, s, flags=re.IGNORECASE)

    # Prettify some HTML for better plaintext readability
    s = re.sub(r"<b>(.*?)</b>", r"*\1*", s, flags=re.IGNORECASE)  # bold
    s = re.sub(r"<i>(.*?)</i>", r"/\1/", s, flags=re.IGNORECASE)  # italic
    s = re.sub(r"<u>(.*?)</u>", r"_\1_", s, flags=re.IGNORECASE)  # underline

    # Replace < and > entities last
    s = s.replace(r"&lt;", "<")
    s = s.replace(r"&gt;", ">")

    return s

def normalize_search_string(s):
    """Normalize a string for searching.

    This converts subscripts and superscripts into their regular equivalents.
    """
    s = " ".join(unicodedata.normalize("NFKC", s).strip().split())
    for k, v in config.subscript_to_regular.items():
        s = s.replace(k, v)
    for k, v in config.superscript_to_regular.items():
        s = s.replace(k, v)
    return s

def search_string_to_fragments(s, *, sort):
    """Convert search string `s` into `(case_sensitive_fragments, case_insensitive_fragments)`.

    `sort`: if `True`, sort the fragments (in each set) from longest to shortest.

    Incremental fragment search, like in Emacs HELM (or in Firefox address bar):
      - "cat photo" matches "photocatalytic".
      - Lowercase search term means case-insensitive for that term (handled in functions
        that perform search, such as `update_search` and `update_info_panel`).
    """
    search_terms = [normalize_search_string(x.strip()) for x in s.split()]
    is_case_sensitive = [x.lower() != x for x in search_terms]
    case_sensitive_fragments = [x for x, sens in zip(search_terms, is_case_sensitive) if sens]
    case_insensitive_fragments = [x for x, sens in zip(search_terms, is_case_sensitive) if not sens]
    if sort:
        case_sensitive_fragments = list(sorted(case_sensitive_fragments, key=lambda x: -len(x)))  # longest to shortest
        case_insensitive_fragments = list(sorted(case_insensitive_fragments, key=lambda x: -len(x)))  # longest to shortest
    return case_sensitive_fragments, case_insensitive_fragments

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

# --------------------------------------------------------------------------------
# GUI font loading utilities

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
    # infinity symbol ∞
    dpg.add_font_chars([0x221e])
    # normal subgroup symbols ⊲, ⊳ (useful also as GUI arrows and similar)
    dpg.add_font_range(0x22b2, 0x22b3)
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
    """Callback for `dpg_markdown` to load a font. Called whenever a new font size or family is needed.

    This calls our `setup_font_ranges` so that special characters work.
    """
    if not isinstance(size, (int, float)):
        raise ValueError(f"markdown_add_font_callback: `size`: expected `int` or `float`, got `{type(size)}` with value `{size}`")
    with dpg.font(file, size, parent=parent, **kwargs) as font:
        setup_font_ranges()
    return font

# --------------------------------------------------------------------------------
# GUI utilities

def has_child_items(widget):
    """Return whether `widget` (DPG tag or ID) has child items in any of its slots."""
    for slot in range(4):
        if len(dpg.get_item_children(widget, slot=slot)):
            return True
    return False

def get_widget_pos(widget):
    """Return `widget`'s (DPG tag or ID) position `(x0, y0)`, in viewport coordinates.

    This papers over the fact that most items support `dpg.get_item_rect_min`,
    but e.g. with child windows, one needs to use `dpg.get_item_pos` instead.
    """
    try:
        x0, y0 = dpg.get_item_rect_min(widget)
    except KeyError:  # some items don't have `rect_min` (e.g. child windows)
        x0, y0 = dpg.get_item_pos(widget)
    return x0, y0

def get_widget_size(widget):
    """Return `widget`'s (DPG tag or ID) on-screen size `(width, height)`, in pixels.

    This papers over the fact that most items support `dpg.get_item_rect_size`,
    but e.g. child windows store their size in the item configuration instead.
    """
    try:
        w, h = dpg.get_item_rect_size(widget)
    except KeyError:  # e.g. child window
        config = dpg.get_item_configuration(widget)
        w = config["width"]
        h = config["height"]
    return w, h

def get_widget_relative_pos(widget, reference):
    """Return `widget`'s (DPG tag or ID) position, measured relative to the `reference` widget (DPG tag or ID).

    This is handy when you need child window coordinates (use the child window as `reference`).
    """
    x0, y0 = get_widget_pos(widget)  # in viewport coordinates  # tag
    x0_c, y0_c = get_widget_pos(reference)  # in viewport coordinates
    x0_local = x0 - x0_c
    y0_local = y0 - y0_c
    return x0_local, y0_local

def is_mouse_inside_widget(widget):
    """Return whether the mouse cursor is inside `widget` (DPG ID or tag)."""
    x0, y0 = get_widget_pos(widget)
    w, h = get_widget_size(widget)
    m = dpg.get_mouse_pos(local=False)  # in viewport coordinates
    if m[0] < x0 or m[0] >= x0 + w or m[1] < y0 or m[1] >= y0 + h:
        return False
    return True

def wait_for_resize(widget, wait_frames_max=10):
    """Wait (calling `dpg.split_frame()`) until the on-screen size of `widget` (DPG tag or ID) changes.

    If `wait_frames_max` frames have elapsed without the size changing, return.

    Return `True` if the size changed, `False` otherwise.
    """
    waited = 0
    old_size = get_widget_size(widget)
    while waited < wait_frames_max:
        dpg.split_frame()  # let the autosize happen
        waited += 1

        new_size = get_widget_size(widget)
        if new_size != old_size:
            logger.debug(f"wait_for_resize: waited {waited} frame{'s' if waited != 1 else ''} for resize of DPG widget {widget}")
            return True
    else:
        logger.debug(f"wait_for_resize: timeout ({wait_frames_max} frames) when waiting for resize of DPG widget {widget}")
    return False

def compute_tooltip_position_scalar(*, algorithm, cursor_pos, tooltip_size, viewport_size, offset=20):
    """Compute x or y position for a tooltip. (Either one of them; hence "scalar".)

    This positions the tooltip elegantly, trying to keep it completely within the DPG viewport area.
    This is mostly useful for tooltips triggered by custom code, such as for a scatterplot dataset in a plotter.

    `algorithm`: str, one of "snap", "snap_old", "smooth".
                 "snap": Right/bottom side if the tooltip fits there, else left/top side.
                 "snap_old": Right/bottom side when the cursor is at the left/top side of viewport, else left/top side.
                 "smooth": Cursor at left edge -> right/bottom side; cursor at right edge -> left/top side; in between,
                           smoothly varying as a function of the cursor position. For the perfectionists.

                 If unsure, try "snap" for the x coordinate, and "smooth" for the y coordinate; usually looks good.

    `cursor_pos`: tuple `(x, y)`, mouse cursor position, in viewport coordinates.
    `tooltip_size`: tuple `(width, height)`, size of the tooltip window, in pixels.
    `viewport_size`: tuple `(width, height)`, size of the DPG viewport (or equivalently, primary window), in pixels.
    `offset`: int. This allows positioning the tooltip a bit off from `cursor_pos`, so that the mouse cursor won't
              immediately hover over it when the tooltip is shown.

              This is important, because in DPG a tooltip is a separate window, so this would prevent further
              mouse hover events of the actual window under the tooltip from being triggered (until the mouse
              exits the tooltip area).

    Usage::

        mouse_pos = dpg.get_mouse_pos(local=False)  # in viewport coordinates
        tooltip_size = dpg.get_item_rect_size(my_tooltip_window)  # after `dpg.split_frame()` if needed
        w, h = dpg.get_item_rect_size(my_primary_window)
        xpos = compute_tooltip_position_scalar(algorithm="snap",
                                               cursor_pos=mouse_pos[0],
                                               tooltip_size=tooltip_size[0],
                                               viewport_size=w)
        ypos = compute_tooltip_position_scalar(algorithm="smooth",
                                               cursor_pos=mouse_pos[1],
                                               tooltip_size=tooltip_size[1],
                                               viewport_size=h)
        dpg.set_item_pos(my_tooltip_window, [xpos, ypos])
    """
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

def get_pixels_per_plotter_data_unit(plot_widget, xaxis, yaxis):
    """Estimate pixels per DPG plotter data unit, for conversion between viewport space and data space.

    `plot_widget`: dpg tag or ID, the plotter widget (`dpg.plot`).
    `xaxis`: dpg tag or ID, the x axis widget of the plotter (`dpg.add_plot_axis(dpg.mvXAxis, ...)`).
    `yaxis`: dpg tag or ID, the y axis widget of the plotter (`dpg.add_plot_axis(dpg.mvYAxis, ...)`).

    This is subtly wrong, because the plot widget includes also the space for the axis labels and such.
    But there seems to be no way to get pixels per data unit from a plot in DPG (unless using a custom series,
    which we don't).

    Raven uses this for estimation of on-screen distances in data space. For that purpose this is good enough.

    Returns the tuple `(pixels_per_data_unit_x, pixels_per_data_unit_y)`.

    Note that if axes are not equal aspect, the x/y results may be different.
    """
    # x0, y0 = dpg.get_item_rect_min("plot")
    pixels_w, pixels_h = dpg.get_item_rect_size(plot_widget)
    xmin, xmax = dpg.get_axis_limits(xaxis)  # in data space
    ymin, ymax = dpg.get_axis_limits(yaxis)  # in data space
    data_w = xmax - xmin
    data_h = ymax - ymin
    if data_w == 0 or data_h == 0:  # no data in view (can happen e.g. if the plot hasn't been rendered yet)
        return [0.0, 0.0]
    pixels_per_data_unit_x = pixels_w / data_w
    pixels_per_data_unit_y = pixels_h / data_h
    return pixels_per_data_unit_x, pixels_per_data_unit_y

# --------------------------------------------------------------------------------
# GUI widget search utilities

# NOTE: Zero is a valid DPG ID. Hence in our filter functions, we use `None` to mean "no match", and otherwise return the item (DPG tag or ID) as-is.

def is_completely_below_target_y(widget, *, target_y):
    """Return whether `widget` (DPG tag or ID) is completely at or below `target_y`, in viewport coordinates."""
    if widget is None:
        return None
    x0, y0 = get_widget_pos(widget)

    # logger.debug(f"is_completely_below_target_y: widget {widget} (tag '{dpg.get_item_alias(widget)}', type {dpg.get_item_type(widget)}): x0, y0 = {x0}, {y0}, target_y = {target_y}, match = {y0 >= target_y}")

    if y0 >= target_y:
        return widget
    return None

def is_completely_above_target_y(widget, *, target_y):
    """Return whether `widget` (DPG tag or ID) is completely above `target_y`, in viewport coordinates."""
    if widget is None:
        return None
    x0, y0 = get_widget_pos(widget)
    w, h = get_widget_size(widget)

    widget_y_last = y0 + h - 1
    if widget_y_last < target_y:
        return widget
    return None

def is_partially_below_target_y(widget, *, target_y):
    """Return whether `widget` (DPG tag or ID) is at least partially at or below `target_y`, in viewport coordinates."""
    if widget is None:
        return None
    x0, y0 = get_widget_pos(widget)
    w, h = get_widget_size(widget)

    widget_y_last = y0 + h - 1
    if widget_y_last >= target_y:
        return widget
    return None

def is_partially_above_target_y(widget, *, target_y):
    """Return whether `widget` (DPG tag or ID) is at least partially above `target_y`, in viewport coordinates."""
    if widget is None:
        return None
    x0, y0 = get_widget_pos(widget)

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

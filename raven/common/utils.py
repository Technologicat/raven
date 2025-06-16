"""Miscellaneous general utilities."""

__all__ = ["absolutize_filename", "strip_ext", "make_cache_filename", "validate_cache_mtime", "create_directory",
           "make_blank_index_array",
           "UnionFilter",
           "format_bibtex_author", "format_bibtex_authors", "unicodize_basic_markup",
           "normalize_search_string", "search_string_to_fragments", "search_fragment_to_highlight_regex_fragment"]

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

import numpy as np

from . import stringmaps

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
    substitute_sub = functools.partial(_substitute_chars, stringmaps.regular_to_subscript, "sub")
    substitute_sup = functools.partial(_substitute_chars, stringmaps.regular_to_superscript, "sup")
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
    for k, v in stringmaps.subscript_to_regular.items():
        s = s.replace(k, v)
    for k, v in stringmaps.superscript_to_regular.items():
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
        s = s.replace(digit, f"({digit}|{stringmaps.regular_to_subscript_numbers[digit]}|{stringmaps.regular_to_superscript_numbers[digit]})")
    return s

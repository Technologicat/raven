"""Miscellaneous general utilities."""

__all__ = ["absolutize_filename", "strip_ext", "make_cache_filename", "validate_cache_mtime", "create_directory",
           "make_blank_index_array",
           "UnionFilter",
           "environ_override",
           "format_bibtex_author", "format_bibtex_authors",
           "normalize_whitespace", "normalize_unicode",
           "unicodize_basic_markup",
           "normalize_search_string", "search_string_to_fragments", "search_fragment_to_highlight_regex_fragment",
           "chunkify_text"]

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

import contextlib
import functools
import io
import os
import pathlib
import re
import threading
from typing import Callable, Dict, List, Optional, Union
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

class UnionFilter(logging.Filter):  # Why isn't this thing in the stdlib?  TODO: very general utility, move to `unpythonic`
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
                                              logging.Filter("raven.common.gui.animation"),
                                              logging.Filter("raven.common.bgtask"),
                                              logging.Filter("raven.common.utils"),
                                              logging.Filter("raven.visualizer.importer"),
                                              logging.Filter("raven.vendor.file_dialog.fdialog")))
        """
        self.filters = filters
    def filter(self, record):
        return any(f.filter(record) for f in self.filters)

_environ_lock = threading.Lock()
@contextlib.contextmanager
def environ_override(**bindings):  # TODO: very general utility, move to `unpythonic`
    """Context manager: Temporarily override OS environment variable(s).

    When the `with` block exits, the previous state of the environment is restored.

    Thread-safe, but blocks if the lock is already taken - only one set of overrides
    can be active at any one time.
    """
    with _environ_lock:
        # remember old values, if any
        old_bindings = {key: os.environ[key] for key in bindings.keys() if key in os.environ}
        try:
            # apply overrides
            for key, value in bindings.items():
                os.environ[key] = value
            # let the caller do its thing
            yield
        finally:
            # all done - restore old environment
            for key in bindings.keys():
                if key in old_bindings:  # restore old value
                    os.environ[key] = old_bindings[key]
                else:  # this key wasn't there in the previous state, so pop it
                    os.environ.pop(key)

# --------------------------------------------------------------------------------
# BibTeX utilities

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

# --------------------------------------------------------------------------------
# String utilities

def normalize_whitespace(s: str) -> str:
    """Normalize whitespace in a string, by replacing any consecutive whitespace by a single space.
    """
    # # https://stackoverflow.com/questions/46501292/normalize-whitespace-with-python
    return " ".join(s.strip().split())

def normalize_unicode(s: str) -> str:  # SillyTavern-extras/server.py
    """Normalize a Unicode string.

    Convert `s` into NFKC form (see `unicodedata.normalize`).
    """
    # https://stackoverflow.com/questions/16467479/normalizing-unicode
    return unicodedata.normalize("NFKC", s)

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
    """Convert simple HTML/LaTeX markup into Unicode, as far as reasonably possible.

    Apply `normalize_unicode` (which see), and then convert
    applicable parts of HTML and LaTeX (e.g. superscripts, subscripts)
    to their Unicode equivalents.
    """
    s = normalize_unicode(s)

    # Remove some common LaTeX encodings
    s = s.replace(r"\%", "%")
    s = s.replace(r"\$", "$")

    # Replace some HTML entities
    s = s.replace(r"&le;", "≤")
    s = s.replace(r"&ge;", "≥")
    s = s.replace(r"&apos;", "'")
    s = s.replace(r"&quot;", '"')
    s = s.replace(r"&Auml;", "Ä")
    s = s.replace(r"&auml;", "ä")
    s = s.replace(r"&Ouml;", "Ö")
    s = s.replace(r"&ouml;", "ö")
    s = s.replace(r"&Aring;", "Å")
    s = s.replace(r"&aring;", "å")

    # Replace HTML with Unicode in chemical formulas (e.g. "CO₂", "NOₓ") and math (e.g. "x²")
    substitute_sub = functools.partial(_substitute_chars, stringmaps.regular_to_subscript, "sub")
    substitute_sup = functools.partial(_substitute_chars, stringmaps.regular_to_superscript, "sup")
    s = re.sub(r"<sub>(.*?)</sub>", substitute_sub, s, flags=re.IGNORECASE)
    s = re.sub(r"<sup>(.*?)</sup>", substitute_sup, s, flags=re.IGNORECASE)

    # Prettify some HTML for better plaintext readability
    s = re.sub(r"<b>(.*?)</b>", r"*\1*", s, flags=re.IGNORECASE)  # bold
    s = re.sub(r"<i>(.*?)</i>", r"/\1/", s, flags=re.IGNORECASE)  # italic
    s = re.sub(r"<u>(.*?)</u>", r"_\1_", s, flags=re.IGNORECASE)  # underline

    # Replace < and > entities last (so that HTML tags process correctly)
    s = s.replace(r"&lt;", "<")
    s = s.replace(r"&gt;", ">")

    return s

def normalize_search_string(s):
    """Normalize a string for use in text search.

    Apply `normalize_unicode` and then `normalize_whitespace` (which see).
    Then convert subscripts and superscripts into their regular equivalents.
    E.g. "O₂" -> "O2",  "x²" -> "x2".
    """
    # TODO: search string normalization: we could additionally apply the `dehyphen` package here.
    s = normalize_whitespace(normalize_unicode(s))
    for k, v in stringmaps.subscript_to_regular.items():
        s = s.replace(k, v)
    for k, v in stringmaps.superscript_to_regular.items():
        s = s.replace(k, v)
    return s

def search_string_to_fragments(s, *, sort):
    """Convert search string `s` into `(case_sensitive_fragments, case_insensitive_fragments)`.

    This first applies `normalize_search_string`, which see.

    `sort`: if `True`, sort the fragments (in each set) from longest to shortest.

    Incremental fragment search, like in Emacs HELM, or in Firefox address bar:
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

def chunkify_text(text: str, chunk_size: int, overlap: int, extra: float, trimmer: Optional[Callable] = None) -> List[Dict]:
    """Sliding-window text chunker with overlap, e.g. for chunking documents for fine-grained search.

    See also `raven.librarian.hybridir.merge_contiguous_spans`, which does unchunking (the inverse operation)
    for its search results.

    `text`: The text to be chunked.

    `chunk_size`: The length of one chunk, in characters (technically, Unicode codepoints,
                  because Python's internal string format).

                  The final chunk may be up to `extra` larger, to avoid leaving a very short chunk at the end
                  (if the length of `text` did not divide well with `chunk_size`).

    `extra`:   Orphan control parameter, as fraction of `chunk_size`, to avoid leaving a very small amount
               of text into a chunk of its own at the end of the document (in the common case where the length
               of the document does not divide evenly by `chunk_size`).

               E.g. `extra=0.4` allows placing an extra 40% of `chunk_size` of text into the last chunk of the
               document. Hence the remainder of text at the end of the document is split into a separate small
               chunk only if that extra 40% is not enough to accommodate it. If it fits into that, we instead
               make the previous chunk larger (by up to 40%), and place the remainder there.

    `overlap`: How much of the end of the previous chunk should be included in the next chunk,
               to avoid losing context at the seams.

               E.g. if `chunk_size` is 2000 characters and you want a 25% overlap, set `overlap=500`.

               For non-overlapping fixed-size chunking, set `overlap=0`.

    `trimmer`: Optional callback to clean up the start/end of a chunk, e.g. to a whole-sentence
               or whole-word boundary.

               Signature: str -> (str, int)

               The `trimmer` receives three arguments:
                  `overlap`: the `overlap` argument above, passed through.
                             You'll need this if you want to trim at the beginning of the chunk (see below).
                  `mode`: one of "first", "middle", "last"
                          "first" means this is the first chunk, so the beginning MUST NOT be trimmed.
                          "middle" means this chunk is in anywhere in the middle.
                          "last" means this is the last chunk, so the end MUST NOT be trimmed.
                  `text`: the text of the chunk before trimming

               The `trimmer` must return a tuple `(trimmed_chunk, offset)`, where `offset` means
               how many characters were trimmed from the beginning. If you trimmed the end only,
               then return `offset=0`.

               Trim only, DO NOT make any other edits!

               Note that when a trimmer is in use:
                   - The final size of any given chunk, after trimming, may be smaller than `chunk_size`.
                   - `overlap` is counted backward from the end of the *trimmed* chunk.
                   - If the beginning is trimmed more than there is overlap, then some text will be dropped.
                     It is highly recommended to avoid doing so.

               An NLP pipeline can be useful as a component for building a high-quality trimmer.

    Returns a list of chunks of the form
        `{"text": actual_content, "chunk_id": running_number, "offset": start_offset_in_original_text}`.

    The `chunk_id` is provided primarily just for information and for debugging.
    The chunks are numbered 0, 1, ...

    The offsets can be used e.g. for unchunking search results (see `merge_contiguous_spans`
    in `raven.librarian.hybridir` for an example).

    If `text` is at most `chunk_size` characters in length, returns a single chunk in the same format.
    """
    # TODO: better `extra` mechanism: adjust chunk size instead, to spread the extra content evenly?

    if len(text) <= (1 + extra) * chunk_size:
        return [{"text": text, "chunk_id": 0, "offset": 0}]

    chunks = []
    chunk_id = 0
    start = 0
    is_last = False
    while start < len(text):
        if len(text) - start <= (1 + extra) * chunk_size:
            chunk = text[start:]
            is_last = True
        else:
            chunk = text[start:start + chunk_size]

        if trimmer is not None:
            if start == 0:
                mode = "first"
            elif is_last:
                mode = "last"
            else:
                mode = "middle"
            chunk, offset = trimmer(overlap, mode, chunk)
            start = start + offset

        chunks.append({"text": chunk,
                       "chunk_id": chunk_id,
                       "offset": start})
        if is_last:
            break
        delta = len(chunk) - overlap
        if delta <= 0:
            assert False
        start += delta
        chunk_id += 1
    return chunks

"""Miscellaneous general utilities."""

__all__ = ["absolutize_filename", "canonical_path",
           "strip_ext", "make_cache_filename", "validate_cache_mtime", "create_directory",
           "user_directory",
           "open_file", "open_in_file_manager",
           "make_blank_index_array", "bail",
           "notify",
           "bibtex_header_key", "bibtex_field_value", "bibtex_unbalanced_field_names",
           "bibtex_brace_repair_candidates",
           "format_bibtex_author", "format_bibtex_authors",
           "normalize_whitespace", "normalize_unicode",
           "unicodize_basic_markup",
           "normalize_search_string", "search_string_to_fragments", "make_search_matcher",
           "search_fragment_to_highlight_regex_fragment",
           "chunkify_text"]

import logging
logger = logging.getLogger(__name__)

import atexit
import functools
import io
import itertools
import os
import pathlib
import re
import subprocess
import sys
from typing import Any, Callable, Dict, List, NoReturn, Optional, Union
import unicodedata

import numpy as np

from . import stringmaps

# --------------------------------------------------------------------------------
# File utilities

def absolutize_filename(filename: str) -> str:
    """Convert `filename` to an absolute filename, following symlinks.

    Use when you want the *file itself*, however it was reached — deduplicating two paths that name one
    file, or recording where something really lives. Use `canonical_path` instead when the path is an
    identity within a directory tree, since resolving can move it out of that tree.
    """
    return str(pathlib.Path(filename).expanduser().resolve())

def canonical_path(path: Union[str, pathlib.Path]) -> pathlib.Path:
    """Convert `path` to an absolute, lexically normalized path, **without** following symlinks.

    The counterpart to `absolutize_filename`, and the right one whenever a path serves as an *identity
    relative to some root* — a document id built as "where this sits under the documents directory", a
    check that a file lies within a permitted tree. Resolving a symlink can relocate the path outside
    that root, so the identity is lost and the containment check fails on exactly the arrangement it
    was meant to allow.

    Nothing is given up by not resolving: the OS follows symlinks on open and on `os.stat`, so reading
    a file or asking its size and mtime through the unresolved path still reaches the real one.

    `..` segments are removed lexically (`os.path.abspath`), so the result never escapes upward through
    a symlinked directory the way a naive join would.
    """
    return pathlib.Path(os.path.abspath(pathlib.Path(path).expanduser()))

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
    return stat_result_orig.st_mtime_ns <= stat_result_cache.st_mtime_ns

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

# The XDG names for the directories a "places" shortcut list offers, keyed by the English folder name that
# is also the fallback. `XDG_DOWNLOAD_DIR` is singular where the others are plural; that is the spec, not a
# typo here.
_XDG_USER_DIRS = {"Desktop": "XDG_DESKTOP_DIR",
                  "Downloads": "XDG_DOWNLOAD_DIR",
                  "Documents": "XDG_DOCUMENTS_DIR",
                  "Music": "XDG_MUSIC_DIR",
                  "Pictures": "XDG_PICTURES_DIR",
                  "Videos": "XDG_VIDEOS_DIR"}

def _xdg_user_dirs_file() -> pathlib.Path:
    """Where the XDG user-directory definitions live."""
    config_home = os.environ.get("XDG_CONFIG_HOME") or "~/.config"
    return pathlib.Path(config_home).expanduser() / "user-dirs.dirs"

def _read_xdg_user_dir(xdg_key: str) -> pathlib.Path | None:
    """Look `xdg_key` up in the user's `user-dirs.dirs`, or `None` if it is not answered there.

    The file is shell fragments meant to be sourced — `XDG_PICTURES_DIR="$HOME/Kuvat"` — so an exported
    value takes precedence over the file, that being what sourcing it would have produced.
    """
    from_env = os.environ.get(xdg_key)
    if from_env:
        return pathlib.Path(os.path.expandvars(from_env)).expanduser()

    path = _xdg_user_dirs_file()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:  # no such file, or unreadable — neither is exceptional, the file is optional
        return None

    for line in lines:
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() != xdg_key:
            continue
        value = value.strip().strip('"').strip("'")
        if not value:
            return None  # present but unset: nothing to say, so fall back like a missing key
        return pathlib.Path(os.path.expandvars(value)).expanduser()
    return None

def user_directory(name: str) -> pathlib.Path:
    """Where the user's `name` directory actually is. `name` is the English folder name — "Pictures", "Home", ...

    Accepts "Home" and the six keys of `_XDG_USER_DIRS`; anything else resolves to `~/<name>`, which is also
    what every name falls back to when the platform has nothing better to say.

    The returned path is not checked for existence — a user may genuinely have no `Videos` directory, and
    whether that means "skip this shortcut" or "offer it anyway" is the caller's decision, not this one's.
    """
    # Only Linux and the BSDs rename these directories on disk; `~/Pictures` is `~/Kuvat` on a Finnish
    # desktop, and joining `~` with an English name finds nothing. Windows and macOS localize the *displayed*
    # name and leave the directory itself in English, so the fallback is the right answer there.
    home = pathlib.Path("~").expanduser()
    if name == "Home":
        return home
    if sys.platform not in ("win32", "darwin"):
        xdg_key = _XDG_USER_DIRS.get(name)
        if xdg_key is not None:
            from_xdg = _read_xdg_user_dir(xdg_key)
            if from_xdg is not None:
                return from_xdg
    return home / name

def _os_open(path: str | pathlib.Path) -> None:
    """Hand `path` to the operating system's default open mechanism.

    Cross-platform dispatch: `xdg-open` (Linux/*BSD), `open` (macOS), `os.startfile` (Windows). The OS handler
    decides what "open" means for the path — a file opens in its default application, a directory in the file
    manager — so `open_file` and `open_in_file_manager` share this one dispatcher and differ only in the intent
    they name at their call sites.

    Raises `OSError` on any failure, so a caller has a single type to catch and turn into a non-intrusive
    message. That covers a missing target (`FileNotFoundError`, an `OSError` subclass), a missing opener binary
    (likewise), and the opener running but reporting failure — no handler registered for the type — which the
    subprocess helpers surface as `CalledProcessError`; that one is not an `OSError`, so it is caught and
    re-raised as one (`raise from`, cause preserved).
    """
    resolved = pathlib.Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"_os_open: no such path: '{resolved}'")
    target = str(resolved)
    try:
        if sys.platform == "win32":
            os.startfile(target)  # Windows-only API; this branch never runs on other platforms
        elif sys.platform == "darwin":
            subprocess.run(["open", target], check=True)
        else:
            subprocess.run(["xdg-open", target], check=True)
    except subprocess.CalledProcessError as exc:
        raise OSError(f"_os_open: the platform opener failed on '{target}' (exit code {exc.returncode}).") from exc

def open_file(path: str | pathlib.Path) -> None:
    """Open a file in the operating system's default application for its type (image viewer, PDF reader, ...).

    Cross-platform (see `_os_open`). Raises `OSError` if the file is gone or no application is registered for
    the type — callers surface that as a transient message, not a modal dialog.
    """
    _os_open(path)

def open_in_file_manager(path: str | pathlib.Path) -> None:
    """Open a directory in the operating system's file manager.

    Cross-platform (see `_os_open`). Intended for directories (the RAG document drop folder, a chat's datastore
    directory, the image-sidecar directory); passing a file would open it in its default application instead.
    Raises `OSError` if the directory is gone or the platform launch fails.
    """
    _os_open(path)

# --------------------------------------------------------------------------------
# Misc utilities

def make_blank_index_array() -> np.array:
    """Make a blank array of the same type as that used for slicing an array in NumPy."""
    return np.array([], dtype=np.int64)

def bail(exitcode: int = 0) -> NoReturn:
    """Terminate the calling process immediately, skipping interpreter finalization.

    Runs registered `atexit` handlers explicitly (Librarian chat persistence,
    `logging.shutdown`, ...), then hard-exits via `os._exit`.

    This sidesteps C-extension teardown, which SIGBUSes on macOS during DearPyGui's
    static destructor / dlclose phase — the SIGBUS otherwise triggers the macOS crash
    reporter dialog.
    """
    atexit._run_exitfuncs()   # honor the atexit contract before bailing
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exitcode)

# --------------------------------------------------------------------------------
# Observer callbacks

def notify(what: str,
           maybe_callback: Callable | None,
           *args,
           _default: Any = None,
           _reraise: tuple[type[BaseException], ...] = (),
           **kwargs) -> Any:
    """Call an observer callback, treating a failure in it as the observer's problem, not the caller's.

    `what`: the callback's parameter name, for the log line.
    `maybe_callback`: the observer's function, or `None` for "nobody is listening".
    `*args`, `**kwargs`: passed to the callback.
    `_default`: returned in place of the callback's answer when it is absent or raised.
    `_reraise`: exception types to let through, for the ones that are not observer failures at all —
                a cancellation arriving through whichever frame happens to be on the stack, typically.

    Returns the callback's answer, or `_default`. Anything raised outside `_reraise` is logged with its
    traceback and swallowed.
    """
    # The leading underscores are load-bearing. This forwards keyword arguments to somebody else's
    # function, so every name in its own signature is a name the callback cannot use — and since Raven's
    # convention is to pass by name, that is a real collision rather than a theoretical one: `default` is
    # an ordinary thing for a callback to be given. The underscore marks these two as the wrapper's own.
    # For an API that reports progress to whoever asked for the work — an event-callback protocol, of which
    # Raven has several. The observer is watching the work; it is not doing it. So a fault in one costs its
    # own output and nothing else, and the operation it was watching finishes and returns its result.
    #
    # Letting one abort the operation instead is worse than the missing notification twice over: the work is
    # lost, and it is lost with a diagnosis pointing at whatever the operation does rather than at the
    # listener. A caller that classifies a raised exception by *what it was doing* will then record a
    # failure of that, which is how a chat reply came to be replaced by a backend-error message on
    # 2026-08-27 after a GUI repaint hit a stale widget.
    #
    # The `reraise` list is what keeps this from being a blanket `except Exception` around user code: it is
    # for exceptions that are *control*, not failure, and are therefore not the observer's to be blamed for.
    if maybe_callback is None:
        return _default
    try:
        return maybe_callback(*args, **kwargs)
    except _reraise:
        raise
    except Exception:
        logger.exception(f"notify: the `{what}` callback raised; continuing without it. Traceback follows.")
        return _default

# --------------------------------------------------------------------------------
# BibTeX utilities

# --------------------------------------------------------------------------------
# Reading BibTeX by its surface syntax, for when the parser has refused
#
# These are not a second BibTeX parser and must not grow into one. They exist for the case where
# `bibtexparser` has already declined a record - a stray brace in a field value ends the value early and
# takes the rest of the record with it - and something useful can still be said about the wreckage. Three
# callers, wanting three different things from the same shape:
#
#   - `papers.burstbib`, naming a record from its header line, which it does before parsing anything.
#   - `librarian.chatutil`, salvaging a title so a document still gets a legible label.
#   - `visualizer.importer`, naming the fields whose braces look wrong, so the user can go and fix the data.
#
# They live here rather than in `papers` because two of the three callers are not paper tooling.

def bibtex_header_key(headerline: str) -> str:
    """Get the record key out of a BibTeX header line: `@article{WOS:000123,` -> `WOS:000123`.

    Returns `""` if `headerline` has no key to give. Verbatim - see `papers.burstbib.get_slug` for the
    variant sanitized for use as a filename.
    """
    start = headerline.find("{")
    end = headerline.rfind(",")
    return headerline[start + 1:end] if -1 < start < end else ""

# One `Key = ` at the start of a line, which is the shape every BibTeX writer in practice emits.
_BIBTEX_FIELD_LINE = re.compile(r"^\s*([A-Za-z][\w-]*)\s*=")

def bibtex_field_value(text: str, key: str) -> str:
    """Get the value of the BibTeX field `key` from `text`, by pattern. `""` if there is none to find.

    Case-insensitive, because the key case is not dependable: a Web of Science export writes `Title = {...}`,
    the BibTeX literature writes `title = {...}`. Reads only a value that opens and closes on its own line,
    which is what the writers emit and what keeps this from turning into a parser.
    """
    pattern = re.compile(r"^\s*" + re.escape(key) + r"\s*=\s*\{(.*)\}\s*,?\s*$",
                         re.IGNORECASE | re.MULTILINE)
    match = pattern.search(text)
    return match.group(1).strip("{} ") if match else ""

def bibtex_unbalanced_field_names(text: str) -> List[str]:
    """Names of the fields in `text` whose own line opens more braces than it closes.

    A shortlist of suspects rather than a verdict: a field value that legitimately spans several lines (an
    `Affiliation` listing one author per line) is unbalanced line by line too. It is still a far shorter
    list than the record, which is the difference between "this file has a broken record somewhere" and
    something the user can act on.
    """
    names = []
    for line in text.splitlines():
        match = _BIBTEX_FIELD_LINE.match(line)
        if match and line.count("{") != line.count("}"):
            names.append(match.group(1))
    return names

def bibtex_brace_repair_candidates(text: str, max_candidates: int = 200) -> List[str]:
    """Propose repairs for a BibTeX record `text` whose braces do not balance. Most promising first.

    Each candidate escapes some of the record's braces as `\\{` / `\\}`; the text is otherwise identical,
    character for character. Returns `[]` when there is nothing sensible to propose.

    **These are guesses, and the caller decides by parsing them.** Deciding *which* brace is the stray one
    needs to know where each field value begins and ends, which needs a parser — and the record in hand is
    precisely the one no parser will read, so that route closes. The way out is to stop trying to be right
    and start being checkable: propose from surface syntax, then let `bibtexparser` adjudicate. The
    proposals only have to *contain* the truth, not identify it, and a wrong guess costs nothing because it
    fails to parse and is discarded. Callers should take the first candidate that yields an entry.

    **Escaping, never completion.** An unmatched brace in a field value is nearly always a literal one that
    belongs to the text: mathematics reaching a BibTeX file through a PDF extractor arrives as things like
    `{0 <= rho <= 1`, set-builder notation whose closing brace the extractor dropped. `\\{` is what BibTeX
    means by a literal brace in running text, so escaping states what the data actually is, and no character
    is added or removed. Supplying a missing partner instead would invent a grouping the author never wrote,
    and since nothing says *where* that partner belonged, the invention is unconstrained. So a record that
    lost a field value's *terminator*, rather than gaining a stray literal, is not repairable here — every
    candidate will fail the caller's check, which is the correct outcome and not a defect.

    Two cheap facts do the pruning, and between them the real cases collapse to a single candidate:

    - **The sign of the imbalance says which brace to look for.** A surplus of `{` means a stray opener, so
      only openers are eligible; a surplus of `}`, only closers.
    - **Some braces are structurally certain and never eligible** — the `@type{` on the header line, the one
      opening each `Key = {` field line, and the record's own closing `}` on its own line. Those are the
      ones a naive scan would blame first, and escaping any of them destroys the record.
    """
    lines = text.splitlines(keepends=True)
    surplus = text.count("{") - text.count("}")
    wanted = "{" if surplus > 0 else "}"
    if surplus == 0:
        return []

    # Structurally certain braces, which no candidate may touch.
    reserved = set()
    for i, line in enumerate(lines):
        match = _BIBTEX_FIELD_LINE.match(line)
        if match:
            col = line.find("{", match.end())
            if col != -1:
                reserved.add((i, col))
        elif line.lstrip().startswith("@"):
            col = line.find("{")
            if col != -1:
                reserved.add((i, col))
        elif line.strip() == "}":
            reserved.add((i, line.index("}")))

    eligible = [(i, j) for i, line in enumerate(lines) for j, ch in enumerate(line)
                if ch == wanted and (i, j) not in reserved and not line[:j].endswith("\\")]
    if len(eligible) < abs(surplus):
        return []
    # A stray literal sits *inside* a value, so it falls after that value's opening brace and before its
    # terminator. Which way to order the guesses therefore depends on what is being hunted: a surplus
    # opener is more likely the later of the candidates, a surplus closer the earlier one. Ordering only
    # decides which guess the oracle sees first, but where more than one would parse, first is what wins.
    ordered = sorted(eligible, reverse=(wanted == "{"))
    combinations = list(itertools.islice(itertools.combinations(ordered, abs(surplus)), max_candidates + 1))
    if len(combinations) > max_candidates:  # too tangled to guess at; let the caller report it instead
        return []

    candidates = []
    for chosen in combinations:
        repaired = list(lines)
        for i, j in sorted(chosen, reverse=True):  # from the end, so earlier positions stay valid
            repaired[i] = repaired[i][:j] + "\\" + repaired[i][j:]
        candidates.append("".join(repaired))
    return candidates

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

    The result is for reading, so LaTeX markup is converted to Unicode the same way titles and
    abstracts are - "H{\\"a}m{\\"a}l{\\"a}inen" is a spelling of "Hämäläinen", not a name. Callers
    wanting the name as written should keep the raw BibTeX field alongside; the Visualizer importer
    stores it as `bibtex_author` for exactly this reason, so export stays lossless.
    """
    try:
        authors_list = [unicodize_basic_markup(format_bibtex_author(author)) for author in authors]
    except ValueError:
        logger.warning("format_bibtex_authors: failed, caught exception", exc_info=True)
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

# LaTeX accent commands → Unicode combining diacritic.
# Applied to the following base letter; NFC at the end composes the result into
# the precomposed form where one exists (e.g. "ä" → "ä").
# Forms: `\X{c}` (required for letter-based accents `c v u H k r`)
#    and `\Xc` (for the non-letter accents `" ' ` ^ ~ = .`).
_latex_accent_to_combining = {
    '"': "̈",  # diaeresis / umlaut:      \"a  → ä
    "'": "́",  # acute:                   \'e  → é
    "`": "̀",  # grave:                   \`a  → à
    "^": "̂",  # circumflex:              \^o  → ô
    "~": "̃",  # tilde:                   \~n  → ñ
    "=": "̄",  # macron:                  \=a  → ā
    ".": "̇",  # dot above:               \.z  → ż
    "c": "̧",  # cedilla:                 \c{c} → ç
    "v": "̌",  # caron (háček):           \v{s} → š
    "u": "̆",  # breve:                   \u{a} → ă
    "H": "̋",  # double acute:            \H{o} → ő
    "k": "̨",  # ogonek:                  \k{a} → ą
    "r": "̊",  # ring above:              \r{a} → å
}

# LaTeX single-token ligatures / special letters → Unicode.
_latex_ligatures = {
    r"\aa": "å", r"\AA": "Å",
    r"\ae": "æ", r"\AE": "Æ",
    r"\oe": "œ", r"\OE": "Œ",
    r"\o": "ø", r"\O": "Ø",
    r"\l": "ł", r"\L": "Ł",
    r"\ss": "ß",
    r"\i": "ı", r"\j": "ȷ",
}

def _apply_latex_accent(match_obj):
    """Replace a `\\X{c}` or `\\Xc` LaTeX accent match with `c + combining-mark`."""
    accent, letter = match_obj.group(1), match_obj.group(2)
    combining = _latex_accent_to_combining.get(accent)
    if combining is None:  # unknown accent; leave as-is (defensive — regex only matches known accents)
        return match_obj.group(0)
    return letter + combining

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

    # LaTeX single-token ligatures (`\ae`, `\oe`, `\o`, `\l`, `\ss`, `\aa`, `\i`, `\j`, …).
    # Must run before brace stripping. The idiomatic way to write e.g. "ønly" in
    # BibTeX is `{\o}nly` — the braces terminate the `\o` command. Once we strip the
    # braces, `{\o}nly` becomes `\only`, and `\o` followed by a letter no longer matches.
    # The right-side letter-lookahead prevents false matches inside longer identifiers.
    for cmd, repl in _latex_ligatures.items():
        s = re.sub(re.escape(cmd) + r"(?![a-zA-Z])", repl, s)

    # LaTeX accent commands of the form `\X{c}` (braced argument). Must run before
    # brace stripping — otherwise the braces disappear and `\c{c}` collapses to the
    # ambiguous `\cc`. The non-letter accents (`\"a`, `\'e`, …) can also appear
    # without braces, and are handled in a second pass below after brace stripping.
    # `\w` (not `[a-zA-Z]`) so we match e.g. `\"{ı}` — which arises from `\"{\i}`
    # after the ligature pass above has turned `\i` into dotless-i (U+0131).
    s = re.sub(r"""\\(["'`^~=.cvuHkr])\{(\w)\}""", _apply_latex_accent, s)

    # The same letter-named accents, space-terminated instead of braced (`\c e`, `\k a`, `\v s`).
    # A LaTeX control word ends at the first non-letter, so the space *is* the terminator and this
    # is exactly equivalent to the braced form. In `.bib` files it is if anything the commoner
    # spelling, because the idiom is to wrap the whole thing in a case-protecting group:
    # `Tr{\c e}bicki`, not `Tr\c{e}bicki`. Must run before brace stripping for the same reason the
    # braced pass does — afterwards `\c e` has lost the group that told us where it ended.
    # Only the letter-named accents: for `\"a` and friends the command name is punctuation, which
    # self-terminates, so a space there would be a literal space rather than a separator.
    s = re.sub(r"""\\([cvuHkr])\s+(\w)""", _apply_latex_accent, s)

    # Strip BibTeX case-preservation grouping braces (`{Word}`, `{ACRONYM}`, and nested
    # forms like `{{AutoPBL}}`). `bibtexparser` is a pure parser: it hands us the raw
    # field value with the grouping braces still in. biblatex/bibtex would strip these
    # at format time; we never format, so we strip them here.
    #
    # Literal escaped braces (`\{`, `\}` — produced by `raven.papers.utils.bibtex_escape`
    # for actual `{`/`}` characters in source text) must survive this pass. We park
    # them on private-use-area sentinels, strip the grouping braces, then restore.
    # Private-use characters are normalization-stable and appear nowhere in real text.
    s = s.replace(r"\{", "").replace(r"\}", "")
    s = s.replace("{", "").replace("}", "")
    s = s.replace("", "{").replace("", "}")

    # LaTeX accent commands of the form `\Xc` (no braces). Only the non-letter accents:
    # the letter-named ones (`\c`, `\v`, `\u`, `\H`, `\k`, `\r`) need *some* separator from their
    # argument, brace or space, and both spellings were handled above.
    s = re.sub(r"""\\(["'`^~=.])(\w)""", _apply_latex_accent, s)

    # Remove LaTeX escapes (including those produced by `raven.papers.utils.bibtex_escape`).
    # `\{` and `\}` are not listed here — they were handled by the brace-stripping pass above.
    s = s.replace(r"\%", "%")
    s = s.replace(r"\$", "$")
    s = s.replace(r"\#", "#")
    s = s.replace(r"\&", "&")

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
    s = s.replace(r"&nbsp;", " ")

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

    # `&amp;` after every other entity, which is what keeps the decoding single-pass: a source that
    # escaped its own markup writes `&amp;lt;` for a literal "&lt;", and decoding the ampersand first
    # would turn that into "<" — the text saying something it does not say. Decoded last, it comes out
    # as the literal "&lt;" the author wrote.
    #
    # Reached at all because `\&` is unescaped near the top of this function, so a BibTeX file's
    # `Q\&amp;A` arrives here as `Q&amp;A`. That is the commonest of these in a database export by a
    # wide margin, and without this line it survived into abstracts, titles and journal names.
    s = s.replace(r"&amp;", "&")

    # The LaTeX sequence `\"{\i}` (→ "naïve") uses dotless-i purely as a typesetting
    # trick — the intended letter is i, written dotless so the diaeresis dots don't
    # collide with the letter's own dot. Unicode encodes the common combinations as
    # precomposed "i with accent" characters (ï, í, ì, î, …) but has no precomposed
    # "dotless-i with accent" — so if we leave the dotless form in place, the NFC
    # pass below can't compose the sequence. Swap dotless → dotted whenever a
    # combining diacritic follows, so NFC can do its job. Same story for \j → ȷ.
    s = re.sub("ı([̀-ͯ])", r"i\1", s)
    s = re.sub("ȷ([̀-ͯ])", r"j\1", s)

    # Final NFC pass composes the `base + combining-mark` sequences produced by
    # the LaTeX accent passes (e.g. "ä" → "ä") into their precomposed forms
    # where Unicode has one. The initial NFKC pass only applied to the input;
    # combining marks introduced afterwards need a second round.
    s = unicodedata.normalize("NFC", s)

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

def make_search_matcher(s):
    """Compile search string `s` into a predicate `matches(text) -> bool`.

    The predicate is the incremental fragment search `search_string_to_fragments` describes: every fragment
    must appear somewhere in `text`, in any order, and a lowercase fragment matches case-insensitively while
    one carrying an uppercase letter matches exactly. An empty search string yields a predicate that accepts
    everything, so "no query" needs no special case at the call site.

    Compiling once and testing many is the point: splitting the search string is per-query work, and the
    predicate is what runs per candidate.

    Note `text` is matched as given. Where the corpus has a normalized form (as Visualizer's entries do),
    pass that — `s` is normalized by `normalize_search_string`, so an unnormalized `text` can fail to match
    on exactly the characters normalization exists to reconcile.
    """
    case_sensitive_fragments, case_insensitive_fragments = search_string_to_fragments(s, sort=False)  # all must match, so sorting buys nothing

    def matches(text):
        text_lowercase = text.lower()
        return (all(fragment in text_lowercase for fragment in case_insensitive_fragments) and
                all(fragment in text for fragment in case_sensitive_fragments))
    return matches

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

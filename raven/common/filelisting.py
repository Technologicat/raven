"""Directory listing for a file browser: enumerate, filter, sort. No GUI.

This is the data half of a file dialog. It answers "what is in this directory, given these filters, in this
order?" and hands back plain objects; turning those into table rows, or into grid tiles, is somebody else's
job.

**Why it is separate.** A listing that exists only as widgets cannot be tested, and cannot be rendered two
ways — and a sort has to recover, from the widgets, the values it was built from. Keeping the entries as
data costs nothing and avoids all three. Splitting the operation from its dialog is the same move
`raven.librarian.cleanup` makes away from `cleanup_dialog`, and for the same reason.

**Directories are resolved explicitly, never against the process's current directory.** `list_directory`
takes the directory to list and builds every path from it. This matters beyond tidiness: a listing that
consults `os.getcwd()` for some of its answers and its argument for the rest is correct only while the two
agree, and reports a directory's contents with every entry misclassified when they do not.
"""

__all__ = ["FileEntry", "SortKey",
           "list_directory",
           "format_size", "format_mtime",
           "is_hidden"]

import ctypes
import dataclasses
import enum
import logging
import os
import time
from collections.abc import Callable
from typing import Optional

logger = logging.getLogger(__name__)

# The `Type` column's values. Strings rather than a bool so they can be displayed and sorted as-is, and
# because this is what the dialog has always shown.
KIND_DIR = "Dir"
KIND_FILE = "File"


class SortKey(enum.Enum):
    """Which column a listing is ordered by."""
    NAME = "name"
    DATE = "date"
    KIND = "kind"
    SIZE = "size"


@dataclasses.dataclass(frozen=True)
class FileEntry:
    """One row of a listing, in whatever view is rendering it.

    `path` is absolute, and is the entry's **stable identity**: names repeat across directories and indices
    move whenever the listing is re-filtered or re-sorted, so anything remembering *which* entry it was on —
    a keyboard cursor, a selection — remembers this.

    `size` is `None` where there is no meaningful answer: a directory whose size was not computed, or an
    entry that could not be read. `mtime` is `None` on the same footing.
    """
    name: str
    path: str
    kind: str  # KIND_DIR or KIND_FILE
    is_hidden: bool
    mtime: Optional[float]
    size: Optional[int]
    is_parent: bool = False  # the ".." entry, which sorts first and is nobody's file

    def get_is_dir(self) -> bool:
        return self.kind == KIND_DIR
    is_dir = property(fget=get_is_dir, doc="Whether this entry is a directory. `..` counts as one.")


def _has_hidden_attribute(path: str) -> bool:
    """Whether Windows marks `path` hidden. Always `False` elsewhere."""
    if os.name != "nt":
        return False
    try:
        FILE_ATTRIBUTE_HIDDEN = 0x2
        attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
        return bool(FILE_ATTRIBUTE_HIDDEN & attrs)
    except Exception as exc:
        logger.debug(f"_has_hidden_attribute: cannot read attributes of '{path}', treating as not hidden: {type(exc)}: {exc}")
        return False


def is_hidden(path: str) -> bool:
    """Whether `path` is hidden: a leading dot anywhere, plus the hidden attribute on Windows."""
    return os.path.basename(os.path.abspath(path)).startswith(".") or _has_hidden_attribute(path)


def format_size(size: Optional[int]) -> str:
    """Render a byte count for display: `"1 MB"`, `"512 B"`, or `"-"` when there is no answer."""
    if size is None:
        return "-"
    for unit, size_limit in (("TB", 2**40), ("GB", 2**30), ("MB", 2**20), ("KB", 2**10), ("B", 1)):
        if size >= size_limit:
            return f"{size / size_limit:.0f} {unit}"
    return "0 B"


def format_mtime(mtime: Optional[float]) -> str:
    """Render a modification time for display, or `"-"` when there is no answer."""
    if mtime is None:
        return "-"
    return time.ctime(mtime)


def _directory_size(path: str) -> Optional[int]:
    """Total size of everything under `path`. `None` if it could not be walked.

    Note this reads the whole subtree, so it is opt-in at the call site — on a large tree it is far and away
    the most expensive thing a listing can do.
    """
    total = 0
    try:
        for dirpath, _dirnames, filenames in os.walk(path):
            for filename in filenames:
                try:
                    total += os.path.getsize(os.path.join(dirpath, filename))
                except OSError:  # vanished mid-walk, or unreadable; the rest of the total still stands
                    continue
    except OSError as exc:
        logger.debug(f"_directory_size: cannot walk '{path}': {type(exc)}: {exc}")
        return None
    return total


def _make_entry(directory: str, name: str, *, dir_sizes: bool) -> Optional[FileEntry]:
    """Build one `FileEntry`, or `None` if the entry cannot be classified.

    Returns `None` rather than raising for a broken symlink or a file that vanished between the directory
    read and this call, both of which are ordinary rather than exceptional in a directory someone else is
    also writing to.
    """
    path = os.path.join(directory, name)
    try:
        if os.path.isdir(path):
            kind = KIND_DIR
            size = _directory_size(path) if dir_sizes else None
        elif os.path.isfile(path):
            kind = KIND_FILE
            size = os.path.getsize(path)
        else:  # broken symlink, socket, device node, or gone since we listed the directory
            return None
        mtime = os.path.getmtime(path)
    except OSError as exc:
        logger.debug(f"_make_entry: cannot stat '{path}', omitting it: {type(exc)}: {exc}")
        return None

    return FileEntry(name=name, path=os.path.abspath(path), kind=kind,
                     is_hidden=is_hidden(path), mtime=mtime, size=size)


def _sort_value(entry: FileEntry, sort_key: SortKey):
    """The value `entry` is ordered by, with a type-stable fallback for the `None`s."""
    if sort_key is SortKey.NAME:
        return entry.name.lower()
    if sort_key is SortKey.DATE:
        return entry.mtime if entry.mtime is not None else 0.0
    if sort_key is SortKey.KIND:
        return entry.kind
    return entry.size if entry.size is not None else -1  # SortKey.SIZE


def list_directory(directory: str,
                   *,
                   show_hidden: bool = False,
                   dirs_only: bool = False,
                   include_parent: bool = True,
                   name_filter: Optional[Callable[[str], bool]] = None,
                   type_filter: Optional[Callable[[str], bool]] = None,
                   sort_key: SortKey = SortKey.NAME,
                   descending: bool = False,
                   dir_sizes: bool = False) -> list[FileEntry]:
    """List `directory`, filtered and sorted, as `FileEntry` objects.

    `show_hidden`: include dotfiles (and Windows-hidden entries).
    `dirs_only`: omit files entirely, for a directory-picker.
    `include_parent`: prepend the `..` entry. It is always first and is never filtered out — it is the only
        way up, so a name filter that hid it would strand the user in the directory.
    `name_filter`: predicate on the entry's name; `None` accepts everything. See
        `raven.common.utils.make_search_matcher` for the dialog's incremental fragment search.
    `type_filter`: predicate on the entry's name, applied to **files only** — a type filter selects among
        files and must not hide the directories you navigate through to reach them.
    `sort_key`, `descending`: the ordering. **Directories always precede files**, in both directions;
        reversing sorts within each group rather than interleaving them, which is what every file manager
        does and what the dialog has always done.
    `dir_sizes`: compute directory sizes by walking each subtree. Off by default, and expensive.

    Unreadable or vanished entries are omitted with a debug log rather than raising, so a directory
    containing one broken symlink still lists.
    """
    entries = []
    for name in os.listdir(directory):  # OSError propagates: the caller knows which directory it asked for
        entry = _make_entry(directory, name, dir_sizes=dir_sizes)
        if entry is None:
            continue
        if entry.is_hidden and not show_hidden:
            continue
        if not entry.is_dir:  # noqa: SIM102 -- two separate concerns: is it admissible at all, and does the query select it
            if dirs_only:
                continue
            if type_filter is not None and not type_filter(entry.name):
                continue
        if name_filter is not None and not name_filter(entry.name):
            continue
        entries.append(entry)

    # Directories first in both directions. `reverse=True` flips the whole ordering, so the group rank is
    # flipped alongside it to leave the groups where they were.
    dir_rank = {KIND_DIR: 1, KIND_FILE: 0} if descending else {KIND_DIR: 0, KIND_FILE: 1}
    entries.sort(key=lambda entry: (dir_rank[entry.kind], _sort_value(entry, sort_key)),
                 reverse=descending)

    if include_parent:
        parent_path = os.path.abspath(os.path.join(directory, os.pardir))
        entries.insert(0, FileEntry(name=os.pardir, path=parent_path, kind=KIND_DIR,
                                    is_hidden=False, mtime=None, size=None, is_parent=True))

    return entries

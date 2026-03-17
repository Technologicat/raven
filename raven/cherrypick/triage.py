"""Triage state management and file operations for raven-cherrypick.

Manages the three-state triage system (neutral/cherry/lemon) and the
virtual directory that merges ``base/``, ``base/cherries/``, and
``base/lemons/`` into a single sorted image list.
"""

__all__ = ["TriageState", "ImageEntry", "TriageManager"]

import logging
import pathlib
import shutil
from enum import Enum
from typing import Optional, Union

from ..common.image.utils import IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)

CHERRY_DIR = "cherries"
LEMON_DIR = "lemons"


class TriageState(Enum):
    """Triage classification for an image."""
    NEUTRAL = "neutral"
    CHERRY = "cherry"
    LEMON = "lemon"


class ImageEntry:
    """An image in the virtual directory.

    Attributes:
        filename:  The bare filename (e.g. ``"IMG_1234.jpg"``).
        state:     Current triage state.
        base_dir:  The root folder opened by the user.
    """
    __slots__ = ("filename", "state", "base_dir")

    def __init__(self, filename: str, state: TriageState, base_dir: pathlib.Path):
        self.filename = filename
        self.state = state
        self.base_dir = base_dir

    @property
    def path(self) -> pathlib.Path:
        """Absolute path to the image file in its current location."""
        if self.state is TriageState.CHERRY:
            return self.base_dir / CHERRY_DIR / self.filename
        elif self.state is TriageState.LEMON:
            return self.base_dir / LEMON_DIR / self.filename
        else:
            return self.base_dir / self.filename

    def __repr__(self) -> str:
        return f"ImageEntry({self.filename!r}, {self.state.value})"


def _is_image(path: Union[pathlib.Path, str]) -> bool:
    """Return True if *path* looks like a supported image file."""
    path = pathlib.Path(path)
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _subdir(base: Union[pathlib.Path, str], name: str) -> pathlib.Path:
    """Return ``base / name``, creating the directory if it doesn't exist."""
    d = pathlib.Path(base) / name
    d.mkdir(exist_ok=True)
    return d


class TriageManager:
    """Manages the image list and triage file operations for one folder.

    The virtual directory merges ``base/``, ``base/cherries/``, and
    ``base/lemons/``, sorted by filename.  Moving a file between
    subdirectories does not change its grid position (sort key is the
    bare filename, stable across triage state changes).
    """

    def __init__(self, base_dir: Union[pathlib.Path, str]):
        self.base_dir = pathlib.Path(base_dir).resolve()
        self.images: list[ImageEntry] = []
        self._index: dict[str, int] = {}  # filename -> position in self.images
        self.scan()

    # ------------------------------------------------------------------
    # Directory scanning
    # ------------------------------------------------------------------

    def scan(self) -> None:
        """(Re-)scan the base directory and its triage subdirectories.

        Populates ``self.images`` sorted by filename.  Existing triage
        state in ``cherries/`` and ``lemons/`` is preserved.
        """
        entries: dict[str, TriageState] = {}

        # Scan base directory (neutral images).
        for f in self.base_dir.iterdir():
            if _is_image(f):
                entries[f.name] = TriageState.NEUTRAL

        # Scan triage subdirectories.
        for subdir, state in ((CHERRY_DIR, TriageState.CHERRY),
                              (LEMON_DIR, TriageState.LEMON)):
            d = self.base_dir / subdir
            if d.is_dir():
                for f in d.iterdir():
                    if _is_image(f):
                        if f.name in entries:
                            # Same filename in base AND subdir — keep the triaged one.
                            logger.warning("TriageManager.scan: duplicate filename %r in both %s/ and %s/; "
                                           "using the triaged copy.",
                                           f.name, self.base_dir.name, subdir)
                        entries[f.name] = state

        # Build sorted list.
        self.images = [ImageEntry(fn, st, self.base_dir)
                       for fn, st in sorted(entries.items())]
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self._index = {e.filename: i for i, e in enumerate(self.images)}

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> ImageEntry:
        return self.images[idx]

    def index_of(self, filename: str) -> Optional[int]:
        """Return the grid position of *filename*, or ``None``."""
        return self._index.get(filename)

    def count(self, state: TriageState) -> int:
        """Count images in a given triage state."""
        return sum(1 for e in self.images if e.state is state)

    # ------------------------------------------------------------------
    # Triage operations
    # ------------------------------------------------------------------

    def set_state(self, idx_or_indices: Union[int, list[int]],
                  new_state: TriageState) -> Union[Optional[str], list[str]]:
        """Change the triage state of one or more images.

        *idx_or_indices* is a single grid position or a list of positions.

        Physically moves each file to the appropriate subdirectory.

        For a single index, returns ``None`` on success or an error message.
        For a list, returns a list of error messages (may be empty).
        """
        if isinstance(idx_or_indices, list):
            errors = []
            for idx in idx_or_indices:
                err = self._set_state_single(idx, new_state)
                if err is not None:
                    errors.append(err)
            return errors
        return self._set_state_single(idx_or_indices, new_state)

    def _set_state_single(self, idx: int, new_state: TriageState) -> Optional[str]:
        """Change the triage state of the image at position *idx*."""
        entry = self.images[idx]
        if entry.state is new_state:
            return None  # no-op

        old_path = entry.path

        # Compute destination path.
        if new_state is TriageState.CHERRY:
            dest_dir = _subdir(self.base_dir, CHERRY_DIR)
        elif new_state is TriageState.LEMON:
            dest_dir = _subdir(self.base_dir, LEMON_DIR)
        else:
            dest_dir = self.base_dir

        dest_path = dest_dir / entry.filename

        # Collision check.
        if dest_path.exists():
            msg = f"TriageManager.set_state: cannot move {entry.filename}: destination already exists in {dest_dir.name}/"
            logger.error(msg)
            return msg

        # Move.
        try:
            shutil.move(str(old_path), str(dest_path))
        except OSError as exc:
            msg = f"TriageManager.set_state: failed to move {entry.filename}: {exc}"
            logger.error(msg)
            return msg

        entry.state = new_state
        logger.info("TriageManager.set_state: %s → %s", entry.filename, new_state.value)
        return None

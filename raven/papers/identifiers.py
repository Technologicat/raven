"""arXiv identifier parsing and extraction from filenames.

Provides regex-based extraction of arXiv IDs from PDF filenames, version
parsing, and de-duplication (keeping the latest version of each paper).

Supports all three arXiv ID eras:

- **New-style 5-digit** (Jan 2015 onwards): ``YYMM.NNNNN``
- **New-style 4-digit** (Apr 2007 – Dec 2014): ``YYMM.NNNN``
- **Old-style** (pre-Apr 2007): ``subject-class/YYMMNNN``
  (in filenames, the ``/`` is commonly replaced by ``-``, ``_``, or ``.``)

The CLI entry point (``raven-arxiv2id``) lists unique arXiv IDs found in a directory
of PDF files.
"""

from __future__ import annotations

__all__ = [
    "ARXIV_NEW_ID_RE",
    "ARXIV_OLD_ID_RE",
    "extract_id",
    "split_version",
    "strip_version",
    "list_subfolders",
    "list_pdf_files",
    "extract_ids_from_filenames",
    "collect_latest_ids",
    "main",
]

import argparse
import os
import pathlib
import re
from typing import Union

from unpythonic import uniqify

from .. import __version__

# New-style arXiv IDs: YYMM.NNNN (2007-2014) or YYMM.NNNNN (2015+).
# Digit lookaround instead of \b so that IDs embedded between underscores,
# letters, or hyphens in filenames still match.
ARXIV_NEW_ID_RE = re.compile(r"(?<!\d)(\d{4}\.\d{4,5})(v\d+)?(?!\d)")

# Old-style arXiv IDs: subject-class/YYMMNNN (pre-Apr 2007).
# In filenames, the canonical ``/`` is commonly replaced by ``-``, ``_``,
# or ``.``.  Subject classes: ``math``, ``hep-th``, ``cs.AI``, etc.
# Length limits on each segment keep false positives low.
ARXIV_OLD_ID_RE = re.compile(
    r"(?<![a-zA-Z])"                        # not preceded by a letter
    r"([a-z]{1,7}(?:-[a-z]{1,3})?(?:\.[A-Z][A-Z])?)"  # subject class
    r"[-_./]"                               # separator
    r"(\d{7})"                              # 7-digit paper number
    r"(v\d+)?"                              # optional version
    r"(?!\d)",                              # not followed by a digit
)


def extract_id(filename: str) -> str | None:
    """Extract an arXiv identifier from *filename*, or return ``None``.

    Tries new-style IDs (``YYMM.NNNN(N)``) first, then old-style
    (``subject-class/YYMMNNN``).  Old-style IDs are returned in canonical
    form with ``/`` regardless of the separator used in the filename.
    """
    # New-style (more common in active collections)
    matches = ARXIV_NEW_ID_RE.findall(filename)
    matches = ["".join(parts) for parts in matches]
    if matches:
        assert len(matches) == 1
        return matches[0]
    # Old-style
    matches = ARXIV_OLD_ID_RE.findall(filename)
    if matches:
        assert len(matches) == 1
        subject, number, version = matches[0]
        return f"{subject}/{number}{version}"
    return None


def split_version(raw_id: str) -> tuple[str, int]:
    """Split an arXiv ID into ``(base, version_int)``.

    IDs without a version suffix are assumed to be version 1.

    >>> split_version("2103.12345v3")
    ('2103.12345', 3)
    >>> split_version("2103.12345")
    ('2103.12345', 1)
    """
    parts = raw_id.split("v")
    if len(parts) == 1:
        return raw_id.strip(), 1
    base, version = parts
    return base.strip(), int(version.strip())


def strip_version(raw_id: str) -> str:
    """Return the arXiv ID with any version suffix removed.

    >>> strip_version("2103.12345v2")
    '2103.12345'
    >>> strip_version("hep-ex/0307015v1")
    'hep-ex/0307015'
    """
    return re.sub(r"v\d+$", "", raw_id)


# ---- Filesystem scanning --------------------------------------------------

def list_subfolders(path: str) -> list[str]:
    """Recursively list all subdirectories under *path*.

    Directories whose names start with ``00_`` are skipped.
    """
    paths: list[str] = []
    for root, dirs, files in os.walk(path):
        paths.append(root)
        new_dirs = [x for x in dirs if not x.startswith("00_")]
        dirs.clear()
        dirs.extend(new_dirs)
    paths = list(sorted(uniqify(str(pathlib.Path(p).expanduser().resolve()) for p in paths)))
    return paths


def list_pdf_files(path: Union[pathlib.Path, str]) -> list[str]:
    """Return sorted list of PDF filenames in *path* (non-recursive)."""
    return list(sorted(
        filename for filename in os.listdir(path)
        if filename.lower().endswith(".pdf")
    ))


# The ``canonize`` feature is used by `raven.papers.download`, which see.
def extract_ids_from_filenames(filenames: list[str],
                               canonize: bool = False) -> list[tuple[str, str]]:
    """Extract arXiv IDs from a list of filenames.

    Returns ``[(arxiv_id, filename), ...]`` for files that contain a
    recognizable arXiv ID.

    *canonize*: add an implicit ``v1`` to IDs that have no version part.
    """
    ids_and_paths = [(raw_id, p) for p in filenames if (raw_id := extract_id(p))]
    if canonize:
        final: list[tuple[str, str]] = []
        for raw_id, p in ids_and_paths:
            base, version = split_version(raw_id)
            final.append((f"{base}v{version}", p))
        return final
    return ids_and_paths


def collect_latest_ids(input_dir: str) -> list[tuple[str, str]]:
    """Scan *input_dir* recursively; return unique arXiv IDs in sorted order.

    When the same paper (same base ID) appears under different filenames
    or in different subdirectories, only the newest version is kept.

    Returns ``[(raw_id, filename), ...]`` sorted by ID.  The filename is
    the basename of the PDF where the ID was matched (not a full path).

    Directories whose names start with ``00_`` are skipped (see
    :func:`list_subfolders`).
    """
    arxiv_pdf_files: dict[str, tuple[str, str, int]] = {}
    for path in list_subfolders(input_dir):
        pdf_files = list_pdf_files(path)
        if not pdf_files:
            continue
        ids_and_paths = extract_ids_from_filenames(pdf_files)
        base_ids_and_versions = [split_version(raw_id) for raw_id, p in ids_and_paths]

        # Pick the latest version for each identifier discovered so far
        for (raw_id, p), (base_id, version) in zip(ids_and_paths, base_ids_and_versions):
            if base_id not in arxiv_pdf_files:
                arxiv_pdf_files[base_id] = (raw_id, p, version)
            else:
                _, _, recorded_version = arxiv_pdf_files[base_id]
                if version > recorded_version:
                    arxiv_pdf_files[base_id] = (raw_id, p, version)

    return [(raw_id, filename) for raw_id, filename, _ in sorted(arxiv_pdf_files.values())]


# ---- CLI -------------------------------------------------------------------

def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="List identifiers of arXiv papers in the specified directory. "
        "The papers are assumed to be PDF files with the arXiv identifier "
        "(yymm.xxxxx) somewhere in the filename. Only unique identifiers are "
        "returned (even if duplicated under different filenames, as long as "
        "the identifier is the same). To avoid duplicates, in case of "
        "multiple versions of the same paper (yymm.xxxxxvN), only the most "
        "recent version is returned.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-i", "--input-dir", dest="input_dir", default=None,
                        type=str, metavar="input_dir",
                        help="Input directory containing arXiv PDF file(s).")
    parser.add_argument('-v', '--version', action='version',
                        version=('%(prog)s ' + __version__))
    parser.add_argument("-V", "--verbose", dest="verbose", action="store_true",
                        default=False,
                        help="Print also the filename for each match, for "
                        "debugging your collection.")
    opts = parser.parse_args()

    for identifier, filename in collect_latest_ids(opts.input_dir or "."):
        if opts.verbose:
            print(identifier, filename)
        else:
            print(identifier)


if __name__ == "__main__":
    main()

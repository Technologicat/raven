"""Shared utilities for bibliography tools."""

from __future__ import annotations

__all__ = ["deduplicate_arxiv_ids", "bibtex_escape", "bibtex_unescape",
           "tsv_cell", "write_tsv"]

import csv
import pathlib
import re
from collections.abc import Iterable

from . import identifiers


def deduplicate_arxiv_ids(arxiv_ids: list[str]) -> list[str]:
    """Deduplicate arXiv IDs, keeping the highest version of each paper.

    IDs without a version suffix are treated as version 1.
    Preserves the order of first occurrence.

    >>> deduplicate_arxiv_ids(["2103.12345v1", "2103.12345v3", "2103.12345v2"])
    ['2103.12345v3']
    >>> deduplicate_arxiv_ids(["2103.12345", "2103.12345v2"])
    ['2103.12345v2']
    """
    best: dict[str, tuple[str, int, int]] = {}  # base → (raw_id, version, first_index)
    for i, raw_id in enumerate(arxiv_ids):
        base, version = identifiers.split_version(raw_id)
        if base not in best or version > best[base][1]:
            first_index = best[base][2] if base in best else i
            best[base] = (raw_id, version, first_index)
    return [raw_id for raw_id, _version, _idx in sorted(best.values(), key=lambda t: t[2])]


def bibtex_escape(s: str) -> str:
    r"""Escape BibTeX-special characters in a field value.

    Handles the characters that cause ``bibtexparser`` or BibTeX/LaTeX
    to choke when they appear unescaped inside ``{...}``-delimited field values.

    Use `bibtex_unescape` to reverse this transformation for display.
    """
    # Order matters: backslash first (so we don't double-escape the backslashes
    # we're about to introduce), then everything else.
    s = s.replace("\\", "\\\\")
    s = s.replace("{", r"\{")
    s = s.replace("}", r"\}")
    s = s.replace("[", "{[}")
    s = s.replace("]", "{]}")
    s = s.replace("&", r"\&")
    s = s.replace("%", r"\%")
    s = s.replace("#", r"\#")
    s = s.replace("$", r"\$")
    return s


def bibtex_unescape(s: str) -> str:
    r"""Reverse `bibtex_escape` — convert LaTeX escapes back to plain text.

    Intended for display purposes (e.g. in the Raven GUI). Not a general
    LaTeX-to-Unicode converter — only handles the escapes that `bibtex_escape`
    produces.
    """
    s = s.replace(r"\$", "$")
    s = s.replace(r"\#", "#")
    s = s.replace(r"\%", "%")
    s = s.replace(r"\&", "&")
    s = s.replace("{[}", "[")
    s = s.replace("{]}", "]")
    s = s.replace(r"\}", "}")
    s = s.replace(r"\{", "{")
    s = s.replace("\\\\", "\\")
    return s


def tsv_cell(value: object) -> str:
    """`value` as one TSV cell: every run of whitespace, tabs and newlines included, collapsed to a space.

    A tab would end the cell and a newline the row, so a cell carrying either would misalign the file.
    """
    return re.sub(r"\s+", " ", str(value)).strip()


def write_tsv(path: pathlib.Path,
              comments: Iterable[str],
              columns: Iterable[str],
              rows: Iterable[Iterable[object]]) -> None:
    """Write a TSV report: a `# ` line per comment, then the column header, then one line per row.

    Every cell goes through `tsv_cell`, so each row is one line. A cell containing a `"` is quoted, with
    the `"` doubled — the CSV convention a spreadsheet's import follows, which reads a bare `"` as the start
    of a quoted cell and runs it on into the rows below.
    """
    with open(path, "w", encoding="utf-8", newline="") as f:
        for comment in comments:
            f.write(f"# {tsv_cell(comment)}\n")
        writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        writer.writerow([tsv_cell(column) for column in columns])
        writer.writerows([tsv_cell(cell) for cell in row] for row in rows)

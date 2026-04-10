"""Shared utilities for bibliography tools."""

from __future__ import annotations

__all__ = ["bibtex_escape", "bibtex_unescape", "deduplicate_arxiv_ids"]


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

"""Shared utilities for bibliography tools."""

from __future__ import annotations

__all__ = ["bibtex_escape"]


def bibtex_escape(s: str) -> str:
    """Escape BibTeX-special characters in a field value.

    Handles the characters that cause ``bibtexparser`` or BibTeX itself to
    choke when they appear unescaped inside field values.
    """
    # Order matters: backslash first, then braces, then the rest.
    s = s.replace("\\", "\\\\")
    s = s.replace("{", "{{")
    s = s.replace("}", "}}")
    s = s.replace("[", "{[}")
    s = s.replace("&", r"\&")
    s = s.replace("%", r"\%")
    return s

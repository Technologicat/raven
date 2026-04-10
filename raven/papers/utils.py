"""Shared utilities for bibliography tools."""

from __future__ import annotations

__all__ = ["bibtex_escape", "bibtex_unescape"]


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

"""Shared utilities for bibliography tools."""

from __future__ import annotations

__all__ = ["deduplicate_arxiv_ids", "bibtex_escape", "bibtex_unescape",
           "normalize_doi", "paper_url"]

import re

from . import identifiers

# The seven Unicode dashes, folded to ASCII `-` in a DOI. Publishers' exports disagree about which one a
# DOI containing a hyphen should use, and two records whose DOIs differ by an en-dash are one paper.
_DASHES = "‐‑‒–—―−"
_DASH_TABLE = str.maketrans({dash: "-" for dash in _DASHES})

# Everything a database might put in front of the DOI itself.
_DOI_PREFIX_PATTERN = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*|info:doi/)+", re.IGNORECASE)


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


def normalize_doi(maybe_raw: str | None) -> str | None:
    """The comparison key for a DOI, or `None` if the value is not one.

    Lowercased, stripped of whatever resolver prefix the exporting database put in front of it, with the
    Unicode dashes folded to ASCII and trailing sentence punctuation removed.

    Returns `None` for anything that does not look like a DOI, which a `doi` field regularly holds — an
    empty string, `n/a`, a publisher's landing-page URL. Those must not become a match key: they are
    equal to each other across unrelated records, and would merge papers that have nothing to do with
    one another.
    """
    if not maybe_raw:
        return None
    value = _DOI_PREFIX_PATTERN.sub("", str(maybe_raw).strip().strip("{}").strip())
    value = value.translate(_DASH_TABLE).lower()
    value = "".join(value.split())  # a DOI has no internal whitespace; a line-wrapped export has some
    value = value.rstrip(".,;:")
    # A DOI is `10.`, a registrant code of four or more digits, a slash, and a non-empty suffix
    # (ISO 26324). No upper bound on the digits, there being none in the standard — the corpus this was
    # built against uses four and five. Anything else in a `doi` field is something other than a DOI,
    # whatever the field is called.
    if not re.match(r"^10\.\d{4,}/\S+$", value):
        return None
    return value


def paper_url(maybe_doi: str | None, maybe_url: str | None) -> str:
    """A link to the paper a record describes: by its DOI if it has one, else its `url` field, else `""`.

    `maybe_doi`, `maybe_url`: the record's `doi` and `url` fields, as found, or `None` when absent.

    The DOI is the one `normalize_doi` returns, so a `doi` field holding a resolver URL, a `doi:` prefix
    or something that is not a DOI at all is handled as it is there. That form is lowercased, which still
    resolves: DOIs are case-insensitive. A `url` is returned only if it is an HTTP(S) address.
    """
    maybe_normalized = normalize_doi(maybe_doi)
    if maybe_normalized is not None:
        return f"https://doi.org/{maybe_normalized}"
    url = (maybe_url or "").strip().strip("{}").strip()
    if url.lower().startswith(("http://", "https://")):
        return url
    return ""

"""Convert arXiv Atom feed entries to BibTeX.

Used by `raven.papers.search` to format arXiv API search results.
"""

from __future__ import annotations

__all__ = ["entries_to_bibtex"]

import re

import bibtexparser
from bibtexparser.model import Entry, Field
from bibtexparser import Library

from . import identifiers


def _make_key(entry) -> str:
    """Generate a BibTeX key from an arXiv feed entry.

    Format: ``LastName_YYYY_arXivID`` — guaranteed unique by the arXiv ID.
    """
    # arXiv ID from the entry URL, e.g. "http://arxiv.org/abs/2103.12345v2"
    arxiv_id = entry.id.split("/abs/")[-1]
    arxiv_id = identifiers.strip_version(arxiv_id)
    # Old-style IDs contain a slash (hep-ex/0307015) — replace for BibTeX safety
    arxiv_id = arxiv_id.replace("/", "_")

    # First author's last name
    authors = entry.get("authors", [])
    if authors:
        name = authors[0].get("name", "Unknown")
        last_name = name.split()[-1]
        last_name = re.sub(r"[^a-zA-Z]", "", last_name)
    else:
        last_name = "Unknown"

    year = entry.published[:4]
    return f"{last_name}_{year}_{arxiv_id}"


def _clean_whitespace(text: str) -> str:
    """Collapse runs of whitespace (including newlines) to single spaces."""
    return " ".join(text.split())


def entries_to_bibtex(entries: list) -> str:
    """Convert a list of feedparser arXiv entries to a BibTeX string."""
    library = Library()

    for entry in entries:
        arxiv_id = entry.id.split("/abs/")[-1]
        arxiv_id = identifiers.strip_version(arxiv_id)

        year = entry.published[:4]
        authors = " and ".join(a.get("name", "") for a in entry.get("authors", []))
        title = _clean_whitespace(entry.get("title", ""))
        abstract = _clean_whitespace(entry.get("summary", ""))

        fields = [
            Field("author", authors),
            Field("title", title),
            Field("year", year),
            Field("eprint", arxiv_id),
            Field("archiveprefix", "arXiv"),
            Field("abstract", abstract),
        ]

        # Primary category
        primary = entry.get("arxiv_primary_category", {})
        if term := primary.get("term"):
            fields.append(Field("primaryclass", term))

        # DOI (may be absent)
        if doi := entry.get("arxiv_doi"):
            fields.append(Field("doi", doi))

        # Journal reference (may be absent)
        if journal_ref := entry.get("arxiv_journal_ref"):
            fields.append(Field("journal", journal_ref))

        key = _make_key(entry)
        library.add(Entry("article", key, fields=fields))

    return bibtexparser.write_string(library)

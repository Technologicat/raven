"""Read and write BibTeX.

Writing: `entries_to_bibtex` converts arXiv Atom feed entries, used by `raven.papers.search` to format
arXiv API search results.

Reading: `parse_file` and `parse_string` wrap `bibtexparser` with the one middleware chain Raven reads
BibTeX through, so every consumer gets the same normalization.
"""

from __future__ import annotations

__all__ = ["parse_file", "parse_string",
           "entries_to_bibtex"]

import pathlib
import re

import bibtexparser
from bibtexparser.model import Entry, Field
from bibtexparser import Library

from . import identifiers


def _reader_middleware() -> list:
    """The middleware chain Raven reads BibTeX through, as a fresh list per call.

    Each link earns its place:

      - `NormalizeFieldKeys` because the key case is not dependable - a Web of Science export writes
        `Title = {...}`, the BibTeX literature writes `title = {...}`.
      - `SeparateCoAuthors` then `SplitNameParts`, in that order, because the second raises without the
        first. Between them they turn one `author` string into name parts that survive "Ludwig van
        Beethoven", "Brinch Hansen, Per" and "Beeblebrox, IV, Zaphod".

    Fresh instances rather than a shared module-level list, because a middleware is free to carry
    per-parse state and sharing one across concurrent parses would be a bug that only shows up under
    load.
    """
    return [bibtexparser.middlewares.NormalizeFieldKeys(),
            bibtexparser.middlewares.SeparateCoAuthors(),
            bibtexparser.middlewares.SplitNameParts()]


def parse_file(filename: str | pathlib.Path) -> Library:
    """Parse the BibTeX file `filename`, returning a `bibtexparser` `Library`.

    Raises whatever `bibtexparser` raises; a caller that wants to treat unparseable input as a normal
    outcome should catch it. Note a `Library` can also come back *partly* parsed - unreadable records
    land in `library.failed_blocks` rather than raising, so a successful return is not a promise that
    every record was understood.
    """
    return bibtexparser.parse_file(str(filename), append_middleware=_reader_middleware())


def parse_string(text: str) -> Library:
    """Parse `text` as BibTeX, returning a `bibtexparser` `Library`.

    `parse_file`, which see, for the error behaviour - it is the same.
    """
    return bibtexparser.parse_string(text, append_middleware=_reader_middleware())


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

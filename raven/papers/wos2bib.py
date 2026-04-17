#!/usr/bin/env python
"""Convert Web of Science plain text export file(s) to BibTeX.

BibTeX output is printed on stdout.

Usage::

  python wos2bib.py input1.txt ... inputn.txt >output.bib
"""

from __future__ import annotations

__all__ = ["ptmap", "record_to_bibtex_entry", "records_to_library", "main"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .. import __version__

import argparse
from typing import Iterable

from unpythonic import timer

import bibtexparser
from bibtexparser.model import Entry, Field
import wosfile

from .utils import bibtex_escape

ptmap = {"J": "article",
         "B": "book"}  # TODO: "S" (series), "P" (patent)


def _format_author_addresses(author_addresses) -> str:
    """Normalize WOS author-affiliation data (field C1) to a single string.

    *author_addresses* may be:

    - a ``dict`` of ``{author_name: [affiliation, ...]}``,
    - a plain ``list`` of affiliation strings (no authors attached),
    - falsy (``None`` / empty), in which case the result is ``""``,
    - any other shape, treated as unrecognized and reported by the caller.
    """
    if not author_addresses:
        return ""
    if isinstance(author_addresses, dict):
        def affiliate(author, affiliations):
            affiliations = list(set(affiliations))  # dedupe per author
            return f"{author}, {'. '.join(affiliations)}"
        return "\n".join(affiliate(k, v) for k, v in author_addresses.items())
    if isinstance(author_addresses, list):
        return ". ".join(list(set(author_addresses)))
    return ""  # unrecognized — caller logs, result empty


def record_to_bibtex_entry(rec) -> tuple[Entry | None, str | None]:
    """Convert a single WOS record to a ``bibtexparser`` Entry.

    *rec* is any object with a dict-like ``.get(key, default=None)`` and an
    ``author_address`` attribute (this matches ``wosfile.Record``; tests can
    substitute a plain stub).

    Returns ``(entry, None)`` on success, or ``(None, reason)`` when a required
    field (UT, PT, AU, PY, TI) is missing — in which case *reason* is a short
    human-readable string for logging.
    """
    accession_number = rec.get("UT")
    if accession_number is None:
        return None, "unique identifier missing"

    publication_type = rec.get("PT")
    if not publication_type:
        return None, f"entry '{accession_number}': no publication type specified"

    authors_list = rec.get("AU")
    if not authors_list:
        return None, f"entry '{accession_number}': no authors specified"

    year_published = rec.get("PY")
    if year_published is None:
        return None, f"entry '{accession_number}': no year specified"

    title = rec.get("TI")
    if title is None:
        return None, f"entry '{accession_number}': no title specified"

    authors_str = " and ".join(authors_list)

    publication_name = rec.get("SO")
    volume = rec.get("VL")
    issue = rec.get("IS")
    page_beginning = rec.get("BP")
    page_end = rec.get("EP")
    pages = f"{page_beginning}-{page_end}" if page_beginning is not None and page_end is not None else ""
    doi = rec.get("DI")
    wos_categories_list = rec.get("WC")
    author_addresses_str = _format_author_addresses(getattr(rec, "author_address", None))
    cited_references = rec.get("CR")
    n_cited_references = rec.get("NR")
    abstract = rec.get("AB")

    fields = [Field(key="Author", value=f"{{{authors_str}}}"),
              Field(key="Year", value=f"{{{year_published}}}"),
              Field(key="Title", value=f"{{{title}}}")]
    if publication_name is not None:
        fields.append(Field(key="Journal", value=f"{{{bibtex_escape(publication_name)}}}"))
    if volume is not None:
        fields.append(Field(key="Volume", value=f"{{{volume}}}"))
    if issue is not None:
        fields.append(Field(key="Number", value=f"{{{issue}}}"))  # yes, "Number".
    if pages != "":
        fields.append(Field(key="Pages", value=f"{{{pages}}}"))
    if doi is not None:
        fields.append(Field(key="DOI", value=f"{{{doi}}}"))
    if wos_categories_list:
        fields.append(Field(key="Web-Of-Science-Categories",
                            value=f"{{{bibtex_escape('; '.join(wos_categories_list))}}}"))
    if abstract is not None:
        fields.append(Field(key="Abstract", value=f"{{{bibtex_escape(abstract)}}}"))
    if author_addresses_str != "":
        fields.append(Field(key="Affiliation", value=f"{{{bibtex_escape(author_addresses_str)}}}"))
    if cited_references is not None:
        newline = "\n"
        fields.append(Field(key="Cited-References",
                            value=f"{{{bibtex_escape(newline.join(cited_references))}}}"))
    if n_cited_references is not None and n_cited_references > 0:
        fields.append(Field(key="Number-Of-Cited-References", value=f"{{{n_cited_references}}}"))

    entry = Entry(entry_type=ptmap[publication_type],
                  key=accession_number,
                  fields=fields)
    return entry, None


def records_to_library(records: Iterable) -> tuple[bibtexparser.Library, list[str]]:
    """Convert an iterable of WOS records to a BibTeX library.

    Returns ``(library, skip_reasons)``: *library* contains every record that
    converted cleanly; *skip_reasons* lists one short string per skipped record
    (missing required field), in the order the skips occurred.
    """
    library = bibtexparser.Library()
    skip_reasons: list[str] = []
    for rec in records:
        entry, reason = record_to_bibtex_entry(rec)
        if entry is None:
            skip_reasons.append(reason)
        else:
            library.add(entry)
    return library, skip_reasons


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="""Convert Web of Science plain text export (.wos/.txt) to BibTeX.""",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version=('%(prog)s ' + __version__))
    parser.add_argument(dest="filenames", nargs="+", default=None, type=str, metavar="wos", help="Web of Science (WOS) plain text file(s) to parse")
    opts = parser.parse_args()

    logger.info(f"Reading input file{'s' if len(opts.filenames) != 1 else ''} {opts.filenames}...")
    with timer() as tim:
        # https://webofscience.help.clarivate.com/en-us/Content/export-records.htm
        # https://images.webofknowledge.com/images/help/WOS/hs_wos_fieldtags.html
        library, skip_reasons = records_to_library(wosfile.records_from(opts.filenames))
    for reason in skip_reasons:
        logger.warning(f"    Skipping entry: {reason}")
    print(bibtexparser.writer.write(library))
    logger.info(f"    Done in {tim.dt:0.6g}s.")
    if skip_reasons:
        logger.info(f"    {len(skip_reasons)} entries skipped (see full log above).")


if __name__ == "__main__":
    main()

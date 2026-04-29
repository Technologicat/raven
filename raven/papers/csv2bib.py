#!/usr/bin/env python
"""Convert CSV (comma-separated values) file(s) to BibTeX.

BibTeX output is printed on stdout.

Usage::

  python csv2bib.py input1.csv ... inputn.csv >output.bib

Example input (each indent here represents a single tab character, "\\t"):

Author                       Year    Title             Abstract
A. Author and B. Coauthor    2026    Our Study         Blah blah blah...
McOther, C.                  2026    Some Other Study  Bla bla bla...

Format:

  - The first line MUST be a header containing the column names. Data from each column is populated into a BibTeX field of the same name.
  - Author names MUST be in BibTeX format, and separated by the literal lowercase word "and".
    - Each author name can have up to four parts (first, von, jr., last).
    - Each author name must be in one of three formats:
          First von Last ("First Last" if no "von" part)
          von Last, First ("Last, First" if no "von" part)
          von Last, Jr., First
    - for details, see: https://www.bibtex.com/f/author-field/
  - Fields used by Raven-visualizer:
        Author, Year, Title, Abstract
  - All fields are transcribed, but fields not listed above are not used by Raven-visualizer.
  - For your own item tracking purposes, providing an "Url" or "Doi" field (where available), or some other unique identifier, can be useful.
"""

from __future__ import annotations

__all__ = ["rows_to_library", "main"]

import logging
logger = logging.getLogger(__name__)

from .. import __version__

import argparse
import uuid
from typing import Callable

import bibtexparser
from bibtexparser.model import Entry, Field

from ..common import readcsv
from .utils import bibtex_escape


def _default_key() -> str:
    return str(uuid.uuid4())


def rows_to_library(rows: list[dict[str, str]], key_fn: Callable[[], str] = _default_key) -> bibtexparser.Library:
    """Turn CSV rows (already parsed into dicts) into a ``bibtexparser.Library``.

    Each row becomes an ``@article`` entry whose BibTeX fields are the row's
    (column-name, value) pairs.  Values are passed through :func:`bibtex_escape`
    and wrapped in braces.

    *key_fn*: zero-argument callable producing a BibTeX key for each entry.
    Defaults to ``str(uuid.uuid4())``; override for deterministic output
    (e.g. in tests).
    """
    library = bibtexparser.Library()
    for row in rows:
        fields = [Field(key=k, value=f"{{{bibtex_escape(v)}}}") for k, v in row.items()]
        entry = Entry(entry_type="article", key=key_fn(), fields=fields)
        library.add(entry)
    return library


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="""Convert CSV (comma-separated values) files to BibTeX. The first line must be a header with field names; fields with those names will be populated in the output.""",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version=('%(prog)s ' + __version__))
    parser.add_argument(dest="filenames", nargs="+", default=None, type=str, metavar="csv", help="CSV file(s) to parse")
    parser.add_argument('-d', '--delimiter', dest="delimiter", type=str, metavar="x", default=None, help="Column delimiter. If omitted, autodetects tab/semicolon.")
    parser.add_argument('--log', metavar='PATH', default=None,
                        help='mirror stderr log to this file (overwritten each run)')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='root logger level (default: INFO)')
    opts = parser.parse_args()

    from ..common import logsetup
    logsetup.configure(level=getattr(logging, opts.log_level),
                       logfile=opts.log)

    logger.info(f"Reading input file{'s' if len(opts.filenames) != 1 else ''} {opts.filenames}...")
    all_rows: list[dict] = []
    for filename in opts.filenames:
        all_rows.extend(readcsv.parse_csv(filename, has_header=True, delimiter=opts.delimiter))

    logger.info("Creating BibTeX database...")
    library = rows_to_library(all_rows)
    print(bibtexparser.writer.write(library))


if __name__ == "__main__":
    main()

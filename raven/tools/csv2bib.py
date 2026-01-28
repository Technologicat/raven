#!/usr/bin/env python
"""Convert CSV (comma-separated values) file(s) to BibTeX.

BibTeX output is printed on stdout.

Usage::

  python csv2bib.py input1.csv ... inputn.csv >output.bib

Example input (each indent here represents a single tab character, "\t"):

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

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .. import __version__

import argparse
import uuid

import bibtexparser
from bibtexparser.model import Entry, Field

from ..common import readcsv

def bibtex_escape(s: str):
    s = s.replace("\\", "\\\\")
    s = s.replace("{", "{{")
    s = s.replace("}", "}}")
    s = s.replace("[", "{[}")
    s = s.replace("&", r"\&")
    s = s.replace("%", r"\%")
    return s

def main():
    parser = argparse.ArgumentParser(description="""Convert CSV (comma-separated values) files to BibTeX. The first line must be a header with field names; fields with those names will be populated in the output.""",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version=('%(prog)s ' + __version__))
    parser.add_argument(dest="filenames", nargs="+", default=None, type=str, metavar="csv", help="CSV file(s) to parse")
    parser.add_argument('-d', '--delimiter', dest="delimiter", type=str, metavar="x", default=None, help="Column delimiter. If omitted, autodetects tab/semicolon.")
    opts = parser.parse_args()

    # Read in all input files
    logger.info(f"Reading input file{'s' if len(opts.filenames) != 1 else ''} {opts.filenames}...")
    all_entries = []
    for filename in opts.filenames:
        entries = readcsv.parse_csv(filename,
                                    has_header=True,
                                    delimiter=opts.delimiter)
        all_entries.extend(entries)

    # Create the BibTeX database
    logger.info("Creating BibTeX database...")
    library = bibtexparser.Library()
    for entry in entries:
        fields = []
        for key, value in entry.items():
            fields.append(Field(key=key, value=f"{{{bibtex_escape(value)}}}"))
        slug = str(uuid.uuid4())
        entry = Entry(entry_type="article",
                      key=slug,
                      fields=fields)
        library.add(entry)
    print(bibtexparser.writer.write(library))

if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""CLI shell for the PDF-to-BibTeX converter.

This is the `raven-pdf2bib` console-script entry point. It parses CLI args and configures logging *before*
importing `raven.papers.pdf2bib`, which reaches the NLP stack and the LLM client — several seconds that
`--help`, or a mistyped option, would otherwise wait through before argparse could answer.

USAGE:

    raven-pdf2bib --slug CONF2024 --year 2024 -i abstracts -o done 1>entries.bib

This writes `entries.bib`, and moves each input PDF into `done` once its BibTeX entry has been printed, so a
large batch can be continued later.
"""

__all__ = ["main"]

import argparse
import logging

from .. import __version__
from ..client import config as client_config  # the help text shows its defaults
from ..librarian import config as librarian_config  # likewise


def main() -> None:
    parser = argparse.ArgumentParser(description="""Convert PDF conference abstracts into a BibTeX database. Extracts the PDF text and processes it with an OpenAI compatible LLM.""",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--backend-url", dest="backend_url", default=librarian_config.llm_backend_url, type=str, metavar="url", help=f"LLM backend to talk to, overriding the configured one (default: '{librarian_config.llm_backend_url}').")
    parser.add_argument("--server-url", dest="server_url", default=None, type=str, metavar="url", help=f"Raven server to talk to, overriding the configured one (default: '{client_config.raven_server_url}'). Used for dehyphenating extracted abstracts.")

    conf = parser.add_argument_group("conference info", "Metadata for the conference; injected into all generated BibTeX entries.")
    conf.add_argument("--slug", dest="conference_slug", required=True, type=str, metavar="SLUG", help="Short conference identifier for BibTeX entry keys (e.g. ECCOMAS2024).")
    conf.add_argument("--year", dest="conference_year", required=True, type=str, metavar="YEAR", help="Conference year (e.g. 2024).")
    conf.add_argument("--booktitle", dest="conference_booktitle", default=None, type=str, metavar="TITLE", help="Full conference title for the BibTeX booktitle field (optional).")
    conf.add_argument("--note", dest="conference_note", default=None, type=str, metavar="NOTE", help="Conference note, e.g. dates and location (optional).")
    conf.add_argument("--url", dest="conference_url", default=None, type=str, metavar="URL", help="Conference URL (optional).")

    parser.add_argument("-s", "--success", dest="success_filename", type=str, metavar="success.bib", help="Output BibTeX file for successful entries (default stdout). Will be appended to.")
    parser.add_argument("-f", "--failed", dest="failed_filename", type=str, metavar="failed.bib", help="Output BibTeX file for failed entries (default: send these too to the success output). Will be appended to. As detected by heuristics, requiring manual verification/fixes.")
    parser.add_argument("-r", "--retries", dest="retries", default=3, type=int, metavar="x", help="Up to this many attempts (default: 3) will be made at the various processing steps for author extraction, when the processing fails. The number set here includes the initial attempt, so '-r 3' means 'try, and then retry up to twice if needed'. Attempts are counted separately for each processing step; each step gets this many retries if needed. This often helps get the LLM unstuck, especially if it starts overthinking and fails to produce a final response within the maximum token limit for a reply.")
    parser.add_argument("-l", "--log", metavar="log.txt", default=None, help="Output logfile, for a copy of the console log (overwritten each run). Useful for seeing what went wrong in each specific failed entry.")
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='root logger level (default: INFO)')
    parser.add_argument("-o", "--output-dir", dest="output_dir", default=None, type=str, metavar="dir", help="directory to move done files into (optional; allows easily continuing later). If also `-of` is specified, then only successful files will be moved to the `-o` directory; failed files will be moved to the `-of` directory.")
    parser.add_argument("-of", "--failed-output-dir", dest="failed_output_dir", default=None, type=str, metavar="dir", help="directory to move failed done files into (optional; allows easily continuing later)")
    parser.add_argument("-i", "--input-dir", dest="input_dir", default=None, type=str, metavar="input_dir", help="Input directory containing PDF file(s) to import (will be scanned recursively, skipping output dirs)")
    parser.add_argument('-v', '--version', action='version', version=('%(prog)s ' + __version__))
    opts = parser.parse_args()

    from ..common import logsetup  # noqa: PLC0415 -- not needed to parse
    logsetup.configure(level=getattr(logging, opts.log_level),
                       logfile=opts.log,
                       allow=["raven.papers.pdf2bib"])  # the library's own log only

    from . import pdf2bib  # noqa: PLC0415 -- heavy, and imported after logging is set up
    pdf2bib.run(opts)


if __name__ == "__main__":
    main()

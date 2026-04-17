"""Burst a BibTeX file into individual entries.

These individual BibTeX record files are useful for feeding into Raven-librarian's document database,
especially if the records contain an abstract.
"""

# TODO: Robustify. Ideally, do this with `bibtexparser` (need to create a single-entry library for each output).

from __future__ import annotations

__all__ = ["is_headerline", "get_slug", "burst_bibtex", "main"]

from .. import __version__

import argparse
import io
import pathlib

from ..common import stringmaps
from ..common import utils as common_utils


def is_headerline(line: str) -> str | bool:
    """Detect whether *line* is a BibTeX record header. Return stripped *line* or ``False``."""
    line = line.strip()
    if line.startswith("@") and line.endswith(","):
        return line
    return False


def get_slug(headerline: str) -> str:
    """Get the BibTeX unique identifier from a BibTeX record header line.

    The slug is sanitized for use as a filename.
    """
    start_of_slug = headerline.find("{")
    end_of_slug = headerline.rfind(",")
    if start_of_slug == -1 or end_of_slug == -1:
        assert False
    slug = headerline[(start_of_slug + 1):end_of_slug]
    # Make safe for filename, to tolerate broken `.bib` files (users not familiar
    # with BibTeX may have used e.g. a DOI or an URL as the slug)
    slug = "".join(c for c in slug if c.isalnum() or c in stringmaps.filename_safe_nonalphanum)
    return slug


def burst_bibtex(source: str) -> list[tuple[str, str]]:
    """Split a BibTeX file's contents into individual records.

    Returns ``[(slug, record_text), ...]`` — one pair per ``@type{key,...}``
    record found in *source*, in the order they appear.  Each ``record_text``
    is the full record (from its ``@type{...,`` header up to, but not
    including, the next record's header or end of input), and *slug* is the
    BibTeX key from the header, sanitized for use as a filename (see
    :func:`get_slug`).

    Anything before the first header line is silently skipped, matching the
    CLI's tolerance for leading comments, blank lines, or ``@preamble`` /
    ``@string`` blocks that don't look like entries.
    """
    records: list[tuple[str, str]] = []
    buf = io.StringIO(source)
    record = io.StringIO()
    slug: str | None = None

    # Sync to the first real record
    while True:
        line = buf.readline()
        if line == "":  # EOF before any record
            return records
        if is_headerline(line):
            slug = get_slug(line)
            record.write(line)
            break

    # Accumulate remaining lines into records
    while True:
        line = buf.readline()
        if line == "":  # EOF — flush the last record
            assert slug is not None
            records.append((slug, record.getvalue()))
            return records
        if is_headerline(line):  # start of next record — flush the current one
            assert slug is not None
            records.append((slug, record.getvalue()))
            record = io.StringIO()
            slug = get_slug(line)
            record.write(line)
        else:
            record.write(line)


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="""Burst a BibTeX file into individual entries, for Raven-librarian's document database.""",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(dest="filenames", nargs="+", default=None, type=str, metavar="myreferences.bib", help="BibTeX file(s) to burst")
    parser.add_argument("-o", "--output-dir", dest="output_dir", default=None, type=str, metavar="output_dir", help="Output directory to write the individual BibTeX records in.")
    parser.add_argument('-v', '--version', action='version', version=('%(prog)s ' + __version__))
    parser.add_argument("-V", "--verbose", dest="verbose", action="store_true", default=False, help="Print progress messages.")
    opts = parser.parse_args()

    if opts.output_dir is None:
        opts.output_dir = "."
    output_dir = pathlib.Path(opts.output_dir).expanduser().resolve()
    if opts.verbose:
        print(f"Creating output directory '{opts.output_dir}' (resolved to '{str(output_dir)}')")
    common_utils.create_directory(output_dir)

    def get_output_path(slug: str) -> pathlib.Path:
        primary_output_path = output_dir / f"{slug}.bib"
        output_path = primary_output_path
        counter = 2
        while output_path.exists():
            output_path = output_dir / f"{slug}_{counter}.bib"
            counter += 1
        if counter > 2:
            print(f"    '{str(primary_output_path)}' already exists, writing to '{str(output_path)}' instead.")
        return output_path

    for input_filename in opts.filenames:
        input_path = pathlib.Path(input_filename).expanduser().resolve()
        if opts.verbose:
            print(f"Processing '{input_filename}' (resolved to '{str(input_path)}')")
        source = input_path.read_text()
        for slug, record_text in burst_bibtex(source):
            output_path = get_output_path(slug)
            if opts.verbose:
                print(f"    Writing '{str(output_path)}'")
            output_path.write_text(record_text)


if __name__ == "__main__":
    main()

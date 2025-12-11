"""Burst a BibTeX file into individual entries.

These individual BibTeX record files are useful for feeding into Raven-librarian's document database,
especially if the records contain an abstract.
"""

# TODO: Robustify. Ideally, do this with `bibtexparser` (need to create a single-entry library for each output).

from .. import __version__

import argparse
import io
import pathlib
from typing import Union

from ..common import stringmaps
from ..common import utils as common_utils

def main() -> None:
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

    for input_filename in opts.filenames:
        input_path = pathlib.Path(input_filename).expanduser().resolve()
        if opts.verbose:
            print(f"Processing '{input_filename}' (resolved to '{str(input_path)}')")
        with open(input_path, "r") as input_file:
            def is_headerline(line) -> Union[str, bool]:
                """Detect whether `line` is a BibTeX record header. Return `line` or `False`."""
                line = line.strip()
                if line.startswith("@") and line.endswith(","):
                    return line
                return False

            def get_slug(headerline):
                """Get the BibTeX unique identifier from a BibTeX record header line."""
                start_of_slug = headerline.find("{")
                end_of_slug = headerline.rfind(",")
                if start_of_slug == -1 or end_of_slug == -1:
                    assert False
                slug = headerline[(start_of_slug + 1):end_of_slug]
                # Make safe for filename, to tolerate broken `.bib` files (users not familiar with BibTeX may have used e.g. a DOI or an URL as the slug)
                slug = "".join(c for c in slug if c.isalnum() or c in stringmaps.filename_safe_nonalphanum)
                return slug

            def get_output_path(slug):
                primary_output_path = output_dir / f"{slug}.bib"
                output_path = primary_output_path

                # make unique path
                counter = 2
                while output_path.exists():
                    output_path = output_dir / f"{slug}_{counter}.bib"
                    counter += 1

                if counter > 2:
                    print(f"    '{str(primary_output_path)}' already exists, writing to '{str(output_path)}' instead.")

                return output_path

            def sync():
                """Find first BibTeX record header."""
                while True:
                    line = input_file.readline()
                    if line == "":  # EOF
                        return None
                    if is_headerline(line):
                        return line

            # Sync to first BibTeX record in this file
            record = io.StringIO()
            headerline = sync()
            if headerline is None:
                continue  # next file
            slug = get_slug(headerline)
            record.write(headerline)

            # Process all records
            while True:
                line = input_file.readline()
                if line == "":  # EOF
                    output_path = get_output_path(slug)
                    if opts.verbose:
                        print(f"    Writing '{str(output_path)}'")
                    with open(output_path, "w") as output_file:
                        output_file.write(record.getvalue())
                    break

                if is_headerline(line):  # start of next record
                    output_path = get_output_path(slug)
                    if opts.verbose:
                        print(f"    Writing '{str(output_path)}'")
                    with open(output_path, "w") as output_file:
                        output_file.write(record.getvalue())
                    record = io.StringIO()
                    headerline = line
                    slug = get_slug(headerline)
                    record.write(headerline)
                else:  # part of current record
                    record.write(line)

if __name__ == "__main__":
    main()

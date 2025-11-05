"""Get the arXiv identifiers for all arXiv PDF files in the given directory.

We assume that the identifier (yymm.xxxxx) is somewhere in the filename.

The non-verbose (default) output can be handed to `arxiv2bib` (https://github.com/nathangrigg/arxiv2bib)
to create a BibTeX bibliography of those papers (by auto-downloading metadata from arXiv).
"""

from .. import __version__

import argparse
import os
import pathlib
import re
from typing import List

from unpythonic import uniqify

def list_subfolders(path: str) -> List[str]:
    # blacklist = []  # ["stuff"]  # don't descend into these directories, at any level. Directories with names beginning with "00_" are automatically ignored.
    paths = []
    for root, dirs, files in os.walk(path):
        paths.append(root)

        new_dirs = [x for x in dirs if not x.startswith("00_")]
        dirs.clear()
        dirs.extend(new_dirs)
        # for x in blacklist:
        #     if x in dirs:
        #         dirs.remove(x)
    paths = list(sorted(uniqify(str(pathlib.Path(p).expanduser().resolve()) for p in paths)))
    return paths

def list_pdf_files(path):
    return list(sorted(filename for filename in os.listdir(path) if filename.lower().endswith(".pdf")))

arxiv_identifier = re.compile(r"\b(\d\d\d\d\.\d\d\d\d\d)(v\d+)?\b")
def get_arxiv_identifier(path: str) -> str:
    filename = os.path.basename(path)
    matches = re.findall(arxiv_identifier, filename)
    matches = [''.join(x) for x in matches]  # [("yymm.xxxxx", "v2"), ...] -> ["yymm.xxxxxv2", ...]
    if matches:
        assert len(matches) == 1
        return matches[0]
    return None

def split_arxiv_identifier(raw_identifier: str) -> int:
    splitted = raw_identifier.split("v")
    if len(splitted) == 1:  # no version
        return raw_identifier.strip(), 1
    base, version = splitted
    return base.strip(), int(version.strip())

def main() -> None:
    parser = argparse.ArgumentParser(description="""List identifiers of arXiv papers in the specified directory. The papers are assumed to be PDF files with the arXiv identifier (yymm.xxxxx) somewhere in the filename. Only unique identifiers are returned (even if duplicated under different filenames, as long as the identifier is the same). To avoid duplicates, in case of multiple versions of the same paper (yymm.xxxxxvN), only the most recent version is returned.""",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input-dir", dest="input_dir", default=None, type=str, metavar="input_dir", help="Input directory containing arXiv PDF file(s).")
    parser.add_argument('-v', '--version', action='version', version=('%(prog)s ' + __version__))
    parser.add_argument("-V", "--verbose", dest="verbose", action="store_true", default=False, help="Print also the filename for each match, for debugging your collection ('where did that 3904.36424 come from, it is not the year 2039 yet?).")
    opts = parser.parse_args()

    if opts.input_dir is None:
        opts.input_dir = "."

    arxiv_pdf_files = {}
    paths = list_subfolders(opts.input_dir)
    for path in sorted(paths):
        pdf_files = list_pdf_files(path)
        if not pdf_files:
            continue
        arxiv_pdf_files_in_this_dir = [(raw_id, p) for p in pdf_files if (raw_id := get_arxiv_identifier(p))]
        base_ids_and_versions = [split_arxiv_identifier(raw_id) for raw_id, p in arxiv_pdf_files_in_this_dir]

        # Pick the latest version for each identifier discovered so far
        for (raw_id, p), (base_id, version) in zip(arxiv_pdf_files_in_this_dir, base_ids_and_versions):
            if (base_id not in arxiv_pdf_files):
                arxiv_pdf_files[base_id] = (raw_id, p, version)
            else:
                recorded_raw_id, recorded_p, recorded_version = arxiv_pdf_files[base_id]
                if version > recorded_version:
                    arxiv_pdf_files[base_id] = (raw_id, p, version)

    arxiv_pdf_files = list(sorted(arxiv_pdf_files.values()))
    for identifier, filename, version in arxiv_pdf_files:
        if opts.verbose:
            print(identifier, filename)
        else:
            print(identifier)

if __name__ == "__main__":
    main()

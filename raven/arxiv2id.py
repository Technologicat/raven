"""Get the arXiv identifiers for all arXiv PDF files in the current directory.

We assume that the identifier (yymm.xxxxx) is somewhere in the filename.

The non-verbose (default) output can be handed to `arxiv2bib` (https://github.com/nathangrigg/arxiv2bib)
to create a BibTeX bibliography of those papers (by auto-downloading metadata from arXiv).
"""

import argparse
import operator
import os
import pathlib
import re

from unpythonic import uniqify

def list_subfolders(path):
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
def get_arxiv_identifier(path):
    filename = os.path.basename(path)
    matches = re.findall(arxiv_identifier, filename)
    matches = [''.join(x) for x in matches]  # [("yymm.xxxxx", "v2"), ...] -> ["yymm.xxxxxv2", ...]
    if matches:
        assert len(matches) == 1
        return matches[0]
    return None

def main():
    parser = argparse.ArgumentParser(description="""List identifiers of arXiv papers in the specified directory. The papers are assumed to be PDF files with the arXiv identifier (yymm.xxxxx) somewhere in the filename. Only unique identifiers are returned (even if duplicated under different filenames, as long as the identifier is the same).""",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input-dir", dest="input_dir", default=None, type=str, metavar="input_dir", help="Input directory containing arXiv PDF file(s).")
    parser.add_argument("-V", "--verbose", dest="verbose", action="store_true", default=False, help="Print also the filename for each match, for debugging your collection ('where did that 3904.36424' come from, it's not the year 2039 yet?).")
    opts = parser.parse_args()

    if opts.input_dir is None:
        opts.input_dir = "."

    arxiv_pdf_files = []
    paths = list_subfolders(opts.input_dir)
    for path in sorted(paths):
        pdf_files = list_pdf_files(path)
        if not pdf_files:
            continue
        arxiv_pdf_files_in_this_dir = [(arxiv_id, p) for p in pdf_files if (arxiv_id := get_arxiv_identifier(p))]
        arxiv_pdf_files.extend(arxiv_pdf_files_in_this_dir)

    arxiv_pdf_files = list(uniqify(sorted(arxiv_pdf_files),
                                   key=operator.itemgetter(0)))
    for identifier, filename in arxiv_pdf_files:
        if opts.verbose:
            print(identifier, filename)
        else:
            print(identifier)

if __name__ == "__main__":
    main()

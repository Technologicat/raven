#!/usr/bin/env python
"""Build the folder the file dialog's documentation screenshots are taken in.

A small, tidy directory of readable names — subfolders, images, two PDFs, a few text files — so that both of
the dialog's looks, the list view and the thumbnail grid, have something to show, and nothing in the shot is
anybody's own files or home path. Everything is made from files tracked in this repository, so any clone can
rebuild it, and the result is the same every time: the same names, the same contents, and the same
modification dates, since the dialog's *Date* column shows them.

    python scripts/make_screenshot_demo_folder.py               # into /tmp/raven-demo
    python scripts/make_screenshot_demo_folder.py --dest DIR

An existing destination is replaced.
"""

import argparse
import datetime
import os
import pathlib
import shutil
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent

# Destination name -> tracked source. Order is the order the dates are handed out in.
COPIES = {
    "aria.png": "raven/avatar/assets/characters/other/aria1.png",
    "Backdrops/anime-plains.png": "raven/avatar/assets/backdrops/anime-plains.png",
    "Backdrops/cyberspace.png": "raven/avatar/assets/backdrops/cyberspace.png",
    "Backdrops/study.png": "raven/avatar/assets/backdrops/study.png",
    "raven-logo.png": "img/logo.png",
    "chat-tree-diagram.png": "img/chattree-diagram.png",
    "embedding-space.png": "img/embedding_space_ai.png",
    "data-pipeline.png": "img/raven-data-processing-pipeline-ai.png",
    "ai-agent.png": "img/ai-agent.png",
    "hybrid-search.png": "img/raven-search.png",
}
# Illustrations and assets only, never the screenshots in `img/`: those are retaken every release, and a
# folder built from them would change with them, which is the one thing this folder must not do.

# Destination name -> tracked image, written out as a one-page PDF.
PDFS = {
    "Papers/embedding-space-explained.pdf": "img/embedding_space_ai.png",
    "Papers/data-processing-pipeline.pdf": "img/raven-data-processing-pipeline-ai.png",
}

TEXTS = {
    "reading-list.md": ("# Reading list\n\n"
                        "- Hybrid search: BM25 + embeddings, fused by reciprocal rank\n"
                        "- Chat trees and branching histories\n"),
    "Notes/meeting-notes.txt": ("Questions for the next meeting:\n"
                                "- Which corpus to index first?\n"
                                "- Subtitle language for the demo?\n"),
    "Notes/ideas.txt": "Things to try with the avatar: zoom, bloom, the CRT filter.\n",
}

# The first file's date; each later one is a few hours on, so the Date column reads as a folder built up
# over a week or so rather than as one instant.
EPOCH = datetime.datetime(2026, 9, 14, 9, 30)
STEP = datetime.timedelta(hours=7, minutes=13)


def build(dest: pathlib.Path) -> list[pathlib.Path]:
    """Build the folder at `dest`, replacing whatever is there. Returns the files written, in date order."""
    from PIL import Image  # only here; the rest of this script needs nothing outside the stdlib

    if dest.exists():
        shutil.rmtree(dest)
    written = []

    def target(name: str) -> pathlib.Path:
        path = dest / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    for name, source in COPIES.items():
        shutil.copyfile(REPO / source, target(name))
        written.append(dest / name)
    for name, source in PDFS.items():
        with Image.open(REPO / source) as image:
            # Dated explicitly: left alone, Pillow stamps the current second into the file, so two builds a
            # second apart would differ. A `struct_time`, which is what its PDF writer can serialize.
            image.convert("RGB").save(target(name), "PDF", resolution=150.0,
                                      creationDate=EPOCH.timetuple(), modDate=EPOCH.timetuple())
        written.append(dest / name)
    for name, text in TEXTS.items():
        target(name).write_text(text, encoding="utf-8")
        written.append(dest / name)

    for i, path in enumerate(written):
        stamp = (EPOCH + i * STEP).timestamp()
        os.utime(path, (stamp, stamp))
    # The folders last, since writing into one moves its date to now; the destination itself last of all.
    for i, folder in enumerate(sorted({path.parent for path in written if path.parent != dest})):
        stamp = (EPOCH + i * STEP).timestamp()
        os.utime(folder, (stamp, stamp))
    stamp = (EPOCH + len(written) * STEP).timestamp()
    os.utime(dest, (stamp, stamp))
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dest", type=pathlib.Path, default=pathlib.Path("/tmp/raven-demo"),
                        help="where to build it (default: %(default)s); replaced if it exists")
    args = parser.parse_args()
    written = build(args.dest)
    print(f"Wrote {len(written)} files to {args.dest}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

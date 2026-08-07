"""Repair BibTeX records that a parser refuses, by escaping stray braces in field values.

The records this rescues are the ones a PDF extractor produced: mathematics reaches a BibTeX file as
things like `{0 <= rho <= 1`, set-builder notation whose closing brace was dropped somewhere in the
pipeline.

Which of the two braces went missing decides what happens next, and the two failures are nothing alike:

- **A surplus opening brace** leaves the field value with no terminator, so the parser reads on looking
  for one and eventually gives up on the record. That is the loud failure, and the recoverable one: the
  whole record is missing from anything reading the file — title, authors and all — but it is *reported*
  missing, and the following records survive because the parser resynchronizes at the next `@`. This is
  what the tool repairs.
- **A surplus closing brace** ends the value early instead, and the record still parses. Nothing is
  reported, because as far as the parser is concerned nothing went wrong: the value is quietly truncated
  at the stray brace, and every field *after* it in the record is dropped. A record can lose its title
  this way and still look like a perfectly good record.

So the tool sees only the first kind, since the second never reaches `failed_blocks` to be repaired.

Raven's own readers repair what they read, so a broken record still reaches the document database and the
Visualizer. This tool exists for the other half: the `.bib` file is the user's own artifact, shared with
collaborators and read by tools that are not Raven, so there has to be a way to fix the file itself. It
writes a new file by default, and touches the original only when asked to.
"""

from __future__ import annotations

__all__ = ["repair_bibtex", "main"]

from .. import __version__

import argparse
import logging
import pathlib
import sys

from ..common import utils as common_utils
from . import bibtex


def repair_bibtex(source: str) -> tuple[str, list[str], list[str]]:
    """Repair the unparseable records in the BibTeX file contents `source`.

    Returns `(repaired_source, recovered_keys, unrecovered_descriptions)`, where the descriptions name
    each record that could not be repaired and the fields that look responsible.

    Records that already parse are copied through untouched, byte for byte. Only a failed record is
    rewritten, and only ever by escaping braces — see `bibtex.repair_record` for why that is the one
    edit worth making unsupervised.
    """
    library = bibtex.parse_string(source)
    if not library.failed_blocks:
        return source, [], []

    # Each block is located by searching for its own text rather than by indexing `splitlines()` with its
    # reported `start_line`. Those two count lines differently: `start_line` counts newlines, while
    # `str.splitlines` also breaks on the other Unicode line boundaries - `\x1c`, `\x1d`, `\x85`, ` `
    # and friends - so a file containing any of them yields more "lines" than it has newlines, and every
    # block after the first one drifts. A bibliography extracted from PDFs is exactly where such
    # characters turn up: the ECCOMAS 2024 file carries five `\x1c` and one `\x1d`, which is enough to
    # splice a repair six lines into the previous record. Searching for `raw`, which is verbatim, sidesteps
    # the disagreement entirely, and the offset it returns also gives a line number worth printing.
    located = []
    cursor = 0
    for failed_block in library.failed_blocks:
        start = source.find(failed_block.raw, cursor)
        if start == -1:  # should not happen; skip rather than corrupt the file on a guess
            continue
        located.append((start, failed_block.raw))
        cursor = start + len(failed_block.raw)

    pieces, recovered, unrecovered = [], [], []
    cursor = 0
    for start, raw in located:
        key = common_utils.bibtex_header_key(raw.lstrip().split("\n", 1)[0]) or "?"
        maybe_repaired = bibtex.repair_record(raw)

        pieces.append(source[cursor:start])
        pieces.append(raw if maybe_repaired is None else maybe_repaired)
        cursor = start + len(raw)

        if maybe_repaired is None:
            unbalanced = common_utils.bibtex_unbalanced_field_names(raw)
            suspects = f", suspect field(s): {', '.join(unbalanced)}" if unbalanced else ""
            unrecovered.append(f"'{key}' at line {source.count(chr(10), 0, start) + 1}{suspects}")
        else:
            recovered.append(key)

    pieces.append(source[cursor:])
    return "".join(pieces), recovered, unrecovered


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="""Repair BibTeX records whose field values contain unbalanced braces, which a parser refuses to read. Writes a new file unless asked to edit in place.""",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-o", "--output-suffix", dest="output_suffix", default="_fixed", type=str, metavar="suf",
                        help="Suffix for naming output files (file.bib -> file_fixed.bib).")
    parser.add_argument("-i", "--in-place", dest="in_place", action="store_true", default=False,
                        help="Write the repair back to the input file instead of to a new one. Your bibliography is yours, so this never happens unless you ask for it.")
    parser.add_argument("-n", "--dry-run", dest="dry_run", action="store_true", default=False,
                        help="Report what would be repaired, and write nothing.")
    parser.add_argument(dest="filenames", nargs="+", default=None, type=str, metavar="myreferences.bib",
                        help="BibTeX file(s) to repair.")
    opts = parser.parse_args()

    # `bibtexparser` logs a warning for each record it cannot read, and this tool reports the same records
    # itself with a line number and the fields to look at. Two accounts of the same problem, interleaved,
    # is worse than one. Set here rather than in `repair_bibtex`, because muting someone else's logger is a
    # decision an application gets to make and a library does not.
    logging.getLogger("bibtexparser").setLevel(logging.ERROR)

    total_recovered = total_unrecovered = 0
    for filename in opts.filenames:
        path = pathlib.Path(filename).expanduser().resolve()
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"{path}: cannot read ({type(exc).__name__}: {exc})", file=sys.stderr)
            continue

        repaired, recovered, unrecovered = repair_bibtex(source)
        total_recovered += len(recovered)
        total_unrecovered += len(unrecovered)

        for key in recovered:
            print(f"{path.name}: repaired '{key}'")
        for description in unrecovered:
            print(f"{path.name}: could not repair {description}", file=sys.stderr)

        if not recovered:
            print(f"{path.name}: nothing to repair.")
            continue
        if opts.dry_run:
            print(f"{path.name}: {len(recovered)} record(s) would be repaired; nothing written (--dry-run).")
            continue

        out_path = path if opts.in_place else path.with_name(f"{path.stem}{opts.output_suffix}{path.suffix}")
        out_path.write_text(repaired, encoding="utf-8")
        print(f"{path.name}: {len(recovered)} record(s) repaired -> {out_path}")

    if total_unrecovered:
        # A record that lost its field terminator rather than gaining a stray brace cannot be repaired by
        # escaping, and guessing where the missing brace belonged is not something to do to someone's data.
        print(f"\n{total_unrecovered} record(s) need a look by hand: the brace that would balance them is "
              f"missing rather than stray, and nothing here can know where it belonged.", file=sys.stderr)
    if total_recovered or total_unrecovered:
        print(f"Total: {total_recovered} repaired, {total_unrecovered} left for you.")


if __name__ == "__main__":  # pragma: no cover
    main()

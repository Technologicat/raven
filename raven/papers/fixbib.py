"""Repair BibTeX records that a parser refuses, and say what was wrong with the ones that stay refused.

Two faults are repaired, and they come from opposite ends of the toolchain.

**Stray braces in a field value** are what a PDF extractor produces: mathematics reaches a BibTeX file as
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

**A field named twice** is what a database export produces: a ProQuest record carries a separate `annote`
for its copyright statement, its last-updated date and its subject terms, and BibTeX has no way to say
that, so the parser rejects the entry whole. The repeats are merged into one field rather than thinned to
one, since each holds something different and choosing between them is not a repair.

The second fault is worth knowing about for how *much* of a file it can take: 1598 of the 6934 records in
one real multi-database export, none of them reported as anything more specific than "unparseable".

Whatever is left is described rather than guessed at, one line per record, naming the fault and the fields
that carry it. A record whose author is written `Bloggs, PhD, MSc, Joan` cannot be repaired here — BibTeX
gives a name two commas and that one uses three — and saying so is more useful than a silent omission.

Raven's own readers repair what they read, so a broken record still reaches the document database and the
Visualizer. This tool exists for the other half: the `.bib` file is the user's own artifact, shared with
collaborators and read by tools that are not Raven, so there has to be a way to fix the file itself. It
writes a new file by default, and touches the original only when asked to.
"""

from __future__ import annotations

__all__ = ["KIND_UNBALANCED_BRACES", "KIND_DUPLICATE_FIELD_KEYS", "KIND_UNREADABLE",
           "RepairReport", "repair_bibtex", "main"]

from .. import __version__

import argparse
import collections
import dataclasses
import logging
import pathlib
import sys

from bibtexparser.model import DuplicateFieldKeyBlock

from ..common import utils as common_utils
from . import bibtex

KIND_UNBALANCED_BRACES = "unbalanced braces"
KIND_DUPLICATE_FIELD_KEYS = "duplicate field keys"
KIND_UNREADABLE = "unreadable"


@dataclasses.dataclass(frozen=True)
class RepairReport:
    """One record `repair_bibtex` found fault with, and what it made of it.

    Frozen, and a record rather than this codebase's usual `env`, because these are counted and grouped
    by `kind` — a mistyped field should fail rather than read as `None` and quietly drop out of a tally.

    Fields:

    `key`: The record's BibTeX key, or `"?"` where the header line was too damaged to yield one.
    `line`: 1-based line number in the *input*, so it points at what the user still has open.
    `kind`: One of the `KIND_*` constants — what was wrong, not what was done about it.
    `detail`: The specifics: which fields repeat, which look unbalanced, or the parser's own complaint.
    """
    key: str
    line: int
    kind: str
    detail: str

    def describe(self) -> str:
        """A one-line account of this record, for a terminal. Does not name the file."""
        return f"'{self.key}' at line {self.line}: {self.kind} ({self.detail})"


def _diagnose(failed_block, key: str, line: int) -> RepairReport:
    """Say what is wrong with one record `bibtexparser` refused.

    Runs before any repair is attempted and describes the *fault*, so the same report serves whether the
    record goes on to be repaired or not.
    """
    if isinstance(failed_block, DuplicateFieldKeyBlock):
        repeated = ", ".join(sorted(failed_block.duplicate_keys))
        return RepairReport(key, line, KIND_DUPLICATE_FIELD_KEYS, f"repeats {repeated}")

    unbalanced = common_utils.bibtex_unbalanced_field_names(failed_block.raw)
    if unbalanced:
        return RepairReport(key, line, KIND_UNBALANCED_BRACES,
                            f"suspect field(s): {', '.join(unbalanced)}")
    # No field opens more braces than it closes, so the fault is something else the parser named itself.
    # Its own words beat a guess here: "Cannot split the following name `Bloggs, PhD, MSc, Joan` into
    # parts: Too many commas" says exactly which line to go and edit.
    return RepairReport(key, line, KIND_UNREADABLE, str(failed_block.error).replace("\n", " "))


def _repair(failed_block) -> str | None:
    """Repair one record, choosing the repair by what is wrong with it. `None` if it stays unreadable."""
    if isinstance(failed_block, DuplicateFieldKeyBlock):
        return bibtex.repair_duplicate_field_keys(failed_block.raw, failed_block.duplicate_keys)
    return bibtex.repair_record(failed_block.raw)


def repair_bibtex(source: str) -> tuple[str, list[RepairReport], list[RepairReport]]:
    """Repair the unparseable records in the BibTeX file contents `source`.

    Returns `(repaired_source, recovered, unrecovered)`, two lists of `RepairReport` describing the
    records that were repaired and the records that were not. Both name the fault, so a caller can
    report *why* a file was unreadable and not only how much of it was.

    Records that already parse are copied through untouched, byte for byte. Only a failed record is
    rewritten, and only ever by escaping braces or by merging fields it names twice — see
    `bibtex.repair_record` and `bibtex.repair_duplicate_field_keys` for why those two edits are safe to
    make unsupervised, where inventing a missing brace or discarding one of two values would not be.
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
        located.append((start, failed_block))
        cursor = start + len(failed_block.raw)

    pieces, recovered, unrecovered = [], [], []
    cursor = 0
    for start, failed_block in located:
        raw = failed_block.raw
        key = common_utils.bibtex_header_key(raw.lstrip().split("\n", 1)[0]) or "?"
        report = _diagnose(failed_block, key, source.count("\n", 0, start) + 1)
        maybe_repaired = _repair(failed_block)

        pieces.append(source[cursor:start])
        pieces.append(raw if maybe_repaired is None else maybe_repaired)
        cursor = start + len(raw)

        (unrecovered if maybe_repaired is None else recovered).append(report)

    pieces.append(source[cursor:])
    return "".join(pieces), recovered, unrecovered


def _summarize_by_kind(reports: list[RepairReport]) -> str:
    """Render a per-fault tally, e.g. `1596 duplicate field keys, 3 unbalanced braces`."""
    counts = collections.Counter(report.kind for report in reports)
    return ", ".join(f"{count} {kind}" for kind, count in counts.most_common())


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="""Repair BibTeX records a parser refuses to read: field values whose braces do not balance, and entries naming the same field twice. Writes a new file unless asked to edit in place.""",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-o", "--output-suffix", dest="output_suffix", default="_fixed", type=str, metavar="suf",
                        help="Suffix for naming output files (file.bib -> file_fixed.bib).")
    parser.add_argument("-i", "--in-place", dest="in_place", action="store_true", default=False,
                        help="Write the repair back to the input file instead of to a new one. Your bibliography is yours, so this never happens unless you ask for it.")
    parser.add_argument("-n", "--dry-run", dest="dry_run", action="store_true", default=False,
                        help="Report what would be repaired, and write nothing.")
    parser.add_argument("-l", "--list", dest="list_repairs", action="store_true", default=False,
                        help="Name every record that was repaired, not just how many. A database export can need this a thousand times over, so the list is off by default.")
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

        if opts.list_repairs:
            for report in recovered:
                print(f"{path.name}: repaired {report.describe()}")
        # The unrecovered are always named. They are the ones asking for the user's own time, and there
        # are few of them by nature -- a fault common enough to fill a screen is one worth teaching the
        # tool to repair.
        for report in unrecovered:
            print(f"{path.name}: could not repair {report.describe()}", file=sys.stderr)

        if recovered:
            print(f"{path.name}: repaired {len(recovered)} record(s) — {_summarize_by_kind(recovered)}.")
        if unrecovered:
            print(f"{path.name}: {len(unrecovered)} record(s) still unreadable — {_summarize_by_kind(unrecovered)}.")
        if not recovered:
            if not unrecovered:
                print(f"{path.name}: nothing to repair.")
            continue
        if opts.dry_run:
            print(f"{path.name}: nothing written (--dry-run).")
            continue

        out_path = path if opts.in_place else path.with_name(f"{path.stem}{opts.output_suffix}{path.suffix}")
        out_path.write_text(repaired, encoding="utf-8")
        print(f"{path.name}: written to {out_path}")

    if total_unrecovered:
        # Each remaining fault needs a decision rather than an edit: where a missing brace belonged, or
        # which commas in `Bloggs, PhD, MSc, Joan` separate name parts and which separate credentials.
        # Those are answerable, but not from the file alone, so they are named and left alone.
        print(f"\n{total_unrecovered} record(s) need a look by hand: what would fix them is not "
              f"recoverable from the text.", file=sys.stderr)
    if total_recovered or total_unrecovered:
        print(f"Total: {total_recovered} repaired, {total_unrecovered} left for you.")


if __name__ == "__main__":  # pragma: no cover
    main()

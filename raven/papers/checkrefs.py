"""Check a bibliography against a reference list someone has written from it, and say where they disagree.

The other paper tools turn a `.bib` into a better one. This one changes nothing: it compares a corpus with a
document made from it — a manuscript's reference list, an appendix table, a coauthor's working list — and
reports the records the document leaves out and the entries it has with no record behind them.

    raven-checkrefs corpus.bib manuscript.odt      # corpus_checkrefs.xlsx beside the .bib

A review has to show that the studies it analysed are the ones it cites, and curation by hand drifts: a
record is dropped from the document and not from the corpus, or a title is corrected in one place only. A
failed match is also often a bad field — a placeholder title, keywords where the title should be — so the
check finds data faults in the `.bib` as a side effect.

**How a record is matched to an entry**, strongest evidence first:

  1. **DOI**, from the record's `doi` field, its `url` or its key, against the first DOI in the entry.
  2. **Title**: the record's normalized title appears whole in the normalized entry, and in no other entry
     still free.
  3. **Proposed**: what is left on both sides is paired one-to-one, best `title_coverage` first. A proposal
     is reported with its score and the runner-up's, and counted apart from the matches. The margin over
     the runner-up is what tells a respelled title from a coincidence, where the score alone cannot: a wrong
     candidate can score high on a title made of common words, but it rarely leaves the right one far
     behind.

Deterministic and offline. The document is read with `raven.common.docextract`, so anything it reads works:
`.odt`, `.docx`, `.pdf`, `.txt`, `.md`.
"""

from __future__ import annotations

__all__ = ["DocumentEntry", "split_entries", "entries_of",
           "Reference", "references_of",
           "title_coverage",
           "MATCHED_BY_DOI", "MATCHED_BY_TITLE", "PROPOSED", "NOT_IN_DOCUMENT", "NO_RECORD", "Row",
           "check",
           "AUDIT_COLUMNS", "write_audit",
           "main"]

import argparse
import dataclasses
import difflib
import logging
import pathlib
import re
import sys

from bibtexparser import Library

from ..common import tabular

from .. import __version__

from . import bibtex
from . import deduplicate
from . import utils as papers_utils

logger = logging.getLogger(__name__)

# A list item starts a line with `12.`, `12)` or `[12]`. A run of them counting up from 1 makes the document a
# numbered list; a number out of that run is a wrapped line that happens to start with one (an article number,
# a page range), and is part of the entry above it.
_NUMBERED_PATTERN = re.compile(r"^[ \t]*(?:\[(\d+)\]|(\d+)[.)])[ \t]+", re.MULTILINE)

# Matching runs shorter than this do not count toward `title_coverage`. A long entry contains most short
# letter sequences somewhere, so without a floor a title would be "covered" by unrelated text.
_MIN_RUN = 4


@dataclasses.dataclass(frozen=True)
class DocumentEntry:
    """One entry of the document's reference list. `number` counts from 1 in document order."""
    number: int
    text: str
    maybe_doi: str | None
    maybe_key: str | None  # `deduplicate.normalize_title` of the whole entry


def split_entries(text: str) -> tuple[list[str], str]:
    """The entries of a reference list, and how it was split: `"numbered"` or `"lines"`.

    A numbered list (`1.`, `1)`, `[1]`) is split at the numbers, so an entry wrapped over several lines stays
    one entry, and whatever precedes the first number — a heading — is dropped. The numbers must count up from
    1; a line starting with any other number continues the entry above it. Anything else is taken as one entry
    per non-empty line, which is how a document extractor returns a paragraph each.
    """
    starts = []
    for match in _NUMBERED_PATTERN.finditer(text):
        if int(match.group(1) or match.group(2)) == len(starts) + 1:
            starts.append(match)
    if len(starts) >= 2:
        ends = [match.start() for match in starts[1:]] + [len(text)]
        return [" ".join(text[start.end():end].split()) for start, end in zip(starts, ends)], "numbered"
    return [" ".join(line.split()) for line in text.splitlines() if line.strip()], "lines"


def _first_doi(text: str) -> str | None:
    maybe_doi = bibtex.find_doi(text)
    return papers_utils.normalize_doi(maybe_doi) if maybe_doi is not None else None


def entries_of(texts: list[str]) -> list[DocumentEntry]:
    """`DocumentEntry` for each entry text, numbered in order."""
    return [DocumentEntry(number=i, text=text, maybe_doi=_first_doi(text),
                          maybe_key=deduplicate.normalize_title(text))
            for i, text in enumerate(texts, start=1)]


@dataclasses.dataclass(frozen=True)
class Reference:
    """One record of the `.bib`, reduced to what matching uses."""
    key: str
    title: str
    maybe_doi: str | None
    maybe_title_key: str | None


def references_of(library: Library) -> list[Reference]:
    """A `Reference` per record of `library`. The DOI is looked for as `bibtex.entry_doi` does."""
    out = []
    for entry in library.entries:
        maybe_found = bibtex.entry_doi(entry)
        maybe_doi = papers_utils.normalize_doi(maybe_found[1]) if maybe_found is not None else None
        title = bibtex.readable_text(bibtex.entry_field(entry, "title"))
        out.append(Reference(key=entry.key, title=title, maybe_doi=maybe_doi,
                             maybe_title_key=deduplicate.normalize_title(title)))
    return out


def title_coverage(title_key: str, entry_key: str) -> float:
    """How much of a title an entry contains: the share of `title_key` found in `entry_key`, in order, 0 to 1.

    Both arguments are `deduplicate.normalize_title` keys. 1.0 means the title appears whole. Unlike
    `difflib.SequenceMatcher.ratio()`, which `raven-deduplicate` uses to compare two titles, this is
    one-sided: an entry carries authors, year and venue as well as the title, and those must not count
    against it.
    """
    if not title_key or not entry_key:
        return 0.0
    matcher = difflib.SequenceMatcher(None, title_key, entry_key, autojunk=False)
    return sum(block.size for block in matcher.get_matching_blocks() if block.size >= _MIN_RUN) / len(title_key)


MATCHED_BY_DOI = "matched by DOI"
MATCHED_BY_TITLE = "matched by title"
PROPOSED = "proposed"
NOT_IN_DOCUMENT = "not in the document"
NO_RECORD = "no record"


@dataclasses.dataclass(frozen=True)
class Row:
    """One row of the audit: a record, an entry, or a record paired with an entry.

    `score` and `runner_up` are set for a proposal, and for a record left unmatched (its best candidate,
    whose entry is not named because it was no match). `runner_up` is the best score among the other
    entries, so `score - runner_up` is the margin a person reads a proposal by.
    """
    status: str
    maybe_reference: Reference | None
    maybe_entry: DocumentEntry | None
    score: float | None = None
    runner_up: float | None = None


def _best_two(reference: Reference, entries: list[DocumentEntry],
              scores: dict[tuple[str, int], float], maybe_chosen: int | None = None) -> tuple[float, float]:
    """The best coverage of `reference` over `entries`, and the best among the rest (or over all but the
    chosen entry, when `maybe_chosen` names one)."""
    ranked = sorted((scores[reference.key, entry.number] for entry in entries
                     if entry.number != maybe_chosen), reverse=True)
    if maybe_chosen is not None:
        return scores[reference.key, maybe_chosen], (ranked[0] if ranked else 0.0)
    return (ranked[0] if ranked else 0.0), (ranked[1] if len(ranked) > 1 else 0.0)


def check(references: list[Reference], entries: list[DocumentEntry]) -> list[Row]:
    """Match `references` to `entries`, and return the audit's rows in the order a person should read them:
    proposals by margin (smallest first), then records not in the document, entries with no record, and the
    matches by entry number.
    """
    by_doi: dict[str, DocumentEntry] = {}
    for entry in entries:
        if entry.maybe_doi:
            by_doi.setdefault(entry.maybe_doi, entry)

    matched: list[Row] = []
    used: set[int] = set()
    pending: list[Reference] = []
    for reference in references:
        maybe_entry = by_doi.get(reference.maybe_doi) if reference.maybe_doi else None
        if maybe_entry is None:
            pending.append(reference)
            continue
        matched.append(Row(MATCHED_BY_DOI, reference, maybe_entry))
        used.add(maybe_entry.number)

    # A title matches an entry it appears in whole, but only one that is still free and the only free one it
    # appears in. A short title ("Development") appears whole in many entries; taking the first of them would
    # take another record's entry. Repeating until nothing changes lets longer titles claim their entries
    # first, which leaves the short one a single place to go. A title still ambiguous at the end goes on to
    # the proposals, which pair one-to-one and report the margin.
    while True:
        leftover: list[Reference] = []
        progress = False
        for reference in pending:
            free = [entry for entry in entries
                    if entry.number not in used and reference.maybe_title_key and entry.maybe_key
                    and reference.maybe_title_key in entry.maybe_key]
            if len(free) == 1:
                matched.append(Row(MATCHED_BY_TITLE, reference, free[0]))
                used.add(free[0].number)
                progress = True
            else:
                leftover.append(reference)
        pending = leftover
        if not progress:
            break

    # The proposals. Scored against every entry, not only the free ones: the runner-up has to include an
    # entry another record already took, or a duplicate record would look like a confident proposal.
    # Quadratic in the leftovers by the entries, which is small whenever most records match exactly.
    scores = {(reference.key, entry.number): title_coverage(reference.maybe_title_key or "", entry.maybe_key or "")
              for reference in leftover for entry in entries}
    # A record is proposed only to the entry it covers best in the whole document, and only if that entry is
    # free and strictly ahead of every other. That is the margin applied as a rule, with no number to choose:
    # a record whose best entry another record took is most likely a duplicate, and pairing it with some
    # leftover entry instead would offer two unrelated texts as a match. Two records wanting one entry are
    # settled by score.
    candidates = []
    for reference in leftover:
        maybe_best = max(entries, key=lambda entry: scores[reference.key, entry.number], default=None)
        if maybe_best is None or maybe_best.number in used:
            continue
        score, runner_up = _best_two(reference, entries, scores, maybe_chosen=maybe_best.number)
        if score > runner_up:
            candidates.append((score, runner_up, reference, maybe_best))
    paired_references: set[str] = set()
    proposed: list[Row] = []
    for score, runner_up, reference, entry in sorted(candidates, key=lambda candidate: -candidate[0]):
        if entry.number in used:
            continue
        paired_references.add(reference.key)
        used.add(entry.number)
        proposed.append(Row(PROPOSED, reference, entry, score, runner_up))

    missing = [Row(NOT_IN_DOCUMENT, reference, None, *_best_two(reference, entries, scores))
               for reference in leftover if reference.key not in paired_references]
    orphans = [Row(NO_RECORD, None, entry) for entry in entries if entry.number not in used]
    proposed.sort(key=lambda row: row.score - row.runner_up)
    matched.sort(key=lambda row: row.maybe_entry.number)
    return proposed + missing + orphans + matched


AUDIT_COLUMNS = ("status", "key", "title in the .bib", "entry", "entry in the document",
                 "score", "runner-up", "decision")


def write_audit(path: pathlib.Path, rows: list[Row], *, bib_path: str, document_path: str, split_mode: str) -> None:
    """Write the audit to `path`, as TSV, `.xlsx` or `.ods` by its extension, with lines naming the tool
    version and the inputs: comment lines above the header in a TSV, a *Notes* sheet in a spreadsheet.

    The `decision` column is left empty for a person to mark the proposals.
    """
    def cell(maybe_value: float | None) -> str:
        return "" if maybe_value is None else f"{maybe_value:.3f}"

    tabular.write_table(path,
                        [f"raven-checkrefs {__version__}",
                         f"bibliography: {bib_path}",
                         f"document: {document_path}, split into entries by {split_mode}",
                         "score: the share of the record's title found in the entry; runner-up: the best "
                         "score among the other entries. Read a proposal by the margin between them."],
                        AUDIT_COLUMNS,
                        [(row.status,
                          row.maybe_reference.key if row.maybe_reference else "",
                          row.maybe_reference.title if row.maybe_reference else "",
                          row.maybe_entry.number if row.maybe_entry else "",
                          row.maybe_entry.text if row.maybe_entry else "",
                          cell(row.score), cell(row.runner_up), "")
                         for row in rows])


def _report(rows: list[Row], n_references: int, n_entries: int, split_mode: str) -> None:
    """Print the counts, and each row a person should look at."""
    def count(status: str) -> int:
        return sum(1 for row in rows if row.status == status)

    # Counts behind labels, so that no verb has to agree with them.
    print(f"records in the .bib: {n_references}; entries in the document: {n_entries} ({split_mode})")
    print(f"  matched: {count(MATCHED_BY_DOI) + count(MATCHED_BY_TITLE)} "
          f"(by DOI {count(MATCHED_BY_DOI)}, by title {count(MATCHED_BY_TITLE)})")
    print(f"  proposed, to check by hand: {count(PROPOSED)}")
    print(f"  records not in the document: {count(NOT_IN_DOCUMENT)}")
    print(f"  entries with no record: {count(NO_RECORD)}")
    for row in rows:
        if row.status == PROPOSED:
            print(f"\n  proposed ({row.score:.3f}, runner-up {row.runner_up:.3f}): {row.maybe_reference.key}\n"
                  f"    .bib:     {row.maybe_reference.title}\n    document: {row.maybe_entry.number}. {row.maybe_entry.text}")
        elif row.status == NOT_IN_DOCUMENT:
            print(f"\n  not in the document: {row.maybe_reference.key}\n    {row.maybe_reference.title}")
        elif row.status == NO_RECORD:
            print(f"\n  no record: {row.maybe_entry.number}. {row.maybe_entry.text}")


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Check a BibTeX file against a reference list written from "
                                                 "it — a manuscript's references, an appendix table — and "
                                                 "write an audit of where they disagree. Changes nothing.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-o", "--output", default=None, metavar="PATH",
                        help="Write the audit here; its extension chooses the format. Default: "
                             "<bib name>_checkrefs.<format> beside the .bib.")
    parser.add_argument("--format", default="xlsx", choices=tabular.FORMATS,
                        help="Format of the audit, when --output does not name one: Excel, OpenDocument or TSV.")
    parser.add_argument("--no-audit", action="store_true", default=False,
                        help="Print the report only, and write nothing.")
    parser.add_argument("--log", metavar="PATH", default=None,
                        help="Write the log here instead of to stderr.")
    parser.add_argument("--log-level", default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
                        help="How much to log.")
    parser.add_argument("bib", metavar="references.bib", help="The bibliography.")
    parser.add_argument("document", metavar="document",
                        help="The document holding the reference list: .odt, .docx, .pdf, .txt, .md.")
    opts = parser.parse_args()

    from ..common import logsetup
    logsetup.configure(level=getattr(logging, opts.log_level), logfile=opts.log)
    logging.getLogger("bibtexparser").setLevel(logging.ERROR)  # unreadable records are `raven-fixbib`'s business

    from ..common import docextract  # reads every office format, so not imported for `--help`

    bib_path = pathlib.Path(opts.bib).expanduser().resolve()
    document_path = pathlib.Path(opts.document).expanduser().resolve()
    try:
        library = bibtex.parse_file(bib_path, split_names=False)
        texts, split_mode = split_entries(docextract.extract_text(document_path))
    except (OSError, docextract.DocumentExtractionError) as exc:
        print(f"cannot read the input: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)

    references = references_of(library)
    entries = entries_of(texts)
    rows = check(references, entries)
    _report(rows, len(references), len(entries), split_mode)

    if not opts.no_audit:
        out = (pathlib.Path(opts.output).expanduser().resolve() if opts.output
               else bib_path.parent / f"{bib_path.stem}_checkrefs.{opts.format}")
        write_audit(out, rows, bib_path=str(bib_path), document_path=str(document_path), split_mode=split_mode)
        print(f"\nwrote {out}")


if __name__ == "__main__":  # pragma: no cover
    main()

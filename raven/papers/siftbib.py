"""Sift a bibliography: keep the records that meet a mechanical criterion, and account for the rest.

A literature search hands back records of wildly uneven completeness. Some carry an abstract and some
carry nothing but a title, and a record with nothing to read cannot be screened — not because it is off
topic, which nobody can tell, but because there is no text to form a view about. Removing those is a
separate act from judging relevance, and this tool is the separate act.

    raven-siftbib corpus.bib --require abstract

The criterion is a parameter rather than a policy. `--require` names a field the record must carry,
`--min-chars` a length it must reach, and `--where` a comparison its value must satisfy as a number
(`--where 'year>=2022'`, for the strays a search engine returns from before the search's first year). Each
may be given more than once, and a record is kept when it satisfies all of them. That is what keeps this from being an abstract-specific tool that grows a second
flag the first time a corpus needs `--require year`.

The sifted `.bib` is what you came for, and the audit beside it — a record per line with the
criterion it failed — is what lets you stand behind it: a review has to account for the records it
removed before screening, and a filtered file alone gives neither a reader nor its own author any way to
tell a good cut from a bad one. Two outputs, and the bibliography is the one with lasting value; the
audit is a record of due diligence.

**Deliberately not a relevance filter.** Whether a record is *about* the right subject is a judgement, it
needs a reader or a model, and it belongs in a different tool. Everything here is deterministic: the same
bibliography and the same flags produce the same two files, on any machine, with no network and no model.

Sits alongside its siblings rather than inside them, because the three ask different questions of a
bibliography: `raven-fixbib` asks whether a record can be *read*, `raven-deduplicate` whether two records
name one study, and this one whether a record can be *used*. It shares `deduplicate`'s audit format,
both of them removing records, and `fixbib`'s command line, both of them writing one output per input
where `deduplicate` merges its inputs into one corpus.
"""

from __future__ import annotations

__all__ = ["Criterion", "require_field", "min_chars", "parse_min_chars",
           "numeric_field", "compare", "parse_where",

           "DroppedRecord", "AUDIT_COLUMNS", "sift", "write_audit",

           "main"]

import argparse
import dataclasses
import logging
import math
import operator
import pathlib
import re
import sys
from collections.abc import Callable

import bibtexparser
from bibtexparser.model import Entry

from ..common import tabular

from .. import __version__

from . import bibtex
from . import utils as papers_utils

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Criterion:
    """One mechanical test a record must pass, and the words for what failing it means.

    `describe` names the criterion for the run's report ("has an abstract"); `reason` is what lands in the
    audit against a record that failed it ("no abstract"). Two strings rather than one because they read
    in opposite directions — the report says what was asked of every record, the audit says what this
    record lacked.
    """
    describe: str
    reason: str
    holds: Callable[[Entry], bool]


def require_field(field: str) -> Criterion:
    """A record must carry `field`, non-empty."""
    return Criterion(describe=f"has a non-empty `{field}`",
                     reason=f"no {field}",
                     holds=lambda entry: bool(bibtex.entry_field(entry, field)))


def min_chars(field: str, length: int) -> Criterion:
    """A record's `field` must run to at least `length` characters.

    The case this exists for is a field that is present and useless. Publishers routinely export a
    truncated teaser as the abstract — a couple of sentences ending mid-word — which `require_field`
    accepts and a reader cannot screen on any more than they could screen on nothing.
    """
    return Criterion(describe=f"`{field}` of at least {length} characters",
                     reason=f"{field} shorter than {length} characters",
                     holds=lambda entry: len(bibtex.entry_field(entry, field)) >= length)


def parse_min_chars(spec: str) -> Criterion:
    """Build a `min_chars` criterion from a `FIELD=N` command-line argument.

    Raises `ValueError` on anything that is not that shape, so the CLI can report it as a bad argument
    rather than silently applying a criterion the caller did not mean.
    """
    field, separator, number = spec.partition("=")
    if not separator or not field.strip():
        raise ValueError(f"expected FIELD=N, got {spec!r}")
    try:
        length = int(number)
    except ValueError:
        raise ValueError(f"expected FIELD=N with N a whole number, got {spec!r}") from None
    if length < 0:
        raise ValueError(f"a length cannot be negative: {spec!r}")
    return min_chars(field.strip(), length)


#: The comparisons `parse_where` reads, longest first so that `>=` is not read as `>`; each with its test, and
#: the comparison a failing record satisfies instead, which is what the audit says against it.
_COMPARISONS = {"<=": (operator.le, ">"), ">=": (operator.ge, "<"), "==": (operator.eq, "!="),
                "!=": (operator.ne, "=="), "<": (operator.lt, ">="), ">": (operator.gt, "<=")}
_WHERE = re.compile(r"^\s*([^\s<>=!]+)\s*(" + "|".join(re.escape(op) for op in _COMPARISONS) + r")\s*(\S.*?)\s*$")


def _number(text: str) -> float | None:
    """`text` as a finite number, or `None` when it is not one."""
    try:
        value = float(text)
    except ValueError:
        return None
    # `float` also accepts "nan" and "inf". A year of NaN fails every comparison, which would put it in the
    # audit under the comparison's reason rather than as what it is, not a number.
    return value if math.isfinite(value) else None


def numeric_field(field: str) -> Criterion:
    """A record's `field` must be a number, as a year or a page count is."""
    return Criterion(describe=f"`{field}` is a number",
                     reason=f"{field} not a number",
                     holds=lambda entry: _number(bibtex.entry_field(entry, field)) is not None)


def compare(field: str, op: str, number: str) -> Criterion:
    """A record's `field`, read as a number, must compare to `number` by `op`, one of `<`, `<=`, `>`, `>=`, `==`
    and `!=`.

    `number` is given as text, the way it was written, and the criterion's descriptions repeat it as written.
    A field that is missing or not a number fails; to tell those apart in an audit, put `require_field` and
    `numeric_field` before this one, as `parse_where` does.

    Raises `ValueError` on an unknown `op`, or a `number` that is not one.
    """
    if op not in _COMPARISONS:
        raise ValueError(f"unknown comparison {op!r}; expected one of {', '.join(_COMPARISONS)}")
    threshold = _number(number)
    if threshold is None:
        raise ValueError(f"expected a number to compare `{field}` with, got {number!r}")
    test, failing = _COMPARISONS[op]

    def holds(entry: Entry) -> bool:
        value = _number(bibtex.entry_field(entry, field))
        return value is not None and test(value, threshold)
    return Criterion(describe=f"`{field}` {op} {number}",
                     reason=f"{field} {failing} {number}",
                     holds=holds)


def parse_where(spec: str) -> list[Criterion]:
    """Build the criteria for a `FIELD OP NUMBER` command-line argument, such as `year>=2022`.

    Three criteria, in the order `sift` tries them, so that a removed record is audited under the one that
    applies: the field is present, it is a number, and it compares as asked.

    Raises `ValueError` on anything that is not that shape.
    """
    match = _WHERE.match(spec)
    if match is None:
        raise ValueError(f"expected FIELD OP NUMBER with OP one of {', '.join(_COMPARISONS)}, e.g. year>=2022; "
                         f"got {spec!r}")
    field, op, number = match.groups()
    return [require_field(field), numeric_field(field), compare(field, op, number)]


@dataclasses.dataclass(frozen=True)
class DroppedRecord:
    """One record that did not survive, and the first criterion it failed.

    `link` is where to read the paper (see `raven.papers.utils.paper_url`), and `abstract` is the record's
    own, in full: a reviewer deciding whether a removal was right should not have to open the `.bib`.
    """
    key: str
    reason: str
    title: str
    venue: str
    link: str = ""
    abstract: str = ""


AUDIT_COLUMNS = ("key", "reason", "title", "venue", "link", "abstract")

def sift(library: bibtexparser.Library,
         criteria: list[Criterion]) -> tuple[bibtexparser.Library, list[DroppedRecord]]:
    """Split a library into the records meeting every criterion, and an account of those that do not.

    Returns `(kept, dropped)`. A record is dropped on the *first* criterion it fails, and that is the one
    named in its `DroppedRecord` — a reviewer acting on the audit wants one reason to act on, and knowing
    that a record also lacks a year does not change what to do about its missing abstract.

    Filtering is at the block level, so a preamble, `@string` definitions and comments between records
    survive into the kept library. An empty `criteria` keeps everything, which is the honest answer to
    being asked for no criteria; the CLI refuses that case earlier, where it can say so.
    """
    kept = bibtexparser.Library()
    dropped = []
    for block in library.blocks:
        if not isinstance(block, Entry):
            kept.add(block)
            continue
        failed = next((criterion for criterion in criteria if not criterion.holds(block)), None)
        if failed is None:
            kept.add(block)
        else:
            dropped.append(DroppedRecord(key=block.key,
                                         reason=failed.reason,
                                         title=bibtex.readable_text(bibtex.entry_field(block, "title")),
                                         venue=bibtex.entry_venue(block),
                                         link=papers_utils.paper_url(bibtex.entry_field(block, "doi"),
                                                                     bibtex.entry_field(block, "url")),
                                         abstract=bibtex.readable_text(bibtex.entry_field(block, "abstract"))))
    return kept, dropped


def write_audit(path: pathlib.Path, dropped: list[DroppedRecord],
                sources: list[str], criteria: list[Criterion]) -> None:
    """Write the audit to `path`, as TSV, `.xlsx` or `.ods` by its extension, with lines naming the tool version,
    inputs and criteria: comment lines above the header in a TSV, a *Notes* sheet in a spreadsheet.

    The header is what makes the file citable: a method section says which tool removed these records and
    on what test, and "the script said so" is not a method section.
    """
    tabular.write_table(path,
                        [f"raven-siftbib {__version__}",
                         f"input: {'; '.join(sources)}",
                         f"kept records satisfying: {'; '.join(criterion.describe for criterion in criteria)}",
                         f"records removed: {len(dropped)}"],
                        AUDIT_COLUMNS,
                        [(record.key, record.reason, record.title, record.venue, record.link, record.abstract)
                         for record in dropped])


def _report(kept: int, dropped: list[DroppedRecord], criteria: list[Criterion]) -> None:
    """Print what the run did, grouped by the criterion each record failed."""
    total = kept + len(dropped)
    print(f"{total} records: {kept} kept, {len(dropped)} removed")
    for criterion in criteria:
        count = sum(1 for record in dropped if record.reason == criterion.reason)
        if count:
            share = 100 * count / total if total else 0.0
            print(f"  {count:>6}  ({share:4.1f}%)  {criterion.reason}")


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Sift a BibTeX file: keep the records meeting a "
                                                 "mechanical criterion, and write an audit of the rest. "
                                                 "A record with no abstract cannot be screened by a "
                                                 "reader, which is a different matter from being off "
                                                 "topic; this tool decides only the first.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--require", dest="require", action="append", default=None, metavar="FIELD",
                        help="Keep only records carrying a non-empty FIELD. May be given more than once; "
                             "a record must satisfy all of them.")
    parser.add_argument("--min-chars", dest="min_chars", action="append", default=None, metavar="FIELD=N",
                        help="Keep only records whose FIELD runs to at least N characters. For a field "
                             "that is present and useless — a publisher's truncated teaser abstract, say.")
    parser.add_argument("--where", dest="where", action="append", default=None, metavar="FIELD OP NUMBER",
                        help="Keep only records whose FIELD is a number comparing to NUMBER by OP, one of "
                             "<, <=, >, >=, == and !=, such as 'year>=2022' (quoted, for the shell). A record "
                             "whose FIELD is missing or not a number is removed, and the audit says which.")
    parser.add_argument("-o", "--output-suffix", dest="output_suffix", default="_sifted", type=papers_utils.output_suffix,
                        metavar="suf", help="Suffix for naming output files (file.bib -> file_sifted.bib).")
    parser.add_argument("--audit-suffix", dest="audit_suffix", default="_removed", type=str,
                        metavar="suf", help="Suffix for naming audit files (file.bib -> file_removed.xlsx).")
    parser.add_argument("--audit-format", dest="audit_format", default="xlsx", choices=tabular.FORMATS,
                        help="Format of the audit files: Excel, OpenDocument or TSV.")
    # A directory rather than a path, because this tool sifts each input on its own: there are as many
    # outputs and audits as there are inputs, and no single path can name them. It covers both files
    # rather than the audit alone — redirecting one and leaving the other beside the input is a
    # combination with no use, and would put an asymmetry inside the tool to no purpose.
    parser.add_argument("--out-dir", dest="out_dir", default=None, type=str, metavar="DIR",
                        help="Write the sifted bibliographies and their audits here, instead of beside each input file. Created if it does not exist.")
    parser.add_argument("--no-audit", dest="no_audit", action="store_true", default=False,
                        help="Do not write the audit. A removal cannot be read back out of the sifted file, so this discards the only record of what went.")
    parser.add_argument("-n", "--dry-run", dest="dry_run", action="store_true", default=False,
                        help="Report what would be removed, and write nothing.")
    parser.add_argument("--log", metavar="PATH", default=None,
                        help="Write the log here instead of to stderr.")
    parser.add_argument("--log-level", default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
                        help="How much to log.")
    parser.add_argument(dest="filenames", nargs="+", default=None, type=str, metavar="myreferences.bib",
                        help="BibTeX file(s) to sift.")
    opts = parser.parse_args()

    from ..common import logsetup
    logsetup.configure(level=getattr(logging, opts.log_level), logfile=opts.log)

    # `bibtexparser` logs a warning per record it cannot read. Those are `raven-fixbib`'s business, and
    # this tool's report is about what it removed on purpose; two accounts interleaved is worse than one.
    logging.getLogger("bibtexparser").setLevel(logging.ERROR)

    criteria = [require_field(field) for field in (opts.require or [])]
    try:
        criteria += [parse_min_chars(spec) for spec in (opts.min_chars or [])]
        criteria += [criterion for spec in (opts.where or []) for criterion in parse_where(spec)]
    except ValueError as exc:
        parser.error(str(exc))

    # Refused rather than defaulted. A default criterion would be this tool having an opinion about what
    # a usable record is, which is exactly the judgement it exists to leave with the caller — and a run
    # with no criteria silently copies the file, which looks like it worked.
    if not criteria:
        parser.error("no criteria given; nothing would be removed. "
                     "Say what a record must have, e.g. --require abstract")

    for filename in opts.filenames:
        path = pathlib.Path(filename).expanduser().resolve()
        try:
            library = bibtex.parse_file(str(path), split_names=False)
        except OSError as exc:
            print(f"{path}: cannot read ({type(exc).__name__}: {exc})", file=sys.stderr)
            continue

        kept, dropped = sift(library, criteria)
        print(f"\n{path}")
        _report(len(kept.entries), dropped, criteria)

        if opts.dry_run:
            continue
        directory = pathlib.Path(opts.out_dir).expanduser().resolve() if opts.out_dir else path.parent
        directory.mkdir(parents=True, exist_ok=True)
        out_path = directory / f"{path.stem}{opts.output_suffix}{path.suffix}"
        out_path.write_text(bibtex.write_string(kept), encoding="utf-8")
        print(f"  wrote {out_path}")
        if not opts.no_audit:
            audit_path = directory / f"{path.stem}{opts.audit_suffix}.{opts.audit_format}"
            write_audit(audit_path, dropped, [str(path)], criteria)
            print(f"  wrote {audit_path}")


if __name__ == "__main__":  # pragma: no cover
    main()

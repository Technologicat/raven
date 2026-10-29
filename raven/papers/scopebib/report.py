"""What became of each record, and the files a run leaves: the audits, the worklist, and the filtered `.bib`.

Everything here is computed from answers already saved, so writing the reports asks nothing of a model, and
the definition's rules are applied here rather than when the fields were extracted: a changed rule takes
effect at the next write.

**A record's outcome** (`outcomes`) is where it ended up: removed, sent to a person, kept, or pending. It
is pending while a stage it needs has not answered for it yet — a drop the reviewer has not read, a keep
whose fields have not been extracted — so a partial run removes and keeps only what it has settled.

**The files** (`write_reports`) are named by stage, so that listing the directory shows the pipeline in
order:

    00a-filled.<fmt>          the abstracts filled from Crossref, and the records it could not help
    00b-sifted.<fmt>          what the sift removed, and on which criterion
    00c-preprocessed.bib      the input as filled and sifted: what the judge reads
    01-auto-judged.<fmt>      every judged record: the verdict, the tests, the judge's reason
    02a-auto-reviewed.<fmt>   the judge's drops, and a sample of its keeps as a control, with any case
                              the reviewer made that a record belongs in scope after all
    02b-ruled.<fmt>           the judge's keeps: their fields and quotes, and whether the rules kept, held or removed each
    03a-CHECK-BY-HAND.<fmt>   every record sent to a person, with why
    03b-removed.<fmt>         every record removed, by which stage and why
    03c-kept.bib              the records that survived every stage
    04a-hand-decided.<fmt>    every decision a person made, on 03a-CHECK-BY-HAND and against the run
    04b-final.bib             the kept records, and those a person kept

The 04 files are written by `finish`, from a marked copy of the worklist. `write_reports` refuses to run
over them unless told to set them aside: kept under their own names, they would look current while describing
an older worklist, and they hold a person's decisions, which nothing should remove.

The directory is shared with whatever else the user keeps there, so a write removes only files bearing one
of these exact names that it did not write this time — a previous run's report in another format, or a
stage that did not run. A copy saved under any other name is left alone.
"""

__all__ = ["FILL_AUDIT", "SIFT_AUDIT", "PREPROCESSED_BIB", "JUDGE_AUDIT", "REVIEW_AUDIT", "EXTRACT_AUDIT",
           "CHECK_BY_HAND", "REMOVED_AUDIT", "KEPT_BIB", "HAND_AUDIT", "FINAL_BIB", "OWNED_NAMES",
           "REMOVED", "TO_A_PERSON", "KEPT", "PENDING",
           "Outcome", "outcomes",
           "SiftResult", "select",
           "FinishedRunError", "finished_outputs", "refuse_if_finished", "move_finished_aside", "write_reports",
           "KEEP", "DROP", "FinishError", "finish",
           "PILOT_REPORT", "KEEP_SAMPLE", "title_bounds_note", "write_pilot_report"]

import dataclasses
import datetime
import pathlib
import random
from collections.abc import Iterable, Sequence

import bibtexparser
from bibtexparser.model import Entry

from ...common import tabular
from .. import bibtex
from .. import fillbib
from .. import siftbib
from . import definition as scope_definition
from . import extract as scope_extract
from . import records as scope_records
from . import review as scope_review
from . import rules as scope_rules

# The stems of the tables, written in whichever of `tabular.FORMATS` the caller asks for, and the two `.bib`s.
FILL_AUDIT = "00a-filled"
SIFT_AUDIT = "00b-sifted"
PREPROCESSED_BIB = "00c-preprocessed.bib"
JUDGE_AUDIT = "01-auto-judged"
REVIEW_AUDIT = "02a-auto-reviewed"
EXTRACT_AUDIT = "02b-ruled"
CHECK_BY_HAND = "03a-CHECK-BY-HAND"
REMOVED_AUDIT = "03b-removed"
KEPT_BIB = "03c-kept.bib"
HAND_AUDIT = "04a-hand-decided"
FINAL_BIB = "04b-final.bib"

_TABLES = (FILL_AUDIT, SIFT_AUDIT, JUDGE_AUDIT, REVIEW_AUDIT, EXTRACT_AUDIT, CHECK_BY_HAND, REMOVED_AUDIT, HAND_AUDIT)

#: Every file name `write_reports` can write, and so every name it may remove as stale.
OWNED_NAMES = frozenset([f"{stem}.{fmt}" for stem in _TABLES for fmt in tabular.FORMATS] + [PREPROCESSED_BIB, KEPT_BIB, FINAL_BIB])

# Outcomes.
REMOVED = "removed"
TO_A_PERSON = "to a person"
KEPT = "kept"
PENDING = "pending"

# The columns every table ends with: what a reader needs to check a call without opening the `.bib`.
_RECORD_COLUMNS = ("authors", "year", "title", "venue", "link", "abstract")


@dataclasses.dataclass(frozen=True)
class Outcome:
    """Where one record ended up.

    `action`: `REMOVED`, `TO_A_PERSON`, `KEPT` or `PENDING`.
    `stage`: what decided it — `"judge"`, `"review"` or `"rules"` — or, for a pending record, what it is
             waiting on.
    `decided_by`: the test that fired or the rule that matched, `""` where neither did; the key reports
                  group by, so that each test's calls sit together.
    `reason`: one line for a person, saying why.
    """
    action: str
    stage: str
    decided_by: str
    reason: str


def _tests_true(answer: dict) -> list[str]:
    return [name for name, value in answer["tests"].items() if value is True]


def _judge_reason(answer: dict) -> str:
    true = _tests_true(answer)
    unanswered = [name for name, value in answer["tests"].items() if value is None]
    said = (f"{', '.join(true)} true" if true else "") + (f"; {', '.join(unanswered)} unanswered" if unanswered else "")
    return f"{said.lstrip('; ')}, {answer['confidence']} confidence from the {answer['source']}: {answer['why']}"


def outcomes(records: Iterable[scope_records.Record], *,
             definition: scope_definition.Definition,
             final: dict[str, dict],
             reviews: dict[str, dict],
             extractions: dict[str, dict]) -> dict[str, Outcome]:
    """Each record's outcome, by key.

    `final`: the judge's final answers, as `raven.papers.scopebib.judge.final_answers` gives them.
    `reviews`: the reviewer's answers by key.
    `extractions`: the extractor's answers by key, as `raven.papers.scopebib.extract.normalize` gives them.

    A record the judge has not settled is pending on the judge; a drop the reviewer has not read, on the
    review; a keep with no extraction, on the extraction, unless the definition names no fields, when a keep
    is final.
    """
    reviewed = scope_review.apply_reviews(final, reviews)
    out = {}
    for record in records:
        answer = reviewed.get(record.key)
        if answer is None:
            out[record.key] = Outcome(PENDING, "judge", "", "not yet judged")
            continue
        verdict = answer["verdict"]
        first_test = next(iter(_tests_true(answer)), "")
        if verdict == "drop":
            if record.key not in reviews:
                out[record.key] = Outcome(PENDING, "review", first_test, "dropped by the judge, not yet reviewed")
            else:
                out[record.key] = Outcome(REMOVED, "judge", first_test, _judge_reason(answer))
        elif verdict == "review":
            if answer.get("rescued"):
                out[record.key] = Outcome(TO_A_PERSON, "review", first_test,
                                          f"the judge dropped it ({_judge_reason(answer)}); "
                                          f"the reviewer's case that it belongs in scope: {answer['case']}")
            else:
                out[record.key] = Outcome(TO_A_PERSON, "judge", first_test,
                                          f"the judge was not sure enough to drop it: {_judge_reason(answer)}")
        elif not definition.fields:
            out[record.key] = Outcome(KEPT, "judge", "", "")
        elif record.key not in extractions:
            out[record.key] = Outcome(PENDING, "extract", "", "kept by the judge, fields not yet extracted")
        else:
            decision = scope_rules.decide(extractions[record.key], definition)
            label = decision.label or ""
            if decision.action == "remove":
                out[record.key] = Outcome(REMOVED, "rules", label,
                                          _rule_evidence(label, extractions[record.key], definition))
            elif decision.action == "hold":
                # The rule's label is `decided_by` already, so the reason starts where the label leaves off.
                evidence = _rule_evidence(label, extractions[record.key], definition)
                out[record.key] = Outcome(TO_A_PERSON, "rules", label,
                                          f"{decision.note}; {evidence}" if decision.note else evidence)
            else:
                out[record.key] = Outcome(KEPT, "rules", label, "")
    return out


def _rule_fields(label: str, definition: scope_definition.Definition) -> tuple[str, ...]:
    rule = next((rule for rule in definition.rules if rule.label == label), None)
    return tuple(rule.when) if rule is not None else ()


def _rule_evidence(label: str, extraction: dict, definition: scope_definition.Definition) -> str:
    """The fields a rule matched on, with their quotes: `level = school ("a primary school")`."""
    parts = []
    for name in _rule_fields(label, definition):
        quote = extraction["quotes"].get(name, {}).get("text", "").strip(scope_extract.QUOTE_MARKS + " ")
        parts.append(f"{name} = {extraction['fields'].get(name)}" + (f' ("{quote}")' if quote else ""))
    return "; ".join(parts)


# --------------------------------------------------------------------------------
# Writing

@dataclasses.dataclass(frozen=True)
class SiftResult:
    """What stage 0 did: the records it removed, and on which criteria (see `raven.papers.siftbib`)."""
    dropped: list[siftbib.DroppedRecord]
    criteria: list[siftbib.Criterion]


def _record_cells(record: scope_records.Record) -> tuple:
    return (record.authors, record.year, record.title, record.venue, record.link, record.abstract)


def _reading_order(record: scope_records.Record) -> tuple:
    """A reference list's order, so that a reader can find a record they know."""
    return (record.authors.casefold(), record.year, record.title.casefold())


def _share(count: int, total: int) -> str:
    return f"{count} of {total} ({100 * count / total:.1f}%)" if total else f"{count} of 0"


def _cell(value: bool | None) -> str:
    return {True: "true", False: "false"}.get(value, "no answer")


def _yes_no(value: bool | None) -> str:
    return {True: "yes", False: "no"}.get(value, "no answer")


def select(library: bibtexparser.Library, keys: set[str]) -> bibtexparser.Library:
    """`library` with only the entries in `keys`. Every other block — a preamble, `@string`s — is kept."""
    out = bibtexparser.Library()
    for block in library.blocks:
        if not isinstance(block, Entry) or block.key in keys:
            out.add(block)
    return out


class FinishedRunError(ValueError):
    """Writing the reports again would remove a finished run's outputs, which `set_aside_finished` must ask for."""


def finished_outputs(out_dir: str | pathlib.Path) -> list[pathlib.Path]:
    """The files `finish` wrote that are in `out_dir`, in name order: what writing the reports would remove."""
    out_dir = pathlib.Path(out_dir)
    names = [f"{HAND_AUDIT}.{fmt}" for fmt in tabular.FORMATS] + [FINAL_BIB]
    return sorted(out_dir / name for name in names if (out_dir / name).exists())


def refuse_if_finished(out_dir: str | pathlib.Path) -> None:
    """Raise `FinishedRunError`, naming them, when `out_dir` holds a finished run's outputs."""
    found = finished_outputs(out_dir)
    if found:
        raise FinishedRunError(f"{pathlib.Path(out_dir)} holds a finished run ({', '.join(path.name for path in found)}): "
                               f"writing the reports again would remove those. --set-aside-finished moves them "
                               f"aside under a dated name, and --decisions with the set-aside hand decisions "
                               f"carries them into the new run")


def move_finished_aside(out_dir: str | pathlib.Path, *,
                        maybe_now: datetime.datetime | None = None) -> list[tuple[pathlib.Path, pathlib.Path]]:
    """Move a finished run's outputs aside, to `<name>, set aside <date time><ext>`, and return `(old, new)`.

    The new names are not ones the tool owns, so nothing it writes later removes them. An existing name gets
    ` (2)`, ` (3)` and so on.
    """
    stamp = (maybe_now or datetime.datetime.now()).strftime("%Y-%m-%d %H.%M.%S")
    moved = []
    for path in finished_outputs(out_dir):
        target = path.with_name(f"{path.stem}, set aside {stamp}{path.suffix}")
        n = 2
        while target.exists():
            target = path.with_name(f"{path.stem}, set aside {stamp} ({n}){path.suffix}")
            n += 1
        path.rename(target)
        moved.append((path, target))
    return moved


def write_reports(out_dir: str | pathlib.Path, *,
                  library: bibtexparser.Library,
                  records: Sequence[scope_records.Record],
                  definition: scope_definition.Definition,
                  final: dict[str, dict],
                  reviews: dict[str, dict],
                  control: Sequence[str],
                  extractions: dict[str, dict],
                  provenance: Sequence[str],
                  table_format: str = "tsv",
                  maybe_sift: SiftResult | None = None,
                  maybe_fill: list[fillbib.AuditRow] | None = None,
                  set_aside_finished: bool = False) -> list[pathlib.Path]:
    """Write a run's reports into `out_dir`, remove the stale ones, and return the paths written.

    `library`: the parsed `.bib` the judge read, after the sift; the kept `.bib` is cut from it.
    `records`: its records (`raven.papers.scopebib.records.records_of`), in file order.
    `final`, `reviews`, `extractions`: as `outcomes` takes them.
    `control`: the keys of the reviewer's control group (`raven.papers.scopebib.review.draw_control`).
    `provenance`: lines heading every table — the tool's version, the input, the definition, and the model
                  and instrument of each stage.
    `table_format`: one of `raven.common.tabular.FORMATS`.
    `maybe_sift`: what stage 0 removed, or `None` where no sift ran.
    `maybe_fill`: the audit of stage 0's fill (`raven.papers.fillbib.fill`), or `None` where none ran.
    `set_aside_finished`: move a finished run's outputs aside first (`move_finished_aside`), rather than refuse.

    A stage that has answered nothing writes no table. Raises `FinishedRunError`, writing nothing, when
    `out_dir` holds a finished run's outputs and `set_aside_finished` is false: kept under their own names,
    they would describe an older worklist while looking current.
    """
    if table_format not in tabular.FORMATS:
        raise ValueError(f"write_reports: unknown table format {table_format!r}; expected one of {tabular.FORMATS}")
    if set_aside_finished:
        move_finished_aside(out_dir)
    refuse_if_finished(out_dir)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_key = {record.key: record for record in records}
    result = outcomes(records, definition=definition, final=final, reviews=reviews, extractions=extractions)
    written = []

    def table(stem: str, comments: list[str], columns: Sequence[str], rows: Iterable[Sequence]) -> None:
        path = out_dir / f"{stem}.{table_format}"
        tabular.write_table(path, [*provenance, *comments], columns, rows)
        written.append(path)

    if maybe_fill is not None:
        table(FILL_AUDIT,
              ["abstracts from: Crossref (api.crossref.org); copyright of each abstract rests with its publisher "
               "or authors",
               f"records listed: {len(maybe_fill)}"],
              fillbib.AUDIT_COLUMNS, fillbib.audit_rows(maybe_fill))

    if maybe_sift is not None:
        sift = maybe_sift
        table(SIFT_AUDIT,
              [f"kept records satisfying: {'; '.join(criterion.describe for criterion in sift.criteria)}",
               f"records removed: {len(sift.dropped)}"],
              siftbib.AUDIT_COLUMNS,
              [(record.key, record.reason, record.title, record.venue, record.link, record.abstract)
               for record in sift.dropped])

    if maybe_fill is not None or maybe_sift is not None:
        path = out_dir / PREPROCESSED_BIB
        path.write_text(bibtex.write_string(library), encoding="utf-8")
        written.append(path)

    if final:
        _judge_table(table, final=final, by_key=by_key, definition=definition)
    reviewed_keys = [key for key in final if key in reviews]
    if reviewed_keys:
        _review_table(table, final=final, reviews=reviews, control=control, by_key=by_key)
    if extractions:
        _extract_table(table, extractions=extractions, by_key=by_key, definition=definition, result=result)
    _outcome_tables(table, result=result, by_key=by_key, maybe_sift=maybe_sift)

    kept = {key for key, outcome in result.items() if outcome.action == KEPT}
    path = out_dir / KEPT_BIB
    path.write_text(bibtex.write_string(select(library, kept)), encoding="utf-8")
    written.append(path)

    for name in sorted(OWNED_NAMES - {path.name for path in written}):
        (out_dir / name).unlink(missing_ok=True)
    return written


_VERDICT_ORDER = {"drop": 0, "review": 1, "keep": 2}


def _judge_table(table, *, final: dict[str, dict], by_key: dict[str, scope_records.Record],
                 definition: scope_definition.Definition) -> None:
    tests = [test.name for test in definition.tests]
    counts = {verdict: sum(1 for answer in final.values() if answer["verdict"] == verdict) for verdict in _VERDICT_ORDER}
    keys = sorted(final, key=lambda key: (_VERDICT_ORDER[final[key]["verdict"]],
                                          ",".join(_tests_true(final[key])),
                                          *_reading_order(by_key[key])))
    tally = "; ".join(f"{verdict}: {_share(count, len(final))}" for verdict, count in counts.items())
    table(JUDGE_AUDIT,
          [f"records judged: {len(final)}; {tally}",
           "verdict is the judge's own; a drop the reviewer made a case for goes to a person, which is in "
           f"{CHECK_BY_HAND}"],
          ("key", "verdict", "confidence", "pass", "tests true", *tests, "why", "truncated", *_RECORD_COLUMNS),
          [(key, final[key]["verdict"], final[key]["confidence"], final[key]["source"],
            ", ".join(_tests_true(final[key])),
            *(_cell(final[key]["tests"].get(name)) for name in tests),
            final[key]["why"],
            {True: "yes", False: "no"}.get(final[key].get("truncated"), ""),
            *_record_cells(by_key[key]))
           for key in keys])


def _review_table(table, *, final: dict[str, dict], reviews: dict[str, dict], control: Sequence[str],
                  by_key: dict[str, scope_records.Record]) -> None:
    all_drops = [key for key, answer in final.items() if answer["verdict"] == "drop"]
    all_controls = [key for key in control if key in final]
    drops = [key for key in all_drops if key in reviews]
    controls = [key for key in all_controls if key in reviews]

    def cases(keys: list[str]) -> int:
        return sum(1 for key in keys if reviews[key]["belongs"] is True)

    def unread(n_all: int, n_read: int) -> str:
        return f" ({n_all - n_read} more not yet read)" if n_all > n_read else ""

    rows = []
    for group, keys in (("drop", drops), ("control", controls)):
        for key in sorted(keys, key=lambda key: (reviews[key]["belongs"] is not True,
                                                 ",".join(_tests_true(final[key])),
                                                 *_reading_order(by_key[key]))):
            answer = final[key]
            rows.append((group, key, _yes_no(reviews[key]["belongs"]), reviews[key]["case"],
                         answer["verdict"], answer["confidence"], ", ".join(_tests_true(answer)), answer["why"],
                         *_record_cells(by_key[key])))
    table(REVIEW_AUDIT,
          [f"the reviewer found a case for {_share(cases(drops), len(drops))} drops"
           f"{unread(len(all_drops), len(drops))}, "
           f"and for {_share(cases(controls), len(controls))} control records"
           f"{unread(len(all_controls), len(controls))}",
           "read that comparison first: a reviewer that finds a case as often in the control as among the "
           "drops is not telling them apart",
           "the control is records the judge kept, drawn at the drops' mix of confidence and shown to the "
           "reviewer unlabelled; it is labelled only here, and the reviewer's answer about it changes nothing"],
          ("group", "key", "reviewer found a case", "reviewer's case that it belongs", "judge verdict", "judge confidence", "tests true", "judge why",
           *_RECORD_COLUMNS),
          rows)


def _extract_table(table, *, extractions: dict[str, dict], by_key: dict[str, scope_records.Record],
                   definition: scope_definition.Definition, result: dict[str, Outcome]) -> None:
    field_columns = []
    for field in definition.fields:
        field_columns.append(field.name)
        if field.quote:
            field_columns += [f"{field.name} quote", f"{field.name} quote found"]

    def field_cells(extraction: dict) -> list:
        cells = []
        for field in definition.fields:
            cells.append(extraction["fields"].get(field.name, ""))
            if field.quote:
                quote = extraction["quotes"].get(field.name, {})
                cells += [quote.get("text", ""), {True: "yes", False: "no"}.get(quote.get("found"), "")]
        return cells

    # Only the judge's keeps: the ledger can hold extractions of records a later judge sent elsewhere.
    keys = [key for key in extractions if key in by_key and result[key].stage == "rules"]
    decisions = {key: scope_rules.decide(extractions[key], definition) for key in keys}
    order = {"remove": 0, "hold": 1, scope_rules.KEEP: 2}
    keys.sort(key=lambda key: (order[decisions[key].action], decisions[key].label or "", *_reading_order(by_key[key])))
    counts = {action: sum(1 for key in keys if decisions[key].action == action) for action in order}
    tally = "; ".join(f"{action}: {_share(count, len(keys))}" for action, count in counts.items())
    table(EXTRACT_AUDIT,
          ["what each record the judge kept says, field by field, as a model read it from the title and abstract; "
           "the definition's rules act on these fields, removing a record, holding it for a person, or keeping it",
           "a quote is the record's own words for a value, and 'quote found' says whether those words are "
           "actually in the record: a removal resting on a quote that is not found is held instead",
           f"records extracted: {len(keys)}; {tally}",
           "the rules are the definition's as it stands, applied when this was written; a value the model gave "
           "outside a field's vocabulary reads as not_stated, and is shown under 'unrecognized'"],
          ("key", "action", "rule", "note", *field_columns, "unrecognized", *_RECORD_COLUMNS),
          [(key, decisions[key].action, decisions[key].label or "", decisions[key].note,
            *field_cells(extractions[key]),
            "; ".join(f"{name}: {value}" for name, value in extractions[key]["raw"].items()),
            *_record_cells(by_key[key]))
           for key in keys])


_STAGE_ORDER = {"sift": 0, "judge": 1, "review": 2, "rules": 3}


def _outcome_tables(table, *, result: dict[str, Outcome], by_key: dict[str, scope_records.Record],
                    maybe_sift: SiftResult | None) -> None:
    def order(key: str) -> tuple:
        outcome = result[key]
        return (_STAGE_ORDER[outcome.stage], outcome.decided_by, *_reading_order(by_key[key]))

    counts = {action: sum(1 for outcome in result.values() if outcome.action == action)
              for action in (REMOVED, TO_A_PERSON, KEPT, PENDING)}
    summary = f"of {len(result)} records: " + "; ".join(f"{action}: {count}" for action, count in counts.items())

    to_a_person = sorted((key for key, outcome in result.items() if outcome.action == TO_A_PERSON), key=order)
    table(CHECK_BY_HAND,
          [summary,
           f"each of these {len(to_a_person)} needs a person's decision; the 'decision' column is for that",
           f"this file is rewritten on every run: save a marked copy under a name of your own, such as "
           f"{CHECK_BY_HAND}-marked"],
          ("key", "sent by", "decided by", "why", "decision", *_RECORD_COLUMNS),
          [(key, result[key].stage, result[key].decided_by, result[key].reason, "", *_record_cells(by_key[key]))
           for key in to_a_person])

    removed = sorted((key for key, outcome in result.items() if outcome.action == REMOVED), key=order)
    rows = []
    sift_count = 0
    if maybe_sift is not None:
        sift_count = len(maybe_sift.dropped)
        rows += [(record.key, "sift", record.reason, record.reason, "", "", record.title, record.venue, record.link,
                  record.abstract)
                 for record in maybe_sift.dropped]
    rows += [(key, result[key].stage, result[key].decided_by, result[key].reason, *_record_cells(by_key[key]))
             for key in removed]
    table(REMOVED_AUDIT,
          [summary,
           f"records removed: {sift_count + len(removed)}" + (f", of which {sift_count} by the sift" if maybe_sift else "")],
          ("key", "removed by", "decided by", "why", *_RECORD_COLUMNS),
          rows)


# --------------------------------------------------------------------------------
# A person's decisions

#: The decisions a person writes in the worklist's `decision` column, in any case. A blank cell drops too.
KEEP = "keep"
DROP = "drop"


class FinishError(ValueError):
    """A marked worklist `finish` cannot act on, with every row or key at fault named."""


def _trailing_blank_warning(decisions: list[str], row_numbers: list[int]) -> str | None:
    """A warning when the blanks at the end of a sheet marking only its keeps look like marking that stopped
    partway.

    The tail is suspect when it is the longest run of blanks in the sheet, since stopping partway makes it
    so; how long that is depends on the person's own marking, so no fixed number is assumed.
    """
    runs, run = [], 0
    for decision in decisions:
        run = run + 1 if decision == "" else 0
        runs.append(run)
    tail = runs[-1] if runs else 0
    if not tail:
        return None
    if KEEP not in decisions:
        return (f"no row is marked '{KEEP}', so all {len(decisions)} records will be dropped; "
                "if that is what you meant, this is normal")
    if tail <= max(runs[:-tail], default=0):
        return None
    return (f"the last {tail} rows, from row {row_numbers[-tail]} on, have no decision, so they "
            f"will be dropped; that is the longest blank run in the sheet, which is how marking that stopped "
            f"partway looks. If you meant to drop them, this is normal")


def finish(out_dir: str | pathlib.Path, *,
           sheet: str | pathlib.Path,
           library: bibtexparser.Library,
           records: Sequence[scope_records.Record],
           definition: scope_definition.Definition,
           final: dict[str, dict],
           reviews: dict[str, dict],
           extractions: dict[str, dict],
           provenance: Sequence[str],
           table_format: str = "tsv",
           require_every_decision: bool = False) -> tuple[list[pathlib.Path], list[str]]:
    """Apply a person's decisions on a marked copy of the worklist, and write `FINAL_BIB` and `HAND_AUDIT`.

    `sheet`: the marked worklist, in any of `raven.common.tabular.FORMATS`. Only its `key` and `decision`
             columns are required. The `decision` column marks either the keeps alone, `KEEP` with a blank
             dropping, or every row, `KEEP` or `DROP`; case and surrounding space do not matter. A `note`
             column, where the person added one, is carried into the audit.
             A row may also name a record the run kept or removed outright: marked `KEEP` or `DROP`, it
             overrides the run, and a blank there is no decision.
    `library`, `records`, `definition`, `final`, `reviews`, `extractions`, `provenance`, `table_format`:
             as `write_reports` takes them; the worklist is worked out from them afresh.
    `require_every_decision`: refuse a blank decision rather than reading it as a drop, so that every row
                              must be marked.

    Returns `(written, warnings)`: the paths written, and what the person should be told — every override
    applied, a row naming a record the run does not hold, which is ignored, and a run of blank rows at the end
    that looks like marking that stopped partway.

    Raises `FinishError`, writing nothing, when the run is not finished (a record is pending), when the sheet
    has no `key` or `decision` column, when a decision is neither `KEEP`, `DROP` nor blank, when a record of
    the worklist is missing from the sheet, or when a decision is blank in a sheet that marks a `DROP`, or
    with `require_every_decision`.
    """
    if table_format not in tabular.FORMATS:
        raise ValueError(f"finish: unknown table format {table_format!r}; expected one of {tabular.FORMATS}")
    result = outcomes(records, definition=definition, final=final, reviews=reviews, extractions=extractions)
    pending = sorted(key for key, outcome in result.items() if outcome.action == PENDING)
    if pending:
        raise FinishError(f"the run is not finished: {len(pending)} records are still waiting on a stage, "
                          f"e.g. {', '.join(pending[:5])}")
    rows = tabular.read_table(sheet)
    if rows and not {"key", "decision"} <= rows[0].keys():
        raise FinishError(f"'{sheet}' has no 'key' and 'decision' columns; is it a copy of {CHECK_BY_HAND}?")

    # A spreadsheet numbers the header row 1, so the first record is row 2. Rows are numbered by their place
    # in the sheet, which interleaves the worklist with any overrides a person added.
    numbered = list(enumerate(rows, start=2))
    worklist = {key for key, outcome in result.items() if outcome.action == TO_A_PERSON}
    warnings, problems, decided = [], [], {}
    unknown = [row["key"] for _, row in numbered if row["key"] not in result]
    if unknown:
        warnings.append(f"{len(unknown)} rows name records this run does not hold, and are ignored: "
                        f"{', '.join(unknown[:5])}" + (", ..." if len(unknown) > 5 else ""))
    # A row on a record the run settled outright overrides the run when it is marked. A blank there is no
    # decision at all: "a blank drops" is the worklist's convention, and must never remove a record the run kept.
    overrides = {}
    for number, row in numbered:
        key = row["key"]
        if key not in result or key in worklist:
            continue
        decision = row["decision"].strip().lower()
        if decision not in (KEEP, DROP, ""):
            problems.append(f"row {number} ({key}): '{row['decision']}' is neither '{KEEP}', '{DROP}' nor blank")
        elif decision and (decision == KEEP) != (result[key].action == KEPT):
            overrides[key] = row
    worked = [(number, row) for number, row in numbered if row["key"] in worklist]
    decisions = [row["decision"].strip().lower() for _, row in worked]
    for (number, row), decision in zip(worked, decisions):
        if decision not in (KEEP, DROP, ""):
            problems.append(f"row {number} ({row['key']}): '{row['decision']}' is neither '{KEEP}', "
                            f"'{DROP}' nor blank")
        decided[row["key"]] = row
    missing = sorted(worklist - decided.keys())
    if missing:
        problems.append(f"{len(missing)} records sent to a person are not in the sheet, so it is not this run's "
                        f"worklist or rows were deleted: {', '.join(missing[:5])}" + (", ..." if len(missing) > 5 else ""))
    # A sheet marks either its keeps alone, a blank dropping, or every row. One `drop` says the person is
    # marking every row, and a blank beside it is then a row not yet decided.
    blank = [str(number) for (number, _), decision in zip(worked, decisions) if decision == ""]
    listed = f"rows {', '.join(blank[:10])}" + (", ..." if len(blank) > 10 else "")
    if blank and DROP in decisions:
        problems.append(f"the sheet marks some rows '{DROP}', so a blank is a row not yet decided, and "
                        f"{len(blank)} are blank ({listed}). Either decide those, or remove every '{DROP}' so "
                        f"that a blank means drop")
    elif blank and require_every_decision:
        problems.append(f"{len(blank)} rows have no decision ({listed})")
    if problems:
        raise FinishError("; ".join(problems))
    if maybe_warning := _trailing_blank_warning(decisions, [number for number, _ in worked]):
        warnings.append(maybe_warning)
    for key, row in overrides.items():
        warnings.append(f"overriding the run: {key}, {result[key].action} by the {result[key].stage} stage, "
                        f"now {row['decision'].strip().lower()} by hand")

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_key = {record.key: record for record in records}
    hand_kept = {key for key, row in decided.items() if row["decision"].strip().lower() == KEEP}
    restored = {key for key, row in overrides.items() if row["decision"].strip().lower() == KEEP}
    unkept = overrides.keys() - restored
    rows_of = {**decided, **overrides}
    keys = sorted(rows_of, key=lambda key: (key not in hand_kept | restored, key in overrides,
                                            _STAGE_ORDER[result[key].stage], result[key].decided_by,
                                            *_reading_order(by_key[key])))
    path = out_dir / f"{HAND_AUDIT}.{table_format}"
    tabular.write_table(path,
                        [*provenance,
                         f"decisions read from: {sheet}",
                         f"of {len(decided)} records sent to a person: kept {len(hand_kept)}, "
                         f"dropped {len(decided) - len(hand_kept)}",
                         f"the run overridden by hand: {len(restored)} removals restored, "
                         f"{len(unkept)} keeps dropped"],
                        ("key", "decision", "the run had", "sent by", "decided by", "why", "note", *_RECORD_COLUMNS),
                        [(key, KEEP if key in hand_kept | restored else DROP, result[key].action, result[key].stage,
                          result[key].decided_by, result[key].reason, rows_of[key].get("note", ""),
                          *_record_cells(by_key[key]))
                         for key in keys])
    written = [path]
    kept = ({key for key, outcome in result.items() if outcome.action == KEPT} - unkept) | hand_kept | restored
    path = out_dir / FINAL_BIB
    path.write_text(bibtex.write_string(select(library, kept)), encoding="utf-8")
    written.append(path)
    for fmt in tabular.FORMATS:
        if fmt != table_format:
            (out_dir / f"{HAND_AUDIT}.{fmt}").unlink(missing_ok=True)
    return written, warnings


# --------------------------------------------------------------------------------
# The pilot report

#: The pilot's one report, in whichever table format: everything a calibration run hands a person to read.
PILOT_REPORT = "PILOT"

#: How many of the judge's keeps the pilot report shows with their fields, drawn by the run's seed.
KEEP_SAMPLE = 20

_SECTIONS = ("1 drops", "2 sent to review", "3 keeps, a sample", "4 title-only")


def title_bounds_note(records: Sequence[scope_records.Record], measures: dict[str, scope_records.TitleMeasure]) -> str:
    """How many titles the title bounds flag as too thin to judge from, against what they flagged in calibration."""
    flagged = sum(1 for record in records if measures[record.key].uninformative)
    low, high = scope_records.CALIBRATED_FLAG_SHARE
    return (f"title bounds (at least {scope_records.MIN_CONTENT_WORDS} content words, density at least "
            f"{scope_records.MIN_CONTENT_DENSITY:.2f}) flag {_share(flagged, len(records))} titles as too thin to "
            f"judge from, against {100 * low:.1f}–{100 * high:.1f}% across the calibration corpora")


def write_pilot_report(out_dir: str | pathlib.Path, *,
                       records: Sequence[scope_records.Record],
                       measures: dict[str, scope_records.TitleMeasure],
                       definition: scope_definition.Definition,
                       final: dict[str, dict],
                       reviews: dict[str, dict],
                       extractions: dict[str, dict],
                       provenance: Sequence[str],
                       seed: int,
                       table_format: str = "tsv") -> pathlib.Path:
    """Write the pilot report, `PILOT_REPORT`, into `out_dir`, and return its path.

    One table, a `section` column first, in the order a person should read it — what a mistake costs:

      1. **The judge's drops**, each with the test that fired, its reason, and the reviewer's case that it belongs
         in scope, where it made one. A false drop loses a record silently, so these are read first, all of them.
      2. **The records the judge sent to review.** Mostly obvious drops means the tests are too timid;
         mostly obvious keeps, too eager.
      3. **A sample of the keeps**, `KEEP_SAMPLE` of them drawn by `seed`, with their fields and what the
         rules did. Anything here that plainly should have gone is a test that is missing.
      4. **The title-only records**, however they ended, being the inputs most likely to break a rubric. Where
         there are none, one row says so rather than the section vanishing.

    Within a section, rows are grouped by what decided them — the test, or the rule — then in reference-list
    order. The table's notes head it with `title_bounds_note`.

    `records`, `measures`: the pilot's records and their title measures.
    `final`, `reviews`, `extractions`: as `write_reports` takes them.
    """
    if table_format not in tabular.FORMATS:
        raise ValueError(f"write_pilot_report: unknown table format {table_format!r}; expected one of "
                         f"{tabular.FORMATS}")
    by_key = {record.key: record for record in records}
    field_columns = [field.name for field in definition.fields]

    def row(section: str, key: str) -> tuple:
        record, answer = by_key[key], final.get(key)
        maybe_review = reviews.get(key)
        maybe_extraction = extractions.get(key)
        decision = scope_rules.decide(maybe_extraction, definition) if maybe_extraction else None
        decided_by = (", ".join(_tests_true(answer)) if answer and answer["verdict"] != "keep"
                      else (decision.label or "") if decision else "")
        return (section, key, decided_by,
                answer["verdict"] if answer else "not judged", answer["confidence"] if answer else "",
                answer["source"] if answer else "", answer["why"] if answer else "",
                {True: "case found", False: "no case"}.get(maybe_review["belongs"], "") if maybe_review else "",
                maybe_review["case"] if maybe_review else "",
                decision.action if decision else "", decision.note if decision else "",
                *(maybe_extraction["fields"].get(name, "") if maybe_extraction else "" for name in field_columns),
                "yes" if measures[key].uninformative else "no",
                *_record_cells(record))

    def ordered(keys) -> list[str]:
        return sorted(keys, key=lambda key: (row("", key)[2], *_reading_order(by_key[key])))

    drops = [key for key, answer in final.items() if answer["verdict"] == "drop"]
    sent = [key for key, answer in final.items() if answer["verdict"] == "review"]
    keeps = sorted(key for key, answer in final.items() if answer["verdict"] == "keep")
    sampled = random.Random(seed).sample(keeps, min(KEEP_SAMPLE, len(keeps)))
    title_only = [record.key for record in records if not record.abstract]
    rows = [row(_SECTIONS[0], key) for key in ordered(drops)]
    rows += [row(_SECTIONS[1], key) for key in ordered(sent)]
    rows += [row(_SECTIONS[2], key) for key in ordered(sampled)]
    rows += [row(_SECTIONS[3], key) for key in ordered(title_only)]
    if not title_only:
        rows.append((_SECTIONS[3], "", "", "", "", "", "none in this pilot: every record has an abstract",
                     *[""] * (5 + len(field_columns) + len(_RECORD_COLUMNS))))
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{PILOT_REPORT}.{table_format}"
    tabular.write_table(path,
                        [*provenance,
                         title_bounds_note(records, measures),
                         f"{len(records)} records: {len(drops)} dropped by the judge, {len(sent)} sent to review, "
                         f"{len(keeps)} kept, of which {len(sampled)} shown; {len(title_only)} title-only",
                         "read the sections in order: a false drop costs most, as it loses a record silently"],
                        ("section", "key", "decided by", "verdict", "confidence", "pass", "why",
                         "reviewer", "reviewer's case that it belongs", "rules", "rules' note", *field_columns, "thin title",
                         *_RECORD_COLUMNS),
                        rows)
    for fmt in tabular.FORMATS:
        if fmt != table_format:
            (out_dir / f"{PILOT_REPORT}.{fmt}").unlink(missing_ok=True)
    return path

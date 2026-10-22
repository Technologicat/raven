"""What became of each record, and the files a run leaves: the audits, the worklist, and the filtered `.bib`.

Everything here is computed from answers already saved, so writing the reports asks nothing of a model, and
the definition's rules are applied here rather than when the fields were extracted: a changed rule takes
effect at the next write.

**A record's outcome** (`outcomes`) is where it ended up: removed, sent to a person, kept, or pending. It
is pending while a stage it needs has not answered for it yet — a drop the reviewer has not read, a keep
whose fields have not been extracted — so a partial run removes and keeps only what it has settled.

**The files** (`write_reports`) are named by stage, so that listing the directory shows the pipeline in
order:

    00-sift-audit.<fmt>      what the sift removed, and on which criterion
    00-sifted.bib            what it kept: the judge's input
    01-judge-audit.<fmt>     every judged record: the verdict, the tests, the judge's reason
    02-review-audit.<fmt>    the drops and the control, with the reviewer's case
    03-extract-audit.<fmt>   the keeps: their fields, their quotes, the rule that acted
    04-CHECK-BY-HAND.<fmt>   every record sent to a person, with why
    04-removed-audit.<fmt>   every record removed, by which stage and why
    04-kept.bib              the records that survived every stage

The directory is shared with whatever else the user keeps there, so a write removes only files bearing one
of these exact names that it did not write this time — a previous run's report in another format, or a
stage that did not run. A copy saved under any other name is left alone.
"""

__all__ = ["SIFT_AUDIT", "SIFTED_BIB", "JUDGE_AUDIT", "REVIEW_AUDIT", "EXTRACT_AUDIT",
           "CHECK_BY_HAND", "REMOVED_AUDIT", "KEPT_BIB", "OWNED_NAMES",
           "REMOVED", "TO_A_PERSON", "KEPT", "PENDING",
           "Outcome", "outcomes",
           "SiftResult", "write_reports"]

import dataclasses
import pathlib
from collections.abc import Iterable, Sequence

import bibtexparser
from bibtexparser.model import Entry

from ...common import tabular
from .. import bibtex
from .. import siftbib
from . import definition as scope_definition
from . import extract as scope_extract
from . import records as scope_records
from . import review as scope_review
from . import rules as scope_rules

# The stems of the tables, written in whichever of `tabular.FORMATS` the caller asks for, and the two `.bib`s.
SIFT_AUDIT = "00-sift-audit"
SIFTED_BIB = "00-sifted.bib"
JUDGE_AUDIT = "01-judge-audit"
REVIEW_AUDIT = "02-review-audit"
EXTRACT_AUDIT = "03-extract-audit"
CHECK_BY_HAND = "04-CHECK-BY-HAND"
REMOVED_AUDIT = "04-removed-audit"
KEPT_BIB = "04-kept.bib"

_TABLES = (SIFT_AUDIT, JUDGE_AUDIT, REVIEW_AUDIT, EXTRACT_AUDIT, CHECK_BY_HAND, REMOVED_AUDIT)

#: Every file name `write_reports` can write, and so every name it may remove as stale.
OWNED_NAMES = frozenset([f"{stem}.{fmt}" for stem in _TABLES for fmt in tabular.FORMATS] + [SIFTED_BIB, KEPT_BIB])

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
                                          f"the reviewer's case for it: {answer['case']}")
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


def _filtered(library: bibtexparser.Library, keys: set[str]) -> bibtexparser.Library:
    """`library` with only the entries in `keys`. Every other block — a preamble, `@string`s — is kept."""
    out = bibtexparser.Library()
    for block in library.blocks:
        if not isinstance(block, Entry) or block.key in keys:
            out.add(block)
    return out


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
                  maybe_sift: SiftResult | None = None) -> list[pathlib.Path]:
    """Write a run's reports into `out_dir`, remove the stale ones, and return the paths written.

    `library`: the parsed `.bib` the judge read, after the sift; the kept `.bib` is cut from it.
    `records`: its records (`raven.papers.scopebib.records.records_of`), in file order.
    `final`, `reviews`, `extractions`: as `outcomes` takes them.
    `control`: the keys of the reviewer's control group (`raven.papers.scopebib.review.draw_control`).
    `provenance`: lines heading every table — the tool's version, the input, the definition, and the model
                  and instrument of each stage.
    `table_format`: one of `raven.common.tabular.FORMATS`.
    `maybe_sift`: what stage 0 removed, or `None` where no sift ran.

    A stage that has answered nothing writes no table.
    """
    if table_format not in tabular.FORMATS:
        raise ValueError(f"write_reports: unknown table format {table_format!r}; expected one of {tabular.FORMATS}")
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_key = {record.key: record for record in records}
    result = outcomes(records, definition=definition, final=final, reviews=reviews, extractions=extractions)
    written = []

    def table(stem: str, comments: list[str], columns: Sequence[str], rows: Iterable[Sequence]) -> None:
        path = out_dir / f"{stem}.{table_format}"
        tabular.write_table(path, [*provenance, *comments], columns, rows)
        written.append(path)

    if maybe_sift is not None:
        sift = maybe_sift
        table(SIFT_AUDIT,
              [f"kept records satisfying: {'; '.join(criterion.describe for criterion in sift.criteria)}",
               f"records removed: {len(sift.dropped)}"],
              siftbib.AUDIT_COLUMNS,
              [(record.key, record.reason, record.title, record.venue, record.link, record.abstract)
               for record in sift.dropped])
        path = out_dir / SIFTED_BIB
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
    path.write_text(bibtex.write_string(_filtered(library, kept)), encoding="utf-8")
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
    drops = [key for key, answer in final.items() if answer["verdict"] == "drop" and key in reviews]
    controls = [key for key in control if key in reviews and key in final]

    def cases(keys: list[str]) -> int:
        return sum(1 for key in keys if reviews[key]["belongs"] is True)

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
          [f"the reviewer found a case for {_share(cases(drops), len(drops))} drops, "
           f"and for {_share(cases(controls), len(controls))} control records",
           "read that comparison first: a reviewer that finds a case as often in the control as among the "
           "drops is not telling them apart",
           "the control is records the judge kept, drawn at the drops' mix of confidence and shown to the "
           "reviewer unlabelled; it is labelled only here, and the reviewer's answer about it changes nothing"],
          ("group", "key", "reviewer found a case", "reviewer's case", "judge verdict", "judge confidence", "tests true", "judge why",
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

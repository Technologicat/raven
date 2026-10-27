"""The scope judge: which records are positively off topic, by the tests a definition names.

Two passes, each a `raven.librarian.sweep` with a ledger of its own:

  - **Titles** (`ask_titles`), in batches: every record, from its title and venue alone.
  - **Abstracts** (`ask_abstracts`), for the records the first pass cannot settle (`needs_escalation`):
    low confidence, a title too thin to judge from, or any test answered true at all.

A record's final answer is the abstract pass's where it had one, and the title pass's otherwise
(`final_answers`). Its verdict (`verdict_of`) is a drop for a test answered true at high confidence, or at
medium after reading the abstract; any other true, or a test left unanswered, sends it to review, and a
record with no test true is kept.

The prompt is the tool's frame (`FRAME`, or the definition's own), the definition's tests, and its hints.
The instrument (`instrument_material`) covers all of it, so a change to any of them asks again; the two
passes share one fingerprint and keep one ledger each (`ledgers`).
"""

__all__ = ["FRAME", "ABSTRACT_CHARS", "CONFIDENCES",
           "rubric", "instrument_material", "ledgers",
           "title_item", "abstract_item", "title_digest", "abstract_digest",
           "ask_titles", "ask_abstracts",
           "normalize", "temper_for_truncation", "verdict_of", "needs_escalation",
           "todo_titles", "todo_abstracts", "final_answers"]

import json
import logging
import pathlib
from collections.abc import Iterable

from unpythonic.env import env

from ...common import netutil
from ...librarian import agent, sweep
from . import definition as scope_definition
from . import records as scope_records

logger = logging.getLogger(__name__)

#: The tool's own framing of the judge's task, for any corpus; a definition's `frame` replaces it.
FRAME = """\
These records came out of a literature search for: {scope}. Every one of them already matched that search. \
Your job is to find the ones that matched it by accident - the ones that are demonstrably about something else.

So do NOT ask whether each record is on topic. Ask whether there is positive evidence that it is off topic, \
and answer each of the tests below separately: a record can fail any one of them and pass the others.

Unless a test says otherwise, silence is not evidence: if a record does not say what a test asks about, \
answer false for that test and lower your confidence.

A short title is mostly absence, whatever the work is, and absence you cannot interpret is not evidence. An \
identifier is not a description: given a bare proper noun, an acronym, a project name, a journal section, a \
heading, or an identifier such as an arXiv number or a DOI, you do NOT know what the work is about - answer \
false to every test, with low confidence. Do not claim to recognize what an unfamiliar name refers to; a \
guess dressed as recognition is worse than admitting the title says nothing.

Some records carry where they were published, shown as "[published in: ...]". Use it: it is often the \
strongest evidence there is when the title is short, since a venue named for a field tells you the field. \
But it says where the work appeared, not what the work is: a book review in a journal of the right field is \
still a book review.

A phrase can match in a sense the search did not mean. Judge the work, not the phrase that matched.

Report your confidence honestly. A generic title gives no evidence either way: answer false to every test, \
with low confidence. Do not guess confidently."""

_TITLES = """\
You are screening a literature-search result for records that are not about the topic it was searching for.

{rubric}

For each numbered item below you are given a title, and where the record names one, where it was published. \
For each item, answer:
  "i"           the item's number, copied exactly
{test_lines}
  "confidence"  "high", "medium" or "low" - how sure you are, given only this
  "why"         at most twelve words, why you answered that way

Answer with a JSON array of objects and nothing else. One object per item, in order, no commentary, no \
markdown fences.

Items:
{items}
"""

_ABSTRACTS = """\
You are screening a literature-search result for records that are not about the topic it was searching for.

{rubric}

Each record below is given with its abstract as well as its title. They arrive here for two different \
reasons: some because the title said too little to judge from, others because a first pass found a test \
true, and that is re-checked against the fuller text before it stands. So do not assume the title is \
uninformative - read both, and let whichever says more decide. In particular, an abstract that says LESS \
than the title does not make the record unjudgeable: judge it from the title.
Judge each record only on its own text; they are unrelated to one another and their order means nothing.

An abstract marked TRUNCATED is a publisher's preview that breaks off mid-sentence. Judge it only on what is \
actually there - whatever it has not reached is not evidence of anything, so lower your confidence rather \
than concluding from what is missing.

For each item, answer:
  "i"           the item's number, copied exactly
{test_lines}
  "confidence"  "high", "medium" or "low"
  "why"         at most twelve words
  "truncated"   true if the abstract stops in the middle - it ends mid-sentence, or breaks off before the \
work has said what it did. Judge the text, not its length: a short abstract that finishes its point is NOT \
truncated.

Answer with a JSON array of objects and nothing else. One object per item, in order, no commentary, no \
markdown fences.

{items}
"""

#: How much of an abstract the abstract pass sends: more than an abstract normally runs to, a cap against
#: the record whose "abstract" holds a whole introduction.
ABSTRACT_CHARS = 4000

CONFIDENCES = ("high", "medium", "low")


def rubric(definition: scope_definition.Definition) -> str:
    """The judge's instructions for `definition`: the frame, the tests, and the hints."""
    frame = definition.frame if definition.frame is not None else FRAME.format(scope=definition.scope)
    tests = "\n".join(f'  "{test.name}"  {test.asks}' for test in definition.tests)
    parts = [frame, f"The tests:\n{tests}"]
    if definition.hints:
        parts.append(f"About this search in particular:\n{definition.hints}")
    return "\n\n".join(parts)


def _test_lines(definition: scope_definition.Definition) -> str:
    return "\n".join(f'  "{test.name}"  true or false, as above' for test in definition.tests)


def instrument_material(definition: scope_definition.Definition) -> str:
    """Everything that decides what a judge's answer means, for `raven.librarian.sweep.fingerprint`."""
    return json.dumps({"definition": scope_definition.judge_material(definition),
                       "frame": FRAME, "titles": _TITLES, "abstracts": _ABSTRACTS,
                       "abstract_chars": ABSTRACT_CHARS},
                      ensure_ascii=False, sort_keys=True)


def ledgers(directory: str | pathlib.Path, fingerprint: str) -> tuple[sweep.Ledger, sweep.Ledger]:
    """The title pass's ledger and the abstract pass's, under one instrument's fingerprint."""
    return (sweep.Ledger(directory, "judge-titles", fingerprint),
            sweep.Ledger(directory, "judge-abstracts", fingerprint))


# --------------------------------------------------------------------------------
# What the model is shown of a record

def title_item(i: int, record: scope_records.Record) -> str:
    """Record `record` as the title pass shows it, numbered `i` in its batch."""
    return f"{i}. {record.title}" + (f"\n   [published in: {record.venue}]" if record.venue else "")


def abstract_item(i: int, record: scope_records.Record) -> str:
    """Record `record` as the abstract pass shows it, numbered `i` in its batch."""
    marked = " (TRUNCATED)" if scope_records.looks_truncated(record.abstract) else ""
    venue = f"Published in: {record.venue}\n" if record.venue else ""
    abstract = record.abstract[:ABSTRACT_CHARS] if record.abstract else "(none)"
    return f"--- item {i} ---\nTitle: {record.title}\n{venue}Abstract{marked}: {abstract}\n"


def title_digest(record: scope_records.Record) -> str:
    """The `raven.librarian.sweep.digest` of `record` as the title pass shows it, whatever its place in a batch."""
    return sweep.digest(title_item(0, record))


def abstract_digest(record: scope_records.Record) -> str:
    """The `raven.librarian.sweep.digest` of `record` as the abstract pass shows it."""
    return sweep.digest(abstract_item(0, record))

# --------------------------------------------------------------------------------
# Asking

def _ask(llm_settings: env, prompt: str, *, batch: list[tuple[str, scope_records.Record]],
         definition: scope_definition.Definition, source: str,
         maybe_abort: netutil.Abort | None) -> tuple[dict, list[agent.TurnRecord]]:
    """Ask one batch, and read the answers back by item number. An answer whose number does not resolve is
    dropped rather than guessed at, and the sweep records its record as failed."""
    record = agent.ask_record(llm_settings, prompt, maybe_abort=maybe_abort)
    if record.truncated:
        reason = "the reply was cut off at the token cap; its traces hold what there was"
        return {key: sweep.Failure(reason) for key, _ in batch}, [record]
    try:
        answers = agent.parse_json_reply(record.reply or "")
    except ValueError:
        answers = None
    if not isinstance(answers, list):
        logger.warning(f"_ask: expected a JSON array from the {source} pass, got {type(answers).__name__}; "
                       "the reply is in the traces")
        return {}, [record]
    out = {}
    for answer in answers:
        if not isinstance(answer, dict) or "i" not in answer:
            continue
        try:
            i = int(answer["i"])
        except (TypeError, ValueError):
            continue
        if 0 <= i < len(batch):
            out[batch[i][0]] = normalize(answer, definition, source)
    return out, [record]


def ask_titles(llm_settings: env, batch: list[tuple[str, scope_records.Record]],
               maybe_abort: netutil.Abort | None = None, *,
               definition: scope_definition.Definition) -> tuple[dict, list[agent.TurnRecord]]:
    """The title pass over one batch of `(key, record)`.

    With `definition` bound — `functools.partial(ask_titles, definition=...)` — the `ask_batch` of a
    `raven.librarian.sweep.Sweep`.
    """
    items = "\n".join(title_item(i, record) for i, (_, record) in enumerate(batch))
    prompt = _TITLES.format(rubric=rubric(definition), test_lines=_test_lines(definition), items=items)
    return _ask(llm_settings, prompt, batch=batch, definition=definition, source="title", maybe_abort=maybe_abort)


def ask_abstracts(llm_settings: env, batch: list[tuple[str, scope_records.Record]],
                  maybe_abort: netutil.Abort | None = None, *,
                  definition: scope_definition.Definition) -> tuple[dict, list[agent.TurnRecord]]:
    """The abstract pass over one batch of `(key, record)`, as `ask_titles`. Each answer is tempered for a
    truncated abstract (`temper_for_truncation`)."""
    items = "\n".join(abstract_item(i, record) for i, (_, record) in enumerate(batch))
    prompt = _ABSTRACTS.format(rubric=rubric(definition), test_lines=_test_lines(definition), items=items)
    answers, turns = _ask(llm_settings, prompt, batch=batch, definition=definition, source="abstract",
                          maybe_abort=maybe_abort)
    records_by_key = dict(batch)
    return {key: answer if isinstance(answer, sweep.Failure) else temper_for_truncation(answer, records_by_key[key])
            for key, answer in answers.items()}, turns


# --------------------------------------------------------------------------------
# Reading the answers

def normalize(answer: dict, definition: scope_definition.Definition, source: str) -> dict:
    """One answer with its fields in the shapes the rest relies on.

    Each test is `True`, `False`, or `None` where the reply gave something else; a confidence outside
    `CONFIDENCES` reads as `"low"`; `source` is the pass that answered, `"title"` or `"abstract"`.
    """
    def boolean(value):
        return value if value in (True, False) else None
    confidence = str(answer.get("confidence") or "").strip().lower()
    out = {"tests": {test.name: boolean(answer.get(test.name)) for test in definition.tests},
           "confidence": confidence if confidence in CONFIDENCES else "low",
           "why": " ".join(str(answer.get("why") or "").split())[:200],
           "source": source}
    if source == "abstract":
        out["truncated"] = answer.get("truncated") is True
    return out


def temper_for_truncation(answer: dict, record: scope_records.Record) -> dict:
    """An answer resting on a publisher's teaser, at low confidence whatever the model said.

    By `verdict_of`, that sends any test answered true to review rather than out. Either signal suffices:
    the abstract visibly ends in an ellipsis (`scope_records.looks_truncated`), or the model said it stops.
    """
    # Decided here rather than left to the model, which reads the TRUNCATED marker and concludes from the
    # absence anyway: the prototype measured a high-confidence drop of a record whose 511-character preview
    # never reached the method the drop was about.
    if not (scope_records.looks_truncated(record.abstract) or answer.get("truncated")):
        return answer
    return {**answer, "confidence": "low", "why": f"abstract is a truncated preview: {answer['why']}"}


def verdict_of(answer: dict) -> str:
    """`"drop"`, `"review"` or `"keep"` for one answer.

    A test answered true drops the record at high confidence, and at medium confidence when the abstract
    pass gave it, having read the fuller text; any other true sends it to review. A test left unanswered
    sends it to review too, a gap being something for a person rather than a keep.
    """
    # Medium drops from the abstract pass are trusted because the reviewer goes over every drop and sends
    # the ones it can make a case for to review. Measured on the prototype's corpus: it contested 0.8% of
    # the high-confidence drops and 2.8% of the medium ones, so sending every medium drop to a person would
    # have spent about 250 hand checks to find 7 records with a case.
    tests = answer["tests"].values()
    if True in tests:
        trusted = answer["confidence"] == "high" or (answer["confidence"] == "medium" and answer["source"] == "abstract")
        return "drop" if trusted else "review"
    return "review" if None in tests else "keep"


def needs_escalation(answer: dict, measure: scope_records.TitleMeasure) -> bool:
    """Whether a title pass's answer is asked again with the abstract: low confidence, a title too thin to
    judge from, or any test answered true."""
    # The third trigger is the asymmetry rather than a second guess at the confidence. A false keep costs a
    # reader one line; a false drop removes a record and leaves nothing behind to notice it by. In the
    # prototype, confidence from a title alone carried no information about whether a drop was right: a
    # case for keeping was found for 3.9% of the drops made confidently from a title, against 0.3% of those
    # made confidently after the abstract.
    return (answer["confidence"] == "low"
            or measure.uninformative
            or True in answer["tests"].values())


# --------------------------------------------------------------------------------
# What is left, and what is settled

def _digests(records: list[scope_records.Record], digest) -> dict[str, str]:
    return {record.key: digest(record) for record in records}


def _answers(records: list[scope_records.Record], *, titles: sweep.Ledger,
             abstracts: sweep.Ledger) -> tuple[dict[str, dict], dict[str, dict]]:
    """Each pass's answers about `records` as they read now."""
    return (titles.answers(_digests(records, title_digest)),
            abstracts.answers(_digests(records, abstract_digest)))


def todo_titles(records: Iterable[scope_records.Record], *, titles: sweep.Ledger) -> list[scope_records.Record]:
    """The records the title pass has not answered, as they read now.

    An answer counts only if it was written with `title_digest` of the record as it reads now, so a sweep over
    `titles` passes `digest=title_digest`, and the abstract pass's `abstract_digest`; one written without is
    asked again. The same holds for `todo_abstracts` and `final_answers`.
    """
    records = list(records)
    answered = titles.answers(_digests(records, title_digest))
    return [record for record in records if record.key not in answered]


def todo_abstracts(records: Iterable[scope_records.Record], *, titles: sweep.Ledger, abstracts: sweep.Ledger,
                   measures: dict[str, scope_records.TitleMeasure]) -> list[scope_records.Record]:
    """The records whose title answer escalates and that the abstract pass has not answered, as they read now."""
    records = list(records)
    by_title, by_abstract = _answers(records, titles=titles, abstracts=abstracts)
    return [record for record in records
            if record.key in by_title and record.key not in by_abstract
            and needs_escalation(by_title[record.key], measures[record.key])]


def final_answers(records: Iterable[scope_records.Record], *, titles: sweep.Ledger, abstracts: sweep.Ledger,
                  measures: dict[str, scope_records.TitleMeasure]) -> dict[str, dict]:
    """Each settled record's final answer, with its `verdict`, by key.

    The abstract pass's where the record escalated and it has answered; the title pass's where the record
    did not escalate. A record still waiting on either pass is absent, so a partial run reports only what
    it has settled. An answer given about a record's text as it read before an edit is not settled.
    """
    records = list(records)
    by_title, by_abstract = _answers(records, titles=titles, abstracts=abstracts)
    out = {}
    for record in records:
        if record.key not in by_title:
            continue
        answer = by_title[record.key]
        if needs_escalation(answer, measures[record.key]):
            if record.key not in by_abstract:
                continue
            answer = by_abstract[record.key]
        out[record.key] = {**answer, "verdict": verdict_of(answer)}
    return out

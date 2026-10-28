"""The reviewer: a second pass over the judge's drops, asked the opposite question.

Re-asking "was this verdict right?" invites agreement. This asks instead whether a credible case exists
that the record belongs in scope, and a drop it makes a case for goes to a person rather than out
(`apply_reviews`). Its answer is a case in a line, checkable against the record in seconds.

**The reviewer is told nothing about where the records came from**: not that they were judged, nor dropped,
nor by what. It sees a record and a question. Nor is it shown the definition's tests or hints, being a
pass independent of the judge's. It is shown the scope and the scope notes: what belongs is the user's to say,
and a reviewer arguing against that says nothing about the judge.

**A reviewer that finds a case for everything is not discriminating**, it only looks thorough. So a
control group of kept records can be mixed in, unlabelled (`draw_control`), and the case rate for the drops
read against the rate for the control. The control is drawn at the drops' own mix of confidence, since a
control easier than what it is compared against flatters the reviewer.

A `raven.librarian.sweep`, batched, with one ledger (`ledger`) named for the instrument
(`instrument_material`).
"""

__all__ = ["ABSTRACT_CHARS", "instrument_material", "ledger", "item", "digest", "ask_reviews", "draw_control", "apply_reviews"]

import collections
import json
import logging
import pathlib
import random

from unpythonic.env import env

from ...common import netutil
from ...librarian import agent, sweep
from . import definition as scope_definition
from . import records as scope_records

logger = logging.getLogger(__name__)

#: How much of an abstract the reviewer is sent; as the judge's abstract pass.
ABSTRACT_CHARS = 4000

_INSTRUCTIONS = """\
You are helping assemble a collection of the literature on: {scope}.

For each numbered record below, answer one question: is there a credible case that this record belongs in \
that collection?

A credible case rests on what the record actually says. It is not enough that the record could conceivably \
touch the topic - it has to give you something to point at. If you find yourself reaching, there is no \
case, and saying so is the useful answer.

For each record, answer:
  "i"        the record's number, copied exactly
  "belongs"  true if a credible case exists, false if not
  "case"     if true, the case in at most twenty words, pointing at what the record says. If false, at \
most twenty words on what is missing.

Answer with a JSON array of objects and nothing else. One object per record, in order, no commentary, no \
markdown fences.

{items}
"""


def instrument_material(definition: scope_definition.Definition) -> str:
    """Everything that decides what a review means: the scope, its notes and this module's prompt."""
    return json.dumps({"scope": definition.scope, "instructions": _INSTRUCTIONS, "abstract_chars": ABSTRACT_CHARS}
                      | scope_definition.scope_notes_material(definition),
                      ensure_ascii=False, sort_keys=True)


def ledger(directory: str | pathlib.Path, fingerprint: str) -> sweep.Ledger:
    """The reviewer's ledger under an instrument's fingerprint."""
    return sweep.Ledger(directory, "review", fingerprint)


def item(i: int, record: scope_records.Record) -> str:
    """Record `record` as the reviewer shows it, numbered `i` in its batch."""
    venue = f"Published in: {record.venue}\n" if record.venue else ""
    abstract = record.abstract[:ABSTRACT_CHARS] if record.abstract else "(none)"
    return f"--- record {i} ---\nTitle: {record.title}\n{venue}Abstract: {abstract}\n"


def digest(record: scope_records.Record) -> str:
    """The `raven.librarian.sweep.digest` of `record` as the reviewer shows it, whatever its place in a batch."""
    return sweep.digest(item(0, record))


def ask_reviews(llm_settings: env, batch: list[tuple[str, scope_records.Record]],
                maybe_abort: netutil.Abort | None = None, *,
                definition: scope_definition.Definition) -> tuple[dict, list[agent.TurnRecord]]:
    """The reviewer over one batch of `(key, record)`.

    With `definition` bound, the `ask_batch` of a `raven.librarian.sweep.Sweep`. Each answer is
    `{"belongs": True | False | None, "case": str}`, `None` where the reply gave something else. An answer
    whose number does not resolve is dropped rather than guessed at, and a reply cut off at the token cap
    fails the whole batch, to be asked again.
    """
    prompt = _INSTRUCTIONS.format(scope=definition.scope,
                                  items="\n".join(item(i, record) for i, (_, record) in enumerate(batch)))
    prompt = scope_definition.with_scope_notes(prompt, definition)
    turn = agent.ask_record(llm_settings, prompt, maybe_abort=maybe_abort)
    if turn.truncated:
        reason = "the reply was cut off at the token cap; its traces hold what there was"
        return {key: sweep.Failure(reason) for key, _ in batch}, [turn]
    try:
        answers = agent.parse_json_reply(turn.reply or "")
    except ValueError:
        answers = None
    if not isinstance(answers, list):
        logger.warning(f"ask_reviews: expected a JSON array, got {type(answers).__name__}; the reply is in the traces")
        return {}, [turn]
    out = {}
    for answer in answers:
        if not isinstance(answer, dict) or "i" not in answer:
            continue
        try:
            i = int(answer["i"])
        except (TypeError, ValueError):
            continue
        if 0 <= i < len(batch):
            belongs = answer.get("belongs")
            out[batch[i][0]] = {"belongs": belongs if belongs in (True, False) else None,
                                "case": " ".join(str(answer.get("case") or "").split())[:300]}
    return out, [turn]


def draw_control(final: dict[str, dict], *, size: int, seed: int) -> list[str]:
    """Keys of kept records for a control group: `size` of them, at the drops' mix of confidence.

    `final`: the judge's final answers by key, as `raven.papers.scopebib.judge.final_answers` gives them.
    `seed`: the draw is reproducible from it, so a control can be drawn again to compare two reviewers.

    Where a confidence level has too few keeps, it gives what it has, and the shortfall is logged.
    """
    drops = [answer for answer in final.values() if answer["verdict"] == "drop"]
    if not drops or size <= 0:
        return []
    mix = collections.Counter(answer["confidence"] for answer in drops)
    rng = random.Random(seed)
    drawn = []
    for confidence, count in sorted(mix.items()):
        wanted = round(size * count / len(drops))
        pool = sorted(key for key, answer in final.items()
                      if answer["verdict"] == "keep" and answer["confidence"] == confidence)
        if len(pool) < wanted:
            logger.warning(f"draw_control: {wanted} kept records wanted at {confidence} confidence, "
                           f"only {len(pool)} there")
        drawn += rng.sample(pool, min(wanted, len(pool)))
    return drawn


def apply_reviews(final: dict[str, dict], reviews: dict[str, dict]) -> dict[str, dict]:
    """The judge's final answers with the reviewer's rescues applied: a drop it made a case for goes to review.

    A rescued answer carries `"rescued": True` and the reviewer's `case`. A drop the reviewer has not yet
    answered stays a drop: a partial review reports what it has found.
    """
    out = {}
    for key, answer in final.items():
        review = reviews.get(key)
        if answer["verdict"] == "drop" and review is not None and review["belongs"] is True:
            answer = {**answer, "verdict": "review", "rescued": True, "case": review["case"]}
        out[key] = answer
    return out

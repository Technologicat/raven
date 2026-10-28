"""The extractor: what a kept record says, field by field, for the rules to act on.

The judge keeps a record whenever no test can be positively established, so a keep carries no reason of its
own. Asking for fields rather than a verdict makes the keeps auditable: a rule over the fields can then
remove what they positively rule out, with the record's own words as the evidence (`raven.papers.scopebib.rules`).

Each field of the definition is one answer per record: a value of its vocabulary, `not_stated` included,
or for a free-text field a few words. A field marked `quote` comes with the words from the record that
settle it, which are checked against the record's text (`normalize`).

A `raven.librarian.sweep`, batched, with one ledger (`ledger`) named for the instrument
(`instrument_material`): the definition's fields and this module's prompt.
"""

__all__ = ["ABSTRACT_CHARS", "QUOTE_MARKS", "instrument_material", "ledger", "item", "digest", "ask_fields", "normalize"]

import json
import logging
import pathlib

from unpythonic.env import env

from ...common import netutil
from ...librarian import agent, sweep
from . import definition as scope_definition
from . import records as scope_records

logger = logging.getLogger(__name__)

#: How much of an abstract the extractor is sent; as the judge's abstract pass.
ABSTRACT_CHARS = 4000

#: The quotation marks a model may wrap a quote in, which are not part of it: straight and typographic, double and single.
QUOTE_MARKS = "\"'“”‘’"

_INSTRUCTIONS = """\
You are reading records from a literature search for: {scope}. You are recording what each record says. You \
are NOT deciding whether any of them belongs anywhere - do not screen, include or exclude. Report what is \
there.

For each numbered record, answer:
  "i"  the record's number, copied exactly
{field_lines}

"not_stated" is a real answer, and often the correct one. Where the record does not say, answer \
"not_stated" rather than inferring from the topic, the venue or what would be usual - if the text does not \
say, you do not know.

A quote must be words that actually appear in the record, copied exactly. If you cannot find any, answer "" \
- do not paraphrase and do not compose a sentence. A value other than "not_stated" with an empty quote \
beside it is a contradiction, so let the quote decide.

Answer with a JSON array of objects and nothing else. One object per record, in order, no commentary, no \
markdown fences. Every value is a string in double quotes.

{items}
"""


def _field_lines(definition: scope_definition.Definition) -> str:
    lines = []
    for field in definition.fields:
        if field.values is None:
            cap = f", at most {field.max_words} words" if field.max_words else ""
            lines.append(f'  "{field.name}"  {field.asks}, in a few words{cap}')
            continue
        values = "\n".join(f"      {value}: {description}" for value, description in field.values.items())
        lines.append(f'  "{field.name}"  {field.asks}. One of:\n{values}')
        if field.quote:
            needing = ", ".join(value for value in field.values if value in field.quote)
            lines.append(f'  "{field.name}_quote"  the words in the record that settle "{field.name}", quoted '
                         f'exactly, or "". Needed when the answer is one of: {needing}')
    return "\n".join(lines)


def instrument_material(definition: scope_definition.Definition) -> str:
    """Everything that decides what an extraction means, for `raven.librarian.sweep.fingerprint`."""
    return json.dumps({"definition": scope_definition.extract_material(definition),
                       "instructions": _INSTRUCTIONS, "abstract_chars": ABSTRACT_CHARS},
                      ensure_ascii=False, sort_keys=True)


def ledger(directory: str | pathlib.Path, fingerprint: str) -> sweep.Ledger:
    """The extractor's ledger under an instrument's fingerprint."""
    return sweep.Ledger(directory, "extract", fingerprint)


def item(i: int, record: scope_records.Record) -> str:
    """Record `record` as the extractor shows it, numbered `i` in its batch."""
    venue = f"Published in: {record.venue}\n" if record.venue else ""
    abstract = record.abstract[:ABSTRACT_CHARS] if record.abstract else "(none)"
    return f"--- record {i} ---\nTitle: {record.title}\n{venue}Abstract: {abstract}\n"


def digest(record: scope_records.Record) -> str:
    """The `raven.librarian.sweep.digest` of `record` as the extractor shows it, whatever its place in a batch."""
    return sweep.digest(item(0, record))


def ask_fields(llm_settings: env, batch: list[tuple[str, scope_records.Record]],
               maybe_abort: netutil.Abort | None = None, *,
               definition: scope_definition.Definition) -> tuple[dict, list[agent.TurnRecord]]:
    """The extractor over one batch of `(key, record)`.

    With `definition` bound — `functools.partial(ask_fields, definition=...)` — the `ask_batch` of a
    `raven.librarian.sweep.Sweep`. An answer whose number does not resolve is dropped rather than guessed at,
    and a reply cut off at the token cap fails the whole batch, to be asked again.
    """
    prompt = _INSTRUCTIONS.format(scope=definition.scope, field_lines=_field_lines(definition),
                                  items="\n".join(item(i, record) for i, (_, record) in enumerate(batch)))
    prompt = scope_definition.with_scope_notes(prompt, definition)
    turn = agent.ask_record(llm_settings, prompt, maybe_abort=maybe_abort)
    if turn.truncated:
        reason = "the reply was cut off at the token cap; its traces hold what there was"
        return {key: sweep.Failure(reason) for key, _ in batch}, [turn]
    try:
        parsed = agent.parse_json_reply(turn.reply or "")
    except ValueError:
        parsed = None
    answers = scope_records.answer_list(parsed)
    if answers is None:
        logger.warning(f"ask_fields: expected a JSON array, got {type(parsed).__name__}; the reply is in the traces")
        return {}, [turn]
    out = {}
    for answer in answers:
        i = scope_records.item_number(answer, len(batch))
        if i is not None:
            key, record = batch[i]
            out[key] = normalize(answer, definition, record)
    return out, [turn]


def normalize(answer: dict, definition: scope_definition.Definition, record: scope_records.Record) -> dict:
    """One extraction with its vocabularies and quotes checked.

    Returns `{"fields": {name: value}, "quotes": {name: {"text", "found"}}, "raw": {name: value}}`. A closed
    field's value outside its vocabulary becomes `not_stated`, the original kept in `raw` so a systematic
    mismatch shows; a free-text value is cut to its word cap. A quote is `found` when its words, whitespace
    folded, appear in the record's title, venue or abstract.
    """
    haystack = " ".join(f"{record.title} {record.venue} {record.abstract}".split()).lower()
    fields, quotes, raw = {}, {}, {}
    for field in definition.fields:
        value = " ".join(str(answer.get(field.name) or "").split())
        if field.values is None:
            words = value.split()
            fields[field.name] = " ".join(words[:field.max_words] if field.max_words else words)
            continue
        by_lowercase = {known.lower(): known for known in field.values}
        if value.lower() in by_lowercase:
            fields[field.name] = by_lowercase[value.lower()]
        else:
            fields[field.name] = scope_definition.NOT_STATED
            raw[field.name] = value
        if field.quote:
            text = " ".join(str(answer.get(f"{field.name}_quote") or "").split())
            # A model quoting often wraps the quote in quotation marks of its own, which are not in the record.
            bare = text.strip(QUOTE_MARKS + " ")
            quotes[field.name] = {"text": text, "found": bool(bare) and bare.lower() in haystack}
    return {"fields": fields, "quotes": quotes, "raw": raw}

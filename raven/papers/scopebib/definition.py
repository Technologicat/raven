"""The definition file: what `raven-scopebib` asks of one corpus, and what it does with the answers.

A TOML file, normally `<bib stem>.scope.toml` beside the `.bib`, holding everything that is about *this*
corpus — the scope question the search asked, the judge's tests, the extractor's fields, and the rules over
those fields — while the tool supplies everything that is not:

    scope = "studies on different aspects of the use of AI agents in higher education"
    scope_notes = "What belongs and what does not, at length. Optional; every stage reads it."
    hints = "Domain cautions for the judge, beyond what each test says. Optional; the judge's alone."
    frame = "Replaces the tool's own framing of the judge's task. Optional, and rarely wanted."

    [[test]]               # one per way of being off topic; true means positive evidence of it
    name = "no_ai"
    asks = "true ONLY if the work is positively about something with no artificial intelligence in it..."

    [[field]]              # what a kept record says
    name = "level"
    asks = "the educational level the work is set at"
    quote = ["school"]     # an answer of `school` must come with the record's words that show it, such
                           # as "pupils in two schools", which are checked against the record; `true` is
                           # every value, and a value asserting an absence has no words to quote
    [field.values]         # the answers it may give: answer = "what it means", as the model is told.
                           # `not_stated` is added to every list
    higher_education = "a university, college, polytechnic or their students and faculty"
    school = "primary or secondary schooling, K-12, pupils"

    [[field]]
    name = "ai_role"
    asks = "what the AI does in this work, or 'none'"
    max_words = 8          # no `values`: free text, kept for reading, never acted on by a rule

    [[rule]]               # first match wins; AND across fields, OR within a list
    label = "school or vocational level, stated in the text"
    when = { level = ["school", "vocational"] }
    action = "remove"      # or "hold", for a person to look at

`load_definition` reads and checks it, naming the file and the entry in every error. The prose is
normalized paragraph by paragraph, so that rewrapping a description changes neither what the model is sent
nor the instrument's fingerprint (`judge_material`, `extract_material`).
"""

__all__ = ["NOT_STATED", "ACTIONS",
           "Test", "Field", "Rule", "Definition",
           "DefinitionError", "load_definition",
           "SCOPE_NOTES_HEADING", "with_scope_notes", "scope_notes_material",
           "judge_material", "extract_material"]

import dataclasses
import json
import pathlib
import tomllib
from typing import Any

#: The value every closed vocabulary has without listing it: the record does not say.
NOT_STATED = "not_stated"

#: What a rule can do with a record it matches.
ACTIONS = ("remove", "hold")


@dataclasses.dataclass(frozen=True)
class Test:
    """One way of being off topic. `asks` is the description the judge is given."""
    name: str
    asks: str


@dataclasses.dataclass(frozen=True)
class Field:
    """One thing the extractor reports about a record.

    `values`: the closed vocabulary, value -> description, `NOT_STATED` last; `None` for free text.
    `quote`: the values that must quote the words that settle them; empty for none. A value that asserts an
             absence, such as "not set in education at all", has nothing to quote, and is left out.
    `max_words`: the length cap on a free-text answer, `None` for a closed one.
    """
    name: str
    asks: str
    values: dict[str, str] | None
    quote: frozenset[str]
    max_words: int | None


@dataclasses.dataclass(frozen=True)
class Rule:
    """What to do with a record whose extracted fields match `when`: field -> the values that match."""
    label: str
    when: dict[str, frozenset[str]]
    action: str


@dataclasses.dataclass(frozen=True)
class Definition:
    """A loaded definition file. `scope_notes` and `hints` are `""` when the file has none; `frame` is `None`
    unless the file replaces the tool's own."""
    path: pathlib.Path
    scope: str
    scope_notes: str
    hints: str
    frame: str | None
    tests: tuple[Test, ...]
    fields: tuple[Field, ...]
    rules: tuple[Rule, ...]


class DefinitionError(ValueError):
    """The definition file cannot be used. The message names the file and the entry."""


_TOP_KEYS = {"scope", "scope_notes", "hints", "frame", "test", "field", "rule"}
_TEST_KEYS = {"name", "asks"}
_FIELD_KEYS = {"name", "asks", "values", "quote", "max_words"}
_RULE_KEYS = {"label", "when", "action"}


def _prose(text: str) -> str:
    """Whitespace collapsed within each paragraph, paragraphs separated by one blank line."""
    paragraphs = [" ".join(paragraph.split()) for paragraph in text.replace("\r\n", "\n").split("\n\n")]
    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph)


def load_definition(path: str | pathlib.Path) -> Definition:
    """Read and check the definition file at `path`. Raises `DefinitionError` naming what is wrong."""
    path = pathlib.Path(path)
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise DefinitionError(f"{path}: not valid TOML: {exc}") from exc
    except OSError as exc:
        raise DefinitionError(f"{path}: cannot be read: {type(exc)}: {exc}") from exc

    def fail(where: str, message: str) -> None:
        raise DefinitionError(f"{path}: {where}: {message}")

    def check_keys(where: str, table: Any, allowed: set[str], required: set[str]) -> None:
        if not isinstance(table, dict):
            fail(where, f"expected a table, got {type(table).__name__}")
        if unknown := sorted(set(table) - allowed):
            # TOML reads a key written after a `[[table]]` header as that table's own, so a top-level key
            # placed below the first test lands in the test. Say so, since "unknown key" alone misleads.
            if misplaced := sorted(set(unknown) & (_TOP_KEYS - {"test", "field", "rule"})):
                fail(where, f"{', '.join(misplaced)} belong{'s' if len(misplaced) == 1 else ''} at the top "
                            f"level: move {'it' if len(misplaced) == 1 else 'them'} above the first [[...]] entry")
            fail(where, f"unknown key{'s' if len(unknown) != 1 else ''} {', '.join(unknown)}; "
                        f"known: {', '.join(sorted(allowed))}")
        if missing := sorted(required - set(table)):
            fail(where, f"missing {', '.join(missing)}")

    def text(where: str, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            fail(where, "must be a non-empty string")
        return _prose(value)

    def name(where: str, value: Any, seen: set[str]) -> str:
        if not isinstance(value, str) or not value.isidentifier():
            fail(where, f"`name` must be a word made of letters, digits and underscores, got {value!r}")
        if value in seen:
            fail(where, f"the name {value!r} is used twice")
        seen.add(value)
        return value

    def entries(key: str) -> list:
        value = raw.get(key, [])
        if not isinstance(value, list):
            fail(key, f"must be written as [[{key}]] entries")
        return value

    def label(kind: str, i: int, entry: Any) -> str:
        return f"{kind} {i}" + (f" '{entry['name']}'" if isinstance(entry, dict) and "name" in entry else "")

    check_keys("the top level", raw, _TOP_KEYS, {"scope", "test"})
    scope = text("scope", raw["scope"])
    scope_notes = text("scope_notes", raw["scope_notes"]) if "scope_notes" in raw else ""
    hints = text("hints", raw["hints"]) if "hints" in raw else ""
    frame = text("frame", raw["frame"]) if "frame" in raw else None

    tests = []
    seen: set[str] = set()
    for i, entry in enumerate(entries("test"), start=1):
        where = label("test", i, entry)
        check_keys(where, entry, _TEST_KEYS, _TEST_KEYS)
        tests.append(Test(name(where, entry["name"], seen), text(f"{where}, asks", entry["asks"])))
    if not tests:
        fail("test", "at least one [[test]] is needed: the judge has nothing to ask otherwise")

    fields = []
    seen = set()
    for i, entry in enumerate(entries("field"), start=1):
        where = label("field", i, entry)
        check_keys(where, entry, _FIELD_KEYS, {"name", "asks"})
        field_name = name(where, entry["name"], seen)
        quote = entry.get("quote", False)
        is_value_list = isinstance(quote, list) and all(isinstance(v, str) for v in quote)
        if not (isinstance(quote, bool) or is_value_list):
            fail(where, "`quote` must be true, false, or a list of the values that must be quoted")
        if quote and "values" not in entry:
            fail(where, "`quote` needs `values`: only a closed vocabulary is acted on by a rule, so a quote "
                        "on free text would protect nothing")
        if "values" in entry:
            if "max_words" in entry:
                fail(where, "`max_words` is for free text; a field with `values` is a closed vocabulary")
            raw_values = entry["values"]
            if not isinstance(raw_values, dict) or not raw_values:
                fail(where, "`values` must be a table of value = \"description\", with at least one value")
            values = {}
            for value, description in raw_values.items():
                if not value.isidentifier():
                    fail(where, f"the value {value!r} must be a word made of letters, digits and underscores")
                if value == NOT_STATED:
                    fail(where, f"`{NOT_STATED}` is part of every vocabulary already; leave it out")
                values[value] = text(f"{where}, value {value!r}", description)
            if quote is True:
                quoted = frozenset(values)
            elif quote is False:
                quoted = frozenset()
            else:
                if NOT_STATED in quote:
                    fail(where, f"`{NOT_STATED}` has nothing to quote; leave it out of `quote`")
                if unknown := sorted(set(quote) - set(values)):
                    fail(where, f"`quote` names {', '.join(map(repr, unknown))}, not among the values")
                quoted = frozenset(quote)
            values[NOT_STATED] = "the record does not say"
            fields.append(Field(field_name, text(f"{where}, asks", entry["asks"]), values, quoted, None))
        else:
            max_words = entry.get("max_words")
            if max_words is not None and (not isinstance(max_words, int) or isinstance(max_words, bool)
                                          or max_words < 1):
                fail(where, "`max_words` must be a positive whole number")
            fields.append(Field(field_name, text(f"{where}, asks", entry["asks"]), None, frozenset(), max_words))

    closed = {field.name: field for field in fields if field.values is not None}
    rules = []
    for i, entry in enumerate(entries("rule"), start=1):
        where = f"rule {i}" + (f" '{entry['label']}'" if isinstance(entry, dict) and "label" in entry else "")
        check_keys(where, entry, _RULE_KEYS, _RULE_KEYS)
        if entry["action"] not in ACTIONS:
            fail(where, f"`action` must be one of {', '.join(ACTIONS)}, got {entry['action']!r}")
        when = entry["when"]
        if not isinstance(when, dict) or not when:
            fail(where, "`when` must name at least one field, as { field = [\"value\", ...] }")
        matched = {}
        for field_name, wanted in when.items():
            if field_name not in closed:
                known = ", ".join(sorted(closed)) or "none"
                fail(where, f"`when` names {field_name!r}, which is not a field with `values` (those are: {known})")
            if not isinstance(wanted, list) or not wanted or not all(isinstance(v, str) for v in wanted):
                fail(where, f"`when.{field_name}` must be a list of values")
            if unknown := sorted(set(wanted) - set(closed[field_name].values)):
                fail(where, f"`when.{field_name}` names {', '.join(map(repr, unknown))}, which "
                            f"{'is' if len(unknown) == 1 else 'are'} not among its values")
            matched[field_name] = frozenset(wanted)
        rules.append(Rule(text(f"{where}, label", entry["label"]), matched, entry["action"]))

    return Definition(path=path, scope=scope, scope_notes=scope_notes, hints=hints, frame=frame,
                      tests=tuple(tests), fields=tuple(fields), rules=tuple(rules))


#: What precedes the scope notes in every stage's prompt.
SCOPE_NOTES_HEADING = "What this covers, and what it does not:"


def with_scope_notes(prompt: str, definition: Definition) -> str:
    """`prompt` with the definition's scope notes inserted after its first paragraph, which names the scope.

    Unchanged when the definition has none. Every stage that is told the scope places its notes this way.
    """
    if not definition.scope_notes:
        return prompt
    head, sep, rest = prompt.partition("\n\n")
    return f"{head}\n\n{SCOPE_NOTES_HEADING}\n{definition.scope_notes}" + (f"\n\n{rest}" if sep else "")


def scope_notes_material(definition: Definition) -> dict:
    """The scope notes' part of a stage's instrument material: `{}` when there are none."""
    # Absent rather than an empty value, so that a definition without notes keeps the fingerprints it had
    # before the key existed, and with them the answers saved under those fingerprints.
    return {"scope_notes": [SCOPE_NOTES_HEADING, definition.scope_notes]} if definition.scope_notes else {}


def judge_material(definition: Definition) -> str:
    """The definition's half of the judge's instrument: the scope, its notes, the hints, the frame and the tests.

    The judge adds its own prompt to it. Normalized, so a change to a comment or to how the prose is
    wrapped leaves it as it was, and any change to what the judge is asked does not.
    """
    return json.dumps({"scope": definition.scope, "hints": definition.hints, "frame": definition.frame,
                       "tests": [[test.name, test.asks] for test in definition.tests]}
                      | scope_notes_material(definition),
                      ensure_ascii=False, sort_keys=True)


def extract_material(definition: Definition) -> str:
    """The definition's half of the extractor's instrument: the scope, its notes and the fields.

    The rules are not in it: they act on the answers when the reports are written, so changing one asks
    nothing again.
    """
    return json.dumps({"scope": definition.scope,
                       "fields": [[field.name, field.asks, field.values, sorted(field.quote), field.max_words]
                                  for field in definition.fields]}
                      | scope_notes_material(definition),
                      ensure_ascii=False, sort_keys=True)

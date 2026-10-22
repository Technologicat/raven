"""Tests for `raven.papers.scopebib.replay`: bringing the prototype judge's answers into the pass ledgers."""

import json

import pytest

from raven.papers.scopebib import definition as scope_definition
from raven.papers.scopebib import judge, replay

DEFINITION = """
scope = "studies of ravens"

[[test]]
name = "not_birds"
asks = "true if the work is positively not about birds"
"""


@pytest.fixture
def definition(tmp_path):
    path = tmp_path / "ravens.scope.toml"
    path.write_text(DEFINITION, encoding="utf-8")
    return scope_definition.load_definition(path)


def prototype_file(tmp_path, *entries):
    path = tmp_path / "judged.jsonl"
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")
    return path


def line(key, source, not_birds=False, confidence="high", why="w"):
    return {"key": key, "not_birds": not_birds, "confidence": confidence, "why": why, "source": source}


class TestImport:
    def test_each_pass_goes_to_its_own_ledger_in_the_tool_s_shape(self, tmp_path, definition):
        titles, abstracts = judge.ledgers(tmp_path, replay.fingerprint_of("the prototype's prompts"))
        count = replay.import_prototype_judgements(
            prototype_file(tmp_path, line("a", "title", not_birds=True), line("a", "abstract"), line("b", "title")),
            titles=titles, abstracts=abstracts, definition=definition)
        assert count == replay.ImportCount(titles=2, abstracts=1)
        assert titles.answers()["a"] == {"tests": {"not_birds": True}, "confidence": "high", "why": "w",
                                         "source": "title"}
        assert abstracts.answers()["a"]["tests"] == {"not_birds": False}
        assert "b" not in abstracts.answers()

    def test_the_latest_answer_per_key_and_pass_wins(self, tmp_path, definition):
        titles, abstracts = judge.ledgers(tmp_path, "proto000")
        replay.import_prototype_judgements(
            prototype_file(tmp_path, line("a", "title", why="first"), line("a", "title", why="second")),
            titles=titles, abstracts=abstracts, definition=definition)
        assert titles.answers()["a"]["why"] == "second"
        assert len(titles.path.read_text().splitlines()) == 1, "only the latest is written, not the history"

    def test_a_withheld_verdict_reads_as_unanswered_and_so_as_review(self, tmp_path, definition):
        titles, abstracts = judge.ledgers(tmp_path, "proto000")
        replay.import_prototype_judgements(prototype_file(tmp_path, line("a", "title", not_birds=None)),
                                           titles=titles, abstracts=abstracts, definition=definition)
        assert judge.verdict_of(titles.answers()["a"]) == "review"

    def test_the_answers_stay_under_the_replayed_instrument(self, tmp_path, definition):
        replayed = judge.ledgers(tmp_path, replay.fingerprint_of("the prototype's prompts"))
        replay.import_prototype_judgements(prototype_file(tmp_path, line("a", "title")),
                                           titles=replayed[0], abstracts=replayed[1], definition=definition)
        own = judge.ledgers(tmp_path, replay.fingerprint_of("the tool's prompts"))
        assert own[0].answers() == {}, "a run under another instrument must not see them"


class TestRefusals:
    def test_answers_for_other_tests_are_refused_by_name(self, tmp_path, definition):
        titles, abstracts = judge.ledgers(tmp_path, "proto000")
        with pytest.raises(ValueError, match=r"judged.jsonl:1: no answer for the test not_birds"):
            replay.import_prototype_judgements(prototype_file(tmp_path, {"key": "a", "source": "title", "no_ai": True}),
                                               titles=titles, abstracts=abstracts, definition=definition)

    def test_a_line_that_is_not_an_answer_is_refused_by_number(self, tmp_path, definition):
        titles, abstracts = judge.ledgers(tmp_path, "proto000")
        path = tmp_path / "judged.jsonl"
        path.write_text(json.dumps(line("a", "title")) + "\n{not json\n", encoding="utf-8")
        with pytest.raises(ValueError, match=r"judged.jsonl:2: not a JSON line"):
            replay.import_prototype_judgements(path, titles=titles, abstracts=abstracts, definition=definition)
        path.write_text(json.dumps({"key": "a", "source": "both"}) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="needs a `key` and a `source`"):
            replay.import_prototype_judgements(path, titles=titles, abstracts=abstracts, definition=definition)


def test_fingerprint_of_is_stable_and_follows_its_texts():
    assert replay.fingerprint_of("a", "b") == replay.fingerprint_of("a", "b")
    assert replay.fingerprint_of("a", "b") != replay.fingerprint_of("ab")
    assert len(replay.fingerprint_of("x")) == 8


EXTRACTOR_DEFINITION = DEFINITION + """
[[field]]
name = "level"
asks = "the level"
quote = ["school"]
[field.values]
school = "school"
not_applicable = "not in education"

[[field]]
name = "learning"
asks = "whether a person learns"
[field.values]
yes = "a person learns"
no = "only software learns"
"""


class TestExtractions:
    @pytest.fixture
    def definition(self, tmp_path):
        path = tmp_path / "levels.scope.toml"
        path.write_text(EXTRACTOR_DEFINITION, encoding="utf-8")
        return scope_definition.load_definition(path)

    @staticmethod
    def corpus(*keys):
        from raven.papers.scopebib import records
        return {key: records.Record(key, "Pupils in a school", "", "", "") for key in keys}

    def test_the_prototype_s_spellings_are_translated(self, tmp_path, definition):
        from raven.papers.scopebib import extract
        path = tmp_path / "extracted.jsonl"
        lines = [{"key": "a", "v": "p", "level": "school", "learning": True, "evidence": '"a school"'},
                 {"key": "b", "v": "p", "level": "not_applicable", "learning": "unclear", "evidence": ""},
                 {"key": "elsewhere", "v": "p", "level": "school", "learning": False, "evidence": ""}]
        path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
        ledger = extract.ledger(tmp_path, "proto000")
        written = replay.import_prototype_extractions(path, ledger=ledger, definition=definition,
                                                      records=self.corpus("a", "b"))
        assert written == 2, "a key not in the corpus is skipped"
        a, b = ledger.answers()["a"], ledger.answers()["b"]
        assert a["fields"] == {"level": "school", "learning": "yes"}
        assert a["quotes"]["level"] == {"text": '"a school"', "found": True}, "evidence is the quoted field's quote"
        assert b["fields"]["learning"] == scope_definition.NOT_STATED and b["raw"] == {"learning": "unclear"}

    def test_two_quoted_fields_cannot_take_one_evidence(self, tmp_path, definition):
        import dataclasses
        from raven.papers.scopebib import extract
        doubled = dataclasses.replace(definition, fields=tuple(dataclasses.replace(f, quote=frozenset({"yes"}))
                                                               if f.name == "learning" else f
                                                               for f in definition.fields))
        with pytest.raises(ValueError, match="which one it belongs to cannot be told"):
            replay.import_prototype_extractions(tmp_path / "unread.jsonl", ledger=extract.ledger(tmp_path, "p"),
                                                definition=doubled, records={})

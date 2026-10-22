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

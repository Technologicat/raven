"""Tests for `raven.papers.scopebib.review`: the reviewer, its control, and what its rescues do."""

import collections
import json

import pytest

from raven.librarian import agent, sweep
from raven.papers.scopebib import definition as scope_definition
from raven.papers.scopebib import records as scope_records
from raven.papers.scopebib import review


@pytest.fixture
def definition(tmp_path):
    path = tmp_path / "ravens.scope.toml"
    path.write_text('scope = "studies of ravens"\nhints = "a hint"\n\n[[test]]\nname = "not_birds"\n'
                    'asks = "true if positively not about birds"\n', encoding="utf-8")
    return scope_definition.load_definition(path)


RECORD = scope_records.Record(key="k", title="Ravens in winter", abstract="We counted ravens.", venue="", link="")


def script(monkeypatch, *replies):
    prompts = []

    def ask_record(llm_settings, prompt, maybe_abort=None, **kw):
        prompts.append(prompt)
        reply = replies[len(prompts) - 1]
        return type("T", (), {"reply": reply or "[{", "truncated": reply is None})()
    monkeypatch.setattr(agent, "ask_record", ask_record)
    return prompts


class TestAsking:
    def test_the_reviewer_is_told_the_scope_and_nothing_about_provenance(self, definition, monkeypatch):
        prompts = script(monkeypatch, "[]")
        review.ask_reviews(None, [("k", RECORD)], definition=definition)
        text = prompts[0].lower()
        assert "studies of ravens" in text and "ravens in winter" in text
        for word in ("drop", "judge", "verdict", "not_birds", "a hint"):
            assert word not in text, f"the reviewer must not learn about the judging: {word!r}"

    def test_the_reviewer_is_told_the_scope_notes_and_still_not_the_hints(self, definition, monkeypatch, tmp_path):
        path = tmp_path / "noted.scope.toml"
        path.write_text('scope = "studies of ravens"\nscope_notes = "Crows are out."\nhints = "a hint"\n\n'
                        '[[test]]\nname = "not_birds"\nasks = "true if positively not about birds"\n', encoding="utf-8")
        noted = scope_definition.load_definition(path)
        prompts = script(monkeypatch, "[]", "[]")
        review.ask_reviews(None, [("k", RECORD)], definition=noted)
        review.ask_reviews(None, [("k", RECORD)], definition=definition)
        assert "Crows are out." in prompts[0] and "a hint" not in prompts[0]
        assert scope_definition.SCOPE_NOTES_HEADING not in prompts[1], "no notes, no heading"
        assert review.instrument_material(noted) != review.instrument_material(definition)

    def test_answers_are_read_by_number(self, definition, monkeypatch):
        script(monkeypatch, json.dumps([{"i": 0, "belongs": True, "case": " Counts  ravens. "},
                                        {"i": 3, "belongs": False, "case": "no such record"}]))
        answers, turns = review.ask_reviews(None, [("k", RECORD)], definition=definition)
        assert answers == {"k": {"belongs": True, "case": "Counts ravens."}} and len(turns) == 1

    def test_a_belongs_that_is_not_a_boolean_is_unanswered(self, definition, monkeypatch):
        script(monkeypatch, json.dumps([{"i": 0, "belongs": "maybe", "case": ""}]))
        answers, _ = review.ask_reviews(None, [("k", RECORD)], definition=definition)
        assert answers["k"]["belongs"] is None

    def test_a_cut_off_reply_fails_the_batch(self, definition, monkeypatch):
        script(monkeypatch, None)
        answers, _ = review.ask_reviews(None, [("k", RECORD)], definition=definition)
        assert isinstance(answers["k"], sweep.Failure)


def final(drops_by_confidence, keeps_by_confidence):
    out = {}
    for verdict, by_confidence in (("drop", drops_by_confidence), ("keep", keeps_by_confidence)):
        for confidence, n in by_confidence.items():
            for i in range(n):
                out[f"{verdict}-{confidence}-{i}"] = {"verdict": verdict, "confidence": confidence}
    return out


class TestControl:
    def test_the_control_is_drawn_from_keeps_at_the_drops_mix(self):
        answers = final({"high": 30, "medium": 10}, {"high": 100, "medium": 100, "low": 100})
        control = review.draw_control(answers, size=20, seed=1)
        mix = collections.Counter(answers[key]["confidence"] for key in control)
        assert mix == {"high": 15, "medium": 5}, "a uniform draw would give about a third at each level"
        assert all(answers[key]["verdict"] == "keep" for key in control)

    def test_the_same_seed_draws_the_same_control(self):
        answers = final({"high": 5}, {"high": 50})
        assert review.draw_control(answers, size=5, seed=3) == review.draw_control(answers, size=5, seed=3)
        assert review.draw_control(answers, size=5, seed=3) != review.draw_control(answers, size=5, seed=4)

    def test_a_thin_stratum_gives_what_it_has(self, caplog):
        answers = final({"medium": 10}, {"medium": 2})
        with caplog.at_level("WARNING"):
            assert len(review.draw_control(answers, size=5, seed=1)) == 2
        assert "only 2 there" in caplog.text

    def test_no_drops_no_control(self):
        assert review.draw_control(final({}, {"high": 5}), size=5, seed=1) == []


class TestRescues:
    def test_a_drop_with_a_case_goes_to_review_and_says_why(self):
        answers = {"a": {"verdict": "drop"}, "b": {"verdict": "drop"}, "c": {"verdict": "keep"}}
        reviews = {"a": {"belongs": True, "case": "it is about ravens"}, "b": {"belongs": False, "case": "no"},
                   "c": {"belongs": True, "case": "a control's case changes nothing"}}
        after = review.apply_reviews(answers, reviews)
        assert after["a"] == {"verdict": "review", "rescued": True, "case": "it is about ravens"}
        assert after["b"] == {"verdict": "drop"} and after["c"] == {"verdict": "keep"}

    def test_an_unreviewed_or_unanswered_drop_stays_a_drop(self):
        answers = {"a": {"verdict": "drop"}, "b": {"verdict": "drop"}}
        after = review.apply_reviews(answers, {"b": {"belongs": None, "case": ""}})
        assert after["a"]["verdict"] == after["b"]["verdict"] == "drop"

"""Tests for `raven.papers.scopebib.judge`: the prompt, reading the answers, the verdict, and the two passes."""

import functools
import json
import types

import pytest

from raven.librarian import agent, sweep
from raven.papers.scopebib import definition as scope_definition
from raven.papers.scopebib import judge
from raven.papers.scopebib import records as scope_records

DEFINITION_TEXT = """
scope = "studies of ravens"
hints = "Ravens and crows are easy to confuse."

[[test]]
name = "not_birds"
asks = "true ONLY if the work is positively not about birds"

[[test]]
name = "wrong_bird"
asks = "true ONLY if the work is positively about a bird other than the raven"
"""


@pytest.fixture
def definition(tmp_path):
    path = tmp_path / "ravens.scope.toml"
    path.write_text(DEFINITION_TEXT, encoding="utf-8")
    return scope_definition.load_definition(path)


def record(key, title="A raven study", abstract="", venue=""):
    return scope_records.Record(key=key, title=title, abstract=abstract, venue=venue, link="")


def answer(not_birds=False, wrong_bird=False, confidence="high", source="title", **extra):
    return {"tests": {"not_birds": not_birds, "wrong_bird": wrong_bird}, "confidence": confidence,
            "why": "because", "source": source, **extra}


INFORMATIVE = scope_records.TitleMeasure(content=5, informative=6, uninformative=False)
THIN = scope_records.TitleMeasure(content=1, informative=2, uninformative=True)


class _Turn:
    """Stands in for an `agent.TurnRecord`: the reply, and whether the token cap cut it off."""

    def __init__(self, reply, truncated=False):
        self.reply = reply
        self.truncated = truncated
        self.reasoning = ()

    def to_dict(self):
        return {"reply": self.reply, "truncated": self.truncated}


def script(monkeypatch, replies):
    """Make `agent.ask_record` answer with `replies` in turn, and keep the prompts it was asked."""
    prompts = []

    def ask_record(llm_settings, prompt, maybe_abort=None, **kw):
        prompts.append(prompt)
        reply = replies.pop(0)
        return reply if isinstance(reply, _Turn) else _Turn(reply)
    monkeypatch.setattr(agent, "ask_record", ask_record)
    return prompts


class TestThePrompt:
    def test_the_frame_names_the_scope_and_the_tests_and_hints_follow(self, definition):
        text = judge.rubric(definition)
        assert "a literature search for: studies of ravens." in text
        assert '"not_birds"  true ONLY if the work is positively not about birds' in text
        assert text.index("The tests:") < text.index("About this search in particular:")
        assert text.endswith("Ravens and crows are easy to confuse.")

    def test_a_definition_s_frame_replaces_the_tool_s(self, definition):
        import dataclasses
        own = dataclasses.replace(definition, frame="Judge these birds.")
        text = judge.rubric(own)
        assert text.startswith("Judge these birds.") and "literature search" not in text

    def test_the_scope_notes_follow_the_scope_and_precede_the_tests(self, definition):
        import dataclasses
        text = judge.rubric(dataclasses.replace(definition, scope_notes="Crows are out."))
        assert text.index("a literature search for: studies of ravens.") < text.index("Crows are out.") < text.index("The tests:")
        assert scope_definition.SCOPE_NOTES_HEADING not in judge.rubric(definition)

    def test_no_hints_no_section(self, definition):
        import dataclasses
        assert "About this search" not in judge.rubric(dataclasses.replace(definition, hints=""))

    def test_the_title_pass_asks_every_test_and_shows_the_venue(self, definition, monkeypatch):
        prompts = script(monkeypatch, ["[]"])
        judge.ask_titles(None, [("a", record("a", venue="Corvid Quarterly")), ("b", record("b"))], definition=definition)
        assert '"not_birds"  true or false, as above' in prompts[0]
        assert '"wrong_bird"  true or false, as above' in prompts[0]
        assert "0. A raven study\n   [published in: Corvid Quarterly]\n1. A raven study" in prompts[0]

    def test_the_abstract_pass_marks_a_teaser_and_says_when_there_is_none(self, definition, monkeypatch):
        prompts = script(monkeypatch, ["[]"])
        judge.ask_abstracts(None, [("a", record("a", abstract="It begins ...")), ("b", record("b"))], definition=definition)
        assert "Abstract (TRUNCATED): It begins ..." in prompts[0]
        assert "Abstract: (none)" in prompts[0]

    def test_a_one_record_batch_answered_with_a_bare_object_is_read(self, definition, monkeypatch):
        # The shape a lone record's answer once came back in: the answer itself, not an array of one.
        script(monkeypatch, ['{"i": 0, "not_birds": true, "wrong_bird": false, "confidence": "high", "why": "x"}'])
        answers, _ = judge.ask_abstracts(None, [("a", record("a", abstract="Lizards."))], definition=definition)
        assert set(answers) == {"a"} and answers["a"]["tests"]["not_birds"] is True

    def test_the_instrument_follows_the_tests_but_not_their_wrapping(self, tmp_path, definition):
        def material(text):
            path = tmp_path / "other.scope.toml"
            path.write_text(text, encoding="utf-8")
            return judge.instrument_material(scope_definition.load_definition(path))
        assert material(DEFINITION_TEXT) == judge.instrument_material(definition)
        rewrapped = DEFINITION_TEXT.replace('"Ravens and crows are easy to confuse."',
                                            '"""\nRavens and crows\nare easy to confuse.\n"""')
        assert material(rewrapped) == judge.instrument_material(definition)
        assert material(DEFINITION_TEXT.replace("other than the raven", "other than a corvid")) != \
            judge.instrument_material(definition)


class TestReadingTheReply:
    def test_answers_are_read_back_by_number(self, definition, monkeypatch):
        script(monkeypatch, [json.dumps([{"i": 1, "not_birds": True, "wrong_bird": False, "confidence": "high", "why": "fish"},
                                         {"i": 0, "not_birds": False, "wrong_bird": False, "confidence": "medium", "why": "ok"},
                                         {"i": 7, "not_birds": True, "wrong_bird": True, "confidence": "high", "why": "no such item"}])])
        answers, turns = judge.ask_titles(None, [("a", record("a")), ("b", record("b"))], definition=definition)
        assert set(answers) == {"a", "b"}, "an answer whose number does not resolve is dropped, not guessed at"
        assert answers["b"]["tests"] == {"not_birds": True, "wrong_bird": False}
        assert answers["a"]["confidence"] == "medium" and answers["a"]["source"] == "title"
        assert len(turns) == 1

    def test_a_reply_cut_off_at_the_cap_fails_the_whole_batch(self, definition, monkeypatch):
        script(monkeypatch, [_Turn('[{"i": 0, "not_birds": false', truncated=True)])
        answers, turns = judge.ask_titles(None, [("a", record("a")), ("b", record("b"))], definition=definition)
        assert all(isinstance(value, sweep.Failure) for value in answers.values()) and set(answers) == {"a", "b"}
        assert len(turns) == 1, "the turn is kept, to read what came back"

    def test_a_reply_that_does_not_parse_answers_nothing_and_keeps_its_turn(self, definition, monkeypatch):
        script(monkeypatch, ["I think they are all about ravens."])
        answers, turns = judge.ask_titles(None, [("a", record("a"))], definition=definition)
        assert answers == {} and len(turns) == 1

    def test_normalize(self, definition):
        out = judge.normalize({"not_birds": "yes", "wrong_bird": False, "confidence": "Very", "why": " a\n b "},
                              definition, "abstract")
        assert out == {"tests": {"not_birds": None, "wrong_bird": False}, "confidence": "low", "why": "a b",
                       "source": "abstract", "truncated": False}


class TestTruncation:
    def test_a_visible_teaser_lowers_the_confidence(self):
        tempered = judge.temper_for_truncation(answer(not_birds=True, source="abstract", truncated=False),
                                               record("a", abstract="It begins ..."))
        assert tempered["confidence"] == "low" and tempered["why"].startswith("abstract is a truncated preview")
        assert judge.verdict_of(tempered) == "review", "a drop resting on a teaser goes to a person"

    def test_the_model_s_own_report_is_enough(self):
        tempered = judge.temper_for_truncation(answer(source="abstract", truncated=True), record("a", abstract="Whole."))
        assert tempered["confidence"] == "low"

    def test_a_complete_abstract_is_left_alone(self):
        original = answer(not_birds=True, source="abstract", truncated=False)
        assert judge.temper_for_truncation(original, record("a", abstract="Whole.")) == original


class TestVerdict:
    @pytest.mark.parametrize("given, expected", [
        (answer(not_birds=True, confidence="high"), "drop"),
        (answer(not_birds=True, confidence="medium", source="abstract"), "drop"),
        (answer(not_birds=True, confidence="medium", source="title"), "review"),
        (answer(not_birds=True, confidence="low", source="abstract"), "review"),
        (answer(wrong_bird=True, confidence="low"), "review"),
        (answer(not_birds=None, confidence="high"), "review"),
        (answer(confidence="low"), "keep"),
        (answer(confidence="high"), "keep"),
    ])
    def test_a_true_drops_only_when_confident_enough(self, given, expected):
        assert judge.verdict_of(given) == expected

    @pytest.mark.parametrize("given, measure, escalates", [
        (answer(confidence="high"), INFORMATIVE, False),  # the control: nothing to look at again
        (answer(confidence="low"), INFORMATIVE, True),
        (answer(confidence="high"), THIN, True),
        (answer(not_birds=True, confidence="high"), INFORMATIVE, True),
        (answer(wrong_bird=True, confidence="medium"), INFORMATIVE, True),
    ])
    def test_escalation(self, given, measure, escalates):
        assert judge.needs_escalation(given, measure) is escalates


class TestThePasses:
    """Both passes on the real sweep, against a scripted model."""

    def test_a_run_settles_every_record_and_a_rerun_asks_nothing(self, llm_settings, definition, tmp_path, monkeypatch):
        recs = [record("keep_me", "Ravens in winter"), record("drop_me", "Salmon farming"),
                record("thin", "Notes"), record("teaser", "Corvids?", abstract="We study ...")]
        measures = {"keep_me": INFORMATIVE, "drop_me": INFORMATIVE, "thin": THIN, "teaser": INFORMATIVE}
        titles_reply = json.dumps([
            {"i": 0, "not_birds": False, "wrong_bird": False, "confidence": "high", "why": "ravens"},
            {"i": 1, "not_birds": True, "wrong_bird": False, "confidence": "high", "why": "fish"},
            {"i": 2, "not_birds": False, "wrong_bird": False, "confidence": "high", "why": "?"},
            {"i": 3, "not_birds": False, "wrong_bird": False, "confidence": "low", "why": "unclear"}])
        abstracts_reply = json.dumps([
            {"i": 0, "not_birds": True, "wrong_bird": False, "confidence": "high", "why": "fish farm", "truncated": False},
            {"i": 1, "not_birds": False, "wrong_bird": False, "confidence": "high", "why": "ravens", "truncated": False},
            {"i": 2, "not_birds": True, "wrong_bird": False, "confidence": "high", "why": "no birds", "truncated": False}])
        script(monkeypatch, [titles_reply, abstracts_reply])
        titles, abstracts = judge.ledgers(tmp_path, "test0000")

        def run(ask, todo, ledger, digest):
            return sweep.Sweep(llm_settings, functools.partial(ask, definition=definition),
                               [(r.key, r) for r in todo], ledger, batch_size=10, digest=digest).run()

        assert run(judge.ask_titles, judge.todo_titles(recs, titles=titles), titles,
                   judge.title_digest).outcome is sweep.completed
        escalated = judge.todo_abstracts(recs, titles=titles, abstracts=abstracts, measures=measures)
        assert [r.key for r in escalated] == ["drop_me", "thin", "teaser"], "every true, a thin title, low confidence"
        assert run(judge.ask_abstracts, escalated, abstracts, judge.abstract_digest).outcome is sweep.completed

        final = judge.final_answers(recs, titles=titles, abstracts=abstracts, measures=measures)
        assert {key: a["verdict"] for key, a in final.items()} == {
            "keep_me": "keep", "drop_me": "drop", "thin": "keep", "teaser": "review"}
        assert final["drop_me"]["source"] == "abstract" and final["keep_me"]["source"] == "title"
        assert final["teaser"]["confidence"] == "low", "the teaser's confident drop was tempered"

        assert judge.todo_titles(recs, titles=titles) == [] and judge.todo_abstracts(recs, titles=titles, abstracts=abstracts, measures=measures) == []

    def test_a_record_waiting_on_the_abstract_pass_is_not_yet_settled(self, tmp_path):
        titles, abstracts = judge.ledgers(tmp_path, "test0000")
        recs = [record("a"), record("b")]
        titles.write_answer("a", answer(not_birds=True), maybe_digest=judge.title_digest(recs[0]))
        titles.write_answer("b", answer(), maybe_digest=judge.title_digest(recs[1]))
        final = judge.final_answers(recs, titles=titles, abstracts=abstracts, measures={"a": INFORMATIVE, "b": INFORMATIVE})
        assert set(final) == {"b"}
        assert types.SimpleNamespace(**final["b"]).verdict == "keep"

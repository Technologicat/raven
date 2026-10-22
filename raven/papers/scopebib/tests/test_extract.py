"""Tests for `raven.papers.scopebib.extract` and `raven.papers.scopebib.rules`: the fields, and acting on them."""

import json

import pytest

from raven.librarian import agent, sweep
from raven.papers.scopebib import definition as scope_definition
from raven.papers.scopebib import extract, rules
from raven.papers.scopebib import records as scope_records

DEFINITION = """
scope = "studies of ravens"

[[test]]
name = "not_birds"
asks = "true if the work is positively not about birds"

[[field]]
name = "setting"
asks = "where the birds were studied"
quote = ["city", "forest"]
[field.values]
city = "an urban area"
forest = "woodland"
nowhere = "no birds were observed at all"

[[field]]
name = "observed"
asks = "whether live birds were observed"
[field.values]
yes = "live birds"
no = "no live birds"

[[field]]
name = "method"
asks = "how the birds were studied"
max_words = 3

[[rule]]
label = "urban, stated in the text"
when = { setting = ["city"] }
action = "remove"

[[rule]]
label = "no birds at all, and none observed"
when = { setting = ["nowhere"], observed = ["no"] }
action = "remove"

[[rule]]
label = "no birds at all"
when = { setting = ["nowhere"] }
action = "hold"
"""

RECORD = scope_records.Record(key="k", title="Ravens of the city park", abstract="We tagged birds in a park.",
                              venue="Corvid Quarterly", link="")


@pytest.fixture
def definition(tmp_path):
    path = tmp_path / "ravens.scope.toml"
    path.write_text(DEFINITION, encoding="utf-8")
    return scope_definition.load_definition(path)


def extraction(definition, **answer):
    return extract.normalize(answer, definition, RECORD)


class TestNormalize:
    def test_a_value_is_matched_without_regard_to_case(self, definition):
        assert extraction(definition, setting="CITY")["fields"]["setting"] == "city"

    def test_a_value_outside_the_vocabulary_is_not_stated_and_kept_raw(self, definition):
        out = extraction(definition, setting="suburb")
        assert out["fields"]["setting"] == scope_definition.NOT_STATED
        assert out["raw"]["setting"] == "suburb"

    def test_free_text_is_cut_to_its_cap(self, definition):
        assert extraction(definition, method="leg bands and radio tags")["fields"]["method"] == "leg bands and"

    @pytest.mark.parametrize("quote, found", [
        ("city park", True),
        ('"City Park"', True),          # the model's own quotation marks, and another case
        ("“the old city park”", False),  # marks stripped; "the old city park" is not in it
        ("the downtown core", False),
        ("", False),
    ])
    def test_a_quote_is_checked_against_the_record(self, definition, quote, found):
        out = extraction(definition, setting="city", setting_quote=quote)
        assert out["quotes"]["setting"]["found"] is found
        assert out["quotes"]["setting"]["text"] == quote, "the quote is kept as given, for a reader"

    def test_only_a_field_with_values_to_quote_has_a_quote(self, definition):
        assert set(extraction(definition, setting="city")["quotes"]) == {"setting"}


class TestRules:
    @pytest.mark.parametrize("answer, expected", [
        ({"setting": "city", "setting_quote": "city park"}, ("remove", "urban, stated in the text")),
        ({"setting": "nowhere", "observed": "no"}, ("remove", "no birds at all, and none observed")),
        ({"setting": "nowhere", "observed": "yes"}, ("hold", "no birds at all")),
        ({"setting": "forest", "setting_quote": "park"}, ("keep", None)),
        ({}, ("keep", None)),
    ])
    def test_the_first_rule_that_matches_decides(self, definition, answer, expected):
        decision = rules.decide(extraction(definition, **answer), definition)
        assert (decision.action, decision.label) == expected and decision.note == ""

    def test_a_removal_whose_quote_is_not_in_the_record_is_held(self, definition):
        decision = rules.decide(extraction(definition, setting="city", setting_quote="downtown"), definition)
        assert decision.action == "hold" and decision.label == "urban, stated in the text"
        assert "quote for setting is not in the record" in decision.note

    def test_a_value_with_nothing_to_quote_is_removed_without_one(self, definition):
        # "nowhere" asserts an absence and is not in `quote`, so its empty quote is no reason to hold.
        decision = rules.decide(extraction(definition, setting="nowhere", observed="no"), definition)
        assert decision.action == "remove"


class TestAsking:
    def test_the_prompt_names_each_field_its_values_and_which_need_a_quote(self, definition, monkeypatch):
        prompts = []

        def ask_record(llm_settings, prompt, maybe_abort=None, **kw):
            prompts.append(prompt)
            return type("T", (), {"reply": "[]", "truncated": False})()
        monkeypatch.setattr(agent, "ask_record", ask_record)
        extract.ask_fields(None, [("k", RECORD)], definition=definition)
        assert "nowhere: no birds were observed at all" in prompts[0]
        assert '"setting_quote"' in prompts[0] and "Needed when the answer is one of: city, forest" in prompts[0]
        assert '"observed_quote"' not in prompts[0]
        assert "at most 3 words" in prompts[0]
        assert "Published in: Corvid Quarterly" in prompts[0]

    def test_answers_are_read_by_number_and_a_cut_off_reply_fails_the_batch(self, definition, monkeypatch):
        replies = [json.dumps([{"i": 0, "setting": "city", "setting_quote": "city", "observed": "yes", "method": "x"},
                               {"i": 5, "setting": "forest"}]), None]

        def ask_record(llm_settings, prompt, maybe_abort=None, **kw):
            reply = replies.pop(0)
            return type("T", (), {"reply": reply or "[{", "truncated": reply is None})()
        monkeypatch.setattr(agent, "ask_record", ask_record)
        answers, _ = extract.ask_fields(None, [("k", RECORD)], definition=definition)
        assert answers["k"]["fields"]["setting"] == "city" and set(answers) == {"k"}
        answers, _ = extract.ask_fields(None, [("k", RECORD)], definition=definition)
        assert isinstance(answers["k"], sweep.Failure)

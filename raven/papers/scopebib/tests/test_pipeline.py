"""Tests for `raven.papers.scopebib.pipeline`: what is left to ask, whether a backend is needed, and asking it."""

import collections
import json
import re

import pytest

from raven.librarian import agent
from raven.papers import bibtex
from raven.papers.scopebib import definition as scope_definition
from raven.papers.scopebib import pipeline, runfile
from raven.papers.scopebib import records as scope_records
from raven.papers.scopebib.runfile import StagePlan

DEFINITION = """
scope = "studies of ravens"

[[test]]
name = "not_birds"
asks = "true if the work is positively not about birds"

[[field]]
name = "setting"
asks = "where the birds were studied"
quote = ["city"]
[field.values]
city = "an urban area"
forest = "woodland"

[[rule]]
label = "urban"
when = { setting = ["city"] }
action = "remove"
"""

# Ravens are kept, crows are not birds as far as this scope goes, and the city ravens are removed by the rule.
TITLES = {"r1": "Ravens of the north", "r2": "Ravens in the city", "r3": "Ravens at dawn", "r4": "Raven nests",
          "r5": "Ravens and snow", "c1": "Crows at the dump", "c2": "Crows in parliament"}


class _Turn:
    def __init__(self, reply):
        self.reply, self.truncated = reply, False

    def to_dict(self):
        return {"reply": self.reply}


class ScriptedModel:
    """Answers each pass by what its prompt asks, and counts the calls by pass."""

    def __init__(self, *, skip_titles=()):
        self.calls = collections.Counter()
        self.skip_titles = set(skip_titles)

    def __call__(self, llm_settings, prompt, maybe_abort=None, **kwargs):
        if "You are recording what each record says" in prompt:
            self.calls["extract"] += 1
            titles = re.findall(r"--- record (\d+) ---\nTitle: (.*)", prompt)
            answers = [{"i": int(i), "setting": "city" if "city" in title else "forest",
                        "setting_quote": "the city" if "city" in title else ""} for i, title in titles]
        elif "credible case" in prompt:
            self.calls["review"] += 1
            titles = re.findall(r"--- record (\d+) ---\nTitle: (.*)", prompt)
            answers = [{"i": int(i), "belongs": False, "case": "nothing to point at"} for i, _ in titles]
        elif "--- item" in prompt:
            self.calls["abstracts"] += 1
            titles = re.findall(r"--- item (\d+) ---\nTitle: (.*)", prompt)
            answers = [{"i": int(i), "not_birds": "Crow" in title, "confidence": "high", "why": "w",
                        "truncated": False} for i, title in titles]
        else:
            self.calls["titles"] += 1
            titles = re.findall(r"^(\d+)\. (.*)$", prompt, re.MULTILINE)
            answers = [{"i": int(i), "not_birds": "Crow" in title, "confidence": "high", "why": "w"}
                       for i, title in titles if title not in self.skip_titles]
        return _Turn(json.dumps(answers))


@pytest.fixture
def definition(tmp_path):
    path = tmp_path / "ravens.scope.toml"
    path.write_text(DEFINITION, encoding="utf-8")
    return scope_definition.load_definition(path)


def bib(path, titles=TITLES, *, without_abstract=()):
    path.write_text("".join(f"@article{{{key},\n  title = {{{title}}},\n"
                            + ("" if key in without_abstract else f"  abstract = {{About {title}.}},\n") + "}\n\n"
                            for key, title in titles.items()), encoding="utf-8")
    return path


@pytest.fixture
def corpus(tmp_path):
    library = bibtex.parse_file(bib(tmp_path / "in.bib"), split_names=False)
    records = scope_records.records_of(library)
    measures = {record.key: scope_records.TitleMeasure(4, 4, False) for record in records}
    return pipeline.Corpus(library, records, None, measures)


@pytest.fixture
def model(monkeypatch):
    scripted = ScriptedModel()
    monkeypatch.setattr(agent, "ask_record", scripted)
    return scripted


@pytest.fixture
def connections(llm_settings):
    """A `connect` that counts how often it was called."""
    counter = collections.Counter()

    def connect():
        counter["connect"] += 1
        return llm_settings
    connect.counter = counter
    return connect


def run(corpus, definition, tmp_path, connect, *, stages=runfile.STAGES, plans=None, **kwargs):
    kwargs.setdefault("control_size", 2)
    kwargs.setdefault("seed", 42)
    return pipeline.run(corpus, definition, stages=stages, plans=plans or {}, ledger_dir=tmp_path / "ledgers",
                        connect=connect, **kwargs)


class TestARun:
    def test_every_stage_runs_in_order_and_the_answers_settle(self, corpus, definition, tmp_path, model, connections):
        result = run(corpus, definition, tmp_path, connections)
        state = result.state
        assert {key: answer["verdict"] for key, answer in state.final.items()} == {
            **{key: "keep" for key in TITLES if key.startswith("r")}, "c1": "drop", "c2": "drop"}
        assert set(state.reviews) == {"c1", "c2", *state.control} and len(state.control) == 2
        assert set(state.extractions) == {key for key in TITLES if key.startswith("r")}
        assert all(not items for items in state.todo.values())
        assert result.connected and connections.counter["connect"] == 1
        assert {stage: plan.maybe_model for stage, plan in result.stages.items()} == dict.fromkeys(runfile.STAGES,
                                                                                                    "test-model")
        assert result.models["judge"].startswith("test-model"), "described from the ledger's header"

    def test_a_run_again_asks_nothing(self, corpus, definition, tmp_path, model, connections):
        run(corpus, definition, tmp_path, connections)
        asked = sum(model.calls.values())
        run(corpus, definition, tmp_path, connections)
        assert sum(model.calls.values()) == asked

    def test_until_stops_after_the_stage_named(self, corpus, definition, tmp_path, model, connections):
        result = run(corpus, definition, tmp_path, connections, stages=("judge",))
        assert model.calls["review"] == model.calls["extract"] == 0
        assert result.state.todo["review"], "the reviewer's work is known once the judge is settled"


class TestConnecting:
    def test_a_replay_with_every_instrument_named_never_connects(self, corpus, definition, tmp_path, model,
                                                                 connections):
        first = run(corpus, definition, tmp_path, connections)
        assert connections.counter["connect"] == 1
        replay = run(corpus, definition, tmp_path, connections, plans=first.stages)
        assert connections.counter["connect"] == 1, "everything is answered under named instruments"
        assert not replay.connected and replay.state.final == first.state.final
        assert replay.stages == first.stages, "the models come from the ledgers' headers"

    def test_a_named_instrument_with_no_ledger_is_reported_with_where_it_was_looked_for(self, corpus, definition,
                                                                                        tmp_path, model, connections):
        plans = {stage: StagePlan(None, "deadbeef") for stage in runfile.STAGES}
        result = run(corpus, definition, tmp_path, connections, stages=(), plans=plans)
        assert not result.connected
        assert [message.split(":")[0] for message in result.messages] == list(runfile.STAGES)
        assert all(str(tmp_path / "ledgers") in message for message in result.messages)
        first = run(corpus, definition, tmp_path, connections)
        replay = run(corpus, definition, tmp_path, connections, stages=(), plans=first.stages)
        assert replay.messages == [], "the control: ledgers that exist are not reported"

    def test_an_unnamed_instrument_connects_even_with_nothing_to_ask(self, corpus, definition, tmp_path, model,
                                                                     connections):
        run(corpus, definition, tmp_path, connections)
        run(corpus, definition, tmp_path, connections, stages=())
        assert connections.counter["connect"] == 2


class TestRefusing:
    def test_a_named_instrument_that_is_not_todays_is_read_only(self, corpus, definition, tmp_path, model,
                                                                connections):
        with pytest.raises(pipeline.PipelineError, match="instrument deadbeef is not today's.*read-only"):
            run(corpus, definition, tmp_path, connections, plans={"judge": StagePlan(None, "deadbeef")})
        assert not model.calls, "refused before asking anything"

    def test_a_named_instrument_that_is_todays_may_finish_its_work(self, corpus, definition, tmp_path, model,
                                                                   connections, llm_settings):
        live = pipeline.materials(definition)["judge"]
        from raven.librarian import sweep
        plans = {"judge": StagePlan(None, sweep.fingerprint(llm_settings, live))}
        run(corpus, definition, tmp_path, connections, stages=("judge",), plans=plans)
        assert model.calls["titles"] > 0

    def test_a_stage_wanting_another_model_is_refused_and_the_earlier_ones_offered(self, corpus, definition,
                                                                                   tmp_path, model, connections):
        with pytest.raises(pipeline.PipelineError, match="wants the model 'other'.*--until judge"):
            run(corpus, definition, tmp_path, connections, plans={"review": StagePlan("other", "abcd1234")})
        assert not model.calls

    def test_an_unfindable_stage_outside_the_run_is_reported_rather_than_refused(self, corpus, definition,
                                                                                 tmp_path, model, connections):
        result = run(corpus, definition, tmp_path, connections, stages=("judge",),
                     plans={"extract": StagePlan("other")})
        assert any("extract's answers are not in the reports" in message for message in result.messages)
        with pytest.raises(pipeline.PipelineError, match="cannot find the answers of extract"):
            run(corpus, definition, tmp_path, connections, plans={"extract": StagePlan("other")})


class TestWaiting:
    def test_the_reviewer_and_extractor_wait_for_the_judge_to_settle(self, corpus, definition, tmp_path,
                                                                     monkeypatch, connections):
        scripted = ScriptedModel(skip_titles={TITLES["r3"]})
        monkeypatch.setattr(agent, "ask_record", scripted)
        result = run(corpus, definition, tmp_path, connections)
        assert scripted.calls["review"] == scripted.calls["extract"] == 0
        assert result.state.control == []
        assert any("wait until the judge" in message for message in result.messages)
        assert any("1 records failed" in message for message in result.messages)


class TestReplicates:
    def test_the_first_replicate_is_the_plain_material(self, definition):
        assert pipeline.materials(definition, replicate=1) == pipeline.materials(definition)
        assert pipeline.materials(definition, replicate=2)["judge"] != pipeline.materials(definition)["judge"]

    def test_a_second_replicate_asks_again_into_ledgers_of_its_own(self, corpus, definition, tmp_path, model,
                                                                   connections):
        first = run(corpus, definition, tmp_path, connections)
        asked = model.calls["titles"]
        second = run(corpus, definition, tmp_path, connections, replicate=2)
        assert model.calls["titles"] == 2 * asked
        assert first.stages["judge"].maybe_instrument != second.stages["judge"].maybe_instrument
        again = run(corpus, definition, tmp_path, connections, plans=first.stages)
        assert again.state.final == first.state.final, "the first replicate's ledgers are untouched"


class _Token:
    def __init__(self, text):
        self.text, self.lemma_, self.is_alpha = text, text.lower(), text.isalpha()


class _NLP:
    def analyze(self, texts):
        return [[_Token(word) for word in text.split()] for text in texts]


# Given rather than defaulted: the default lists come with the ML stack, which CI does not install.
STOPWORDS = frozenset({"of", "the", "in", "at", "and"})


class TestLoadingTheCorpus:
    def test_the_sift_runs_first_and_is_accounted_for(self, tmp_path):
        path = bib(tmp_path / "in.bib", without_abstract={"r1"})
        corpus = pipeline.load_corpus(path, nlp=_NLP(), require=["abstract"], stopwords=STOPWORDS)
        assert "r1" not in {record.key for record in corpus.records}
        assert [record.key for record in corpus.maybe_sift.dropped] == ["r1"]
        assert set(corpus.measures) == {record.key for record in corpus.records}

    def test_no_criteria_no_sift(self, tmp_path):
        assert pipeline.load_corpus(bib(tmp_path / "in.bib"), nlp=_NLP(), stopwords=STOPWORDS).maybe_sift is None

    def test_a_sample_is_seeded_and_ignores_the_files_order(self, tmp_path):
        forward = bib(tmp_path / "a.bib")
        backward = bib(tmp_path / "b.bib", dict(reversed(TITLES.items())))

        def drawn(path, seed):
            corpus = pipeline.load_corpus(path, nlp=_NLP(), maybe_sample=3, seed=seed, stopwords=STOPWORDS)
            return {record.key for record in corpus.records}
        assert drawn(forward, 1) == drawn(backward, 1) and len(drawn(forward, 1)) == 3
        assert any(drawn(forward, seed) != drawn(forward, 1) for seed in range(2, 6)), \
            "the control: another seed draws another sample"

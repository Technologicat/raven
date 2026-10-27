"""Tests for `raven.papers.scopebib.pipeline`: what is left to ask, whether a backend is needed, and asking it."""

import collections
import json
import re

import pytest

from raven.librarian import agent
from raven.papers import bibtex, fillbib
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
    """Answers each pass by what its prompt asks, and counts the calls by pass, and the titles each saw."""

    def __init__(self, *, skip_titles=()):
        self.calls = collections.Counter()
        self.seen = collections.defaultdict(list)
        self.skip_titles = set(skip_titles)

    def __call__(self, llm_settings, prompt, maybe_abort=None, **kwargs):
        if "You are recording what each record says" in prompt:
            self.calls["extract"] += 1
            titles = re.findall(r"--- record (\d+) ---\nTitle: (.*)", prompt)
            self.seen["extract"].extend(title for _, title in titles)
            answers = [{"i": int(i), "setting": "city" if "city" in title else "forest",
                        "setting_quote": "the city" if "city" in title else ""} for i, title in titles]
        elif "credible case" in prompt:
            self.calls["review"] += 1
            titles = re.findall(r"--- record (\d+) ---\nTitle: (.*)", prompt)
            self.seen["review"].extend(title for _, title in titles)
            answers = [{"i": int(i), "belongs": False, "case": "nothing to point at"} for i, _ in titles]
        elif "--- item" in prompt:
            self.calls["abstracts"] += 1
            titles = re.findall(r"--- item (\d+) ---\nTitle: (.*)", prompt)
            self.seen["abstracts"].extend(title for _, title in titles)
            answers = [{"i": int(i), "not_birds": "Crow" in title, "confidence": "high", "why": "w",
                        "truncated": False} for i, title in titles]
        else:
            self.calls["titles"] += 1
            titles = re.findall(r"^(\d+)\. (.*)$", prompt, re.MULTILINE)
            self.seen["titles"].extend(title for _, title in titles)
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

    def test_a_record_edited_between_runs_is_asked_again_and_only_by_the_passes_that_show_it(
            self, corpus, definition, tmp_path, model, connections):
        run(corpus, definition, tmp_path, connections)
        edited = dict(TITLES, c1="Crows at the landfill")  # a title edited: every pass shows the title
        library = bibtex.parse_file(bib(tmp_path / "edited.bib", edited), split_names=False)
        records = [scope_records.Record(**{**record.__dict__, "abstract": "About ravens, at length."})
                   if record.key == "r3" else record  # an abstract edited: only the extractor shows it to a keep
                   for record in scope_records.records_of(library)]
        model.seen.clear()
        run(pipeline.Corpus(library, records, None, corpus.measures), definition, tmp_path, connections)
        assert model.seen["titles"] == ["Crows at the landfill"]
        assert model.seen["review"] == ["Crows at the landfill"], "a drop is reviewed"
        assert model.seen["extract"] == ["Ravens at dawn"]

    @staticmethod
    def _forget_digests(ledger_dir):
        """Make the ledgers look written before answers carried a digest."""
        for path in ledger_dir.glob("*.jsonl"):
            lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            path.write_text("".join(json.dumps({k: v for k, v in line.items() if k != "d"}) + "\n" for line in lines),
                            encoding="utf-8")

    def test_old_ledgers_stamped_from_their_corpus_ask_nothing_again(self, corpus, definition, tmp_path, model,
                                                                     connections):
        run(corpus, definition, tmp_path, connections)
        self._forget_digests(tmp_path / "ledgers")
        stamped = pipeline.stamp_ledgers(corpus.records, tmp_path / "ledgers")
        assert sum(stamped.values()) > 0 and all(not name.endswith("-traces.jsonl") for name in stamped)
        asked = sum(model.calls.values())
        run(corpus, definition, tmp_path, connections)
        assert sum(model.calls.values()) == asked

    def test_old_ledgers_unstamped_are_asked_again(self, corpus, definition, tmp_path, model, connections):
        # The control for the test above: without the stamp, the same ledgers answer nothing.
        run(corpus, definition, tmp_path, connections)
        self._forget_digests(tmp_path / "ledgers")
        model.seen.clear()
        run(corpus, definition, tmp_path, connections)
        assert sorted(model.seen["titles"]) == sorted(TITLES.values())

    def test_a_record_edited_after_the_stamped_run_is_still_asked_again(self, corpus, definition, tmp_path, model,
                                                                        connections):
        run(corpus, definition, tmp_path, connections)
        self._forget_digests(tmp_path / "ledgers")
        pipeline.stamp_ledgers(corpus.records, tmp_path / "ledgers")
        edited = dict(TITLES, c1="Crows at the landfill")
        library = bibtex.parse_file(bib(tmp_path / "edited.bib", edited), split_names=False)
        model.seen.clear()
        run(pipeline.Corpus(library, scope_records.records_of(library), None, corpus.measures), definition, tmp_path,
            connections)
        assert model.seen["titles"] == ["Crows at the landfill"]

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
    def test_a_named_instrument_that_is_not_the_current_one_is_read_only(self, corpus, definition, tmp_path,
                                                                          model, connections):
        with pytest.raises(pipeline.PipelineError, match="instrument deadbeef is not the one the current definition and model make.*read-only"):
            run(corpus, definition, tmp_path, connections, plans={"judge": StagePlan(None, "deadbeef")})
        assert not model.calls, "refused before asking anything"

    def test_a_named_instrument_that_is_the_current_one_may_finish_its_work(self, corpus, definition, tmp_path,
                                                                            model, connections, llm_settings):
        live = pipeline.materials(definition)["judge"]
        from raven.librarian import sweep
        plans = {"judge": StagePlan(None, sweep.fingerprint(llm_settings, live))}
        run(corpus, definition, tmp_path, connections, stages=("judge",), plans=plans)
        assert model.calls["titles"] > 0

    def test_a_stage_wanting_another_model_is_refused_and_the_earlier_ones_offered(self, corpus, definition,
                                                                                   tmp_path, model, connections):
        with pytest.raises(pipeline.PipelineError, match="wants the model 'other'.*cannot change by itself.*--until judge"):
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

    @staticmethod
    def _bib_with_dois(path):
        path.write_text("@article{bare,\n  title = {Ravens unexplained},\n  doi = {10.1234/bare}\n}\n\n"
                        "@article{whole,\n  title = {Ravens explained},\n  doi = {10.1234/whole},\n"
                        "  abstract = {We explain ravens.}\n}\n", encoding="utf-8")
        return path

    def test_the_fill_runs_before_the_sift(self, tmp_path, monkeypatch):
        asked = []

        def fetch(dois, *, maybe_cache_path=None, **kwargs):
            asked.append((list(dois), maybe_cache_path))
            return {"10.1234/bare": "Ravens are explained at last.", "10.1234/whole": "We explain ravens."}, {}
        monkeypatch.setattr(fillbib, "fetch_crossref", fetch)
        path = self._bib_with_dois(tmp_path / "in.bib")
        corpus = pipeline.load_corpus(path, nlp=_NLP(), require=["abstract"], stopwords=STOPWORDS, fill=True,
                                      maybe_crossref_cache=tmp_path / "crossref.jsonl")
        assert {record.key: record.abstract for record in corpus.records}["bare"] == "Ravens are explained at last."
        assert [(row.key, row.outcome) for row in corpus.maybe_fill] == [("bare", fillbib.FILLED)]
        assert corpus.maybe_sift.dropped == [], "filled first, the bare record has an abstract when the sift asks"
        assert asked == [(["10.1234/bare", "10.1234/whole"], tmp_path / "crossref.jsonl")]

    def test_without_fill_crossref_is_not_asked(self, tmp_path, monkeypatch):
        # The control for the test above: the same corpus, and the bare record goes to the sift.
        monkeypatch.setattr(fillbib, "fetch_crossref", lambda *args, **kwargs: pytest.fail("Crossref was asked"))
        corpus = pipeline.load_corpus(self._bib_with_dois(tmp_path / "in.bib"), nlp=_NLP(), require=["abstract"],
                                      stopwords=STOPWORDS)
        assert corpus.maybe_fill is None
        assert [record.key for record in corpus.maybe_sift.dropped] == ["bare"]

    def test_a_failed_lookup_stops_the_run_and_names_the_doi(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fillbib, "fetch_crossref",
                            lambda dois, **kwargs: ({"10.1234/whole": None}, {"10.1234/bare": "ConnectionError: x"}))
        with pytest.raises(pipeline.PipelineError, match=r"1 DOI \(10\.1234/bare\).*ConnectionError"):
            pipeline.load_corpus(self._bib_with_dois(tmp_path / "in.bib"), nlp=_NLP(), stopwords=STOPWORDS, fill=True)

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


class FakeSwitcher:
    """Stands in for LM Studio: records each load, and makes the backend report the model just loaded."""

    def __init__(self, llm_settings, monkeypatch, *, available=("test-model", "other-model")):
        from raven.librarian import llmclient
        from unpythonic.env import env
        self.base, self.loads, self.available_models = llm_settings, [], set(available)
        self.current = llm_settings.model_record
        monkeypatch.setattr(llmclient, "resolve_model_info", lambda backend_url, flavor: self.current)
        self._env = env

    def available(self):
        return self.available_models

    def load(self, model):
        self.loads.append(model)
        record = self._env(**{**dict(self.base.model_record.items()), "model_id": model, "label": model})
        self.current = record
        settings = self._env(**dict(self.base.items()))
        settings.model_record = record
        return settings


class TestSwitching:
    def test_a_stage_wanting_another_model_runs_with_it_and_the_first_comes_back(
            self, corpus, definition, tmp_path, model, connections, llm_settings, monkeypatch):
        switcher = FakeSwitcher(llm_settings, monkeypatch)
        recorded = []
        result = run(corpus, definition, tmp_path, connections, plans={"extract": StagePlan("other-model")},
                     make_switcher=lambda settings: switcher, on_instruments=recorded.append)
        assert switcher.loads == ["other-model", "test-model"], "loaded for the extractor, then put back"
        assert model.calls["extract"] > 0
        assert result.stages["extract"].maybe_model == "other-model"
        assert result.stages["extract"].maybe_instrument not in (None, result.stages["judge"].maybe_instrument)
        assert recorded[0]["extract"].maybe_instrument is None, "not known until its model is loaded"
        assert recorded[-1]["extract"].maybe_instrument == result.stages["extract"].maybe_instrument

    def test_stages_sharing_a_model_switched_to_each_learn_their_instrument(
            self, corpus, definition, tmp_path, model, connections, llm_settings, monkeypatch):
        # The judge's switch loads the model the review and the extractor want too, so neither switches again:
        # each must still learn its own instrument under it, or it asks into a ledger nobody can find.
        switcher = FakeSwitcher(llm_settings, monkeypatch)
        plans = {stage: StagePlan("other-model") for stage in runfile.STAGES}
        result = run(corpus, definition, tmp_path, connections, plans=plans, make_switcher=lambda settings: switcher)
        assert switcher.loads == ["other-model", "test-model"], "loaded once, for all three, then put back"
        assert model.calls["review"] > 0 and model.calls["extract"] > 0
        instruments = [result.stages[stage].maybe_instrument for stage in runfile.STAGES]
        assert None not in instruments and len(set(instruments)) == 3
        assert not list((tmp_path / "ledgers").glob("*-None*.jsonl"))

    def test_a_model_the_backend_lacks_is_refused_before_any_work(self, corpus, definition, tmp_path, model,
                                                                  connections, llm_settings, monkeypatch):
        switcher = FakeSwitcher(llm_settings, monkeypatch)
        with pytest.raises(pipeline.PipelineError, match="no model 'missing-model'"):
            run(corpus, definition, tmp_path, connections, plans={"review": StagePlan("missing-model")},
                make_switcher=lambda settings: switcher)
        assert not model.calls and switcher.loads == []

    def test_a_stage_with_nothing_to_ask_loads_nothing(self, corpus, definition, tmp_path, model, connections,
                                                       llm_settings, monkeypatch):
        switcher = FakeSwitcher(llm_settings, monkeypatch)
        run(corpus, definition, tmp_path, connections, stages=("judge",), plans={"extract": StagePlan("other-model")},
            make_switcher=lambda settings: switcher)
        assert switcher.loads == [], "the extractor is not run, so its model is not loaded"

    def test_the_first_model_comes_back_even_when_a_stage_is_interrupted(self, corpus, definition, tmp_path, model,
                                                                         connections, llm_settings, monkeypatch):
        switcher = FakeSwitcher(llm_settings, monkeypatch)

        def interrupted(llm_settings, prompt, maybe_abort=None, **kwargs):
            if "You are recording what each record says" in prompt:
                raise KeyboardInterrupt
            return model(llm_settings, prompt, maybe_abort, **kwargs)
        monkeypatch.setattr(agent, "ask_record", interrupted)
        with pytest.raises(KeyboardInterrupt):
            run(corpus, definition, tmp_path, connections, plans={"extract": StagePlan("other-model")},
                make_switcher=lambda settings: switcher)
        assert switcher.loads == ["other-model", "test-model"]

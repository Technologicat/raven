"""Tests for `raven.papers.scopebib.cli`: resolving a command line, and a run from start to `finish`."""

import collections
import json
import re

import pytest

from raven.common import tabular
from raven.librarian import agent
from raven.papers.scopebib import cli, report, runfile
from raven.papers.scopebib import records as scope_records
from raven.papers.scopebib.tests.test_pipeline import DEFINITION, STOPWORDS, ScriptedModel, _NLP, bib


def parse(*argv):
    return cli._parser().parse_args(list(argv))


@pytest.fixture
def files(tmp_path):
    definition = tmp_path / "ravens.scope.toml"
    definition.write_text(DEFINITION, encoding="utf-8")
    return bib(tmp_path / "corpus.bib"), definition


class TestResolving:
    def test_a_first_run_from_the_command_line(self, files, tmp_path):
        corpus, definition = files
        plan = cli.resolve(parse("run", str(corpus), "--definition", str(definition), "--out-dir",
                                 str(tmp_path / "out"), "--until", "review"))
        assert plan.stages == ("judge", "review")
        assert (plan.reports_dir, plan.ledger_dir) == (tmp_path / "out", tmp_path / "out" / "ledgers")
        settings = plan.run_file.settings
        assert (settings.maybe_seed, settings.maybe_control, settings.maybe_replicate, settings.maybe_format) == \
               (42, 40, 1, "xlsx")
        assert settings.maybe_ledgers == tmp_path / "out" / "ledgers", "recorded, so a run file can name it"

    def test_a_run_file_supplies_the_rest_and_the_command_line_wins(self, files, tmp_path):
        corpus, definition = files
        path = tmp_path / "out" / "run.toml"
        runfile.write_run_file(path, runfile.RunFile(runfile.Settings(maybe_bib=corpus, maybe_definition=definition,
                                                                      maybe_seed=7, maybe_control=10), {}))
        plan = cli.resolve(parse("report", "--run-file", str(path), "--seed", "8", "--model", "judge=big"))
        assert plan.reports_dir == path.parent, "the run file's directory, so passing it back writes in place"
        assert (plan.run_file.settings.maybe_seed, plan.run_file.settings.maybe_control) == (8, 10)
        assert plan.run_file.stages["judge"].maybe_model == "big"
        assert plan.stages == ()

    def test_a_pilot_writes_its_reports_apart_and_shares_the_ledgers(self, files, tmp_path):
        corpus, definition = files
        plan = cli.resolve(parse("run", str(corpus), "--definition", str(definition), "--out-dir",
                                 str(tmp_path / "out"), "--pilot"))
        assert plan.reports_dir == tmp_path / "out" / "pilot" and plan.ledger_dir == tmp_path / "out" / "ledgers"
        assert plan.run_file.settings.maybe_pilot == 50

    def test_a_sample_on_the_command_line_replaces_a_recorded_pilot(self, files, tmp_path):
        corpus, definition = files
        path = tmp_path / "run.toml"
        runfile.write_run_file(path, runfile.RunFile(runfile.Settings(maybe_bib=corpus, maybe_definition=definition,
                                                                      maybe_pilot=20), {}))
        settings = cli.resolve(parse("run", "--run-file", str(path), "--sample", "5")).run_file.settings
        assert (settings.maybe_sample, settings.maybe_pilot) == (5, None)

    @pytest.mark.parametrize("argv, complaint", [
        (("run", "--definition", "d.toml", "--out-dir", "o"), "no .bib"),
        (("run", "c.bib", "--out-dir", "o"), "no definition file.*c.scope.toml"),
        (("run", "c.bib", "--definition", "d.toml"), "no output directory"),
        (("run", "c.bib", "--definition", "d.toml", "--out-dir", "o", "--sample", "0"), "at least 1"),
    ])
    def test_what_is_missing_is_named(self, argv, complaint):
        with pytest.raises(cli.UsageError, match=complaint):
            cli.resolve(parse(*argv))

    def test_the_definition_defaults_to_the_one_beside_the_bib(self, files, tmp_path):
        corpus, definition = files
        beside = corpus.with_name(f"{corpus.stem}.scope.toml")
        beside.write_text(definition.read_text(encoding="utf-8"), encoding="utf-8")
        plan = cli.resolve(parse("run", str(corpus), "--out-dir", str(tmp_path / "out")))
        assert plan.run_file.settings.maybe_definition == beside

    def test_finish_takes_the_sheet_and_not_a_bib(self):
        opts = parse("finish", "marked.xlsx", "--run-file", "run.toml")
        assert (opts.sheet, opts.bib) == ("marked.xlsx", None)


@pytest.fixture
def model(monkeypatch):
    scripted = ScriptedModel()
    monkeypatch.setattr(agent, "ask_record", scripted)
    # Given rather than defaulted: the default lists come with the ML stack, which CI does not install.
    monkeypatch.setattr(scope_records, "_default_stopwords", lambda: STOPWORDS)
    return scripted


@pytest.fixture
def connect(llm_settings):
    counter = collections.Counter()

    def connect():
        counter["connect"] += 1
        return llm_settings
    connect.counter = counter
    return connect


class TestARun:
    def test_a_run_writes_its_reports_and_its_own_recipe(self, files, tmp_path, model, connect, capsys):
        corpus, definition = files
        plan = cli.resolve(parse("run", str(corpus), "--definition", str(definition), "--out-dir",
                                 str(tmp_path / "out"), "--control", "2"))
        assert cli.execute(plan, nlp=_NLP(), connect=connect) == 0
        out = tmp_path / "out"
        assert {f"{report.CHECK_BY_HAND}.xlsx", report.KEPT_BIB, "run.toml"} <= {p.name for p in out.iterdir()}
        recorded = runfile.load_run_file(out / "run.toml")
        assert recorded.settings.maybe_control == 2
        assert all(plan.maybe_model == "test-model" and plan.maybe_instrument for plan in recorded.stages.values())

        # The recipe alone repeats the run, and asks nothing: every instrument is named in it.
        kept_before = (out / report.KEPT_BIB).read_text(encoding="utf-8")
        asked = sum(model.calls.values())
        assert cli.execute(cli.resolve(parse("run", "--run-file", str(out / "run.toml"))), nlp=_NLP(),
                           connect=connect) == 0
        assert connect.counter["connect"] == 1 and sum(model.calls.values()) == asked
        assert (out / report.KEPT_BIB).read_text(encoding="utf-8") == kept_before
        assert "no backend was needed" in capsys.readouterr().out

    def test_a_pilots_recipe_reruns_the_pilot_in_place(self, files, tmp_path, model, connect):
        corpus, definition = files
        cli.execute(cli.resolve(parse("run", str(corpus), "--definition", str(definition), "--out-dir",
                                      str(tmp_path / "out"), "--pilot", "3", "--control", "1")),
                    nlp=_NLP(), connect=connect)
        pilot = tmp_path / "out" / "pilot"
        plan = cli.resolve(parse("report", "--run-file", str(pilot / "run.toml")))
        assert plan.reports_dir == pilot and plan.ledger_dir == tmp_path / "out" / "ledgers"
        cli.execute(plan, nlp=_NLP(), connect=connect)
        assert not (pilot / "pilot").exists()
        assert sorted(path.name for path in pilot.iterdir()) == [f"{report.PILOT_REPORT}.xlsx", "run.toml"]

    def test_finish_from_a_marked_copy(self, files, tmp_path, model, connect, capsys):
        corpus, definition = files
        out = tmp_path / "out"
        cli.execute(cli.resolve(parse("run", str(corpus), "--definition", str(definition), "--out-dir", str(out),
                                      "--control", "2")), nlp=_NLP(), connect=connect)
        worklist = tabular.read_table(out / f"{report.CHECK_BY_HAND}.xlsx")
        marked = tmp_path / "marked.tsv"
        tabular.write_table(marked, [], ["key", "decision"], [(row["key"], "keep") for row in worklist])
        assert cli.execute(cli.resolve(parse("finish", str(marked), "--run-file", str(out / "run.toml"))),
                           nlp=_NLP(), connect=connect, sheet=str(marked)) == 0
        assert (out / report.FINAL_BIB).exists()
        assert (out / f"{report.CHECK_BY_HAND}.xlsx").exists(), "finish does not rewrite the reports"


class FickleModel(ScriptedModel):
    """The scripted model, which also calls the ravens at dawn not birds."""

    def __call__(self, llm_settings, prompt, maybe_abort=None, **kwargs):
        turn = super().__call__(llm_settings, prompt, maybe_abort, **kwargs)
        if "Ravens at dawn" in prompt and "credible case" not in prompt and "recording what" not in prompt:
            answers = json.loads(turn.reply)
            for answer in answers:
                if "not_birds" in answer and re.search(rf"(?m)^(--- item )?{answer['i']}(\. | ---\nTitle: )Ravens at dawn",
                                                       prompt):
                    answer["not_birds"] = True
            turn.reply = json.dumps(answers)
        return turn


class TestCompare:
    def test_a_replicate_against_the_first_run(self, files, tmp_path, model, connect, monkeypatch):
        corpus, definition = files
        first, second = tmp_path / "first", tmp_path / "second"
        cli.execute(cli.resolve(parse("run", str(corpus), "--definition", str(definition), "--out-dir", str(first),
                                      "--control", "2")), nlp=_NLP(), connect=connect)
        monkeypatch.setattr(agent, "ask_record", FickleModel())
        cli.execute(cli.resolve(parse("run", str(corpus), "--definition", str(definition), "--out-dir", str(second),
                                      "--control", "2", "--replicate", "2", "--ledgers", str(first / "ledgers"))),
                    nlp=_NLP(), connect=connect)
        main, disagreements = cli.compare_runs(first / "run.toml", second / "run.toml", nlp=_NLP())
        assert main == second / "compare-with-first.xlsx"
        verdict = next(row for row in tabular.read_table(main) if row["measure"] == "verdict")
        assert verdict["records"] == "7" and float(verdict["agreement"]) == pytest.approx(6 / 7, abs=1e-3)  # written to three places
        rows = tabular.read_table(disagreements)
        assert {(row["key"], row["first run"], row["second run"]) for row in rows if row["measure"] == "verdict"} == \
               {("r3", "keep", "drop")}

    def test_a_run_file_without_its_instruments_is_refused(self, files, tmp_path):
        corpus, definition = files
        path = tmp_path / "run.toml"
        runfile.write_run_file(path, runfile.RunFile(runfile.Settings(maybe_bib=corpus, maybe_definition=definition), {}))
        with pytest.raises(cli.UsageError, match="no instrument named"):
            cli.compare_runs(path, path, nlp=_NLP())

"""Tests for `raven.papers.scopebib.runfile`: which model and instrument each stage asks with."""

import pathlib

import pytest

from raven.papers.scopebib import runfile
from raven.papers.scopebib.runfile import RunFile, Settings, StagePlan


def write(tmp_path, text):
    path = tmp_path / "run.toml"
    path.write_text(text, encoding="utf-8")
    return path


class TestLoading:
    def test_a_top_level_model_serves_every_stage_not_naming_its_own(self, tmp_path):
        plans = runfile.load_run_file(write(tmp_path, 'model = "small"\n\n[judge]\nmodel = "big"\n'
                                                      'instrument = "0970b39f"\n')).stages
        assert plans == {"judge": StagePlan("big", "0970b39f"), "review": StagePlan("small"),
                         "extract": StagePlan("small")}

    def test_an_empty_file_leaves_every_stage_to_the_loaded_model(self, tmp_path):
        empty = RunFile(Settings(), {stage: StagePlan() for stage in runfile.STAGES})
        assert runfile.load_run_file(write(tmp_path, "")) == empty

    def test_the_settings_and_a_path_relative_to_the_file(self, tmp_path):
        (tmp_path / "runs").mkdir()
        path = tmp_path / "runs" / "run.toml"
        path.write_text('[run]\nbib = "../corpus.bib"\nrequire = ["abstract"]\nmin_chars = ["abstract=200"]\n'
                        'seed = 7\ncontrol = 0\nformat = "xlsx"\n', encoding="utf-8")
        assert runfile.load_run_file(path).settings == Settings(maybe_bib=(tmp_path / "corpus.bib").resolve(),
                                                                require=("abstract",), min_chars=("abstract=200",),
                                                                maybe_seed=7, maybe_control=0, maybe_format="xlsx")

    @pytest.mark.parametrize("text, complaint", [
        ('[judges]\nmodel = "m"\n', "unknown key 'judges'"),
        ('[judge]\nmodle = "m"\n', r"\[judge\]: unknown key 'modle'"),
        ('[judge]\ninstrument = "0970B39"\n', "eight hex digits"),
        ('[review]\nmodel = 3\n', "non-empty string"),
        ('judge = "m"\n', "must be a table"),
        ('[judge\n', "not valid TOML"),
        ('[run]\nsample = 0\n', r"\[run\] sample: expected a whole number of at least 1"),
        ('[run]\ncontrol = true\n', "expected a whole number"),
        ('[run]\nmin_chars = ["abstract"]\n', "expected FIELD=N"),
        ('[run]\nformat = "csv"\n', "expected one of"),
        ('[run]\nuntil = "judge"\n', r"\[run\]: unknown key 'until'"),
        ('[run]\nserver_nlp = "yes"\n', "expected true or false"),
    ])
    def test_a_mistake_is_named_with_its_file(self, tmp_path, text, complaint):
        path = write(tmp_path, text)
        with pytest.raises(runfile.RunFileError, match=complaint) as caught:
            runfile.load_run_file(path)
        assert str(path) in str(caught.value)


class TestOverrides:
    def test_an_override_replaces_one_stage_and_leaves_the_rest(self):
        run_file = RunFile(Settings(), {"judge": StagePlan("a", "0970b39f"), "review": StagePlan("a")})
        out = runfile.with_overrides(run_file, models=[runfile.parse_override("judge=b", option="--model")],
                                     instruments=[runfile.parse_override("review=14036139", option="--instrument")])
        assert out.stages == {"judge": StagePlan("b", "0970b39f"), "review": StagePlan("a", "14036139"),
                              "extract": StagePlan()}

    def test_a_given_option_replaces_a_setting_and_an_absent_one_does_not(self):
        run_file = RunFile(Settings(maybe_seed=1, maybe_control=40), {})
        out = runfile.with_overrides(run_file, settings={"maybe_seed": 2, "maybe_control": None})
        assert (out.settings.maybe_seed, out.settings.maybe_control) == (2, 40)

    @pytest.mark.parametrize("spec, option, complaint", [
        ("judge", "--model", "STAGE=VALUE"),
        ("judgee=m", "--model", "unknown stage 'judgee'"),
        ("judge=nothex12", "--instrument", "eight hex digits"),
    ])
    def test_a_bad_override_is_refused(self, spec, option, complaint):
        with pytest.raises(runfile.RunFileError, match=complaint):
            runfile.parse_override(spec, option=option)


class TestWriting:
    def test_a_written_file_reads_back_the_same(self, tmp_path):
        run_file = RunFile(Settings(maybe_bib=(tmp_path / "corpus.bib").resolve(),
                                    maybe_definition=(tmp_path / "out" / "d.scope.toml").resolve(),
                                    require=("abstract",), min_chars=("abstract=200",), maybe_sample=50,
                                    maybe_pilot=50, maybe_seed=42, maybe_control=40, maybe_replicate=2,
                                    maybe_format="ods", maybe_server_nlp=True, maybe_context_length=131072),
                           {"judge": StagePlan('a "quoted"\\model\tname\x01', "0970b39f"),
                            "review": StagePlan("b"), "extract": StagePlan()})
        path = tmp_path / "out" / "run.toml"
        runfile.write_run_file(path, run_file, comments=["raven-scopebib 0.0", "a second line"])
        assert runfile.load_run_file(path) == run_file
        text = path.read_text(encoding="utf-8")
        assert text.startswith(f"{runfile.GENERATED}\n# raven-scopebib 0.0\n# a second line\n\n[run]\n")
        assert 'bib = "../corpus.bib"' in text and 'definition = "d.scope.toml"' in text

    def test_a_run_directory_moved_with_its_input_still_resolves(self, tmp_path):
        old, new = tmp_path / "old", tmp_path / "new"
        (old / "out").mkdir(parents=True)
        runfile.write_run_file(old / "out" / "run.toml", RunFile(Settings(maybe_bib=old / "corpus.bib"), {}))
        old.rename(new)
        assert runfile.load_run_file(new / "out" / "run.toml").settings.maybe_bib == (new / "corpus.bib").resolve()

    def test_a_path_sharing_only_the_root_is_written_whole(self, tmp_path):
        path = tmp_path / "run.toml"
        elsewhere = pathlib.Path(tmp_path.anchor) / "corpus.bib"
        runfile.write_run_file(path, RunFile(Settings(maybe_bib=elsewhere, maybe_definition=tmp_path / "d.toml"), {}))
        text = path.read_text(encoding="utf-8")
        assert f'bib = "{elsewhere.as_posix()}"' in text
        assert 'definition = "d.toml"' in text, "the control: a path sharing a directory stays relative"

    def test_every_stage_is_written_in_full_rather_than_by_default(self, tmp_path):
        path = tmp_path / "run.toml"
        runfile.write_run_file(path, RunFile(Settings(), {stage: StagePlan("same") for stage in runfile.STAGES}))
        text = path.read_text(encoding="utf-8")
        assert text.count('model = "same"') == 3 and not text.startswith("model")


class TestArchiving:
    def test_a_hand_written_file_is_kept_aside_and_a_written_one_replaced(self, tmp_path):
        path = write(tmp_path, '# my choice of models\nmodel = "m"\n')
        assert runfile.write_run_file(path, RunFile(Settings(), {})) == tmp_path / "run_orig.toml"
        assert (tmp_path / "run_orig.toml").read_text(encoding="utf-8").startswith("# my choice")
        assert runfile.write_run_file(path, RunFile(Settings(), {})) is None, "the tool's own file is replaced"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["run.toml", "run_orig.toml"]

    def test_an_archive_is_never_written_over(self, tmp_path):
        (tmp_path / "run_orig.toml").write_text("first", encoding="utf-8")
        path = write(tmp_path, "second")
        assert runfile.write_run_file(path, RunFile(Settings(), {})) == tmp_path / "run_orig-2.toml"
        assert (tmp_path / "run_orig.toml").read_text(encoding="utf-8") == "first"
        assert (tmp_path / "run_orig-2.toml").read_text(encoding="utf-8") == "second"

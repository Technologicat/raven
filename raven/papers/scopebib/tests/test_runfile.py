"""Tests for `raven.papers.scopebib.runfile`: which model and instrument each stage asks with."""

import pytest

from raven.papers.scopebib import runfile
from raven.papers.scopebib.runfile import StagePlan


def write(tmp_path, text):
    path = tmp_path / "run.toml"
    path.write_text(text, encoding="utf-8")
    return path


class TestLoading:
    def test_a_top_level_model_serves_every_stage_not_naming_its_own(self, tmp_path):
        plans = runfile.load_run_file(write(tmp_path, 'model = "small"\n\n[judge]\nmodel = "big"\n'
                                                      'instrument = "0970b39f"\n'))
        assert plans == {"judge": StagePlan("big", "0970b39f"), "review": StagePlan("small"),
                         "extract": StagePlan("small")}

    def test_an_empty_file_leaves_every_stage_to_the_loaded_model(self, tmp_path):
        assert runfile.load_run_file(write(tmp_path, "")) == {stage: StagePlan() for stage in runfile.STAGES}

    @pytest.mark.parametrize("text, complaint", [
        ('[judges]\nmodel = "m"\n', "unknown key 'judges'"),
        ('[judge]\nmodle = "m"\n', r"\[judge\]: unknown key 'modle'"),
        ('[judge]\ninstrument = "0970B39"\n', "eight hex digits"),
        ('[review]\nmodel = 3\n', "non-empty string"),
        ('judge = "m"\n', "must be a table"),
        ('[judge\n', "not valid TOML"),
    ])
    def test_a_mistake_is_named_with_its_file(self, tmp_path, text, complaint):
        path = write(tmp_path, text)
        with pytest.raises(runfile.RunFileError, match=complaint) as caught:
            runfile.load_run_file(path)
        assert str(path) in str(caught.value)


class TestOverrides:
    def test_an_override_replaces_one_stage_and_leaves_the_rest(self):
        plans = {"judge": StagePlan("a", "0970b39f"), "review": StagePlan("a")}
        out = runfile.with_overrides(plans, models=[runfile.parse_override("judge=b", option="--model")],
                                     instruments=[runfile.parse_override("review=14036139", option="--instrument")])
        assert out == {"judge": StagePlan("b", "0970b39f"), "review": StagePlan("a", "14036139"),
                       "extract": StagePlan()}

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
        plans = {"judge": StagePlan('a "quoted"\\model\tname\x01', "0970b39f"), "review": StagePlan("b"),
                 "extract": StagePlan()}
        path = tmp_path / "out" / "run.toml"
        runfile.write_run_file(path, plans, comments=["raven-scopebib 0.0", "a second line"])
        assert runfile.load_run_file(path) == plans
        assert path.read_text(encoding="utf-8").startswith("# raven-scopebib 0.0\n# a second line\n\n[judge]\n")

    def test_every_stage_is_written_in_full_rather_than_by_default(self, tmp_path):
        path = tmp_path / "run.toml"
        runfile.write_run_file(path, {stage: StagePlan("same") for stage in runfile.STAGES})
        text = path.read_text(encoding="utf-8")
        assert text.count('model = "same"') == 3 and not text.startswith("model")

"""Tests for raven.papers.deduplicate_runfile: what a `raven-deduplicate` run records, and running from it."""

import shutil
import sys

import pytest

from raven.papers import deduplicate as dd
from raven.papers import deduplicate_runfile as runfile

SOURCE = """@article{a,
  title = {A Study of Learning Analytics},
  author = {Smith, Jane},
  year = {2024},
  doi = {10.1234/x}
}

@article{b,
  title = {A study of learning analytics},
  author = {Smith, Jane},
  year = {2024},
  doi = {10.1234/x}
}
"""


class TestRoundTrip:
    def test_what_is_written_reads_back(self, tmp_path):
        recorded = runfile.RunFile({"filenames": [tmp_path / "in" / "search.bib"], "output": tmp_path / "out.bib",
                                    "audit_format": "ods", "judge": True, "no_audit": False,
                                    "backend_url": "http://localhost:1234"},
                                   {"pairs": {"model": "some-model", "instrument": "1a2b3c4d"}})
        runfile.write(tmp_path / "run.toml", recorded)
        assert runfile.load(tmp_path / "run.toml") == recorded

    def test_paths_are_kept_relative_so_the_folder_can_move(self, tmp_path):
        here = tmp_path / "here"
        here.mkdir()
        runfile.write(here / "run.toml", runfile.RunFile({"filenames": [here / "search.bib"],
                                                          "output": here / "out.bib"}))
        assert '"search.bib"' in (here / "run.toml").read_text(encoding="utf-8")
        moved = tmp_path / "moved"
        shutil.move(here, moved)
        assert runfile.load(moved / "run.toml").settings["filenames"] == [moved / "search.bib"]

    def test_an_unknown_setting_is_refused_naming_the_file(self, tmp_path):
        (tmp_path / "run.toml").write_text('[run]\njudje = true\n', encoding="utf-8")
        with pytest.raises(runfile.RunFileError, match="run.toml.*'judje'"):
            runfile.load(tmp_path / "run.toml")


class TestRunningFromIt:
    def _main(self, monkeypatch, *args):
        monkeypatch.setattr(sys, "argv", ["raven-deduplicate", *map(str, args)])
        dd.main()

    def test_a_run_writes_its_run_file_beside_the_output(self, monkeypatch, tmp_path):
        source = tmp_path / "search.bib"
        source.write_text(SOURCE, encoding="utf-8")
        self._main(monkeypatch, source, "-o", tmp_path / "deduped.bib", "--audit-format", "ods")
        settings = runfile.load(tmp_path / "deduped_run.toml").settings
        assert settings["filenames"] == [source] and settings["output"] == tmp_path / "deduped.bib"
        assert settings["audit_format"] == "ods" and settings["judge"] is False

    def test_the_run_file_alone_runs_it_again(self, monkeypatch, tmp_path):
        source = tmp_path / "search.bib"
        source.write_text(SOURCE, encoding="utf-8")
        self._main(monkeypatch, source, "-o", tmp_path / "deduped.bib")
        first = (tmp_path / "deduped.bib").read_text(encoding="utf-8")
        (tmp_path / "deduped.bib").unlink()
        self._main(monkeypatch, "--run-file", tmp_path / "deduped_run.toml")
        assert (tmp_path / "deduped.bib").read_text(encoding="utf-8") == first

    def test_the_command_line_overrides_the_run_file(self, monkeypatch, tmp_path):
        source = tmp_path / "search.bib"
        source.write_text(SOURCE, encoding="utf-8")
        self._main(monkeypatch, source, "-o", tmp_path / "deduped.bib", "--audit-format", "ods")
        self._main(monkeypatch, "--run-file", tmp_path / "deduped_run.toml", "--audit-format", "tsv")
        assert (tmp_path / "deduped_audit.tsv").exists()
        assert runfile.load(tmp_path / "deduped_run.toml").settings["audit_format"] == "tsv", \
            "the run file records the run as it now is"

    def test_a_dry_run_keeps_no_run_file(self, monkeypatch, tmp_path):
        source = tmp_path / "search.bib"
        source.write_text(SOURCE, encoding="utf-8")
        self._main(monkeypatch, source)
        assert list(tmp_path.glob("*_run.toml")) == []

    def test_the_judges_record_survives_a_run_rewriting_its_run_file(self, monkeypatch, tmp_path):
        """The run file is rewritten as each run starts; what the judge recorded must not be lost by that."""
        source = tmp_path / "search.bib"
        source.write_text(SOURCE, encoding="utf-8")
        self._main(monkeypatch, source, "-o", tmp_path / "deduped.bib")
        path = tmp_path / "deduped_run.toml"
        runfile.write(path, runfile.with_instrument(runfile.load(path), "pairs", model="m", instrument="1a2b3c4d"))
        self._main(monkeypatch, source, "-o", tmp_path / "deduped.bib")
        assert runfile.load(path).instruments["pairs"]["instrument"] == "1a2b3c4d"

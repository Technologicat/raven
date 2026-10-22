"""Tests for `raven.papers.scopebib.report`: where each record ends up, and the files a run leaves."""

import pytest

from raven.common import tabular
from raven.papers import bibtex
from raven.papers import siftbib
from raven.papers.scopebib import definition as scope_definition
from raven.papers.scopebib import judge, report
from raven.papers.scopebib import records as scope_records

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
nowhere = "no birds were observed at all"

[[rule]]
label = "urban"
when = { setting = ["city"] }
action = "remove"

[[rule]]
label = "no birds at all"
when = { setting = ["nowhere"] }
action = "hold"
"""

KEYS = ("dropped", "unreviewed", "rescued", "unsure", "city", "nowhere", "forest", "unextracted", "unjudged")

BIB = '@string{rv = "Raven Review"}\n\n' + "".join(
    f"@article{{{key},\n  title = {{About {key}}},\n  author = {{Corvus, {key.title()}}},\n  year = {{2024}},\n"
    f"  journal = rv,\n  abstract = {{Text of {key}.}},\n}}\n\n" for key in KEYS)


def answer(not_birds, confidence="high", source="abstract"):
    out = {"tests": {"not_birds": not_birds}, "confidence": confidence, "why": "a reason", "source": source}
    return {**out, "verdict": judge.verdict_of(out)}


def extraction(setting, found=True):
    return {"fields": {"setting": setting}, "quotes": {"setting": {"text": "the city", "found": found}}, "raw": {}}


@pytest.fixture
def definition(tmp_path):
    path = tmp_path / "ravens.scope.toml"
    path.write_text(DEFINITION, encoding="utf-8")
    return scope_definition.load_definition(path)


@pytest.fixture
def run(tmp_path):
    """A run with one record at every point a record can stop."""
    path = tmp_path / "in.bib"
    path.write_text(BIB, encoding="utf-8")
    library = bibtex.parse_file(path, split_names=False)
    final = {"dropped": answer(True), "unreviewed": answer(True), "rescued": answer(True),
             "unsure": answer(True, "low", "title"),
             **{key: answer(False) for key in ("city", "nowhere", "forest", "unextracted")}}
    reviews = {"dropped": {"belongs": False, "case": "no case"},
               "rescued": {"belongs": True, "case": "it counts ravens"},
               "forest": {"belongs": True, "case": "a control record the reviewer liked"}}
    extractions = {"city": extraction("city"), "nowhere": extraction("nowhere"), "forest": extraction("forest"),
                   "dropped": extraction("forest")}  # an extraction of a record the judge did not keep
    return dict(library=library, records=scope_records.records_of(library), final=final, reviews=reviews,
                extractions=extractions)


def outcomes_of(run, definition):
    return report.outcomes(run["records"], definition=definition, final=run["final"], reviews=run["reviews"],
                           extractions=run["extractions"])


def write(run, definition, out_dir, **kwargs):
    kwargs.setdefault("control", ["forest"])
    kwargs.setdefault("provenance", ["raven-scopebib test"])
    return report.write_reports(out_dir, definition=definition, **run, **kwargs)


class TestOutcomes:
    def test_every_point_a_record_can_stop(self, run, definition):
        got = {key: (outcome.action, outcome.stage) for key, outcome in outcomes_of(run, definition).items()}
        assert got == {"dropped": (report.REMOVED, "judge"),
                       "unreviewed": (report.PENDING, "review"),
                       "rescued": (report.TO_A_PERSON, "review"),
                       "unsure": (report.TO_A_PERSON, "judge"),
                       "city": (report.REMOVED, "rules"),
                       "nowhere": (report.TO_A_PERSON, "rules"),
                       "forest": (report.KEPT, "rules"),
                       "unextracted": (report.PENDING, "extract"),
                       "unjudged": (report.PENDING, "judge")}

    def test_a_removal_whose_quote_is_not_in_the_record_goes_to_a_person(self, run, definition):
        run["extractions"]["city"] = extraction("city", found=False)
        outcome = outcomes_of(run, definition)["city"]
        assert (outcome.action, outcome.decided_by) == (report.TO_A_PERSON, "urban")
        assert "not in the record" in outcome.reason

    def test_with_no_fields_a_keep_is_final(self, run, tmp_path):
        path = tmp_path / "bare.scope.toml"
        path.write_text(DEFINITION.split("[[field]]")[0], encoding="utf-8")
        bare = scope_definition.load_definition(path)
        assert outcomes_of(run, bare)["unextracted"].action == report.KEPT

    def test_a_reason_says_what_fired_and_the_case_for_a_rescue(self, run, definition):
        got = outcomes_of(run, definition)
        assert got["dropped"].decided_by == "not_birds" and "not_birds true, high confidence" in got["dropped"].reason
        assert "it counts ravens" in got["rescued"].reason
        assert got["city"].reason == 'setting = city ("the city")', "the reason picks up where `decided_by` leaves off"
        assert got["nowhere"].reason == 'setting = nowhere ("the city")'

    def test_a_quote_the_model_wrapped_is_not_quoted_twice(self, run, definition):
        run["extractions"]["city"]["quotes"]["setting"]["text"] = "“the city”"
        assert outcomes_of(run, definition)["city"].reason == 'setting = city ("the city")'


class TestFiles:
    def test_the_kept_bib_holds_the_survivors_and_its_strings(self, run, definition, tmp_path):
        write(run, definition, tmp_path / "out")
        kept = bibtex.parse_file(tmp_path / "out" / report.KEPT_BIB, split_names=False)
        assert [entry.key for entry in kept.entries] == ["forest"]
        assert "rv" in kept.strings_dict

    def test_the_worklist_and_the_removals(self, run, definition, tmp_path):
        write(run, definition, tmp_path / "out")
        worklist = tabular.read_table(tmp_path / "out" / f"{report.CHECK_BY_HAND}.tsv")
        assert sorted(row["key"] for row in worklist) == ["nowhere", "rescued", "unsure"]
        assert all(row["decision"] == "" for row in worklist)
        removed = tabular.read_table(tmp_path / "out" / f"{report.REMOVED_AUDIT}.tsv")
        assert [(row["key"], row["removed by"]) for row in removed] == [("dropped", "judge"), ("city", "rules")]

    def test_the_sift_is_accounted_for_in_the_removals(self, run, definition, tmp_path):
        sift = report.SiftResult(dropped=[siftbib.DroppedRecord(key="bare", reason="no abstract", title="Bare",
                                                                venue="")],
                                 criteria=[siftbib.require_field("abstract")])
        written = {path.name for path in write(run, definition, tmp_path / "out", maybe_sift=sift)}
        assert {f"{report.SIFT_AUDIT}.tsv", report.SIFTED_BIB} <= written
        removed = tabular.read_table(tmp_path / "out" / f"{report.REMOVED_AUDIT}.tsv")
        assert removed[0]["key"] == "bare" and removed[0]["removed by"] == "sift"

    def test_the_review_audit_labels_the_control_and_compares_the_rates(self, run, definition, tmp_path):
        write(run, definition, tmp_path / "out")
        path = tmp_path / "out" / f"{report.REVIEW_AUDIT}.tsv"
        rows = tabular.read_table(path)
        assert {(row["group"], row["key"]) for row in rows} == {("drop", "dropped"), ("drop", "rescued"),
                                                                 ("control", "forest")}
        assert "a case for 1 of 2 (50.0%) drops, and for 1 of 1 (100.0%) control" in path.read_text(encoding="utf-8")

    def test_the_extract_audit_covers_only_the_judges_keeps(self, run, definition, tmp_path):
        write(run, definition, tmp_path / "out")
        rows = tabular.read_table(tmp_path / "out" / f"{report.EXTRACT_AUDIT}.tsv")
        assert sorted(row["key"] for row in rows) == ["city", "forest", "nowhere"]

    def test_a_stage_with_no_answers_writes_no_table(self, run, definition, tmp_path):
        run["extractions"] = {}
        written = {path.name for path in write(run, definition, tmp_path / "out")}
        assert f"{report.EXTRACT_AUDIT}.tsv" not in written and f"{report.JUDGE_AUDIT}.tsv" in written

    @pytest.mark.parametrize("table_format", tabular.FORMATS)
    def test_every_table_format_is_written(self, run, definition, tmp_path, table_format):
        write(run, definition, tmp_path / "out", table_format=table_format)
        assert len(tabular.read_table(tmp_path / "out" / f"{report.CHECK_BY_HAND}.{table_format}")) == 3


class TestOwnership:
    def test_a_stale_report_goes_and_a_users_copy_stays(self, run, definition, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        stale = [out / f"{report.JUDGE_AUDIT}.ods",  # an earlier run in another format
                 out / f"{report.SIFT_AUDIT}.tsv"]   # a stage that did not run this time
        mine = [out / f"{report.CHECK_BY_HAND}-marked.tsv", out / "notes.txt"]
        for path in stale + mine:
            path.write_text("x", encoding="utf-8")
        write(run, definition, out)
        assert not any(path.exists() for path in stale)
        assert all(path.read_text(encoding="utf-8") == "x" for path in mine)

    def test_every_name_written_is_owned(self, run, definition, tmp_path):
        sift = report.SiftResult(dropped=[], criteria=[siftbib.require_field("abstract")])
        written = write(run, definition, tmp_path / "out", maybe_sift=sift)
        assert {path.name for path in written} <= report.OWNED_NAMES


@pytest.fixture
def finished(run):
    """The same run with nothing pending: the worklist is `rescued`, `unsure` and `nowhere`."""
    run["reviews"]["unreviewed"] = {"belongs": False, "case": "no case"}
    run["extractions"]["unextracted"] = extraction("forest")
    run["final"]["unjudged"] = answer(False)
    run["extractions"]["unjudged"] = extraction("forest")
    return run


def mark(path, *rows, columns=("key", "decision", "note")):
    tabular.write_table(path, [], columns, rows)
    return path


def finish(run, definition, out_dir, sheet, **kwargs):
    return report.finish(out_dir, sheet=sheet, definition=definition, provenance=["raven-scopebib test"],
                         **{key: value for key, value in run.items()}, **kwargs)


class TestFinish:
    def test_a_person_keeps_and_a_blank_drops(self, finished, definition, tmp_path):
        sheet = mark(tmp_path / "marked.tsv", ("rescued", " Keep ", "looked fine"), ("unsure", "", ""),
                     ("nowhere", "", ""))
        written, warnings = finish(finished, definition, tmp_path / "out", sheet)
        final_bib = bibtex.parse_file(tmp_path / "out" / report.FINAL_BIB, split_names=False)
        assert sorted(entry.key for entry in final_bib.entries) == ["forest", "rescued", "unextracted", "unjudged"]
        audit = {row["key"]: row for row in tabular.read_table(tmp_path / "out" / f"{report.HAND_AUDIT}.tsv")}
        assert {key: row["decision"] for key, row in audit.items()} == {"rescued": "keep", "unsure": "drop",
                                                                        "nowhere": "drop"}
        assert audit["rescued"]["note"] == "looked fine"

    def test_an_unknown_decision_is_refused_and_its_row_named(self, finished, definition, tmp_path):
        sheet = mark(tmp_path / "marked.tsv", ("rescued", "kep", ""), ("unsure", "", ""), ("nowhere", "", ""))
        with pytest.raises(report.FinishError, match="row 2 \\(rescued\\): 'kep'"):
            finish(finished, definition, tmp_path / "out", sheet)
        assert not (tmp_path / "out" / report.FINAL_BIB).exists()

    def test_a_record_missing_from_the_sheet_is_refused(self, finished, definition, tmp_path):
        sheet = mark(tmp_path / "marked.tsv", ("rescued", "keep", ""), ("unsure", "keep", ""))
        with pytest.raises(report.FinishError, match="not in the sheet.*nowhere"):
            finish(finished, definition, tmp_path / "out", sheet)

    def test_a_row_the_run_has_since_settled_is_ignored_with_a_warning(self, finished, definition, tmp_path):
        sheet = mark(tmp_path / "marked.tsv", ("forest", "", ""), ("rescued", "keep", ""), ("unsure", "keep", ""),
                     ("nowhere", "keep", ""))
        _, warnings = finish(finished, definition, tmp_path / "out", sheet)
        assert any("no longer sends to a person" in warning and "forest" in warning for warning in warnings)
        final_bib = bibtex.parse_file(tmp_path / "out" / report.FINAL_BIB, split_names=False)
        assert "forest" in [entry.key for entry in final_bib.entries]

    def test_an_unfinished_run_is_refused(self, run, definition, tmp_path):
        sheet = mark(tmp_path / "marked.tsv", ("rescued", "keep", ""))
        with pytest.raises(report.FinishError, match="not finished"):
            finish(run, definition, tmp_path / "out", sheet)

    def test_a_sheet_marking_every_row_drops_on_drop(self, finished, definition, tmp_path):
        sheet = mark(tmp_path / "marked.tsv", ("rescued", "keep", ""), ("unsure", "DROP", ""), ("nowhere", "drop", ""))
        finish(finished, definition, tmp_path / "out", sheet, require_every_decision=True)
        kept = {entry.key for entry in bibtex.parse_file(tmp_path / "out" / report.FINAL_BIB, split_names=False).entries}
        assert "rescued" in kept and not {"unsure", "nowhere"} & kept

    def test_a_blank_beside_a_drop_is_an_undecided_row(self, finished, definition, tmp_path):
        sheet = mark(tmp_path / "marked.tsv", ("rescued", "keep", ""), ("unsure", "drop", ""), ("nowhere", "", ""))
        with pytest.raises(report.FinishError, match=r"1 are blank \(rows 4\).*remove every 'drop'"):
            finish(finished, definition, tmp_path / "out", sheet)

    def test_a_blank_is_refused_on_request(self, finished, definition, tmp_path):
        sheet = mark(tmp_path / "marked.tsv", ("rescued", "keep", ""), ("unsure", "", ""), ("nowhere", "", ""))
        finish(finished, definition, tmp_path / "out", sheet)  # the control: a blank drops by default
        with pytest.raises(report.FinishError, match=r"2 rows have no decision \(rows 3, 4\)"):
            finish(finished, definition, tmp_path / "out", sheet, require_every_decision=True)

    def test_a_sheet_in_a_spreadsheet_format_is_read(self, finished, definition, tmp_path):
        sheet = mark(tmp_path / "marked.xlsx", ("rescued", "keep", ""), ("unsure", "", ""), ("nowhere", "", ""))
        finish(finished, definition, tmp_path / "out", sheet, table_format="ods")
        assert (tmp_path / "out" / f"{report.HAND_AUDIT}.ods").exists()

    def test_rewriting_the_reports_removes_the_05_files(self, finished, definition, tmp_path):
        sheet = mark(tmp_path / "marked.tsv", ("rescued", "keep", ""), ("unsure", "", ""), ("nowhere", "", ""))
        written, _ = finish(finished, definition, tmp_path / "out", sheet)
        assert all(path.exists() for path in written)
        write(finished, definition, tmp_path / "out")
        assert not any(path.exists() for path in written)


class TestTrailingBlanks:
    def warning(self, *decisions):
        return report._trailing_blank_warning(list(decisions), first_row=2)

    def test_a_tail_longer_than_any_earlier_blank_run_is_suspect(self):
        assert "the last 3 rows, from row 5 on" in self.warning("keep", "", "keep", "", "", "")

    def test_a_tail_no_longer_than_an_earlier_run_is_not(self):
        assert self.warning("", "", "keep", "", "") is None

    def test_no_keep_at_all_says_everything_drops(self):
        assert "all 2 records will be dropped" in self.warning("", "")

    def test_a_sheet_ending_in_a_keep_is_not_suspect(self):
        assert self.warning("", "", "", "keep") is None

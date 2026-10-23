"""Tests for `raven.papers.scopebib.compare`: Cohen's kappa, what is compared, and the tables."""

import pytest

from raven.common import tabular
from raven.papers.scopebib import compare
from raven.papers.scopebib import definition as scope_definition
from raven.papers.scopebib import records as scope_records


class TestKappa:
    def test_the_worked_example(self):
        # Wikipedia's example: 50 items, both say yes to 20 and no to 15, and they split 5 and 10 the other ways.
        # Observed agreement 0.7; by chance 0.5 * 0.6 + 0.5 * 0.4 = 0.5; kappa (0.7 - 0.5) / (1 - 0.5) = 0.4.
        pairs = [("yes", "yes")] * 20 + [("yes", "no")] * 5 + [("no", "yes")] * 10 + [("no", "no")] * 15
        assert compare.cohens_kappa(pairs) == pytest.approx(0.4)

    def test_perfect_agreement_across_categories_is_one(self):
        assert compare.cohens_kappa([("a", "a"), ("b", "b")]) == pytest.approx(1.0)

    def test_agreement_no_better_than_chance_is_zero(self):
        pairs = [("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")]
        assert compare.cohens_kappa(pairs) == pytest.approx(0.0)

    @pytest.mark.parametrize("pairs", [[], [("a", "a"), ("a", "a")]])
    def test_it_is_not_defined_without_items_or_with_one_answer_throughout(self, pairs):
        assert compare.cohens_kappa(pairs) is None


DEFINITION = """
scope = "studies of ravens"

[[test]]
name = "not_birds"
asks = "true if the work is positively not about birds"

[[field]]
name = "setting"
asks = "where"
[field.values]
city = "urban"
forest = "woods"

[[field]]
name = "method"
asks = "how"
max_words = 3
"""


@pytest.fixture
def definition(tmp_path):
    path = tmp_path / "d.scope.toml"
    path.write_text(DEFINITION, encoding="utf-8")
    return scope_definition.load_definition(path)


def answer(verdict, not_birds, confidence="high"):
    return {"verdict": verdict, "tests": {"not_birds": not_birds}, "confidence": confidence, "source": "title"}


def fields(setting, method="tagging"):
    return {"fields": {"setting": setting, "method": method}, "quotes": {}, "raw": {}}


class TestMeasures:
    def test_each_stage_over_the_records_both_runs_answered(self, definition):
        a = {"final": {"k1": answer("keep", False), "k2": answer("drop", True), "only_a": answer("keep", False)},
             "reviews": {"k2": {"belongs": False}},
             "extractions": {"k1": fields("city")}}
        b = {"final": {"k1": answer("keep", False), "k2": answer("keep", False, "low")},
             "reviews": {"k2": {"belongs": True}},
             "extractions": {"k1": fields("forest", "counting")}}
        got = {(m.stage, m.name): m for m in compare.measures(a, b, definition_a=definition, definition_b=definition)}
        assert set(got) == {("judge", "verdict"), ("judge", "not_birds"), ("judge", "confidence"),
                            ("review", "belongs"), ("extract", "setting")}, "a free-text field has nothing to agree on"
        assert [key for key, _, _ in got["judge", "verdict"].pairs] == ["k1", "k2"], "only_a was answered once"
        assert got["judge", "verdict"].agreement == pytest.approx(0.5)
        assert got["extract", "setting"].pairs == [("k1", "city", "forest")]

    def test_a_test_only_one_definition_has_is_not_compared(self, definition, tmp_path):
        path = tmp_path / "other.scope.toml"
        path.write_text(DEFINITION.replace("not_birds", "not_ravens"), encoding="utf-8")
        other = scope_definition.load_definition(path)
        names = [m.name for m in compare.measures({"final": {}, "reviews": {}, "extractions": {}},
                                                  {"final": {}, "reviews": {}, "extractions": {}},
                                                  definition_a=definition, definition_b=other)]
        assert "not_birds" not in names and "not_ravens" not in names and "verdict" in names


class TestTables:
    def test_the_measures_and_the_disagreements(self, tmp_path):
        compared = [compare.Measure("judge", "verdict", [("k1", "keep", "keep"), ("k2", "drop", "keep")]),
                    compare.Measure("review", "belongs", [])]
        records = [scope_records.Record("k2", "A title", "", "", "")]
        main, disagreements = compare.write_comparison(tmp_path / "compare-with-a.tsv", compared, records=records,
                                                       provenance=["test"])
        rows = tabular.read_table(main)
        assert (rows[0]["agreement"], rows[1]["kappa"]) == ("0.5", "not defined")
        assert tabular.read_table(disagreements) == [{"stage": "judge", "measure": "verdict", "key": "k2",
                                                      "first run": "drop", "second run": "keep",
                                                      "title": "A title"}]

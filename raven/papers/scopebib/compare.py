"""How far two runs agree: test-retest reliability, or one model against another.

Each stage is compared over the records both runs answered, measure by measure (`measures`): the judge's
verdict, each of its tests and its confidence; the reviewer's `belongs`; each closed field the extractor
filled. For each, the share of records on which the two agree, and Cohen's kappa (`cohens_kappa`), which
discounts the agreement two raters would reach by chance given how often each gives each answer — so a
test that is false for nearly every record does not look reliable merely by being false twice.

Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological
Measurement* 20(1), 37–46. https://en.wikipedia.org/wiki/Cohen%27s_kappa
"""

__all__ = ["cohens_kappa", "Measure", "measures", "write_comparison"]

import collections
import pathlib
from collections.abc import Sequence
from typing import NamedTuple

from ...common import tabular
from . import definition as scope_definition
from . import records as scope_records


def cohens_kappa(pairs: Sequence[tuple[object, object]]) -> float | None:
    """Cohen's kappa for two raters' answers, one `(a, b)` pair per item.

    `None` where it is not defined: no items, or both raters giving one and the same answer throughout,
    when agreement by chance is certain and there is nothing left for agreement to measure.
    """
    n = len(pairs)
    if not n:
        return None
    observed = sum(1 for a, b in pairs if a == b) / n
    counts_a = collections.Counter(a for a, _ in pairs)
    counts_b = collections.Counter(b for _, b in pairs)
    expected = sum(counts_a[category] * counts_b[category] for category in counts_a) / (n * n)
    if expected == 1:
        return None
    return (observed - expected) / (1 - expected)


class Measure(NamedTuple):
    """One thing compared: `stage` and `name` say what, `pairs` holds `(key, a, b)` per record both answered."""
    stage: str
    name: str
    pairs: list[tuple[str, object, object]]

    def _get_agreement(self) -> float | None:
        return sum(1 for _, a, b in self.pairs if a == b) / len(self.pairs) if self.pairs else None
    agreement = property(fget=_get_agreement, doc="The share of records on which the two agree; `None` for none.")

    def _get_kappa(self) -> float | None:
        return cohens_kappa([(a, b) for _, a, b in self.pairs])
    kappa = property(fget=_get_kappa, doc="Cohen's kappa over the pairs (`cohens_kappa`).")


def _pairs(a: dict[str, object], b: dict[str, object], value) -> list[tuple[str, object, object]]:
    return [(key, value(a[key]), value(b[key])) for key in sorted(a.keys() & b.keys())]


def measures(a: dict, b: dict, *,
             definition_a: scope_definition.Definition,
             definition_b: scope_definition.Definition) -> list[Measure]:
    """What to compare between runs `a` and `b`, each a dict of `final`, `reviews` and `extractions` as
    `raven.papers.scopebib.pipeline.State` holds them.

    Tests and fields are compared where both definitions have them; a field only where it is closed, free
    text having no categories to agree on.
    """
    out = [Measure("judge", "verdict", _pairs(a["final"], b["final"], lambda answer: answer["verdict"]))]
    tests_b = {test.name for test in definition_b.tests}
    for test in definition_a.tests:
        if test.name in tests_b:
            out.append(Measure("judge", test.name,
                               _pairs(a["final"], b["final"], lambda answer, name=test.name: answer["tests"].get(name))))
    out.append(Measure("judge", "confidence", _pairs(a["final"], b["final"], lambda answer: answer["confidence"])))
    out.append(Measure("review", "belongs", _pairs(a["reviews"], b["reviews"], lambda answer: answer["belongs"])))
    closed_b = {field.name for field in definition_b.fields if field.values is not None}
    for field in definition_a.fields:
        if field.values is not None and field.name in closed_b:
            out.append(Measure("extract", field.name,
                               _pairs(a["extractions"], b["extractions"],
                                      lambda answer, name=field.name: answer["fields"].get(name))))
    return out


def _number(value: float | None) -> float | str:
    return round(value, 3) if value is not None else "not defined"


def write_comparison(path: str | pathlib.Path, compared: Sequence[Measure], *,
                     records: Sequence[scope_records.Record],
                     provenance: Sequence[str]) -> list[pathlib.Path]:
    """Write the measures to `path`, and the records the runs disagreed on beside it, `-disagreements` added to
    its name. Returns both paths.

    `records`: for the titles in the disagreements.
    `provenance`: lines heading both tables — which two runs, with what.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    titles = {record.key: record.title for record in records}
    notes = [*provenance,
             "agreement: the share of the records both runs answered on which they gave the same answer",
             "kappa: Cohen's kappa, agreement beyond what chance would give at the rates each run gives each "
             "answer; 1 is perfect, 0 no better than chance; not defined where both runs gave a single answer "
             "throughout. See https://en.wikipedia.org/wiki/Cohen%27s_kappa"]
    tabular.write_table(path, notes, ("stage", "measure", "records", "agreement", "kappa"),
                        [(m.stage, m.name, len(m.pairs), _number(m.agreement), _number(m.kappa)) for m in compared])
    disagreements = path.with_name(f"{path.stem}-disagreements{path.suffix}")
    rows = [(m.stage, m.name, key, str(a), str(b), titles.get(key, ""))
            for m in compared for key, a, b in m.pairs if a != b]
    # The count behind a label, so that no noun has to agree with it.
    tabular.write_table(disagreements, [*provenance, f"disagreements, one row per record and measure: {len(rows)}"],
                        ("stage", "measure", "key", "first run", "second run", "title"), rows)
    return [path, disagreements]

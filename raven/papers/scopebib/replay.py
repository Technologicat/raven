"""Bringing answers made elsewhere into the tool's ledgers, so a run can use them without asking again.

The source is the prototype in `investigations/aokk-corpus-scope/`. `judge_scope.py` kept one JSON line per
answer, `{"key", <one boolean per test>, "confidence", "why", "source"}`, `source` being `"title"` or
`"abstract"` for the pass that answered, a later line for the same key and pass superseding an earlier one.
`extract_fields.py` kept one flat line per record, `{"key", "v", <one value per field>, "evidence"}`.
`review_drops.py` wrote a table per slice of the drops, with the columns `group`, `belongs`, `key` and
`case`, `group` being `"dropped"` or `"control"`.

Replayed answers keep the instrument that made them. They are written under a fingerprint the caller names —
`fingerprint_of`, over the prompts that produced them, is the honest choice — and never under the tool's
own, so a report built from them says which judge decided, and a run under the tool's instrument still asks
everything afresh.
"""

__all__ = ["fingerprint_of", "ImportCount", "import_prototype_judgements", "import_prototype_extractions",
           "import_prototype_reviews"]

import hashlib
import json
import pathlib
from collections.abc import Iterable
from typing import NamedTuple

from ...common import tabular
from ...librarian import sweep
from . import definition as scope_definition
from . import extract
from . import judge
from . import records as scope_records


def fingerprint_of(*texts: str) -> str:
    """A fingerprint for answers made by an instrument described by `texts`, typically its prompts. Eight hex
    digits, the shape of `raven.librarian.sweep.fingerprint`'s."""
    return hashlib.sha256("\x00".join(texts).encode("utf-8")).hexdigest()[:8]


class ImportCount(NamedTuple):
    """How many answers went into each ledger."""
    titles: int
    abstracts: int


def import_prototype_judgements(path: str | pathlib.Path, *, titles: sweep.Ledger, abstracts: sweep.Ledger,
                                definition: scope_definition.Definition) -> ImportCount:
    """Write the prototype judge's answers in `path` into the two pass ledgers.

    `titles`, `abstracts`: the ledgers to write, normally `judge.ledgers(directory, fingerprint)` under the
                           replayed instrument's fingerprint.
    `definition`: whose tests the answers are read against. Each test must be a field of every answer, by
                  name; a `null` reads as unanswered, as the prototype stored a withheld verdict.

    Only the latest answer per key and pass is written. Raises `ValueError` on a line that is not an answer,
    naming the line, or when the file's tests do not match the definition's.
    """
    path = pathlib.Path(path)
    latest: dict[tuple[str, str], dict] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: not a JSON line ({exc})") from exc
        if not isinstance(entry, dict) or "key" not in entry or entry.get("source") not in ("title", "abstract"):
            raise ValueError(f"{path}:{number}: not a judge's answer: needs a `key` and a `source` of "
                             f"\"title\" or \"abstract\"")
        if missing := [test.name for test in definition.tests if test.name not in entry]:
            raise ValueError(f"{path}:{number}: no answer for the test{'s' if len(missing) != 1 else ''} "
                             f"{', '.join(missing)}; the definition's tests are "
                             f"{', '.join(test.name for test in definition.tests)}")
        latest[(entry["key"], entry["source"])] = entry
    counts = {"title": 0, "abstract": 0}
    for (key, source), entry in latest.items():
        (titles if source == "title" else abstracts).write_answer(key, judge.normalize(entry, definition, source))
        counts[source] += 1
    return ImportCount(titles=counts["title"], abstracts=counts["abstract"])


def import_prototype_extractions(path: str | pathlib.Path, *, ledger: sweep.Ledger,
                                 definition: scope_definition.Definition,
                                 records: dict[str, scope_records.Record]) -> int:
    """Write the prototype extractor's answers in `path` into `ledger`. Returns how many were written.

    `ledger`: normally `extract.ledger(directory, fingerprint)` under the replayed instrument's fingerprint.
    `definition`: whose fields the answers are read against, by name.
    `records`: the corpus, by key, against which each quote is checked. A line for a key not in it is
               skipped, and counted out of the return value.

    Three translations, each the prototype's spelling of something the tool spells differently: a JSON
    `true`/`false` becomes `"yes"`/`"no"` where the field's vocabulary has those; any value outside a
    vocabulary becomes `not_stated`, kept in `raw` (the prototype's `"unclear"`); and the prototype's single
    `"evidence"` becomes the quote of the definition's one quoted field. Raises `ValueError` where that
    field is not unique, or a line lacks a field the definition names.
    """
    path = pathlib.Path(path)
    quoted = [field.name for field in definition.fields if field.quote]
    if len(quoted) > 1:
        raise ValueError(f"the prototype kept one `evidence` per record, and the definition quotes "
                         f"{', '.join(quoted)}: which one it belongs to cannot be told")
    written = 0
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: not a JSON line ({exc})") from exc
        if not isinstance(entry, dict) or "key" not in entry:
            raise ValueError(f"{path}:{number}: not an extraction: needs a `key`")
        if missing := [field.name for field in definition.fields if field.name not in entry]:
            raise ValueError(f"{path}:{number}: no value for {', '.join(missing)}")
        if entry["key"] not in records:
            continue
        answer = {}
        for field in definition.fields:
            value = entry[field.name]
            if isinstance(value, bool) and field.values is not None and {"yes", "no"} <= set(field.values):
                value = "yes" if value else "no"
            answer[field.name] = value
        if quoted:
            answer[f"{quoted[0]}_quote"] = entry.get("evidence", "")
        ledger.write_answer(entry["key"], extract.normalize(answer, definition, records[entry["key"]]))
        written += 1
    return written


def import_prototype_reviews(paths: Iterable[str | pathlib.Path], *, ledger: sweep.Ledger) -> int:
    """Write the prototype reviewer's answers in the tables at `paths` into `ledger`. Returns how many.

    `ledger`: normally `review.ledger(directory, fingerprint)` under the replayed instrument's fingerprint.

    Both groups are written, the control's answers being what the reviewer's discrimination is read
    against. A `belongs` of `yes` or `no` becomes `True` or `False`, anything else `None`. Raises
    `ValueError` for a table lacking one of the columns.
    """
    written = 0
    for path in map(pathlib.Path, paths):
        rows = tabular.read_table(path)
        if rows and (missing := sorted({"group", "belongs", "key", "case"} - set(rows[0]))):
            raise ValueError(f"{path}: no column{'s' if len(missing) != 1 else ''} {', '.join(missing)}; "
                             f"is this a table `review_drops.py` wrote?")
        for row in rows:
            belongs = {"yes": True, "no": False}.get(row["belongs"].strip().lower())
            ledger.write_answer(row["key"], {"belongs": belongs, "case": " ".join(row["case"].split())[:300]})
            written += 1
    return written

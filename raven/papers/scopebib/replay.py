"""Bringing answers made elsewhere into the tool's ledgers, so a run can use them without asking again.

The source is the prototype in `investigations/aokk-corpus-scope/`: `judge_scope.py` kept one JSON line per
answer, `{"key", <one boolean per test>, "confidence", "why", "source"}`, `source` being `"title"` or
`"abstract"` for the pass that answered, a later line for the same key and pass superseding an earlier one.

Replayed answers keep the instrument that made them. They are written under a fingerprint the caller names —
`fingerprint_of`, over the prompts that produced them, is the honest choice — and never under the tool's
own, so a report built from them says which judge decided, and a run under the tool's instrument still asks
everything afresh.
"""

__all__ = ["fingerprint_of", "ImportCount", "import_prototype_judgements"]

import hashlib
import json
import pathlib
from typing import NamedTuple

from ...librarian import sweep
from . import definition as scope_definition
from . import judge


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

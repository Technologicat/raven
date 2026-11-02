"""Score `raven-scopebib` runs against a person's decisions: where each run agrees with people, and where not.

    python investigations/aokk-corpus-scope/score_against_decisions.py DECISIONS.tsv RUN_DIR [RUN_DIR ...]

`DECISIONS`: any table with `key` and `decision` columns, `keep` or `drop` in any case.
`RUN_DIR`: a run's folder, holding its `03a-CHECK-BY-HAND`, `03b-removed` and `03c-kept` tables.

Per run, a table of the person's decision against the run's outcome (removed, sent to a person, kept), and
every record on which the two disagree outright: kept what the person dropped, or removed what they kept. A
record the run sent to a person disagrees with nobody, and is counted apart.

Read the score with its reference in mind: decisions made on lists a run already sorted lean toward that run's
model, and they cover mainly the records a run or a person found hard.
"""

import argparse
import collections
import pathlib
import sys

from raven.common import tabular
from raven.papers.scopebib import report

OUTCOMES = ("removed", "to a person", "kept")


def read_table_any(run_dir: pathlib.Path, stem: str) -> list[dict[str, str]]:
    """The table `stem` in `run_dir`, in whichever format the run wrote."""
    for ext in tabular.FORMATS:
        path = run_dir / f"{stem}.{ext}"
        if path.exists():
            return tabular.read_table(path)
    raise SystemExit(f"{run_dir}: no table '{stem}'")


def outcomes_of(run_dir: pathlib.Path) -> tuple[dict[str, str], dict[str, str]]:
    """The run's outcome and a citation-like label for every record, by key."""
    outcomes, labels = {}, {}
    for stem, outcome in ((report.REMOVED_AUDIT, "removed"), (report.CHECK_BY_HAND, "to a person"), (report.KEPT_AUDIT, "kept")):
        for row in read_table_any(run_dir, stem):
            if row.get("sent by") == report.SENT_BY_EARLIER_DECISION:
                continue
            outcomes[row["key"]] = outcome
            labels[row["key"]] = f"{row.get('authors', '')[:24]} {row.get('year', '')}: {row.get('title', '')[:70]}"
    return outcomes, labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("decisions", help="a table with 'key' and 'decision' columns")
    parser.add_argument("runs", nargs="+", help="run folders")
    opts = parser.parse_args()

    decisions = {row["key"]: row["decision"].strip().lower() for row in tabular.read_table(opts.decisions)
                 if row["decision"].strip().lower() in ("keep", "drop")}
    print(f"{len(decisions)} decisions")
    for run in map(pathlib.Path, opts.runs):
        outcomes, labels = outcomes_of(run)
        counts = collections.Counter()
        kept_dropped, removed_kept, absent = [], [], []
        for key, decision in decisions.items():
            if key not in outcomes:
                absent.append(key)
                continue
            outcome = outcomes[key]
            counts[decision, outcome] += 1
            if decision == "drop" and outcome == "kept":
                kept_dropped.append(key)
            elif decision == "keep" and outcome == "removed":
                removed_kept.append(key)
        scored = len(decisions) - len(absent)
        print(f"\n===== {run.name}: {len(outcomes)} records; {scored} decisions scored, {len(absent)} on records not in the run")
        print(f"{'':>16}" + "".join(f"{o:>14}" for o in OUTCOMES))
        for decision in ("keep", "drop"):
            print(f"{'person: ' + decision:>16}" + "".join(f"{counts[decision, o]:>14}" for o in OUTCOMES))
        agreeing = counts["keep", "kept"] + counts["drop", "removed"]
        print(f"agreeing outright: {agreeing}; sent to a person: {counts['keep', 'to a person'] + counts['drop', 'to a person']}; "
              f"disagreeing outright: {len(kept_dropped) + len(removed_kept)}")
        for title, keys in (("removed what the person kept", removed_kept), ("kept what the person dropped", kept_dropped)):
            print(f"{title}:")
            for key in keys:
                print(f"    {labels[key]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Score a drop review against what the judge itself said, cell by cell.

`review_drops.py` prints one rate per group, which answers *is this reviewer discriminating at all* and
nothing finer. Two things that number cannot show, and both decide how much its findings are worth:

- **Whether the control was easier than what it is compared against.** The controls are kept records, and
  two thirds of the kept pool is high-confidence, while the drops worth reviewing are the hedged ones. A
  wide separation drawn against an easy control is partly measuring that gap rather than the reviewer.
  Splitting the control by the judge's own confidence is what tells the two apart — and a reviewer that
  finds a case for the confident keeps while hedging on the hedged ones is *agreeing with the judge*,
  which reads as a weak control and is the opposite.
- **Which drops are contested.** A drop reached from the title alone and one reached after reading the
  full abstract rest on different evidence, so a case found in each is not worth the same. They are
  separate cells here rather than one rate.

    python score_review.py drop-review-0-400.tsv drop-review-924-1124.tsv

Several slices may be scored at once. Their *dropped* cells are pooled, since coverage of the drop list is
the point of taking more slices; their controls are not, because a run may draw its control uniformly or
per stratum and pooling two sampling designs would give a rate describing neither.
"""

import argparse
import collections
import csv
import json
import pathlib
import sys

# Below this, the two groups are close enough that the reviewer is not telling them apart, and nothing it
# says about any single record carries weight. It is `review_drops`' own threshold, restated where the
# cells that explain a failure are in view.
MIN_SEPARATION = 0.20


def load_answers(path: pathlib.Path) -> dict[str, dict]:
    """The judge's final answer per citekey. A later line supersedes an earlier one."""
    answers = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            answer = json.loads(line)
            answers[answer["key"]] = answer
    return answers


def load_rows(path: pathlib.Path) -> list[dict]:
    return list(csv.DictReader(path.read_text(encoding="utf-8").splitlines(), delimiter="\t"))


def cell_of(row: dict, answers: dict[str, dict]) -> str:
    """Which cell of the judge's own output a reviewed record came from.

    Kept records are split by confidence alone: every keep was reached the same way. Dropped ones are
    split by confidence *and* by whether pass 2 ever read the abstract, which is the sharper of the two.
    """
    answer = answers.get(row["key"])
    if answer is None:
        return "(unjudged)"
    if row["group"] == "kept":
        return f"keep/{answer['confidence']}"
    return f"drop/{answer['confidence']}/{answer['source']}"


def rate_table(rows: list[dict], answers: dict[str, dict]) -> None:
    counts = collections.Counter((cell_of(row, answers), row["belongs"] == "yes") for row in rows)
    print(f"  {'the judge said':<28}{'n':>6}{'case found':>12}{'rate':>9}")
    for name in sorted({cell for cell, _ in counts}):
        yes = counts[(name, True)]
        n = yes + counts[(name, False)]
        print(f"  {name:<28}{n:>6}{yes:>12}{100 * yes / n:>8.1f}%")


def separation(rows: list[dict], answers: dict[str, dict]) -> None:
    """The comparison the run's own report makes, plus the one it cannot: against confident keeps only."""
    kept = [row for row in rows if row["group"] == "kept"]
    dropped = [row for row in rows if row["group"] == "dropped"]
    if not (kept and dropped):
        return
    def rate(subset):
        return sum(1 for row in subset if row["belongs"] == "yes") / len(subset)
    kept_rate, dropped_rate = rate(kept), rate(dropped)
    print(f"\n  kept {100 * kept_rate:.1f}%  vs  dropped {100 * dropped_rate:.1f}%   "
          f"(separation {100 * (kept_rate - dropped_rate):.1f} points)")

    # Restricting the control to the judge's confident keeps asks the narrower question: can this reviewer
    # see a case where the judge was sure there was one? A control that fails *that* is a broken
    # instrument. One that fails only on the hedged keeps is agreeing with the hedge, which is not a
    # failure at all — and the pooled rate cannot distinguish the two.
    confident = [row for row in kept if answers.get(row["key"], {}).get("confidence") == "high"]
    if confident and len(confident) < len(kept):
        print(f"  against high-confidence keeps only: {len(confident)} records, "
              f"{100 * rate(confident):.1f}%  (separation {100 * (rate(confident) - dropped_rate):.1f} "
              f"points)")
    if kept_rate - dropped_rate < MIN_SEPARATION:
        print("\n  The two rates are close: this reviewer is not telling the groups apart. Read the cells\n"
              "  above before any finding — a control made mostly of hedged keeps looks like this too.")


# Which cell a contested record came from decides how likely it is to be a real miss, and the two are far
# apart. A drop reached from the title alone is contested mostly because the title named a subject and the
# abstract named a university setting — the judge never saw the setting. A drop reached from the abstract
# is contested mostly because the reviewer reached for a tenuous educational angle. So the title-only cell
# is read first: it is where the corpus actually loses studies.
CELL_PRIORITY = {"drop/high/title": 0, "drop/medium/abstract": 1, "drop/high/abstract": 2}


def report_coverage(slices: list[tuple[pathlib.Path, list[dict]]], dropped_path: pathlib.Path) -> None:
    """How much of the drop list these slices actually cover, and whether they overlap.

    Slices are cut by `--skip`/`--limit`, so a mistyped offset leaves a gap or reviews something twice, and
    both are invisible afterwards: the rates still look fine, the contested list is still sorted, and the
    only symptom is a coverage claim that is quietly wrong. Cheap to check, so it is checked rather than
    computed by hand.
    """
    lines = [line for line in dropped_path.read_text(encoding="utf-8").splitlines()
             if not line.startswith("#")]
    all_keys = [row["key"] for row in csv.DictReader(lines, delimiter="\t") if row.get("key")]
    seen = collections.Counter(row["key"] for _, rows in slices for row in rows
                               if row["group"] == "dropped")
    covered = sum(1 for key in all_keys if key in seen)
    print(f"\ncoverage of {dropped_path.name}: {covered} of {len(all_keys)} dropped records "
          f"({100 * covered / len(all_keys):.1f}%)")
    twice = [key for key, n in seen.items() if n > 1]
    stray = [key for key in seen if key not in set(all_keys)]
    if twice:
        print(f"  {len(twice)} reviewed more than once — the slices overlap: {', '.join(twice[:5])}")
    if stray:
        print(f"  {len(stray)} reviewed but not in this drop list — a stale file somewhere")
    if not (twice or stray):
        print("  no overlaps, no strays: each reviewed record appears exactly once")


def write_contested(rows: list[dict], answers: dict[str, dict], path: pathlib.Path) -> int:
    """The hand-check list: every dropped record the reviewer made a case for, worst cell first.

    An empty first column to mark in, matching the judge's own review table. The judge's reason and the
    reviewer's case sit side by side because that pairing is the whole check — one of them is wrong, and
    which one is usually obvious from the title alone.
    """
    contested = [row for row in rows if row["group"] == "dropped" and row["belongs"] == "yes"]
    contested.sort(key=lambda row: (CELL_PRIORITY.get(cell_of(row, answers), 9), row["key"]))
    with path.open("w", encoding="utf-8") as f:
        f.write("mark\tcell\ttest\tkey\ttitle\tjudge_said\treviewer_said\n")
        for row in contested:
            answer = answers.get(row["key"], {})
            tests = ",".join(name for name in ("no_ai", "not_education", "wrong_level")
                             if answer.get(name) is True)
            f.write(f"\t{cell_of(row, answers)}\t{tests}\t{row['key']}\t{row['title']}\t"
                    f"{answer.get('why', '')}\t{row['case']}\n")
    return len(contested)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reviews", nargs="+", help="one or more drop-review TSVs")
    parser.add_argument("--judged", default=None, metavar="PATH",
                        help="the judge's state JSONL (default: judged.jsonl beside this script)")
    parser.add_argument("--dropped", nargs="?", const="dropped.tsv", default=None, metavar="PATH",
                        help="the full drop list, to report how much of it these slices cover and whether "
                             "they overlap. A mistyped --skip is invisible in the rates")
    parser.add_argument("--contested", nargs="?", const="contested.tsv", default=None, metavar="PATH",
                        help="also write the hand-check list: every dropped record a case was made for, "
                             "across all slices given, worst cell first, with an empty column to mark in")
    opts = parser.parse_args()

    here = pathlib.Path(__file__).resolve().parent
    answers = load_answers(pathlib.Path(opts.judged) if opts.judged else here / "judged.jsonl")

    slices = [(pathlib.Path(path), load_rows(pathlib.Path(path))) for path in opts.reviews]
    for path, rows in slices:
        print(f"\n{path.name}: {len(rows)} reviewed\n")
        rate_table(rows, answers)
        separation(rows, answers)

    if len(slices) > 1:
        pooled = [row for _, rows in slices for row in rows if row["group"] == "dropped"]
        yes = sum(1 for row in pooled if row["belongs"] == "yes")
        print(f"\nall slices, dropped records only: {len(pooled)} reviewed, {yes} defended "
              f"({100 * yes / len(pooled):.1f}%)\n")
        rate_table(pooled, answers)
        print("\n  Controls are not pooled: the slices drew them by different designs, so a combined\n"
              "  control rate would describe neither. Read each slice's own separation above.")

    if opts.dropped:
        dropped = pathlib.Path(opts.dropped)
        report_coverage(slices, dropped if dropped.is_absolute() or dropped.parent != pathlib.Path(".")
                        else here / dropped)

    if opts.contested:
        path = pathlib.Path(opts.contested)
        if not path.is_absolute() and path.parent == pathlib.Path("."):
            path = here / path
        n = write_contested([row for _, rows in slices for row in rows], answers, path)
        print(f"\nwrote {path}  ({n} records to hand-check)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""How far the runs of `group_studies.py` agree with each other, at the theme and at the sub-theme level.

    python investigations/llm-thematic-grouping/compare_groupings.py LEDGER.jsonl [LEDGER.jsonl ...]

Given ledgers of several conditions (one aim each), each is named by its directory, and beside each
condition's own pairwise table comes a condition-by-condition table: the mean agreement between runs of two
conditions, and on the diagonal, between runs within one. A condition that changes the grouping shows as
agreeing less with another than either does with itself.

Two measures per pair of runs, at each level, both indifferent to what a group is called:

- **The omega index** (Collins & Dent 1988), the adjusted Rand index generalized to groupings in which a
  study may sit in several groups: over every pair of studies, how often the two runs agree on *how many*
  groups the pair shares, corrected for chance. 1 is identical, 0 is chance level.
  https://doi.org/10.1207/s15327906mbr2302_6
- **The adjusted Rand index on primary placements** (Hubert & Arabie 1985), each study at the one place the
  model marked primary: the familiar number, on the partition the Visualizer plots.
  https://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index

Also a summary per run — themes, sub-themes, studies placed more than once, and the reply's problems.
"""

import argparse
import collections
import itertools
import json
import pathlib
import sys

from sklearn.metrics import adjusted_rand_score


def load_runs(ledger_path: pathlib.Path) -> dict[str, dict]:
    """The latest successful answer per run id in the ledger at `ledger_path`."""
    runs = {}
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "id" not in entry:
            continue
        if entry.get("status") == "ok":
            runs[entry["id"]] = entry["answer"]
        else:
            runs.pop(entry["id"], None)
    return dict(sorted(runs.items()))


def memberships(themes: list[dict], *, level: str) -> dict[str, set]:
    """Study key -> the set of groups it is placed in; a group is a theme index, or a (theme, sub-theme) pair."""
    out = collections.defaultdict(set)
    for t, theme in enumerate(themes):
        for s, sub in enumerate(theme["subthemes"]):
            for p in sub["studies"]:
                out[p["key"]].add(t if level == "theme" else (t, s))
    return out


def primaries(themes: list[dict], *, level: str) -> dict[str, object]:
    """Study key -> its primary group."""
    out = {}
    for t, theme in enumerate(themes):
        for s, sub in enumerate(theme["subthemes"]):
            for p in sub["studies"]:
                if p["primary"]:
                    out[p["key"]] = t if level == "theme" else (t, s)
    return out


def omega(a: dict[str, set], b: dict[str, set], keys: list[str]) -> float:
    """The omega index of two overlapping groupings of `keys`."""
    shared_a, shared_b = [], []
    for x, y in itertools.combinations(keys, 2):
        shared_a.append(len(a.get(x, set()) & a.get(y, set())))
        shared_b.append(len(b.get(x, set()) & b.get(y, set())))
    m = len(shared_a)
    observed = sum(1 for i, j in zip(shared_a, shared_b) if i == j) / m
    count_a, count_b = collections.Counter(shared_a), collections.Counter(shared_b)
    expected = sum(count_a[j] * count_b[j] for j in count_a) / m ** 2
    return 1.0 if expected == 1.0 else (observed - expected) / (1.0 - expected)


def ari(a: dict[str, object], b: dict[str, object], keys: list[str]) -> float:
    def labels(p):
        index = {}
        return [index.setdefault(p.get(k), len(index)) for k in keys]
    return adjusted_rand_score(labels(a), labels(b))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("ledgers", nargs="+", help="grouping-<fingerprint>.jsonl files written by group_studies.py")
    opts = parser.parse_args()

    conditions = {}
    for path in map(pathlib.Path, opts.ledgers):
        runs = load_runs(path)
        if not runs:
            print(f"{path}: no successful runs")
            continue
        conditions[path.resolve().parent.name] = runs
    if not conditions:
        return 1
    keys = sorted(next(iter(next(iter(conditions.values())).values()))["order"])
    for condition, runs in conditions.items():
        print(f"===== {condition}")
        summarize(runs, keys)
    if len(conditions) > 1:
        across(conditions, keys)
    return 0


def summarize(runs: dict[str, dict], keys: list[str]) -> None:
    """Per run, its shape and its themes; then the runs' pairwise agreement at both levels."""
    for run_id, answer in runs.items():
        placed = memberships(answer["themes"], level="theme")
        multi = sum(1 for k in keys if sum(1 for t in answer["themes"] for s in t["subthemes"]
                                           for p in s["studies"] if p["key"] == k) > 1)
        sizes = sorted((len(placed_keys) for placed_keys in
                        [{p["key"] for s in t["subthemes"] for p in s["studies"]} for t in answer["themes"]]),
                       reverse=True)
        print(f"{run_id}: {len(answer['themes'])} themes {sizes}, "
              f"{sum(len(t['subthemes']) for t in answer['themes'])} sub-themes, "
              f"{multi} of {len(keys)} studies placed more than once, "
              f"{sum(1 for k in keys if len(placed.get(k, ())) > 1)} under more than one theme, "
              f"{len(answer['problems'])} problems")
        for problem in answer["problems"]:
            print(f"    {problem}")
        for t, theme in enumerate(answer["themes"], start=1):
            print(f"    {t}. {theme['name']}: " + "; ".join(f"{s['name']} ({len(s['studies'])})" for s in theme["subthemes"]))

    ids = list(runs)
    if len(ids) < 2:
        print()
        return
    for level in ("theme", "sub-theme"):
        print(f"\n{level} level, pairwise: omega index / ARI on primary placements")
        member = {r: memberships(runs[r]["themes"], level=level) for r in ids}
        primary = {r: primaries(runs[r]["themes"], level=level) for r in ids}
        print("        " + "".join(f"{r:>14}" for r in ids))
        omegas, aris = [], []
        for r1 in ids:
            cells = []
            for r2 in ids:
                if r1 == r2:
                    cells.append(f"{'—':>14}")
                    continue
                o, a = omega(member[r1], member[r2], keys), ari(primary[r1], primary[r2], keys)
                if r1 < r2:
                    omegas.append(o)
                    aris.append(a)
                cells.append(f"{o:>7.2f} /{a:>5.2f}")
            print(f"{r1:>8}" + "".join(cells))
        if omegas:
            print(f"mean over {len(omegas)} pairs: omega {sum(omegas) / len(omegas):.2f}, ARI {sum(aris) / len(aris):.2f}")
    print()


def across(conditions: dict[str, dict[str, dict]], keys: list[str]) -> None:
    """Mean agreement between the runs of each pair of conditions, and within each on the diagonal."""
    names = list(conditions)
    width = max(max(len(n) for n in names), len("-0.00 / -0.00")) + 2
    for level in ("theme", "sub-theme"):
        print(f"===== across conditions, {level} level: mean omega index / mean ARI on primary placements")
        member = {(c, r): memberships(a["themes"], level=level) for c in names for r, a in conditions[c].items()}
        primary = {(c, r): primaries(a["themes"], level=level) for c in names for r, a in conditions[c].items()}
        print(" " * width + "".join(f"{n:>{width}}" for n in names))
        for c1 in names:
            cells = []
            for c2 in names:
                pairs = [(x, y) for x in member if x[0] == c1 for y in member if y[0] == c2 and x != y]
                if c1 == c2:
                    pairs = [(x, y) for x, y in pairs if x < y]
                if not pairs:
                    cells.append(f"{'—':>{width}}")
                    continue
                o = sum(omega(member[x], member[y], keys) for x, y in pairs) / len(pairs)
                a = sum(ari(primary[x], primary[y], keys) for x, y in pairs) / len(pairs)
                cells.append(f"{f'{o:.2f} / {a:.2f}':>{width}}")
            print(f"{c1:>{width}}" + "".join(cells))
        print()


if __name__ == "__main__":
    sys.exit(main())

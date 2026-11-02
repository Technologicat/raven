"""Pool the themes of several grouping runs into one catalogue, for people to edit.

Different runs of `group_studies.py` find different themes, and a complete list may need several of them, so the
unit combined here is the theme rather than the grouping. Averaging whole groupings would outvote a theme found
in only a few runs, which may be the one worth having.

    python investigations/llm-thematic-grouping/build_catalogue.py LEDGER.jsonl --bib STUDIES.bib --out catalogue

Every theme of every run becomes a point; two are close when their studies overlap (Jaccard), and average linkage
cut at `--cut` groups them into entries. Each entry says in how many runs it was found, what the runs called it,
and which studies it holds, core (in at least half of the entry's themes) or occasional.

Writes `<out>.md`, the page to read, and `<out>.json`, the entries as data. The page also lists the pairs of entries
that nearly merged, so the cut's effect can be read and overruled, and how many new entries each further run added,
which says whether more runs would be likely to add themes.
"""

import argparse
import collections
import json
import pathlib
import random
import sys

import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import compare_groupings  # noqa: E402 -- the bundle's own scripts are not a package
import group_studies  # noqa: E402 -- as above

#: Share of an entry's themes a study must sit in to count as its core.
CORE_SHARE = 0.5

#: Report a pair of entries as nearly merged when their closest themes overlap at least this much.
NEAR_MISS = 0.3


def themes_of(runs: dict[str, dict]) -> list[dict]:
    """Every theme of every run, as `{"run", "name", "description", "keys"}`, `keys` every study placed in it."""
    return [{"run": run_id, "name": theme["name"], "description": theme.get("description", ""),
             "keys": frozenset(p["key"] for sub in theme["subthemes"] for p in sub["studies"])}
            for run_id, answer in runs.items() for theme in answer["themes"]]


def jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def entries_of(themes: list[dict], *, cut: float) -> list[list[int]]:
    """Group `themes` by the overlap of their studies: lists of theme indices, the most widely found first."""
    n = len(themes)
    distance = np.array([[1.0 - jaccard(a["keys"], b["keys"]) for b in themes] for a in themes])
    np.fill_diagonal(distance, 0.0)
    labels = hierarchy.fcluster(hierarchy.linkage(squareform(distance, checks=False), method="average"),
                                t=cut, criterion="distance")
    groups = collections.defaultdict(list)
    for i in range(n):
        groups[labels[i]].append(i)
    return sorted(groups.values(), key=lambda g: (-len({themes[i]["run"] for i in g}), -len(g)))


def describe(group: list[int], themes: list[dict], n_runs: int) -> dict:
    """An entry as data: its support, the names the runs gave it, most typical first, and its studies."""
    members = [themes[i] for i in group]
    counts = collections.Counter(key for theme in members for key in theme["keys"])
    # The most typical theme overlaps most with the others in its entry, and its name goes first.
    typical = sorted(members, key=lambda t: -sum(jaccard(t["keys"], o["keys"]) for o in members))
    return {"support": len({t["run"] for t in members}), "n_runs": n_runs, "n_themes": len(members),
            "names": [t["name"] for t in typical], "description": typical[0]["description"],
            "runs": sorted({t["run"] for t in members}),
            "core": sorted((k for k, c in counts.items() if c / len(members) >= CORE_SHARE), key=lambda k: (-counts[k], k)),
            "occasional": sorted((k for k, c in counts.items() if c / len(members) < CORE_SHARE), key=lambda k: (-counts[k], k)),
            "share": {k: c / len(members) for k, c in counts.items()}}


def saturation(themes: list[dict], groups: list[list[int]], run_ids: list[str], *, trials: int = 200) -> list[float]:
    """Mean number of entries first seen at the 1st, 2nd, ... run, over random orders of the runs."""
    runs_of = [{themes[i]["run"] for i in group} for group in groups]
    total = np.zeros(len(run_ids))
    rng = random.Random(0)
    for _ in range(trials):
        order = rng.sample(run_ids, len(run_ids))
        seen = [False] * len(groups)
        for position, run_id in enumerate(order):
            for g, runs in enumerate(runs_of):
                if not seen[g] and run_id in runs:
                    seen[g] = True
                    total[position] += 1
    return list(total / trials)


def near_misses(themes: list[dict], groups: list[list[int]]) -> list[tuple[int, int, float]]:
    """Pairs of entries (their positions) whose closest themes overlap at least `NEAR_MISS`, closest first."""
    pairs = []
    for a in range(len(groups)):
        for b in range(a + 1, len(groups)):
            best = max(jaccard(themes[i]["keys"], themes[j]["keys"]) for i in groups[a] for j in groups[b])
            if best >= NEAR_MISS:
                pairs.append((a, b, best))
    return sorted(pairs, key=lambda p: -p[2])


def page(entries: list[dict], names: dict[str, str], *, source: str, cut: float, new_per_run: list[float],
         nearly: list[tuple[int, int, float]], unplaced: list[str]) -> str:
    """The catalogue as Markdown."""
    def cite(keys: list[str]) -> str:
        return "; ".join(names[k] for k in keys) or "—"

    out = [f"# Draft theme catalogue: {source}", "",
           f"Pooled from the themes of {entries[0]['n_runs']} runs, grouped by how much their studies overlap (cut at {cut}).",
           "For people to keep, merge, split and rename. **A low count is not a verdict**: a theme found in few runs may",
           "still be one the review needs.", "",
           "| # | Entry, by its most typical name | Found in | Core studies |", "|---|---|---|---|"]
    for i, e in enumerate(entries, start=1):
        out.append(f"| {i} | {e['names'][0]} | {e['support']} of {e['n_runs']} | {len(e['core'])} |")
    out += ["", "## The entries", ""]
    for i, e in enumerate(entries, start=1):
        out += [f"### {i}. {e['names'][0]}", "",
                f"Found in {e['support']} of {e['n_runs']} runs.", "", f"*{e['description']}*", ""]
        others = list(dict.fromkeys(e["names"][1:]))
        if others:
            out += ["Other names: " + "; ".join(others[:4]) + ("; …" if len(others) > 4 else ""), ""]
        out += [f"- **Core ({len(e['core'])}):** {cite(e['core'])}"]
        if e["occasional"]:
            out += [f"- **Occasional ({len(e['occasional'])}):** " +
                    "; ".join(f"{names[k]} {e['share'][k]:.0%}" for k in e["occasional"])]
        out.append("")
    out += ["## Entries that nearly merged", "",
            "A pair here may be one theme split in two, or two themes one run lumped together.", ""]
    out += [f"- {a + 1} and {b + 1}, closest themes overlapping {overlap:.0%}" for a, b, overlap in nearly] or ["None."]
    out += ["", "## Would more runs add themes?", "",
            "Mean new entries added by each further run, over random orders of the runs: " +
            ", ".join(f"{x:.1f}" for x in new_per_run) + ".", ""]
    if unplaced:
        out += ["## Studies in no entry's core", "", cite(unplaced), ""]
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("ledger", help="a grouping-<fingerprint>.jsonl written by group_studies.py")
    parser.add_argument("--bib", required=True, help="the grouped .bib, to name studies by their citations")
    parser.add_argument("--out", required=True, help="where to write: <out>.md and <out>.json")
    parser.add_argument("--cut", type=float, default=0.7,
                        help="average-linkage distance at which themes stop merging, 1 - Jaccard (default: 0.7)")
    opts = parser.parse_args()

    runs = compare_groupings.load_runs(pathlib.Path(opts.ledger))
    studies = group_studies.load_studies(pathlib.Path(opts.bib))
    names = {s.key: s.citation for s in studies}
    themes = themes_of(runs)
    groups = entries_of(themes, cut=opts.cut)
    entries = [describe(group, themes, len(runs)) for group in groups]
    core = {k for e in entries for k in e["core"]}
    text = page(entries, names, source=pathlib.Path(opts.bib).name, cut=opts.cut,
                new_per_run=saturation(themes, groups, list(runs)), nearly=near_misses(themes, groups),
                unplaced=sorted(k for k in names if k not in core))
    out = pathlib.Path(opts.out)
    md, data = out.with_name(f"{out.name}.md"), out.with_name(f"{out.name}.json")
    md.write_text(text, encoding="utf-8")
    data.write_text(json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(themes)} themes from {len(runs)} runs -> {len(entries)} entries: {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

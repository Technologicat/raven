"""Compare LLM cluster keywords with and without the phrase evidence, on one corpus.

Usage:

    python investigations/cluster-keywords/compare_llm_runs.py CORPUS [--show 15]

Reads `data/CORPUS.pickle` (LLM keywords, no evidence, from `run_llm_imports.sh`), `data/CORPUS-evidence.pickle`
(with the evidence) and, where it exists, `data/CORPUS-repeat.pickle` (no evidence, run again), all from
`run_llm_evidence_imports.sh`. The three must have the same clusters, which agglomerative clustering on cached
embeddings gives.

Reports, and writes to `out/CORPUS-llm-evidence.md`:

    agreement   per cluster, the overlap of two runs' keyword sets (Jaccard), averaged over clusters: the two
                runs without the evidence against each other (the noise floor, where a repeat exists), and
                with the evidence against without
    anchoring   the share of a run's keywords found among the cluster's phrases (the 30 the evidence lists),
                with the evidence and without
    labels      per cluster, side by side: the phrase label, and the LLM's without and with the evidence

Keywords are compared lowercased, with a final "s" dropped, so "Large Language Models" matches "large
language model".
"""

import argparse
import pathlib
import pickle
import random
import statistics

HERE = pathlib.Path(__file__).resolve().parent


def norm(keyword: str) -> str:
    k = keyword.strip().lower()
    return k[:-1] if k.endswith("s") and not k.endswith(("ss", "ics")) else k


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = {norm(k) for k in a}, {norm(k) for k in b}
    return len(sa & sb) / len(sa | sb) if sa | sb else 1.0


def anchored(keywords: list[str], phrases: list[str]) -> float:
    pool = {norm(p) for p in phrases}
    return sum(1 for k in keywords if norm(k) in pool) / len(keywords) if keywords else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus")
    parser.add_argument("--show", type=int, default=15, help="clusters shown, of the largest and of a random draw each")
    args = parser.parse_args()

    def load(suffix):
        path = HERE / "data" / f"{args.corpus}{suffix}.pickle"
        if not path.exists():
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    without, with_ev, repeat = load(""), load("-evidence"), load("-repeat")
    if without is None or with_ev is None:
        raise SystemExit(f"need data/{args.corpus}.pickle and data/{args.corpus}-evidence.pickle")
    for other, name in ((with_ev, "evidence"), (repeat, "repeat")):
        if other is not None and list(other["labels"]) != list(without["labels"]):
            raise SystemExit(f"the {name} run's clusters differ from the first run's; the labels cannot be compared")

    lists = with_ev["cluster_keyword_lists"]
    n = with_ev["n_vis_clusters"]
    clusters = [c for c in range(n) if lists[c]]
    phrases = {c: [p for p, _ in lists[c]] for c in clusters}
    kw = {"without": without["vis_keywords_by_cluster"], "with": with_ev["vis_keywords_by_cluster"]}
    if repeat is not None:
        kw["repeat"] = repeat["vis_keywords_by_cluster"]

    def mean(values):
        return statistics.fmean(values) if values else float("nan")

    lines = [f"# LLM keywords with and without the phrase evidence: {args.corpus}", "",
             f"{len(without['vis_data'])} records, {n} clusters.", "",
             "| comparison | mean keyword overlap (Jaccard) |", "|---|---|"]
    if repeat is not None:
        lines.append(f"| without, run twice (noise floor) | {mean([jaccard(kw['without'][c], kw['repeat'][c]) for c in clusters]):.2f} |")
    lines.append(f"| with against without | {mean([jaccard(kw['with'][c], kw['without'][c]) for c in clusters]):.2f} |")
    if repeat is not None:
        lines.append(f"| with against the repeat | {mean([jaccard(kw['with'][c], kw['repeat'][c]) for c in clusters]):.2f} |")
    lines += ["", "| run | keywords found among the cluster's phrases |", "|---|---|"]
    for name in kw:
        lines.append(f"| {name} | {mean([anchored(kw[name][c], phrases[c]) for c in clusters]):.0%} |")
    lines.append("")

    sizes = {c: sum(1 for x in without["labels"] if x == c) for c in clusters}
    by_size = sorted(clusters, key=lambda c: -sizes[c])
    largest = by_size[:args.show]
    rest = [c for c in by_size if c not in largest]
    drawn = sorted(random.Random(0).sample(rest, min(args.show, len(rest))), key=lambda c: -sizes[c])
    records = without["vis_data"]
    for title, cids in (("The largest clusters", largest), ("A random draw of the rest", drawn)):
        lines += [f"## {title}", ""]
        for c in cids:
            lines += [f"### Cluster {c} ({sizes[c]} records)", "",
                      f"- **phrases**: {', '.join(phrases[c][:6])}",
                      f"- **llm, without**: {', '.join(kw['without'][c])}"]
            if "repeat" in kw:
                lines.append(f"- **llm, without, again**: {', '.join(kw['repeat'][c])}")
            lines.append(f"- **llm, with evidence**: {', '.join(kw['with'][c])}")
            members = [r for r in records if r.cluster_id == c]
            for r in random.Random(c).sample(members, min(3, len(members))):
                lines.append(f"  - *{r.title}*")
            lines.append("")

    out = HERE / "out" / f"{args.corpus}-llm-evidence.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:14]))
    print(f"\nReport: {out}")


if __name__ == "__main__":
    main()

"""Which clusters of a corpus are worth a person's reading after a new import? A short list, picked by script.

Reading every cluster takes long for little gain, so the clusters read on arXiv calibrate which to pick. What made
a cluster worth reading there: the phrase and LLM labels disagreeing, a theme gained or lost between runs, the
corpus-wide term leading, a duplicate spelling. The picks, each with its reason:

    changed     the clusters whose LLM label changed most between the earlier import and the new one, or whose
                phrase label did (spelling folds, capitals)
    disagree    the clusters where the new phrase and LLM labels share the fewest keywords
    largest     the largest clusters, which hold the most records
    read        on arXiv, clusters the maintainer read on 2026-11-24, the most changed of them, as regression
                checks; cheap to judge, being known

Usage, from the repository root, once `run_new_imports.sh` has run:

    python investigations/cluster-keywords/shortlist_clusters.py CORPUS [CORPUS ...] [--per-reason 4]

Reads `data/CORPUS-evidence-2026-11-23.pickle` (the earlier import), `data/CORPUS-evidence.pickle` and
`data/CORPUS-hybrid.pickle`; the three must have the same clusters. Writes `out/CORPUS-shortlist.md`, each cluster
with why it was picked, its labels side by side, and a few of its titles.
"""

import argparse
import pathlib
import pickle
import random

from compare_llm_runs import jaccard

HERE = pathlib.Path(__file__).resolve().parent

# Clusters of the arXiv corpus the maintainer read on 2026-11-24 (this bundle's README).
READ = {"arxiv": {2, 3, 9, 15, 16, 19, 20, 27, 33, 35, 36, 37, 38, 44, 45, 49, 51, 52, 54, 56, 57, 59, 60, 65}}


def load(name):
    with open(HERE / "data" / f"{name}.pickle", "rb") as f:
        return pickle.load(f)


def phrase_label(dataset, cluster, n=6):
    lists = dataset.get("cluster_keyword_lists")
    return [p for p, _ in lists[cluster][:n]] if lists and lists[cluster] else []


def shortlist(corpus, per_reason):
    old, new, hybrid = load(f"{corpus}-evidence-2026-11-23"), load(f"{corpus}-evidence"), load(f"{corpus}-hybrid")
    for other, name in ((new, "evidence"), (hybrid, "hybrid")):
        if list(other["labels"]) != list(old["labels"]):
            raise SystemExit(f"{corpus}: the {name} import's clusters differ from the earlier one's; nothing to compare")
    n = new["n_vis_clusters"]
    sizes = {c: sum(1 for x in new["labels"] if x == c) for c in range(n)}
    clusters = [c for c in range(n) if sizes[c]]

    llm_change = {c: 1 - jaccard(old["vis_keywords_by_cluster"][c], new["vis_keywords_by_cluster"][c]) for c in clusters}
    phrase_change = {c: 1 - jaccard(phrase_label(old, c), phrase_label(new, c)) for c in clusters}
    disagreement = {c: 1 - jaccard(phrase_label(new, c), new["vis_keywords_by_cluster"][c]) for c in clusters}

    def top(score, k, among=None):
        pool = [c for c in (among if among is not None else clusters) if score[c] > 0]
        return sorted(pool, key=lambda c: (-score[c], -sizes[c]))[:k]

    reasons = {}

    def add(cs, reason):
        for c in cs:
            reasons.setdefault(c, []).append(reason)

    change = {c: max(llm_change[c], phrase_change[c]) for c in clusters}
    add(top(change, per_reason), "changed")
    add(top(disagreement, per_reason), "disagree")
    add(sorted(clusters, key=lambda c: -sizes[c])[:max(2, per_reason // 2)], "largest")
    if corpus in READ:
        add(top(change, per_reason, among=[c for c in clusters if c in READ[corpus]]), "read")

    lines = [f"# Clusters worth reading: {corpus}", "",
             f"{len(new['vis_data'])} records, {len(clusters)} clusters; {len(reasons)} picked. *changed*: the label "
             f"moved most since the earlier import; *disagree*: the phrase and LLM labels share least; *largest*; "
             f"*read*: read before, the most changed of those.", ""]
    for c in sorted(reasons, key=lambda c: -sizes[c]):
        lines += [f"## Cluster {c} ({sizes[c]} records): {', '.join(reasons[c])}", "",
                  f"- **phrases, new**: {', '.join(phrase_label(new, c))}",
                  f"- **phrases, earlier**: {', '.join(phrase_label(old, c))}",
                  f"- **llm with evidence, new**: {', '.join(new['vis_keywords_by_cluster'][c])}",
                  f"- **llm with evidence, earlier**: {', '.join(old['vis_keywords_by_cluster'][c])}",
                  f"- **hybrid, new**: {', '.join(hybrid['vis_keywords_by_cluster'][c])}"]
        members = [r for r in new["vis_data"] if r.cluster_id == c]
        for r in random.Random(c).sample(members, min(3, len(members))):
            lines.append(f"  - *{r.title}*")
        lines.append("")
    out = HERE / "out" / f"{corpus}-shortlist.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"{corpus}: {len(reasons)} of {len(clusters)} clusters picked -> {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpora", nargs="+")
    parser.add_argument("--per-reason", type=int, default=4, help="clusters picked for each reason, at most")
    args = parser.parse_args()
    for corpus in args.corpora:
        shortlist(corpus, args.per_reason)


if __name__ == "__main__":
    main()

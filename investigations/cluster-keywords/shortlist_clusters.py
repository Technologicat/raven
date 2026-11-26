"""Which clusters of a corpus are worth a person's reading after a new import? A short list, picked by script.

Each cluster shows its keywords as the Visualizer shows them, **Themes** (the LLM's) and **Quoted** (the phrases),
through the app's own formatter (`raven.visualizer.entry_renderer.format_cluster_keywords`), beside the same from
the previous import.

Reading every cluster takes long for little gain, so the clusters read on arXiv calibrate which to pick. What made
a cluster worth reading there: the phrase and LLM keywords disagreeing, a theme gained or lost between runs, the
corpus-wide term leading, a duplicate spelling. The picks, each with its reason:

    changed     the clusters whose Themes and Terms changed most since the previous import
    disagree    the clusters where the LLM's keywords and the phrases share the fewest terms
    largest     the largest clusters, which hold the most records
    read        on arXiv, clusters the maintainer read on 2026-11-24, the most changed of them, as regression
                checks; cheap to judge, being known

Usage, from the repository root, once `run_new_imports.sh` has run:

    python investigations/cluster-keywords/shortlist_clusters.py CORPUS [CORPUS ...] [--per-reason 4]
                                                                 [--previous 2026-11-24]

Reads `data/CORPUS-evidence.pickle` and `data/CORPUS-evidence-PREVIOUS.pickle`; the two must have the same
clusters. Writes `out/CORPUS-shortlist.md`, each cluster with why it was picked, its keywords now and before, and a
few of its titles.
"""

import argparse
import pathlib
import pickle
import random

from unpythonic.env import env

from raven.visualizer import entry_renderer

from compare_llm_runs import jaccard

HERE = pathlib.Path(__file__).resolve().parent

# Clusters of the arXiv corpus the maintainer read on 2026-11-24 (this bundle's README).
READ = {"arxiv": {2, 3, 9, 15, 16, 19, 20, 27, 33, 35, 36, 37, 38, 44, 45, 49, 51, 52, 54, 56, 57, 59, 60, 65}}


def load(name):
    with open(HERE / "data" / f"{name}.pickle", "rb") as f:
        dataset = pickle.load(f)
    # An import from before the LLM's whole list was kept: its label is the LLM's list, cut at six.
    if dataset.get("cluster_llm_keywords") is None and dataset.get("clusters_keyword_method") == "llm":
        dataset["cluster_llm_keywords"] = dataset["vis_keywords_by_cluster"]
    return dataset


def keyword_lines(dataset, cluster):
    """The cluster's keyword lines as the Visualizer shows them, as `[(caption, text), ...]`."""
    return entry_renderer.format_cluster_keywords(env(**dataset), cluster)


def terms_of(lines):
    return [t for _, text in lines for t in text.strip("[]").split(", ") if t]


def shortlist(corpus, per_reason, previous):
    prev, new = load(f"{corpus}-evidence-{previous}"), load(f"{corpus}-evidence")
    if list(prev["labels"]) != list(new["labels"]):
        raise SystemExit(f"{corpus}: the previous import's clusters differ from the new one's; nothing to compare")
    n = new["n_vis_clusters"]
    sizes = {c: sum(1 for x in new["labels"] if x == c) for c in range(n)}
    clusters = [c for c in range(n) if sizes[c]]

    lines_new = {c: keyword_lines(new, c) for c in clusters}
    lines_prev = {c: keyword_lines(prev, c) for c in clusters}
    change = {c: 1 - jaccard(terms_of(lines_prev[c]), terms_of(lines_new[c])) for c in clusters}
    lists = new.get("cluster_keyword_lists") or [[]] * n
    llm = new.get("cluster_llm_keywords") or [[]] * n
    disagreement = {c: 1 - jaccard([p for p, _ in lists[c][:15]], llm[c]) for c in clusters}

    def top(score, k, among=None):
        pool = [c for c in (among if among is not None else clusters) if score[c] > 0]
        return sorted(pool, key=lambda c: (-score[c], -sizes[c]))[:k]

    reasons = {}

    def add(cs, reason):
        for c in cs:
            reasons.setdefault(c, []).append(reason)

    add(top(change, per_reason), "changed")
    add(top(disagreement, per_reason), "disagree")
    add(sorted(clusters, key=lambda c: -sizes[c])[:max(2, per_reason // 2)], "largest")
    if corpus in READ:
        add(top(change, per_reason, among=[c for c in clusters if c in READ[corpus]]), "read")

    def show(lines):
        return [f"  - **{caption}**: {text}" if caption else f"  - {text}" for caption, text in lines]

    lines = [f"# Clusters worth reading: {corpus}", "",
             f"{len(new['vis_data'])} records, {len(clusters)} clusters; {len(reasons)} picked. *changed*: the keywords "
             f"moved most since the import of {previous}; *disagree*: the LLM's keywords and the phrases share least; "
             f"*largest*; *read*: read before, the most changed of those.", ""]
    for c in sorted(reasons, key=lambda c: -sizes[c]):
        lines += [f"## Cluster {c} ({sizes[c]} records): {', '.join(reasons[c])}", "", "- **Now**"]
        lines += show(lines_new[c])
        lines += [f"- **Before** ({previous})"] + show(lines_prev[c])
        members = [r for r in new["vis_data"] if r.cluster_id == c]
        for r in random.Random(c).sample(members, min(3, len(members))):
            lines.append(f"- *{r.title}*")
        lines.append("")
    out = HERE / "out" / f"{corpus}-shortlist.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"{corpus}: {len(reasons)} of {len(clusters)} clusters picked -> {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpora", nargs="+")
    parser.add_argument("--per-reason", type=int, default=4, help="clusters picked for each reason, at most")
    parser.add_argument("--previous", default="2026-11-24", help="the date suffix of the import to compare against")
    args = parser.parse_args()
    for corpus in args.corpora:
        shortlist(corpus, args.per_reason, args.previous)


if __name__ == "__main__":
    main()

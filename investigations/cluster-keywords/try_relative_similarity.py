"""Does ranking terms by their similarity to their own cluster *relative to the others* undo the single-word bias?

`measure_single_words.py` found that the phrase labeller's similarity order demotes single-word terms far more than
their specificity warrants: a short text embeds far from every centroid. This builds each cluster's label as
`keyphrases.label_groups` does, with the similarity order computed three ways:

    raw        cosine to the cluster's centroid, as shipped
    relative   that cosine minus the term's mean cosine to all the clusters' centroids
    z          the same difference divided by the standard deviation of those cosines

and reports, per corpus and variant: of the terms in a pool's first three places by c-TF-IDF, the share of the
single words the label keeps (the bias), the same for longer terms, and how many labels change. Then, for the
arXiv corpus, every label that a variant changes, side by side, for reading.

Usage, from the repository root:

    python investigations/cluster-keywords/try_relative_similarity.py

Uses the corpora and caches of `measure_single_words.py`. Writes `out/relative-similarity.md`.
"""

import collections
import pathlib
import pickle

import numpy as np

from raven.client import api
from raven.client import config as client_config
from raven.common import keyphrases
from raven.common.text import stopwords
from raven.visualizer import importer

import measure_single_words as msw

HERE = pathlib.Path(__file__).resolve().parent
VARIANTS = ("raw", "relative", "z")


def labels_by_variant(name, caches, embedder, shown=6, pool_size=30):
    """`{variant: {cluster: [display term, ...]}}`, the bias counts, and the display map."""
    with open(HERE / "data" / f"{name}-evidence.pickle", "rb") as f:
        dataset = pickle.load(f)
    vectors = np.concatenate([np.load(c, allow_pickle=True)["all_vectors"] for c in caches])
    records = dataset["vis_data"]
    assert len(vectors) == len(records), f"{name}: the caches do not match the records"
    display = dataset["keyphrase_display"]
    terms = [set(getattr(r, "keyphrases", None) or []) for r in records]
    groups = [int(r.cluster_id) if r.cluster_id >= 0 else None for r in records]

    ranked = keyphrases.rank_terms(terms, groups, min_support=2)
    umbrella = keyphrases.umbrella_terms(terms, groups, min_support=2, threshold_fraction=0.1)
    members = collections.defaultdict(list)
    for idx, g in enumerate(groups):
        if g is not None:
            members[g].append(idx)
    cluster_ids = sorted(members)
    centroids = np.stack([vectors[members[g]].mean(axis=0) for g in cluster_ids])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

    heads, pools = {}, {}
    for g in cluster_ids:
        candidates = [item for item in ranked.get(g, []) if item[0] not in stopwords.bare_keyword_stopwords]
        heads[g] = keyphrases._pick(candidates, 1)
        pools[g] = [item for item in candidates
                    if item[0] not in umbrella and not any(keyphrases._nested(item[0], t) for t, _ in heads[g])][:pool_size]
    unique = sorted({t for items in pools.values() for t, _, _ in items})
    embedded = np.asarray(embedder.encode(unique), dtype=np.float32)
    embedded /= np.linalg.norm(embedded, axis=1, keepdims=True)
    term_vectors = dict(zip(unique, embedded))

    labels = {v: {} for v in VARIANTS}
    bias = {v: collections.defaultdict(lambda: [0, 0]) for v in VARIANTS}  # word class -> [kept, total]
    for g in cluster_ids:
        pool = pools[g]
        if not pool:
            for v in VARIANTS:
                labels[v][g] = [display.get(t, t) for t, _ in heads[g]]
            continue
        to_all = np.stack([term_vectors[t] for t, _, _ in pool]) @ centroids.T
        own = to_all[:, cluster_ids.index(g)]
        scores = {"raw": own,
                  "relative": own - to_all.mean(axis=1),
                  "z": (own - to_all.mean(axis=1)) / np.maximum(to_all.std(axis=1), 1e-9)}
        for v in VARIANTS:
            order = np.argsort(-scores[v], kind="stable")
            fused = keyphrases.fuse([pool, [pool[k] for k in order]])
            label = heads[g] + keyphrases._pick(fused, shown - len(heads[g]), fixed=heads[g])
            labels[v][g] = [display.get(t, t) for t, _ in label]
            chosen = {t for t, _ in label}
            for t, _, _ in pool[:3]:
                wc = "1" if len(t.split()) == 1 else "2+"
                bias[v][wc][0] += t in chosen
                bias[v][wc][1] += 1
    return labels, bias


def main():
    api.initialize(raven_server_url=client_config.raven_server_url,
                   maybe_network_retry=client_config.network_retry_batch,
                   raven_api_key_file=client_config.raven_api_key_file)
    embedder = importer._make_embedder()

    def share(kt):
        return f"{kt[0]}/{kt[1]} ({kt[0] / kt[1]:.0%})" if kt[1] else "–"

    lines = ["# Ranking by similarity relative to the other clusters", "",
             "Of the terms in a pool's first three places by c-TF-IDF, the share the six-term label keeps, single "
             "words and longer terms; and how many labels differ from the shipped (raw) ones.", "",
             "| corpus | variant | single words kept | longer terms kept | labels changed |", "|---|---|---|---|---|"]
    arxiv_labels = None
    for name, caches in msw.CORPORA.items():
        labels, bias = labels_by_variant(name, caches, embedder)
        for v in VARIANTS:
            changed = sum(1 for g in labels["raw"] if labels[v][g] != labels["raw"][g])
            lines.append(f"| {name} | {v} | {share(bias[v]['1'])} | {share(bias[v]['2+'])} | "
                         f"{changed} of {len(labels['raw'])} |")
        print("\n".join(lines[-3:]), flush=True)
        if name == "arxiv":
            arxiv_labels = labels

    lines += ["", "## arXiv: the labels a variant changes", ""]
    for g in sorted(arxiv_labels["raw"]):
        if all(arxiv_labels[v][g] == arxiv_labels["raw"][g] for v in VARIANTS):
            continue
        lines += [f"### Cluster {g}", ""] + [f"- **{v}**: {', '.join(arxiv_labels[v][g])}" for v in VARIANTS] + [""]
    out = HERE / "out" / "relative-similarity.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"\nReport: {out}")


if __name__ == "__main__":
    main()

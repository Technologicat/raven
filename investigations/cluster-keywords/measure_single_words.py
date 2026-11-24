"""Does the phrase labeller's similarity order work against single-word terms?

Cluster 33 of the arXiv corpus lost *hallucination*: second by c-TF-IDF, at the cluster's top share, yet 24th of 30
by cosine similarity to the cluster's centroid, so the fusion dropped it. Its two bare single words sat lowest in
that order. This asks whether that holds across corpora.

For each dataset, rebuilding each cluster's tail pool as `keyphrases.label_groups` does, three measures by the
number of words in a term (1, 2, 3 or more; a hyphenated word counts as one):

    cosine        the mean cosine between a pooled term and its cluster's centroid
    percentile    the term's place in its own cluster's similarity order, 0 lowest and 1 highest, averaged; this
                  removes differences between clusters
    top-3 kept    of the terms in the first three places of a pool by c-TF-IDF, the share that the shown label
                  (six terms, head included) keeps; the effect that matters for a reader
    own rank      the control: where the term places its own cluster among all the clusters' centroids, 1 when
                  its own is the nearest and 0 when the farthest, averaged. A generic term sits near many
                  centroids and places its own low; a specific one places it high whatever its absolute cosine.
                  So where single words hold their own rank while falling in percentile, the fall comes from the
                  embedding of a short text rather than from genericness.

Usage, from the repository root:

    python investigations/cluster-keywords/measure_single_words.py

Reads the `-evidence` datasets in `data/` and their embeddings caches; embeds the pooled terms through
Raven-server, or locally where it does not answer. Writes `out/single-words.md`.
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

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
R = ROOT / "00_stuff" / "rawdata"
HYDROGEN = R / "100000_most_relevant_refs_of_hydrogen_productionzip"

# Each corpus's embeddings caches, in the order the import read their BibTeX files (`run_llm_evidence_imports.sh`).
CORPORA = {
    "arxiv": [ROOT / "00_stuff/datasets/ai_papers/ai_papers_202611_embeddings_cache.npz"],
    "aokk": [R / "AOKK/scopebib-2026-11-full/run-without-retracted/kept-and-held_embeddings_cache.npz"],
    "eccomas": [ROOT / "00_stuff/datasets/ECCOMAS2024/success_final_manually_fixed_and_added_missing_abstracts_embeddings_cache.npz"],
    "hydrogen": [HYDROGEN / "savedrecs_embeddings_cache.npz", HYDROGEN / "savedrecs-2_embeddings_cache.npz",
                 HYDROGEN / "savedrecs-3_embeddings_cache.npz"],
    "biology": [R / "biology_geography_health_education/202508/biology_embeddings_cache.npz"],
    "geography": [R / "biology_geography_health_education/202512/geography_embeddings_cache.npz"],
}


def word_class(term: str) -> str:
    n = len(term.split())
    return "1" if n == 1 else "2" if n == 2 else "3+"


def measure(name, caches, embedder, shown=6, pool_size=30):
    with open(HERE / "data" / f"{name}-evidence.pickle", "rb") as f:
        dataset = pickle.load(f)
    vectors = np.concatenate([np.load(c, allow_pickle=True)["all_vectors"] for c in caches])
    records = dataset["vis_data"]
    if len(vectors) != len(records):
        raise SystemExit(f"{name}: {len(vectors)} vectors for {len(records)} records; the caches do not match")
    terms = [set(getattr(r, "keyphrases", None) or []) for r in records]
    groups = [int(r.cluster_id) if r.cluster_id >= 0 else None for r in records]

    ranked = keyphrases.rank_terms(terms, groups, min_support=2)
    umbrella = keyphrases.umbrella_terms(terms, groups, min_support=2, threshold_fraction=0.1)
    members = collections.defaultdict(list)
    for idx, g in enumerate(groups):
        if g is not None:
            members[g].append(idx)

    heads, pools = {}, {}
    for g in members:
        candidates = [item for item in ranked.get(g, []) if item[0] not in stopwords.bare_keyword_stopwords]
        heads[g] = keyphrases._pick(candidates, 1)
        head_terms = [t for t, _ in heads[g]]
        pools[g] = [item for item in candidates
                    if item[0] not in umbrella and not any(keyphrases._nested(item[0], t) for t in head_terms)][:pool_size]

    unique = sorted({t for items in pools.values() for t, _, _ in items})
    embedded = np.asarray(embedder.encode(unique), dtype=np.float32)
    embedded /= np.linalg.norm(embedded, axis=1, keepdims=True)
    term_vectors = dict(zip(unique, embedded))

    centroids = {}
    for g, idxs in members.items():
        c = vectors[idxs].mean(axis=0)
        centroids[g] = c / np.linalg.norm(c)
    cluster_ids = sorted(centroids)
    all_centroids = np.stack([centroids[g] for g in cluster_ids])

    cosines, percentiles = collections.defaultdict(list), collections.defaultdict(list)
    own_ranks = collections.defaultdict(list)
    top3 = collections.defaultdict(lambda: [0, 0])  # [kept, total]
    for g, idxs in members.items():
        pool = pools[g]
        if len(pool) < 2:
            continue
        centroid = centroids[g]
        pool_vectors = np.stack([term_vectors[t] for t, _, _ in pool])
        sim = pool_vectors @ centroid
        to_all = pool_vectors @ all_centroids.T  # (terms, clusters)
        own = cluster_ids.index(g)
        for k, (t, _, _) in enumerate(pool):
            nearer = int((to_all[k] > to_all[k, own]).sum())
            own_ranks[word_class(t)].append(1.0 - nearer / (len(cluster_ids) - 1))
        order = np.argsort(-sim, kind="stable")
        place = np.empty(len(pool))
        place[order] = np.arange(len(pool))
        for k, (t, _, _) in enumerate(pool):
            cosines[word_class(t)].append(float(sim[k]))
            percentiles[word_class(t)].append(1.0 - place[k] / (len(pool) - 1))
        fused = keyphrases.fuse([pool, [pool[k] for k in order]])
        label = {t for t, _ in keyphrases._pick(fused, shown - len(heads[g]), fixed=heads[g])}
        for t, _, _ in pool[:3]:
            top3[word_class(t)][0] += t in label
            top3[word_class(t)][1] += 1
    return cosines, percentiles, top3, own_ranks


def main():
    api.initialize(raven_server_url=client_config.raven_server_url,
                   maybe_network_retry=client_config.network_retry_batch,
                   raven_api_key_file=client_config.raven_api_key_file)
    embedder = importer._make_embedder()

    lines = ["# Does the similarity order work against single-word terms?", "",
             "By the number of words in a pooled term. `n` is how many pooled terms; *top-3 kept* is the share of the "
             "terms in a pool's first three places by c-TF-IDF that the six-term label keeps.", "",
             "| corpus | words | n | cosine | percentile | top-3 kept | own rank |", "|---|---|---|---|---|---|---|"]
    for name, caches in CORPORA.items():
        cosines, percentiles, top3, own_ranks = measure(name, caches, embedder)
        for wc in ("1", "2", "3+"):
            if cosines[wc]:
                kept, total = top3[wc]
                kept_text = f"{kept}/{total} ({kept / total:.0%})" if total else "–"
                lines.append(f"| {name} | {wc} | {len(cosines[wc])} | {np.mean(cosines[wc]):.3f} | "
                             f"{np.mean(percentiles[wc]):.2f} | {kept_text} | {np.mean(own_ranks[wc]):.2f} |")
        print("\n".join(lines[-3:]), flush=True)
    out = HERE / "out" / "single-words.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"\nReport: {out}")


if __name__ == "__main__":
    main()

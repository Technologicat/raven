"""Does the embedding rerank favour generic terms, and is that because cluster centroids lie near the corpus mean?

For each dataset given, two measures, from the caches `keyword_probe.py --rerank` leaves behind:

    centroid closeness   the mean cosine between each cluster's centroid and the corpus mean
    generality bias      per cluster, the Spearman correlation between a pooled term's similarity to the
                         cluster's centroid and the number of clusters the term is present in (in two or
                         more records), averaged over clusters; positive means the rerank prefers terms
                         many clusters share. Raw, and with the centering `keyword_probe.rerank` applies.
    clusters per pick    how many clusters the six terms a method picks are present in, on average:
                         c-TF-IDF's top six against the rerank's

Usage:

    python investigations/cluster-keywords/measure_centroids.py DATASET.pickle [...]

Run `keyword_probe.py --rerank` on each dataset first; this script embeds nothing.
"""

import collections
import pathlib
import pickle
import sys

import numpy as np
import scipy.stats

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import keyword_probe as kp  # noqa: E402 -- after the path setup


def measure(dataset_path: pathlib.Path, pool: int = 30) -> dict:
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)
    records = dataset["vis_data"]
    labels = [entry.cluster_id for entry in records]
    members = collections.defaultdict(list)
    for idx, label in enumerate(labels):
        if label >= 0:
            members[label].append(idx)

    candidates = kp.extract(dataset_path, records)
    phrases = kp.fold_plurals(candidates["phrase"])
    phrase_ab, _display = kp.expand_abbreviations(phrases, candidates["abbreviations"])
    ranked = kp.ctfidf(phrase_ab, labels)

    stem = dataset_path.stem
    with open(HERE / "cache" / f"{stem}-docvectors.pickle", "rb") as f:
        doc_cache = pickle.load(f)
    with open(HERE / "cache" / f"{stem}-termvectors.pickle", "rb") as f:
        term_cache = pickle.load(f)
    doc_vectors = np.stack([doc_cache[kp.record_text(entry)] for entry in records])

    # How many clusters each term is present in, by the same bar as the umbrella rule.
    tf = collections.defaultdict(collections.Counter)
    for terms, label in zip(phrase_ab, labels):
        if label >= 0:
            tf[label].update(terms)
    presence = collections.Counter()
    for counts in tf.values():
        presence.update(t for t, c in counts.items() if c >= 2)

    pools = {cid: [t for t, _, _ in ranked.get(cid, []) if t not in kp.BARE_STOPWORDS][:pool] for cid in members}
    all_terms = sorted({t for ts in pools.values() for t in ts})
    term_matrix = np.stack([term_cache[t] for t in all_terms])
    index = {t: k for k, t in enumerate(all_terms)}

    corpus_mean = kp.unit(doc_vectors.mean(axis=0))
    closeness = [float(kp.unit(doc_vectors[m].mean(axis=0)) @ corpus_mean) for m in members.values()]

    bias = {}
    for name, centered in (("raw", False), ("centered", True)):
        doc_mean = doc_vectors.mean(axis=0) if centered else 0.0
        term_mean = term_matrix.mean(axis=0) if centered else 0.0
        rhos = []
        for cid, terms in pools.items():
            if len(terms) < 5:
                continue
            centroid = kp.unit(doc_vectors[members[cid]].mean(axis=0) - doc_mean)
            vectors = kp.unit(term_matrix[[index[t] for t in terms]] - term_mean)
            relevance = vectors @ centroid
            generality = [presence[t] for t in terms]
            if len(set(generality)) > 1:
                rhos.append(scipy.stats.spearmanr(relevance, generality).statistic)
        bias[name] = (float(np.mean(rhos)), len(rhos))

    # The direct question: are the six terms each method picks present in more clusters? Mean presence
    # of the picks, averaged over clusters, for c-TF-IDF's own top six and for the rerank's.
    def mean_presence(labelling):
        return float(np.mean([np.mean([presence[t] for t, _ in kws]) for kws in labelling.values() if kws]))
    picks = {"ctfidf": {cid: kp.pick(ranked.get(cid, []), 6, suppress_nested=True, exclude=kp.BARE_STOPWORDS)
                        for cid in members}}
    for name, centered in (("rerank", False), ("rerank-c", True)):
        picks[name] = kp.rerank(ranked, members, doc_vectors, term_cache, 6, pool=pool, centered=centered,
                                bare=kp.BARE_STOPWORDS)
    presence_of_picks = {name: mean_presence(labelling) for name, labelling in picks.items()}
    return {"clusters": len(members), "closeness": float(np.mean(closeness)), "bias": bias,
            "picks": presence_of_picks}


def main():
    print("| dataset | clusters | centroid closeness | generality bias, raw | generality bias, centered "
          "| clusters per pick: c-TF-IDF | rerank | rerank, centered |")
    print("|---|---|---|---|---|---|---|---|")
    for arg in sys.argv[1:]:
        path = pathlib.Path(arg)
        r = measure(path)
        print(f"| {path.stem} | {r['clusters']} | {r['closeness']:.3f} | {r['bias']['raw'][0]:+.2f} | "
              f"{r['bias']['centered'][0]:+.2f} | {r['picks']['ctfidf']:.1f} | {r['picks']['rerank']:.1f} | "
              f"{r['picks']['rerank-c']:.1f} |", flush=True)


if __name__ == "__main__":
    main()

"""Does embedding a term in its context undo the single-word bias?

`try_relative_similarity.py` found that a bare single word can embed in its everyday sense: *hallucination*, in the
arXiv corpus's retrieval-augmented generation cluster, lies nearest the physics papers' centroid. This builds each
cluster's label as `keyphrases.label_groups` does, with the terms for the similarity order embedded three ways:

    bare        the term alone, as shipped
    head        the term followed by the cluster's head term in parentheses: "hallucination (retrieval-augmented
                generation)"
    sentences   the mean of the embeddings of up to `--max-sentences` sentences of the cluster's records (title or
                abstract) that contain the term, matched without case, hyphens or a final plural "s"; the bare term
                where none matches

and reports, per corpus and variant: of the terms in a pool's first three places by c-TF-IDF, the share of the
single words and of longer terms the label keeps, and how many labels change; then, for the arXiv corpus, where
*hallucination* lands in cluster 33, and every label a variant changes, side by side.

Usage, from the repository root:

    python investigations/cluster-keywords/try_context_embedding.py [--corpora arxiv aokk ...] [--max-sentences 5]

Uses the corpora and caches of `measure_single_words.py`. Writes `out/context-embedding.md`.
"""

import argparse
import collections
import pathlib
import pickle
import re

import numpy as np

from raven.client import api
from raven.client import config as client_config
from raven.common import keyphrases
from raven.common.text import stopwords
from raven.visualizer import importer

import measure_single_words as msw

HERE = pathlib.Path(__file__).resolve().parent
VARIANTS = ("bare", "head", "sentences")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def term_pattern(term: str) -> re.Pattern:
    """Matches `term` in running text, without case, hyphen against space, and a final plural "s"."""
    words = [re.escape(w) for w in term.replace("-", " ").split()]
    return re.compile(r"\b" + r"[\s-]+".join(words) + r"s?\b", re.IGNORECASE)


def unit(m):
    return m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-12)


def run(name, caches, embedder, max_sentences, shown=6, pool_size=30):
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

    heads, pools = {}, {}
    for g in cluster_ids:
        candidates = [item for item in ranked.get(g, []) if item[0] not in stopwords.bare_keyword_stopwords]
        heads[g] = keyphrases._pick(candidates, 1)
        pools[g] = [item for item in candidates
                    if item[0] not in umbrella and not any(keyphrases._nested(item[0], t) for t, _ in heads[g])][:pool_size]

    # The texts to embed, per variant, keyed by (cluster, term); embedded in one batch per variant.
    sentences_by_cluster = {}
    for g in cluster_ids:
        sents = []
        for idx in members[g]:
            r = records[idx]
            sents.append(str(r.title))
            sents.extend(s for s in SENTENCE_END.split(str(getattr(r, "abstract", "") or "")) if s.strip())
        sentences_by_cluster[g] = sents

    term_vectors = {}
    bare_terms = sorted({t for items in pools.values() for t, _, _ in items})
    bare = dict(zip(bare_terms, unit(np.asarray(embedder.encode(bare_terms), dtype=np.float32))))
    head_texts = {(g, t): f"{t} ({display.get(heads[g][0][0], heads[g][0][0])})"
                  for g in cluster_ids if heads[g] for t, _, _ in pools[g]}
    keys = sorted(head_texts)
    head_vecs = unit(np.asarray(embedder.encode([head_texts[k] for k in keys]), dtype=np.float32))
    term_vectors["head"] = dict(zip(keys, head_vecs))

    sentence_texts, owners = [], []
    for g in cluster_ids:
        for t, _, _ in pools[g]:
            pattern = term_pattern(t)
            hits = [s for s in sentences_by_cluster[g] if pattern.search(s)][:max_sentences]
            for s in hits:
                sentence_texts.append(s)
                owners.append((g, t))
    sentence_vecs = unit(np.asarray(embedder.encode(sentence_texts), dtype=np.float32)) if sentence_texts else []
    by_owner = collections.defaultdict(list)
    for key, v in zip(owners, sentence_vecs):
        by_owner[key].append(v)
    n_fallback = 0
    term_vectors["sentences"] = {}
    for g in cluster_ids:
        for t, _, _ in pools[g]:
            if by_owner[(g, t)]:
                term_vectors["sentences"][(g, t)] = unit(np.mean(by_owner[(g, t)], axis=0))
            else:
                term_vectors["sentences"][(g, t)] = bare[t]
                n_fallback += 1

    def vector(variant, g, t):
        return bare[t] if variant == "bare" else term_vectors[variant][(g, t)]

    labels = {v: {} for v in VARIANTS}
    bias = {v: collections.defaultdict(lambda: [0, 0]) for v in VARIANTS}
    places = {}  # (variant, cluster, term) -> place in the similarity order
    for g in cluster_ids:
        pool = pools[g]
        if not pool:
            for v in VARIANTS:
                labels[v][g] = [display.get(t, t) for t, _ in heads[g]]
            continue
        centroid = unit(vectors[members[g]].mean(axis=0))
        for v in VARIANTS:
            sim = np.stack([vector(v, g, t) for t, _, _ in pool]) @ centroid
            order = np.argsort(-sim, kind="stable")
            for place, k in enumerate(order):
                places[(v, g, pool[k][0])] = place
            fused = keyphrases.fuse([pool, [pool[k] for k in order]])
            label = heads[g] + keyphrases._pick(fused, shown - len(heads[g]), fixed=heads[g])
            labels[v][g] = [display.get(t, t) for t, _ in label]
            chosen = {t for t, _ in label}
            for t, _, _ in pool[:3]:
                wc = "1" if len(t.split()) == 1 else "2+"
                bias[v][wc][0] += t in chosen
                bias[v][wc][1] += 1
    n_terms = sum(len(p) for p in pools.values())
    return labels, bias, places, n_fallback, n_terms


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpora", nargs="*", default=list(msw.CORPORA))
    parser.add_argument("--max-sentences", type=int, default=5)
    args = parser.parse_args()

    api.initialize(raven_server_url=client_config.raven_server_url,
                   maybe_network_retry=client_config.network_retry_batch,
                   raven_api_key_file=client_config.raven_api_key_file)
    embedder = importer._make_embedder()

    def share(kt):
        return f"{kt[0]}/{kt[1]} ({kt[0] / kt[1]:.0%})" if kt[1] else "–"

    lines = ["# Embedding a term in its context", "",
             f"Of the terms in a pool's first three places by c-TF-IDF, the share the six-term label keeps; how many "
             f"labels differ from the shipped (bare) ones. Up to {args.max_sentences} sentences per term.", "",
             "| corpus | variant | single words kept | longer terms kept | labels changed | no sentence found |",
             "|---|---|---|---|---|---|"]
    arxiv = None
    for name in args.corpora:
        labels, bias, places, n_fallback, n_terms = run(name, msw.CORPORA[name], embedder, args.max_sentences)
        for v in VARIANTS:
            changed = sum(1 for g in labels["bare"] if labels[v][g] != labels["bare"][g])
            fallback = f"{n_fallback} of {n_terms}" if v == "sentences" else ""
            lines.append(f"| {name} | {v} | {share(bias[v]['1'])} | {share(bias[v]['2+'])} | "
                         f"{changed} of {len(labels['bare'])} | {fallback} |")
        print("\n".join(lines[-3:]), flush=True)
        if name == "arxiv":
            arxiv = (labels, places)

    if arxiv is not None:
        labels, places = arxiv
        lines += ["", "## arXiv: *hallucination* in cluster 33, place in the similarity order (of 30)", ""]
        lines += [f"- **{v}**: {places.get((v, 33, 'hallucination'), 'not in the pool')}" for v in VARIANTS]
        lines += ["", "## arXiv: the labels a variant changes", ""]
        for g in sorted(labels["bare"]):
            if all(labels[v][g] == labels["bare"][g] for v in VARIANTS):
                continue
            lines += [f"### Cluster {g}", ""] + [f"- **{v}**: {', '.join(labels[v][g])}" for v in VARIANTS] + [""]
    out = HERE / "out" / "context-embedding.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[lines.index("") + 1:][:0]))
    print(f"\nReport: {out}")


if __name__ == "__main__":
    main()

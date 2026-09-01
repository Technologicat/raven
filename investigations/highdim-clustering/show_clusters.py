#!/usr/bin/env python
"""Fit one HDBSCAN configuration in high-D and print each cluster's titles, for a reader to judge.

The sweep says how many clusters a configuration finds and how tight they are. It cannot say whether
a cluster is *about one thing*, which is the question that decides whether the map is any good — so
this prints the titles nearest each cluster's medoid and lets a domain reader answer it.

Usage:

    python show_clusters.py --vectors PATH.npz --dataset PATH.pickle \\
        [--min-cluster-size 5] [--min-samples 1] [--method eom] [--pca 0] \\
        [--assign-outliers] [--titles 8]
"""

import argparse
import sys

import numpy as np
from sklearn.cluster import HDBSCAN, AgglomerativeClustering

import clusterlab


def llm_keywords_for_clusters(texts_by_cluster, *, backend_url=None):
    """Ask the configured LLM for keywords describing each cluster.

    `texts_by_cluster`: `dict` of `{cluster_id: [entry text, ...]}`, each text as
                        `raven.visualizer.importer` formats it for this purpose — the title, and the
                        abstract after it where there is one.

    Returns `{cluster_id: [keyword, ...]}`. A cluster the model declines to characterize gets
    `["<unknown topic>"]`, which is what the importer records in that case.

    This deliberately reuses the Visualizer's own prompt and the same one-shot `agent.turn` call the
    importer's `llm` keyword mode makes, rather than asking the question a second way. The point of the
    exercise is to read what the app would show, so a prompt of our own would answer a question nobody
    is going to ask again.
    """
    from raven.librarian import agent, config as librarian_config, llmclient
    from raven.visualizer import config as visualizer_config

    url = backend_url or librarian_config.llm_backend_url
    if not llmclient.test_connection(url):
        raise SystemExit(f"no LLM backend answering at {url}")
    llm_settings = llmclient.setup(backend_url=url, quiet=True)
    print(f"keywording with {llm_settings.model_id} at {url}", file=sys.stderr)

    keywords = {}
    for cluster_id, texts in sorted(texts_by_cluster.items()):
        # Same separator the importer uses: a blank line can occur inside an abstract, two cannot.
        body = "\n\n\n".join(t.strip() for t in texts)
        prompt = f"{visualizer_config.clusters_llm_keyword_extraction_prompt}\n-----\n\n{body}"
        record = agent.turn(llm_settings,
                            prompt,
                            use_character_card=False,
                            tools_enabled=False,
                            internet_enabled=False,
                            docs_enabled=False,
                            markup=None)
        reply = (record.reply or "").strip()
        if reply.lower() == "keyword extraction failed":
            keywords[cluster_id] = ["<unknown topic>"]
        else:
            keywords[cluster_id] = [k.strip() for k in reply.split(",") if k.strip()]
        print(f"  cluster {cluster_id}: {', '.join(keywords[cluster_id])}", file=sys.stderr, flush=True)
    return keywords


def drop_undersized_clusters(labels, min_size):
    """Relabel clusters smaller than `min_size` as outliers, and renumber the survivors from 0.

    `labels`: rank-1 `np.array` of cluster IDs. Not mutated.

    Returns a new rank-1 `np.array`, outliers marked -1.
    """
    out = np.full(len(labels), -1, dtype=np.int64)
    next_id = 0
    for cid in np.unique(labels):
        members = (labels == cid)
        if int(members.sum()) >= min_size:
            out[members] = next_id
            next_id += 1
    return out


def cluster_mean_directions(vectors, labels):
    """Unit mean direction of each cluster, in cluster-ID order. Outliers are ignored.

    Stands in for HDBSCAN's medoids where the clusterer does not provide any — near enough for ranking
    an outlier's nearest cluster, which is all they are used for here.

    Returns a rank-2 `np.array` of shape `[n_clusters, dim]`, empty when nothing was clustered.
    """
    cluster_ids = np.unique(labels[labels >= 0])
    if len(cluster_ids) == 0:
        return np.zeros((0, vectors.shape[1]))
    return clusterlab.normalize(np.stack([vectors[labels == cid].mean(axis=0) for cid in cluster_ids]))


def assign_outliers_to_nearest_medoid(vectors, labels, medoids, *, min_similarity=None):
    """Give every outlier the label of the cluster whose medoid it is most similar to.

    `vectors`: rank-2 `np.array` of shape `[N, highdim]`, L2-normalized.
    `labels`: rank-1 `np.array` of cluster IDs, outliers marked -1. Not mutated.
    `medoids`: rank-2 `np.array` of shape `[n_clusters, highdim]`, HDBSCAN's `medoids_`.
    `min_similarity`: `float` or `None`. When given, an outlier whose best cosine similarity falls
                      below this stays an outlier, so that a point genuinely unlike everything in the
                      corpus is still reported as such.

    Returns `(new_labels, best_similarities)`, where `best_similarities` holds, for each formerly
    outlying point, the cosine similarity to the medoid it was assigned to (`nan` elsewhere).
    """
    labels = np.array(labels, copy=True)
    best_sims = np.full(len(labels), np.nan)
    outlier_idxs = np.flatnonzero(labels == -1)
    if len(outlier_idxs) == 0 or len(medoids) == 0:
        return labels, best_sims

    medoids = clusterlab.normalize(medoids)
    sims = clusterlab.normalize(vectors[outlier_idxs]) @ medoids.T
    winners = np.argmax(sims, axis=1)
    winning_sims = sims[np.arange(len(outlier_idxs)), winners]

    accepted = np.ones(len(outlier_idxs), dtype=bool) if min_similarity is None else (winning_sims >= min_similarity)
    labels[outlier_idxs[accepted]] = winners[accepted]
    best_sims[outlier_idxs] = winning_sims
    return labels, best_sims


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vectors", required=True, help="importer embedding cache (*_embeddings_cache.npz)")
    parser.add_argument("--dataset", required=True, help="Raven-visualizer dataset pickle, for the titles")
    parser.add_argument("--center", action="store_true", help="remove the corpus mean direction before fitting")
    parser.add_argument("--pca", type=int, default=0, help="fit in this many principal components (0 = raw space)")
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--algorithm", default="hdbscan", choices=["hdbscan", "agglomerative"],
                        help="which clusterer to look at (default: hdbscan)")
    parser.add_argument("--min-cluster-size", type=int, default=5,
                        help="HDBSCAN's `min_cluster_size`; for agglomerative, the size below which a "
                             "cluster is reported as outliers instead")
    parser.add_argument("--min-samples", type=int, default=1, help="HDBSCAN only")
    parser.add_argument("--method", default="eom", help="HDBSCAN's cluster_selection_method")
    parser.add_argument("--epsilon", type=float, default=0.0, help="HDBSCAN only")
    parser.add_argument("--n-clusters", type=int, default=None, help="agglomerative: cut the tree at this many")
    parser.add_argument("--distance-threshold", type=float, default=None,
                        help="agglomerative: cut the tree at this distance instead of at a count")
    parser.add_argument("--linkage", default="average", help="agglomerative linkage (default: average)")
    parser.add_argument("--assign-outliers", action="store_true", help="give outliers their nearest medoid's label")
    parser.add_argument("--min-similarity", type=float, default=None, help="floor for --assign-outliers")
    parser.add_argument("--titles", type=int, default=8, help="how many titles to print per cluster")
    parser.add_argument("--llm-keywords", action="store_true",
                        help="ask the configured LLM for keywords describing each cluster, using the "
                             "Visualizer's own `clusters_keyword_method='llm'` prompt")
    parser.add_argument("--backend-url", default=None,
                        help="LLM backend to use for --llm-keywords, overriding the configured one")
    parser.add_argument("--max-prompt-entries", type=int, default=60,
                        help="cap on how many of a cluster's entries go into its keyword prompt "
                             "(default 60); the ones nearest the cluster centre are kept")
    args = parser.parse_args()

    vectors, model_name = clusterlab.load_vectors(args.vectors)
    entries = clusterlab.load_entries(args.dataset)
    titles = [title for title, _ in entries]
    if len(titles) != len(vectors):
        raise SystemExit(f"dataset has {len(titles)} entries but the embedding cache has {len(vectors)}; "
                         "they must come from the same import")
    print(f"{len(vectors)} entries, {vectors.shape[1]} dimensions, embedded by {model_name}", file=sys.stderr)

    original = clusterlab.normalize(vectors)  # the reference space, used for scoring and for the medoid ordering
    fit_vectors = original
    if args.center:
        fit_vectors = clusterlab.center(fit_vectors)
    if args.pca:
        fit_vectors, kept = clusterlab.pca_reduce(fit_vectors, args.pca)
        print(f"PCA to {args.pca} components keeps {kept:.1%} of the variance", file=sys.stderr)
        fit_vectors = clusterlab.normalize(fit_vectors)

    if args.algorithm == "hdbscan":
        clusterer = HDBSCAN(min_cluster_size=args.min_cluster_size,
                            min_samples=args.min_samples,
                            cluster_selection_method=args.method,
                            cluster_selection_epsilon=args.epsilon,
                            metric=args.metric,
                            store_centers="medoid",
                            copy=True)
        clusterer.fit(fit_vectors)
        labels = clusterer.labels_
        medoids = clusterer.medoids_
    else:
        if (args.n_clusters is None) == (args.distance_threshold is None):
            raise SystemExit("agglomerative needs exactly one of --n-clusters or --distance-threshold")
        labels = AgglomerativeClustering(n_clusters=args.n_clusters,
                                         distance_threshold=args.distance_threshold,
                                         metric=args.metric,
                                         linkage=args.linkage).fit_predict(fit_vectors)
        # Agglomerative assigns every point, so it has no noise concept of its own. Cutting a tree deep
        # enough to be useful leaves a tail of two- and three-member clusters that no reader would call
        # a topic, so undersized ones are reported as outliers here — that is the "honest outliers"
        # arrangement the write-up recommends, and this is where it gets looked at rather than scored.
        labels = drop_undersized_clusters(labels, args.min_cluster_size)
        medoids = cluster_mean_directions(fit_vectors, labels)

    stats = clusterlab.cluster_size_stats(labels)
    print(f"\nfitted: {stats['n_clusters']} clusters, {stats['n_outliers']} outliers "
          f"({stats['outlier_fraction']:.1%})", file=sys.stderr)

    if args.assign_outliers:
        # The medoids come back in the fitting space, so the similarities that decide the assignment
        # must be computed there too — mixing a PCA-space medoid with a full-space point would rank
        # by an inner product between two different bases.
        labels, best_sims = assign_outliers_to_nearest_medoid(fit_vectors, labels, medoids,
                                                              min_similarity=args.min_similarity)
        assigned = np.isfinite(best_sims)
        stats = clusterlab.cluster_size_stats(labels)
        print(f"after outlier assignment: {stats['n_clusters']} clusters, {stats['n_outliers']} outliers "
              f"({stats['outlier_fraction']:.1%})", file=sys.stderr)
        if assigned.any():
            sims = best_sims[assigned]
            print(f"  similarity of an outlier to its winning medoid: "
                  f"min {sims.min():.3f}, median {np.median(sims):.3f}, max {sims.max():.3f}",
                  file=sys.stderr)

    print(f"  compactness {clusterlab.mean_intra_cluster_similarity(original, labels):.3f}, "
          f"nearest-cluster {clusterlab.mean_nearest_cluster_similarity(original, labels):.3f}\n",
          file=sys.stderr)

    keywords_by_cluster = {}
    if args.llm_keywords:
        # Every member goes to the model where the cluster is small enough, because the keywords are meant
        # to describe the cluster and a sample of the three titles printed below would describe the sample.
        # A big cluster cannot: on a crowded corpus the largest runs to hundreds of entries, and with
        # abstracts attached that is a prompt of a quarter of a million tokens. So the biggest clusters are
        # capped, keeping the entries nearest the centre - the ones that most define what the cluster is
        # about, which is the question being asked. Clusters at or below the cap are unaffected, and on the
        # corpora here that is most of them.
        texts_by_cluster = {}
        capped = []
        for cid in np.unique(labels[labels >= 0]):
            members = np.flatnonzero(labels == cid)
            if len(members) > args.max_prompt_entries:
                center = clusterlab.normalize(original[members].mean(axis=0, keepdims=True))
                members = members[np.argsort(-(original[members] @ center[0]))][:args.max_prompt_entries]
                capped.append(int(cid))
            texts_by_cluster[int(cid)] = [clusterlab.format_for_keyword_extraction(*entries[i])
                                          for i in members]
        if capped:
            print(f"{len(capped)} cluster(s) capped at {args.max_prompt_entries} entries for keywording: "
                  f"{capped}", file=sys.stderr)
        keywords_by_cluster = llm_keywords_for_clusters(texts_by_cluster, backend_url=args.backend_url)

    # Print clusters largest first, and within a cluster the titles closest to its centre, so that the
    # first lines a reader sees are the ones that most nearly define what the cluster is about.
    for cid in sorted(np.unique(labels[labels >= 0]), key=lambda c: -np.sum(labels == c)):
        members = np.flatnonzero(labels == cid)
        center = clusterlab.normalize(original[members].mean(axis=0, keepdims=True))
        order = members[np.argsort(-(original[members] @ center[0]))]
        print(f"--- cluster {cid}  ({len(members)} entries) ---")
        if int(cid) in keywords_by_cluster:
            print(f"    KEYWORDS: {', '.join(keywords_by_cluster[int(cid)])}")
        for idx in order[:args.titles]:
            print(f"    {titles[idx]}")
        print()

    n_outliers = int(np.sum(labels == -1))
    if n_outliers:
        print(f"--- outliers ({n_outliers} entries) ---")
        for idx in np.flatnonzero(labels == -1)[:args.titles]:
            print(f"    {titles[idx]}")


if __name__ == "__main__":
    main()

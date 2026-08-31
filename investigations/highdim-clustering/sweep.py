#!/usr/bin/env python
"""Sweep HDBSCAN hyperparameters over a corpus's high-dimensional embeddings.

Answers: which `(min_cluster_size, min_samples, cluster_selection_method, cluster_selection_epsilon)`
give a labelling worth showing a reader, when the clustering is done in the embedding space rather
than in the 2D map.

Usage:

    python sweep.py --vectors PATH.npz [--pca 50] [--metric cosine|euclidean] [--out results.tsv]

Prints a table and, with `--out`, writes it as TSV.
"""

import argparse
import itertools
import sys
import time

from sklearn.cluster import HDBSCAN

import clusterlab


def run_one(vectors, scoring_vectors, *, min_cluster_size, min_samples, method, epsilon, metric):
    """Fit one HDBSCAN configuration. Returns a row `dict` of settings, metrics and wall time.

    `vectors`: what to fit on — the original embeddings, or a PCA projection of them.
    `scoring_vectors`: what to judge the resulting labelling in. Always the original embedding space,
                       so that configurations fitted in different PCA subspaces stay comparable:
                       compactness and separation measured inside a 10-dimensional projection are on
                       a different scale from the same quantities measured in 1024, and reading the
                       two side by side would rank the projection rather than the clustering.
    """
    t0 = time.perf_counter()
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                        cluster_selection_method=method,
                        cluster_selection_epsilon=epsilon,
                        metric=metric,
                        store_centers="medoid",
                        copy=True)  # sklearn 1.10 makes this the default; set it now to silence the FutureWarning
    clusterer.fit(vectors)
    dt = time.perf_counter() - t0

    row = {"min_cluster_size": min_cluster_size,
           "min_samples": min_samples,
           "method": method,
           "epsilon": epsilon}
    row.update(clusterlab.cluster_size_stats(clusterer.labels_))
    row["intra_sim"] = clusterlab.mean_intra_cluster_similarity(scoring_vectors, clusterer.labels_)
    row["nearest_sim"] = clusterlab.mean_nearest_cluster_similarity(scoring_vectors, clusterer.labels_)
    row["seconds"] = dt
    return row


COLUMNS = ["min_cluster_size", "min_samples", "method", "epsilon",
           "n_clusters", "n_outliers", "outlier_fraction",
           "size_min", "size_median", "size_max",
           "intra_sim", "nearest_sim", "seconds"]


def format_row(row):
    """Render one result row as a fixed-width line, for reading in a terminal."""
    return (f"{row['min_cluster_size']:>4} {row['min_samples']:>4} {row['method']:>5} {row['epsilon']:>6.3f} "
            f"{row['n_clusters']:>6} {row['outlier_fraction']:>7.1%} "
            f"{row['size_min']:>5} {row['size_median']:>7} {row['size_max']:>6} "
            f"{row['intra_sim']:>8.3f} {row['nearest_sim']:>8.3f} {row['seconds']:>7.2f}")


HEADER = (f"{'mcs':>4} {'ms':>4} {'meth':>5} {'eps':>6} "
          f"{'clust':>6} {'noise':>7} {'min':>5} {'median':>7} {'max':>6} "
          f"{'intra':>8} {'nearest':>8} {'sec':>7}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vectors", required=True, help="importer embedding cache (*_embeddings_cache.npz)")
    parser.add_argument("--center", action="store_true", help="remove the corpus mean direction before fitting")
    parser.add_argument("--pca", type=int, default=0, help="reduce to this many principal components first (0 = no PCA)")
    parser.add_argument("--metric", default="cosine", help="HDBSCAN metric (default: cosine)")
    parser.add_argument("--min-cluster-size", type=int, nargs="+", default=[5, 10, 15, 20, 30, 50])
    parser.add_argument("--min-samples", type=int, nargs="+", default=[1, 2, 5, 10])
    parser.add_argument("--method", nargs="+", default=["eom", "leaf"])
    parser.add_argument("--epsilon", type=float, nargs="+", default=[0.0])
    parser.add_argument("--out", default=None, help="write the results as TSV to this path")
    args = parser.parse_args()

    vectors, model_name = clusterlab.load_vectors(args.vectors)
    print(f"{len(vectors)} vectors, {vectors.shape[1]} dimensions, embedded by {model_name}", file=sys.stderr)

    vectors = clusterlab.normalize(vectors)
    scoring_vectors = vectors  # judge every configuration in the original space, whatever it was fitted in
    if args.center:
        vectors = clusterlab.center(vectors)
    if args.pca:
        vectors, kept = clusterlab.pca_reduce(vectors, args.pca)
        print(f"PCA to {args.pca} components keeps {kept:.1%} of the variance", file=sys.stderr)
        vectors = clusterlab.normalize(vectors)

    print(HEADER)
    rows = []
    for mcs, ms, method, eps in itertools.product(args.min_cluster_size, args.min_samples,
                                                  args.method, args.epsilon):
        if ms > mcs:  # `min_samples` above `min_cluster_size` is not a meaningful setting
            continue
        row = run_one(vectors, scoring_vectors, min_cluster_size=mcs, min_samples=ms, method=method,
                      epsilon=eps, metric=args.metric)
        rows.append(row)
        print(format_row(row), flush=True)

    if args.out:
        with open(args.out, "w") as f:
            f.write("\t".join(COLUMNS) + "\n")
            for row in rows:
                f.write("\t".join(str(row[c]) for c in COLUMNS) + "\n")
        print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

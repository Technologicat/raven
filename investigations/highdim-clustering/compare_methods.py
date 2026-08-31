#!/usr/bin/env python
"""Compare clustering methods for the high-dimensional map, on one yardstick.

The sweep tunes HDBSCAN. This asks the prior question: is a density model the right one for literature
embeddings at all, or does a centroid model fit them better? The centroid entries stand in for MSSC —
UTU's Clust-Splitter solves the same objective and is what a positive result here would justify wiring.

Every method is scored in the same space (the raw normalized embeddings), so the numbers are
comparable across methods and against the current 2D baseline:

    compactness   mean cosine of a clustered point to its own cluster's mean direction (higher better)
    nearest       mean cosine between a cluster's mean direction and the nearest other one (lower better)
    gap           compactness - nearest. The single number to read: positive means clusters are more
                  like themselves than like each other, which is the whole claim a cluster makes.
    coverage      fraction of the corpus that landed in some cluster

Read `gap` and `coverage` together. A method that clusters only the easy tenth of the corpus can post
a fine gap while saying nothing about the rest, and a method that assigns everything pays for the
coverage in gap.

Usage:

    python compare_methods.py --vectors PATH.npz [--dataset PATH.pickle] [--out results.tsv]
"""

import argparse
import sys
import time

import numpy as np

import clusterlab


def score(scoring_vectors, labels, seconds, name):
    """Build one comparison row from a labelling."""
    stats = clusterlab.cluster_size_stats(labels)
    intra = clusterlab.mean_intra_cluster_similarity(scoring_vectors, labels)
    nearest = clusterlab.mean_nearest_cluster_similarity(scoring_vectors, labels)
    return {"method": name,
            "n_clusters": stats["n_clusters"],
            "coverage": 1.0 - stats["outlier_fraction"],
            "size_median": stats["size_median"],
            "size_max": stats["size_max"],
            "compactness": intra,
            "nearest": nearest,
            "gap": intra - nearest,
            "seconds": seconds}


COLUMNS = ["method", "n_clusters", "coverage", "size_median", "size_max",
           "compactness", "nearest", "gap", "seconds"]

HEADER = (f"{'method':<34} {'clust':>6} {'cover':>6} {'med':>5} {'max':>6} "
          f"{'compact':>8} {'nearest':>8} {'gap':>7} {'sec':>6}")


def format_row(row):
    return (f"{row['method']:<34} {row['n_clusters']:>6} {row['coverage']:>5.0%} "
            f"{row['size_median']:>5} {row['size_max']:>6} "
            f"{row['compactness']:>8.3f} {row['nearest']:>8.3f} {row['gap']:>+7.3f} {row['seconds']:>6.2f}")


def hdbscan_labels(fit_vectors, *, min_cluster_size, min_samples, method):
    from sklearn.cluster import HDBSCAN
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                        cluster_selection_method=method, metric="cosine",
                        store_centers="medoid", copy=True)
    clusterer.fit(fit_vectors)
    return clusterer.labels_


def kmeans_labels(fit_vectors, k):
    """Spherical k-means: on L2-normalized input, the k-means objective ranks by cosine.

    This is the MSSC objective Clust-Splitter solves, at the quality Lloyd's algorithm reaches.
    A real solver would find a better optimum of the same objective, so treat this as a floor on
    what a centroid model can do here rather than as its ceiling.
    """
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(fit_vectors)


def agglomerative_labels(fit_vectors, k):
    from sklearn.cluster import AgglomerativeClustering
    return AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(fit_vectors)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vectors", required=True)
    parser.add_argument("--dataset", default=None, help="dataset pickle, to score the current 2D baseline too")
    parser.add_argument("--k", type=int, nargs="+", default=[20, 50, 100, 150])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    vectors, model_name = clusterlab.load_vectors(args.vectors)
    print(f"{len(vectors)} vectors, {vectors.shape[1]} dimensions, embedded by {model_name}\n", file=sys.stderr)
    raw = clusterlab.normalize(vectors)
    centered = clusterlab.center(raw)

    rows = []

    if args.dataset:
        import pickle
        with open(args.dataset, "rb") as f:
            dataset = pickle.load(f)
        rows.append(score(raw, np.asarray(dataset["labels"]), float("nan"), "CURRENT (HDBSCAN in 2D)"))

    for space_name, fit_vectors in (("raw", raw), ("centered", centered)):
        for mcs, ms, method in ((5, 1, "eom"), (5, 1, "leaf"), (10, 1, "eom")):
            t0 = time.perf_counter()
            labels = hdbscan_labels(fit_vectors, min_cluster_size=mcs, min_samples=ms, method=method)
            rows.append(score(raw, labels, time.perf_counter() - t0,
                              f"HDBSCAN {space_name} mcs={mcs} ms={ms} {method}"))

    for space_name, fit_vectors in (("raw", raw), ("centered", centered)):
        for k in args.k:
            t0 = time.perf_counter()
            labels = kmeans_labels(fit_vectors, k)
            rows.append(score(raw, labels, time.perf_counter() - t0, f"k-means {space_name} k={k}"))

    for k in args.k:
        t0 = time.perf_counter()
        labels = agglomerative_labels(centered, k)
        rows.append(score(raw, labels, time.perf_counter() - t0, f"agglomerative centered k={k}"))

    print(HEADER)
    for row in rows:
        print(format_row(row), flush=True)

    if args.out:
        with open(args.out, "w") as f:
            f.write("\t".join(COLUMNS) + "\n")
            for row in rows:
                f.write("\t".join(str(row[c]) for c in COLUMNS) + "\n")
        print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

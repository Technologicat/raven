#!/usr/bin/env python
"""Negative control for the method comparison: does HDBSCAN win, or does it just pick the easy points?

`compare_methods.py` scores a density method that clusters a quarter of the corpus against centroid
methods that cluster all of it, at different cluster counts. Two confounds ride along, and either
alone would explain the whole result:

  - **Coverage.** HDBSCAN's clustered points are by construction the ones in dense regions. Any method
    restricted to those points should look good.
  - **Cluster count.** Separation is measured against the *nearest* other cluster, so cutting a corpus
    into more pieces can only bring the pieces closer together.

So this re-runs the centroid methods on exactly the subset HDBSCAN clustered, at exactly the cluster
count HDBSCAN chose. If HDBSCAN still wins there, the win is about the partition rather than about
which points it declined to answer for.

Usage:

    python matched_control.py --vectors PATH.npz
"""

import argparse
import sys

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans, AgglomerativeClustering

import clusterlab


def gap(vectors, labels):
    """Compactness minus nearest-cluster similarity, the single comparison number."""
    intra = clusterlab.mean_intra_cluster_similarity(vectors, labels)
    nearest = clusterlab.mean_nearest_cluster_similarity(vectors, labels)
    return intra, nearest, intra - nearest


def size_weighted_gap(vectors, labels):
    """`gap`, but with the separation term weighted by cluster size.

    The plain `gap` mixes two weightings: compactness averages over *points*, separation over
    *clusters*. That hands a two-member cluster the same vote on separation as an 800-member one, and
    a singleton the largest vote of all — its compactness is exactly 1.0 against its own mean, and
    sitting alone in a sparse region it can be far from every other centre. A method that leaves a
    tail of slivers therefore scores well for having done so.

    Weighting separation by size closes that hole while changing nothing about a balanced partition.
    Read it beside `gap`: a method whose advantage disappears here had its advantage in the slivers.
    """
    vectors = clusterlab.normalize(vectors)
    cluster_ids = np.unique(labels[labels >= 0])
    if len(cluster_ids) < 2:
        return float("nan")
    centers = clusterlab.normalize(np.stack([vectors[labels == cid].mean(axis=0) for cid in cluster_ids]))
    sizes = np.array([int(np.sum(labels == cid)) for cid in cluster_ids])
    gram = centers @ centers.T
    np.fill_diagonal(gram, -np.inf)
    nearest = float(np.average(gram.max(axis=1), weights=sizes))
    intra = clusterlab.mean_intra_cluster_similarity(vectors, labels)
    return intra - nearest


def drop_undersized(labels, min_size):
    """Relabel clusters below `min_size` as outliers. Returns `(new_labels, n_clusters_left)`."""
    out = np.array(labels, copy=True)
    for cid in np.unique(labels[labels >= 0]):
        if int(np.sum(labels == cid)) < min_size:
            out[labels == cid] = -1
    return out, int(len(np.unique(out[out >= 0])))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vectors", required=True)
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument("--method", default="eom")
    parser.add_argument("--center", action="store_true")
    parser.add_argument("--dataset", default=None,
                        help="dataset pickle; also scores the shipped 2D labelling against its own matched floor")
    args = parser.parse_args()

    vectors, model_name = clusterlab.load_vectors(args.vectors)
    raw = clusterlab.normalize(vectors)
    fit_vectors = clusterlab.center(raw) if args.center else raw
    print(f"{len(raw)} vectors, embedded by {model_name}; "
          f"fitting HDBSCAN mcs={args.min_cluster_size} ms={args.min_samples} {args.method}"
          f"{' on centered vectors' if args.center else ''}\n", file=sys.stderr)

    hdb = HDBSCAN(min_cluster_size=args.min_cluster_size, min_samples=args.min_samples,
                  cluster_selection_method=args.method, metric="cosine",
                  store_centers="medoid", copy=True).fit(fit_vectors)

    clustered = np.flatnonzero(hdb.labels_ >= 0)
    k = int(len(np.unique(hdb.labels_[hdb.labels_ >= 0])))
    print(f"HDBSCAN clustered {len(clustered)} of {len(raw)} points ({len(clustered) / len(raw):.1%}) "
          f"into {k} clusters.\n"
          f"Every row below is scored on those same {len(clustered)} points, at k={k}.\n", file=sys.stderr)

    subset_raw = raw[clustered]
    subset_fit = fit_vectors[clustered]

    print(f"{'labelling':<40} {'compact':>8} {'nearest':>8} {'gap':>8} {'size-wt':>8} {'drop<5':>8} {'k left':>7} {'≤2':>4}")
    rows = [("HDBSCAN (the partition under test)", hdb.labels_[clustered]),
            (f"k-means on the same points, k={k}", KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(subset_fit)),
            (f"agglomerative on the same points, k={k}", AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(subset_fit)),
            (f"random labelling, k={k}", np.random.default_rng(42).integers(0, k, size=len(clustered)))]
    for name, labels in rows:
        intra, nearest, g = gap(subset_raw, labels)
        weighted = size_weighted_gap(subset_raw, labels)
        trimmed, k_left = drop_undersized(labels, 5)
        _, _, g_trimmed = gap(subset_raw, trimmed)
        slivers = sum(1 for cid in np.unique(labels[labels >= 0]) if int(np.sum(labels == cid)) <= 2)
        print(f"{name:<40} {intra:>8.3f} {nearest:>8.3f} {g:>+8.3f} {weighted:>+8.3f} "
              f"{g_trimmed:>+8.3f} {k_left:>7} {slivers:>4}")

    print("\n`gap` is the headline; `size-wt` and `drop<5` are the robustness checks, and a method whose\n"
          "lead survives only in `gap` earned it from the sliver clusters counted in the last column.",
          file=sys.stderr)

    print("\nThe random row is the floor: any labelling scoring near it has found nothing, and a\n"
          "fixture where every method lands there could not have told them apart in the first place.",
          file=sys.stderr)

    if args.dataset:
        # The shipped 2D labelling has its own coverage and its own cluster count, so its floor is not
        # the one above: a random labelling into more clusters scores worse whatever it is measuring.
        # Comparing it to a floor built at *its* k and *its* covered subset is the only fair reading.
        import pickle
        with open(args.dataset, "rb") as f:
            baseline = np.asarray(pickle.load(f)["labels"])
        covered = np.flatnonzero(baseline >= 0)
        baseline_k = int(len(np.unique(baseline[covered])))
        print(f"\nShipped 2D labelling: {len(covered)} of {len(baseline)} points "
              f"({len(covered) / len(baseline):.1%}) into {baseline_k} clusters.", file=sys.stderr)
        print(f"{'labelling':<40} {'compact':>8} {'nearest':>8} {'gap':>8}")
        for name, labels in (("HDBSCAN in 2D (what ships today)", baseline[covered]),
                             (f"random labelling, k={baseline_k}",
                              np.random.default_rng(42).integers(0, baseline_k, size=len(covered)))):
            intra, nearest, g = gap(raw[covered], labels)
            print(f"{name:<40} {intra:>8.3f} {nearest:>8.3f} {g:>+8.3f}")


if __name__ == "__main__":
    main()

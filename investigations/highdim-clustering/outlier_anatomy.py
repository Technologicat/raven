#!/usr/bin/env python
"""Are the outliers genuinely apart from every cluster, or merely between two of them?

The map leaves some records unplaced, and the two possible reasons want different responses. If an
outlier sits in a sparse region of the embedding, being unplaced is a fact about the corpus and the map
is right to say so. If it sits between two clusters - close to both, committed to neither - then it is
the clustering that failed to place it, and a rule for breaking the tie would recover it.

Three measurements separate those:

    nearest        cosine to the closest cluster's mean direction. Low for a genuinely isolated point,
                   high for one sitting between clusters.
    margin         cosine to the closest cluster minus cosine to the second closest. Near zero means
                   the point is equidistant between two clusters, which is the "missed" case.
    local density  mean cosine to its own k nearest neighbours among *all* points, clustered or not.
                   This is the one that does not depend on the clustering at all, so it is the honest
                   test of whether the region is sparse.

Read against the clustered points as a baseline: an outlier population that matches them on local
density but not on margin was missed, and one that is sparser on local density is genuinely apart.

Note what an outlier *is* here depends on the method. Under agglomerative with a minimum size, an
outlier is a member of a cluster too small to show - the algorithm placed it, and we declined the answer
- so the size distribution of those sub-threshold clusters is reported too. Under HDBSCAN it is real
density-noise. The two are not the same thing and the same word covers both.

Usage:

    python outlier_anatomy.py --vectors PATH.npz [--algorithm agglomerative|hdbscan]
"""

import argparse
import sys

import numpy as np
from sklearn.cluster import AgglomerativeClustering, HDBSCAN

import clusterlab


def describe(name, values):
    """One line of distribution, for comparing two populations by eye."""
    if len(values) == 0:
        return f"    {name:<16} (none)"
    q = np.percentile(values, [10, 50, 90])
    return f"    {name:<16} n={len(values):>5}  p10 {q[0]:+.3f}  median {q[1]:+.3f}  p90 {q[2]:+.3f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vectors", required=True)
    parser.add_argument("--algorithm", default="agglomerative", choices=["agglomerative", "hdbscan"])
    parser.add_argument("--n-clusters", type=int, default=100, help="agglomerative cut level")
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--neighbours", type=int, default=10, help="k for the local density measure")
    args = parser.parse_args()

    vectors, model_name = clusterlab.load_vectors(args.vectors)
    original = clusterlab.normalize(vectors)
    fit_vectors = clusterlab.center(original)
    print(f"{len(original)} vectors, embedded by {model_name}", file=sys.stderr)

    if args.algorithm == "agglomerative":
        raw = AgglomerativeClustering(n_clusters=args.n_clusters, metric="cosine",
                                      linkage="average").fit_predict(fit_vectors)
        labels = np.full(len(raw), -1)
        next_id = 0
        subthreshold_sizes = []
        for cid in np.unique(raw):
            members = raw == cid
            if int(members.sum()) >= args.min_cluster_size:
                labels[members] = next_id
                next_id += 1
            else:
                subthreshold_sizes.append(int(members.sum()))
    else:
        labels = HDBSCAN(min_cluster_size=args.min_cluster_size, min_samples=1,
                         cluster_selection_method="leaf", metric="cosine", copy=True).fit_predict(fit_vectors)
        subthreshold_sizes = None

    outliers = np.flatnonzero(labels == -1)
    clustered = np.flatnonzero(labels >= 0)
    cluster_ids = np.unique(labels[labels >= 0])
    print(f"{args.algorithm}: {len(cluster_ids)} clusters, {len(outliers)} outliers "
          f"({len(outliers) / len(labels):.1%})\n")

    if subthreshold_sizes is not None:
        sizes = np.array(sorted(subthreshold_sizes, reverse=True))
        print(f"outliers come from {len(sizes)} sub-threshold clusters (min size {args.min_cluster_size}); "
              f"sizes: {sizes[:15].tolist()}{' ...' if len(sizes) > 15 else ''}")
        print(f"  singletons {int((sizes == 1).sum())}, "
              f"just-missed (size {args.min_cluster_size - 1}) {int((sizes == args.min_cluster_size - 1).sum())}\n")

    # Cosine to each cluster's mean direction, for every point.
    centers = clusterlab.normalize(np.stack([original[labels == cid].mean(axis=0) for cid in cluster_ids]))
    sims = original @ centers.T
    ordered = np.sort(sims, axis=1)
    nearest = ordered[:, -1]
    margin = ordered[:, -1] - ordered[:, -2]

    # Local density, computed against every point and so independent of the clustering. Chunked, because
    # the full similarity matrix is not needed at once and would be large on a big corpus.
    k = args.neighbours
    density = np.empty(len(original))
    for start in range(0, len(original), 2048):
        block = original[start:start + 2048] @ original.T
        block[np.arange(len(block)), np.arange(start, start + len(block))] = -np.inf  # not one's own neighbour
        density[start:start + len(block)] = np.sort(block, axis=1)[:, -k:].mean(axis=1)

    print("cosine to the nearest cluster center:")
    print(describe("clustered", nearest[clustered]))
    print(describe("outliers", nearest[outliers]))
    print("\nmargin (nearest cluster minus second nearest) - near zero means 'between two clusters':")
    print(describe("clustered", margin[clustered]))
    print(describe("outliers", margin[outliers]))
    print(f"\nlocal density (mean cosine to {k} nearest neighbours, any label) - the clustering-independent test:")
    print(describe("clustered", density[clustered]))
    print(describe("outliers", density[outliers]))

    if args.algorithm == "agglomerative":
        # Is a small cluster a real topic or a residue? Average linkage merges what is left over, so a
        # cluster can be small because its members belong together, or small because they belong nowhere
        # else. Their mean pairwise similarity tells the two apart where their size cannot - and the
        # threshold can be read off the big clusters rather than picked, which is what makes this usable
        # on a corpus nobody has looked at yet.
        def cohesion(members):
            block = original[members] @ original[members].T
            return block[~np.eye(len(members), dtype=bool)].mean()

        by_size = {int(cid): np.flatnonzero(raw == cid) for cid in np.unique(raw)}
        big = [cohesion(m) for m in by_size.values() if len(m) >= args.min_cluster_size]
        small = [(cid, m, cohesion(m)) for cid, m in by_size.items()
                 if 2 <= len(m) < args.min_cluster_size]
        if big and small:
            floor = float(np.percentile(big, 10))
            kept = [s for s in small if s[2] >= floor]
            print("\ncohesion (mean pairwise cosine within a cluster) - is a small cluster a topic or a residue?")
            print(f"    clusters >= {args.min_cluster_size}: median {np.median(big):+.3f}, p10 {floor:+.3f}")
            print(f"    clusters  < {args.min_cluster_size}: median {np.median([s[2] for s in small]):+.3f}, "
                  f"min {min(s[2] for s in small):+.3f}")
            print("    keeping a small cluster when its cohesion reaches the big clusters' p10:")
            print(f"        {len(kept)} of {len(small)} survive, recovering {sum(len(s[1]) for s in kept)} "
                  f"of {len(outliers)} outliers")


if __name__ == "__main__":
    main()

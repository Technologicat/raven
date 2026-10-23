#!/usr/bin/env python
"""Which small-cluster rule to ship: the cohesion floor with a size threshold, or one without.

Compares, at one cut of the agglomerative tree, what each rule keeps:

  `p10-small`   finding 10's rule: clusters under `--min-size` are kept when their cohesion reaches the
                membership-weighted 10th percentile; larger clusters are kept unconditionally.
  `p10-all`     the same floor applied to every cluster, which needs no size and drops whole topics.
  `background`  a cluster is kept when its cohesion exceeds the corpus's mean pairwise cosine: its
                members are more alike than two records picked at random. No size, no percentile.

Cohesion is the mean pairwise cosine within a cluster, on the uncentered normalized vectors, as in
`outlier_anatomy.py`. The background is estimated from a random sample of pairs.

    python investigations/highdim-clustering/cohesion_rules.py --vectors V.npz [V2.npz ...] --cut 100
"""

import argparse
import sys

import numpy as np
from sklearn.cluster import AgglomerativeClustering

import clusterlab


def background_similarity(vectors, n_pairs=200_000, seed=0):
    """Mean cosine between two distinct records picked at random. `vectors` L2-normalized."""
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(vectors), n_pairs)
    j = rng.integers(0, len(vectors), n_pairs)
    keep = i != j
    return float(np.mean(np.sum(vectors[i[keep]] * vectors[j[keep]], axis=1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vectors", nargs="+", required=True, help="one corpus; several caches are concatenated")
    parser.add_argument("--cut", type=int, default=100)
    parser.add_argument("--min-size", type=int, default=5, help="for the p10-small rule only")
    args = parser.parse_args()

    original = clusterlab.normalize(np.concatenate([clusterlab.load_vectors(v)[0] for v in args.vectors]))
    raw = AgglomerativeClustering(n_clusters=args.cut, metric="cosine",
                                  linkage="average").fit_predict(clusterlab.center(original))
    members = {int(cid): np.flatnonzero(raw == cid) for cid in np.unique(raw)}
    cohesion = {cid: clusterlab.cohesion(original, m) for cid, m in members.items() if len(m) >= 2}
    floor = clusterlab.cohesion_floor(original, raw)
    background = background_similarity(original)
    print(f"{len(original)} records, cut {args.cut}: {len(members)} raw clusters, "
          f"{sum(len(m) == 1 for m in members.values())} singletons; p10 floor {floor:.3f}, "
          f"background {background:.3f}", file=sys.stderr)

    rules = {"p10-small": lambda cid: len(members[cid]) >= args.min_size or cohesion[cid] >= floor,
             "p10-all": lambda cid: cohesion[cid] >= floor,
             "background": lambda cid: cohesion[cid] > background}
    print("rule        clusters  coverage   dropped clusters (size:cohesion), largest first")
    for name, keeps in rules.items():
        kept = [cid for cid in cohesion if keeps(cid)]
        dropped = sorted((cid for cid in cohesion if not keeps(cid)), key=lambda cid: -len(members[cid]))
        covered = sum(len(members[cid]) for cid in kept) / len(original)
        shown = ", ".join(f"{len(members[cid])}:{cohesion[cid]:.2f}" for cid in dropped[:8])
        print(f"{name:10s}  {len(kept):8d}  {covered:7.1%}   {len(dropped)}: {shown}{' ...' if len(dropped) > 8 else ''}")


if __name__ == "__main__":
    main()

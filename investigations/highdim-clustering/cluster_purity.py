#!/usr/bin/env python
"""How tight are the importer's agglomerative clusters? Purity per cluster, across corpora.

    python investigations/highdim-clustering/cluster_purity.py NAME=EMBEDDINGS.npz [NAME=EMBEDDINGS.npz ...] [--plot PNG]

Each corpus is clustered as the importer clusters it (`raven.visualizer.importer._agglomerative_labels`, with the
configured cut and minimum size). A cluster's purity: for each member, the share of its `k` nearest neighbours in
the centered embedding space that are members too, averaged over the cluster, where `k` is the cluster's size minus
one, capped at `K_MAX`. Scale-free: 1 means the members are each other's nearest neighbours, whatever the cluster's
size; a random group of the same size would score about `k / N`.

The question is whether purities fall into a tight group and a loose one at a common level across corpora, which
would put a threshold in the gap rather than at a number chosen by eye.
"""

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import clusterlab  # noqa: E402 -- beside this script

K_MAX = 10


def purities(vectors: np.ndarray, labels: np.ndarray) -> list[tuple[int, int, float]]:
    """`(cluster id, size, purity)` per cluster, outliers (-1) excluded. `vectors` centered and normalized."""
    sims = vectors @ vectors.T
    np.fill_diagonal(sims, -np.inf)
    order = np.argsort(-sims, axis=1)[:, :K_MAX]
    out = []
    for cid in sorted(set(labels.tolist()) - {-1}):
        members = np.where(labels == cid)[0]
        k = min(len(members) - 1, K_MAX)
        inside = np.isin(order[members, :k], members).mean()
        out.append((cid, len(members), float(inside)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("corpora", nargs="+", metavar="NAME=EMBEDDINGS.npz")
    parser.add_argument("--plot", help="draw each corpus's purity histogram into this PNG")
    opts = parser.parse_args()

    from raven.visualizer import config as visualizer_config
    from raven.visualizer import importer

    bins = np.linspace(0.0, 1.0, 11)
    results = {}
    for spec in opts.corpora:
        name, path = spec.split("=", 1)
        vectors, _model = clusterlab.load_vectors(path)
        labels = np.asarray(importer._agglomerative_labels(vectors,
                                                           n_clusters=visualizer_config.clustering_n_clusters,
                                                           min_cluster_size=visualizer_config.clustering_min_cluster_size))
        centered = importer._centered(vectors)
        rows = purities(centered, labels)
        results[name] = rows
        if not rows:
            print(f"{name}: {len(vectors)} records, no cluster reached the minimum size; every record is an outlier")
            continue
        p = np.array([r[2] for r in rows])
        sizes = np.array([r[1] for r in rows])
        hist, _ = np.histogram(p, bins=bins)
        print(f"{name}: {len(vectors)} records, {len(rows)} clusters, {int((labels == -1).sum())} outliers")
        print(f"    purity quantiles 10/25/50/75/90%: " + " ".join(f"{q:.2f}" for q in np.quantile(p, [.1, .25, .5, .75, .9])))
        print(f"    histogram by 0.1:  " + " ".join(f"{h:3d}" for h in hist))
        print(f"    clusters of 5-9 records, median purity {np.median(p[sizes < 10]) if (sizes < 10).any() else float('nan'):.2f};"
              f" of 10 or more, {np.median(p[sizes >= 10]) if (sizes >= 10).any() else float('nan'):.2f}")

    if opts.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(results)
        cols = 3
        fig, axes = plt.subplots((n + cols - 1) // cols, cols, figsize=(12, 3 * ((n + cols - 1) // cols)), squeeze=False)
        for ax, (name, rows) in zip(axes.flat, results.items()):
            ax.hist([r[2] for r in rows], bins=np.linspace(0, 1, 21), color="tab:blue")
            ax.set_title(f"{name} ({len(rows)} clusters)", fontsize=9)
            ax.set_xlim(0, 1)
        for ax in list(axes.flat)[n:]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(opts.plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""How well does a 2D map keep the high-dimensional clusters together? Several layouts of one dataset, scored.

    python investigations/highdim-clustering/map_layout.py DATASET.pickle EMBEDDINGS.npz [--out-dir DIR]

`DATASET.pickle` is an importer dataset whose labels are the reference (an agglomerative import, whose clusters were
found in the embedding space); `EMBEDDINGS.npz` is the importer's embedding cache for the same records.

The score is neighbour agreement: for each clustered record, the share of its `K` nearest neighbours on the map that
belong to its own cluster, averaged. The same measured in the embedding space (centered, cosine) is the ceiling a
layout could reach, and a random layout is the floor. Each layout is also drawn, coloured by cluster, into `--out-dir`.
"""

import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import clusterlab  # noqa: E402 -- beside this script

K = 10


def neighbour_agreement(points: np.ndarray, labels: np.ndarray, *, metric: str = "euclidean") -> float:
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=K + 1, metric=metric).fit(points)
    _, idx = nn.kneighbors(points)
    clustered = labels != -1
    same = (labels[idx[:, 1:]] == labels[:, None]).mean(axis=1)
    return float(same[clustered].mean())


def tsne(vectors: np.ndarray, *, exaggeration: float, perplexity: float = 30.0) -> np.ndarray:
    import openTSNE
    return np.asarray(openTSNE.TSNE(n_components=2, perplexity=perplexity, exaggeration=exaggeration, metric="cosine",
                                    negative_gradient_method="fft", initialization="pca", n_jobs=8,
                                    random_state=42).fit(vectors))


def draw(points: np.ndarray, labels: np.ndarray, path: pathlib.Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 8), dpi=100)
    outlier = labels == -1
    ax.scatter(points[outlier, 0], points[outlier, 1], s=6, c="lightgray")
    ax.scatter(points[~outlier, 0], points[~outlier, 1], s=8, c=labels[~outlier], cmap="tab20")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset")
    parser.add_argument("embeddings")
    parser.add_argument("--out-dir", default=".")
    opts = parser.parse_args()
    out_dir = pathlib.Path(opts.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import pickle
    with open(opts.dataset, "rb") as f:
        dataset = pickle.load(f)
    labels = np.asarray(dataset["labels"])
    vectors, _model = clusterlab.load_vectors(opts.embeddings)
    if len(vectors) != len(labels):
        raise SystemExit(f"{len(labels)} records in the dataset and {len(vectors)} in the cache; they must be one import")
    vectors = clusterlab.normalize(vectors)
    centered = clusterlab.center(vectors)
    print(f"{len(labels)} records, {len(set(labels.tolist()) - {-1})} clusters, {int((labels == -1).sum())} outliers")

    print(f"ceiling, the embedding space itself: {neighbour_agreement(centered, labels, metric='cosine'):.3f}")
    rng = np.random.default_rng(0)
    print(f"floor, a random layout:              {neighbour_agreement(rng.normal(size=(len(labels), 2)), labels):.3f}")

    layouts = {"importer's map (fitted on a sample)": np.asarray(dataset["lowdim_data"])}
    for name, vecs, exaggeration in (("full t-SNE, exaggeration 1", vectors, 1.0),
                                     ("full t-SNE, exaggeration 4", vectors, 4.0),
                                     ("full t-SNE on centered vectors, exaggeration 1", centered, 1.0),
                                     ("full t-SNE on centered vectors, exaggeration 4", centered, 4.0)):
        t0 = time.perf_counter()
        layouts[name] = tsne(vecs, exaggeration=exaggeration)
        print(f"  ({name}: {time.perf_counter() - t0:.1f} s)")
    for i, (name, points) in enumerate(layouts.items()):
        score = neighbour_agreement(points, labels)
        print(f"{score:.3f}  {name}")
        draw(points, labels, out_dir / f"layout-{i}.png", f"{name}: agreement {score:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

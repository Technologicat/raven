#!/usr/bin/env python
"""Shared loading and scoring helpers for the high-dimensional clustering experiments.

Not a script; `sweep.py` and `show_clusters.py` import this.

The instrument reads the importer's own embedding cache (`*_embeddings_cache.npz`, written by
`raven.visualizer.importer._get_highdim_semantic_vectors`) rather than re-embedding, so a sweep
costs seconds instead of minutes and every configuration sees byte-identical input.
"""

import numpy as np


def load_vectors(npz_path):
    """Load an importer embedding cache.

    Returns `(vectors, model_name)`, where `vectors` is a rank-2 `np.array` of shape `[N, highdim]`.
    """
    data = np.load(npz_path, allow_pickle=True)
    return data["all_vectors"], str(data["embedding_model"])


def load_titles(pickle_path):
    """Load the entry titles from a Raven-visualizer dataset file, in dataset order.

    The order matches the embedding cache, because both are built by concatenating the per-input-file
    lists in the same order.

    Returns a `list` of `str`.
    """
    return [title for title, _abstract in load_entries(pickle_path)]


def load_entries(pickle_path):
    """Load `(title, abstract)` for each entry of a Raven-visualizer dataset file, in dataset order.

    `abstract` is the empty string where the record has none. Order is as `load_titles`, which see.

    Returns a `list` of `(str, str)`.
    """
    import pickle
    with open(pickle_path, "rb") as f:
        dataset = pickle.load(f)
    return [(entry.title, entry.abstract or "") for entry in dataset["vis_data"]]


def format_for_keyword_extraction(title, abstract):
    """Render one entry the way the keyword-extraction prompt expects it.

    Mirrors `raven.visualizer.importer._format_entry_for_keyword_extraction`, deliberately rather than
    calling it: importing that module runs its top-level setup, which validates devices and can open an
    LLM connection, none of which a read-only script should trigger. Three lines of duplication buys
    that, and this comment is here so the two are kept in step.
    """
    return f"{title}.\n\n{abstract}" if abstract else title


def normalize(vectors):
    """L2-normalize each row of `vectors`. Returns a new `float64` array.

    The upcast is load-bearing rather than tidiness. The importer caches embeddings in whatever dtype
    the embedding device was configured with, so a cache can arrive as `float16` — which carries about
    three decimal digits, enough to move the third decimal of a similarity, and enough to overflow a
    variance reduction over a few million pairs. Every similarity here is quoted to three decimals, so
    the arithmetic is done at a width that supports them.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def center(vectors):
    """Subtract the corpus mean direction from each row, then re-normalize.

    Sentence embeddings are anisotropic: a large shared component points every vector in roughly the
    same direction, which compresses all pairwise cosine similarities into a narrow band near the top
    of the range. That band is what a density-based clusterer has to find structure in, so removing
    the shared component widens the range the clustering actually works over.

    Returns a new array.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    return normalize(vectors - vectors.mean(axis=0, keepdims=True))


def pca_reduce(vectors, n_components):
    """Project `vectors` onto their first `n_components` principal components.

    Returns `(reduced, explained_variance_ratio_cumulative)`, the second being the fraction of total
    variance the kept components account for.
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(vectors)
    return reduced, float(np.sum(pca.explained_variance_ratio_))


def cluster_size_stats(labels):
    """Summarize a labelling.

    `labels`: rank-1 `np.array` of cluster IDs, outliers marked -1.

    Returns an `dict` with `n_clusters`, `n_outliers`, `outlier_fraction`, and the smallest, median
    and largest cluster sizes (0 when there are no clusters).
    """
    labels = np.asarray(labels)
    n_outliers = int(np.sum(labels == -1))
    cluster_ids = np.unique(labels[labels >= 0])
    sizes = np.array([int(np.sum(labels == cid)) for cid in cluster_ids], dtype=np.int64)
    return {"n_clusters": int(len(cluster_ids)),
            "n_outliers": n_outliers,
            "outlier_fraction": n_outliers / max(1, len(labels)),
            "size_min": int(sizes.min()) if len(sizes) else 0,
            "size_median": int(np.median(sizes)) if len(sizes) else 0,
            "size_max": int(sizes.max()) if len(sizes) else 0}


def mean_intra_cluster_similarity(vectors, labels):
    """Mean cosine similarity of each clustered point to its own cluster's mean direction.

    A compactness number, averaged over clustered points and ignoring outliers. It says how tight the
    clusters are; it says nothing about whether they are *separated*, so read it beside
    `mean_nearest_cluster_similarity` rather than alone.

    Returns `float`, or `nan` when nothing was clustered.
    """
    vectors = normalize(np.asarray(vectors, dtype=np.float64))
    labels = np.asarray(labels)
    sims = []
    for cid in np.unique(labels[labels >= 0]):
        members = vectors[labels == cid]
        center = normalize(members.mean(axis=0, keepdims=True))
        sims.append(members @ center[0])
    if not sims:
        return float("nan")
    return float(np.mean(np.concatenate(sims)))


def mean_nearest_cluster_similarity(vectors, labels):
    """Mean cosine similarity between each cluster's mean direction and its nearest other cluster's.

    A separation number: lower is better-separated. Read beside `mean_intra_cluster_similarity` —
    a configuration that raises compactness by shattering the data into near-duplicates of each
    other will show it here.

    Returns `float`, or `nan` when there are fewer than two clusters.
    """
    vectors = normalize(np.asarray(vectors, dtype=np.float64))
    labels = np.asarray(labels)
    cluster_ids = np.unique(labels[labels >= 0])
    if len(cluster_ids) < 2:
        return float("nan")
    centers = normalize(np.stack([vectors[labels == cid].mean(axis=0) for cid in cluster_ids]))
    gram = centers @ centers.T
    np.fill_diagonal(gram, -np.inf)
    return float(np.mean(gram.max(axis=1)))


def cohesion(vectors, members):
    """Mean pairwise cosine similarity among the rows `members` of `vectors`, self-pairs excluded.

    `vectors` must be L2-normalized. `members`: index array of at least two rows; a singleton has no
    pairs, which is why the floor of two in `filter_by_cohesion` is structural.

    Returns `float`.
    """
    block = vectors[members] @ vectors[members].T
    return float(block[~np.eye(len(members), dtype=bool)].mean())


def cohesion_floor(vectors, raw_labels, percentile=10):
    """The cohesion a cluster must reach to count as a topic rather than a residue.

    The `percentile` of every cluster's cohesion, weighted by membership: each cluster of two or more
    contributes its cohesion once per member, so the result is the cohesion of the cluster that the
    `percentile`-th clustered record sits in. Large clusters dominate the weight, which is what lets this
    stand in for "the corpus's established clusters" without a size threshold deciding which those are.

    `vectors`: L2-normalized, as given to `cohesion`. `raw_labels`: a cut of the tree, every record
    labelled, no outlier marker.

    Returns `float`, or `nan` when no cluster has two members.
    """
    raw_labels = np.asarray(raw_labels)
    values = []
    for cid in np.unique(raw_labels):
        members = np.flatnonzero(raw_labels == cid)
        if len(members) >= 2:
            values.append(np.full(len(members), cohesion(vectors, members)))
    if not values:
        return float("nan")
    return float(np.percentile(np.concatenate(values), percentile))


def filter_by_cohesion(vectors, raw_labels, percentile=10):
    """Mark as outliers (-1) the singletons and every cluster less coherent than `cohesion_floor`.

    Surviving clusters are renumbered 0, 1, ... in order of their raw ID.

    Returns `(labels, floor)`.
    """
    raw_labels = np.asarray(raw_labels)
    floor = cohesion_floor(vectors, raw_labels, percentile)
    labels = np.full(len(raw_labels), -1)
    next_id = 0
    for cid in np.unique(raw_labels):
        members = np.flatnonzero(raw_labels == cid)
        if len(members) >= 2 and cohesion(vectors, members) >= floor:
            labels[members] = next_id
            next_id += 1
    return labels, floor

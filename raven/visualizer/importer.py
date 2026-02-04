#!/usr/bin/env python
"""Extract BibTeX data for visualization. This can put an entire field of science into one picture.

This script performs analysis and writes the visualization data file. See `app.py` to plot the results.

This module is both a standalone command-line app, as well as an importable module for Raven-visualizer,
used by its *Import BibTeX* window.
"""

__all__ = ["init",
           "start_task", "has_task", "cancel_task",
           "result_successful", "result_cancelled", "result_errored"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .. import __version__

logger.info(f"Raven-importer version {__version__} loading.")

logger.info("Loading libraries...")
from unpythonic import timer
with timer() as tim:
    import argparse
    import atexit
    import collections
    import copy
    import itertools
    import math
    import os
    import pathlib
    import pickle
    import sys
    import threading
    import traceback
    from typing import Optional

    import bibtexparser

    from unpythonic.env import env
    from unpythonic import box, dyn, ETAEstimator, islice, make_dynvar, sym, timer, uniqify

    # # To connect to the live REPL:  python -m unpythonic.net.client localhost
    # from unpythonic.net import server
    # server.start(locals={"main": sys.modules["__main__"]})

    import numpy as np
    # import pandas as pd

    import torch

    from sklearn.cluster import HDBSCAN

    from ..client import api
    from ..client import config as client_config
    from ..client import mayberemote

    from ..common import bgtask
    from ..common import deviceinfo
    from ..common import nlptools
    from ..common import utils as common_utils

    from . import config as visualizer_config
logger.info(f"    Done in {tim.dt:0.6g}s.")

if visualizer_config.clusters_keyword_method == "llm" or visualizer_config.summarize:
    logger.info("LLM backend needed (for cluster keywords and/or summarization). Setting up connection.")
    from ..librarian import config as librarian_config
    from ..librarian import llmclient
    llm_backend_url = librarian_config.llm_backend_url
    if not llmclient.test_connection(llm_backend_url):
        sys.exit(255)
    llm_settings = llmclient.setup(backend_url=llm_backend_url)

# --------------------------------------------------------------------------------
# Inits that must run before we proceed any further

deviceinfo.validate(visualizer_config.devices)  # modifies in-place if CPU fallback needed

# The extended stopword set (with custom additional stopwords tuned for English-language scientific text).
extended_stopwords = copy.copy(nlptools.default_stopwords)
extended_stopwords.update(x.lower() for x in visualizer_config.custom_stopwords)

# For us (Raven-visualizer importer), running Raven-server is optional.
#
# Using the server avoids loading an extra copy of (possibly large) NLP models in VRAM,
# especially if also Raven-librarian is running simultaneously.
#
# If the server is running, we'll call into its NLP modules.
# If not, no big deal - we'll run in standalone mode, loading the NLP models locally. See `mayberemote`.
#
# It doesn't hurt to always initialize the API. This doesn't yet connect to the server.
# TODO: Maybe Raven-visualizer should init the API in its main module, for consistency with the other apps. OTOH, this is a main module too, for the command-line importer.
api.initialize(raven_server_url=client_config.raven_server_url,
               raven_api_key_file=client_config.raven_api_key_file,
               tts_playback_audio_device=client_config.tts_playback_audio_device,
               stt_capture_audio_device=client_config.stt_capture_audio_device)

# --------------------------------------------------------------------------------
# Common helpers

def update_status_and_log(msg, *, log_indent=0):
    """Log `msg` at info level.

    Additionally, if the optional GUI status update function has been set,
    call that function with `msg` as only argument.

    `msg`: str, the message to display/log.
    `log_indent`: int, indent level for the log message (4 spaces per level).
    """
    dyn.maybe_update_status(msg)  # see the `make_dynvar` call further below; the default behavior is a no-op
    logger.info((4 * log_indent * " ") + msg)


# # https://github.com/sciunto-org/python-bibtexparser/issues/467
# from bibtexparser.library import Library
# from bibtexparser.model import Block, Entry
# class NormalizeFieldNames(bibtexparser.middlewares.middleware.BlockMiddleware):
#     def __init__(self, allow_inplace_modification: bool = True):
#         super().__init__(allow_inplace_modification=allow_inplace_modification,
#                          allow_parallel_execution=True)
#
#     def transform_entry(self, entry: Entry, library: "Library") -> Union[Block, Collection[Block], None]:
#         for field in entry.fields:
#             field.key = field.key.lower()
#         return entry

# # TODO: For some reason, this doesn't see the duplicates. The `seen` set is collected correctly (judging by its `len` after reading in the .bib file), but the warning never triggers though there are duplicate blocks.
#
# from typing import Collection, Union
# from bibtexparser.middlewares import BlockMiddleware
# from bibtexparser.library import Library
# from bibtexparser.model import Block, Entry
# class DetectDuplicateKeys(BlockMiddleware):
#     def __init__(self, allow_inplace_modification: bool = True):
#         super().__init__(allow_inplace_modification=allow_inplace_modification,
#                          allow_parallel_execution=False)
#         self.seen = set()
#
#     def transform_entry(self, entry: Entry, library: "Library") -> Union[Block, Collection[Block], None]:
#         if entry.key in self.seen:
#             logger.warning(f"Duplicate BibTeX entry key detected: '{entry.key}'")
#         self.seen.add(entry.key)
#         return entry

# --------------------------------------------------------------------------------

def parse_input_files(*filenames):
    """Read in and parse BibTeX files `filenames`, and return an `unpythonic.env` containing the parsed data."""
    resolved_filenames = list(uniqify(str(pathlib.Path(fn).expanduser().resolve()) for fn in filenames))

    logger.info(f"Reading input file{'s' if len(resolved_filenames) != 1 else ''}...")
    bibtex_entries_by_filename = {}
    with timer() as tim:
        for j, filename in enumerate(resolved_filenames, start=1):
            update_status_and_log(f"[{j} out of {len(resolved_filenames)}] Reading {filename}...", log_indent=1)
            filename = str(pathlib.Path(filename).expanduser().resolve())
            library = bibtexparser.parse_file(filename,
                                              append_middleware=[bibtexparser.middlewares.NormalizeFieldKeys(),
                                                                 bibtexparser.middlewares.SeparateCoAuthors(),
                                                                 bibtexparser.middlewares.SplitNameParts()])
            bibtex_entries_by_filename[filename] = library.entries
    logger.info(f"    Done in {tim.dt:0.6g}s.")

    logger.info("Extracting data from input records...")
    parsed_data_by_filename = {}
    dehyphenator = None
    with timer() as tim:
        n_total_entries = sum(len(entries) for entries in bibtex_entries_by_filename.values())
        progress.set_micro_count(n_total_entries)
        for j, (filename, entries) in enumerate(bibtex_entries_by_filename.items(), start=1):
            update_status_and_log(f"[{j} out of {len(bibtex_entries_by_filename)}] Extracting data from {filename}...", log_indent=1)
            parsed_data_by_filename[filename] = []
            for entry in entries:
                fields = entry.fields_dict

                if _is_cancelled():
                    return

                # Validate presence of mandatory fields
                entry_valid = True
                for field in ("author", "year", "title"):
                    if field not in fields or not fields[field].value:
                        logger.warning(f"Skipping entry '{entry.key}', reason: no {field}")
                        entry_valid = False
                        break
                if not entry_valid:
                    progress.tick()
                    continue

                authors_str = common_utils.format_bibtex_authors(fields["author"].value)
                year = fields["year"].value
                title = common_utils.normalize_whitespace(common_utils.unicodize_basic_markup(fields["title"].value))

                # abstract is optional
                if "abstract" in fields and fields["abstract"].value:
                    abstract = fields["abstract"].value
                    if visualizer_config.dehyphenate:
                        if dehyphenator is None:  # delayed init - load only if needed, on first use
                            dehyphenator = mayberemote.Dehyphenator(allow_local=True,
                                                                    model_name=visualizer_config.dehyphenation_model,
                                                                    device_string=visualizer_config.devices["sanitize"]["device_string"])
                        abstract = dehyphenator.dehyphenate(abstract)
                    abstract = common_utils.unicodize_basic_markup(abstract)
                else:
                    abstract = None

                # TODO: "keywords" may be populated (though it always isn't). We don't use them anyway, as we extract our own via NLP.
                # TODO: WOS exports may also have "keywords-plus"

                # TODO: Preserve what other fields? Or include a full dump of `entry` as-is? Or its `fields_dict`?

                # Preserving the full author list in BibTeX format allows us to BibTeX export interesting entries from the GUI.
                parsed_data_by_filename[filename].append(env(author=authors_str,
                                                             bibtex_author=fields["author"].value,
                                                             year=year,
                                                             title=title,
                                                             abstract=abstract))
                progress.tick()
    n_entries_total = sum(len(entries) for entries in bibtex_entries_by_filename.values())
    logger.info(f"    {n_entries_total} total entries processed.")
    logger.info(f"    Done in {tim.dt:0.6g}s [avg {n_entries_total / tim.dt:0.6g} entries/s].")

    return env(parsed_data_by_filename=parsed_data_by_filename,
               n_entries_total=n_entries_total,
               resolved_filenames=resolved_filenames)


def get_highdim_semantic_vectors(input_data):
    """Compute (or read from disk cache) the semantic embedding vectors for the given dataset.

    `input_data`: output of `parse_input_files`, which see.

    NOTE: The semantic vectors produced by the embedder live on the surface of the unit hypersphere
    in the high-dimensional space - i.e., each vector is a direction in that space. If you need to
    compare them, you can use cosine similarity.
    """
    logger.info("Preparing semantic space...")

    # Caches are per input file, to make it fast to concatenate new files to the dataset.
    embeddings_cache_filenames = {fn: common_utils.make_cache_filename(fn, "embeddings_cache", "npz") for fn in input_data.resolved_filenames}

    all_vectors_by_filename = {}
    embedder = None
    progress.set_micro_count(len(input_data.parsed_data_by_filename))
    for j, (filename, entries) in enumerate(input_data.parsed_data_by_filename.items(), start=1):
        if _is_cancelled():
            return

        update_status_and_log(f"[{j} out of {len(input_data.parsed_data_by_filename)}] Preparing semantic vectors for {filename}...", log_indent=1)

        # TODO: clear memory between input files to avoid running out of RAM/VRAM when we have lots of files

        # Check embeddings cache for this input file
        embeddings_cache_filename = embeddings_cache_filenames[filename]
        cache_state = "unknown"
        if os.path.exists(embeddings_cache_filename):
            logger.info(f"        Checking cached embeddings '{embeddings_cache_filename}'...")
            if common_utils.validate_cache_mtime(embeddings_cache_filename, filename):  # is cache valid, judging by mtime
                with timer() as tim:
                    embeddings_cached_data = np.load(embeddings_cache_filename)
                logger.info(f"            Done in {tim.dt:0.6g}s.")
                embedding_model_for_this_cache = embeddings_cached_data["embedding_model"]
                if embedding_model_for_this_cache == visualizer_config.embedding_model:
                    cache_state = "ok"
                else:
                    cache_state = "cache file has different embedding model"
            else:
                cache_state = "cache file out of date, original file changed"
        else:
            cache_state = "no such file"
        if cache_state == "ok":  # use cache
            logger.info(f"        Using cached embeddings '{embeddings_cache_filename}'...")
            all_vectors = embeddings_cached_data["all_vectors"]
        else:  # no cache
            logger.info(f"        No cached embeddings '{embeddings_cache_filename}', reason: {cache_state}")
            logger.info("        Computing embeddings...")
            if embedder is None:  # delayed init - load only if needed, on first use
                embedder = mayberemote.Embedder(allow_local=True,
                                                model_name=visualizer_config.embedding_model,
                                                device_string=visualizer_config.devices["embeddings"]["device_string"],
                                                dtype=visualizer_config.devices["embeddings"]["dtype"])
            logger.info("        Encoding...")
            with timer() as tim:
                def format_entry_for_vectorization(entry: env) -> str:
                    # return entry.title  # original solution - with mpnet, this works best (and we don't always necessarily have an abstract)
                    # return entry.abstract  # early versions with snowflake used this
                    # return " ".join(entry.keywords)  # meh, not all entries have keywords

                    # Maybe best of both worlds?
                    if entry.abstract:
                        return entry.title + ".\n\n" + entry.abstract
                    else:
                        return entry.title

                all_inputs = [format_entry_for_vectorization(entry) for entry in entries]
                all_vectors = embedder.encode(all_inputs)
                # Round-trip to force truncation, if needed.
                # This matters to make the "cluster centers" coincide with the original datapoints when clustering is disabled.
                if embedder.is_local():  # TODO: We can only do this when running locally (otherwise, the device can be different from what we expect). What is the correct solution in the remote case?
                    all_vectors = torch.tensor(all_vectors,
                                               device=visualizer_config.devices["embeddings"]["device_string"],
                                               dtype=visualizer_config.devices["embeddings"]["dtype"])
                    all_vectors = all_vectors.detach().cpu().numpy()
            logger.info(f"            Done in {tim.dt:0.6g}s [avg {len(all_inputs) / tim.dt:0.6g} entries/s].")

            logger.info(f"        Caching embeddings for this dataset to '{embeddings_cache_filename}'...")
            with timer() as tim:
                np.savez_compressed(embeddings_cache_filename, all_vectors=all_vectors, embedding_model=visualizer_config.embedding_model)
            logger.info(f"            Done in {tim.dt:0.6g}s.")
        all_vectors_by_filename[filename] = all_vectors
        progress.tick()
    all_vectors = np.concatenate(list(all_vectors_by_filename.values()))

    return all_vectors


def cluster_highdim_semantic_vectors(all_vectors, max_n=10000):
    """Cluster the semantic vectors in the high-dimensional space.

    `all_vectors`: output of `get_highdim_semantic_vectors`, which see.

    `max_n`: int, maximum number of vectors to use for clustering.

             If the dataset is larger than `max_n`, then `max_n` vectors are sampled uniformly at random.

             Typically, in a large dataset, many items are very similar. This improves performance,
             reducing both run time and memory use.

    Return a tuple `(unique_vs, n_clusters)`, where:

        `unique_vs`: Rank-2 `np.array` of shape `[N, highdim]`, a sample of representative vectors,
                     stratified across detected clusters. Here `N = min(len(all_vectors), max_n)`.

        `n_clusters`: `int`, the number of detected clusters.

    NOTE: This is a first-step clustering, which gathers a stratified sample of high-dimensional vectors across the
          detected clusters. The final clustering for visualization is performed later, in 2D.

          This data is better than using just the cluster medoids for training the dimension reduction mapping.
          The `max_n` limit keeps the dimension reduction training reasonably fast.

    Raises `RuntimeError` if the high-dimensional vectors are so spread out that not even one cluster is detected.
    """
    update_status_and_log("Detecting semantic clusters...", log_indent=1)
    logger.info(f"        Full dataset has {len(all_vectors)} data points.")
    with timer() as tim:
        # The semantic vectors represent directions in the latent space, so we can compare them using the cosine metric.
        clusterer = HDBSCAN(min_cluster_size=5, min_samples=1, cluster_selection_method="leaf", metric="cosine", store_centers="medoid")
        # clusterer = HDBSCAN(min_cluster_size=5, min_samples=1, cluster_selection_method="leaf", metric="euclidean", store_centers="medoid", algorithm="kd_tree")
        # clusterer = HDBSCAN(min_cluster_size=20, min_samples=5, metric="cosine", store_centers="medoid")  # n_jobs=6,  # n_jobs=-1,
        if len(all_vectors) <= max_n:
            logger.info(f"        Dataset is small ({len(all_vectors)} ≤ {max_n}), using the full dataset.")
            fit_idxs = np.arange(len(all_vectors), dtype=np.int64)
            fit_vectors = all_vectors
        else:  # Too many (HDBSCAN runs out of memory on 32GB for ~50k).
            # Take a random set of 10k entries, and hope we hit all the important clusters.
            logger.info(f"        Dataset is large ({len(all_vectors)} > {max_n}), picking {max_n} random entries for cluster detection.")
            fit_idxs = np.random.randint(len(all_vectors), size=max_n)
            fit_vectors = all_vectors[fit_idxs]

        dyn.maybe_update_status(f"Detecting semantic clusters using {len(fit_vectors)} samples...")
        clusterer.fit(fit_vectors)
        unique_vs = clusterer.medoids_
        n_clusters = np.shape(unique_vs)[0]
        n_outliers = sum(clusterer.labels_ == -1)

        # Pick representative points for training the dimension reduction.
        if n_clusters > 0:
            logger.info(f"        Detected {n_clusters} seed clusters in high-dimensional space, with {n_outliers} outlier data points (out of {len(fit_vectors)} data points used).")

            # Instead of using the medoids, pick a few random representative points from each cluster. We just take the first up-to-k points for now.
            # TODO: This discards the medoids (`clusterer.medoids_`). Include them, too?
            points_by_cluster = [all_vectors[fit_idxs[clusterer.labels_ == label]] for label in range(n_clusters)]  # gather sublists by cluster, discard outliers (label -1)
            samples_by_cluster = [points[:20, :] for points in points_by_cluster]
            unique_vs = np.concatenate(samples_by_cluster)
            logger.info(f"        Picked a total of {len(unique_vs)} representative points from the detected seed clusters.")
        else:
            logger.info("        No clusters detected in high-dimensional data. Cannot train dimension reduction for this dataset. Canceling.")
            raise RuntimeError("No clusters detected in high-dimensional data. Cannot train dimension reduction for this dataset. Canceling.")
    logger.info(f"        Done in {tim.dt:0.6g}s.")

    return unique_vs, n_clusters


def reduce_dimension(unique_vs, n_clusters, all_vectors):
    """Reduce the dimensionality of the semantic vectors from the high-dimensional space into 2D.

    `unique_vs`: output of `cluster_highdim_semantic_vectors`, which see.
    `n_clusters`: output of `cluster_highdim_semantic_vectors`, which see.
    `all_vectors`: output of `get_highdim_semantic_vectors`, which see.

    Returns `lowdim_data`, a rank-2 `np.array` of shape `[N, 2]`, containing a 2D point for each entry.
    In the 2D representation, semantically similar entries are mapped near each other, facilitating
    semantic exploration.

    NOTE: The only structure in the resulting 2D map is that similar entries appear near each other.
          Long distances in the 2D map are meaningless. Similarly, gaps are meaningless.

    NOTE: It is meaningless to try to interpolate between clusters to guess what could be there.
          This is not a projection of a metric space, but rather just an arbitrary, compressed
          representation of a given dataset. Essentially, the high-dimensional data manifold
          (in this case, hypersphere surface) is forcibly crumpled into a planar 2D representation,
          using the available plot area efficiently.

    The dimension reducer runs in a single thread on CPU and may take a long time (minutes).
    """
    logger.info("Dimension reduction for visualization...")
    update_status_and_log(f"Loading dimension reduction library for '{visualizer_config.vis_method}'...", log_indent=1)
    progress.set_micro_count(3)  # load library, train mapping, apply mapping
    with timer() as tim:
        if visualizer_config.vis_method == "tsne":
            # t-distributed Stochastic Neighbor Embedding
            #
            # We use the empirical settings by Gove et al. (2022) that generally (across 691 different datasets)
            # prioritize accurate neighbors (rather than accurate distances).
            #
            # Gove et al. write:
            #
            #   If those hyperparameters don’t produce good visualizations, try using perplexity in the range 2-16,
            #   exaggeration in the range 1-8, and learning rate in the range 10-640. We found that accurate visualizations
            #   tended to have hyperparameters in these ranges. To guide your exploration, you can first try perplexity
            #   near 16 or n/100 (where n is the number of data points); exaggeration near 1; and learning rate near 10 or n/12.
            #
            # Blog post, with link to preprint PDF (Gove et al., 2022. New Guidance for Using t-SNE: Alternative Defaults,
            # Hyperparameter Selection Automation, and Comparative Evaluation):
            #   https://twosixtech.com/new-guidance-for-using-t-sne/
            #
            # See also Böhm et al. (2022), which discusses the similarities between the nonlinear projections produced by
            # t-SNE, UMAP, and laplacian eigenmaps.
            #   https://arxiv.org/abs/2007.08902
            #
            # See also:
            #   https://pgg1610.github.io/blog_fastpages/python/data-visualization/2021/02/03/tSNEvsUMAP.html
            import openTSNE
            trans = openTSNE.TSNE(n_components=2,
                                  perplexity=max(16.0, len(unique_vs) / 100.0),
                                  exaggeration=1.0,
                                  learning_rate=10.0,
                                  metric="cosine",
                                  n_iter=500,
                                  # n_jobs=6,
                                  # n_jobs=n_jobs,
                                  # initialization="pca",
                                  # initialization="spectral",
                                  random_state=42)
        elif visualizer_config.vis_method == "umap":
            # UMAP (Uniform Manifold Approximation and Projection) assumes that the data is uniformly distributed
            # on a Riemannian manifold; the Riemannian metric is locally constant (at least approximately);
            # and that the manifold is locally connected. It attempts to preserve the topological structure
            # of this manifold.
            #
            # When the assumptions hold, it typically produces very nice-looking results, but it is
            # very, very much (multiple times) slower than t-SNE. See the link to Böhm et al. (2022) above;
            # with appropriate hyperparameter values, t-SNE can produce results that look like those from UMAP.
            #
            # See McInnes et al. (2020, revised v3 of paper originally published in 2018):
            #   https://arxiv.org/abs/1802.03426
            #
            # NOTE: UMAP needs to load TensorFlow, whereas the rest of the importer uses PyTorch.
            import umap
            trans = umap.UMAP(n_components=2,
                              # n_neighbors=max(1, len(unique_vs) // 2),
                              n_neighbors=100,
                              metric="cosine",
                              min_dist=0.8,
                              # n_jobs=20,
                              random_state=42,
                              low_memory=False)
        else:
            raise ValueError(f"Unknown `config.vis_method` '{visualizer_config.vis_method}'; valid: 'tsne', 'umap'; please check your `raven/visualizer/config.py` and restart the app.")
    progress.tick()
    logger.info(f"        Done in {tim.dt:0.6g}s.")

    if _is_cancelled():
        return

    with timer() as tim:
        update_status_and_log(f"Learning dimension reduction into 2D from the detected {n_clusters} semantic clusters, using {len(unique_vs)} representative points...", log_indent=1)
        trans = trans.fit(unique_vs)
        # trans = trans.fit(all_vectors)  # DEBUG: high quality result, but extremely expensive! (several minutes for 5k points)
    progress.tick()
    logger.info(f"        Done in {tim.dt:0.6g}s.")

    if _is_cancelled():
        return

    with timer() as tim:
        update_status_and_log(f"Applying learned dimension reduction to full dataset [n = {len(all_vectors)}]...", log_indent=1)
        lowdim_data = trans.transform(all_vectors)
        # lowdim_reprs = trans.transform(unique_vs)  # DEBUG: where did our representative points end up in the 2D representation?
    progress.tick()
    logger.info(f"        Done in {tim.dt:0.6g}s.")

    return lowdim_data


def cluster_lowdim_data(input_data, lowdim_data):
    """Cluster the low-dimensional (2D) data.

    `input_data`: output of `parse_input_files`, which see.
    `lowdim_data`: output of `reduce_dimension`, which see.

    Returns (`vis_data`, `labels`, `n_vis_clusters`, `n_vis_outliers`), where:
        - `vis_data`: list, concatenated entries from all input files in `input_data` (this is what the visualizer reads)
        - `labels`: rank-1 `np.array`, cluster ID (0-based) for each item in `vis_data`. Outliers get the cluster ID -1.
        - `n_vis_clusters`: int, how many clusters were detected
        - `n_vis_outliers`: int, how many data points were left without any cluster

    Mutates `input_data`, adding the fields `item.cluster_id` and `item.cluster_probability` (confidence value).
    """
    # Judging by paper titles in the test dataset, the initial high-dimensional clustering (and then training the dimension reduction using those points)
    # seems to "seed" the 2D clusters reasonably, so that the t-SNE fit, using the representative points only, maps the full dataset into the 2D space
    # in a semantically sensible manner.
    update_status_and_log("Detecting clusters in 2D semantic map...", log_indent=1)

    # Concatenate data from individual input files into one large dataset. This is what we will visualize.
    vis_data = list(itertools.chain.from_iterable(input_data.parsed_data_by_filename.values()))

    with timer() as tim:
        vis_clusterer = HDBSCAN(min_cluster_size=10, min_samples=2, cluster_selection_method="leaf", metric="euclidean", store_centers="medoid")
        vis_clusterer.fit(lowdim_data)
        n_vis_clusters = np.shape(vis_clusterer.medoids_)[0]
        n_vis_outliers = sum(vis_clusterer.labels_ == -1)
        # Tag the data points with the cluster IDs. Outliers get the tag -1.
        for idx, entry in enumerate(vis_data):
            entry.cluster_id = vis_clusterer.labels_[idx]
            entry.cluster_probability = vis_clusterer.probabilities_[idx]
        if n_vis_clusters > 0:
            logger.info(f"        Detected {n_vis_clusters} clusters in 2D, with {n_vis_outliers} outlier data points (out of {len(lowdim_data)} data points total).")
    logger.info(f"        Done in {tim.dt:0.6g}s.")

    return vis_data, vis_clusterer.labels_, n_vis_clusters, n_vis_outliers


def format_entry_for_keyword_extraction(entry: env) -> str:
    """Format a BibTeX entry into plain text for the keyword extraction step.

    Output format:

        If the entry has an abstract:

            Entry title here.

            Blah blah blah...
            Blah blah...

        The full-stop and the two newlines after the title are added.
        Both the title and the abstract are pasted as-is.

        If the entry does NOT have an abstract:

            Entry title here

        The title is pasted as-is.

    Note that in either case, authors and year are not mentioned.
    This is because they are not relevant for keyword extraction.
    """
    # return entry.title
    if entry.abstract:
        return entry.title + ".\n\n" + entry.abstract
    else:
        return entry.title

def extract_keywords(input_data, max_vis_kw=6):
    """Extract keywords for the dataset.

    This is based on a combination of word frequency analysis (ignoring stopwords) and named entity recognition (NER),
    the latter powered by a NLP model.

    `input_data`: output of `parse_input_files`, which see.
    `max_vis_kw`: how many highest-frequency keywords (after ignoring stopwords) to keep for `item.vis_keywords` (see below).

    Returns a `dict`: `{keyword0: count0, ...}`, sorted by number of occurrences, descending.
    The counts are measured across the whole dataset.

    Additionally, `input_data` is mutated to add the keyword fields to each item:
        `item.keywords` is in the same format as the return value of `extract_keywords` (but for keywords of that one item).
        `item.entities` is a `set` of `str`, because entities do not have the number of occurrences available.
        `item.vis_keywords` is a `list` of `str`, which contains the `max_vis_kw` highest-frequency keywords in that item,
                            and all detected named entities that did not already occur among those highest-frequency keywords.

    Currently, Raven only uses the per-item `vis_keywords` to compute the per-cluster `vis_keywords` (which are shown in the visualizer GUI).
    """
    logger.info("NLP analysis...")

    # Caches are per input file, so that it is fast to concatenate new files to the dataset.
    nlp_cache_filenames = {fn: common_utils.make_cache_filename(fn, "nlp_cache", "pickle") for fn in input_data.resolved_filenames}
    nlp_cache_version = 1

    all_keywords_by_filename = {}
    nlp = None
    progress.set_micro_count(len(input_data.parsed_data_by_filename))
    for j, (filename, entries) in enumerate(input_data.parsed_data_by_filename.items(), start=1):
        if _is_cancelled():
            return

        update_status_and_log(f"[{j} out of {len(input_data.parsed_data_by_filename)}] NLP analysis for {filename}...", log_indent=1)

        nlp_cache_filename = nlp_cache_filenames[filename]
        cache_state = "unknown"
        if os.path.exists(nlp_cache_filename):
            logger.info(f"        Checking cached NLP data '{nlp_cache_filename}'...")
            if common_utils.validate_cache_mtime(nlp_cache_filename, filename):  # is cache valid, judging by mtime
                with timer() as tim:
                    with open(nlp_cache_filename, "rb") as nlp_cache_file:
                        nlp_cached_data = pickle.load(nlp_cache_file)
                logger.info(f"            Done in {tim.dt:0.6g}s.")
                if nlp_cached_data["version"] == nlp_cache_version:
                    cache_state = "ok"
                else:
                    cache_state = f"cache file has version {nlp_cached_data['version']}, expected {nlp_cache_version}"
            else:
                cache_state = "cache file out of date, original file changed"
        else:
            cache_state = "no such file"
        if cache_state == "ok":
            logger.info(f"        Using cached NLP data {nlp_cache_filename}...")
            all_keywords_for_this_file = nlp_cached_data["all_keywords"]
            entry_keywords_for_this_file = nlp_cached_data["entry_keywords"]
            entry_entities_for_this_file = nlp_cached_data["entry_entities"]
            logger.info(f"        {len(all_keywords_for_this_file)} keywords collected from {filename} (cached).")

            all_keywords_by_filename[filename] = all_keywords_for_this_file
        else:
            logger.info(f"        No cached NLP data '{nlp_cache_filename}', reason: {cache_state}")
            logger.info("        Extracting keywords...")
            if nlp is None:
                update_status_and_log("Loading NLP pipeline for keyword analysis...", log_indent=2)
                nlp = mayberemote.NLP(allow_local=True,
                                      model_name=visualizer_config.spacy_model,
                                      device_string=visualizer_config.devices["nlp"]["device_string"])
                update_status_and_log(f"[{j} out of {len(input_data.parsed_data_by_filename)}] NLP analysis for {filename}...", log_indent=1)  # restore old message  # TODO: DRY log messages

            logger.info("        Computing keyword set...")
            logger.info(f"            Running NLP pipeline for data from {filename}...")
            with timer() as tim:
                all_texts_for_nlp_for_this_file = [format_entry_for_keyword_extraction(entry) for entry in entries]

                # The pipe-batched version processed about 25 entries per second on an RTX 3070 Ti mobile GPU.
                # TODO: Can we minibatch the NLP pipelining to save VRAM when using the Transformers model?
                # all_docs_for_this_file = [nlp.analyze(text) for text in all_texts_for_nlp_for_this_file]
                all_docs_for_this_file = nlp.analyze(all_texts_for_nlp_for_this_file)
                # all_docs_for_this_file = []
                # for j, text in enumerate(all_texts_for_nlp_for_this_file):
                #     if j % 100 == 0:
                #         logger.info(f"    {j + 1} out of {len(all_texts_for_nlp_for_this_file)}...")
                #     all_docs_for_this_file.append(nlp.analyze(text))
            logger.info(f"                Done in {tim.dt:0.6g}s [avg {input_data.n_entries_total / tim.dt:0.6g} entries/s].")

            # TODO: Should we trim the keywords across the whole dataset? We currently trim each input file separately.
            logger.info(f"            Frequency analysis across all documents from NLP pipeline results for data from {filename}...")
            with timer() as tim:
                all_keywords_for_this_file = nlptools.count_frequencies(all_docs_for_this_file,
                                                                        stopwords=extended_stopwords)
                all_keywords_for_this_file = dict(sorted(all_keywords_for_this_file.items(), key=lambda kv: -kv[1]))  # sort by number of occurrences, descending
                # all_keywords_for_this_file = " ".join(sorted(all_keywords_for_this_file.keys()))
                # logger.info(f"keywords collected from {filename}: {all_keywords_for_this_file}")  # DEBUG
            logger.info(f"                Done in {tim.dt:0.6g}s.")

            logger.info(f"        {len(all_keywords_for_this_file)} keywords collected from {filename}.")
            all_keywords_by_filename[filename] = all_keywords_for_this_file

            # logger.info(f"Keywords for {filename}, in order of frequency (descending):")
            # logger.info(all_keywords_for_this_file)
            # logger.info("The same keywords, alphabetically:")
            # alphabetized_keywords_debug = dict(sorted(all_keywords_for_this_file.items(), key=lambda kv: kv[0]))
            # logger.info(", ".join(alphabetized_keywords_debug.keys()))  # DEBUG

            # Per-document analysis.
            logger.info(f"        Per-document frequency analysis and NER (named entity recognition) for data from {filename}...")
            fa_times = []
            ner_times = []
            with timer() as tim:
                entry_keywords_for_this_file = []
                entry_entities_for_this_file = []
                for entry, doc in zip(entries, all_docs_for_this_file):
                    if _is_cancelled():
                        return

                    # This is essentially just dumb occurrence counting, aided by an extended list of stopwords manually tuned for scientific texts.
                    # Everything smarter occurs further below, in `collect_cluster_keywords`, where we actually analyze this data to ignore uselessly
                    # common words (when determining keywords for clusters).

                    # Frequency analysis.
                    with timer() as tim2:
                        kws = nlptools.count_frequencies(doc,
                                                         stopwords=extended_stopwords)
                    fa_times.append(tim2.dt)

                    # Named entity recognition (NER).
                    # TODO: Update named entities to the cluster keywords, too. The thing is, we don't have frequency information for named entities, so we can't decide which are the most relevant ones. TODO: Now we do. Fix this.
                    with timer() as tim2:
                        ents = set(nlptools.detect_named_entities(doc,
                                                                  stopwords=extended_stopwords).keys())
                    ner_times.append(tim2.dt)

                    # for cache saving (and unified handling)
                    entry_keywords_for_this_file.append(kws)
                    entry_entities_for_this_file.append(ents)
            logger.info(f"            Done in {tim.dt:0.6g}s (frequency analysis {math.fsum(fa_times):0.6g}s, NER {math.fsum(ner_times):0.6g}s).")

            # Save into cache
            logger.info(f"        Caching NLP data '{nlp_cache_filename}'...")
            with timer() as tim:
                nlp_cached_data = {"version": nlp_cache_version,
                                   "entry_keywords": entry_keywords_for_this_file,
                                   "entry_entities": entry_entities_for_this_file,
                                   "all_keywords": all_keywords_for_this_file}
                with open(nlp_cache_filename, "wb") as nlp_cache_file:
                    pickle.dump(nlp_cached_data, nlp_cache_file)
            logger.info(f"            Done in {tim.dt:0.6g}s.")

        # Populate the detected keywords and named entities into the parsed data items.
        #
        logger.info(f"        Tagging data from {filename} with keywords...")
        with timer() as tim:
            for entry, kws, ents in zip(entries, entry_keywords_for_this_file, entry_entities_for_this_file):
                entry.keywords = kws  # counter
                entry.entities = ents  # no frequency information, so just a set
        logger.info(f"            Done in {tim.dt:0.6g}s.")
        progress.tick()

    # Merge the keyword data from all input files.
    #
    update_status_and_log("Combining keywords from all input files...", log_indent=1)
    with timer() as tim:
        all_keywords = collections.defaultdict(lambda: 0)
        for filename, kws in all_keywords_by_filename.items():
            logger.info(f"        Combining keywords from {filename}...")
            for kw, count in kws.items():
                all_keywords[kw] += count
        all_keywords = dict(sorted(all_keywords.items(), key=lambda kv: -kv[1]))  # sort by number of occurrences, descending
    logger.info(f"        Done in {tim.dt:0.6g}s.")
    logger.info(f"        {len(all_keywords)} unique keywords collected across the whole dataset.")

    logger.info("        Top 20 keywords across the whole dataset (in order of decreasing frequency):")
    for k, (kw, count) in enumerate(islice(all_keywords.items())[:20], start=1):
        logger.info(f"            {k:2d}. {kw} ({count})")

    # Postprocess the keywords, for use in visualization.
    #
    # This part is cheap, so we always recompute it, so that we can mod the algorithm easily.
    #
    # TODO: Currently the visualization keywords for individual entries (`entry.vis_keywords`) are unused. We use vis keywords for clusters, computed further below.
    #
    update_status_and_log("Tagging data with visualization keywords...", log_indent=1)
    with timer() as tim:
        for filename, entries in input_data.parsed_data_by_filename.items():
            logger.info(f"        Tagging visualization keywords for {filename}...")
            for entry in entries:
                kws = entry.keywords
                ents = entry.entities

                # Find the highest-frequency keywords for this entry.
                kws = {k: v for k, v in kws.items() if k in all_keywords}  # Drop keywords that did not appear in the trimmed keyword analysis across the whole dataset.
                kws = list(sorted(kws.keys(),
                                  key=lambda word: all_keywords.get(word, 0),  # sort by frequency in full corpus, descending
                                  reverse=True))
                kws = kws[:max_vis_kw]

                # Add in the named entities.
                ents = {x for x in ents if x.lower() not in kws}  # omit entities already present in the highest-frequency keywords
                ents = list(sorted(ents))

                # Human-readable list of the most important detected keywords for this entry. Currently unused.
                entry.vis_keywords = kws + ents
    logger.info(f"        Done in {tim.dt:0.6g}s.")

    # for filename, entries in parsed_data_by_filename.items():  # DEBUG - prints the full dataset!
    #    for entry in entries:
    #        logger.info(f"{entry.author} ({entry.year}): {entry.title}")
    #        logger.info(f"    {entry.abstract}" if entry.abstract else "[no abstract]")
    #
    #        logger.info(f"    {entry.vis_keywords}")
    #        logger.info("")

    return all_keywords

def collect_cluster_keywords(vis_data, n_vis_clusters, all_keywords, max_vis_kw=6, fraction=0.1):
    """Collect a set of keywords for each visualization cluster (2D), based on the per-entry detected keywords.

    `vis_data`: output of `cluster_lowdim_data`, which see.
    `n_vis_clusters`: output of `cluster_lowdim_data`, which see.

    `all_keywords`: Used when `config.clusters_keyword_method == "frequencies"`.

                    Output of `extract_keywords`, which see.

    `max_vis_kw`: how many keywords to keep for each cluster.

    `fraction`: Used when `config.clusters_keyword_method == "frequencies"`.

                IMPORTANT. The source of the keyword suggestion magic. See `nlptools.suggest_keywords`.

    Returns `vis_keywords_by_cluster`, a list, where the `k`th item is a list of keywords (`str`) for cluster ID `k`.
    For each cluster, the keywords are sorted by number of occurrences (descending) across the whole dataset.
    """
    update_status_and_log("Extracting keywords for each cluster...", log_indent=1)

    if visualizer_config.clusters_keyword_method == "frequencies":
        with timer() as tim:
            logger.info("        Collecting keywords from data points in each cluster (`suggest_keywords` will treat each cluster as a 'document')...")
            keywords_by_cluster = [collections.Counter() for _ in range(n_vis_clusters)]
            for entry in vis_data:
                if entry.cluster_id >= 0:  # not an outlier
                    # Update cluster keyword counters from this entry.
                    # NOTE: We operate on the already-filtered data that excludes stopwords (see `extract_keywords` for details).
                    # TODO: We could also use `entry.cluster_probability` for something here.
                    # TODO: Including entities would be nice, but they don't currently have frequency information.
                    keywords_by_cluster[entry.cluster_id].update(entry.keywords)  # inject keywords of this entry to the keywords of the cluster this entry belongs to

            # Here, each cluster is a "document" for the purposes of suggesting keywords.
            vis_keywords_by_cluster = nlptools.suggest_keywords(per_document_frequencies=keywords_by_cluster,
                                                                corpus_frequencies=all_keywords,
                                                                threshold_fraction=fraction,
                                                                max_keywords=max_vis_kw)
        logger.info(f"        Done in {tim.dt:0.6g}s.")
    elif visualizer_config.clusters_keyword_method == "llm":
        # First, group entries by cluster
        entries_by_cluster = collections.defaultdict(list)  # {cluster_id0: [entry0, ...], ...}
        for entry in vis_data:
            if entry.cluster_id >= 0:  # not an outlier
                entries_by_cluster[entry.cluster_id].append(entry)

        eta_estimator = ETAEstimator(total=len(entries_by_cluster), keep_last=50)
        vis_keywords_by_cluster = []
        for cluster_id, entries in sorted(entries_by_cluster.items()):  # default sort is fine, since the key is the cluster ID
            logger.info(f"        Extracting keywords for cluster #{cluster_id} (number of clusters: {len(entries_by_cluster.items())}); {eta_estimator.formatted_eta}")
            # Use two blank lines as an entry separator (works also when the abstract has paragraph breaks; also clearly associates which title goes with which abstract).
            cluster_texts = "\n\n\n".join(format_entry_for_keyword_extraction(entry).strip() for entry in entries)
            prompt = f"{visualizer_config.clusters_llm_keyword_extraction_prompt}\n-----\n\n{cluster_texts}"

            logger.info(f"        LLM prompt for cluster #{cluster_id}:\n{prompt}")

            # Ask the LLM to provide keywords
            raw_output_text, scrubbed_output_text = llmclient.perform_throwaway_task(llm_settings,
                                                                                     instruction=prompt,
                                                                                     on_progress=llmclient.make_console_progress_handler("."))

            logger.info(f"        LLM output (raw) for cluster #{cluster_id}:\n{raw_output_text}")
            logger.info(f"        LLM output (final answer) for cluster #{cluster_id}:\n{scrubbed_output_text}")

            # TODO: wrap this in a retry mechanism (up to 3 times?)
            if scrubbed_output_text.strip().lower() == "keyword extraction failed":
                logger.warning(f"        The LLM could not identify a common theme for cluster #{cluster_id}.")
                cluster_keywords = ["<unknown topic>"]
            else:
                cluster_keywords = [keyword.strip() for keyword in scrubbed_output_text.split(",")]
            vis_keywords_by_cluster.append(cluster_keywords)
            eta_estimator.tick()
    else:
        error_msg = f"Unknown cluster keyword method '{visualizer_config.clusters_keyword_method}'; valid: 'frequencies', 'llm'. Please check your `raven.visualizer.config`."
        logger.error(f"collect_cluster_keywords: {error_msg}")
        raise ValueError(error_msg)

    return vis_keywords_by_cluster


def summarize(input_data):
    """Summarize each item of the dataset, using an LLM.

    Requires an LLM backend and `visualizer_config.summarize = True`.

    `input_data`: output of `parse_input_files`, which see.

    No return value.

    Mutates `input_data`, adding an `item.summary` field to each item. The field value is:
        - `str`, the summary, if there was an abstract (so a summary could be created).
        - `None`, if there was no abstract (so a summary could not be created).
    """
    logger.info("Summarizing abstracts using LLM...")
    n_total_entries = sum(len(entries) for entries in input_data.parsed_data_by_filename.values())
    progress.set_micro_count(n_total_entries)
    for k, (filename, entries) in enumerate(input_data.parsed_data_by_filename.items(), start=1):
        logger.info(f"    [input file {k} out of {len(input_data.parsed_data_by_filename)}] Summarizing abstracts from {filename}...")
        with timer() as tim:
            for j, entry in enumerate(entries, start=1):
                if _is_cancelled():
                    return

                if entry.abstract:
                    update_status_and_log(f"[input file {k} out of {len(input_data.parsed_data_by_filename)}] Summarizing entry {j} out of {len(entries)}: {entry.author} ({entry.year}): {entry.title}",
                                          log_indent=2)
                    entry_text = f"Title: {entry.title}\n\nAbstract: {entry.abstract}"
                    prompt = f"{visualizer_config.summarize_llm_prompt}\n-----\n\n{entry_text}"

                    raw_output_text, scrubbed_output_text = llmclient.perform_throwaway_task(llm_settings,
                                                                                             instruction=prompt,
                                                                                             on_progress=llmclient.make_console_progress_handler("."))

                    logger.info(f"        LLM output (raw):\n{raw_output_text}")
                    logger.info(f"        LLM output (final answer):\n{scrubbed_output_text}")

                    if scrubbed_output_text.strip().lower() == "summarization failed":
                        logger.warning(f"        The LLM could not summarize entry: {entry.author} ({entry.year}): {entry.title}")
                        summary = None
                    else:
                        summary = scrubbed_output_text.strip()
                else:
                    update_status_and_log(f"[input file {k} out of {len(input_data.parsed_data_by_filename)}] Skipping entry {j} out of {len(entries)} (no abstract to summarize): {entry.author} ({entry.year}): {entry.title}",
                                          log_indent=2)
                    summary = None
                entry.summary = summary
                progress.tick()
        logger.info(f"        Done in {tim.dt:0.6g}s [avg {len(entries) / tim.dt:0.6g} entries/s].")

# --------------------------------------------------------------------------------
# Background task management

# TODO: Return value for whether the background task succeeded or failed? (Can use the `task_env.done_callback` mechanism for this.)

status_box = box("")  # status message for GUI to pull
status_lock = threading.Lock()

result_successful = sym("successful")
result_cancelled = sym("cancelled")
result_errored = sym("errored")

class Progress:
    def __init__(self) -> None:
        """Progress counter for currently running importer task."""
        self.reset()

    def reset(self) -> None:
        """Reset the progress counter."""
        self._macrosteps_done = 0
        self._macrosteps_count = 8  # parse, hiD vectors, hiD cluster, reduce, 2D cluster, entry keywords, cluster keywords, LLM summarize.

        # Microsteps take place within the current macrostep.
        self._microsteps_done = 0
        self._microsteps_count = 1

    def tick(self) -> None:
        """Increment progress counter by one microstep within the current macrostep."""
        self._microsteps_done += 1

    def tock(self) -> None:
        """Increment progress counter to the start of the next macrostep."""
        self._macrosteps_done += 1
        self._microsteps_done = 0
        self._microsteps_count = 1

    def set_micro_count(self, newcount: int) -> None:
        """Set the number of microsteps in the current macrostep."""
        self._microsteps_count = newcount

    def _get(self) -> Optional[float]:
        if has_task():
            # We partition the progress bar so that each macrostep gets the same amount of space.
            # The microsteps within the current macrostep then contribute a fractional part
            # that can be used to provide smoother motion (when progress information inside a given macrostep is available).
            return (self._macrosteps_done + (self._microsteps_done / self._microsteps_count)) / self._macrosteps_count
        return None
    value = property(_get, doc="Progress of currently running import task as a float in [0, 1], or `None` when no task is running.")
progress = Progress()

def discard_message(new_msg):
    pass
make_dynvar(maybe_update_status=discard_message)

bg = None
task_manager = None
def init(executor):
    """Initialize this module. Must be called before `start_task` can be used.

    `executor`: A `ThreadPoolExecutor` or something duck-compatible with it.
                Used for running the background task.
    """
    global bg
    global task_manager
    if bg is not None:  # already initialized?
        return
    bg = executor
    try:
        task_manager = bgtask.TaskManager(name="bibtex_importer",
                                          mode="concurrent",
                                          executor=bg)
        def clear_background_tasks():
            task_manager.clear(wait=False)  # signal background tasks to exit
        atexit.register(clear_background_tasks)
    except Exception:
        bg = None
        task_manager = None
        raise

def start_task(started_callback, done_callback, output_filename, *input_filenames) -> bool:
    """Spawn a background task to convert BibTeX files into a Raven-visualizer dataset file.

    `started_callback`: callable or `None`.

                       If provided, must take a single `unpythonic.env.env` argument.
                       Called when the task actually starts running (it may first
                       have to wait in the queue for a short while, depending on
                       available resources in the executor).

                       The return value of `started_callback` is ignored.

                       Useful e.g. for updating GUI status to show the task has started.

    `done_callback`: callable or `None`.

                     If provided, must take a single `unpythonic.env.env` argument.
                     Called when the task exits.

                     If the task completed successfully:
                       - `env.result_code` will be `result_successful`.
                       - `env.cancelled` will be `False`.

                     If the task was cancelled (by calling `cancel_task`):
                       - `env.result_code` will be `result_cancelled`.
                       - `env.cancelled` will be `True`.

                     If the task completed successfully:
                       - `env.result_code` will be `result_errored`.
                       - `env.cancelled` will be `True` (since the task did not run to completion).
                       - `env.exc` will contain the exception instance.

                     The return value of `done_callback` is ignored.

    `output_filename`: The name of the Raven-visualizer dataset file to write.

    `input_filenames`: The name(s) of the input BibTeX file(s)
                       from which to create the Raven-visualizer dataset.

    Return value is `True` if the task was successfully submitted, and `False` otherwise.
    Task submission may fail if the module has not been initialized, or if an importer
    task is already running.

    The task proceeds asynchronously. To check if it is still running, call `has_task`.
    """
    logger.info("start_task: entered.")
    if task_manager is None:
        logger.warning("start_task: no `task_manager`, canceling. Maybe `importer.init()` has not been called?")
        return False
    if has_task():  # Only allow one importer task to be spawned simultaneously, because it takes a lot of GPU/CPU resources.
        logger.info("start_task: an importer task is already running, canceling.")
        return False

    def update_status(new_msg):
        with status_lock:
            status_box << new_msg

    def importer_task(task_env):
        logger.info(f"importer_task: {task_env.task_name}: entered.")
        if task_env.cancelled:  # if cancelled while waiting in queue -> we're done.
            logger.info(f"importer_task: {task_env.task_name}: cancelled (from task queue)")
            return
        try:
            if started_callback is not None:
                logger.info(f"importer_task: {task_env.task_name}: `started_callback` exists, calling it now.")
                started_callback(task_env)
            with dyn.let(task_env=task_env):
                logger.info(f"importer_task: {task_env.task_name}: entering `import_bibtex` function.")
                import_bibtex(update_status, output_filename, *input_filenames)  # get args from closure, no need to have them in `task_env`
                logger.info(f"importer_task: {task_env.task_name}: done.")
        # Used to be VERY IMPORTANT, to not silently swallow uncaught exceptions from background task.
        # But now `TaskManager._done_callback` does this. However, we need to update the GUI with the
        # error message.
        except Exception as exc:
            logger.warning(f"importer_task: {task_env.task_name}: exited with exception {type(exc)}: {exc}")
            # traceback.print_exc()  # DEBUG; `TaskManager._done_callback` now does this.
            exc_msg = exc.args[0] if (hasattr(exc, "args") and exc.args and exc.args[0]) else f"{type(exc)} (see log for details)"  # show exception message if available, else the type
            update_status(f"Error during import: {exc_msg}")
            task_env.result_code = result_errored
            task_env.exc = exc  # for debugging
            raise
        else:
            if not task_env.cancelled:
                task_env.result_code = result_successful
                finished = "complete"
            else:
                task_env.result_code = result_cancelled
                finished = "cancelled"
            update_status(f"[Import {finished}. To start a new one, select files, and then click the play button.]")
        finally:
            progress.reset()

    update_status("Importer task queued, waiting to start.")
    task_manager.submit(importer_task, env(done_callback=done_callback))  # `task_manager` needs the `done_callback` to be in the `task_env`.
    logger.info("start_task: importer task submitted.")
    return True

def _is_cancelled():
    """Internal function, for the task to check whether it has been cancelled while it is still running."""
    if "task_env" in dyn and dyn.task_env.cancelled:
        return True
    return False

def has_task():
    """Return whether an importer task currently exists.

    This is useful for e.g. enabling/disabling the GUI button to start the importer.
    We only allow one importer task to be spawned simultaneously, because it takes
    a lot of GPU/CPU resources.
    """
    if task_manager is None:
        return False
    return task_manager.has_tasks()

def cancel_task():
    """Cancel the running importer task, if any."""
    if task_manager is None:
        return
    task_manager.clear(wait=True)  # we must wait for the task to exit so that its `done_callback` gets triggered

# --------------------------------------------------------------------------------
# The actual BibTeX importer function (BibTeX to Raven-visualizer dataset)

def import_bibtex(status_update_callback, output_filename, *input_filenames) -> None:
    """Import BibTeX file(s) into a Raven-visualizer dataset.

    This is the synchronous, foreground function that actually performs the task,
    which is mainly useful in a CLI tool that doesn't mind blocking the main thread
    until the import is done.

    To run the import in the background (e.g. in a GUI app), use `start_task` instead.

    `status_update_callback`: callable or `None`.

                              If provided, must take a single `str` argument.
                              Used for sending human-readable status messages
                              while the importer runs.

                              When the importer finishes, it will send a
                              blank string as the final status update.

                              Return value is ignored.

    `output_filename`: The name of the Raven-visualizer dataset file to write.

    `input_filenames`: The name(s) of the input BibTeX file(s)
                       from which to create the Raven-visualizer dataset.

    No return value.

    Filenames are automatically converted to absolute paths via `parse_input_files`,
    which see.
    """
    if status_update_callback is not None:
        maybe_update_status = status_update_callback
    else:
        maybe_update_status = discard_message

    with dyn.let(maybe_update_status=maybe_update_status):  # dynamic assignment is the clean solution to pass the status update function to anything we call while this block is running.
        # --------------------------------------------------------------------------------
        # Prepare input data

        input_data = parse_input_files(*input_filenames)
        progress.tock()

        if _is_cancelled():
            return False

        # --------------------------------------------------------------------------------
        # Prepare filenames

        # Figures are for the whole input dataset, which consists of all input files.
        # For the visualizer, gather all input filenames into a string that can be used in the filenames of the output figures to easily identify where they came from.
        all_input_filenames_list = [common_utils.strip_ext(os.path.basename(fn)) for fn in input_data.resolved_filenames]
        all_input_filenames_str = "_".join(all_input_filenames_list)

        # --------------------------------------------------------------------------------
        # Prepare the high-dimensional semantic space (embedding space, latent space)

        all_vectors = get_highdim_semantic_vectors(input_data)
        progress.tock()

        if _is_cancelled():
            return False

        # --------------------------------------------------------------------------------
        # Dimension reduction hiD -> 2D

        unique_vs, n_clusters = cluster_highdim_semantic_vectors(all_vectors)  # find a stratified sample in the high-dimensional space, for training
        progress.tock()
        if _is_cancelled():
            return False
        lowdim_data = reduce_dimension(unique_vs, n_clusters, all_vectors)  # train the dimension reduction mapping, map the full dataset
        progress.tock()
        if _is_cancelled():
            return False
        vis_data, labels, n_vis_clusters, n_vis_outliers = cluster_lowdim_data(input_data, lowdim_data)  # find visualization clusters in 2D
        progress.tock()
        if _is_cancelled():
            return False

        # --------------------------------------------------------------------------------
        # Find representative keywords via NLP analysis over the whole dataset

        if visualizer_config.extract_keywords:
            all_keywords = extract_keywords(input_data)
        else:
            logger.info("Keyword extraction disabled, skipping NLP analysis.")
            all_keywords = {}
        progress.tock()
        if _is_cancelled():
            return False

        # --------------------------------------------------------------------------------
        # Find a set of keywords for each cluster

        if visualizer_config.extract_keywords:
            vis_keywords_by_cluster = collect_cluster_keywords(vis_data, n_vis_clusters, all_keywords)
        else:
            vis_keywords_by_cluster = []
        progress.tock()
        if _is_cancelled():
            return False

        # --------------------------------------------------------------------------------
        # Write LLM summary for each item

        if visualizer_config.summarize:
            summarize(input_data)  # mutates its input
        else:
            logger.info("LLM summarization disabled, skipping.")
        progress.tock()

        # Do not need to allow cancellation after this point, because all that is left is to save the results.

        # --------------------------------------------------------------------------------
        # Save the resulting Raven-visualizer dataset file

        logger.info(f"Saving visualization datafile {output_filename}...")

        # Be sure to save the values of any settings that affect data availability and interpretation! (E.g. `extract_keywords` -> whether annotations and word cloud can be plotted from this data.)

        output_file_version = 1  # must be a version supported by the visualizer
        with timer() as tim:
            output_data = {"version": output_file_version,
                           "all_input_filenames_raw": input_data.resolved_filenames,  # actual paths
                           "all_input_filenames_list": all_input_filenames_list,  # just the filenames (no path)
                           "all_input_filenames_str": all_input_filenames_str,  # concatenated, "file1_file2_..._fileN", for naming output figures for this combination of input files
                           "embedding_model": visualizer_config.embedding_model,
                           "vis_method": visualizer_config.vis_method,  # dimension reduction method
                           "n_vis_clusters": n_vis_clusters,  # number of clusters detected
                           "n_vis_outliers": n_vis_outliers,  # number of outlier points, not belonging to any cluster
                           "labels": labels,
                           "vis_data": vis_data,  # list, concatenated entries from all input files
                           "lowdim_data": lowdim_data,  # rank-2 `np.array` of shape `[N, 2]`, 2D points from the semantic mapping, after dimension reduction
                           "keywords_available": visualizer_config.extract_keywords,
                           "all_keywords": all_keywords,
                           "vis_keywords_by_cluster": vis_keywords_by_cluster}
            with open(output_filename, "wb") as output_file:
                pickle.dump(output_data, output_file)
        logger.info(f"    Done in {tim.dt:0.6g}s.")

        return True

# --------------------------------------------------------------------------------
# Main program (when run as a standalone command-line tool)

def main() -> None:
    logger.info("Settings (for LOCAL models):")
    logger.info(f"    Embedding model: {visualizer_config.embedding_model}")
    logger.info(f"        Dimension reduction method: {visualizer_config.vis_method}")
    logger.info(f"    Extract keywords: {visualizer_config.extract_keywords}")
    logger.info(f"        NLP model (spaCy): {visualizer_config.spacy_model}")
    logger.info(f"    Summarize via LLM: {visualizer_config.summarize}")

    parser = argparse.ArgumentParser(description="""Convert BibTeX file(s) into a Raven-visualizer dataset file.""",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version=('%(prog)s ' + __version__))
    parser.add_argument(dest="output_filename", type=str, metavar="out", help="Output, Raven-visualizer dataset file")
    parser.add_argument(dest="input_filenames", nargs="+", default=None, type=str, metavar="bib", help="Input, BibTeX file(s) to parse")
    opts = parser.parse_args()

    if opts.output_filename.endswith(".bib"):
        print(f"Output filename '{opts.output_filename}' looks like an input filename. Cancelling. Please check usage summary by running this program with the '-h' (or '--help') option.")
        sys.exit(1)

    try:
        with timer() as tim:
            import_bibtex(None, opts.output_filename, *opts.input_filenames)
    except Exception:
        logger.warning(f"Error after {tim.dt:0.6g}s total:")
        traceback.print_exc()
    else:
        logger.info(f"All done in {tim.dt:0.6g}s total.")

if __name__ == "__main__":
    main()

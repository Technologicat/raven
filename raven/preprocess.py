#!/usr/bin/env python
"""Extract BibTeX data for visualization. This can put an entire field of science into one picture.

This script performs analysis and writes the visualization data file. See `app.py` to plot the results.

We use `bibtexparser` v2.x. To install, e.g.::
    pip install bibtexparser --pre --upgrade --user
For more, see::
    https://github.com/sciunto-org/python-bibtexparser
"""

__all__ = ["init",
           "start_task", "has_task", "cancel_task",
           "result_successful", "result_cancelled", "result_errored"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import argparse
import collections
import functools
import itertools
import math
import os
import pathlib
import pickle
import re
import sys
import threading
import traceback
from typing import Dict, List, Union

import bibtexparser

# import nltk
import spacy

from unpythonic.env import env
from unpythonic import box, dyn, islice, make_dynvar, sym, timer, uniqify

# # To connect to the live REPL:  python -m unpythonic.net.client localhost
# from unpythonic.net import server
# server.start(locals={"main": sys.modules["__main__"]})

import numpy as np
# import pandas as pd

import torch
import transformers

from sklearn.cluster import HDBSCAN

from . import bgtask
from . import config
from . import utils

# TODO: spacy doesn't support NumPy 2.0 yet (August 2024), so staying with NumPy 1.26 for now.
# stopwords = nltk.corpus.stopwords.words("english")
from spacy.lang.en import English
nlp_en = English()
stopwords = nlp_en.Defaults.stop_words

def isstopword(word: str) -> bool:
    # word = lemmatizer.lemmatize(word.lower())  # NLTK
    word = word.lower()
    return word in stopwords or word in config.custom_stopwords

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

# --------------------------------------------------------------------------------
# TL;DR: AI-based summarization

if config.summarize:
    summarization_pipeline = transformers.pipeline("summarization", model=config.summarization_model, device=config.device_string, torch_dtype=config.torch_dtype)
else:
    summarization_pipeline = None

def tldr(text: str) -> str:
    """Return AI-based summary of `text`.

    The input must fit into the model's context window.
    """
    # Produce raw summary
    summary = utils.unicodize_basic_markup(summarization_pipeline(f"{config.summarization_prefix}{text}",
                                                                  min_length=50,  # tokens
                                                                  max_length=100)[0]["summary_text"])

    # Postprocess the summary

    # Decide whether we need to remove an incomplete sentence at the end
    end = -1 if summary[-1] != "." else None

    # Normalize whitespace at sentence borders
    parts = summary.split(".")
    parts = [x.strip() for x in parts]
    parts = [x for x in parts if len(x)]
    summary = ". ".join(parts[:end]) + "."
    summary = re.sub(r"(\d)\. (\d)", r"\1.\2", summary)  # Fix decimal numbers broken by the punctuation fixer

    # Normalize whitespace around commas
    parts = summary.split(",")
    parts = [x.strip() for x in parts]
    parts = [x for x in parts if len(x)]
    summary = ", ".join(parts)
    summary = re.sub(r"(\d)\, (\d)", r"\1,\2", summary)  # Fix numbers with American thousands separators, broken by the punctuation fixer

    # Capitalize start of first sentence
    summary = summary[0].upper() + summary[1:]

    return summary

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

                authors_str = utils.format_bibtex_authors(fields["author"].value)
                year = fields["year"].value
                title = utils.unicodize_basic_markup(fields["title"].value)

                # abstract is optional
                if "abstract" in fields and fields["abstract"].value:
                    abstract = utils.unicodize_basic_markup(fields["abstract"].value)
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
    embeddings_cache_filenames = {fn: utils.make_cache_filename(fn, "embeddings_cache", "npz") for fn in input_data.resolved_filenames}

    all_vectors_by_filename = {}
    sentence_embedder = None
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
            if utils.validate_cache_mtime(embeddings_cache_filename, filename):  # is cache valid, judging by mtime
                with timer() as tim:
                    embeddings_cached_data = np.load(embeddings_cache_filename)
                logger.info(f"            Done in {tim.dt:0.6g}s.")
                embedding_model_for_this_cache = embeddings_cached_data["embedding_model"]
                if embedding_model_for_this_cache == config.embedding_model:
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
            if sentence_embedder is None:  # delayed init - load only if needed, on first use
                logger.info("        Loading SentenceTransformer...")
                with timer() as tim:
                    from sentence_transformers import SentenceTransformer
                    try:
                        sentence_embedder = SentenceTransformer(config.embedding_model, device=config.device_string)
                        final_device = config.device_string
                    except RuntimeError as exc:
                        logger.warning(f"get_highdim_semantic_vectors: exception while loading SentenceTransformer (will try again in CPU mode): {type(exc)}: {exc}")
                        try:
                            sentence_embedder = SentenceTransformer(config.embedding_model, device="cpu")
                            final_device = "cpu"
                        except RuntimeError as exc:
                            logger.warning(f"get_highdim_semantic_vectors: failed to load SentenceTransformer: {type(exc)}: {exc}")
                            raise
                logger.info(f"            Done in {tim.dt:0.6g}s.")

            logger.info(f"        Encoding (on device '{final_device}')...")
            with timer() as tim:
                all_inputs = [entry.title for entry in entries]  # with mpnet, this works best (and we don't always necessarily have an abstract)
                # all_inputs = [entry.abstract for entry in entries]  # testing with snowflake
                # all_inputs = [" ".join(entry.keywords) for entry in entries]
                all_vectors = sentence_embedder.encode(all_inputs,
                                                       show_progress_bar=True,
                                                       convert_to_numpy=True,
                                                       normalize_embeddings=True)
                # Round-trip to force truncation, if needed.
                # This matters to make the "cluster centers" coincide with the original datapoints when clustering is disabled.
                all_vectors = torch.tensor(all_vectors, device=config.device_string, dtype=config.torch_dtype)
                all_vectors = all_vectors.detach().cpu().numpy()
            logger.info(f"            Done in {tim.dt:0.6g}s [avg {len(all_inputs) / tim.dt:0.6g} entries/s].")

            logger.info(f"        Caching embeddings for this dataset to '{embeddings_cache_filename}'...")
            with timer() as tim:
                np.savez_compressed(embeddings_cache_filename, all_vectors=all_vectors, embedding_model=config.embedding_model)
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
    update_status_and_log(f"Loading dimension reduction library for '{config.vis_method}'...", log_indent=1)
    progress.set_micro_count(3)  # load library, train mapping, apply mapping
    with timer() as tim:
        if config.vis_method == "tsne":
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
        elif config.vis_method == "umap":
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
            # NOTE: UMAP needs to load TensorFlow, whereas the rest of the preprocessor uses PyTorch.
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
            raise ValueError(f"Unknown `config.vis_method` '{config.vis_method}'; valid: 'tsne', 'umap'; please check your `raven/config.py` and restart the app.")
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


def rank_keyword(word: str, counts: dict) -> int:
    """Sort key: rank `word` (str) by number of occurrences in `counts` (dict, str -> int), descending.

    Words that do not appear in `counts` are sorted to the end.
    """
    if word not in counts:
        return float("+inf")
    return -counts[word]

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
    nlp_cache_filenames = {fn: utils.make_cache_filename(fn, "nlp_cache", "pickle") for fn in input_data.resolved_filenames}
    nlp_cache_version = 1

    all_keywords_by_filename = {}
    nlp_pipeline = None
    progress.set_micro_count(len(input_data.parsed_data_by_filename))
    for j, (filename, entries) in enumerate(input_data.parsed_data_by_filename.items(), start=1):
        if _is_cancelled():
            return

        update_status_and_log(f"[{j} out of {len(input_data.parsed_data_by_filename)}] NLP analysis for {filename}...", log_indent=1)

        nlp_cache_filename = nlp_cache_filenames[filename]
        cache_state = "unknown"
        if os.path.exists(nlp_cache_filename):
            logger.info(f"        Checking cached NLP data '{nlp_cache_filename}'...")
            if utils.validate_cache_mtime(nlp_cache_filename, filename):  # is cache valid, judging by mtime
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
            if nlp_pipeline is None:
                logger.info("        Loading spaCy...")
                with timer() as tim2:
                    # NOTE: make sure to `source env.sh` first, or this won't find the CUDA runtime.
                    try:
                        spacy.require_gpu()
                        update_status_and_log("NLP model will run on GPU.", log_indent=2)
                    except Exception as exc:
                        logger.warning(f"extract_keywords: exception while enabling GPU: {type(exc)}: {exc}")
                        spacy.require_cpu()
                        update_status_and_log("NLP model will run on CPU.", log_indent=2)
                    try:
                        nlp_pipeline = spacy.load(config.spacy_model)
                    except OSError:
                        # https://stackoverflow.com/questions/62728854/how-to-place-spacy-en-core-web-md-model-in-python-package
                        update_status_and_log("Downloading language model for spaCy (don't worry, this will only happen once)...", log_indent=2)
                        from spacy.cli import download
                        download(config.spacy_model)
                        nlp_pipeline = spacy.load(config.spacy_model)
                    update_status_and_log(f"[{j} out of {len(input_data.parsed_data_by_filename)}] NLP analysis for {filename}...", log_indent=1)  # restore old message  # TODO: DRY log messages
                    # analysis = nlp_pipeline.analyze_pipes(pretty=True)  # print pipeline overview
                    # nlp_pipeline.disable_pipe("parser")
                    # nlp_pipeline.enable_pipe("senter")
                logger.info(f"            Done in {tim2.dt:0.6g}s.")

            # Apply standard tricks from information retrieval:
            #   - Drop useless stopwords ("the", "of", ...), which typically dominate the word frequency distribution
            #   - Cut the (long!) tail of the distribution
            # => Obtain words that appear at an intermediate frequency (not too common, not too rare).
            #    These usually describe the text usefully.

            # # Old implementation using NLTK
            # lemmatizer = nltk.stem.WordNetLemmatizer()
            # def extract_word_counts(text: Union[str, List[str]]) -> collections.Counter:
            #     if isinstance(text, str):
            #         def filter_tokens(tokens):
            #             out = []
            #             for x in tokens:
            #                 if x.pos_ in ("ADP", "AUX", "CCONJ", "DET", "NUM", "PRON", "PUNCT", "SCONJ"):
            #                     continue
            #                 if not x.lemma_.isalnum():
            #                     continue
            #                 x = x.lemma_.lower()
            #                 # x = lemmatizer.lemmatize(x.lower())  # NLTK
            #                 if isstopword(x) or len(x) < 3:
            #                     continue
            #                 out.append(x)
            #             return out
            #         return collections.Counter(filter_tokens(nlp_pipeline(text)))
            #         # return collections.Counter(filter_tokens(nltk.word_tokenize(text)))
            #     else:  # List[str]
            #         word_counts = collections.Counter()
            #         for item in text:
            #             word_counts.update(extract_word_counts(item))
            #         return word_counts

            # New implementation using spaCy
            def extract_word_counts(things: Union[List[spacy.tokens.token.Token],
                                                  List[List[spacy.tokens.token.Token]]]) -> collections.Counter:
                if not things:
                    return collections.Counter()
                if isinstance(things[0], spacy.tokens.token.Token):  # token list
                    def filter_tokens(tokens):
                        out = []
                        for x in tokens:
                            # https://spacy.io/usage/linguistic-features
                            # https://universaldependencies.org/u/pos/
                            #     ADJ: adjective
                            #     ADP: adposition
                            #     ADV: adverb
                            #     AUX: auxiliary
                            #     CCONJ: coordinating conjunction
                            #     DET: determiner
                            #     INTJ: interjection
                            #     NOUN: noun
                            #     NUM: numeral
                            #     PART: particle
                            #     PRON: pronoun
                            #     PROPN: proper noun
                            #     PUNCT: punctuation
                            #     SCONJ: subordinating conjunction
                            #     SYM: symbol
                            #     VERB: verb
                            #     X: other
                            if x.pos_ in ("ADP", "AUX", "CCONJ", "DET", "NUM", "PRON", "PUNCT", "SCONJ"):  # filter out parts of speech that are useless as keywords
                                continue
                            if not x.lemma_.isalnum():  # filter out punctuation
                                continue
                            x = x.lemma_.lower()
                            if isstopword(x) or len(x) < 3:
                                continue
                            out.append(x)
                        return out
                    return collections.Counter(filter_tokens(things))
                else:  # list of token lists
                    word_counts = collections.Counter()
                    for sublist in things:
                        word_counts.update(extract_word_counts(sublist))
                    return word_counts

            # def trim_word_counts(word_counts: Dict[str, int], p: float = 0.05) -> Dict[str, int]:
            #     """`p`: tail cutoff, as proportion of the largest count.
            #
            #             The resulting cutoff count is automatically rounded to the nearest integer.
            #             Words that appear only once are always trimmed (to do only this, use `p=0`).
            #     """
            #     max_count = max(word_counts.values())
            #     tail_cutoff_count = max(2, round(p * max_count))
            #     representative_words = {x: word_counts[x] for x in word_counts if word_counts[x] >= tail_cutoff_count}
            #     return representative_words

            # from unpythonic import islice
            # def trim_word_counts(word_counts: Dict[str, int]) -> Dict[str, int]:
            #     k = 10  # drop this many most common words
            #     keys = islice(word_counts.keys())[k:]
            #     m = 3  # drop any word with fewer occurrences than this
            #     representative_words = {x: word_counts[x] for x in keys if word_counts[x] >= m}
            #     return representative_words

            def trim_word_counts(word_counts: Dict[str, int]) -> Dict[str, int]:
                # U = int(0.2 * input_data.n_entries_total)  # drop any word with at least this many occurrences
                U = float("+inf")  # upper limit disabled
                L = 2  # drop any word with fewer occurrences than this
                representative_words = {x: word_counts[x] for x in word_counts if L <= word_counts[x] <= U}
                return representative_words

            def format_entry_for_keyword_extraction(entry: env) -> str:
                # return entry.title
                if entry.abstract:
                    return entry.title + ".\n\n" + entry.abstract
                else:
                    return entry.title

            logger.info("        Computing keyword set...")
            logger.info(f"            Running NLP pipeline for data from {filename}...")
            with timer() as tim:
                all_texts_for_nlp_for_this_file = [format_entry_for_keyword_extraction(entry) for entry in entries]

                # The pipe-batched version processed about 25 entries per second on an RTX 3070 Ti mobile GPU.
                # TODO: Can we minibatch the NLP pipelining to save VRAM when using the Transformers model?
                # all_docs_for_this_file = [nlp_pipeline(text) for text in all_texts_for_nlp_for_this_file]
                all_docs_for_this_file = list(nlp_pipeline.pipe(all_texts_for_nlp_for_this_file))
                # all_docs_for_this_file = []
                # for j, text in enumerate(all_texts_for_nlp_for_this_file):
                #     if j % 100 == 0:
                #         logger.info(f"    {j + 1} out of {len(all_texts_for_nlp_for_this_file)}...")
                #     all_docs_for_this_file.append(nlp_pipeline(text))
            logger.info(f"                Done in {tim.dt:0.6g}s [avg {input_data.n_entries_total / tim.dt:0.6g} entries/s].")

            # TODO: Should we trim the keywords across the whole dataset? We currently trim each input file separately.
            logger.info("            Extracting keywords from NLP pipeline results...")
            with timer() as tim:
                all_keywords_for_this_file = trim_word_counts(extract_word_counts(all_docs_for_this_file))
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

            # Tag the entries.
            logger.info(f"        Trimming word counts and tagging named entities for data from {filename}...")
            fa_times = []
            ner_times = []
            with timer() as tim:
                entry_keywords_for_this_file = []
                entry_entities_for_this_file = []
                for entry, doc in zip(entries, all_docs_for_this_file):
                    if _is_cancelled():
                        return

                    with timer() as tim2:
                        kws = trim_word_counts(extract_word_counts(doc))
                    fa_times.append(tim2.dt)

                    # Named entity recognition (NER).
                    # TODO: Update named entities to the cluster keywords, too. The thing is, we don't have frequency information for named entities, so we can't decide which are the most relevant ones.
                    with timer() as tim2:
                        # pos_tags = nltk.pos_tag(nltk.word_tokenize(text))
                        # tree = nltk.ne_chunk(pos_tags, binary=True)
                        # ents = set()
                        # for subtree in tree:
                        #     if isinstance(subtree, nltk.Tree):
                        #         ent = " ".join([word for word, tag in subtree.leaves()])
                        #         # label = subtree.label()
                        #         if not isstopword(ent):  # omit single-word entities that are in stopwords
                        #             ents.add(ent)
                        #         # logger.info(f"Entity: {ent}, Label: {label}")

                        # spaCy's pipeline (which we anyway use for keyword detection above) gives much better results here.
                        ents = set()
                        for ent in doc.ents:
                            if ent.label_ in ("CARDINAL", "DATE", "MONEY", "QUANTITY", "PERCENT", "TIME"):
                                continue
                            if isstopword(ent.text):
                                continue
                            # print(f"Entity: {ent.text}, Label: {ent.label_}")  # DEBUG
                            ents.add(ent.text)
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
                kws = list(sorted(kws.keys(), key=functools.partial(rank_keyword, counts=all_keywords)))
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
    `all_keywords`: output of `extract_keywords`, which see.

    `max_vis_kw`: how many keywords to keep for each cluster.

    `fraction`: float, cleaning parameter, important.

                The aim of this parameter is to help omit useless keywords from visualization, trying to focus on
                what distinguishes each cluster from the others. There's a fine balance here, drawing on the old
                NLP observation that words with intermediate frequencies best describe a dataset:

                  - Filtering out keywords present in *all* other clusters drops only a few of the most common words,
                    leaving uselessly large overlaps between the keyword sets. This happens even after we account for
                    stopwords, because the typical use case of Raven is that the dataset discusses a single broad
                    umbrella topic (one field of science).

                  - Filtering out keywords present in *any* other cluster leaves only noise.

                  - So we set a threshold `fraction` such that if at least that fraction of all clusters have a
                    particular keyword, then that keyword is considered uselessly common for the purposes of
                    distinguishing clusters, and dropped.

                 The default is to ignore keywords that appear in at least 10% of all clusters.
                 But small datasets produce very few clusters, so the final formula that allows
                 also such edge cases is::

                     threshold_n = max(min(5, n_vis_clusters), math.ceil(fraction * n_vis_clusters))

                 Any keyword that appears in `threshold_n` or more clusters is ignored.

    Returns `vis_keywords_by_cluster`, a list, where the `k`th item is a list of keywords (`str`) for cluster ID `k`.
    For each cluster, the keywords are sorted by number of occurrences (descending) across the whole dataset.
    """
    update_status_and_log("Extracting keywords for each cluster...", log_indent=1)
    progress.set_micro_count(n_vis_clusters)
    with timer() as tim:
        logger.info("        Collecting keywords from data points in each cluster...")
        keywords_by_cluster = [collections.Counter() for _ in range(n_vis_clusters)]
        for entry in vis_data:
            if entry.cluster_id >= 0:  # not an outlier
                # Update cluster keyword counters from this entry.
                # NOTE: We operate on the already-filtered data that excludes stopwords (see `extract_keywords` for details).
                # TODO: We could also use `entry.cluster_probability` for something here.
                # TODO: Including entities would be nice, but they don't currently have frequency information.
                keywords_by_cluster[entry.cluster_id].update(entry.keywords)  # inject keywords of this entry to the keywords of the cluster this entry belongs to

        threshold_n = max(min(5, n_vis_clusters), math.ceil(fraction * n_vis_clusters))  # how many clusters must have a particular keyword for that keyword to be considered uselessly common
        cluster_counts_by_keyword = collections.Counter()
        for cluster_id, kws in enumerate(keywords_by_cluster):
            for kw in kws.keys():
                cluster_counts_by_keyword[kw] += 1  # found one more cluster that has this keyword
        keywords_common_to_most_clusters = {kw for kw, count in cluster_counts_by_keyword.items() if count >= threshold_n}
        logger.info(f"        Found {len(keywords_common_to_most_clusters)} common (useless) keywords, shared between {threshold_n} clusters out of {n_vis_clusters}. Threshold fraction = {fraction:0.2g}.")

        # # Detect keywords common to *all* clusters - not helpful.
        # keywords_common_to_all_clusters = set(keywords_by_cluster[0].keys())
        # for kws in keywords_by_cluster[1:]:
        #     kws = set(kws.keys())
        #     keywords_common_to_all_clusters.intersection_update(kws)

        # Keep the highest-frequency keywords detected in each cluster. Hopefully this will compactly describe what the cluster is about.
        logger.info("        Ranking and filtering results...")
        vis_keywords_by_cluster = []
        for cluster_id, kws in enumerate(keywords_by_cluster):
            kws = set(kws.keys())

            # # Drop keywords present in *any* other cluster - not helpful.
            # other_cluster_keywords = keywords_by_cluster[:cluster_id] + keywords_by_cluster[cluster_id + 1:]
            # for other_kws in other_cluster_keywords:
            #     kws = kws.difference(other_kws)

            # Drop keywords common with too many other clusters.
            # NOTE: We build the set first (further above), and filter against the complete set, to treat all clusters symmetrically.
            #       We don't want filler words to end up in the first cluster just because it was processed first.
            kws = kws.difference(keywords_common_to_most_clusters)

            kws = list(sorted(kws, key=functools.partial(rank_keyword, counts=all_keywords)))
            kws = kws[:max_vis_kw]
            vis_keywords_by_cluster.append(kws)
            progress.tick()
    logger.info(f"        Done in {tim.dt:0.6g}s.")

    return vis_keywords_by_cluster


def summarize(input_data):
    """Summarize each item of the dataset, using an AI model for abstractive summarization.

    NOTE: For a large dataset, this will take a long, long time!

    `input_data`: output of `parse_input_files`, which see.

    No return value.

    Mutates `input_data`, adding an `item.summary` field to each item. The field value is:
        - `str`, the summary, if there was an abstract (so a summary could be created).
        - `None`, if there was no abstract (so a summary could not be created).
    """
    logger.info("Summarizing abstracts using AI...")
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
                    summary = tldr(entry.abstract)
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
    def __init__(self):
        """Progress counter for currently running import task."""
        self.reset()

    def reset(self):
        self._macrosteps_done = 0
        self._macrosteps_count = 8  # parse, hiD vectors, hiD cluster, reduce, 2D cluster, entry keywords, cluster keywords, summarize.

        # Microsteps take place within the current macrostep.
        self._microsteps_done = 0
        self._microsteps_count = 1

    def tick(self):
        self._microsteps_done += 1

    def tock(self):
        self._macrosteps_done += 1
        self._microsteps_done = 0
        self._microsteps_count = 1

    def set_micro_count(self, newcount):
        self._microsteps_count = newcount

    def _get(self):
        if has_task():
            # We partition the progress bar so that each macrostep gets the same amount of space; the microsteps within the current macrostep then contribute a fractional part.
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
    except Exception:
        bg = None
        task_manager = None
        raise

def start_task(started_callback, done_callback, output_filename, *input_filenames) -> bool:
    """Spawn a background task to convert BibTeX files into a visualization dataset file.

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

    `output_filename`: The name of the visualization dataset file to write.

    `input_filenames`: The name(s) of the input BibTeX file(s)
                       from which to create the visualization dataset.

    Return value is `True` if the task was successfully submitted, and `False` otherwise.
    Task submission may fail if the module has not been initialized, or if a preprocessor
    task is already running.

    The task proceeds asynchronously. To check if it is still running, call `has_task`.
    """
    logger.info("start_task: entered.")
    if task_manager is None:
        logger.warning("start_task: no `task_manager`, canceling. Maybe `preprocess.init()` has not been called?")
        return False
    if has_task():  # Only allow one preprocessor task to be spawned simultaneously, because it takes a lot of GPU/CPU resources.
        logger.info("start_task: a preprocessor task is already running, canceling.")
        return False

    def update_status(new_msg):
        with status_lock:
            status_box << new_msg

    def preprocessor_task(task_env):
        logger.info(f"preprocessor_task: {task_env.task_name}: entered.")
        if task_env.cancelled:  # if cancelled while waiting in queue -> we're done.
            logger.info(f"preprocessor_task: {task_env.task_name}: cancelled (from task queue)")
            return
        try:
            if started_callback is not None:
                logger.info(f"preprocessor_task: {task_env.task_name}: `started_callback` exists, calling it now.")
                started_callback(task_env)
            with dyn.let(task_env=task_env):
                logger.info(f"preprocessor_task: {task_env.task_name}: entering `preprocess` function.")
                preprocess(update_status, output_filename, *input_filenames)  # get args from closure, no need to have them in `task_env`
                logger.info(f"preprocessor_task: {task_env.task_name}: done.")
        except Exception as exc:  # VERY IMPORTANT, to not silently swallow uncaught exceptions from background task
            logger.warning(f"preprocessor_task: {task_env.task_name}: exited with exception {type(exc)}: {exc}")
            traceback.print_exc()  # DEBUG
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

    update_status("Preprocessor task queued, waiting to start.")
    task_manager.submit(preprocessor_task, env(done_callback=done_callback))  # `task_manager` needs the `done_callback` to be in the `task_env`.
    logger.info("start_task: preprocessor task submitted.")
    return True

def _is_cancelled():
    """Internal function, for the task to check whether it has been cancelled while it is still running."""
    if "task_env" in dyn and dyn.task_env.cancelled:
        return True
    return False

def has_task():
    """Return whether a preprocessor task currently exists.

    This is useful for e.g. enabling/disabling the GUI button to start the preprocessor.
    We only allow one preprocessor task to be spawned simultaneously, because it takes
    a lot of GPU/CPU resources.
    """
    if task_manager is None:
        return False
    return task_manager.has_tasks()

def cancel_task():
    """Cancel the running preprocessor task, if any."""
    if task_manager is None:
        return
    task_manager.clear(wait=True)  # we must wait for the task to exit so that its `done_callback` gets triggered

# --------------------------------------------------------------------------------
# The actual preprocessing function (data importer that creates the visualization dataset)

def preprocess(status_update_callback, output_filename, *input_filenames) -> None:
    """Preprocess input files into a visualization dataset.

    This is the synchronous, foreground function that actually performs the task.
    To process in the background, use `start_task` instead.

    `status_update_callback`: callable or `None`.

                              If provided, must take a single `str` argument.
                              Used for sending human-readable status messages
                              while the preprocessor runs.

                              When the preprocessor finishes, it will send a
                              blank string as the final status update.

                              Return value is ignored.

    `output_filename`: The name of the visualization dataset file to write.

    `input_filenames`: The name(s) of the input BibTeX file(s)
                       from which to create the visualization dataset.

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
        all_input_filenames_list = [utils.strip_ext(os.path.basename(fn)) for fn in input_data.resolved_filenames]
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

        if config.extract_keywords:
            all_keywords = extract_keywords(input_data)
        else:
            logger.info("Keyword extraction disabled, skipping NLP analysis.")
            all_keywords = {}
        progress.tock()
        if _is_cancelled():
            return False

        # --------------------------------------------------------------------------------
        # Find a set of keywords for each cluster

        if config.extract_keywords:
            vis_keywords_by_cluster = collect_cluster_keywords(vis_data, n_vis_clusters, all_keywords)
        else:
            vis_keywords_by_cluster = []
        progress.tock()
        if _is_cancelled():
            return False

        # --------------------------------------------------------------------------------
        # Write AI summary for each item (EXPERIMENTAL / EXPENSIVE)

        if config.summarize:
            summarize(input_data)  # mutates its input
        else:
            logger.info("AI summarization disabled, skipping.")
        progress.tock()

        # Do not need to allow cancellation after this point, because all that is left is to save the results.

        # --------------------------------------------------------------------------------
        # Save the resulting visualization dataset file

        logger.info(f"Saving visualization datafile {output_filename}...")

        # Be sure to save the values of any settings that affect data availability and interpretation! (E.g. `extract_keywords` -> whether annotations and word cloud can be plotted from this data.)

        output_file_version = 1  # must be a version supported by the visualizer
        with timer() as tim:
            output_data = {"version": output_file_version,
                           "all_input_filenames_raw": input_data.resolved_filenames,  # actual paths
                           "all_input_filenames_list": all_input_filenames_list,  # just the filenames (no path)
                           "all_input_filenames_str": all_input_filenames_str,  # concatenated, "file1_file2_..._fileN", for naming output figures for this combination of input files
                           "embedding_model": config.embedding_model,
                           "vis_method": config.vis_method,  # dimension reduction method
                           "n_vis_clusters": n_vis_clusters,  # number of clusters detected
                           "n_vis_outliers": n_vis_outliers,  # number of outlier points, not belonging to any cluster
                           "labels": labels,
                           "vis_data": vis_data,  # list, concatenated entries from all input files
                           "lowdim_data": lowdim_data,  # rank-2 `np.array` of shape `[N, 2]`, 2D points from the semantic mapping, after dimension reduction
                           "keywords_available": config.extract_keywords,
                           "all_keywords": all_keywords,
                           "vis_keywords_by_cluster": vis_keywords_by_cluster}
            with open(output_filename, "wb") as output_file:
                pickle.dump(output_data, output_file)
        logger.info(f"    Done in {tim.dt:0.6g}s.")

        return True

# --------------------------------------------------------------------------------
# Main program (when run as a standalone command-line tool)

def main() -> None:
    logger.info("Settings:")
    logger.info(f"    Compute device '{config.device_string}' ({config.device_name}), data type {config.torch_dtype}")
    if torch.cuda.is_available():
        logger.info(f"        {torch.cuda.get_device_properties(config.device_string)}")
        logger.info(f"        Compute capability {'.'.join(str(x) for x in torch.cuda.get_device_capability(config.device_string))}")
        logger.info(f"        Detected CUDA version {torch.version.cuda}")
    logger.info(f"    Embedding model: {config.embedding_model}")
    logger.info(f"        Dimension reduction method: {config.vis_method}")
    logger.info(f"    Extract keywords: {config.extract_keywords}")
    logger.info(f"        NLP model (spaCy): {config.spacy_model}")
    logger.info(f"    Summarize via AI: {config.summarize}")
    logger.info(f"        AI summarization model: {config.summarization_model}")

    parser = argparse.ArgumentParser(description="""Convert BibTeX file(s) into a Raven visualization dataset file.""",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(dest="output_filename", type=str, metavar="out", help="Output, visualization dataset file")
    parser.add_argument(dest="input_filenames", nargs="+", default=None, type=str, metavar="bib", help="Input, BibTeX file(s) to parse")
    opts = parser.parse_args()

    if opts.output_filename.endswith(".bib"):
        print(f"Output filename '{opts.output_filename}' looks like an input filename. Cancelling. Please check usage summary by running this prorgram with the '-h' option.")
        sys.exit(1)

    try:
        with timer() as tim:
            preprocess(None, opts.output_filename, *opts.input_filenames)
    except Exception:
        logger.warning(f"Error after {tim.dt:0.6g}s total:")
        traceback.print_exc()
    else:
        logger.info(f"All done in {tim.dt:0.6g}s total.")

if __name__ == "__main__":
    main()

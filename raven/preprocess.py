#!/usr/bin/env python
"""Extract BibTeX data for visualization. This can put an entire field of science into one picture.

This script performs analysis and writes the visualization data file. See `app.py` to plot the results.

We use `bibtexparser` v2.x. To install, e.g.::
    pip install bibtexparser --pre --upgrade --user
For more, see::
    https://github.com/sciunto-org/python-bibtexparser
"""

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# # Output log messages only from this module. Handy for debugging.
# # https://stackoverflow.com/questions/17275334/what-is-a-correct-way-to-filter-different-loggers-using-python-logging
# for handler in logging.root.handlers:
#     handler.addFilter(logging.Filter(__name__))

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
from typing import Dict, List, Union

import bibtexparser

# import nltk
import spacy

from unpythonic.env import env
from unpythonic import islice
from unpythonic import timer
from unpythonic import uniqify

# # To connect to the live REPL:  python -m unpythonic.net.client localhost
# import sys
# from unpythonic.net import server
# server.start(locals={"main": sys.modules["__main__"]})

import numpy as np
# import pandas as pd

import torch
import transformers

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
# Main program

def main(opts) -> None:
    # --------------------------------------------------------------------------------
    # Prepare input data

    resolved_filenames = list(uniqify(str(pathlib.Path(fn).expanduser().resolve()) for fn in opts.filenames))

    logger.info(f"Reading input file{'s' if len(resolved_filenames) != 1 else ''}...")
    bibtex_entries_by_filename = {}
    with timer() as tim:
        for filename in resolved_filenames:
            logger.info(f"    Reading {filename}...")
            filename = str(pathlib.Path(filename).expanduser().resolve())
            library = bibtexparser.parse_file(filename,
                                              append_middleware=[bibtexparser.middlewares.NormalizeFieldKeys(),
                                                                 bibtexparser.middlewares.SeparateCoAuthors(),
                                                                 bibtexparser.middlewares.SplitNameParts()])
            bibtex_entries_by_filename[filename] = library.entries
    logger.info(f"    Done in {tim.dt:0.6g}s.")

    logger.info("Extracting fields from each entry...")
    parsed_data_by_filename = {}
    with timer() as tim:
        for filename, entries in bibtex_entries_by_filename.items():
            logger.info(f"    Extracting from {filename}...")
            parsed_data_by_filename[filename] = []
            for entry in entries:
                fields = entry.fields_dict

                try:
                    authors_list = [utils.format_bibtex_author_name(author) for author in fields["author"].value]
                except ValueError as exc:
                    logger.warning(f"Skipping entry '{entry.key}', reason: {str(exc)}")
                    continue

                if len(authors_list) >= 3:
                    authors_str = f"{authors_list[0]} et al."
                elif len(authors_list) == 2:
                    authors_str = f"{authors_list[0]} and {authors_list[1]}"
                elif len(authors_list) == 1:
                    authors_str = authors_list[0]
                else:  # empty author list
                    logger.warning(f"Skipping entry '{entry.key}', reason: empty authors list")
                    continue

                year = fields["year"].value
                title = utils.unicodize_basic_markup(fields["title"].value)
                abstract = utils.unicodize_basic_markup(fields["abstract"].value)

                # TODO: "keywords" may be populated (though it always isn't)
                # TODO: WOS exports may also have "keywords-plus"

                parsed_data_by_filename[filename].append(env(author=authors_str,
                                                             year=year,
                                                             title=title,
                                                             abstract=abstract))
    n_entries_total = sum(len(entries) for entries in bibtex_entries_by_filename.values())
    logger.info(f"    {n_entries_total} total entries processed.")
    logger.info(f"    Done in {tim.dt:0.6g}s [avg {n_entries_total / tim.dt:0.6g} entries/s].")

    # --------------------------------------------------------------------------------
    # Prepare filenames

    # Caches are per input file, so that it is fast to concatenate new files to the dataset.
    embeddings_cache_filenames = {fn: utils.make_cache_filename(fn, "embeddings_cache", "npz") for fn in resolved_filenames}
    nlp_cache_filenames = {fn: utils.make_cache_filename(fn, "nlp_cache", "pickle") for fn in resolved_filenames}
    nlp_cache_version = 1

    # Figures are for the whole input dataset, which consists of all input files.
    # For the visualizer, gather all input filenames into a string that can be used in the filenames of the output figures to easily identify where they came from.
    all_input_filenames_list = [utils.strip_ext(os.path.basename(fn)) for fn in resolved_filenames]
    all_input_filenames_str = "_".join(all_input_filenames_list)

    # --------------------------------------------------------------------------------
    # Prepare semantic space (embedding space)

    logger.info("Preparing semantic space...")

    all_vectors_by_filename = {}
    sentence_embedder = None
    for filename, entries in parsed_data_by_filename.items():
        logger.info(f"    Preparing semantic space for {filename}...")

        # TODO: clear memory between input files to avoid running out of RAM/VRAM when we have lots of files

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
        if cache_state == "ok":
            logger.info(f"        Using cached embeddings '{embeddings_cache_filename}'...")
            all_vectors = embeddings_cached_data["all_vectors"]
        else:
            logger.info(f"        No cached embeddings '{embeddings_cache_filename}', reason: {cache_state}")
            logger.info("        Computing embeddings...")
            if sentence_embedder is None:
                logger.info("        Loading SentenceTransformer...")
                with timer() as tim:
                    from sentence_transformers import SentenceTransformer
                    sentence_embedder = SentenceTransformer(config.embedding_model, device=config.device_string)
                logger.info(f"            Done in {tim.dt:0.6g}s.")

            logger.info("        Encoding...")
            with timer() as tim:
                all_inputs = [entry.title for entry in entries]  # with mpnet, this works best
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
    all_vectors = np.concatenate(list(all_vectors_by_filename.values()))

    # --------------------------------------------------------------------------------
    # Compute semantic clusters.
    #
    # The semantic vectors produced by the embedder live on the surface of the unit hypersphere in the hiD space - i.e., each vector is
    # a direction in that space. So we can compare semantic vectors using cosine similarity.
    #
    logger.info("    Seeding clusters in hiD...")
    logger.info(f"        Full dataset has {len(all_vectors)} data points.")
    with timer() as tim:
        from sklearn.cluster import HDBSCAN
        clusterer = HDBSCAN(min_cluster_size=5, min_samples=1, cluster_selection_method="leaf", metric="cosine", store_centers="medoid")
        # clusterer = HDBSCAN(min_cluster_size=5, min_samples=1, cluster_selection_method="leaf", metric="euclidean", store_centers="medoid", algorithm="kd_tree")
        # clusterer = HDBSCAN(min_cluster_size=20, min_samples=5, metric="cosine", store_centers="medoid")  # n_jobs=6,  # n_jobs=-1,
        if len(all_vectors) <= 10000:
            logger.info("        Dataset is small, using the full dataset.")
            fit_idxs = np.arange(len(all_vectors), dtype=np.int64)
            fit_vectors = all_vectors
        else:  # Too many (HDBSCAN runs out of memory on 32GB for ~50k).
            # Take a random set of 10k entries, and hope we hit all the important clusters.
            logger.info("        Dataset is large, picking 10000 random entries for cluster detection.")
            fit_idxs = np.random.randint(len(all_vectors), size=10000)
            fit_vectors = all_vectors[fit_idxs]
        clusterer.fit(fit_vectors)
        unique_vs = clusterer.medoids_
        n_clusters = np.shape(unique_vs)[0]
        n_outliers = sum(clusterer.labels_ == -1)

        if n_clusters > 0:
            logger.info(f"        Detected {n_clusters} seed clusters in hiD, with {n_outliers} outlier data points (out of {len(fit_vectors)} data points used).")

            # EXPERIMENTAL: instead of using the medoids, pick a few random representative points from each cluster (we just take the first up-to-k points now)
            # TODO: This discards the medoids for now. Include them, too.
            points_by_cluster = [all_vectors[fit_idxs[clusterer.labels_ == label]] for label in range(n_clusters)]  # automatically discards outliers (label -1)
            samples_by_cluster = [points[:20, :] for points in points_by_cluster]
            unique_vs = np.concatenate(samples_by_cluster)
            logger.info(f"        Picked a total of {len(unique_vs)} representative points from the detected seed clusters.")
        else:
            logger.info("        No clusters detected in hiD data. Don't know how to visualize this dataset. Canceling.")
            import sys
            sys.exit(1)
            unique_vs = all_vectors
    logger.info(f"        Done in {tim.dt:0.6g}s.")

    # --------------------------------------------------------------------------------
    # Find representative keywords by word frequency over the whole dataset

    if config.extract_keywords:
        logger.info("NLP analysis...")

        all_keywords_by_filename = {}
        nlp_pipeline = None
        for filename, entries in parsed_data_by_filename.items():
            logger.info(f"    NLP analysis for {filename}...")

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
                        spacy.require_gpu()
                        nlp_pipeline = spacy.load(config.spacy_model)
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
                    # U = int(0.2 * n_entries_total)  # drop any word with at least this many occurrences
                    U = float("+inf")  # upper limit disabled
                    L = 2  # drop any word with fewer occurrences than this
                    representative_words = {x: word_counts[x] for x in word_counts if L <= word_counts[x] <= U}
                    return representative_words

                def format_entry_for_keyword_extraction(entry: env) -> str:
                    # return entry.title
                    return entry.title + ".\n\n" + entry.abstract

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
                logger.info(f"                Done in {tim.dt:0.6g}s [avg {n_entries_total / tim.dt:0.6g} entries/s].")

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

                # Tag the abstracts.
                logger.info(f"        Trimming word counts and tagging named entities for data from {filename}...")
                fa_times = []
                ner_times = []
                with timer() as tim:
                    entry_keywords_for_this_file = []
                    entry_entities_for_this_file = []
                    for entry, doc in zip(entries, all_docs_for_this_file):
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

        # Merge the keyword data from all input files.
        #
        logger.info("    Combining keywords from all input files...")
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
        def rank_keyword(word: str, keyword_counts: dict) -> int:
            """Sort key: rank `word` by number of occurrences in `keyword_counts`, descending.

            Words that do not appear in `keyword_counts` are sorted to the end.
            """
            if word not in keyword_counts:
                return float("+inf")
            return -keyword_counts[word]

        logger.info("    Tagging data with visualization keywords...")
        with timer() as tim:
            for filename, entries in parsed_data_by_filename.items():
                logger.info(f"        Tagging visualization keywords for {filename}...")
                for entry in entries:
                    kws = entry.keywords
                    ents = entry.entities

                    # Find the highest-frequency keywords for this entry.
                    kws = {k: v for k, v in kws.items() if k in all_keywords}  # Drop keywords that did not appear in the trimmed keyword analysis across the whole dataset.
                    kws = list(sorted(kws.keys(), key=functools.partial(rank_keyword, keyword_counts=all_keywords)))
                    kws = kws[:6]

                    # Add in the named entities.
                    ents = {x for x in ents if x.lower() not in kws}  # omit entities already present in the highest-frequency keywords
                    ents = list(sorted(ents))

                    # Human-readable list of the most important detected keywords for this entry. Currently unused.
                    entry.vis_keywords = kws + ents
        logger.info(f"        Done in {tim.dt:0.6g}s.")

        # for filename, entries in parsed_data_by_filename.items():  # DEBUG - prints the full dataset!
        #    for entry in entries:
        #        logger.info(f"{entry.author} ({entry.year}): {entry.title}")
        #        logger.info(f"    {entry.abstract}")
        #
        #        # with timer() as tim:
        #        #     logger.info(f"    TL;DR: {tldr(entry.abstract)}")  # DEBUG / EXPERIMENTAL / EXPENSIVE
        #        # logger.info(f"    TL;DR done in {tim.dt:0.6g}s.")
        #
        #        logger.info(f"    {entry.vis_keywords}")
        #        logger.info("")
    else:
        logger.info("Keyword extraction disabled, skipping NLP analysis.")
        all_keywords = {}

    # --------------------------------------------------------------------------------
    # Summarization (EXPERIMENTAL / EXPENSIVE)

    if config.summarize:
        logger.info("Summarizing each abstract using AI...")
        for filename, entries in parsed_data_by_filename.items():
            logger.info(f"    Summarizing abstracts from {filename}...")
            with timer() as tim:
                for entry in entries:
                    summary = tldr(entry.abstract)
                    entry.summary = summary
            logger.info(f"        Done in {tim.dt:0.6g}s [avg {len(entries) / tim.dt:0.6g} entries/s].")
    else:
        logger.info("AI summarization disabled, skipping.")

    # TODO: Split dimension reduction and visualization into a separate program.
    # UMAP needs to load TensorFlow, and after this point our script no longer needs the PyTorch stuff.

    logger.info("Dimension reduction for visualization...")
    logger.info(f"    Loading dimension reduction library for '{config.vis_method}'...")
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
            # See McInnes et al. (2020, revised v3 of paper originally published in 2018):
            #   https://arxiv.org/abs/1802.03426
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
            assert False
    logger.info(f"        Done in {tim.dt:0.6g}s.")
    with timer() as tim:
        logger.info(f"    Learning hiD->2D dimension reduction from the detected {n_clusters} clusters, using {len(unique_vs)} representative points...")
        trans = trans.fit(unique_vs)
        # trans = trans.fit(all_vectors)  # DEBUG: high quality result, but extremely expensive! (several minutes for 5k points)
    logger.info(f"        Done in {tim.dt:0.6g}s.")
    with timer() as tim:
        logger.info(f"    Applying learned dimension reduction to full dataset [n = {n_entries_total}]...")
        lowdim_data = trans.transform(all_vectors)
        # lowdim_centroids = trans.transform(unique_vs)  # DEBUG: where did our representative points end up in 2D?
    logger.info(f"        Done in {tim.dt:0.6g}s.")

    # Cluster the low-dimensional data.
    #
    # Judging by paper titles, the initial hiD clustering seems to "seed" the 2D clusters correctly,
    # so that the t-SNE fit to the representative points maps the full data into the 2D space in a
    # semantically sensible manner.
    #
    logger.info("    Clustering [final, in 2D]...")

    # Concatenate data from individual input files into one large dataset. This is what we will visualize.
    vis_data = list(itertools.chain.from_iterable(parsed_data_by_filename.values()))

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

    # Find a set of keywords for each cluster
    if config.extract_keywords:
        logger.info("    Extracting keywords for each cluster...")
        with timer() as tim:
            logger.info("        Collecting keywords from data points in each cluster...")
            keywords_by_cluster = [collections.Counter() for _ in range(n_vis_clusters)]
            for entry in vis_data:
                if entry.cluster_id >= 0:  # not an outlier
                    # Update cluster keyword counters from this entry.
                    # TODO: We could also use `entry.cluster_probability` for something here.
                    # TODO: Including entities would be nice, but they don't currently have frequency information.
                    keywords_by_cluster[entry.cluster_id].update(entry.keywords)  # inject keywords of this entry to the keywords of the cluster this entry belongs to

            # Omit useless keywords from visualization; try to focus on what distinguishes each cluster from the others.
            # There's a fine balance here, drawing on the old NLP observation that words with intermediate frequencies best describe a dataset.
            #
            #   - Filtering out keywords present in *all* other clusters drops only a few of the most common words, leaving uselessly large overlaps between the keyword sets.
            #     This happens even after we have accounted for stopwords, because by assumption, the dataset discusses a single broad umbrella topic (one field of science).
            #   - Filtering out keywords present in *any* other cluster leaves only noise.
            #   - So we must set a threshold fraction such that if at least that fraction of all clusters have a particular keyword, then that keyword is considered
            #     uselessly common for the purposes of distinguishing clusters.
            #
            # fraction = 0.25
            fraction = 0.1
            threshold_n = max(min(5, n_vis_clusters), math.ceil(fraction * n_vis_clusters))  # Very small datasets may produce very few clusters.
            cluster_counts_by_keyword = collections.Counter()
            for cluster_id, kws in enumerate(keywords_by_cluster):
                for kw in kws.keys():
                    cluster_counts_by_keyword[kw] += 1  # found one more cluster that has this keyword
            keywords_common_to_most_clusters = {kw for kw, count in cluster_counts_by_keyword.items() if count >= threshold_n}
            logger.info(f"        Found {len(keywords_common_to_most_clusters)} common (useless) keywords at threshold fraction {fraction:0.2g} (shared between {threshold_n} clusters out of {n_vis_clusters}).")

            # keywords_common_to_all_clusters = set(keywords_by_cluster[0].keys())
            # for kws in keywords_by_cluster[1:]:
            #     kws = set(kws.keys())
            #     keywords_common_to_all_clusters.intersection_update(kws)

            # Keep the highest-frequency keywords detected in each cluster. Hopefully this will show what the cluster is about.
            # But remove keywords common with other clusters, keep only uniques (filter after the fact, to make it symmetric; we don't want filler words to end up in the first cluster).
            logger.info("        Ranking and filtering results...")
            vis_keywords_by_cluster = []
            for cluster_id, kws in enumerate(keywords_by_cluster):
                kws = set(kws.keys())

                # # Drop keywords present in any other cluster
                # other_cluster_keywords = keywords_by_cluster[:cluster_id] + keywords_by_cluster[cluster_id + 1:]
                # for other_kws in other_cluster_keywords:
                #     kws = kws.difference(other_kws)

                kws = kws.difference(keywords_common_to_most_clusters)

                kws = list(sorted(kws, key=functools.partial(rank_keyword, keyword_counts=all_keywords)))
                kws = kws[:6]
                vis_keywords_by_cluster.append(kws)
        logger.info(f"        Done in {tim.dt:0.6g}s.")
    else:
        vis_keywords_by_cluster = []

    logger.info(f"Saving visualization datafile {opts.output_filename}...")

    # Be sure to save the values of any settings that affect data availability and interpretation! (E.g. `extract_keywords` -> whether annotations and word cloud can be plotted from this data.)

    output_file_version = 1
    with timer() as tim:
        output_data = {"version": output_file_version,
                       "all_input_filenames_raw": resolved_filenames,  # actual paths
                       "all_input_filenames_list": all_input_filenames_list,  # just the filenames (no path)
                       "all_input_filenames_str": all_input_filenames_str,  # "file1_file2_..._fileN"
                       "embedding_model": config.embedding_model,
                       "vis_method": config.vis_method,  # dimension reduction method
                       "n_vis_clusters": n_vis_clusters,  # number of clusters detected
                       "labels": vis_clusterer.labels_,
                       "vis_data": vis_data,
                       "lowdim_data": lowdim_data,
                       "keywords_available": config.extract_keywords,
                       "all_keywords": all_keywords,
                       "vis_keywords_by_cluster": vis_keywords_by_cluster}
        with open(opts.output_filename, "wb") as output_file:
            pickle.dump(output_data, output_file)
    logger.info(f"    Done in {tim.dt:0.6g}s.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Extract relevant fields from BibTeX file(s), for abstract summarization.""",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(dest="output_filename", type=str, metavar="out", help="Output file to save analysis data in")
    parser.add_argument(dest="filenames", nargs="+", default=None, type=str, metavar="bib", help="BibTeX file(s) to parse")
    opts = parser.parse_args()

    if opts.output_filename.endswith(".bib"):
        print(f"Output filename '{opts.output_filename}' looks like an input file. Cancelling. Please check up-to-date usage summary by running this script with the '-h' option.")
        sys.exit(1)

    # --------------------------------------------------------------------------------

    logger.info("Settings:")
    logger.info(f"    Device '{config.device_string}' ({torch.cuda.get_device_name(config.device_string)}), data type {config.torch_dtype}")
    logger.info(f"        {torch.cuda.get_device_properties(config.device_string)}")
    # logger.info(f"        Compute capability {'.'.join(str(x) for x in torch.cuda.get_device_capability(device_string))}")
    logger.info(f"    Embedding model: {config.embedding_model}")
    logger.info(f"        Dimension reduction method: {config.vis_method}")
    logger.info(f"    Extract keywords: {config.extract_keywords}")
    logger.info(f"        NLP model (spaCy): {config.spacy_model}")
    logger.info(f"    Summarize via AI: {config.summarize}")
    logger.info(f"        AI summarization model: {config.summarization_model}")

    with timer() as tim:
        main(opts)
    logger.info(f"All done in {tim.dt:0.6g}s total.")

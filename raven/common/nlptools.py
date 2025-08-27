"""NLP (natural language processing) tools.

- Stopword set management.
- Word frequency analysis tools, useful for keyword detection.
- NLP backend loading and functions to call the models.
  - model caching: load only one copy of each model in the same process (if requested for the same device with the same dtype).
  - device management (which device to load on), with automatic CPU fallback if loading on GPU fails.

Backends:
  - spaCy NLP pipeline
  - classification (text sentiment)
  - dehyphenation of broken text (e.g. as extracted from a PDF)
  - embeddings (semantic embedding of sentences for use as vector DB keys)
  - summarization via specialized AI model
"""

__all__ = {"default_stopwords",
           "load_spacy_pipeline",
           "serialize_spacy_pipeline", "deserialize_spacy_pipeline",
           "serialize_spacy_docs", "deserialize_spacy_docs",
           "load_classifier", "classify",
           "load_dehyphenator", "dehyphenate",
           "load_embedder", "embed_sentences",
           "load_summarizer", "summarize",
           "load_translator", "translate",
           "count_frequencies", "detect_named_entities",
           "suggest_keywords"}

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import collections
import copy
import math
import operator
import re
from typing import Container, Dict, List, Optional, Tuple, Union

import numpy as np

import torch  # import torch before spaCy, so that spaCy finds the GPU (otherwise spaCy will complain that CuPy is not installed, which is not true)

from transformers import pipeline
from sentence_transformers import SentenceTransformer

import spacy
from spacy.tokens import DocBin

import flair
import dehyphen

from . import utils as common_utils
from . import numutils

# --------------------------------------------------------------------------------
# Stopword management

def _load_stopwords() -> List[str]:
    """Load a default list of stopwords (English default list from spaCy)."""
    from spacy.lang.en import English
    nlp_en = English()
    stopwords = nlp_en.Defaults.stop_words
    return stopwords

# We apply stopwords case-insensitively, so lowercase them now.
default_stopwords = set(x.lower() for x in _load_stopwords())

# --------------------------------------------------------------------------------
# NLP backend: spaCy NLP pipeline

# For this backend, we provide no utility functions; do what you want with the `nlp` object.

_spacy_pipelines = {}
def load_spacy_pipeline(model_name: str, device_string: str):
    """Load and return a spaCy NLP pipeline.

    `model_name`: spaCy NLP model name. This is NOT a HuggingFace model name, but is auto-downloaded (by spaCy) on first use.

                  See:
                      https://spacy.io/models

                  For English specifically, usually "en_core_web_sm" (CPU-friendly) or "en_core_web_trf" (Transformer model,
                  GPU highly recommended) are good choices.

    `device_string`: as in Torch, e.g. "cpu", "cuda", or "cuda:0".

    If the specified model is already loaded on the same device (identified by `device_string`), return the already-loaded instance.
    """
    cache_key = (model_name, device_string)
    if cache_key in _spacy_pipelines:
        logger.info(f"load_spacy_pipeline: '{model_name}' is already loaded on device '{device_string}', returning it.")
        return _spacy_pipelines[cache_key]
    logger.info(f"load_spacy_pipeline: Loading '{model_name}' on device '{device_string}'.")

    if device_string.startswith("cuda"):
        if ":" in device_string:
            _, gpu_id = device_string.split(":")
            gpu_id = int(gpu_id)
        else:
            gpu_id = 0

        try:
            spacy.require_gpu(gpu_id=gpu_id)
            logger.info("load_spacy_pipeline: spaCy will run on GPU (if possible).")
        except Exception as exc:
            logger.warning(f"load_spacy_pipeline: exception while enabling GPU for spaCy: {type(exc)}: {exc}")
            spacy.require_cpu()
            logger.info("load_spacy_pipeline: spaCy will run on CPU.")
            device_string = "cpu"
            cache_key = (model_name, device_string)
    else:
        spacy.require_cpu()
        logger.info("load_spacy_pipeline: spaCy will run on CPU.")
        device_string = "cpu"
        cache_key = (model_name, device_string)

    try:
        nlp = spacy.load(model_name)
    except OSError:
        # https://stackoverflow.com/questions/62728854/how-to-place-spacy-en-core-web-md-model-in-python-package
        logger.info(f"load_spacy_pipeline: Downloading spaCy model '{model_name}' (don't worry, this will only happen once)...")
        from spacy.cli import download
        download(model_name)
        nlp = spacy.load(model_name)

    logger.info(f"load_spacy_pipeline: Loaded spaCy model '{model_name}' on device '{device_string}'.")
    _spacy_pipelines[cache_key] = nlp
    return nlp

# The following four functions are based on:
#   https://spacy.io/usage/saving-loading
#   https://spacy.io/usage/processing-pipelines
#   https://spacy.io/api/language#config
#
def serialize_spacy_pipeline(nlp) -> Tuple[str, bytes]:
    """Utility to serialize a spaCy NLP pipeline so that it can be sent over the network.

    NOTE: This is NOT the function you want for typical use cases. This prepares the pipeline
          itself for sending, whereas in a client/server setup, one typically wants to send just
          the analyzed documents to avoid the need to instantiate a GPU-based pipeline on the client.

          See `serialize_spacy_docs`.

    `nlp`: return value of `load_spacy_pipeline`

    Returns `(config_str, data_bytes)`, where:
      `config_str` is the pipeline config as a string.
      `data_bytes` is the binary data that stores all non-config parts of the pipeline.

    Both values are needed at deserialization time.
    """
    config_str = nlp.config.to_str()
    data_bytes = nlp.to_bytes()
    return config_str, data_bytes

def deserialize_spacy_pipeline(config_str: str, data_bytes: bytes):
    """Utility to deserialize a spaCy NLP pipeline that was earlier serialized with `serialize_spacy_pipeline`.

    NOTE: This is NOT the function you want for typical use cases. This reconstructs the pipeline
          itself, whereas in a client/server setup, one typically wants to receive just the analyzed
          documents to avoid the need to instantiate a GPU-based pipeline on the client.

          See `deserialize_spacy_docs`.

    `config_str`, `data_bytes`: return values of `serialize_spacy_pipeline`.

    Returns the NLP pipeline object.

    NOTE: Currently, this does NOT update our pipeline cache, because the model name and device string are lost.
    """
    config = spacy.Config().from_str(config_str)
    lang_cls = spacy.util.get_lang_class(config["nlp"]["lang"])
    nlp = lang_cls.from_config(config)
    nlp.from_bytes(data_bytes)
    return nlp

def serialize_spacy_docs(docs: Union[List[spacy.tokens.token.Token],
                                     List[List[spacy.tokens.token.Token]]]) -> bytes:
    """Serialize one or more spaCy NLP pipeline outputs (spaCy "documents") so that they can be sent over the network.

    `docs`: One or more documents.
        If `docs` is a list of tokens, it is treated as one document.
        Otherwise `docs` is treated as a list of documents (where each document is a list of tokens).

    Returns a `bytes` object containing the serialized data.

    Example::

        doc1 = nlp("The quick brown fox jumps over the lazy dog.")
        data1 = serialize_spacy_docs(doc1)  # serialize one document

        doc2 = nlp("This is another document.")
        data2 = serialize_spacy_docs([doc1, doc2])  # serialize multiple documents

        # ...in another process, later...

        docs1 = deserialize_spacy_docs(data1, lang="en")
        assert len(docs1) == 1  # one document received
        doc = docs1[0]  # -> a copy of the data of `doc1`

        # Process the analysis results the same way as if you had called `doc = nlp(text)` in this process.
        for token in doc:
            print(token.text, token.lemma_)

        docs2 = deserialize_spacy_docs(data2, lang="en")
        assert len(docs2) == 2  # two documents received
        for doc in docs2:
            for token in doc:
                print(token.text, token.lemma_)

    NOTE: As noted in spaCy documentation, if you have multiple documents that were processed with the same pipeline,
    it is more efficient to send all at once, because then e.g. the vocabulary needs to be serialized only once
    for the whole set of documents. This uses spaCy's `DocBin` to do that.

    See:
        https://spacy.io/usage/saving-loading
    """
    doc_bin = DocBin()
    if isinstance(docs[0], spacy.tokens.token.Token):  # list of tokens, i.e. single document
        doc_bin.add(docs)
    else:  # list of list of tokens, i.e. multiple documents
        for doc in docs:
            doc_bin.add(doc)
    return doc_bin.to_bytes()

def deserialize_spacy_docs(docs_bytes: bytes, lang: str) -> List[List[spacy.tokens.token.Token]]:
    """Deserialize one or more spaCy NLP pipeline outputs (spaCy "documents") that were earlier serialized with `serialize_spacy_docs`.

    `docs_bytes`: return value of `serialize_spacy_docs`.

    `lang`: spaCy language code, e.g. "en" for English.

            This should match the language of the spaCy NLP pipeline that analyzed the documents at the sending end.

    Returns a `list` of spaCy documents (even if there is only one document stored in `docs_bytes`).

    The documents are loaded into a blank pipeline of language `lang`.

    You can then read the analysis output as usual (e.g. `[token.lemma_ for token in doc]`).

    See example in docstring of `serialize_spacy_docs`.
    """
    nlp = spacy.blank(lang)
    doc_bin = DocBin().from_bytes(docs_bytes)
    docs = list(doc_bin.get_docs(nlp.vocab))
    return docs

# --------------------------------------------------------------------------------
# NLP backend: sentiment classification

_classifiers = {}
def load_classifier(model_name: str, device_string: str, torch_dtype: Union[str, torch.dtype]) -> pipeline:
    """Load and return a text sentiment classification model.

    `model_name`: HuggingFace model name. Auto-downloaded on first use.
                  Try e.g. "joeddav/distilbert-base-uncased-go-emotions-student" for a 28-emotion model.

                  See:
                      https://huggingface.co/tasks/text-classification

    `device_string`: as in Torch, e.g. "cpu", "cuda", or "cuda:0".
    `torch_dtype`: e.g. "float32", "float16" (on GPU), or `torch.float16` (same thing).

    If the specified model is already loaded on the same device (identified by `device_string`),
    with the same dtype, then return the already-loaded instance.
    """
    cache_key = (model_name, device_string, str(torch_dtype))
    if cache_key in _classifiers:
        logger.info(f"load_classifier: '{model_name}' (with dtype '{str(torch_dtype)}') is already loaded on device '{device_string}', returning it.")
        return _classifiers[cache_key]
    logger.info(f"load_classifier: Loading '{model_name}' (with dtype '{str(torch_dtype)}') on device '{device_string}'.")

    try:
        device = torch.device(device_string)
        classifier = pipeline("text-classification",
                              model=model_name,
                              top_k=None,
                              device=device,
                              torch_dtype=torch_dtype)
    except RuntimeError as exc:
        logger.warning(f"load_classifier: exception while loading classifier (will try again in CPU mode): {type(exc)}: {exc}")
        try:
            device_string = "cpu"
            torch_dtype = "float32"
            cache_key = (model_name, device_string, str(torch_dtype))
            classifier = pipeline("text-classification",
                                  model=model_name,
                                  top_k=None,
                                  device=device,
                                  torch_dtype=torch_dtype)
        except RuntimeError as exc:
            logger.warning(f"load_classifier: failed to load classifier: {type(exc)}: {exc}")
            raise
    logger.info(f"load_classifier: Loaded model '{model_name}' (with dtype '{str(torch_dtype)}') on device '{device_string}'.")
    _classifiers[cache_key] = classifier
    return classifier

def classify(classifier: pipeline, text: str) -> list:
    """Classify the sentiment of `text`.

    `classifier`: return value of `load_classifier`

    Returns a list in the format:

        [{"label": emotion0, "score": confidence0},
         ...]

    sorted by score, descending.
    """
    output = classifier(text,
                        truncation=True,
                        max_length=classifier.model.config.max_position_embeddings)[0]
    return list(sorted(output,
                       key=operator.itemgetter("score"),
                       reverse=True))

# --------------------------------------------------------------------------------
# NLP backend: dehyphenation

_dehyphenators = {}
def load_dehyphenator(model_name: str, device_string: str) -> dehyphen.FlairScorer:
    """Load and return the dehyphenator, for fixing broken text (e.g. as extracted from PDFs).

    If the specified model is already loaded on the same device (identified by `device_string`), return the already-loaded instance.

    `model_name`: Flair-NLP contextual embeddings model, see:
        https://github.com/flairNLP/flair/blob/master/resources/docs/embeddings/FLAIR_EMBEDDINGS.md
        https://github.com/flairNLP/flair/blob/master/flair/embeddings/token.py

        This model is loaded by the `dehyphen` package; omit the "-forward" or "-backward" part
        of the model name, those are added automatically.

        At first, try "multi", it should support 300+ languages. If that doesn't perform adequately,
        then look at the docs.

    `device_string`: as in Torch, e.g. "cpu", "cuda", or "cuda:0".

    See:
        https://github.com/pd3f/dehyphen
        https://github.com/flairNLP/flair
    """
    cache_key = (model_name, device_string)
    if cache_key in _dehyphenators:
        logger.info(f"load_dehyphenator: '{model_name}' is already loaded on device '{device_string}', returning it.")
        return _dehyphenators[cache_key]
    logger.info(f"load_dehyphenator: Loading '{model_name}' on device '{device_string}'.")

    # Flair requires "no weights only load" mode for Torch; but this is insecure, so only enable it temporarily while loading the Flair model.
    #   https://github.com/flairNLP/flair/issues/3263
    #   https://github.com/pytorch/pytorch/blob/main/torch/serialization.py#L1443
    logger.warning("load_dehyphenator: Temporarily forcing Torch into 'no weights only' load mode for Flair-NLP compatibility. The mode will be disabled immediately after Flair-NLP is loaded. The security warning is normal.")
    with common_utils.environ_override(TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1"):
        try:
            # How to set CPU/GPU mode for Flair (used by `dehyphen`).
            # This needs to be done *before* instantiating the model.
            #   https://github.com/flairNLP/flair/issues/464
            flair.device = device_string  # TODO, FIXME, UGH!
            scorer = dehyphen.FlairScorer(lang=model_name)
        except RuntimeError as exc:
            logger.warning(f"load_dehyphenator: exception while loading dehyphenator (will try again in CPU mode): {type(exc)}: {exc}")
            try:
                device_string = "cpu"
                cache_key = (model_name, device_string)
                flair.device = device_string  # TODO, FIXME, UGH!
                scorer = dehyphen.FlairScorer(lang=model_name)
            except Exception as exc:
                logger.warning(f"load_dehyphenator: failed to load dehyphenator: {type(exc)}: {exc}")
                raise
    logger.info(f"load_dehyphenator: Loaded model '{model_name}' on device '{device_string}'.")
    _dehyphenators[cache_key] = scorer
    return scorer

def _join_paragraphs(scorer: dehyphen.FlairScorer, candidate_paragraphs: List[str]) -> List[str]:  # TODO: Should probably move this into `dehyphen`
    """Internal function; input/output format is as produced by `dehyphen.text_to_format`.

    Essentially, `[[lines, of paragraph, one], [lines, of, paragraph, two], ...]`, where each of the lines is a string.

    `scorer`: return value of `load_dehyphenator`
    """
    if len(candidate_paragraphs) >= 2:
        out = []
        candidate1 = candidate_paragraphs[0]
        j = 1

        # handle blank lines at beginning of input
        while not len(candidate1):  # no lines in this paragraph?
            candidate1 = candidate_paragraphs[j]
            j += 1

            # all of input is blank lines?
            if j == len(candidate_paragraphs):
                out.append(candidate1)
                return out

        while True:
            candidate2 = candidate_paragraphs[j]

            # # DEBUG
            # print("=" * 80)
            # print(j)
            # print("-" * 80)
            # print("LEFT:")
            # print(candidate1)
            # print("-" * 80)
            # print("RIGHT:")
            # print(candidate2)

            combined = scorer.is_split_paragraph(candidate1, candidate2)

            # # DEBUG
            # print("-" * 80)
            # print(f"combined: {combined}")  # essentially `candidate1 + candidate2` (if `dehyphen` thinks it wasn't complete) or `None` (if it thinks it was complete)
            # print("-" * 80)
            # print("LEFT is probably complete, committing" if combined is None else "LEFT is probably NOT complete, joining")

            if j == len(candidate_paragraphs) - 1:  # end of text: commit whatever we have left
                if combined is None:  # candidate1 is a complete paragraph (candidate2 starts a new paragraph)
                    out.append(candidate1)
                    out.append(candidate2)
                else:
                    out.append(combined)
                break
            else:  # general case: commit only when a paragraph is completed
                if combined is None:  # candidate1 is a complete paragraph (candidate2 starts a new paragraph)
                    out.append(candidate1)
                    candidate1 = candidate2
                else:  # keep combining
                    candidate1 = combined
                j += 1
    else:
        out = copy.copy(candidate_paragraphs)
    return out

def dehyphenate(scorer: dehyphen.FlairScorer, text: Union[str, List[str]]) -> Union[str, List[str]]:
    """Dehyphenate broken text (e.g. as extracted from a PDF), via perplexity analysis using a character-level AI model for NLP.

    `scorer`: return value of `load_dehyphenator`
    `text`: one or more texts to dehyphenate.

    Returns `str` (one input) or `list` of `str` (more inputs).
    """
    def doit(text: str) -> str:
        # Don't send if the input is a single character, to avoid crashing `dehyphen`.
        if len(text) == 1:
            return text
        data = dehyphen.text_to_format(text)
        data = scorer.dehyphen(data)
        data = _join_paragraphs(scorer, data)
        paragraphs = [dehyphen.format_to_paragraph(lines) for lines in data]
        output_text = "\n\n".join(paragraphs)
        return output_text
    if isinstance(text, list):
        output_text = [doit(item) for item in text]
    else:  # str
        output_text = doit(text)
    return output_text

# --------------------------------------------------------------------------------
# NLP backend: semantic embeddings (vectorization)

_embedders = {}
def load_embedder(model_name: str, device_string: str, torch_dtype: Union[str, torch.dtype]) -> SentenceTransformer:
    """Load and return a semantic embedding model (for e.g. vector storage).

    `model_name`: HuggingFace model name supported by the `sentence_transformers` package. Auto-downloaded on first use.

                  For general use, try e.g. "sentence-transformers/all-mpnet-base-v2" or "Snowflake/snowflake-arctic-embed-l".

                  If you need to embed questions and answers to those questions near each other, try e.g.
                  "sentence-transformers/multi-qa-mpnet-base-cos-v1".

                  See:
                      https://sbert.net/docs/sentence_transformer/pretrained_models.html
                      https://huggingface.co/tasks/sentence-similarity

    `device_string`: as in Torch, e.g. "cpu", "cuda", or "cuda:0".
    `torch_dtype`: e.g. "float32", "float16" (on GPU), or `torch.float16` (same thing).

    If the specified model is already loaded on the same device (identified by `device_string`),
    with the same dtype, then return the already-loaded instance.
    """
    cache_key = (model_name, device_string, str(torch_dtype))
    if cache_key in _embedders:
        logger.info(f"load_embedder: '{model_name}' (with dtype '{str(torch_dtype)}') is already loaded on device '{device_string}', returning it.")
        return _embedders[cache_key]
    logger.info(f"load_embedder: Loading '{model_name}' (with dtype '{str(torch_dtype)}') on device '{device_string}'.")

    try:
        embedder = SentenceTransformer(model_name,
                                       device=device_string,
                                       model_kwargs={"torch_dtype": torch_dtype})
    except RuntimeError as exc:
        logger.warning(f"load_embedder: exception while loading SentenceTransformer (will try again in CPU mode): {type(exc)}: {exc}")
        try:
            device_string = "cpu"
            torch_dtype = "float32"  # probably (we need a cache key, so let's use this)
            cache_key = (model_name, device_string, str(torch_dtype))
            embedder = SentenceTransformer(model_name, device=device_string)
        except RuntimeError as exc:
            logger.warning(f"load_embedder: failed to load SentenceTransformer: {type(exc)}: {exc}")
            raise
    logger.info(f"load_embedder: Loaded model '{model_name}' (with dtype '{str(torch_dtype)}') on device '{device_string}'.")
    _embedders[cache_key] = embedder
    return embedder

def embed_sentences(embedder: SentenceTransformer, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
    """Embed (vectorize) one or more sentences using a semantic embedding AI model.

    `embedder`: return value of `load_embedder`
    `text`: one (str) or more (List[str]) texts to embed

    Returns a `list` for one input, a `list` of `list`s for more inputs. This is to keep the output easily JSONable
    (NumPy arrays aren't), to facilitate easily sending the data over the network.
    """
    vectors: Union[np.array, List[np.array]] = embedder.encode(text,
                                                               show_progress_bar=True,  # on console running this app
                                                               convert_to_numpy=True,
                                                               normalize_embeddings=True)
    # NumPy arrays are not JSON serializable, so convert to Python lists
    if isinstance(vectors, np.ndarray):
        vectors = vectors.tolist()
    else:  # isinstance(vectors, list) and all(isinstance(x, np.ndarray) for x in vectors)
        vectors = [x.tolist() for x in vectors]
    return vectors

# --------------------------------------------------------------------------------
# NLP backend: summarization

_summarizers = {}
def load_summarizer(model_name: str,
                    device_string: str,
                    torch_dtype: Union[str, torch.dtype],
                    summarization_prefix: str = "") -> Tuple[pipeline, str]:
    """Load a text summarizer.

    This is a small AI model specialized to the task of summarization ONLY, not a general-purpose LLM.

    `model_name`: HuggingFace model name. Try e.g. "Qiliang/bart-large-cnn-samsum-ChatGPT_v3".

    `device_string`: as in Torch, e.g. "cpu", "cuda", or "cuda:0".
    `torch_dtype`: e.g. "float32", "float16" (on GPU), or `torch.float16` (same thing).

    `summarization_prefix`: Some summarization models need input to be formatted like
         "summarize: Actual text goes here...". This sets the prefix, which in this example is "summarize: ".
         For whether you need this and what the value should be, see the model card for your particular model.

    NOTE: To use `summarize`, you also need a spaCy NLP pipeline; see `load_spacy_pipeline`.
    """
    cache_key = (model_name, device_string, str(torch_dtype))
    if cache_key in _summarizers:
        return _summarizers[cache_key]

    try:
        device = torch.device(device_string)
        summarizer = pipeline("summarization",
                              model=model_name,
                              device=device,
                              torch_dtype=torch_dtype)
    except RuntimeError as exc:
        logger.warning(f"load_summarizer: exception while loading summarizer (will try again in CPU mode): {type(exc)}: {exc}")
        try:
            device_string = "cpu"
            torch_dtype = "float32"  # probably (we need a cache key, so let's use this)
            cache_key = (model_name, device_string, str(torch_dtype))
            summarizer = pipeline("summarization",
                                  model=model_name,
                                  device=device,
                                  torch_dtype=torch_dtype)
        except RuntimeError as exc:
            logger.warning(f"load_embedder: failed to load summarizer: {type(exc)}: {exc}")
            raise
    logger.info(f"load_summarizer: model '{model_name}' context window is {summarizer.tokenizer.model_max_length} tokens.")
    logger.info(f"load_summarizer: Loaded model '{model_name}' (with dtype '{str(torch_dtype)}') on device '{device_string}'.")
    _summarizers[cache_key] = (summarizer, summarization_prefix)  # save the given prompt prefix with the cached model so they stay together
    return summarizer, summarization_prefix

def _summarize_chunk(summarizer: Tuple[pipeline, str], text: str) -> str:
    """Internal function. Summarize a piece of text that fits into the summarization model's context window.

    If the text does not fit, raises `IndexError`. See `_summarize_chunked` to handle arbitrary length texts.
    """
    text_summarization_pipe, text_summarization_prefix = summarizer
    text = f"{text_summarization_prefix}{text}"  # some summarizer AIs require a prompt (e.g. "summarize: The actual text to be summarized...")

    tokens = text_summarization_pipe.tokenizer.tokenize(text)  # may be useful for debug...
    length_in_tokens = len(tokens)  # ...but this is what we actually need to set up the summarization lengths semsibly
    logger.info(f"_summarize_chunk: Input text length is {len(text)} characters, {length_in_tokens} tokens.")

    if length_in_tokens > text_summarization_pipe.tokenizer.model_max_length:
        logger.info(f"_summarize_chunk: Text to be summarized does not fit into model's context window (text length {len(text)} characters, {length_in_tokens} tokens; model limit {text_summarization_pipe.tokenizer.model_max_length} tokens).")
        raise IndexError  # and let `_summarize_chunked` handle it
    if length_in_tokens <= 20:  # too short to summarize?
        return text

    # TODO: summary length: sensible limits that work for very short (one sentence) and very long (several pages) texts. This is optimized for a paragraph or a few at most.
    lower_limit = min(20, length_in_tokens)  # try to always use at least this many tokens in the summary
    upper_limit = min(120, length_in_tokens)  # and always try to stay under this limit
    max_length = numutils.clamp(length_in_tokens // 2, ell=lower_limit, u=upper_limit)
    min_length = numutils.clamp(length_in_tokens // 10, ell=lower_limit, u=upper_limit)
    logger.info(f"_summarize_chunk: Setting summary length guidelines as min = {min_length} tokens, max = {max_length} tokens.")

    summary = text_summarization_pipe(
        text,
        truncation=False,
        min_length=min_length,
        max_length=max_length,
    )[0]["summary_text"]
    return summary

def _summarize_chunked(summarizer: Tuple[pipeline, str], nlp_pipe, text: str) -> str:
    """Internal function. Summarize a text that may require chunking before it fits into the summarization model's context window."""
    try:
        return _summarize_chunk(summarizer, text)
    except IndexError:
        logger.info(f"_summarize_chunked: input text (length {len(text)} characters) is long; cutting text in half at a sentence boundary and summarizing the halves separately.")

        with nlp_pipe.select_pipes(enable=['tok2vec', "parser", "senter"]):  # process faster by enabling only needed modules; https://stackoverflow.com/a/74907505
            doc = nlp_pipe(text)
        sents = list(doc.sents)
        mid = len(sents) // 2
        firsthalf = " ".join(str(sent).strip() for sent in sents[:mid])
        secondhalf = " ".join(str(sent).strip() for sent in sents[mid:])
        # print("=" * 80)
        # print("Splitting long text:")
        # print("-" * 80)
        # print(firsthalf)
        # print("-" * 80)
        # print(secondhalf)
        # print("-" * 80)
        return " ".join(
            [_summarize_chunked(summarizer, nlp_pipe, firsthalf),
             _summarize_chunked(summarizer, nlp_pipe, secondhalf)]
        )

        # # Sentence-splitting the output of the general sliding-window chunkifier doesn't seem to work that well here. It's easier to correctly split into sentences when we have all the text available at once.
        #
        # def full_sentence_trimmer(overlap, mode, text):
        #     @memoize  # from `unpythonic`
        #     def get_sentences(text):
        #         with nlp_pipe.select_pipes(enable=['tok2vec', "parser", "senter"]):  # process faster by enabling only needed modules; https://stackoverflow.com/a/74907505
        #             doc = nlp_pipe(text)
        #         return list(doc.sents)
        #
        #     offset = 0
        #     tmp = text.strip()  # ignore whitespace at start/end of chunk when detecting incomplete sentences
        #     if mode != "first":  # allowed to trim beginning?
        #         # Lowercase letter at the start of the chunk -> probably not the start of a sentence.
        #         if tmp[0].upper() != tmp[0]:
        #             sents = get_sentences(text)
        #             first_sentence_len = len(sents[0])
        #             offset = min(overlap, first_sentence_len)  # Prefer to keep incomplete sentence when there's not enough chunk overlap to trim it without losing text.
        #             text = text[offset:]
        #     if mode != "last":  # allowed to trim end?
        #         # No punctuation mark at the end of the chunk -> probably not a complete sentence.
        #         if tmp[-1] not in (".", "!", "?"):
        #             sents = get_sentences(text)
        #             last_sentence_len = len(sents[-1])
        #             text = text[:-last_sentence_len]
        #     return text, offset
        #
        # chunks = common_utils.chunkify_text(text,
        #                                     chunk_size=len(text) // 2,
        #                                     overlap=0,
        #                                     extra=0.2,
        #                                     trimmer=full_sentence_trimmer)
        # summary = " ".join(_summarize_chunked(summarizer, nlp_pipe, chunk["text"]) for chunk in chunks)
        # return summary

def summarize(summarizer: Tuple[pipeline, str], nlp_pipe, text: str) -> str:
    """Return an abstractive summary of input text.

    This uses an AI summarization model (see `raven.server.config.summarization_model`),
    plus some heuristics to minimally clean up the result.

    `summarizer`: return value of `load_summarizer`
    `nlp_pipe`: return value of `load_spacy_pipeline` (used for sentence-boundary splitting during chunking)
    """
    def normalize_sentence(sent: str) -> str:
        """Given a sentence, remove surrounding whitespace and capitalize the first word."""
        sent = str(sent).strip()  # `sent` might actually originally be a spaCy output
        sent = sent[0].upper() + sent[1:]
        return sent
    def sanitize(text: str) -> str:
        """Preprocess `text` for summarization.

        Specifically:
          - Normalize Unicode representation to NFKC
          - Normalize whitespace at sentence boundaries (as detected by the loaded spaCy NLP model)
          - Capitalize start of each sentence
          - Drop incomplete last sentence if any, but only if there's more than one sentence in total.
        """
        text = common_utils.normalize_unicode(text)
        text = text.strip()

        # Detect possible incomplete sentence at the end.
        #   - Summarizer AIs sometimes do that, especially if they run into the user-specified output token limit too soon.
        #   - The input text may have been cut off before it reaches us. When this happens, some summarizer AIs become confused.
        #     (E.g. Qiliang/bart-large-cnn-samsum-ChatGPT_v3, given the input:
        #          " The quick brown fox jumped over the lazy dog.  This is the second sentence! What?! This incomplete sentence"
        #      focuses only on the fact that the last sentence is incomplete, and reports that as the summary.)
        end = -1 if text[-1] not in (".", "!", "?") else None

        # Split into sentences via NLP. (This is the sane approach.)
        with nlp_pipe.select_pipes(enable=['tok2vec', "parser", "senter"]):  # Process faster by enabling only needed modules; https://stackoverflow.com/a/74907505
            doc = nlp_pipe(text)
        sents = list(doc.sents)
        if end is not None and len(sents) == 1:  # If only one sentence, keep it even if incomplete.
            end = None
        text = " ".join(normalize_sentence(sent) for sent in sents[:end])
        return text

    # Prompt the summarizer to write the raw summary (AI magic happens here)
    text = sanitize(text)
    summary = _summarize_chunked(summarizer, nlp_pipe, text)

    # Rudimentary check against AI hallucination: summarizing a very short text sometimes fails with the AI making up more text than there is in the original.
    if len(summary) > len(text):
        return text

    # Postprocess the summary
    summary = sanitize(summary)

    # At this point, depending on the AI model, we still sometimes have the spacing for the punctuation as "Blah blah blah . Bluh bluh..."

    # Normalize whitespace at full-stops (periods)
    parts = summary.split(".")
    has_period_at_end = summary.endswith(".")  # might have "!" or "?" instead
    parts = [x.strip() for x in parts]
    parts = [x for x in parts if len(x)]
    summary = ". ".join(parts) + ("." if has_period_at_end else "")
    summary = re.sub(r"(\d)\. (\d)", r"\1.\2", summary)  # Fix decimal numbers broken by the punctuation fix

    # Normalize whitespace at commas
    parts = summary.split(",")
    parts = [x.strip() for x in parts]
    parts = [x for x in parts if len(x)]
    summary = ", ".join(parts)
    summary = re.sub(r"(\d)\, (\d)", r"\1,\2", summary)  # Fix numbers with American thousands separators, broken by the punctuation fix

    # Convert some very basic markup (e.g. superscripts/subscripts) into their Unicode equivalents.
    summary = common_utils.unicodize_basic_markup(summary)

    return summary

# --------------------------------------------------------------------------------
# NLP backend: machine translation

_translators = {}
def load_translator(model_name: str, device_string: str, torch_dtype: Union[str, torch.dtype]) -> pipeline:
    """Load and return a machine translator (for one or more natural languages to another).

    `model_name`: HuggingFace model name. Auto-downloaded on first use.
                  Try e.g. "Helsinki-NLP/opus-mt-tc-big-en-fi" for English to Finnish.

                  See:
                      https://huggingface.co/tasks/translation

    `device_string`: as in Torch, e.g. "cpu", "cuda", or "cuda:0".
    `torch_dtype`: e.g. "float32", "float16" (on GPU), or `torch.float16` (same thing).

    If the specified model is already loaded on the same device (identified by `device_string`),
    with the same dtype, then return the already-loaded instance.

    NOTE: To use `translate`, you also need a spaCy NLP pipeline; see `load_spacy_pipeline`.
    """
    cache_key = (model_name, device_string, str(torch_dtype))
    if cache_key in _translators:
        logger.info(f"load_translator: '{model_name}' (with dtype '{str(torch_dtype)}') is already loaded on device '{device_string}', returning it.")
        return _translators[cache_key]
    logger.info(f"load_translator: Loading '{model_name}' (with dtype '{str(torch_dtype)}') on device '{device_string}'.")

    try:
        device = torch.device(device_string)
        translator = pipeline("translation",
                              model=model_name,
                              device=device,
                              torch_dtype=torch_dtype)
    except RuntimeError as exc:
        logger.warning(f"load_translator: exception while loading translator (will try again in CPU mode): {type(exc)}: {exc}")
        try:
            device_string = "cpu"
            torch_dtype = "float32"
            cache_key = (model_name, device_string, str(torch_dtype))
            translator = pipeline("translation",
                                  model=model_name,
                                  device=device,
                                  torch_dtype=torch_dtype)
        except RuntimeError as exc:
            logger.warning(f"load_translator: failed to load translator: {type(exc)}: {exc}")
            raise
    logger.info(f"load_translator: model '{model_name}' context window is {translator.tokenizer.model_max_length} tokens.")
    logger.info(f"load_translator: Loaded model '{model_name}' (with dtype '{str(torch_dtype)}') on device '{device_string}'.")
    _translators[cache_key] = translator
    return translator

# # TODO: Add support for "translation prefix" like `summarize` has? This would e.g. allow using t5-base as a translator.
# def _translate_chunk(translator: pipeline, text: str) -> str:
#     """Internal function. Translate a piece of text that fits into the translation model's context window.
#
#     If the text does not fit, raises `IndexError`. See `_translate_chunked` to handle arbitrary length texts.
#     """
#     tokens = translator.tokenizer.tokenize(text)
#     length_in_tokens = len(tokens)
#     logger.info(f"_translate_chunk: Input text length is {len(text)} characters, {length_in_tokens} tokens.")
#
#     if length_in_tokens > translator.tokenizer.model_max_length:
#         logger.info(f"_translate_chunk: Text to be translated does not fit into model's context window (text length {len(text)} characters, {length_in_tokens} tokens; model limit {translator.tokenizer.model_max_length} tokens).")
#         raise IndexError  # and let `_translate_chunked` handle it
#
#     output = translator(text)
#     print(output)
#     translation = output[0]["translation_text"]
#     return translation

def _translate_chunked(translator: pipeline, nlp_pipe, text: str) -> str:
    """Internal function. Translate a text that may require chunking before it fits into the translation model's context window."""
    # Translate one sentence at a time (reliable and keeps the chunks short).
    # TODO: Can sometimes give bad results:
    #   - If the sentence splitting is bad (e.g. scientific abstract with [ABC1992] inline citations can confuse spaCy)
    #   - If the sentences depend on larger context to make sense.
    with nlp_pipe.select_pipes(enable=['tok2vec', "parser", "senter"]):  # process faster by enabling only needed modules; https://stackoverflow.com/a/74907505
        doc = nlp_pipe(text)
    sents = list(doc.sents)
    outputs = translator([str(sent).strip() for sent in sents])
    translations = [output["translation_text"] for output in outputs]
    return " ".join(translations)

    # # TODO: This currently misses some sentences for multi-sentence inputs.
    # try:
    #     return _translate_chunk(translator, text)
    # except IndexError:
    #     logger.info(f"_translate_chunked: input text (length {len(text)} characters) is long; cutting text in half at a sentence boundary and translating the halves separately.")
    #
    #     with nlp_pipe.select_pipes(enable=['tok2vec', "parser", "senter"]):  # process faster by enabling only needed modules; https://stackoverflow.com/a/74907505
    #         doc = nlp_pipe(text)
    #     sents = list(doc.sents)
    #     mid = len(sents) // 2
    #     firsthalf = " ".join(str(sent).strip() for sent in sents[:mid])
    #     secondhalf = " ".join(str(sent).strip() for sent in sents[mid:])
    #     return " ".join(
    #         [_translate_chunked(translator, nlp_pipe, firsthalf),
    #          _translate_chunked(translator, nlp_pipe, secondhalf)]
    #     )

def translate(translator: pipeline, nlp_pipe, text: Union[str, List[str]]) -> Union[str, List[str]]:
    """Translate `text` to another natural language.

    `translator`: return value of `load_translator`
    `nlp_pipe`: return value of `load_spacy_pipeline` (used for sentence-boundary splitting during chunking)
    `text`: one (str) or more (List[str]) texts to translate

    Returns `str` (one input) or `list` of `str` (more inputs).
    """
    def doit(text: str) -> str:
        return _translate_chunked(translator, nlp_pipe, text)
    if isinstance(text, list):
        output_text = [doit(item) for item in text]
    else:  # str
        output_text = doit(text)
    return output_text

# --------------------------------------------------------------------------------
# Frequency analysis

def count_frequencies(tokens: Union[List[spacy.tokens.token.Token],
                                    List[List[spacy.tokens.token.Token]]],
                      lemmatize: bool = True,
                      stopwords: Container = default_stopwords,
                      min_length: int = 3,
                      min_occurrences: int = 2) -> Dict[str, int]:
    """Count the number of occurrences of each word in document(s).

    The result will be aggregated over the whole input, resulting in one big dict.

    `tokens`: Tokenized text content of one or more documents, in spaCy format.

    `lemmatize`: Whether to lemmatize words first.

                 If `True`, all words are lemmatized first before further analysis.
                 This is usually the right thing to do.

                 If `False`, the words are processed as is. Each word form counts separately.

                 When `lemmatize=True`, the results will be in lemmatized form. E.g. then the words
                 "process", "processing", and "process" all lemmatize into "process", and thus
                 they will all count as occurrences of "process".

                 Note lemmatization may sometimes produce silly results. E.g. the publisher name
                 "Elsevier" lemmatizes into "Elsevi", because the name looks like the comparative
                 form of an adjective. Similarly "Springer" lemmatizes into "Spring".

    `stopwords`: See `default_stopwords`.

                 You can also provide your own. The stopwords must be in lowercase.

                 If you don't want to stopword anything, use `stopwords=set()`.

    `min_length`: Minimum length to accept a word, in characters (Unicode codepoints).
                  Words shorter than this are ignored.

    `min_occurrences`: How many times a word must appear, across the whole corpus,
                       before it is accepted. Words rarer than this are dropped
                       from the final results.

    Returns a dict `{word0: count0, ...}`, ordered by count (descending, i.e. more common words first).

    Example::

        # Here each string is the fulltext of one document
        documents = ["blah blah",
                     "foo bar baz",
                     ...]

        nlp = load_spacy_pipeline("en_core_web_sm", "cuda:0")
        tokenss = list(nlp.pipe(documents))

        all_frequencies = count_frequencies(tokenss)  # across all documents
        per_document_frequencies = [count_frequencies(tokens) for tokens in tokenss]  # separately for each document
    """
    # Apply standard tricks from information retrieval:
    #   - Drop useless stopwords ("the", "of", ...), which typically dominate the word frequency distribution
    #   - Cut the (long!) tail of the distribution
    # => Obtain words that appear at an intermediate frequency (not too common, not too rare).
    #    These usually describe the text usefully.

    # # Old implementation using NLTK
    # stopwords = nltk.corpus.stopwords.words("english")
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
    #                 if lemmatize:
    #                     x = lemmatizer.lemmatize(x)
    #                 x = x.lower()
    #                 if len(x) < min_length or x in stopwords:
    #                     continue
    #                 out.append(x)
    #             return out
    #         return collections.Counter(filter_tokens(nltk.word_tokenize(text)))
    #     else:  # List[str]
    #         word_counts = collections.Counter()
    #         for item in text:
    #             word_counts.update(extract_word_counts(item))
    #         return word_counts

    # New implementation using spaCy
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

            if lemmatize:
                x = x.lemma_
            x = x.lower()

            if len(x) < min_length or x.lower() in stopwords:
                continue

            out.append(x)
        return out
    def extract_word_counts(things: Union[List[spacy.tokens.token.Token],
                                          List[List[spacy.tokens.token.Token]]]) -> collections.Counter:
        """Count the number of occurrences of each unique word in a document or documents.

        The result is aggregated over the whole input.

        A document is a list of spaCy tokens.

        `things`: list of documents, or a single document.
        """
        if not things:
            return collections.Counter()
        if isinstance(things[0], spacy.tokens.token.Token):  # list of tokens, i.e. single document
            return collections.Counter(filter_tokens(things))
        else:  # list of list of tokens, i.e. multiple documents
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
        representative_words = {x: word_counts[x] for x in word_counts if word_counts[x] >= min_occurrences}
        return representative_words

    frequencies = trim_word_counts(extract_word_counts(tokens))
    frequencies = dict(sorted(frequencies.items(),
                              key=operator.itemgetter(1),
                              reverse=True))
    return frequencies


def detect_named_entities(tokens: Union[List[spacy.tokens.token.Token],
                                        List[List[spacy.tokens.token.Token]]],
                          stopwords: Container = default_stopwords) -> Dict[str, int]:
    """Named entity recognition (NER).

    Same input format as in `count_frequencies`, which see.

    Returns a dict `{entity0: count0, ...}`, ordered by count (descending, i.e. more common entities first).
    """
    def _ner_impl(tokens):
        if not tokens:
            return collections.Counter()
        if isinstance(tokens[0], spacy.tokens.token.Token):  # list of tokens, i.e. single document
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
            ents = collections.Counter()
            for ent in tokens.ents:
                if ent.label_ in ("CARDINAL", "DATE", "MONEY", "QUANTITY", "PERCENT", "TIME"):
                    continue
                if ent.text in stopwords:
                    continue
                # print(f"Entity: {ent.text}, Label: {ent.label_}")  # DEBUG
                ents[ent.text] += 1
            return ents
        else:  # list of list of tokens, i.e. multiple documents
            ents = collections.Counter()
            for sublist in tokens:
                ents.update(_ner_impl(sublist))
            return ents
    named_entities = _ner_impl(tokens)
    named_entities = dict(sorted(named_entities.items(),
                                 key=operator.itemgetter(1),
                                 reverse=True))
    return named_entities


def suggest_keywords(per_document_frequencies: List[Dict[str, int]],
                     corpus_frequencies: Optional[Dict[str, int]] = None,
                     threshold_fraction: float = 0.1,
                     max_keywords: Optional[int] = None) -> List[List[str]]:
    """Data-adaptively suggest per-document keywords that distinguish between documents of the corpus.

    `per_document_frequencies`: list of outputs of `count_frequencies`, one element per document.
                                See the example there.

    `corpus_frequencies`: output of `count_frequencies`, across the whole corpus.

                          The per-document data is analyzed against the corpus data. For each document,
                          any words that do not appear in the corpus data will be ignored, regardless
                          of their frequencies in the individual document.

                          If `corpus_frequencies` is not supplied, one will be automatically aggregated
                          from the per-document frequency data.

                          Supplying the corpus data separately can be useful e.g. if you need to filter it,
                          or when looking at a subset of documents from a larger corpus (while analyzing
                          keywords against that larger corpus).

    `threshold_fraction`: IMPORTANT. The source of the keyword suggestion magic. See details below.

    `max_keywords`: In the final results, return at most this many keywords for each document.

                    If not specified, all words remaining after threshold filtering, in that document's
                    entry in `per_document_frequencies`, will be treated as keywords.

    Returns `[[doc0_kw0, doc0_kw1, ...], [doc1_kw0, ...], ...]`, where each list is ordered by count
    (descending, i.e. more common words first).

    If you want the frequencies, each of the words is a key to your `per_document_frequencies`,
    so you can do this::

        per_document_keyword_frequencies = []
        for words, frequencies in zip(suggested_keywords, per_document_frequencies):
            per_document_keyword_frequencies.append({word: frequencies[word] for word in words})

    **Details**

    This function aims to pick, from the given frequency data, words that distinguish each document
    from the other documents in the corpus. Words that are uselessly common across different documents
    are dropped.

    There's a fine balance here, drawing on the old NLP observation that words with intermediate frequencies
    best describe a dataset:

      - Filtering out words present in *all* other documents drops only a few of the most common words,
        leaving uselessly large overlaps between the suggested keyword sets. This happens even after we
        account for stopwords (see `count_frequencies`), because a typical corpus discusses a single
        broad umbrella topic.

      - Filtering out words present in *any* other document leaves only noise. Documents that talk about
        the same topic tend to use some of the same words.

      - Hence, enter the `threshold_fraction` parameter. If at least that fraction of all documents have
        a particular word, then that word is considered uselessly common for the purposes of distinguishing
        between documents (within this corpus), and dropped.

    The default is to ignore words that appear in at least 10% of all documents. But since it is possible
    that there are just a few documents, the final formula that allows also such edge cases is::

        threshold_n = max(min(5, n_documents), math.ceil(threshold_fraction * n_documents))

    Any word that appears in `threshold_n` or more documents is ignored.
    """
    # Collect word frequencies over whole corpus, if not supplied, from the frequencies over each document.
    user_supplied_corpus_frequencies = True
    if corpus_frequencies is None:
        user_supplied_corpus_frequencies = False
        corpus_frequencies = collections.Counter()
        for frequencies in per_document_frequencies:
            corpus_frequencies.update(frequencies)

    # Find how many document(s) each word in the per-document frequencies appears in.
    n_documents = len(per_document_frequencies)
    threshold_n = max(min(5, n_documents), math.ceil(threshold_fraction * n_documents))
    document_counts_by_word = collections.Counter()
    for frequencies in per_document_frequencies:
        for word in frequencies.keys():
            document_counts_by_word[word] += 1  # found one more document that has this word
    excessively_common_words = {word for word, count in document_counts_by_word.items() if count >= threshold_n}
    logger.info(f"suggest_keywords: Found {len(excessively_common_words)} words shared between {threshold_n} documents out of {n_documents}. Threshold fraction = {threshold_fraction:0.2g}. Total unique words = {len(corpus_frequencies)}.")

    # # Detect words common to *all* documents - not helpful.
    # words_common_to_all_documents = set(per_document_frequencies[0].keys())
    # for kws in per_document_frequencies[1:]:
    #     kws = set(kws.keys())
    #     words_common_to_all_documents.intersection_update(kws)

    # Keep the highest-frequency words detected in each document. Hopefully this will compactly describe what the document is about.
    logger.info("suggest_keywords: Ranking and filtering results...")
    per_document_keywords = []
    for document_index, frequencies in enumerate(per_document_frequencies):
        words = set(frequencies.keys())

        # # Drop words present in *any* other document - not helpful.
        # other_documents_words = per_document_frequencies[:document_index] + per_document_frequencies[document_index + 1:]
        # for other_words in other_documents_words:
        #     words = words.difference(other_words)
        # words = list(words)

        # Drop words common with too many other documents. This is helpful.
        # NOTE: We build the `excessively_common_words` set first (further above), and filter against the complete set, to treat all documents symmetrically.
        #       We don't want filler words to end up in the first document's keywords just because it was processed first.
        words = list(words.difference(excessively_common_words))

        # Drop words not in the corpus data.
        # If the corpus data was autogenerated, all per-document words appear there, so we only need to do this if the corpus data was user-supplied.
        if user_supplied_corpus_frequencies:  # if the corpus data was autogenerated, all per-document words appear there, so we only need to check this if the corpus data was user-supplied.
            words = [word for word in words if word in corpus_frequencies.keys()]

        # Sort by frequency *in this document*, more common first.
        filtered_frequencies = {word: frequencies[word] for word in words}
        filtered_frequencies = list(sorted(filtered_frequencies.items(),
                                           key=operator.itemgetter(1),
                                           reverse=True))
        words = [word for word, count in filtered_frequencies]

        # # Sort by frequency across whole corpus, more common first.
        # words = list(sorted(words,
        #                     key=lambda word: corpus_frequencies.get(word, 0),
        #                     reverse=True))

        # If we're cutting, keep the words with the highest number of occurrences.
        if max_keywords is not None:
            words = words[:max_keywords]

        per_document_keywords.append(words)
    return per_document_keywords

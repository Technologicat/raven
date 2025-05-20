"""Initialization for NLP (natural language processing) backends.

The main features are model caching (load only one copy of each) and device management (which device to load on).
"""

__all__ = {"default_stopwords", "extended_stopwords",
           "load_pipeline",
           "load_embedding_model"}

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import collections
import copy
from typing import Container, Dict, List, Union

# NLP
from sentence_transformers import SentenceTransformer
import spacy

from . import config
# --------------------------------------------------------------------------------

def _load_stopwords() -> List[str]:
    """Load a default list of stopwords (English default list from spaCy)."""
    from spacy.lang.en import English
    nlp_en = English()
    stopwords = nlp_en.Defaults.stop_words
    return stopwords

# We apply stopwords case-insensitively, so lowercase them now.
default_stopwords = set(x.lower() for x in _load_stopwords())

extended_stopwords = copy.copy(default_stopwords)
extended_stopwords.update(x.lower() for x in config.custom_stopwords)

# --------------------------------------------------------------------------------

_nlp_pipelines = {}
def load_pipeline(model_name: str):
    """Load and return the spaCy NLP pipeline.

    If the specified model is already loaded, return the already-loaded instance.
    """
    if model_name in _nlp_pipelines:
        logger.info(f"load_pipeline: '{model_name}' is already loaded, returning it.")
        return _nlp_pipelines[model_name]
    logger.info(f"load_pipeline: Loading '{model_name}'.")

    device_string = config.devices["nlp"]["device_string"]
    if device_string.startswith("cuda"):
        if ":" in device_string:
            _, gpu_id = device_string.split(":")
            gpu_id = int(gpu_id)
        else:
            gpu_id = 0

        try:
            spacy.require_gpu(gpu_id=gpu_id)
            logger.info("load_pipeline: spaCy will run on GPU (if available).")
        except Exception as exc:
            logger.warning(f"load_pipeline: exception while enabling GPU for spaCy: {type(exc)}: {exc}")
            spacy.require_cpu()
            logger.info("load_pipeline: spaCy will run on CPU.")
    else:
        spacy.require_cpu()
        logger.info("load_pipeline: spaCy will run on CPU.")

    try:
        nlp_pipeline = spacy.load(model_name)
    except OSError:
        # https://stackoverflow.com/questions/62728854/how-to-place-spacy-en-core-web-md-model-in-python-package
        logger.info(f"load_pipeline: Downloading spaCy model '{model_name}' (don't worry, this will only happen once)...")
        from spacy.cli import download
        download(model_name)
        nlp_pipeline = spacy.load(model_name)

    logger.info(f"load_pipeline: Loaded spaCy model '{model_name}'.")
    _nlp_pipelines[model_name] = nlp_pipeline
    return nlp_pipeline

# Cache the embedding models (to load only one copy of each model)
_embedding_models = {}
def load_embedding_model(model_name: str):
    """Load and return the embedding model (for vector storage).

    If the specified model is already loaded, return the already-loaded instance.
    """
    if model_name in _embedding_models:
        logger.info(f"load_embedding_model: '{model_name}' is already loaded, returning it.")
        return _embedding_models[model_name]
    logger.info(f"load_embedding_model: Loading '{model_name}'.")

    try:
        device_string = config.devices["embeddings"]["device_string"]
        embedding_model = SentenceTransformer(model_name, device=device_string)
    except RuntimeError as exc:
        logger.warning(f"load_embedding_model: exception while loading SentenceTransformer (will try again in CPU mode): {type(exc)}: {exc}")
        try:
            device_string = "cpu"
            embedding_model = SentenceTransformer(model_name, device=device_string)
        except RuntimeError as exc:
            logger.warning(f"load_embedding_model: failed to load SentenceTransformer: {type(exc)}: {exc}")
            raise
    logger.info(f"load_embedding_model: Loaded model '{model_name}' on device '{device_string}'.")
    _embedding_models[model_name] = embedding_model
    return embedding_model

# --------------------------------------------------------------------------------

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

    `stopwords`: See `default_stopwords` and `extended_stopwords`.
                 The stopwords must be in lowercase.

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

        nlp = load_pipeline("en_core_web_sm")
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
    frequencies = dict(sorted(frequencies.items(), key=lambda kv: -kv[1]))
    return frequencies


def ner(tokens: Union[List[spacy.tokens.token.Token],
                      List[List[spacy.tokens.token.Token]]],
        stopwords: Container = default_stopwords) -> Dict[str, int]:
    """Named entity recognition.

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
    named_entities = dict(sorted(named_entities.items(), key=lambda kv: -kv[1]))
    return named_entities

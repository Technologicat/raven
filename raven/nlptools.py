"""NLP (natural language processing) tools.

- Stopword set management.
- Backend loading.
  - Model caching (load only one copy of each model in the same process).
  - Device management (which device to load on), with automatic CPU fallback if loading on GPU fails.
- Frequency analysis tools.
"""

__all__ = {"default_stopwords", "extended_stopwords",
           "load_pipeline",
           "load_embedding_model",
           "count_frequencies", "detect_named_entities",
           "suggest_keywords"}

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import collections
import copy
import math
import operator
from typing import Container, Dict, List, Optional, Union

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

# The extended stopword set (with custom additional stopwords tuned for English-language scientific text) is used by Raven's BibTeX importer (`preprocess.py`).
extended_stopwords = copy.copy(default_stopwords)
extended_stopwords.update(x.lower() for x in config.custom_stopwords)

# --------------------------------------------------------------------------------

_pipelines = {}
def load_pipeline(model_name: str):
    """Load and return the spaCy NLP pipeline.

    If the specified model is already loaded, return the already-loaded instance.
    """
    if model_name in _pipelines:
        logger.info(f"load_pipeline: '{model_name}' is already loaded, returning it.")
        return _pipelines[model_name]
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
            logger.info("load_pipeline: spaCy will run on GPU (if possible).")
        except Exception as exc:
            logger.warning(f"load_pipeline: exception while enabling GPU for spaCy: {type(exc)}: {exc}")
            spacy.require_cpu()
            logger.info("load_pipeline: spaCy will run on CPU.")
    else:
        spacy.require_cpu()
        logger.info("load_pipeline: spaCy will run on CPU.")

    try:
        nlp = spacy.load(model_name)
    except OSError:
        # https://stackoverflow.com/questions/62728854/how-to-place-spacy-en-core-web-md-model-in-python-package
        logger.info(f"load_pipeline: Downloading spaCy model '{model_name}' (don't worry, this will only happen once)...")
        from spacy.cli import download
        download(model_name)
        nlp = spacy.load(model_name)

    logger.info(f"load_pipeline: Loaded spaCy model '{model_name}'.")
    _pipelines[model_name] = nlp
    return nlp

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

"""Tests for `raven.common.nlptools` — runs the full ML stack in-process.

Marked as `ml`-tier: skipped on CI (the `-m "not ml"` filter in the workflows)
and further guarded by `importorskip` so the file also cleanly skips in any
environment without the heavy ML deps.

All fixtures load real small models on CPU once per session, so the whole file
amortizes model load cost across many cheap assertions.
"""

import pytest

# Skip the whole file when any of the heavy deps are missing (e.g. CI).
pytest.importorskip("spacy")
pytest.importorskip("flair")
pytest.importorskip("dehyphen")
pytest.importorskip("sentence_transformers")
pytest.importorskip("transformers")

pytestmark = pytest.mark.ml

from raven.common import nlptools  # noqa: E402 -- must come after the importorskip guard above

# Model names: the canonical defaults from `raven/server/config.py`, except the
# embedder where we pick the lighter `-m` variant (440 MB) over `-l` (1.3 GB)
# to keep the test suite snappy — both are listed side-by-side in config.py.
SPACY_MODEL = "en_core_web_sm"
CLASSIFIER_MODEL = "joeddav/distilbert-base-uncased-go-emotions-student"
DEHYPHEN_MODEL = "multi"
EMBED_MODEL = "Snowflake/snowflake-arctic-embed-m"
TRANSLATOR_MODEL = "Helsinki-NLP/opus-mt-tc-big-en-fi"

# CPU keeps tests reproducible across dev machines regardless of GPU state, and
# avoids contention with any other GPU work happening in parallel on the box.
DEVICE = "cpu"
DTYPE = "float32"

# --------------------------------------------------------------------------------
# Session-scoped fixtures: load each model once, reuse across all tests.

@pytest.fixture(scope="session")
def spacy_pipe():
    return nlptools.load_spacy_pipeline(SPACY_MODEL, DEVICE)

@pytest.fixture(scope="session")
def classifier():
    return nlptools.load_classifier(CLASSIFIER_MODEL, DEVICE, DTYPE)

@pytest.fixture(scope="session")
def dehyphenator():
    return nlptools.load_dehyphenator(DEHYPHEN_MODEL, DEVICE)

@pytest.fixture(scope="session")
def embedder():
    return nlptools.load_embedder(EMBED_MODEL, DEVICE, DTYPE)

@pytest.fixture(scope="session")
def translator():
    return nlptools.load_translator(TRANSLATOR_MODEL, DEVICE, DTYPE,
                                    source_lang="en", target_lang="fi")

# --------------------------------------------------------------------------------
# Stopwords

def test_default_stopwords_is_lowercase_set():
    assert isinstance(nlptools.default_stopwords, set)
    assert len(nlptools.default_stopwords) > 0
    # Canonical English stopwords should be present.
    assert "the" in nlptools.default_stopwords
    assert "and" in nlptools.default_stopwords
    # Nothing uppercase — they are normalized before storage.
    assert all(w == w.lower() for w in nlptools.default_stopwords)

# --------------------------------------------------------------------------------
# spaCy pipeline: loader caching, analyze, serializers

def test_load_spacy_pipeline_caches(spacy_pipe):
    again = nlptools.load_spacy_pipeline(SPACY_MODEL, DEVICE)
    assert again is spacy_pipe  # identity, not just equality

def test_spacy_analyze_single_str(spacy_pipe):
    docs = nlptools.spacy_analyze(spacy_pipe, "The quick brown fox jumps over the lazy dog.")
    assert isinstance(docs, list)
    assert len(docs) == 1
    tokens = [t.text for t in docs[0]]
    assert "fox" in tokens
    assert "dog" in tokens

def test_spacy_analyze_multiple(spacy_pipe):
    docs = nlptools.spacy_analyze(spacy_pipe, ["Hello world.", "Goodbye world."])
    assert len(docs) == 2

def test_spacy_analyze_with_pipes(spacy_pipe):
    text = "The first sentence is here. This is the second one. And a third."
    docs = nlptools.spacy_analyze(spacy_pipe, text, pipes=["tok2vec", "parser", "senter"])
    assert len(docs) == 1
    sents = list(docs[0].sents)
    assert len(sents) == 3

def test_serialize_deserialize_spacy_docs_single(spacy_pipe):
    docs = nlptools.spacy_analyze(spacy_pipe, "Alice met Bob in Paris.")
    data = nlptools.serialize_spacy_docs(docs[0])  # single-Doc path
    assert isinstance(data, list) and len(data) == 1
    assert data[0]["lang"] == "en"
    assert "doc" in data[0]  # pristine spaCy Doc.to_json output
    assert "vectors" not in data[0]  # default: no vectors on the wire
    recovered = nlptools.deserialize_spacy_docs(data)
    assert len(recovered) == 1
    assert [t.text for t in recovered[0]] == [t.text for t in docs[0]]

def test_serialize_deserialize_spacy_docs_multiple(spacy_pipe):
    docs = nlptools.spacy_analyze(spacy_pipe, ["First text.", "Second text."])
    data = nlptools.serialize_spacy_docs(docs)
    assert len(data) == 2
    recovered = nlptools.deserialize_spacy_docs(data)
    assert len(recovered) == 2
    for orig, rec in zip(docs, recovered):
        assert [t.text for t in orig] == [t.text for t in rec]
        assert [t.lemma_ for t in orig] == [t.lemma_ for t in rec]
        assert [t.pos_ for t in orig] == [t.pos_ for t in rec]

def test_deserialize_spacy_docs_multilingual():
    """Deserializer handles multiple languages in a single batch.

    Forward-compat check: the wire format puts `lang` on each item so that a future
    server configuration could load multiple pipelines (e.g. English + Finnish) and
    route texts per language within a single request. No live model download needed
    — `spacy.blank(lang)` works for any language code spaCy recognizes.

    Also exercises the vocab cache path in `deserialize_spacy_docs` (same lang reused
    across items should `spacy.blank()` only once).
    """
    import spacy
    items = [
        {"lang": "en",
         "doc": spacy.blank("en")("Hello world.").to_json()},
        {"lang": "fi",
         "doc": spacy.blank("fi")("Hei maailma.").to_json()},
        {"lang": "en",  # same lang again — should hit the vocab cache
         "doc": spacy.blank("en")("Second English doc.").to_json()},
    ]
    recovered = nlptools.deserialize_spacy_docs(items)
    assert len(recovered) == 3
    assert recovered[0].lang_ == "en"
    assert recovered[1].lang_ == "fi"
    assert recovered[2].lang_ == "en"
    assert [t.text for t in recovered[0]] == ["Hello", "world", "."]
    assert [t.text for t in recovered[1]] == ["Hei", "maailma", "."]
    assert [t.text for t in recovered[2]] == ["Second", "English", "doc", "."]

def test_serialize_spacy_docs_doc_json_schema(spacy_pipe):
    """Contract: `item["doc"]` is pristine `Doc.to_json()` output.

    Locks in the documented wire-format promise so that non-Python clients (e.g. a
    future JS avatar frontend) can rely on the standard spaCy JSON schema.
    """
    docs = nlptools.spacy_analyze(spacy_pipe, "Alice met Bob in Paris.")
    data = nlptools.serialize_spacy_docs(docs)
    doc_json = data[0]["doc"]
    # Top-level keys match spaCy's Doc.to_json schema.
    assert "text" in doc_json
    assert "tokens" in doc_json
    assert "ents" in doc_json
    assert "sents" in doc_json
    # Per-token dict keys match spaCy's schema (no extra or renamed fields).
    for token in doc_json["tokens"]:
        assert set(token.keys()) >= {"id", "start", "end", "tag", "pos", "morph", "lemma", "dep", "head"}

def test_serialize_deserialize_spacy_docs_with_vectors(spacy_pipe):
    """Round-trip `doc.tensor` / `token.vector` when `with_vectors=True`.

    `en_core_web_sm` has no vocab vectors (dim 0), but `tok2vec` populates `doc.tensor`;
    `token.vector` falls through to `doc.tensor[i]` when vocab has none.
    """
    import numpy as np
    docs = nlptools.spacy_analyze(spacy_pipe, "Alice met Bob in Paris.")
    data = nlptools.serialize_spacy_docs(docs, with_vectors=True)
    assert "vectors" in data[0]
    assert data[0]["vectors"]["dim"] == docs[0].tensor.shape[1]
    recovered = nlptools.deserialize_spacy_docs(data)
    # Tensor round-trips bit-for-bit (float32 → base64 → float32 is lossless).
    assert np.array_equal(recovered[0].tensor, docs[0].tensor)
    # `token.vector` falls through to `doc.tensor[i]` on the blank-vocab reconstructed Doc.
    for orig_tok, rec_tok in zip(docs[0], recovered[0]):
        assert np.array_equal(orig_tok.vector, rec_tok.vector)

def test_serialize_deserialize_spacy_pipeline(spacy_pipe):
    config_str, data_bytes = nlptools.serialize_spacy_pipeline(spacy_pipe)
    assert isinstance(config_str, str)
    assert isinstance(data_bytes, bytes)
    nlp2 = nlptools.deserialize_spacy_pipeline(config_str, data_bytes)
    # Should produce analyses compatible in shape with the original.
    orig = list(spacy_pipe.pipe(["The cat sat on the mat."]))[0]
    copy = list(nlp2.pipe(["The cat sat on the mat."]))[0]
    assert [t.text for t in copy] == [t.text for t in orig]

# --------------------------------------------------------------------------------
# Classifier

def test_load_classifier_caches(classifier):
    again = nlptools.load_classifier(CLASSIFIER_MODEL, DEVICE, DTYPE)
    assert again is classifier

def test_classify_returns_sorted_descending(classifier):
    output = nlptools.classify(classifier, "I am so happy and excited about this wonderful news!")
    assert isinstance(output, list)
    assert len(output) > 0
    # Every entry has label + score, and the list is sorted descending by score.
    for entry in output:
        assert set(entry.keys()) >= {"label", "score"}
    scores = [entry["score"] for entry in output]
    assert scores == sorted(scores, reverse=True)
    # The top prediction for clearly joyful text should be in the positive cluster.
    assert output[0]["label"] in {"joy", "excitement", "admiration", "love", "amusement", "pride"}

# --------------------------------------------------------------------------------
# Dehyphenator

def test_load_dehyphenator_caches(dehyphenator):
    again = nlptools.load_dehyphenator(DEHYPHEN_MODEL, DEVICE)
    assert again is dehyphenator

def test_dehyphenate_single_char_passthrough(dehyphenator):
    # Early exit path: single character is returned as-is (guards against `dehyphen` crashing).
    assert nlptools.dehyphenate(dehyphenator, "x") == "x"

def test_dehyphenate_single_line_passthrough(dehyphenator):
    # Early exit path: <=1 newline means there's nothing to fix.
    s = "A single line of text."
    assert nlptools.dehyphenate(dehyphenator, s) == s

def test_dehyphenate_list_input(dehyphenator):
    # List path maps the helper over each item; single-char elements hit the passthrough.
    out = nlptools.dehyphenate(dehyphenator, ["x", "y"])
    assert out == ["x", "y"]

def test_dehyphenate_fixes_broken_word(dehyphenator):
    # A hyphen-broken paragraph should get reassembled. We don't assert an exact
    # string (the model's perplexity-based joiner has minor leeway), just that
    # "evaluation" (the fixed word) ends up present as a contiguous token.
    broken = ("The system performs automatic eval-\n"
              "uation of the incoming documents.\n"
              "It then writes the results to disk.")
    out = nlptools.dehyphenate(dehyphenator, broken)
    assert isinstance(out, str)
    assert "evaluation" in out

# --------------------------------------------------------------------------------
# Embedder

def test_load_embedder_caches(embedder):
    again = nlptools.load_embedder(EMBED_MODEL, DEVICE, DTYPE)
    assert again is embedder

def test_embed_sentences_single(embedder):
    vec = nlptools.embed_sentences(embedder, "A quick brown fox.")
    assert isinstance(vec, list)
    assert len(vec) > 0
    assert all(isinstance(x, float) for x in vec)
    # Normalized embeddings have unit L2 norm.
    norm = sum(x * x for x in vec) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-3)

def test_embed_sentences_multiple(embedder):
    vecs = nlptools.embed_sentences(embedder, ["Hello world.", "Goodbye world."])
    assert isinstance(vecs, list)
    assert len(vecs) == 2
    assert all(isinstance(v, list) for v in vecs)
    # Both entries should have the same dimensionality.
    assert len(vecs[0]) == len(vecs[1]) > 0

# --------------------------------------------------------------------------------
# Translator

def test_load_translator_caches(translator):
    again = nlptools.load_translator(TRANSLATOR_MODEL, DEVICE, DTYPE,
                                     source_lang="en", target_lang="fi")
    assert again is translator

def test_translate_single(translator, spacy_pipe):
    out = nlptools.translate(translator, spacy_pipe, "The cat sits on the mat.")
    assert isinstance(out, str)
    assert len(out) > 0
    # Heuristic: Finnish output shouldn't be identical to the English input.
    assert out.lower() != "the cat sits on the mat."

def test_translate_list(translator, spacy_pipe):
    outs = nlptools.translate(translator, spacy_pipe,
                              ["Hello, world.", "The sun is shining today."])
    assert isinstance(outs, list)
    assert len(outs) == 2
    assert all(isinstance(s, str) and len(s) > 0 for s in outs)

def test_translate_chunked_multisentence(translator, spacy_pipe):
    # Exercises the sentence-splitting chunker path directly.
    text = "The first point is clear. The second follows from it. A third and final point closes."
    out = nlptools._translate_chunked(translator, spacy_pipe, text)
    assert isinstance(out, str)
    # Should contain three joined fragments (roughly — we don't count Finnish sentences, just length).
    assert len(out) > 0

# --------------------------------------------------------------------------------
# Frequency / NER / keyword suggestion

@pytest.fixture(scope="module")
def sample_docs(spacy_pipe):
    texts = [
        "Finite element methods discretize partial differential equations. "
        "The finite element approach is widely used in structural analysis.",
        "Machine learning algorithms train on large datasets. "
        "Neural networks power modern machine learning systems.",
        "Climate models simulate atmospheric circulation. "
        "The climate system exhibits strong nonlinear dynamics.",
    ]
    return nlptools.spacy_analyze(spacy_pipe, texts)

def test_count_frequencies_basic(sample_docs):
    freqs = nlptools.count_frequencies(sample_docs[0], min_occurrences=1, min_length=3)
    # Lemmatized form of "methods"/"method" should collapse to "method".
    assert "method" in freqs or "methods" in freqs
    # Descending order.
    counts = list(freqs.values())
    assert counts == sorted(counts, reverse=True)

def test_count_frequencies_lemmatize_flag(sample_docs):
    freqs_lemmatized = nlptools.count_frequencies(sample_docs[0], lemmatize=True, min_occurrences=1, min_length=3)
    freqs_raw = nlptools.count_frequencies(sample_docs[0], lemmatize=False, min_occurrences=1, min_length=3)
    # Raw (non-lemmatized) form can surface plurals separately; at least one word from
    # the lemmatized set should not appear literally in the raw-form counts.
    assert set(freqs_lemmatized.keys()) != set(freqs_raw.keys())

def test_count_frequencies_stopwords(sample_docs):
    # A custom stopword list drops the listed words.
    freqs = nlptools.count_frequencies(sample_docs[0],
                                       stopwords={"finite", "element"},
                                       min_occurrences=1,
                                       min_length=3)
    assert "finite" not in freqs
    assert "element" not in freqs

def test_count_frequencies_min_length(sample_docs):
    freqs = nlptools.count_frequencies(sample_docs[0], min_length=20, min_occurrences=1)
    assert freqs == {}  # no 20-character words

def test_count_frequencies_min_occurrences(sample_docs):
    # Threshold of 99 filters out everything.
    freqs = nlptools.count_frequencies(sample_docs[0], min_occurrences=99)
    assert freqs == {}

def test_count_frequencies_list_of_docs(sample_docs):
    # Aggregates over all docs.
    freqs = nlptools.count_frequencies(sample_docs, min_occurrences=1, min_length=3)
    assert len(freqs) > 0

def test_count_frequencies_empty():
    assert nlptools.count_frequencies([]) == {}

def test_detect_named_entities(spacy_pipe):
    # Sentences packed with clear entities.
    docs = nlptools.spacy_analyze(spacy_pipe,
                                  "Alice met Bob in Paris. Later, Alice visited London with Charlie.")
    ents = nlptools.detect_named_entities(docs[0])
    assert isinstance(ents, dict)
    # At least some of the canonical names should come through.
    found = {e.lower() for e in ents.keys()}
    assert {"alice", "bob", "paris"} & found

def test_detect_named_entities_list_input(spacy_pipe):
    docs = nlptools.spacy_analyze(spacy_pipe,
                                  ["Alice visited Paris.", "Bob went to London."])
    ents = nlptools.detect_named_entities(docs)
    assert isinstance(ents, dict)
    assert len(ents) >= 2

def test_detect_named_entities_empty():
    assert nlptools.detect_named_entities([]) == {}

def test_suggest_keywords_distinguishes_documents(sample_docs):
    per_doc = [nlptools.count_frequencies(d, min_occurrences=1, min_length=3) for d in sample_docs]
    keywords = nlptools.suggest_keywords(per_doc)
    assert len(keywords) == 3
    # Each document gets a distinct top keyword from its own subject matter.
    doc0_text = "finite element discretize"
    doc1_text = "machine learning neural"
    doc2_text = "climate atmospheric circulation"
    # At least one keyword of each doc should be subject-specific.
    assert any(kw in doc0_text for kw in keywords[0])
    assert any(kw in doc1_text for kw in keywords[1])
    assert any(kw in doc2_text for kw in keywords[2])

def test_suggest_keywords_max_keywords(sample_docs):
    per_doc = [nlptools.count_frequencies(d, min_occurrences=1, min_length=3) for d in sample_docs]
    keywords = nlptools.suggest_keywords(per_doc, max_keywords=2)
    assert all(len(kws) <= 2 for kws in keywords)

def test_suggest_keywords_with_supplied_corpus(sample_docs):
    per_doc = [nlptools.count_frequencies(d, min_occurrences=1, min_length=3) for d in sample_docs]
    # A deliberately narrow corpus — words absent from it are dropped from results.
    corpus = {"machine": 5, "learning": 4, "climate": 3}
    keywords = nlptools.suggest_keywords(per_doc, corpus_frequencies=corpus)
    for kws in keywords:
        for kw in kws:
            assert kw in corpus

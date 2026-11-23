"""Compare cluster keyword methods on an existing Visualizer dataset.

Reads a dataset pickle as the importer wrote it (records with their title, abstract and cluster ID, plus the
cluster keywords the importer stored), extracts candidate terms afresh with spaCy, and writes a report
setting three methods side by side:

    stored    the keywords in the dataset, as the importer made them
    unigram   single nouns and proper nouns, scored by class-based TF-IDF
    phrase    noun phrases, scored by class-based TF-IDF, nested phrases suppressed

Usage:

    python investigations/cluster-keywords/keyword_probe.py DATASET.pickle [--top 6] [--show 15]

The extracted candidates are cached under `cache/` beside this script, keyed by the dataset's path and
mtime, so a second run over the same dataset costs seconds. The report goes to `out/`.
"""

import argparse
import collections
import hashlib
import json
import math
import pathlib
import pickle
import random
import sys
import unicodedata

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))  # run from anywhere; import the tree this script is in

from raven.common import nlptools  # noqa: E402 -- after the path setup
from raven.common.text import stopwords as raven_stopwords  # noqa: E402 -- after the path setup

STOPWORDS = set(nlptools.default_stopwords) | {x.lower() for x in raven_stopwords.academic_stopwords}

BARE_STOPWORDS = raven_stopwords.bare_keyword_stopwords

# Parts of speech that may not start or end a phrase. A noun chunk arrives with its determiner ("a novel
# method"), and sometimes a number or a pronoun; those say nothing about the topic.
EDGE_POS = {"DET", "PRON", "NUM", "PUNCT", "ADP", "CCONJ", "SCONJ", "PART", "SYM", "AUX", "X", "SPACE"}
MAX_PHRASE_WORDS = 4
MIN_CHARS = 3


# --------------------------------------------------------------------------------
# Candidate extraction

def sentence_case(title: str) -> str:
    """Lowercase a title written in Title Case or ALL CAPS, and return any other title as it is.

    A tagger reads a capitalized word as a proper noun, and a proper noun is not lemmatized: "Rectangular
    Plates" stays plural, and the same phrase in an abstract is counted as a different term. Lowercasing
    loses the acronyms in such a title, which costs less than splitting every term in it.
    """
    words = [w for w in title.split() if w[:1].isalpha()]
    if len(words) < 4:
        return title
    capitalized = sum(1 for w in words if w[0].isupper())
    if capitalized / len(words) > 0.6:
        return title.lower()
    return title


def record_text(entry) -> str:
    """Title and abstract, as the importer joins them for analysis, with the title sentence-cased."""
    title = sentence_case(entry.title or "")
    abstract = entry.abstract or ""
    # NFKC folds the compatibility forms: "TiO₂" and "TiO2" are one term.
    if title and abstract:
        return unicodedata.normalize("NFKC", f"{title}.\n\n{abstract}")
    return unicodedata.normalize("NFKC", title or abstract)


def unigrams(doc) -> set[str]:
    """The nouns and proper nouns of a document, lemmatized, as `nlptools.count_frequencies` filters them —
    except that one occurrence is enough, where the importer's per-record count needs two."""
    out = set()
    for token in doc:
        if token.pos_ not in ("NOUN", "PROPN") or not token.lemma_.isalnum():
            continue
        word = token.lemma_.lower()
        if len(word) >= MIN_CHARS and word not in STOPWORDS:
            out.add(word)
    return out


def phrase_words(tokens) -> list[str]:
    """The words of a phrase, lowercased, with the head lemmatized.

    Hyphenated compounds stay whole: "retrieval-augmented" is one word. Only the head is lemmatized:
    "rectangular plates" is "rectangular plate", while a modifier's lemma is often another word ("moving
    plate" would become "move plate", "cracked beam" "crack beam").
    """
    words = []
    for k, token in enumerate(tokens):
        form = token.lemma_.lower() if k == len(tokens) - 1 else token.text.lower()
        if words and (token.text == "-" or words[-1].endswith("-")):
            words[-1] += form
        else:
            words.append(form)
    return words


def abbreviations(doc) -> list[tuple[str, str]]:
    """Abbreviations a document defines, as `(short, long)`, both as phrase terms.

    The algorithm of Schwartz and Hearst (2003, *A simple algorithm for identifying abbreviation
    definitions in biomedical text*, https://psb.stanford.edu/psb-online/proceedings/psb03/schwartz.pdf):
    in "long form (SF)", the short form is one token of 2–10 characters starting with a letter or digit
    and containing a letter (and here, two capitals or a digit), and the long form is the shortest run of the words before the parenthesis,
    at most min(|SF| + 5, 2 |SF|) of them, in which every character of the short form occurs in order,
    its first character starting a word.
    """
    found = []
    for k, token in enumerate(doc):
        if token.text != "(" or k + 2 >= len(doc) or doc[k + 2].text != ")":
            continue
        short = doc[k + 1].text
        if not (2 <= len(short) <= 10 and short[0].isalnum() and any(ch.isalpha() for ch in short)):
            continue
        # Not in Schwartz and Hearst, who work on biomedical text: an abbreviation has two capitals or a
        # digit ("SRL", "ChatGPT", "TiO2"). Without this, "the gas evolved at the anode (oxygen)" defines
        # the word "oxygen", and every phrase containing it expands.
        if sum(1 for ch in short if ch.isupper()) < 2 and not any(ch.isdigit() for ch in short):
            continue
        letters = [ch.lower() for ch in short if ch.isalnum()]
        if short.endswith("s") and len(letters) > 2 and short[:-1].isupper():  # "LLMs" defines "LLM"
            letters = letters[:-1]
        window = min(len(letters) + 5, 2 * len(letters))
        before = [t for t in doc[max(0, k - window):k]]
        # Shortest suffix of the window in which the letters occur in order, the first starting a word.
        for start in range(len(before) - 1, -1, -1):
            span = before[start:]
            if span[0].is_punct or span[0].is_stop:
                continue
            text = " ".join(t.text.lower() for t in span)
            if not text.startswith(letters[0]):
                continue
            pos = 0
            for ch in letters:
                pos = text.find(ch, pos)
                if pos < 0:
                    break
                pos += 1
            else:
                if span[-1].pos_ in ("NOUN", "PROPN") and len(span) >= 2 or "-" in text:
                    found.append((" ".join(letters), " ".join(phrase_words(span))))
                break
    return [("".join(short.split()), long) for short, long in found]


def phrases(doc) -> set[str]:
    """The noun phrases of a document, lemmatized, with each one's head-final sub-phrases.

    English puts a noun phrase's head at its end, so "a fine-tuned large language model" also yields "large
    language model", "language model" and "model": each is a phrase in its own right, and the long form
    rarely repeats across records where the shorter ones do. Which of them is worth showing is left to the
    scoring.
    """
    out = set()
    for chunk in doc.noun_chunks:
        tokens = list(chunk)
        while tokens and (tokens[0].pos_ in EDGE_POS or tokens[0].lemma_.lower() in STOPWORDS):
            tokens.pop(0)
        while tokens and (tokens[-1].pos_ in EDGE_POS or tokens[-1].lemma_.lower() in STOPWORDS):
            tokens.pop()
        # Punctuation inside a chunk, other than a hyphen, separates two phrases: "large language models
        # (LLMs" is "large language models" and "LLMs". Keep the head's side.
        cut = max((k for k, t in enumerate(tokens) if t.is_punct and t.text != "-"), default=-1)
        tokens = tokens[cut + 1:]
        if not tokens or tokens[-1].pos_ not in ("NOUN", "PROPN"):
            continue
        words = phrase_words(tokens)
        for start in range(max(0, len(words) - MAX_PHRASE_WORDS), len(words)):
            if words[start] in STOPWORDS:  # "dark- and photo-fermentation" yields "and photo-fermentation"
                continue
            phrase = " ".join(words[start:])
            if len(phrase) < MIN_CHARS or phrase in STOPWORDS:
                continue
            if not any(ch.isalpha() for ch in phrase):
                continue
            out.add(phrase)
    return out


def extract(dataset_path: pathlib.Path, records) -> dict:
    """Candidate sets per record, `{"unigram": [set, ...], "phrase": [set, ...]}`, cached."""
    stat = dataset_path.stat()
    key = hashlib.sha256(f"{dataset_path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}".encode()).hexdigest()[:16]
    cache_path = HERE / "cache" / f"{dataset_path.stem}-{key}.pickle"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"Extracting candidates from {len(records)} records (spaCy, CPU)...", flush=True)
    nlp = nlptools.load_spacy_pipeline("en_core_web_sm", "cpu")
    result = {"unigram": [], "phrase": [], "abbreviations": []}
    texts = [record_text(entry) for entry in records]
    for j, doc in enumerate(nlp.pipe(texts, batch_size=64)):
        if j % 1000 == 0:
            print(f"    {j} / {len(texts)}", flush=True)
        result["unigram"].append(unigrams(doc))
        result["phrase"].append(phrases(doc))
        result["abbreviations"].append(abbreviations(doc))

    cache_path.parent.mkdir(exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    return result


def plural_map(vocabulary: set[str]) -> dict[str, str]:
    """`{plural: singular}` for the terms of `vocabulary` that `fold_plurals` folds."""
    return {term: term[:-1] for term in vocabulary
            if term.endswith("s") and not term.endswith("ss") and term[:-1] in vocabulary}


def fold_plurals(candidates: list[set[str]]) -> list[set[str]]:
    """Map a term ending in "s" to the same term without it, where the corpus also has that form.

    The tagger reads a capitalized word as a proper noun and leaves it unlemmatized, so "Large Language
    Models" in an abstract stays plural beside "large language model" elsewhere; an acronym's plural,
    "LLMs", is not lemmatized at all. Folding only onto a form the corpus actually contains keeps a word
    that merely ends in "s" ("process", "physics") as it is.
    """
    mapping = plural_map(set().union(*candidates))
    return [{mapping.get(term, term) for term in terms} for terms in candidates]


def expand_abbreviations(candidates: list[set[str]],
                         definitions: list[list[tuple[str, str]]]) -> tuple[list[set[str]], dict[str, str]]:
    """Merge each abbreviation with its long form, word by word, in every record.

    A definition two or more records make applies corpus-wide, including to the many records that use the
    short form without defining it; one made by a single record applies to that record only, so that a
    paper calling its system CARE does not take over the word "care" everywhere else. Abbreviations inside
    a phrase expand too, so "generative ai" and "generative artificial intelligence" are one term.

    Returns the candidates in canonical form, and `{canonical: display}`. The display form is the spelling
    the corpus uses most, except for an abbreviation: that shows as its short form only when most records
    using it do not define it. A corpus that uses "ChatGPT" unexplained knows it; one that defines "FWP"
    wherever it appears does not, and neither will its reader, who sees "fast weight programmer".
    """
    counts = collections.defaultdict(collections.Counter)
    for pairs in definitions:
        for short, long in set(pairs):
            counts[short][long] += 1
    corpus_map = {short: longs.most_common(1)[0][0] for short, longs in counts.items()
                  if sum(longs.values()) >= 2}

    def expand(term: str, mapping: dict[str, str]) -> str:
        words = []
        for word in term.split():
            replacement = mapping.get(word, word)
            if replacement != word:  # one level of nesting: "genai" -> "generative ai" -> "... artificial intelligence"
                replacement = " ".join(mapping.get(w, w) if w != word else w for w in replacement.split())
            words.append(replacement)
        return " ".join(words)

    expanded = []
    for terms, pairs in zip(candidates, definitions):
        mapping = dict(corpus_map)
        mapping.update(dict(pairs))
        expanded.append({term: expand(term, mapping) for term in terms})
    # Fold plurals again: a long form arrives as written, and "Large Language Models (LLMs)" is plural.
    folds = plural_map({c for record in expanded for c in record.values()})

    surface = collections.defaultdict(collections.Counter)
    short_use = collections.defaultdict(lambda: [0, 0])  # canonical -> [records defining it, records not]
    out = []
    for record, pairs in zip(expanded, definitions):
        defined_here = {short for short, _ in pairs}
        canonical = set()
        for term, c in record.items():
            c = folds.get(c, c)
            surface[c][term] += 1
            canonical.add(c)
            if term in corpus_map or term in defined_here:  # the bare short form of an abbreviation
                short_use[c][0 if term in defined_here else 1] += 1
        out.append(canonical)
    display = {}
    for c, forms in surface.items():
        defining, undefined = short_use.get(c, (0, 0))
        if defining and undefined < defining:  # mostly explained where used: the reader needs the long form
            long_forms = [(n, f) for f, n in forms.items() if " " in f or "-" in f]
            display[c] = max(long_forms)[1] if long_forms else c
        else:
            display[c] = forms.most_common(1)[0][0]
    return out, display


# --------------------------------------------------------------------------------
# Scoring

def ctfidf(candidates: list[set[str]], labels: list[int], *, min_support: int = 2,
           weighting: str = "bertopic") -> dict[int, list[tuple[str, float, float]]]:
    """Class-based TF-IDF over clusters (Grootendorst 2022, BERTopic), on record counts.

    `candidates`: each record's candidate terms, as a set.
    `labels`: each record's cluster ID; negative is an outlier, which counts towards the corpus totals but
              belongs to no class.
    `min_support`: a term must appear in at least this many records of a cluster to be ranked for it.

    The weight of term t in cluster c is

        W(t, c) = tf(t, c) * log(1 + A / f(t))

    where tf(t, c) is the number of the cluster's records containing t, f(t) the number of records in the
    whole corpus containing t, and A the average number of term occurrences per cluster. Counting records
    rather than occurrences keeps one abstract that repeats a word from deciding its cluster's label, and
    it is what makes tf(t, c) / |c| the share of the cluster the term covers.

    `weighting="kl"` scores instead by the term's contribution to the KL divergence of the cluster from
    the corpus, over the binary event "a record contains t":

        W(t, c) = p log(p / q),    p = tf(t, c) / |c|,    q = f(t) / N

    the cluster's share times the log of the lift. A term no more common in the cluster than in the corpus
    scores zero or below, however much of the cluster it covers, which is what an umbrella term of a
    single-topic corpus is.

    Returns `{cluster_id: [(term, weight, share), ...]}`, best first.
    """
    corpus_df = collections.Counter()
    for terms in candidates:
        corpus_df.update(terms)

    cluster_tf = collections.defaultdict(collections.Counter)
    cluster_size = collections.Counter()
    for terms, label in zip(candidates, labels):
        if label >= 0:
            cluster_tf[label].update(terms)
            cluster_size[label] += 1

    avg_terms = sum(sum(tf.values()) for tf in cluster_tf.values()) / len(cluster_tf)
    n_records = len(candidates)

    def weight(count, size, term):
        if weighting == "bertopic":
            return count * math.log(1.0 + avg_terms / corpus_df[term])
        p, q = count / size, corpus_df[term] / n_records
        return p * math.log(p / q)

    ranked = {}
    for label, tf in cluster_tf.items():
        scored = [(term, weight(count, cluster_size[label], term), count / cluster_size[label])
                  for term, count in tf.items() if count >= min_support]
        scored.sort(key=lambda item: (-item[1], item[0]))  # ties by term, so a run does not depend on set order
        ranked[label] = scored
    return ranked


def nested(a: str, b: str) -> bool:
    """Whether one phrase is the other's word sequence or a contiguous part of it."""
    wa, wb = a.split(), b.split()
    if len(wa) > len(wb):
        wa, wb = wb, wa
    return any(wb[i:i + len(wa)] == wa for i in range(len(wb) - len(wa) + 1))


def umbrella_terms(candidates: list[set[str]], labels: list[int], *, min_support: int = 2,
                   threshold_fraction: float = 0.1) -> set[str]:
    """Terms present in too many clusters to say how any one of them stands out.

    The rule `nlptools.suggest_keywords` applies to the stored keywords, with clusters as its documents: a
    term present in at least max(min(5, n), ceil(threshold_fraction n)) of the n clusters is excluded.
    Present means in at least `min_support` of the cluster's records, the bar a term must clear to be
    ranked for a cluster at all.
    """
    tf = collections.defaultdict(collections.Counter)
    for terms, label in zip(candidates, labels):
        if label >= 0:
            tf[label].update(terms)
    n = len(tf)
    threshold_n = max(min(5, n), math.ceil(threshold_fraction * n))
    presence = collections.Counter()
    for counts in tf.values():
        presence.update(term for term, count in counts.items() if count >= min_support)
    return {term for term, count in presence.items() if count >= threshold_n}


def pick(ranked: list[tuple[str, float, float]], top: int, *, suppress_nested: bool,
         keep_longer: float = 0.5, exclude: frozenset[str] = frozenset()) -> list[tuple[str, float]]:
    """The first `top` terms as `(term, share)`, with nested phrases suppressed if asked, and the terms in
    `exclude` skipped.

    When a candidate and a term already taken are nested in each other, only one of them is kept. The
    shorter one always covers at least as much of the cluster, every record holding "latin hypercube
    sampling" also holding "sampling", so ranking alone keeps the shorter and loses the name. So a longer
    phrase replaces a shorter one already taken when it covers at least `keep_longer` of what the shorter
    one does; otherwise the candidate is skipped.
    """
    chosen = []
    for term, _weight, share in ranked:
        if term in exclude:
            continue
        if suppress_nested:
            clash = next((j for j, (other, _) in enumerate(chosen) if nested(term, other)), None)
            if clash is not None:
                other, other_share = chosen[clash]
                if len(term.split()) > len(other.split()) and share >= keep_longer * other_share:
                    if not any(nested(term, x) for j, (x, _) in enumerate(chosen) if j != clash):
                        chosen[clash] = (term, share)
                continue
        chosen.append((term, share))
        if len(chosen) == top:
            break
    return chosen


def pick_mixed(ranked: list[tuple[str, float, float]], top: int, *, n_common: int, umbrella: set[str],
               bare: frozenset[str]) -> list[tuple[str, float]]:
    """`n_common` terms from the full ranking, then the rest from the ranking with umbrella terms removed.

    The two pools of the keyword-pools brief, from one set of scores: the first says what the cluster is
    (in a single-topic corpus, often a term other clusters share), the rest how it stands out.
    """
    common = pick(ranked, n_common, suppress_nested=True, exclude=bare)
    taken = {term for term, _ in common}
    rest = [item for item in ranked
            if item[0] not in taken and not any(nested(item[0], t) for t in taken)]
    distinctive = pick(rest, top - len(common), suppress_nested=True, exclude=bare | umbrella)
    return common + distinctive


# --------------------------------------------------------------------------------
# Embedding rerank

def make_embedder():
    """The importer's embedding model, through Raven-server when it answers, else locally."""
    from raven.client import api
    from raven.client import config as client_config
    from raven.client import mayberemote
    from raven.visualizer import config as visualizer_config
    api.initialize(raven_server_url=client_config.raven_server_url,
                   maybe_network_retry=client_config.network_retry_batch,
                   raven_api_key_file=client_config.raven_api_key_file)
    return mayberemote.Embedder(allow_local=True,
                                model_name=visualizer_config.embedding_model,
                                device_string=client_config.devices["embeddings"]["device_string"],
                                dtype=client_config.devices["embeddings"]["dtype"])


def embed(embedder, texts: list[str], *, batch_size: int = 256) -> np.ndarray:
    """Unit vectors for `texts`, one row each."""
    rows = [np.asarray(embedder.encode(texts[k:k + batch_size])) for k in range(0, len(texts), batch_size)]
    vectors = np.concatenate(rows).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def cached_vectors(embedder_box: list, cache_path: pathlib.Path, texts: list[str]) -> dict[str, np.ndarray]:
    """`{text: vector}`, embedding only the texts not already in the cache file."""
    cache = {}
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    missing = [t for t in dict.fromkeys(texts) if t not in cache]
    if missing:
        if not embedder_box:
            embedder_box.append(make_embedder())
        print(f"Embedding {len(missing)} texts...", flush=True)
        cache.update(zip(missing, embed(embedder_box[0], missing)))
        cache_path.parent.mkdir(exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
    return cache


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def mmr(relevance: np.ndarray, vectors: np.ndarray, top: int, *, diversity: float = 0.5) -> list[int]:
    """Maximal marginal relevance (Carbonell and Goldstein 1998): indices of `top` items, each chosen to
    maximize (1 - diversity) * relevance - diversity * (its largest similarity to an item already chosen).
    `vectors` are unit rows, so similarity is a dot product.
    """
    chosen = [int(np.argmax(relevance))]
    similarity = vectors @ vectors.T
    while len(chosen) < min(top, len(relevance)):
        redundancy = similarity[:, chosen].max(axis=1)
        score = (1.0 - diversity) * relevance - diversity * redundancy
        score[chosen] = -np.inf
        chosen.append(int(np.argmax(score)))
    return chosen


def rerank(ranked: dict[int, list[tuple[str, float, float]]], members: dict[int, list[int]],
           doc_vectors: np.ndarray, term_vectors: dict[str, np.ndarray], top: int, *, pool: int, centered: bool,
           bare: frozenset[str]) -> dict[int, list[tuple[str, float]]]:
    """Each cluster's c-TF-IDF pool reordered by similarity to the cluster's centroid, picked by MMR.

    After KeyBERT (Grootendorst 2020, https://github.com/MaartenGr/KeyBERT), with the pool taken from the
    c-TF-IDF ranking rather than from all n-grams. `centered=True` subtracts the corpus mean from the
    documents and the mean of all pooled terms from the terms before comparing, so that a term scores by
    how it points where this cluster differs from the corpus rather than where the whole corpus points.
    """
    pools = {cid: [item for item in ranked.get(cid, []) if item[0] not in bare][:pool] for cid in members}
    all_terms = sorted({term for items in pools.values() for term, _, _ in items})
    term_matrix = np.stack([term_vectors[t] for t in all_terms]) if all_terms else np.zeros((0, doc_vectors.shape[1]))
    doc_mean = doc_vectors.mean(axis=0) if centered else 0.0
    term_mean = term_matrix.mean(axis=0) if centered and len(all_terms) else 0.0
    index = {t: k for k, t in enumerate(all_terms)}
    out = {}
    for cid, items in pools.items():
        if not items:
            out[cid] = []
            continue
        centroid = unit(doc_vectors[members[cid]].mean(axis=0) - doc_mean)
        vectors = unit(term_matrix[[index[t] for t, _, _ in items]] - term_mean)
        relevance = vectors @ centroid
        out[cid] = [(items[k][0], items[k][2]) for k in mmr(relevance, vectors, top)]
    return out


def relevance_order(items: list[tuple[str, float, float]], centroid: np.ndarray,
                    term_vectors: dict[str, np.ndarray]) -> list[tuple[str, float, float]]:
    """`items` sorted by their term's cosine similarity to `centroid`, a unit vector, most similar first."""
    if not items:
        return []
    vectors = unit(np.stack([term_vectors[t] for t, _, _ in items]))
    order = np.argsort(-(vectors @ centroid), kind="stable")
    return [items[k] for k in order]


def fuse(rankings: list[list[tuple[str, float, float]]], *, k: int = 60) -> list[tuple[str, float, float]]:
    """Reciprocal rank fusion (Cormack, Clarke and Buettcher 2009): each term scores the sum of 1 / (k + rank)
    over the rankings that hold it, and the fused ranking is by that score. k = 60 is theirs."""
    score = collections.Counter()
    item_of = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            score[item[0]] += 1.0 / (k + rank)
            item_of[item[0]] = item
    return [item_of[t] for t, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))]


def interleave(rankings: list[list[tuple[str, float, float]]]) -> list[tuple[str, float, float]]:
    """The rankings' items taken in turn, first of each, then second of each, a term only the first time."""
    out, seen = [], set()
    for row in zip(*rankings):
        for item in row:
            if item[0] not in seen:
                seen.add(item[0])
                out.append(item)
    return out


# --------------------------------------------------------------------------------
# Measures

def shares_by_literal_match(keywords: list[str], member_candidates: list[set[str]]) -> list[float]:
    """Each keyword's share of a cluster, by literal match against the members' candidate sets."""
    n = len(member_candidates)
    return [sum(1 for terms in member_candidates if kw in terms) / n for kw in keywords]


def summarize(labels_by_cluster: dict[int, list[str]], members: dict[int, list[int]], candidates: list[set[str]]) -> dict:
    """Across-cluster distinctiveness and within-cluster coverage of a labelling."""
    occurrences = collections.Counter(kw for kws in labels_by_cluster.values() for kw in kws)
    slots = sum(len(kws) for kws in labels_by_cluster.values())
    coverage = []
    for cid, kws in labels_by_cluster.items():
        member_sets = [candidates[i] for i in members[cid]]
        coverage.append(sum(1 for terms in member_sets if any(kw in terms for kw in kws)) / len(member_sets))
    return {"slots": slots,
            "distinct": len(occurrences),
            "repeated": sum(1 for count in occurrences.values() if count > 1),
            "most_repeated": occurrences.most_common(8),
            "mean_coverage": sum(coverage) / len(coverage)}


# --------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", type=pathlib.Path)
    parser.add_argument("--top", type=int, default=6, help="keywords per cluster (default: 6, as the importer)")
    parser.add_argument("--rerank", action="store_true", help="add the embedding rerank (needs the embedder)")
    parser.add_argument("--pool", type=int, default=30, help="for --rerank: candidates per cluster from c-TF-IDF (default: 30)")
    parser.add_argument("--show", type=int, default=15, help="clusters shown in full, of the largest and of a random draw each")
    args = parser.parse_args()

    with open(args.dataset, "rb") as f:
        dataset = pickle.load(f)
    records = dataset["vis_data"]
    labels = [entry.cluster_id for entry in records]
    members = collections.defaultdict(list)
    for idx, label in enumerate(labels):
        if label >= 0:
            members[label].append(idx)

    candidates = extract(args.dataset, records)
    definitions = candidates["abbreviations"]
    candidates = {space: fold_plurals(candidates[space]) for space in ("unigram", "phrase")}
    candidates["phrase-ab"], display = expand_abbreviations(candidates["phrase"], definitions)

    # Stored LLM keywords are phrases in their own capitalization; they are matched against the phrase
    # candidates, which undercounts them, a keyword being an abstraction over its cluster.
    stored = {cid: [kw.lower() for kw in dataset["vis_keywords_by_cluster"][cid]] for cid in members}
    stored_space = "phrase-ab" if any(" " in kw for kws in stored.values() for kw in kws) else "unigram"
    # Fold the stored keywords' plurals as the candidates' were, so "transformers" matches "transformer".
    stored_vocabulary = set().union(*candidates[stored_space]) | {kw for kws in stored.values() for kw in kws}
    stored_folds = plural_map(stored_vocabulary)
    stored = {cid: [stored_folds.get(kw, kw) for kw in kws] for cid, kws in stored.items()}
    unigram_ranked = ctfidf(candidates["unigram"], labels)
    phrase_ab_ranked = ctfidf(candidates["phrase-ab"], labels)
    umbrella = umbrella_terms(candidates["phrase-ab"], labels)
    methods = {
        "stored": {cid: list(zip(kws, shares_by_literal_match(kws, [candidates[stored_space][i] for i in members[cid]])))
                   for cid, kws in stored.items()},
        "unigram": {cid: pick(unigram_ranked.get(cid, []), args.top, suppress_nested=False) for cid in members},
        "phrase-ab": {cid: pick(phrase_ab_ranked.get(cid, []), args.top, suppress_nested=True) for cid in members},
        "phrase-cut": {cid: pick(phrase_ab_ranked.get(cid, []), args.top, suppress_nested=True, exclude=umbrella)
                       for cid in members},
        "phrase-bare": {cid: pick(phrase_ab_ranked.get(cid, []), args.top, suppress_nested=True,
                                  exclude=BARE_STOPWORDS) for cid in members},
        "phrase-mix1": {cid: pick_mixed(phrase_ab_ranked.get(cid, []), args.top, n_common=1, umbrella=umbrella,
                                        bare=BARE_STOPWORDS) for cid in members},
        "phrase-mix2": {cid: pick_mixed(phrase_ab_ranked.get(cid, []), args.top, n_common=2, umbrella=umbrella,
                                        bare=BARE_STOPWORDS) for cid in members},
    }
    if args.rerank:
        embedder_box = []
        stem = args.dataset.stem
        doc_texts = [record_text(entry) for entry in records]
        doc_cache = cached_vectors(embedder_box, HERE / "cache" / f"{stem}-docvectors.pickle", doc_texts)
        doc_vectors = np.stack([doc_cache[t] for t in doc_texts])
        pooled = {t for cid in members for t, _, _ in
                  [item for item in phrase_ab_ranked.get(cid, []) if item[0] not in BARE_STOPWORDS][:args.pool]}
        term_vectors = cached_vectors(embedder_box, HERE / "cache" / f"{stem}-termvectors.pickle", sorted(pooled))
        for name, centered in (("rerank", False), ("rerank-c", True)):
            methods[name] = rerank(phrase_ab_ranked, members, doc_vectors, term_vectors, args.top, pool=args.pool,
                                   centered=centered, bare=BARE_STOPWORDS)
        # The two pools again, with the distinctive one reordered by the rerank: one term from the full
        # c-TF-IDF ranking, then the rest from the ranking with the umbrella terms and that term removed.
        common = {cid: pick(phrase_ab_ranked.get(cid, []), 1, suppress_nested=True, exclude=BARE_STOPWORDS)
                  for cid in members}
        distinctive = {cid: [item for item in phrase_ab_ranked.get(cid, [])
                             if item[0] not in umbrella and not any(nested(item[0], t) for t, _ in common[cid])]
                       for cid in members}
        more = {t for cid in members for t, _, _ in
                [item for item in distinctive[cid] if item[0] not in BARE_STOPWORDS][:args.pool]} - set(term_vectors)
        if more:
            term_vectors = cached_vectors(embedder_box, HERE / "cache" / f"{stem}-termvectors.pickle",
                                          sorted(set(term_vectors) | more))
        for name, centered in (("mix-rerank", False), ("mix-rerank-c", True)):
            tails = rerank(distinctive, members, doc_vectors, term_vectors, args.top - 1, pool=args.pool,
                           centered=centered, bare=BARE_STOPWORDS)
            methods[name] = {cid: common[cid] + tails[cid] for cid in members}
        # The hybrids: one common term, then the distinctive pool ordered two ways -- by c-TF-IDF, and by
        # similarity to the cluster's centroid -- and combined, alternately or by rank fusion.
        for name, combine in (("hybrid-alt", interleave), ("hybrid-rrf", fuse)):
            methods[name] = {}
            for cid in members:
                by_ctfidf = [item for item in distinctive[cid] if item[0] not in BARE_STOPWORDS][:args.pool]
                centroid = unit(doc_vectors[members[cid]].mean(axis=0))
                by_relevance = relevance_order(by_ctfidf, centroid, term_vectors)
                tail = pick(combine([by_ctfidf, by_relevance]), args.top - 1, suppress_nested=True)
                methods[name][cid] = common[cid] + tail
    labels_canonical = {name: {cid: [kw for kw, _ in kws] for cid, kws in labelling.items()} for name, labelling in methods.items()}
    for name in [n for n in methods if n.startswith(("phrase", "rerank", "mix", "hybrid"))]:
        methods[name] = {cid: [(display.get(kw, kw), share) for kw, share in kws] for cid, kws in methods[name].items()}
    coverage_space = {"stored": stored_space, "unigram": "unigram", "phrase-ab": "phrase-ab", "phrase-cut": "phrase-ab", "phrase-bare": "phrase-ab",
                      "phrase-mix1": "phrase-ab", "phrase-mix2": "phrase-ab", "rerank": "phrase-ab", "rerank-c": "phrase-ab",
                      "mix-rerank": "phrase-ab", "mix-rerank-c": "phrase-ab",
                      "hybrid-alt": "phrase-ab", "hybrid-rrf": "phrase-ab"}

    lines = [f"# Cluster keywords: {args.dataset.name}", "",
             f"{len(records)} records, {len(members)} clusters, {sum(1 for x in labels if x < 0)} outliers; "
             f"stored method: {dataset.get('clusters_keyword_method')}; top {args.top}.", "",
             "| method | slots | distinct | repeated across clusters | mean coverage |",
             "|---|---|---|---|---|"]
    summaries = {}
    for name, labelling in methods.items():
        s = summarize(labels_canonical[name], members, candidates[coverage_space[name]])
        summaries[name] = s
        lines.append(f"| {name} | {s['slots']} | {s['distinct']} ({s['distinct'] / max(1, s['slots']):.0%}) | "
                     f"{s['repeated']} | {s['mean_coverage']:.0%} |")
    lines.append("")
    lines.append("*Coverage*: the share of a cluster's records containing at least one of its keywords, averaged over clusters.")
    lines.append("")
    lines.append(f"Umbrella terms excluded by `phrase-cut` ({len(umbrella)}): "
                 + ", ".join(sorted(display.get(t, t) for t in umbrella)))
    lines.append("")
    for name, s in summaries.items():
        lines.append(f"- most repeated, {name}: " + ", ".join(f"{display.get(kw, kw) if name.startswith('phrase') else kw} ×{n}"
                                                               for kw, n in s["most_repeated"]))
    lines.append("")

    by_size = sorted(members, key=lambda cid: -len(members[cid]))
    largest = by_size[:args.show]
    rng = random.Random(0)
    rest = [cid for cid in by_size if cid not in largest]
    drawn = sorted(rng.sample(rest, min(args.show, len(rest))), key=lambda cid: -len(members[cid]))

    def show(title, cids):
        lines.extend([f"## {title}", ""])
        for cid in cids:
            lines.append(f"### Cluster {cid} ({len(members[cid])} records)")
            lines.append("")
            for name, labelling in methods.items():
                lines.append(f"- **{name}**: " + ", ".join(f"{kw} ({share:.0%})" for kw, share in labelling[cid]))
            sample = rng.sample(members[cid], min(3, len(members[cid])))
            for idx in sample:
                lines.append(f"  - *{records[idx].title}*")
            lines.append("")

    show("The largest clusters", largest)
    show("A random draw of the rest", drawn)

    out_path = HERE / "out" / f"{args.dataset.stem}.md"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump({name: {str(cid): [kw for kw, _ in kws] for cid, kws in labelling.items()}
                   for name, labelling in methods.items()}, f, indent=1)
    print("\n".join(lines[:16 + len(methods)]))
    print(f"\nReport: {out_path}")


if __name__ == "__main__":
    main()

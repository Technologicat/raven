"""Keyphrases for the groups of a corpus: what each group is about, in noun phrases.

A group is anything a corpus is divided into: the clusters of a semantic map, the years of a bibliography.
Each group gets a label, its terms ranked from the most telling down, each term with the share of the group's
records containing it.

The method, from first to last:

  1. **Candidates** (`extract`, per record): the noun phrases of a parsed document, with the head-final
     sub-phrases of each ("large language model", "language model", "model"), and the abbreviations the
     document defines, found with the algorithm of Schwartz and Hearst (2003).
  2. **Normalization** (`normalize`, over the corpus): plurals folded onto a singular the corpus contains, and
     each abbreviation merged with its long form, displayed in whichever the corpus's readers would know.
  3. **Ranking** (`rank_terms`): class-based TF-IDF over the groups (Grootendorst 2022, BERTopic), on record
     counts, so that a term's score within a group also gives the share of the group it covers.
  4. **The label** (`label_groups`): the group's best term, then a tail from the terms that say how it stands
     out, ordered twice, by the ranking above and by similarity to the group's centroid in an embedding space
     (after KeyBERT, Grootendorst 2020), and the two orders combined by reciprocal rank fusion (Cormack,
     Clarke and Buettcher 2009). Terms present in too many groups are left out of the tail.

The measurements that chose each step are in the repository's `investigations/cluster-keywords/`.

Nothing here loads a model: the caller passes parsed documents, record vectors, and a function that embeds
terms. So the module itself needs only NumPy.
"""

__all__ = ["EXTRACTION_VERSION",
           "sentence_case", "nfkc", "extract",
           "casing_table", "recase", "normalize",
           "rank_terms", "umbrella_terms", "fuse",
           "label_groups",
           "top_terms",
           "without_umbrella_terms", "shared_last", "lead_with"]

import collections
import math
import re
import unicodedata
from typing import Callable, Container, Hashable, Sequence

import numpy as np

from .text import stopwords as raven_stopwords

# A cache of `extract`'s output is valid only for the version that wrote it. Bump this when extraction changes.
EXTRACTION_VERSION = 4

# Parts of speech that may not start or end a phrase. A noun chunk arrives with its determiner ("a novel
# method"), and sometimes a number or a pronoun; those say nothing about the topic.
_EDGE_POS = frozenset({"DET", "PRON", "NUM", "PUNCT", "ADP", "CCONJ", "SCONJ", "PART", "SYM", "AUX", "X", "SPACE"})
_NOUN_POS = ("NOUN", "PROPN")
_CASED_POS = frozenset({"NOUN", "PROPN", "ADJ"})
_MAX_PHRASE_WORDS = 4
_MIN_CHARS = 3


# --------------------------------------------------------------------------------
# Candidates, per record

def sentence_case(title: str) -> str:
    """Lowercase a title written in Title Case or ALL CAPS, and return any other title as it is.

    A title of fewer than four words is returned as it is, there being too little to tell its case from.
    """
    # A tagger reads a capitalized word as a proper noun, and does not lemmatize one: "Rectangular Plates"
    # stays plural, and is counted as another term than the same phrase in an abstract. Lowercasing loses the
    # acronyms in such a title, which costs less than splitting every term in it.
    words = [w for w in title.split() if w[:1].isalpha()]
    if len(words) < 4:
        return title
    if sum(1 for w in words if w[0].isupper()) / len(words) > 0.6:
        return title.lower()
    return title


def nfkc(text: str) -> str:
    """Unicode NFKC normalization, for preparing a text before parsing it for `extract`: "TiO₂" becomes
    "TiO2"."""
    return unicodedata.normalize("NFKC", text)


def _head_form(token) -> str:
    """A noun's lemma, lowercased, except where spaCy's small English pipeline lemmatizes wrongly.

    "data" keeps its spelling, the pipeline making it "datum". So does a word in "-ics", which is either a
    singular mass noun ("physics", which the pipeline makes "physic") or a plural ("characteristics"); the
    corpus-level `normalize` tells the two apart.
    """
    text = token.text.lower()
    if text.endswith("ics") or text == "data":
        return text
    return token.lemma_.lower()


# Punctuation that ends the word before it rather than starting one: "verification, validation".
_GLUED_PUNCT = frozenset({",", ".", ";", ":", "!", "?", ")", "]"})


def _phrase_words(tokens, *, cased: bool = False) -> list[str]:
    """The words of a phrase, lowercased, with the head lemmatized and hyphenated compounds kept whole.

    `cased`: give the words as written instead, neither lowercased nor lemmatized; word for word the same split.

    A possessive stays on its word ("prisoner's"), and punctuation such as a comma on the word before it.
    """
    # Only the head is lemmatized: a modifier's lemma is often another word ("moving plate" would become
    # "move plate", "cracked beam" "crack beam").
    words: list[str] = []
    for k, token in enumerate(tokens):
        if cased:
            form = token.text
        else:
            form = _head_form(token) if k == len(tokens) - 1 else token.text.lower()
        if words and (token.text == "-" or words[-1].endswith("-") or token.tag_ == "POS" or
                      (token.is_punct and token.text in _GLUED_PUNCT)):
            words[-1] += form
        else:
            words.append(form)
    return words


def _phrases(doc, stopwords: Container[str]) -> dict[str, str]:
    """The noun phrases of a parsed document, with each one's head-final sub-phrases, as `{term: as written}`."""
    out = {}
    for chunk in doc.noun_chunks:
        tokens = list(chunk)
        # A token joined to the next by a hyphen is the first half of a word ("test-time compute"), not a word
        # of its own, so being a stopword does not strip it; stripping it would leave "time compute".
        while tokens and (tokens[0].pos_ in _EDGE_POS or tokens[0].lemma_.lower() in stopwords) and \
                not (len(tokens) > 1 and tokens[1].text == "-" and not tokens[0].whitespace_):
            tokens.pop(0)
        # A stopword at the end is the head of the phrase. Qualified by a noun it names something ("tool use",
        # "sensitivity analysis") and stays; qualified by an adjective it does not ("remarkable ability") and
        # goes, as does a stopword standing alone, below.
        while tokens and (tokens[-1].pos_ in _EDGE_POS or
                          (tokens[-1].lemma_.lower() in stopwords and
                           (len(tokens) < 2 or tokens[-2].pos_ not in _NOUN_POS))):
            tokens.pop()
        # Punctuation inside a chunk, other than a hyphen, separates two phrases: "large language models (LLMs"
        # is "large language models" and "LLMs". The head's side is kept.
        cut = max((k for k, t in enumerate(tokens) if t.is_punct and t.text != "-"), default=-1)
        tokens = tokens[cut + 1:]
        if not tokens or tokens[-1].pos_ not in _NOUN_POS:
            continue
        # English puts a noun phrase's head at its end, so each head-final run of words is a phrase in its
        # own right; the long form rarely repeats across records where the shorter ones do, and which of them
        # is worth showing is left to the ranking.
        words = _phrase_words(tokens)
        written = _phrase_words(tokens, cased=True)
        for start in range(max(0, len(words) - _MAX_PHRASE_WORDS), len(words)):
            if words[start] in stopwords:  # "dark- and photo-fermentation" yields "and photo-fermentation"
                continue
            if not any(ch.isalpha() for ch in words[start]):  # "the Apache 2.0 license" yields "2.0 license"
                continue
            phrase = " ".join(words[start:])
            if len(phrase) >= _MIN_CHARS and phrase not in stopwords:
                out.setdefault(phrase, " ".join(written[start:]))
    return out


def _abbreviations(doc) -> list[tuple[str, str, str]]:
    """Abbreviations a parsed document defines, as `(short, long, short as written)`: "SRL" gives `("srl",
    "self-regulated learning", "SRL")`.

    The algorithm of Schwartz and Hearst (2003, *A simple algorithm for identifying abbreviation definitions in
    biomedical text*, https://psb.stanford.edu/psb-online/proceedings/psb03/schwartz.pdf): in "long form (SF)",
    the short form is one token of 2–10 characters starting with a letter or digit, and the long form is the
    shortest run of the words before the parenthesis, at most min(|SF| + 5, 2 |SF|) of them, in which every
    character of the short form occurs in order, its first starting a word. Added here: the short form must
    have two capitals or a digit ("SRL", "ChatGPT", "TiO2").
    """
    # The capitals rule is not Schwartz and Hearst's: without it, "the gas evolved at the anode (oxygen)"
    # defines the word "oxygen", and every phrase containing it expands.
    found = []
    for k, token in enumerate(doc):
        if token.text != "(" or k + 2 >= len(doc) or doc[k + 2].text != ")":
            continue
        short = doc[k + 1].text
        if not (2 <= len(short) <= 10 and short[0].isalnum() and any(ch.isalpha() for ch in short)):
            continue
        if sum(1 for ch in short if ch.isupper()) < 2 and not any(ch.isdigit() for ch in short):
            continue
        letters = [ch.lower() for ch in short if ch.isalnum()]
        written = short
        if short.endswith("s") and len(letters) > 2 and short[:-1].isupper():  # "LLMs" defines "LLM"
            letters = letters[:-1]
            written = short[:-1]
        window = min(len(letters) + 5, 2 * len(letters))
        before = list(doc[max(0, k - window):k])
        for start in range(len(before) - 1, -1, -1):  # the shortest suffix of the window that fits
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
                if (span[-1].pos_ in _NOUN_POS and len(span) >= 2) or "-" in text:
                    found.append(("".join(letters), " ".join(_phrase_words(span)), written))
                break
    return found


def extract(doc, *, stopwords: Container[str]) -> dict:
    """The candidate terms of one parsed document, for `normalize`.

    `doc`: a spaCy `Doc` with a dependency parse (noun chunks need one), from a pipeline such as
           `en_core_web_sm`. A document deserialized from `raven.common.nlptools.serialize_spacy_docs`
           output serves as well.
    `stopwords`: words that are not a term on their own; typically `nlptools.default_stopwords` together with
                 `raven.common.text.stopwords.academic_stopwords`.

    Returns a picklable `dict`: `"phrases"`, a sorted list of the document's phrase terms; `"cased"`, how it writes
    those whose spelling differs from the term, `{term: as written}`; `"definitions"`, the abbreviations it
    defines, as `(short, long, short as written)`; `"nouns"`, a sorted list of the nouns it uses, as written; and
    `"casing"`, how it capitalizes its nouns and adjectives where a sentence does not start, as
    `{lowercased word: the form most of its occurrences take}`, for `casing_table`.

    The text a document was parsed from is best prepared with `sentence_case` on its title, and Unicode NFKC
    normalization, so that "TiO₂" and "TiO2" are one term.
    """
    phrases = _phrases(doc, stopwords)
    return {"phrases": sorted(phrases),
            "cased": {term: written for term, written in phrases.items() if written != term},
            "definitions": _abbreviations(doc),
            "nouns": sorted({t.text.lower() for t in doc if t.pos_ in _NOUN_POS}),
            "casing": _casing(doc)}


def _casing(doc) -> dict[str, str]:
    """`{lowercased word: its usual form}` for the nouns and adjectives of `doc` not starting a sentence, counted in
    its abstract: after the first blank line, where there is one."""
    # A sentence's first word is capitalized whatever it is, so it says nothing about the word. A title is left
    # out, being sentence-cased before extraction, which lowercases every proper name in a Title Case title.
    #
    # Every occurrence of a noun or adjective counts, whatever its part of speech there: a capitalized part of a
    # compound ("AI-Based") is tagged a proper noun, while "based" in "based on" is a verb, so counting nouns and
    # adjectives alone hands every vote to the capitalized form.
    tokens = list(doc)
    breaks = [k for k, t in enumerate(tokens) if t.is_space and "\n\n" in t.text]
    region = tokens[breaks[0] + 1:] if breaks else tokens
    nominal = {t.text.lower() for t in region if t.pos_ in _CASED_POS}
    forms: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for t in region:
        word = t.text.lower()
        if word in nominal and not t.is_sent_start and any(ch.isalpha() for ch in t.text):
            forms[word][t.text] += 1
    return {word: max(c.items(), key=lambda kv: (kv[1], kv[0] == word))[0] for word, c in forms.items()}


# --------------------------------------------------------------------------------
# Normalization, over the corpus

# Fields of study in "-ics", a singular mass noun however often the "-ic" form heads a phrase: "first-order kinetic"
# is written, and still "kinetics" is no plural.
_ICS_MASS_NOUNS = ("acoustics", "analytics", "dynamics", "economics", "electronics", "ethics", "genetics", "informatics",
                   "kinematics", "kinetics", "linguistics", "logistics", "mathematics", "mechanics", "omics", "optics",
                   "physics", "politics", "robotics", "semantics", "statics", "statistics")

# British spellings and their American counterparts, word-final unless anchored otherwise, for the spelling classes.
_BRITISH = ((re.compile(r"isation"), "ization"), (re.compile(r"is(e|ed|es|er|ers|ing)$"), r"iz\1"),
            (re.compile(r"ys(e|ed|es|ing)$"), r"yz\1"), (re.compile(r"ell(ing|ed|er|ers)$"), r"el\1"),
            (re.compile(r"our(s|ed|ing)?$"), r"or\1"), (re.compile(r"(?<=[^aeiou])re(s|d)?$"), r"er\1"))


def _spelling_class(term: str) -> str:
    """A key shared by the spellings of one term: hyphens, spaces and a possessive dropped, British spellings
    made American. "prisoner's dilemma", "fibre-reinforced polymer" and "fiber reinforced polymer" share theirs
    with "prisonerdilemma" and so on."""
    words = []
    for word in term.split():
        word = word.removesuffix("'s")
        for pattern, replacement in _BRITISH:
            word = pattern.sub(replacement, word)
        words.append(word)
    return "".join(words).replace("-", "")


def _plural_map(vocabulary: set[str]) -> dict[str, str]:
    """`{plural: singular}` for the terms ending in "s" whose singular is also in `vocabulary`."""
    # Folding only onto a form the corpus contains keeps a word that merely ends in "s" ("process") as it is.
    return {term: term[:-1] for term in vocabulary
            if term.endswith("s") and not term.endswith(("ss", "ics")) and term[:-1] in vocabulary}


def casing_table(extracted: Sequence[dict]) -> dict[str, str]:
    """How the corpus capitalizes its words: `{lowercased word: form}`, for `recase`.

    `extracted`: `extract`'s output for every record of the corpus.

    Each record votes for the form it mostly writes a word in, away from the start of a sentence, and the form
    with the most votes wins, lowercase on a tie. So "Solomonoff" and "LoRA" keep their capitals, and a word
    some papers capitalize for emphasis does not.
    """
    votes: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for item in extracted:
        for word, form in item.get("casing", {}).items():
            votes[word][form] += 1
    return {word: max(c.items(), key=lambda kv: (kv[1], kv[0] == word))[0]
            for word, c in votes.items() if max(c.items(), key=lambda kv: (kv[1], kv[0] == word))[0] != word}


def recase(term: str, casing: dict[str, str]) -> str:
    """`term` capitalized as the corpus capitalizes its words (`casing_table`).

    A word the table knows takes the corpus's form. Another keeps its own, unless the term is in title case
    throughout ("Foundation Models") or capitalized as a sentence starts ("A posteriori error estimation"), as an
    LLM writes; then a word capitalized only by that is lowercased. A word with capitals past its first letter
    ("LLMs", "iOS") always keeps them.
    """
    words = term.split()
    title_cased = len(words) > 0 and all(w[:1].isupper() for w in words)

    def one(part: str, first: bool) -> str:
        if not part:
            return part
        if part.lower() in casing:
            return casing[part.lower()]
        if any(ch.isupper() for ch in part[1:]):
            return part
        return part.lower() if title_cased or first else part

    return " ".join("-".join(one(p, k == 0 and j == 0) for j, p in enumerate(w.split("-")))
                    for k, w in enumerate(words))


def normalize(extracted: Sequence[dict], *, short_form_share: float = 0.5) -> tuple[list[set[str]], dict[str, str]]:
    """Each record's terms in canonical form, with the form each is displayed in.

    `extracted`: `extract`'s output for every record of the corpus, in order.
    `short_form_share`: an abbreviation displays as its short form when at least this share of the records
                        using it leave it undefined; otherwise as its long form.

    Returns `(terms, display)`: `terms[i]` is the set of record `i`'s canonical terms, and `display` maps each
    canonical term to the spelling a label shows.

    What is folded together:

      - **A final word in "-ics"** onto "-ic", where the "-ic" form ends a phrase of the corpus, the head of a
        noun phrase: "a key characteristic" occurs, so "characteristics" is a plural, while nobody writes "a
        physic". A field of study is never folded ("kinetics", "dynamics"), though "first-order kinetic" occurs.
      - **A plural** onto its singular, where the corpus contains the singular. This catches what lemmatizing
        misses, a capitalized word read as a proper noun ("Large Language Models") or an acronym's plural.
      - **An abbreviation** with its long form, word by word, where the word is written as the abbreviation is,
        so "SRL" and "self-regulated learning" are one term, and "generative AI" and "generative artificial
        intelligence" another, while "Co" stays cobalt where "CO" is defined as carbon monoxide. A definition
        made by two or more records applies corpus-wide; one made by a single record applies to that record only.
      - **Spellings of one term**, where the corpus has more than one, onto the one most of its records use:
        a hyphen, a space or nothing between two words ("retrieval-augmented generation", "retrieval augmented
        generation"; "bit-width", "bitwidth"), a possessive ("prisoner's dilemma", "prisoner dilemma"), and a
        British and an American spelling ("modelling", "modeling"). This also joins "re-cover" and "recover",
        and "metre" and "meter", a price paid knowingly, being rare.

    The display form is capitalized as the corpus capitalizes its words (`casing_table`, `recase`).
    """
    # Why a single record's definition stays local: a paper calling its system CARE must not take over the
    # word "care" everywhere else.
    heads_seen = {t.split()[-1] for item in extracted for t in item["phrases"]}

    def fold_ics(term: str) -> str:
        words = term.split()
        if (words[-1].endswith("ics") and words[-1][:-1] in heads_seen and
                not words[-1].endswith(_ICS_MASS_NOUNS)):
            words[-1] = words[-1][:-1]
        return " ".join(words)

    raw = [{t: fold_ics(t) for t in item["phrases"]} for item in extracted]
    folds = _plural_map({c for record in raw for c in record.values()})

    # A definition, `(short, long, short as written)`, applies to a word written as its short form is.
    definitions = [list(item["definitions"]) for item in extracted]
    counts = collections.defaultdict(collections.Counter)
    for triples in definitions:
        for short, long, written in set(triples):
            counts[(short, written)][long] += 1
    corpus_map = {key: longs.most_common(1)[0][0] for key, longs in counts.items() if sum(longs.values()) >= 2}

    def expand(term: str, written: str, mapping: dict[tuple[str, str], str]) -> str:
        words, as_written = term.split(), written.split()
        if len(as_written) != len(words):  # cannot happen by construction; then match on the term alone
            as_written = words
        # A long form is stored lowercased, so a short form nested in one ("generative ai") matches on its letters.
        nested = {short: long for (short, _written), long in mapping.items()}
        out = []
        for word, form in zip(words, as_written):
            if len(form) > 2 and form.endswith("s") and form[:-1].isupper():  # "LLMs"
                form = form[:-1]
            replacement = mapping.get((word, form), word)
            if replacement != word:  # one level of nesting: "GenAI" -> "generative ai" -> "... artificial intelligence"
                replacement = " ".join(nested.get(w, w) if w != word else w for w in replacement.split())
            out.append(replacement)
        return " ".join(out)

    expanded = []
    for item, record, triples in zip(extracted, raw, definitions):
        mapping = dict(corpus_map)
        mapping.update({(short, written): long for short, long, written in triples})
        cased = item.get("cased", {})
        expanded.append({term: expand(folds.get(c, c), cased.get(term, term), mapping) for term, c in record.items()})
    # Fold plurals again: a long form arrives as written, and "Large Language Models (LLMs)" is plural.
    folds = _plural_map({c for record in expanded for c in record.values()})

    # Then hyphens and joins: each spelling class goes to its most common member, counted in records. Last, so
    # that it sees the plural and abbreviation folds' output, where variants that differed in more than that
    # have already met.
    record_counts = collections.Counter(c for record in expanded for c in {folds.get(c, c) for c in record.values()})
    by_spelling = collections.defaultdict(list)
    for c in record_counts:
        by_spelling[_spelling_class(c)].append(c)
    hyphen_folds = {c: max(members, key=lambda m: (record_counts[m], m))
                    for members in by_spelling.values() if len(members) > 1 for c in members}

    surface: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    short_use: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])  # [records defining it, records not]
    terms_out = []
    corpus_shorts = {short for short, _written in corpus_map}
    for record, triples in zip(expanded, definitions):
        defined_here = {short for short, _long, _written in triples}
        canonical = set()
        for term, c in record.items():
            c = folds.get(c, c)
            c = hyphen_folds.get(c, c)
            surface[c][term] += 1
            canonical.add(c)
            if term in corpus_shorts or term in defined_here:  # the bare short form of an abbreviation
                short_use[c][0 if term in defined_here else 1] += 1
        terms_out.append(canonical)

    # An abbreviation the corpus uses unexplained is one its readers know ("chatgpt"); one it defines wherever
    # it appears is one they would not ("fwp"), so that one shows spelled out ("fast weight programmer").
    casing = casing_table(extracted)
    display = {}
    for c, forms in surface.items():
        defining, undefined = short_use.get(c, (0, 0))
        if defining and undefined < short_form_share * (defining + undefined):
            long_forms = [(n, f) for f, n in forms.items() if " " in f or "-" in f]
            display[c] = max(long_forms)[1] if long_forms else c
        else:
            display[c] = forms.most_common(1)[0][0]
        display[c] = recase(display[c], casing)
    return terms_out, display


# --------------------------------------------------------------------------------
# Ranking

def _group_counts(terms: Sequence[set[str]], groups: Sequence[Hashable | None]):
    """Per group, how many of its records contain each term, and the group's size."""
    tf: dict[Hashable, collections.Counter] = collections.defaultdict(collections.Counter)
    size: collections.Counter = collections.Counter()
    for record_terms, group in zip(terms, groups):
        if group is not None:
            tf[group].update(record_terms)
            size[group] += 1
    return tf, size


def rank_terms(terms: Sequence[set[str]], groups: Sequence[Hashable | None], *,
               min_support: int = 2) -> dict[Hashable, list[tuple[str, float, float]]]:
    """Each group's terms ranked by class-based TF-IDF, as `{group: [(term, weight, share), ...]}`, best first.

    `terms`: each record's terms, as `normalize` returns them.
    `groups`: each record's group; `None` for a record in no group, which still counts toward the corpus.
    `min_support`: a term must occur in at least this many of a group's records to be ranked for it.

    The weight of term t in group g is tf(t, g) · log(1 + A / f(t)), where tf(t, g) is the number of g's
    records containing t, f(t) the number of the corpus's records containing it, and A the average number of
    term occurrences per group (Grootendorst 2022). `share` is tf(t, g) / |g|. Ties break by term.
    """
    # Counting records rather than occurrences keeps one abstract repeating a word from deciding its group's
    # label, and is what makes the share a share.
    corpus_df: collections.Counter = collections.Counter()
    for record_terms in terms:
        corpus_df.update(record_terms)
    tf, size = _group_counts(terms, groups)
    if not tf:
        return {}
    average = sum(sum(counts.values()) for counts in tf.values()) / len(tf)
    ranked = {}
    for group, counts in tf.items():
        scored = [(term, count * math.log(1.0 + average / corpus_df[term]), count / size[group])
                  for term, count in counts.items() if count >= min_support]
        scored.sort(key=lambda item: (-item[1], item[0]))
        ranked[group] = scored
    return ranked


def umbrella_terms(terms: Sequence[set[str]], groups: Sequence[Hashable | None], *,
                   min_support: int = 2, threshold_fraction: float = 0.1) -> set[str]:
    """Terms present in too many groups to say how any one of them stands out.

    A term is present in a group when at least `min_support` of the group's records contain it, and too common
    when present in at least max(min(5, n), ceil(threshold_fraction n)) of the n groups. With fewer than two
    groups there is nothing to stand out from, and no term is.
    """
    tf, _size = _group_counts(terms, groups)
    n = len(tf)
    if n < 2:
        return set()
    threshold = max(min(5, n), math.ceil(threshold_fraction * n))
    presence: collections.Counter = collections.Counter()
    for counts in tf.values():
        presence.update(term for term, count in counts.items() if count >= min_support)
    return {term for term, count in presence.items() if count >= threshold}


def fuse(rankings: Sequence[Sequence[tuple]], *, k: int = 60) -> list[tuple]:
    """Reciprocal rank fusion (Cormack, Clarke and Buettcher 2009) of rankings whose items start with a term.

    Each term scores the sum of 1 / (k + rank) over the rankings holding it, ranks counted from 1, and the
    fused ranking orders the items by that score, ties by term. k = 60 is the authors'.
    """
    score: collections.Counter = collections.Counter()
    item_of = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            score[item[0]] += 1.0 / (k + rank)
            item_of[item[0]] = item
    return [item_of[t] for t, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))]


# --------------------------------------------------------------------------------
# The label

def _nested(a: str, b: str) -> bool:
    """Whether one phrase is the other's word sequence or a contiguous part of it."""
    wa, wb = a.split(), b.split()
    if len(wa) > len(wb):
        wa, wb = wb, wa
    return any(wb[i:i + len(wa)] == wa for i in range(len(wb) - len(wa) + 1))


def _pick(ranked: Sequence[tuple[str, float, float]], top: int, *,
          fixed: Sequence[tuple[str, float]] = (), keep_longer: float = 0.5) -> list[tuple[str, float]]:
    """Up to `top` terms of `ranked` as `(term, share)`, two terms nested in each other counting as one.

    Of a nested pair, the shorter covers at least as much of the group, every record holding "latin hypercube
    sampling" also holding "sampling", so ranking alone keeps the shorter and loses the name. A longer phrase
    therefore replaces a shorter one already taken when it covers at least `keep_longer` of what the shorter
    one does.

    `fixed`: terms already taken elsewhere, which a candidate nested in one of them is skipped for and which
             are never replaced. They are not part of the result.
    """
    chosen: list[tuple[str, float]] = []
    if top <= 0:
        return chosen
    for term, _weight, share in ranked:
        if any(_nested(term, other) for other, _ in fixed):
            continue
        clash = next((j for j, (other, _) in enumerate(chosen) if _nested(term, other)), None)
        if clash is not None:
            other, other_share = chosen[clash]
            if (len(term.split()) > len(other.split()) and share >= keep_longer * other_share and
                    not any(_nested(term, x) for j, (x, _) in enumerate(chosen) if j != clash)):
                chosen[clash] = (term, share)
            continue
        chosen.append((term, share))
        if len(chosen) == top:
            break
    return chosen


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def label_groups(terms: Sequence[set[str]], groups: Sequence[Hashable | None], *,
                 display: dict[str, str],
                 record_vectors: np.ndarray,
                 embed_terms: Callable[[list[str]], np.ndarray],
                 top: int = 30, shown: int = 6, pool: int = 30,
                 umbrella_fraction: float = 0.1,
                 min_support: int = 2,
                 exclude: Container[str] = raven_stopwords.bare_keyword_stopwords
                 ) -> dict[Hashable, list[tuple[str, float]]]:
    """Each group's label: up to `top` terms, best first, as `{group: [(display term, share), ...]}`.

    `terms`, `display`: as `normalize` returns them.
    `groups`: each record's group; `None` for a record in no group.
    `record_vectors`: one embedding per record, in the order of `terms`; a group's centroid is their mean.
    `embed_terms`: embeds a list of terms in the same space as `record_vectors`, one row each.
    `top`: how many terms to return per group.
    `shown`: how many of them a label shows. The first `shown` are picked as a label on their own, and the rest
             continue the list without changing them, so the list always begins with the label.
    `pool`: how many of a group's distinctive terms the tail is chosen from.
    `umbrella_fraction`: the `threshold_fraction` of `umbrella_terms`.
    `min_support`: as for `rank_terms`.
    `exclude`: terms never given, though phrases containing them may be; by default the words that say
               nothing as a keyword on their own (`raven.common.text.stopwords.bare_keyword_stopwords`).

    The first term is the group's best by class-based TF-IDF, saying what the group is; in a corpus on one
    subject it is often a term other groups share. The rest come from the terms that say how the group stands
    out: its ranking with the umbrella terms (`umbrella_terms`) removed, cut to `pool`, ordered once by that
    ranking and once by each term's cosine similarity to the group's centroid, and the two orders fused
    (`fuse`). Two terms nested in each other count as one.
    """
    # The two orders fail in opposite places: the ranking alone fills an abstract group's label with the
    # ordinary words most of it uses ("interest, world"), and the similarity alone reaches for narrow terms in
    # a technical one. A term high in both survives the fusion; one high in only one does not.
    ranked = rank_terms(terms, groups, min_support=min_support)
    umbrella = umbrella_terms(terms, groups, min_support=min_support, threshold_fraction=umbrella_fraction)

    members: dict[Hashable, list[int]] = collections.defaultdict(list)
    for idx, group in enumerate(groups):
        if group is not None:
            members[group].append(idx)

    shown = min(shown, top)
    heads, pools = {}, {}
    for group in members:
        candidates = [item for item in ranked.get(group, []) if item[0] not in exclude]
        heads[group] = _pick(candidates, min(1, top))
        head_terms = [t for t, _ in heads[group]]
        pools[group] = [item for item in candidates
                        if item[0] not in umbrella and not any(_nested(item[0], t) for t in head_terms)][:pool]

    to_embed = sorted({t for items in pools.values() for t, _, _ in items})
    term_vectors = dict(zip(to_embed, _unit(np.asarray(embed_terms(to_embed), dtype=np.float32)))) if to_embed else {}

    labels = {}
    for group, idxs in members.items():
        by_rank = pools[group]
        if by_rank:
            centroid = _unit(np.asarray(record_vectors[idxs], dtype=np.float32).mean(axis=0))
            similarity = np.stack([term_vectors[t] for t, _, _ in by_rank]) @ centroid
            by_similarity = [by_rank[k] for k in np.argsort(-similarity, kind="stable")]
            fused = fuse([by_rank, by_similarity])
            tail = _pick(fused, shown - len(heads[group]), fixed=heads[group])
            taken = {t for t, _ in tail}
            tail += _pick([item for item in fused if item[0] not in taken], top - len(heads[group]) - len(tail),
                          fixed=heads[group] + tail)
        else:
            tail = []
        labels[group] = [(display.get(t, t), share) for t, share in heads[group] + tail]
    return labels


def top_terms(record_terms: Sequence[set[str]], top: int, *,
              exclude: Container[str] = raven_stopwords.bare_keyword_stopwords) -> list[tuple[str, int]]:
    """The most common terms of a set of records, as `[(term, count), ...]`, most common first.

    For a summary of an arbitrary selection, such as a word cloud: `count` is how many of the records contain the
    term. Two terms nested in each other count as one, by the rule a group's label uses, so the two agree: a
    longer phrase stands in for a shorter one it contains where it is found in at least half as many records.

    `record_terms`: the selected records' terms, as `normalize` returns them.
    `top`: how many terms to return at most.
    `exclude`: as for `label_groups`.
    """
    counts: collections.Counter = collections.Counter()
    for terms in record_terms:
        counts.update(terms)
    ranked = sorted(((term, count, count) for term, count in counts.items() if term not in exclude),
                    key=lambda item: (-item[1], item[0]))
    return [(term, int(count)) for term, count in _pick(ranked, top)]


def _comparable(term: str) -> str:
    """`term` as compared for repeats: lowercased, hyphens as spaces, the last word's plural "s" dropped."""
    words = term.lower().replace("-", " ").split()
    if words and words[-1].endswith("s") and not words[-1].endswith(("ss", "ics")):
        words[-1] = words[-1][:-1]
    return " ".join(words)


def without_umbrella_terms(lists: Sequence[Sequence[str]], *, threshold_fraction: float = 0.1) -> list[list[str]]:
    """Each of `lists` without the terms too many of the lists share, as `umbrella_terms` counts them.

    `lists`: one list of terms per group, such as an LLM's keywords for each cluster.

    A term is shared by a list when the list holds it, compared without case, hyphens or a final plural "s". So
    *large language models*, which an LLM names for most clusters of a corpus about them, goes from every list,
    leaving what tells the clusters apart. Order is kept.
    """
    groups = list(range(len(lists)))
    umbrella = umbrella_terms([{_comparable(t) for t in terms} for terms in lists], groups,
                              min_support=1, threshold_fraction=threshold_fraction)
    return [[t for t in terms if _comparable(t) not in umbrella] for terms in lists]


def shared_last(lists: Sequence[Sequence[str]], *, keep: int = 1) -> list[list[str]]:
    """Each of `lists` with the terms other lists share too moved to its end, except the first `keep` of them.

    Terms are compared as `without_umbrella_terms` compares them, and order is kept otherwise. So where an LLM
    gives *Education for Sustainable Development* and *curriculum* to several clusters, a cluster keeps the
    first as its lead and shows its own *SDG integration* before the second, and a label cut short shows what
    tells the clusters apart.
    """
    support = collections.Counter(c for terms in lists for c in {_comparable(t) for t in terms})
    out = []
    for terms in lists:
        own, kept, moved = [], 0, []
        for t in terms:
            if support[_comparable(t)] > 1:
                if kept >= keep:
                    moved.append(t)
                    continue
                kept += 1
            own.append(t)
        out.append(own + moved)
    return out


def lead_with(leading: Sequence[str], ranked: Sequence[tuple[str, float]], top: int, *,
              display: dict[str, str] | None = None, nested: bool = True) -> list[str]:
    """A label of at most `top` terms: `leading`, then terms of `ranked` that add something new.

    `leading`: terms to put first, as written, such as an LLM's keywords.
    `ranked`: `(term, share)` pairs, best first, such as one group's list from `label_groups`.
    `display`: `normalize`'s map from canonical terms to their display forms, when `ranked` holds display forms.
               Each term is then compared in both forms, so an LLM's *low-rank adaptation* is a repeat of a
               phrase displayed as *lora*, and the other way round.
    `nested`: whether a term nested in one already taken, or one in it, counts as a repeat. True suits one label,
              where *deep reinforcement learning* after *reinforcement learning* spends a slot on little; False
              keeps only exact repeats out, for two lists shown side by side, where a one-word term such as
              *intelligence* must not take every phrase containing it.

    A term of `ranked` is skipped when it repeats a term already taken, or is nested in one, or one in it,
    compared without case, hyphens or a final plural "s": after *reinforcement learning*, neither
    *Reinforcement Learning* nor *deep reinforcement learning* adds a term.
    """
    other_form = {}
    for canonical, shown in (display or {}).items():
        if canonical != shown:
            other_form.setdefault(_comparable(shown), _comparable(canonical))
            other_form.setdefault(_comparable(canonical), _comparable(shown))

    def forms(term: str) -> list[str]:
        c = _comparable(term)
        return [c, other_form[c]] if c in other_form else [c]

    label = list(leading[:top])
    taken = [f for t in label for f in forms(t)]
    for term, _share in ranked:
        if len(label) >= top:
            break
        if any(c == t or (nested and _nested(c, t)) for c in forms(term) for t in taken):
            continue
        label.append(term)
        taken.extend(forms(term))
    return label

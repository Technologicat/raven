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

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))  # run from anywhere; import the tree this script is in

from raven.common import nlptools  # noqa: E402 -- after the path setup
from raven.common.text import stopwords as raven_stopwords  # noqa: E402 -- after the path setup

STOPWORDS = set(nlptools.default_stopwords) | {x.lower() for x in raven_stopwords.academic_stopwords}

# Candidates for a list of words useless as a bare one-word keyword, while useful inside a phrase:
# "potential" against "potential flow", "field" against "field theory". Applied only to a single-word
# candidate, never to a phrase's edges. Proposed from the labels of six corpora, 2026-11-20, not yet adopted.
BARE_STOPWORDS = {"background", "challenge", "change", "concept", "concern", "conclusion", "effect",
                  "example", "field", "framework", "future", "issue", "level", "limitation", "method",
                  "outcome", "paper", "participant", "people", "potential", "problem", "strategy", "topic"}

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

    Returns the candidates in canonical form, and `{canonical: display}`, where the display form is the
    spelling the corpus uses most, so a label says "chatgpt" rather than its expansion.
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
    out = []
    for record in expanded:
        canonical = set()
        for term, c in record.items():
            c = folds.get(c, c)
            surface[c][term] += 1
            canonical.add(c)
        out.append(canonical)
    display = {c: forms.most_common(1)[0][0] for c, forms in surface.items()}
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
        scored.sort(key=lambda item: -item[1])
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
                                  exclude=frozenset(BARE_STOPWORDS)) for cid in members},
        "phrase-mix": {cid: pick_mixed(phrase_ab_ranked.get(cid, []), args.top, n_common=2, umbrella=umbrella,
                                       bare=frozenset(BARE_STOPWORDS)) for cid in members},
    }
    labels_canonical = {name: {cid: [kw for kw, _ in kws] for cid, kws in labelling.items()} for name, labelling in methods.items()}
    for name in ("phrase-ab", "phrase-cut", "phrase-bare", "phrase-mix"):
        methods[name] = {cid: [(display.get(kw, kw), share) for kw, share in kws] for cid, kws in methods[name].items()}
    coverage_space = {"stored": stored_space, "unigram": "unigram", "phrase-ab": "phrase-ab", "phrase-cut": "phrase-ab", "phrase-bare": "phrase-ab", "phrase-mix": "phrase-ab"}

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

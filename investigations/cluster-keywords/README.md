# Cluster keywords without an LLM

Whether a classical keyworder can label the Visualizer's clusters well: noun phrases as candidates, scored
by class-based TF-IDF, in place of single nouns counted under a 10% cutoff. This is the first two steps of
`briefs/visualizer-keyword-pools-brief.md`, *An improved classical mode, worth trying*. The third, the
embedding rerank, is not yet tried.

It bears on more than the cluster labels: per-year keywords for colouring by year
(`briefs/visualizer-filter-by-date-and-recolour-brief.md`) need a method that scales to a year holding
thousands of records, where the LLM keyworder's prompt does not.

## The script

- **`keyword_probe.py DATASET.pickle`**: reads a dataset as the importer wrote it, extracts candidate terms
  afresh with spaCy (`en_core_web_sm`, CPU), and writes `out/<dataset>.md`, setting the methods below side by
  side: a summary table, then the largest clusters and a random draw of the rest, each keyword with its share
  of the cluster and three member titles. Candidates are cached in `cache/`; both are gitignored, as they
  hold corpus text.

## The methods compared

| name | candidates | scoring |
|---|---|---|
| `stored` | what the dataset holds: the importer's frequency keywords | count, words in ≥10% of clusters dropped |
| `unigram` | nouns and proper nouns, lemmatized | c-TF-IDF |
| `phrase` | noun phrases with their head-final sub-phrases | c-TF-IDF, nested phrases suppressed |
| `phrase-kl` | as `phrase` | share × log(lift), the term's KL contribution |
| `phrase-ab` | as `phrase`, abbreviations merged with their long forms | c-TF-IDF, nested phrases suppressed |

**c-TF-IDF** is Grootendorst's class-based TF-IDF from BERTopic ([2022](https://arxiv.org/abs/2203.05794)),
W(t, c) = tf(t, c) · log(1 + A / f(t)), here on record counts rather than occurrences: tf is how many of the
cluster's records contain the term, so tf / |c| is the share of the cluster it covers, which the report shows
beside each keyword. A term must occur in two records of a cluster to be ranked for it.

**Abbreviations** are found with Schwartz and Hearst's algorithm
([2003](https://psb.stanford.edu/psb-online/proceedings/psb03/schwartz.pdf)) and merged with their long form,
inside phrases too, displayed in whichever spelling the corpus uses more. A definition made by two or more
records applies corpus-wide, one made by a single record only to that record.

## Results, 2026-11-20

Four corpora in three fields: hydrogen production (11,974 records, 390 clusters), AOKK (5,007, 183),
ECCOMAS 2024 (2,519, 116), and a small mechanics collection of mostly titles (531, 24). Six keywords per
cluster. *Distinct* is the share of label slots holding a keyword no other slot holds; *coverage* the share
of a cluster's records containing at least one of its keywords, averaged over clusters.

| corpus | `stored` distinct / coverage | `unigram` | `phrase` | `phrase-kl` | `phrase-ab` |
|---|---|---|---|---|---|
| hydrogen | 51% / 76% | 66% / 85% | 83% / 78% | 61% / 92% | 83% / 78% |
| AOKK | 71% / 36% | 58% / 88% | 67% / 81% | 54% / 87% | 70% / 81% |
| ECCOMAS | 77% / 62% | 68% / 90% | 79% / 85% | 77% / 87% | 79% / 86% |
| mechanics | 89% / 25% | 81% / 84% | 88% / 75% | 86% / 76% | 88% / 75% |

**Read the labels, not only the table.** The table cannot see the difference that matters most: on hydrogen,
`stored` labels a cluster *pome, effluent, palm, mill, hrt, thermoanaerobacterium*, and `phrase-ab` labels
it *palm oil mill effluent, chemical oxygen demand, cod removal, response surface methodology*; *iodine,
thermochemical, bunsen, nuclear* becomes *bunsen reaction, iodine-sulfur process, hydrogen iodide*.

- **Phrases are the large lever**, as the brief expected: names of methods, processes and materials arrive
  whole. The phrase methods are the most distinctive on hydrogen and ECCOMAS; on AOKK, `phrase-ab` is level
  with `stored` (70% against 71%) at more than twice its coverage.
- **c-TF-IDF lifts coverage over the stored method everywhere**, most on AOKK. There, the 10% cutoff drops
  every term a large cluster's records share (on one cluster of 69 papers on LLMs, the stored keywords cover
  between 0% and 9% of it each), leaving the label to leftovers.
- **`phrase-kl` was rejected.** Expected to penalize a single-topic corpus's umbrella terms; it does the
  opposite, a share times a log favouring coverage, and it is the least distinctive phrase method on every
  corpus.
- **Abbreviation merging frees the slots an acronym spends beside its expansion** (*srl* and
  *self-regulated learning*, *genai* and *generative ai*, *pome*), which the distinct share understates, since
  a duplicate pair within one label is not a repeat across clusters.

Fixes found on the way, each from a real label rather than an invented case:

- **A Title Case title is sentence-cased first**: the tagger reads capitalized words as proper nouns and
  leaves them unlemmatized, so *Rectangular Plates* was another term than *rectangular plate*.
- **Only a phrase's head is lemmatized**: lemmatizing modifiers made *moving plate* "move plate".
- **A longer phrase replaces a nested shorter one** when it covers at least half as much: the shorter always
  covers more, so ranking alone kept *sampling* and lost *Latin hypercube sampling*. The half is a choice,
  not a measurement.
- **Plurals fold onto a singular the corpus contains**, which catches unlemmatized capitals and acronym
  plurals (*LLMs*); text is NFKC-normalized (*TiO₂*).
- **A short form needs two capitals or a digit**, an addition to Schwartz and Hearst: "(oxygen)" after a
  description defined the word *oxygen*, and expanded every phrase containing it.

## Umbrella terms and the two pools, 2026-11-20

Run again on six datasets, the latest of each where it exists: hydrogen (`out-2026-11`), the latest AOKK
(2,032 records, 74 clusters), ECCOMAS, and the three education corpora, biology (165 records, 6 clusters),
geography (128, 7; stored keywords from the LLM) and health education (40, 4; LLM theme names). Three more
variants, all on the `phrase-ab` scores:

- `phrase-cut`: the maintainer's original rule, the one `nlptools.suggest_keywords` applies, with clusters as
  documents: a term present (in two or more records) in at least max(min(5, n), ⌈0.1 n⌉) clusters is dropped.
- `phrase-bare`: a list of words dropped only as a bare one-word keyword, never inside a phrase (*potential*
  against *potential flow*). Candidates in the script's `BARE_STOPWORDS`, not yet adopted.
- `phrase-mix`: the keyword-pools brief's two pools from one set of scores — the best two terms of the full
  ranking, then four with the umbrella terms removed.

| corpus | `stored` distinct / coverage | `phrase-ab` | `phrase-cut` | `phrase-mix` |
|---|---|---|---|---|
| hydrogen | 51% / 76% | 83% / 78% | 84% / 75% | 84% / 77% |
| AOKK, latest | 85% / 64% | 82% / 92% | 94% / 71% | 92% / 84% |
| ECCOMAS | 77% / 62% | 79% / 86% | 86% / 76% | 84% / 81% |
| biology | 81% / 77% | 64% / 99% | 94% / 85% | 86% / 96% |
| geography | 71% / 48% | 64% / 100% | 98% / 90% | 88% / 97% |
| health education | — | 83% / 100% | 88% / 100% | 88% / 100% |

(Geography's stored coverage is a literal match of LLM keywords, which undercounts them.)

- **On a single-topic corpus, each pure method fails one way.** `phrase-ab` fills a label with the corpus's
  own vocabulary (*education, student, tool*); `phrase-cut` removes the term that defines the cluster
  (*self-regulated learning*, *generative ai*) and leaves a tail at a few percent of coverage each.
- **`phrase-mix` reads best of the four**: *self-regulated learning, self-regulation, forethought,
  monitoring, scaffolding, self-reflection*; *intelligent tutoring system, learning process, knowledge
  tracing, neural network, knowledge state*. Its second common slot is often generic (*student*, *tool*),
  so one common term may be the better split.
- **`phrase-bare` barely moves the statistics**, as expected: a dropped word makes room for the next.
- **The very small corpora stay weak** under every method, a few clusters of tens of records each giving
  c-TF-IDF little to contrast.

## Open

- **Generic nouns remain**: *framework*, *limitation*, *participant*, structured-abstract headings
  (*background*, *conclusion*). Several have a useful sense inside a phrase, which `raven.common.text.stopwords`
  records (*potential flow*, *field theory*), so the candidate list applies to bare terms only. To be checked
  word by word before adoption.
- **"state of the art" yields *art***, and *GIS* did not merge with *geographical information system*.
- **Not yet measured against the LLM keyworder**, which the brief asks for; none of these datasets stores
  LLM keywords.
- **The embedding rerank with MMR**, the brief's third step.

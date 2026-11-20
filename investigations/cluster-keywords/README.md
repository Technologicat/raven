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

## Open

- **Generic nouns remain**: *review*, *potential*, *field*, *limitation*, *framework*, *participant*, and
  structured-abstract headings (*background*, *conclusion*). A stopword question, or the embedding rerank's.
- **The umbrella terms of a single-topic corpus** (*generative ai* in 22 of AOKK's 183 labels) are
  distinctive against nothing, the whole corpus being about them.
- **Not yet measured against the LLM keyworder**, which the brief asks for; none of these datasets stores
  LLM keywords.
- **The embedding rerank with MMR**, the brief's third step.

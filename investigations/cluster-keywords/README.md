# Cluster keywords without an LLM

Whether a classical keyworder can label the Visualizer's clusters well: noun phrases as candidates, scored
by class-based TF-IDF, in place of single nouns counted under a 10% cutoff. This is the first two steps of
`briefs/visualizer-keyword-pools-brief.md`, *An improved classical mode, worth trying*. The third, the
embedding rerank, is not yet tried.

It bears on more than the cluster labels: per-year keywords for colouring by year
(`briefs/visualizer-filter-by-date-and-recolour-brief.md`) need a method that scales to a year holding
thousands of records, where the LLM keyworder's prompt does not.

## The script

- **`measure_centroids.py DATASET.pickle ...`**: whether the rerank favours terms many clusters share,
  from the caches `keyword_probe.py --rerank` leaves; embeds nothing.
- **`run_llm_imports.sh`**: imports the test corpora with LLM cluster keywords, for the comparison.
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
  against *potential flow*). Adopted as `raven.common.text.stopwords.bare_keyword_stopwords`, after checking each word for a useful
  sense standing alone (*future*, *method* and *paper* have one, and stay off it).
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

## The embedding rerank, 2026-11-20

`--rerank` adds the brief's third step: a pool of each cluster's 30 best terms by c-TF-IDF (bare-keyword
stopwords removed) is scored by cosine similarity to the cluster's centroid in the importer's embedding
space, then picked by maximal marginal relevance ([Carbonell & Goldstein,
1998](https://doi.org/10.1145/290941.291025)), after KeyBERT (Grootendorst 2020,
[repository](https://github.com/MaartenGr/KeyBERT)). The diversity weight is 0.5, a choice. Two shapes, each
raw and *centered* (documents on the corpus mean, terms on the mean of all pooled terms, so a term scores by
where this cluster differs from the corpus):

- `rerank`, `rerank-c`: the whole label from the reranked pool.
- `mix-rerank`, `mix-rerank-c`: one term from c-TF-IDF, then the distinctive pool (umbrella terms removed)
  reranked. `phrase-mix1` is the same split without the rerank.

| corpus | `phrase-mix1` distinct / coverage | `rerank` | `rerank-c` | `mix-rerank` | `mix-rerank-c` |
|---|---|---|---|---|---|
| hydrogen | 84% / 76% | 73% / 71% | 77% / 76% | 74% / 76% | 78% / 77% |
| AOKK, latest | 94% / 81% | 66% / 89% | 75% / 91% | 90% / 81% | 93% / 82% |
| ECCOMAS | 85% / 78% | 84% / 69% | 82% / 79% | 85% / 73% | 87% / 75% |
| biology | 92% / 92% | 75% / 90% | 92% / 91% | 81% / 90% | 94% / 88% |
| geography | 93% / 93% | 71% / 90% | 88% / 96% | 90% / 93% | 93% / 94% |
| health education | 88% / 100% | 92% / 97% | 96% / 100% | 100% / 99% | 100% / 99% |

- **The rerank alone reads worse on AOKK and hydrogen**: AOKK's labels fill with *learning process*,
  *educator*, *task*, and hydrogen's repeat *photocatalytic hydrogen production* in 15 clusters. Why is
  not settled; see the measurement below, which rejects the first explanation offered.
- **On the distinctive pool it does what it was for**: the tail stops picking noise for its rarity. AOKK's
  *gender*, *effect size*, *base* give way to *technology acceptance model*, *bayesian knowledge tracing*,
  *learner model*, *prompt engineering*, *educational data mining*; geography's tail becomes *spatial
  thinking skill*, *geography teaching*, *sustainability competency*, *inquiry-based learning*, close to its
  LLM keywords.
- **The cost is coverage within the tail**: the rerank prefers specific phrases at 3–5% of a cluster, and
  sometimes drops a good term, as SRL's phases *forethought* and *monitoring*.
- **Not settled by the table.** The labels have to be read, and the LLM imports (`run_llm_imports.sh`,
  running) are for comparing against.

### Does the rerank favour generic terms? (`measure_centroids.py`)

The explanation first written here was that a single-topic corpus's cluster centroids lie near the corpus's
generic phrases, so similarity to a centroid rewards them. Measured on the six corpora: *centroid closeness*
is the mean cosine between each cluster's centroid and the corpus mean; *generality bias* the Spearman
correlation, within each cluster's pool of 30, between a term's similarity to the centroid and the number
of clusters it is present in; *clusters per pick* that number averaged over the six terms each method picks.

| dataset | centroid closeness | generality bias, raw / centered | clusters per pick: c-TF-IDF / rerank / rerank centered |
|---|---|---|---|
| hydrogen | 0.843 | −0.06 / −0.05 | 7.4 / 9.1 / 13.8 |
| AOKK, latest | 0.884 | −0.15 / −0.31 | 16.5 / 17.2 / 19.7 |
| ECCOMAS | 0.854 | −0.29 / −0.29 | 11.9 / 5.4 / 11.9 |
| biology | 0.944 | +0.17 / −0.15 | 4.8 / 3.9 / 3.8 |
| geography | 0.934 | +0.21 / −0.16 | 5.2 / 3.7 / 3.6 |
| health education | 0.945 | +0.03 / −0.13 | 3.0 / 1.7 / 2.5 |

- **The explanation does not hold in general.** Within a pool, the terms nearest the centroid tend to be the
  *less* shared ones on every large corpus; the rerank's picks are more generic than c-TF-IDF's on
  hydrogen, about as generic on AOKK, and less generic on the other four.
- **Centroid closeness separates the corpora little**: AOKK, the single-topic one, is highest of the large
  three, by a few hundredths. The small corpora score highest because each of their few clusters is a large
  share of the corpus.
- **Centering makes the picks more generic on hydrogen and AOKK**, the opposite of its intent.

## The hybrids, 2026-11-23

Read on the arXiv corpus, the five largest clusters (the maintainer, 2026-11-23): no one method won.
`phrase-mix1` was acceptable or best on four, `mix-rerank` on three or four, and they failed on different
clusters. `phrase-mix` fails on an abstract cluster, where the terms covering most of it are ordinary words
(*interest, world, machine, definition* for papers on AGI and AI risk); `mix-rerank` fails on a concrete
technical one, reaching for narrow terms (it missed *verifiable reward* in the RLVR cluster) or, centered,
unexplained ones (*fwp*).

So two hybrids, each one common term and then the distinctive pool ordered two ways, by c-TF-IDF and by
similarity to the cluster's centroid:

- `hybrid-alt`: the two orderings taken in turn.
- `hybrid-rrf`: reciprocal rank fusion ([Cormack, Clarke and Büttcher,
  2009](https://doi.org/10.1145/1571941.1572114)), a term scoring 1 / (60 + rank) in each ordering, so a term
  high in only one loses to a term fair in both.

`hybrid-rrf` on the same five: *reinforcement learning, reasoning capability, large reasoning model, llm
reasoning, reasoning performance, verifiable reward*; *diffusion model, generative model, stable diffusion,
image generation, image synthesis, gan*; *transformer, language modeling, recurrent model, recurrent neural
network, attention mechanism, linear attention*. 94% distinct at 82% coverage, as `phrase-mix1`.

Two fixes in the same round:

- **An abbreviation shows as its long form where most records using it define it**, and as its short form
  where most use it unexplained: *fwp* became *fast weight programmer*, *chatgpt* stays. The half is a choice,
  and a setting (`--short-form-share`).
- **The stored keywords' plurals are folded before matching**, as the candidates' are: the LLM writes
  *transformers* and *diffusion models*, which matched nothing. Its arXiv coverage is 70%, not 38%; the
  earlier tables here undercount it the same way.

### MMR over the fused ranking (`hybrid-rrf-mmr`)

Tried for the near-duplicates `hybrid-rrf` keeps (*recurrent model* beside *recurrent neural network*):
maximal marginal relevance over the fused ranking, relevance the rank scaled to [0, 1], diversity 0.3 (a
choice), with the common term counted as already chosen. **It changes almost nothing**: the duplicate pairs
stay, and where a pick moves it is no better (*synthesis* for *image synthesis*'s slot). The pairs are
close, at cosines of 0.94 (*recurrent model* / *recurrent neural network*) and 0.92 (*image generation* /
*image synthesis*), but distinct related terms are close too (0.77 for *consciousness* / *existential
risk*), so a stronger diversity weight chosen from these few pairs would be fitted to them. Whether
near-duplicates separate from related terms at all wants the distribution over every label first.

### One term per modifier (`hybrid-rrf-word`)

`hybrid-rrf` can spend a label on one facet: the RLVR cluster read *reinforcement learning, reasoning
capability, large reasoning model, llm reasoning, reasoning performance, reasoning ability*. `hybrid-rrf-word`
skips a candidate sharing a modifier word with a term already taken, the common term included, where a
modifier is a word before the head in either term (`shares_modifier`). Each run writes its firings to
`out/<dataset>-firings.txt`.

- **It fires often**: 104 times on arXiv, changing 36 of 68 labels; a third to two thirds of the labels on
  every large corpus.
- **The RLVR cluster becomes** *reinforcement learning, reasoning capability, verifiable reward, supervised
  fine-tuning, deepseek-r1, policy optimization*; the transformer cluster loses *recurrent neural network*
  beside *recurrent model* and gains *self-attention, sequence length*.
- **It costs a name where the shared word is part of one**: *stable diffusion* gives way to *diffusion
  model*, and the slot goes to the bare *synthesis*.
- **Some firings separate distinct terms**: *data analysis* / *data visualisation*, and *perceived ease of
  use* / *perceived usefulness*, two constructs of the technology acceptance model.

## Open

**Which method, not yet decided** (the maintainer, 2026-11-20): no clear winner among `phrase-mix1`,
`phrase-mix2`, `mix-rerank` and `mix-rerank-c`. The `phrase-mix` labels read more easily; the `mix-rerank`
ones find specific concepts, as *bayesian knowledge tracing*. Judging on AOKK is confounded, its clusters
being unfamiliar to the person reading them, so the next judgement is on the arXiv corpus, which the
maintainer knows:

1. `keyword_probe.py --rerank data/arxiv.pickle` (imported 2026-11-20 with LLM keywords, from
   `00_stuff/datasets/ai_papers/ai_papers_202611.bib`), and the maintainer reads the labels.
2. The same on the other LLM imports in `data/`, for the comparison against the LLM keyworder.
3. Then pick the method, and plan it into the importer; after that, colouring by year v0
   (`briefs/visualizer-filter-by-date-and-recolour-brief.md`).

- **The LLM keyworder given the classical terms as evidence**: the terms with their shares, beside the
  sampled abstracts, framed as data rather than as a suggestion, so that any anchoring pulls toward the
  corpus. The two fail in opposite directions (classical terms cannot abstract, as the evaluation cluster
  shows; the LLM's wander, as *AI safety* in the diffusion cluster), and the statistics cover the whole
  cluster where the prompt holds at most 60 records. Its bias wants measuring: the same clusters with and
  without the evidence. Importer work, not this probe's.
- **"state of the art" yields *art***, and *GIS* did not merge with *geographical information system*.
- **Against the LLM keyworder**: `run_llm_imports.sh` imports each corpus with LLM keywords on agglomerative
  clusters into `data/`, one corpus at a time, resuming after the last one finished.

# Cluster keywords without an LLM

Whether a classical keyworder can label the Visualizer's clusters well: noun phrases as candidates, scored
by class-based TF-IDF, in place of single nouns counted under a 10% cutoff. This is the first two steps of
`briefs/visualizer-keyword-pools-brief.md`, *An improved classical mode, worth trying*. The third, the
embedding rerank, is not yet tried.

It bears on more than the cluster labels: per-year keywords for colouring by year
(`briefs/visualizer-filter-by-date-and-recolour-brief.md`) need a method that scales to a year holding
thousands of records, where the LLM keyworder's prompt does not.

## The script

- **`try_context_embedding.py`**: the labels with each term embedded in context, beside its cluster's head term
  or as the cluster's sentences containing it, against the bare term.
- **`try_relative_similarity.py`**: the labels with the similarity order corrected per term, two ways, against
  the shipped order, on the same datasets.
- **`measure_single_words.py`**: whether the similarity order demotes single-word terms, on the six `-evidence`
  datasets and their embeddings caches, with each term's specificity as the control.
- **`measure_centroids.py DATASET.pickle ...`**: whether the rerank favours terms many clusters share,
  from the caches `keyword_probe.py --rerank` leaves; embeds nothing.
- **`run_new_imports.sh`**: each corpus again after the changes of 2026-11-24, with LLM keywords given the
  evidence and with hybrid keywords; resumable one import at a time, smallest corpora first. Before the first
  run, the previous evidence datasets are renamed `CORPUS-evidence-2026-11-23.pickle` (and `.log`), so that
  they are kept for comparison and not taken as done. `import_with_evidence.py --method hybrid` is what runs
  the hybrid half.
- **`run_llm_evidence_imports.sh`**: the same corpora with the LLM given the phrase evidence, through
  **`import_with_evidence.py`**, which switches `clusters_llm_keyword_evidence` for one run without touching the
  machine's configuration; and arXiv a second time without it, for the noise floor.
- **`compare_llm_runs.py CORPUS`**: the LLM's keywords with the evidence against without, by keyword overlap
  between runs and by how many of them come from the cluster's phrases, and side by side for reading.
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

## The LLM given the phrases as evidence, 2026-11-23

`clusters_llm_keyword_evidence` ends each cluster's prompt with its 30 phrases and their shares, framed as data
about the whole cluster and not as suggested keywords; the records the prompt holds are a sample when a cluster
is big, where the phrases count every record. The question is whether that grounds the LLM's labels, which
abstract well and sometimes seem to wander, without anchoring them onto the phrases, which cannot abstract.

- **Measured against the imports of 2026-11-20**, same clusters (agglomerative on cached embeddings), the
  prompt otherwise unchanged.
- **The noise floor first**: the LLM samples, so two runs of one prompt already differ. arXiv runs a second
  time without the evidence, and the evidence's effect is read against that difference.
- **arXiv** (68 clusters; `out/arxiv-llm-evidence.md`):

  | comparison | mean keyword overlap (Jaccard) |
  |---|---|
  | without, run twice (the noise floor) | 0.40 |
  | with against without | 0.38 |
  | with against the repeat | 0.40 |

  | run | keywords found among the cluster's phrases |
  |---|---|
  | without | 22% |
  | without, again | 21% |
  | with | 31% |

  - **Which keywords a label gets changes no more than between two runs without the evidence**: the overlap
    with it is the overlap of the noise floor. So no measurable wholesale anchoring.
  - **Its pull toward the phrases is modest**: a share of 31% against 21–22%, about half a keyword per label.
  - **The changes go where the earlier review found the LLM wandering.** The diffusion cluster loses the safety
    term both runs without gave it (*AI Safety*, *generative AI safety*) and gains *Stable Diffusion*, *image
    editing*, *generative models* — but that safety term turned out to be grounded (below); the distillation cluster loses the *reinforcement learning* of both runs
    without, for *model compression*; the attack cluster gains *backdoor attacks*. The evaluation cluster keeps
    its abstractions (*LLM-as-a-judge*, *theory of mind*) rather than taking the GPT-family terms the phrases
    lead with.
  - These are readings of a few clusters, not a measurement; the maintainer's read decides.
- **The diffusion cluster's *AI safety* is a real minority, not a wander** (checked 2026-11-24, on the maintainer's
  question). Five of its 51 records are about the safety and misuse of image generators: *Glaze* (protecting
  artists from style mimicry), *IMPRESS* (unauthorized data usage), *Red-Teaming the Stable Diffusion Safety
  Filter*, *Extracting Training Data from Diffusion Models*, and a paper on generative AI's feedback loop with the
  internet. The cluster is under 60 records, so the model saw all of it, and both runs' reasoning in the logs
  (`data/arxiv.log`, `data/arxiv-repeat.log`) lists these papers, counts them at five or six, and weighs safety
  against the four GAN papers for the last of six places.
  - **So the evidence pushed out an abstraction over different wordings in favour of frequent phrases.** The five
    say it as protection, privacy, style mimicry, a safety filter; no one phrase reaches a share the evidence list
    would show. That is the anchoring the measurement was looking for, in one label: the abstraction the LLM is
    there to make, traded for terms close to ones the label already has (*Stable Diffusion* and *image
    generation*, beside *diffusion models* and *text-to-image generation*).
  - **Whether a facet of a tenth of the cluster deserves one of six places** is a judgement about labels rather
    than a fact about either method.

### The maintainer's read of `out/arxiv-llm-evidence.md`, 2026-11-24

- **Clusters 60 and 37: fine.**
- **Cluster 44: the LLM catches *LLM-as-a-judge*, which looks like an important theme of the cluster**, and is
  markedly better than the phrases, which put the GPT series forward (*chatgpt, gpt-3, gpt-4 performance,
  gpt-4.5*). All three LLM runs have it, with the evidence and without; 6 of the 33 records mention judging.
- **Cluster 38: every method points the same way** (quantization, low-rank adaptation, efficient inference), **and
  the run with the evidence reads best.** Another win for it.
- **Cluster 57: a win like 38's.** Here the evidence also dropped the umbrella: both runs without it lead with
  *large language models*, and the run with it does not, taking *multilingual language models* instead (the
  phrases have *multilingual llm*).
- **Cluster 33: the phrases miss *hallucination*, which the LLM names in all three runs.** Measured the same day
  (the cluster's two orderings recomputed with the real embedder):
  - **The phrase method found it and the fusion demoted it.** By c-TF-IDF it is the cluster's second term, at a
    share of 40% (as high as the head's), so first in the tail's pool; by cosine similarity to the centroid it is
    24th of 30 (0.45, where the pool runs 0.45–0.64), and RRF puts it 10th. The fusion is built to drop a term
    high in only one order, and here that drops the right one.
  - **The bare single words sit at the bottom of the similarity order**: *hallucination* 0.452, *characteristic*
    0.448, against 0.51–0.64 for the multi-word phrases. One cluster, so a hint rather than a finding: a one-word
    embedding may be systematically far from any centroid, which would bias the fusion against single-word
    concepts as such.
  - **A separate defect: the label carries *retrieval-augmented generation* and *retrieval augmented generation*
    both.** `normalize` folds plurals and abbreviations but not hyphen and space variants, and the nesting test
    in `label_groups` compares the raw strings, so neither notices they are one term.
  - **The single-word bias, measured on all six corpora** (`measure_single_words.py`, `out/single-words.md`; the
    hyphen defect is fixed, the same day). It holds everywhere, and mostly as an effect of embedding a short
    text rather than of single words being generic:
    - **In their own cluster's similarity order, single words sit at the 29th–44th percentile**, two-word terms
      at the 52nd–72nd, longer ones at the 63rd–84th.
    - **Of the terms in a pool's first three places by c-TF-IDF, the label keeps 45–59% of the single words**,
      against 57–80% of the two-word terms and 78–100% of longer ones.
    - **The control, specificity: where a term places its own cluster among all the clusters' centroids.** A
      generic term sits near many and places its own low. Single words place it nearly as high as phrases do:
      0.79 against 0.81 on ECCOMAS, 0.80 against 0.81 on hydrogen, with a larger gap on arXiv and AOKK (0.69 and
      0.68 against 0.77). On the two small corpora, with 8 and 10 clusters, it does not separate at all.
    - **So the similarity order demotes single words far more than their specificity warrants.** A single word
      is somewhat more generic on the single-topic corpora, which accounts for some of the fall, not most.
    - **A fix would compare each term's similarity to its own centroid with its similarity to the others**,
      removing a per-term offset, as the control does. That is different from the centering rejected on
      2026-11-23, which moved the centroids.
    - **Probed the same day, on the stored datasets** (`try_relative_similarity.py`, `out/relative-similarity.md`):
      the order by cosine minus the term's mean cosine to all centroids, and the same as a z-score.
      - **It lifts single words on the four large corpora**: kept from the top three 49% → 65% on AOKK, 45% →
        53–56% on ECCOMAS, 47% → 60% on hydrogen, 59% → 60–63% on arXiv; longer terms unchanged.
      - **It changes nearly every label**, many by reordering only (clusters 2, 19, 20). Cluster 44 gains *judge*
        and *assessment*, which point at the LLM-as-a-judge theme; cluster 49 trades *cognition, advancement* for
        *survey, scientific discovery*, mixed.
      - **It does not bring back *hallucination*, the case that started this**: 23rd–25th of 30 in every
        variant. The bare word embeds in its everyday sense: of the 68 centroids, the nearest to it is cluster
        2, the physics and cosmology papers, and 44 are nearer to it than its own cluster. A per-term offset
        cannot correct a word read in the wrong sense.
      - **So the length effect is partly a sense effect**: a phrase carries its own context, a single word does
        not. The lead this leaves is embedding a term in context (as it occurs in the cluster's records, or
        beside the cluster's head term) rather than correcting its similarity afterwards.
    - **Probed next, the same day: embedding the term in context** (`try_context_embedding.py`,
      `out/context-embedding.md`), two ways: the term beside its cluster's head term in parentheses, and the mean
      embedding of up to five of the cluster's sentences containing it (the bare term where none does, about 4% of
      terms).

      | corpus | single words kept: bare / head / sentences | longer terms kept: bare / head / sentences |
      |---|---|---|
      | arXiv | 59% / 61% / 71% | 80% / 82% / 81% |
      | AOKK | 49% / 53% / 62% | 82% / 78% / 80% |
      | ECCOMAS | 45% / 52% / 59% | 79% / 73% / 80% |
      | hydrogen | 47% / 49% / 56% | 70% / 66% / 71% |
      | biology | 55% / 50% / 80% | 80% / 80% / 70% |
      | geography | 56% / 56% / 69% | 62% / 50% / 88% |

      - **Sentences lift single words on every corpus, by 7–25 points, and leave longer terms about where they
        were**; the head term does little and costs the longer terms on the large corpora.
      - **Cluster 33 gets *hallucination* back** with sentences: 5th of 30 in the similarity order, against 24th,
        and in the label.
      - **But the labels change throughout, and not only for the better** (arXiv, read the same day): 44 gains
        *judge, agreement, reliability*, toward its LLM-as-a-judge theme; 20 loses *Solomonoff induction* for
        *predictor*; 65 gains *license* and *long document*; 2 turns more generic (*star, measurement,
        detection*). A sentence embedding says what the sentence is about, so a generic word in a typical
        sentence scores as typical of the cluster.
      - Not decided; the maintainer reads `out/context-embedding.md`.
    - **Joined spellings now fold too** (*bitwidth*, *bit-width*), since the same day.
- **Cluster 65: both methods find good terms, and different ones.** The phrases quote the Qwen family
  (*qwen3-omni, qwen series, qwen-image*), the LLM generalizes it to *open-source models* (in the run with the
  evidence only). Both correct and useful. Similarly a *GPT* term could have been useful in cluster 44, had it
  not crowded out the cluster's real topic there.
- **Capitalization is inconsistent in both methods, in opposite ways** (general observation). The LLM title-cases
  keywords at random (*Foundation Models* beside *vision-language models* in one list) but spells proper names
  right; the phrases lowercase everything, proper names included: cluster 20's *solomonoff induction*, where the
  LLM, given that phrase as evidence, wrote *Solomonoff induction*.
- **Cluster 45: the LLM's keywords are more descriptive, at the right level of abstraction** (*computational
  expressivity, chain-of-thought reasoning, compositionality*), than the phrases (*multi-digit multiplication,
  quantifier*). Little difference with the evidence and without; the evidence version, winning overall, is the
  pick here too.
- **The rest of the read**, the same day:
  - **The LLM with the evidence is the pick, with no strong preference otherwise**: 16, 54; and 36 and 35, where
    the evidence version reads the most fluently of the four, which makes it faster for a person to read.
  - **The LLM wins by its level of abstraction**: 52 (the evidence version stronger), 15 (all three LLM runs).
  - **The LLM catches what the phrases miss**: 56's *character training* (all three runs); 59's *representation
    geometry* (two runs of three, one with the evidence and one without).
  - **Cluster 9: *glue*, which is GLUE**; the capitalization rule of the same day takes care of it. The evidence
    version good again.
  - **Cluster 3: *disentanglement*, a minority theme?** Two of its 9 records mention it, and only one LLM run (one
    without the evidence) names it. It is not the cluster's alone: cluster 42 has 1 of 8 records, 6 and 18 have
    1 of 9 each, and cluster 1 has 2 of 70. At 2 of 9 it is most concentrated here, though at these counts one
    record more or less would reorder that.
  - **Cluster 51: no winner, the LLM's better; but nothing says *tool calling***, the feature's usual name, so a
    keyword search for it would miss the cluster. The papers write *tool use* (five of its six records). A case
    for the LLM aliases of the keyword-search item in `TODO.md`.
- **Tally over the arXiv read**: the evidence version is the pick, or as good, in most clusters; it was worse in
  one (19, which lost a grounded abstraction), and the LLM beat the phrases on abstraction in most of the rest.
- **Reviewing the next imports: only the highest-impact clusters** (decided by the maintainer, the same day).
  Reading every cluster takes long for little gain, so the clusters already read calibrate which to pick. What
  made a cluster worth reading so far: the phrase and LLM labels disagreeing (19, 2, 33, 44, 45, 65), a minority
  theme gained or lost between runs (19, 2), the corpus-wide term leading (38, 44, 57), and a duplicate spelling
  (33, 38). So the picks are, from a script rather than by eye: the clusters whose label changes most between
  the old import and the new; those where the phrase and LLM labels overlap least; the few largest; and the ones
  already read, as regression checks, being cheap to judge.
- **Cluster 27: searching *RSI* should find it** (labelled *recursive self-improvement*, a term the corpus never
  abbreviates). Filed as keyword search with abbreviations, `TODO.md`.
- **Cluster 49: the run with the evidence reads best**, less generic than the alternatives.
- **Cluster 19, the diffusion cluster**: its *AI safety* is a grounded minority theme (above).
- **Cluster 2: *science of science*, also a minority, in a cluster that is all minorities.** Its 45 records are the
  corpus's papers that are not about AI: black holes and wormholes, cosmology, JWST and Sgr A* observations,
  numerical PDE schemes, category theory, the Fermi paradox, space elevators. Three are metascience (*Growth rates
  of modern science*, *The decline of disruptive science and technology*, *The strain on scientific publishing*).
  The model's reasoning in both logs calls the set "very diverse", counts those three as a theme, and so gives
  each facet a place; with no dominant theme, every keyword is a minority. Here the evidence kept it, as
  *scientific progress*. The phrase label (*equation, universe, physics, orbit, energy, black hole*) is the
  generic one.
  - **Its purity is the corpus's highest**: 0.83, against a median of 0.41. Purity counts how many of a record's
    nearest neighbours share its cluster, so in a corpus about AI, papers about anything else have only each other
    for neighbours. **Purity measures how separate a cluster is, not how coherent**, and a catch-all of outliers
    scores best on it. Worth remembering wherever purity is shown or used as a cutoff.
- **The other five corpora** (`out/<corpus>-llm-evidence.md`), with no repeat run, so read against arXiv's
  noise floor of 0.40:

  | corpus | clusters | overlap, with against without | from the phrases, without → with |
  |---|---|---|---|
  | biology | 10 | 0.50 | 18% → 20% |
  | geography | 8 | 0.50 | 12% → 19% |
  | ECCOMAS | 84 | 0.35 | 22% → 33% |
  | AOKK, latest | 74 | 0.45 | 13% → 24% |
  | hydrogen | 87 | 0.47 | 19% → 28% |

  The same shape everywhere: the overlap is around arXiv's noise floor, and the share taken from the phrases
  rises by two to eleven points. Each corpus's own noise floor is unmeasured, so a difference of a few
  hundredths between corpora says nothing.
- Next, the maintainer reads the reports, starting with `out/arxiv-llm-evidence.md`, and decides whether the
  evidence goes on by default; colouring by year v0 follows.
- **The README's word cloud screenshot shows words only**, and wants retaking with phrases before 0.2.11.

## A hybrid mode: the LLM's abstraction leading, the phrases' specifics after, 2026-11-24

Decided (the maintainer): build `clusters_keyword_method = "hybrid"` now, the context being warm. Phrases
quote and an LLM abstracts, so each takes the part of the label it is good at.

- **Done:** `keyphrases.lead_with(leading, ranked, top)`, with tests: the leading terms, then ranked terms that
  neither repeat one already taken nor nest in one or contain one (compared without case, hyphens or a final
  plural "s").
- **Wired into the importer, as designed (2026-11-24)**, with tests. One thing the tests found: the evidence
  went into the prompt whenever phrase lists existed, which in hybrid mode is always; it now follows
  `clusters_llm_keyword_evidence` alone. Not yet imported on a corpus. The design, for the record:
  - `"hybrid"` joins `KEYWORD_METHODS`; the LLM branch runs as for `"llm"`, then each cluster's label is
    `lead_with(llm_keywords[:n], phrase_list, max_vis_kw)`, with `n` a setting,
    `clusters_keyword_hybrid_llm_count = 2` (the 2 is a choice, unmeasured).
  - The phrase lists are computed whenever the method is `"hybrid"` or the evidence is on (today only the
    latter, in `_collect_cluster_keywords`).
  - Hybrid follows `clusters_llm_keyword_evidence` as `"llm"` does.
  - Falls back to `"phrases"` with no backend (`_effective_keyword_method`); a cluster the LLM gives up on
    (`"<unknown topic>"`) gets its phrase label.
  - `llm_backed_stages` treats `"hybrid"` as needing the LLM; its setting text names both methods.
  - Tests, the config's method description, the Visualizer README's keyword bullets, a changelog entry.
- **Preview on arXiv** (`out/arxiv-hybrid-preview.txt`, from the evidence run's stored keywords and lists, no
  new import): it reads well where the LLM's first two are a theme (*artificial general intelligence*, *LLM
  security*, *knowledge distillation*, *diffusion models, Stable Diffusion*). Two weaknesses to weigh before
  or after wiring:
  - **The LLM's first terms can be the corpus's umbrella** (*large language models* leads three clusters
    here), which spends a slot on what every cluster is about.
  - **The phrases' own head can follow as a near-repeat in other words** (*LLM security*, then *attack*); the
    phrase head says what the cluster is, which the LLM already did, so skipping it in hybrid may be right.

### Hybrid's first weakness, met on cluster 44, and the direction chosen (2026-11-24)

Hybrid with two LLM keywords leading would lose cluster 44's *LLM-as-a-judge*, the term the maintainer judged the
cluster's best (third or fifth in the LLM's lists), behind *large language models* and *LLM evaluation*, and fill
the rest with the GPT-series phrases.

- **Decided (the maintainer): drop the LLM's corpus-wide terms before taking its first n, and have the LLM order
  its list**, rather than raising `clusters_keyword_hybrid_llm_count` blindly.
  - **Ordering needs care in the wording.** The keyword-pools brief measured that the model front-loads its
    general terms, positions 13–15 of fifteen being the most distinctive on all five corpora; "most important
    first" may well strengthen that. What is wanted is closer to "most specific to this cluster first".
- **Built the same day: `keyphrases.without_umbrella_terms`**, used by hybrid before it takes the LLM's first
  two. On the stored arXiv evidence run it drops only *large language models* (23 clusters) and *language
  models* (8), and cluster 44 then leads with *LLM evaluation, LLM-as-a-judge*. An abbreviation was then not a
  repeat to `lead_with`, so cluster 38 read *low-rank adaptation*, then the phrases' *lora*; fixed the same day,
  `lead_with` comparing each phrase in its canonical form as well as its displayed one.
- **The question behind the share: are six keywords enough, and at which cluster sizes?** Two questions in one:
  how many to show, which the header and tooltip limit and the planned expandable list relaxes; and how many to
  extract and store, which nothing limits. Unmeasured. A cheap first measurement from data already stored: per
  cluster, how many of its phrases it takes to cover most of its records (greedy, on each record's `keyphrases`),
  against cluster size, on all six corpora. Phrases undercount abstractions, so this bounds from below.

## Open

**Decided: `hybrid-rrf` ships as it is** (the maintainer, 2026-11-23). Read across the arXiv corpus, its
labels mostly read as if a person or an LLM had picked them. Tuning stops here: `hybrid-rrf-word` conflates
distinct terms (*data analysis* / *data visualisation*), and narrowing it to generic heads needs a judgement
of what is generic, which is an LLM-shaped problem. Next, in order: wire `hybrid-rrf` into the importer; the
LLM keyworder given the classical terms as evidence (below), with this same test set; then colouring by
year v0 and v1 (`briefs/visualizer-filter-by-date-and-recolour-brief.md`).

**The importer wiring, settled the same day** (the maintainer): the method lives in a new module,
`raven.common.keyphrases`, generic over the grouping so that v1's years use it too; it **replaces**
`clusters_keyword_method = "frequencies"` as `"phrases"`, the default and the fallback when the LLM backend
is down; and the dataset **stores each cluster's top 30 terms with their shares** beside the six shown, for
the expandable list the keyword-pools brief plans and for v1. The probe stays as it is, the record of what
was measured; the module is the maintained copy.

**Earlier, 2026-11-20:** no clear winner among `phrase-mix1`, `phrase-mix2`, `mix-rerank` and `mix-rerank-c`. The `phrase-mix` labels read more easily; the `mix-rerank`
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
  shows; the LLM's wander, as *AI safety* in the diffusion cluster seemed to, though it turned out grounded),
  and the statistics cover the whole
  cluster where the prompt holds at most 60 records. Its bias wants measuring: the same clusters with and
  without the evidence. Importer work, not this probe's.
- **Each mention counted once, under the most specific form shown.** The nested-phrase rule lets a dominant
  longer form replace a shorter one, so the papers writing only the bare shorter form (*language model*, where
  others write *large* or *small language model*) lose their word. Keeping it needs per-record bookkeeping;
  if it is ever added, it goes into the labels and the phrase word cloud alike, which share the rule (the
  maintainer, 2026-11-23).
- **"state of the art" yields *art***, and *GIS* did not merge with *geographical information system*.
- **Against the LLM keyworder**: `run_llm_imports.sh` imports each corpus with LLM keywords on agglomerative
  clusters into `data/`, one corpus at a time, resuming after the last one finished.

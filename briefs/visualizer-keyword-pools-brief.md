# Brief: common and distinctive keyword pools, and a dataset keyword dialog

**Filed 2026-09-01.** Cluster keywords currently answer one question where a reader has two, and the
corpus-level keyword data the importer already saves is not shown anywhere. These are the same problem
seen from two ends, and the mechanism below covers both.

Not sprint-scheduled. Pull it into `researchers-night/` if the map's labels turn out to matter for the
exhibit.

## The problem

**A cluster's keywords are asked to do two jobs, and the shared ones do only the first.** A reader wants
to know *what is this cluster about* and *how does it differ from its neighbours*. Measured over the 61
agglomerative clusters of the arXiv AI corpus (2026-08-31): `Large Language Models` appears in **34** of
them. It answers the first question every time and the second never. The next most repeated keyword
appears 5 times, so the distribution is close to bimodal — a handful of corpus-wide terms, and a long
tail of terms belonging to one cluster each.

Both halves are signal. That every cluster is about language models is worth knowing once; it is not
worth repeating 34 times in the place where a reader is trying to tell two clusters apart (Juha).

**Raven already has both instruments, mis-framed as alternatives.** `clusters_keyword_method` chooses
between them, and they do not answer the same question:

- `"frequencies"` → `nlptools.suggest_keywords(per_document_frequencies=…, corpus_frequencies=all_keywords, …)`.
  This is TF-IDF-shaped: it scores a cluster's terms against the corpus background, so it already
  **discriminates**.
- `"llm"` → each cluster is described in isolation. It captures **identity** well and structurally
  cannot discriminate, because the model never sees the other clusters.

## The mechanism

**The split is postprocessing over data already in every dataset file.** Nothing needs to be recomputed
at import time, and the second half of this brief needs no importer change at all.

The quantity that separates the two pools is a keyword's **cluster document frequency** (CDF): in how
many clusters does it appear? It is a count over `vis_keywords_by_cluster`, which is saved.

**Rank by it; do not threshold on it** (settled 2026-09-01 — the first draft of this brief proposed a
threshold, and Juha asked whether this could be made adaptive the way the cohesion criterion had been.
It can, and the adaptive form has no parameter at all.)

```
idf(k)           = log(n_clusters / CDF(k))          # high = distinctive to few clusters
displayed(c)     = sort(vis_keywords_by_cluster[c], by -idf, ties keep the model's own order)[:6]
common pool      = the low-idf end of the same ranking, shown once at corpus level
```

**The display budget does the cutting, and a budget is a layout constraint rather than a tuning knob.**
Extract 12, rank, show 6. There is no number for anyone to choose, and the rule adapts by itself: on
arXiv, where a single term is corpus-wide, the ranking barely moves most clusters; on AOKK, where a
dozen are, it reorders aggressively.

**The model's own ordering is not the one wanted, and measurably so.** Asked to describe a cluster, it
leads with the general topic — which is correct for *identity* and exactly wrong for *discrimination*.
Measured on AOKK's 83 clusters: **80 of them would change displayed order** under IDF ranking.

| cluster | as the model gave it | ranked by IDF |
|---|---|---|
| 0 | **Generative AI**, Higher Education, Teaching and Learning, … | Educational Transformation, Teaching and Learning, …, **Generative AI** |
| 5 | Self-Regulated Learning, **Generative AI**, Educational Chatbots, … | Metacognitive Support, Educational Chatbots, …, **Generative AI** |
| 15 | Knowledge Tracing, **Intelligent Tutoring Systems**, … | Knowledge Tracing, Knowledge Graphs, …, **Intelligent Tutoring Systems** |

`Generative AI` scores IDF 0.98, appearing in 31 of 83 clusters, and sinks to last wherever it appears.
IDF ran 0.98 to 4.42 across that corpus, so the spread is ample.

**The tail is where this can fail, and canonicalization is what stands in the way.** Ranking by rarity
inherits the classical frequency-analysis failure: a term occurring once scores highest, and "occurs
once" is what a genuinely specific topic looks like *and* what noise looks like (Juha, 2026-09-01). The
distinctiveness figures in this brief cannot tell those apart, and neither can a string test — measured
on three corpora, 82–87% of distinct keywords occur in exactly one cluster, and a subset/superset check
flags 14–25% of those as possible variants while catching `logical reasoning` against `reasoning` and
`photobiological hydrogen production` against `hydrogen production`, which are *specializations* and
precisely what should rank first.

So the only step that can separate a variant from a specialization is the canonicalization pass, which
judges semantic equivalence rather than string overlap. **It must therefore run before IDF is computed**
— a spelling variant that ought to have merged otherwise scores maximum rarity and leads the display.
The importer's ordering already satisfies this, canonicalization being the last thing the `llm` branch
does and the ranking happening at load; it is recorded because nothing in the code says why the order
matters.

**Both halves of that were then measured, 2026-09-01, and the tail holds up.**

*How much of it is spelling variants:* almost none. Running the shipped canonicalization over the five
labelled corpora merges 6–45 keywords, of which only **0–12 were singletons** — arXiv 2, AOKK 0,
hydrogen 3, banichuk 1, ECCOMAS 12. The singleton fraction barely moves. The reason is structural:
variants are overwhelmingly of *common* terms, because a keyword must be used across many clusters
before anyone spells it two ways. So the ordering constraint above is correct and was never silently
corrupting the ranking, and the distinctiveness figures were inflated by at most ~6%, and by 0–1% on
three corpora of five.

*How much of it is noise:* 6–20%, the rest being real topics. Asking the model to classify each
singleton against the full vocabulary (`judge_singletons.py`):

| corpus | topic | generic | duplicate | artifact |
|---|---|---|---|---|
| AOKK | **94%** | 4% | 2% | 0% |
| ECCOMAS | 89% | 4% | 6% | 2% |
| banichuk | 87% | 12% | 2% | 0% |
| hydrogen | 83% | 14% | 2% | 1% |
| arXiv | 80% | 13% | 7% | 0% |

**That judge has a ground-truth check, which is what makes it worth reading at all.** ECCOMAS's five
artifacts are exactly the fragments of the one malformed prose reply found independently earlier the
same day — *"Based on the five abstracts provided"*, *"and ocular biomechanics."* — and the judge
returned zero artifacts on the three corpora separately verified to have none. It also produced visible
false positives: hydrogen's *Surface production* and *Volume production* are plasma-physics terms for
how negative ions are made, not noise. So: useful for a rough split, not a number to build on.

**The residual risk is `generic`, not `artifact`.** A vague term occurring in exactly one cluster scores
maximum rarity and leads the display — which is the failure the ranking exists to prevent, arriving by
the back door. At 4–14% it is small, and it is the part worth watching if the tail ever looks wrong.

What the classical case does not transfer is the prior: there the terms are raw tokens and a hapax is
usually a typo, where here they are model-written topic labels over clusters, so a singleton is usually
a real specific topic — measured at 80–94%. Not immune; differently distributed.

### Unify case deterministically before asking the model anything

Most of what the canonicalization pass actually does is fix case — `AI safety` → `AI Safety`,
`Chain-of-thought` → `Chain-of-Thought`. A rule does that more reliably than a model and costs nothing,
which leaves the model only the acronym and synonym decisions that need judgement (Juha, 2026-09-01).
`judge_singletons.unify_case` is the shape: group by casefold, keep the most frequent surface form, ties
to the one that sorts first so the result does not depend on iteration order. Worth moving into
`importer._canonicalize_cluster_keywords` as a pre-step.

Apply the same normalization to whatever the model returns, so a verdict on `Water Splitting` is not
dropped because the vocabulary spells it `Water splitting`.

**Ties matter and break sensibly.** Most keywords appear in exactly one cluster and so share the top IDF;
breaking those ties by the model's own order means its relevance judgement decides among equally
distinctive terms. Two rankings, each used for what it is good at.

*Same idiom as `nlptools.suggest_keywords`, which already scores the `"frequencies"` method's keywords
against a corpus background — so this is the house move rather than a new one.*

Three consequences worth having:

- **No dataset format change**, so every dataset already on disk gains the feature on load.
- **No LLM cost.** Discrimination over LLM-produced keywords needs no second pass; it is counting.
- **The threshold becomes a live GUI setting** rather than something baked in at import. Changing it
  re-partitions instantly, which is the right affordance for a number nobody can pick correctly in
  advance.

**Clusters keep their full keyword list.** A common keyword still describes the cluster truthfully — it
is a fact about the data, not noise — so it is retained and de-prioritized rather than removed (Juha).
The ordering is what changes: distinctive first.

### Where each pool is shown

| surface | shows | why |
|---|---|---|
| cluster annotation / tooltip / info panel | distinctive keywords first, ~6 of them | limited space; this is the *label*, and the reader is here to tell clusters apart |
| the new dataset dialog | the common pool, framed as *what this dataset is about* | said once, at corpus level, instead of 34 times |
| the same dialog | the full `all_keywords` table with counts, copyable | this is the *inspection* view, where "everything, with numbers" belongs |

**The dialog is an instrument, not a readout** (Juha, 2026-09-01). Each keyword row is clickable and
**selects**, honouring the modifier that picks the selection mode — so the existing replace / add /
subtract / intersect algebra in `selection.py` composes into boolean queries over keywords, and no query
language has to be invented. Select the clusters carrying `AI`, add `chatbot`, add `generative`, then
invert, and what remains is the candidate out-of-scope material. That is how the four AI-less clusters
recorded in `briefs/researchers-night/aokk-corpus-scope-classification-brief.md` were found, and it
wants to be a feature rather than a script somebody ran once.

### Raise the extraction count to 12, then split

**Measured 2026-09-01, and the number follows from what the split costs rather than from taste.** The
requirement is *six discriminating keywords to show*, so keywords burnt on corpus-common terms do not
count toward it. Counting the burn per cluster on the labelled corpora:

Measured on all five corpora, at CDF ≥ 10% of clusters:

| corpus | records | clusters | corpus-common keywords per cluster (median / p90 / max) | must extract for six distinctive |
|---|---|---|---|---|
| arXiv AI | 958 | 61 | 1 / 1 / 1 | 7 |
| ECCOMAS | 2519 | 84 | 1 / 1 / 3 | 7 |
| **AOKK** | 5007 | 83 | **2 / 4 / 5** | **10** |
| hydrogen 1–3 | 11973 | 82 | 1 / 1 / 1 | 7 |
| banichuk | 531 | 28 | 0 / 1 / 3 | 7 |

**Four of five need only seven. AOKK is the sole hard case, and not for the reason expected.** The guess
was that *crowding* drives the burn, AOKK being the most crowded corpus by median cosine distance
(0.516). Hydrogen is nearly as crowded (0.558) and behaves like the loose ones, and it is also the
largest corpus here — so neither crowding nor size predicts this. What AOKK has is a **single-topic
search**: every cluster is about AI in education, so `generative ai` and `educational technology` recur
by construction. Hydrogen production is a broad application domain spanning many subfields.

**Ten covers every measured case. Ship fifteen** — measured on all five corpora, 2026-09-01, with the
same result on every one.

| corpus | asked 12 → returned | asked 15 → returned | distinctiveness 1–12 | **distinctiveness 13–15** |
|---|---|---|---|---|
| arXiv AI | 11.5 | 14.1 | 0.83 | **1.00** |
| ECCOMAS | 11.8 | 14.5 | 0.65 | **0.87** |
| AOKK | 11.8 | 14.2 | 0.40 | **0.52** |
| hydrogen 1–3 | 11.2 | 14.3 | 0.66 | 0.68 |
| banichuk | 10.9 | 14.4 | 0.81 | **0.97** |

**Positions 13–15 are more distinctive than 1–12 on all five, without exception.** That is the opposite
of padding, and it is the IDF finding seen from the other end: the model front-loads the general terms,
so whatever it adds last is what is specific to that cluster. The extra three land exactly where the
design wants them. Hydrogen is the narrowest margin (+0.02, effectively flat) and still not negative.

**It never pads to fill the quota.** Asked for fifteen it returns 14.1–14.5 on every corpus; asked for
twelve, 10.9–11.8. Roughly `n − 0.7` either way, which is the shape of a model answering with what it
has rather than with what it was asked for.

*Grounding across that boundary is mixed — it falls on arXiv (0.77 → 0.50), banichuk and AOKK, and holds
on ECCOMAS and hydrogen. Not read as invention. The check is near-saturated where clusters are large,
and arXiv's are the smallest here at 52 members, so its haystack is smallest and an abstraction over the
titles has least chance of appearing verbatim — which is what a good keyword often is. Recorded rather
than buried, being the one measure not pointing the same way.*

The claim being made is **"Qwen3.6 tolerates fifteen"**, and that is sufficient: it is the model Raven
runs (Juha, 2026-09-01). It need not hold for models in general, and the failure if a future one pads is
not silent — see the note above on malformed versus merely wrong.

*What none of this bounds: all five corpora are English-language and STEM-ish, which is where this work
happens. A corpus narrower than AOKK would push the requirement higher and nothing here says by how
much.*

*The percentages here are not the threshold the mechanism uses — there isn't one. They are three
readings of "how generic counts as generic", used to bracket how many of a cluster's keywords are
corpus-common and therefore sink under the ranking. The count needs bracketing rather than solving,
which is why a spread of strictness levels is the right way to read it, and why the extraction count
survived the threshold's removal unchanged.*

**The model can supply twelve, and does not pad to reach it.** `keyword_count_probe.py` ran twelve
clusters at both settings:

- asked for six → **5.8** returned on average; asked for twelve → **11.2**. It returns fewer when it has
  fewer (one cluster gave 10), rather than filling a quota.
- distinctiveness within the twelve-run barely falls from the head to the tail: **0.44** for positions
  1–6 against **0.40** for 7–12. Padding would collapse.
- the tails read as real topics, several *more* specific than typical head keywords: *Cognitive Load,
  Human-AI Teams, AI Transparency, Delegation, Metacognition, Human-AI Symbiosis*; *Anthropomorphism,
  Trust in AI*; *Emotion Recognition, Pedagogical Agents*; *Self-efficacy, Goal setting*.

Two cautions about those numbers, because both invite over-reading. The fall from the six-run's head
(0.53) to the twelve-run's head (0.44) is **confounded** — more keywords per cluster means more chances
of collision, so part of that is arithmetic rather than quality, and only the within-run comparison is
clean. And *grounding* (whether a keyword's words appear in the cluster's own titles) came back at 1.00
and 0.97, which is **near-saturated and therefore weak**: a 944-title haystack contains almost any word,
so it would only have been informative had it come back low.

**Expect less than six extra discriminators, though.** The tails carry corpus-common terms too —
`Educational Technology`, `Personalized Learning`, `AI Literacy` recur across them — so some of the
extra budget lands in the common pool rather than in the cluster's own label. That is the split working
as intended, not a fault, but it means the yield from 6→12 is smaller than the raw count suggests.

### The dialog

Opened from a toolbutton. Two sections, in this order:

1. **What this dataset is about** — the common pool, plus the head of `all_keywords`. Framed as a
   description of the corpus rather than presented as keywords, because the head of that list is a
   domain stopword list in disguise: AOKK's top five are `student` (9025), `learning` (8899),
   `education` (5342), `learn` (4282), `model` (4052). That *is* the signal — it says what the corpus
   is — but a reader who is shown it under the heading "keywords" will read it as a failure.
2. **The full table**, keyword and count, sorted descending, in a clipboard-copyable form. TSV pastes
   into a spreadsheet, which is what a researcher will do with it.

**`all_keywords` is already saved and read by nothing.** It is `{lemma: count}` over the whole corpus —
8918 entries for AOKK — written at `importer.py:1467`, and `grep` finds no consumer anywhere in
`raven/visualizer/` outside the importer. The importer prints the first 20 of it to the log and that is
the only place it has ever been seen. So this section is pure GUI work over an existing field.

**It does not duplicate the word cloud.** That is built from `entry.keywords` of the *currently
selected* points (`word_cloud.py:143`), so it is selection-scoped, visual and approximate. This is
corpus-scoped, numeric and exact. Different questions.

## The parser needs a guard, and the ranking makes that urgent rather than cosmetic

**Measured across 338 clusters of five corpora, 2026-09-01: one reply came back as prose rather than as
a comma-separated list** — a bulleted "here is a structured summary of the research" — despite the
prompt asking for a list and saying a program would read it. That is 0.3%, and the importer has no
validation: `_collect_cluster_keywords` checks only for the literal failure sentinel and otherwise
splits on commas, so a prose reply becomes twenty-odd fragments of a sentence, recorded as that
cluster's keywords.

**Under IDF ranking that failure is promoted rather than buried.** Garbage fragments appear in exactly
one cluster, so they score the maximum IDF and sort to the *front* of what the cluster displays. A rule
that ranks by rarity is exactly the wrong rule to combine with an unvalidated parser, so the guard is
part of this design rather than a separate tidy-up.

**The same reasoning does *not* extend to a model that pads**, and it is worth being clear about why,
because the two failures look alike from the code's side. Padding would also rank first, being rare —
but unlike a parse failure it produces keywords that are *well-formed and useless*, which no validation
can detect and which a reader spots immediately: the labels on the map stop meaning anything and
somebody asks why (Juha, 2026-09-01). That belongs in the user manual, not in a guard. The distinction
is whether the output is malformed or merely wrong, and only the first is a program's business.

The shape of the guard is not subtle — a keyword list whose entries run to sentence length, or carry
Markdown bullets or colons, is not a keyword list — and `importer.py` already carries a
`# TODO: wrap this in a retry mechanism (up to 3 times?)` at that exact spot. Rejecting and retrying is
what the TODO asks for, and the rejection test is cheap.

*Note the burn measurements above are unaffected: recomputed with and without the malformed cluster,
ECCOMAS reads median 1, p90 1, max 3 either way, because the garbage terms are all singletons and so
never join the common pool.*

## Settled

- ~~**Does the model produce twelve meaningful keywords?**~~ **Yes** — measured 2026-09-01, see above.
- ~~**Should topic keywords be filtered to nouns and proper nouns?**~~ **Done**, shipped 2026-09-01:
  `nlptools.count_frequencies` now defaults to `accepted_pos=("NOUN", "PROPN")`. The verbs of academic
  prose crowded the head of the frequency list and discriminate nothing.
  - The adjective question is **sequenced rather than closed**. They are genuinely mixed — `generative`,
    `collaborative`, `conversational` are topical, `effective`, `significant`, `specific` are filler —
    but the measurement found a discriminator worth keeping: **the topical ones are also tagged NOUN a
    fair share of the time**, because they get used substantively, while the filler ones essentially
    never are (`generative` 155 ADJ / 110 NOUN, against `effective` 101 ADJ / 0 NOUN). So the noun
    filter already recovers much of the topical signal and excludes the filler outright.
  - Revisit including ADJ **after this brief's split exists**, since that mechanism is what would demote
    the filler automatically. The word cloud is what argues against it today, having nothing that filters.

## Open

- ~~**What threshold?**~~ **Dissolved 2026-09-01 — there is no threshold.** Ranking by IDF and cutting
  at the display budget does the job without one, and adapts per corpus by itself. The measurements that
  led here are kept because they say what the ranking is up against: arXiv's cluster-document-frequency
  distribution is bimodal (34, then 5), where AOKK's is a smooth gradient (31, 21, 20, 17, 17, 16, 13,
  12, 11, 11 …) with no gap for any threshold to find. A rule that needed a cutoff would have had to be
  tuned per corpus; this one does not.
- ~~**Does the split apply to `"frequencies"` mode too?**~~ **Not worth building for it** (Juha,
  2026-09-01: that method "tends to give low-quality keywords"). It already discriminates, being
  TF-IDF-shaped, so the per-cluster half would be redundant even if its keywords were good. The dialog
  applies either way, since `all_keywords` is produced regardless of which method labels the clusters.
  - Worth noting what happened here rather than only the outcome: keyword extraction is a long-standing
    NLP problem that frequency methods solved indifferently and an LLM solves well. The `"frequencies"`
    method stays as the offline path for an import with no backend, not as a peer.
- **Does the corpus-level display want the low-IDF terms, or the raw frequency head, or both?**
  **Probably both, and the decision waits for something to look at** (Juha). They are different lists —
  the first is "what every cluster mentions", the second is "what the corpus is made of" — and both have
  a claim on the phrase *what this dataset is about*. Cheap to show both, and a judgement that wants a
  GUI in front of it rather than a table in a brief.

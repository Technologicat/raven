# Retrieval evaluation

Known-item evaluation sets for `raven.librarian.hybridir`, and the harnesses that score them.

These are instruments, not probes: they are committed and re-scored whenever a lever changes, which is the
whole point of having them — a measurement nobody can repeat settles an argument once and then rots.

This exists because retrieval-quality arguments are otherwise undecidable. The motivating complaint —
*the hybrid rank does not track how good a result is* — names a ranking failure, and no amount of reading
the code settles whether a proposed change helps. The design work it feeds is
`briefs/summer_2026_librarian_extension/09_retrieval-query-side-brief.md`, where every lever is a
hypothesis about a different part of the score-vs-relevance curve this measures.

## The trick, and what it costs

Full relevance judgments are expensive: twenty questions against twenty results each is four hundred
human judgments before anyone learns anything. So instead of judging results, we **generate questions
from known documents**. Sample an abstract, write a question that abstract answers, and the label comes
free — that document is relevant to that question.

The metric is then "did the retriever find the paper the question was written from", as recall@k and MRR,
with no judging at all.

Two honest limitations, because they are structural rather than fixable:

- **Precision is understated.** In a corpus of 12k papers on one topic, other papers will often also
  answer the question, and they count as misses here. Absolute numbers are a floor, not an estimate.
- **It is nonetheless unbiased across configurations**, which is the property actually needed. Comparing
  score-aware fusion against plain RRF requires only the same questions and the same gold documents on
  both sides.

**And a third, which only shows up once a number is read off the set rather than compared across it.** A
generated question is written *with the passage in front of the generator*, so it names the right entities
and asks about something the passage demonstrably contains. A real query is written from fading memory —
"there was one where they went to Switzerland, I think?" — and is vaguer, wronger and shorter. So the
generated questions are systematically **easier** than the real ones they stand in for.

That costs nothing when comparing two retrieval configurations, since both face the same questions. It
matters a great deal when a *threshold* is read off where the on-corpus scores sit: real queries will score
below the measured on-corpus distribution, so a cut calibrated here sits too high and rejects real
questions that the corpus can answer. Any threshold taken from this set is an upper bound on the safe one.

The observation is Juha's, 2026-08-05, and arrived as a joke about needing a working search engine in order
to write the questions that test the search engine. It is not circular — using retrieval to surface
candidates that are then *verified by reading* breaks the loop, and that is how both hand-written probes so
far were checked. But it does say what the hand-written probes are for: they are a spot-check on the
instrument, not a scaling path, because human recall of a corpus runs out long before the question count
does.

Full judgments can be layered on later by pooling the top-N of each configuration and judging the union.
This set is the seed for that, not a competitor to it.

**And the judging needs a corpus in a native area.** Both limitations above trace to the same thing:
hydrogen production is an application domain for this team rather than its home field, so "did it find the
paper the question came from" is the only question answerable without a subject-matter judge. The fix is
not a better metric — it is a corpus in a literature the evaluator knows from the inside. The hand-curated BibTeX database of axially moving materials (assembled by hand, ~2007-2016) is
the candidate: small, curated rather than exported, and in a literature the maintainer knows well enough to
say whether the right papers came back. Known-item retrieval measures whether the retriever can find a
planted document; only a reader who knows the field can measure whether it found the *useful* ones, which
is the thing the tool claims to do.

This is a *different* need from the one the fiction corpus below answers, and the two should not be
confused: fiction is there to test whether a constant survives a change of collection, and could be a
literature nobody here had read without losing any of that. Judging needs the opposite property. (The
fiction corpus happens to have it — the maintainer has read all 19 — so it could seed pooled judgments
too. That is opportunity, not design.)

## The hazard to watch

The questions are LLM-written, and a question that reuses its abstract's distinctive phrasing turns the
task into string matching. The generator prompt forbids verbatim phrases, and `evaluate.py` reports the
**keyword-only baseline** for exactly this reason:

- If BM25 alone scores near-perfect, the questions are too easy. Regenerate the set; do not draw
  conclusions from it.
- If BM25 alone matches or beats the fusion, that is the finding — RRF is losing information rather than
  adding it.

## Two corpora, and why

Every conclusion drawn from a single corpus that takes the form of *a tuned constant* is a conclusion about
that corpus. Librarian indexes whatever the user drops in the folder, so a threshold measured on one
collection has to be shown to survive another before it can ship as a default. Hence a second, deliberately
distant one.

| set | corpus | questions | what it is for |
|---|---|---|---|
| `questions.json` | ~12k Web of Science records, hydrogen production | 99 | the original: comparing retrieval configurations |
| `fiction_questions.json` | 19 Optimalverse stories saved from fimfiction.net, ~2.2M characters | ~100 on-corpus + ~30 adjacent | whether a threshold travels |

Prose fiction is about as far from scientific abstracts as a document set gets while still being something
someone might plausibly index, which is what makes it a fair adversary rather than a token second sample.

**Each set doubles as the other's negatives.** A hydrogen question is unanswerable from a corpus of pony
fiction and vice versa, by construction — so scoring either corpus gets ~100 real, well-formed negatives for
free, in place of hand-written probes. The fiction set additionally carries an `adjacent` group generated
from 13 stories deliberately *held out* of the index: same universe, same site, same generator, same
prompts, differing only in that the answer is not in the corpus. That is as hard as a negative gets, and it
is a lower bound by construction — fan fiction in one shared universe overlaps, so a held-out question may
be answerable from an indexed story, which mislabels it and biases *against* the signal being measured.

## Corpus and copyright

**Neither corpus is in this repository, and neither may be** — one is copyrighted third-party abstracts, the
other copyrighted third-party fiction. Both live in Librarian's documents directory on the developer machine.

What *is* committed is the question sets, and the rule is the same for both: **generated text plus
identifiers, never source text.** A question is new writing that the generator was explicitly told not to
phrase in the source's words, and the labels are WoS accession numbers in one set and filenames in the
other. Anyone with the same corpus can reproduce the scores; anyone without it gets a question set and no
answers, which is the correct outcome. Note that neither generator writes the sampled passage into its
output, and neither should be changed to — that is the line that keeps the sets publishable.

Consequence for regeneration: the sampling seed is fixed in each generator, so a rerun draws the same papers
or passages. It does *not* make the questions identical — the model is sampled at temperature, so a rerun
produces different questions about the same sources. Changing the seed changes the sources too, and scores
across different sets are not comparable either way. Both generators checkpoint after every question, so an
interrupted run leaves a valid, shorter set rather than nothing.

## Running it

```bash
# Generate (needs an LLM backend; ~30 calls)
python investigations/retrieval/make_questions.py <llm_base_url> <model> [n_focused] [n_rambling]

# Score (needs raven-server for spaCy + embeddings, and the local document index)
python investigations/retrieval/evaluate.py [k]
python investigations/retrieval/sharpness.py [k]
```

The two scoring scripts answer different questions, and the second one needs a set the first one cannot
supply:

- **`evaluate.py` compares retrieval configurations** — does this change to how a query is built or fused
  find the gold document more often? Output is recall@k / MRR per condition, plus per-question ranks in
  `results.json`. This is what settled lever 3.
- **`sharpness.py` scores a *diagnostic signal*** — given a query, can we tell from its own score
  distribution whether it found anything? Output is AUROC per candidate signal, plus per-query signal
  values in `sharpness_results.json`. It asks that twice: once against retrieval success over the
  known-item questions, and once against 16 hand-written off-corpus probes carried in the script itself,
  because every generated question is answerable by construction and the interesting case is the one that
  is not. This is what settled lever 1.

Both read the index and do not write to it, so either is safe to run against a live Librarian installation.

## Baseline, 2026-07-28

99 questions (77 focused, 22 rambling) against the 11974-record corpus, k=20, questions generated by
Qwen3.6-35B-A3B. Five generation attempts were dropped (the thinking model exhausted its budget and
returned empty), so the set is 99 rather than the 104 requested.

| condition | R@1 | R@5 | R@20 | MRR |
|---|---|---|---|---|
| hybrid (RRF, as shipped) | 0.39 | 0.61 | 0.78 | 0.486 |
| keyword only (BM25) | 0.32 | 0.52 | 0.68 | 0.411 |
| semantic only (vector) | 0.26 | 0.46 | 0.66 | 0.363 |

Split by question shape:

| hybrid | focused (n=77) | rambling (n=22) |
|---|---|---|
| MRR | 0.535 | 0.315 |
| R@5 | 0.66 | 0.41 |
| R@1 | 0.44 | 0.23 |

Gold-document rank histogram, hybrid:
`{1: 39, 2: 6, 3: 10, 4: 4, 5: 1, 6: 2, 7: 3, 8: 1, 9: 1, 13: 1, 14: 3, 16: 1, 17: 2, 18: 1, 20: 2}`,
with 22 absent from the top 20. Retrieval either nails it or misses hard: 39 of 99 at rank 1, 22 of 99
nowhere in range, and a thin middle.

What this supports:

1. **The set is valid.** BM25 alone reaches R@1 = 0.32, far from perfect, so the questions did not come
   out as string matches on their source abstracts — the hazard of generating them from the documents.
2. **Fusion earns its place.** The hybrid leads both single engines on every metric, by a margin
   (MRR +18% relative over BM25) that a set this size can carry.
3. **Long, wandering messages retrieve far worse than focused ones** — MRR 0.315 against 0.535, R@5 0.41
   against 0.66. This is the centroid-dilution failure that multi-query decomposition targets, and it is
   the largest effect in the data.

### Superseded: the n=30 run

The first run, at 30 questions, reported two findings. One replicated and one did not, which is the whole
argument for having grown the set:

- *Replicated:* the focused/rambling gap (then MRR 0.562 vs 0.292, on n=8 rambling).
- *Overturned:* "the hybrid trails both single engines at R@5 while leading at R@20", which read as
  rank-only fusion promoting mediocre-but-agreed-upon documents. At n=30 the hybrid scored R@5 0.53
  against 0.60 and 0.60; at n=99 it scores 0.61 against 0.52 and 0.46. The effect reversed. It was two
  questions wide and it was noise. The n=30 write-up flagged it as such and said to grow the set before
  acting on it — which is the only reason no work was built on top of it.

Worth keeping as a calibration on how much a 30-question known-item set can carry: enough for a factor-of-two
effect, not enough for a few points of R@5.

## What the set has decided so far

- **2026-08-05 — multi-query retrieval (brief 09, lever 3): rejected as specified.** Splitting a rambling
  message into sentences and fusing every result set alongside the whole message measured *worse* than not
  splitting: rambling MRR 0.315 → 0.286, R@20 0.64 → 0.50. Focused questions were identical to three
  decimals, which is both the control working and the evidence that the plumbing is sound — a fault in the
  per-query indexing or the fusion would have moved both subsets.

  Cause: a rambling message yields five to seven subqueries, so the whole-message query holds one vote in
  seven, and the context sentences outvoting it agree with each other about the general topic. RRF rewards
  agreement, so it promotes exactly the generically-topical documents the brief set out to demote.

  The diagnosis it was built from is untouched — rambling questions really do retrieve at half the MRR of
  focused ones, still the largest effect in this data. Only the remedy is refuted. The machinery ships
  defaulted off (`HybridIR.query(multi_query=...)`) pending lever 1's confidence signal, which would drop
  the subqueries that found nothing in particular. Per-question ranks for all four conditions are in
  `results.json`.

  **This is the case the set exists for.** The change was plausible, well-argued, and measured as the
  largest opportunity in the data — and it was wrong, in a direction nobody predicted, for a reason that
  only shows up in numbers. Two hours of implementation and three minutes of scoring beat any amount of
  arguing about it.

- **2026-08-05 — document-level questions are a distinct failure class, and the signal detects them.** Three
  hand-written probes against the fiction corpus, from a reader who had read it. Two asked about content in
  a passage ("which story has the protagonist travel to Switzerland?", "…go undercover to sabotage a
  competing AGI project?"): both retrieved the right story at rank 1, at best vector similarity 0.45–0.54.
  The third asked about a property of a whole story ("which one is set offline, in America?"): it retrieved
  the wrong story, at 0.38–0.40 — clearly below the other two.

  The reason is structural rather than a ranking failure. **No chunk says "this story is set offline in
  America."** That fact is distributed over the whole document, and the index holds 1000-character chunks,
  so the question is unanswerable in the form asked even though the corpus plainly contains the answer. A
  reader answers it by having read the thing; retrieval would need document-level metadata or a summary
  layer, which is a different mechanism from anything in brief 09.

  Two things follow. It is **evidence for the confidence signal**, which separated "the corpus can answer
  this" from "the corpus contains this but cannot be asked this way" without being built for that
  distinction. And it is a **product gap worth its own treatment**: "which of my documents is the one
  about X" is an ordinary thing to ask a document database, and chunk RAG structurally cannot serve it.

- **2026-08-05 — the retrieval confidence signal (brief 09, lever 1): the level, not the shape.** Brief 09
  designed a `min_p`-style reading of the *shape* of a query's score distribution, having rejected an
  absolute threshold on cosine distance. Measured, the shape reading is anti-correlated with what it was
  built for: an off-corpus query reads *sharper* than an on-corpus one (mean 0.92 against 0.53), because
  with nothing genuinely matching, an accidental best hit stands clear of an already-low field. The
  rejected design wins outright — absolute best vector similarity separates the 99 known-item questions
  from 16 off-corpus probes at AUROC 0.99, with a cut at 0.45 rejecting 13 of 16 probes and none of the 99.

  The two readings turn out to answer different questions, which is the finding worth keeping. **Level**
  (best vector similarity) says whether the corpus can answer this at all; **shape** (keyword `best/mean`,
  AUROC 0.73) says whether a query the corpus *can* answer landed well. Asking either one the other's
  question gets a confident wrong answer.

  What this cannot say: whether 0.45 travels. The objection to absolute thresholds was always that the
  scale of "close" belongs to the collection, and one hydrogen corpus cannot test that — which is the
  sharpest argument yet for the second, native-area corpus described above. The probes are also
  hand-written rather than sampled, and the only hard group among them ("adjacent" — real science, not in
  this corpus) has four members.

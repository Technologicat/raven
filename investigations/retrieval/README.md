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

**The same shape turned up twice on the same day**, which is what makes it a property of the work rather
than a quip. The second: reading the IR literature would give us ideas for improving retrieval, and combing
literature is what Raven is *for* — so the tool we would use to find out how to fix retrieval is the
retrieval we are fixing. It breaks the same way, by the same move: retrieval surfaces candidates, reading
decides. Recorded because the resolution is the useful part — a loop like this is only vicious if you let
the tool's output stand as the verdict instead of as the shortlist.

Acted on the same evening, entirely with tooling Raven already ships (`raven-arxiv-search` →
`raven-burstbib` → `raven-indexer`). The query, kept so the sweep is reproducible:

```
("score normalization" OR "score calibration" OR "threshold")
  AND ("information retrieval" OR "dense retrieval" OR "retrieval augmented generation")
```

157 entries, at `00_stuff/datasets/ir_literature/` (gitignored — arXiv metadata, not repo content).

**And it comes with a coverage caveat worth stating before anyone treats it as a literature review.** The
year distribution is 2012 at the earliest and 125 of 157 in 2024–2026. That is arXiv working as designed:
IR's threshold-setting and score-normalization classics come from the TREC filtering track and the
distributed-IR/collection-fusion line, which predate preprint culture in this field and live in SIGIR and
TREC proceedings rather than on arXiv. So this set is a source of *ideas to test on our own stack*, which
is what it was gathered for, and not evidence about what is already known. Anything from the older
literature has to be reached another way.

Three hits look directly relevant to the open levers, on titles alone and unread as yet: *DAT: Dynamic
Alpha Tuning for Hybrid Retrieval in RAG* (per-query dense/BM25 weighting — levers 1 and 3 meet), *The
Overlooked Role of Graded Relevance Thresholds in Multilingual Dense Retrieval* (both live questions at
once), and *BalanceRAG: Joint Risk Calibration for Cascaded RAG*.

**This set is also the obvious fifth corpus, and `raven-arxiv-download` would make it the one shape the
other four do not have.** The current matrix is short-scientific (hydrogen and arXiv abstracts),
very-short-scientific (banichuk titles) and long-narrative (fiction). Nothing is *long scientific* — which
is Librarian's actual pitch, a researcher dropping PDFs into the folder.

The gap matters for a specific reason rather than for symmetry: **chunking barely runs in any corpus we
have.** An abstract is one to three chunks, a title is one, so the sliding window, the overlap, the
contiguous-chunk merging and within-document ranking are all essentially untested by this harness — every
result so far is close to one-chunk-per-document retrieval wearing a chunking engine's clothes. A fulltext
paper is dozens of chunks, and is the case where a wrong answer can come from the right document.

Cheap to build from here (`raven-arxiv-download` over the 157 identifiers already in hand), and it brings
a second question with it: a fulltext corpus of *IR papers about thresholds* is a corpus whose own subject
is what we are measuring, so the hand-written probes for it can be written from genuine curiosity rather
than generated — the first set where the reader would be asking real questions with a stake in the answer.

**Better still, and it needs no download at all.** The `ai_papers` set was not gathered by searching arXiv;
it was built the other way round — arXiv IDs parsed out of PDF *filenames* on disk, then queried for their
metadata. So **the 1268 fulltext PDFs already exist**, on the personal machine rather than this one, which
makes the transfer the only cost.

That is the stronger experiment, because it is *controlled*. Indexing those PDFs gives the same 1268
documents already indexed as abstracts — same topic, same corpus, same identities, same gold labels, and
even the same generated question set applies unchanged. Document length is then the only variable that
moved. Every corpus comparison up to now has confounded length with topic and genre; this one would not,
and it is the only way to attribute anything measured to chunking rather than to subject matter.

**Dedup is a prerequisite, and the obvious rule is the wrong one.** The PDFs on disk include several
versions of some papers (arXiv IDs carry a `vN` suffix). The *abstract* corpus does not — checked, 1268
files and 1268 distinct IDs — so nothing measured so far is affected by this; it is purely a constraint on
building the fulltext set.

The reflex would be "keep the highest version per paper". That silently destroys the experiment. The
abstract corpus pinned *particular* versions (`2410.07866v5`, `0706.3639v1`, …) and the document ID is the
filename, which is what every gold label keys on. Selecting by newest would produce a corpus whose IDs no
longer match, so the question set would have to be regenerated and the comparison would stop being
controlled — which was the entire reason to prefer this corpus. **Select the PDFs by exact filename match
against the 1268 `.bib` files**, and treat any paper whose pinned version is missing from disk as an
exclusion to report rather than a version to substitute.

Two things to expect, worth writing down before it is run. On-corpus similarity should *rise*, since a
question's answer is likely stated somewhere in a full paper and only gestured at in its abstract — which
would put the fulltext case on the opposite side of the constant from the titles case, and a single
threshold would then have to straddle both. And known-item scoring gets harder to interpret: with dozens
of chunks per document, "found the right document" stops being the whole question, since a wrong passage
from the right paper is a failure the current metric scores as a success.

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
python investigations/retrieval/sharpness.py <hydrogen|fiction> [k]
python investigations/retrieval/run_probes.py [k]
python investigations/retrieval/calibrate.py [hydrogen|fiction]
```

The scoring scripts answer different questions, and each needs something the others cannot supply:

- **`evaluate.py` compares retrieval configurations** — does this change to how a query is built or fused
  find the gold document more often? Output is recall@k / MRR per condition, plus per-question ranks in
  `results.json`. This is what settled lever 3.
- **`run_probes.py` scores the hand-written probe set** in `probes.json` — nine information needs with
  human-verified labels, each in several phrasings. It covers what the generated sets structurally cannot:
  questions stratified by *where the answer lives* (in a chunk, in the document but stated, in the document
  but only exhibited, or outside the corpus entirely), and phrasing sensitivity with ground truth held
  constant. Output is per-class hit rates and per-probe similarity spreads, plus `probe_results.json`.
- **`sharpness.py` scores a *diagnostic signal*** — given a query, can we tell from its own score
  distribution whether it found anything? Output is AUROC per candidate signal, plus per-query signal
  values in `sharpness_results.json`. It asks that twice: once against retrieval success over the
  known-item questions, and once against 16 hand-written off-corpus probes carried in the script itself,
  because every generated question is answerable by construction and the interesting case is the one that
  is not. This is what settled lever 1.
- **`calibrate.py` picks the operating point** for whichever collection is currently indexed, using only the
  universal probes — so unlike everything above it needs **no labelled questions**, which is what makes it
  the shape a shipping implementation could take. Output is the probe distribution and a recommended cut;
  pass a corpus name and it also reports what that cut would have cost on the labelled set.

All of them read the index and do not write to it, so any is safe to run against a live Librarian
installation. **None of them can check that the corpus you named is the one indexed** — pointing a run at
the wrong index silently relabels every question rather than failing.

### Running the full sweep: finish every question set *before* scoring any corpus

`sharpness.py` builds a corpus's negatives from *every other corpus's question file that exists at the
moment it runs*. That is what makes the cross-corpus negatives free, and it is also a trap: a run performed
while another set is still being generated silently scores against a partial one, and the resulting numbers
are not comparable with a later run that saw the whole thing.

This is not hypothetical — the hydrogen re-run on 2026-08-05 picked up 17 of an eventual 100 banichuk
questions, and nothing in its output said so. The output file records `n_negative`, which is the only
place the discrepancy shows, and only if you go looking.

So the sweep is two phases, and they do not interleave:

1. Generate **all** question sets. They need only the source directories, so this is independent of which
   corpus is indexed and several can run at once if the LLM backend allows it.
2. Then, for each corpus in turn: swap it into the documents/index slot, `raven-indexer`, and
   `sharpness.py <corpus>`.

Only after phase 2 completes for every corpus is the comparison table trustworthy. Re-running one corpus
alone, later, reintroduces exactly the inconsistency this avoids.

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

## Resolved: the third corpus falsified both predictions, and the second retraction is mine from this morning

Fiction tested the calibration against a corpus as *far* from hydrogen electrolysis abstracts as a document
set gets. That is the easy direction, and it is the one that flatters the signal: fiction and abstracts
differ in genre, register, sentence length and vocabulary all at once, so almost anything would separate
them. The near case is the one that decides whether the mechanism is real — 686 arXiv AI/ML abstracts,
which share genre, register, length distribution and academic phrasing with the hydrogen corpus, and differ
**only in topic**.

Stated in advance, so the run can falsify it rather than confirm whatever happens:

1. **Cross-corpus AUROC will fall materially below the 0.9997/0.9998 measured against fiction.** If it does
   not — if a near well separates as cleanly as a far one — then the signal is reading *topic* rather than
   *register*, which is better news than expected and would widen where it can ship.
2. **The p75 calibration will still beat any fixed constant**, because its mechanism does not depend on how
   similar the corpora are: a probe about sourdough is off-corpus for all three collections. What should
   shrink is the *margin* — the gap between the on-corpus distribution and the near-corpus negatives.
3. **The recommended cut for arXiv AI will land near hydrogen's 0.382 rather than fiction's 0.367**, since
   the probes are being scored against the same genre of text.

Prediction 1 is the one that matters. It is also the one most likely to be wrong in the interesting
direction.

### The result: 1268 arXiv AI/ML abstracts, indexed 2026-08-05

**Prediction 1 was wrong, in exactly the direction named as interesting.** The near well separates as
cleanly as the far one — arXiv on-corpus questions against hydrogen questions score **AUROC 0.999**, and
against fiction questions **0.999**. Identical. The two negative distributions sit on top of each other
(hydrogen 0.232–0.464, fiction 0.201–0.470), so sharing genre, register, length and academic phrasing with
the corpus buys a query essentially nothing.

**So the signal reads topical match, not register.** That is worth more than the prediction was: the
standing worry about any absolute cut was that "the scale of close belongs to the collection". If the
score is about whether the query is *about* what the corpus is about, then its scale is far more
corpus-independent than that argument assumed.

**Prediction 2 was also wrong, and it was this morning's finding.** p75-of-probes does not survive contact
with a third corpus: on arXiv it cuts at 0.282 and lets **72.8%** of off-corpus negatives through. It
looked good on hydrogen and fiction because on both, p75 of the probe distribution happened to land near
the on-corpus floor. Redone with all three corpora, and scored on the worst case rather than the average —
which is what a shipped default has to survive:

| estimator | worst on-corpus lost | worst negatives missed |
|---|---|---|
| p75 of probes *(this morning's pick)* | 3.4% | **72.8%** |
| p90 of probes | 6.8% | 14.6% |
| max of probes | 11.4% | 11.8% |
| fixed 0.45 *(the hydrogen pick)* | **27.3%** | 6.9% |
| **fixed 0.40** | **6.8%** | **15.3%** |

**A fixed 0.40 matches the best probe-calibrated estimator on both axes.** So calibration-from-probes buys
essentially nothing, and this morning's conclusion was an artifact of comparing it against 0.45 — a
constant chosen on hydrogen alone, and therefore the wrong baseline. Comparing a new mechanism against a
badly-chosen incumbent is how a mechanism gets adopted on someone else's mistake.

**The two falsifications are one finding.** Prediction 1 failing is *why* prediction 2 fails: a signal that
reads topic rather than register does not have a per-collection scale, so there is little for a
per-collection calibration to recover. What survives from this morning is only the narrow claim — that
**0.45 specifically** does not travel, costing 27.3% of answerable questions on fiction — not the mechanism
built on top of it.

The caveat that killed p75 applies to 0.40 as well, and is now stated before rather than after: three
corpora chose it, and a fourth could unseat it. The difference is that a constant makes a weaker claim than
a mechanism, so it has less to be wrong about. `calibrate.py` is kept as the instrument that produced the
table, with its recommendation removed.

## Open: a fourth corpus stresses document *length*, and another prediction written first

The three corpora so far vary topic and genre while holding one thing constant: every document carries a
few hundred words of prose. The axially-moving-materials bibliography (`00_stuff/rawdata/banichuk_references.bib`,
541 records, 1766–2013, shape documented in brief 11) breaks that — **only 4 of its 541 records carry an
abstract**, because it was typed by hand between 2007 and 2016, partly predating routine online abstracts.
It is titles, authors and years. That is not a degenerate case invented to be hard; it is what a hand-built
BibTeX database looks like, and answering "which paper was the one about X" over one is a plausible thing
to want.

Written before the run:

1. **On-corpus similarity will be systematically lower than on any of the three abstract corpora**, whose
   medians run 0.519–0.670. A QA-type embedder maps a question near its *answer*; a title names a topic
   without answering anything, and gives roughly a tenth of the surface to match against.
2. **Therefore the shipping constant of 0.40 will lose materially more than its current worst case of
   6.8%** — and if the on-corpus median lands near 0.40, then "one global constant" needs a document-length
   caveat, and the per-collection idea comes back in a form that has nothing to do with off-corpus probes.
3. **The keyword arm will degrade more than the vector arm.** BM25 on a twelve-token document has almost no
   term-frequency signal to work with.

Prediction 2 is the one with consequences. Note it is a *different* mechanism from the one refuted above:
that one calibrated against off-corpus probes and failed; this one would key on a property of the documents
themselves, which is measurable at index time without any probes at all.

A hazard specific to this corpus, to check rather than assume: questions generated from a title alone have
much less room to avoid reusing the title's distinctive words than questions generated from an abstract, so
`check_leakage.py` matters more here than it did.

**That hazard was backwards, and looking for it found the real one (2026-08-05).** Leakage measured on the
first 31 titles-only questions is the *lowest* of any set — longest shared run 2 words, against hydrogen's
6 and fiction's worse. The reason is obvious in hindsight: a ten-word title offers almost nothing to echo,
while an abstract offers paragraphs of it. For reference, `check_leakage.py` now covers all four sets:

| set | max shared run | share at 6+ words |
|---|---|---|
| hydrogen | 6 | 2% |
| arxiv-ai | 5 | 0% |
| banichuk (first 31) | 2 | 0% |

What the zero-overlap cases turned up instead is **a known-item validity problem, not a copying one**. Six
questions share *no* word with their title, and inspection shows all six are good paraphrases — one even
translates a German title correctly. But two of them come from bare textbook titles, and those paraphrase
into questions that identify no particular document:

- *Fluid Mechanics* → "How do liquids and gases flow and respond to applied forces?"
- *Exploratory Data Analysis* → "What methods are recommended for initially inspecting and visualizing raw
  datasets to uncover hidden patterns before formal statistical modeling?"

Both are faithful to their source and neither is answerable *only* by it. The generator's docstring already
warns that known-item scoring understates precision because other documents may also answer a question;
a titles-only corpus makes that much worse, because a generic title gives nothing specific to build a
discriminating question from. So expect this corpus's retrieval numbers to be a floor with a wider gap
beneath them than the abstract corpora's.

It also suggests a mechanism for something already measured: scored against the *hydrogen* index, banichuk
questions sit higher than any other off-corpus group (median 0.372, max 0.558, against fiction's 0.266 and
0.494). Generic questions match more things everywhere. Topical proximity — structural mechanics is nearer
to hydrogen-production engineering than fan fiction is — would explain it equally well, so the two are
confounded on the present evidence and the mechanism is *not* established. What would separate them: score
the generic questions and the specific ones separately against an unrelated index.

## What the set has decided so far

- **2026-08-05 (superseded the same day — see the arXiv section above) — "per collection" means calibrated
  at index time, not typed into a config file.** Retained because the reasoning is a clean example of a
  result that is real on its evidence and wrong on a wider sample: everything below is correctly measured
  on hydrogen and fiction, and the estimator it selects fails on a third corpus that had not been built
  yet. The self-flagged caveat — "the estimator was chosen on the same two corpora it is scored on" — is
  exactly what cashed. The
  fiction run left the off-corpus cut as a per-collection setting without saying what such a number is made
  of. A user-typed threshold is not an answer: the value cannot be derived from anything a user knows, and
  the failure is silent in the direction that hurts — an answerable question marked ungrounded reads as a
  confident refusal.

  The way out is that **the two sides of the calibration are not equally available**. On-corpus questions do
  not exist at index time (chunks can stand in, but a chunk is long and expository where a question is short
  and oblique — the mismatch that makes the dramatized-text probes fail). Off-corpus queries, by contrast,
  are corpus-*independent* by definition: a probe about sourdough is off-corpus for every collection anyone
  would build. So run a fixed probe set against the new index and put the cut at the top of the resulting
  distribution. `calibrate.py` does this and needs no labels.

  Measured on both corpora, against the fixed 0.45 the hydrogen run proposed:

  | estimator | hydrogen: on-corpus lost | fiction: on-corpus lost |
  |---|---|---|
  | fixed 0.45 | 0 / 99 (0.0%) | 24 / 88 (27.3%) |
  | max of probes | 3 / 99 (3.0%) | 10 / 88 (11.4%) |
  | **p75 of probes** | **0 / 99 (0.0%)** | **3 / 88 (3.4%)** |
  | median + 2σ | 4 / 99 (4.0%) | 14 / 88 (15.9%) |

  Far negatives are still caught at 112/117 and 99/99 respectively, so the gain is not bought by giving up
  the detection the signal exists for. **Anchoring on the maximum is the trap**: the top probe is an outlier
  ("What is the capital of Mongolia?" scores 0.479 against hydrogen abstracts, 0.074 clear of the next), so
  any estimator resting on it inherits one query's bad luck.

  Two things the table does not say. The probe set is **12 queries**, so a quantile of it is itself noisy —
  widening it is the obvious next step if this ships. And the estimator was chosen on the same two corpora
  it is scored on, which is how a tuned constant returns under a new name; a third collection is the test.

  A side finding worth keeping: **pleasantries are the hard universal negative for a narrative corpus.**
  Against fiction the three top-scoring probes are all conversational filler ("Could you say a bit more
  about that?", 0.412) — dialogue looks like dialogue. Against hydrogen abstracts they sit mid-pack. A
  calibration probe set built only from factual off-topic questions would therefore under-estimate the cut
  on exactly the corpus type where the cut matters most.

- **2026-08-05 — the cross-corpus direction that had not been run: hydrogen is also 1.000.** The fiction run
  measured on-corpus against 99 hydrogen questions; the reverse — 99 hydrogen questions against 117 fiction
  ones, on the hydrogen index — was never run, and was carried in the notes as "symmetric, mostly
  confirmatory". **That was sloppy**: the two runs query *different indexes*, so every similarity value
  differs and nothing forces the two AUROCs to agree. Run properly it comes out at 1.000, matching fiction.
  The claim is now measured rather than assumed, and the assumption was the kind that is right until it
  isn't.

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

- **2026-08-05 — on narrative prose, retrieval works when the query shares a word with the text, and fails
  when it describes what the text dramatizes.** Six hand-written probes against the fiction corpus, from a
  reader who had read it. Sorted by whether the thing asked about is *named* in the prose:

  | probe | distinctive term in the text | result |
  |---|---|---|
  | the Switzerland trip | yes, "Switzerland" ×19 | correct story at rank 1, sim 0.45–0.51 |
  | the bots discovering Marxism | yes, "Marx" ×16 | correct story at rank 1–2 |
  | the undercover job against a rival AGI | partly, "infiltration" | correct story at rank 1, sim 0.475 |
  | a holdout talking to his uninterested father | **no** — the scene is dialogue | **12th of 19**, sim 0.442 |
  | which story is set offline in America | **no** — "America" never appears | not retrieved, sim 0.38–0.40 |
  | which story is an Asimov pastiche | **no** — and the referent is outside the corpus entirely | not retrieved, sim 0.36–0.40 — **and it should not be.** No reasonable definition of retrieval covers this: the fact being asked for is not in the corpus in any form |
  | the same story, asked by its content | yes, "entropy" ×4 | correct story at rank 1, sim 0.46–0.65 |
  | which story runs until the end of time | yes — "endless eternities" in the prose, and the chapter titles escalate "An Hour" … "A Millennium", "An Eternity", "A Yoctosecond" | correct story at rank 1 on both phrasings |

  **That last row only works because of a bug fixed the same morning**, which is worth recording as the
  concrete value of the fix rather than as a coincidence. The chunk that answers it sits at offset 313500
  of `Spiraling Upwards`, and until that day the HTML extractor returned the first 20999 characters of that
  document's 366881 — one chapter of fifteen. The answering text did not exist in the index. What the query
  would have got instead is the failure mode the rest of this table documents: a confident wrong story,
  with nothing marking it as wrong.

  Also note *how* the document-level question succeeds here when "set offline in America" did not. The
  structure of the whole story is legible in this one because its chapter headings are **text** — they say
  "An Eternity" — and headings survive extraction (they are recovered per article by the same fix). A
  document-level property that the document happens to state is retrievable; one it merely exhibits is not.

  The last two rows are the same document, and the pair is the cleanest thing in the table: `The Last
  Optimization` is unfindable as *an Asimov pastiche* and trivially findable as *a story about entropy*.
  The allusion queries also score the two lowest similarities of the whole session, so the confidence
  signal calls it correctly — and calling it correctly here means reporting nothing, which is the answer.

  Note the 0.646 on "Celestia tries to defeat entropy over trillions of years", the highest on-corpus
  reading on this corpus and close to hydrogen's median. That query nearly quotes the story's opening
  ("she had tried repeatedly and multiply, for hundreds of trillions of original-Earth years"). So the
  genre gap is really a *register* gap: narrative scores low against a question, and high against a query
  phrased in its own terms. Which is the same effect as the "generated questions are easier" limitation
  above, seen from the other side — a reader working from memory does not phrase queries that way.

  The holdout row is the one that rules out the cheap explanations. `Just Be Happy` *is*
  the story that probe describes — its opening chapter is a holdout under a truck while his father asks
  what cutie marks are, his brother having already uploaded. Scored per document with everything else held
  out, its best chunk sits at 0.442, **below eleven stories that are not about that at all**, `Caelum Est
  Conterrens` leading at 0.550. So this is not fusion, not ranking, and not the document-length effect
  (which was tested separately and came out roughly proportional to chunk share, i.e. not a bias): the
  embedder itself places the right passage further from the query than eleven wrong ones.

  **The mechanism, inferred from the pattern rather than measured directly:** the embedder is
  `multi-qa-mpnet`, trained on question↔answer-passage pairs, and the query is analytical where the passage
  is dramatized. Stories that *discuss* holdouts and uploading in expository prose beat the one that *shows*
  one.

  **Corrected the same day, and the correction matters more than the original claim.** An earlier version of
  this entry concluded that the fusion must therefore be leaning on BM25, since the rows that work all have
  a rare term to grab. A later probe refutes the general form of that: "the protagonist discovers Equestria
  is a doubly linked list of voxels" retrieves the right passage at rank 4 out of a 2977-chunk index, and
  the passage contains not one of those words — it says chest, void, blocks, cells, *connections*. The
  investigator had by then failed to find that same passage with a dozen hand-written greps, for exactly
  the reason the embedder did not: grep searches the asker's vocabulary.

  So the rule is not "shared words or nothing". It is that the embedder bridges concept to text **when the
  text is itself expository** — Light Sparks reasoning explicitly about grids, coordinates and adjacency is
  as abstract as the query — and fails when the text is purely dramatized, as `Just Be Happy`'s opening
  dialogue is. Both observations stand; only the generalisation drawn from the first was wrong.

  This also supplies the mechanism for the threshold finding below, which was flagged there as unverified
  speculation: abstracts are expository and state things the way a question asks for them, narrative prose
  does not, so the fiction corpus's on-corpus similarity sits roughly 0.2 lower *throughout* rather than on
  hard queries only. That is a property of the genre, which is stable per collection — so it argues for a
  per-collection threshold rather than against thresholds.

  The remedy is not on the query side of brief 09: a description-to-dramatization gap is closed by
  something that has *read* the corpus, which is the RAG tool surface letting the model author the query,
  or a document-level summary layer. Worth knowing before more effort goes into query rewriting.

- **2026-08-05 — the fiction run: the threshold does not travel, and the signal only does the coarse job.**
  88 generated on-corpus questions against 144 negatives (99 hydrogen questions, 29 held-out-story
  "adjacent" questions, 16 hand-written probes), scored on the fiction index. `check_leakage.py` first: 0
  of 117 questions share a 6-word run with their source, longest run anywhere is 4 words, so the on-corpus
  distribution is not inflated by copying and the numbers below can be read as levels.

  **The constant does not travel, and the size of the failure is the point:**

  | | on-corpus min | median | max | cut 0.45 rejects |
  |---|---|---|---|---|
  | hydrogen | 0.460 | 0.670 | 0.823 | **0 of 99** |
  | fiction | 0.352 | 0.519 | 0.804 | **24 of 88** |

  A global 0.45 marks **27% of answerable fiction questions as ungrounded**. To reject none of them the cut
  has to come down to 0.35, where it would also stop rejecting most of what it is for. This is what the
  second corpus was built to find out, and it is a clean answer.

  **But the signal itself works — for the coarse discrimination only.** Split by how far the negatives are:

  | negatives | n | AUROC |
  |---|---|---|
  | the 99 hydrogen questions (different field entirely) | 99 | **1.000** |
  | hand-written off-topic and science probes | 12 | **1.000** |
  | held-out Optimalverse stories (same universe, absent document) | 29 | **0.742** |
  | all of the above | 144 | 0.947 |

  So "this conversation has moved off your corpus" is detected essentially perfectly, on both corpora. "This
  particular document is not in your corpus, though its neighbours are" is barely detected: the adjacent
  group runs 0.353–0.637 against on-corpus 0.352–0.804, and at any cut preserving the on-corpus questions
  it rejects none of them. That is a lower bound — same-universe fan fiction overlaps, so some adjacent
  questions are genuinely answerable from indexed stories and are mislabelled — but the overlap is far too
  wide to be explained by contamination alone.

  **This rescues lever 1's original consumer.** Brief 10's grounding marker exists because "what is 2 + 2?"
  returned electrolysis documents and read as grounded. That is the *far* case, at AUROC 1.000. The marker
  does not need to know that one paper is missing; it needs to know the conversation left the corpus. So
  the signal ships — per collection, and advertised as coarse.

  **Measurement A is degenerate here and should not be quoted:** 84 of 88 questions found their gold
  document, so its AUROCs rest on 4 negatives. Expected — 19 documents at k=20 makes known-item retrieval
  nearly free — and the reason the probe set exists.

  **A note on the probes versus the generated set, because they disagreed and the generated set is right.**
  The hand-written probes are stratified to *include* cases designed to break the signal, and reading them
  as representative produced an over-pessimistic conclusion earlier in the day (see the crossing pair
  below, which is real but is two hand-picked points). The generated set is the representative sample; the
  probes are the failure-mode catalogue. Both are worth keeping, and neither substitutes for the other.

- **2026-08-05 — the level signal crosses, and this time both labels are verified.** The retracted version
  of this claim rested on a guessed label. The probe set (`probes.json`, scored by `run_probes.py`) supplies
  it properly, because two of its probes have settled ground truth pointing opposite ways:

  | probe | truth | retrieval | best similarity |
  |---|---|---|---|
  | `runs-until-end-of-time` | answerable, in the corpus | **gold at rank 1, both phrasings** | 0.387–0.409 |
  | `asimov-pastiche` | not in the corpus in any form | correctly finds nothing | 0.356–**0.395** |

  **0.395 > 0.387.** A question the corpus answers *perfectly* scores below one it cannot answer at all,
  and no threshold placed anywhere separates them. The `holdout-and-father` probe closes the pincer from
  the other side: 0.550–0.636, the highest readings in the set, on a probe that retrieves the gold document
  at rank 1 **zero** times out of two.

  So on this corpus the level is not merely differently-calibrated — high when wrong, low when right. That
  is the consumer lever 1 was built for, and it is the reading the generated run now has to confirm or
  overturn at proper sample size. Note what is *not* claimed: measurement A already found the level weak at
  predicting retrieval success (AUROC 0.563), so the surprise is not that it fails there. The surprise is
  the on-corpus/off-corpus crossing, which is measurement B's own job.

- **2026-08-05 — one information need, three phrasings, 0.17 of confidence signal between them.** The
  reader asked which story has the protagonist work out what data structure Equestria is built from,
  remembering the answer — a doubly linked list of voxels — but not the story. Three phrasings of that one
  question, against the same index:

  | phrasing | best vector similarity |
  |---|---|
  | "…discovers it is a doubly linked list of voxels" | 0.344 — the lowest reading of the session |
  | "…what data structure Equestria is built from?" | 0.431 |
  | "experimenting to find out how the simulated world is represented internally" | 0.510 |

  All three returned 20 confident results. **The spread is 0.166, which is comparable to the entire gap
  between the two corpora that any threshold has to straddle** — and it comes from nothing but how the
  same question was worded. A fixed cut anywhere in that range accepts or rejects the same information
  need depending on the user's phrasing, which is not a property a grounding marker can be built on
  without saying so out loud.

  **Ground truth, established only after two wrong answers from the investigator:** the scene is in the
  original `Friendship is Optimal`. Light Sparks, studying magic at Celestia's suggestion, is given a
  puzzle cube with one sapphire block; probing it and then his own treasure chest, he finds "one hundred
  blocks in, the command to get the next adjacent block failed", and concludes that "Equestria didn't have
  a geometry, but only had **connections**. You couldn't give a 3D coordinate for a block." The reader's
  "doubly linked list of voxels" is an exact paraphrase of a passage that contains none of those words.

  **Two claims were made and retracted along the way, and both are worth keeping visible.** First, that the
  answer was verifiably in the held-out `The Advocate` — from a single keyword hit which, read, is about
  chip design. A conclusion was built on that label (that on-corpus and off-corpus distributions *overlap*,
  since 0.510 beat the verified-answerable Switzerland probe at 0.450); it fell with the label. Second,
  that the scene was in neither corpus at all — also wrong, and wrong the same way: the searches looked for
  the *reader's* vocabulary (`cube`, `hole`, `voxel`, `linked list`) where the text says chest, void,
  blocks, cells, connections. One search did look for `cube`, and filtered it by co-occurrence with "hole"
  — the text says "void", so the filter discarded the answer.

  **The retrieval engine found it on every phrasing, including the one with no shared vocabulary at all.**
  "The protagonist discovers Equestria is a doubly linked list of voxels" returns the right document at
  rank 2 and the right passage at rank 4; the text's own wording does better still, but the concept-level
  query works. That is the reverse of the investigator's experience with grep, and it is the clearest
  single demonstration in this directory of what the semantic arm is *for*.

  Worth keeping as a note on method: **a labelled item with a wrong label is worse than no item**, because
  it silently biases every score computed from the set afterwards — which is why this one was held out of
  `fiction_questions.json` while its label was in doubt.

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

  **A harder class sits behind it, and retrieval is the wrong tool for that one entirely.** "Which of these
  is an Asimov pastiche?" is answerable in a second by anyone who has read *The Last Question* — the story
  is `The Last Optimization`, it opens on entropy and closes on "And friendship was magic", echoing "And
  there was light". The word "Asimov" occurs nowhere in it. The answer is not distributed across the
  document the way the setting was; it is not in the corpus **at all**. It lives in the reader's knowledge
  of a work outside the corpus, and no indexing strategy over these files can recover it.

  That is worth separating from the document-level gap, because it has a different remedy: the *model*
  already knows *The Last Question*, so the answer comes from world knowledge applied to retrieved text,
  not from better retrieval. It marks the boundary of what the retrieval layer should be asked to do —
  and, usefully, it is a case where the confidence signal reporting "nothing here" is the *correct*
  behaviour rather than a failure to be tuned away.

  **All three probes came from hazy recollection, and one of them dissolved on inspection.** A fourth,
  "which is the post-apocalyptic village story", retrieved `My Life In Fimbria` on strong surface evidence
  (132 mentions of "village", "from before the fall of man and my retreat into hiding") — and the reader's
  verdict was that Fimbria is really the story where the protagonist edits the world and it turns out not
  to be the real Equestria Online. Contested ground truth, so it is not usable as a labelled item. Which is
  the limitation above meeting practice: human-recalled probes are a spot-check, and some of them do not
  survive being checked.

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

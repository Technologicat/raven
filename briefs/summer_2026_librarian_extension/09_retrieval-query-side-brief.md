# Brief: fix the query side of retrieval before reaching for a reranker

**What:** four changes to how `HybridIR` builds and fuses a query, plus the measurement setup that
decides between them. None of them needs a new model, and one of them removes information the pipeline
currently throws away.

**Why now:** the motivating complaint is that *the hybrid rank does not map cleanly to how good a result
is* — the observed failure being less topical matches from elsewhere in the corpus outscoring the ones
that actually answer the question. Reranking is the standard answer, and it is queued. But reranking is
also the most expensive answer, it wants VRAM on the machines that have least of it, and it is aimed at
the *scoring* stage — so it is worth knowing what the *query* stage is currently costing us first.

**Not a replacement for reranking.** Everything here shrinks the problem reranking has to solve, and
makes it measurable. Some of it may make reranking unnecessary at the current corpus size. That is an
outcome, not a claim.

## A shipped feature now depends on this (2026-07-28)

This started as quality work with no hard deadline. It has since acquired a caller. Brief 10's grounding
marker — *"this reply had nothing from your documents to stand on"* — needs to distinguish **matches
arrived** from **matches were any good**, and on its first test drive it could not: asked "what is 2 + 2?",
retrieval returned hydrogen-electrolysis documents, the marker read that as grounding, and stayed silent.

The signal it needs is **lever 1 plus the confidence signal below** — absolute scores surviving fusion, and
the `min_p`-style survivor count read off the distribution. Until those exist there is nothing to threshold
on, and against a real corpus the marker will almost never fire, because retrieval nearly always returns
*something*. That is a silent failure, not a visible one.

**Measured 2026-08-05, and only half of that sentence survived:** the survivor count is the wrong signal
for this consumer and points the wrong way, while the absolute vector similarity separates a question the
corpus can answer from one it cannot at AUROC 0.99. The marker wants the *level*, not the shape. See
"Built and measured" under lever 1.

So the ordering has changed: lever 1 and the confidence signal are now the front of this brief's queue,
ahead of the multi-query decomposition that was going to be built first. (The alternative route does not
run through here at all — inline citations, where the model reports its own grounding instead of Raven
inferring it. Worth weighing against lever 1 rather than assuming this one wins.)

**And the marker is not the only thing the signal pays for: a no-confidence match set can be dropped
instead of injected.** Injected matches are not free. They are prompt-processed on every turn they appear
in, and a question the corpus cannot answer still produces a full batch of best-of-a-bad-lot matches — tens
of thousands of tokens of noise, paid for as latency before the reply begins. Today the only remedy is
manual: switch the **Documents** toggle off when the conversation moves off-corpus, which is what the
toggle's tooltip now advises. A survivor count that says "nothing here cleared the bar" turns that into
something Raven can decide per turn, which is the same decision the user is currently making by hand and by
guess. So the confidence signal buys a correct marker, a cheaper turn, and one less thing to remember —
worth counting when weighing it against the citations route, which fixes only the marker.

## The finding that motivates the rest

Verified by reading `hybridir.py`, not inferred:

- The two engines' quality signals are applied as **absolute cutoffs, per engine, before fusion**
  (`keyword_score_threshold`, `semantic_distance_threshold`).
- Fusion is then **pure reciprocal rank fusion** (`reciprocal_rank_fusion`), which sums `1 / (rank + K)`
  over each engine's *position* list. `K = 60`.

So a chunk that scraped past the BM25 threshold at 0.11 and a chunk scoring 30 both arrive at fusion as
"rank 1", and both contribute exactly `1/61`. **The mapping from score to quality is discarded one line
before the rank that is supposed to carry it.** The complaint is not that the ranking is imperfect; it is
that the pipeline is asked to rank using information it deleted.

This is also why the symptom is *less topical matches winning*: when the query is a poor fit for the
corpus, both engines still return their best-of-a-bad-batch, still ranked 1..n, and RRF cannot tell that
batch apart from a good one. It gets worse the more candidates are retrieved — which is precisely the
direction reranking wants to push. RRF over 200 candidates is mostly position noise in the tail.

Note that `alpha` is **not** the knob for that widening, though it looks like one. It exists to compensate
for the span-joiner: adjacent matched chunks from the same document are merged, which *reduces* the
result count, so retrieval over-fetches by `alpha` to land near `k` after merging. Giving it a second job
("retrieve broadly for the reranker") would tangle two unrelated corrections in one number. A reranking
stage wants its own retrieve-wide parameter, with `alpha` still doing its merge compensation on top.

## Measured baseline (2026-07-28)

An evaluation set now exists — `investigations/retrieval/`, 99 known-item questions against the 11974-record
corpus — and it reorders what follows. Numbers and method are in that directory's README.

**What it establishes:**

- **Long, wandering messages retrieve far worse than focused ones** — MRR 0.315 against 0.535, R@5 0.41
  against 0.66. This is the largest effect in the data, it is lever 3's target, and it replicated when the
  set grew from 30 questions to 99. On that evidence **lever 3 should be built first**, not third.
- **Fusion earns its place.** The hybrid leads both single engines on every metric (MRR 0.486 against
  0.411 for BM25 and 0.363 for the vector arm).

**What it retracted.** The first run, at n=30, showed the hybrid *trailing* both single engines at R@5
while leading at R@20 — read at the time as rank-only fusion promoting mediocre-but-agreed-upon documents,
which is the behaviour this brief's opening section predicts. At n=99 the effect reversed outright (R@5
0.61 against 0.52 and 0.46). **It was noise, two questions wide.**

That costs lever 1 one of its two supports, and the brief should be honest about which one is gone. The
*structural* finding stands, because it is a property of the code rather than a measurement: fusion
discards the scores, so the rank cannot carry quality information it was never given. What is gone is the
claim that this is currently *hurting* — measured, RRF is doing its job well. So lever 1's remaining
deliverable is the **confidence signal**, which is about detecting a bad query rather than about ranking
good ones, and which the retraction does not touch. Replacing RRF itself has no measured motive.

The set is cheap to re-score (minutes) and moderately expensive to regenerate (over an hour of GPU time at
n≈100), so every lever below should be evaluated against it rather than argued about — and note that a
30-question set was enough to carry a factor-of-two effect and not enough to carry a few points of R@5.

**One caveat that constrains the whole brief, and it is easy to forget:** that corpus is one user's
hydrogen collection. Librarian indexes whatever the user puts in the directory — another science of the
year, AI papers, fanfiction. Any conclusion here that takes the form of *a tuned constant* is a conclusion
about hydrogen, not about retrieval. See the split under lever 1.

## The four levers

Ordered cheapest first, which is *not* the order to build them in — the measured baseline above puts
lever 3 first on evidence. Cost and priority are different questions, and both are worth seeing.

### 1. Let the scores survive fusion

RRF is scale-free by design, which is its whole appeal — no calibration between BM25 scores and cosine
distances, and no tuning against a corpus nobody has seen. That is the right default for a tool whose
document set is chosen by the user, and it should stay the default.

But "raw scores are incomparable" is true of one engine and not the other, and the asymmetry is the
opening:

- **Cosine distance is already calibrated.** With a normalized embedding model, 0.8 means roughly the same
  thing for every query — which is why `semantic_distance_threshold` works as an absolute cutoff at all.
- **BM25 is not.** Its scores move with IDF, document length and query length, so they are meaningful only
  within one query's result set.

So the two survive differently, and the split is the actionable part:

**Corpus-independent, safe to ship as a default:**

- **A query-local confidence signal, read off the shape of the score distribution.** This answers the
  question nothing currently answers — *did this query find anything, or is its best result merely the
  head of a flat list?* — and it does so without any constant to tune. See below; it went through two
  worse designs before arriving here, and both are recorded because the reasons they failed are the
  reasons this one works.

#### The confidence signal, and the two designs it replaces

The first attempt was **an absolute cutoff on vector distance**, on the grounds that cosine distance is
calibrated by the embedding model. It is broken, for a reason worth keeping: applied per document, it
discards exactly the results BM25 exists to catch — a rare proper noun, an acronym, an exact term the
embedder has no signal for and the keyword arm nails. Gating those away removes the point of having two
engines. Whatever this signal is, it must be **per query, advisory** — never a per-document filter.

The second attempt was **calibrating a threshold against the corpus**: sample random chunk pairs at index
time, take the background distance distribution, express the cutoff as a percentile of it. That handles
the fact that the scale of "close" is a property of the collection — everything in a narrow corpus sits in
one neighbourhood, so a constant that suits hydrogen abstracts is wrong for fanfiction. But a corpus is
not a fixed object: documents are added, updated and deleted whenever the user drops a file in the magic
directory, so the statistic goes stale continuously. It could ride the existing `commit()` lifecycle
cheaply enough, but growth exposes a second flaw that recomputation does not fix — a corpus that has
become *bimodal* has no single correct scale, so a global percentile is not stale so much as meaningless.

What survives both objections is to stop asking about the corpus at all and ask only about **this query's
own returned scores**, of which there are already `alpha * k`:

- A query that found something produces a **sharp head and a long tail** — the top result stands off from
  the rest.
- A query that found nothing produces a **flat list** — everything equidistant, and the "best" result is
  best by noise.

This is `min_p` sampling, transplanted: keep what scores at least `min_p` times the best score. Note the
inversion before implementing it, because it is counterintuitive — on a *flat* distribution the bar is low
and nearly everything survives, on a sharp one almost nothing does. So the filter runs backwards for our
purposes and the **survivor count is the signal**: few survivors means a confident query, many means noise.

The properties that make this the one to build: no constant fitted to a corpus, no persisted statistic, no
recalibration, and immunity to the corpus growing, shrinking or changing character — because it only ever
compares a query's results to each other. It survives an embedder swap for the same reason.

Its honest limit: it cannot tell "flat because nothing matches" from "flat because everything matches
equally well". The second case has no ranking problem to solve, so the failure is benign.

**Borrow the idea, not the number.** `min_p` operates on a proper probability distribution — normalized,
summing to 1, bounded above by 1 — and none of that is true of BM25 scores, which are unnormalized and
unbounded. The ratio-to-best test still works, and is in fact scale-free in a way `min_p` itself is not,
which is exactly why it survives a corpus changing character or an embedder being swapped. But it also
means the tuned values in circulation for LLM sampling (roughly 0.02 to 0.1) carry no information about
what to use here: they were fitted against a distribution with different properties. The value has to come
off the evaluation set.

**Implementation trap:** this must run on the **per-engine raw scores, before fusion**. RRF output is
`1/(rank + K)` by construction, so its distribution has the same shape whatever was retrieved — running a
shape test on it measures the arithmetic, not the corpus. One more reason the raw scores have to survive
to the fusion boundary.

#### Built and measured 2026-08-05: the shape loses to the level, and the rejected design wins

`score_sharpness` was implemented as specified above and scored against `investigations/retrieval/`
(`sharpness.py`, which is the apparatus for everything in this subsection). Two questions were asked,
because the consumers below want different things, and they came back with different answers.

**Question A — does sharpness predict retrieval success?** Over the 99 known-item questions, is the
signal higher where the gold document was found than where it was missed? Reported as AUROC — the
probability that a success outranks a failure, so 0.5 discriminates nothing.

| signal | all (77/22) | focused (63/14) | rambling (14/8) |
|---|---|---|---|
| keyword `best/mean` | **0.734** | 0.671 | **0.893** |
| keyword sharpness @ 0.7 | 0.730 | **0.681** | 0.750 |
| vector `best/mean` | 0.650 | 0.621 | 0.688 |
| vector best similarity | 0.563 | 0.588 | 0.616 |

Moderate, real, and strongest exactly where the brief expected the problem to be — the rambling
subset, where a bad query is the whole failure mode. Note the *n* before leaning on that 0.893.

**Question B — does sharpness separate on-corpus from off-corpus?** The known-item questions were
written *from* the corpus, so every one of them is answerable, and question A structurally cannot see
the case brief 10's grounding marker exists for. So the 99 questions were scored against 16
hand-written probes with no answer in a hydrogen corpus: eight plainly off-topic, four conversational
pleasantries, and four *adjacent* — real science, plausibly phrased, still not in this corpus, which
is the only one of the three that is a hard test.

| signal | vs. off-topic (8) | vs. pleasantry (4) | vs. adjacent (4) |
|---|---|---|---|
| **vector best similarity** | **0.996** | **1.000** | **0.987** |
| keyword best score | 0.986 | 1.000 | 0.924 |
| keyword sharpness @ 0.9 | 0.684 | 0.535 | 0.444 |
| vector sharpness @ 0.9 | 0.142 | 0.331 | 0.201 |

**The shape reading does not merely fail here, it points the wrong way**, and the mechanism is
measured rather than guessed: an off-corpus query reads *sharper* than an on-corpus one (mean vector
sharpness @ 0.9 of 0.92 against 0.53) while its level is less than half as high (mean best similarity
0.31 against 0.67). With nothing genuinely matching, the accidental best hit stands well clear of an
already-low field; a question the corpus can answer pulls twenty genuinely related chunks that all sit
near the top and therefore look flat. So sharpness is *anti*-correlated with the thing it was invented
to detect. The brief's own stated limit — that it cannot tell flat-because-nothing-matches from
flat-because-everything-matches — turns out to be the benign half of a worse problem.

**And the winner is the design this section opens by rejecting.** Absolute vector distance was
dismissed on the grounds that, applied per document, it discards exactly what BM25 exists to
catch. That objection stands and is untouched — but it was an argument against a *per-document
filter*, and this is the per-query advisory reading the same paragraph demanded instead. The
separation is wide enough to be usable rather than merely significant:

| group | n | min | median | max |
|---|---|---|---|---|
| known-item questions | 99 | 0.460 | 0.670 | 0.823 |
| adjacent probes | 4 | 0.386 | 0.452 | 0.509 |
| off-topic + pleasantries | 12 | 0.160 | 0.320 | 0.479 |

A cut at 0.45 rejects 13 of 16 probes and **none** of the 99 questions; at 0.50 it rejects 15 of 16
and 4 of 99. That is a real operating point, not a curve-fitting artifact.

**What this does not establish, and the caution is the whole of it.** This is one corpus, and the
argument against absolute thresholds was never that they do not work — it was that *the scale of close
is a property of the collection*. Nothing here tests that: a single hydrogen corpus cannot show whether
0.45 travels to fanfiction. So the threshold is corpus-dependent until a second corpus says otherwise,
which is precisely the measurement the evaluation set's README already lists as needed, and the reason
to want a curated set in a literature the maintainer knows from the inside. Two further limits worth
stating plainly: the 16 probes are hand-written by the implementer rather than sampled, and the
"adjacent" column — the only hard one — rests on four negatives.

**So the levers split, and this is the durable finding.** Two signals, answering two questions, for
two different consumers:

- **Level** (best vector similarity) answers *can this corpus answer this at all?* — the grounding
  marker, dropping a no-confidence match set instead of injecting it, and the per-subquery gate that
  lever 3 is waiting on. A pleasantry scores 0.32 against a question's 0.67.
- **Shape** (keyword `best/mean`) answers *given a corpus that can answer it, did this query land
  well?* — adaptive `k`, and the reranking triage. It says nothing about the first question and must
  not be asked it.

`score_sharpness` ships as the shape reading, with no consumer in the retrieval path, because the
shape's own consumer (adaptive `k`) has not been measured yet. The level has no implementation at all
yet; it is one `max()` over the vector arm's candidates, compared against the constant the arXiv
section below settles on.

#### Verdict, 2026-08-05: the level ships as one global constant, and only as a coarse signal

*(This heading said "per collection" until a third corpus was measured later the same day. The two
subsections below are in the order they were written — the fiction result, then the arXiv result that
overturned its conclusion while leaving its evidence intact. Read both; the second does not repeat the
first.)*

A second corpus was built to settle the one thing the hydrogen numbers could not — whether an absolute cut
is a fact about retrieval or a fact about that collection. 19 Optimalverse stories, 2.2M characters, as far
from scientific abstracts as a document set gets while remaining something someone might index. Method and
full numbers in `investigations/retrieval/`; the answer is in two parts.

**The constant does not travel.** On-corpus best vector similarity runs 0.460–0.823 on hydrogen and
0.352–0.804 on fiction, medians 0.670 against 0.519. The 0.45 cut that rejected *none* of 99 hydrogen
questions rejects **24 of 88** fiction ones. Shipping it globally would have marked 27% of answerable
questions on a narrative corpus as ungrounded — a confident wrong answer, which is worse than the silent
failure it was meant to fix. **This read as making it a per-collection setting** — see the arXiv section
below, where a third corpus shows the right conclusion is a *different* global constant rather than a
per-collection one.

**The signal detects the coarse case and little else.** Against negatives from a different field entirely it
is essentially perfect, and this is now measured in both directions rather than assumed symmetric: 0.9998
scoring 88 fiction questions against 99 hydrogen ones on the fiction index, 0.9997 the other way round on
the hydrogen index. (The two runs query different indexes, so every similarity value differs and nothing
forced them to agree — the reverse direction was run on 2026-08-05 for that reason.) Against *adjacent*
negatives — questions written from Optimalverse stories deliberately held out
of the index, same universe and site and generator — it is 0.742, and the distributions overlap so widely
that no cut preserving the on-corpus questions rejects any of them.

**A third corpus settles it: one constant near 0.40, and the per-collection machinery is not needed
(2026-08-05).** This section was written twice in one day, and the first version is worth stating because
the correction is the useful part.

The first attempt reasoned that "per collection" must mean *calibrated*, not configured — a threshold
cannot be derived from anything a user knows, so a knob hands them a calibration problem dressed as a
preference. Calibrate it at index time from a fixed off-corpus probe set, the argument went, since a probe
about sourdough is off-corpus for every collection anyone would build. Measured against the fixed 0.45,
p75-of-probes cut fiction's losses from 24/88 to 3/88 at no cost on hydrogen. Two corpora agreed.

**A third corpus — 1268 arXiv AI/ML abstracts — refuted it.** On arXiv, p75-of-probes cuts at 0.282 and
lets **72.8%** of off-corpus negatives through. Scored on the worst case across all three, which is what a
shipped default has to survive:

| estimator | worst on-corpus lost | worst negatives missed |
|---|---|---|
| p75 of probes | 3.4% | **72.8%** |
| p90 of probes | 6.8% | 14.6% |
| fixed 0.45 | **27.3%** | 6.9% |
| **fixed 0.40** | **6.8%** | **15.3%** |

A fixed 0.40 matches the best probe-calibrated estimator on both axes, so the calibration machinery buys
nothing. The first result was an artifact of comparing against 0.45 — a constant chosen on hydrogen alone,
and so the wrong incumbent to beat.

**And the reason is the more useful finding.** arXiv AI is the *near* well: same genre, same register, same
length distribution as the hydrogen abstracts, differing only in topic. It separates at AUROC 0.999 —
identical to fiction, the far well — and the two negative distributions lie on top of each other. So the
signal reads *topical match*, not register. The standing objection to any absolute cut was that the scale
of "close" belongs to the collection; a score that measures whether the query is about what the corpus is
about does not have much of a per-collection scale to begin with. That is why one constant works, and it is
why the second attempt was solving a problem smaller than it looked.

**What to ship:** ~~one constant, near 0.40~~ — see the next subsection. Three corpora chose 0.40, the
caveat above said a fourth could unseat it, and a fourth did, within hours.

#### Rescope, 2026-08-06: reranking is the lever the measurements point at

Taking stock with four corpora measured, against what this brief set out to do. The brief exists because
Raven should have better retrieval; it is ours, so the sub-goals are not binding.

**1. Detect that the corpus cannot answer the query — not shippable.** The ordering is excellent (AUROC
0.99+ on every corpus, near and far) and the absolute calibration is not transferable, as the four-corpus
result below records. There is no honest global cut, and no label-free way found so far to derive a local
one. Parked, not abandoned: the fulltext experiment may yet explain the corpus-to-corpus scale shift.

**2. Adaptive `k` from distribution shape — real, weak, and aimed at the wrong deficit.** The shape reading
does predict whether a query's gold document lands at rank 1, which is the job it was actually built for
(as opposed to on/off-corpus detection, where it was refuted). But choosing a signal per corpus is how
today's two mistakes happened, so what matters is the worst case across corpora:

| signal | worst | hydrogen | arxiv-ai | banichuk | fiction |
|---|---|---|---|---|---|
| `keyword sharpness @ 0.5` | **0.635** | 0.650 | 0.646 | 0.707 | 0.635 |
| `keyword best/mean` | 0.605 | **0.785** | 0.669 | 0.637 | 0.605 |

The headline 0.785 is hydrogen-specific; the best *consistent* signal is `score_sharpness` at ratio 0.5,
worst-case 0.635. Above chance, but not enough to size a context budget on — and it is the wrong lever
anyway, which the next table shows.

**3. Neural reranking — this is where the measured headroom is.** Retrieval as it stands, k=20:

| corpus | gold at rank 1 | gold found within k=20 |
|---|---|---|
| hydrogen | 39% | 78% |
| arxiv-ai | 42% | 84% |
| banichuk | 9% | 42% |
| fiction | 70% | 95% |

**The right document is usually already in the candidate set; it is just not at the top.** On the abstract
corpora that is a 39→78 and 42→84 gap, and closing it is the textbook cross-encoder job — reorder the
candidates you already have. Nothing about adaptive `k` addresses it, and a `k` that *shrank* would make it
worse by discarding gold documents currently sitting at ranks 2–20.

So the rescope: **reranking is the deliverable, adaptive `k` is deferred, and off-corpus detection is
parked.** Adaptive `k` becomes interesting again *after* reranking, and for the opposite reason to the
original one — if reranking concentrates the relevant material at the top, a smaller `k` becomes safe, and
the same shape signal would then be sizing a genuinely better-ordered list.

**Falsifiable predictions, written before the run:**

- Reranking helps **banichuk most in relative terms**. A bi-encoder must compress a ten-word title into a
  vector that also has to sit near the question's vector; a cross-encoder reads the question and the title
  together and never forms that vector. Titles are where the bi-encoder is structurally weakest, and
  banichuk's 9% rank-1 is the worst number in the table.
- Reranking helps **fiction least**, at 70% rank-1 already, with the least room above it.
- It should move `holdout-and-father` (`investigations/retrieval/probes.json`), which is a bi-encoder
  failure by construction: the content is in a chunk, the query is analytical, the passage is dialogue.
- It will **not** fix `offline-in-america` or `asimov-pastiche`. Neither has a passage to score highly —
  one is a document-level property never stated, the other is not in the corpus at all. A reranker that
  appeared to fix them would mean the labels are wrong, not that it is better than expected.

**A bonus worth testing but not promising:** a cross-encoder emits a *relevance* score rather than a
similarity, trained against graded judgements, so it may be far more comparable across corpora than cosine
distance turned out to be. If so, sub-goal 1 comes back for free on the reranker's score instead. Stated as
a hypothesis; today's lesson is that plausible mechanisms need measuring.

##### Correction to the adaptive-`k` verdict: the measurement above answers a different question

Juha's framing, 2026-08-06, and it invalidates the dismissal rather than softening it. Adaptive `k` was
weighed above as a *context budget* decision — how many results to spend on this query. The more
interesting version is **detecting whether a question is specific or general against this corpus**:

- *"What is the drift rate of the Kelvin-3 microarray?"* — one document answers it; more results are noise.
- *"What component models are needed for a green hydrogen value chain, assuming photovoltaics?"* — a
  synthesis question, where the answer is assembled from many documents and a small `k` truncates it.

**The 0.635 figure says nothing about this**, and cannot, because **every question in all four sets is a
known-item question by construction**. Each was written *from* one passage and is answerable by one
document; `rambling` wanders on the way but still lands on a single gold document. Specificity has no
variance in the eval data, so what was measured is *difficulty* — will this query's one right answer reach
rank 1 — which is a different variable that happens to be measurable with the sets we have.

**And the mechanism points the right way this time**, which is worth stating carefully given how the same
signal was refuted for off-corpus detection. `score_sharpness` measures how concentrated a query's matches
are. For on/off-corpus that was backwards: a query with nothing matching reads *sharp*, because an
accidental best hit stands clear of an already-low field. For specific-vs-general the semantics line up —
a general question genuinely matches many documents well, giving a flat distribution and low sharpness;
a specific one concentrates. Same primitive, third use, and the first where its meaning matches the job.

**What it needs is a question class that does not exist yet**: `broad` or `synthesis`, written to be
answerable only by combining several documents, with *a set* of gold documents rather than one. That also
means a different metric — recall over the gold set, not rank of the gold document — so it is a genuine
addition to the harness rather than a new prompt. Worth building, and it is the piece that would let
sub-goal 2 be evaluated on its own terms for the first time.

**And it looks like a third catch-22** — knowing which documents *jointly* answer a question is what the
retrieval under test is for. It is not, and the escape is the one the existing sets already use: the
known-item design never searches for a question's answer, it **writes the question from the document**, so
the label holds by construction. The same trick generalizes directly. Show a model several documents, ask
for a question that needs *all* of them, and the gold set is the documents you showed it.

Most of the machinery is already there. `make_questions.py`'s `rambling` path samples groups of three and
asks for a question only the TARGET answers; the synthesis variant asks for one that needs the whole group,
and records `gold` as the group instead of the target. Prompt and gold field change; the sampler does not.

Two design points that do need deciding:

- **Sample related documents, not random ones.** Three unrelated abstracts force an artificial question
  that no user would ask. Nearest neighbours in embedding space are the cheap grouping — seed on one
  document, take its neighbours — and note this *uses* retrieval without being circular, because it is
  forming the group, not finding an answer to a pre-existing question. (Visualizer clusters would also
  serve, at the cost of depending on that pipeline.)
- **The metric understates, in the familiar way.** Other documents in the corpus may also contribute to a
  synthesis answer while not being in the gold set, so recall over that set is a floor — exactly as
  known-item recall is a floor on precision. Same limitation, same shape, and worth stating in the set's
  own documentation rather than rediscovering.

##### And the recall@k curve is only meaningful where k is small relative to the corpus

Juha's point, and it bounds the experiment above: recall rises with `k` until `k` reaches the corpus size,
where it hits 100% and the first-stage ranking has contributed nothing — the reranker would be doing all
the work on an unranked pile. So the curve is informative only while `k` is a small fraction of the
collection.

**The denominator is the chunk count, not the document count** — Raven's retrieval is chunk-level
throughout, so `k` counts chunks and a corpus of few long documents is not a small collection. Fiction's 19
documents hold 2977 chunks (measured from the live index, not estimated), which makes it one of the *larger*
collections here:

| corpus | documents | chunks | k=200 as a share |
|---|---|---|---|
| hydrogen | 11974 | ~36000 | 0.6% |
| arxiv-ai | 1268 | ~3070 | 6.5% |
| fiction | 19 | **2977** | 6.7% |
| banichuk | 541 | 541 | **37%** |

So only **banichuk** is compromised, and for a reason specific to it: its records are sub-chunk-size, so
chunks and documents coincide and there is no multiplier. Fiction, which the document count made look
degenerate, is fine — comparable to arxiv-ai. (A bytes-divided-by-stride estimate gets banichuk wrong for
the same reason, predicting 229; chunks never span documents, so the floor is one per document.)

**And that table contains a result the fulltext experiment was going to be needed for.** On-corpus median
similarity is *monotonic in chunk count* across all four — 541 → 0.395, 2977 → 0.519, 3070 → 0.549,
36000 → 0.670 — and the decisive pair is fiction against arxiv-ai. They sit next to each other in chunk
count (2977 vs ~3070) and next to each other in similarity (0.519 vs 0.549), while differing completely in
document shape: 19 long narrative documents against 1268 short scientific ones. If *text per chunk* drove
the score, those two should differ; they barely do. If *number of chunks searched* drove it — a best-of-N
effect, needing no embedding story at all — they should match, and they do.

That is four points and a confound (genre still varies with everything else), so it is evidence rather than
settlement. But it shifts the prior substantially toward best-of-N, and it sharpens the fulltext
prediction into something crisper: indexing the same 1268 papers as fulltext takes them from ~3070 chunks
to perhaps 60000, so best-of-N predicts on-corpus similarity **rising to around hydrogen's 0.670**, while
text-per-chunk predicts it barely moving. One run, and the two hypotheses are far apart.

The saturation point is itself the number worth having, and not only as a cost bound: **it is where the
cheap first stage stops earning its keep.** If recall@200 ≫ recall@20 on hydrogen, the ranking is weak and
the reranker is carrying the result; if they are close, the first stage is doing real work and the
reranker is refining rather than rescuing. That distinction decides how much to invest in each stage, and
it comes free with the same four runs.

##### Retrieve deep, then rerank: the candidate set should be much larger than 20

Also Juha's, same date, and it is the standard retrieve-and-rerank shape rather than anything exotic: fetch
a wide, cheap candidate set, then spend an expensive model only on reordering it. Local BM25 and vector
queries are cheap, so `k=20` for the *candidate* stage is a limit inherited from the days when `k` was also
what reached the LLM, and there is no reason for the two to be the same number.

The measured case for it is strong: gold is at rank 1 for 39/42/9/70% but within k=20 for 78/84/42/95%, so
a large fraction of the misses are already ordering failures rather than retrieval failures. banichuk at
42% within k=20 has the most to gain.

~~and **going deeper can only add more of them**, since recall@k is monotone in k.~~ **Struck 2026-08-06 —
measured false, and the way it is false matters.** Recall@k is monotone in `k` *within one retrieval*, but
the sweep depth is not a truncation of a fixed ranking: under reciprocal rank fusion the candidate lists
feed the fusion, so retrieving deeper changes the order of what was already there. Measured below.

The constraint is reranker latency, which is linear in candidate count, so **the first measurement needs no
reranker at all**: run `evaluate.py` at k = 20, 50, 100, 200 and find where recall saturates. Reranking 200
candidates is pointless if recall@100 already has them, and that curve costs four cheap runs per corpus.
Do that before choosing the candidate depth, then size the rerank stage against the latency budget.

**Where the reranker runs is a second measurement, not an assumption.** GPU preferred, CPU as the fallback
— the candidate model is small enough that CPU may well be sufficient at a sane candidate depth, which is
what made it attractive in the first place, but "may well be" is exactly the kind of claim this brief has
been burned on twice today. Time it at each candidate depth on both, and let the numbers pick the default.
Note the GPU is not free here: it already holds the embedding model, and `briefs/06`'s VRAM accounting is
the place that has to absorb another resident model.

Model candidate and cost argument are in the reranking section further down. Scope for the 0.2.8 window:
rerank a candidate set deeper than the k=20 that reaches the LLM, measure with `evaluate.py` across all
four corpora, and ship it behind a config toggle. **The measured curve below sets that depth at 100** —
hydrogen reaches 89.9% there against 74.7% at 20, and the latency cost is bounded by a number we can time
before committing to it.

**Order of work, because the obvious order is wrong.** The three open threads are the fulltext experiment,
the recall@k curve, and the reranker itself, and the temptation is to take them in that order since the
fulltext corpus is the newest arrival. That puts the longest pole in front of the only ship-critical item:

1. **Start the fulltext build first and walk away** — 170 downloads at arXiv's rate limit (the count was
   243 until two counting errors in the drift measurement were found; see the investigation README), then
   extraction and indexing against 4 minutes for the same papers' abstracts. It is hours of machine time
   and minutes of human time, so it should be running in the background, not blocking. *Started
   2026-08-06.*
2. **Then the recall@k curve** (k = 20, 50, 100, 200, on hydrogen and arxiv-ai). No new code, four cheap
   runs, and it is what sizes the reranker's candidate stage — so it gates the deliverable. *Done
   2026-08-06 — measured above, and it clears the reranker to proceed.* One correction to the plan as
   written: it needed a single sweep at k=200 rather than four runs, since `sharpness.py` records the gold
   rank and every shallower k reads off the same file.
3. **Then the reranker**, which is the deliverable.
4. **Synthesis questions last.** They feed adaptive `k`, which this section defers, so they are genuinely
   fourth unless the reranker lands early.

The fulltext experiment answers a *science* question — which mechanism sets the similarity level — and
nothing in the reranker depends on its answer. Worth having, not worth waiting on.

##### The curve, measured (2026-08-06): the reranker has room, and depth is not free

`sharpness.py <corpus> 200 --db-dir …` against the parked indexes, read with `recall_curve.py`. 99
on-corpus questions each.

| k | 1 | 5 | 10 | 20 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|---|
| **hydrogen** | 38.4% | 56.6% | 66.7% | 74.7% | 84.8% | 89.9% | 96.0% |
| **arxiv-ai** | 43.4% | 65.7% | 75.8% | 80.8% | 92.9% | 98.0% | 100.0% |

**This is the case for the reranker, in one line:** the gold document is findable 96% / 100% of the time,
and *first* 38% / 43% of the time. That ~57-point gap is ordering, which is exactly what a cross-encoder
reorders. On hydrogen, 4 questions of 99 never surface at all — a hard ceiling the reranker cannot lift,
but a low one; on arxiv-ai there is no ceiling at all.

**Read the two curves at different discounts, because `k` means different things in them.** Measured chunk
counts (the denominator, since retrieval is chunk-level):

| corpus | documents | chunks | k=200 as a share |
|---|---|---|---|
| hydrogen | 11974 | 31600 | 0.6% |
| arxiv-ai | 1268 | 2596 | 7.7% |
| fiction | — | 2977 | 6.7% |
| banichuk | — | 542 | 37% |

hydrogen's 96% at 0.6% of the corpus is a real retrieval result. arxiv-ai's 100% at 7.7% is a weaker
claim wearing the same notation — "read a thirteenth of the corpus and you have everything" is closer to
enumeration than to search. The two are not a replication of each other, and quoting them as a pair
without this column would overstate both. **banichuk cannot be swept deep at all**: k=200 is 37% of it, so
its k=20 figure (42.4%) is the last honest point on that curve.

**Neither curve saturates where it matters.** hydrogen's last step gains *more* than the one before it
(+6.1 after +5.1), so the "find where recall saturates" plan does not terminate in this range; arxiv-ai
saturates only by exhausting a small corpus. Choose the candidate depth against the latency budget
instead, and treat these as floors rather than plateaus.

**Focused and rambling questions separate on hydrogen, and stay separated.** Focused: 42.9% @1 rising to
98.7% @200. Rambling: 22.7% @1 rising to 86.4% @200 — the gap does not close with depth, which says the
vague questions are not merely ranked worse; a share of them have no single document that answers them,
the synthesis-question case arriving unbidden in a known-item test set.

On arxiv-ai the same split converges instead (rambling reaches 100% by k=100, focused by k=200), which is
what the corpus-size column predicts: with 2596 chunks, depth eventually finds everything regardless of
how the question was phrased. So the *separation* is the hydrogen result, and arxiv-ai neither confirms
nor contradicts it — the corpus is too small for the question to be asked there.

**And the finding that changes the experiment design: retrieval depth perturbs the ranking.** The same 99
questions against the same index scored 78% within k=20 when swept at k=20, and 74.7% when swept at k=200.
Diffing the per-question ranks (the old file is in git, so this is exact rather than inferred):

- 39 of 99 gold ranks changed.
- Of those in the top 20 at k=20, **5 fell out** at k=200 (to 21, 22, 23, 23, 33) and **2 entered** — net
  −3 questions, which is the whole 78 → 74.7 difference.
- Of the 72 in the top 20 under both, **16 moved**, in both directions (14→6, 20→5, and 1→4).

The mechanism is RRF: fusion reads the candidate lists, so lengthening them re-weights the result. Two
consequences for the reranker work, and the second is the one that would have quietly invalidated a
comparison:

1. **Benign for retrieve-deep-then-rerank.** Every displaced document landed at rank ≤ 33, so a k=100
   candidate set still contains all of them. Deep retrieval loses nothing the reranker will not see.
2. **The baseline must be the shipped configuration, k=20 with no reranker — not k=200 with no reranker.**
   The latter is 3 points worse at the top through fusion alone, so measuring against it would credit the
   reranker with recovering damage the experiment itself caused. Compare end-to-end pipelines, not stages.

#### The fourth corpus unseats the constant (2026-08-06): ship nothing yet

A titles-only bibliography — 541 hand-typed records, 303 bytes median, the shape a working researcher's
own BibTeX file actually has — puts on-corpus similarity at median **0.395**, against 0.519–0.670 for the
three corpora the constant was chosen on. At 0.40 it rejects **53 of 99** answerable questions. Scored
across all four, no constant survives: 0.35 loses 32.3% and misses 32.3%, 0.40 loses 53.5%, 0.45 loses
78.8%. Every value fails badly on *some* corpus a user might plausibly have.

The three-corpus conclusion was not wrong about its three corpora. It was drawn from a sample that happened
to hold the deciding variable fixed — the same failure as the two-corpus one, one level up, and the second
time in a day that a result died of the sample rather than of the reasoning.

**What decides it is the size of the matched unit, not of the document.** Chunking is ~1000 characters, and
the ordering follows chunk content rather than document length: fiction's documents are 45 kB yet it sits
mid-table, because its *chunks* are window-sized and so look like abstracts. banichuk is the only corpus
whose chunks fall far below the window. That is measurable at index time with no probes and no labels,
which is what makes it a candidate mechanism rather than a knob — but it is **confounded with corpus size**
(hydrogen scores 0.670 at 11974 documents against arXiv's 0.549 at 1268, on near-identical document shape),
and these four corpora cannot separate the two.

**So the grounding marker in brief 10 does not get a threshold from here yet.** What it can have, and what
is unchanged by all of this, is the *ordering* result: on-corpus questions separate from off-corpus ones at
AUROC 0.99+ on every corpus tested, near and far. The signal is real; only its absolute calibration is
unsettled. Options, in the order they should be tried:

1. **Run the fulltext experiment** (`investigations/retrieval/README.md`), which separates the two
   mechanisms because they predict opposite outcomes. Cheap — the PDFs already exist, and the question set
   and gold labels transfer unchanged.
2. **If chunk size is the driver**, the threshold becomes a function of a measurable index-time statistic
   rather than a constant, and the per-collection idea returns in a form that has nothing to do with the
   off-corpus probes that failed.
3. **Only then** pick an operating point, and state which corpus shapes it was validated on.

Shipping a constant now would mean a Librarian user with a hand-built bibliography — the exact person this
feature is for — seeing half their answerable questions marked ungrounded.

**The constant belongs to the embedder, and the embedder is scheduled to change.** Every number in this
section is a cosine similarity under `multi-qa-mpnet-base-cos-v1`, the currently configured
`qa_embedding_model`. The nomic-embed v1.5 migration in brief 06 replaces it with a different model, and a
different model puts its similarities on a different scale. That alone invalidates 0.40; nothing subtler is
needed to make the point.

(Asymmetric retrieval is *not* the reason, and it is worth saying so because the phrase invites the
mistake. A QA-type model like the current mpnet is already asymmetric in task — trained on question↔passage
pairs, which is exactly the "map a question near its answer" property this brief has been leaning on
throughout. What nomic changes is the *mechanism*: the asymmetry moves into explicit `search_query:` /
`search_document:` prefixes over shared weights, rather than being baked into what the weights were trained
for. Brief 06 rates that lift "small" and "marginal in context". So the migration is a scale change to
re-measure, and separately a new way to get retrieval quietly wrong — a caller that forgets the prefix gets
degraded embeddings with no error, which is brief 06's known Hindsight limitation.)

So the constant carries an explicit dependency: **re-measure it when the embedder changes, before shipping
the new embedder.** The cost of doing so is now small and that is the point of having built the harness —
four indexed corpora, one `sharpness.py <corpus>` run each, and the comparison table falls out. What must
not happen is the embedder swap landing while a stale 0.40 stays in the config, since the failure mode is
silent: a threshold in the wrong place does not error, it just starts calling answerable questions
ungrounded (or stops catching off-corpus ones), and nothing in the app would report that.

**That is still enough for the consumer this brief was blocked on.** Brief 10's grounding marker exists
because "what is 2 + 2?" returned electrolysis documents and read as grounding. That is the far case. The
marker does not need to know that one paper is missing; it needs to know the conversation left the corpus,
and that is measured at 1.000 — and at 0.999 against a corpus in the same genre, which is the case that
would have been easy to assume was harder. It ships as one constant near 0.40, described as coarse, and the
docs should not claim it can tell a missing document from a present one.

**What is refuted rather than deferred:** the `min_p`-style *shape* reading this section was built around.
It is anti-correlated with the on-corpus/off-corpus distinction on both corpora — an off-corpus query reads
*sharper*, because with nothing genuinely matching, an accidental best hit stands clear of an already-low
field. `score_sharpness` is kept, with no consumer in the retrieval path, because it is a correct
implementation of a reading that turns out to answer a different question (see below).

**And a finding that constrains any threshold, not just this one:** six phrasings of one verified question,
all retrieving the gold document at rank 1, span 0.229 of signal — wider than the gap between the two
corpora. The most conversational phrasing ("I remember one of them going to Switzerland — which was that?")
scores lowest, at 0.370. So the signal moves more with wording than with whether the corpus can answer,
which caps how sharp any cut can usefully be and is an argument for reporting a band rather than a verdict.

#### What consumes the signal

A signal with no consumer is a statistic. The through-line for choosing among these: **the shape of a
score distribution can justify deciding how much effort to spend; it is thinner evidence for deciding what
to say.** Effort decisions first, speech decisions only with measurement behind them.

Organized by *which condition fires*, since that is what an implementer needs, and since the two
conditions turn out to want quite different things.

##### Sharp head — the query worked

**Set `k` per query (build this first).** `docs_num_results` is a fixed constant today, applied
identically to a query that nailed one paper and a query that found mush. Let the shape set it: a sharp
head means take the head — three, five — and a flat list means take few or none. This is the *fewer,
better passages* outcome the reranking item wants, reached with no reranker, no VRAM, and no prompt
change, and it pays in exactly the currency the reranking item is denominated in (context, KV cache,
attention). It is also **evaluable offline against `investigations/retrieval/`** — "does adaptive `k` cost
recall?" needs no LLM in the loop and runs in seconds per configuration, so it can be settled before it is
shipped.

**Deepen the pool with pseudo-relevance feedback (worth trying; see lever 4).** A confident pass 1 is the
condition under which expanding the query from its own top results is safe, and it can surface papers the
user's phrasing missed. Note that this pulls the *opposite* way from adaptive `k` on the same trigger —
one widens the search, the other narrows what gets injected — and that is coherent rather than
contradictory: PRF improves the pool, adaptive `k` decides how much of the improved pool to spend context
on. Implement them in that order and the interaction is benign.

##### Flat list — the query did not work

**Tell the model (build this, after the RAG tool surface).** A line in the retrieval tool result — "the
best match for this query scored weakly" — is *data* rather than instruction, which is the right side of
the taxonomy in `done/08_context-injects-brief.md`.

**On the tool route, telling the model and triggering the re-query are the same act.** A model that may
author its own query does not need Raven to decide when; it needs to know that the first pass was thin.
That is the whole mechanism, and it is why this consumer is gated on the RAG tool surface rather than
merely sequenced after it — before the tool exists, telling the model its matches are weak informs it of
a problem it has no means to act on, which is how Q11's rejected mitigation earned its 29000 characters of
deliberation.

Worth being precise about what this adds over Q11, since the two look redundant and are not. They are
independent detectors of overlapping conditions, and each catches what the other misses:

- **The model's own reading** notices that the documents are on-topic but do not contain the fact. That is
  semantic, requires having read them, and fires with no signal from us — it is what Q11 observed.
- **The score shape** notices that nothing scored well, without reading anything. It catches the case
  where the retrieved text *looks* plausible and the corpus simply had nothing closer to offer.

So the signal hands the model something it cannot get by reading: how good this match is relative to what
the corpus could have produced. Additive, not duplicative.

**Measure the wording before shipping it**, for the reason above — this is exactly the shape of thing that
misfired in Q11. The probe already exists (`manual_tests/absent_fact.py`); run both temperatures and read
the reasoning length, not just the verdict.

**Not the no-match bypass — until it moves.** Firing the bypass on a flat result set is tempting and is
currently wrong: a flat set is the case where a second query is *most* likely to rescue the turn, and the
bypass ends the turn before that can happen. Once the bypass moves to the end of the agent loop — decided,
see "Related" below — the model gets its second query first and this becomes a safe consumer.

**Corpus-dependent, do not ship as a tuned constant:**

- **Convex combination**, `alpha * norm(bm25) + (1 - alpha) * norm(vector)` with per-query min-max
  normalization. This preserves the score *shape* that RRF flattens: a query where BM25's top hit towers
  over the rest keeps that gap. The hybrid-retrieval literature reports it beating RRF when `alpha` can be
  tuned in-domain, and losing when it cannot — *(recollection, not verified; check the source before this
  sentence is relied on)* — which is precisely the trade Raven cannot make globally, because the domain is
  whatever the user indexed.
- **RRF's `K`** (currently the paper's default of 60, not a tuned value). Smaller `K` sharpens the
  top-rank advantage. Cheap to sweep, and equally corpus-specific.

If either of these is wanted, its home is a per-collection setting rather than a global constant — which
is one more thing the queued docs-DB *scopes* work would make possible.

**A caveat that catches people, including the author of this brief:** per-query min-max normalization does
*not* solve the absolute-quality problem. It stretches every result set to [0, 1], so a uniformly terrible
set still yields a confident-looking 1.0 at the top — the exact failure being chased. The absolute signal
has to come from the vector arm's raw distance. Pair the two changes; shipping the normalization alone
reproduces the bug in a new coat.

### 2. Route the two arms differently

One query string currently serves both engines (`_query_body` takes a single `query`), and they want
opposite things:

- **The vector arm wants natural text.** It embeds with the `qa` role, which maps questions near their
  answers. A long question is fine input; a keyword salad is not.
- **The BM25 arm wants discriminative terms**, which is what `_tokenize` produces — except that it calls
  `text.lower()` *before* the spaCy analysis, discarding the capitalization the tagger uses to recognize a
  proper noun. Keep tokens tagged `PROPN`, or carrying internal capitals or digits, verbatim, and lowercase
  only what remains.

Note the second half is not hypothetical hygiene: the tagger is a neural model reading context, and it does
mangle names — "Elsevier" in a copyright line is tagged `ADJ` and lemmatized to "elsevi" (spaCy 3.8.14 /
en_core_web_sm 3.8.0). A keyword search for an unusual proper noun is lossy today.

### 3. Ask several small questions instead of one big one

The user's message is not a query. With the multiline composer it can be an essay, and an essay embeds to
a centroid that is near nothing in particular — which produces exactly the observed symptom, since a
centroid's nearest neighbours are chosen by *average* topicality rather than by answering anything.

Split the message (per sentence, or per question mark, or per paragraph) and fuse the per-query result
sets. A match then has to be good for *something specific that was asked*, not merely near the mean of
everything asked. It costs one embedding call per sentence.

**This is now the measured priority, not a guess.** The evaluation set's `rambling` questions — several
sentences of wandering context ending in one specific question, which is what the multiline composer
actually produces — retrieve at 0.292 MRR against 0.562 for focused ones. Nothing else in the brief has a
measured effect that size. Build this first.

### Built 2026-08-05, and it does not work as specified

Implemented as written above — whole message plus every qualifying sentence, all fused together — and
measured against `investigations/retrieval/`. It is **worse than not splitting**, on the subset it was built
for:

| condition | n | R@1 | R@5 | R@20 | MRR |
|---|---|---|---|---|---|
| *rambling*, whole message only | 22 | 0.23 | 0.41 | 0.64 | 0.315 |
| *rambling*, whole + subqueries | 22 | 0.18 | 0.45 | **0.50** | **0.286** |
| *focused*, whole message only | 77 | 0.44 | 0.66 | 0.82 | 0.535 |
| *focused*, whole + subqueries | 77 | 0.44 | 0.66 | 0.82 | 0.535 |

The focused rows are identical to three decimals, which is the control working: those questions do not
split, so nothing should move, and nothing does. That is also what rules out a plumbing bug — a fault in the
per-query indexing or the fusion would have moved both.

**The mechanism, which the numbers make legible.** A rambling message yields five to seven subqueries, so
the whole-message query holds one vote in seven. And the context sentences it is outvoted by *agree with
each other*, because they are all about the general topic — so RRF, which rewards agreement across lists,
promotes the generically topical documents over the one that answers the question. **That is this brief's
own opening complaint, reproduced by the fix for it.** The dilution mechanism is the one Juha predicted for
conversational sentences (see above); it turns out not to need pleasantries, because ordinary context
sentences do it too.

**What this does and does not retract.** The *diagnosis* stands and is untouched: rambling messages really
do retrieve at half the MRR of focused ones, and that is still the largest effect in the data. What is
refuted is the *remedy as specified* — "split it and fuse everything" — because equal votes for every
sentence is not a mechanism for finding the question, it is a mechanism for amplifying the context.

**The machinery is kept, defaulted off** (`HybridIR.query(multi_query=...)`, `split_into_subqueries`). The
split, the batched multi-query retrieval and the single flat fusion are all reusable and all measured as
correct; only the policy over them is wrong. **Juha, 2026-08-05: the jury is out until lever 1 exists** —
a per-subquery confidence test would drop the sentences that found nothing in particular, which is precisely
the set doing the damage here, and it is the same signal the conversational-sentence problem wants. So this
should be re-measured with lever 1 in hand before anything is concluded about multi-query as an idea.

Candidates to try at that point, cheapest first, each one re-scored against the set:

- **Per-subquery confidence gating** (lever 1). The principled one, and the reason to wait.
- **Far fewer subqueries.** Cap at two or three rather than eight, preferring the sentence carrying the
  question mark and the last sentence — the brief's own recency prior, which the implementation applied only
  as a tiebreak at the cap rather than as the selection rule.
- **Weight the whole-message query above its parts** in the fusion, so that it cannot be outvoted by the
  material it already contains.

**Its failure mode is the mirror image of the one it fixes**, and it has to be designed around rather than
discovered later. Splitting throws away exactly the context that a *short* question depends on. "I'm
working on alkaline electrolyzers. What is the specific energy consumption?" splits into a second sentence
that is about nothing at all — the topic lived in the first one, and the query that needed it no longer
has it.

The fix is free, because nothing says the split has to be a partition: **query with the whole message
*and* with each part, and fuse all of the result sets.** The whole-message query carries the context, the
per-part queries carry the specificity, and neither has to be right about which shape the message is —
which matters because that is not knowable in advance. Cost is one extra embedding call and one extra
BM25 pass per part, both cheap next to a generation.

Note what this does *not* fix: a message whose question is short and whose context is long enough to
dilute the whole-message embedding is still hard, because both queries are individually weak. That case
wants a query built by something that has read the message — which is pass 2, the RAG tool call, not
anything in this brief.

**The conversational sentence is the failure mode to design around, and it creates a dependency on lever 1**
(raised by Juha, 2026-08-05, while lever 3 was being built). A chat message is not all query: "Good evening!
How are you doing today? I've been reading about alkaline electrolyzers — what is their specific energy
consumption?" splits into three pieces, two of which are social. Those retrieve whatever happens to sit
nearest them in a corpus about something else, and then vote in the fusion with the same weight as the piece
that asked the actual question.

A minimum word count is the obvious guard and it is not sufficient: it catches "Good evening!" and lets "How
are you doing today?" through, which is five words of pure noise. Nor is a stoplist of pleasantries the
answer — it is unmaintainable, language-specific, and wrong the moment someone asks a genuine short question.

Two things bound the damage in the meantime, and it is worth knowing they are bounds rather than fixes. The
whole-message query is always in the fusion, so a social piece can dilute but never replace. And RRF rewards
*agreement across lists*: a noise subquery's hits are scattered and corroborated by nothing, so each collects
a single `1/(1+K)` vote, while a document the query set actually agrees on collects several. The cost lands
in the tail slots rather than at the top.

**The fix is lever 1's confidence signal, applied per subquery rather than per turn: a subquery whose own
score distribution is flat does not get a vote.** That is the same shape test, at a different granularity,
and it dissolves the conversational-sentence case without anyone maintaining a list of greetings — a
pleasantry against a technical corpus produces a textbook flat distribution. So the two levers compose more
tightly than the ordering above suggests: lever 3 shipped alone is bounded-but-diluted, and lever 1 is what
makes it clean. Whoever builds lever 1 should treat this as one of its consumers.

Recency is a usable prior when a cheaper cut is wanted: the last paragraph, or the sentence carrying the
question mark, is what the user is actually asking.

### 4. Pseudo-relevance feedback — for depth, not for rescue

Classical IR's answer to a weak query (Rocchio / RM3): run it, take the top few results, harvest their
high-IDF terms, re-query with the expansion. No model, no VRAM, one extra BM25 pass.

The lineage is worth knowing, because it explains the failure mode better than the algorithm does.
*Relevance feedback* — Rocchio, 1971, in Salton's SMART system — formalized what searchers had always done
by hand: read the good hits, notice the vocabulary the field actually uses, search again with it. The
step that makes it work is the human *looking* first. **Pseudo-**relevance feedback is that loop with the
looking removed: assume the top k were relevant, and skip the asking.
(https://en.wikipedia.org/wiki/Relevance_feedback)

So query drift is not a quirk of the algorithm; it is the missing judgment step, showing up exactly where
you would expect. Nobody mines a page of junk results for better search terms. Which is what the
confidence signal restores — it is not a new safeguard bolted on, it is the discrimination that "pseudo"
threw away, recovered from the score distribution instead of from a person.

**It is not a rescue mechanism, and an earlier draft of this brief had it wired backwards.** PRF is an
*amplifier*: it expands the query using whatever pass 1 returned, so when those results are not relevant
it drags the query further from the target — the classical failure known as **query drift**. So it must
*not* fire on the flat-distribution signal, which is precisely "pass 1 found noise". It fires on a
**sharp** one, to deepen a match that already worked.

Re-wired that way it has a real use case, and it is one of Librarian's central ones: **the literature
review sweep.** "Show me everything on X" is a *recall* question, and the user's phrasing is one sampling
of the vocabulary a field uses — PRF re-queries with the field's own words, harvested from the papers the
first query did find. That is exactly the gap between "the papers whose abstracts happen to match how the
user phrased it" and "the papers about the thing".

**Warning for whoever evaluates it: `investigations/retrieval/` cannot score this lever.** The set is
known-item — one gold document per question — so a mechanism that surfaces *more* relevant papers around
an already-found one scores exactly zero improvement, and may score worse if the extra results displace
the gold document. A null result there is not evidence against PRF; it is the metric being blind to the
thing PRF does. Measuring it needs set-level relevance (pooled judgments over the union of
configurations), which is the layer that directory's README describes as future work.

The general objection that *does* stick, and applies to anything else proposed in this slot: **building a
better query without reading the results is doing the LLM's job with 2010s tools.** Query construction
from extracted named entities is that idea in a different costume. Once pass 1 has *failed*, the thing
that can write a better query is the thing that can read — pass 2, the RAG tool call. PRF survives only
because it does something different: it does not rescue a failed query, it broadens a successful one.

*Not to be confused with lever 2*, which also uses spaCy's tagging. Using NER to *build* a query is
superseded; using POS tags to *stop destroying* the query we already have — so that `Elsevier` survives
tokenization instead of becoming `elsevi` — is not query construction at all. That one stays.

#### Second consumer: the tail of a result set that already worked

The literature-review sweep above is a *recall* use — find more of the papers. There is a second one that
is about *precision*, and it applies to every ordinary question rather than only to sweeps: when the top
match is good and the distribution is sharp, the remaining `k-1` slots are usually filled with noise that
merely happened to rank. PRF can spend those slots on more good matches instead, at zero LLM latency.

This is the framing that survives the RAG tool surface, and the rescue framing is the one that doesn't.
A model that can re-query will out-write any term-harvesting heuristic, because it has read the results —
but it cannot cheaply tell that positions 3–10 are junk. It just reads them and gets diluted. Discarding
a weak tail from the score distribution is a machine's job; writing a better question is the model's.

Note this needs a *different metric* from the sweep, and the known-item set cannot supply either: it has
one gold document per question, so it can say nothing about how well positions 2..k are spent.

#### Build it to measure it, not to ship it

The expansion is a pure function of a query and a result set, so it can be prototyped inside
`investigations/retrieval/` and scored offline, with no change to Librarian at all. Do that first. Promotion
into the live retrieval path is a separate decision that should follow an **in-conversation** measurement
taken *after* the RAG tool surface lands — because the tool changes what a weak pass 1 costs, and
therefore changes what automatic expansion is worth. The standalone number cannot see that interaction.

## Keep the levers; retarget them (Juha, 2026-08-05)

**Decision: retrieval quality goes back to the drawing board, and none of the machinery is reverted.** Both
levers were built, measured, and found not to do the job they were specified for. Neither is wrong code, and
both answer questions worth answering — just not the ones that motivated them. Reverting would spend the
build cost twice, since the next attempt needs the same parts.

What exists, and what it is now good for:

- **`score_sharpness`** — reads the shape of a score distribution, scale-free, no fitted constant. Refuted as
  an on-corpus/off-corpus detector, where it points the wrong way. It is the best *available* predictor of
  whether a query that the corpus can answer will land well (`vector best/mean` and the sharpness family
  lead measurement A on both corpora), which is what **adaptive `k`** wants — spend context on five results
  when the head is sharp and on fifteen when it is not. That consumer is unbuilt and unmeasured.
- **`split_into_subqueries` and the batched multi-query path** — the split, the per-query retrieval and the
  single flat fusion all measured correct; only equal-weight fusion over every sentence is wrong. Its most
  likely use is no longer decomposition-for-recall but **per-subquery gating** once there is a signal worth
  gating on, and the level is not it at the sentence granularity.
- **`query(return_extra_info=True).per_query`** — raw candidate scores before the thresholds. This is the
  part with no plausible replacement: any future scoring work needs the scores that fusion discards, and
  they now survive to the fusion boundary.

**What the drawing board has to account for**, from the fiction measurements — these are properties of the
problem rather than of any lever:

- The embedder bridges concept to text when the text is **expository** and fails when it is **dramatized**.
  A query describing a scene misses the scene and hits stories that discuss the topic abstractly.
- **Document-level questions** ("which of my documents is the one about X") are unanswerable by chunk
  retrieval unless the document happens to *state* the property in text. See `TODO.md`.
- **Phrasing moves the scores more than the corpus does**, so anything read off an absolute level is
  softer than it looks.

Which together point away from the query side: the remaining gaps want something that has *read* the
corpus — the RAG tool surface, or a document-level summary layer — rather than a better query string.

**That is stage 3 of `VISION.md`, reached from the opposite direction, and the convergence is worth naming.**
Stage 3 — *LLM as first-pass reviewer: read all of the selection against one question* — was specified from
the researcher's workflow, and that document already says of it, flatly, "That is not a retrieval problem."
These measurements are the same conclusion arrived at by exhausting the retrieval side and seeing what is
left over. Two of the three properties above are what a per-document reading pass simply dissolves: a
document-level question is trivial for something that has read the document, and a dramatized scene is
findable by a reader who is not matching vocabulary. Note too that stage 3's half-built piece is
**per-document summarization — shipped code, currently switched off** because it was built to run over a
whole dataset at import time rather than over a selection; and a summary layer is also exactly what the
document-level gap wants indexed. Whether those are one mechanism or two is a real design question, and not
one to answer here.

### Next, when this is picked up again (agreed 2026-08-05)

Two things, in this order:

1. **Ship the levers for what they measured well** — the off-corpus detector, at collection granularity,
   into brief 10's grounding marker and the drop-the-injects decision. This part is *finished*: measured on
   two corpora, with a limitation that is known and must be stated rather than discovered (coarse — it sees
   a conversation leaving the corpus, not a document missing from it).
2. **Work out what is still available on the retrieval side**, before conceding the remainder to stage 3.
   Open, and deliberately not pre-answered here. Candidates this brief already carries: **the MiniLM
   reranker**, which is now the strongest of them and has its own section above — CPU-only, named, and with
   a falsifiable prediction attached; lever 2's tokenizer fix, which none of today's results touch and which
   is independently motivated; adaptive `k`, which is `score_sharpness`'s surviving consumer and is
   offline-evaluable; lever 4's PRF. Plus whatever the fiction measurements suggest that nobody has thought
   of yet — that is the part worth arriving at with a clear head rather than by continuing down this list.

## Reranking: there is a named candidate, and it is now measurable (2026-08-05)

**Test this as part of this brief rather than after it.** The model is
`cross-encoder/ms-marco-MiniLM-L6-v2` — 23M parameters, CPU. It is recorded in
`monday-2026-08-03-checklist.md`, which is otherwise stale, so it is repeated here to keep it from being
lost with that file.

Two things changed today that turn it from a queued idea into an experiment that can be run:

- **CPU-only removes the objection that blocked it.** The cost table below prices a reranker in VRAM on
  laptop dGPUs where 8 GB is already crowded. At 23M on CPU there is nothing to displace; the card stays
  with the LLM. What needs measuring is latency over a shortlist, not whether it fits.
- **The known domain-shift risk is now testable, on two corpora rather than argued about.** MS MARCO is web
  search queries. Scientific abstracts are one shift from that and narrative fiction is another, and brief
  13 predicted exactly this use for the fiction corpus before it existed ("out-of-domain in three ways the
  current eval set cannot test … the MiniLM reranker is MS MARCO-trained"). Both corpora are now indexed,
  with question sets and harnesses.

**Which harness measures what, because they do not overlap and picking wrong wastes the run:**

- **`evaluate.py` on hydrogen** gives the recall/MRR numbers. A reranker is a retrieval configuration, which
  is what that harness was built to compare, and 12k documents give the known-item task real discrimination.
- **`run_probes.py` on fiction** gives the domain-shift and failure-class answer. Known-item is degenerate
  there (19 documents at k=20), so recall says nothing — but the probe set is stratified by failure class
  and reports per-class hit rates, which is the shape the question actually has.

**A specific prediction worth writing down before running it**, since it is the strongest reason to expect
a reranker to earn its place here rather than a general hope for better ranking. A cross-encoder reads query
and passage *together*, which is precisely the thing a bi-encoder cannot do — and today's sharpest measured
failure is the description-to-dramatization gap, where `holdout-and-father` retrieves its gold document at
rank 1 **zero times of two** and the gold's best chunk scores below eleven stories that are not about it.
That failure is a bi-encoder failure by construction. If the cross-encoder does not move that probe, the
mechanism is not what we think it is, and that is worth learning early. If it does, it is evidence for
reranking on exactly the corpus type where the query-side levers ran out.

Note the reranker also cannot fix the two classes that are not ranking problems: `document-level-unstated`
and `intertextual` stay unfixed, because there is no passage to score highly.

## What reranking is still for, after the levers

Sharpening the query does not make ranking *comparative*. A cross-encoder reads the query and a candidate
*together* and answers "does this answer that", which no bag-of-words score and no independent embedding
can do. If the residual failures after 1–4 are still "plausible-looking passage that does not answer the
question", that is a cross-encoder's job and nothing above substitutes for it.

Cost, for the retrieve-200-keep-20 shape, as **estimates to measure, not figures to trust**:

| option | VRAM | latency over 200 candidates |
|---|---|---|
| cross-encoder (BERT-base class, ~280M) on GPU | ~0.5–1 GB | order 0.1–0.3 s |
| same, on CPU | none | order seconds — needs measuring before it is dismissed or adopted |
| through the main LLM | none extra | ~50k tokens of prompt to rank, plus a full KV miss, twice |

The 1–7B neural rerankers that produce the good results are out of budget on the laptop dGPUs (8 GB is
crowded; 16 GB might absorb a couple), which is the constraint that makes the query side worth exhausting
first.

**Where the reranker sits relative to the span-joiner** is a design point, not a detail. Span-joining
stays — merged spans are what make a result readable, to a human and to a model alike — so the question is
whether scoring happens before or after the merge:

- *Score chunks, then merge.* Fits the existing machinery exactly: the merge already scores a span as the
  max over its chunks, so a reranked chunk score flows through unchanged. Also keeps every candidate
  within a cross-encoder's 512-token window.
- *Score merged spans.* Scores the thing that actually reaches the prompt, which is the more honest
  target — but a merged span runs from ~1000 to several thousand characters, so the long ones do not fit
  the window and would need truncation or chunk-wise scoring anyway, which collapses back into the first
  option.

So the first is probably right, and the reason is worth stating: it is the one that leaves span-joining
alone.

**Load-on-demand is an architectural question, not a local one.** Raven-server currently keeps every model
resident. Loading a reranker per turn wants uniform load-on-demand across the server modules, and on a
laptop the PCIe link (x8 on a dGPU, x4 over Thunderbolt) may make it a non-starter — that is a prior from
past experience, explicitly unverified, and it belongs on the measurement list rather than in a design
decision.

## Measuring it

None of the above is decidable by argument, and the failure being chased is a *ranking* failure, so it
needs labels:

- **The corpus exists**: the `.bib` entries already loaded into Librarian's document database (~12k). Real
  data, real queries, and the failure was observed on it.
- **The labels don't.** Twenty to thirty questions, each marked against which returned spans actually
  answer it, is enough to plot score against relevance and settle the finding above empirically: *is the
  rank informative at all, and where does it stop being informative?* That plot is the artifact this brief
  is really asking for — every lever above is a hypothesis about a different part of its shape.
- Keep the questions in the repo, the corpus out of it (copyright).

Do this before implementing any of 1–4, because two of them (score-aware fusion, multi-query) can be
evaluated offline against a labelled set in seconds, with no LLM in the loop at all.

## Related, not in scope

- **RAG access via tool-call** (`TODO.md`) — pass 2, where the model authors the query. Composes with all
  of the above; lever 4 is its cheap cousin.
- **Moving the no-match bypass to the end of the agent loop** — **decided, to be done with the RAG tool
  work.** The guard currently conflates two jobs that were the same job when there was only one pass:
  *whether the model runs* and *what the user is allowed to see*. Only the second is the anti-confabulation
  property. Let the model run and re-query; if the loop terminates with nothing grounded and speculation is
  off, substitute the no-match message then. The user still never sees an ungrounded answer, the model
  gets its second query in the case where a better query is most likely to help, and the guard costs
  nothing on the turns where retrieval worked. Not in this brief because it is a `scaffold` change, but it
  is a precondition for consumer 4 above.
- **Reranking itself** (`TODO_DEFERRED.md`) — this brief is what should run first.

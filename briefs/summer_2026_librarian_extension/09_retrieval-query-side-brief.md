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

An evaluation set now exists — `evaluation/retrieval/`, 30 known-item questions against the 11974-record
corpus — and it reorders what follows. Numbers and method are in that directory's README; the two results
that change the plan:

- **Long, wandering messages retrieve at roughly half the MRR of focused ones** (0.292 against 0.562), and
  on those the fusion is *beaten by the vector arm alone*. This is the largest effect measured, and it is
  lever 3's target. On that evidence lever 3 should be built first, not third.
- **The fusion trails both single engines at R@5 while leading at R@20.** Exactly the shape rank-only
  fusion predicts. But the gap is two questions wide at n=30 — grow the set to ~100 before acting on it.

The set is cheap to grow (about four minutes per run) and cheap to re-score, so every lever below should
be evaluated against it rather than argued about.

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
attention). It is also **evaluable offline against `evaluation/retrieval/`** — "does adaptive `k` cost
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
the taxonomy in `08_context-injects-brief.md`.

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

**Warning for whoever evaluates it: `evaluation/retrieval/` cannot score this lever.** The set is
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

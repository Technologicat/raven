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

## The four levers, cheapest first

### 1. Let the scores survive fusion

RRF is scale-free by design, which is its whole appeal — no calibration between BM25 scores and cosine
distances. The cheap fix keeps that property and adds back only what is missing:

- **Use RRF for ordering and the raw scores for admission.** The thresholds already exist; make them do
  more than a floor. A *relative* cutoff (drop anything below some fraction of the best score that engine
  returned this query) distinguishes "top of a good batch" from "top of a bad one", which is the exact
  distinction absolute thresholds cannot make.
- Or **weight each engine's RRF contribution by its normalized score**, which is a two-line change and
  keeps the rank-based backbone.

Either way the point is that a query which found nothing good should *return* less, rather than returning
its best rubbish with a confident rank.

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
everything asked. This is the single change most likely to move the metric, and it costs one embedding
call per sentence.

Recency is a usable prior when a cheaper cut is wanted: the last paragraph, or the sentence carrying the
question mark, is what the user is actually asking.

### 4. Pseudo-relevance feedback — a better query, for free

Classical IR's answer to "the query was bad" (Rocchio / RM3): run the query, take the top few results,
harvest their high-IDF terms, re-query with the expansion. No model, no VRAM, one extra BM25 pass.

It is worth naming here because it is *the pre-LLM version of the RAG tool call* — the same "read the
first pass, ask a better question" move, done statistically instead of with a generation. Where the tool
gives a smarter query for the price of a generation, PRF gives a somewhat better one for microseconds. The
two compose: PRF sharpens pass 1, so the tool gets a better starting point on the turns it fires.

## What reranking is still for, after all four

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
- **Moving the no-match bypass to the end of the agent loop** — so that a bad pass-1 query can be rescued
  by a pass-2 one instead of ending the turn. Only matters once the tool exists.
- **Reranking itself** (`TODO_DEFERRED.md`) — this brief is what should run first.

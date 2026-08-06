# Retrieval evaluation: findings

A standing summary of what the retrieval investigation has decided, organized by decision rather than by
the order things were tried. `README.md` in this directory remains the research notes — the historical
record, including the wrong turns and the reasoning that produced each result, which is where to look for
*why* something is claimed. This file is where to look for *what* is claimed.

Current as of **2026-08-06**. Corpora, instruments and the raw per-question data are described in the
README; the scripts are in this directory.

---

## 1. What this changed in the product

| change | evidence | confidence |
|---|---|---|
| **`docs_num_results` 20 → 50** | recall@k 74.7% → 84.8% on 12k abstracts, at +1.7 s prefill; the next doubling buys a third as much for four times the wait | measured on one corpus, one model; the prefill side is linear and should transfer by token count |
| **`docs_max_result_length` = 2000** | results had no length bound, so per-turn prefill was unbounded in principle and varied by an order of magnitude in practice | the bound is arithmetic; the *value* is a policy choice, and it also recovers 87% of the recall gap to unmerged chunks on fulltext |
| **`keyword_weight`, `rrf_k`, `merge` as query parameters** | the best arm weight differs by corpus (≈0.1 on titles, ≈0.5–0.6 on abstracts); single-arm modes were previously unreachable | the *shape* is solid on hydrogen at n=293; the cross-corpus claim is not yet (see §3) |
| **Extraction repairs UTF-16 surrogates** | three of 1268 arXiv PDFs failed to index, reported as success | root-caused and fixed |
| **Rescan keys on `document_id`, not absolute path** | moving or renaming the documents directory aborted the startup scan | root-caused and fixed |

Nothing else measured this sprint reached the bar for shipping.

---

## 2. What is settled, and settled negatively

These are closed. Reopening one needs new evidence, not new argument.

**Off-corpus detection by a fixed similarity threshold — dead.** No constant survives four corpora. A cut
at 0.40 loses 6.8% of answerable questions on the worst corpus of three, and 53.5% once a titles-only
corpus is included. Probe-calibrated per-collection thresholds do no better than the best constant.
*What survives:* the coarse case works essentially perfectly (AUROC ≈ 1.0 for "the conversation left the
corpus"), and that is what the grounding badge needs. The fine case — "this particular document is
missing, though its neighbours are here" — is barely detectable at AUROC 0.74 and should not be attempted.

**Multi-query decomposition — refuted.** Splitting a rambling message into sentences and fusing every
result set measured *worse* on exactly the case it targeted: rambling MRR 0.315 → 0.286. Cause: the
whole-message query becomes one vote in seven, and the context sentences agree with each other about the
general topic, so RRF promotes the generically-topical documents the change was meant to demote. Ships
disabled; the diagnosis that motivated it (rambling questions retrieve at half the MRR of focused ones)
stands untouched.

**Cross-encoder reranking — no configuration helped.** Two models (22.7M and 278M parameters, different
training data), three placements. Reranking the *fused* list is the worst option on all three corpora and
that result generalizes: do not do it. Reranking a single arm and fusing afterwards recovers most of the
loss but still does not beat plain fusion. The mechanism is that fusing two cheap independent signals
beats one expensive model's opinion — RRF's value is evidence diversity, and collapsing it is the cost.

**Per-query arm selection — no signal found.** The oracle is wide (+7.1 to +12.1 points at @20 over fixed
fusion), so the opportunity is real. But score sharpness and a scale-free standardized-top statistic both
sit at AUROC ≈ 0.53 on the two corpora with headroom. Note the sharpness variant is not merely weak but
ill-formed: it presumes a score whose zero means "no match", true of BM25 and false of cosine similarity.

**The RRF constant `K` — noise.** At n=99, `K=60 → K=10` looked like a consistent 9:2 improvement pooled
across corpora (p=0.065) and was written up as the sprint's leading candidate. At n=293 it is 11:10,
p=1.000. See §5.

---

## 3. What is open, in the order worth attacking

**Summing chunk scores instead of taking the best one — the one positive lead.** *Measured 2026-08-06,
promising and not established.* A document currently ranks where its single best chunk ranked. Summing
normalized chunk scores instead, on the **fulltext** corpus:

| condition | @20 | @50 | gained | lost | p |
|---|---|---|---|---|---|
| RRF, best-chunk *(baseline)* | 85.4% | 93.9% | — | — | — |
| min-max scores + **sum**, w=0.5 | **89.0%** | 95.1% | 15 | 6 | **0.078** |
| min-max scores + sum, w=0.3 | 88.6% | 94.7% | 14 | 6 | 0.115 |
| min-max scores + sum, w=0.7 | 89.0% | 93.5% | 16 | 7 | 0.093 |

+3.6 points at @20, with all three weights leaning the same way at roughly 2.4:1. The mechanism is
plausible and was **predicted before the run**: a paper matching the query in five places is more likely
to be the source than one matching once, and the best-chunk rule cannot see the difference. The same
conditions are a clean null on hydrogen (11 gained, 10 lost) — as they must be, since an abstract is one
to three chunks and the two rules nearly coincide there. A prediction that could have failed and did not.

**It held when the sample doubled**, which is the test `K=10` failed. At n≈136 it was 15 gained against 6
lost; at n=283, **17 against 8, p=0.108, +3.1 points**. The ratio and the effect size are stable where
`K=10` went from 4:1 to 11:10 under the same treatment. Still not significant at 0.05, and the evidential
weight is in the pattern rather than the p-value: three weights agreeing, a stable effect across two
sample sizes, a control corpus behaving as predicted, and the two neighbouring rules failing for reasons
that make sense.

**The control: same papers, same questions, indexed as abstracts — a clean null.** `minmax/sum/0.5`
scores 87.0% against a baseline of 86.6% on the abstract index, 9 gained against 8 lost, p=1.000, with a
length ratio of 1.06× against fulltext's 1.64×. `sum` differs from `max` only when a document yields
several matching chunks, so it *must* be a null on one-to-three-chunk abstracts. It is. The mechanism is
confirmed by control rather than by argument.

**Neighbouring rules, and both fail informatively.** `mean` — relevance *density*, `sum` divided by the
number of matching chunks — was worth testing because normalizing away the length advantage could have
gone either way. It is **catastrophic**: 69.3% against the baseline's 85.2%, 14 gained against 59 lost,
p < 0.001, worse than `count`. The reason is clear once seen: `mean` rewards documents where *few but
good* chunks matched, so one lucky chunk scoring 0.9 outranks five chunks averaging 0.7. It inverts the
evidence-accumulation signal that makes `sum` work.

Taken together the three rules say something specific: what helps is **accumulating evidence**, combining
quality and quantity. Quantity alone (`count`) is bad, quality alone (`mean`, and `max` to a lesser
degree) leaves information on the table, and only `sum` uses both.

> ### Do not normalize the sum by the number of chunks
>
> Stated as its own heading because it is the correction a future reader is most likely to propose, and it
> reads as obviously right: *a document with more chunks has more chances to accumulate score, so surely
> the total should be divided by how many matched.* That is `mean`, it was measured, and it is the worst
> rule tested — worse than ignoring the scores altogether.
>
> **This is not a hypothetical naive reader.** The proposal came from `hybridir`'s author, minutes before
> it was tested, which is the point: it is what someone who understands the system best proposes, because
> the length worry it responds to is genuine. The correction is wrong anyway, and only measuring said so.
>
> **The number of matching chunks is signal, not a nuisance factor to be divided out.** A paper that
> matches the query in five places is more relevant than one that matches in one, and dividing by the
> count deletes exactly that. What looks like a normalization is the removal of the evidence.
>
> The length worry that motivates the proposal is real and was checked separately (below); it is simply
> not answered by dividing. `sum` is *less* length-biased than the shipped rule, so there is nothing
> needing correction.

**Treat as promising rather than established.** This is structurally the situation that produced the
retracted `K=10` result this morning — a p just under 0.1 with several related cells agreeing — and the
resemblance is worth keeping in view. What differs, and none of it is significance:

- the mechanism was named in the script's docstring **before** the run;
- the corpus where it should *not* work (abstracts) is a clean null, by control rather than by argument;
- it **survived a doubling of the sample**, which is precisely what `K=10` did not;
- the length confound was measured and acquits it;
- the two neighbouring rules fail, in opposite directions, for reasons the mechanism predicts.

Five independent lines agreeing is a different kind of evidence from one p-value, and worth more than the
p-value suggests — but it is still one corpus shape.

**What a replicating corpus would have to be**, since the constraint is not obvious: **many documents *and*
long ones**, both at once. `sum` and `max` can only differ where a document yields several matching chunks,
which needs long documents; and recall@20 must not saturate, which needs enough documents that 20 is a
small fraction. Fiction fails the second (19 documents), banichuk and both abstract corpora fail the first
(one to three chunks). That is a happy constraint rather than an awkward one, because "many long documents"
is exactly Librarian's target case — a researcher's folder of PDFs.

#### Run 2026-08-06: the dose-response test refutes the mechanism

Rather than seek a second corpus, split the 1268 fulltext papers by gold-document length and ask whether
`sum`'s advantage *grows* with length. Under "accumulate evidence across positions" it must: near zero for
the shortest papers, largest for the longest. It does the opposite.

| gold length (chars) | n | baseline @20 | `sum` @20 | delta | `top3` delta |
|---|---|---|---|---|---|
| 18,891 – 52,396 | 73 | 80.8% | 89.0% | **+8.2** | **+9.6** |
| 52,584 – 72,137 | 73 | 86.3% | 90.4% | +4.1 | +5.5 |
| 72,877 – 108,010 | 73 | 89.0% | 91.8% | +2.7 | +2.7 |
| 108,699 – 477,090 | 76 | 85.5% | 82.9% | **−2.6** | **−3.9** |

**Monotonically decreasing, and negative on the longest quartile.** The proposed mechanism is refuted. The
aggregate +3.1 points is a *mixture* — a large gain on short papers minus a loss on long ones — which means
it depends on this corpus's length distribution and would not transfer to a collection weighted
differently. That is precisely the failure a second corpus of the same kind would have hidden, and it is
the argument for preferring the within-corpus test.

**Bounding the accumulation does not rescue it.** `top-N` sums only a document's best N chunks, so `max` is
N=1 and `sum` is N=∞; the two failure ends suggested an optimum between them. Measured, `top3` scores
88.8% at @20 (19 gained, 9 lost, p=0.087, MRR 0.641 against the baseline's 0.581) — the best cell in the
grid — but its length profile is the *same shape, slightly amplified*. So the loss on long documents is not
caused by unbounded summing.

**What can honestly be said**, and it is less than this morning:

- The *family* of multi-chunk rules — `sum`, `top2`, `top3`, `top5` — all land at 87.8–88.8% with 16–19
  gained against 8–9 lost, where `max` (= top-1) sits at 86.4%. "Accumulate more than one chunk; the exact
  N barely matters" is the robust form. `top3` being best is a winner picked from 42 cells and should not
  be quoted as such.
- The gain is real on short-to-medium documents and reverses on long ones. **It should not ship as a
  blanket default on this evidence.**
- No verified explanation replaces the refuted one. A plausible story — that a question written from an
  abstract has its answer concentrated in one place in a long paper, so `max` is right there, while
  accumulation promotes other long papers matching diffusely — is *only* a story, and this investigation
  has spent the day demonstrating what those are worth.
- The shape suggests the rule may want to be length-dependent, which is the same "no configuration is best
  independent of the data" conclusion this file keeps reaching. That is a hypothesis, not a plan.

**The length confound was checked rather than argued away, and it acquits `sum`.** The worry is that
summing rewards documents with more chunks, i.e. longer ones, so the gain could be a length prior wearing
a relevance costume. Measuring the mean length of the documents each condition promotes into its top 20,
against a corpus mean of 94,545 characters:

| condition | mean length of top 20 | vs corpus |
|---|---|---|
| **RRF, best-chunk (shipped)** | 154,853 | **1.64×** |
| min-max + sum, w=0.5 | 151,197 | 1.60× |
| min-max + max, w=0.5 | 139,943 | 1.48× |
| z-score + sum, w=0.5 | 130,115 | **1.38×** |
| count (any weight) | 159,860 | 1.69× |

`sum` is *less* length-biased than what ships, so its gain is not bought that way.

**And the shipped configuration is itself strongly length-biased, which nobody was looking for.** RRF with
best-chunk aggregation promotes documents 64% longer than the corpus average. The mechanism is obvious
once stated: "a document ranks where its best chunk ranked" is a **maximum over N samples**, and a maximum
over more samples is higher in expectation. Best-chunk aggregation *is* a length prior — so the intuition
that `sum` would be the length-biased rule is exactly backwards, and the least biased condition measured
(z-score + sum, 1.38×) is a summing one.

Whether 1.64× is harmful is a separate question this does not answer: gold documents are sampled uniformly
here, so retrieval is over-selecting long documents relative to the corpus, but on a real query a long
paper may genuinely be likelier to be relevant. It is recorded because a systematic length preference in
the shipped ranking is worth knowing about either way.

**Fiction cannot test `sum` at document level, and is inconclusive at passage level.** With 19 documents
at `k=20` every condition returns every document, so document recall is 100.0% for all nineteen conditions
and the length check is degenerate too. Scored by passage coverage instead — the metric that corpus does
support — mean coverage does not move (43.1% for the baseline against 42.0% for `sum`), while
near-complete coverage roughly doubles (5.7% → 10.2% of questions above 90%). That is 5 questions against
9 out of 88: a hint, not a result, and pointing the same way as the fulltext finding without adding much
to it.

**Fiction cannot test this, and the reason is structural.** It is the obvious second corpus to try, having
dozens of chunks per document like fulltext but a completely different genre — which would separate "helps
on long documents" from "helps on scientific papers". Run, it returns **100.0% at @20 for all nineteen
conditions, 0 gained and 0 lost everywhere**: with 19 documents and `k=20` every configuration returns
every document, so nothing can discriminate. Only MRR moves, and an MRR-only difference is not an
improvement here (§5).

The question survives the failed measurement. Asking it of fiction needs the metric that corpus *can*
support — passage **coverage**, which is not saturated (55.2% for merged spans) — i.e. whether ordering by
summed chunk scores puts more of the answering passage in front of the model. That means running
`passage_recall.py`'s coverage measurement over a `sum`-ordered result list, which is a wiring job between
two existing scripts rather than a new experiment.

**Using the *main model* as the reranker — untested, and it splits into three configurations.** Juha's
question, 2026-08-06: is having the LLM order the results better or worse than injecting them raw, as
happens now?

*On the relevance channel the prediction is that it loses, and more decisively than the cross-encoder
did.* The measured cause of the cross-encoder's failure is that collapsing independent evidence into one
model's opinion costs more than the opinion is worth. A same-model reranker is the **maximally dependent**
case: its selection errors and its answering errors come from the same weights, so they are perfectly
correlated. A cross-encoder at least contributes a different model's biases; the main model contributes
none.

*But there is a second channel, and it has nothing to do with relevance.* At `k=50` the retrieved block is
~18k tokens, and models attend poorly to the middle of long contexts (Liu et al. 2023,
[arXiv:2307.03172](https://arxiv.org/abs/2307.03172)). If good material is effectively invisible where it
sits, a pass that moves it helps — not by judging relevance better, but by re-siting it where attention
lands. Those two channels pull opposite ways and a single "does it help" measurement cannot separate them.

Hence three configurations, of which only the first two are usually considered:

| | what it does | expectation |
|---|---|---|
| (a) inject all `k` raw | what ships | baseline |
| (b) LLM-rerank, inject top-N | filters | loses information the answer pass would have used, *and* pays a full extra pass |
| (c) LLM-rerank, reorder all `k` | keeps everything, best material to the ends | the one worth testing: can win on attention, cannot lose information |

**(c) is the configuration to test, and nobody proposed it** — the reranking literature assumes filtering
because it was built for systems that cannot afford to send everything. Raven can. And if the benefit is
attentional rather than relevance-based, there is a cheaper variant still: **reorder by the retrieval score
already computed** — best-ranked first and last, worst in the middle — which costs no extra pass at all. If
that recovers most of the gain, the LLM pass is unnecessary, and the whole question turns out to be about
*placement* rather than *judgment*.

**The cost objection, and it may rule the LLM pass out before quality is even measured** (Juha,
2026-08-06): a rerank pass is a **KV-cache double miss**. It is a *cold* prompt — it does not share the
system-prompt-and-history prefix, so it reuses nothing — and it carries the same ~18k-token retrieved
block that the answer pass will then prefill again. The block is uncacheable either way, so reordering
costs nothing extra downstream, but nothing is saved either: **the retrieved block is prefilled twice.**

At the measured ~5000 tokens/s that turns 3.4 s into roughly 7 s per turn, which is almost exactly the
`k=100` figure in §1 — a cost already rejected there as not worth its recall. So the LLM rerank starts at
a latency this project has *already declined to pay*, before any benefit is demonstrated.

**And it is worse than "one extra pass", which retires the obvious mitigation** (Juha, same day —
correcting the paragraph above, which had proposed it). At concurrency 1 and local-machine VRAM the backend
holds **one** cached prefix. So the sequence is:

1. the turn begins with the chat-history prefix cached from the previous turn;
2. the rerank pass is a different prompt, and prefilling it **overwrites the slot**;
3. the answer pass finds no history cache and prefills *the entire conversation* again.

The cost is therefore not *the rerank pass*; it is *the rerank pass plus the loss of a history cache that
would otherwise have been free*. On a chat carrying 60k tokens of history, today's design prefills ~18k
and this becomes 3k + 60k + 18k ≈ 81k — roughly **16 s against 3.4 s**.

**The preview idea does not help**, and that is the instructive part: shrinking the rerank prompt to ~3k
tokens attacks the smallest term. A 3k-token intervening prompt evicts the slot exactly as thoroughly as an
18k one, so the dominant cost — re-prefilling the history — is unchanged. The mitigation addressed the
number that was easiest to see rather than the one that dominates.

It also attacks the injection design head-on. Raven places retrieved context immediately before the user's
latest message *precisely* so the history prefix stays cacheable; a rerank pass discards that every turn.
**On a single-slot local backend this looks disqualifying regardless of how good the reranking is**, unless
the backend can hold several prefixes (more VRAM, or cache offload to system RAM) — which is a deployment
property, not something the retrieval layer can arrange.

**Which leaves the free variant, and makes it the whole experiment:** reorder by the retrieval score
already computed — best-ranked first and last, weakest in the middle — costing **no extra pass and no cache
eviction**. If the benefit is attentional rather than judgmental, this captures it for nothing. Its
existence is also what makes the LLM version falsifiable: run the free variant first, and any LLM pass then
has to beat *it* while carrying a 5× latency penalty, not merely beat the unordered baseline.

Measurable with the known-item harness exactly as the cross-encoder was: rerank the `k=50` list with the
main model and score recall@5 against gold. The free reordering variant needs no LLM at all and can be run
against the existing sweeps; the LLM variants queue behind question generation.

**Score-based fusion instead of rank-based — a null when the arms are balanced, a gain when they are
not.** *Measured 2026-08-06, and the second half was found by a control run late that evening.*

The first reading was a flat null. `CombSUM` over per-query-normalized scores measures **equal** to RRF on
hydrogen (74.4% against 73.0%, 13 gained against 9 lost, p = 0.52) and the picture repeats on the arXiv
corpora. Then the banichuk control — run to check `sum` aggregation, where it is provably a no-op because
every document is a single chunk — showed score fusion **beating** RRF: 43.7% against 41.0% at @20, 12
gained against 4 lost, **p = 0.077**. With every aggregation rule coinciding on that corpus, the difference
is purely `CombSUM` against RRF.

**The pattern across corpora is the mechanism, and it is not curve-fitting**:

| corpus | BM25 arm MRR | vector arm MRR | arms | score fusion vs RRF |
|---|---|---|---|---|
| hydrogen | 0.414 | 0.375 | comparable | null |
| arxiv-ai | comparable | | comparable | null |
| **banichuk** | **0.090** | **0.201** | **very unequal** | **+2.7, p = 0.077** |

**Score fusion self-weights, and RRF cannot.** When one arm is much weaker, RRF still gives it an equal
vote *by position* — rank 1 of a bad list is indistinguishable from rank 1 of a good one, which is the same
information-discarding that makes the fused rank blind to result quality in the first place. Score fusion
lets a weak arm's genuinely low scores contribute proportionally less, automatically, **with no tuning
constant**. That is the "cheap, no tuning" shape this investigation is selecting for, and it means the
earlier null was not a property of score fusion but of the corpora it was measured on: both are collections
where the two arms happen to be evenly matched.

It also connects to the corpus-dependent arm weight (§3, `keyword_weight`): score fusion is doing
approximately what a per-corpus weight would do, but per *query* and without anyone having to choose the
number. Whether it does it well enough to make the knob unnecessary is the obvious next question.

*Still to check before believing it:* one corpus, p = 0.077, and banichuk is the corpus this file has
repeatedly found to be unlike the others. Fiction is the natural fourth test (arms 0.692 against 0.814,
moderately unequal) but its document-level recall saturates, so it would need the passage-coverage metric.

*What this does not close:* score fusion produces a fused **value** where RRF produces a reciprocal-rank
artifact, so the argument from *calibrated confidence* survives its failure to improve *ranking*. Those
are separate consumers and only the first was measured. If a confidence signal is wanted later, this is
the cheaper foundation even at equal ranking quality.

**Counting matching chunks — dead.** Ranking a document by how many chunks matched, ignoring how well,
costs 37 points on hydrogen (8 gained against 116 lost, p < 0.001). Neutral on fulltext, which is a
corpus-shape interaction rather than a rescue.

**A stronger embedder.** Every number here comes from `multi-qa-mpnet-base-cos-v1`. The Nomic-embed v1.5
migration is already planned. Reranking was the "better model" attack on *ordering* and lost; a better
embedder attacks *candidate generation*, which is upstream of everything and is where the recall curve
locates the losses. Costs a re-index per corpus.

**HyDE** (Gao et al. 2022, [arXiv:2212.10496](https://arxiv.org/abs/2212.10496)). Distinct from the
refuted multi-query work: that split a query, this replaces it with a hypothetical *answer* before
embedding. It attacks the register mismatch the hand-written probes documented — an analytical question
against dramatized prose — which this investigation independently concluded needs "something that has read
the corpus".

> **Score the hypothetical answer with a *symmetric* embedding role, not the `qa` role** (Juha,
> 2026-08-06). The `qa` role is trained for the asymmetry it names — a short question mapped near a long
> answer — and HyDE's entire move is to make the query passage-like. Embedding a synthetic passage as
> though it were a question fights the technique with the model. Raven's server already exposes
> embedding roles, so this is a parameter rather than a second model. Worth measuring both ways, since
> the prediction is cheap to falsify.

**Document summaries as extra indexed chunks.** Document-level questions ("which of these is set offline
in America") are a structural failure of chunk RAG: no chunk states the property. One LLM-written summary
per document, indexed alongside its chunks, addresses it directly. **This is wanted anyway** for VISION.md
stage 3 (the LLM as first-pass recommender and reviewer), so the retrieval benefit rides on work that has
its own justification — which changes its cost, not just its value.

**Chunk size and overlap.** Never varied once, in any experiment. The corpus comparison shows chunk size
interacts strongly with document shape, so this may be the largest single-parameter effect available. Costs
a re-index per setting.

> **Ship them as config options with a loud warning, and pick a defensible default** (Juha, 2026-08-06).
> These cannot be changed after a database is built — the stored chunks *are* the setting — so the comment
> has to say so where someone would edit it. Exposing them is not a hard sell in this class of tool;
> Hindsight and comparable IR systems ship the same knobs. The default does not need to be optimal, only
> good enough across the main use cases, which is what the corpus matrix here is for.

**Reference-list stripping for scientific PDFs.** A bibliography is several hundred other papers' titles
with nothing marking it as different from the body, so a query matching paper X also matches every paper
citing X. Predicted to depress fulltext precision and possibly recall; untested here, but reported to have
helped in a prior private implementation elsewhere.

**Per-corpus arm weight, and whether it can be calibrated at index time.** The value is corpus-dependent on
present evidence, but that evidence is n=99 on two corpora — the same footing `K=10` fell off. Regrowing
those sets is queued. If it survives, generating a few dozen questions from the corpus at ingest and
scoring both arms would pick the weight with labels free by construction, and the same run would yield the
on-corpus similarity distribution the threshold work needed.

---

## 3b. Next actions, as of 2026-08-06 end of day

In order. The first group needs no GPU and no decisions; the second is the new corpus; the third is what
the sprint's live leads still need.

**Ready to run, with one exception noted under item 1:**

1. **Free result reordering** — no LLM, no extra pass. Settles whether the LLM-rerank question is about
   *placement* rather than *judgment*, and any LLM version afterwards has to beat this rather than the
   unordered baseline (§3).

   **"Free" describes the technique, not the measurement, and the measurement does not exist yet**
   (checked 2026-08-06). Reordering changes only the order the retrieved results are presented in — the
   set is identical — so every metric in this investigation is invariant to it *by construction*: gold
   rank, recall@k, passage coverage and the signal AUROCs all read the retrieval, and the retrieval does
   not change. Scoring these arms means scoring what the model *answered*, which is the first
   answer-quality question in the sprint. Nothing here does that: all 19 scripts are retrieval-side, and
   the question sets store `gold` and `gold_titles` but no gold answer text, so there is no cheap
   extractive check either.

   So the honest cost is one generation plus one judgment per question per arm, with the gold passage as
   the reference — LLM-as-judge, on the same local backend, which also makes it the first eval here whose
   result depends on the judge. That is a half-day of harness before any arm runs, and it is a decision
   about what we are willing to trust rather than a build detail. Worth settling before the harness is
   written, not during.

   **Two effects compete here and the experiment has to separate them** (Juha, 2026-08-06). *Lost in the
   middle* says put the best material at the ends. But a ranked result list is best-first **by
   convention**, and models are trained on that convention — so best-first-and-last is *off-distribution*,
   and a model reading what it takes to be a ranked list may discount the tail whatever is in it. Moving
   good material there could mislead rather than rescue.

   So at least three arms, and the third is the one that tries to have both:

   | arm | ordering | tests |
   |---|---|---|
   | (a) best-first | as now | baseline, on-distribution |
   | (b) best-first-and-last | strongest at both ends | pure lost-in-the-middle mitigation, off-distribution |
   | (c) best-first, explicitly numbered | as now, plus visible rank labels | keeps the convention while telling the model the ordering, so it need not infer relevance from position |

   If (b) wins, attention placement dominates. If (c) wins, the problem was that the model could not *tell*
   the list was ranked. If (a) holds, position was never the issue.

2. **`score_fusion.py fiction`** — the fourth corpus for the self-weighting hypothesis (score fusion helps
   when the arms are unequal). Needs the passage-coverage metric, since document recall saturates at 19
   documents.

   *The 13 held-out stories could be indexed to make it 32*, but that is probably the wrong trade: 32
   documents at `k=20` still saturates (62% of the corpus), so it does not fix the metric problem, while
   indexing them destroys their current role as the *adjacent* negative group — the hardest negatives in
   the whole set, and irreplaceable. They are also unread, so they bring no human ground truth. Keep them
   held out; use the coverage metric.

**Bring up the ECCOMAS 2024 corpus** (`00_stuff/datasets/ECCOMAS2024`, 2520 conference abstracts):

3. **Do not re-run the PDF→BibTeX extraction.** It cost **1–2 weeks of GPU time**, and the result is
   hand-corrected and stamped golden:
   `success_final_manually_fixed_and_added_missing_abstracts.bib`. Start from that file. The PDFs are kept
   for provenance, not for reprocessing.
4. `raven-burstbib` the golden `.bib` → `documents_eccomas`, `raven-indexer` → `rag_index_eccomas`, add a
   `CORPORA` entry to `make_questions.py`, generate a question set (~300, matching the others), and run the
   standard sweep.
5. **The reason it is worth the setup**: at 2520 documents it sits inside the bracket where adaptive `k`
   stops working (works at 1268, fails at 11974), so it is the direct probe of where that transition is.
   Secondarily it is the only *dirty-provenance* corpus in the set.

**What the live leads still need:**

6. `sum` aggregation is length-dependent — gains on short documents, loses on the longest — so it does not
   ship as a blanket default. The open question is whether a **length-aware** rule is worth having, and
   that wants a mechanism before another grid sweep.
7. Adaptive `k` is confirmed on small corpora and dead on large ones. Building it means shipping the domain
   of validity with it, and the large-corpus answer is stratification, which needs clustering.

**Do not score any corpus while a question set for any corpus is being generated** — the scorers read every
question file at startup, and two runs minutes apart have already seen different `n`.

---

## 4. Two things the measurements agree on, from different directions

**Breadth and depth are a trade, and the retriever cannot know which a query needs.** Bare chunks reach
more distinct documents per character (+11.0 points of document recall at a tight budget on fulltext);
merged spans deliver more of the passage that answers the question (55.2% against 43.3% coverage on
prose). Each favours the arm one would expect. Nothing in the query says which matters.

**The narrow/broad distinction is the one worth detecting**, and it arrived independently three times: as
the arm-selection oracle, as the breadth/depth trade, and as the adaptive-`k` hypothesis. All three want
the same missing capability rather than three separate features.

### Measured 2026-08-06 (late): adaptive `k` pays — but only while the corpus is small enough

The synthesis question class was built for exactly this, and `synthesis_recall.py` asks whether broad
questions keep gaining from larger `k` after narrow ones have stopped. Mean recall over each question's
gold *set* (4 documents for synthesis, 1 for focused):

| corpus | docs | group | k=5 | k=20 | k=50 | k=100 | k=200 | gain 5→200 |
|---|---|---|---|---|---|---|---|---|
| **arxiv-ai** | 1,268 | focused | 71.2% | 86.3% | 94.0% | 97.4% | **100.0%** | +28.8 |
| | | synthesis | 17.3% | 39.1% | 52.6% | 63.5% | **74.4%** | **+57.1** |
| **hydrogen** | 11,974 | focused | 62.4% | 75.5% | 84.3% | 88.2% | 92.6% | +30.1 |
| | | synthesis | 3.3% | 6.6% | 9.9% | 19.7% | 24.3% | **+21.1** |

**The two corpora disagree, and the disagreement is the finding.**

On arxiv-ai the hypothesis holds cleanly: focused questions **saturate at 100%** by k=200 while synthesis
is still climbing steeply — twice the gain, and nowhere near done. That is precisely the curve separation
adaptive `k` exists to exploit, and it is the first confirmation of a stated prediction in this sprint.

On hydrogen it fails, for the reason **Juha predicted this morning** before any of it was built: at 11,974
documents k=200 is 1.7% of the corpus, so a broad question's relevant set outruns any conversational `k`.
Synthesis recall reaches 24.3% and is barely moving. More `k` is a thin sample of the relevant set, not
coverage of it — which is exactly the note recorded in brief 09 at his prompting, now measured rather than
reasoned.

**So adaptive `k` is worth building, with its domain of validity attached**: it pays on small and medium
collections and stops paying on large ones, where stratified sampling is the real answer and `k` is only
the affordable approximation. Shipping it without that caveat would produce a feature that works on the
demo corpus and quietly does nothing on a researcher's 12k-record library.

#### And this promotes clustering from a nice-to-have to a prerequisite

Juha's reading, and it is the consequence that matters most: **the hydrogen corpus wants stratification,
and nothing available today can give it that.** A broad question there has a relevant set that no
conversational `k` samples adequately — 24.3% of the planted documents at k=200, still climbing, with the
next doubling unaffordable on prefill grounds. Raising `k` is not a partial solution to that; it is the
wrong axis.

What a broad question needs is a *few results from each region* of the relevant material rather than the
top-`k` by score, which oversamples whichever region ranks highest. That requires cluster structure over
the corpus — which the Visualizer pipeline already builds and the Librarian side does not have, and which
the **corpus scopes and unified DB** work is what makes reachable.

**The measurement also brackets when it starts to matter.** Adaptive `k` works at 1,268 documents and
fails at 11,974, so the transition sits somewhere between — and since the mechanism is `k` as a *fraction*
of the collection, it should track corpus size rather than any property of the subject matter. That is an
actionable design number rather than a vague "large corpora are harder": a few thousand documents is where
top-`k` retrieval stops being able to answer broad questions at all.

**And a corpus that lands inside the bracket already exists** (Juha, 2026-08-06): **ECCOMAS 2024**, 2520
conference abstracts at `00_stuff/datasets/ECCOMAS2024`, with a hand-corrected `.bib`. At 2520 documents it
sits between the corpus where adaptive `k` works and the one where it does not, which makes it the direct
probe of where the transition is — and it is a *drop-in* fifth corpus, since the `.bib` carries abstracts
in the same shape the other sets use (`raven-burstbib` → `raven-indexer` → a `CORPORA` entry).

Measured shape: 1637 characters per abstract on average (median 1606, p75 1923, max 8371), about 2.2 chunks
each, ~5500 chunks for the collection.

**Two other things make it the most representative corpus here, both about provenance rather than size.**
It is *dirty in the way real collections are dirty*: free-form PDFs where most but not all follow the
mandatory template, some missing titles, some missing author names, one a verbatim copy of the template
itself, and several authored in Word with text layers broken badly enough to need `ocrmypdf`. Building the
`.bib` took `raven-pdf2bib` with Qwen 3 for ~2200 of them and hand-fixing for the remaining ~300. Every
other corpus here arrived clean — WoS exports, arXiv metadata, a hand-typed bibliography — so this is the
only one that exercises the ingestion path the way a conference organizer's dump would.

Its abstracts are also **verbatim author text**, not generated: `pdf2bib`'s prompt says *"Do NOT summarize
or reword the input in any way. Just copy the main text… AS-IS"*, and the pipeline additionally strips the
conference template boilerplate, author affiliations and addresses, and any reference list. So the field is
the author's own abstract body with the packaging removed — which is what makes it usable as ground truth
rather than as a model's paraphrase of one.

**ECCOMAS 2026 is expected too, and the pair is worth more than the sum.** Two editions of the same
conference, same template, same community, two years apart, gives a **near-duplicate axis** no other corpus
here has: the same research groups presenting successive work on the same topics. That is the case where
retrieval has to distinguish *this* paper from its own sequel, which is both a realistic failure mode for a
researcher's collection and untested by every set currently in use — all of which contain at most incidental
near-duplicates. Together the two editions would also be ~5000 documents, further into the
adaptive-`k` bracket.

*Access is a governance question, not a technical one.* The abstracts are public by design, but public does
not imply scrapable, and the hosting site's stance is unknown. The 2024 set came **from the organizers**
directly, and Juha has a coworker on the ECCOMAS managing board — so the route to ask is short, and the
answer may well be a bulk export, which is politer than scraping and yields cleaner data than rendering
pages would. Ask before fetching; do not infer permission from a `robots.txt`.

Which reorders the roadmap. Clustering was justified as an interface and organization feature; it is now
also the only route to broad-question retrieval on the collection sizes Raven is built for.

#### It also confirms `VISION.md` stage 3, from the opposite direction

Stage 3 says of the broad-question case: *"That is not a retrieval problem. The selection was already made,
by a person, on the map. Ranking two hundred hand-picked documents and keeping the top five would discard
ninety percent of what the user deliberately chose."* The document is from this week, but the stage 3 idea
dates to 2024 and the reasoning is from the workflow rather than from any measurement. Tonight's result
reaches the same place from the retrieval side — a broad question over 12k documents retrieves 24.3% of its
relevant set at k=200, and the next doubling is unaffordable on prefill. Two independent routes to "stop
trying to retrieve this".

So there are two answers to the broad question, and they are complements rather than rivals: **stratified
retrieval** (needs clustering) for when the user has not selected anything, and **map-and-reduce over a
human selection** (stage 3) for when they have. The second is stronger where it applies, because a
human-verified selection beats an algorithmically sampled one — and today's results are a long argument for
distrusting algorithmic selection at this corpus size.

#### Summarization now has four independent justifications

Worth stating together, because it changes the cost case rather than merely adding a use (Juha, 2026-08-06):

1. **Stage 3 map-and-reduce** — its original purpose. Shipped code in the Visualizer importer, switched off
   because it was built to run over an entire dataset at import rather than over a selection.
2. **Document-level retrieval failures** — the class this investigation found chunk RAG structurally cannot
   serve, where no chunk states the property being asked about ("which story is set offline in America").
   An indexed summary states it.
3. **An independent view for evidence fusion** — axis 6 of the accumulation list: a summary is a different
   representation of the document, not merely extra coverage of it.
4. **Token budget** — summaries instead of full text at high `k`, which is directly valuable now that
   prefill is the measured constraint on `k`.

**But "summarize lazily" does not save the retrieval cases, and the numbers are now measured** (Juha,
2026-08-06). Tonight's question generation ran at **27 s per item** with an *abstract-sized* prompt and a
short output. A fulltext paper is ~95k characters — roughly 24k tokens, so ~5 s of prefill before any
generation — putting a real summary at 15–20 s per document. Therefore:

| case | documents | lazy cost | verdict |
|---|---|---|---|
| retrieval, per query at `k=50` | 50 | **12–17 min** | broken, not slow |
| stage 3, per selection | ~200 | ~1 hour | **fine** — it is already designed as a batch job with progress, cancellation and resumption |

So the four uses split cleanly. **Stage 3 can be lazy; retrieval cannot.** Anything a query touches has to
be precomputed, which puts summarization back where it was disqualified — an indexing step over the whole
corpus — and makes *speed* the deliverable rather than an optimization to do afterwards.

Where the speed has to come from, since the constraint is generation throughput rather than prefill:

- **A small dedicated model, not the chat LLM.** The summarizer does not need a 35B MoE. This is the
  `raven-server` three-layer pattern's natural shape — a served module with its own model, sized for the
  job.
- **Batching, i.e. concurrency > 1.** Single-stream generation is the whole problem; throughput scales
  with batch size on the same card. The chat path runs at concurrency 1 by necessity, an indexing pass
  does not.
- **Skip what is already a summary.** An abstract *is* one — so the abstract corpora need no summarization
  at all, and the scope collapses to fulltext. For a scientific PDF there is a stronger shortcut still:
  the abstract is *inside the document*, so "summarize this paper" is often "extract its abstract", which
  is nearly free and better than a generated summary.
- **Cap the output.** Summary length is a setting; generation time is linear in it.

That last pair matters most for the corpus sizes here: it is the difference between summarizing 12k
documents and summarizing the ~1.3k that actually need it.

##### Fiction is the hard case, and the small-model path has history worth knowing

Extraction beats generation wherever an author-written summary exists, which covers scientific papers.
**Fiction has neither**: no abstract, documents that may not fit a context window (median 45k characters,
tail to 366k), and *two* use cases rather than one — with and without spoilers, which has no standard
solution at all.

Raven tried the small-dedicated-summarizer path before, and the git history is the feasibility study
(`git log -S summarize`). Models tried: `ArtifactAI/led_base_16384_arxiv_summarization`,
`ArtifactAI/led_large_16384_arxiv_summarization`, `Falconsai/text_summarization`, settling on
`Qiliang/bart-large-cnn-samsum-ChatGPT_v3` as the best of them. Sub-second on a 200-word abstract, but the
quality at the compression ratio wanted was judged inadequate, and BART's 1024-token window forced a
spaCy sentence-boundary splitter as a preprocessing step. The long-context LED options were on the list
and lost on *quality*, not length.

**The task then was scientific abstracts condensed to 1–2 sentences, for stage 3 — and that pairing was
correctly matched, which makes the negative stronger than it first appears.** An arXiv-tuned summarization
model, evaluated on arXiv abstracts, at roughly 10:1 compression, is the model doing exactly what it was
built for. It still lost. Fiction was never tested, by any of them. (An earlier version of this section
described the models as "mismatched" and inferred the negative was therefore weak. That was wrong in the
direction that flatters the idea.)

**Then the ecosystem removed the abstraction.** The module was deleted in `5bc503d` (Feb 2026) — not by
preference: *"'summarization' task is gone from the transformers side. Major social signal that LLMs have
eaten this use case."* The `translate` module met the same fate the same day and now loads its model
directly, which is the working precedent in this repo for how a revival would have to be built. Some of
the models tried are also simply gone from HF now.

**And for stage 3 specifically, a seq2seq summarizer is the wrong *shape*, not merely lower quality.**
Stage 3 asks each document a *question* — "which of these describes a computationally lightweight yet
reasonably accurate model I could use in a value chain?" — and wants a summary that answers **that**. A
dedicated summarization model cannot take a query at all; it performs generic compression and nothing
else. So the small-model path could only ever have served the *generic tldr* variant, never the
query-conditioned one that stage 3 is built around. That is an architectural exclusion rather than a
quality judgement, and it holds however good the model gets.

**Fiction, by contrast, was never tested at all**, and it is the case where a specialized model still has
a plausible claim: generic compression is genuinely what is wanted there, so the shape objection above
does not apply. A **BookSum**-tuned LED on narrative is the untested combination:
[`pszemraj/led-large-book-summary`](https://huggingface.co/pszemraj/led-large-book-summary) fine-tunes
`allenai/led-large-16384` on BookSum, which is long-form narrative — novels, plays and stories — with
human-written abstractive summaries at paragraph, chapter and book granularity. 16,384 tokens is ~64k
characters, so the median fiction story fits in one pass and the long tail maps onto *chapter*-level
summarization rather than an arbitrary splitter. Siblings:
[`led-base-book-summary`](https://huggingface.co/pszemraj/led-base-book-summary) and
[`long-t5-tglobal-xl-16384-book-summary`](https://huggingface.co/pszemraj/long-t5-tglobal-xl-16384-book-summary).

**Where that leaves the four uses**, which is a cleaner split than "revive it or don't":

| use | needs | verdict |
|---|---|---|
| stage 3 map-and-reduce | a *query-conditioned* summary | **LLM only** — a summarizer model cannot take a question |
| generic tldr of abstracts | 10:1 generic compression | **LLM** — the matched specialized model was measured and lost |
| scientific *fulltext* for retrieval | a document-level summary | **extraction**, not generation — the abstract is inside the paper |
| scientific documents that *are* abstracts | a shorter form | **generation** — there is nothing shorter to extract |
| **fiction for retrieval** | generic compression of long narrative | **open** — never tested, and the one case a specialized model still fits |

**The extraction row is narrower than it first looked** (Juha, 2026-08-06). It holds for documents that
*contain* an author-written summary of themselves — journal papers and preprints, where the abstract sits
inside the fulltext. It does **not** hold for scientific documents that *are* abstracts: a conference
abstract is already the shortest form its author wrote, so wanting a one-sentence version means generating
one. The ECCOMAS 2024 set below is 2520 of exactly that, and it is a normal scientific corpus rather than
an edge case.

So the split is not "scientific versus everything else" but **"does the document contain a summary of
itself?"** — and only the journal-paper shape does. Extraction is the special case twice over.

**And "only fiction" is the wrong way to read that last row.** Fiction stands in for *long-form text with
no author-written summary*, which is the **general** case — novels, memoirs, reports, transcripts, manuals,
legal documents, meeting notes. The scientific paper is the **special** case: it is unusual in shipping
with an author-written abstract that can simply be extracted. So the open question is not a corner of the
problem; it is everything except the one document type that happens to summarize itself.

That is also why the fiction corpus earns its place as an instrument rather than as a joke. Raven indexes
whatever the user drops in the folder, and domain-agnosticism is structural here rather than a
nice-to-have — a summarization story that works only where an abstract already exists is not a
summarization story.

Worth an evaluation rather than a decision, then, with the burden correctly placed: the standing prior is
that only a 30B-class LLM is competent at this compression ratio, and it was formed on a *correctly
matched* test that the specialized model lost — so the prior is well-founded rather than an artifact of a
bad pairing. Polished previous-generation tooling can beat a developing current generation, and its
hardware profile favours local deployment. But the specialized model has to earn its way back in, on a
corpus type where nothing has been measured yet.

**One caveat on the metric, which cuts against the synthesis numbers specifically.** Set recall asks
whether *these four* documents came back, and a broad question over a large corpus has many documents that
answer it equally well — so the absolute synthesis levels are a severe floor, more so than the known-item
understatement elsewhere in this file, and more so on the larger corpus. The *shape* comparison is the
robust part: if larger `k` were bringing in more of the relevant set, the planted documents would arrive
with it.

**A process note, since it nearly went the other way.** Primed by a day of null results, and with hydrogen
run first and delivering one, the tidy conclusion was already forming. The second corpus reversed it. The
order the corpora happened to run in was very nearly the finding.

**A structural note.** Retrieval resembles sorting: there is no configuration that is best independent of
the data, which is why the useful output is a distribution across corpora rather than a champion setting.
That argues for what this work converged on anyway — honest defaults, exposed knobs, and worst-case
reporting rather than average-case claims.

### What to look for: cheap independent evidence, combined without tuning

The design target is a simple technique with a disproportionate gain — `min_p` in LLM sampling is the
reference example, and BM25 + vector + RRF is the one already in Raven (Juha, 2026-08-06). Stated as a
selection criterion it is sharper than "try promising things", and it explains this sprint's results
better than the individual measurements do.

What those two share: **two or more cheap signals that fail independently, combined with no tuning.** BM25
and embeddings fail on different queries — one on paraphrase, the other on rare exact terms — and RRF
exploits that without a fitted parameter. `min_p` reads the shape of the distribution it is given instead
of imposing a constant on it.

Read that way, **`sum` aggregation is the same idea on a different axis.** RRF says *a document found by
two independent engines is better*; `sum` says *a document matching in several independent places is
better*. Accumulating independent evidence, within a document rather than across engines. That is a reason
to weight it slightly above its p-value, and it predicts `mean`'s failure exactly: dividing by the number
of matches discards the independence that the accumulation was exploiting.

#### Where else is there independent evidence to accumulate?

If the principle is the attack, the generative question is what *else* fails independently. Enumerated
2026-08-06; the first two are the ones already in hand, and the rest are untested.

1. **Across engines** — BM25 and embeddings fail on different queries. *Shipped, as RRF.*
2. **Within a document, across positions** — several chunks of one document matching. *The `sum` lead.*
3. ~~**Within a document, across engines at different positions.**~~ **Tested 2026-08-06 — a null.** The
   idea: RRF fuses at chunk level, so a document found by both arms *in the same chunk* scores exactly
   like one found by BM25 in the introduction and by the vector arm three pages later, and the second
   looks like better evidence — two independent engines *and* two independent locations.

   Implemented without a tuning constant as `armsum`: sum a document's chunk scores *within* each arm,
   rank documents per arm, then RRF the two document rankings. Measured on the fulltext corpus at n=295 it
   scores 85.8% against the baseline's 85.4%, with **3 discordant questions in 295** (2 gained, 1 lost).
   The two orderings are very nearly identical, so the hypothesized structure is either absent or already
   captured by chunk-level fusion.

   Cheap to have eliminated — ten minutes, no new retrieval — which is the argument for keeping this list
   rather than reasoning about each idea in isolation.
4. **Across query views — worth testing, and *not* free as first claimed.** Embed the same query under both
   the `qa` and the symmetric role and fuse the two result sets: two views of one query, failing
   differently. Distinct from the refuted multi-query work, which split a message into *fragments* that
   then outvoted the whole — here every view is the whole query, so that failure mode does not apply.

   **Correction to the cost, 2026-08-06.** The roles are different *models* — `default` is
   `snowflake-arctic-embed-l`, `qa` is `multi-qa-mpnet-base-cos-v1` — so the corpus must be embedded
   twice and carry a second vector collection. Query-time cost stays small (one extra query embedding,
   one extra vector search); index-time cost is a full second embedding pass. That puts it in the same
   gated class as (5), not ahead of it. The original "no cost" claim skipped the index side.

   The prior against it is worth stating so the measurement can overrule it: a *question* should embed far
   from its *answer* under a symmetric model, which is exactly what the asymmetric `qa` role exists to fix.
   Juha's own reading, flagged by him as a "surely" — and priors of that shape have lost twice already in
   this investigation.

5. **Across chunk scales — good idea, blocked on storage.** Index at, say, 500 and 2000 characters and fuse
   both. Small chunks find precise facts, large chunks find diffuse topics. The appeal is that it
   **dissolves the chunk-size sweep rather than winning it**: instead of choosing a size, index at two and
   let the disagreement be informative. No query-time model, no tuning constant.

   *Gated on the datastore work* (`TODO_DEFERRED.md`, "The docs DB stores each document's full text *and*
   its chunks"). The fulltext corpus already produces a 587 MB `fulldocs/data.json` inside a 2.6 GB index;
   a second scale roughly doubles it, which is not a reasonable thing to ask of a user before that store
   is fixed. Revisit once it is.

6. **Across document representations — blocked on the summary work.** The same 1268 papers exist as
   abstracts *and* fulltext, and an abstract says what a paper is *about* where the fulltext says what it
   *contains*. But that pair is an artifact of this investigation, not something a user has: in the general
   case the second representation has to be *made*, which is the LLM summary pass of (§3). What this adds
   is a better argument for building it — a summary is an **independent view** of the document, not merely
   extra coverage of it.

7. ~~**Across query terms.**~~ **Tested 2026-08-06 — a null, and it corrects the reasoning behind it.**
   The idea: how many *distinct* query terms a document matches, as against matching one term strongly.
   Fused as a third ranked list so there was no constant to tune. On the fulltext corpus at n=295:

   | condition | @20 | @50 | MRR | gained | lost | p |
   |---|---|---|---|---|---|---|
   | bm25 + vector *(baseline)* | 85.4% | 93.9% | 0.581 | — | — | — |
   | + coverage as a third arm | **85.4%** | 92.5% | 0.569 | 6 | 6 | **1.000** |
   | coverage alone | 72.9% | 83.1% | 0.405 | 11 | 48 | < 0.001 |

   **The signal is real but not independent.** It has genuine spread — the best candidate averages 46.3%
   coverage against the worst at 3.8% — and ranks meaningfully alone at 72.9%, so this is not a broken
   measurement. It simply carries nothing BM25 does not already have.

   The justification was wrong, which is the more useful part. It claimed that BM25's per-term saturation
   fails to reward breadth across terms. But BM25 *is a sum over matched query terms*: breadth is precisely
   what it accumulates, and saturation bounds each term's *depth*, which makes it **more** coordination-like
   rather than less. The proposal was to add to BM25 a property BM25 is largely made of.

**And one to rule out, for the reason that makes the principle work.** Corroboration from a document's
*semantic neighbours* is tempting and should not be done: neighbours are correlated by construction — that
is what makes them neighbours — so accumulating over them double-counts one piece of evidence rather than
adding independent pieces. The independence is load-bearing, not decorative.

It also re-ranks the open work in §3, which was ordered by expected gain rather than by this criterion:

- **Fits well** — `sum` aggregation (a one-line change, no new model, no user-visible knob); reference-list
  stripping (a heuristic at extraction, no new model); chunk size and overlap (one constant, if a win
  exists); a stronger embedder (a drop-in component swap, though it costs a re-index).
- **Fits badly, whatever its merit** — HyDE (an extra LLM call in the query path, so latency on every
  turn) and document summaries (an LLM pass per document at ingest). Both may work; neither is cheap, and
  the summaries only earn their place because VISION.md stage 3 wants them anyway.

And a caution the criterion carries with it: the sprint's null results cluster almost entirely in one
family — *query-side* levers, which is what brief 09 scoped. The one thing that moved is a **structural**
choice nobody had questioned, sitting in how chunk evidence becomes document evidence. That is where to
look next: not better tuning of the parts that were designed deliberately, but the defaults that were
never decisions in the first place.

---

## 5. Method: what this harness can and cannot decide

**Known-item evaluation.** Questions are generated *from* known documents, so labels are free and no
judging is needed. Precision is understated by construction — other documents may also answer a question
and count as misses — but the comparison across configurations is unbiased, which is the property needed.

**Generated questions are easier than real ones.** They are written with the passage in view, so they name
the right entities. Any *threshold* read off this set is an upper bound on the safe one.

**n=99 could not decide anything at the sizes that matter, and cost a false positive.** McNemar discards
concordant pairs, so 99 questions left single-digit discordant counts. `K=10` survived at that size and
died at n=293. Hydrogen is now at 293; arxiv-ai and banichuk are queued.

**An MRR-only improvement is not an improvement here.** Two separate candidates — document-level fusion and
a smaller `K` — moved MRR by ~0.03 without changing membership of the top 20. MRR is head-weighted, and
`k=50` ships: what reaches the model is membership, not order within it. Apply this test before running a
paired comparison, not after.

**Read a metric's semantics before writing a mechanism onto it.** Two claims were published and retracted
in one afternoon for this reason. Passage-level recall read 39.8% under a metric that required covering a
4000-character passage's *start point* with 1000-character chunks — which also silently favoured longer
spans, manufacturing an arm difference. By interval overlap it is 89.8%; by the measurement that matches
how results are used (how much of the passage reaches the model at all) it is 55.2%. Same data, three
numbers, and only the third answers the question anyone cares about.

**Do not score while a question set is being generated.** The scorers read the question files at startup
and the generator appends to them, so two runs minutes apart saw n=128 and n=130.

**A silent zero is the characteristic failure.** Scoring the fulltext corpus against gold labels naming
`.bib` files while the index held `.pdf` files reported a clean 0.0% at every budget, with retrieval
plainly returning 100 results per query. It reads as a corpus that cannot answer its own questions. Compare
on `sharpness.document_key`, and distrust any zero that arrives without an error.

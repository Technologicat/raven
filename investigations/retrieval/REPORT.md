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

**Two ways out, and both are worth measuring before the full version:**

- **Rerank on candidate *previews*, not full chunks.** The pass has to judge relevance, not answer the
  question, so it does not need the whole text: a document id plus the first hundred characters of each
  candidate is ~3k tokens rather than ~18k. That drops the extra prefill from ~3.4 s to well under a
  second and makes the idea viable again. It also matches what a human skimming a result list does.
- **Reorder by the retrieval score already computed** — best-ranked first and last, weakest in the middle
  — which costs **no extra pass at all**. If the benefit is attentional rather than judgmental, this
  captures it for free, and its existence is what makes the LLM version falsifiable: run the free variant
  first, and the LLM pass has to beat *it*, not the unordered baseline.

Measurable with the known-item harness exactly as the cross-encoder was: rerank the `k=50` list with the
main model and score recall@5 against gold. The free reordering variant needs no LLM at all and can be run
against the existing sweeps; the LLM variants queue behind question generation.

**Score-based fusion instead of rank-based — no gain in ranking quality.** *Measured 2026-08-06.* This
was ranked as the most promising untried idea, on the reasoning that RRF fuses positions and discards
scores, which is exactly why "the hybrid rank does not track how good a result is". `CombSUM` over
per-query-normalized scores measures **equal** to RRF: the best cell reaches 74.4% against 73.0% on
hydrogen, paired 13 gained against 9 lost, p = 0.52, and the picture repeats on fulltext. Both min-max and
z-score normalization, at three weights each.

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

## 4. Two things the measurements agree on, from different directions

**Breadth and depth are a trade, and the retriever cannot know which a query needs.** Bare chunks reach
more distinct documents per character (+11.0 points of document recall at a tight budget on fulltext);
merged spans deliver more of the passage that answers the question (55.2% against 43.3% coverage on
prose). Each favours the arm one would expect. Nothing in the query says which matters.

**The narrow/broad distinction is the one worth detecting**, and it arrived independently three times: as
the arm-selection oracle, as the breadth/depth trade, and as the adaptive-`k` hypothesis. All three want
the same missing capability rather than three separate features. Whether it *pays* is measurable and not
yet measured — the synthesis question class was built for exactly this, and `synthesis_recall.py` asks
whether broad questions keep gaining from larger `k` after narrow ones have stopped. If they do not,
detection buys nothing however well it works.

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

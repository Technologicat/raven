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

**Score-based fusion instead of rank-based.** *Untried, and the most direct attack on the founding
complaint.* RRF fuses positions and discards scores, which is precisely why "the hybrid rank does not
track how good a result is". A convex combination of normalized scores (z-scored BM25 with cosine) needs
no re-index and no new model, and would produce the calibrated confidence number §2's threshold work
wanted and never obtained. The whole sprint tuned parameters *inside* RRF without questioning it.

**Document-level score aggregation.** A document currently ranks where its single best chunk ranked.
Sum-of-chunk-scores and count-of-matching-chunks are untested, and they differ most on long documents —
where a paper matching in five places is plausibly more relevant than one matching once, and the present
rule cannot tell them apart. Cheap; the fulltext corpus exists to test it.

**A stronger embedder.** Every number here comes from `multi-qa-mpnet-base-cos-v1`. The Nomic-embed v1.5
migration is already planned. Reranking was the "better model" attack on *ordering* and lost; a better
embedder attacks *candidate generation*, which is upstream of everything and is where the recall curve
locates the losses. Costs a re-index per corpus.

**HyDE** (Gao et al. 2022, [arXiv:2212.10496](https://arxiv.org/abs/2212.10496)). Distinct from the
refuted multi-query work: that split a query, this replaces it with a hypothetical *answer* before
embedding. It attacks the register mismatch the hand-written probes documented — an analytical question
against dramatized prose — which this investigation independently concluded needs "something that has read
the corpus".

**Document summaries as extra indexed chunks.** Document-level questions ("which of these is set offline
in America") are a structural failure of chunk RAG: no chunk states the property. One LLM-written summary
per document, indexed alongside its chunks, addresses it directly.

**Chunk size and overlap.** Never varied once, in any experiment. The corpus comparison shows chunk size
interacts strongly with document shape, so this may be the largest single-parameter effect available. Costs
a re-index per setting.

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

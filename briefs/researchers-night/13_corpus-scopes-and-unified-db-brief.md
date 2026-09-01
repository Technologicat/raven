# Brief 13 (draft): corpus scopes and the unified document DB

**Status: rough draft, 2026-08-03.** Not a design. This exists to give the material a home with a number on
it, so that the design session has somewhere to start and the decisions already made do not have to be
re-derived. The design itself is expected to come back from a claude.ai session as a filled-in version of
this brief.

**Priority raised 2026-08-07 (Juha): "the symlink dance is becoming unbearable."** Switching corpora means
repointing `documents` and `rag_index` by hand, and the retrieval work has been doing it several times a day
across five collections. Brief 12 then added a third directory to the dance (`document_sidecars`), and every
new capability that is per-corpus adds a fourth. So the cost of not having scopes now grows with the feature
set rather than staying flat, which is what moved this up the queue.

Three things now wait on it explicitly, and each is queued rather than designed around:

- **Cross-corpus sidecar GC** (brief 12). Content-addressed derived text would dedup across corpora, but
  then "is this file orphaned?" is a union query over every index that exists. A scope-aware index supplies
  that answer instead of reconstructing it.
- **"Autosearch off, tools still on"** — the middle setting for the Documents toggle. Incoherent until a
  scope can inject a topic TOC, because a model cannot sensibly decide to search a corpus it knows nothing
  about. See §4, which already says the TOC is blocked on the same work.
- **Large-corpus retrieval.** Adaptive `k` was measured to pay below ~1.3k documents and to be dead at
  ~12k, where a broad question's relevant set outruns any conversational `k`. The answer there is stratified
  sampling, which needs the clustering this brief covers.

**Provenance markers** are carried over from the 2026-08-01 design session and are load-bearing — the point
of this document is that a later reader can tell what is settled from what is merely proposed:

- **[D]** — decided.
- **[N]** — from Juha's notes, carried as-is.
- **[P]** — Claude's proposal, **not agreed**. Argue with these before building on them.
- **[X]** — proposed and **retracted**, kept with the reason so it is not re-proposed.

**Timing.** 04, 05, 06 and 09 to implement, then v0.2.8, then demo polish through Researchers' Night
(26 September 2026). The design session lands after that unless something slips forward.

**What this gates.** [D] Scopes and the unified DB are the prerequisite for the corpus TOC (§4 below) and for
most of what the corpus-interrogation sketch wants. Brief 12 (derived artifact store) deliberately does *not*
depend on this and can be built in parallel — it says so itself, and confines itself to keying, storage,
regeneration and GC.

---

## 1. What a scope is

- **[D] Scopes want to be tags, not directories.** Agreed 2026-08-01, from the same pet peeve as photo and
  music collections: there are multiple valid categorizations, a directory tree can represent only one, and
  multi-category membership otherwise needs symlinks. Directory-drop stays the low-friction **constructor** —
  a folder generates a tag, nested folders generate nested tags — but the tag is the thing. Saved Visualizer
  selections live in the same namespace as peers rather than in a separate concept.
  - Supersedes the tentative "scope key = the dataset's file path", which assumed a Visualizer dataset is one
    monolithic `.bib`.
- **[P] Hierarchical scopes: the fit belongs to the scope you opened; children are a filter or a highlight
  over it, never their own fit.** Open `arxiv_ai` and `2026_07_new_studies` is a colour on the parent map.
  Fit two scopes independently and you need Procrustes to compare them — a problem you can decline to create.
  "View together" is union, "what's new since June" is difference, and neither needs a new fit.
  - **Consequence: Procrustes gets less urgent.** Still wanted for deliberate refit-and-realign, but no longer
    load-bearing for the add-new-papers workflow that motivated it (brief 11 item 4).

### The attachment scope: membership computed, not configured

Worked out 2026-08-05, from the other end — asking how the AI could search a chat's own attachments the way a
human hits Ctrl+F in a long document. It lands here rather than beside here, because what it needs is an
ordinary scope whose membership happens to be *computed*, and that is a shape this machinery should have
anyway (a saved Visualizer selection is the same idea with a different generator).

- **[N] Content-addressing removes the hard question before it is asked.** A sidecar filename is a content
  hash, so it already *is* a stable document ID. Index each attachment **once, globally**. There is then no
  per-chat index, and "what even is a chat in a multiverse" — the question that makes this look intractable —
  never arises at index time.
- **[P] "This chat's attachments" is a query-time filter, not an index.** It is the set of sidecar filenames
  reachable from HEAD: exactly `chattree.linearize_up(HEAD)` walked for `text_file` parts, which is the walk
  `llmclient.count_branch_tokens` already does and the mark phase `textfilestore.sidecar_refs_in_payload`
  already implements (union it with `imagestore`'s, as the datastore's `sidecar_extractor` does).
  - Branch-correct by construction: an attachment on a sibling branch is simply not in the reachable set.
  - Nothing to reindex on branch, reroll, or delete.
  - One page fetched twice on two branches is *one* index entry, because the hashes match.
- **[D] The retriever already supports it.** `hybridir.HybridIR.query` takes
  `include_documents: Optional[List[str]]` — "search only in the specified documents". Verified 2026-08-05.
  It was put there in 2025 against exactly this class of need, and searching across a Visualizer *selection*
  once the unified DB lands is the same mechanism with a different filter set. So this costs a filter
  computation, not a retriever change.
- **[D] This and the offset-reader are a pair, not alternatives.** They do different jobs, in the order a
  person does them: search locates a match and reports where in the document it sits, and the reader then
  fetches the surrounding span — the same gesture as clicking a search result and reading around it. That is
  already the contract between `search_documents` and `fetch_document`, whose docstring states it outright,
  so applying it to attachments adds no new concept.
  - **Consequence, and it is the strongest argument for one `read_document` rather than two tools:** once
    attachments are a searchable scope, an attachment and a knowledge-base document are the *same kind of
    thing* at query time. Two tool pairs would then be two spellings of one operation, differing only in
    which handle they accept.
  - The reader half is scoped in `TODO_DEFERRED.md`, "A fetched web page is budgeted as a user attachment,
    not as a speculative fetch", under v2.
- **Open:** whether an attachment scope is *visible* as a scope in the UI or only reachable by the AI; and
  whether indexing every attachment globally wants a retention policy, since the index then outlives the
  chats whose attachments produced it.

## 2. When to cluster, and what a map is

**[N] Visualizer: a dataset is a document scope, or several.** Clustering is expensive, so the question is
when it runs. Everything in this section is [P] and wants arguing with.

- **[P] Stratify by invalidation domain.** *Tier 0*, per-document: extraction, embedding, per-document NLP,
  per-document summary — shared across scopes, so a delete costs nothing and an add costs one unit. *Tier 1*,
  per-scope and global: the dimension-reduction fit, the HDBSCAN fit, cluster labels, corpus frequency stats.
  - Brief 12's derived-artifact store *is* the tier-0 cache. That is why the two briefs meet.
- **[P] The map is a materialized view with an as-of time, not a build target.** Procrustes exists because
  spatial memory is worth protecting, and a map that silently refits destroys exactly what Procrustes was
  added to preserve. So show staleness rather than fixing it: *"map built 3 days ago; 47 placed since, 2
  removed"*.
- **[P] Three visible per-document states**: **fitted**, **placed** (arrived later, projected through the
  existing fit), **pending**. Placement is out-of-sample `transform` plus cosine-to-medoid — which is brief 11
  item 3, already planned, doing double duty.
- **[P] Refit trigger: mean cosine-to-nearest-medoid over the placed points.** The same number as the planned
  novelty detection, aggregated. Crossing the threshold starts a background refit, Procrustes-aligns it, and
  *offers* the swap. Never swap without consent.
- **[P] One rule applied twice**: auto-build when there is nothing to lose (cold start — also the case the
  time-to-competence metric cares about), place when there is something to lose, refit on request or on
  measured drift.
- **[P] Label clusters from the medoid-nearest-k documents, not from all members.** Stabilises tier-1 labels
  under membership churn that does not move the medoid. Costs some accuracy; worth testing.
- **[P] A scope needs a readiness ladder, not an `is_indexing` boolean**: embedded → a map is possible; NLP
  done → labels are possible; summaries done → interrogation is possible.

### Clustering implementation, if it becomes the bottleneck

- **[N] GPU-accelerated clusterers?**
  - **[P] Do PCA preprocessing first** — brief 11 item 2, free, and 768→50 before UMAP is a large enough
    constant factor that the CPU path may stop being the bottleneck. Measure after that, not before.
  - **[P] cuML** has GPU HDBSCAN and UMAP, sklearn-compatible, with UMAP the bigger win — but it drags in
    RAPIDS, is CUDA-version-pinned, and collides with the "easy install with a chosen CUDA version" deferred
    item. Take the dependency only if measurement demands it.
- **[X] ~~Test UTU's Clust-Splitter.~~ Measured 2026-09-01 without wrapping it, and the answer is no —
  not for clustering quality.** The question this item existed to settle was whether MSSC loses to
  density- and linkage-based methods because the *objective* is wrong for literature embeddings, or
  because Lloyd's algorithm finds bad optima of a fine objective. Those predict the same table, so
  k-means's poor showing could not distinguish them.

  **Ward linkage settles it, and costs ten minutes rather than an f2py wrap of 8975 lines of Fortran.**
  Ward minimizes within-cluster sum of squares agglomeratively — the same objective Clust-Splitter
  solves, reached by a better search than Lloyd's. Scored on the matched control of
  `investigations/highdim-clustering/`:

  | | arXiv | AOKK |
  |---|---|---|
  | average-linkage (the method chosen) | **+0.067** | **+0.089** |
  | Ward — MSSC objective, greedy search | +0.048 | +0.049 |
  | k-means — MSSC objective, Lloyd's | +0.019 | +0.042 |

  So **part of k-means's weakness really was the solver** — Ward beats it, by 2.5× on arXiv, which is
  the point that was in Clust-Splitter's favour and it is real. But **the objective still loses when
  solved better**: Ward lands between k-means and average-linkage on both corpora. Overtaking
  average-linkage would need another 40–80% over Ward on an objective Ward already optimizes
  competently, which is a great deal to ask of a solver.

  *Two caveats, so nobody re-opens this on a technicality: sklearn's Ward requires euclidean, so it is
  not a cosine-MSSC — on unit vectors the two are closely related (`‖a−b‖² = 2−2cos`) but not identical
  — and Ward is greedy, so it is a lower bound on what a real MSSC solver reaches. Neither gap looks
  like 40–80%.*

  **What stays open is the other reason to want it**, and it is not about cluster quality: MSSC yields
  *optimized* centroids, so placing a new document on an existing map is an exact `argmin` over k.
  Agglomerative gives centroids computed after the fact, which were never optimized for that. If the
  incremental-map work (brief 11 item 4, Procrustes) needs exact placement and post-hoc centroids prove
  inadequate, revisit **on that ground**. The `subgrad_help_b` bug report to the authors stands
  regardless.

- **[N] ~~Test UTU's Clust-Splitter.~~** *Original note, kept for the references and the March 2026
  context.* Turku group (Lampainen, Karmitsa, Joki,
  Mäkelä), MSSC via LMBM, 8975 lines of Fortran, incremental in k. Conclusion then: **f2py-wrap it as-is
  first** to test fit before considering a PyTorch port — the objective/subgradient is O(n·k·d) and dominates
  roughly 1000:1, but internal callbacks make that layer awkward to reach through f2py. A suspected bug at
  `subgrad_help_b` line 1308 (`a(j,i)` for the value, `b(j,i)` for the gradient) is still to be raised with
  the authors.
  - **[D] References**, supplied 2026-08-05 — until then this note carried authors and method from a March
    2026 conversation with no way to reach the source, which blocked both the f2py-wrap test and the
    `subgrad_help_b` bug report:
    - Code: [github.com/jmlamp/Clust-Splitter](https://github.com/jmlamp/Clust-Splitter)
    - Paper: [arXiv:2505.04389](https://arxiv.org/abs/2505.04389)
  - **[P] Its incrementality is in k, not in data** — easy to conflate under a "when to cluster" heading.
  - **[P] But centroid-based fits the placed/fitted model better than density-based.** MSSC gives real
    centers, so placing a new point is an exact `argmin` over k rather than `approximate_predict`, and brief
    11 item 3 is already retrofitting a centroid model onto HDBSCAN output. Against: HDBSCAN's variable
    density and its *noise* concept model something real — literature embeddings are not spherical, and
    forcing every document into a cluster is a lie you then have to look at on the map. Plus MSSC needs k.
  - **[P] Brief 11 item 5 (added 2026-08-05) adds a practical argument to that modelling one.** The
    authoritative clustering should move into the high-dimensional space, where HDBSCAN's memory is what
    forces the current 10k sample — so an algorithm that fits the whole corpus in that space wins something
    the present 2D arrangement never made visible. Keep the two choices separate: *which space* is brief
    11's, *which algorithm* is this one. The first changes the constraints on the second.

## 3. What a document is — settled enough to stop re-litigating

- **[D] "A document is a file" stands.** Clean and simple in a way the alternatives are not.
- **[X] ~~Ingest `.bib` as a container: N documents at `foo.bib#entry-key`, with locators into the original
  file.~~** Retracted 2026-08-01 (Juha's objection, accepted):
  - The DB indexes **chunks** and the RAG tool surface handles **documents**; containers make it three levels.
  - Record-boundary documents sharing one file need offsets and lengths — a **new mechanism**.
  - The user *will* insert an entry in the middle.
  - The only win was removing a CLI step, which does not pay for the above.
- **[D] `raven-burstbib` was only ever an import tool**, and a monolithic `.bib` is more common than
  single-entry files. If the items are already loaded, export from *that*, not from the tool.
- **[D] Fix the burst-step friction without the container mechanism**: burst on ingest into a **visible**
  sibling directory (`foo.bib.d/`), original untouched, shown in the UI, one-click undo. Addresses the "too
  much magic" objection at a fraction of the machinery, and keeps a document a file.
- **[D] Updating a `.bib` must regenerate `.bib.d/` selectively.** Hash each record's normalized text at burst
  time, rewrite only what changed. This hashing is **internal to the burst step** and never leaks downstream,
  which is what keeps it compatible with having retracted the container idea.
  - **Not rewriting unchanged files is load-bearing, not an optimization.** If the burst touches every file,
    mtimes bump and the entire downstream per-file cache invalidates — the hash comparison would have been
    built and then thrown away one layer down.
  - **Deleted records leave orphans in `.bib.d/`**, with the same recover-or-delete affordances the sidecar
    orphan problem needs. Another argument for brief 12's unification.
- **[D] `.bib.d/` files are named by BibTeX slug, not by hash.** Slugs are unique within a `.bib`, so
  per-directory uniqueness holds, and a slug is legible in a file manager. Needs light filename sanitization.
- **[X] ~~Content-hash *document IDs*.~~** Withdrawn 2026-08-01 after being weakened twice. The reorg case is
  fair (the same reorganization invalidates an rsync backup), and **dedup across scopes is solved by tags,
  not by hashing** — one file with three tags is one document with one embedding. Content hashing survives as
  a *dedup check*; it is the **identity scheme** that died.
- **[X] ~~Do the ID scheme inside the Nomic window because it is the cheapest moment.~~** Retracted: one 12k
  pile plus a couple of small ones is hours, not weeks, and the cost is the same whenever it happens.

## 4. Corpus TOC for the model — blocked on the above

- **[P] Use cluster keywords as the scope description.** The Visualizer already computes them per scope, so
  the TOC is derived rather than hand-written and stays fresh at the same cadence as the map — the drift
  metric governs both. Scope names alone are much weaker: they say a topic exists but nothing about extent or
  granularity, which is most of what the model needs in order to decide whether looking is worth a turn.
- **[P] Size split**: names plus document counts always (~10 tokens each, and the counts do real work — "3
  docs" and "12,431 docs" warrant different behaviour), with cluster-level detail behind a `describe_scope`
  tool. Makes hierarchical scopes fall out: top-level names always, expand on demand.
- **[P] Placement**: system prompt, stable across a session, cached once. The opposite of the tool-loadout
  churn problem that motivated the tool-budget fix.
- **[P] Wording trap**: a TOC invites reasoning about absence ("no scope about X, therefore nothing about X"),
  but a paper about X may sit inside `hydrogen_papers` without being prominent enough to surface as a cluster
  label. Frame it as **what is prominent, not what is present** — one sentence, cheap now, expensive later as
  a confabulated "we have nothing on that".
- **[P] Autodetect the retrieval mode from the scores, with an optional model-set hint.** The score
  distribution tells you about the corpus; the utterance tells you about the task. "What do these papers say
  about X" and "find me the paper about X" can produce near-identical distributions and want different k, and
  autodetect cannot see that difference because it is not in the data. Hint optional and usually absent, not a
  required parameter the model reasons about on every call.

## 5. Test corpora

- **[N] The Friendship is Optimal add-on stories**, which ship as HTML. Genuinely useful beyond the joke: it
  is mode 2 in the corpus-interrogation sketch (a corpus the reader already knows), "which add-on is which" is
  a known-item retrieval task, and it is **out-of-domain in three ways the current eval set cannot test** —
  the stopword list is tuned for scientific text, `format_entry_for_keyword_extraction` assumes bibliographic
  fields, and the embedding model is trained on question-answer pairs rather than narrative prose. (The
  third reason was originally "the MiniLM reranker is MS MARCO-trained"; there is no reranker — it was
  measured and rejected on 2026-08-06, see `investigations/retrieval/REPORT.md` §2. The out-of-domain
  argument survives the substitution, and in fact this corpus already demonstrated it: on fiction,
  retrieval fails exactly when the query describes what the prose *dramatizes* rather than states.)
- The **axially-moving-materials bibliography** is the in-domain counterpart, and brief 11 records its
  measured shape and what kind of verdict it can give. Between them they cover in-domain and out-of-domain.

---

## What the design session has to settle

Collected so the session has an agenda rather than a pile:

1. **The tag data model.** Nesting, the directory-drop constructor, how a saved Visualizer selection becomes a
   tag, and what happens to a tag when its constructing directory changes.
2. **Which of §2's [P] items survive.** They form a coherent story — materialized view, three states, drift
   trigger, readiness ladder — but the story has not been argued against, and it is the expensive part to
   build.
3. **Where the scope↔document relation lives**, given that brief 12 owns derived artifacts and hybridir owns
   the index. Three stores currently believe different things about who is authoritative.
4. **Whether the TOC (§4) is one mechanism or two** — the always-on names-and-counts, and `describe_scope`.
5. **Migration.** Existing hybridir datastores and existing Visualizer datasets both predate all of this.

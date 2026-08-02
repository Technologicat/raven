# Monday 2026-08-03 — pick-up checklist

From the 2026-08-01 evening design session (Juha + Claude, claude.ai), plus Juha's own notes dumped in that
session, revised after his corrections. Goal for Monday: thin the brief pile and repair the maps of the work.

**Provenance marking**, since the session mixed several kinds of item:

- **[D]** — decided, or already decided and merely recorded here.
- **[N]** — from Juha's notes, carried over as-is.
- **[P]** — Claude's proposal, not agreed. Argue with these before acting.
- **[X]** — proposed and **retracted** during the session. Kept with the reason, so it isn't re-proposed.

---

## 1. Repair the maps (do first — cheap, and everything else reads them)

- [x] **[D] Fix the stale Visualizer SLOC claims.** *Done 2026-08-03.* All three updated, plus two things
      found alongside: the `~700 lines per module` guideline had been stranded on the `app.py` bullet (lifted
      to the section, where it applies to all of them), and `raven/visualizer/CLAUDE.md`'s "no tests" rationale
      was half-expired — the refactor it was meant to protect has landed, so what remains is pinning the new
      module boundaries. Still zero test files under `raven/visualizer/`, verified. The `app.py` refactor
      landed (measured in a fresh clone:
      `app.py` 1912, `info_panel.py` 1518, plus `selection`, `plotter`, `annotation`, `word_cloud`,
      `entry_renderer`, `app_state` split out). Three places still describe a 4427-line god object, and CC will
      keep reading whichever it hits first:
  - `CLAUDE.md:223` — "4427 lines, monolithic"
  - `raven/visualizer/CLAUDE.md:6` — same number in the module map
  - `TODO.md:166` — "currently a god object (~4k SLOC)", still **[High]**; largely done. The "extract the info
        panel" sub-item is closed (it is now its own module at 1518). Reduce to whatever genuinely remains.
- [x] **[D] Add a top-level `briefs/done/`.** *Done 2026-08-03*, created by the `ai-act-article-50-summary.md`
      move below. Only `summer_2026_librarian_extension/` had one, which is how the SLOC drift above survived
      unnoticed.
  - **Follow-up, not done: the rest of the archival sweep.** The top level still holds a lot of finished work —
        the whole `xdot-viewer-*` set, `speech-extract-to-common-implementation.md`, `cherrypick-spec.md`, the
        two performance audits, and notably **`visualizer-refactoring.md`**, which is the brief for the refactor
        that just landed and is the direct cause of the drift fixed above. Each needs a done/not-done judgement,
        so it is a sit-down rather than a sweep.
  - **One dependency to check first**: `raven/visualizer/CLAUDE.md` points at `briefs/visualizer-refactoring.md`
        for the `app.py` section map. Archiving it moves that path.
- [ ] **[P] While in there: check whether other "current state" claims in `CLAUDE.md` have drifted.** The three
      above were all one refactor. Test-coverage and known-issues lists are the likely next candidates.
  - *Partly done 2026-08-03*: the Visualizer "no tests" claim is **still true** (verified — no test files under
        `raven/visualizer/`), but its rationale was half-expired and has been rewritten. The project-level
        test-coverage and known-issues lists have **not** been audited.

## 2. Brief reorganization

- [x] **[D] `briefs/ai-act-article-50-summary.md` → `briefs/done/`.** *Done 2026-08-03.*
- [x] **[D] `briefs/atmospheric-dust.md` and `briefs/crt-display.md` → the summer sprint folder.** *Done
      2026-08-03.* Moved to the sprint folder, **not** to `done/` — both are queued for Researchers' Night.
      (Juha's original note was unclear on this; corrected 2026-08-01.)
- [x] **[D] `briefs/visualizer-importer-rework.md` → the sprint, probably as `11_`.** *Done 2026-08-03* as
      `11_visualizer-importer-rework-brief.md`. Had been forgotten; it carries Nomic, PCA preprocessing,
      cosine-to-medoid outlier assignment, and Procrustes.
- [x] **[D] While renumbering 11, amend its item 1 — the Nomic choice is a fork, not a version bump.**
      *Done 2026-08-03, and the fork stands.* Searched rather than assumed: **no v2-aligned vision encoder
      turned up** — the vision encoder is consistently paired with v1.5, and Nomic's `nomic-embed-multimodal-3b`
      is a standalone visual-document-retrieval model with its own latent space rather than a v2 companion.
      Recorded in the brief as absence-of-evidence from a search, not a checked changelog, so it wants
      re-verifying before the choice is committed. **The decision itself is still open** — this only removed
      the escape hatch. Item 1

      currently names `nomic-embed-text-v1.5` + `nomic-embed-vision-v1.5`, justified by *"unified text+image
      embedding space"*. But **`nomic-embed-vision` is aligned to text-v1.5, not to v2**, and v2-moe is the
      multilingual one (~100 languages, 1.6B pairs, explicitly cross-lingual; Apache 2.0; 475M total / 305M
      active). **First check whether a v2-aligned vision encoder exists** — if it does, the fork dissolves. If
      not:
  - **v1.5** — figures rank against text natively in one collection; English-centric.
  - **v2-moe** — Finnish, Japanese and other non-English material works; images lose the direct route and go
        through the description pivot instead (which exists anyway, since images have OCR and description
        channels regardless).
  - **This is bigger than the multilingual-lyrics case that surfaced it.** JAMK's own context is Finnish, and
        multilingual scientific literature is a real corpus property rather than a niche. Worth deciding
        deliberately.
  - **Related, and it interacts with item 2**: v2-moe is Matryoshka-trained, so vectors truncate 768→256 with
        claimed minimal degradation — 3× less embedding storage. It does **not** simply subsume the PCA
        preprocessing step, though: Matryoshka truncation is a fixed property of the model's training, while
        PCA is *corpus-adaptive* and item 2's stated purpose is measuring this corpus's effective
        dimensionality. They may still compose.
  - **[P]** Worth noting the folder is named `summer_2026_librarian_extension` but is about to hold two avatar
        briefs and an importer brief. Either rename it or accept it as a sprint-by-date rather than by
        component — but decide, rather than letting the name quietly stop being true.
- [ ] **[D] `model-lineup-autumn-2026.md`**: Gemma 12B, not Qwen.
  - *Not done 2026-08-03 — the note is too terse to act on without guessing.* The doc's open questions include
        *"Whether **Qwen3.6-12B** is worth installing"*, so the likely reading is that the 12B model actually
        worth considering is **Gemma**, and Qwen has no 12B — but that is inference from one adjacent line, not
        something checked. The body is otherwise "Qwen across the board" with Gemma 4 as the multilingual
        backup, so the correction could instead be about the *recommendation*. Needs one sentence from Juha
        before editing, since the two readings change different paragraphs.
- [ ] **[D] `context-inject-shape-measurements.md`**: "what Raven ships today" is out of date; Qwen 9B is not
      the weakest; not reported in all tables; Raven's document search is now a tool the model can call.
  - *Not done 2026-08-03 — deliberately, because this one is a measurements record and the obvious edit would
        corrupt it.* What changed is the **product**, not the data: the runs happened, and "Raven ships one
        user-role message per inject" and "document search is not a tool the model can call" were both true
        when measured. Overwriting them in place would silently restate history. The right shape is a dated
        "what has changed since" preamble that leaves the body intact, plus in-place *markers* where a claim
        would otherwise mislead a reader mid-table.
  - **One of the four is a genuine erratum rather than drift, and needs checking**: "Qwen3.5-9B is the weakest
        model in the set" is stated absolutely, but Gemma4-E4B appears in some tables and not others. So the
        claim is true for the tables where E4B is absent and wrong where it is present — which is exactly the
        "not reported in all tables" item, and the two should be fixed together.

### Turn / round terminology

*Not started 2026-08-03: this is a code sweep gated on an unagreed [P], so it is not a doc quick win. The
naming decision has to land first, and it renames things across the codebase, not just in prose.*

- [ ] **[D] Make usage consistent everywhere.** Current split:
  - *Briefs*: a **turn** may consist of several **rounds** when the model calls tools.
  - *Codebase* (approximately): a **round** is user+AI; an individual AI message is a **turn**; the high-level
        sequence of AI and tool messages has **no name**.
- [ ] **[P] Recommendation: the briefs are already standard usage; change the code.**
  - **turn** = one participant's contribution — including the whole tool loop for an assistant turn. This is
        what the codebase currently leaves unnamed, and brief 10's existing "tool-call round cap" already
        assumes it.
  - **round** = one iteration of the agent loop within a turn (model call → tool calls → results).
  - **exchange** = user turn + assistant turn, if the pairing needs a name (this is today's code "round").
  - Check `ai_turn` against the result — under this convention the name is already correct if it runs the loop.
- [x] **[D] House term is "scaffold", not "harness"**, for Raven's own agent loop (Seth Herd influence).
      *Done 2026-08-03.* Three places had lost it, and `raven.librarian.scaffold` already exists, so the
      inconsistency was internal:
  - `TODO_DEFERRED.md` — heading is now "Headless scaffold mode for `ai_turn`"; the body's "the harness is
        not a third frontend" and "a harness with a scripted backend" became "the headless mode" and "a
        headless driver".
  - `TODO_DEFERRED.md` — the proposed `raven.librarian.agentharness` is now "a thin headless driver module
        beside `scaffold`", left **deliberately unnamed**: `scaffold` already owns the concept, so the name
        should follow what the module turns out to do.
  - `briefs/design/product-identity-sketch.md` — citation updated to match the new heading.
  - **Leave alone**: `TODO.md:423` and `product-identity-sketch.md:53` use "generic agent harness" for *other
        people's* products, which is the right word there. `evaluation/retrieval/README.md` and
        `TODO_DEFERRED.md:3121` mean *test* harness — different sense.

### `raven.papers.bibtex` — consolidate the readers

*Not started 2026-08-03: code, not documentation. Small and well-scoped, but it changes two call sites and
wants `ruff` plus the suite, so it belongs in the morning rather than in a doc pass.*

- [ ] **[D] The module has a writer and no reader**, so reader code is duplicated against raw `bibtexparser`.
      Verified in a fresh clone:
  - `raven/papers/bibtex.py` — `__all__ = ["entries_to_bibtex"]`. Writer only.
  - `raven/visualizer/importer.py:169` — `bibtexparser.parse_file(...)` with middleware
        `[NormalizeFieldKeys(), SeparateCoAuthors(), SplitNameParts()]`
  - `raven/librarian/chatutil.py:479` — `bibtexparser.parse_string(...)` with the **identical** middleware triple
  - (`raven/papers/wos2bib.py` uses `bibtexparser.model` for *writing*; not part of this.)
  - Shape: add `parse_file` / `parse_string` to `raven.papers.bibtex` carrying the one canonical middleware
        list, and point both call sites at it.

## 3. Librarian behaviour changes decided

- [ ] **[D] Tools should error out informatively when the budget is exhausted, not be withdrawn.** Replaces the
      v1 "final round with no tools" approach for stopping a runaway agent loop (Qwen going into deep research
      unasked). Two reasons:
  - **Avoids KV-cache burn** from a mid-turn tool-loadout change — the measurable one, and the original motivation.
  - **Better distribution fit**: history referencing a tool no longer in the schema is off-distribution, whereas
        tool *errors* are well represented in training. Possibly related to the Q11 measurement (literal
        `<tool_call>` emission) — a hypothesis, not a finding.
  - Keep the "no more calls this turn" note in the error payload.
- [ ] **[N] Document injects: expose offset/length** so the model can locate the truncated middle. Same for RAG
      results, so it can look around a hit.
  - **[P]** Also makes the consulted-docs list span-exact rather than document-level.
  - **[P]** Wants extraction to be deterministic and cached, or the offsets aren't stable.
  - **[X] ~~"and Article 50 export cites exactly what was seen"~~** — wrong. Article 50 here is only about
        marking AI messages as AI-generated. See §7 on the two senses of "provenance".
- [ ] **[N] Consulted-docs list**: add offset and length; add "previously consulted" **for disambiguation** —
      so the model doesn't read the list as referring to the current turn. Consistency then forces the inject
      ordering: list first, then the current turn's autosearch results.
- [ ] **[N] "No sources consulted" marker — only when Docs is ON.**
- [ ] **[N] Is the separate Speculation toggle still needed?** If Docs ON implies marking as appropriate,
      probably not.
  - **[P]** Check first: does the marker distinguish "Docs ON, nothing retrieved" from "Docs OFF, answering
        from weights"? Different epistemic states; if the marker collapses them, the toggle still carries
        information.
- [ ] **[N] Compaction for Researchers' Night**: interaction with attachments; always keep the first two turns.
  - **[P]** Watch the sidecar interaction — compacting away a turn that referenced an attachment leaves the
        sidecar live but unreferenced in visible history.
- [ ] **[X] ~~Tell the model which term pulled a document in.~~** Retracted 2026-08-01 (Juha): the term is
      already in the snippet and in the user's question, and the injection needs to look like standard search
      results for distribution-fit reasons — which is the same argument used *for* the tool-error change above,
      so it applies here too.

## 4. Retrieval (brief 09 territory)

- [ ] **[P] The `min_p` survivor count measures peakedness, not relevance — check before building on it.**
      For "what is 2+2?" the best score is low and everything else is comparably low, so the ratio to best is
      near 1 for many chunks → high survivor count → reads as "broad, well-covered". That is the exact case the
      marker exists to catch. Peakedness separates "one clear winner" from "many similar", but "broad query
      with real coverage" and "query about nothing" are both flat.
- [ ] **[P] Two floors, each in its own currency.** Brief 09 already notes the engines are asymmetric; use that
      rather than trying to calibrate BM25 scores:
  - **Semantic**: absolute cosine floor. Answers "is anything here at all".
  - **BM25**: not a score floor — **did any high-IDF query term match?** IDF is a corpus property known at
        index time, so it is absolutely calibrated even though the scores are not. This is what rescues
        proper-name queries, where cosine is mediocre and an exact rare-term hit is the real signal (Juha's
        objection, 2026-08-01).
  - Inject if either passes.
- [ ] **[P] The same two measurements route the retrieval mode**, so the marker work pays for this for free:
  - below both floors → nothing there; fire the marker, skip the injection, save the tokens
  - above floor, peaked (or high-IDF exact match) → **selective** mode; rerank, small k
  - above floor, flat → **coverage** mode; stratify, large k
- [ ] **[N] Good k for scientific work.** Reviewing a cluster or user selection may need k≈100 abstracts at
      once; a non-specific search over a 10k corpus needs k≈100 for even 1% coverage. Not needle-in-haystack.
  - **[P] Top-k is the wrong tool for the coverage regime** — ranking by similarity oversamples one region, so
        top-100 over 10k documents lands in two clusters. Use **cluster-stratified sampling**: medoid plus
        nearest few per cluster, weighted by cluster mass. Only possible because map and retrieval are in one
        system.
  - **[P] Do not rerank the coverage set** — it re-concentrates what stratification spread out. MiniLM-L6 for
        selective mode, stratification for coverage mode; not the same knob.
- [ ] **[D] Summaries: do *not* switch `visualizer_config.summarize` on at import.** 10k abstracts would take
      hours-to-days rather than ~15 minutes, both with GPU. Disqualifying.
  - **[P] But summarize lazily on retrieval instead.** Coverage mode needs summaries for the ~100 documents
        actually retrieved, not for the 10k. Cache per document, amortize across queries; cost is then bounded
        by what gets looked at rather than by corpus size. Converges on what the interrogation sketch already
        asks for — lift `summarize` out of the importer into the library, run it against a scope. Payoff at
        k=100: ~40k tokens of abstracts → ~10k of summaries.
- [ ] **[N] MiniLM reranker** (`cross-encoder/ms-marco-MiniLM-L6-v2`, 23M, CPU).
  - **[P]** Good for the VRAM story — CPU keeps the card for the LLM. But MS MARCO-trained on web queries, so
        domain shift to scientific abstracts is real. Measure against `evaluation/retrieval`, which is mode 2
        in the interrogation sketch ("a corpus the reader already knows") doing its intended job.
- [ ] **[D] Autosearch vs. tool-call: undecided.** The two cover different moments (autosearch = the user asked
      something the corpus obviously bears on; tool = the model realises mid-reasoning it should check). That
      *was* the intention; the blocker is that retrieval isn't good enough yet, which is what this section is
      about.
  - **[P] Possible third option**: keep unconditional autosearch, gate the *injection* on the floors above.
        The cost is the injected tokens, not the retrieval — two decisions currently made as one. Degrades
        safely: a too-low floor reproduces today's behaviour rather than regressing.

## 5. Corpus / scopes / unified DB — design session, realistically post-Researchers' Night

**Timing, honestly**: 04, 05, 06 and 09 to implement, then v0.2.8, then demo polish through 26 September. The
design session lands after that unless something slips forward. Recorded here so it isn't re-derived.

- [ ] **[D] Scopes and the unified DB are the prerequisite** for the cluster-keyword TOC (§6) and most of the rest.
- [ ] **[D] Scopes want to be tags, not directories.** Agreed 2026-08-01 — the same pet peeve as photo and
      music collections: multiple valid categorizations, directories can represent only one, and multi-category
      membership otherwise needs symlinks. Directory-drop stays the low-friction *constructor* (a folder
      generates a tag, nested folders generate nested tags); saved Visualizer selections live in the same
      namespace as peers. Supersedes the tentative "scope key = the dataset's file path" idea, which assumed a
      Visualizer dataset = one monolithic `.bib`.
- [ ] **[P] Hierarchical scopes: the fit belongs to the scope you opened; children are a filter/highlight over
      it, never their own fit.** Open `arxiv_ai`, and `2026_07_new_studies` is a colour on the parent map. Fit
      two scopes independently and you need Procrustes to compare them — a problem you can decline to create.
      "View together" is union; "what's new since June" is difference; neither needs a new fit.
  - **Consequence: Procrustes gets *less* urgent.** Still wanted for deliberate refit-and-realign, but no
        longer load-bearing for the add-new-papers workflow that motivated it. One item off the snowball.
- [ ] **[N] Visualizer: dataset = document scope (or several).** When to cluster (expensive)?
  - **[P] Stratify by invalidation domain.** Tier 0 (per-document): extraction, embedding, per-doc NLP, per-doc
        summary — shared across scopes, delete costs nothing, add costs one unit. Tier 1 (per-scope, global):
        dim-reduction fit, HDBSCAN fit, cluster labels, corpus frequency stats.
  - **[P] The map is a materialized view with an as-of time, not a build target.** Procrustes exists because
        spatial memory is worth protecting; a map that silently refits destroys what Procrustes was added to
        preserve. Show staleness, don't fix it: *"map built 3 days ago; 47 placed since, 2 removed"*.
  - **[P] Three visible per-document states**: **fitted**, **placed** (arrived later, projected through the
        existing fit), **pending**. Placement = out-of-sample `transform` + cosine-to-medoid — which is rework
        item 3, already planned, doing double duty.
  - **[P] Refit trigger = mean cosine-to-nearest-medoid over the placed points.** Same number as the planned
        novelty detection, aggregated. Crossing threshold → background refit, Procrustes-align, offer the swap;
        never swap without consent.
  - **[P] One rule, applied twice**: auto-build when there is nothing to lose (cold start — also the case the
        time-to-competence metric cares about), place when there is, refit on request or measured drift.
  - **[P] Label clusters from the medoid-nearest-k documents, not all members.** Stabilises tier-1 labels under
        membership churn that doesn't move the medoid. Costs some accuracy — worth testing.
  - **[P] A scope needs a readiness ladder, not an `is_indexing` boolean**: embedded → map possible; NLP done →
        labels possible; summaries done → interrogation possible.
- [ ] **[N] GPU-accelerated clusterers?**
  - **[P] Do PCA preprocessing first** — rework item 2, free, and 768→50 before UMAP is a large enough constant
        factor that the CPU path may stop being the bottleneck. Measure after that.
  - **[P] cuML** has GPU HDBSCAN and UMAP (sklearn-compatible), UMAP the bigger win — but drags in RAPIDS,
        CUDA-version-pinned, colliding with the "easy install with a chosen CUDA version" deferred item. Take
        the dependency only if measurement demands it.
- [ ] **[N] Test UTU's Clust-Splitter.** Context from March 2026: Turku group (Lampainen, Karmitsa, Joki,
      Mäkelä), MSSC via LMBM, 8975 lines of Fortran, incremental in k. Conclusion then: **f2py-wrap as-is
      first** to test fit before considering a PyTorch port — objective/subgradient is O(n·k·d) and dominates
      ~1000:1, but internal callbacks make that layer awkward to reach through f2py. Suspected bug at
      `subgrad_help_b` line 1308 (`a(j,i)` for the value, `b(j,i)` for the gradient) still to raise with the
      authors.
  - **[P] Its incrementality is in k, not in data** — easy to conflate under a "when to cluster" heading.
  - **[P] But centroid-based fits the placed/fitted model better than density-based.** MSSC gives real centers,
        so placing a new point is exact `argmin` over k, not `approximate_predict`; rework item 3 is already
        retrofitting a centroid model onto HDBSCAN output. Against: HDBSCAN's variable density and its *noise*
        concept model something real — literature embeddings aren't spherical, and forcing every document into
        a cluster is a lie you then have to look at on the map. Plus MSSC needs k.

### Derived artifacts: the sidecar mechanism *is* the tier-0 cache

Raised 2026-08-01 (Juha), from noticing that image DB documents will need stored OCR text and a rescaled
image, and that this is the same kind of thing as a burst `.bib.d/`. **This is core logic, so it wants
designing before images land in the DB rather than discovered afterwards.**

- [ ] **[D] Downscaled image, OCR text, extracted PDF/office text, per-record `.bib`, embedding are all one
      kind of object**: derived from a source artifact, expensive, content-determined, not authoritative,
      regenerable. The sidecar store is where several of them already live.
- [ ] **[D] There are currently three storage answers for that one kind of object.** Verified in a fresh clone,
      and the point of the design work is to pick one:
  - *Chat attachment text* — **not persisted**. `textfilestore` / `sidecarstore.sidecar_to_text` extracts on
        demand and memoizes on the content-addressed filename, so a chat with an attached PDF re-extracts it at
        most once per process.
  - *DB document text* — **persisted inline** in hybridir's own datastore: `_read` calls the
        `docextract.extract_text` callback at ingest and passes the result to `add(document_id, path, text)`.
  - *Attachment bytes* (image, PDF, office) — **content-addressed sidecar**.
- [ ] **[P] Unify the key shape, not the storage layout.** The key that covers every case:

        (source content hash, derivation kind, derivation version)

      The version stamp is the part that pays later and is nearly free now. `extract_keywords` already does
      this locally (`nlp_cache_version = 1`); generalizing it means the Nomic switch invalidates
      embedding-derived artifacts and leaves OCR text and extracted PDF text alone. Today each cache invents
      its own versioning or has none, which is what turns a model swap into a full reprocess.
- [ ] **[P] Share the mechanism, separate the stores.** Chat sidecars are GC'd by reference from chattree
      nodes; DB sidecars would be GC'd by reference from the document index. Two lifecycles. One
      implementation, two reference-map providers, two stores — a collector that has to understand both graphs
      at once is where this kind of code goes bad.
  - **Encouraging**: the backend-agnostic layer partly exists already. `sidecarstore` holds the shared
        mechanics (URL scheme, provenance skeleton, byte ingestion, GC content-walk) with `imagestore` and
        `textfilestore` as two specializations mirroring each other at three public operations each, and the GC
        mark phase is documented as composable by set union over per-kind reference interpreters. A DB store is
        a **third specialization plus a third reference-map provider**, following an existing in-tree pattern —
        not a backend-agnostic layer invented from scratch.
- [ ] **[P] Don't force one placement policy.** The chat store is content-addressed because dedup matters and
      nobody browses it; `foo.bib.d/smith2024.bib` is legible in a file manager, and DB documents *are*
      browsed. That difference looks essential rather than accidental. Unify the derivation registry; leave
      placement pluggable.
- [ ] **[P] The "off-diagonal import cells" are derivations, not import paths.** "PDF page range as images" and
      "text off an image" derive from an already-ingested document, which is the better frame because it means
      they **compose**: PDF → page image → OCR text is two registered derivations, with the intermediate cached
      like anything else. Once the registry exists, adding one is registering a producer function.
  - **Downloading a document from a URL is the genuinely different one**: that is *acquisition*, not
        derivation, and it is a write path into the indexed corpus — which `corpus-interrogation-sketch.md`
        already flags as wanting deliberate treatment (token, size cap, extension filter, off by default)
        rather than inheriting the inference API's trusted-network posture.
- [ ] **[N] Spreadsheets are not the next format in the cascade** — already correctly filed as its own design
      question at `TODO_DEFERRED.md:1018` and `:2549`. A naive `.xlsx` text dump is column soup: it pollutes
      BM25 with header repetition and embeds to nothing meaningful. `.ods` may be nearly free (`odfpy` is
      already a dependency for `.odt`/`.odp`), but that is a reader, not the design question.
- [ ] **[D] Format coverage as of this week**: plain text, PDF, `.docx`, `.odt`/`.odp`, saved HTML. **The
      talk's "not just papers" constraint is now satisfied literally, not just in principle** — worth stating
      without hedging in the vision doc.

- [ ] **[D] This needs its own brief, and it does *not* depend on scopes.** Brief-shaped rather than
      sketch-shaped: the mechanism is mostly known (`sidecarstore` exists, the specialization pattern exists,
      the key shape is the open part). So it can be written and built while the corpus design session is still
      pending — which un-blocks images-in-DB, which in turn gates several things.
  - **Draft exists** as `12_derived-artifact-store-brief.md`, with three proposed decisions (D1–D3) and six
        open questions (O1–O6).
  - **[D] It is v0.2.9 work and does not gate v0.2.8.** The release cuts once brief 09, the webfetch
        attachment work, and the queued UX fixes are in.
  - **[D] Webfetch sidecarring lands *first*, from the already-running CC session, and that is deliberate.**
        Renegotiating an in-flight compacted session costs more friction than the retrofit does. The
        chat-attachment pattern it follows is already the right one, so it will arrive as another consumer of
        `sidecarstore`/`textfilestore` rather than a fourth invented scheme — retrofit is *extract the inline
        extraction call into a named producer and register it*. Added to brief 12's O3.
  - What the brief must settle: which store is authoritative for derived text; the registry's key shape,
        version stamping and producer registration; the two reference-map providers and how the GC mark phase
        composes; per-store placement policy (content-addressed vs. filesystem-legible); migration for existing
        hybridir datastores and existing chat sidecars; and what a version bump does — lazy re-derive on next
        access, or an eager sweep.
- [ ] **[P] Recommendation for "which shape wins": the content-addressed sidecar, with the inline copy
      demoted rather than deleted.** Derived text becomes a sidecar keyed by `(source hash, kind, version)`;
      the chat path becomes look-up-and-compute-on-miss (still lazy, but persistent across processes); hybridir
      stops being the authority.
  - **Three consumers already agree and hybridir is the outlier**, which is why this is evidence rather than
        preference: `textfilestore`'s docstring argues no-text-inline explicitly ("the datastore stays small
        even for a large PDF"), and the deferred webfetch item makes the identical argument at
        `TODO_DEFERRED.md:534` ("a smaller datastore, since the JSON keeps a `sidecar:` reference rather than
        the text").
  - **What hybridir actually stores, verified**: `add()` persists the **full plaintext**, and
        `_prepare_document_for_indexing` derives chunks that each carry their own `text` plus `chunk_id` and
        `offset` — with `extra=0.4` overlap, so the chunk set alone is ~1.4× the document, stored alongside the
        full copy.
  - **So the amplification caveat is weaker than first argued**: the BM25 rebuild works off the tokenized
        chunks already in the datastore, not off the full text. The full copy is needed only for re-chunking on
        a full rebuild (rare, already expensive) and possibly for `fetch_document` returning a whole document —
        both servable from a sidecar. **The chunks stay** (they *are* the search results); the full inline copy
        is the redundant one, and scalability wants it gone rather than merely demoted. Treat demotion as a
        stepping stone, not the destination.
- [ ] **[D] `.bib.d/` files are named by BibTeX slug, not by hash.** Slugs are unique within a `.bib`, so
      per-directory uniqueness holds, and a slug is legible in a file manager — the "don't force one placement
      policy" point applied to the case it was made for. Needs light filename sanitization. **Content hashing
      itself survives as dedup**; what was withdrawn (below) is hash-as-filename and hash-as-primary-ID.
- [ ] **[X] ~~Content-hash *document IDs*.~~** Withdrawn 2026-08-01 after being weakened twice, rather than
      carried further. The reorg case is fair (the same reorganization invalidates an rsync backup), and
      **dedup across scopes is solved by tags, not by hashing** — one file with three tags is one document with
      one embedding. Content hashing remains fine as a dedup check for genuinely duplicate files; it is the
      *identity scheme* that died.

### Storage: what a document *is* — settled enough to stop re-litigating

- [ ] **[D] "A document is a file" stands.** Clean and simple in a way the alternatives are not.
- [ ] **[X] ~~Ingest `.bib` as a container: N documents at `foo.bib#entry-key`, with locators into the original
      file.~~** Retracted 2026-08-01 (Juha's objection, accepted). Reasons, recorded so this isn't re-proposed:
  - The DB indexes **chunks** and the RAG tool surface handles **documents**; containers make it three levels.
  - Record-boundary documents sharing one file need offsets and lengths — **a new mechanism**.
  - The user *will* insert an entry in the middle. (A content-hash ID would survive that — identities stable,
        only locators move — but defending the mechanism is not the same as justifying it.)
  - The only win was removing a CLI step, which is not worth the above.
- [ ] **[D] `raven-burstbib` was only ever an import tool**, and a monolithic `.bib` is more common than
      single-entry files. If the items are already loaded, export from *that*, not from the tool.
- [ ] **[D] Fix the burst-step friction without the container mechanism**: burst on ingest into a **visible**
      sibling directory (`foo.bib.d/`), original untouched, shown in the UI, one-click undo. Addresses the
      "too much magic" objection (nothing appears invisibly) at a fraction of the machinery, and keeps a
      document a file. Per-entry sidecars are the other candidate; not obviously better than just favouring
      individual-record `.bib`s as input.
- [ ] **[D] Updating a `.bib` must regenerate `.bib.d/` selectively**, or the whole point is lost to
      reprocessing unchanged records. Mechanism: hash each record's normalized text at burst time, rewrite only
      what changed. Note this hashing is **internal to the burst step** and never leaks downstream, which is
      what keeps it compatible with having retracted the container idea. Two things to get right:
  - **Not rewriting unchanged files is load-bearing, not an optimization.** If the burst touches every file,
        mtimes bump and the entire downstream per-file cache invalidates — the hash comparison would have been
        built and then thrown away one layer down.
  - **Deleted records leave orphans in `.bib.d/`.** The sidecar orphan problem again, including the
        recover-or-delete affordances. Another argument for the unification above.
- [ ] **[N] Test corpus for scopes, once they exist**: the Friendship is Optimal add-on stories, which ship as
      HTML. Genuinely useful beyond the joke — it is mode 2 in the interrogation sketch (a corpus the reader
      already knows, i.e. the evaluation-grade one), "which add-on is which" is a known-item retrieval task, and
      it is **out-of-domain** in three ways the current eval set cannot test: the stopword list is tuned for
      scientific text, `format_entry_for_keyword_extraction` assumes bibliographic fields, and the MiniLM
      reranker is MS MARCO-trained.
- [ ] **[X] ~~Do the ID scheme inside the Nomic window because it's the cheapest moment.~~** Retracted: one 12k
      pile plus a couple of small ones is hours, not weeks, and the cost is the same whenever it happens. The
      only residue is "no cache-migration code needed", which is worth little when discarding and rebuilding is
      cheap anyway.

## 6. Corpus TOC for the model

- [ ] **[P] Use cluster keywords as the scope description.** Visualizer already computes them per scope, so the
      TOC is derived rather than hand-written and stays fresh at the same cadence as the map — the drift metric
      governs both. Scope names alone are much weaker: they say a topic exists but nothing about extent or
      granularity, which is most of what the model needs to decide whether looking is worth a turn.
      **Blocked on scopes + unified DB (§5).**
- [ ] **[P] Size split**: names + document counts always (~10 tokens each; the counts do real work — "3 docs"
      and "12,431 docs" warrant different behaviour), cluster-level detail behind a `describe_scope` tool.
      Makes hierarchical scopes fall out: top-level names always, expand on demand.
- [ ] **[P] Placement**: system prompt, stable across a session, cached once. The opposite of the tool-loadout
      churn problem in §3.
- [ ] **[P] Wording trap**: a TOC invites reasoning about absence ("no scope about X, therefore nothing about
      X"), but a paper about X may sit inside `hydrogen_papers` without being prominent enough to surface as a
      cluster label. Frame it as **what's prominent, not what's present** — one sentence, cheap now, expensive
      later as a confabulated "we have nothing on that".
- [ ] **[P] Autodetect the retrieval mode from the scores, with an optional model-set hint.** The score
      distribution tells you about the corpus; the utterance tells you about the task. "What do these papers say
      about X" and "find me the paper about X" can produce near-identical distributions and want different k —
      autodetect can't see that difference because it isn't in the data. Hint optional and usually absent, not a
      required parameter the model reasons about on every call.

## 7. Vision documents — after the pile is thinned

- **[P] Three audiences, one writing effort.** Write the **team + project manager** version properly; the other
  two are extractions.
  - *Team/PM*: corpus framing, the pillars, workflow walkthrough. Retro-future register in small doses — it
        explains design decisions that otherwise look arbitrary.
  - *Funding*: the gap and the evidence. The identity sketch's strongest line is already there — demonstrated
        feasibility is a different object from asserted feasibility — which turns "not competing with the
        frontier labs" from an apology into the point. Pair with the confidentiality constraint and ECCOMAS.
  - *README*: opens with the plain-language sentence that already exists in `TODO_DEFERRED.md` — *put your
        documents in a folder, get a map of them, then ask questions about the parts you care about*. The
        current one fails for a diagnosable reason: it explains the constellation before the purpose, so a
        reader meets seven components before learning what any of it is for.
  - **Hard rule**: README describes **what ships today**; short "where this is going" section linking to the
        vision doc in-repo. Also serves channel 2 (legibility) directly.
- **[D] The spine is *the corpus* — currently, and as post-hoc justification that happens to fit.** Juha's
  framing, 2026-08-01, and worth keeping in that register rather than promoting it to a founding principle.
  - **[P] So write down what would falsify it**, one sentence, in the doc. If the lab-assistant track grows
        toward instrument status and experiment control, the corpus stops being the unifying noun and the
        unifier becomes something more like *the set of things Aria can address*. That makes the claim testable
        rather than merely fitting.
  - **[D] And the falsifier has arrived early, from an unexpected direction** (Juha, 2026-08-01): home
        automation. *"Aria, could you play something adventurous?"* is still corpus-shaped — find me things in
        a collection. ***"What's playing?"* is not.** That is **state in a controlled system**, and it is
        structurally identical to asking a lab instrument what it is currently doing. Media control and
        instrument control are the same falsification case in different clothes, which makes this cheap
        evidence available *now* rather than after the lab-assistant track matures. The corpus framing covers
        "find me things"; it does not cover "what is the state of the world, and change it."

### Home automation / MCP thread

- **[D] First two MCP use cases: media player control and weather info.** Media control is a good way to
  prototype the shape of the lab assistant with remote desk work — real state, real actions, low stakes when
  it misfires, and it exercises the read-state / take-action split that instruments will need without the
  consequences. Brief 04 (MCP client) is queued right after 09 and the v0.2.8 release.
- **[P] Don't inject the library.** Thousands of tracks is the same problem §6 already solves: names-and-counts
  TOC ("Nordic jazz, 340 tracks") plus a `search_library` tool. Identical to what document search became.
- **[D] The retrieval does not transfer as cleanly as the storage does.** A track's "document text" is a label
  — artist, title, album, maybe genre and year — and embedding that gives almost nothing to match *"adventurous"*
  against, because mood is exactly the attribute metadata lacks. LLM-based enrichment fails precisely where it
  is needed: the model cannot tell you a Nordic jazz artist is restless if it does not know who they are. Real
  options are audio-side models or user tags — which lands back on the tag system, from the same pet peeve
  about music collections that started that thread.
- **[P] But the modality gap is narrower than it first looks, and brief 12 is why.** Whisper already exists in
  the stack, a coworker has floated `raven-transcribe` for podcasts, and a music description model would be a
  welcome addition. **Every one of those is a registered producer under brief 12's key shape** — derivation
  kinds `transcript` and `description`, keyed on `(source hash, kind, producer params)` exactly like extracted
  text. So the vision-doc limit is not *"Raven maps things that carry text"* but the weaker and more accurate
  **"Raven maps things that can be made to carry text"** — with the derivation registry as the mechanism that
  makes them. Podcasts become documents; music becomes documents; the corpus spine survives *that* part.
  What it does not survive is `what's playing?`.

### Provenance means two different things — do not conflate

- **[D]** Raven treats both, and they need different handling:
  - **Disclosure** — the EU AI Act Article 50 sense: this message/artifact was AI-generated. Compliance-shaped,
        applies per message.
  - **Provenance** — the research sense: which source material did this claim come from? This is the one
        researchers care about, it is the standard term in research data management, and it should keep the word.
  - **[P]** Note the collision already in the tree: `07_export-provenance-brief.md` is the *disclosure* one.
        Worth renaming, or at least a line in it saying which sense it means.
  - **[P]** In the vision doc, the **provenance** sense is the one that deserves to be a named pillar — for
        screening it is the value proposition rather than compliance overhead, since a first-pass reviewer whose
        output cannot be traced to specific documents is useless for a review you have to defend to a referee.
- **[P] Lead with the measurements.** The MoE result especially — 2.75× against a predicted 3–9×, with the
  reason named — because a document reporting a prediction coming in *low* reads as measurement rather than
  advocacy, and that carries the credibility of everything else. The VRAM ledger is the other: "all nine
  modules with the avatar running is ~2.9 GiB; the LLM is the constraint" is the single number anyone asking
  about hardware needs.

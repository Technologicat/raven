# Importer Rework Plan

Bundled changes to the import pipeline (`importer.py` / `raven-importer`):

1. **Nomic-embed migration**: Replace snowflake-arctic + mpnet with Nomic-embed-text (unified text embeddings) and, for future image search, a vision encoder. VRAM savings + a unified embedding space.

   **Amended 2026-08-03: the model choice is a fork, not a version bump.** Checked against Nomic's published
   lineup rather than assumed:

   - **`nomic-embed-vision-v1.5` is aligned to `nomic-embed-text-v1.5`**, and that alignment is the whole
     reason to want it: text and image embeddings land in one space, so a text query ranks figures directly,
     and one collection holds both.
   - **`nomic-embed-text-v2-moe` is the multilingual model** — MoE, 475M total / 305M active, ~100 languages,
     trained on 1.6B pairs, Matryoshka-truncatable 768→256. It is a different model, not a newer v1.5.
   - **No v2-aligned vision encoder turned up.** Two searches found the vision encoder consistently paired
     with v1.5 and nothing pairing it with v2; Nomic's separate `nomic-embed-multimodal-3b` is a standalone
     visual-document-retrieval model with its own latent space, not a v2 companion. This is absence of
     evidence from a search rather than a checked changelog, so re-verify before committing — but plan for
     the fork standing.

   **Amended 2026-08-05: the migration carries one measurement, whichever branch is taken.** Brief 09
   settles Librarian's off-corpus detection on an absolute cosine threshold near 0.40 — a number denominated
   in `multi-qa-mpnet-base-cos-v1`'s similarity scale. Any replacement model puts its similarities on a
   different scale, so **swapping the embedder invalidates that constant**, and the failure is silent: a
   threshold in the wrong place does not error, it starts calling answerable questions ungrounded, or stops
   catching off-corpus ones, and nothing in the app reports either. Re-measure before the swap ships.

   The cost is small, which is what the retrieval harness was built for — four indexed corpora (hydrogen,
   arXiv AI, fan fiction, and a titles-only bibliography), one `sharpness.py <corpus>` run each, and the
   comparison table falls out. Method in `investigations/retrieval/README.md`.

   Expect more than a rescaling: mpnet dates from 2021, so a 2024 model may move retrieval quality itself
   rather than only the axis it is measured on. The titles-only corpus is where the current embedder is
   weakest, and so the case most likely to improve.

   *(This trigger is repeated in brief 06, because the two briefs disagree about which lands first — 06 has
   the migration shipping with the Hindsight standup, and this brief has it in the importer rework. Whoever
   gets there first needs to see it, so it is written in both rather than referenced from one.)*

   So the two branches, and they are mutually exclusive as long as that holds:

   - **v1.5** — figures rank against text natively in one collection. English-centric.
   - **v2-moe** — Finnish, Japanese and other non-English material works, which matters because JAMK's own
     context is Finnish and multilingual scientific literature is a corpus property rather than a niche.
     Images lose the direct route and reach text queries through the description pivot instead — which exists
     anyway, since images carry OCR and description channels regardless (see brief 12).

   **Related, and it interacts with item 2:** v2-moe's Matryoshka training truncates vectors 768→256 with
   claimed minimal degradation, i.e. 3× less embedding storage. It does **not** subsume item 2's PCA step,
   though: Matryoshka truncation is fixed at training time, while PCA is *corpus-adaptive* and item 2's stated
   purpose is measuring this corpus's effective dimensionality. They may still compose.
2. **PCA preprocessing**: Reduce embedding dimensionality (e.g. 768 → 50) before UMAP/t-SNE. Measure effective dimensionality of the corpus — if first 50 components capture >95% variance, downstream quality should be nearly identical but faster.
3. **Cosine-to-medoid outlier assignment**: HDBSCAN noise points assigned to the cluster whose medoid has highest cosine similarity, instead of leaving them unassigned.
4. **Procrustes alignment**: When adding new papers to an existing dataset, re-embed the full combined corpus, then use SVD on correspondence points (papers present in both old and new embeddings) to find the optimal rotation matrix R. Apply R to align the new embedding with the old one. Preserves spatial memory while allowing new clusters to appear. Side benefit: novelty detection (new papers far from existing clusters may indicate field-expanding work).

5. **Cluster once, in high-D — the clusters the user sees are currently computed in 2D, and that is a defect.**
   Added 2026-08-05. The pipeline runs HDBSCAN twice, and the run that produces the visible answer is the
   wrong one:

   - `cluster_highdim_semantic_vectors` (importer.py) fits HDBSCAN with `metric="cosine"` on the
     high-dimensional embeddings — but **discards its labels**. They are used only to stratify a sample of
     up to 20 points per cluster, which then trains the dimension reduction. Only `unique_vs` and a cluster
     *count* come back.
   - `cluster_lowdim_data` then fits HDBSCAN again with `metric="euclidean"` on the 2D output, and *those*
     labels become `entry.cluster_id` — what the visualizer colours, labels, and builds word clouds from.

   So cluster membership is decided after the projection, which is exactly where the information needed to
   decide it has been destroyed. t-SNE and UMAP preserve local neighbourhoods and are explicit that
   inter-cluster distances and densities in the output are not faithful; crowding forces genuinely separate
   high-D groups to overlap, and can split one group across the plane. Clustering that output measures the
   projection as much as the corpus. The high-D fit, which does have the geometry to answer the question,
   is already being computed and then thrown away.

   The fix in principle is to keep the high-D labels and let the 2D map be presentation only. Two things
   make it more than a re-wiring, and both need settling before it is scheduled:

   - **The high-D fit currently runs on a sample** (`max_n=10000`, random beyond that, because HDBSCAN
     exhausts 32 GB at ~50k). Labels for the *whole* dataset need either a fit over everything, or an
     assignment step for the unsampled remainder — which is item 3's cosine-to-medoid machinery, so the
     two items compose rather than compete.
   - **The parameters are not transferable.** The high-D fit uses `min_cluster_size=5, min_samples=1`
     chosen to seed the projection; the 2D fit uses `10` and `2` chosen to look right on a map. Promoting
     the high-D labels to visible output means re-tuning against what a reader should see, not against
     what trains t-SNE well.

   Also note this interacts with item 2: PCA to ~50 dimensions before the projection would give a *third*
   space to cluster in, and a middle one is arguably the right home — enough dimensions to keep the
   geometry, few enough for HDBSCAN to scale past the sampling limit. Worth measuring rather than assuming;
   the bibliography below is the instrument, since "is this cluster about one thing" is exactly the
   judgement it supports.

## How to tell whether any of this made things better

Every item above changes where documents land on the map, and none of them has an obvious metric. Cluster
quality is a judgement, so the evaluation instrument has to be **a corpus whose correct answer someone
already knows** — mode 2 in `briefs/design/corpus-interrogation-sketch.md`.

**We have one: the axially-moving-materials bibliography** at `00_stuff/rawdata/banichuk_references.bib`
(gitignored, local — it is research data, not repo content). Juha knows this literature well enough to say
by inspection whether a clustering came out right, which is the property that makes it an instrument rather
than merely data. Recorded 2026-08-03, because it had only ever been mentioned in conversation.

**What kind of verdict to expect from it**, stated so nobody over-reads the results. The bibliography
accumulated over a working career and includes entries added by coauthors — Juha does not recall every
record in it, and some he never chose. So the ground truth is **topical coherence, not per-paper recall**:
the question it answers well is "is this cluster about one thing, and is it the thing the label says",
which a domain reader can judge from titles alone. It cannot answer "is *this specific paper* in the right
place" for the parts of the corpus he did not put there himself.

Its shape, measured rather than assumed, because two of these change what it can test:

- **541 records, spanning 1766–2013.** The long tail matters: an embedding trained on modern text meeting
  18th- and 19th-century mechanics titles is itself worth watching.
- **Only 2 records carry an abstract**, and this is structural rather than an oversight: the database was
  **typed by hand between 2007 and 2016**, partly predating routine online abstracts, and many sources were
  dead-tree. So the abstracts are not sitting somewhere waiting to be re-exported — for most of these
  records they were never in machine-readable form at all.
  - **Enrichment via Crossref was considered and measured, 2026-08-03. It does not pay.** Only 17% of the
    records (93 of 541) carry a DOI at all, and of 20 sampled DOIs exactly **one** has an abstract in
    Crossref. The reason is publisher policy rather than corpus age: the corpus is ~75% Elsevier, which
    deposits no abstracts (0/15 sampled), Springer 0/3, and the single hit was Wiley. Resolving DOIs for
    the remaining 83% via bibliographic query would raise the denominator, not the ~5% yield. Expect a
    dozen abstracts for an afternoon's work, against 541 records — it would not move the corpus out of the
    title-only regime, so it would not even change what the corpus tests. Written down because it is a
    reasonable idea that looks better than it measures.

  Take the title-only regime as the thing under test: `importer.get_highdim_semantic_vectors` embeds
  `title + abstract` when it can and falls back to the bare title otherwise, and that fallback is the path
  this corpus exercises end to end. It is a real and common regime — plenty of hand-built bibliographies
  look like this — but it is *not* the abstract-rich one a Web of Science export gives you. A change that
  helps one may not help the other, so test both and label which is which.
- **Mixed-language author names**, ~4% carrying LaTeX diacritics, concentrated in Nordic, German and Polish
  authors — which is why it also surfaced the author-name decoding bug fixed on 2026-08-03.

Complementary to `investigations/retrieval/`, which evaluates the *retrieval* side against a known corpus
for brief 09. This one evaluates the *map*. Both are mode 2; neither substitutes for the other.

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
- **Only 2 records carry an abstract.** So this corpus exercises the **title-only** regime —
  `importer.get_highdim_semantic_vectors` embeds `title + abstract` when it can and falls back to the bare
  title otherwise. That is a real and common regime (plenty of bibliographies look like this), but it is
  *not* the abstract-rich one a Web of Science export gives you, and a change that helps one may not help
  the other. Test both, and say which.
- **Mixed-language author names**, ~4% carrying LaTeX diacritics, concentrated in Nordic, German and Polish
  authors — which is why it also surfaced the author-name decoding bug fixed on 2026-08-03.

Complementary to `investigations/retrieval/`, which evaluates the *retrieval* side against a known corpus
for brief 09. This one evaluates the *map*. Both are mode 2; neither substitutes for the other.

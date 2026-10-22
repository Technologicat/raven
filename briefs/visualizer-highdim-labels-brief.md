# Brief: the Visualizer's cluster labels from high-D, v1

*Status: decided, not started. Written 2026-10-22 for a one-day build on 2026-10-23, ahead of grouping a
corpus of about 150 records in the Visualizer on 2026-10-26.*

The v1 of brief 11's item 5, *Cluster once, in high-D*. The method is settled by
`investigations/highdim-clustering/README.md`, *What this suggests for the rework*; this brief only scopes
the build. Read that section first — it is short, and every parameter below comes from it.

## What v1 is

The labels the user sees come from the embedding space, and the 2D map becomes presentation only.

- **Fit on centered vectors** (finding 7), **agglomerative, average linkage, cosine, the tree cut at 100**
  — a resolution, not a cluster count.
- **A cluster of one is an outlier.** A small cluster is kept when its cohesion (mean pairwise cosine)
  clears the **10th percentile of every cluster's cohesion, weighted by membership** (finding 10). No size
  knob; the floor of two is structural.
- **Handed downstream in the shape the 2D labels have today**, so keyword extraction, the plot and the
  dataset format are untouched. The seam is in `importer.py`: `_cluster_lowdim_data`'s labels are what get
  replaced.

## Scope guard

Labels from high-D, and nothing else changes. Anything about how a cluster that is scattered across the 2D
map should *look* goes on a list for after Monday rather than into this build (finding 8 says to decide
that presentation deliberately, which is not a same-day decision).

## Open, in the order to settle them

1. **150 records is outside everything measured, and both rules break there.** The smallest corpus in the
   investigation is banichuk at 531, itself unrepresentative. Cut at 100, 150 records leave about 1.5 per
   cluster, so most would be singletons and therefore outliers; and the cohesion floor calibrates against
   the corpus's own clusters, which barely exist at that cut. The cut needs an answer for small corpora.
   **Measure it on real data, and real small corpora exist** (maintainer, 2026-10-22): the education
   corpora under `00_stuff/rawdata/biology_geography_health_education/` — `202508/biology.bib` (163
   records) and `202512/geography.bib` (135), every record with an abstract. Same family and size as
   Monday's, so they are the calibration set, ahead of subsampling the large corpora (which gives a sparse
   sample of a big field rather than a small field).
   **Monday's corpus is the third of the family**, `202506/health_education.bib` (119 records, every one
   with an abstract; maintainer, 2026-10-22 — it may have moved by Monday). **Keep it out of the
   calibration**: settle the rule on biology and geography, then apply it to health education as the
   held-out case, so that the first time it meets that corpus is the honest test.
   **It needs cleaning first** (maintainer: a raw file from someone new to BibTeX). Parsed on 2026-10-22,
   115 of 119 records read, and every one of those has a title and an abstract. The four that do not: two
   parse failures, at least one from a DOI URL used as the citekey (`@article{https://doi.org/…`), and two
   duplicate keys, a placeholder `doi-numero` reused and a DOI-URL key appearing twice, which may also be a
   duplicate record. Try `raven-fixbib` first, then `raven-deduplicate`; anything they cannot repair is a
   case for them rather than for a hand edit.
   **Both have a human-checked grouping, which makes them yardsticks** (maintainer, 2026-10-22):
   - **Geography**: `202512/geography_clusters.txt` is the final human-checked result, which became a
     table in the geography paper.
   - **Biology**: the human-checked result is **Table 6** of the published paper,
     <https://www.mdpi.com/2227-7102/16/2/201>. The `clustered*.txt` files in `202508/` are not known to
     match it, so transcribe the table rather than trusting them, and say where they differ if they do.
     `202508/` holds a manuscript dated 2025-11-13, as `.odt` and `.pdf`, which is probably the paper:
     extract the table from there, and check it against the published Table 6, since a manuscript can
     predate the final edits. The maintainer can supply the final `.odt` or `.pdf` if it differs.

   A human grouping is a reference to compare against, not a target to fit: **there is no correct grouping
   of these records** (maintainer, 2026-10-22), as the investigation found for cluster counts, where
   nothing in the data picks a number. Do not choose a
   number against examples written for the purpose. Whatever rule comes out must give the same answer as cut-at-100 on the corpora where 100 was
   measured, or say why it should not.
2. **Port the cohesion rule from the prototype, not from this summary — and note the prototype has only
   half of it.** `investigations/highdim-clustering/outlier_anatomy.py` computes cohesion and the
   size-split floor (the 10th percentile over clusters of five or more). The membership-weighted floor,
   which is the one to ship, appears in the README only as a comparison table against the size-split one.
   So write the weighted floor new, and check it against that table before trusting it: it should
   reproduce the arXiv AI and AOKK rows (0.516 and 0.510) from the same fits.
3. **The `float16` question** (the README's last open item): whether fitting straight off the cached
   `float16` vectors changes the result. Minutes; upcast if it does.
4. **If time allows: replace the high-D HDBSCAN sampler** (`_cluster_highdim_semantic_vectors`), which
   picks the 2D reduction's training sample, with the agglomerative labels. Try it on the hydrogen corpus
   (21378 records). If it does not land the same day, it stays out: the old sampler still works, and the
   pending change is set aside whole.

## Known safe

- **Nothing reads the 2D clusterer's medoids except the importer itself** (maintainer, 2026-10-22), so a
  high-D cluster scattered in 2D breaks no label placement elsewhere.
- **Cost**: 1 min 44 s and 3.9 GB at 21378 records (the README); a few hundred records is instant.

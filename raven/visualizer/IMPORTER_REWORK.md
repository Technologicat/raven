# Importer Rework Plan

Bundled changes to the import pipeline (`importer.py` / `raven-importer`):

1. **Nomic-embed migration**: Replace snowflake-arctic + mpnet with Nomic-embed-text v1.5 (unified text embeddings) and Nomic-embed-vision v1.5 (future: image search). VRAM savings + unified text+image embedding space.
2. **PCA preprocessing**: Reduce embedding dimensionality (e.g. 768 → 50) before UMAP/t-SNE. Measure effective dimensionality of the corpus — if first 50 components capture >95% variance, downstream quality should be nearly identical but faster.
3. **Cosine-to-medoid outlier assignment**: HDBSCAN noise points assigned to the cluster whose medoid has highest cosine similarity, instead of leaving them unassigned.
4. **Procrustes alignment**: When adding new papers to an existing dataset, re-embed the full combined corpus, then use SVD on correspondence points (papers present in both old and new embeddings) to find the optimal rotation matrix R. Apply R to align the new embedding with the old one. Preserves spatial memory while allowing new clusters to appear. Side benefit: novelty detection (new papers far from existing clusters may indicate field-expanding work).

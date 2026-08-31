# Clustering the map in high-D instead of in 2D

Whether the Visualizer's clusters should be found in the embedding space rather than in the 2D
projection, what that costs, and which algorithm to use once the fit moves there.

Measured 2026-08-31, against four corpora — of which **three count**; see the next section. No Visualizer
code was changed; this is the measurement that says what to change.

The question comes from `briefs/researchers-night/11_visualizer-importer-rework-brief.md` item 5, which
argued on principle that clustering the projection measures the projection. It does, and by how much is
now a number. Two of the same brief's other items — 2 (PCA preprocessing) and 3 (cosine-to-medoid
outlier assignment) — do **not** survive contact with the data, which is the part worth reading before
scheduling either.

## The corpora, and which of them count

Four corpora, but they do not carry equal weight. **AOKK, hydrogen and arXiv AI are representative of
what gets fed into Raven; banichuk is not** (Juha, 2026-08-31) — it is an old hand-built bibliography,
titles only, and pathological rather than merely small.

That is measured, not just asserted. Abstract coverage counted from the `.bib` files, and the geometry
from the embedding caches (1500-point sample for the pairwise figures):

| corpus | n | with abstract | ‖mean vector‖ | pairwise cosine mean | p1 – p99 |
|---|---|---|---|---|---|
| AOKK multisource | 5007 | 83% | 0.701 | 0.488 | 0.304 – 0.868 |
| hydrogen production | 21378 | 100% | 0.668 | 0.449 | 0.277 – 0.643 |
| arXiv AI | 958 | 100% | 0.657 | 0.431 | 0.234 – 0.620 |
| banichuk | 531 | **0.4%** (2 of 541) | **0.900** | **0.810** | 0.626 – 0.927 |

The three representative corpora agree closely on geometry. Banichuk is the outlier on every column:
almost no abstracts, and a mean vector of norm 0.900 — nearly all of every embedding pointing one way,
with all pairwise similarities crushed into a narrow high cone. So its margins between methods are tiny
by construction, and finding 5 shows them landing inside the arithmetic noise. **Read it as a
corroborating case at best, and never as a tie-breaker.**

Brief 11 designates banichuk as the evaluation instrument for the map, and that is not in conflict with
this: it is the instrument because Juha can judge its literature, which is about *ground truth being
available*, a different axis from being representative. The brief already anticipated the split — "test
both and label which is which" — and this section is that labelling.

### A precision trap in the caches, found here

**The importer caches embeddings in whatever dtype the embedding device was configured with**, so dtype
varies per cache with no flag saying so: AOKK's is `float64`, and banichuk's, arXiv AI's and hydrogen's
are all `float16`. Sixteen-bit floats carry about three decimal digits, which is exactly the precision
these similarities are quoted to.

It mattered. Normalizing at the cache's own dtype moved banichuk's k-means gap from +0.011 to **+0.019**,
which flips its ordering against HDBSCAN (+0.017) — a conclusion reversed by rounding. `clusterlab.normalize`
now upcasts to `float64` before doing anything, and every number in this write-up is post-fix. AOKK is
unaffected, being `float64` already, so findings 1–4 and 7 (all AOKK-only) never depended on it.

## What ships today

`importer.py` fits HDBSCAN twice, and the run that produces the visible answer is the wrong one.
`_cluster_highdim_semantic_vectors` fits with `metric="cosine"` on the 1024-dimensional embeddings and
then **discards its labels**, keeping only a stratified sample of points to train t-SNE on.
`_cluster_lowdim_data` fits again with `metric="euclidean"` on the 2D output, and *those* labels become
`entry.cluster_id` — what the app colours, labels and builds word clouds from.

## The metric, and why it needs a control

Cluster quality is a judgement, so the numbers here only narrow the field; the titles decide. The single
number used throughout is the **gap**:

    compactness   mean cosine of a clustered point to its own cluster's mean direction   (higher better)
    nearest       mean cosine between a cluster's mean direction and its nearest other   (lower better)
    gap           compactness - nearest

Positive means clusters are more like themselves than like each other, which is the whole claim a
cluster makes. **Everything is scored in the raw normalized embedding space**, whatever space the fit
ran in, or configurations fitted in different subspaces would not be comparable.

The gap alone is not enough, and reading it alone would have produced the wrong recommendation here.
It moves with both **coverage** (a method that clusters only the dense quarter of a corpus is scored
only on the easy points) and **cluster count** (cutting a corpus into more pieces can only bring the
pieces closer together). `matched_control.py` exists for that reason: it re-runs the competing methods
on *exactly* the subset HDBSCAN clustered, at *exactly* the cluster count HDBSCAN chose, and adds a
random labelling as a floor. Without it, HDBSCAN looked far better than it is — see finding 5.

## Findings

**1. The defect is real, and it is worth about 2.3× in headroom.** Every labelling is scored against a
random labelling matched to its own coverage and cluster count, since floors differ:

| labelling | clusters | coverage | gap | its own random floor | above floor |
|---|---|---|---|---|---|
| HDBSCAN in 2D — *what ships today* | 183 | 74% | −0.143 | −0.248 | +0.105 |
| HDBSCAN in high-D, `mcs=5 ms=1 eom` | 37 | 26% | +0.069 | −0.173 | +0.242 |
| agglomerative, matched coverage and k | 37 | 26% | +0.102 | −0.173 | +0.275 |

The shipped map is not noise — it sits meaningfully above its floor. But its gap is **negative**: its
183 clusters are closer to each other, in the embedding space, than their own members are to their
centres. Inspection shows why — clusters 96 (LLMs in education), 143 (AI in higher education) and 181
(chatbots for learning) are near-duplicates of one another, split across the plane. That is exactly the
failure brief 11 predicted from t-SNE's crowding.

*Caveat on "above floor": the three rows have different coverage and cluster counts, so their floors are
not the same quantity. The column is a sanity check on direction, not a calibrated ratio.*

**2. The high-D clusters are topically coherent.** This is the judgement the numbers cannot make.
`clusters_raw_eom_5_1.txt` holds all 37 with their titles; every one is about one thing — AI in higher
education, self-regulated learning, LLMs in education, human-AI decision-making and XAI, medical
education, inclusive and special education, learning assistants, pedagogical agents, GenAI in language
learning. `clusters_centered_eom_5_1.txt` (78 clusters) is finer-grained and holds up at that
granularity too, down to six-entry clusters on clinical decision support and on social robots in
education.

**3. Brief item 2's premise is false for these corpora.** The brief proposed PCA 768→50 on the
hypothesis that "if first 50 components capture >95% variance, downstream quality should be nearly
identical". Measured: **50 components capture 53%**, 100 capture 67%, and even 300 reach only 90%. The
corpus is not low-dimensional in the variance sense.

More to the point, PCA makes the clustering *worse* where it counts. It lowers the noise fraction only
by splitting, and the resulting clusters are less separated in the real space:

| fit space | clusters | noise | gap |
|---|---|---|---|
| raw 1024-D | 37 | 74.4% | **+0.069** |
| PCA-100 | 92 | 63.8% | +0.029 |
| PCA-50 | 105 | 60.9% | +0.011 |
| PCA-10 | 133 | 54.5% | **−0.037** |

So brief 11's speculation that a middle space is "arguably the right home" does not hold here. PCA may
still be worth having as a *speed* measure for the t-SNE/UMAP step, which is a separate question this
did not test.

**4. Brief item 3, as stated, destroys what moving to high-D buys.** Assigning every HDBSCAN noise point
to its nearest medoid by cosine takes coverage to 100% and the gap from **+0.069 to −0.147** — back to
the shipped 2D map's quality. The reason is visible in the assignment itself: the median outlier sits at
**0.64** similarity to its winning medoid, against ~0.90 within clusters. Those points genuinely do not
belong to any cluster. If outlier assignment is wanted, it needs a similarity floor, and the floor is
doing all the work.

**5. k-means is consistently last. Agglomerative leads on the headline number, but that lead is
partly an artifact and survives on only one corpus.** At matched coverage and matched cluster count:

| corpus | n | k | coverage | | HDBSCAN | k-means | agglomerative | random |
|---|---|---|---|---|---|---|---|---|
| AOKK multisource | 5007 | 37 | 26% | gap | +0.069 | +0.034 | **+0.102** | −0.173 |
| | | | | size-weighted | +0.113 | −0.038 | **+0.120** | −0.173 |
| | | | | drop clusters <5 | +0.069 | +0.031 | **+0.101** | −0.173 |
| hydrogen production | 21378 | 536 | 21% | gap | +0.033 | +0.026 | **+0.076** | −0.194 |
| | | | | size-weighted | +0.027 | +0.012 | **+0.036** | −0.201 |
| | | | | drop clusters <5 | +0.033 | +0.020 | **+0.042** | −0.203 |
| arXiv AI | 958 | 20 | 19% | gap | +0.031 | +0.019 | **+0.067** | −0.199 |
| | | | | size-weighted | +0.018 | +0.005 | **+0.023** | −0.203 |
| | | | | drop clusters <5 | **+0.031** | +0.020 | +0.022 | −0.199 |
| *banichuk — not representative* | 531 | 24 | 51% | gap | *+0.017* | *+0.019* | *+0.025* | *−0.065* |

**The plain gap has a weighting hole, and agglomerative is the method that exploits it.** Compactness
averages over *points*; separation averages over *clusters*. So a two-member cluster gets the same vote
on separation as an 800-member one, and a singleton gets the best deal available — its compactness is
exactly 1.0 against its own mean, and alone in a sparse region it is far from every other centre.
Average linkage chains, so it leaves exactly that tail. Sliver counts (clusters of ≤2 members), where
HDBSCAN produced **none on any corpus**:

| corpus | k | agglomerative slivers | k left after dropping <5 |
|---|---|---|---|
| AOKK | 37 | 7 (3 singletons) | 27 |
| hydrogen | 536 | **140** | 330 |
| arXiv AI | 20 | 3 | 14 |

Two checks close the hole — weighting separation by cluster size, and discarding clusters under 5
members before scoring. Under them:

- **AOKK**: agglomerative's lead holds under both (+0.120 vs +0.113; +0.101 vs +0.069). Real.
- **hydrogen**: holds, but narrowly (+0.036 vs +0.027; +0.042 vs +0.033) and at the cost of 140 slivers.
- **arXiv AI**: collapses. +0.067 becomes +0.023 size-weighted — a 0.005 lead — and **+0.022 against
  HDBSCAN's +0.031** once slivers go, so HDBSCAN wins. The apparent factor-of-two was the slivers.

So the honest reading: **k-means is last under every check on every corpus**, which is robust and is the
part that speaks to Clust-Splitter. **Agglomerative leads on two of three representative corpora under
the robust checks and loses the third**, by margins a quarter the size of the headline number. It is the
better bet, not a settled answer — and its chaining is a defect in its own right, since a quarter of its
hydrogen output was clusters of one or two papers.

An earlier version of this write-up claimed "three for three, ahead by roughly a factor of two". That
was the unweighted number believed too readily, and it is precisely the artifact the `≤2` column exists
to expose. In the *unmatched* comparison
(`method_comparison.tsv`) agglomerative looks poor and HDBSCAN looks dominant — because there
agglomerative is made to cover 100% of the corpus while HDBSCAN answers for a quarter of it. Most of
HDBSCAN's apparent advantage was coverage, not partition quality.

The two larger corpora are run with `leaf` selection here, not `eom`, because `eom` **collapses** on both
(finding 6) and a two-cluster split of 21378 records — 21367 in one, 6 in the other — tests nothing about
map quality whatever gap it posts. That is worth stating plainly rather than leaving in the table: an
earlier version of this write-up carried the `eom` numbers for hydrogen (HDBSCAN +0.101, agglomerative
+0.296) and they were an artifact of a fixture that could not discriminate.

**k-means standing in for MSSC is the relevant reading for Clust-Splitter** (brief 13). A real MSSC
solver would find a better optimum of the k-means objective, so the k-means row is a floor on the
centroid model rather than its ceiling — but it is last on all three representative corpora and under
every one of the three checks, which is the most robust result in this table. It is also the only method
that goes *below the random floor's neighbourhood* on any check: −0.038 size-weighted on AOKK, where the
random labelling scores −0.173 and both rivals are positive. That is evidence against the centroid model
fitting literature embeddings, and it is worth having before anyone f2py-wraps 9k lines of Fortran. On
banichuk it edges ahead of HDBSCAN by 0.002, which is the noise rather than a result.

The caveat that keeps this from being decisive: Lloyd's algorithm is a weak solver, and "the objective is
wrong for this data" and "this solver finds bad optima of it" predict the same table. Distinguishing them
needs the real solver, which is the wrapping job itself.

**6. HDBSCAN's cluster count is unstable across corpora at fixed hyperparameters, to the point of
collapsing.** The same `mcs=5, ms=1, eom` gives:

| corpus | n | clusters | coverage |
|---|---|---|---|
| AOKK multisource | 5007 | 37 | 26% |
| arXiv AI | 958 | **2** | 87% |
| hydrogen production | 21378 | **2** | 100% |
| *banichuk — not representative* | 531 | *24* | *51%* |

**On two of the three representative corpora, `eom` collapses**, and hydrogen's two clusters hold 21367
records and 6. That is not a map. For a tool that imports whatever bibliography a user hands it, this is
a serious drawback independent of partition quality: there is no setting that is right for the next
corpus, and the failure is silent — the importer would log "2 clusters detected" and carry on to build a
word cloud out of it.

Note which way discounting banichuk cuts here. Banichuk is one of only two corpora where `eom` behaves,
so leaning on it would have made HDBSCAN look considerably more dependable than it is. Agglomerative cut
at a target count or a distance threshold is predictable in a way this is not. `sweep_hydrogen.tsv` has
the `eom`/`leaf` comparison at that size.

**7. The embeddings are strongly anisotropic, and centering helps.** The corpus mean vector has norm
**0.70** — most of every embedding points in one shared direction — which compresses pairwise cosines
into a narrow band (mean 0.49, sd 0.10). Removing the mean direction widens that band (mean 0.00,
sd 0.16). Fitting on centered vectors moves AOKK from 37 clusters / 74.4% noise to **78 clusters /
68.5% noise**, at a small separation cost (gap +0.069 → +0.060), and the extra clusters are coherent
(finding 2). *The link between the anisotropy measurement and the clustering improvement is inference,
not something measured — both facts are solid, the arrow between them is not tested.*

**8. No method reaches high coverage and positive separation together.** Across HDBSCAN, k-means,
agglomerative at fixed k, and agglomerative cut by distance threshold, the gap goes negative once
coverage passes roughly 50%. This looks like a property of the corpora rather than of any algorithm:
literature embeddings are a dense continuum with a few dense knots in it, and a labelling that assigns
every paper to a cluster is drawing boundaries through the continuum. **The honest map has outliers.**
That is a product decision as much as a technical one, since the current map shows 74% coverage and the
high-D fit would show less.

## What this suggests for the rework

Not yet decided — this is the measurement, and the shape of the change is Juha's call. What the numbers
support:

- **Move the authoritative fit to high-D and make the 2D map presentation only.** Finding 1. The high-D
  fit is already being computed and thrown away.
- **Fit on centered vectors.** Finding 7. Cheap, and it buys granularity.
- **Try agglomerative average-linkage as the fitting method**, with undersized clusters kept as
  outliers — but as a candidate to evaluate, not as a settled answer. Findings 5 and 6. It leads on two
  of three representative corpora under the robust checks and loses the third, and its chaining produces
  a tail of one- and two-paper clusters that the min-size filter exists to absorb. The strongest part of
  the case against HDBSCAN is not the gap at all, it is finding 6: a method that silently returns two
  clusters for 21378 papers cannot be pointed at an arbitrary user bibliography.
  - **The decision this actually needs is a reader's verdict, not another metric.** The gap has now been
    wrong three times in this investigation, each time in a way that looked like a result. arXiv AI is
    the corpus to settle it on, being both representative and one Juha can judge.
- **Do not adopt brief item 3 unconditionally**, and treat brief item 2 as a speed measure rather than
  a quality one. Findings 3 and 4.
- **Expect fewer clusters and more outliers than the current map shows**, and decide deliberately how to
  present that. Finding 8.

## Open, and deliberately not answered here

- **The 32 GB constraint.** `maia` has 32 GB and must keep working. HDBSCAN with `metric="cosine"` and
  agglomerative with `metric="cosine"` both materialize a full pairwise distance matrix — one such
  matrix is 3.7 GB at 21378 records and would be ~20 GB at 50k, which is what forces `importer.py`'s
  current `max_n=10000` sample. **Measured**: `matched_control.py` on the 21378-record hydrogen corpus
  peaked at **9.1 GB RSS** in 2 min 51 s, but that run holds several such matrices at once (it fits
  HDBSCAN, k-means and agglomerative in one process), so it is an upper bound on any single method
  rather than a per-method figure. The n² scaling is the part that matters, and it says 50k does not
  fit in 32 GB by this route.
  - Since the cached vectors are L2-normalized, euclidean and cosine are monotonically related and give
    an identical HDBSCAN hierarchy under `leaf` selection, which may open a tree-based path — **untested,
    and the equivalence argument does not extend to `eom`**, whose stability sums are not invariant under
    a monotone change of metric. Note also that a `ball_tree` in 1024 dimensions degenerates toward brute
    force, so the saving would be in memory rather than in time.
- **A domain reader's verdict — the one thing that would actually settle the algorithm choice.** Brief 11
  designates banichuk for this job because Juha can judge that literature, but banichuk is the corpus
  this write-up otherwise discounts, so the two roles pull apart. **arXiv AI resolves it** (Juha,
  2026-08-31: "somewhat familiar with it"): it is representative *and* readable, which banichuk is not
  and AOKK is not. `arxiv_hdbscan_centered_leaf.txt` and `arxiv_agglomerative_centered_k100.txt` are
  written for exactly this, and the question to put to them is not "are these clusters tight" — the
  numbers already answer that, and disagree with each other — but **"which of these two would I rather
  navigate?"**
- **Fulltext.** The arXiv corpus is the only one with fulltext PDFs on disk
  (`~/.config/raven/librarian/documents_arxiv_fulltext`, noted by Juha 2026-08-31). Everything here
  embeds `title + abstract`, which is what the importer does. Whether clustering on fulltext-derived
  embeddings changes any of this is untested and is a separate question from the one asked here — but
  arXiv AI is now the corpus that could answer it, since it is the only one holding both.
- **Clust-Splitter itself.** Finding 5 tests the *model* via k-means, not the solver. Wrapping it stays
  a separate job (brief 13).
- **Whether the cached `float16` costs anything in the importer itself.** It cost a reversed ordering
  *here* before the upcast, and the importer clusters straight off those same arrays. Whether HDBSCAN's
  own fit is sensitive to it was not tested — sklearn may upcast internally — but it is cheap to check
  and worth checking before the rework, since the answer would apply to the shipped pipeline rather
  than only to this apparatus.

## Scripts

| script | what it answers |
|---|---|
| `clusterlab.py` | shared loading, normalization, centering, PCA and the scoring metrics. Not a script |
| `sweep.py` | which HDBSCAN hyperparameters give what, in a chosen space. `--center`, `--pca N`, `--metric` |
| `show_clusters.py` | one configuration's clusters with their titles, nearest-the-centre first, for judging coherence. `--assign-outliers` tests brief item 3 |
| `compare_methods.py` | HDBSCAN vs k-means vs agglomerative vs the shipped 2D labelling, on one yardstick. **Unmatched — read with `matched_control.py`, not alone** |
| `matched_control.py` | the negative control: the same comparison at matched coverage and matched cluster count, against a random floor. This is the one that settled finding 5 |

## Data

The two AOKK title dumps are **gitignored on purpose** — they list records from a corpus that lives
under `00_stuff/`, which is gitignored research data rather than repo content, and this repository is
public. The arXiv dumps are committed, those titles being public on arxiv.org already. Everything else
here is aggregate numbers. Regenerate the missing two with the commands under "Reproducing".

| file | what it holds |
|---|---|
| `sweep_cosine_full.tsv` | the full HDBSCAN grid in raw 1024-D (AOKK) |
| `sweep_centered.tsv` | the same grid on centered vectors |
| `sweep_pca{10,20,50,100}.tsv` | the same grid in PCA subspaces |
| `sweep_hydrogen.tsv` | `eom` against `leaf` at 21378 records, where `eom` collapses to two clusters |
| `method_comparison.tsv` | the unmatched cross-method table |
| `clusters_raw_eom_5_1.txt` | AOKK, 37 clusters with titles, raw space. **Local, not committed** |
| `clusters_centered_eom_5_1.txt` | AOKK, 78 clusters with titles, centered space. **Local, not committed** |
| `arxiv_hdbscan_centered_leaf.txt` | arXiv AI, HDBSCAN: 40 clusters, 40% coverage, gap +0.007 |
| `arxiv_agglomerative_centered_k100.txt` | arXiv AI, agglomerative cut at 100 then filtered: 61 clusters, **90% coverage**, gap −0.067 |

**The last two are the pair to read against each other**, and they are the reason the algorithm choice
is being left to a reader. They are the two ends of finding 8's trade — HDBSCAN answering for 40% of the
corpus with a (barely) positive gap, agglomerative answering for 90% of it with a negative one — and the
metric prefers the first while the second is plainly the better map: 61 balanced clusters (median 12,
max 52) on LLM alignment surveys, diffusion models, chain-of-thought, RLHF, jailbreaking, RL for
reasoning, and the AI-limits critique literature, each recognisably one topic.

That is worth stating as a limit on everything above: **the gap penalises coverage structurally**, so it
ranks a cautious labelling over a useful one, and at 90% coverage it stops tracking what a reader means
by a good map. It was the right instrument for the question this investigation opened with — *is 2D
clustering worse than high-D clustering, all else equal* — and it is the wrong one for choosing an
operating point.

## Reproducing

The scripts read the importer's own embedding caches, so nothing is re-embedded and every configuration
sees byte-identical input. Corpora used (all gitignored local research data):

```bash
V=00_stuff/rawdata/AOKK/multisource/tekoalyagentti_tutkimus_deduped_embeddings_cache.npz
D=00_stuff/datasets/AOKK/multisource.pickle

python investigations/highdim-clustering/sweep.py --vectors $V
python investigations/highdim-clustering/matched_control.py --vectors $V --dataset $D
python investigations/highdim-clustering/show_clusters.py --vectors $V --dataset $D --center
```

The other corpora are `00_stuff/rawdata/banichuk_references_embeddings_cache.npz`,
`00_stuff/rawdata/ai_papers_202510_embeddings_cache.npz`, and the hydrogen set, whose five per-file
caches under `00_stuff/rawdata/100000_most_relevant_refs_of_hydrogen_productionzip/` concatenate to
21378 records.

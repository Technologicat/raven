# Brief: common and distinctive keyword pools, and a dataset keyword dialog

**Filed 2026-09-01.** Cluster keywords currently answer one question where a reader has two, and the
corpus-level keyword data the importer already saves is not shown anywhere. These are the same problem
seen from two ends, and the mechanism below covers both.

Not sprint-scheduled. Pull it into `researchers-night/` if the map's labels turn out to matter for the
exhibit.

## The problem

**A cluster's keywords are asked to do two jobs, and the shared ones do only the first.** A reader wants
to know *what is this cluster about* and *how does it differ from its neighbours*. Measured over the 61
agglomerative clusters of the arXiv AI corpus (2026-08-31): `Large Language Models` appears in **34** of
them. It answers the first question every time and the second never. The next most repeated keyword
appears 5 times, so the distribution is close to bimodal — a handful of corpus-wide terms, and a long
tail of terms belonging to one cluster each.

Both halves are signal. That every cluster is about language models is worth knowing once; it is not
worth repeating 34 times in the place where a reader is trying to tell two clusters apart (Juha).

**Raven already has both instruments, mis-framed as alternatives.** `clusters_keyword_method` chooses
between them, and they do not answer the same question:

- `"frequencies"` → `nlptools.suggest_keywords(per_document_frequencies=…, corpus_frequencies=all_keywords, …)`.
  This is TF-IDF-shaped: it scores a cluster's terms against the corpus background, so it already
  **discriminates**.
- `"llm"` → each cluster is described in isolation. It captures **identity** well and structurally
  cannot discriminate, because the model never sees the other clusters.

## The mechanism

**The split is postprocessing over data already in every dataset file.** Nothing needs to be recomputed
at import time, and the second half of this brief needs no importer change at all.

The quantity that separates the two pools is a keyword's **cluster document frequency** (CDF): in how
many clusters does it appear? It is a count over `vis_keywords_by_cluster`, which is saved.

```
common pool      = {k : CDF(k) >= threshold_fraction * n_clusters}
distinctive(c)   = [k for k in vis_keywords_by_cluster[c] if k not in common pool]
```

Three consequences worth having:

- **No dataset format change**, so every dataset already on disk gains the feature on load.
- **No LLM cost.** Discrimination over LLM-produced keywords needs no second pass; it is counting.
- **The threshold becomes a live GUI setting** rather than something baked in at import. Changing it
  re-partitions instantly, which is the right affordance for a number nobody can pick correctly in
  advance.

**Clusters keep their full keyword list.** A common keyword still describes the cluster truthfully — it
is a fact about the data, not noise — so it is retained and de-prioritized rather than removed (Juha).
The ordering is what changes: distinctive first.

### Where each pool is shown

| surface | shows | why |
|---|---|---|
| cluster annotation / tooltip / info panel | distinctive keywords first, ~6 of them | limited space; this is the *label*, and the reader is here to tell clusters apart |
| the new dataset dialog | the common pool, framed as *what this dataset is about* | said once, at corpus level, instead of 34 times |
| the same dialog | the full `all_keywords` table with counts, copyable | this is the *inspection* view, where "everything, with numbers" belongs |

### Raise the extraction count, then split

Six keywords is a label budget, not an inspection budget, and once the common ones are filtered out a
cluster can be left with very few. So extract **10–12** and let the split decide what is shown.

**This needs measuring before it is built** — see the open question below. Human keyword sets run to
5–6, so a request for twelve may be out of distribution for the model and may be answered with padding
rather than with six more real keywords.

### The dialog

Opened from a toolbutton. Two sections, in this order:

1. **What this dataset is about** — the common pool, plus the head of `all_keywords`. Framed as a
   description of the corpus rather than presented as keywords, because the head of that list is a
   domain stopword list in disguise: AOKK's top five are `student` (9025), `learning` (8899),
   `education` (5342), `learn` (4282), `model` (4052). That *is* the signal — it says what the corpus
   is — but a reader who is shown it under the heading "keywords" will read it as a failure.
2. **The full table**, keyword and count, sorted descending, in a clipboard-copyable form. TSV pastes
   into a spreadsheet, which is what a researcher will do with it.

**`all_keywords` is already saved and read by nothing.** It is `{lemma: count}` over the whole corpus —
8918 entries for AOKK — written at `importer.py:1467`, and `grep` finds no consumer anywhere in
`raven/visualizer/` outside the importer. The importer prints the first 20 of it to the log and that is
the only place it has ever been seen. So this section is pure GUI work over an existing field.

**It does not duplicate the word cloud.** That is built from `entry.keywords` of the *currently
selected* points (`word_cloud.py:143`), so it is selection-scoped, visual and approximate. This is
corpus-scoped, numeric and exact. Different questions.

## Open

- **Does the model produce twelve meaningful keywords, or six and six of padding?** Cheap to measure:
  run one corpus at 6 and at 12 and compare the tails. Do this before raising the number.
- **What threshold?** The arXiv distribution (34, then 5) is bimodal enough that anything from ~10% to
  ~50% of clusters separates it cleanly, so the value is not critical *there*. A topically narrow corpus
  like AOKK will put much more into the common pool, which is correct behaviour and worth looking at
  before fixing a default.
- **Does the split apply to `"frequencies"` mode too?** That method already discriminates, so the
  per-cluster half is largely redundant for it. The dialog applies either way.
- **Should topic keywords be filtered to nouns and proper nouns?** `learning` (NOUN, lemma `learning`)
  and `learn` (VERB, lemma `learn`) are counted separately, which is spaCy behaving correctly — a
  deverbal noun is its own lexeme, and merging them would lose the distinction between "machine
  learning" and "students learn". Verified 2026-09-01 on en_core_web_sm 3.8.0. The verb sense is close
  to a stopword here, so a POS filter would sharpen the list; it would also drop useful adjectives, so
  it wants trying rather than assuming.

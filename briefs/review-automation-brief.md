# Automating a literature review: what the two current studies need next

*Status: collecting (2026-10-29). Item 10 has a date (a prototype early in the week of 2026-11-02); the rest is unscheduled, and the order below is a first proposal.*

Two studies are using Raven's paper tools end to end, and between them they have turned up the next layer
of work:

- **AOKK** — AI agents in higher education: a multi-database search of a few thousand records, screened with
  `raven-scopebib`, with a team screening by hand in parallel.
- **Health education** — sustainability education in health education, the third study in a series after
  biology and geography: a Scopus search of about a hundred records, screened with `raven-scopebib` and by
  hand on 2026-10-28/29, with the final set due early in the week of 2026-11-02.

The detail of each item lives in the entry named, in `TODO.md` or `TODO_DEFERRED.md`; this brief says how
they fit together and in what order. Settled decisions that left no diff go in the log at the end.

## Screening

1. **Carry a person's decisions from one run to the next.** Done by hand twice on 2026-10-29: a decisions
   file keyed by record, the next run's check-by-hand list pre-filled from it, and a list of the decided
   records the new run settled the other way — which `finish` cannot reach, since they never reach the
   worklist. The same thing is what the AOKK team needs when their hand labels arrive: a new list with
   their decisions already filled in, so nothing is redone. Not filed yet; this brief is its first entry.
2. **`finish` overturning a removal** — `TODO_DEFERRED.md`, *`raven-scopebib finish` cannot overturn a
   removal*. The other half of item 1: a decision against a run's outright keep or removal, applied by the
   tool and recorded in the hand audit, rather than edited into the final bib by hand.
3. **Preparing a definition, and drafting one from notes** — `TODO_DEFERRED.md`, *Preparing a
   `raven-scopebib` definition file, as a manual section*, which also carries the `raven-scopebib draft`
   idea. Both definitions so far were drafted by an agent from the researchers' observations.
4. **Stratified samples for replicates** — `TODO_DEFERRED.md`, *Stratified samples for `raven-scopebib`
   replicates*. For the test-retest figure a methods section reports; gated on a full run's verdicts, which
   AOKK's full run supplies.

## Preparing the corpus

5. **More abstract sources than Crossref** — `TODO.md`, *`raven-fillbib` v2*. PubMed is named there for the
   health-education corpus in particular.
6. **`raven-deduplicate --judge` onto the shared sweep** — `TODO_DEFERRED.md`, of that name.
7. **Words broken mid-line** — `TODO_DEFERRED.md`, *Dehyphenate a word broken mid-line…*. Two cases in the
   health-education corpus so far.

## After screening

8. **Per-study attributes from the fulltexts**: school levels and countries, the table the two earlier
   studies built by hand — `TODO.md`, *Per-study attributes from the fulltext*. The health-education set,
   counted by hand as well, is its calibration test. The same pass can flag a screening decision that the
   fulltext disagrees with.
9. **Per-cluster cohesion in the Visualizer** — `TODO_DEFERRED.md`, under *Visualizer: quality flags…*. For
   reading the accepted set's map: on the health-education corpus the 2D map misrepresented which cluster
   was the real one.

## Analysis

These two are the maintainer's (2026-10-29), and the first comes first in the whole brief: a prototype is
wanted early in the week of 2026-11-02, for the health-education study.

10. **Clustering the accepted set by the research questions, with an LLM.** A small accepted set does not
    cluster by embedding: the health-education run d's 40 records gave no cluster at all, in high
    dimensions or in 2D. But 40 titles and abstracts are 9,702 words, 12,171 tokens to the served model, which fits
    in one prompt with room to spare. So: give the model the study's research questions and every abstract
    at once, and ask it to group the studies in the light of those questions, naming each group and saying
    why each study is in it. Run it several times, to see how stable the grouping is. For this study it
    would replace the HDBSCAN clusters.
    - **The two earlier studies are its test set, and their sizes fit too**: geography's bibliography is 133
      records, 35,088 tokens of titles and abstracts; biology's is 161, 49,884 (counted 2026-10-29 with
      the served model's tokenizer). Both have a cluster table made by hand — geography's in the study's
      materials, biology's in its published paper — so the grouping can be compared with people before it
      is used on health education. Compare on the set each table covers: the geography paper screened to
      127, against the bibliography's 133.
    - **How far it scales is open**: at about 300 tokens a record the context holds roughly 400, but how
      well the model reads a prompt of 100k tokens and more, as against how much fits, is unmeasured, and
      its thinking grows with the set.
    - **To settle in the prototype**: how the runs are compared, since a clustering has no fixed labels to
      match — an agreement measure that ignores which name a group got (the adjusted Rand index is the
      usual one), plus a read of where the runs differ; and what is shown in the paper — the groups of one
      run, or what is stable across them.
11. **Inductive and deductive content analysis, agentified**, after Elo & Kyngäs, *The qualitative content
    analysis process* (Journal of Advanced Nursing, 2008).
    - **The test set exists**: the coauthor did this by hand for the biology and geography studies, and both
      published papers record the method and the results (archived in
      `00_stuff/rawdata/biology_geography_health_education/published/`). Run it on those two first,
      compare, and only then on the health-education set.
    - **It reads fulltexts**, one at a time, which ties it to item 8's fetching. The maintainer expects at
      least a representative set from the coauthor in the week of 2026-11-02.
    - **Related reading, beside the Elo & Kyngäs PDF**: Elo et al. (2014) on the trustworthiness of
      qualitative content analysis, and Borse et al. (2025, arXiv 2508.14764) on inter-rater reliability
      between LLMs and human raters in qualitative analysis — the question this item puts to the test.

## Decision log

- **2026-10-29**: this brief created, to hold the review-automation items of both studies as one set.
- **2026-10-29**: items 10 and 11 added by the maintainer; item 10 is the nearest deadline, a prototype
  early in the week of 2026-11-02, and replaces the HDBSCAN clusters for the health-education study.

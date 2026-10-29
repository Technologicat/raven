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
    - **The output side is two settings, not a limit**: the backend caps a reply at 16k tokens per round
      today, and the cap can be raised; the chat template's `low` effort switch asks for less thinking
      (maintainer, 2026-10-29). A grouping of 161 records with a reason each, thinking included, is the case
      to watch.
    - **The test sets have to be recovered first.** No file on disk is the set a cluster table covers.
      Biology's paper reviewed 131 studies (165 after deduplication, then 34 rejected by hand); the newest
      bibliography, `202508/biology.bib`, has 163 entries, 2 of them unreadable, and 9 more records were
      split out of it in December 2025 into `biology_missing.bib` and `biology_broken_entries.bib`, some
      still unreadable. Geography's table covers 127 of its bibliography's 133. `raven-checkrefs` against
      each paper's own list of the studies, over the union of the files, names the covered set and flags
      what the bibliographies lack; `raven-fixbib` repairs what it can of the unreadable records first.
      - **The lists to check against**: each paper lists its studies apart from its references (maintainer,
        2026-10-29). In the biology paper that is *Appendix A. Clustered Studies (\* Not Reviewed in the
        Content Analysis)*, which gives the clustered set and, by its asterisks, the reviewed 131 in one
        place. The geography paper has no such list (the maintainer checked the published PDF, 2026-10-29),
        so its set is recovered from *Table 5*, pp. 22–24: each cluster's studies as author–year citations,
        beside the item topics with their counts and the cluster's number and top-level topic. The text
        layer runs the table's columns together, so read it from page images instead (`pdftoppm`, pp.
        22–24), by a vision model, which keeps the layout (maintainer, 2026-10-29); the model served now
        reads images. Match each citation to its entry in the reference list by surname and year, then
        run `raven-checkrefs` on those entries.
      - **How the table counts**: a cluster's *Count* is its total over all its rows, and an *Outliers*
        row is studies the clustering put in that cluster and the authors excluded from it by hand
        (maintainer, 2026-10-29: #1's 16 is 11 under *Integration of sustainability* plus 5 outliers). So
        the checks are that *Count* sums to 127 and the outlier rows to the paper's 13. For item 10, an
        outlier is a study whose place people disputed, not one that belongs nowhere; score it apart.
      - **A vision model reads it**: p. 22 transcribed correctly on the first try (2026-10-29), 33
        citations against a reading by eye, printed punctuation kept and a row cut at the page foot
        marked as continuing; 28 s a page. Two checks come free: the paper states "a total of 127 clustered studies", and the table
        gives a count per cluster. The table is also item 10's ground truth, study by study.
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

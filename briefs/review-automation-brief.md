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

1. **Carry a person's decisions from one run to the next** — done 2026-10-29: `--decisions` on `run` and
   `report` fills the new worklist from any sheet of decisions, a decision the new run contradicts first and
   highlighted, and `--set-aside-finished` keeps the finished run's files. The AOKK team's hand labels can
   arrive this way, as a sheet keyed by record.
2. **`finish` overturning the run** — done 2026-10-29: a row added to the marked sheet, a key and `keep` or
   `drop`, overrides an outright removal or keep, is listed in the output and recorded in the hand audit.
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
        marked as continuing; 28 s a page. **All three pages done the same day**, kept with the paper in
        `published/geography-table5/` (gitignored): every count check passes — rows against their *n*,
        clusters against *Count*, 127 in all, 13 outliers. One page returned each row's citations as one
        `; `-joined string, which the assembler splits.
      - **One study sits in two clusters, on purpose**: Boehm et al. (2024) is in both #0 and #6, so the
        table's 127 places cover 126 distinct studies. The authors placed it twice because it belongs under
        both themes (maintainer, 2026-10-29); the paper's text does not say so. A ground truth that allows
        a study two places means item 10's comparison must allow it too: either placement agrees with
        people, and so does both.
      - **Biology's set is recovered** (2026-10-29): `published/biology_bibliometric.bib`, Table 6's 165
        studies (167 places; Hamman 2018 and Faizah et al. 2024 are placed twice, as the paper's text says),
        and `published/biology_content_analysis.bib`, the 131 the appendix does not asterisk — the asterisk
        marks only studies left out of the content analysis, all 165 being in the bibliometric one
        (maintainer). The appendix lists one study more, Sterling (2013), which is in no cluster and has no
        record: likely a slip in the appendix (maintainer, 2026-11-02). Table 6 dates Baena-Morales et al. 2023, the publisher 2024. The vision model misread six
        of Table 6's 165 names, which the text layer corrected: trust the image for structure, the text for
        spelling.
      - **Geography's set is recovered** (2026-10-29): `published/geography_bibliometric.bib`, Table 5's 126
        studies. The paper does list its studies apart, as the first 127 entries of its reference list,
        set off from the rest only by a line of dots on p. 57 (confirmed by the maintainer); a separator
        that easy to miss is why it was taken for one list. `raven-checkrefs` paired 126 of the 127 with
        records, three of them proposals well clear of their runners-up. The 127th, Yli-Panula et al.
        (2020), is in no cluster and has no record — the same shape as biology's Sterling (2013), and
        likely a slip too (maintainer, 2026-11-02). The content analysis reviewed 114: every clustered place but the 13
        outliers, rejected by hand (the paper, p. 25). So `published/geography_content_analysis.bib` holds
        113 studies, Boehm et al. being two of the 114 places.
      - **And one slip**: the table cites Boehn & Hamann (2016), where the reference list has 2011, and
        2011 is right — the article's own running heads say *European Journal of Geography 1 (2011)*
        (checked by the maintainer, 2026-10-29). The 2016 comes from the bibliography's record, so the table
        was written from the bib. A surname-and-year match misses it; `raven-checkrefs` matches on titles
        too. Two checks come free: the paper states "a total of 127 clustered studies", and the table
        gives a count per cluster. The table is also item 10's ground truth, study by study.
    - **What it produces: a grouping shaped like Table 5** — themes, sub-themes within them, a reason per
      study, a study allowed under two themes. Coding each study per research question is the other
      reading of "by the research questions", and grows into item 11 instead (decided 2026-10-29).
    - **Which level of the tables is the human one**: the sub-themes exist because the Visualizer's
      clusters turned out to hold several themes each (maintainer, 2026-10-29). Every name in the tables
      is human, at both levels, the Visualizer's keywords serving only as a sanity check — frequency-based
      for biology, which predates the LLM keyword mode and found them of little use, and LLM-made for
      geography, which found them good and named the topics by hand anyway; what came from
      the algorithm is the top level's *membership*, which studies were grouped together. The sub-themes
      are a valid concept for classification in their own right. So compare at both levels, and read
      where they differ: a grouping that splits or merges the top-level clusters differently need not
      disagree with people.
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
    - **Compare the 3.8 27B with Flash Next here** (maintainer, 2026-11-03). On the short judge batches the two nearly
      coincide; reading whole documents with looser categories is a longer task, where a gap may open.
    - **It reads fulltexts**, one at a time, which ties it to item 8's fetching. The maintainer expects at
      least a representative set from the coauthor in the week of 2026-11-02.
    - **Related reading, beside the Elo & Kyngäs PDF**: Elo et al. (2014) on the trustworthiness of
      qualitative content analysis, and Borse et al. (2025, arXiv 2508.14764) on inter-rater reliability
      between LLMs and human raters in qualitative analysis — the question this item puts to the test.

## Decision log

- **2026-10-29**: this brief created, to hold the review-automation items of both studies as one set.
- **2026-10-29**: items 10 and 11 added by the maintainer; item 10 is the nearest deadline, a prototype
  early in the week of 2026-11-02, and replaces the HDBSCAN clusters for the health-education study.
- **2026-10-29**: item 10 produces a grouping shaped like the earlier studies' tables (themes and
  sub-themes); coding per research question goes to item 11. Considered and set aside for the prototype:
  doing both at once, which has nothing on hand to score the coding against.
- **2026-10-29**: a person marks one sheet, the copy of `03a-CHECK-BY-HAND`, for every decision: wrong
  removals and keeps are pasted in from `03b-removed` and `03c-kept`, which share its columns, and that
  sheet is what `--decisions` carries. Considered and set aside: `finish` reading several sheets, one per
  table marked — more places for a decision to live, for no gain. The sheets' column order and widths are
  still to tune, from the maintainer's notes on marking at 1080p (expected the week of 2026-11-02).
- **2026-10-30**, item 10's prototype, settled before building (maintainer):
  - **The research questions are context, not the themes.** The themes are what the studies are about,
    matching the published tables in spirit; one theme per question would be item 11's coding.
  - **The number of themes is left open** for the first runs; steering it is for later, if wanted.
  - **The records are shuffled per run**, with the seed recorded: order effects are a confound. What remains
    between runs is not only sampling noise — the problem may admit several good groupings, and a run may
    find any one of them (maintainer).
  - **Placing a study under two themes is allowed in the grouping**, as the earlier tables do.
  - **Citations read "Author (2024)", "Author & Another (2024)", "Author et al. (2024)"**, from
    `raven.common.utils.format_bibtex_authors`, which already does the et-al form.
  - **Agreement**: pair co-membership over overlapping groups (the omega index, Collins & Dent 1988), with
    ARI on each study's primary placement beside it; both at the top level and the sub-theme level.
  - **Effort**: LM Studio caps the thinking at 12k tokens (a server setting, 2026-10-29), so a reply cannot
    think its way past the 16k output cap. Run at `xhigh` first, and drop to `low` if thinking crowds it.
  - **The grouping is plottable in the Visualizer**, with made-up 2D coordinates rather than an embedding:
    one circle per theme, its studies on the perimeter, the circles' centres scattered as Librarian's
    attachment piles are (a maximin Latin square), so the plot reads at once as an LLM grouping laid out by
    hand rather than as a t-SNE map. **For the plot, each study gets one place** — the Visualizer's promise
    is that each point is a unique record — so the model marks one placement per study as primary.
  - **Home**: `investigations/llm-thematic-grouping/`; a tool once the method holds on biology and geography.
  - **Whether the research questions help is one of the things the prototype is to find out** (maintainer,
    2026-10-30), so they are a condition to ablate, not a fixed input.
  - **For the general tool: the guidance is a general aim**, a background explanation of how to group, which
    a set of research questions is one case of (maintainer, 2026-10-30).
- **2026-10-30**, after the first evening's three conditions (results in `investigations/llm-thematic-grouping/`):
  - **The topic line stays**: it names the review's scope, setting and population, without handing the model
    categories to sort by, which the research questions did. Asking the model *not* to make one theme per
    question did not hold.
  - **Themes only**: sub-themes agreed at 0.27–0.36 under every condition, so people read and name them, as
    in the earlier studies. A second pass per theme stays possible.
  - **Programme continuity** ("sustainability" as a programme lasting) came out as a theme of its own under
    every condition. Those studies are in scope on the hand screening's reading; an instruction to place them
    by what else they are about is a hypothesis, measured against a control without it.
  - **Slips in a reply get a repair turn**, naming each one and asking for the whole corrected JSON, up to
    two. Considered and set aside: tools for declaring themes and placing studies — about forty tool rounds a
    run, tool schemas to build, and the model losing the single global view a grouping needs, for a
    guarantee the repair turn gives too.
  - **Ten runs a condition** from here on, five having left differences of 0.1 a lead rather than a result.
- **2026-10-31**: **the programme-continuity instruction is dropped.** It renamed the theme more than it
  dissolved it, and cost stability; the studies in it share a topic of their own, implementing programmes
  in schools.
- **2026-11-02**, the health-education study's open questions (maintainer):
  - **The implementation theme stays**: it is a real theme in the dataset, which settles the dropped
    programme-continuity instruction of 2026-10-31.
  - **Biology's Sterling (2013) and geography's Yli-Panula et al. (2020) are likely slips in the papers' study
    lists.** For health education, `raven-checkrefs` is run against the study list, to avoid such slips.
- **2026-11-03** (maintainer):
  - **Item 10's approach is to become a tool in the near future**: the method in
    `investigations/llm-thematic-grouping/METHOD.md`, finding themes, pooling them into a catalogue, people editing it, and
    applying it. Its shape waits on how the catalogue worked with the coauthor.
  - **Noted, with no expectation of doing it: whether a larger model's general knowledge changes what it does.** A larger model
    would be expected to store more trivia, and recall is easy to test; what counts is a case where knowing an obscure fact
    leads to a better approach to a problem, and designing good cases for that is hard.
- **2026-11-05** (maintainer), for the tool, unscheduled:
  - **An ensembled pilot**: several runs, so that a result can say what is stable, as item 10's runs do.
  - **Recommending a sample size**: invert the bound, so that a person names the precision wanted and gets the number of records to
    rate. Part of `VISION.md`'s obligation to measure the error rate. Advice in round numbers, which are easy to remember and
    cost a record of extra work at most: 15, 30, 60, 150, 300 records for 20, 10, 5, 2, 1 percent (maintainer).
- **2026-11-05, for the papers' methodology sections** (maintainer): write the mathematics at step-by-step detail, every term and
  every substitution shown, as `VISION.md`'s *How to read such a number* now does for its Bayes calculation. It makes it hard for a
  math bug to hide. Also to do then: check whether PRISMA has an extension for AI-assisted reviews, get a statistician's read if we
  can find one, and have a model from another family review the method before submission.
- **2026-11-04/05, working with a team that meets every two weeks** (maintainer):
  - **Batch rather than block.** Do whatever can be done independently, and collect questions for one reply, so that an issue in 5% of
    the work does not hold up the other 95%. Coordinating with a team that meets rarely is chess by mail; a stream of questions is a
    GUI full of modal dialogs.
  - **So an open scope question is included and tagged, not decided.** The team then decides a whole tagged set at once, seeing what it
    holds. With 200 tagged records or fewer, they can go in the review sheet as one continuous run. If nearly every record carries a
    tag, it becomes a custom field in the output `.bib`, with a filter step after the decision. Not built yet: extraction gives the
    tags, and writing them into the `.bib` and filtering on them are small additions.
  - **The maintainer's reads on the open questions, not checked with the team:** perceptions out, reviews in (useful on building such
    systems), informal out. The first two are tagged and asked.
  - **Messages to the team start from the maintainer's register.** Edits to a draft softened assertions into the conditional, broke up
    dense paragraphs, and put thanks and context before any request: colleagues at the next meeting, rather than findings presented.
- **2026-11-05, the review sheet's UX** (maintainer): the team replaced `mark` with their own columns: decision, whether the abstract was
  enough, how AI was used, and a reason. Those become the sheet's columns. Column order and widths still to tune, starting from what
  causes the most friction in practice. Get the tool working first.
- **From the AOKK meeting of 2026-11-03, for the paper; the notes themselves are gitignored:** cite the 3–5 most relevant articles per
  cluster, with the whole clustering in an appendix of links or references. Publish the clusters as an interactive artifact, without
  abstracts, since the bibliography cannot be published (technology open; check the journal's practice). A per-country table, from the
  fulltext tool that health education needs anyway (item 8). "PRISMA-inspired", and check how the biology and geography papers worded
  this. A case study asking the material about two research questions. Present the review system as a methodological strength without
  displacing the pedagogical contribution.
- **2026-11-05, small things for later:** `VISION.md` gets concrete numbers once the two studies are published. When sample sizing reaches
  code, its docstring may cite Daniels (US Air Force, about 1952), who found no average pilot: a note on sizing pilots, in a straight
  face. `raven-scopebib` has no `--declare`, so a run on a backend not reporting quantization leaves it out of the methods line; low
  priority, the 27B reporting its own.
- **2026-11-09, the PRISMA extension check for the papers' methodology** (from the note of 2026-11-05): PRISMA-trAIce (Holst et al., JMIR AI
  2025;4:e80247) is **not endorsed by the PRISMA Executive**, who say it did not follow their process (Moher et al., "PRISMA-trAIce: A Name
  Without Endorsement", letter to the editor, JMIR AI 2026;5:e104210, doi:10.2196/104210). The same letter says official consensus-based
  guidance on AI tools is being developed and will be hosted on the PRISMA website, so check <https://www.prisma-statement.org/extensions>
  again before submission.
  L-PRISMA (Shailendra et al., arXiv:2603.19236, accepted at ICMET 2025) is a proposal from one use case with no measure of what it misses,
  and does not fit our method, but its flow diagram counts records screened and studies included by GenAI apart from the manual ones,
  which is the box PRISMA 2020 lacks for records a tool keeps.
  **AOKK's study is a scoping review, so its reporting guideline is PRISMA-ScR** (Tricco et al., Ann Intern Med 2018;169(7):467–473). It
  prescribes no diagram boxes: item 17 asks for the numbers screened, assessed and included, "with reasons for exclusions at each stage,
  ideally using a flow diagram", item 9 for the selection process, and item 5 for a protocol and its registration only if they exist. It
  does not mention automation. Levac et al. (Implement Sci 2010;5:69) recommend at least two researchers each independently reviewing
  abstracts, meeting at the beginning, midpoint and end, and refining the search if needed.
- **2026-11-09, a second family over the removals** (maintainer agreed, to do after AOKK's blind sample): a judge from another model family reads
  only the records the tool removed, and any disagreement goes to a person, the part of Levac et al.'s "third reviewer" a tool can play. What a
  second reviewer buys is independent errors, so two runs of one model, or our judge and reviewer stages, do not count as two; the paper says
  "LLM screener" and leaves the ontology alone. One more judge pass over the removals. It can only take records out of the removed pile, so the
  number of in-scope records wrongly removed can only fall, but the bound on the removal rate then describes the earlier pile, and a second
  round re-measures it.

# Open threads, November 2026

*A working list, opened 2026-11-16 because a single day had more threads in flight than anyone could hold in
their head. It points at where each thread's detail lives; it is not the place for that detail. Take an item
off when it closes, and delete the file when it is empty.*

## Done 2026-11-20: year filtering, for the AOKK review

`raven-siftbib --where 'year>=2022'`, the same in `raven-scopebib`'s sift, and live year filtering in the
Visualizer with a reset per filter. Also a zoom race on dataset load fixed, and *Interface design* (Nielsen,
Shneiderman) in `raven-style-guide.md`. Decisions: `briefs/visualizer-filter-by-date-and-recolour-brief.md`.

## Done 2026-11-19: the health-education methods draft, first version

See *Waiting on others* for what it still needs.

## Done 2026-11-18: material for an executive summary

The three infographics are in `img/` and embedded in the README and the apps' manuals (papers, Librarian,
Visualizer); the light-background plot on its own is `img/visualizer-plot-light.png`, Misc grayed, with a 2% margin.

## Done 2026-11-26: colouring the map by year, v0, and the keyword shortlist read

- **Colour by** cluster or year (F2, Ctrl+B), with the tooltip and the info panel grouping the same way, and the
  year before each tooltip title.
- **The colour bar**, now a sparkline of what is shown, its heights compressed data-adaptively (Ward Larson,
  Rushmeier and Piatko), naming the column under the mouse and selecting it on a click (`raven.visualizer.colorscale`).
- **Search by field**, a sigil to type and a prefix to read: `#7`/`cluster:7`, `~2022`/`year:2022`, ranges, `#misc`,
  `~none`. Author search will be `@smith`/`author:smith` (`TODO.md`).
- **The shortlist read** found eight keyword faults, fixed as one bundle with Themes putting shared terms last
  (`investigations/cluster-keywords/README.md`).

## Next, 2026-11-27

- **First thing: the maintainer reads the keyword rules draft**, `briefs/reference/cluster-keywords.md`, the *Considered and
  rejected* section especially.
- **The re-import started at 16:10 on 2026-11-26** with all of the day's fixes; the previous datasets are kept as
  `investigations/cluster-keywords/data/*-2026-11-25.*`. Check it finished (`*.log` there), then make the shortlists
  again (`shortlist_clusters.py CORPUS --previous 2026-11-25`) for a quick read of what the bundle changed, and settle
  the rules draft against it.
- **The re-import finished at 23:23**, all twelve datasets (geography's LLM import retried after one failed name
  lookup of the server). One regression, on arXiv alone: cluster 4 gave up in both runs, with 4 and 3 malformed
  replies in all. The reply is a *Common theme:* paragraph, then *Suggested keywords:* and a bulleted list, which the
  list parser rejects, as it allows a header line but no prose before it. Next: read the bulleted block under a
  heading that names keywords; how much prose to tolerate is the design call.
- **Then colouring by year's v1**: keywords per year at import (`briefs/visualizer-filter-by-date-and-recolour-brief.md`).
- **Filed 2026-11-26**: `FileDialog` changes the process's working directory (`TODO_DEFERRED.md`).

## In progress: cluster keywords (2026-11-24)

`investigations/cluster-keywords/README.md` has the detail. Built today: `clusters_keyword_method = "hybrid"` (the
LLM's corpus-wide terms do not lead; an abbreviation counts as a repeat of its long form), the LLM's whole keyword
list kept in the dataset, hyphen and joined spellings folded, and keywords capitalized as the corpus does. The
maintainer has read the whole arXiv evidence comparison.

- **The new imports are done** (evening of 2026-11-24, all six corpora, LLM with evidence and hybrid).
- **Decided the same day**: the evidence is on by default; embedding terms by their sentences is rejected.
- **Built 2026-11-25**: a cluster shows **Themes** (the LLM's) and **Quoted** (the phrases; first called *Terms*) on two lines, in the
  info panel and the tooltip (the shortlist read's first finding: six terms cannot hold both); the LLM is asked for
  up to fifteen keywords; *test-time* compounds kept whole; `#7` in the search field finds cluster 7; the selection
  heading reads *68 clusters + Misc*.
- **In progress: the shortlist read**, `investigations/cluster-keywords/out/CORPUS-shortlist.md`; arXiv clusters 1
  and 7 read.
- **Done, evening of 2026-11-25: the re-import**, with the hyphen fix and the fifteen-keyword lists; the shortlists
  are made again from it.
- **Built 2026-11-26**: the extraction bundle (eight faults) and Themes putting shared terms last; the re-import
  waits on the maintainer's ping.
- **Then a document of the final keyword rules** (the maintainer, 2026-11-26: a lot of latent knowledge goes into
  them): each rule with the failure it prevents, from extraction to display. A reference document beside the code,
  `briefs/reference/cluster-keywords.md` (moved there 2026-11-27 from beside the module, describing how the rules
  are rather than what to build), linked from the module and from the Visualizer README. Reviewed 2026-11-27.
- **Still open**: the hybrid's ordering prompt (option 3), to be measured against tonight's import.

## Ready to pick up, no deadline

- **Matching the three reviews' study lists against the fulltext set** that arrived on 2026-11-17:
  `investigations/health-education-review/README.md`, *Open*. Also what the per-country table reads from.
- **Backend flavors for gufo and strata**, with a reasoning-effort setting and its selector in Librarian: built
  and tested on the local AI server. `TODO_DEFERRED.md`, the stock-take item.
- **Release 0.2.11 before the health-education paper comes out**, so that its methods can cite a released
  version (2026-11-19). No date yet; it follows the paper's schedule. The `release` skill has the process.
  - **What remains important for it** (the maintainer, 2026-11-25): **all-fields search**, with the full author
    list shown and searchable (`TODO.md`, *Author search* and *Fragment search across multiple fields*), and
    **colouring the map by year**. Also the README screenshots, retaken.
  - **Added 2026-11-27, for the health-education and AOKK studies**: a Themes mode for the word cloud (done the same
    day), and a large, printable version of the colour bar's histogram, with the word cloud's UX (`TODO.md`).
- **All-fields search with one-line snippets** around each match and a toggle to the full abstract, designed
  2026-11-25: `TODO.md`, *Fragment search across multiple fields*. M.
- **Colouring the Visualizer's map by year: v0 done 2026-11-26.** Then v1 with per-year keywords, before 0.2.11
  is released. `briefs/visualizer-filter-by-date-and-recolour-brief.md`, *Decided: colouring by year*.
- **The year filter's two-handle range slider**, to replace the two fields later: a drawlist and custom code, as
  DPG has none. Same brief.
- **A light theme across the constellation**, with a roomier layout as an option, starting with a UI design
  session on the principles listed there: `TODO.md`, Cross-cutting. M or L.
- **Datasets of 100k records in the Visualizer**: level-of-detail rendering for the plot, on the model of
  Barnes–Hut chunking. `TODO.md`, the Visualizer's display section.
- **The pipeline figure's redraw**, for 0.2.12 or 0.2.13: `TODO.md`, the figure and screenshot sweep.

## Waiting on a decision

- **Aria's voice: how to measure from here.** Settled: two axes, and the absurd prompts kept with a setup turn.
  Open: ranking within a prompt against its cost in the rater's attention; the steganosaur and AGItater
  humour tests; whether to skip further judge tuning and go to step 4, the candidate card compared by hand.
  `investigations/aria-register/README.md`, *The fourth blind sample*.

## Waiting on others

- **The health-education methods description**: first draft done and reviewed on 2026-11-19
  (`investigations/health-education-review/METHODS-DRAFT.md`), with the run-to-run variation measured on the
  adopted prompt. Waiting on the coauthor for the two further exclusions, the reasons for the authors' own ten
  exclusions if the flowchart keeps the breakdown, and the word clouds. Two small fixes in the manuscript's
  Table 5 are noted in the study's README.

- **The AOKK scoping review: year filtering is done (2026-11-20); next, clusters from the final keeps.**
  - **The clusters wait twice**: on the team's keep-sample ratings, which fix the final keeps, and on our design
    discussion of LLM grouping at about 2,000 records (`investigations/llm-thematic-grouping/README.md`, *Next*:
    the map's clusters as bins, a revision step over cluster summaries, batched allocation with reasons).
  - **With the final keeps**: comparing year-specific maps against the whole map filtered by year, for the
    trends. Both routes exist (`raven-siftbib --where 'year==2023'` per slice, or the live filter).
  - **2026-11-24: the keep sample is rated**, which unblocks rules v2; teachers as AI users are out; a table of the
    clusters is wanted for the article; and the number of clusters is open, perhaps to be swept.
  - **Done 2026-11-25: the run with the new rules** (`AOKK/scopebib-2026-11-25-rules-v3/`): of 3,765 records,
    2,588 removed, 495 to a person, 682 kept. Next, the team's: the hand check of `03a-CHECK-BY-HAND.xlsx`, which
    fixes the final keeps. `investigations/aokk-corpus-scope/README.md`, *The run of 2026-11-25*.
  - **Then, on the final keeps**: the cluster table export (a Visualizer feature, `TODO.md`; its hotkey needs the
    help card split) and the cluster-count sweep.
  `investigations/aokk-corpus-scope/README.md`, the meetings of 2026-11-19 and 2026-11-24, points to each.

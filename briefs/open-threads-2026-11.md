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

## Done 2026-11-27: the printable histogram, recovery from a server restart, and the keyword fixes

- **The keyword rules document**, `briefs/reference/cluster-keywords.md`, read and revised by the maintainer, with a
  reference list. The arXiv regression (cluster 4) is fixed: the keyword parser reads a bulleted list under a
  heading that names keywords.
- **Keyword fixes**: an abbreviation most papers define shows as *fast weight programmer (FWP)*; every word votes
  on its casing across the corpus (`keyphrases.EXTRACTION_VERSION` 5), meant to fix AOKK's *LLM-Based*: tonight's
  import is the first to show whether it does.
- **The word cloud's Themes mode**, with a *Shared themes* checkbox; both kept between sessions.
- **The printable histogram (F12)**: the colour bar's counts as a matplotlib figure, on adaptive, log or linear
  scales (Shift+F12), saved as PNG, PDF or SVG. Its scales, ticks and labels are in `raven.common.numutils`
  (`histogram_adjustment`, `log_scale`, `decade_ticks`, `spread_labels`, `data_labels`), the adaptive one linear in
  the CDF as in SAVU's `hdrplot.m`, which the colour bar's sparkline now follows too. The help card gained a
  *Windows* page. One background setting, `figure_background_color`, for the histogram and the word cloud, and
  `colormap_ranges` per background lightness for them and the plot.
- **Recovery from a Raven-server restart**: the client keeps each avatar's desired state (`api.avatar_restore`),
  batch work retries with backoff, and a dropped feed resumes. Tested live, `kill -9` included.
- **The Visualizer reports a dataset it cannot open**, from the command line too, where it hung.
- **Filed**: porting the rest of SAVU, and its pool-opening ensemble as a test and a JS toy (`TODO_DEFERRED.md`); an
  interactive HTML export of a dataset for supplemental material (`TODO.md`).

## Next, Monday 2026-11-30

- **Tonight's re-import**, started 15:00 on 2026-11-27 with extraction version 5, from a worktree frozen at
  `29d1745b` (`.claude/worktrees/reimport`, run with `PYTHONPATH` pointing into it). The previous datasets are kept as
  `investigations/cluster-keywords/data/*-2026-11-26b.*`. Check all twelve finished (the `*.log` there), remove the
  worktree (`git worktree remove .claude/worktrees/reimport`), and make the shortlists again
  (`shortlist_clusters.py CORPUS --previous 2026-11-26b`) for the maintainer to read; AOKK's *-Based* capitals are
  the thing to check.
- **The histogram's next steps**, decided 2026-11-27: count the selected records, as the word cloud does, and in
  year mode follow the year filter along the bottom. Open: what an empty selection shows (`TODO.md`).
- **`FileDialog` changes the process's working directory** (`TODO_DEFERRED.md`): it bit twice on 2026-11-27, so it
  wants fixing rather than working around a third time.
- **Help cards: where a secondary window's keys go**, across the apps. The Visualizer's keyboard page now names only
  the key that opens each window, its other keys on a *Windows* page (2026-11-27). Librarian lists its audio input
  panel's keys (F9, then D, A, Esc, …) on the keyboard page under the opener, its chat graph on a page of its own.
  Settle the rule (`raven-style-guide.md` → *Pages*), then bring the apps in line.
- **Then colouring by year's v1**: keywords per year at import (`briefs/visualizer-filter-by-date-and-recolour-brief.md`).

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
  - **Added 2026-11-27, for the health-education and AOKK studies**: a Themes mode for the word cloud and a
    printable histogram, both done the same day; the histogram's next steps are under *Next*.
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

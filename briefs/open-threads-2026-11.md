# Open threads, November 2026

*A working list, opened 2026-11-16 because a single day had more threads in flight than anyone could hold in
their head. It points at where each thread's detail lives; it is not the place for that detail. Take an item
off when it closes, and delete the file when it is empty.*

## Due Wednesday 2026-11-18: material for an executive summary

**State at the end of 2026-11-17**: item 1's draft is done and waiting for the maintainer's comments:
`img/raven-librarian-infographic.svg`, not yet embedded anywhere; its PNG is made once it is agreed, as the papers
one's was (`inkscape … --export-width=3200`). Items 2 and 3 are not started.

The audience is managers and funders, alongside researchers; they expect a light theme. `TODO.md`, Cross-cutting,
has the entry.

1. **Raven-librarian's infographic**, first, as it needs nothing from the maintainer. Outline proposed, not yet
   agreed:
   - *"Ask your documents; keep every branch."* At the centre, a conversation branching where a reply was
     rerolled, adapted from `img/chattree-diagram.svg`.
   - Around it, what feeds the answers: the user's documents (hybrid search), web search and tools,
     attachments, and the optional voiced avatar.
   - "Why" cards along the bottom: a local LLM, nothing leaves the machine unless asked, every branch kept. Not
     "answers cite their sources": Librarian stores the search matches in a reply's metadata and does not cite
     them in the reply yet (maintainer, 2026-11-16). Citations are filed in `TODO.md`, "Inline citations,
     validated".
2. **The light-background Visualizer plot**, together with the maintainer: the plotter's colours set through the
   REPL (they are settings already), a screenshot of the plot, cropped. The colormap range fix (below) first, if
   there is time, as Viridis's yellow end is weak on white.
3. **Raven-visualizer's infographic**, around that plot. Outline proposed, not yet agreed:
   - *"Take 10,000 studies, find the ones worth reading"* (the README's goal is 10k+). The light plot at the
     centre.
   - The import as a short line of stations on the left: BibTeX, semantic embedding, clusters, the 2D map,
     keywords per cluster.
   - What to do with the map, as callouts on the right: search, select by brush or cluster, the info panel, a
     cluster's word cloud, the display filters.
   - "Why" cards: local and private, tens of thousands of records, keyboard first.

All three in the style of `img/raven-papers-infographic.svg` (16:9, a solid light background, an SVG source with
a 2× PNG beside it); its design record is `briefs/design/papers-infographic-sketch.md`.

## Ready to pick up, no deadline

- **The health-education study's methods section**: a free-form description of the processing, in English, for the
  authors to write the manuscript's section from. The record it draws on, with the decisions settled on 2026-11-17 and
  what is still open, is `investigations/health-education-review/README.md`; its *Open* section is the next steps:
  the methods description itself, two small fixes in the manuscript's Table 5, and matching the three reviews'
  study lists against the fulltext set that arrived on 2026-11-17.
- **The colormap's dark end against the dark plot background**: a range setting and one line in
  `plotter.colormap_position`, the bound chosen from screenshots side by side. `TODO.md`, the Visualizer's
  display section. Cost S.
- **Date-range filtering, recolouring, embeddings in the dataset**: `briefs/visualizer-filter-by-date-and-recolour-brief.md`,
  opened and not yet discussed.
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

- **The AOKK scoping review**: the team's ratings of the keep sample and their decisions on the open record
  groups, then rules v2 and probably a new LLM run. `investigations/aokk-corpus-scope/README.md`.

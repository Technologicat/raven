# Open threads, November 2026

*A working list, opened 2026-11-16 because a single day had more threads in flight than anyone could hold in
their head. It points at where each thread's detail lives; it is not the place for that detail. Take an item
off when it closes, and delete the file when it is empty.*

## Done 2026-11-18: material for an executive summary

The three infographics are in `img/` and embedded in the README and the apps' manuals (papers, Librarian,
Visualizer); the light-background plot on its own is `img/visualizer-plot-light.png`, Misc grayed, with a 2% margin.

## Ready to pick up, no deadline

- **The health-education study's methods section, next (from 2026-11-19)**: a free-form description of the
  processing, in English, for the authors to write the manuscript's section from, as
  `investigations/health-education-review/METHODS-DRAFT.md`. The record it draws on, with the decisions settled on 2026-11-17 and
  what is still open, is `investigations/health-education-review/README.md`; its *Open* section is the next steps:
  the methods description itself, two small fixes in the manuscript's Table 5, and matching the three reviews'
  study lists against the fulltext set that arrived on 2026-11-17.
- **Release 0.2.11 before the health-education paper comes out**, so that its methods can cite a released
  version (2026-11-19). No date yet; it follows the paper's schedule. The `release` skill has the process.
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

- **The AOKK scoping review**: the team's ratings of the keep sample, then rules v2 and probably a new LLM run.
  The meeting of 2026-11-19 settled the corpus's edges (the excluded ProQuest collection out, ERIC at 247) and
  asked for year-based work: a year criterion in `raven-siftbib` (S, ready to build), year-specific maps compared with live
  year filtering, and whether LLM grouping scales to the final keeps (about 2,000), design to be discussed first. `investigations/aokk-corpus-scope/README.md`, that meeting's
  section, points to each.

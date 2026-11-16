# Sketch: an infographic of `raven.papers`

*Status: requested 2026-11-13 (maintainer): an executive-summary graphic of what the paper tools can do, why they
are useful, and how they fit a researcher's workflow, ideally a pretty one. The open questions were settled on
2026-11-16 (below, under Open), and the graphic was drawn the same day: `img/raven-papers-infographic.svg`, the
source, edited by hand or in Inkscape, and its PNG export beside it, which the README shows. A Claude-ism in the
first draft's agent card ("run the tools from their manuals") was reworded to "the manuals are written for it",
which does not claim an agent integration Raven does not ship (maintainer). Infographics of the Visualizer and
Librarian in the same style are wanted next (`TODO.md`, Cross-cutting).*

## What it would say

**The message**, the papers manual's own: the part of a literature review that happens before anyone reads
anything, done with tools that keep a record of every decision.

**The workflow, as stages**, with the tools in each:

1. **Gather.** arXiv: search, identifiers, BibTeX, full texts (`raven-arxiv-search`, `raven-arxiv2id`,
   `raven-arxiv2bib`, `raven-arxiv-download`). Database exports into BibTeX: `raven-wos2bib`, `raven-csv2bib`,
   `raven-pdf2bib`.
2. **Clean.** `raven-fixbib` repairs what the databases hand over; `raven-deduplicate` merges one paper found by
   several databases, auditing every merge; `raven-fillbib` fills missing and truncated abstracts from Crossref;
   `raven-siftbib` sets aside what cannot be screened.
3. **Screen.** `raven-scopebib`: an LLM applies the researcher's own definition of scope, a reviewer reads its
   removals, extracted tags drive rules, and the doubtful go to a person; checked with a blind sample, and runs
   compared with each other.
4. **Explore** (Raven beyond `raven.papers`): the Visualizer's semantic map of the result; Librarian for questions
   over the documents.
5. **Write and verify.** `raven-checkrefs` compares a manuscript's reference list with the `.bib`.

**Why it is worth using**, as a strip along the bottom: auditable (every merge, removal and hold has its reason in
a table); reproducible (run files, saved answers); local and private (it runs on the researcher's own machine);
agent-friendly (an agent can run it, from the manuals); the PRISMA flowchart's numbers come out as a by-product.

## How it could look

- **A. A metro map**: the tools as stations, an arXiv line and a databases line merging at `raven-deduplicate`
  and running on through cleaning and screening to the map and the manuscript. Shows the workflow and the
  optional branches at once. The proposal of 2026-11-13.
- **B. A funnel**: the record count shrinking stage by stage, PRISMA-like. Makes the value obvious at a glance,
  but only with a real run's numbers.
- **C. Swimlanes**: what the researcher does, and what the tools do. The clearest about who does what; the least
  visual.

A and B combine: the metro map, with a small funnel inset.

## Open, and settled (maintainer, 2026-11-16)

- **Audience**: researchers new to the tools, and management too: aimed at researchers, trimmed if it needs to
  be. Tool names stay, as some detail gives the diagram credibility.
- **Format**: SVG, horizontal rather than A4, as the main targets are Raven's README and PowerPoint slides; so
  16:9. On a solid light background, which GitHub's dark mode would otherwise turn unreadable.
- **The funnel's numbers**: illustrative, labelled as such.
- **The look**: the metro map (A), with the funnel (B) as bars under its trunk line.

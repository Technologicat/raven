# Brief: filtering the Visualizer by date, and recolouring it

*Status: opened 2026-11-16, not yet discussed. The scope comes from the display-filters brief
([`done/visualizer-display-filters-brief.md`](done/visualizer-display-filters-brief.md)), which built the layer
these reuse and left them for later as a natural extension rather than part of its own work.*

The display filters gave each record a label it is shown with (`current_label_id`) and a visibility (shown,
grayed, hidden), kept in `raven.visualizer.displaystate`, separately from what the importer found. The two uses
so far are the purity cutoff and the Misc selector. This brief is for the next uses of the same layer.

## Wanted by the AOKK review (2026-11-19)

Year-specific maps, to see how the topics changed from 2022 to 2026
(`investigations/aokk-corpus-scope/README.md`, the meeting of 2026-11-19). Two routes, and the team named both:

- **Prefilter, then import**: sift the bibliography by year and import each slice. It needs no new Visualizer code,
  only `raven-siftbib --where 'year==2023'` per slice, or two `--where` for a range (since 2026-11-20). Each slice is its own map, with clusters
  of its own, so the slices' clusters do not correspond to each other.
- **Filter live**: candidate 1 below, over one map of the whole corpus. The clusters stay the same across years,
  which is what a comparison over time wants.

**Both, and the difference between them, is the point** (the maintainer, 2026-11-19): independent maps per year
against the whole map filtered by year. Does the comparison reveal anything new about the data?

## What is already there

- **The layer**: `displaystate`, with `interactable` deciding what can be hovered and selected; the plot is
  refilled from it (`plotter.refresh_series`) and never re-sorted.
- **The rules the first brief set**, which a new filter inherits rather than renegotiates: leaving "you see all
  the data" is always visible (the filter pill and the plot title's count); **Show all** resets every filter
  at once; only records in full colour can be interacted with; search counts what it cannot show; and the
  selection keeps records a filter sets aside, bringing them back when they are shown again.
- **The shared pill**, `raven.common.gui.pill`, which a second filter's state can be told through.

## The candidates

1. **Date-range filtering**: a minimum and a maximum year, records outside it grayed or hidden.
   - The year is on every record as the importer read it. A record without one carries Raven's placeholder,
     which `raven.visualizer.importer.is_missing_field` tells from a value a database supplied; the
     importer's docstring names this filter as the reason that helper exists. What the filter does with
     a record that has no year is a decision to make.
   - A second filter brings the first question the layer has not yet had to answer: how two filters combine,
     and which of them the pill and Show all speak for.
2. **Recolouring by another criterion**, such as date or first author: another way of filling the series
   from per-record state, the label being one choice among several.
   - The colours currently follow cluster labels, with the legend in the info panel and the tooltip's cluster
     headers. Recolouring by a continuous quantity (a year) and by a categorical one (an author) want
     different colour maps and a different legend.
3. **Storing the embeddings in the dataset**: they live in the importer's cache file today. Needed the first
   time a feature wants them when a dataset loads, rather than at import.

## Decided: the year filter (2026-11-20)

**Date-range filtering comes first** (the maintainer, 2026-11-19): it is next for the AOKK review, together with
the year criterion in `raven-siftbib` (done 2026-11-20). Recolouring and the embeddings in the dataset come later.

Settled with the maintainer on 2026-11-20:

- **Filters combine by AND, over three states**: a record is shown in full colour only if every filter shows it,
  hidden if any filter hides it, and grayed otherwise. The most restrictive filter wins.
- **So each filter computes its own verdict per record, and one function combines them.** Today
  `displaystate.apply_misc_mode` writes `visibility` directly and resets every other record to `SHOWN`, which a
  second filter would overwrite; its logic stays, and where its result lands changes.
- **The filter**: a minimum and a maximum year, defaulting to the dataset's own range and reset when a dataset
  loads; and a mode for the records outside it, grayed or hidden, cycled like the Misc control. Leaving the full
  range is what switches it off, so there is no shown mode. *Show all* resets it with the others.
- **A record with no year follows the out-of-range records**, as a missing year fails `raven-siftbib --where`.
  The importer's placeholder marks one (`raven.visualizer.importer.is_missing_field`).
- **Two integer fields for the range, for now**, to get to testing quickly. **A two-handle range slider later**
  (the maintainer): DPG has none, so it is a drawlist and custom code.

Settled in live testing, the same day:

- **The year controls are a second row**, right-aligned, and **Grayed points selectable** sits at its left end,
  under the purity readout, as it applies to whatever any filter grays. The purity slider is labelled **Misc
  cutoff**.
- **Keys**: **Ctrl+E** (era) focuses the year mode selector, which then takes **Left / Right** to slide the range
  a year keeping its width, **Shift+** to move the first year and **Ctrl+** the last, and **Up / Down / Home / End**
  for the mode. **F5** switches the mode from anywhere.

**A reset per filter, beside *Show all*** (the maintainer, 2026-11-20). Both kinds are needed, and not as
alternatives: *Show all* discards the whole view, which is right only when that is what the user wants ("I'm done
with this view, back to the whole dataset"). Undoing one filter must never require *Show all* followed by
remembering and redoing the others. That is *recognition rather than recall*, the sixth of
[Nielsen's usability heuristics](https://www.nngroup.com/articles/ten-usability-heuristics/) (Nielsen and Molich 1990,
refined 1994), and Shneiderman's "reduce short-term memory load" ([*Eight Golden Rules*](https://www.cs.umd.edu/users/ben/goldenrules.html));
a reset per filter is Nielsen's *user control and freedom*, an exit from one action rather than from all of them.
`raven-style-guide.md` → *Interface design* maps both lists to Raven as a whole.

- **A reset leaves its filter's mode alone**, in both filters. The years reset to the dataset's span; the Misc
  reset sets the cutoff to 0%.
- **At the left end of each filter's controls**, styled like the search field's clear button, with a tooltip saying
  what it resets.
- **No hotkeys for now.**

## Decided: what comes after the year filter (2026-11-20)

- **Colouring by year is in scope for 0.2.11** (the maintainer): candidate 2 above, for the year only. A
  continuous quantity wants a sequential colour map and a legend of its own, in place of the cluster colours.
  Not yet designed. Colouring by a categorical criterion, such as the first author, is not part of it.
- **Time at a finer granularity than a year comes in a v2**, perhaps H1/2027: timestamps down to nanoseconds since
  the epoch, as a system log carries, with a general time-range picker in the GUI. It needs the dataset format
  changed, as BibTeX gives a year only, so it is filed with the format redesign (`TODO.md`, "Data file format").
  - **Per-year keywords stop being precomputable there** (the maintainer): an arbitrary time range has no fixed
    groups. LLM keywords do not add up over a bin, so they would want precomputed bins at a fixed base
    granularity, or generating on demand.
  - **Frequency keywords add up over any bin**, as each record's keyword counts are already in the dataset (which
    is how the word cloud sums a selection), **but are useless as they stand** (the maintainer). The fix is planned
    in `briefs/visualizer-keyword-pools-brief.md`, *An improved classical mode, worth trying*: phrases as candidates, then c-TF-IDF,
    then an embedding rerank. The first two keep the additivity, needing only each bin's term counts against the
    corpus totals, so an improved frequency keyworder would serve arbitrary time ranges. The rerank needs each
    bin's centroid embedding, one more reason for the embeddings in the dataset.
- **The embeddings in the dataset move to the format redesign** too (candidate 3); same item.
- **The two-handle range slider** replaces the two year fields later (*Decided: the year filter*, above).
- **The comparison of year-specific maps against the filtered whole map waits for the AOKK team's final keeps**,
  and is done with them.

## Decided: colouring by year (2026-11-20)

Settled with the maintainer:

- **A *Colour by* selector, *Cluster* or *Year***, on the year row, with **F2** to switch and the same keyboard
  handling as the other selectors.
- **The plot's configured colormap**, reassigned to the years: a change of meaning, not of palette.
- **A colour bar on the year row**, "2012 ▬▬▬ 2026", from the same colormap, shown only while colouring by year: a
  small drawlist of a rectangle per year. What the colours mean is on screen.
- **A record with no year takes the Misc colour.** Misc is "no cluster placement detected", so "no year" is its
  natural generalization; it stays distinct from grayed, which means filtered out.
- **The reading views group by what the plot colours by** (the maintainer): in year mode the tooltip and the info
  panel group by year, so the group headers are the legend. One notion of "the current grouping" serves the plot,
  the tooltip and the panel alike, and the two views never disagree.
  - Within a group, the order stays the reference list's: authors, year, title.
  - **The year in front of each title in the tooltip, always**, in both modes: "(2022) Teaching a conversational
    agent…". A record with no year shows none. A strict improvement on its own (the maintainer).
  - **The info panel's cluster features become group features in year mode**: Ctrl+N, Ctrl+P and Ctrl+U go by
    year, F7 and the header buttons select a year. Its widget lookups, anchors and scroll-to-cluster key on the
    group rather than on the cluster ID.
  - **Purity has no per-year counterpart**, so a year header carries none.
- **Keywords per year are part of v1, not v0** (the maintainer). v0 groups by year with plain headers; the feature
  is not complete without the keywords.
  - **Computed at import time**, as the cluster keywords are: the years are fixed groups, known when the dataset
    is made. The word cloud is the other case, dynamic by design: it works on whatever the user selects, so it
    cannot be precomputed, and computing per-year keywords on display the same way would be too slow.
  - **Recent years can hold thousands of records**, which bears on the LLM keyworder's prompt size.
  - So they are stored in the dataset, which ties v1 to the dataset format (`TODO.md`, "Data file format"), or to
    an extra field in today's pickle.

## Open

Colouring by year's v1: per-year keywords at import, and how the LLM keyworder copes with a year of thousands of
records.

# Brief: filtering the Visualizer by date, and recolouring it

*Status: opened 2026-11-16, not yet discussed. The scope comes from the display-filters brief
([`done/visualizer-display-filters-brief.md`](done/visualizer-display-filters-brief.md)), which built the layer
these reuse and left them for later as a natural extension rather than part of its own work.*

The display filters gave each record a label it is shown with (`current_label_id`) and a visibility (shown,
grayed, hidden), kept in `raven.visualizer.displaystate`, separately from what the importer found. The two uses
so far are the purity cutoff and the Misc selector. This brief is for the next uses of the same layer.

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

## Open

Everything above, starting with which of the three comes first.

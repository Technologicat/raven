# Brief: the Visualizer's display filters, starting with cluster purity

*Status: done 2026-11-16. Decided 2026-11-11; steps 1–3 built and their layout tuned the same day; step 4's
display half (4a) 2026-11-13; its interaction half (4b) and the shared pill (5) 2026-11-16, all five
live-tested. The later work on the same layer has a brief of its own (the last section).*

The Visualizer is an interactive view on the data, so the user should be able to set aside what they take to be
noise, and later filter and recolour by other criteria (date, first author). This brief scopes the layer that
makes that possible and its first two uses: showing loose clusters as Misc, and graying out or hiding Misc.

## Why purity, and why the user chooses the cutoff

A cluster's *purity* is the share of its members' nearest neighbours in the embedding space that are members
too. `investigations/highdim-clustering/README.md`, *Cluster purity*, measured it across ten corpora: it spreads
broadly in every corpus, at a level set by the corpus, with no gap between tight and loose clusters. So no
cutoff can be derived, and none is shipped. The GUI states purity in words and lets the user choose.

## The steps

1. **The importer computes purity and stores it in the dataset**: per cluster, and for the whole dataset (on
   average over clustered records, the share of a record's nearest neighbours in its own cluster). For every
   clustering method, measured in the centered embedding space. A dataset without it (imported earlier) shows
   no purity readouts and no purity slider.
2. **Purity in words, in the GUI.** A dataset line: *"Clusters in this dataset overlap: on average, 35% of a
   record's closest neighbours are in its own cluster."* And per cluster, in the info panel's cluster header.
   Stop here for a look in the running app.
3. **The display-state layer, and the purity slider.**
   - Each entry keeps the importer's `cluster_id` and gains `current_label_id`, the label it is currently shown
     with, initialized from `cluster_id`. Every GUI consumer that reads `entry.cluster_id` today reads
     `current_label_id` instead.
   - **Record indices never move.** `data_idx` indexes the `sorted_*` arrays, sorted by the importer's label
     once at load; selection, hover, search and the info panel all hold it. A DPG scatter series takes plain
     lists, so a series is refilled from any set of points, contiguous or not.
   - The slider sets `current_label_id` to `cluster_id`, or to -1 (Misc) for a cluster whose purity is below
     the cutoff, then refills the series and triggers the existing info panel and tooltip rebuilds.
   - The slider sits in a row with the dataset purity readout, above the plot (which fills the rest of the
     column): the short readout on the left, "Purity cutoff for Misc" and the slider flush right, and what both
     mean in the slider's tooltip, a control being what a reader hovers. The tooltip's cluster headers carry
     each cluster's purity, as the info panel's do.
4. **Visibility: gray out or hide.** Each record is shown normally, grayed, or hidden. Two halves, a commit and a
   look in the running app each:
   - **4a, the display (built 2026-11-13).** `displaystate` carries a visibility per record. A Misc selector in the
     row (a combo, *shown / grayed / hidden*, with F4 / Shift+F4 and Ctrl+M as Raven's combos have) grays out or
     hides Misc, clusters below the purity cutoff included. Grayed records are one gray series created first, so
     beneath everything in colour. The pill and the plot title say how many records are in focus (grayed) or on
     screen (hidden), and the pill's click is Show all.
   - **4b, the interaction (next).** Out of hover, brush and lasso, "select this cluster" and the info panel:
     hidden records always, grayed ones unless a *grayed points selectable* checkbox is on, shown in the row only
     while something is grayed. Graying or hiding records that are selected drops them from the selection as one
     undoable step. Search counts what it cannot show, as *"7 matches, 3 hidden"*. About five sites across three
     modules (the hover lookup, the selection tools, search), each with a test. Settled 2026-11-16 (maintainer):
     - Unchecking *grayed points selectable* drops grayed records from the selection too, like graying does. The
       checkbox is always in the row, disabled while nothing is grayed, so that the row does not shift.
     - A grayed search match keeps the search highlight, being on screen; selecting the search results takes
       only what can be interacted with, and the count names hidden matches only.
     - **The undo history keeps the raw selection, and the filters apply on the fly** (after live testing):
       a filter change records nothing, so records grayed or hidden leave the selection and come back when
       shown again; the next selection change builds on what is shown. The first version recorded the drop as
       a step, and undo and redo then lost the records for good. Rejected too: undo entries carrying the
       filter state, which would roll the filters back along with an older plain selection change.
5. **A shared pill widget, in `raven.common.gui`** (maintainer, 2026-11-13): after the Visualizer's pill, built
   first to learn the shape of the common surface. Librarian's three status pills are app code too and would move
   onto it. Settled 2026-11-16 (maintainer):
   - `raven.common.gui.pill.Pill`: an icon, a button carrying the label, a self-sizing tooltip on the button,
     and a `shown` property the layout code reads instead of mirroring it in a flag of its own.
   - Three kinds: *ok* (green, steady), *alert* (caution colour, the icon pulsing: act on me) and *notice*
     (caution colour, steady: read this, nothing is wrong). The themes and the pulsator are shared, so pills
     alerting side by side pulse in phase.
   - The Visualizer's filter pill is a *notice*: a display filter is a state the user chose.
   - Librarian's transient notice in the composer row becomes a pill too, with no icon (`icon=None`).
   - Placement, the click action, the captions and the click flash stay with the app.

## The rules

- **"You see all the data" is the default, and leaving it is always visible.** This is Tufte's warning about a
  y axis that does not start at zero, generalized. Whenever anything is grayed or hidden:
  - a clickable pill under the plot says so: *"Showing 1,422 of 2,032 records · Show all"*, in the style of
    Librarian's status pills;
  - the plot title carries the same count, since it stays in view when the row does not.
  With everything shown, neither appears.
- **Show all resets every filter at once.** A single filter is undone through its own control.
- **Grayed is the default, hiding the stronger option.** Graying keeps the shape of the whole corpus in view.
- **Only what is in full colour can be interacted with**: hover, brush and lasso, the info panel. Graying out is
  a soft hide, for focusing on what is not grayed. A live toggle makes grayed points interactive again, for
  "wait, why is this one grayed out?".
- **Hidden points cannot be interacted with at all.** Hiding records that are selected takes them out of the
  selection, and showing them again brings them back (step 4b).
- **Search counts what it cannot show**: *"7 matches, 3 hidden"*, as the "…N more" boxes in Librarian's chat
  graph do. A search on a filtered view must not silently find less than the data holds.

## Later, on the same layer

Date-range filtering, recolouring by other criteria, and storing the embeddings in the dataset moved to their own
brief, [`visualizer-filter-by-date-and-recolour-brief.md`](../visualizer-filter-by-date-and-recolour-brief.md),
when this one closed (2026-11-16).

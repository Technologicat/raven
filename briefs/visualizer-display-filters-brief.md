# Brief: the Visualizer's display filters, starting with cluster purity

*Status: decided 2026-11-11. Steps 1–3 built and their layout tuned the same day; step 4 next. Date filtering
and recolouring are later work that reuses the layer from step 3.*

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
4. **Visibility: gray out or hide.** Each record is shown normally, grayed, or hidden. First use: a checkbox to
   gray out or hide Misc.

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
- **Hidden points cannot be interacted with at all.** Hiding records that are selected drops them from the
  selection, as one undoable step, so Ctrl+Z restores both.
- **Search counts what it cannot show**: *"7 matches, 3 hidden"*, as the "…N more" boxes in Librarian's chat
  graph do. A search on a filtered view must not silently find less than the data holds.

## Later, on the same layer

- **Date range filtering**: a minimum and maximum year, grayed or hidden outside it.
- **Recolouring** by other criteria (date, first author): another way of filling the series from
  `current_label_id`-style per-record state.
- **Storing the embeddings in the dataset** (they are in the importer's cache file today), the first time a
  feature needs them at load time.

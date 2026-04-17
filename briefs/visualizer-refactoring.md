# Visualizer Refactoring Notes

Living document for the Visualizer's monolithic-`app.py` split-up. Updated as modules get extracted.

## Progress

| # | Module | Status | Commit | Lines |
|---|---|---|---|---|
| 0 | `app_state.py` (shared namespace) | ✓ landed | 3896312 | 45 |
| 1 | `word_cloud.py` | ✓ landed | 3896312 | 259 |
| 2 | `selection.py` | ✓ landed | b525816 | 262 |
| 3 | `plotter.py` | ✓ landed | 158d52f | 266 |
| 4 | `annotation.py` (tooltip) | pending | — | ~350 |
| 5 | `info_panel.py` | pending | — | ~1450 |
| 6 | `entry_renderer.py` (dedup pass) | pending | — | — |

`app.py` line count: 4421 → 3795 (−626 lines / −14%) after three extractions.

Actual order differs from the brief's priority order: we went smallest-first (word_cloud → selection → plotter) to validate the extraction workflow on low-risk pieces before tackling the larger annotation/info-panel pair.

## Conventions and lessons from the first three extractions

### The `app_state` namespace

Unlike xdot_viewer and cherrypick (where shared state is a dict inside their single `app.py`), the Visualizer's split needs *cross-module* shared access for state that `app.py` historically held as bare module-level globals (`dataset`, `themes_and_fonts`, `bg` executor, `selection_data_idxs_box`, …).

`raven/visualizer/app_state.py` holds a single `unpythonic.env.env` instance, imported by every submodule as `from .app_state import app_state`. Every cross-module reference becomes `app_state.foo` — explicit, greppable, no circular imports. The docstring in `app_state.py` lists the currently-expected fields and their owners.

Entries get added as each extraction surfaces a new cross-module dependency (each extraction so far added 0–4 entries). Entries leave once a later class-wrap pass bundles related state under a single instance.

### Naming: drop the redundant prefix

Functions extracted into a module `foo.py` drop any `foo_` prefix they carried when they lived in `app.py` — the module namespace already provides that context. Examples:

- `update_selection` → `selection.update`
- `update_selection_highlight` → `selection.update_highlight`
- `keyboard_state_to_selection_mode` → `selection.keyboard_state_to_mode`
- `reset_plotter_zoom` → `plotter.reset_zoom`
- `load_data_into_plotter` → `plotter.load_dataset`
- `update_word_cloud` → `word_cloud.update`
- `show_save_word_cloud_dialog` → `word_cloud.show_save_dialog`

### DPG item access by string tag, not Python name

Items created with `tag="word_cloud_window"` in the GUI-layout section get referenced everywhere else by the string tag `"word_cloud_window"`, not the Python binding created by `as word_cloud_window`. Removes one coupling axis — extracted modules don't need to know about the Python variable that `app.py`'s GUI setup used.

### Module-local task managers, lazy init

`bgtask.TaskManager` needs `app_state.bg` (the thread-pool executor) which is created near the end of `app.py`'s startup. Rather than ordering the import to match (fragile) or requiring an explicit `init()` call, extracted modules that own a task manager create it lazily on first use (`word_cloud._get_task_manager`). Cheap, no ordering dependency.

### Tokenizer-based name migration tool

`briefs/tools_visualizer_rewrite_to_app_state.py` takes `(path, name1 name2 …)` and rewrites every bare-name occurrence of each `name` into `app_state.name`, using Python's `tokenize` module so occurrences inside string literals, comments, and f-string text parts are correctly left alone. Handles `global` declaration removal and skips `def NAME(…)` / `class NAME(…)` / kwarg-name positions (`foo(name=name)`).

**Gotcha**: does not track function-parameter shadowing (a `def f(dataset): …` whose body legitimately uses the parameter gets its parameter renamed too, breaking the function). Rename the parameter to something else before running the tool. Encountered once so far (`load_data_into_plotter(dataset)` → `(ds)`).

### `from . import X` ordering

New submodule imports go in alphabetical order in `app.py`'s import block, placed after `importer` (the other local import). Current sequence:

```python
from . import importer
from . import plotter
from . import selection
from . import word_cloud
```

## Remaining section map (post-plotter extraction; line numbers approximate as of 158d52f)

| Lines | Section |
|-------|---------|
| 1–95 | Imports, logging setup |
| 97–98 | *(plotter utilities — extracted)* |
| 100–171 | Modal window utilities, `is_any_modal_window_visible`, app_state registration |
| 173–213 | DPG bootup (context, fonts, themes, textures, viewport) |
| 215–247 | Dataset loading orchestration (`clear_background_tasks`, `reset_app_state`, `open_file`) |
| 249–405 | File dialogs (4 instances: open dataset, save word cloud, open BibTeX, save import) |
| 407–500 | BibTeX importer integration |
| 502–834 | Animations, search, live updates |
| 836–1407 | GUI layout creation |
| 1410–1471 | Shared helpers (`get_entries_for_selection`, `format_cluster_annotation`) |
| 1472–1796 | **Annotation tooltip** (`update_mouse_hover`, `_update_annotation`) |
| 1798–3242 | **Info panel** (~1450 lines — the biggest piece) |
| 3244–3335 | Help window |
| 3337–3371 | GUI resize handler |
| 3377–3723 | Event handlers |
| 3725–3795 | App lifecycle (exit cleanup, argparse, executor setup, render loop) |

## Tooltip / Info Panel: Shared Structure, Parallel Code

The tooltip (`_update_annotation`, ~300 lines) and info panel (`_update_info_panel`, ~720 lines) share a rendering vocabulary but implement it independently. This is the key duplication that `entry_renderer.py` should address — the parallel code differs mainly in *what* to render per item (compact vs. full), not in *how* to gather and organize the data.

| Concern | Tooltip | Info panel |
|---------|---------|------------|
| Data gathering | `get_entries_for_selection(data_idxs, max_n=10)` | `get_entries_for_selection(selection_data_idxs, max_n=100)` |
| Cluster headers | Title + keywords (text only) | Title + keywords + item count + nav buttons |
| Per-item display | Selection/search icons + title (text only) | 2x2 button group + title (plain or MD-highlighted) + abstract |
| Search highlighting | Icon color (bright/dim) | MD `<font color>` regex substitution + icon |
| Content management | Double-buffered group swap | Double-buffered group swap + scroll anchoring |
| Cancellation | Build number, `task_env.cancelled` | Build number, `task_env.cancelled` |
| Navigation metadata | `annotation_data_idxs` list | 6 dicts/lists for forward/reverse lookups |
| Report generation | None | Plain text + Markdown `StringIO` |
| Thread safety | `annotation_content_lock` (RLock) | `info_panel_content_lock` (RLock) |

## Refactoring Plan

**Goal**: Split `app.py` into a layered module structure analogous to Librarian.

Original priority-ordered list from the pre-refactoring plan, with completion status:

1. ✓ **`word_cloud.py`** — Generation, display, save (done smallest-first; was priority 6).
2. ✓ **`selection.py`** — Undo/redo stack, modes, highlight update (was priority 4).
3. ✓ **`plotter.py`** — Plotter utilities + dataset loading + highlight series (was priority 5).
4. **`annotation.py`** (tooltip) — Extract `update_mouse_hover`, `_update_annotation`, `clear_mouse_hover`, the annotation content lock, `annotation_data_idxs`. ~350 lines. **Future**: wrap in a class.
5. **`info_panel.py`** — Extract the info panel renderer, scroll anchoring, navigation, clipboard/report generation, search result tracking. ~1450 lines of self-contained complexity. Key globals to encapsulate: `info_panel_entry_title_widgets`, `info_panel_widget_to_data_idx`, `info_panel_widget_to_display_idx`, `info_panel_search_result_widgets`, `cluster_ids_in_selection`, `report_plaintext`, `report_markdown`, the content lock, and the build-number counter. **Future**: wrap in a class.
6. **`entry_renderer.py`** (shared abstraction for #4 and #5) — Extract `get_entries_for_selection`, `format_cluster_annotation`, and the shared item-rendering vocabulary (cluster header rendering, selection/search status icons, search fragment highlighting). Dedup pass over annotation and info_panel once both are in their own modules.
7. **`app.py`** — Reduced to thin orchestrator: DPG bootup, GUI layout, event handlers, search logic (~60 lines, coordinates info panel + tooltip), wiring, render loop.

Class-wrapping per-subsystem state (undo/redo instance for selection, info-panel instance, etc.) is a **separate later pass per module** — never combined with extraction.

**Constraints**: The DPG container stack is global and not thread-safe. The codebase already handles this by using explicit `parent=` on all `dpg.add_*` calls from background threads. This pattern must be preserved.

**Testing**: The Visualizer has zero automated tests; writing meaningful ones requires the refactor to be completed first (same strategy as raven-xdot-viewer). Each extraction is verified by running `raven-visualizer 00_stuff/datasets/100.pickle` and exercising the affected features by hand. After all modules are extracted, a comprehensive verification pass runs the checklist below before we start adding automated tests.

## Final verification checklist

After the last extraction lands, before starting automated test work, run `raven-visualizer 00_stuff/datasets/100.pickle` and walk through the following. Each group corresponds to one subsystem — a fault points directly at the module to investigate.

### Plotter

- [ ] Dataset loads and renders as a semantic map (scatter plot with cluster colours)
- [ ] Pan works (middle-drag on the plot)
- [ ] Zoom works (scroll wheel)
- [ ] Reset zoom button restores the full view
- [ ] Reset zoom hotkey works
- [ ] Mouse-hover highlights the hovered datapoint(s)
- [ ] File → open dataset loads a different `.pickle` file and repaints the plot
- [ ] After loading a second file: old cluster colour themes are cleaned up (no theme leak — DPG item count not growing by the number of clusters each reload)

### Selection

- [ ] Click in empty area with LMB-drag: replace-mode selection (new selection replaces old)
- [ ] Shift+LMB-drag: add to selection
- [ ] Ctrl+LMB-drag: subtract from selection
- [ ] Ctrl+Shift+LMB-drag: intersect with selection
- [ ] "Select visible" toolbar button works
- [ ] Undo button reverts to previous selection
- [ ] Redo button re-applies the undone change
- [ ] Undo history cleared when loading a new dataset
- [ ] Selecting clears on dataset reload

### Word cloud

- [ ] F10 (or the toolbar button) toggles the word cloud window
- [ ] Window shows correct keywords for the current selection
- [ ] Word cloud live-updates as selection changes (only if window is visible)
- [ ] Same-selection re-toggle: no recompute, just re-shows the window
- [ ] Save-as-PNG dialog opens
- [ ] Saved PNG file exists on disk and matches the displayed cloud

### Annotation / tooltip (after extraction)

- [ ] Tooltip appears on hover over a datapoint, showing cluster header + item titles
- [ ] Tooltip follows the mouse / repositions correctly at plot edges
- [ ] Tooltip disappears when the mouse leaves the plot
- [ ] Modal windows (help card, file dialogs) suppress the tooltip
- [ ] Tooltip double-buffering: no flicker during rapid mouse movement
- [ ] Search highlighting in tooltip matches info panel's

### Info panel (after extraction)

- [ ] Content area shows entries matching the current selection
- [ ] Scroll position is preserved across selection changes (scroll anchoring)
- [ ] Cluster navigation buttons (up/down between clusters) work
- [ ] Per-entry buttons work: copy to clipboard, scroll-to-in-plot, select, (others as applicable)
- [ ] "Copy report to clipboard" produces plain text and Markdown variants
- [ ] Search results are highlighted in the entry titles (MD `<font color>`)
- [ ] Search navigation (next/previous match) cycles correctly
- [ ] Current-item indicator follows scroll position

### Search

- [ ] Typing in the search field filters visible entries
- [ ] Matching datapoints highlighted on the plot (search-results scatter series)
- [ ] Ctrl+F (or hotkey) focuses the search field
- [ ] Clear-search button resets both the search field and the highlight

### Cross-subsystem interactions

- [ ] Selection change triggers info panel update
- [ ] Selection change triggers word cloud update (if visible)
- [ ] Selection change triggers tooltip refresh
- [ ] Opening a modal hides the tooltip; closing the modal restores it
- [ ] Opening a new dataset resets: selection, undo history, info panel, word cloud, plot, themes

### App-level

- [ ] F1 / help card opens and closes
- [ ] BibTeX importer integration: open a `.bib` file, import runs, output `.pickle` saves, progress bar + status updates
- [ ] Dragging the OS window corner resizes the GUI without layout breakage
- [ ] Clean exit: no traceback on window-close, tasks cancelled cleanly
- [ ] App starts with no arguments (blank dataset; opens to "no dataset loaded" state)
- [ ] App starts with a dataset argument (loads straight into it)

### Performance sanity checks (subjective)

- [ ] Info panel update latency on a 100-item dataset feels responsive
- [ ] Word cloud generation on a 100-item selection completes in a reasonable time
- [ ] No visible frame drops while scrolling the info panel
- [ ] Tooltip appears within one frame of hover

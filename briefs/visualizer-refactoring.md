# Visualizer Refactoring Notes

Living document for the Visualizer's monolithic-`app.py` split-up. Updated as modules get extracted.

## Progress

| # | Module | Status | Commit | Lines |
|---|---|---|---|---|
| 0 | `app_state.py` (shared namespace) | ✓ landed | 3896312 | 45 |
| 1 | `word_cloud.py` | ✓ landed | 3896312 | 259 |
| 2 | `selection.py` | ✓ landed | b525816 | 262 |
| 3 | `plotter.py` | ✓ landed | 158d52f | 266 |
| 4 | `annotation.py` (tooltip) | ✓ landed | 9526347 | 420 |
| 5 | `info_panel.py` | pending | — | ~1450 |
| 6 | `entry_renderer.py` (dedup pass) | pending | — | — |

`app.py` line count: 4421 → 3486 (−935 lines / −21%) after four extractions.

Actual order differs from the brief's priority order: we went smallest-first (word_cloud → selection → plotter) to validate the extraction workflow on low-risk pieces before tackling the larger annotation/info-panel pair.

## Conventions and lessons from the first four extractions

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
- `update_mouse_hover` → `annotation.update` (+ alias `app_state.update_mouse_hover = annotation.update` so `selection` and event handlers don't need to import `annotation`)
- `_update_annotation` → `annotation._render_worker`
- `clear_mouse_hover` kept as-is (describes the semantic effect — clears both the tooltip and the plot-highlight series; dropping `mouse_hover` would muddle that)
- `annotation_content_lock` → `annotation.content_lock`, `annotation_data_idxs` → `annotation.data_idxs`, `annotation_task_manager` → owned & lazy-created inside `annotation`, exposed via `annotation.clear_tasks(wait)`

### DPG item access by string tag, not Python name

Items created with `tag="word_cloud_window"` in the GUI-layout section get referenced everywhere else by the string tag `"word_cloud_window"`, not the Python binding created by `as word_cloud_window`. Removes one coupling axis — extracted modules don't need to know about the Python variable that `app.py`'s GUI setup used.

### Exception: swap-rebuilt widgets need a Python-variable handle

The tooltip's content group is created once at startup, then replaced on every hover update (build in a hidden group → show it → delete the old one → reassign the alias). For this "current slot" widget, the *string-alias* approach used elsewhere does not work reliably: `dpg.delete_item(alias_str)` followed by `dpg.set_item_alias(new_id, alias_str)` leaves the alias in a state where later lookups resolve to `0`, and the next `configure_item` call raises `SystemError: Item not found: 0`. What does work is `dpg.delete_item(old_id)` (by int widget ID) followed by `dpg.set_item_alias(new_id, alias_str)` — same as the info-panel content swap at `app.py:2822`. So `annotation.py` keeps a module-local `_current_group` holding the current widget ID, initializes it in `build_window()`, and reassigns it during the swap. The `"annotation_group"` alias is still set for debug-registry readability, but it's not on the hot path. (Also documented as DPG Pitfall #6 in `CLAUDE.md`.)

### Module-local task managers, lazy init

`bgtask.TaskManager` needs `app_state.bg` (the thread-pool executor) which is created near the end of `app.py`'s startup. Rather than ordering the import to match (fragile) or requiring an explicit `init()` call, extracted modules that own a task manager create it lazily on first use (`word_cloud._get_task_manager`). Cheap, no ordering dependency.

### Tokenizer-based name migration tool

`briefs/tools_visualizer_rewrite_to_app_state.py` takes `(path, name1 name2 …)` and rewrites every bare-name occurrence of each `name` into `app_state.name`, using Python's `tokenize` module so occurrences inside string literals, comments, and f-string text parts are correctly left alone. Handles `global` declaration removal and skips `def NAME(…)` / `class NAME(…)` / kwarg-name positions (`foo(name=name)`).

**Gotcha**: does not track function-parameter shadowing (a `def f(dataset): …` whose body legitimately uses the parameter gets its parameter renamed too, breaking the function). Rename the parameter to something else before running the tool. Encountered once so far (`load_data_into_plotter(dataset)` → `(ds)`).

### `from . import X` ordering

New submodule imports go in alphabetical order in `app.py`'s import block, placed after `importer` (the other local import). Current sequence:

```python
from . import annotation
from . import config as visualizer_config
from . import importer
from . import plotter
from . import selection
from . import word_cloud
```

## Remaining section map (post-annotation extraction; line numbers approximate as of 9526347)

| Lines | Section |
|-------|---------|
| 1–94 | Imports, logging setup |
| 95–96 | *(plotter utilities — extracted)* |
| 97–101 | *(selection management — extracted)* |
| 102–142 | Modal window utilities, `is_any_modal_window_visible`, app_state registration |
| 143–184 | DPG bootup (context, fonts, themes, textures, viewport) |
| 185–247 | Dataset loading orchestration (`clear_background_tasks`, `reset_app_state`, `open_file`) |
| 248–335 | File dialog init + "Open file" dialog |
| 336–504 | BibTeX importer integration (+ save-word-cloud / save-import / open-import dialog callbacks) |
| 505–841 | Animations, search (`search_string_box`, `update_search`), live updates |
| 842–1409 | GUI layout creation |
| 1410–1475 | Shared helpers (`get_entries_for_selection`, `format_cluster_annotation`) |
| 1476–1481 | *(annotation tooltip — extracted; `annotation.build_window()` call + `app_state.update_mouse_hover` publishing)* |
| 1482–2932 | **Info panel** (~1450 lines — the biggest piece) |
| 2933–3025 | Help window |
| 3026–3065 | GUI resize handler |
| 3066–3415 | Event handlers (mouse click / drag / wheel / move, keyboard) |
| 3416–3428 | App exit cleanup |
| 3429–3486 | App lifecycle (argparse, executor setup, initial file open, render loop) |

## Tooltip / Info Panel: Shared Structure, Parallel Code

The tooltip (`annotation._render_worker`, ~300 lines) and info panel (`_update_info_panel`, ~720 lines) share a rendering vocabulary but implement it independently. This is the key duplication that `entry_renderer.py` should address — the parallel code differs mainly in *what* to render per item (compact vs. full), not in *how* to gather and organize the data.

| Concern | Tooltip (`annotation`) | Info panel |
|---------|---------|------------|
| Data gathering | `app_state.get_entries_for_selection(data_idxs, max_n=10)` | `get_entries_for_selection(selection_data_idxs, max_n=100)` |
| Cluster headers | Title + keywords (text only) | Title + keywords + item count + nav buttons |
| Per-item display | Selection/search icons + title (text only) | 2x2 button group + title (plain or MD-highlighted) + abstract |
| Search highlighting | Icon color (bright/dim) | MD `<font color>` regex substitution + icon |
| Content management | Double-buffered group swap (`_current_group` int ID) | Double-buffered group swap + scroll anchoring |
| Cancellation | Build number, `task_env.cancelled` | Build number, `task_env.cancelled` |
| Navigation metadata | `annotation.data_idxs` list | 6 dicts/lists for forward/reverse lookups |
| Report generation | None | Plain text + Markdown `StringIO` |
| Thread safety | `annotation.content_lock` (RLock) | `info_panel_content_lock` (RLock) |

## Refactoring Plan

**Goal**: Split `app.py` into a layered module structure analogous to Librarian.

Original priority-ordered list from the pre-refactoring plan, with completion status:

1. ✓ **`word_cloud.py`** — Generation, display, save (done smallest-first; was priority 6).
2. ✓ **`selection.py`** — Undo/redo stack, modes, highlight update (was priority 4).
3. ✓ **`plotter.py`** — Plotter utilities + dataset loading + highlight series (was priority 5).
4. ✓ **`annotation.py`** (tooltip) — `annotation.update` (tasker + plot-highlight), `annotation._render_worker` (tooltip-content builder), `annotation.clear_mouse_hover`, `annotation.clear_tasks`, `annotation.build_window`, plus public `content_lock` / `data_idxs`; module-local `_current_group` (int widget ID), `_build_number`, `_task_manager` (lazy). 420 lines. **Future**: wrap in a class.
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

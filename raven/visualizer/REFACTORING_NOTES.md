# Visualizer Refactoring Notes

Pre-refactoring snapshot. Update this file as modules get extracted from `app.py`.

## app.py Section Map

Line numbers are approximate.

| Lines | Section |
|-------|---------|
| 1–96 | Imports, logging setup |
| 100–197 | Plotter utilities (`get_visible_datapoints`, `get_data_idxs_at_mouse`, `reset_plotter_zoom`) |
| 200–421 | Selection management (undo/redo stack, `update_selection` with modes replace/add/subtract/intersect, highlight update) |
| 426–455 | Modal window utilities (`enter_modal_mode`, `exit_modal_mode`) |
| 458–640 | Word cloud (generate from keywords, display, save as PNG) |
| 642–682 | DPG bootup (context, fonts, themes, textures, viewport) |
| 684–869 | Dataset loading (`parse_dataset_file`, sort by cluster, build kd-tree, load into plotter) |
| 871–1033 | File dialogs (4 instances: open dataset, save word cloud, open BibTeX, save import) |
| 1034–1127 | BibTeX importer integration (start/stop, status, progress bar) |
| 1129–1461 | Animations, search, live updates (`PlotterPulsatingGlow`, `CurrentItemControlsGlow`, `update_search`, dimmer overlay, current item tracking) |
| 1463–2034 | **GUI layout creation** (info panel header + navigation + content area, toolbar, search bar, plotter, word cloud window, BibTeX importer window) |
| 2037–2098 | Shared helpers (`get_entries_for_selection`, `format_cluster_annotation`) |
| 2099–2423 | **Annotation tooltip** (`update_mouse_hover`, `_update_annotation` worker, double-buffering, positioning) |
| 2425–3869 | **Info panel** (~1450 lines — the largest section). Subsections: content lock, navigation metadata, widget search predicates, programmatic scroll, scroll anchoring, search result navigation, cluster navigation, clipboard, `_update_info_panel` worker (~720 lines alone), report generation |
| 3871–3962 | Help window (hotkey table, terminology, search help) |
| 3964–3998 | GUI resize handler |
| 4004–4350 | Event handlers (mouse click/move/release/wheel, key down/up, hotkeys dispatcher) |
| 4352–4427 | App lifecycle (exit cleanup, argparse, executor setup, render loop) |

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

**Proposed modules** (priority order, high → low). The count is indicative — fewer modules is better, as long as each has a clear area of responsibility. Target ~700 lines per module as a guideline, not a hard limit; some can be longer when appropriate (e.g. lots of simple related code). Never split just because the line count was exceeded.

1. **`info_panel.py`** — Extract the info panel renderer, scroll anchoring, navigation, clipboard/report generation, search result tracking. This is ~1450 lines of self-contained complexity. Key globals to encapsulate: `info_panel_entry_title_widgets`, `info_panel_widget_to_data_idx`, `info_panel_widget_to_display_idx`, `info_panel_search_result_widgets`, `cluster_ids_in_selection`, `report_plaintext`, `report_markdown`, the content lock, and the build-number counter. **Future**: wrap in a class to support multiple instances. Do this as a separate step after extraction — don't change location and state management at the same time.

2. **`annotation.py`** (tooltip) — Extract `update_mouse_hover`, `_update_annotation`, `clear_mouse_hover`, the annotation content lock, `annotation_data_idxs`. ~350 lines. **Future**: wrap in a class (same reasoning as info panel).

3. **`entry_renderer.py`** (shared abstraction for #1 and #2) — Extract `get_entries_for_selection`, `format_cluster_annotation`, and the shared item-rendering vocabulary (cluster header rendering, selection/search status icons, search fragment highlighting). Both the tooltip and info panel would call into this. This is the key deduplication opportunity — the parallel code in the tooltip and info panel differs mainly in *what* to render per item (compact vs. full), not in *how* to gather and organize the data.

4. **`selection.py`** — Undo/redo stack, `update_selection` with modes, `selection_data_idxs_box`, highlight update. ~220 lines. Clean boundary: only dependency is on `dataset` (for coordinates) and DPG (for highlight scatter series). **Future**: wrap in a class (undo/redo state is a natural instance).

5. **`plotter.py`** — `get_visible_datapoints`, `get_data_idxs_at_mouse`, `reset_plotter_zoom`, `load_data_into_plotter`, dataset parsing, cluster color themes. ~280 lines.

6. **`word_cloud.py`** — Generation, display, save. ~170 lines. Already fairly self-contained.

7. **`app.py`** — Reduced to thin orchestrator: DPG bootup, GUI layout, event handlers, search logic (~60 lines, coordinates info panel + tooltip), wiring, render loop. Event handlers and search are pure consumers of the other modules and belong here (same pattern as Librarian).

**State management**: Currently all module-level globals. The cleanest refactoring path is to keep `dataset` as a module-level global (or a shared `env` namespace), and give each extracted module its own internal state, exposed via a public API (functions, not direct global access). This matches the Librarian pattern where e.g. `chattree` owns the `Forest` and `scaffold` calls into it. If the Visualizer needs persistent state in the future (e.g. last opened file, view settings), follow the Librarian's `appstate` pattern.

**Constraints**: The DPG container stack is global and not thread-safe. The codebase already handles this by using explicit `parent=` on all `dpg.add_*` calls from background threads. This pattern must be preserved.

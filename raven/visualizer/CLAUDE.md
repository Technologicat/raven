# Visualizer — CLAUDE.md

~6134 lines across 3 modules. Monolithic `app.py` needs splitting.

```
app.py       (4427 lines) — GUI app: plotter, info panel, tooltip, selection, search, word cloud, events
importer.py  (1286 lines) — BibTeX import pipeline: parse, embed, cluster, reduce, keywords, LLM summarize
config.py    (421 lines)  — Configuration-as-code (devices, import settings, stopwords, GUI settings)
```

No tests. Writing unit tests for extractable logic is a priority — both to catch regressions during refactoring and to define the API boundaries of the new modules. `importer.py` also serves as a standalone CLI app (`raven-importer`).

## How app.py Is Organized

The code is a deliberate script-style interleaving of function definitions, module-level state, and inline GUI creation. `@call` (from `unpythonic`) scopes temporaries that would otherwise pollute the module namespace. All state lives in module-level globals — `dataset`, `selection_data_idxs_box`, `search_string_box`, `info_panel_entry_title_widgets`, etc.

Lines tagged `# tag` indicate DPG widget tag references (searchable). All widget tags are string literals.

See `briefs/visualizer-refactoring.md` (project root) for the approximate section map of `app.py` (line-number index, pre-refactoring snapshot — update as modules get extracted).

## Key Patterns

**Double-buffered GUI updates**: Both the tooltip and info panel build new content in a hidden DPG group, then swap it in atomically (hide old, show new, `dpg.split_frame()`, delete old, reassign alias). These builds run in background threads — `split_frame()` must never be called from the main thread or it will deadlock the GUI loop. This avoids flickering and handles cancellation (partially-built content is deleted on cancel). Each build gets a unique build number for DPG tag uniqueness (`_buildN` suffix).

**Background task management**: Three `bgtask.TaskManager` instances (annotation, info panel, word cloud), all sequential-mode, sharing one `ThreadPoolExecutor`. Each supports pending-wait (debounce keyboard/mouse input), cancellation of pending tasks, and running-task completion before starting the next. See `raven.common.bgtask` for the full API.

**Selection with undo/redo**: Selection is a boxed `np.array` of indices into `sorted_xxx`. Undo stack is a list of snapshots. Modes: replace, add, subtract, intersect — chosen by keyboard modifier state (none, Shift, Ctrl, Ctrl+Shift). Mouse-draw select defers undo commits until mouse release.

**Scroll anchoring**: When the info panel rebuilds (ship-of-Theseus problem — completely new content), it records screen-y offsets of visible items before the swap, then finds the corresponding items in the new content and restores the scroll position. Multi-anchor: tries several visible items in case the topmost one isn't present after rebuild.

**Per-item button callbacks via closure factories**: `make_copy_entry_to_clipboard(item)`, `make_search_or_select_entry(entry)`, `make_select_cluster(cluster_id)`, `make_scroll_info_panel_to_cluster(display_idx)` — each returns a closure that captures the specific item.

**Widget search via predicates**: `user_data` on DPG widgets stores `(kind, data)` tuples. Predicate functions like `is_entry_title_container_group(item)` check the kind. `widgetfinder.binary_search_widget()` uses these for O(log n) lookups in the info panel widget list.

The tooltip (`_update_annotation`, ~300 lines) and info panel (`_update_info_panel`, ~720 lines) share a rendering vocabulary but implement it independently — see `briefs/visualizer-refactoring.md` for a detailed comparison. The tooltip also renders a help/legend section at its bottom.

## importer.py Structure

Pipeline architecture with caching. Stages: parse BibTeX → compute semantic vectors (cached per file+mtime) → HDBSCAN cluster (high-dim) → dimension reduce (t-SNE/UMAP) → cluster (2D) → extract keywords (NLP, cached) → collect cluster keywords (frequency or LLM) → optional LLM summarize → save dataset.

Uses `unpythonic.dyn` for injecting status update callbacks. Progress tracked via macro/microstep counter with ETA. Background execution via `bgtask.TaskManager`. Optionally connects to raven-server for NLP; falls back to local models via `mayberemote`.

## Importer Rework

Planned changes to the import pipeline (Nomic-embed migration, PCA preprocessing, outlier assignment, Procrustes alignment). See `briefs/visualizer-importer-rework.md` (project root) for details.

## Refactoring

**Goal**: Split `app.py` into a layered module structure analogous to Librarian. See `briefs/visualizer-refactoring.md` (project root) for the detailed plan (proposed modules, state management, constraints).

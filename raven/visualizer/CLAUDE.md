# Visualizer — CLAUDE.md

~7.1k lines across 13 modules. The `app.py` split has landed.

Sizes are rounded to two significant figures, measured **2026-09-01** — they are here for the shape of the
package, not as a figure to quote. Re-measure before quoting one. `python scripts/check_module_maps.py`
checks this table against the package, including whether every module is in it.

```
app.py            (~1.4k) — GUI app: window layout, event wiring, the main render loop
info_panel.py     (~1.6k) — the info panel: content build, scrolling, navigation, anchors
importer.py       (~1.5k) — BibTeX import pipeline: parse, embed, cluster, reduce, keywords, LLM summarize
annotation.py     (~500)  — datapoint annotations and their tooltips
config.py         (~440)  — Configuration-as-code (import settings, models, stopwords, GUI settings).
                            Compute devices live in `raven.client.config.devices` — one map for the
                            constellation, since these stages are `mayberemote` services
plotter.py        (~420)  — the scatter plot: dataset loading, plotter-space queries, the select brush
importer_gui.py   (~430)  — the importer's window, its two file dialogs, and its start/stop lifecycle
selection.py      (~270)  — selection state and the lasso/wand tools
word_cloud.py     (~250)  — word cloud rendering
entry_renderer.py (~190)  — per-entry rendering shared by panel and tooltip
search.py         (~150)  — the title scan, and the three GUI elements reporting its result
importer_cli.py    (~82)  — `raven-importer` entry point
app_state.py       (~58)  — top-level app state containers
```

**Under test as of 2026-09-01** — 289 tests over nine modules, which is every module the coverage plan
covers:

```
tests/test_selection.py       (39) — the four combine modes, undo/redo, scroll anchors, modifier keys
tests/test_plotter.py         (38) — the cluster sort, and the plotter-space queries
tests/test_importer.py        (37) — parsing and record recovery, cluster keywords, progress, the task
tests/test_info_panel.py      (34) — hotkey decisions, the clipboard, cluster navigation, widget kinds
tests/test_entry_renderer.py  (33) — grouping, the `max_n` budget, search highlighter compile and apply
tests/test_importer_gui.py    (31) — the filename tables, the start/stop decision table, the dialogs
tests/test_annotation.py      (30) — the item decoration table, and the guards on showing a tooltip
tests/test_word_cloud.py      (28) — the two render guards, keyword summing, cancellation, saving
tests/test_search.py          (19) — what counts as a match, and the three GUI elements reporting it
```

**Seven of the nine run in CI.** What keeps a module out is a module-level import of something
`.github/workflows/requirements-ci.txt` does not install, and the consequence is a skip that reads as a
pass rather than a failure. SciPy and `wordcloud` were added on 2026-09-01 for exactly this reason, which
brought `test_plotter.py` and `test_word_cloud.py` in. What is left out:

- **`test_info_panel.py`** — `info_panel` imports spaCy.
- **`test_importer.py`** — deliberately: it carries the `ml` marker, and CI runs `-m "not ml"`. The
  importer *is* the ML pipeline, so this one is not a gap. (It would also need scikit-learn and
  `mcpyrate`.)

Note torch *is* in CI — installed from PyTorch's own CPU wheel index by a separate line in the workflows.
That is why it is absent from the requirements file: `--index-url` inside a requirements file is a
file-level option in pip, so it would repoint every package in the list.

`importer_gui` is only clean because it reaches the pipeline through `_importer()` instead of importing it
at the top; `test_importer_gui.py` asserts that structurally, since the consequence of regressing it is a
skip rather than a failure.

**What is deliberately *not* covered, and why**, since a coverage figure hides it: the two big content
builds (`info_panel._update_info_panel`, `annotation._render_worker`) and everything that reads widget
geometry back — scroll positions, "which item is at the top of the panel", the anchoring that survives a
rebuild. Those need rendered frames, so they need a running app rather than a DPG context.
`config.py` needs no tests (configuration-as-code, carrying local overrides), and `app.py` is out of
scope: it lays out the GUI, wires events and animations, and boots up, so anything in there worth
testing is a sign it belongs in another module. `briefs/visualizer-test-coverage-brief.md` is the plan
and records both decisions.

The original rationale was to catch regressions *during* the refactor; that refactor landed without them,
so what these pin is the module boundaries the split created, before feature work leans on them.
`importer.py` also serves as a standalone CLI app (`raven-importer`).

**What the existing test modules establish, so the next one need not rediscover it:**

- **`importer` imports cleanly under pytest** — everything expensive is lazy, and the LLM connection is
  set up at import time only when the config asks for cluster keywords or summaries — so its functions
  can be driven against a `.bib` written into `tmp_path`. It does reach sklearn, torch and spaCy, which
  CI does not install, so the module is guarded with `pytest.importorskip("raven.visualizer.importer")`
  and marked `ml`; without the guard a module-level import failure is a *collection* error and turns the
  matrix red rather than skipping. `python scripts/check_ci_imports.py` is what reports that, and is
  worth running before pushing a new test module here.
- **`entry_renderer` needs no guard at all**, reaching nothing beyond numpy and `unpythonic`, so its
  tests run on every push.
- **A whole window can be built and read back headless.** `test_importer_gui.py` calls
  `importer_gui.build_window()` into a context with an unmapped viewport and then asserts against the real
  widgets — `get_item_label`, `get_item_configuration(...)["enabled"]`, `get_value`, and the table's rows via
  `get_item_children(table, slot=1)`. So a layout module wants a context rather than a stand-in, and only two
  things have to be faked: `app_state.themes_and_fonts` (a bare `env(icon_font_solid=0)` is enough for
  `bind_item_font`) and the `disablable_widget_theme` tag.
  - **What a context cannot answer is anything the last frame decided.** `is_item_visible` stays False until
    something renders — read `get_item_configuration(...)["show"]` instead, and patch `dpg.is_item_visible`
    where a test needs the window to *be* open. **Drawn geometry does not exist at all**: a widget's position
    and size come out of layout, and layout only happens against a mapped viewport. So configuration and item
    state are fair game; anything measured is not. `dpg-notes.md`, "Testing DPG code" has the full ceiling,
    including the one coordinate that survives headless and why it is not the one you want.
- **A module whose DPG use is a handful of calls does not need a context.** `selection` makes four kinds
  of DPG call, and `test_selection.py` monkeypatches a recording stand-in over the module's `dpg` binding
  that delegates everything else to the real toolkit — so assertions are about what the module asked the
  GUI to do, and the key constants are still DPG's own. Per-module contexts are fine where a test really
  needs one (`dpg-notes.md`, "Testing DPG code"); the point is that the heavier Visualizer modules can be
  approached one seam at a time rather than all-or-nothing.

## How app.py Is Organized

The code is a deliberate script-style interleaving of function definitions, module-level state, and inline GUI creation. `@call` (from `unpythonic`) scopes temporaries that would otherwise pollute the module namespace. All state lives in module-level globals — `dataset`, `selection_data_idxs_box`, `search_string_box`, `info_panel_entry_title_widgets`, etc.

Lines tagged `# tag` indicate DPG widget tag references (searchable). All widget tags are string literals.

See `briefs/done/visualizer-refactoring/visualizer-refactoring.md` (project root) for the approximate section map of `app.py` (line-number index, pre-refactoring snapshot — update as modules get extracted).

## Key Patterns

**Double-buffered GUI updates**: Both the tooltip and info panel build new content in a hidden DPG group, then swap it in atomically (hide old, show new, `dpg.split_frame()`, delete old, reassign alias). These builds run in background threads — `split_frame()` must never be called from the main thread or it will deadlock the GUI loop. This avoids flickering and handles cancellation (partially-built content is deleted on cancel). Each build gets a unique build number for DPG tag uniqueness (`_buildN` suffix).

**Background task management**: Three `bgtask.TaskManager` instances (annotation, info panel, word cloud), all sequential-mode, sharing one `ThreadPoolExecutor`. Each supports pending-wait (debounce keyboard/mouse input), cancellation of pending tasks, and running-task completion before starting the next. See `raven.common.bgtask` for the full API.

**Selection with undo/redo**: Selection is a boxed `np.array` of indices into `sorted_xxx`. Undo stack is a list of snapshots. Modes: replace, add, subtract, intersect — chosen by keyboard modifier state (none, Shift, Ctrl, Ctrl+Shift). Mouse-draw select defers undo commits until mouse release.

**Scroll anchoring**: When the info panel rebuilds (ship-of-Theseus problem — completely new content), it records screen-y offsets of visible items before the swap, then finds the corresponding items in the new content and restores the scroll position. Multi-anchor: tries several visible items in case the topmost one isn't present after rebuild.

**Per-item button callbacks via closure factories**: `make_copy_entry_to_clipboard(item)`, `make_search_or_select_entry(entry)`, `make_select_cluster(cluster_id)`, `make_scroll_info_panel_to_cluster(display_idx)` — each returns a closure that captures the specific item.

**Widget search via predicates**: `user_data` on DPG widgets stores `(kind, data)` tuples. Predicate functions like `is_entry_title_container_group(item)` check the kind. `widgetfinder.binary_search_widget()` uses these for O(log n) lookups in the info panel widget list.

The tooltip (`_update_annotation`, ~300 lines) and info panel (`_update_info_panel`, ~720 lines) share a rendering vocabulary but implement it independently — see `briefs/done/visualizer-refactoring/visualizer-refactoring.md` for a detailed comparison. The tooltip also renders a help/legend section at its bottom.

## importer.py Structure

Pipeline architecture with caching. Stages: parse BibTeX → compute semantic vectors (cached per file+mtime) → HDBSCAN cluster (high-dim) → dimension reduce (t-SNE/UMAP) → cluster (2D) → extract keywords (NLP, cached) → collect cluster keywords (frequency or LLM) → optional LLM summarize → save dataset.

Uses `unpythonic.dyn` for injecting status update callbacks. Progress tracked via macro/microstep counter with ETA. Background execution via `bgtask.TaskManager`. Optionally connects to raven-server for NLP; falls back to local models via `mayberemote`.

## Importer Rework

Planned changes to the import pipeline (Nomic-embed migration, PCA preprocessing, outlier assignment, Procrustes alignment). See `briefs/researchers-night/11_visualizer-importer-rework-brief.md` for details — note its item 1 now carries an undecided fork between `nomic-embed-text-v1.5` (shared image-text space) and `v2-moe` (multilingual).

## Refactoring

**Goal**: Split `app.py` into a layered module structure analogous to Librarian. See `briefs/done/visualizer-refactoring/visualizer-refactoring.md` (project root) for the detailed plan (proposed modules, state management, constraints).

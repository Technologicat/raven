# Raven - CLAUDE.md

## Project Overview
Local RAG-based research assistant constellation. Privacy-first, 100% local.

**Components:**
- **Visualizer** (`raven/visualizer/`): BibTeX topic analysis, semantic clustering, keyword extraction. The original app.
- **Librarian** (`raven/librarian/`): LLM chat frontend with tree-structured branching history, hybrid RAG, tool-calling, avatar integration.
- **Server** (`raven/server/`): Web API for GPU-bound ML models. All inference happens here.
- **Client** (`raven/client/`): Python bindings for Server API.
- **Avatar** (`raven/avatar/`): AI-animated anime character (THA3 engine, lipsync, cel animations, video postprocessor).
- **Common** (`raven/common/`): Shared utilities (video processing, audio, GUI widgets, networking).
- **Tools** (`raven/tools/`): CLI utilities (format converters, system checks).

## Architecture

### Server/Client Split
All ML inference in `raven/server/modules/`:
- `tts.py` - Kokoro TTS with phoneme timestamps (needed for lipsync)
- `stt.py` - Whisper speech recognition
- `embeddings.py` - Sentence embeddings (snowflake-arctic)
- `translate.py` - Neural machine translation
- `classify.py` - Sentiment/emotion classification, to control avatar's facial expression
- `sanitize.py` - Text cleanup (dehyphenation etc.)
- `natlang.py` - spaCy NLP analysis
- `websearch.py` - Web search tool for LLM
- `avatar.py`, `avatarutil.py`, `imagefx.py` - Avatar rendering pipeline

Client apps call Server via `raven/client/api.py`. Server can run on different machine.

### Librarian Architecture (Reference for Visualizer Refactoring)

~8000 lines total across 10 modules. Clean layered design, target style for the project.

#### Dependency Layers (bottom → top)

```
Layer 5 - Applications:     app.py (1010 lines), minichat.py (618 lines, deprecated)
Layer 4 - Controller:       chat_controller.py (1705 lines)
Layer 3 - Orchestration:    scaffold.py (556 lines)
Layer 2 - Backends:         llmclient.py (847 lines), hybridir.py (1170 lines)
Layer 1 - Utilities:        chatutil.py (698 lines), appstate.py (299 lines)
Layer 0 - Foundation:       config.py (357 lines), chattree.py (786 lines)
```

Each layer only imports from layers below it. No circular dependencies.

#### Module Details

- **`config.py`** — Configuration-as-code. Module-level constants + functions for dynamic content (system prompt, character card). Template variables (`user`, `char`, `model`, `weekday_and_date`) populated at runtime from LLM backend info. Imports `raven.config` (global) and `raven.common.video.colorspace`.

- **`chattree.py`** — `Forest` (in-memory) and `PersistentForest(Forest)` (JSON-backed). Nodes with parent pointers + children lists. Payload revisioning (multiple versions per node). Thread-safe (`threading.RLock`). Key ops: `create_node`, `linearize_up` (ancestor walk), `copy_subtree`, `delete_subtree`, `reparent_subtree`, `prune_unreachable_nodes`. Auto-saves via `atexit`. Format migration in `_upgrade()`.

- **`chatutil.py`** — Pure functions for message formatting, creation, and cleanup. Handles thought blocks (`<think>...</think>`) via regex — modes: `"discard"`, `"markup"`, `"keep"`. `scrub()` cleans LLM output (thought blocks, persona prefix, formatting quirks). `linearize_chat()` reconstructs linear history from tree. `create_chat_message()` builds OpenAI-format messages. Multiple markup targets (ANSI, Markdown, None).

- **`appstate.py`** — Loads/saves app state (JSON dict) + datastore (`PersistentForest`). On load: refreshes system prompt (overwrites stored version), refreshes greeting node, validates HEAD pointers, migrates old formats. Factory reset fallback if state is corrupted. State dict tracks: HEAD, toggle states (tools/docs/speculate/speech/subtitles), node IDs for system prompt and greeting.

- **`llmclient.py`** — Low-level LLM communication. `setup()` queries backend, builds `env` namespace with personas, tools, sampler params. `invoke()` streams via SSE, detects tool calls, supports stopping strings. `perform_tool_calls()` parses tool_calls JSON, validates, dispatches to registered entrypoints. `perform_throwaway_task()` for one-shot LLM tasks (e.g. keyword extraction). Progress via callbacks. Built-in tools: websearch.

- **`hybridir.py`** — `HybridIR` class: sliding-window chunking with overlap, BM25 keyword search (`bm25s`), ChromaDB vector search, reciprocal rank fusion, contiguous chunk merging. Pending-edit pattern (queue adds/updates/deletes, then `commit()`). `HybridIRFileSystemEventHandler` watches directory via watchdog, auto-commits changes. Background processing via `ThreadPoolExecutor` + `bgtask.TaskManager`. Tokenization: lowercase + lemmatize + stopword removal via spaCy (through raven-server).

- **`scaffold.py`** — High-level orchestration. `user_turn()` creates user message node. `ai_turn()` runs the full AI response pipeline: RAG search → context injection → LLM agent loop (interleave LLM + tool calls until done) → node creation. Anti-hallucination bypass: if RAG finds nothing and speculation is off, creates a "no match" response without calling LLM. Temporary context injects (RAG results, datetime, focus reminders) added at call time, not persisted. Rich event callbacks: `on_docs_start/done`, `on_llm_start/progress/done`, `on_tools_start/done`, `on_nomatch_done`, `on_prompt_ready`.

- **`chat_controller.py`** — GUI controller, the bridge between scaffold and DearPyGui. Classes: `DPGChatMessage` (base, thread-safe MD rendering), `DPGCompleteChatMessage` (stored nodes, with copy/reroll/continue/speak/edit/branch/delete/navigate buttons), `DPGStreamingChatMessage` (live-updating during generation), `DPGLinearizedChatView` (message container). `DPGChatController` wires everything: `chat_round()` → `user_turn()` + `ai_turn()` in background thread. Handles avatar emotion updates, TTS with lipsync, subtitles. Closures for button callbacks.

- **`app.py`** — Main GUI entry point. Two-column layout: left = chat panel + input controls, right = avatar panel + mode toggles. Bottom toolbar for global actions. Startup sequence: DPG init → server/LLM connection → state load → RAG load → GUI build → event loop. Hotkeys (Enter, Ctrl+N/G/S/R/U, F1/F8/F11). Animations: pulsating indicators, button flashes. Dynamic resize handler.

- **`minichat.py`** — CLI REPL. Same core as GUI (reuses scaffold, llmclient). GNU readline, special commands (`!clear`, `!docs`, `!reroll`, etc.). Deprecated but functional.

#### Key Design Patterns

**Layered separation**: Each module has a single clear responsibility. Data storage (`chattree`) knows nothing about LLM or GUI. Protocol (`llmclient`) knows nothing about GUI. Orchestration (`scaffold`) coordinates data+protocol but is GUI-agnostic. Controller (`chat_controller`) adapts scaffold events to DearPyGui. App (`app.py`) is layout and wiring only.

**Event-driven orchestration**: `scaffold.ai_turn()` takes ~15 optional callbacks. The controller passes closures that update GUI state (progress indicators, streaming text, avatar). This keeps scaffold reusable (minichat uses different callbacks for CLI output).

**Background threading**: LLM generation, tool calls, avatar rendering, and RAG indexing all run in background threads via `ThreadPoolExecutor` + `bgtask.TaskManager`. GUI stays responsive. Thread safety via `RLock` on shared data structures.

**Functional style**: Heavy use of closures (especially for GUI callbacks and event handlers). `unpythonic.env` for ad-hoc namespaces. Minimal OOP — classes used for GUI widgets and stateful objects (Forest, HybridIR), but logic functions are standalone.

**OpenAI-compatible message format**: Messages are `{"role": ..., "content": ...}` dicts, wrapped in Raven payloads that add `general_metadata` (timestamp, persona), `generation_metadata` (model, tokens, timing), and `retrieval` (RAG query/results).

#### Data Structures

**Chat node payload**:
```python
{"message": {"role": "user"|"assistant"|"system"|"tool", "content": "...", "tool_calls": [...]},
 "general_metadata": {"timestamp": ns, "datetime": "iso", "persona": "Aria"|None},
 "generation_metadata": {"model": "...", "n_tokens": N, "dt": secs, "status": "success"|"error"},
 "retrieval": {"query": "...", "results": [...]}}
```

**App state**:
```python
{"system_prompt_node_id": "...", "new_chat_HEAD": "...", "HEAD": "...",
 "tools_enabled": True, "docs_enabled": True, "speculate_enabled": False,
 "avatar_speech_enabled": True, "avatar_subtitles_enabled": True}
```

### Visualizer Architecture (Refactoring Target)

~6150 lines across 3 modules. Monolithic `app.py` needs splitting.

```
app.py       (4427 lines) — GUI app: plotter, info panel, tooltip, selection, search, word cloud, events
importer.py  (1295 lines) — BibTeX import pipeline: parse, embed, cluster, reduce, keywords, LLM summarize
config.py    (425 lines)  — Configuration-as-code (devices, import settings, stopwords, GUI settings)
```

No tests. `importer.py` also serves as a standalone CLI app.

#### How app.py Is Organized

The code is a deliberate script-style interleaving of function definitions, module-level state, and inline GUI creation. `@call` (from `unpythonic`) scopes temporaries that would otherwise pollute the module namespace. All state lives in module-level globals — `dataset`, `selection_data_idxs_box`, `search_string_box`, `info_panel_entry_title_widgets`, etc.

Lines tagged `# tag` indicate DPG widget tag references (searchable). All widget tags are string literals.

**Approximate section map** (line numbers are approximate):

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

#### Key Patterns in Visualizer

**Double-buffered GUI updates**: Both the tooltip and info panel build new content in a hidden DPG group, then swap it in atomically (hide old, show new, `dpg.split_frame()`, delete old, reassign alias). This avoids flickering and handles cancellation (partially-built content is deleted on cancel). Each build gets a unique build number for DPG tag uniqueness (`_buildN` suffix).

**Background task management**: Three `bgtask.TaskManager` instances (annotation, info panel, word cloud), all sequential-mode, sharing one `ThreadPoolExecutor`. Each supports pending-wait (debounce keyboard/mouse input), cancellation of pending tasks, and running-task completion before starting the next.

**Selection with undo/redo**: Selection is a boxed `np.array` of indices into `sorted_xxx`. Undo stack is a list of snapshots. Modes: replace, add, subtract, intersect — chosen by keyboard modifier state (none, Shift, Ctrl, Ctrl+Shift). Mouse-draw select defers undo commits until mouse release.

**Scroll anchoring**: When the info panel rebuilds (ship-of-Theseus problem — completely new content), it records screen-y offsets of visible items before the swap, then finds the corresponding items in the new content and restores the scroll position. Multi-anchor: tries several visible items in case the topmost one isn't present after rebuild.

**Per-item button callbacks via closure factories**: `make_copy_entry_to_clipboard(item)`, `make_search_or_select_entry(entry)`, `make_select_cluster(cluster_id)`, `make_scroll_info_panel_to_cluster(display_idx)` — each returns a closure that captures the specific item.

**Widget search via predicates**: `user_data` on DPG widgets stores `(kind, data)` tuples. Predicate functions like `is_entry_title_container_group(item)` check the kind. `widgetfinder.binary_search_widget()` uses these for O(log n) lookups in the info panel widget list.

#### Tooltip / Info Panel: Shared Structure, Parallel Code

The tooltip (`_update_annotation`, ~300 lines) and info panel (`_update_info_panel`, ~720 lines) share a rendering vocabulary but implement it independently:

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

Both also render a help/legend section at the bottom of the tooltip.

#### importer.py Structure

Pipeline architecture with caching. Stages: parse BibTeX → compute semantic vectors (cached per file+mtime) → HDBSCAN cluster (high-dim) → dimension reduce (t-SNE/UMAP) → cluster (2D) → extract keywords (NLP, cached) → collect cluster keywords (frequency or LLM) → optional LLM summarize → save dataset.

Uses `unpythonic.dyn` for injecting status update callbacks. Progress tracked via macro/microstep counter with ETA. Background execution via `bgtask.TaskManager`. Optionally connects to raven-server for NLP; falls back to local models via `mayberemote`.

#### Refactoring Plan

**Goal**: Split `app.py` into a layered module structure analogous to Librarian.

**Proposed modules** (priority order, high → low):

1. **`info_panel.py`** — Extract the info panel renderer, scroll anchoring, navigation, clipboard/report generation, search result tracking. This is ~1450 lines of self-contained complexity. Key globals to encapsulate: `info_panel_entry_title_widgets`, `info_panel_widget_to_data_idx`, `info_panel_widget_to_display_idx`, `info_panel_search_result_widgets`, `cluster_ids_in_selection`, `report_plaintext`, `report_markdown`, the content lock, and the build-number counter.

2. **`annotation.py`** (tooltip) — Extract `update_mouse_hover`, `_update_annotation`, `clear_mouse_hover`, the annotation content lock, `annotation_data_idxs`. ~350 lines.

3. **`entry_renderer.py`** (shared abstraction for #1 and #2) — Extract `get_entries_for_selection`, `format_cluster_annotation`, and the shared item-rendering vocabulary (cluster header rendering, selection/search status icons, search fragment highlighting). Both the tooltip and info panel would call into this. This is the key deduplication opportunity — the parallel code in the tooltip and info panel differs mainly in *what* to render per item (compact vs. full), not in *how* to gather and organize the data.

4. **`selection.py`** — Undo/redo stack, `update_selection` with modes, `selection_data_idxs_box`, highlight update. ~220 lines. Clean boundary: only dependency is on `dataset` (for coordinates) and DPG (for highlight scatter series).

5. **`search.py`** — `update_search`, `search_string_box`, `search_result_data_idxs_box`, fragment parsing, search field coloring. ~60 lines of logic, but interacts with info panel and tooltip via their public APIs.

6. **`plotter.py`** — `get_visible_datapoints`, `get_data_idxs_at_mouse`, `reset_plotter_zoom`, `load_data_into_plotter`, dataset parsing, cluster color themes. ~280 lines.

7. **`word_cloud.py`** — Generation, display, save. ~170 lines. Already fairly self-contained.

8. **`event_handlers.py`** — Mouse and keyboard handlers, hotkey dispatcher. ~360 lines. Depends on most other modules, but is a pure consumer (calls into them, not called by them).

9. **`app.py`** — Reduced to thin orchestrator: DPG bootup, GUI layout, wiring, render loop. Target ~800–1000 lines (mainly the declarative layout).

**State management**: Currently all module-level globals. The cleanest refactoring path is to keep `dataset` as a module-level global (or a shared `env` namespace), and give each extracted module its own internal state, exposed via a public API (functions, not direct global access). This matches the Librarian pattern where e.g. `chattree` owns the `Forest` and `scaffold` calls into it.

**Constraints**: The DPG container stack is global and not thread-safe. The codebase already handles this by using explicit `parent=` on all `dpg.add_*` calls from background threads. This pattern must be preserved.

### Common Subsystems
- `raven/common/video/` - Postprocessor, upscaler (PyTorch Anime4K), colorspace conversions, cel compositor
- `raven/common/audio/` - Player, recorder, codec (PyAV streaming)
- `raven/common/gui/` - Custom DearPyGui widgets (VU meter, GUI animation framework, messagebox)

### Vendored Dependencies
- `tha3/` - Talking Head Anime 3 neural network (avatar animation)
- `DearPyGui_Markdown/` - MD renderer (robustified for background threads, has one remaining URL highlight bug)
- `file_dialog/` - File dialog, extended (sortable, animated OK button, click twice when overwriting)
- `anime4k/` - PyTorch port of Anime4K upscaler (extracts kernels from GLSL), slightly cleaned up
- `kokoro_fastapi/` - Streaming audio writer for TTS over network
- `IconsFontAwesome6.py` - Icon font (note: outdated version)

## Code Style
All new and modified code must follow `raven-style-guide.md` (in the project root).

- Impure functional, Lispy (closures, `unpythonic` patterns)
- `unpythonic` used for: `env` (namespace), `Timer` (benchmarking), occasionally `@call`
- OOP where appropriate (GUI components, stateful objects)
- Config via Python modules (`config.py` files, not YAML/JSON)
- Type hints encouraged but not enforced everywhere
- Contract-style preconditions/postconditions would be useful, but mostly not implemented yet

## Key Patterns

### Message Tree (Librarian)
Messages stored as nodes with `parent` pointers. Branch = HEAD pointer to leaf.
Linear history reconstructed by walking ancestor chain. Branching is cheap.
Nodes have payload revisioning (multiple versions, like GitHub issue comment edits).

### Hybrid RAG
- Semantic: ChromaDB embeddings
- Keyword: bm25s (BM25 algorithm)
- Combined via reciprocal rank fusion
- Sliding-window chunking with overlap
- Contiguous chunk merging in results
- Pending-edit pattern: queue changes, then `commit()` to rebuild indices

### Avatar Lipsync
TTS (Kokoro) provides timestamped phonemes → mapped to mouth morphs → THA3 animator.
This coupling limits TTS engine choices (most don't expose timestamped phoneme data).

### DearPyGui App Structure (Librarian as Reference)
- **Layout**: Two-column (content panel + side panel) + bottom toolbar, all in a single `main_window`.
- **Resize**: `resize_gui()` callback recalculates sizes. Debounced via background task for expensive updates.
- **Themes**: Named themes for button variants, pulsating indicators. Created at module level.
- **Fonts**: Default + icon fonts (FontAwesome), loaded at startup.
- **Animations**: `PulsatingColor` (cyclic) and `ButtonFlash` (one-shot) via `raven.common.gui.animation` global `animator` singleton.
- **Hotkeys**: Registered via `dpg.add_key_*_handler` in a handler registry.
- **Background work**: `ThreadPoolExecutor` for async ops (LLM, avatar, RAG). GUI updates from callbacks. `dpg.split_frame()` for multi-frame ops.

### Startup Sequence (Librarian as Reference)
1. DPG init (context, fonts, themes, viewport)
2. Connect to raven-server (`raven.client.api`)
3. Connect to LLM backend (`llmclient.setup()`)
4. Load persistent state (`appstate.load()`)
5. Load domain-specific backends (RAG: `hybridir.setup()`)
6. Build GUI layout
7. Create controller(s)
8. Initial view render
9. Start DPG event loop

## Current State

### Well-structured (target style)
- `raven/librarian/` - Clean module separation (~8000 lines across 10 modules)

### Needs refactoring
- `raven/visualizer/app.py` - 4427 lines, monolithic, needs splitting into ~9 modules (see Visualizer Architecture above)
- `raven/visualizer/importer.py` - 1295 lines, pipeline architecture, lower priority but could benefit from stage separation

### Test coverage
- Minimal: only `raven/client/tests/test_api.py` and `raven/librarian/tests/test_hybridir.py`
- Visualizer has **zero tests**
- **Priority: expand test coverage** (especially before refactoring Visualizer)

## LLM Backend
Uses text-generation-webui with OpenAI-compatible API.
Recommended model: Qwen3-VL-30B-A3B (24GB+ VRAM) or Qwen3-VL-4B (8GB VRAM).

## Known Issues / TODOs
- Visualizer refactoring needed (see Visualizer Architecture section for plan)
- Test coverage very low (Visualizer has none)
- DearPyGui_Markdown URL highlight bug (threading-related, untracked)
- FontAwesome version outdated
- Hindsight integration pending (PDM dependency conflicts)
- TTS engine expansion limited by phoneme timestamp requirement
- DPG 2.0.0: Page Up/Down key constants changed (mysterious 517/518, `app.py:4316`)
- Many `# TODO: DRY duplicate definitions for labels` scattered through Visualizer `app.py`
- Annotation tooltip help section rebuilt every time (could be static with show/hide)
- `_update_info_panel` race condition: current item highlight sometimes doesn't update immediately after selection change
- Search match scrolling race condition: hammering the button can error out (`app.py:2978`)

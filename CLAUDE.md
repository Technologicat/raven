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
- `summarize.py` - Text summarization
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
- `raven/librarian/` - Clean module separation

### Needs refactoring
- `raven/visualizer/app.py` - ~4000 lines, monolithic, needs splitting

### Test coverage
- Minimal: only `raven/client/tests/test_api.py` and `raven/librarian/tests/test_hybridir.py`
- **Priority: expand test coverage**

## LLM Backend
Uses text-generation-webui with OpenAI-compatible API.
Recommended model: Qwen3-VL-30B-A3B (24GB+ VRAM) or Qwen3-VL-4B (8GB VRAM).

## Known Issues / TODOs
- Visualizer refactoring needed
- Test coverage very low
- DearPyGui_Markdown URL highlight bug (threading-related, untracked)
- FontAwesome version outdated
- Hindsight integration pending (PDM dependency conflicts)
- TTS engine expansion limited by phoneme timestamp requirement

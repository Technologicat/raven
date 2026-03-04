# Librarian — CLAUDE.md

~8000 lines total across 10 modules. Clean layered design, target style for the project.

## Dependency Layers (bottom → top)

```
Layer 5 - Applications:     app.py (1010 lines), minichat.py (618 lines, minimal reference client)
Layer 4 - Controller:       chat_controller.py (1705 lines)
Layer 3 - Orchestration:    scaffold.py (556 lines)
Layer 2 - Backends:         llmclient.py (847 lines), hybridir.py (1170 lines)
Layer 1 - Utilities:        chatutil.py (698 lines), appstate.py (299 lines)
Layer 0 - Foundation:       config.py (357 lines), chattree.py (786 lines)
```

Each layer only imports from layers below it. No circular dependencies.

## Module Details

- **`config.py`** — Configuration-as-code. Module-level constants + functions for dynamic content (system prompt, character card). Template variables (`user`, `char`, `model`, `weekday_and_date`) populated at runtime from LLM backend info. Imports `raven.config` (global) and `raven.common.video.colorspace`.

- **`chattree.py`** — `Forest` (in-memory) and `PersistentForest(Forest)` (JSON-backed). Nodes with parent pointers + children lists. Payload revisioning (multiple immutable versions per node). Thread-safe (`threading.RLock`). Key ops: `create_node`, `linearize_up` (ancestor walk), `copy_subtree`, `delete_subtree`, `reparent_subtree`, `prune_unreachable_nodes`. Auto-saves via `atexit`. Format migration in `_upgrade()`.

- **`chatutil.py`** — Pure functions for message formatting, creation, and cleanup. Handles thought blocks (`<think>...</think>`) via regex — modes: `"discard"`, `"markup"`, `"keep"`. `scrub()` cleans LLM output (thought blocks, persona prefix, formatting quirks). `linearize_chat()` reconstructs linear history from tree. `create_chat_message()` builds OpenAI-format messages. Multiple markup targets (ANSI, Markdown, None).

- **`appstate.py`** — Loads/saves app state (JSON dict) + datastore (`PersistentForest`). On load: refreshes system prompt (overwrites stored version), refreshes greeting node, validates HEAD pointers, fills missing settings with defaults, migrates old formats. Recovers gracefully from partial corruption (dangling HEAD, missing keys); factory reset only if datastore is genuinely empty. State dict tracks: HEAD, toggle states (tools/docs/speculate/speech/subtitles), node IDs for system prompt and greeting.

- **`llmclient.py`** — Low-level LLM communication. `setup()` queries backend, builds `env` namespace with personas, tools, sampler params. `invoke()` streams via SSE, detects tool calls, supports stopping strings. `perform_tool_calls()` parses tool_calls JSON, validates, dispatches to registered entrypoints. `perform_throwaway_task()` for one-shot LLM tasks (e.g. keyword extraction). Progress via callbacks. Built-in tools: websearch.

- **`hybridir.py`** — `HybridIR` class: sliding-window chunking with overlap, BM25 keyword search (`bm25s`), ChromaDB vector search, reciprocal rank fusion, contiguous chunk merging. Pending-edit pattern (queue adds/updates/deletes, then `commit()`). `HybridIRFileSystemEventHandler` watches directory via watchdog, auto-commits changes. Background processing via `bgtask.TaskManager`. Tokenization: lowercase + lemmatize + stopword removal via spaCy (through raven-server).

- **`scaffold.py`** — High-level orchestration; contains the agent loop. `user_turn()` creates user message node. `ai_turn()` runs the full AI response pipeline: linearize chat history (walk parent links from current HEAD to root) → RAG search → context injection → LLM agent loop (interleave LLM + tool calls until done) → node creation. Anti-hallucination bypass: if RAG finds nothing and speculation is off, creates a "no match" response without calling LLM. Temporary context injects (RAG results, datetime, focus reminders) added at call time, not persisted. Rich event callbacks: `on_docs_start/done`, `on_llm_start/progress/done`, `on_tools_start/done`, `on_nomatch_done`, `on_prompt_ready`.

- **`chat_controller.py`** — GUI controller, the bridge between scaffold and DearPyGui. Classes: `DPGChatMessage` (base, thread-safe MD rendering), `DPGCompleteChatMessage` (stored nodes, with copy/reroll/continue/speak/edit/branch/delete/navigate buttons), `DPGStreamingChatMessage` (live-updating during generation), `DPGLinearizedChatView` (message container). `DPGChatController` wires everything: `chat_round()` → `user_turn()` + `ai_turn()` in background thread. Handles avatar emotion updates; delegates TTS with lipsync and subtitles to `raven.client.avatar_controller.DPGAvatarController`. Closures for button callbacks.

- **`app.py`** — Main GUI entry point. Two-column layout: left = chat panel + input controls, right = avatar panel + mode toggles. Bottom toolbar for global actions. Help card (F1). Startup sequence: DPG init → server/LLM connection → state load → RAG load → GUI build → event loop. Hotkeys (Enter, Ctrl+N/G/S/R/U, F1/F8/F11). Animations: pulsating indicators, button flashes. Dynamic resize handler.

- **`minichat.py`** — Minimal CLI REPL. Same backend as GUI (reuses scaffold, llmclient). GNU readline, special commands (`!clear`, `!docs`, `!reroll`, etc.). Serves as a reference client for the backend API layers and works over bare SSH terminals.

## Key Design Patterns

**Layered separation**: Each module has a single clear responsibility. Data storage (`chattree`) knows nothing about LLM or GUI. Protocol (`llmclient`) knows nothing about GUI. Orchestration (`scaffold`) coordinates data+protocol but is GUI-agnostic. The stack branches at scaffold: the GUI path goes scaffold → `chat_controller` (adapts scaffold events to DearPyGui) → `app.py` (layout and wiring). The CLI path goes scaffold → `minichat.py` directly, using scaffold's callbacks for terminal output. This branching is the proof that the backend layers are truly frontend-agnostic.

**Event-driven orchestration**: `scaffold.ai_turn()` takes ~15 optional callbacks. The controller passes closures that update GUI state (progress indicators, streaming text, avatar). This keeps scaffold reusable (minichat uses different callbacks for CLI output).

**Background threading**: LLM generation, tool calls, avatar rendering, and RAG indexing all run in background threads via `raven.common.bgtask.TaskManager` (which wraps `ThreadPoolExecutor`). GUI stays responsive. Thread safety via `RLock` on shared data structures.

**Functional style**: Heavy use of closures (especially for GUI callbacks and event handlers). `unpythonic.env` for ad-hoc namespaces. Minimal OOP — classes used for GUI widgets and stateful objects (Forest, HybridIR), but logic functions are standalone.

**OpenAI-compatible message format**: Messages are `{"role": ..., "content": ...}` dicts, wrapped in Raven payloads that add `general_metadata` (timestamp, persona), `generation_metadata` (model, tokens, timing), and `retrieval` (RAG query/results).

## Data Structures

**Message tree**: Messages stored as nodes with `parent` pointers. HEAD points to the current node — typically a leaf, but can point to an internal node when branching from an existing message. Linear history reconstructed by walking ancestor chain from HEAD to root. Branching is cheap. Nodes have payload revisioning (multiple immutable versions, like GitHub issue comment edits).

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

## Hybrid RAG
- Semantic: ChromaDB embeddings
- Keyword: bm25s (BM25 algorithm)
- Combined via reciprocal rank fusion
- Sliding-window chunking with overlap
- Contiguous chunk merging in results
- Pending-edit pattern: queue changes, then `commit()` to rebuild indices

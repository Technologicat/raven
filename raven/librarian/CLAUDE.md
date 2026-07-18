# Librarian — CLAUDE.md

~11,800 lines total across 13 modules. Clean layered design, target style for the project.

## Dependency Layers (bottom → top)

```
Layer 5 - Applications:     app.py (1620 lines), minichat.py (672 lines, minimal reference client)
Layer 4 - Controller:       chat_controller.py (2406 lines)
Layer 3 - Orchestration:    scaffold.py (845 lines)
Layer 2 - Backends:         llmclient.py (1600 lines), hybridir.py (1397 lines)
Layer 1 - Utilities:        chatutil.py (990 lines), appstate.py (301 lines), imagestore.py (214 lines), textfilestore.py (129 lines)
Layer 0 - Foundation:       config.py (561 lines), chattree.py (931 lines), sidecarstore.py (105 lines)
```

Each layer only imports from layers below it. No circular dependencies.

## Module Details

- **`config.py`** — Configuration-as-code. Module-level constants + functions for dynamic content (system prompt, character card). Template variables (`user`, `char`, `model`, `weekday_and_date`) populated at runtime from LLM backend info. Also image-storage knobs (megapixel cap, keep-original toggle, staging dir) and the per-model VLM image-token-cost table. Imports `raven.config` (global) and `raven.common.video.colorspace`.

- **`chattree.py`** — `Forest` (in-memory) and `PersistentForest(Forest)` (JSON-backed). Nodes with parent pointers + children lists. Payload revisioning (multiple immutable versions per node). Thread-safe (`threading.RLock`). Key ops: `create_node`, `linearize_up` (ancestor walk), `copy_subtree`, `delete_subtree`, `reparent_subtree`, `prune_unreachable_nodes`. Also a content-addressed **attachment sidecar store** — stores any attachment bytes (images *and* text/PDF documents); the directory is `<datastore>.images/` for historical reasons (`store_sidecar`/`read_sidecar`/`sidecar_path`/`sidecar_dir`) — with mark-and-sweep GC (`prune_unreferenced_sidecars`/`unreferenced_sidecars`); the per-payload reference reader is injected at construction (`sidecar_extractor`), so chattree drives the traversal without knowing the message schema. Auto-saves via `atexit`. Format migration in `_upgrade()`.

- **`chatutil.py`** — Pure functions for message formatting, creation, and cleanup. **Content is a list of typed parts** (OpenAI multimodal schema: `{"type": "text"|"image_url"|"text_file", ...}` — `text_file` is Raven's own part type for an attached plain-text/PDF document), not a bare string. Constructors: `create_chat_message()` (string → single text part, with persona), `create_message_from_parts()` (multi-part). Accessors: `content_to_text()` (universal "give me the text" reader — assumes a parts list, raises on a stray string; skips `text_file` parts, whose text is folded in at wire-build time, not shown as message text), `text_content_part`/`image_content_part`/`text_file_content_part`, `normalize_content()` (the one str→parts migration converter). Handles thought blocks (`<think>...</think>`) via regex — modes: `"discard"`, `"markup"`, `"keep"`. `scrub()` cleans LLM output (thought blocks, persona prefix, formatting quirks). `linearize_chat()` reconstructs linear history from tree. Multiple markup targets (ANSI, Markdown, None).

- **`sidecarstore.py`** — Shared foundation for the two per-kind attachment stores (`imagestore`, `textfilestore`). Owns the `SIDECAR_SCHEME` (`"sidecar:"`) constant and the mechanics both kinds duplicate otherwise: `read_source_bytes()` (bytes-or-path ingestion), `base_provenance()` (the four common provenance keys — url/fetched_at/content_type/source — as a fresh dict the caller extends), `sidecar_filename_from_url()` (the scheme-strip both resolvers need, raising on a non-`sidecar:` URL), `content_part_sidecar_refs(payload, part_type)` (the GC mark-phase content-list walk, parameterized by part type). Stdlib-only, no `chatutil`/`chattree`/`config` deps — so it sits beneath every store. Exists so the two kind modules can't drift on the shared bits.

- **`imagestore.py`** — Image-specific sidecar store, on top of `sidecarstore`. Bridges the image codec/Lanczos resampler (`raven.common.image`), the sidecar file store (`chattree`), and the image-storage config. `store_image_as_sidecar()` (decode → downsample-to-cap → re-encode; original kept byte-for-byte to preserve EXIF/ICC when over cap) returns the `image_url` content-part + provenance metadata. `sidecar_url_to_data_url()` resolves a stored `sidecar:<filename>` URL to a `data:` URL for wire-send (a `sidecar:` URL never leaves the datastore). `sidecar_refs_in_payload()` is the GC mark-phase interpreter injected into `chattree` (image parts + the preserved-original `original_sidecar` refs).

- **`textfilestore.py`** — Document-specific sidecar store (plain text / PDF attachments), the file sibling of `imagestore` on the same `sidecarstore` base. `store_file_as_sidecar()` stores the document bytes *verbatim* (no transform, unlike an image) and returns a `text_file` content-part + provenance. A document has no native wire form: `sidecar_to_text()` extracts its plaintext on demand via `raven.common.docextract` (memoized on the content-addressed filename), and `llmclient` folds that into the message text at wire-build time — so any model can use an attached document, no vision capability required. `sidecar_refs_in_payload()` is the `text_file` GC mark interpreter; union it with `imagestore`'s when configuring a datastore's `sidecar_extractor`.

- **`appstate.py`** — Loads/saves app state (JSON dict) + datastore (`PersistentForest`). On load: refreshes system prompt (overwrites stored version), refreshes greeting node, validates HEAD pointers, fills missing settings with defaults, migrates old formats. Recovers gracefully from partial corruption (dangling HEAD, missing keys); factory reset only if datastore is genuinely empty. State dict tracks: HEAD, toggle states (tools/docs/speculate/speech/subtitles), node IDs for system prompt and greeting.

- **`llmclient.py`** — Low-level LLM communication. `setup()` queries backend, builds `env` namespace with personas, tools, sampler params; detects vision capability (`model_is_vlm` tri-state — True/False/None). `invoke()` streams via SSE through a single `StreamParser` emitting typed events (content / reasoning / tool-call), detects tool calls, supports stopping strings; serializes history for the wire and resolves `sidecar:` image URLs to `data:` just before send. `perform_tool_calls()` parses tool_calls JSON, validates, dispatches to registered entrypoints. `perform_throwaway_task()` for one-shot LLM tasks (e.g. keyword extraction). `count_tokens` + `image_token_cost` for the context-fill estimate. Progress via callbacks. Built-in tools: websearch, webfetch.

- **`hybridir.py`** — `HybridIR` class: sliding-window chunking with overlap, BM25 keyword search (`bm25s`), ChromaDB vector search, reciprocal rank fusion, contiguous chunk merging. Pending-edit pattern (queue adds/updates/deletes, then `commit()`). `HybridIRFileSystemEventHandler` watches directory via watchdog, auto-commits changes. Background processing via `bgtask.TaskManager`. Tokenization: lowercase + lemmatize + stopword removal via spaCy (through raven-server).

- **`scaffold.py`** — High-level orchestration; contains the agent loop. `user_turn()` creates the user message node (and stores any staged image attachments as sidecars, recording their provenance). `ai_turn()` runs the full AI response pipeline: linearize chat history (walk parent links from current HEAD to root) → RAG search → context injection → LLM agent loop (interleave LLM + tool calls until done) → node creation. Anti-hallucination bypass: if RAG finds nothing and speculation is off, creates a "no match" response without calling LLM. Temporary context injects (RAG results, datetime, focus reminders) added at call time, not persisted. Rich event callbacks: `on_docs_start/done`, `on_llm_start/progress/done`, `on_tools_start/done`, `on_nomatch_done`, `on_prompt_ready`.

- **`chat_controller.py`** — GUI controller, the bridge between scaffold and DearPyGui. Classes: `DPGChatMessage` (base, thread-safe MD rendering), `DPGCompleteChatMessage` (stored nodes, with copy/reroll/continue/speak/edit/branch/delete/navigate buttons), `DPGStreamingChatMessage` (live-updating during generation), `DPGLinearizedChatView` (message container). `DPGChatController` wires everything: `chat_round()` → `user_turn()` + `ai_turn()` in background thread. Handles avatar emotion updates; delegates TTS with lipsync and subtitles to `raven.client.avatar_controller.DPGAvatarController`. Closures for button callbacks.

- **`app.py`** — Main GUI entry point. Two-column layout: left = chat panel + input controls, right = avatar panel + mode toggles. Bottom toolbar for global actions. Help card (F1). Startup sequence: DPG init → server/LLM connection → state load → RAG load → GUI build → event loop. Hotkeys (Enter, Ctrl+N/G/S/R/U, F1/F8/F11). Animations: pulsating indicators, button flashes. Dynamic resize handler.

- **`minichat.py`** — Minimal CLI REPL. Same backend as GUI (reuses scaffold, llmclient). GNU readline, special commands (`!clear`, `!docs`, `!reroll`, etc.). Serves as a reference client for the backend API layers and works over bare SSH terminals.

## Key Design Patterns

**Layered separation**: Each module has a single clear responsibility. Data storage (`chattree`) knows nothing about LLM or GUI. Protocol (`llmclient`) knows nothing about GUI. Orchestration (`scaffold`) coordinates data+protocol but is GUI-agnostic. The stack branches at scaffold: the GUI path goes scaffold → `chat_controller` (adapts scaffold events to DearPyGui) → `app.py` (layout and wiring). The CLI path goes scaffold → `minichat.py` directly, using scaffold's callbacks for terminal output. This branching is the proof that the backend layers are truly frontend-agnostic.

**Event-driven orchestration**: `scaffold.ai_turn()` takes ~15 optional callbacks. The controller passes closures that update GUI state (progress indicators, streaming text, avatar). This keeps scaffold reusable (minichat uses different callbacks for CLI output).

**Background threading**: LLM generation, tool calls, avatar rendering, and RAG indexing all run in background threads via `raven.common.bgtask.TaskManager` (which wraps `ThreadPoolExecutor`). GUI stays responsive. Thread safety via `RLock` on shared data structures.

**Functional style**: Heavy use of closures (especially for GUI callbacks and event handlers). `unpythonic.env` for ad-hoc namespaces. Minimal OOP — classes used for GUI widgets and stateful objects (Forest, HybridIR), but logic functions are standalone.

**OpenAI-compatible message format**: Messages are `{"role": ..., "content": [...parts...]}` dicts — `content` is a **list of typed parts** (`{"type": "text", ...}` / `{"type": "image_url", ...}` / `{"type": "text_file", ...}`), the OpenAI multimodal schema (plus Raven's own `text_file` part for attached documents) used directly as Raven's internal representation. Read the text out with `chatutil.content_to_text()`; never index `content` as a string. Messages are wrapped in Raven payloads that add `general_metadata` (timestamp, persona, and `sidecars` provenance for attached images and documents), `generation_metadata` (model, tokens, timing), and `retrieval` (RAG query/results).

## Data Structures

**Message tree**: Messages stored as nodes with `parent` pointers. HEAD points to the current node — typically a leaf, but can point to an internal node when branching from an existing message. Linear history reconstructed by walking ancestor chain from HEAD to root. Branching is cheap. Nodes have payload revisioning (multiple immutable versions, like GitHub issue comment edits).

**Chat node payload**:
```python
{"message": {"role": "user"|"assistant"|"system"|"tool",
             "content": [{"type": "text", "text": "..."},                          # zero or more typed parts
                         {"type": "image_url", "image_url": {"url": "sidecar:<sha256>.png"}},
                         {"type": "text_file", "text_file": {"url": "sidecar:<sha256>.pdf", "name": "paper.pdf"}}],
             "reasoning_content": "...",  # optional: separated thinking trace (not in `content`)
             "tool_calls": [...]},
 "general_metadata": {"timestamp": ns, "datetime": "iso", "persona": "<char name>"|None,  # char name from config, not literally "Aria"
                      "sidecars": {"<filename>": {"url": ..., "source": ..., "stored_dimensions": [h, w], "name": ..., ...}}},
 "generation_metadata": {"model": "...", "n_tokens": N, "dt": secs, "status": "success"|"error"},
 "retrieval": {"query": "...", "results": [...]}}
```
`content` is always a parts list post-migration (a legacy string is upgraded once, at load). `general_metadata["sidecars"]` records provenance for attached images and documents, keyed by sidecar filename; `image_url` parts reference an image sidecar and `text_file` parts a document sidecar, both by `sidecar:<filename>` URL. An `image_url` is resolved to a `data:` URL on the wire (VLM-only); a `text_file` has no native wire form — its extracted text is folded into the message's text part at wire-build (`llmclient._serialize_history_for_wire`), so any model can use it.

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

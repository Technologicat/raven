# Librarian — CLAUDE.md

~21k lines across 19 modules.

**The layering is the part worth copying, and it is the only part.** Each layer imports downward and no
further, and that has held while the package grew — which is the property that made this the reference for
the rest of Raven. The module *sizes* are no longer exemplary and should not be read as endorsed: against
the project's ~700-line guideline, `chat_controller.py` is ~5.0k and `app.py` ~2.5k (and `llmclient.py` was ~2.8k until the tools moved out of it).
The growth is recent rather than gradual — `chat_controller.py` gained 44% in the three weeks to
2026-08-24, and the layer map below had been recording sizes 30–45% low for that whole period.

## Terminology: turn, round, exchange

Three words for three different things, and the code and the briefs used to disagree about which was which.
Settled 2026-08-07; this is the authoritative statement, and the code was moved to match it.

- **turn** — one participant's contribution. An assistant turn includes *the whole tool loop*: every model
  call, tool call and tool result, up to and including the reply the user reads. `scaffold.ai_turn` and
  `scaffold.user_turn` are named correctly under this reading, and always were.
- **round** — one iteration of the agent loop *within* an assistant turn: model call → tool calls → results.
  This is what `librarian_config.max_tool_call_rounds` caps.
- **exchange** — a user turn plus the assistant turn answering it. `chat_controller.chat_exchange` runs one.

The trap the old naming set: "round" meant *exchange* in the controller and *agent-loop iteration* in the
scaffold, so the same word named two different scopes one layer apart.

## Dependency Layers (bottom → top)

`python scripts/check_module_maps.py` checks this table against the package — sizes and, more usefully, whether
every module is here at all.

Sizes are rounded to two significant figures, measured **2026-08-27**. Rounded because the figure is here
for the *shape* — where the mass sits, which is what makes the refactoring calls legible — and an exact
number claims a precision that the next commit removes. The previous exact figures were 30–45% low by the
time anyone noticed. Re-measure before quoting one.

```
Layer 5 - Applications:     app.py (~2.5k), minichat.py (~730, minimal reference client),
                            indexer.py (~170, the `raven-indexer` CLI; also where the frontends get their
                            shared `open_document_store`)
Layer 4 - Controller/GUI:   chat_controller.py (~5.0k), cleanup_dialog.py (~410)
Layer 4 - Scripting:        agent.py (~500), the headless sibling of the controller
Layer 3 - Orchestration:    scaffold.py (~1.4k)
Layer 2 - Backends:         llmclient.py (~2.3k), llmtools.py (~990), hybridir.py (~1.9k)
Layer 1 - Utilities:        chatutil.py (~1.6k), appstate.py (~480), cleanup.py (~300),
                            imagestore.py (~270), textfilestore.py (~190)
Layer 0 - Foundation:       config.py (~930), chattree.py (~1.4k), sidecarstore.py (~150),
                            gguftokenizer.py (~350)
```

Each layer only imports from layers below it. No circular dependencies.

## Module Details

- **`config.py`** — Configuration-as-code. Module-level constants + functions for dynamic content (system prompt, character card). Template variables (`user`, `char`, `model`, `context_length`) populated at runtime from LLM backend info. No date among them: this text is built once at app start, so the current date is injected into the system message per turn instead (`scaffold.build_turn_prompt`). Also image-storage knobs (megapixel cap, keep-original toggle, staging dir) and the per-model VLM image-token-cost table. Imports `raven.config` (global) and `raven.common.video.colorspace`.

- **`chattree.py`** — `Forest` (in-memory) and `PersistentForest(Forest)` (JSON-backed). Nodes with parent pointers + children lists. Payload revisioning (multiple immutable versions per node). Thread-safe (`threading.RLock`). Key ops: `create_node`, `linearize_up` (ancestor walk), `copy_subtree`, `delete_subtree`, `reparent_subtree`, `prune_unreachable_nodes`. Also a content-addressed **attachment sidecar store** for any attachment bytes (images *and* text/PDF documents), split by what actually differs between the two classes: `Forest` owns the policy — content addressing, first-write-wins descriptions, mark-and-sweep GC (`prune_unreferenced_sidecars`/`list_unreferenced_sidecars`) — over a dict held in memory, and `PersistentForest` overrides only the members that touch the filesystem, keeping the bytes in `<datastore>.sidecars/`. So a `Forest` carries attachments too, which is what lets a script attach a paper without leaving a file behind. `store_sidecar`/`read_sidecar`/`sidecar_size`/`has_sidecar` work on either; `sidecar_path`/`sidecar_dir` raise on an in-memory store and have only GUI callers left. The per-payload reference reader is injected at construction (`sidecar_extractor`), so chattree drives the GC traversal without knowing the message schema — worth passing even in memory, since nothing else ever reclaims an in-memory sidecar. Auto-saves via `atexit`. Format migration in `_upgrade()`.

- **`chatutil.py`** — Pure functions for message formatting, creation, and cleanup. **Content is a list of typed parts** (OpenAI multimodal schema: `{"type": "text"|"image_url"|"text_file", ...}` — `text_file` is Raven's own part type for an attached plain-text/PDF document), not a bare string. Constructors: `create_chat_message()` (string → single text part, with persona), `create_message_from_parts()` (multi-part). Accessors: `content_to_text()` (universal "give me the text" reader — assumes a parts list, raises on a stray string; skips `text_file` parts, whose text is folded in at wire-build time, not shown as message text), `text_content_part`/`image_content_part`/`text_file_content_part`, `normalize_content()` (the one str→parts migration converter). Handles thought blocks (`<think>...</think>`) via regex — modes: `"discard"`, `"markup"`, `"keep"`. `scrub()` cleans LLM output (thought blocks, persona prefix, formatting quirks). `linearize_chat()` reconstructs linear history from tree. Multiple markup targets (ANSI, Markdown, None).

- **`sidecarstore.py`** — Shared foundation for the two per-kind attachment stores (`imagestore`, `textfilestore`). Owns the `SIDECAR_SCHEME` (`"sidecar:"`) constant and the mechanics both kinds duplicate otherwise: `read_source_bytes()` (bytes-or-path ingestion), `base_provenance()` (the four common provenance keys — url/fetched_at/content_type/source — as a fresh dict the caller extends), `sidecar_filename_from_url()` (the scheme-strip both resolvers need, raising on a non-`sidecar:` URL), `content_part_sidecar_refs(payload, part_type)` (the GC mark-phase content-list walk, parameterized by part type). Stdlib-only, no `chatutil`/`chattree`/`config` deps — so it sits beneath every store. Exists so the two kind modules can't drift on the shared bits.

- **`imagestore.py`** — Image-specific sidecar store, on top of `sidecarstore`. Bridges the image codec/Lanczos resampler (`raven.common.image`), the sidecar file store (`chattree`), and the image-storage config. `store_image_as_sidecar()` (decode → downsample-to-cap → re-encode; original kept byte-for-byte to preserve EXIF/ICC when over cap) returns the `image_url` content-part + provenance metadata. `sidecar_url_to_data_url()` resolves a stored `sidecar:<filename>` URL to a `data:` URL for wire-send (a `sidecar:` URL never leaves the datastore). `sidecar_refs_in_payload()` is the GC mark-phase interpreter injected into `chattree` (image parts + the preserved-original `original_sidecar` refs).

- **`textfilestore.py`** — Document-specific sidecar store (plain text / PDF attachments), the file sibling of `imagestore` on the same `sidecarstore` base. `store_file_as_sidecar()` stores the document bytes *verbatim* (no transform, unlike an image) and returns a `text_file` content-part + provenance. A document has no native wire form: `sidecar_to_text()` extracts its plaintext on demand via `raven.common.docextract` (memoized on the content-addressed filename), and `llmclient` folds that into the message text at wire-build time — so any model can use an attached document, no vision capability required. `sidecar_text_if_extracted()` asks the same question without paying it, returning `None` rather than extracting; that is how the GUI's context-fill readout stays off pypdf, extraction being seconds of work on a thread that also delivers keystrokes. `sidecar_refs_in_payload()` is the `text_file` GC mark interpreter; union it with `imagestore`'s when configuring a datastore's `sidecar_extractor`.

- **`cleanup.py`** — Datastore maintenance, the operation half. **No `dearpygui` in this module's import graph, and it must stay that way**: CI installs no GUI toolkit, so a `dpg` import here makes the whole module uncollectable and the operation untested. (Learned the hard way — the two halves started as one module and CI went red on the first push.) The GUI lives in `cleanup_dialog.py`; pure presentation logic (name shortening, size formatting) stays here. The operation is `prune_unreachable_nodes` → `prune_dead_links` → `prune_unreferenced_sidecars` → `save`, in that order — the sidecar sweep must run *after* the node prune or attachments held only by doomed nodes still look live. `preview_cleanup()` is the dry run for the pair, built on `chattree.list_unreachable_nodes()` plus `list_unreferenced_sidecars(excluding_nodes=...)`; without the exclusion it would under-report exactly the files the cleanup exists to reclaim. `describe_sidecar()` recovers a human-readable name from the sidecar's stored description (`get_sidecar_metadata`), falling back to the content hash. `rescue_to_staging()` copies an orphan out to `config.attachment_staging_dir` before the sweep (copy, not move; identical bytes under the same name are one rescue, different bytes get a ` (2)` suffix). Manual trigger only — no sweep on GUI exit.

- **`cleanup_dialog.py`** — `DPGCleanupDialog`, the GUI half of the above: dry-run preview (image grid + document list, both collapsed by default), per-item and bulk rescue-to-staging, commit. Thumbnails are letterboxed into uniform tiles and decoded on a background task, into a per-dialog texture registry whose tags carry a build counter (DPG frees deleted items lazily). A downsampled image and its preserved original are shown as one entry — `cleanup.preview_cleanup` does the folding, and `SidecarEntry.archival_filename` is what the open and rescue actions act on, matching the chat log.

- **`appstate.py`** — Loads/saves app state (JSON dict) + datastore (`PersistentForest`). On load: refreshes system prompt (overwrites stored version), refreshes greeting node, validates HEAD pointers, fills missing settings with defaults, migrates old formats. Recovers gracefully from partial corruption (dangling HEAD, missing keys); factory reset only if datastore is genuinely empty. State dict tracks: HEAD, toggle states (internet/docs/speech/subtitles), node IDs for system prompt and greeting. Two migration mappings sit next to the defaults: `_RETIRED_FLAGS` drops a flag that went away, `_RENAMED_FLAGS` carries a renamed one's *value* across.

- **`llmclient.py`** — Low-level LLM communication. `setup()` queries backend, builds `env` namespace with personas, tools, sampler params; detects vision capability (`model_is_vlm` tri-state — True/False/None). `invoke()` streams via SSE through a single `StreamParser` emitting typed events (content / reasoning / tool-call), detects tool calls, supports stopping strings; serializes history for the wire and resolves `sidecar:` image URLs to `data:` just before send. `count_tokens` + `image_token_cost` for the context-fill estimate, and the budgeting that fits attachments and fetched documents into what is left. Progress via callbacks. The tools themselves are `llmtools`, which this imports and re-exports.

- **`gguftokenizer.py`** — The exact-token-count tier, for backends that have no token-count endpoint. A llama.cpp-family backend serves a `.gguf`, which carries the model's vocabulary and merges, so any machine holding a copy of the model can count its tokens offline. `find_for_model` picks the file matching what the backend says it is serving (following symlinks, since a model archive shared between backends is usually a tree of them); `load` builds a `tokenizers.Tokenizer` from it. Two constructions, dispatched on the class GGUF names in `tokenizer.ggml.model`: the byte-level BPE most current families share (`gpt2`, e.g. the Qwens), and Gemma's SentencePiece-derived one (`gemma4`), where a space is a word mark rather than a byte, nothing pre-splits the text, and anything outside the vocabulary falls back to its 256 single-byte pieces. They share no part, which is why the dispatch is on the class rather than on a regex. **Nothing is trusted until something checks it**, because a wrong count here is wrong *and* unmarked — the readout drops its `~` when it believes a count is exact, so a bad tokenizer is worse than the estimate it replaces. `load` builds optimistically and then asks the backend to confirm: two short probes, compared by the *difference* between them so the chat template's framing cancels (comparing totals would need that framing to be known, which is the thing `prompt_size_report_looks_whole` exists because we don't). The backend arrives as a `text -> count` callable, so this module makes no network calls and its tests need none. That check subsumes the offline `_VERIFIED_CONSTRUCTIONS` list — it is about the model actually being served — and catches what careful name matching cannot: a file whose *name* matched while its vocabulary belongs to another model, which builds and round-trips perfectly. When the backend cannot be asked, the offline list decides. A name that only *partly* matches is never a match either way, a publisher prefix saying who packaged the file rather than whose vocabulary is in it. `gguf` and `tokenizers` are declared dependencies, imported inside `load` rather than at module scope; that is worth ~46 ms of import time against `llmclient`'s own 1282 ms, so treat it as a convenience (a broken install declines like any other unreadable file) rather than as a constraint to preserve.

- **`llmtools.py`** — The tools the LLM may call, and the registry that offers them. A tool is three things that have to agree: the JSON schema the model is shown (`TOOLS`), the function that runs when it asks (`TOOL_ENTRYPOINTS`), and whether it is offered this turn at all (`maybe_tool_names_for_turn`, over `DOCUMENT_TOOL_NAMES` / `NETWORK_TOOL_NAMES`). They live together because a schema naming an unregistered function, or a tool gated in neither group, is a mismatch that surfaces as the model calling something that does not exist. `perform_tool_calls` is the other half: validate what was asked against the registry, dispatch, and package the results as `role="tool"` messages. The allowlist gating is here rather than at the call site — `webfetch` is the one tool that visits an address the *model* chose, so its refusal path, its per-session approvals and the standing warning against memoizing it sit where a reader changing one sees the others. Every canned string a tool can return is a `CANONICAL_*` constant here, written to be acted on by the model rather than merely displayed. **`llmclient` imports this and not the reverse**; the single reach back is deferred, inside `fetch_document_wrapper`, for the token budget its answer has to fit. `llmclient` re-exports the public names, so `llmclient.TOOLS` and `llmclient.perform_tool_calls` still resolve — where the tools live is not the callers' business.

- **`hybridir.py`** — `HybridIR` class: sliding-window chunking with overlap, BM25 keyword search (`bm25s`), ChromaDB vector search, reciprocal rank fusion, contiguous chunk merging. Pending-edit pattern (queue adds/updates/deletes, then `commit()`). `HybridIRFileSystemEventHandler` watches directory via watchdog, auto-commits changes. Background processing via `bgtask.TaskManager`. Tokenization: lowercase + lemmatize + stopword removal via spaCy (through raven-server).

- **`scaffold.py`** — High-level orchestration; contains the agent loop. `user_turn()` creates the user message node (and stores any staged image attachments as sidecars, recording their provenance). `ai_turn()` runs the full AI response pipeline: linearize chat history (walk parent links from current HEAD to root) → RAG search → context injection → LLM agent loop (interleave LLM + tool calls until done) → node creation. Anti-confabulation: with speculation off, the reply records whether it had grounding material (`generation_metadata["grounded"]`), which the frontends surface as a badge — it does not withhold the reply, since the material's absence cannot distinguish a corpus question from a general-knowledge aside. Temporary context injects (RAG results, datetime, reminders) added at call time, not persisted. Rich event callbacks: `on_docs_start/done`, `on_llm_start/progress/done`, `on_tools_start/done`, `on_prompt_ready`.

- **`agent.py`** — The scripting surface: the agent loop with the events turned inside out. `turn()` runs one assistant turn — optionally posting the user's message first, and building an in-memory `chattree.Forest` if not given one — and returns a `TurnRecord` instead of a node id. It takes **no callbacks**: what a frontend gets as events, a script gets as the record. `describe_turn()` builds the same record by walking a stored branch, which is how a saved `PersistentForest` from an earlier batch is analyzed with the same counting the live path uses. The record fixes in one place the walk every probe used to write out longhand — notably the vocabulary (a **round** is one assistant message asking for tools, however many *calls* it asks for) and the span (one turn, not the whole branch). Two defaults differ from the apps on purpose: `internet_enabled=False`, because a run with tools enabled makes real network calls; and the automatic search runs only when a retriever is supplied. Per-run overrides are fields on `llm_settings` (see `chatutil.default_formatters`), never module globals. `record.generation is None` is how an unattended batch tells a backend failure from a reply — `ai_turn` materializes the failure as an assistant message, which is right for a watching human and invisible at 3 a.m. `use_character_card=False` with `tools_enabled=False` is the other shape this surface offers: no character, no greeting, no per-turn injects, and the character-independent half of the configuration (`config.setup_system_prompt`) as the only system text — which Raven ships empty, so the task instruction is then the root of the chat. That is what the batch extraction tools use (`papers.pdf2bib`, `visualizer.importer`), whose outputs are parsed rather than read.

- **`chat_controller.py`** — GUI controller, the bridge between scaffold and DearPyGui. Classes: `DPGChatMessage` (base, thread-safe MD rendering), `DPGCompleteChatMessage` (stored nodes, with copy/reroll/continue/speak/edit/branch/delete/navigate buttons), `DPGStreamingChatMessage` (live-updating during generation), `DPGLinearizedChatView` (message container). `DPGChatController` wires everything: `chat_exchange()` → `user_turn()` + `ai_turn()` in background thread. Handles avatar emotion updates; delegates TTS with lipsync and subtitles to `raven.client.avatar_controller.DPGAvatarController`. Closures for button callbacks.

- **`app.py`** — Main GUI entry point. Two-column layout: left = chat panel + input controls, right = avatar panel + mode toggles. Bottom toolbar for global actions. Help card (F1). Startup sequence: DPG init → server/LLM connection → state load → RAG load → GUI build → event loop. Hotkeys (Enter, Ctrl+N/G/S/R/U, F1/F8/F11). Animations: pulsating indicators, button flashes. Dynamic resize handler.

- **`indexer.py`** — `raven-indexer`: build or refresh the RAG index over a documents directory, with no GUI. The indexing itself is `hybridir`'s — `setup` already reconciles the index against the directory on construction, and `commit` already reports progress — so what this adds is the part a library used only by long-lived apps never needed: a way to *wait* for the work and then exit. Two things follow from that. Indexing stops being coupled to a runnable desktop frontend, so a GUI-side breakage cannot block a batch run; and `-d/--db-dir` makes swapping corpora a command rather than a ritual, which is what makes measuring against a second corpus thinkable. `open_document_store` is the reusable half and is public for exactly that reason: the frontends each carried their own copy of the same six-argument `hybridir.setup` call, and a third copy is how three call sites drift. Note "refresh" reconciles rather than rebuilds — a corrupt index is fixed by deleting the index directory, not by running this again.

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
`content` is always a parts list post-migration (a legacy string is upgraded once, at load). `general_metadata["sidecars"]` records provenance for attached images and documents, keyed by sidecar filename; `image_url` parts reference an image sidecar and `text_file` parts a document sidecar, both by `sidecar:<filename>` URL. An `image_url` is resolved to a `data:` URL on the wire (VLM-only); a `text_file` has no native wire form — its extracted text is folded into the message's text part at wire-build (`llmclient.serialize_history_for_wire`), so any model can use it.

**App state**:
```python
{"system_prompt_node_id": "...", "new_chat_HEAD": "...", "HEAD": "...",
 "internet_enabled": True, "docs_enabled": True,
 "avatar_speech_enabled": True, "avatar_subtitles_enabled": True}
```

## Where the data lives on disk

Everything sits under `librarian_config.llmclient_userdata_dir` — `~/.config/raven/llmclient/` by default. The paths are config (`llm_datastore_file`, `llm_state_file`), read by both frontends, so the GUI and the CLI share one chat history by construction rather than by two literals happening to agree:

| Path | What |
|---|---|
| `chat.json` | the chat node datastore (`chattree.PersistentForest`). Was `data.json` before 0.2.9; `appstate.load` adopts a file by that name if the configured one is absent *and* the file reads as a forest — the old name is generic enough that going by the name alone would claim a stranger's file |
| `chat.sidecars/` | attachment sidecars for that datastore — content-addressed `<sha256>.<ext>` plus `<file>.meta.json` descriptions. Derived from the datastore filename via `with_suffix(chattree.SIDECAR_SUFFIX)`, so it tracks whatever the datastore is called, which is what keeps two datastores' sidecars apart and therefore the GC correct. Was `.images/`, from when images were the only attachment kind; `PersistentForest` renames it on load |
| `state.json` | app state — HEAD, the system-prompt/greeting node IDs, toggle states |
| `documents/` | the docs-DB drop folder (`llm_docs_dir`); files landing here are ingested into RAG |
| `rag_index/` | the built RAG index (`llm_database_dir`) |
| `api_key.txt` | optional; used if present |

**`chat.json`'s on-disk shape is a flat mapping of node ID → node, with no wrapper key.** Not `{"nodes": {...}}` — the top level *is* the node dict, even though the in-memory attribute is `PersistentForest.nodes`. Node IDs look like `gensym#forest-node:<uuid>`. Within a node, `node["data"]` maps revision → payload, so a script that wants "the message" has to pick a revision (or iterate all of them, which is what the sidecar GC mark phase does). Worth knowing before writing any ad-hoc script against the file: assuming a `"nodes"` key silently yields zero nodes rather than an error.

## Hybrid RAG
- Semantic: ChromaDB embeddings
- Keyword: bm25s (BM25 algorithm)
- Combined via reciprocal rank fusion
- Sliding-window chunking with overlap
- Contiguous chunk merging in results
- Pending-edit pattern: queue changes, then `commit()` to rebuild indices

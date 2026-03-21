# Raven TODO

Covers the full Raven constellation: Visualizer, Librarian, Server, Avatar, XDot Viewer, and shared tooling.

Priority tiers: **[High]** | **[Medium]** | **[Low]** | **[Parked]**

Items marked **[Verify]** should be checked against the current codebase in a CC session before implementing.


---

## Cross-cutting

- **[High]** HF hub: document the env vars that prevent hub checks on Raven startup (for privacy and faster startup). Add recommendation to server docs. Currently not written down anywhere in the project.
  - `HF_HUB_OFFLINE=1` — forces huggingface_hub to use only locally cached models, no network requests at all.
  - `HF_HUB_DISABLE_TELEMETRY=1` — stops telemetry pings only.

- **[High]** Neural reranker for HybridIR: add a reranker stage to avoid needing large k values (k=100 style workarounds). Since we maintain our own HybridIR backend, we can power it up properly.

- **[High]** Revisit logging system: library modules should not reconfigure the logger (verify exact behavior against Python `logging` stdlib docs, but currently each module sets the log level, which is the entrypoint's responsibility). Move logging configuration to entrypoints only. Add a "detailed debug" level at that time for particularly spammy-but-useful log lines (e.g. `SmoothScrolling.render_frame`, `_managed_task`, `binary_search_item`).

- **[Medium]** Flash the search field when focused by hotkey. Currently affects Visualizer main window, fdialog component, and XDot Viewer. Generalize `ButtonFlash` for GUI elements other than buttons.

- **[Medium]** `vis_data` → `entries` rename across the whole constellation, including importers and BibTeX tooling in `raven.tools`.

- **[Medium]** Visualizer↔Librarian integration: allow querying Librarian for documents (set as RAG sources) that are currently selected in Visualizer. Apps communicate over the local network. Core workflow: "show me the cluster structure around this topic" → "now let me drill into those papers conversationally."
  - IPC design: ZeroMQ pub/sub over localhost (or localhost websockets, since raven-server already has a web API layer). IPC is optional — if both apps are running, use it; if not, graceful degradation. Neither app should depend on the other being present.
  - Bidirectional stretch goal: Librarian highlights search results on Visualizer's semantic map. Allows vague natural-language queries to find papers related to a given topic.

- **[Medium]** Large files (images, audio, full PDFs) should be stored separately from the main datastore and linked, not embedded. Currently no large files are used; this is a note for when blob support is added. Applies to both Visualizer dataset files and the Librarian document DB, and to large text files too.

- **[Low]** `deviceinfo` at app bootup should report whether the reported device configuration is for the client or for the server. Add a parameter.


---

## Visualizer

### Refactor (do first)

- **[High]** `raven.visualizer.app` refactor: currently a god object (~4k SLOC). Extracting the info panel (~2k SLOC) is the main candidate. The info tooltip is another good candidate and shares many data sources with the info panel. Prerequisite for most further Visualizer feature work.

- **[Medium]** FP refactor: keep app state in top-level containers, pass in/out explicitly. More FP-idiomatic and facilitates adding unit tests. Do after the `app.py` refactor.


### Search and data access

- **[High]** Author search: show full author list. GUI must be search-aware — when search is active, highlight where the match appears in a long author list (e.g. a 200-name list starting with "Aaltonen" and ending with "Virtanen"; user searching for "Smith" needs to see where it is, not just that it matched).

- **[High]** DOI: record DOI in BibTeX importer; show DOI per item in info panel; per-item button to open official webpage (`https://dx.doi.org/...`); export list of DOIs/URLs for fulltext automation.

- **[Medium]** Fragment search across multiple fields (author, year, abstract, ...); configurable which fields to search. Add checkboxes and a select/unselect-all button below the search bar. Note: the highlighter currently only processes titles and is slow — may not be able to highlight in abstracts without performance work.

- **[Medium]** Semantic orienteering: embed user-typed text, dimension-reduce it, highlight the resulting virtual datapoint in the plotter. Later: support user-given BibTeX entry or PDF file as input.

- **[Medium]** Select cluster by number: useful complement to the wand button for datasets with few clusters.

- **[Medium]** Add GUI filter/search in the help window hotkeys list: incremental fragment search by key or action.

- **[Low]** Word boundary mark (`\b`) for search. UX: what character should the user type as a word boundary?

- **[Low]** BUG: Search result highlight: "Can a" → highlights whole word "Can", then highlights "a" inside it, breaking the outer highlight. Difficult to fix.


### Import and data pipeline

- **[High]** Publish a ready-made dataset for quick-start demo (e.g. AI papers from arXiv, fully public).

- **[High]** HybridIR integration (1): spawn an in-memory `Forest` + HybridIR instance over the BibTeX data for full-text search. Once full BibTeX records are saved in the dataset, this is mostly scripting.

- **[Medium]** Procrustes alignment for incremental dataset updates: when adding new papers to an existing map, use SVD-based rotation from correspondence points to align the new embedding space with the existing one, preserving spatial memory. Document assumptions and limitations (works well when new data is from the same semantic region as the existing dataset; consider a fallback for the unrelated-dataset case). Also bundles Nomic-embed integration and PCA preprocessing.
  - Novelty detection falls out naturally as a byproduct: items with large Procrustes residuals, or items with no close neighbors in the existing dataset, are flagged as novel.
  - UX: add a field to the BibTeX import dialog to specify a base dataset to add to; add a corresponding option to the `raven-importer` CLI tool.

- **[Medium]** BibTeX export: keep full original BibTeX entries in the dataset; export selected items as BibTeX. Check whether the importer already preserves full entries or discards them.

- **[Medium]** Excel import to BibTeX (CSV importer already exists; minor convenience upgrade for pilot users without the CSV workaround). Needs Windows testing.

- **[Medium]** Importer: check that there is at least one item before proceeding; throw a sensible error message if not (currently crashes silently). Triggered by BibTeX files with the Author field missing.

- **[Medium]** Pre-filtering at import time (e.g. by year). Also: add option to re-scan a BibTeX database for new entries added since last import (import only new items, report them as such).

- **[Medium]** BibTeX-encoded umlauts and verbatim braces: handle `{\"o}` → ö etc.; drop BibTeX verbatim braces (`{GPU}` → `GPU`). **[Verify]** against current codebase.

- **[Medium]** More flexible data import: configurable which fields to use for the semantic embedding; user-defined Python hook (`input record → object to embed`) as a plugin API for developers. Consider also: make the stopword list configurable (text file).

- **[Medium]** Time granularity: currently year only. Scientific papers may need month; news analysis needs date; syslogs need nanosecond timestamps. Design for arbitrary granularity.
  - Timeline visualization: show also month/day when available; for log analysis, full timestamps.

- **[Medium]** Web of Science: fix character escape bug (quotes, braces, etc.) breaking import for several files in the hydrogen test set.

- **[Medium]** More import sources: Semantic Scholar, Scopus, ERIC (educational sciences/didactics), and others.

- **[Low]** Data file format: replace `.pickle` with `npz` or similar (not portable across Python/app versions). Also rename dataset vs. NLP cache file extensions to avoid the current `.pickle`/`.pickle` collision.

- **[Low]** Deployability: move user-configurable parts to `~/.raven/visualizer/` (consistent with `~/.raven/` already used by Librarian). Check Windows and macOS conventions.

- **[Low]** Detect and report duplicate entry keys in BibTeX importer (to ease debugging of BibTeX databases).

- **[Low]** For cluster-level keyword detection, de-duplicate words within each abstract before keyword extraction (avoids "keyword spam" from a single abstract dominating cluster keywords).


### Visualization and display

- **[Medium]** Configurable coloring modes: by cluster (current default), by year (newer = brighter), by input BibTeX filename (to see new data at a glance). Store import-source metadata in the dataset. Handle Misc/outlier items for year-coloring (toggle show/hide?).

- **[Medium]** Full report of all selected items, bypassing the info panel bottleneck. Suggested hotkeys: Ctrl+F8 for plain text (whole selection), Ctrl+Shift+F8 for Markdown. Separate the report generator from the info panel renderer (`_update_info_panel`).

- **[Medium]** Show most common keywords: currently printed to console only. Add GUI display, clipboard copy, save with dataset, button to recall at any time.

- **[Medium]** Show full authors in info panel (full author list is already loaded, just not displayed). Same search-aware display challenge as in author search: a 200-name list starting "Aaltonen ... Virtanen" needs to show where "Smith" matched, not just that it did.

- **[Medium]** BibTeX entry type support: show type per entry (article, inproceedings, book, patent, ...); show count by type in current selection; allow filtering by type.

- **[Medium]** Word cloud window: make resizable; add 1:1 button; use Pillow Lanczos for scaling (DPG built-in is nearest-neighbor); selectable color scheme (white background for paper export); move toolbar to top so it stays on-screen if the image is too large; expose size and color settings in GUI (currently only in `config.py`).

- **[Medium]** Settings window: expose `gui_config` in the GUI. Currently only in `config.py`. Note: this is a general gap — most Visualizer settings are not runtime-configurable.

- **[Medium]** Configurable annotation tooltip and info panel: which fields to show, sort by which field.

- **[Medium]** Layout switchable left/right: which side of the screen the info panel is on (for on-site collaboration, physical laptop placement constraints).

- **[Medium]** Show item slug (BibTeX identifier).

- **[Medium]** Per-item buttons in info panel: open DOI webpage; search for other items by same author(s) (rank by number of shared authors, descending).

- **[Medium]** Make the "Search" heading brighter to make it stand out visually.

- **[Medium]** Comparative analysis: place one dataset in the context of another (e.g. own research group within a whole field of science). Which dataset goes on top? How to color-code?

- **[Medium]** Image support in Visualizer: GUI currently handles text only. Needs design work:
  - Annotation tooltip and info panel: show images and/or generated captions
  - Text search over images: embed via Nomic (text+vision aligned space), or generate CLIP/VLM caption at import time and keyword-search the resulting text
  - Rethink what "search" and "keywords" mean for non-text items

- **[Medium]** Visualize how the selection was produced (search history display). E.g. "search 'cat photo', add 'solar', subtract 'vehicle'".

- **[Medium]** Save/load selection for reproducible reports. Especially important once Librarian uses the Visualizer selection to scope RAG (chat histories will be selection-specific). UX needs thinking.

- **[Medium]** Import BibTeX: use multiple columns in the input file table when there are very many input files.

- **[Low]** Make clustering hyperparameters configurable, preferably in the GUI. Put defaults into `raven.visualizer.config`.

- **[Low]** fdialog improvements:
  - Add "go up to parent directory" button
  - Change the "go to default directory" icon to something less confusing
  - In save mode: if the user has picked a unique file extension in the filter combo, use that as the default extension. If multiple extensions or wildcards, use the API-provided default.
  - Ctrl+F hotkey to focus the file name field is not always working. **[Verify]** exact conditions before fixing.

- **[Low]** Drag'n'drop from OS file manager into the Raven window to open a dataset. DPG 2.0.0: not implemented for Linux; Windows add-on exists. Need a cross-platform solution — keep an eye on DPG upstream.

- **[Low]** Live filtering by year (or other fields) in the visualization view, complementing import-time pre-filtering.

- **[Low]** Make all colors configurable. Requires customizing every colorable DPG item (can't query default theme colors). All custom colors are currently chosen to fit DPG's default color scheme.

- **[Low]** Convert filter to selection and vice versa (useful e.g. to select all items from 2020–2024, then invert).

- **[Low]** We can now import items that have no abstract. Generalize handling of arbitrary missing fields once configurable embedding fields are implemented.

- **[Parked]** Highlight visualization improvement: use outline instead of filled circle; brighten the data point's own color rather than using a separate color. Currently working well enough.

- **[Parked]** spaCy NLP for arbitrary input language (especially Finnish).

- **[Parked]** LLM keyword detection Alternative 2: preprocess text by LLM before handing to simple detector. Alternative 3: invert the embedding to find the word/sentence that best describes the cluster. (Alternative 1 — direct LLM — is the current implementation, prototype functional, tested on ~150 items, promising but slow.)


### LLM-assisted features

- **[Medium]** AI summarize: call an LLM to generate a summary report of items in selection. Per-datapoint summarization is already implemented in `raven.visualizer.importer`. See archive section for older design notes (citation validation, seahorse-based validation) that may still contain useful ideas.

- **[Medium]** LLM keyword detection (Alternative 1, current implementation): refinements needed — dataset-level topic analysis from titles, letter-case normalization, cacheable keyword sets (including partial cache of cluster results), progress display in GUI, logging cleanup. Update docs: LLM backend required when keyword extraction mode is "llm"; add low-VRAM mode fallback.

- **[Medium]** HybridIR integration (2): cross-app data integration between Visualizer and Librarian — both apps access the same data. Major design work, deferred. See also conversation logs for design draft.


### macOS support

- **[Medium]** Cmd key substitution for all hotkeys when running on macOS: detect OS at startup, update help and tooltips accordingly.
- **[Medium]** Resolve remaining hotkey conflicts with macOS builtins. Gather empirical data via live video session with pilot user. (Cmd+Shift+M for debug window is working; check others.)
- **[Medium]** Right-click and right-drag features on one-button mouse/trackpad.
- **[Medium]** F-key support on macOS.
- **[Medium]** OS X 10.x: ChromaDB/onnxruntime won't install; `av`/TTS won't install (add `try`/`except`, disable `tts` module gracefully). TTS is irrelevant for Visualizer-only use. Drop 10.x support as soon as pilot user upgrades to a recent macOS.


### Robustness and bug fixing

- **[Medium]** Crash recovery: periodically save crash recovery file (which dataset was open, selection undo history, search status); restore on startup with a non-blocking notification. No crashes yet on the 12k dataset, but peace of mind value is real. Also: unit tests would help here.

- **[Medium]** DPG 2.0.0 regression check (CC session): verify whether the following bugs from DPG 1.x are still reproducible:
  1. Keyboard focus issue: search field not focused visually, but navigation keys still won't operate the info panel
  2. Rare race condition in `hotkeys_callback`: widget lookup fails, DPG attempts to look up widget 0
  3. Ctrl+Z crash in search bar, especially after clearing the search

- **[Low]** Word cloud window shown under toolbutton highlight and info panel dimmer (DPG drawing order issue). Not clear if fixable — brainstorm with CC.

- **[Medium]** Performance: info panel is O(n²) due to the pure-Python Markdown renderer (no better options available), which starts hurting at ~400 items. Consider limiting data shown; also investigate the vendored DPG Markdown library with CC for optimization opportunities.

- **[Low]** Test again in DPG 2.0.0: `fdialog` Ctrl+F hotkey to focus file name field not always working. Test before attempting fix.



---

## Librarian

### Urgent / in-flight

- **[High]** Qwen3.5 thinking toggle: urgent. Models have a thinking toggle that must be set in a specific location (check ooba's implementation; see https://unsloth.ai/docs/models/qwen3.5 for reference). More broadly: support for non-thinking models — Librarian currently assumes `<think>` tag in several places (avatar speaking animation, `llmclient.invoke`). Add a "thinking model" toggle; when enabled, inject initial `<think>` at start of message if model doesn't send it (only when not continuing a previous message). `chatutil.scrub` already handles a missing opening tag when a closing tag is present, but only for the final message.

- **[High]** RAG PDF support: use `pdftotext` (from `poppler-utils`) to extract text, run through `sanitize`, add to RAG index. Store extracted text and link to original document. Handle paragraph break detection in `sanitize`. Generalize to other input formats (images via caption generation, etc.).

- **[High]** Adjustable semantic search match strictness: configurable cosine similarity threshold in HybridIR below which results are dropped. High priority.

- **[High]** File attachments (images): inject full image into LLM context (not RAG). Needed for reading graphs, tables, and equations in papers. Qwen3.5 and similar now support vision input natively.

- **[High]** File attachments (text/PDF): full-document context injection for deep analysis of a single item; complement to RAG, not a replacement. Also add feature to pull an already-ingested item from DB. Orthogonal to RAG — useful when you need to ensure the LLM sees the full document.

- **[High]** Citation tracker GUI: show which documents matched the RAG query; clickable to open each one. Also: validate that LLM-inlined citations (in whatever format we specify) actually point to documents in the RAG result set; flag any that don't.

- **[Low]** `minichat`: remove deprecation note (it will be maintained with Claude Code). Minimal example client, usable over a bare SSH terminal.


### Core features

- **[Medium]** Think blocks: parse properly instead of current regex hack. We already receive one token at a time.

- **[Medium]** Proactive context engineering: move beyond reactive BM25+semantic retrieval toward intelligent context curation. The system should maintain a graph of topical connections and proactively include relevant documents the user didn't explicitly ask for. E.g. "You asked about hydrogen embrittlement — here are the materials science papers you looked at last month." Shallow version (agentic chain-of-thought retrieval over a topic graph) is achievable now; deeper version requires a world model.

- **[Medium]** Document scopes: subdirectory-based filtering; scope selection GUI (checkbox per scope, select/unselect all); tags as the primary scoping mechanism (auto-tag by subdirectory name on ingestion); avoid cross-contamination between work/hobby contexts. Needed for long-term memory too. Currently must manually switch directories for each demo.

- **[Medium]** BM25 migration from `bm25s` to ChromaDB FTS5: gains incremental updates and metadata filtering (needed for scopes); removes full index rebuild at each commit; simplifies `hybridir.py` and removes a dependency. Mitigate tokenization quality loss by storing spaCy-lemmatized text in a dedicated ChromaDB field for FTS5 search. **Low priority** — `bm25s` works, and Raven's dependency policy is already generous.

- **[Medium]** Context compaction: drop and/or summarize old messages when context window fills. Use `raven.llmclient.token_count` to bisect linearized history to find the cut point (accounting for max response length from `settings.request_data["max_tokens"]`). Medium priority — in practice, usually start a new chat before running out, but any serious LLM frontend needs this.

- **[Medium]** Long-term memory: second RAG store indexing chat messages. Tool-call access (search with query, retrieve local neighborhood of a node). Automatic associative memory via autosearch on user's most recent message(s). Return user messages only (not AI replies) to keep the model grounded. **Design TBD — flag for second review round.** Hindsight may be a better backend here.

- **[Medium]** Explicit memory bank: third RAG store, AI-managed. Tool-call access (store/list/search/retrieve; title + content). Customizable system message section for things to remember across every chat. Chunk length may need adjustment (one chunk per memory). **Design TBD — flag for second review round.**

- **[Medium]** Three RAG stores architecture: (1) documents — explicit, user-managed (exists); (2) long-term memory — implicit, system-managed, indexes chat messages (new); (3) explicit memory bank — explicit, AI-managed (new). See memory items above.

- **[Medium]** Context fill meter.

- **[Medium]** Chat HEAD jump undo/redo.


### Chat UI

- **[Medium]** Recent chats list view: still pending. Design is nontrivial in a tree-based storage — consider that each top-level user message constitutes a distinct chat, with the most interesting branches as a second level. UX should faithfully represent what the memory system actually remembers (if only the main branch is remembered, show only that).
  - Chat card: show something distinctive per chat (user's initial message, last branch point, most recent message, tags)
  - Click to switch; double-click to switch and close the list
  - Timeline section separators by date
  - Filter by persona names, tags; tag autocomplete; mass tag editing
  - HybridIR search (since chats will be indexed for memory); show matching snippet

- **[Medium]** Nonlinear chat view / chat graph editor: XDot DPG viewer now exists. Librarian needs to generate `.xdot` code; manual layout (no GraphViz needed for simple chat trees). Limit visible depth (full chat tree at interactive FPS is not feasible). "Jump to chat node by ID" feature needed.

- **[Medium]** Switch HEAD by chat node ID: exported chatlogs report IDs; allow jumping directly to a node; show "not found" error if node doesn't exist in this Librarian instance.

- **[Medium]** Chat panel improvements:
  - Double-buffering for UI calmness during rebuild (not a performance issue, a smoothness issue)
  - Scrollability during LLM stream: add "user touched scroll controls" flag; disable auto-scroll when set; clear flag on appropriate events

- **[Medium]** Save/show full prompt per AI message: save the exact prompt at message-generation time (cannot reconstruct it later — system prompt may have changed, tree datastore doesn't preserve it). Likely needs a separate datastore with full prompt duplication. Show prompt in GUI with token count; copy to clipboard.

- **[Medium]** Ctrl+F find in current chat: incremental fragment search; reuse existing generic infrastructure from Visualizer/XDot viewer.

- **[Medium]** Multiline input.

- **[Medium]** Message editing: use chattree's revision system.

- **[Medium]** Robustness: temporarily disable relevant buttons while AI is writing; re-enable correctly by checking whether the relevant action has a stashed callback for that specific displayed chat message.

- **[Medium]** Fix bug: incomplete thought block (in first response) after Continue. Continuing should resume the incomplete thought block. May have a reproducible case still in the persistent chat tree — investigate.

- **[Low]** Add lockfile so `raven-minichat` and `raven-librarian` can't run simultaneously (prevents losing changes made in one app). Quick CC session.

- **[Low]** minichat: **[Verify]** when retrieval results are `null` in `data.json` — old bug or still present in current codebase? (CC session)


### STT / voice

- **[Medium]** STT: configurable silence level, autostop timeout, VU peak hold time.

- **[Medium]** `raven-transcribe`: command-line tool for transcribing audio files or mic input. (`-p` for prompt, `-o` for output file, stdout by default.) Potential for podcast analysis.

- **[Medium]** Proper name extraction via spaCy NER: extract proper names from chat log, fill into STT prompt as a comma-separated list (improves transcription of names).

- **[Low]** Voice command interface: split transcribed text to words, check first two words for command prefix, trigger command processor for the rest. Low priority.

- **[Low]** Long subtitle splitter: we now have audio length per sentence.

- **[Low]** Edit spoken message before sending.

- **[Low]** Look into quantized whisper-large-v3-turbo to save VRAM (~1.6 GB currently). May need vLLM backend.

- **[Low]** STT known issues (still open):
  - Spurious text generated after speech ends in long audio (see `raven.client.tests.test_api`)
  - Test `stt_transcribe_file` and `stt_transcribe_array`


### Tools

- **[Medium]** Weather tool via open-meteo (https://open-meteo.com/en/docs). Makes Librarian more humanlike as a "voice with internet access" (HCI is a major Raven goal). Medium priority.

- **[Medium]** Calendar tool: get one- or three-month calendar, like the `cal` command-line utility. See Python's `calendar` module.

- **[Medium]** Calculator tool: secure eval limited to math expressions. `eval` itself is unsafe (see notes in archived section). Candidate: https://github.com/danthedeckie/simpleeval.

- **[Medium]** RAG access via tool-call: search the document DB with a given query (optionally scoped), fetch a full document by ID, get available topics/scopes. Keep auto-inject (current scaffold behaviour) alongside tool access — don't replace it.
  - **Investigate:** Qwen currently requires RAG results near the start of the context or comprehension degrades — this kills KV cache hit rate (near-full preprocess each turn since the RAG result set changes). Check whether injecting near the end of the chat still works; newer model versions may have fixed this. No official Qwen documentation for RAG-with-tool-use found yet.

- **[Medium]** Webfetch tool: allow fetching user-provided URLs (conservative security policy — user-provided links considered safe; no autonomous crawling). Useful for throwing AI research blogposts at the model and seeing how they connect to the local paper pile.

- **[Medium]** Websearch: **[Verify]** whether raw URLs are currently saved in tool results. Remaining work: final formatting of results, link crawling to retrieve full result documents (persist to RAG with expiry timeout), figure out in which contexts search result pages should be enabled as RAG data sources.

- **[Medium]** HybridIR pedigree field: auto-remove only documents added by a named scanner instance. Needed for programmatic RAG ingestion (e.g. web pages from websearch).

- **[Medium]** Source attribution for RAG: clickable snippets in GUI based on `document_id`, `offset`, length; clickable link to open full document (spawn external viewer based on file type). See also citation tracker GUI item above.

- **[Medium]** Inline citations: encourage LLM to inline citations in a specified format; validate programmatically that cited IDs exist in the RAG result set; flag invalid citations. Design goal: preserve synthesis (don't force one-paragraph-per-source).

- **[Medium]** MCP support: under consideration; security implications unresolved. Agent skills (CLI-based, "anime maid form factor" — plugging into interfaces designed for human use) are a superior alternative capability-wise, but also more dangerous for the user's computing environment. Keep both options under consideration.

- **[Low]** IBM Granite OCR / vision OCR: low priority. Since writing this item, DeepSeek-OCR and Qwen3.5 native vision have appeared. Evaluate accuracy/speed/model size tradeoff when relevant.

- **[Parked]** Translator upgrade: HPLT v2 (Helsinki-NLP), needs Marian format backend (https://huggingface.co/HPLT/translate-en-fi-v2.0-hplt_opus). Divided on adding a second backend just for this.

- **[Parked]** User persona sampling / prefill: functional utility for local model testing, but deferred for now.


### Avatar (Librarian-side)

- **[Medium]** Avatar on/off toggle: auto-off is implemented; add explicit disable so Librarian won't try to load or run the avatar at all (for low-VRAM setups). What to show in the right panel when avatar is off? (Recent chats list, once that exists?)

- **[Medium]** Avatar: digital glitch effect when switching chat branches. Postprocessor filters already exist; this is a scripting/control task. Think through interaction with the user's postprocessor config. Fits Raven's deliberate cyberpunk aesthetic.

- **[Medium]** Avatar: do more to eliminate stutter while receiving LLM response. Happens especially at first avatar speech in a session and while TTS is rendering in the background. Pushing limits of 3070Ti. Investigate audio buffer size (see `raven.client.util`) and rendering smoothness under high system load.

- **[Medium]** `DPGAvatarRenderer`, `DPGAvatarController`: isolate DPG-specific parts for portability.

- **[Low]** Draw per-character AI chat icons for all characters (e.g. `aria1.png` → `aria1_icon.png`, RGBA 64×64).

- **[Parked]** Avatar vector emotions: blend several emotions by classification values; normalize appropriately. Low priority.


### Robustness

- **[Medium]** Smooth scrolling for linearized chat view. Infrastructure already exists (`raven.common.gui.animation.SmoothScrolling`); just connect it (see Visualizer info panel for example).

- **[Medium]** Don't crash if `tts` module isn't running.

- **[Low]** RAG: **[Verify]** whether chunk full-IDs are listed in retrieval metadata for combined contiguous chunks. (CC session)


---

## Server

- **[Medium]** Server: check for local model before checking HuggingFace Hub.
  - Currently some modules do this, others don't.
  - Important if a model is removed from HF (as happened with the old summarizer).
  - Allows an existing installation to start even when the model is no longer on HF.
  - Better for privacy.
  - Allow disabling (opt-in) for automatic model updates when the HF repo is updated.

- **[Medium]** AI model update UX: currently Server pings HF on startup to check for model updates for everything it loads. Need UX design for the case where a model is superseded by an API-compatible but different-lineage model (the original HF repo won't update). What should happen? Warn? Auto-swap? User-configurable?

- **[Medium]** STT module known issues (see Librarian STT section for details).

- **[Low]** Zip avatar characters for ease of distribution:
  - Include all extra cels, optional animator/postprocessor settings, optional emotion templates.
  - Implement zip loading on server side; add a new web API endpoint.
  - Do this when JS client work starts.


---

## Avatar

- **[High]** Add help cards for: Avatar settings editor, Avatar pose editor.

- **[Low]** Implement JS client for integration of Avatar with other LLM frontends. Needs work on those other frontends, too. Initially, target SillyTavern.

- **[Low]** Update assets for all characters: add at least the eye-waver effect (and possibly other cel-blending cels). Aria is the default character with full feature support. Other characters are lower priority.

- **[Low]** Move data eyes management to server side (currently client-side). Distributes concerns correctly; will simplify JS client slightly. Do this when JS client work starts.


---

## XDot Viewer (`raven-xdot-viewer`)

*(No outstanding items.)*

---

## Tooling (e.g. pdf2bib, csv2bib)

- **[Medium]** `raven-docdb-import` (or similar — check naming convention against existing CLI tools): CLI tool for batch document ingestion into the Librarian document DB. Just run `hybridir.setup` on the same datastore that Librarian uses and wait for the scanner to finish.

- **[Medium]** pdf2bib: conference name is hardcoded. Make it configurable (parameter or config file).

- **[Medium]** pdf2bib: prompt the author extraction step to return a canonical string (e.g. "No authors provided") when no authors are found. Same for title extraction (e.g. "No title provided"; also handle the case where the LLM thinks the title is literally "Abstract").

- **[Medium]** pdf2bib: some LLMs behave erratically when the system date is later than their training cutoff (e.g. refusing tasks, claiming to be in a simulation). Investigate mitigation strategies; may be model-version-specific. Track across model upgrades.

- **[Medium]** pdf2bib overthinking / token-limit mitigation: detect token-limit-exceeded in `raven.librarian.llmclient`, return a status flag in metadata. Consider executive-function simulation via LLM (in the neuropsychology sense: https://en.wikipedia.org/wiki/Executive_functions) as a recovery strategy — but may be superseded by improved model capabilities; monitor before investing time.

- **[High]** csv2bib: add documentation (main README + Visualizer README, section on importing data). New CLI tool added in 0.2.5.


---

## Infrastructure and maintenance

- **[High]** Unit tests. Currently very sparse. Would significantly improve confidence in refactoring.

- **[Low]** Post PR of vendored FileDialog fixes upstream. Raven's extensions have genuine added value worth sharing. Upstream is likely inactive but the PR is worth filing.

- **[Low]** wosfile: consider vendoring our fixed version. Check upstream activity first — may be worth a PR instead.

- **[Low]** Raven technical report (arXiv): document Raven as a citable reference. "Here's a tasteful way to put existing ideas together, plus a GUI app." Needs a CS category endorser.


---

## Archive

*Items considered and decided against, or firmly superseded. Kept for reference.*

- **AI summarize — older design notes**: from the original TODO. May contain useful material for when this feature is implemented:
  - Per-datapoint LLM summarization: condense each abstract into one sentence with the most important main point. (Core implementation done in `raven.visualizer.importer`.)
  - Citation validation via `seahorse-large` (based on `mT5-Large`; 6 models, 5 GB each): https://github.com/google-research-datasets/seahorse
  - Scaffold for guaranteed-correct citations: process each document separately to eliminate cross-contamination; check each summary via LLM for hallucinations ("does all information in this summary come from the original text?"). Build an internal reference list from matched document IDs; append citations programmatically at the end.
  - Newer design (supersedes above for citation tracking): LLM inlines citations freely in a specified format; scaffold validates that cited IDs actually exist in the RAG result set; flags any that don't. Preserves synthesis.

- **SONAR sentence embedder** (https://github.com/facebookresearch/SONAR): evaluated as a potential replacement for the semantic embedder. Decision: Nomic-embed (Apache 2.0, aligned text+vision spaces) selected instead. SONAR's multilingual capabilities are interesting but not currently needed.

- **SaT text segmentation** (https://github.com/segment-any-text/wtpsplit): potential NLP tool for document cleaning. Parked — may be useful later but no current use case.

- **"Detect novelty" (naive approach)**: original idea — novelty as inverse density (sparse regions = novel). Superseded by the Procrustes-based novelty detector, which falls out naturally from the incremental dataset update feature and is more principled.

- **"Importer: allow specifying a dataset to load dimension reduction from" (original)**: the simplest approach to adding new data on top of an existing dataset. Superseded by Procrustes alignment, which is strictly better for the common case (related data). The Procrustes item above documents its assumptions and the fallback for unrelated datasets.

- **PDF conference abstracts robustness item**: added as a reminder to check whether pdf2bib handles this case. Now working correctly. (Conference name still hardcoded — see active TODO item.)

- **System prompt tuning for LLM speculation on/off**: was relevant during early Qwen3 work. Superseded by improved model behavior. Dropped.

- **RAG search data location in chat tree**: where to store RAG results in the chat tree format. Resolved — tracked in metadata. Dropped.

- **Privacy note for STT in Librarian docs**: has been added to documentation. Dropped.

- **"Switch chat from all leaf nodes" feature**: idea was that each leaf node constitutes a potentially interesting HEAD. Not a productive framing — too many leaf nodes for useful UX. Superseded by the recent chats list design, which uses a more principled definition of "distinct chat."

- **Installation instructions TL;DR**: now covered in main README.md. Separate section no longer needed.

- **Misc items: assign to closest cluster in 2D** (original Visualizer item): duplicate of the cosine-to-medoid outlier assignment in the importer rework. Dropped.

- **Calculator tool using `eval`**: `eval` is fundamentally unsafe in Python (e.g. `().__class__.__base__.__subclasses__()[-1].__init__.__globals__['__builtins__']['__import__']('os').system(...)`). See https://stackoverflow.com/questions/64618043. Use `simpleeval` instead — see active TODO item.

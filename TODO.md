# Raven TODO

Covers the full Raven constellation: Visualizer, Librarian, Server, Avatar, XDot Viewer, and shared tooling.

Priority tiers: **[High]** | **[Medium]** | **[Low]** | **[Parked]**

Items marked **[Verify]** should be checked against the current codebase in a CC session before implementing.

## Cross-cutting

- **[High]** HF hub: document the env vars that prevent hub checks on Raven startup (for privacy and faster startup). Add recommendation to server docs. Currently not written down anywhere in the project.
  - `HF_HUB_OFFLINE=1` — forces huggingface_hub to use only locally cached models, no network requests at all.
  - `HF_HUB_DISABLE_TELEMETRY=1` — stops telemetry pings only.

- **[High]** Revisit logging system: library modules should not reconfigure the logger (verify exact behavior against Python `logging` stdlib docs, but currently each module sets the log level, which is the entrypoint's responsibility). Move logging configuration to entrypoints only. Add a "detailed debug" level at that time for particularly spammy-but-useful log lines (e.g. `SmoothScrolling.render_frame`, `_managed_task`, `binary_search_item`).

- **[Medium]** Flash the search field when focused by hotkey. Currently affects Visualizer main window, fdialog component, and XDot Viewer. **The enabler is done** (2026-07-30): `ButtonFlash` is now `WidgetFlash` and animates any widget — a text widget fades its own text color, anything else fades a theme background — with `animation.highlight_widget` as the convenience entry point, alongside `flash_button`. What remains is applying it at the three search fields, which is the actual item.

- **[Medium]** Split `raven/vendor/` into `vendor/` and `forks/`. **Gate: after the FileDialog keyboard work is finished** (Juha, 2026-08-20) — the fork in question is the file dialog, and renaming its package mid-sprint buys nothing.
  - **The defect is a false README, not an untidy name.** `raven/vendor/README.md` opens with *"None of this is Raven's work, and none of it is covered by Raven's own licence."* That is wrong on both counts for `file_dialog`: measured 2026-08-20, it arrived at 1233 lines and is now 5552, with 5037 lines added over 138 commits — roughly 78% of that tree is ours, under Raven's BSD, including 175 tests. A folder whose README cannot be true is the thing to fix; the rename is how.
  - **The criterion is upstream mergeability, not how much we changed.** *Could this still take an upstream update?* Vendored code could — patches reapplied, upstream still worth reading, changes possibly worth sending back. A fork could not, and nobody would look. Cut that way, only `file_dialog` moves: nobody is merging upstream into 138 commits of our own keyboard.
  - **Divergence magnitude is the axis that does not work**, and `DearPyGui_Markdown` is why — 490 lines over 12 commits is too much to read as a snapshot and too little to read as a fork, so sorting by size stalls on it. On the mergeability axis it is not awkward at all: guards added to otherwise-unchanged code, and an upstream release still merges. It stays vendored.
  - **Which is provisional, and deliberately so.** The renderer is due an overhaul (`briefs/markdown-block-rendering-brief.md`, and the `markdown-renderer` cluster in `TODO_DEFERRED.md`). If that work rewrites rather than guards, the answer to the mergeability question changes and the renderer moves too. **Re-ask it when that lands**, rather than assuming this entry settled it.
  - Blast radius: 10 import sites in `raven/`, plus `pyproject.toml`, three docs and eight investigation scripts. Mechanical; `grep -c` afterwards is the check.

- **[Medium]** Sweep the GUI styling constants into one module. Colours, pulse periods, flash durations, dwell and delay times are sprinkled through the tree, and many of them agree with each other by hand rather than by construction. **Gate: after the FileDialog keyboard brief reaches a closable state** — item 7 there adds another shared colour, and doing the sweep first would mean sweeping twice.
  - **The trigger that made it visible:** `raven.common.gui.keyboardmark` is the second constant to be pulled out of the widget that happened to need it first (the default text colour was the first). A third would be a pattern rather than a coincidence.
  - Scale, from one grep: **at least 24 module-level colour/duration constants outside the tests**, plus whatever is inline. A floor, not a count — the pattern only catches `NAME = (...)` and `NAME = 1.5` at module level, so class attributes and literals at the call site are extra.
  - **The design question is not where to put them, it is which ones are styling.** `gui_config` is an `env` inside `raven/librarian/config.py` and `raven/visualizer/config.py`, mixing two different things: what the *project* decides and should agree across apps (a flash duration, the keyboard's blue), and what a *user* sets for their machine (window sizes, paths). Only the first should move; the second is what those files are for.
  - **A hygiene win comes with it.** Those two `config.py` files are the ones edited in place on every dev machine and never staged, so anything shared that lives in them is shared through a file nobody may commit. Moving the styling out shrinks the surface where a personal override sits next to something everyone needs.

- **[Medium]** `vis_data` → `entries` rename across the whole constellation, including importers and BibTeX tooling in `raven.papers`.

- **[Medium]** Visualizer↔Librarian integration: allow querying Librarian for documents (set as RAG sources) that are currently selected in Visualizer. Apps communicate over the local network. Core workflow: "show me the cluster structure around this topic" → "now let me drill into those papers conversationally."
  - IPC design: ZeroMQ pub/sub over localhost (or localhost websockets, since raven-server already has a web API layer). IPC is optional — if both apps are running, use it; if not, graceful degradation. Neither app should depend on the other being present.
  - Bidirectional stretch goal: Librarian highlights search results on Visualizer's semantic map. Allows vague natural-language queries to find papers related to a given topic.

- **[Medium]** Large files (images, audio, full PDFs) should be stored separately from the main datastore and linked, not embedded. Currently no large files are used; this is a note for when blob support is added. Applies to both Visualizer dataset files and the Librarian document DB, and to large text files too.

- **[Low]** `deviceinfo` at app bootup should report whether the reported device configuration is for the client or for the server. Add a parameter.


---

## Visualizer

### Refactor (do first)

- **[Low]** `raven.visualizer.app` refactor: largely done. `app.py` is 1912 lines, with `info_panel`, `selection`, `plotter`, `annotation`, `word_cloud`, `entry_renderer` and `app_state` extracted. What remains is optional rather than blocking: `info_panel.py` is 1518 lines and could split further, and the info tooltip still shares many data sources with the info panel.

- **[Medium — but first decide whether it still applies]** FP refactor: keep app state in top-level containers, pass in/out explicitly. More FP-idiomatic and facilitates adding unit tests.

  The `app.py` refactor it was waiting on has landed, and `app_state.py` arrived with it — but it answers only half of this. `app_state` is a single shared `env()` namespace, so state now lives in a top-level container and every cross-module access is named (`app_state.foo`), which kills the circular imports and the ambiguous bare names. What it does not do is *pass state in and out explicitly*: reads and writes still go to shared mutable module state, so the stated payoff — easier unit tests — is largely unrealized, since a test must still populate and tear down a global namespace.

  So the question is whether the explicit-passing version is still wanted for a DPG app whose event callbacks are inherently global-shaped, or whether `app_state` is the acceptable long-term answer and this item can go.


### Search and data access

- **[High]** Author search: show full author list, in the info panel too, where it is already loaded but not displayed. GUI must be search-aware — when search is active, highlight where the match appears in a long author list (e.g. a 200-name list starting with "Aaltonen" and ending with "Virtanen"; user searching for "Smith" needs to see where it is, not just that it matched).

- **[High]** DOI: record DOI in BibTeX importer; show DOI per item in info panel; per-item button to open official webpage (`https://dx.doi.org/...`); export list of DOIs/URLs for fulltext automation.

- **[Medium]** Fragment search across multiple fields (author, year, abstract, ...); configurable which fields to search. Add checkboxes and a select/unselect-all button below the search bar. Note: the highlighter currently only processes titles and is slow — may not be able to highlight in abstracts without performance work.

- **[Medium]** Semantic orienteering: embed user-typed text, dimension-reduce it, highlight the resulting virtual datapoint in the plotter. Later: support user-given BibTeX entry or PDF file as input.

- **[Medium]** Select cluster by number: useful complement to the wand button for datasets with few clusters.

- **[Medium]** Add GUI filter/search in the help window hotkeys list: incremental fragment search by key or action.

- **[Low]** Word boundary mark (`\b`) for search. UX: what character should the user type as a word boundary?

- **[Low]** BUG: Search result highlight: "Can a" → highlights whole word "Can", then highlights "a" inside it, breaking the outer highlight. Difficult to fix.


### Import and data pipeline

- **[Medium]** Procrustes alignment for adding papers to an existing map, with novelty detection: `briefs/11_visualizer-importer-rework-brief.md`, item 4.

- **[High]** Publish a ready-made dataset for quick-start demo (e.g. AI papers from arXiv, fully public).

- **[High]** HybridIR integration (1): spawn an in-memory `Forest` + HybridIR instance over the BibTeX data for full-text search. Once full BibTeX records are saved in the dataset, this is mostly scripting.

- **[Medium]** BibTeX export: keep full original BibTeX entries in the dataset; export selected items as BibTeX. Check whether the importer already preserves full entries or discards them.

- **[Medium]** Excel import to BibTeX (CSV importer already exists; minor convenience upgrade for pilot users without the CSV workaround). Needs Windows testing.

- **[Medium]** Importer: check that there is at least one item before proceeding; throw a sensible error message if not (currently crashes silently). Triggered by BibTeX files with the Author field missing.

- **[Medium]** Pre-filtering at import time (e.g. by year). Also: add option to re-scan a BibTeX database for new entries added since last import (import only new items, report them as such).

- **[Low]** `papers.utils.bibtex_escape` / `bibtex_unescape` and `common.utils.unicodize_basic_markup` are three implementations of two directions of one transformation — worth checking whether they should be one. (Noticed 2026-07-29, verifying that BibTeX umlauts and verbatim braces already worked.)

- **[Medium]** More flexible data import: configurable which fields to use for the semantic embedding; user-defined Python hook (`input record → object to embed`) as a plugin API for developers. Consider also: make the stopword list configurable (text file).

- **[Medium]** Time granularity: currently year only. Scientific papers may need month; news analysis needs date; syslogs need nanosecond timestamps. Design for arbitrary granularity.
  - Timeline visualization: show also month/day when available; for log analysis, full timestamps.

- **[Medium]** More import sources: Semantic Scholar, Scopus, ERIC (educational sciences/didactics), and others.

- **[Low]** Data file format: replace `.pickle` with `npz` or similar (not portable across Python/app versions). Also rename dataset vs. NLP cache file extensions to avoid the current `.pickle`/`.pickle` collision.

- **[Low]** Deployability: move user-configurable parts to `~/.raven/visualizer/` (consistent with `~/.raven/` already used by Librarian). Check Windows and macOS conventions.

- **[Low]** Detect and report duplicate entry keys in BibTeX importer (to ease debugging of BibTeX databases).

- **[Low]** For cluster-level keyword detection, de-duplicate words within each abstract before keyword extraction (avoids "keyword spam" from a single abstract dominating cluster keywords).


### Visualization and display

- **[Medium]** Configurable coloring modes: by cluster (current default), by year (newer = brighter), by input BibTeX filename (to see new data at a glance). Store import-source metadata in the dataset. Handle Misc/outlier items for year-coloring (toggle show/hide?).

- **[Medium]** Full report of all selected items, bypassing the info panel bottleneck. Suggested hotkeys: Ctrl+F8 for plain text (whole selection), Ctrl+Shift+F8 for Markdown. Separate the report generator from the info panel renderer (`_update_info_panel`).

- **[Medium]** Show most common keywords: currently printed to console only. Add GUI display, clipboard copy, save with dataset, button to recall at any time.

- **[Medium]** BibTeX entry type support: show type per entry (article, inproceedings, book, patent, ...); show count by type in current selection; allow filtering by type.

- **[Medium]** Word cloud window: make resizable; add 1:1 button; use Pillow Lanczos for scaling (DPG's built-in scaling is bilinear with no mipmaps, so it aliases when shrinking); selectable color scheme (white background for paper export); move toolbar to top so it stays on-screen if the image is too large; expose size and color settings in GUI (currently only in `config.py`).

- **[Medium]** Settings window: expose `gui_config` in the GUI. Currently only in `config.py`. Note: this is a general gap — most Visualizer settings are not runtime-configurable.

- **[Medium]** Configurable annotation tooltip and info panel: which fields to show, sort by which field.

- **[Medium]** Layout switchable left/right: which side of the screen the info panel is on (for on-site collaboration, physical laptop placement constraints).

- **[Medium]** Show item slug (BibTeX identifier).

- **[Medium]** Per-item button in info panel: search for other items by same author(s) (rank by number of shared authors, descending). The DOI button is part of the DOI item above.

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

- **[Low]** Drag'n'drop from OS file manager into the Raven window to open a dataset. **The hard part is done**: `raven.common.gui.filedrop` shipped 2026-08-10 and is wired into all six GUI apps, so the paths arrive; what is left for Visualizer is deciding what a dropped file *means* here (open a dataset? add to the current one?) and handling the unsupported-file case. The mechanism and its one real constraint — the callback runs on the render thread, so no `split_frame` and therefore no modal messagebox inside it — are in `dpg-notes.md`; the probes are in `investigations/dpg-dnd/`.

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

- **[Medium]** HybridIR integration (2), a unified Visualizer/Librarian document DB with scopes: see `briefs/13_corpus-scopes-and-unified-db-brief.md`.


### macOS support

- **[Medium]** Cmd key substitution for all hotkeys when running on macOS: detect OS at startup, update help and tooltips accordingly.
- **[Medium]** Resolve remaining hotkey conflicts with macOS builtins. Gather empirical data via live video session with pilot user. (Cmd+Shift+M for debug window is working; check others.)
- **[Medium]** Right-click and right-drag features on one-button mouse/trackpad.
- **[Medium]** F-key support on macOS.
- **[Low]** OS X 10.x: ChromaDB/onnxruntime won't install; `av`/TTS won't install (add `try`/`except`, disable `tts` module gracefully). TTS is irrelevant for Visualizer-only use. Superseded in practice — `TODO_DEFERRED.md`, "Drop the Intel Mac / macOS 10.x install workaround", records the platform as effectively dead (new Macs are Apple Silicon) and proposes removing the README section rather than supporting it. Resolve the two together.


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

- **[High]** Give the tool-call round budget room for a genuine multi-document read. Measured live 2026-07-29 (`briefs/librarian-extension/manual_tests/rag_live_corpus.py`, phase F): asked a follow-up about which documents supported an earlier answer, the model works through `list_consulted_documents`' output one `fetch_document` per round, exhausts `max_tool_call_rounds = 5` on gathering alone, and has no round left to answer. Nine of fourteen sampled turns that reached the cap ended with an **empty assistant message**, against one of ten that did not (24 paired samples, Fisher exact p = 0.013; raw data in `investigations/tool_budget/`). Brief 10's own two features collide here: the provenance list invites reading several documents, the cap budgets five rounds in total.
  - **A mitigation was tried and did not measurably work.** The invocation after the cap carries `chatutil.format_notice_that_tools_are_spent`, on the reasoning that the model was never *told* the gathering was over — it found out by reaching for a tool that was gone. Measured across 12 paired samples per arm, it moved nothing: 8/12 answered with it, 6/12 without, p = 0.68, and the sign flips once restricted to cap-reaching turns. Kept because it is one line and addresses an observed mechanism, but **it is not the fix and must not be mistaken for one**. The cap itself is what correlates with the empty reply, which is what the budget change below has to address.
  - **Landed 2026-08-04, and it is not this item:** past the cap the tools now stay in the schema and a call is refused with an error result (`chatutil.format_error_that_tools_are_spent`), with withdrawal kept as the terminator of last resort after `max_tool_call_refusal_rounds`. That was argued on cache-burn and distribution-fit grounds, not on the empty replies — the model still runs out of budget at the same round, so the measurement above stands and the item below is still the fix. Worth re-running the probe once the budget actually changes: the arms and the resume ledger are in place, so it is a re-run rather than a rebuild.
  - **The fix to make is a larger budget, not a list that discourages reading.** Deciding *not* to make retrieval timid: a user faced with a model reading through the phone book has Ctrl+G and a rephrased request, which is a better remedy than a system that refuses to read thoroughly when thoroughness is what was asked for.
  - **Partly overtaken by events, 2026-08-04: the cap was raised from 5 to 20** on the strength of `investigations/tool_refusal/`, which measured where this model actually stops. That may be enough on its own; re-run the phase F probe before building anything further here.
  - **Two budgets, not one, and both per *turn*.** A single larger `max_tool_call_rounds` loosens the wrong thing too: a `fetch_document` is bounded work against a document already known to exist, while a `search_documents` can be rephrased forever against a corpus that has nothing — which is the failure the cap was added for in the first place (`manual_tests/rag_tool_rescue.py`). So searches keep a small cap and fetches get their own, larger allowance, naturally sized by `docs_num_results` since a search cannot surface more documents than that.
    - **The "forever" did not survive measurement.** Against a corpus containing literally nothing, qwen3.6-35b-a3b rephrased nine or ten times and then gave up unprompted, in 3 of 3 samples with the cap out of reach (`investigations/tool_refusal/`). That is the strongest form of the case this design was built to handle, and the model self-terminated. One model and small n — evidence, not a refutation — but the two-budget split needs a better argument than this one before it is worth its complexity.
    - **The allowance has to be per turn, not per search.** Per search it does not bound anything: `search(X) → fetch(X, 0..9) → search(Y) → fetch(Y, 0..9) → …`, where Y is a keyword the model picked up while reading X's results. Each search would refill the pool, and the recursion is a perfectly reasonable research strategy — which is exactly why it needs a ceiling rather than an argument.
    - Note what that loop *is*: search, read, harvest a term, search again is the shape sold elsewhere as agentic "deep research". So the ceiling is a resource decision — context window, latency, the user's patience — and not a correctness one. Pick the number by what a turn can afford to spend, not by what looks like runaway behaviour, because the runaway and the good version are the same algorithm.

- **[Medium]** The context-fill meter should count prior reasoning on templates that keep it. It counts none, which is right for Qwen (whose template discards prior reasoning) and a large under-report on Gemma (whose template re-sends every stored trace). Counting it unconditionally would be as wrong the other way, so the meter needs a per-model "template retains reasoning" flag, in the `model_is_vlm` family. The debounced prefill corrects the readout once the chat settles; it is the immediate estimate that is off. Measurements in `investigations/thinking-toggle/`.

- **[Low]** Qwen 3.6's `preserve_thinking` (keep every turn's reasoning in the prompt) is unreachable through LM Studio's HTTP APIs, every route measured. The one untried surface is LM Studio's per-model config, the way it evidently sets `enable_thinking`. If it works, the context meter has to learn about the mode at the same time. `investigations/thinking-toggle/`.

- **[Low]** `chatutil.scrub`'s `<think>` repair is a backstop for models that emit malformed or missing tags where reasoning arrives inline. **Keep it.** The July record expected it never to fire on LM Studio, which delivers reasoning on its own channel — but spurious `</think>` tags were seen arriving in the content channel from Qwen 3.5 9B on LM Studio the week of 2026-09-21 (maintainer), so something may still reach it there. Next step, if it matters: scan the chat datastore for `</think>` in stored content, which says how often. The ooba re-test is the other half (`TODO_DEFERRED.md`, "Upgrade oobabooga…"). `investigations/thinking-toggle/`.

- **[Low]** Note for sampler config: **LM Studio honours `min_p` even though its documented parameter list omits it.** Verified behaviourally 2026-07-27 — at temperature 2.0 the unclamped output varies between seeds, while `min_p=0.9` is seed-invariant, as is the documented `top_k=1` control. Recorded because the docs list (model, messages, temperature, top_p, top_k, max_tokens, stream, stop, presence_penalty, frequency_penalty, logit_bias, repeat_penalty, seed) reads as exhaustive and isn't; don't drop a sampler setting on the strength of it. Corollary: LM Studio returns HTTP 200 for unknown parameters, so any future "is this supported?" question needs a behavioural test, not a status code.

- **[Med]** RAG PDF ingestion — polish. The core is done: born-digital PDF text is extracted via `raven.common.docextract` (pypdf) and indexed like any other document. Remaining: run the extracted text through `sanitize` before indexing (PDF text often has hyphenation artifacts and paragraph-break ambiguity); link a search result back to its original document (see `TODO_DEFERRED.md`, "Expose the docs-DB source files behind a reply's RAG citations"); generalize to scanned PDFs (OCR) and to images (caption generation — ties into the Nomic multimodal-search plan).

- **[High]** Adjustable semantic search match strictness: configurable cosine similarity threshold in HybridIR below which results are dropped. High priority.

- **[Medium]** Attach a document that is *already in the docs DB*. Full-document attach itself works (images and text/PDF, brief 03 Half 2) — but only from the filesystem. There is no way to reach into the RAG store and attach one of its documents whole, which is what you want when retrieved chunks aren't enough and the file is already ingested.
  - **Open question: whose affordance is this — the user's, the AI's, or both?** The AI side already has an entry under Tools ("RAG access via tool-call: … fetch a full document by ID"), so if that lands, the model can pull a whole document itself. The user side (pick from the DB in the attach dialog) is the genuinely missing half. Deciding this shapes both: a shared "resolve doc ID → `text_file` content part" path serves both callers, and the GUI picker needs the docs DB to be browsable, which the tool version doesn't.

- **[High]** Inline citations, validated: encourage the LLM to inline citations in a specified format, then check that each cited ID is in the RAG result set, and flag any that are not. Design goal: preserve synthesis — don't force one paragraph per source. The other half — surfacing *which* documents fed a reply, and opening the originals — is specced in `TODO_DEFERRED.md`, "Expose the docs-DB source files behind a reply's RAG citations"; the provenance data is already tracked per turn (the payload's `retrieval` field), just not shown.


### Core features

- **[Medium]** Document-level questions: "which of my documents is the one about X?" Chunk retrieval structurally cannot answer this, and the failure is quiet. The fact that a story is set in the real world in America is distributed over the whole of it; no 1000-character chunk states it, so the query retrieves whichever chunk is nearest the phrasing and returns the wrong document with no indication anything went wrong. Observed 2026-08-05 against the fan-fiction corpus, where two content probes retrieved correctly at similarity 0.45–0.54 and this one retrieved wrongly at 0.38 (`investigations/retrieval/README.md`).
  - This is an ordinary thing to ask a document database, and it is squarely in Librarian's pitch — "find me the paper that was about the moving-web instability" is the same query shape as the one that failed.
  - Wants a document-level layer rather than a better query: per-document summaries or metadata, indexed separately from the chunks, so that a question about a document is matched against something that describes the document. Note the interaction with the corpus-scopes work — a scope is also document-level metadata, and the two probably want one mechanism rather than two.
    - **And with stage 3 of `VISION.md`**, whose half-built piece is per-document summarization — shipped code, currently switched off because it runs over a whole dataset at import time rather than over a selection. A summary layer is what this item wants indexed, so that is three things wanting one mechanism. Worth settling before any of them is built separately.
  - The confidence signal from brief 09 lever 1 detects the case without fixing it, which is worth having in the meantime: the query reads as low-confidence, so it can at least be *reported* as unanswered rather than answered wrongly.

- **[Medium]** A classifier pass over the arXiv paper stash, to separate the AI/ML records from the strays. The corpus at `00_stuff/datasets/ai_papers` accumulated by saving everything to one folder, so a minority of cosmology, astronomy and speculative-physics records are mixed in with the AI research. A short script that asks a local model to label each abstract would give a pure AI set, and the labels are reusable: they are exactly the ground truth a topic-scoping feature would be evaluated against, so this is worth doing even though the corpus works as-is for retrieval evaluation. Raised 2026-08-05.

- **[Medium]** Think blocks: parse properly instead of current regex hack. We already receive one token at a time.

- **[Medium]** Proactive context engineering: move beyond reactive BM25+semantic retrieval toward intelligent context curation. The system should maintain a graph of topical connections and proactively include relevant documents the user didn't explicitly ask for. E.g. "You asked about hydrogen embrittlement — here are the materials science papers you looked at last month." Shallow version (agentic chain-of-thought retrieval over a topic graph) is achievable now; deeper version requires a world model.

- **[Medium]** Document scopes: see `briefs/13_corpus-scopes-and-unified-db-brief.md`.

- **[Medium]** HybridIR: give documents a **title** field. Today a document has only `document_id` (the path relative to `docs_dir`) and its text, so there is nothing to show a user, nothing to hand a model deciding whether a document is worth fetching, and nothing to weight in retrieval. Titles are usually already present in the data and merely unparsed, and the reading of them is now written: `chatutil.document_label` extracts a BibTeX record's `title`/`author`/`year`, or falls back to the first substantial line. What is missing is *storing* the result as a field, which is what search can weight. Two wins, and the second is the larger: a legible label wherever a document is named (the `list_consulted_documents` inject in `briefs/librarian-extension/done/10_rag-tool-surface-brief.md`, a future citation UI), and a field that can be **weighted** in search — a title match is a much stronger relevance signal than a body match, which is index-side work adjacent to brief 09's query-side levers.
  - **Cost is a reindex**, ~1.5 h for the hydrogen dataset. Open question whether to migrate the existing index instead of rebuilding it: cheaper for the user, more code to maintain, and unlike the chat datastore a search index holds no irreplaceable hand-entered content — it is derived data, so nuke-and-rebuild is defensible in a way it would not be for `chattree`.

- **[Medium]** BM25 migration from `bm25s` to ChromaDB FTS5: gains incremental updates and metadata filtering (needed for scopes); removes full index rebuild at each commit; simplifies `hybridir.py` and removes a dependency. Mitigate tokenization quality loss by storing spaCy-lemmatized text in a dedicated ChromaDB field for FTS5 search. **Low priority** — `bm25s` works, and Raven's dependency policy is already generous.

- **[High]** Context compaction: drop and/or summarize old messages when context window fills. Use `raven.llmclient.token_count` to bisect linearized history to find the cut point (accounting for max response length from `settings.request_data["max_tokens"]`). Budgeting details in `TODO_DEFERRED.md`, "Context-window budgeting and conversation compaction (Librarian)".
  - **Raised from Medium 2026-07-29.** "Start a new chat before running out" stops being an answer once a turn can carry several fulltext attachments. Three papers compared against each other (`briefs/design/corpus-interrogation-sketch.md`, mode 3) is ~30k tokens before the discussion begins, and the discussion is the point.
  - **An attachment must not roll out entirely.** It is the reason the conversation exists. So compaction needs a priority order — pinned material, then recent turns, then older turns — rather than a single cut point found by bisection. Note that `llmclient.fit_attachments_to_context` is already a partial answer: it shrinks attachments as the conversation grows, max-min fair between them. What it lacks is the temporal dimension (an attachment nobody has mentioned in thirty turns is not equal to the one under discussion) and any notion of pinning.
  - **Summarizing on the main LLM is a two-way KV cache miss.** Sending a summarization prompt evicts the chat's cached prefix; returning to the chat with a *modified* history presents a new prefix in turn. Worse, since compaction targets the oldest part of the conversation, the first replaced message sits near the front — so nearly the whole prompt is reprocessed. A summarization event is therefore roughly a full prompt reprocess, which is the argument for **granularity**: compact rarely and in large chunks rather than continuously in small ones.
    - Two mitigations already have machinery in the tree. The context-fill indicator predicts *when* compaction will be needed before it is urgent; and `config.context_prefill_idle_delay` already runs a background LLM call while the user is reading, which is exactly the window in which a reprocess is free. Speculative compaction during idle is the natural pairing.
  - **Summaries belong in the chattree, and branching makes that pay.** A summary covering a span of nodes is derived data that must be cached and invalidated with the branch. Storing it against the span rather than the branch means every branch sharing that ancestry reuses it — the shared prefix is exactly where the oldest, most compactable material lives, so the reuse rate should be high. Consequence: building the sent context stops being a linear walk of `linearize_up` and becomes a policy evaluation over the branch (what is pinned, what is summarized, what is dropped), which wants its own module and its own tests rather than growing inside `serialize_history_for_wire`.

- **[Medium]** Memory, as three RAG stores: (1) documents — explicit, user-managed (exists); (2) long-term memory — implicit, system-managed; (3) a memory bank — explicit, AI-managed. **Design TBD for both new ones — flag for a second review round.** Hindsight may be a better backend for either; `briefs/librarian-extension/06_hindsight-standup-brief.md` is where that gets decided.
  - **Long-term memory** indexes chat messages. Tool-call access (search with a query, retrieve the local neighbourhood of a node); automatic associative memory by autosearch on the user's most recent message(s). Return user messages only, not AI replies, to keep the model grounded.
  - **The memory bank** is AI-managed: tool-call access (store / list / search / retrieve; title + content), and a customizable system-message section for things to remember across every chat. Chunk length may need adjusting — one chunk per memory.

- **[Medium]** Chat HEAD jump undo/redo: `TODO_DEFERRED.md`, "Nothing remembers which sibling the reader was on", which has the design.


### Chat UI

- **[High]** Show the raw prompt (the prompt viewer). A window displaying exactly what went on the wire for the current turn, with a copy button (Raven's usual green flash and tooltip acknowledgment). Slated for 2026-08-25; recorded because the decisions below were taken on 2026-08-24 and produced no diff.
  - **Raw text is the default, with a toggle to render it.** The chatlog is already the rendered view, so the point of this window is to be the unrendered one — and Markdown rendering hides the whitespace and delimiters a prompt is opened to inspect. Rendering is still worth offering, since messages typically contain formatting.
  - **With a breakdown, as SillyTavern's has**: system prompt, character card, user profile, RAG results, per-turn injects. This is the part with real work in it — the segments exist only as concatenated text by the time anyone can see them, so `scaffold.build_turn_prompt` has to hand back labelled pieces rather than a string. Sized as one focused session, which is why it is here and not a brief.
    - **Half of that is already labelled, checked 2026-08-25**, which is what keeps the sizing honest. `build_turn_prompt` returns a `List[Dict]`, so the *data* injects are separate messages by construction (`_synthetic_tool_exchange`). Only the *instruction* injects are joined into one string — and `build_system_injects` hands `_add_to_system_message` a **list of texts** to join, so the work is carrying a label alongside pieces that are already separate rather than decomposing a blob. One production caller, `scaffold.py:1116`.
    - **So v1 need not touch the prompt builder at all** (Juha, 2026-08-25): the instruction injects can simply be counted as part of the system prompt, which is where they physically are. That is honest about the wire and leaves the breakdown useful. **v2 is the one that separates them**, and is what the labelled-pieces work above is for — worth having eventually, and not a blocker for a first window.
  - **Window placement decides whether the chatlog needs a marker** (Juha, 2026-08-25). If the window covers the chatlog pane and cannot be moved, nothing else is needed — there is no chatlog on screen to be ambiguous about. If the chatlog stays visible, the message whose prompt is on display has to be marked in it: a pulsating frame, an icon, something. Without that, a window full of prompt text says nothing about *which* turn it belongs to, and the first thing anyone does with this window is compare two turns.
    - The colour and pulse for "this is the one" already exist as `raven.common.gui.keyboardmark`; check whether this is the same signal (the keyboard is here) or a different one that merely wants to look related, before reusing it.
  - `llmclient.serialize_history_for_wire(settings, history, continue_=False, datastore=...)` is the existing tee point, and returns the wire-ready messages. It is what the prompt-size measurements used.

- **[Medium]** Image shapes in the xdot widget, so `raven-xdot-viewer` draws GraphViz `image=` nodes: pieces 1 and 2 of `briefs/xdot-image-shapes-brief.md`. Piece 3, mip selection, landed 2026-09-07.

- **[Medium]** User-level settings: move the configs a *user* legitimately changes out of `config.py` into JSON, and give them settings dialogs. Today every one of them — LLM backend URL, model, docs directory, avatar knobs — is a Python source edit, which is why the tracked `config.py` files carry local overrides on every dev machine and have to be kept out of every commit by hand. Same gap the Visualizer has (the "Settings window: expose `gui_config` in the GUI" item above), so the two want a shared answer rather than two dialogs.
  - ~~**The JSON half**~~ — **done 2026-09-11.** `raven.configoverrides` reads `~/.config/raven/overrides.json`, keyed by config module, applied as the last statement of all eleven config modules; a dotted name reaches into an `env`, so both shapes a config module has are covered by one rule. Shipped defaults stay in `config.py` and the override file wins. See the README's *Configuration* section for the user-facing format.
    - **What this leaves for the dialogs** is the GUI half and nothing else: the file format, the precedence, the type fitting and the reporting of a bad key all exist and are tested. A dialog writes this file.
    - It also closes the leak below by construction — a file that is not in the tree cannot be staged — and the three tracked `config.py` files are clean as of that date.
  - **This needs a degraded startup mode, and that is the part that will bite.** Once a URL lives in a settings dialog, bailing on it means the user cannot reach the one control that would fix the problem: server down, server moved, laptop on a different network, and the app refuses to open far enough to be told so. The connection failure has to become a state the GUI can *run in* — the affected features disabled with a clear reason, settings reachable, and a retry that does not require a restart.
    - ~~**Librarian bails on the LLM backend**~~ — **no longer true, and the pattern to copy.** It is deliberately not a startup gate (`app.py`, just below the `api.test_connection()` call, says so): Librarian opens with no model in sight, shows a status pill above the composer, and reconnects. Past chats, the cleanup dialog and the settings are all reachable meanwhile.
    - **What is still a hard gate is Raven-server** — `api.test_connection()` in `app.py`, `sys.exit(255)`. The asymmetry is the argument for closing it: Librarian already *has* a degraded state for Raven-server, since a server that goes away mid-session raises a row below the mode toggles naming what stopped working, with a retry. So the app can run without Raven-server; it just refuses to *start* without it.
      - **It is not, however, just deleting the gate.** `briefs/raven-server-availability-brief.md` is the design, and this is its item 2 — the brief is not referenced from anywhere else, so look there first rather than starting from this line. **Avatar sessions are persistent on the server** (Juha's flag, and the thing nothing else would have caught): a connect that follows an earlier connect has to unload the stale session first or every reconnect leaks one, which makes reconnect-shaped code the natural owner of the unload rather than teardown — teardown by definition does not run on the path that matters. See also `TODO_DEFERRED.md`, "Librarian leaks its server-side avatar instance when it doesn't exit normally", which shares the premise.
    - **The Visualizer had the same shape and it was fixed on 2026-09-11**, which is worth reading first because it is the smallest instance: with LLM cluster keywords configured and the backend down, the app exited 255 before drawing anything, the pipeline having connected in its own module body while the GUI imports it at module level. The fix moved the check to where a run *starts* and made the failure an exception each frontend places where it has room. See `raven.visualizer.importer._setup_llm_backend`; `tests/test_importer_startup.py` pins the structural half.
  - Not everything should move. `config.py` is configuration-as-code and that is a feature for the parts that are genuinely code (the system prompt builder, the per-model VLM token table, computed paths). The split is "would a user reasonably want to change this without editing Python", not "is it a constant".
  - **"Kept out of every commit by hand" has failed three times, and this is what closes it.** `llm_backend_url` carrying a machine name reached the public repository on 2026-07-29, 2026-08-07 and 2026-09-07 — each time as an override staged along with a legitimate edit to the same file, and each time reverted a commit or two later. The deny rules in the agent's permission settings cover `git add -A`, `-u` and the directory form, and cannot cover this one: the file is where a real setting goes, so today's leak rode along with a genuine new flag. **A rule that blocked it would block the feature work.**
    - So the guard cannot live at the `git add` layer at all. Once the user-tunable values are JSON and `config.py` holds only defaults, the tracked file has nothing personal in it to commit by accident and the override lives somewhere ungitted. That is the fix, and the leak rate is the argument for its priority: it is not only a usability item.
    - The history is not being rewritten. `maia.local` is in ten commits from 2026-07-29 onward; a scrub would invalidate fifteen commit SHAs cited in these documents, orphan whatever PRs are open, and remove a `.local` name that does not resolve off the LAN. What *was* done (2026-09-07) is the part that reaches a reader: the same name is out of the seven live files that carried it.
  - **Then: tools for the AI to read and change Raven's own settings**, which the digital-colleague track wants — a colleague you can ask to turn the avatar off, point at a different documents folder, or switch the send key is a different thing from one you have to configure around. Strictly gated on the JSON move: a tool that edits `config.py` would be rewriting source it is itself running under, whereas a JSON settings file is data, with a schema to validate against and a known set of keys to expose. Design questions when we get there, none of them settled: which settings are exposed at all (the LLM backend URL is the one that can lock the AI out of its own next turn), whether a change needs user confirmation, and whether the AI can see the values it is not allowed to change. Note this raises the same stakes as any actuation tool — see the memory note on where the actuation boundary belongs.


- **[Medium]** Recent chats list view: still pending. Design is nontrivial in a tree-based storage — consider that each top-level user message constitutes a distinct chat, with the most interesting branches as a second level. UX should faithfully represent what the memory system actually remembers (if only the main branch is remembered, show only that).
  - Chat card: show something distinctive per chat (user's initial message, last branch point, most recent message, tags)
  - Click to switch; double-click to switch and close the list
  - Timeline section separators by date
  - Filter by persona names, tags; tag autocomplete; mass tag editing
  - HybridIR search (since chats will be indexed for memory); show matching snippet

- **[Medium]** Multiversal chat view / chat graph editor: **mostly built, 2026-09** — `chatgraph.py` and
  `chatgraph_panel.py`, wired into Librarian behind the *Chat graph* toggle. Done: the view itself, the
  visible-depth limit (as gap boxes), placement in the avatar panel's rect, and pausing the avatar while it
  is covered. **What is left is finding things: "jump to chat node by ID", and search.**
  - **The "emit xdot" decision below was superseded by what shipped**, and the note is kept because it
    argued two things that did not happen. The panel builds an `xdotwidget.graph.Graph` directly — no xdot
    text, no parser in the path — so the parser is *not* being kept alive by the everyday path, and a
    layout bug cannot be dumped to a file and opened in the XDot viewer. Both were real arguments; if
    either still matters, it needs a deliberate answer rather than the assumption that this route provided
    it.
  - The renderer takes a `Graph` of `Node`/`Edge` elements built from `Shape` primitives; `xdotwidget.parser` (xdot text → `Graph`) is one front-end among possible others. So emitting `.xdot` and building the `Graph` directly are both possible, and neither needs the `dot` binary. **Decided 2026-07-29: emit xdot**, provided the parse cost is negligible at chat-tree sizes (check before committing to it — parsing runs on tree change, not per frame, so the bar is low).
    - Two independent reasons, either of which would do. **Keeping the parser alive:** code exercised only by the XDot viewer — a peripheral app someone opens occasionally — can break and stay broken until a user trips over it, whereas the same code on the everyday path fails loudly, immediately, in front of a developer. More shared code on the hot path is buying maintenance, at the cost of a round-trip we can afford. **Debuggability:** a layout bug can be dumped to a file and opened in the XDot viewer, which is worth real time for a view whose entire difficulty is positions.
  - **Placement: the chat tree occupies the avatar panel's rect exactly.** In the classic mode it is toggleable and overlays that panel when open; in a no-avatar mode it simply lives there. So this is *one rect with alternative occupants*, not three layouts — the simplest DPG shape being two child windows sharing the rect the resize handler already computes, shown and hidden, rather than a true overlay with its own z-order.
    - **Pause the avatar while it is covered** — the frames are invisible, so rendering them is pure waste, and pausing hands the classic mode a slice of the same GPU and battery saving the no-avatar mode is for. The mechanism exists: `avatar_renderer.pause(action="pause"/"resume")`, currently driven only by the idle-off timeout in `avatar_controller`'s `emotion_autoreset_task` (`idle_off_timeout`, 15 s by default). What is needed is an **AND gate** — render only when the idle detector says active *and* the avatar is visible at all — since today activity resumes the video unconditionally, covered or not.
      - Watch where the gate goes. The existing pause branch is guarded by `config.idle_timeout is not None`, so with idle-off *disabled* it never runs. A visibility term added inside that condition would be dead exactly for the users who turned auto-off off, who then keep rendering a covered avatar. Visibility has to be able to pause on its own, not only as a term in the idle path.
  - Since we lay out ourselves, keep positions **stable across incremental changes** — the tree gains a node per turn and a branch per reroll, and anything that re-positions existing nodes makes the picture jump while it is being read. Reingold–Tilford tidy-tree layout is the textbook fit and can hold existing nodes in place. (Complements the visible-depth limit above rather than replacing it: depth-limiting bounds the *cost*, stability bounds the *distraction*.)
  - The natural home for this view is the panel a **no-avatar mode** frees up — see `TODO_DEFERRED.md`, "A no-avatar mode…". The two want designing together, since the panel is what the mode varies.

- **[Medium]** Switch HEAD by chat node ID: exported chatlogs report IDs; allow jumping directly to a node; show "not found" error if node doesn't exist in this Librarian instance.

- **[Medium]** Chat panel improvements:
  - Double-buffering for UI calmness during rebuild (not a performance issue, a smoothness issue)
  - Scrollability during LLM stream: add "user touched scroll controls" flag; disable auto-scroll when set; clear flag on appropriate events

- **[Medium]** Save/show full prompt per AI message: save the exact prompt at message-generation time (cannot reconstruct it later — system prompt may have changed, tree datastore doesn't preserve it). Likely needs a separate datastore with full prompt duplication. Show prompt in GUI with token count; copy to clipboard.

- **[Medium]** Bilingual chat display / on-demand translation of user input. Raven is English-only because Qwen (and Gemma, and Gemini) understand Finnish but can't *produce* acceptable Finnish. Translating Finnish *input* into English is feasible — `opus-mt-tc-big-fi-en` is already in `server/config.py`'s `translation_models`, commented out to save VRAM on smaller setups — but it needs UX work, not just the model:
  - A silently-applied wrong translation is worse than no translation, so auto-translated text must be prominently marked as such.
  - The original wording must be preserved in the datastore, never replaced by its translation.
  - Likely shape: "translate this message" / "translate conversation" actions, or a dual-language overlay showing both. Expect a couple of prototypes before settling — this is a UX design problem more than a plumbing one.

- **[Medium]** Robustness: temporarily disable relevant buttons while AI is writing; re-enable correctly by checking whether the relevant action has a stashed callback for that specific displayed chat message.

- **[Medium]** Resume a reply cut off mid-thought: Continue after an interruption during thinking leaves an incomplete thought block, where it should pick up the thinking where it stopped. A defect, not a limitation (Juha, 2026-08-26); continuing a reply cut off *mid-answer* is right as it is. May have a reproducible case still in the persistent chat tree — investigate. `TODO_DEFERRED.md`'s "Edit an AI reply's thinking trace" waits on this.
  - **Probable mechanism, found 2026-07-27 while probing backends.** Continue works by prefilling the partial reply as a trailing assistant message — and a trailing assistant message means the template emits no generation prompt, which is exactly where the thinking prefix comes from. So the continued turn **cannot re-enter the thought channel**: if generation was interrupted mid-thinking, the block has no way to be closed, which is the reported symptom. Corroborated by Juha's recollection of the manual testing that produced it — the interruptions were sometimes during output and sometimes during thinking, and it is the latter that this predicts will break.
  - Still a hypothesis rather than a proven cause; the confirming test is to interrupt deliberately during thinking, then Continue, and check whether the reasoning channel reopens. Worth doing *before* designing a fix, because the fix differs: if this is the cause, resuming mid-thought needs the continuation to re-open the block explicitly (prefill the partial reasoning *inside* an open `<think>`), rather than anything in the renderer.
  - **On Qwen it looks buildable**, both ingredients already measured: an open `<think>\n` prefill forces thinking on, and a trailing assistant message is continued rather than restarted. **Untested is the composition** — an open block that already contains text — and what LM Studio's server-side reasoning parser does with a prompt ending mid-block. One request answers it; probe before building.
  - **A Qwen mechanism, not a general one**: Gemma's template puts its thinking marker in the first system turn, so there is no mirror of the trick there. It needs the same signal as the parser fix in `TODO_DEFERRED.md`, "Streaming thinking is shown as the answer until the closing `</think>` arrives", so build the two together. The full record is `investigations/thinking-toggle/`.

- **[Low]** Per-message backgrounds in the chat log — a tinted panel behind each message, keyed by role, so the eye can separate turns without reading them. Currently role is signalled by the icon and the persona name only.
  - **An abandoned attempt is in the history, and the approach is what to avoid, not to resume.** The original librarian WIP (`ef3a5d9`, Oct 2025) drew a rounded `draw_rectangle` into a drawlist positioned behind each message's container. Being a drawlist rather than a laid-out widget, it had to be told its own geometry: the message's rect size is not known until a frame after it is built, so it needed a deferred frame callback per message — and since `set_frame_callback` holds one callback per frame number, that meant a queue plus a master callback to drain it (`DPGChatMessage.callbacks` / `run_callbacks`, which is what those were for). It did run, a couple of times, during development; it was commented out before the first commit that carries it, and the queue then rode along as dead code until it was removed.
  - **Why it was dropped** (Juha's recollection, so treat it as a lead rather than as a finding): three things had to hold at once — the rect sized correctly, the box behind the message in z-order, and the box scrolling with the chat log — and it was a pick-two-of-three situation, possibly pick-one.
  - That is what makes a *widget* the likelier shape than a drawlist: a container with a background of its own is sized, ordered and scrolled by the toolkit, so all three fall out instead of being maintained. An ImGui child window takes `mvThemeCol_ChildBg` from a theme, one theme per role. Unverified — check whether a per-message child window is affordable at chat-log lengths before committing, since each one is a scroll region and a clip rect.

- **[Low]** minichat: **[Verify]** when retrieval results are `null` in `chat.json` — old bug or still present in current codebase? (CC session)


### STT / voice

- **[Medium]** STT: input-language selector in the GUI. `api.stt_transcribe` / `stt_transcribe_array` already take `language: Optional[str]` (`None` = autodetect) and the server honours it; Librarian's only call site (`app.py`, `stop_recording_audio_message`) just never passes it. So the plumbing exists — what's missing is the control.
  - **Off the Researchers' Night path** (Juha, 2026-08-25), and dropped from [High] with it: Raven is English-only for now, and language selection is future expansion. The mixed-audience argument below is the case for building it *eventually*, not this year — read it that way.
    - What covers the exhibit instead is the operator asking visitors who want to speak to the system directly to ask in English. Worth knowing before this item is re-argued from the mixed-audience premise: that premise is true and is already answered, by instruction rather than by a control.
  - A **combobox**, not a config knob: "Automatic" plus each configured input language. The language has to change *between questioners*, not at startup — a Researchers' Night audience will mix Finnish and English speakers, and switching on the fly is the difference between the mic working for everyone and working for half the room.
  - Read the selection at transcription time (the pattern `_make_open_folder_callback` already uses for directories), so a mid-session change takes effect on the next recording with no restart.
  - The offered list comes from config — Whisper handles ~99 languages, but the demo wants two. Distinct from the *subtitler's* output language (`gui_config.translator_target_lang`); don't conflate them.
  - Edge case: an English-only Whisper build (`whisper-base.en`) makes the selector meaningless. Hide or disable it when the configured `speech_recognition_model` ends in `.en`.
  - Autodetect stays worth offering but shouldn't be the only option: Whisper's language detection is least reliable on short utterances in a noisy room, which is exactly a live Q&A.
  - **Show what was detected.** When the selector is on "Automatic", briefly flash the detected language code somewhere unobtrusive — a corner of the avatar panel is the natural spot. Not cosmetic: a misdetect currently fails *silently*, producing a plausible-looking transcription in the wrong language, and this converts it into something the operator can see and correct. Only worth showing in Automatic mode; when the language is pinned, it's noise. Reuse the existing `animation` flash machinery, and note it stays legible under the colorblind-signaling item since a language code is text rather than a color.
    - **Prerequisite: nothing returns the detected language today.** `server.modules.stt.speech_to_text` returns a bare `str` and `api.stt_transcribe` returns `List[str]`, so the detection is discarded at the engine boundary. Surfacing it means changing the response shape through all three layers (`common.audio.speech.stt` → server module → client API) to carry text *plus* language. First check whether the engine wrapper can expose it at all — Whisper detects the language internally, but whether `common/audio/speech/stt.py`'s `transcribe` can hand it back needs reading. Worth doing now rather than later: the response-shape break is free while Librarian has no outside users, and gets expensive once it does.

- **[High]** Wake-word trigger for voice input. The exhibit case: a visitor speaking directly to Aria reads very differently from an operator typing on their behalf. Higher demo value than anything else in this section, and the only item here that changes *who is talking* rather than how well it is heard.
  - **Pushed past Researchers' Night** (Juha, 2026-08-25) — wanted in the lab, and later this year is fine. What made it uncosted for a four-week window is below: continuous capture fanned out to three consumers, and two interaction styles that have to be tested against real strangers rather than reasoned about. Priority stays [High]; only the deadline came off.
  - **The architectural cost is continuous capture.** `Recorder` is `start`/`stop` on demand, and `pvrecorder` is a single device handle, so a wake word cannot simply open a second stream. It needs one always-on capture fanned out to three consumers: the VU meter, the detector, and — once armed — the recording buffer. `connect_vu_readout` is the existing precedent for a consumer, and since the audio input panel (2026-08-28) the recorder's VU readout is a listener list rather than one slot, so more than one consumer is already possible; what is new is capture that never stops. The panel (F9) also has room below its peak-hold slider for this item's controls.
  - **`pvrecorder` itself is fine — this is about a *different* Picovoice package.** Checked 2026-08-10: `pvrecorder` 1.2.7 is Apache-2.0 with **no dependencies at all** — pure PCM capture, no key, no account, no network. Nothing needs replacing. The AccessKey regime applies to Picovoice's *inference* engines (Porcupine, Cheetah, Leopard, Rhino, Orca).
    - Note the tell, since the licence field does not carry it: `pvrecorder` and `pvporcupine` declare the *identical* Apache-2.0 classifier, and what distinguishes them is that `pvporcupine` 4.0.3 depends on `requests`. A wake-word engine that runs entirely on-device has no reason to need an HTTP client; that dependency is the activation call, visible in the package metadata. **For anything claiming to run locally, read `requires_dist` before the licence field.**
  - **Engine choice, and the obvious one is wrong here.** `pvrecorder` being Picovoice's makes `pvporcupine` the natural technical fit — same frame conventions, designed to be fed from it. But Porcupine has required an AccessKey and online activation since v2.0, its free tier is evaluation-only, and a user is allowed **one unique device**, with reports of containerized runs registering as a new device and locking the account out. Phoning home for activation, bound to a machine, is a single point of failure on exhibit night — the kind that cannot be debugged with a queue of visitors waiting. `openWakeWord` is the usual open alternative (ONNX, on-device, no key); **verify its license and its custom-word training path from the project itself**, since the readily-found comparisons are competitors' marketing pages.
  - **Intended approach: build it, using the STT already present** (Juha, 2026-08-10). A ring buffer of recent audio, gated by VAD or the existing energy threshold, transcribed in short windows and matched against the wake word. This removes both risks above — no licensing question, and no custom-model training for "Aria", which is a non-stock word under every engine and was the schedule risk worth sizing first.
  - **Two interaction styles, and which suits this audience is an open question — test both** (Juha, 2026-08-10). They are different HCI, not one being a better implementation of the other.
    - **Two-phase**, the *Star Trek* form: "Aria" — cue — "what's the atomic number of hydrogen?" The cue is *feedback*: the visitor knows they were heard before committing to a question, and failure is legible. Also cheaper technically, since detection and transcription separate cleanly with no query to reconstruct from the buffer.
    - **Single-breath**: "Aria, what's the atomic number of hydrogen?" Natural, no convention to learn, and it falls out of the ring buffer at no extra cost — when a transcription *begins with* the wake word, the remainder of that utterance is the query.
    - The considerations pull opposite ways and the audience decides it. Strangers with no model of the system arguably need feedback more than naturalness: with single-breath there is nothing to look at until the answer starts, so an unsure visitor repeats themselves mid-transcription and corrupts the query they already gave. Against that, a convention has to be explained, which at an exhibit means signage or the operator saying it each time.
  - **The confirmation channel is worth testing separately from the style, and it wants to be both** (Juha, 2026-08-10). An audio cue can be missed in a loud room; a visual one is missed by anyone not looking at the avatar, which in a lab is most people. Redundant channels for one signal — the same argument the colorblind-signaling item makes about ok/error flashes.
  - **The visual side is buildable with what THA3 already has**, with three pieces of work:
    - **Head and eye morphs** point the character at the user. The complication is the idle animation: it changes body and head angles continuously, so the morph values that mean "looking at you" move with it. **Do not assume the compensation is arithmetic** — THA3 is a neural net, so the eye morphs' effect may *depend on* the head and body values rather than adding to them, an interaction term rather than a scale factor, in which case composition is wrong rather than imprecise. The animator knows the values it *sends*, not what the net does with them.
      - **Measure the error before characterising the mapping.** The question that decides the work is not "what is the mapping" but "is the naive composition's error visible" — gaze tolerance for *looking at you* is generous, and a few degrees reads fine. So: implement naive composition, run the idle animation through its range against a fixed gaze target, screenshot at the extremes, and look for visible drift. Minutes, not a sweep. If it holds, skip the calibration; if it drifts, the sweep is warranted and you already know from which direction. Pose editor plus parameter sweep plus screenshots is the instrument for that second stage.
    - **animefx** can put the visible "\ | /" over the character's head. ~~It currently fires only on emotion changes~~ — **generalized 2026-09-15**: `DPGAvatarController.trigger_animefx(config, "notice")` plays a named effect whatever the emotion, and Librarian's `Ctrl+P` ping already uses it. It waits for the video if the avatar is asleep.
    - **With no head tracking, v1 assumes the user sits about at the centre of the monitor's top edge** (Juha, 2026-09-15), and looks there. The ping is the other place this is wanted: pinging the avatar should eventually have it look at the user too.
    - **Per-character opt-in is a requirement, not a nicety.** The effect looks right on Aria and wrong on the researcher DT, which has no animefx configured. So the effect set belongs in character config rather than being global.
  - **The risk moves rather than disappearing, and it lands on the exhibit's exact condition.** Whisper hallucinates fluently on noise — a crowd murmur transcribes to confident text. A purpose-built KWS engine is trained against hard negatives precisely so it does not; a general ASR model has no such training. So this route trades a licensing problem for a false-accept problem, in a loud room, where a spurious trigger mid-answer is the failure that looks broken. Design in from the start: require the match at the *start* of the utterance, require energy above threshold, require a minimum utterance length, and use whatever no-speech signal the model exposes.
  - **Cost competes with what the demo needs.** Continuous Whisper on a GPU also running the LLM and THA3 is not free. Gating on VAD or the existing silence threshold means detection runs only while someone is actually speaking, which is most of the saving. A two-tier model — small for detection, full after trigger — is the next step at the cost of a second resident model; **record it, do not schedule it**, since gating may make the tier unnecessary.
  - The ring buffer earns its place under either style: with two-phase it still covers the visitor who starts talking before the trigger has registered.
  - **Interacts with the input-language item**: a Finnish/English audience means the wake word must be recognized under both, and Whisper's rendering of "Aria" may differ by decoding language. Worth testing both before the day.
  - **Room constraint, shared with the two items above.** An open-doors evening is loud, and *false accepts are worse than false rejects*. So it wants the same tune-it-in-the-room control the silence-threshold item specifies, and a push-to-talk fallback that can be switched to on the day without a restart. **Design the three STT items together** — they share the constraint and the GUI surface.

- **[High]** Finnish demo path — end-to-end test. The chain that lets a Finnish-speaking audience interact without the LLM ever producing Finnish: Finnish speech → Whisper (multilingual, language selected in the GUI per the item above) → Finnish text → the LLM *understands* it → answers in English → TTS speaks English → the subtitler translates to Finnish. Every hop is believed to work; the whole has never been run. Test the English-input path through the same chain too — a mixed audience is the expected case, not the exception. Two known gaps: `speech_recognition_model` is `openai/whisper-base` (74M, chosen for CPU) which will be rough on Finnish — `whisper-large-v3-turbo` (~1.6 GB) is commented out two lines above in `server/config.py` and is affordable on a mid-VRAM setup — and nobody has tested the chain with a real Finnish question.

- **[Medium]** `raven-transcribe`: command-line tool for transcribing audio files or mic input. (`-p` for prompt, `-o` for output file, stdout by default.) Potential for podcast analysis.

- **[Medium]** Proper name extraction via spaCy NER: extract proper names from chat log, fill into STT prompt as a comma-separated list (improves transcription of names).

- **[Low]** Voice command interface: split transcribed text to words, check first two words for command prefix, trigger command processor for the rest. Low priority.

- **[Medium]** Long subtitle splitter — **for 0.2.10** (Juha, 2026-09-24). The subtitler shows one card per
  sentence, so a long sentence becomes a card of up to ten lines covering half the avatar; professional
  subtitling splits a sentence across several cards. The timing is already there: TTS returns per-word
  timestamps (they drive the lipsync), so each part can go up when its first word is spoken, rather than
  dividing the sentence's audio length evenly. Open question: where to split in the *translated* text, whose
  words do not line up one-to-one with the spoken English — splitting the English first and translating each
  part is the simple answer, at some cost in translation quality across the cut. Seen while recording the
  manual's avatar clips, which were chosen to avoid the long cards.

- **[Low]** Edit spoken message before sending.

- **[Low]** Look into quantized whisper-large-v3-turbo to save VRAM (~1.6 GB currently). May need vLLM backend.

- **[Low]** STT known issues (still open):
  - Spurious text generated after speech ends in long audio (see `raven.client.tests.test_api`)
  - Test `stt_transcribe_file` and `stt_transcribe_array`


### LLM backends

- **[Medium]** An exact token count where neither existing tier reaches (raised by Juha, 2026-08-27; narrowed 2026-09-08). `count_tokens` has three tiers, the first two exact: a configured local tokenizer (`gguftokenizer`, offline and backend-agnostic), then oobabooga's `/v1/internal/token-count`, then a calibrated tokens-per-character ratio, which is the estimate the readout marks with `~`.
  - The gap is where neither exact tier applies: the `.gguf` is not on a local path *and* the backend is not ooba. That is the ordinary case here — LM Studio is what the team uses — so a model on another machine falls straight to the estimate. Two candidates, not exclusive:
    - A **Raven-server endpoint** with a client half in `raven.client.api`, for when the server sits beside the LLM backend and the apps are elsewhere.
    - **Asking the backend to count, offered as a tier of its own.** The mechanism already runs: `gguftokenizer.load` verifies itself with it — two short probes compared by their *difference*, so the chat template's framing cancels — and it is backend-agnostic in a way the ooba tier is not. It is simply not offered for counting. Probably should be (Juha, 2026-09-08). The readout updates on the idle-prefill settle rather than per keystroke, so two round-trips per recount is cheap; unmeasured as a counting path.
  - **Installing ooba to widen the second tier is explicitly not the answer** (Juha, 2026-09-08).

- **[High]** **Support Anthropic-style backends**, alongside the OpenAI-compatible ones. Raised 2026-08-07 (Juha). Two reasons, and the second is the one that outlives the first:

  - **Reach.** A significant fraction of scientific users are on Claude, and today Raven cannot talk to them at all.
  - **Brand neutrality, as a stated position.** *Bring your own backend* — the same stance Raven takes on NVIDIA versus AMD, where development happens on NVIDIA but nothing in the stack is supposed to *require* it. A local-first research tool that works with exactly one vendor's API shape has made a choice it did not mean to make, and the longer only one shape is supported the more the code quietly assumes it.

  **Testable locally, which is what makes this tractable now**: LM Studio serves an Anthropic-compatible endpoint, so the whole thing can be developed and tested against a local model with no API account. (This project has no Anthropic API account to test against — its Claude access is through Claude Code.)

  **What that endpoint does, probed 2026-07-27 against `qwen3.6-35b-a3b`** — and it adds a third reason: **a working per-request thinking toggle, which LM Studio's OpenAI endpoint does not have.**
  - `thinking: {"type": "disabled"}` → content blocks `['text']`; `thinking: {"type": "enabled", "budget_tokens": N}` → `['thinking', 'text']`. A genuine toggle, in Anthropic's own spelling, with no prefill needed.
  - The endpoint **defaults to thinking off**, the opposite of the OpenAI endpoint's default-on. Worth knowing before comparing behaviour across the two.
  - It **streams** — proper SSE, `event: message_start` and Anthropic-shaped events — so `llmclient`'s stream parser has something to attach to.
  - Assistant prefill works there too.
  - Response carries `stop_reason` and `usage.cache_read_input_tokens`, i.e. the real Anthropic shape rather than a thin alias.

  Where it lands in the code: `llmclient.detect_backend_flavor` already probes by *payload shape* and returns `"lmstudio"` / `"oobabooga"` / `"generic"`, and `backend_flavor` already gates request details at a handful of sites (`_resolve_model_info`, the continue flag, the sampler block). So the seam exists. What is genuinely different about the Anthropic shape, and needs designing rather than switching on:

  - **`system` is a top-level request field, not a message with `role="system"`.** Raven's history is a list of role-tagged messages and the system prompt is a node in the chat tree, so the wire builder has to lift it out.
  - **Tool calls and results are content *blocks*** (`tool_use` / `tool_result`) inside a message, rather than a separate `tool_calls` field plus `role="tool"` messages. Raven's content is already a typed-parts list, which is the right shape to map from — but `perform_tool_calls` and the streaming parser both speak the OpenAI spelling today.
  - **The streaming event protocol is different** (`message_start` / `content_block_delta` / …, not OpenAI's `choices[].delta`). `StreamParser` is the one place that would have to grow a second dialect; it already emits typed events, so the parser changes and its consumers do not.
  - **Thinking blocks arrive as their own content block type**, which is closer to what Raven wants than the `reasoning_content` field it normalizes to now.

  The four differences above are from memory of the Anthropic Messages API, not read off the docs in the session that wrote this — check them against the current spec before designing to them, and check what LM Studio's compatibility endpoint actually implements, since a compatibility layer is free to support a subset. The claims about *Raven's* side (what `detect_backend_flavor` returns, where `backend_flavor` is consulted, that `StreamParser` emits typed events) were read from the source.

  Open question worth settling before writing code: whether this is a *flavor* of the existing client or a second client behind a common interface. The flavor gates are cheap while the differences are per-field; a different streaming protocol and a different tool-call representation may be past that line.

### Tools

- **[Medium]** A separate toggle for MCP tools, once brief 04 lands. Different trust surface from either **Internet** or **Documents**, so it wants its own group in `llmclient` (a third alongside `NETWORK_TOOL_NAMES` and `DOCUMENT_TOOL_NAMES`) rather than being folded into one of theirs. The grouping mechanism is in place and takes one more entry; what needs deciding is whether one switch covers every MCP server or each server gets its own, which is a question about how many the user is expected to run.

- **[Medium]** Weather tool, via open-meteo (https://open-meteo.com/en/docs) — makes Librarian more humanlike as a "voice with internet access" (HCI is a major Raven goal). Parked in brief 01 §6. Its sibling, the calculator, shipped as `llmtools.calculate` on `simpleeval`.
  - **The shape wanted** (maintainer, 2026-09-30): answer *"What's the weather like in Tampere, Finland today?"*, plus possibly a 24-hour or a week's forecast as a table. So one place-name-in tool with a horizon, not a set of fine-grained endpoints.
  - **Decided: a built-in** (maintainer, 2026-09-30), over `open-meteo-mcp` once brief 04 lands. The MCP server publishes many narrow tools, which is the wrong grain for that question — each would sit in the tool list every turn, and the model would have to chain geocoding and forecast itself. A specific tool covering the common case beats that, and the app is a librarian rather than a meteorologist. It is small: open-meteo's geocoding and forecast endpoints are free JSON with no key. **It lives on Raven-server** (maintainer, 2026-09-30), beside the web tools, and where brief 13 expects the document DB to move.

- **[Medium]** Calendar tool: get one- or three-month calendar, like the `cal` command-line utility. See Python's `calendar` module.

- **[Medium]** `webfetch`: content-aware extraction per site, so a fetch opens with the part a reader actually wants. Generic extraction takes the page in document order, which on a Wikipedia article means the infobox — a fetch of *Corvidae* opens `| Kingdom: | Animalia |` and spends the whole excerpt on taxonomy boxes before reaching a sentence of prose (observed 2026-08-04). It reads as broken even though nothing failed, and it is exactly the kind of thing an ideal librarian product gets right.
  - **The hook already exists**, so this is extension rather than construction: `webfetch._rewrite_url` returns an optional per-site extractor, and arXiv, Reddit and YouTube already use it (`_extract_arxiv`, the `old.reddit.com` rewrite, `_extract_youtube_transcript`). It is pure and separately unit-testable by design — the rewrite decision is tested without touching the network.
  - **First version: Wikipedia and arXiv.** Wikipedia is the one that is visibly wrong today — lead section first, infobox after or dropped; it has a REST content API that returns exactly the lead, which would sidestep the extraction problem rather than fight it (worth verifying before committing to it — this is recalled, not checked). arXiv already has an extractor and mostly works; the remaining nit is that a Google-permissions blurb precedes the abstract on the HTML rendering, so the abstract starts ~350 characters in.
  - **Then, for scientific users**, roughly in order of how often they will hit them: `doi.org` (currently resolves to a publisher page that is often a login wall — Crossref content negotiation returns real metadata for a DOI whether or not the fulltext is reachable, which turns a useless fetch into a citation); PubMed / PMC (E-utilities for the abstract, PMC OA for fulltext); bioRxiv / medRxiv (public API, and preprints are open by definition); Semantic Scholar (abstract plus a link to any open-access PDF). All four API claims are from memory — check each before building on it.
  - **A second symptom, same root: a page that is not an article at all.** `https://astronomynow.com/2026/` is a year index; generic extraction returned 1398 characters that stop mid-sentence — *"…now there's someone up there from my own country, and while"* — because it latched onto one entry's teaser and ran out. Verified against the server directly, so the truncation is in extraction rather than anywhere downstream (`spaSuspected` false, so no second-tier retry was even attempted). Worth handling explicitly: an index or listing page wants either its list of links (which is what the reader would click) or a refusal saying it is a listing, not half of one item presented as the content. The failure is quiet, which is the dangerous part — the model receives a truncated sentence with no marker and no way to tell it apart from a short article.
    - **`docextract` now detects this failure, and the detector may port over — the threshold does not.** A saved page holding one `<article>` per chapter had the same thing happen: readability extraction chose one block and silently dropped the rest. The fix there compares the extraction against `trafilatura.html2txt` (the whole page, no block selection) and treats a small ratio as truncation rather than boilerplate removal, then re-extracts per `<article>` to keep the Markdown. The *mechanism* is shared with this item and the remedy might be too. The *constant* is not transferable: it was calibrated on locally saved pages, which carry almost no chrome, whereas a live page legitimately loses a fair fraction to navigation, ads and related-article blocks — so reusing 0.5 here could fire the fallback on pages that were extracted correctly. Recalibrate against live fetches before porting.
  - **The general principle worth extracting from the specific cases**: when a site has an API that returns the content as *data*, prefer it to scraping the rendered page. Extraction from HTML is a heuristic recovering structure that was thrown away; an API hands the structure over. That is also what makes these testable — a fixture of the API response, rather than a snapshot of a page that will be redesigned.

  Raised by Juha 2026-08-04, from the Wikipedia excerpt.


- **[Medium]** Websearch: **[Verify]** whether raw URLs are currently saved in tool results. Remaining work: final formatting of results, link crawling to retrieve full result documents (persist to RAG with expiry timeout), figure out in which contexts search result pages should be enabled as RAG data sources.

- **[Medium]** HybridIR pedigree field: auto-remove only documents added by a named scanner instance. Needed for programmatic RAG ingestion (e.g. web pages from websearch).

- **[Medium]** Source attribution for RAG: clickable snippets in GUI based on `document_id`, `offset`, length; clickable link to open full document (spawn external viewer based on file type). Same feature as `TODO_DEFERRED.md`, "Expose the docs-DB source files behind a reply's RAG citations" — that entry carries the current design questions (where the affordance lives, snippet vs. whole document) and notes the `open_file` / `open_in_file_manager` machinery it can reuse.

- **[High]** MCP support: specced in `briefs/librarian-extension/04_librarian-mcp-client-brief.md` — client-side MCP tools registered *alongside* the built-ins, all feeding the existing `perform_tool_calls` loop. Gated on the Hindsight playground (brief 06). Main line for the "digital colleague" track: this is how Librarian reaches the lab's systems. Agent skills (CLI-based, "anime maid form factor" — plugging into interfaces designed for human use) remain a superior alternative capability-wise but more dangerous for the user's computing environment; still under consideration as a separate path.

- **[Low]** IBM Granite OCR / vision OCR: low priority. Since writing this item, DeepSeek-OCR and Qwen3.5 native vision have appeared. Evaluate accuracy/speed/model size tradeoff when relevant.

- **[Parked]** Translator upgrade. Current: `Helsinki-NLP/opus-mt-tc-big-en-fi`, sentence-level only, so it misses whatever needs broader context to disambiguate. Surveyed 2026-07-27; nothing clean is available, so this stays parked until HPLT v2 ships HF weights:
  - **HPLT v2 en-fi** (https://huggingface.co/HPLT/translate-en-fi-v2.0-hplt_opus) — still Marian-format only. The card says "we are working on converting it to the Hugging Face format", with no timeline. Would need a second backend, which is why it was parked in the first place.
  - **HPLT v1.0 en-fi** does ship HF-format weights, but the card documents a conversion defect: the checkpoint "cannot work with transformer versions <4.26 or >4.30" (recommends `transformers==4.28`). Raven-server shares one `transformers` across classify / embeddings / Whisper / translate, so that pin is unaffordable. Dead end — recorded so it isn't re-investigated.
  - **NLLB-200** — CC-BY-NC. Blocked by the commercial partners who want to use Raven, independently of quality.
  - **MADLAD-400** (CC BY 4.0, T5-based, 3B/7B/10B) — the only license-clean transformers-native candidate, but far heavier than a ~200 MB opus-mt for a subtitler, and multilingual rather than (en, fi)-specialized. Worth a spot check, not a plan.
  - **EuroLLM 9B** — tested 2025, output unusable. 28 EU languages in 9B, with Finnish and Estonian the only Fenno-Ugric ones. Don't re-test.
  - **Aya** — 20–30B class. No VRAM headroom: that budget is spent on the LLM itself.
  - If the LLM route is taken at all, the translator has to *be* the main LLM (Gemma 4 handles Finnish better than Qwen 3.6 but is less capable overall). For production the tradeoff favours intelligence — English is fine. Demo requirements differ; see the Finnish demo path under Chat UI.

- **[Parked]** User persona sampling / prefill: functional utility for local model testing, but deferred for now.


### Avatar (Librarian-side)

- **[Medium]** Avatar on/off toggle, so Librarian never loads the avatar on a low-VRAM setup (auto-off already exists): `TODO_DEFERRED.md`, "A no-avatar mode, with the chat tree in the panel the avatar vacates", which also answers what the right panel shows instead.

- **[Low]** Tune the branch-switch glitch's look by eye. The effect shipped 2026-08-25; its parameters, and why its ceiling wants re-checking, are in `briefs/done/researchers-night/README.md`.

- **[Medium]** Avatar: do more to eliminate stutter while receiving LLM response. Happens especially at first avatar speech in a session and while TTS is rendering in the background. Pushing limits of 3070Ti. Investigate audio buffer size (see `raven.client.util`) and rendering smoothness under high system load.
  - **Deprioritized 2026-07-28**: it shows mainly at the start of a session, which a warm-up handles without knowing the cause, and a slight stutter on the first spoken sentence is within what games routinely ship.
  - Hypotheses still worth the eventual look: warmup, GPU contention, or the GIL. The voice is already warmed at startup, so "first speech" warmup is weaker than it sounds; what is still cold at that moment is THA3's first inference at the talking-morph shape, the audio device open, and the postprocessor's first pass. Test whether it survives TTS on the CPU, which would rule contention out.
  - *The moment is not startup.* The stutter is on an avatar that has been streaming frames steadily for a long time — Librarian starts the session well before anything is spoken — so hypotheses about session-start costs are aimed at the wrong moment.
  - *VRAM is the wrong instrument.* A stutter is dropped or late frames; the measurement wanted is **frame inter-arrival timing** across the speech transition, with the caveat that frames reach a client over HTTP, so client-observed timing carries scheduling noise that server-side instrumentation would not.
  - The entry point is easy to get wrong: `avatar_start_talking` is the randomized-mouth *idle* animation, not lipsync. Real speech goes through `raven.client.tts.tts_speak_lipsynced`; for how an application drives it, see `raven.client.avatar_controller.speak_task`.

- **[Medium]** `DPGAvatarRenderer`, `DPGAvatarController`: isolate DPG-specific parts for portability.

- **[Low]** Draw per-character AI chat icons for all characters (e.g. `aria1.png` → `aria1_icon.png`, RGBA 64×64).

- **[Parked]** Avatar vector emotions: blend several emotions by classification values; normalize appropriately. Low priority.


### Robustness

- **[Medium]** Don't crash if `tts` module isn't running.

- **[Low]** RAG: **[Verify]** whether chunk full-IDs are listed in retrieval metadata for combined contiguous chunks. (CC session)


---

## Server

- **[High]** Expand the server config-variant set. `device_string` is already per *module*, not just per config, so any split across two GPUs is a config edit rather than a code change. Existing: default `config.py`, `config_lowvram.py`, `config_avatar_only.py` (avatar testing / settings editor). Wanted, so the right one can be selected on the CLI at server start.

  Two axes: how much VRAM the GPU serving raven-server's modules has, and whether there is a *second* GPU the LLM gets to itself. Name by capability tier, not by hardware — an installing user knows their card's VRAM, not our machines. Proposed (naming still open):

  | Config | Server-module GPU | LLM |
  |---|---|---|
  | `config_lowvram.py` (exists) | ~8 GB | shares the same GPU |
  | `config_midvram.py` | ~16 GB | shares the same GPU |
  | `config_dual_lowvram.py` | ~8 GB | dedicated second GPU |
  | `config_dual_midvram.py` | ~16 GB | dedicated second GPU |

  `config_dual_midvram` is the demo configuration: LLM alone on the larger card, all nine server modules on the internal one. Tiers extend upward (`high` ≈ 24 GB, `extreme` ≈ 32 GB+) as hardware warrants; don't create empty cells in advance.

  **`config.py` stays the default** — a server that won't start without a CLI flag fails the "installable by any half-tech-savvy person" bar. Two ways to fill that role, and they're worth deciding between explicitly:
  - **Auto-tiering default.** `config.py` reads available VRAM at import (`raven.common.deviceinfo` already does the detection) and selects a tier, logging loudly which one it picked and what flag overrides it. Works out of the box, and the log line is where the user learns the explicit configs exist. Cost: more magic to reason about when it guesses wrong.
  - **Fixed conservative default.** `config.py` is simply the lowest tier that's still useful. Predictable and trivially debuggable, at the price of under-using good hardware until the user discovers the flag.

  Either way, document the tier → approximate-GB mapping in a header comment and as a "your hardware → this config" table in the README — picking the right one is the user's first decision after install, and the default only has to be *good enough to start*, not optimal.

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

- **[Low]** Re-measure what the server costs in VRAM, where the 2026-07-28 figures no longer describe what ships: `imagefx` with `crt` and `atmospheric_dust` in its chain (it measured 0.00 with the chain empty), and `embeddings` at the Nomic switch. Peak during use for the other modules only if the LLM budget turns out tight. Figures and method in `investigations/vram/README.md`.


---

## Avatar

- **[Low]** Implement JS client for integration of Avatar with other LLM frontends. Needs work on those other frontends, too. Initially, target SillyTavern.

- **[Low]** Update assets for all characters: add at least the eye-waver effect (and possibly other cel-blending cels). Aria is the default character with full feature support. Other characters are lower priority.


---

## XDot Viewer (`raven-xdot-viewer`)

- **[Low]** A switch on `XDotWidget` turning its pan/zoom animation off, per instance (agreed 2026-09-09, unscheduled).

---

## Papers tooling (e.g. pdf2bib, csv2bib)

- **[Medium]** pdf2bib: prompt the author extraction step to return a canonical string (e.g. "No authors provided") when no authors are found. Same for title extraction (e.g. "No title provided"; also handle the case where the LLM thinks the title is literally "Abstract").

- **[Medium]** Some LLMs behave erratically when the system date is later than their training cutoff (e.g. refusing tasks, claiming to be in a simulation). Investigate mitigation strategies; may be model-version-specific. Track across model upgrades. First seen in pdf2bib, but **not a pdf2bib problem** — it follows the date inject, so it applies anywhere Raven tells a model what day it is, which is every Librarian turn (`scaffold._perform_injects`). Measured groundwork exists: `investigations/context-injects/datetime_inject.py` asks exactly whether a model believes us over its own priors.
  - **The mild form is the one that will persist, and it costs tokens rather than correctness.** Qwen3.6-35B, 2026-08-04, asked about Artemis after a websearch: it reconciled a snippet dated April 2026 against its own priors out loud, at length — *"in reality (2024/2025 knowledge), Artemis II is scheduled for late 2024/early 2025 … I must check if there is real news about Artemis II delaying to 2026 or if the snippet is just a hypothetical/future-dated article or if I am misinterpreting the date"* — before accepting the injected date and proceeding correctly. No refusal, no simulation claim, right answer; just a large slice of the thinking budget spent re-deriving that the present is the present, on every turn where retrieved material carries a date.
  - So the success criterion is not "does it refuse" but "how much deliberation does the date cost". That is measurable with the existing probe, and it is worth measuring per model as the fleet upgrades: the erratic form may be disappearing while the expensive form stays.

- **[Medium]** pdf2bib overthinking / token-limit mitigation: detect token-limit-exceeded in `raven.librarian.llmclient`, return a status flag in metadata. Consider executive-function simulation via LLM (in the neuropsychology sense: https://en.wikipedia.org/wiki/Executive_functions) as a recovery strategy — but may be superseded by improved model capabilities; monitor before investing time.



---

## Infrastructure and maintenance

- **[High]** Unit tests. Currently very sparse. Would significantly improve confidence in refactoring.

- **[Low]** Post PR of adopted FileDialog fixes upstream. Raven's extensions have genuine added value worth sharing. Upstream is likely inactive but the PR is worth filing.
  - **The obstacle is not the patch, it is the dependency.** `fdialog` now imports `raven.common.gui.animation` for its button-flash acknowledgments and `raven.common.utils` for the Find field's matching, so the changes cannot be lifted out as a diff. Upstreaming means either reworking those two call sites to stand alone, or offering the animation framework alongside — which DPG lacks entirely, and which is why it exists here.
  - Worth deciding *what* to offer before doing the work: the parts that stand alone (multi-extension type filters, case-insensitive extension matching, the drives-list fix, the sortable table, the save-mode overwrite confirmation) are separable from the parts that do not.

- **[Low]** Fork kokoro/misaki and bump their Python upper bound (`<3.13` → `<3.15`), then test on 3.13+. The `<3.13` cap may be precautionary rather than reflecting real incompatibility. kokoro appears effectively abandoned upstream, and it's the only TTS engine that provides timestamped phoneme data (required for avatar lipsync). Currently Raven's `requires-python` is narrowed to `<3.13` to accommodate this.
  - **Consider forced alignment before forking anything (noted 2026-07-28, unverified).** The requirement is not "a TTS that reports phoneme timings" but "phoneme timings", and those can be recovered after synthesis: align the generated audio against the text it was generated from, and read the boundaries off the alignment. torchaudio ships a forced-alignment API for this. If it works, **the constraint that pins this whole item disappears** — engine choice reopens to whatever sounds best or runs smallest, kokoro stops being load-bearing, and the Python cap can be lifted by swapping the engine rather than by forking an abandoned one. It would fit the three-layer pattern cleanly (alignment is `raven.common`, engine-agnostic) and the lipsync driver already consumes `WordTiming` objects rather than anything Kokoro-shaped, so its input contract would not change.
    - **Checked 2026-07-28, and the obvious objection does not apply.** The worry was phoneme-level granularity: word alignment is easy, phoneme alignment is not, and the mouth morphs are driven per phoneme. But `lipsync.build_phoneme_stream` already splits each word's timespan *linearly across its phonemes* — Raven has never had phoneme-level timings and does not use them. `WordTiming` carries word, phoneme string, start, end. So the requirement decomposes into **word-level timings** (which `torchaudio.functional.forced_align` provides out of the box, with CPU and CUDA implementations, via `Wav2Vec2FABundle`) and **a phoneme string per word** (G2P, which is independent of the TTS engine). Neither needs a phoneme-aligning model. Alignment would also be running on synthetic speech — clean, no noise — which is the easiest case for a CTC aligner.
    - **The real risk is the phoneme inventory, not the timings.** The morph map does `vocabulary[phoneme]`, and misaki emits IPA (`mˈaɪnd` → `m, ˈ, a, ɪ, n, d`). A replacement G2P must emit a compatible inventory or the vocabulary needs remapping. Check that before anything else.
    - Still to weigh: added latency, and one more resident model (wav2vec2-class, a few hundred MB — the VRAM measurement above says there is room on both cards).
    - Prompted by hitting the same wall twice. KittenTTS: asked upstream, no reply. pocket-tts: **confirmed no native timestamps** (kyutai-labs/pocket-tts issue #66); upstream's position is that they would point users at a pipeline rather than build one in. Note their larger Kyutai TTS 1.6B *does* report word-level timings, so this is a model-tier decision rather than a house policy.
  - **Decision, 2026-08-10: keep both, do not act now.** Kokoro stays as the TTS engine, torchaudio stays as a dependency, and `requires-python` stays capped at `<3.13`. The cost of keeping Kokoro is being pinned below Python 3.13, and that only becomes forcing when 3.12 goes end-of-life in **October 2028** — two years out, during which agentic build-out continues. Charted rather than urgent.
    - **The escape route above has decayed, and this is the part that needed writing down.** The *analysis* stands: Raven has never used phoneme-level timings, so no phoneme-aligning model is needed. What changed is the dependency. **torchaudio stopped shipping** (checked on PyPI 2026-08-10): 2.11.0 on 2026-03-23, released the same day as torch 2.11.0, and nothing since — while torch has shipped 2.12.0, 2.12.1 and 2.13.0. Torch and torchaudio minor versions must match, so **adopting torchaudio now would pin torch to 2.11.0**, which trades the Python cap for a torch cap, and that is the worse of the two: torch pins drag CUDA, THA3, Whisper and the embedding stack along with them.
      - Note the shape of the miss, because it is cheap to avoid and was not part of an otherwise thorough feasibility analysis: the 2026-07-28 note was written four months into torchaudio's silence and asked every question about the *technique* and none about whether the dependency was still being released.
      - **Update 2026-10-01: the decay reversed, and the route is open again.** torchaudio 2.11 turned out to be built forward-compatible rather than abandoned — upstream's README says it works with every future torch, and it was measured running under torch 2.14.1, `forced_align` included. So adopting it no longer pins torch. It is in maintenance mode, which is a weaker reason to hesitate, but not a cap.
    - **Whisper timestamps were considered and rejected** (Juha, 2026-08-10): too slow, and it hallucinates at the tail. The speed objection is the decisive one — alignment sits in the interactive path, so seconds per utterance delays speech onset on every reply, whatever the quality.
    - **The technique survives; only the library was the problem.** torchaudio supplied two things: a wav2vec2 CTC acoustic model, and `forced_align`, which is Viterbi over CTC log-probabilities against known text. The model is available from `transformers`, already present transitively via `sentence_transformers`. The alignment step is a dynamic program over a lattice — small, well-specified, and squarely in-house territory for a codebase that already carries a custom xdot renderer and a GPU Lanczos scaler for similar reasons. So the escape route to re-scope is **CTC forced alignment, with the alignment step buildable in-house if it has to be**. Look for a maintained library first — that is a quick search, and if one exists it is the cheaper answer. What has changed is that a null result is no longer a dead end: the acoustic model is already available, the alignment is a dynamic program, and building it is a normal afternoon rather than a blocker. Worth a probe before it is needed rather than after.
    - **Record the trigger, not only the date.** October 2028 is the *latest* the cap can bite, not the earliest. Any dependency that comes to require 3.13+ brings it forward, and **torch is the likeliest candidate** — at which point Raven is wedged between torch and Kokoro with no room to move. A date invites forgetting; a trigger keeps the check cheap. Concretely: whenever a dependency bump is blocked by `requires-python`, that is this decision coming due, and the forced-alignment route is what to reach for.

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

- **PDF conference abstracts robustness item**: added as a reminder to check whether pdf2bib handles this case. Now working correctly. Conference info is now configurable via CLI options.

- **System prompt tuning for LLM speculation on/off**: was relevant during early Qwen3 work. Superseded by improved model behavior. Dropped.

- **RAG search data location in chat tree**: where to store RAG results in the chat tree format. Resolved — tracked in metadata. Dropped.

- **Privacy note for STT in Librarian docs**: has been added to documentation. Dropped.

- **"Switch chat from all leaf nodes" feature**: idea was that each leaf node constitutes a potentially interesting HEAD. Not a productive framing — too many leaf nodes for useful UX. Superseded by the recent chats list design, which uses a more principled definition of "distinct chat."

- **Installation instructions TL;DR**: now covered in main README.md. Separate section no longer needed.

- **Misc items: assign to closest cluster in 2D** (original Visualizer item): duplicate of the cosine-to-medoid outlier assignment in the importer rework. Dropped.

- **Calculator tool using `eval`**: `eval` is fundamentally unsafe in Python (e.g. `().__class__.__base__.__subclasses__()[-1].__init__.__globals__['__builtins__']['__import__']('os').system(...)`). See https://stackoverflow.com/questions/64618043. Use `simpleeval` instead — see active TODO item.

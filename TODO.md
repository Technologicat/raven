# Raven TODO

Covers the full Raven constellation: Visualizer, Librarian, Server, Avatar, XDot Viewer, and shared tooling.

Priority tiers: **[High]** | **[Medium]** | **[Low]** | **[Parked]**

Items marked **[Verify]** should be checked against the current codebase in a CC session before implementing.


---

## Autumn 2026: Researchers' Night (2026-09-26)

Librarian has to be demo-ready and impressive by then. The date is the only hard constraint in this file, so
individual items carry a "Researchers' Night" note where they feed it. Working backwards from 26 September:

1. **Demo correctness** — the defects a live audience would *see*. The temporary-context-inject package (six
   linked items in `TODO_DEFERRED.md`, listed under its cluster index), the Markdown renderer defects, the
   remaining crash/race items. Not polish: this is whether the demo works.
2. **Demo impressiveness** — `crt` and `atmospheric_dust` (both briefed), the avatar branch-switch glitch, RAG
   reranking for better answers on stage, citation surfacing, colorblind-safe signalling, lorebook if it fits.
### The actual list, as of 2026-07-28 (60 days out)

Phases 1 and 2 above are the framing; this is the concrete list they resolve to after today's work. Kept
here rather than reconstructed each time, because reconstructing it means re-deciding things that were
already decided.

**Phase 1 — correctness. A live audience would see every one of these.**

- **RAG access via tool-call** (see the Tools section). The largest item, the only one that is new
  construction rather than repair, and promoted to phase 1 on measured evidence: asked something the
  documents do not answer, Qwen3.6-27B writes a literal `<tool_call>` block instead of an answer roughly
  one turn in three (Q11 in `investigations/context-injects/context-inject-shape-measurements.md`). Carries two attached
  decisions — the no-match bypass moves to the end of the agent loop, and telling the model its retrieval
  was weak is gated on this landing.
- **Markdown renderer: ATX headings don't render; fenced code blocks don't render; indented bullets
  mis-render.** All three are content defects rather than cosmetic, and Qwen emits headings and code
  fences constantly. The indented-bullet one is *not* fixed — the bullets fix that landed was a different
  bug (stacking at origin in hidden containers), and the render path still has no dedent.
- **Chat view scroll position jumps back down during generation.** Fires on every turn regardless of
  content, unlike the Markdown cases, and removes the one useful thing to do while a thinking model works.
- **Librarian: in-flight AI turn bleeds into a new chat** (turn-sequencing race).
- **Thinking mode is a wall of text on a projector.** Details under Librarian / Chat UI; here is why it is
  phase 1 rather than polish. A thinking model spends most of the turn reasoning, and right now every word
  of it lands in the chat log at full size in front of the audience — the answer, which is the thing being
  demonstrated, arrives underneath a screenful of deliberation and scrolls past. What the demo needs is the
  collapsed-by-default mode plus the pulsating cloud, and the two are one item on stage: hiding the trace
  without an activity signal replaces a wall of text with an app that appears to have frozen for thirty
  seconds, which reads worse. The token/time readout is the optional third — nice for a technical audience
  ("it thought for 8 s and 1400 tokens"), not required.

**Phase 2 — impressiveness.** `crt` and `atmospheric_dust`, both fully briefed and speced. Note `imagefx`
measures 0.00 GiB today only because its filter chain is empty; these go into exactly that chain, so re-run
`investigations/vram/avatar_footprint.py` after they land.

**Explicitly *not* on the demo path**, each for a recorded reason:

- *Avatar stutter* — deprioritized; a warm-up handles it without knowing the cause.
- *Chat view drops a character* — one sighting, no recurrence; an open report, not a known defect, and
  nothing to test a fix against.
- *RAG reranking* — still quality work, not correctness. But the **query-side levers**
  (`briefs/summer_2026_librarian_extension/09`) are no longer in this category: lever 1 and the confidence
  signal are what the grounding marker needs to tell "matches arrived" from "matches were any good", and
  without them it stays silent against a real corpus. That makes them a prerequisite for a phase-1 feature
  that has already shipped, so they move onto the demo path. The alternative route, inline citations, does
  not go through brief 09 at all and should be weighed against it rather than assumed to lose.
- *Brief 03 section D* — wanted, to close out the last unfinished brief; not demo-visible.

**Both "measure early" items are closed** (per-module VRAM, MoE vs dense), so nothing further blocks the
hardware plan. The remaining schedule risk is phase 3 getting eaten, which is what phase 3 always does.

3. **Freeze and rehearse** — no new features; run the demo repeatedly on the actual hardware and fix what that
   surfaces. This window is the one that gets eaten, and it is where the real surprises live. It is also when the
   Librarian marathon restarts, since briefs 04 (MCP) and 06 (Hindsight) are off the deadline path by their own
   framing — deferred by date, not by worth. They remain the main line for the "digital colleague" track.

**Measurements to take early**, because several decisions are blocked on numbers rather than on opinions:

- **Avatar stutter** — **deprioritized 2026-07-28, not a demo blocker.** It shows mainly at the start of a
  session, which a warm-up handles without knowing the cause, and the slight stutter on the first TTS'd
  sentence is within what games routinely ship. Investigate if the other items land with time to spare.

  Kept because the hypotheses are still worth the eventual look: is it warmup, GPU contention, or genuinely
  the GIL? The voice is already warmed at startup, so "first speech" warmup is a weaker hypothesis than it
  sounds; what is still cold at that moment is THA3's first inference at the talking-morph shape, the audio
  device open, and the postprocessor's first pass. Test whether it survives TTS-on-CPU, which would rule
  contention out.

  **Two notes for whoever picks this up, both learned by getting it wrong first:**

  - *The moment matters, and it is not startup.* The stutter is on an avatar that has already been
    streaming frames steadily for a long time — Librarian starts the session well before anything is
    spoken. So any hypothesis about session-start costs (including the ~300 MiB the VRAM probe sees
    arriving during the first frames) is aimed at the wrong moment and can be dropped.
  - *VRAM is the wrong instrument.* A stutter is dropped or late frames; memory usage cannot show it.
    The measurement wanted is **frame inter-arrival timing** across the speech transition — with the
    caveat that frames reach a client over HTTP, so client-observed timing carries scheduling noise
    that server-side instrumentation would not.

  And the entry point, which is easy to get wrong: `avatar_start_talking` is the *randomized-mouth idle*
  animation, not lipsync. Real speech goes through `raven.client.tts.tts_speak_lipsynced`; for how an
  application drives it, see `raven.client.avatar_controller.speak_task`.
- ~~**Per-module VRAM budget**~~ — **measured 2026-07-28 on the 16 GB machine** (`raven-server --vram-report PATH`, instrumentation lives in `deviceinfo.VRAMLedger`). All nine modules resident cost **2.27 GiB** at load, leaving 13.0 GiB of 15.6 free:

  | module | GiB | | module | GiB |
  |---|---|---|---|---|
  | embeddings | 0.83 | | tts | 0.34 |
  | sanitize | 0.63 | | translate | 0.25 |
  | classify | 0.14 | | stt | 0.04 |
  | avatar | 0.03 | | natlang | 0.02 |
  | imagefx | 0.00 | | | |

  So the module set is not the constraint on either card; the LLM is. On 8 GB that leaves ~5.5 GiB before inference peaks, which fits the on-the-road model with room.

  **This is a floor, not the operating footprint**, and two rows say so. `avatar` genuinely loads THA3 eagerly (0.03 is the posing engine), but per-session render buffers and the Anime4K upscaler are allocated when a session starts, not at init. `imagefx` reads 0.00 because it is constructed with an empty filter chain — and `crt` plus `atmospheric_dust` are about to go into that chain. Inference activations are absent throughout.

  **The avatar's running footprint is now measured too** (`investigations/vram/avatar_footprint.py`, which drives a real session over the client API and samples `nvidia-smi` from outside, so it adds no CUDA context of its own): **+386 MiB peak while animating**, against 30 MiB at load — 13x, and the reason load-time figures cannot be trusted for this module. Session creation alone is +80 MiB; the rest arrives once frames flow. Hit 25.4 FPS over 100 frames, i.e. the server's target rate.

  So the whole server side, all nine modules with the avatar running, is **~2.9 GiB**. On 8 GB that leaves ~5 GiB for the LLM, which fits the on-the-road model with its KV cache. The module set is comfortably not the constraint.

  Memory stays resident after unloading a session, which is expected — PyTorch's caching allocator keeps freed blocks reserved rather than returning them to the driver. `nvidia-smi` cannot distinguish that from a leak; doing so needs the server's own allocator stats.

  Still unmeasured: peak during use for the other eight modules (needs a warm-up request each; only worth doing if the LLM budget turns out tight), and the postprocessor filter chain, which is where `crt` and `atmospheric_dust` will live and which currently measures 0.00 because it is empty.

  `embeddings` goes stale at the Nomic switch; the other eight should hold until the demo.
- ~~**MoE vs dense decode speed**~~ — **measured 2026-07-28 on the eGPU: ~110 tok/s for 35B-A3B against ~40
  tok/s for 27B dense, so 2.75×.** That is *below* the 3–9× band predicted from the active-parameter ratio
  (3B vs 27B), which says the fixed per-token overhead compresses it harder than expected — worth remembering
  the next time an active-parameter ratio is used to predict a speedup. Decision: **the MoE is the demo
  model**; 40 tok/s is usable but sluggish for a thinking model in front of an audience, and the 27B is the
  fallback for a problem the MoE can't solve rather than the default. Model lineup in
  `briefs/reference/model-lineup-autumn-2026.md`.

**Demo hardware shape**: LLM alone on the larger card, all nine raven-server modules on the internal one — the
`config_dual_midvram` variant. Reducing context to buy headroom is a weak lever and not worth the cheat: Qwen 3.6
is gated-deltanet at 3:1, so only about a quarter of the layers carry a KV cache that grows with context.

The **eGPU travels to the demo**, so it runs on the top tier rather than on a laptop-class model. Which model
that is, and which one each other machine gets, is settled in `briefs/reference/model-lineup-autumn-2026.md` — Qwen at
every tier, chosen on measurements rather than reputation.


---

## Cross-cutting

- **[High]** HF hub: document the env vars that prevent hub checks on Raven startup (for privacy and faster startup). Add recommendation to server docs. Currently not written down anywhere in the project.
  - `HF_HUB_OFFLINE=1` — forces huggingface_hub to use only locally cached models, no network requests at all.
  - `HF_HUB_DISABLE_TELEMETRY=1` — stops telemetry pings only.

- **[High]** Neural reranker for HybridIR: add a reranker stage to avoid needing large k values (k=100 style workarounds). Since we maintain our own HybridIR backend, we can power it up properly. Design worked out in `TODO_DEFERRED.md`, "RAG: rerank retrieved chunks and inject only the best few" (cross-encoder stage, three-layer `common`/`server`/`mayberemote` shape, VRAM tradeoff).

- **[High]** Revisit logging system: library modules should not reconfigure the logger (verify exact behavior against Python `logging` stdlib docs, but currently each module sets the log level, which is the entrypoint's responsibility). Move logging configuration to entrypoints only. Add a "detailed debug" level at that time for particularly spammy-but-useful log lines (e.g. `SmoothScrolling.render_frame`, `_managed_task`, `binary_search_item`).

- **[Medium]** Flash the search field when focused by hotkey. Currently affects Visualizer main window, fdialog component, and XDot Viewer. **The enabler is done** (2026-07-30): `ButtonFlash` is now `WidgetFlash` and animates any widget — a text widget fades its own text color, anything else fades a theme background — with `animation.highlight_widget` as the convenience entry point, alongside `flash_button`. What remains is applying it at the three search fields, which is the actual item.

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

- **[Done — verified 2026-07-29]** BibTeX-encoded umlauts and verbatim braces. Both halves were already implemented in `common.utils.unicodize_basic_markup`; the item was stale, which is what its own `[Verify]` marker existed to catch. Confirmed against real input: `\"{o}` and unbraced `\"a` give ö/ä, `\'{e}`/`\H{o}`/`\c{c}` give é/ő/ç, the ligatures (`{\o}`, `\ss{}`, `{\ae}`) resolve, `{GPU}` loses its grouping braces, and literal `\{`/`\}` survive brace stripping via private-use sentinels. Also confirms the display path reverses everything `raven-wos2bib` emits, so teaching that converter to escape every field broke nothing. Note for whoever touches this next: `papers.utils.bibtex_escape` / `bibtex_unescape` and `unicodize_basic_markup` are three implementations of two directions of one transformation — worth checking whether they should be one.

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

- **[Low]** fdialog improvements. Five further FileDialog items live in `TODO_DEFERRED.md` (slow open + teardown input-dead-window on huge directories, smart-case Find, image thumbnail previews, multi-extension filter as one labelled item, reduce per-use-site boilerplate); treat the whole set as one work package rather than picking at it from two lists.
  - Add "go up to parent directory" button
  - Change the "go to default directory" icon to something less confusing
  - In save mode: if the user has picked a unique file extension in the filter combo, use that as the default extension. If multiple extensions or wildcards, use the API-provided default.
  - Ctrl+F hotkey to focus the file name field is not always working. **[Verify]** exact conditions before fixing.

- **[Low]** Drag'n'drop from OS file manager into the Raven window to open a dataset. DPG 2.0.0: not implemented for Linux; Windows add-on exists. Need a cross-platform solution — keep an eye on DPG upstream. Fleet-wide framing (it blocks the Librarian attach path too, making the in-app picker the sole entry route) in `TODO_DEFERRED.md`, "OS drag-and-drop of files into DPG apps (cross-platform)".

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
  - **Document scopes are a prerequisite**, not a parallel feature. Unifying the databases means one Librarian corpus holding items from several Visualizer datasets at once, and those have to stay separable — so every item imported from a dataset gets tagged with a scope identifying it, keyed by the dataset's file path. Without that the unified database is a bag with no way back to "the papers in *this* map". See the document-scopes item under Librarian.
  - Related shape problem in the meantime: a `.bib` dropped into `docs_dir` is *one* document however many records it holds, so a whole reference database imports as a single blob — retrievable chunk-wise, useless to fetch. `raven-burstbib` is the current answer (burst it into one file per record first), and unification is the eventual one.


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

- **[High]** Give the tool-call round budget room for a genuine multi-document read. Measured live 2026-07-29 (`briefs/summer_2026_librarian_extension/manual_tests/rag_live_corpus.py`, phase F): asked a follow-up about which documents supported an earlier answer, the model works through `list_consulted_documents`' output one `fetch_document` per round, exhausts `max_tool_call_rounds = 5` on gathering alone, and has no round left to answer. Nine of fourteen sampled turns that reached the cap ended with an **empty assistant message**, against one of ten that did not (24 paired samples, Fisher exact p = 0.013; raw data in `investigations/tool_budget/`). Brief 10's own two features collide here: the provenance list invites reading several documents, the cap budgets five rounds in total.
  - **A mitigation was tried and did not measurably work.** The invocation after the cap carries `chatutil.format_notice_that_tools_are_spent`, on the reasoning that the model was never *told* the gathering was over — it found out by reaching for a tool that was gone. Measured across 12 paired samples per arm, it moved nothing: 8/12 answered with it, 6/12 without, p = 0.68, and the sign flips once restricted to cap-reaching turns. Kept because it is one line and addresses an observed mechanism, but **it is not the fix and must not be mistaken for one**. The cap itself is what correlates with the empty reply, which is what the budget change below has to address.
  - **Landed 2026-08-04, and it is not this item:** past the cap the tools now stay in the schema and a call is refused with an error result (`chatutil.format_error_that_tools_are_spent`), with withdrawal kept as the terminator of last resort after `max_tool_call_refusal_rounds`. That was argued on cache-burn and distribution-fit grounds, not on the empty replies — the model still runs out of budget at the same round, so the measurement above stands and the item below is still the fix. Worth re-running the probe once the budget actually changes: the arms and the resume ledger are in place, so it is a re-run rather than a rebuild.
  - **The fix to make is a larger budget, not a list that discourages reading.** Deciding *not* to make retrieval timid: a user faced with a model reading through the phone book has Ctrl+G and a rephrased request, which is a better remedy than a system that refuses to read thoroughly when thoroughness is what was asked for.
  - **Partly overtaken by events, 2026-08-04: the cap was raised from 5 to 20** on the strength of `investigations/tool_refusal/`, which measured where this model actually stops. That may be enough on its own; re-run the phase F probe before building anything further here.
  - **Two budgets, not one, and both per *turn*.** A single larger `max_tool_call_rounds` loosens the wrong thing too: a `fetch_document` is bounded work against a document already known to exist, while a `search_documents` can be rephrased forever against a corpus that has nothing — which is the failure the cap was added for in the first place (`manual_tests/rag_tool_rescue.py`). So searches keep a small cap and fetches get their own, larger allowance, naturally sized by `docs_num_results` since a search cannot surface more documents than that.
    - **The "forever" did not survive measurement.** Against a corpus containing literally nothing, qwen3.6-35b-a3b rephrased nine or ten times and then gave up unprompted, in 3 of 3 samples with the cap out of reach (`investigations/tool_refusal/`). That is the strongest form of the case this design was built to handle, and the model self-terminated. One model and small n — evidence, not a refutation — but the two-budget split needs a better argument than this one before it is worth its complexity.
    - **The allowance has to be per turn, not per search.** Per search it does not bound anything: `search(X) → fetch(X, 0..9) → search(Y) → fetch(Y, 0..9) → …`, where Y is a keyword the model picked up while reading X's results. Each search would refill the pool, and the recursion is a perfectly reasonable research strategy — which is exactly why it needs a ceiling rather than an argument.
    - Note what that loop *is*: search, read, harvest a term, search again is the shape sold elsewhere as agentic "deep research". So the ceiling is a resource decision — context window, latency, the user's patience — and not a correctness one. Pick the number by what a turn can afford to spend, not by what looks like runaway behaviour, because the runaway and the good version are the same algorithm.

- **[High]** Thinking toggle: let the user turn a thinking model's reasoning on and off. Researched 2026-07-27 — brief 02 closed the *transport* half of this item but none of the request-side half, so the remaining work is narrower and differently shaped than originally written:
  - **Done (brief 02).** Reasoning rides out-of-band: `llmclient.invoke` emits a typed `{"type": "reasoning", ...}` event, storage uses the `reasoning_content` sibling field, `chatutil.upgrade_datastore` migrates old inline `<think>` at load, and minichat / `chat_controller` consume the typed channel instead of sniffing tags. Live traffic no longer depends on parsing `<think>`.
  - **Open, and the actual work.** Nothing anywhere sends `enable_thinking` / `chat_template_kwargs` / `reasoning_effort` — grep across `librarian/` and `client/` returns zero hits. So a model that *supports* toggling can't be told to stop thinking. Needs a GUI toggle and an app-state key (current keys: tools / docs / speculate / speech / subtitles — no thinking entry).
  - **Probed against LM Studio 2026-07-27; the obvious mechanism does not work and the rejected one does.** Measured on `qwen3.5-9b` and `qwen3.6-35b-a3b`:
    - **`chat_template_kwargs` is silently ignored.** Reasoning length was unchanged with the field absent, `false`, and `true` (9B: 823 / 731 / 817 chars; 35B: 614 / 748 / 1281 — the spread is sampling noise, not signal). LM Studio also returns HTTP 200 for a deliberately nonsensical parameter name, so **acceptance never implies support there** — only behaviour does. Matches the open request [lmstudio-bug-tracker#1559](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1559).
    - **The template's own default is thinking OFF.** From the GGUF (`gguf-dump --no-tensors --json`): `enable_thinking is defined and is true` emits an *open* `<think>\n`; every other case emits `<think>\n\n</think>\n\n`. Both models nevertheless think by default through LM Studio, so LM Studio is supplying `enable_thinking=true` itself — the per-model config / UI toggle, not the quant.
    - **Prefill works, and is the mechanism.** Prefilling the assistant turn with `<think>\n\n</think>\n\n` drops reasoning to zero while the answer stays correct (9B 747→0, 35B 502→0). For Qwen that string is **byte-identical to the template's own non-thinking branch** — with no generation prompt to add, the rendered prompt is the same either way — so it reproduces non-thinking mode exactly rather than approximating it.
    - **But the *cause* is prefilling at all, not the prefill's content** (established 2026-07-27 on both families, correcting the earlier claim). A bare `"The"` suppresses reasoning just as completely, and each family's marker suppresses on the other. What actually happens is that a trailing assistant message means no generation prompt, and the generation prompt is where the template puts its thinking prefix.
    - **Use the template-correct string anyway — the wrong one breaks generation.** Content doesn't control *suppression*, but it does control *output quality*, and this is not subtle. Prefilling Gemma's `<|channel>thought\n<channel|>` into `qwen3.5-9b` zeroes the reasoning and then makes the model **echo the question back** instead of answering it; Qwen's own marker answers cleanly. (Asymmetric: Qwen's marker on Gemma was harmless — presumably Gemma reads it as ordinary text, while Gemma's `<|…|>` forms collide with something in Qwen's vocabulary.) So the per-family string is a correctness requirement, not tidiness, and a wrong or missing entry must fail loudly rather than fall back to a default marker.
    - **Consequence for `continue_`, which nobody has looked at:** continuing a reply prefills it, so **the continued turn does not think.** For resuming a truncated answer that is probably wanted. But it is a plausible mechanism for the existing "incomplete thought block after Continue" bug listed under Chat UI — if generation was cut off *mid-thinking*, continuing cannot re-enter the thought channel, so the block stays unclosed. Not proven, but it fits the symptom exactly, and the two should be investigated together.
  - **Design revision.** The toggle is still a per-call parameter on `invoke` (see below), but on LM Studio it controls a *prefill*, not a request field. Keep the flavor-aware mapping: prefill on LM Studio, presumably `chat_template_kwargs` on ooba — **ooba is untested**, and is not installed on the 16 GB machine at all, so that half needs the ooba upgrade first.
  - **Where it goes: a parameter on `invoke`, not a mutation of `settings.request_data`.** `llmclient.setup` assembles `request_data` (~line 465) from `librarian_config.llm_sampler_config`, and `invoke` deep-copies it per call (~line 1094), so the per-call copy already exists and is the natural place to apply the flag. It must *not* be done by mutating the shared settings dict, because `perform_throwaway_task` takes the same `llm_settings` and routes through the same `invoke` — a keyword-extraction call shouldn't spend reasoning tokens just because the user enabled thinking in chat. Settings holds the default; the parameter overrides per call; the user's toggle lives in app state and threads down through scaffold, the same shape tools / docs / speculate already use.
  - **One flavor-aware mapping.** The *mechanism* differs per backend, not merely the field name, and `llmclient` already has `detect_backend_flavor` — so translate "thinking on/off" into either a prefill or a request field once, where the payload is finalized, rather than at each call site.
  - **Gemma 4 (26B-A4B) probed 2026-07-27; same backend answers, opposite history behaviour.** `chat_template_kwargs` ignored, `min_p` honoured, prefill works, Anthropic-endpoint thinking toggle works, streaming works — all as for Qwen. Two differences that matter:
    - **Its template retains prior reasoning; Qwen's discards it.** Same measurement, opposite result: prompt size went 44 → 653 tokens when a ~700-token `reasoning_content` was present in history (Qwen stayed at 38). No `preserve_thinking` involved — the Gemma template has no such key and simply keeps it. Since `_serialize_history_for_wire` sends `reasoning_content` untouched, **this is live, not hypothetical**: on Gemma every stored reasoning trace is re-sent every turn.
      - **Checked 2026-07-27: the meter does not count reasoning at all.** `update_context_fill_indicator` builds its estimate from `chatutil.content_to_text(message.get("content"))`, and reasoning lives in the `reasoning_content` *sibling* field, so it never enters the sum. That is *correct* on Qwen, whose template discards prior reasoning anyway — and a large under-report on Gemma. Mitigated but not fixed by the existing two-stage design: the debounced background prefill replaces the estimate with the backend's exact `prompt_tokens`, so the readout self-corrects once the chat settles; it is the immediate estimate that is wrong, and by more than the "slightly under-reports" its docstring claims.
      - **The fix is model-dependent, which is the interesting part.** Counting reasoning unconditionally would over-report on Qwen as badly as omitting it under-reports on Gemma. So the meter needs to know whether the backend's template retains reasoning history — another capability flag in the `model_is_vlm` family, determined per model rather than guessed.
    - **Thinking can't be forced *on* by prefill.** Gemma's template puts `<|think|>` at the top of the *first system turn*, not at the assistant turn, so the Qwen trick has no mirror image here. Its thinking-off marker is `<|channel>thought\n<channel|>`, emitted at the generation prompt.
  - **Environment for all of the above:** LM Studio **0.4.19 (Build 2)**; unsloth GGUFs for the Qwens, lmstudio-community for Gemma 4. The Gemma quant is not interchangeable — the unsloth build fails to load, and LM Studio's workaround for Gemma's template fires only for the lmstudio-community build with its bundled template unoverridden. Re-record the version when re-probing; these are behaviours of a build, not of a protocol.
  - **Qwen 3.6 wants its thinking history kept, and we currently can't grant that.** The two templates differ on reasoning retention (read from the GGUFs, 2026-07-27):
    - **3.5** renders prior reasoning only when `loop.index0 > ns.last_query_index` — older thinking is always stripped, no opt-out.
    - **3.6** adds a `preserve_thinking` kwarg: `(preserve_thinking is defined and preserve_thinking is true) or (loop.index0 > ns.last_query_index)`. Set it, and *all* prior reasoning is retained. Default is still strip, so this is opt-in.

    `preserve_thinking` is a `chat_template_kwargs` key, which LM Studio ignores — and prefill can't substitute, since this governs how *history* renders rather than how the next turn starts. **Probed exhaustively 2026-07-27; it is unreachable by every route tried.** Measured by prompt size, since a retained ~700-token reasoning blob would be unmissable:

    | history shape / route | prompt tokens |
    |---|---|
    | baseline, no reasoning in history | 38 |
    | `reasoning_content` sibling field | 38 |
    | + `chat_template_kwargs.preserve_thinking = true` | 38 |
    | Anthropic-native `thinking` content block, `/v1/messages` | 40 |

    So prior reasoning is discarded whatever we send and however we ask — the Anthropic endpoint's working *toggle* does not extend to feeding thinking history back. The full-thinking-history mode 3.6 was designed for is simply not available through LM Studio's HTTP APIs. **Remaining open question:** whether LM Studio's per-model config can set it, the way it evidently sets `enable_thinking`; that's the only untried surface.

    Note the cost if it ever becomes reachable — retaining every turn's reasoning enlarges the prompt substantially, which interacts with the context-budgeting and KV-cache items **and with the context-fill meter**: `update_context_fill_indicator` would be under-reporting by the whole reasoning history, so the meter has to learn about this mode at the same time.
  - **Still open, separately:** `chatutil.scrub` retains the `<think>`-repair machinery for models that emit malformed or missing tags — QwQ-32B is the documented case, but recent Qwens reportedly do it too, so this isn't a single-model quirk. The standing question at `chatutil.py:937` asks whether to inject the opening `<think>` into the prompt and how to do that through the API. That framing may be obsolete: on current models the tag is believed to come from the chat template automatically once the right API options are set, making it the *same* knob as the toggle above rather than a separate injection problem. **Research before implementing either** — check Qwen 3.5, Qwen 3.6 and Gemma 4 specifically, since those are what Raven currently supports, and the answer decides whether this is one feature or two.
    - Prefill *is* the escape hatch, contrary to what this item said before the 2026-07-27 probe: injecting the tag as a pre-written assistant turn works fine on LM Studio. Forcing thinking *on* is the mirror image of forcing it off — prefill an open `<think>\n` instead of a closed empty block.
    - **Don't delete the autofixer yet.** It's moot on LM Studio, which delivers reasoning on its own channel, so the repair path never fires there. Ooba is the backend that would exercise it, and the local install is far behind — upgrade it and re-test (see "Upgrade oobabooga and re-check Raven's ooba support" in `TODO_DEFERRED.md`) before concluding the machinery is dead code.

- **[High]** Enable `continue_` on LM Studio. It is currently a no-op there and only works against ooba, on the belief that LM Studio has no prefill. **That belief is wrong** — probed 2026-07-27: a trailing assistant message on the OpenAI-compat endpoint is genuinely continued, not restarted (`"The four seasons are spring, summer,"` → `"autumn, and winter."`), on both `qwen3.5-9b` and `qwen3.6-35b-a3b`. So this is a latent capability Raven isn't using rather than a missing backend feature, and the GUI's continue-after-truncated-reply action can be enabled on the default backend. Shares its mechanism with the thinking toggle above, so land them together.

- **[High]** Anthropic-compatible backend support. Started as a breadth-of-options item; the 2026-07-27 probe promoted it, because **LM Studio's Anthropic endpoint exposes a working per-request thinking toggle that its OpenAI endpoint does not.** Verified against `qwen3.6-35b-a3b`:
  - `thinking: {"type": "disabled"}` → content blocks `['text']`; `thinking: {"type": "enabled", "budget_tokens": N}` → `['thinking', 'text']`. A genuine toggle, in Anthropic's own spelling, with no prefill needed.
  - The endpoint **defaults to thinking off**, the opposite of the OpenAI endpoint's default-on. Worth knowing before comparing behaviour across the two.
  - It **streams** — proper SSE, `event: message_start` and Anthropic-shaped events — so `llmclient`'s stream parser has something to attach to.
  - Assistant prefill works there too.
  - Response carries `stop_reason` and `usage.cache_read_input_tokens`, i.e. the real Anthropic shape rather than a thin alias.

  So this is now two features in one: the clean thinking toggle, *and* letting Raven talk to Anthropic's own API as a backend. The latter still matters on its own — Raven targets scientific research, and a meaningful slice of that community already works through the Anthropic API. Same reasoning as supporting both NVIDIA and AMD: breadth, not preference.

- **[Low]** Note for sampler config: **LM Studio honours `min_p` even though its documented parameter list omits it.** Verified behaviourally 2026-07-27 — at temperature 2.0 the unclamped output varies between seeds, while `min_p=0.9` is seed-invariant, as is the documented `top_k=1` control. Recorded because the docs list (model, messages, temperature, top_p, top_k, max_tokens, stream, stop, presence_penalty, frequency_penalty, logit_bias, repeat_penalty, seed) reads as exhaustive and isn't; don't drop a sampler setting on the strength of it. Corollary: LM Studio returns HTTP 200 for unknown parameters, so any future "is this supported?" question needs a behavioural test, not a status code.

- **[Med]** RAG PDF ingestion — polish. The core is done: born-digital PDF text is extracted via `raven.common.docextract` (pypdf) and indexed like any other document. Remaining: run the extracted text through `sanitize` before indexing (PDF text often has hyphenation artifacts and paragraph-break ambiguity); link a search result back to its original document (see `TODO_DEFERRED.md`, "Expose the docs-DB source files behind a reply's RAG citations"); generalize to scanned PDFs (OCR) and to images (caption generation — ties into the Nomic multimodal-search plan).

- **[High]** Adjustable semantic search match strictness: configurable cosine similarity threshold in HybridIR below which results are dropped. High priority.

- **[Medium]** Attach a document that is *already in the docs DB*. Full-document attach itself works (images and text/PDF, brief 03 Half 2) — but only from the filesystem. There is no way to reach into the RAG store and attach one of its documents whole, which is what you want when retrieved chunks aren't enough and the file is already ingested.
  - **Open question: whose affordance is this — the user's, the AI's, or both?** The AI side already has an entry under Tools ("RAG access via tool-call: … fetch a full document by ID"), so if that lands, the model can pull a whole document itself. The user side (pick from the DB in the attach dialog) is the genuinely missing half. Deciding this shapes both: a shared "resolve doc ID → `text_file` content part" path serves both callers, and the GUI picker needs the docs DB to be browsable, which the tool version doesn't.

- **[High]** Citation tracker GUI: validate that LLM-inlined citations (in whatever format we specify) actually point to documents in the RAG result set; flag any that don't. The other half — surfacing *which* documents fed a reply, and opening the originals — is specced in `TODO_DEFERRED.md`, "Expose the docs-DB source files behind a reply's RAG citations"; the provenance data is already tracked per turn (the payload's `retrieval` field), just not shown.

- **[Low]** `minichat`: remove deprecation note (it will be maintained with Claude Code). Minimal example client, usable over a bare SSH terminal.


### Core features

- **[Medium]** A CLI indexer for the document database: build or refresh the RAG index over a documents directory without starting a GUI. Today the only way to index a corpus is to launch `raven-librarian` and wait, which couples a batch job to a desktop session and to the frontend being in a runnable state — noticed 2026-08-05 while swapping corpora for a retrieval experiment, where an unrelated GUI-side change would have blocked the indexing run. It also makes a corpus swap a manual ritual rather than a command, which is the shape of thing that quietly discourages measuring against a second corpus.
  - Wants roughly the arguments `hybridir.setup` already takes — documents directory, index directory, extractor, embedding model — plus progress on stdout and a non-zero exit on failure. The commit machinery already reports progress and is cancellable per document, so this is a front end over existing parts.
  - Sibling of the scriptable-scaffold item below, and the same argument: these are library layers with exactly one entry point each, and that entry point is a desktop app.

- **[Medium]** Make the scaffold scriptable: a script should be able to do everything the chat clients can, without standing up a GUI or a REPL. Today `perform_throwaway_task` is the only scripting entry point and it is deliberately thin — no datastore, so no attachments, no branch, no retrieval. The two symptoms that keep surfacing are the same gap seen from different sides: an attached document cannot be folded into a throwaway task (`llmclient._serialize_history_for_wire` needs the datastore to resolve the sidecar), and every batch experiment ends up re-implementing a slice of `scaffold.ai_turn` by hand. Supporting attachments in throwaway tasks is the narrow fix; the better one is a scripting-facing surface over the scaffold, which would also cover the manual-test probes under `briefs/*/manual_tests/`.
  - **It is a programming library, not a product.** Visualizer and Librarian are what ships to researchers; the engines under them are useful for things those products do not do, and that is what the surface is for. Which is the diagnosis for the `_perform_injects` problem above rather than a separate observation: it is *already* a library API and has simply never been told so. Being one means `__all__`, docstrings that stand as the documentation, and some statement about what may move — mostly not code.
    - **MCP is not a user-plugin system, and the distinction is worth keeping in writing.** MCP is tool *supply*: an external process offering capabilities inward, which is exactly what an agentic per-document pass wants, and it is wanted (see `04_librarian-mcp-client-brief.md`). User plugins would be app *extension*: third-party code inside the process, with the trust and versioning surface that implies. Not wanted. Conflating the two is how a library grows an extension system by accident.
  - **Worth a brief, and `manual_tests/rag_live_corpus.py` is the specification by example.** Written 2026-07-29 to measure one behaviour, it ended up hand-rolling most of what such a surface would provide, which is the useful evidence: this is not speculative API design, it is code that already exists in the wrong place.
    - **A turn should return what happened, not a node id.** Every probe re-implements the same branch walk afterwards — count the tool nodes by name, count the rounds (an assistant message asking for tools, however many it asks for), collect the reasoning that never reached `content`, find the reply. That walk is the actual result of a turn and it is written out longhand each time, differently, with the round-vs-call distinction got wrong at least once.
    - **The callback wall is the friction.** `ai_turn` takes eleven callbacks that a script passes as `None`, and forgetting one is a `TypeError` — which is how a probe silently rotted when brief 10 removed `on_nomatch_done`. Defaults would have made that a non-event.
    - **Capturing the prompt needs a closure.** The wire history is only reachable through `on_prompt_ready`, so any script that wants to assert on what was actually sent has to build a callback to catch it.
    - **A/B-ing a knob means monkeypatching.** The probe swaps out a `chatutil` formatter to run its control arm. Fine in a probe, and a sign that per-run overrides belong in the scripting surface rather than in module globals.
    - Persistence and resume it now does properly (a `PersistentForest` per sample plus a JSONL ledger), and that part is worth lifting wholesale: these runs take an hour and the machines reboot.
  - **Two entry points, not one.** Surveying the twelve probes under `briefs/summer_2026_librarian_extension/manual_tests/` (2026-07-29) splits them three ways, and only two of the three want anything:
    - **Wire-level probes** — `backend_capabilities`, `gemma4_reasoning_roundtrip`, `vision_check`, `webfetch_*`, `datetime_inject`. These post raw to `/v1/chat/completions` deliberately, because they measure the *backend* and routing through `llmclient` would confound the result. They must keep bypassing Raven; a scripting layer is not for them.
    - **Prompt-shape probes** — `assembled_shape`, `absent_fact`, `rag_placement`, `inject_shapes`. These want the prompt Raven *would* send, and then to send it themselves. All four reach into `scaffold._perform_injects` — a private function, called from four separate files, which makes it a public API that has not been declared one. It gained a parameter this session (`tools_are_spent`) and they survived only because it defaults. So entry point one is **"build the turn's prompt and hand it back"**, no backend involved.
    - **Full-turn probes** — `rag_live_corpus`, `rag_tool_rescue`. Entry point two is **"run the turn and tell me what happened"**, per the shape described above.
    - The two are not layers of each other: the first must not talk to a backend at all, and the second is useless without one. A single "scriptable scaffold" API that only offers the second would leave four probes still reaching through the private door.
  - **The callback wall is not fixed by giving the callbacks defaults.** `ai_turn`'s mandatory callbacks are deliberate, on the fail-fast principle, and they earn that for a GUI client — one that forgets `on_llm_progress` is genuinely broken and should say so loudly. They earn nothing from a script that will never draw a progress bar. So the scripting surface has **no callbacks at all**: the events become the returned record of what happened, which is the thing every probe currently reconstructs by walking the branch afterwards. Keep the wall where it protects someone; do not propagate it to callers it cannot protect.
  - **The second concrete user is `raven-pdf2bib`, and it is not hypothetical — it already shipped.** Extracting 2500 free-form conference PDFs into a BibTeX database (ECCOMAS 2024) is what the throwaway-task mode was built for, and the file shows what that mode does not provide: **eight** `perform_throwaway_task` call sites, each wrapped in its own hand-written `for retry in range(n_retries)` loop with an emptiness check and a warning — the same six lines, eight times, in one 1058-line file. No caching and no resume, so a crash at document 2400 restarts from zero.
    - That is the same primitive `manual_tests/rag_live_corpus.py` hand-rolled tonight (persist per sample, resume from the ledger) and the same one `briefs/design/corpus-interrogation-sketch.md` needs for its map stage. **A per-document LLM pass that retries, caches, resumes and reports progress** — three users, which is the bar for building it rather than the first-user-guesses-at-an-API problem.
    - Which is also the answer to "no point building yet another generic agent harness". This is not one and must not become one: no plugin system, no workflow DSL, no orchestration layer. What is not commodity here is what a generic harness cannot bring — a local corpus with its index, the branching chattree, the provenance machinery. The surface is programmatic access to *those*, and the agentic part is just that a per-document pass may call tools rather than being one shot.

- **[Medium]** Think blocks: parse properly instead of current regex hack. We already receive one token at a time.

- **[Medium]** Proactive context engineering: move beyond reactive BM25+semantic retrieval toward intelligent context curation. The system should maintain a graph of topical connections and proactively include relevant documents the user didn't explicitly ask for. E.g. "You asked about hydrogen embrittlement — here are the materials science papers you looked at last month." Shallow version (agentic chain-of-thought retrieval over a topic graph) is achievable now; deeper version requires a world model.

- **[Medium]** Document scopes: subdirectory-based filtering; scope selection GUI (checkbox per scope, select/unselect all); tags as the primary scoping mechanism (auto-tag by subdirectory name on ingestion); avoid cross-contamination between work/hobby contexts. Needed for long-term memory too. Currently must manually switch directories for each demo.

- **[Medium]** HybridIR: give documents a **title** field. Today a document has only `document_id` (the path relative to `docs_dir`) and its text, so there is nothing to show a user, nothing to hand a model deciding whether a document is worth fetching, and nothing to weight in retrieval. Titles are usually already present in the data and merely unparsed, and the reading of them is now written: `chatutil.document_label` extracts a BibTeX record's `title`/`author`/`year`, or falls back to the first substantial line. What is missing is *storing* the result as a field, which is what search can weight. Two wins, and the second is the larger: a legible label wherever a document is named (the `list_consulted_documents` inject in `briefs/summer_2026_librarian_extension/done/10_rag-tool-surface-brief.md`, a future citation UI), and a field that can be **weighted** in search — a title match is a much stronger relevance signal than a body match, which is index-side work adjacent to brief 09's query-side levers.
  - **Cost is a reindex**, ~1.5 h for the hydrogen dataset. Open question whether to migrate the existing index instead of rebuilding it: cheaper for the user, more code to maintain, and unlike the chat datastore a search index holds no irreplaceable hand-entered content — it is derived data, so nuke-and-rebuild is defensible in a way it would not be for `chattree`.

- **[Medium]** BM25 migration from `bm25s` to ChromaDB FTS5: gains incremental updates and metadata filtering (needed for scopes); removes full index rebuild at each commit; simplifies `hybridir.py` and removes a dependency. Mitigate tokenization quality loss by storing spaCy-lemmatized text in a dedicated ChromaDB field for FTS5 search. **Low priority** — `bm25s` works, and Raven's dependency policy is already generous.

- **[High]** Context compaction: drop and/or summarize old messages when context window fills. Use `raven.llmclient.token_count` to bisect linearized history to find the cut point (accounting for max response length from `settings.request_data["max_tokens"]`). Budgeting details in `TODO_DEFERRED.md`, "Context-window budgeting and conversation compaction (Librarian)".
  - **Raised from Medium 2026-07-29.** "Start a new chat before running out" stops being an answer once a turn can carry several fulltext attachments. Three papers compared against each other (`briefs/design/corpus-interrogation-sketch.md`, mode 3) is ~30k tokens before the discussion begins, and the discussion is the point.
  - **An attachment must not roll out entirely.** It is the reason the conversation exists. So compaction needs a priority order — pinned material, then recent turns, then older turns — rather than a single cut point found by bisection. Note that `llmclient.fit_attachments_to_context` is already a partial answer: it shrinks attachments as the conversation grows, max-min fair between them. What it lacks is the temporal dimension (an attachment nobody has mentioned in thirty turns is not equal to the one under discussion) and any notion of pinning.
  - **Summarizing on the main LLM is a two-way KV cache miss.** Sending a summarization prompt evicts the chat's cached prefix; returning to the chat with a *modified* history presents a new prefix in turn. Worse, since compaction targets the oldest part of the conversation, the first replaced message sits near the front — so nearly the whole prompt is reprocessed. A summarization event is therefore roughly a full prompt reprocess, which is the argument for **granularity**: compact rarely and in large chunks rather than continuously in small ones.
    - Two mitigations already have machinery in the tree. The context-fill indicator predicts *when* compaction will be needed before it is urgent; and `config.context_prefill_idle_delay` already runs a background LLM call while the user is reading, which is exactly the window in which a reprocess is free. Speculative compaction during idle is the natural pairing.
  - **Summaries belong in the chattree, and branching makes that pay.** A summary covering a span of nodes is derived data that must be cached and invalidated with the branch. Storing it against the span rather than the branch means every branch sharing that ancestry reuses it — the shared prefix is exactly where the oldest, most compactable material lives, so the reuse rate should be high. Consequence: building the sent context stops being a linear walk of `linearize_up` and becomes a policy evaluation over the branch (what is pinned, what is summarized, what is dropped), which wants its own module and its own tests rather than growing inside `_serialize_history_for_wire`.

- **[Medium]** Long-term memory: second RAG store indexing chat messages. Tool-call access (search with query, retrieve local neighborhood of a node). Automatic associative memory via autosearch on user's most recent message(s). Return user messages only (not AI replies) to keep the model grounded. **Design TBD — flag for second review round.** Hindsight may be a better backend here.

- **[Medium]** Explicit memory bank: third RAG store, AI-managed. Tool-call access (store/list/search/retrieve; title + content). Customizable system message section for things to remember across every chat. Chunk length may need adjustment (one chunk per memory). **Design TBD — flag for second review round.**

- **[Medium]** Three RAG stores architecture: (1) documents — explicit, user-managed (exists); (2) long-term memory — implicit, system-managed, indexes chat messages (new); (3) explicit memory bank — explicit, AI-managed (new). See memory items above.

- **[Medium]** Context fill meter.

- **[Medium]** Chat HEAD jump undo/redo.


### Chat UI

- **[Medium]** User-level settings: move the configs a *user* legitimately changes out of `config.py` into JSON, and give them settings dialogs. Today every one of them — LLM backend URL, model, docs directory, avatar knobs — is a Python source edit, which is why the tracked `config.py` files carry local overrides on every dev machine and have to be kept out of every commit by hand. Same gap the Visualizer has (the "Settings window: expose `gui_config` in the GUI" item above), so the two want a shared answer rather than two dialogs.
  - **This needs a degraded startup mode, and that is the part that will bite.** Librarian currently checks for the LLM backend at startup and bails if it is absent — perfectly reasonable when the only way to change the URL is to edit source and restart. Once the URL lives in a settings dialog, bailing means the user cannot reach the one control that would fix the problem: server down, server moved, laptop on a different network, and the app refuses to open far enough to be told so. So the connection failure has to become a state the GUI can *run in* — chat disabled with a clear reason, settings reachable, and a retry that does not require a restart.
  - Not everything should move. `config.py` is configuration-as-code and that is a feature for the parts that are genuinely code (the system prompt builder, the per-model VLM token table, computed paths). The split is "would a user reasonably want to change this without editing Python", not "is it a constant".
  - **Then: tools for the AI to read and change Raven's own settings**, which the digital-colleague track wants — a colleague you can ask to turn the avatar off, point at a different documents folder, or switch the send key is a different thing from one you have to configure around. Strictly gated on the JSON move: a tool that edits `config.py` would be rewriting source it is itself running under, whereas a JSON settings file is data, with a schema to validate against and a known set of keys to expose. Design questions when we get there, none of them settled: which settings are exposed at all (the LLM backend URL is the one that can lock the AI out of its own next turn), whether a change needs user confirmation, and whether the AI can see the values it is not allowed to change. Note this raises the same stakes as any actuation tool — see the memory note on where the actuation boundary belongs.

- **[Medium]** Indexing progress needs sub-document granularity and a live clock. `get_indexing_progress_text` advances once per document — `[14 / 186] | file.bib | elapsed 6s, ETA 01:14` — which was invisible while the corpus was Web of Science abstracts at ~1.3 kB each, and is not once a document is a 366 kB story producing several hundred chunks. The indicator then sits unchanged for the whole of one document and reads as hung, which is the worst thing a progress display can do: the user cannot tell a slow job from a dead one. Observed 2026-08-05 indexing 19 fan-fiction stories, four minutes total.
  - Two separate fixes, and the cheaper one is not the counter. **The elapsed/ETA figures should tick on the GUI's poll**, not on the worker's per-document update — the panel already polls once per frame, so it can recompute elapsed itself from a start timestamp and get a live clock for free, whatever the counter is doing. That alone removes the "is it hung" question.
  - **Then the counter itself**, which wants chunk-level reporting from inside the per-document work (chunking and embedding both know how far along they are). Note the same diagnosis as the status-panel item in `TODO_DEFERRED.md`: the numbers exist and are being formatted into prose before anyone outside can use them. Whatever shape this takes should report counts and let the caller render, so both consumers are served once.

- **[Medium]** Recent chats list view: still pending. Design is nontrivial in a tree-based storage — consider that each top-level user message constitutes a distinct chat, with the most interesting branches as a second level. UX should faithfully represent what the memory system actually remembers (if only the main branch is remembered, show only that).
  - Chat card: show something distinctive per chat (user's initial message, last branch point, most recent message, tags)
  - Click to switch; double-click to switch and close the list
  - Timeline section separators by date
  - Filter by persona names, tags; tag autocomplete; mass tag editing
  - HybridIR search (since chats will be indexed for memory); show matching snippet

- **[Medium]** Nonlinear chat view / chat graph editor: XDot DPG viewer now exists. Librarian needs to generate `.xdot` code; manual layout (no GraphViz needed for simple chat trees). Limit visible depth (full chat tree at interactive FPS is not feasible). "Jump to chat node by ID" feature needed.
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

- **[Medium]** Ctrl+F find in current chat: incremental fragment search; reuse existing generic infrastructure from Visualizer/XDot viewer.

- **[Medium]** Message editing: use chattree's revision system.

- **[Medium]** Bilingual chat display / on-demand translation of user input. Raven is English-only because Qwen (and Gemma, and Gemini) understand Finnish but can't *produce* acceptable Finnish. Translating Finnish *input* into English is feasible — `opus-mt-tc-big-fi-en` is already in `server/config.py`'s `translation_models`, commented out to save VRAM on smaller setups — but it needs UX work, not just the model:
  - A silently-applied wrong translation is worse than no translation, so auto-translated text must be prominently marked as such.
  - The original wording must be preserved in the datastore, never replaced by its translation.
  - Likely shape: "translate this message" / "translate conversation" actions, or a dual-language overlay showing both. Expect a couple of prototypes before settling — this is a UX design problem more than a plumbing one.

- **[Medium]** Robustness: temporarily disable relevant buttons while AI is writing; re-enable correctly by checking whether the relevant action has a stashed callback for that specific displayed chat message.

- **[Medium]** Fix bug: incomplete thought block (in first response) after Continue. Continuing should resume the incomplete thought block. May have a reproducible case still in the persistent chat tree — investigate.
  - **Probable mechanism, found 2026-07-27 while probing backends.** Continue works by prefilling the partial reply as a trailing assistant message — and a trailing assistant message means the template emits no generation prompt, which is exactly where the thinking prefix comes from. So the continued turn **cannot re-enter the thought channel**: if generation was interrupted mid-thinking, the block has no way to be closed, which is the reported symptom. Corroborated by Juha's recollection of the manual testing that produced it — the interruptions were sometimes during output and sometimes during thinking, and it is the latter that this predicts will break.
  - Still a hypothesis rather than a proven cause; the confirming test is to interrupt deliberately during thinking, then Continue, and check whether the reasoning channel reopens. Worth doing *before* designing a fix, because the fix differs: if this is the cause, resuming mid-thought needs the continuation to re-open the block explicitly (prefill the partial reasoning *inside* an open `<think>`), rather than anything in the renderer.
  - Land it with the thinking-toggle work above — same mechanism, same per-family marker table, and the same "wrong marker breaks generation" constraint applies.

- **[High]** Thinking trace: three improvements sharing one widget. **On the Researchers' Night phase-1 list** (see the Autumn 2026 section for why the first two are demo *correctness* rather than polish); post-0.2.8.
  - **Report what the thinking cost.** A complete message shows tokens / wall time / average speed; the thought bubble above it shows nothing, so the single largest consumer of a reasoning turn is the one part with no numbers on it. Put the same three under the thought button. The lead (Juha): `invoke` can detect where thinking *ends* — the `StreamParser` already emits reasoning and content as distinct event types, so the boundary is the first content event — and sample the token count and elapsed time there. Worth confirming that the boundary is crisp on every backend before building on it; LM Studio and oobabooga deliver reasoning differently.
  - **Let thinking start collapsed, as a live user preference.** Sometimes the reasoning is what you want to read; more often it is a wall of text between you and the answer. This is a toggle the user flips during a session, not a config knob set at startup — the right setting changes with what they are doing. Note the trace is already visible *as it streams* (blue text), so collapsing by default hides only the replay, which is the part that is in the way.
    - **With thinking hidden, the cloud has to say that thinking is happening**, or the app looks frozen for exactly as long as the model reasons — which on a thinking model is most of the turn. So: pulsate the thought cloud while the model is thinking, settle it to a static color when it finishes. Consistent with the indicator vocabulary the rest of the app already uses (the pulsating INDEXING / DOCS / SYSTEM / WEB indicators), so it needs no explanation to a user who has seen those.
  - Neither is blocked on the other, and neither is blocked on the eager-render question below — but if that gets revisited, this widget is the reason. The thinking-trace toggle renders both states up front and hides one (`_render_text`), which is cheap for a few KB and is why expand/collapse is instant; a long trace makes it less obviously the right trade, and the document-body toggle went the other way (rebuild on demand) for exactly that reason.

- **[Low]** Per-message backgrounds in the chat log — a tinted panel behind each message, keyed by role, so the eye can separate turns without reading them. Currently role is signalled by the icon and the persona name only.
  - **An abandoned attempt is in the history, and the approach is what to avoid, not to resume.** The original librarian WIP (`ef3a5d9`, Oct 2025) drew a rounded `draw_rectangle` into a drawlist positioned behind each message's container. Being a drawlist rather than a laid-out widget, it had to be told its own geometry: the message's rect size is not known until a frame after it is built, so it needed a deferred frame callback per message — and since `set_frame_callback` holds one callback per frame number, that meant a queue plus a master callback to drain it (`DPGChatMessage.callbacks` / `run_callbacks`, which is what those were for). It did run, a couple of times, during development; it was commented out before the first commit that carries it, and the queue then rode along as dead code until it was removed.
  - **Why it was dropped** (Juha's recollection, so treat it as a lead rather than as a finding): three things had to hold at once — the rect sized correctly, the box behind the message in z-order, and the box scrolling with the chat log — and it was a pick-two-of-three situation, possibly pick-one.
  - That is what makes a *widget* the likelier shape than a drawlist: a container with a background of its own is sized, ordered and scrolled by the toolkit, so all three fall out instead of being maintained. An ImGui child window takes `mvThemeCol_ChildBg` from a theme, one theme per role. Unverified — check whether a per-message child window is affordable at chat-log lengths before committing, since each one is a scroll region and a clip rect.

- **[Low]** Add lockfile so `raven-minichat` and `raven-librarian` can't run simultaneously (prevents losing changes made in one app). Quick CC session.

- **[Low]** minichat: **[Verify]** when retrieval results are `null` in `data.json` — old bug or still present in current codebase? (CC session)


### STT / voice

- **[High]** STT: configurable silence level, autostop timeout, VU peak hold time. Needs a GUI, not just config knobs — the noise threshold for auto-stop has to be tunable in the room, on the day. Demo-facing (Researchers' Night, 2026-09-26).

- **[High]** STT: input-language selector in the GUI. `api.stt_transcribe` / `stt_transcribe_array` already take `language: Optional[str]` (`None` = autodetect) and the server honours it; Librarian's only call site (`app.py`, `stop_recording_audio_message`) just never passes it. So the plumbing exists — what's missing is the control.
  - A **combobox**, not a config knob: "Automatic" plus each configured input language. The language has to change *between questioners*, not at startup — a Researchers' Night audience will mix Finnish and English speakers, and switching on the fly is the difference between the mic working for everyone and working for half the room.
  - Read the selection at transcription time (the pattern `_make_open_folder_callback` already uses for directories), so a mid-session change takes effect on the next recording with no restart.
  - The offered list comes from config — Whisper handles ~99 languages, but the demo wants two. Distinct from the *subtitler's* output language (`gui_config.translator_target_lang`); don't conflate them.
  - Edge case: an English-only Whisper build (`whisper-base.en`) makes the selector meaningless. Hide or disable it when the configured `speech_recognition_model` ends in `.en`.
  - Autodetect stays worth offering but shouldn't be the only option: Whisper's language detection is least reliable on short utterances in a noisy room, which is exactly a live Q&A.
  - **Show what was detected.** When the selector is on "Automatic", briefly flash the detected language code somewhere unobtrusive — a corner of the avatar panel is the natural spot. Not cosmetic: a misdetect currently fails *silently*, producing a plausible-looking transcription in the wrong language, and this converts it into something the operator can see and correct. Only worth showing in Automatic mode; when the language is pinned, it's noise. Reuse the existing `animation` flash machinery, and note it stays legible under the colorblind-signaling item since a language code is text rather than a color.
    - **Prerequisite: nothing returns the detected language today.** `server.modules.stt.speech_to_text` returns a bare `str` and `api.stt_transcribe` returns `List[str]`, so the detection is discarded at the engine boundary. Surfacing it means changing the response shape through all three layers (`common.audio.speech.stt` → server module → client API) to carry text *plus* language. First check whether the engine wrapper can expose it at all — Whisper detects the language internally, but whether `common/audio/speech/stt.py`'s `transcribe` can hand it back needs reading. Worth doing now rather than later: the response-shape break is free while Librarian has no outside users, and gets expensive once it does.

- **[High]** Finnish demo path — end-to-end test. The chain that lets a Finnish-speaking audience interact without the LLM ever producing Finnish: Finnish speech → Whisper (multilingual, language selected in the GUI per the item above) → Finnish text → the LLM *understands* it → answers in English → TTS speaks English → the subtitler translates to Finnish. Every hop is believed to work; the whole has never been run. Test the English-input path through the same chain too — a mixed audience is the expected case, not the exception. Two known gaps: `speech_recognition_model` is `openai/whisper-base` (74M, chosen for CPU) which will be rough on Finnish — `whisper-large-v3-turbo` (~1.6 GB) is commented out two lines above in `server/config.py` and is affordable on a mid-VRAM setup — and nobody has tested the chain with a real Finnish question.

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

- **[Medium]** Weather and calculator tools: both parked in brief 01 §6 and specced in `TODO_DEFERRED.md`, "Add built-in calculator and weather LLM tools". Weather via open-meteo (https://open-meteo.com/en/docs) — makes Librarian more humanlike as a "voice with internet access" (HCI is a major Raven goal). Calculator via secure eval limited to math expressions; `eval` itself is unsafe (see notes in the archived section), candidate https://github.com/danthedeckie/simpleeval.

- **[Medium]** Calendar tool: get one- or three-month calendar, like the `cal` command-line utility. See Python's `calendar` module.

- **[Medium]** `webfetch`: content-aware extraction per site, so a fetch opens with the part a reader actually wants. Generic extraction takes the page in document order, which on a Wikipedia article means the infobox — a fetch of *Corvidae* opens `| Kingdom: | Animalia |` and spends the whole excerpt on taxonomy boxes before reaching a sentence of prose (observed 2026-08-04). It reads as broken even though nothing failed, and it is exactly the kind of thing an ideal librarian product gets right.
  - **The hook already exists**, so this is extension rather than construction: `webfetch._rewrite_url` returns an optional per-site extractor, and arXiv, Reddit and YouTube already use it (`_extract_arxiv`, the `old.reddit.com` rewrite, `_extract_youtube_transcript`). It is pure and separately unit-testable by design — the rewrite decision is tested without touching the network.
  - **First version: Wikipedia and arXiv.** Wikipedia is the one that is visibly wrong today — lead section first, infobox after or dropped; it has a REST content API that returns exactly the lead, which would sidestep the extraction problem rather than fight it (worth verifying before committing to it — this is recalled, not checked). arXiv already has an extractor and mostly works; the remaining nit is that a Google-permissions blurb precedes the abstract on the HTML rendering, so the abstract starts ~350 characters in.
  - **Then, for scientific users**, roughly in order of how often they will hit them: `doi.org` (currently resolves to a publisher page that is often a login wall — Crossref content negotiation returns real metadata for a DOI whether or not the fulltext is reachable, which turns a useless fetch into a citation); PubMed / PMC (E-utilities for the abstract, PMC OA for fulltext); bioRxiv / medRxiv (public API, and preprints are open by definition); Semantic Scholar (abstract plus a link to any open-access PDF). All four API claims are from memory — check each before building on it.
  - **A second symptom, same root: a page that is not an article at all.** `https://astronomynow.com/2026/` is a year index; generic extraction returned 1398 characters that stop mid-sentence — *"…now there's someone up there from my own country, and while"* — because it latched onto one entry's teaser and ran out. Verified against the server directly, so the truncation is in extraction rather than anywhere downstream (`spaSuspected` false, so no second-tier retry was even attempted). Worth handling explicitly: an index or listing page wants either its list of links (which is what the reader would click) or a refusal saying it is a listing, not half of one item presented as the content. The failure is quiet, which is the dangerous part — the model receives a truncated sentence with no marker and no way to tell it apart from a short article.
    - **`docextract` now detects this failure, and the detector may port over — the threshold does not.** A saved page holding one `<article>` per chapter had the same thing happen: readability extraction chose one block and silently dropped the rest. The fix there compares the extraction against `trafilatura.html2txt` (the whole page, no block selection) and treats a small ratio as truncation rather than boilerplate removal, then re-extracts per `<article>` to keep the Markdown. The *mechanism* is shared with this item and the remedy might be too. The *constant* is not transferable: it was calibrated on locally saved pages, which carry almost no chrome, whereas a live page legitimately loses a fair fraction to navigation, ads and related-article blocks — so reusing 0.5 here could fire the fallback on pages that were extracted correctly. Recalibrate against live fetches before porting.
  - **The general principle worth extracting from the specific cases**: when a site has an API that returns the content as *data*, prefer it to scraping the rendered page. Extraction from HTML is a heuristic recovering structure that was thrown away; an API hands the structure over. That is also what makes these testable — a fixture of the API response, rather than a snapshot of a page that will be redesigned.

  Raised by Juha 2026-08-04, from the Wikipedia excerpt.

- **[Done 2026-07-29]** RAG access via tool-call: search the document DB with a given query, fetch a full document by ID, list what the conversation has already consulted. Auto-inject kept alongside tool access, as planned. Shipped as `search_documents`, `fetch_document` and `list_consulted_documents`, with a per-name tool gate, an agent-loop round cap, grounding-by-declaration, and a shared truncation budget that also repairs the attachment wire-fold. The design and everything decided while building it are in `briefs/summer_2026_librarian_extension/done/10_rag-tool-surface-brief.md`. **Not yet run against a real corpus** — the manual-test probe uses a stub retriever. One piece of the original scope did not land here: a *scoped* search, and a lister for the available topics/scopes, both of which wait on the document-scopes item above.
  - **Investigated 2026-07-28; the placement question is settled.** Qwen no longer needs the results near the start — the constraint did not reproduce on any current model, at any corpus size. But the obvious alternative, injecting at the *end*, fails differently: with a tool result as the last message the history reads as a paused agent loop, and Qwen 3.6 answers by requesting another search. The placement that works is **immediately before the user's latest message**, which keeps the prefix ahead of it stable *and* leaves the question last (36/36 across three models and two corpus sizes). Now implemented in `scaffold._perform_injects`; full method and numbers in `investigations/context-injects/context-inject-shape-measurements.md`. So the KV-cache objection to auto-inject is gone, and this item is purely about adding the tool.
  - **New motivation from the same sweep (Q11):** asked something the auto-injected matches don't answer, the model reaches for a second, better-aimed search — a query like `"Kelvin-7"` that only exists *after* reading the first pass. Having no such tool, it writes the call out as literal `<tool_call>` text and the user gets that instead of an answer (~1 turn in 3 on Qwen3.6-27B). Talking it out of asking was tried and made things worse, so this is deliberately unmitigated until the tool exists. Note the division of labour this implies: auto-inject buys a zero-latency first pass from a cheap heuristic query, and the tool buys a model-authored refinement — which is also the argument for keeping both.
  - **Designed 2026-07-28; the design is `briefs/summer_2026_librarian_extension/done/10_rag-tool-surface-brief.md`.** Tool surface, the per-name tool gate, the agent-loop cap, grounding-by-declaration, the shared truncation budget, the "mark, don't hide" replacement for the no-match bypass, and the four-commit build order all live there rather than being restated here. Two things that changed from the earlier plan recorded above: the bypass is *deleted* rather than moved (its false-positive rate is near 100% on general-knowledge questions, so it becomes a badge), and `fetch_document` is in scope, because it is also what closes the follow-up hole left by the auto-search's ephemeral results.

- **[Medium]** Websearch: **[Verify]** whether raw URLs are currently saved in tool results. Remaining work: final formatting of results, link crawling to retrieve full result documents (persist to RAG with expiry timeout), figure out in which contexts search result pages should be enabled as RAG data sources.

- **[Medium]** HybridIR pedigree field: auto-remove only documents added by a named scanner instance. Needed for programmatic RAG ingestion (e.g. web pages from websearch).

- **[Medium]** Source attribution for RAG: clickable snippets in GUI based on `document_id`, `offset`, length; clickable link to open full document (spawn external viewer based on file type). Same feature as `TODO_DEFERRED.md`, "Expose the docs-DB source files behind a reply's RAG citations" — that entry carries the current design questions (where the affordance lives, snippet vs. whole document) and notes the `open_file` / `open_in_file_manager` machinery it can reuse.

- **[Medium]** Inline citations: encourage LLM to inline citations in a specified format; validate programmatically that cited IDs exist in the RAG result set; flag invalid citations. Design goal: preserve synthesis (don't force one-paragraph-per-source).

- **[High]** MCP support: specced in `briefs/summer_2026_librarian_extension/04_librarian-mcp-client-brief.md` — client-side MCP tools registered *alongside* the built-ins, all feeding the existing `perform_tool_calls` loop. Gated on the Hindsight playground (brief 06). Main line for the "digital colleague" track: this is how Librarian reaches the lab's systems. Agent skills (CLI-based, "anime maid form factor" — plugging into interfaces designed for human use) remain a superior alternative capability-wise but more dangerous for the user's computing environment; still under consideration as a separate path.

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

- **[High]** Per-module VRAM budget: measure it once, so the config variants below are derived from numbers instead of trial and error on demo day. All unique models load at server startup (`server/config.py`), so the budget is the resident sum across the nine placeable modules — `avatar`, `classify`, `embeddings`, `imagefx`, `natlang`, `sanitize`, `stt`, `translate`, `tts` — plus whatever the LLM backend takes in its own process.

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


---

## Avatar

- **[High]** Prerequisite for both avatar-effect briefs below: **give `_priority` a stated meaning** (`briefs/summer_2026_librarian_extension/crt-display.md` §0). The existing numbers already almost form a scheme — Scene (< 0), Capture (0–5), Signal (5–10), Display (≥ 10), with `0.0` as the moment of capture — so codifying it renumbers nothing. Documentation plus a convention, no code motion; the brief asks for it as its own commit, and both new filters need the Scene band to place themselves in.

- **[High]** `crt` — raster projection simulation (`briefs/summer_2026_librarian_extension/crt-display.md`). Wanted for the Researchers' Night demo (2026-09-26). **Adds** a filter at priority −3.0, in the Scene band: the hologram's own raster is diegetically *in the world*, so it composites early and rides through the capture-stage optics like the character does. It replaces nothing — `scanlines` (13.0) stays as the *viewer's* monitor, a different diegetic layer, and the brief is explicit that this filter must not be described as superseding it. The capture-band effects (`bloom`, `chromatic_aberration`, `vignetting`) are untouched and in fact load-bearing: downstream `bloom` glows the scanlines for free, which is why `glow_strength` defaults to 0.0. `banding` isn't involved.

- **[High]** `atmospheric_dust` — drifting in-air particles (`briefs/summer_2026_librarian_extension/atmospheric-dust.md`). Wanted for the Researchers' Night demo (2026-09-26). Light-catching motes in the avatar's air, priority −2.0, Scene band, for the same diegetic reason. Register is anime-atmospheric (dust in a sunbeam), not game-HUD sparkle. Budget: ≤ 1.5 ms at 1024² against the postprocessor's current ~11 ms stage.

- **[High]** Add help cards for: Avatar settings editor, Avatar pose editor.

- **[Low]** Implement JS client for integration of Avatar with other LLM frontends. Needs work on those other frontends, too. Initially, target SillyTavern.

- **[Low]** Update assets for all characters: add at least the eye-waver effect (and possibly other cel-blending cels). Aria is the default character with full feature support. Other characters are lower priority.


---

## XDot Viewer (`raven-xdot-viewer`)

*(No outstanding items.)*

---

## Papers tooling (e.g. pdf2bib, csv2bib)

- **[Medium]** `raven-docdb-import` (or similar — check naming convention against existing CLI tools): CLI tool for batch document ingestion into the Librarian document DB. Just run `hybridir.setup` on the same datastore that Librarian uses and wait for the scanner to finish.

- **[Medium]** pdf2bib: prompt the author extraction step to return a canonical string (e.g. "No authors provided") when no authors are found. Same for title extraction (e.g. "No title provided"; also handle the case where the LLM thinks the title is literally "Abstract").

- **[Medium]** Some LLMs behave erratically when the system date is later than their training cutoff (e.g. refusing tasks, claiming to be in a simulation). Investigate mitigation strategies; may be model-version-specific. Track across model upgrades. First seen in pdf2bib, but **not a pdf2bib problem** — it follows the date inject, so it applies anywhere Raven tells a model what day it is, which is every Librarian turn (`scaffold._perform_injects`). Measured groundwork exists: `investigations/context-injects/datetime_inject.py` asks exactly whether a model believes us over its own priors.
  - **The mild form is the one that will persist, and it costs tokens rather than correctness.** Qwen3.6-35B, 2026-08-04, asked about Artemis after a websearch: it reconciled a snippet dated April 2026 against its own priors out loud, at length — *"in reality (2024/2025 knowledge), Artemis II is scheduled for late 2024/early 2025 … I must check if there is real news about Artemis II delaying to 2026 or if the snippet is just a hypothetical/future-dated article or if I am misinterpreting the date"* — before accepting the injected date and proceeding correctly. No refusal, no simulation claim, right answer; just a large slice of the thinking budget spent re-deriving that the present is the present, on every turn where retrieved material carries a date.
  - So the success criterion is not "does it refuse" but "how much deliberation does the date cost". That is measurable with the existing probe, and it is worth measuring per model as the fleet upgrades: the erratic form may be disappearing while the expensive form stays.

- **[Medium]** pdf2bib overthinking / token-limit mitigation: detect token-limit-exceeded in `raven.librarian.llmclient`, return a status flag in metadata. Consider executive-function simulation via LLM (in the neuropsychology sense: https://en.wikipedia.org/wiki/Executive_functions) as a recovery strategy — but may be superseded by improved model capabilities; monitor before investing time.

- **[High]** csv2bib: add documentation (main README + Visualizer README, section on importing data). New CLI tool added in 0.2.5.


---

## Infrastructure and maintenance

- **[High]** Unit tests. Currently very sparse. Would significantly improve confidence in refactoring.

- **[Low]** Post PR of vendored FileDialog fixes upstream. Raven's extensions have genuine added value worth sharing. Upstream is likely inactive but the PR is worth filing.

- **[Low]** Fork kokoro/misaki and bump their Python upper bound (`<3.13` → `<3.15`), then test on 3.13+. The `<3.13` cap may be precautionary rather than reflecting real incompatibility. kokoro appears effectively abandoned upstream, and it's the only TTS engine that provides timestamped phoneme data (required for avatar lipsync). Currently Raven's `requires-python` is narrowed to `<3.13` to accommodate this.
  - **Consider forced alignment before forking anything (noted 2026-07-28, unverified).** The requirement is not "a TTS that reports phoneme timings" but "phoneme timings", and those can be recovered after synthesis: align the generated audio against the text it was generated from, and read the boundaries off the alignment. torchaudio ships a forced-alignment API for this. If it works, **the constraint that pins this whole item disappears** — engine choice reopens to whatever sounds best or runs smallest, kokoro stops being load-bearing, and the Python cap can be lifted by swapping the engine rather than by forking an abandoned one. It would fit the three-layer pattern cleanly (alignment is `raven.common`, engine-agnostic) and the lipsync driver already consumes `WordTiming` objects rather than anything Kokoro-shaped, so its input contract would not change.
    - **Checked 2026-07-28, and the obvious objection does not apply.** The worry was phoneme-level granularity: word alignment is easy, phoneme alignment is not, and the mouth morphs are driven per phoneme. But `lipsync.build_phoneme_stream` already splits each word's timespan *linearly across its phonemes* — Raven has never had phoneme-level timings and does not use them. `WordTiming` carries word, phoneme string, start, end. So the requirement decomposes into **word-level timings** (which `torchaudio.functional.forced_align` provides out of the box, with CPU and CUDA implementations, via `Wav2Vec2FABundle`) and **a phoneme string per word** (G2P, which is independent of the TTS engine). Neither needs a phoneme-aligning model. Alignment would also be running on synthetic speech — clean, no noise — which is the easiest case for a CTC aligner.
    - **The real risk is the phoneme inventory, not the timings.** The morph map does `vocabulary[phoneme]`, and misaki emits IPA (`mˈaɪnd` → `m, ˈ, a, ɪ, n, d`). A replacement G2P must emit a compatible inventory or the vocabulary needs remapping. Check that before anything else.
    - Still to weigh: added latency, and one more resident model (wav2vec2-class, a few hundred MB — the VRAM measurement above says there is room on both cards).
    - Prompted by hitting the same wall twice. KittenTTS: asked upstream, no reply. pocket-tts: **confirmed no native timestamps** (kyutai-labs/pocket-tts issue #66); upstream's position is that they would point users at a pipeline rather than build one in. Note their larger Kyutai TTS 1.6B *does* report word-level timings, so this is a model-tier decision rather than a house policy.

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

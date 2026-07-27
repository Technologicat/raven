# Raven TODO

Covers the full Raven constellation: Visualizer, Librarian, Server, Avatar, XDot Viewer, and shared tooling.

Priority tiers: **[High]** | **[Medium]** | **[Low]** | **[Parked]**

Items marked **[Verify]** should be checked against the current codebase in a CC session before implementing.


---

## Cross-cutting

- **[High]** HF hub: document the env vars that prevent hub checks on Raven startup (for privacy and faster startup). Add recommendation to server docs. Currently not written down anywhere in the project.
  - `HF_HUB_OFFLINE=1` — forces huggingface_hub to use only locally cached models, no network requests at all.
  - `HF_HUB_DISABLE_TELEMETRY=1` — stops telemetry pings only.

- **[High]** Neural reranker for HybridIR: add a reranker stage to avoid needing large k values (k=100 style workarounds). Since we maintain our own HybridIR backend, we can power it up properly. Design worked out in `TODO_DEFERRED.md`, "RAG: rerank retrieved chunks and inject only the best few" (cross-encoder stage, three-layer `common`/`server`/`mayberemote` shape, VRAM tradeoff).

- **[High]** Revisit logging system: library modules should not reconfigure the logger (verify exact behavior against Python `logging` stdlib docs, but currently each module sets the log level, which is the entrypoint's responsibility). Move logging configuration to entrypoints only. Add a "detailed debug" level at that time for particularly spammy-but-useful log lines (e.g. `SmoothScrolling.render_frame`, `_managed_task`, `binary_search_item`).

- **[Medium]** Flash the search field when focused by hotkey. Currently affects Visualizer main window, fdialog component, and XDot Viewer. Generalize `ButtonFlash` for GUI elements other than buttons.

- **[Medium]** `vis_data` → `entries` rename across the whole constellation, including importers and BibTeX tooling in `raven.papers`.

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

- **[Medium]** Think blocks: parse properly instead of current regex hack. We already receive one token at a time.

- **[Medium]** Proactive context engineering: move beyond reactive BM25+semantic retrieval toward intelligent context curation. The system should maintain a graph of topical connections and proactively include relevant documents the user didn't explicitly ask for. E.g. "You asked about hydrogen embrittlement — here are the materials science papers you looked at last month." Shallow version (agentic chain-of-thought retrieval over a topic graph) is achievable now; deeper version requires a world model.

- **[Medium]** Document scopes: subdirectory-based filtering; scope selection GUI (checkbox per scope, select/unselect all); tags as the primary scoping mechanism (auto-tag by subdirectory name on ingestion); avoid cross-contamination between work/hobby contexts. Needed for long-term memory too. Currently must manually switch directories for each demo.

- **[Medium]** BM25 migration from `bm25s` to ChromaDB FTS5: gains incremental updates and metadata filtering (needed for scopes); removes full index rebuild at each commit; simplifies `hybridir.py` and removes a dependency. Mitigate tokenization quality loss by storing spaCy-lemmatized text in a dedicated ChromaDB field for FTS5 search. **Low priority** — `bm25s` works, and Raven's dependency policy is already generous.

- **[Medium]** Context compaction: drop and/or summarize old messages when context window fills. Use `raven.llmclient.token_count` to bisect linearized history to find the cut point (accounting for max response length from `settings.request_data["max_tokens"]`). Medium priority — in practice, usually start a new chat before running out, but any serious LLM frontend needs this. Budgeting details in `TODO_DEFERRED.md`, "Context-window budgeting and conversation compaction (Librarian)".

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

- **[Medium]** RAG access via tool-call: search the document DB with a given query (optionally scoped), fetch a full document by ID, get available topics/scopes. Keep auto-inject (current scaffold behaviour) alongside tool access — don't replace it.
  - **Investigate:** Qwen currently requires RAG results near the start of the context or comprehension degrades — this kills KV cache hit rate (near-full preprocess each turn since the RAG result set changes). Check whether injecting near the end of the chat still works; newer model versions may have fixed this. No official Qwen documentation for RAG-with-tool-use found yet.

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

- **[High]** CRT filter improvements. Wanted for the Researchers' Night demo (2026-09-26). The retro-display look is currently *assembled* from separate `Postprocessor` filters — `scanlines`, `banding`, `chromatic_aberration`, `vignetting`, `bloom`, `translucent_display` / `monochrome_display` — with no dedicated CRT filter. What "improvements" covers is to be settled in a brief (discussed separately with claude.ai; not yet written up).

- **[High]** Floating glittering dust: ambient particle effect around the avatar. Wanted for the Researchers' Night demo (2026-09-26). No existing filter covers it; the candidate homes are the addon-cel / animefx layer and the `Postprocessor` chain, and which one it belongs in is a design question for the brief (also discussed separately with claude.ai; not yet written up).

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

- **[Medium]** pdf2bib: some LLMs behave erratically when the system date is later than their training cutoff (e.g. refusing tasks, claiming to be in a simulation). Investigate mitigation strategies; may be model-version-specific. Track across model upgrades.

- **[Medium]** pdf2bib overthinking / token-limit mitigation: detect token-limit-exceeded in `raven.librarian.llmclient`, return a status flag in metadata. Consider executive-function simulation via LLM (in the neuropsychology sense: https://en.wikipedia.org/wiki/Executive_functions) as a recovery strategy — but may be superseded by improved model capabilities; monitor before investing time.

- **[High]** csv2bib: add documentation (main README + Visualizer README, section on importing data). New CLI tool added in 0.2.5.


---

## Infrastructure and maintenance

- **[High]** Unit tests. Currently very sparse. Would significantly improve confidence in refactoring.

- **[Low]** Post PR of vendored FileDialog fixes upstream. Raven's extensions have genuine added value worth sharing. Upstream is likely inactive but the PR is worth filing.

- **[Low]** Fork kokoro/misaki and bump their Python upper bound (`<3.13` → `<3.15`), then test on 3.13+. The `<3.13` cap may be precautionary rather than reflecting real incompatibility. kokoro appears effectively abandoned upstream, and it's the only TTS engine that provides timestamped phoneme data (required for avatar lipsync). Currently Raven's `requires-python` is narrowed to `<3.13` to accommodate this.

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

# TODO triage, pass 5: worksheet

Written 2026-09-30 at the end of passes 1–4, so the proposals carry that session's reading of the items.
**Every row is a proposal, not a decision.** Answer per row — "ok", or what to do instead — and the next
session applies the answers and deletes this file. Reasons marked *(inferred)* are mine rather than
anything an item or the maintainer said.

## A. The [High] items in `TODO.md`

Tiers here are priorities, so this overlaps the prioritization session; do this table first only if it
helps that session start from a cleaner list.

| item | proposal | why |
|---|---|---|
| HF hub: document the env vars (`HF_HUB_OFFLINE`, telemetry) | keep [High]; do soon | S, privacy, and user-facing *(inferred)* |
| Revisit the logging system (library modules must not configure the logger) | **verify first** | part may be done by the fleet-wide logsetup work (`briefs/done/logsetup-fleet-wide.md`) |
| Author search: the full author list, search-aware | → [Medium] | open-ended GUI work, no deadline *(inferred)* |
| DOI in the importer, info panel, button, export | keep [High] | literature reviews lean on DOIs *(inferred)* |
| Publish a quick-start dataset | → [Medium], unless a public release is coming | its value is for new users *(inferred)* |
| HybridIR integration (1): full-text search over BibTeX | → [Medium], read against brief 13 | brief 13's unified DB may absorb it |
| Tool-call round budget for a multi-document read | keep [High]; it is a probe re-run, ~S | cheap to settle |
| Adjustable similarity threshold | → [Low] | brief 09 found no threshold carries across corpora |
| Inline citations, validated | keep [High] | |
| Context compaction | keep [High] | long chats with attachments hit the window |
| Show the raw prompt (the prompt viewer) | keep [High] | decisions already taken; ~one session |
| Wake-word trigger | keep [High] | the item says the priority stays (Juha, 2026-08-25) |
| Finnish demo path, end-to-end test | → [Parked] | the demos run in English |
| Support Anthropic-style backends | keep [High] | |
| MCP support (brief 04) | keep [High]; schedule with 05/06 | |
| Server config variants by VRAM tier | keep [High] | a single modest GPU is a supported configuration |
| Unit tests ("very sparse") | **delete** | stale: 105 test modules; the specific gaps are filed in `TODO_DEFERRED.md` |

## D. Items gated `post-0.2.10`

Every item whose gate says `post-0.2.10`, grouped by cluster. After the 7th that gate stops meaning anything
precise, so each wants a real one: `0.2.11`, `later`, a named precondition, or removal. The summary is the
item's first sentence, cut at 160 characters.

| cluster | item | cost | gate now | what it is | proposal | answer |
|---|---|---|---|---|---|---|
| cherrypick | Cherrypick: crown the winner without leaving the compare cycle | S | post-0.2.10 | `Ctrl+Shift+C` is `_mark_winner`: cherry the current image, lemon the rest of the selection. | 0.2.10 — S and UX, by your rule | |
| document-ingestion | HTML pages whose content is produced by running them | ? | post-0.2.10, with the cluster | `raven.common.docextract` reads HTML through `trafilatura`'s readability extraction, which looks at markup. | → the document-ingestion sketch (to be written) | |
| document-ingestion | Read documents as page images, for figure- and math-heavy sources | ? | post-0.2.10, with the cluster — and the one that bites hardest | Current extraction is **text-layer only**, for PDFs and (as of 0.2.8) office formats alike. | → the document-ingestion sketch | |
| document-ingestion | Same file formats in the docs DB and in chat attachments | ? | post-0.2.10, with the cluster | The docs database and chat attachments should accept the *same* set of formats. | → the document-ingestion sketch | |
| document-ingestion | Spreadsheets in the docs DB and attachments (`.xlsx`, `.ods`) | ? | post-0.2.10, with the cluster | Left out of the office-formats work deliberately: a spreadsheet is a different problem class wearing the same file picker. | → the document-ingestion sketch (it has its own brief already) | |
| document-ingestion | Text out of images, so figures work without a vision model (OCR, and SVG `<text>`) | ? | post-0.2.10, with the cluster | The image → text cell of the 2×2 in the SVG item below: given an image, produce its plain text. | → the document-ingestion sketch | |
| document-ingestion | Vector figures in the docs DB and attachments (`.svg`) | ? | post-0.2.10, with the cluster | Hand-authored figures — problem setups, schematics, diagrams — are commonly SVG, because that is what you get when you draw them yourself for a manuscript ra… | → the document-ingestion sketch | |
| hygiene-sweep | Sweep `## Declined` for decisions whose follow-through was never filed | S | post-0.2.10 | `## Declined` was originally built — by claude.ai, at the start of the project — as a section for *completed or already-decided* items, and only later correc… | 0.2.10 — S, and it keeps the file honest | |
| performance | The thumbnail grid's textures are dynamic, and probably need not be | S | post-0.2.10 | `ThumbnailGrid.set_thumbnail` creates a **dynamic** DPG texture per thumbnail, so a Cherrypick folder of a few hundred images registers a few hundred of them. | 0.2.10? S, but performance rather than UX | |
| polish | Raven's global theme sets three of ImGui's seven rounding vars | S | post-0.2.10 | `raven.common.gui.utils.setup_themes` sets `FrameRounding` 6, `WindowRounding` 8, `ChildRounding` 8, and **`PopupRounding` 6 as of 2026-08-14** — added becau… | 0.2.10? S polish | |
| system-prompt-and-greeting | Make the canned AI greeting optional | M | post-0.2.10 — and a `chatutil` cleanup now waits on it too (2026-08-25) | A new chat opens with a canned greeting from the AI (`raven.librarian.config`, "Names, AI's greeting"). | → with the system-prompt trio, 0.2.11 | |
| ? | Agent skills for Librarian (natural-language workflows over the document database) | ? | post-0.2.10 | Design work deliberately postponed; this records the idea and what is already established about it. | later | |
| ? | Avatar settings editor: custom postprocessor chain ordering | L | post-0.2.10 | **This is a GUI limitation only** — the band-scheme comment above the first filter definition in `raven/common/video/postprocessor.py` establishes that the b… | later | |
| ? | Batch tools: LLM reconnect mid-run | ? | post-0.2.10 | The model-loaded work made `raven-pdf2bib` and `raven-importer` stop at *start time* on both failure states — unreachable, and reachable-with-no-model. | → the per-document LLM pass, which supersedes it (keep as pointer) | |
| ? | Browse *all* attachments in the datastore, not just the orphaned ones | ? | post-0.2.10 | The cleanup dialog (`raven/librarian/cleanup_dialog.py`) turned out to be a decent attachment browser that happens to be filtered to orphans. | 0.2.11? | |
| ? | Client-local avatar animator (licensing-bounded) | ? | post-0.2.10, deprioritized | The avatar animator currently lives only in `raven.server.modules.avatar` under AGPL. | later | |
| ? | Context-window budgeting and conversation compaction (Librarian) | ? | post-0.2.10 | Librarian does not yet budget the prompt against the model's context window, nor compact long conversations. | the prioritization session — it is [High] in `TODO.md`, and L | |
| ? | Datastore scaling: a single `chat.json` (+ flat sidecar dir) won't hold years of chats | ? | post-0.2.10 | Librarian stores *every* chat — all nodes, all payload revisions, across the whole forest — in one `chat.json` (`chattree.PersistentForest`), and every attac… | later; the minute autosave is fine for now | |
| ? | Extract `raven.common` into an upstream library ("corvid") | ? | post-0.2.10 (or —); nothing is waiting on it | Raven's `common/` package has grown into a general-purpose DPG toolkit: GUI widgets (file dialog, markdown, helpcard, xdot widget, animation framework, VU me… | later | |
| ? | Faster PNG decoder | ? | post-0.2.10 | PIL's PNG decode via libpng is slow (~59 ms for a 1 MP image). | with Cherrypick's performance cluster, later | |
| ? | Librarian: open a chat datastore other than the configured default | ? | post-0.2.10 | Librarian loads one datastore, fixed at `librarian_config.llm_datastore_file`. | 0.2.11? | |
| ? | Modernize the Librarian system prompt / character card | ? | post-0.2.10 | The default system prompt (`raven.librarian.config`) reads as dated for current instruction-tuned models — "take a deep breath and think step by step", "beli… | → with the system-prompt trio, 0.2.11 | |
| ? | Remaining server modules without a MaybeRemote | ? | post-0.2.10, scoped down | With `Classifier`, `Translator`, `Postprocessor`, `Upscaler` landed (2026-04-22), the following server modules still don't participate in the MaybeRemote pat… | remove — a navigational note, not a task (roadmap part 4) | |
| ? | Revisit `recenter_window`'s degrade-instead-of-raise policy | ? | post-0.2.10 | `guiutils.recenter_window` passes `required=False` for its offscreen-measure wait, so calling it from the render loop thread warns and centers using whatever… | later | |
| ? | System prompt templating: the user should choose where the per-turn facts go | M | post-0.2.10 | Filed 2026-08-12 to make good on a condition set when the multi-root work landed: today's advice — **do not use `{model}` or `{context_length}` in a card** —… | → with the system-prompt trio, 0.2.11 | |
| ? | TTS reads arXiv IDs digit by digit | ? | post-0.2.10 | Qwen likes to cite arXiv papers by their full identifier, and the TTS then says "twenty twenty six dot zero five ... | 0.2.11? | |
| ? | The ingest pool's concurrency is nominal: pypdf is pure Python | M | post-0.2.10 | Split out 2026-08-12 from "Indexing a large corpus is silent for minutes", whose titular half shipped. | → the offline-processing sketch: import-time work runs offline anyway | |
| ? | Visualizer's importer should read the document database, not just `.bib` files | ? | post-0.2.10 | Visualizer ingests BibTeX databases. | → brief 13 | |
| ? | Web status panel: check on a long job without being at the machine | ? | post-0.2.10 | The motivating case is concrete: a ~12k-abstract hydrogen indexing run, and no way to see how it is doing except the Librarian window and the terminal that l… | → the offline-processing sketch, open question 3 | |
| ? | raven-cherrypick: export image sequence (QOI→PNG batch conversion) | ? | post-0.2.10 | raven-cherrypick is effectively an image viewer with QOI support, which is rare. | later | |

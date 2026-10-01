# Roadmap overview, October 2026

*A snapshot of every open item in `TODO.md`, `TODO_DEFERRED.md` and the open briefs, taken 2026-09-30 as input
for prioritizing the autumn. It orders nothing: prioritizing is a session with the maintainer against this
page. It edits nothing either — the TODO files are as they were, and part 4 is the input for their triage.*

**How to read a line.** `**Title** — what it is · cost · gate · where`. Cost is the item's own where it has one,
`~S`/`~M`/`~L` where a reader estimated it, `?` where nobody has. A gate is omitted when there is none.

**Where.** `TD "…"` is a heading in `TODO_DEFERRED.md`, quoted far enough to grep; `T:NNN` is a line in
`TODO.md` **as of commit `d15a1c78`**, and goes stale with the first edit to that file; a path is a brief.

**What it was built from.** Five readers, one slice each, returning one line per item; merged here by hand.
Duplicates across the three sources are folded into one line citing both. Items that look done, stale or
superseded are left out of parts 1–3 and listed in part 4.

| source | entries | open | done? | stale? | dup? | tentative |
|---|---|---|---|---|---|---|
| `TODO_DEFERRED.md` | 188 | 140 | 9 | 9 | 5 | 23 |
| `TODO.md` | 225 (13 archived) | 140 | 29 | 10 | 30 | 3 |
| open briefs and sketches | 15 briefs, 4 sketches, 1 sprint of 3 | — | 2 | — | — | — |

The TODO.md duplicates are mostly against `TODO_DEFERRED.md` or a brief, so the two "open" counts overlap.
Merged, this page holds roughly 250 distinct open items.


## 1. The large pieces, and what gates what

The briefs are where the autumn's size lives. Each line is the brief as a whole; sub-lines are its separately
schedulable steps. Statuses are the briefs' own.

### Corpus and document pipeline

- **Per-document LLM pass** (`briefs/per-document-llm-pass-brief.md`) — a shared primitive running one question
  per item over thousands of documents: resumable JSONL ledger, instrument fingerprint, stop on backend failure,
  batching, push/pull progress, cancellation · gate `next` · **design decided** (six of seven questions
  2026-09-29, progress 2026-09-30) · ~M–L. Eight users wait on it.
  - Step 0: one model record in `llmclient`, read from LM Studio's `/api/v1/models` and shared with Librarian's
    label · "moderate", own commit · **ready to build**
  - The pass itself, sibling to `raven.librarian.agent`, exposing `maybe_abort` on `agent.turn` · ~M · ready
  - Then: port `investigations/aokk-corpus-scope/extract_fields.py`; then importer `_summarize` and
    `raven-pdf2bib`'s eight hand-rolled retry loops; title-per-document waits for brief 13
- **Corpus filter** (`briefs/corpus-filter-brief.md`) — generalize the AOKK scope-classification prototype into
  a `raven.papers` tool flagging off-topic records with reasons, for human review · ~L, plus "a day or two of
  GPU time, unattended" for the study re-run · **gated on the per-document pass** · seven questions, some
  decided 2026-09-30. **The AOKK methodology numbers wait on this**, so the chain is
  *per-document pass → corpus filter → AOKK numbers*. The chain is in both briefs and in neither TODO file.
- **Importer rework, brief 11** (`briefs/11_visualizer-importer-rework-brief.md`) — "after Yrityspäivä, and
  near-term" · ~L overall
  - Item 5: cluster once, in high-D (spec in `investigations/highdim-clustering/`) · 1–2 days · **specified and
    measured, ready**. The importer's remaining tests are to be written with it.
  - Item 1: the Nomic migration · ~M · fork decided 2026-09-30 (v1.5 unless something now offers both), and
    must re-measure Librarian's off-corpus threshold before the swap · **goes in during brief 13's build**
    (2026-10-01)
  - Item 4: Procrustes alignment for adding papers to a map · ~M · less urgent once scopes exist (brief 13).
    Absorbs `T:473`.
- **Derived artifact store, brief 12** (`briefs/12_derived-artifact-store-brief.md`) — one key shape and one
  regeneration/GC mechanism for everything computed from a source · "scheduled for v0.2.10" · ~L · six open
  questions O1–O6 to decide first; does not depend on 13
  - Core, including D1 (hybridir's inline full text to a seekable sidecar) · ~M–L
  - Producers one at a time — OCR, VLM description, vision embedding (waits on 11's fork), thumbnail · ~S each
  - Webfetch retrofit as a registered producer · "expected to be small"
- **Corpus scopes and the unified document DB, brief 13** (`briefs/13_corpus-scopes-and-unified-db-brief.md`)
  — scopes as tags, one DB behind both apps, a corpus TOC for the model, retiring the automatic search · "rough
  draft, unscheduled, but needed this year" · ~L+, and the brief warns it expands to fill its schedule
  - **A drydock build** (2026-10-01): several weeks with Raven not operable, so **a release is cut just before
    it starts**. The multimodal embedder goes in during it; the unified DB unlocks the Librarian↔Visualizer
    integration
  - **A design session first**, six-item agenda · ~S–M · not held
  - §4a: publish a scope TOC, measure whether the model searches on its own, then drop the automatic search
  - *Where the database lives*: behind Raven-server or not — reads together with server autostart
  - Attachment scope (search a chat's own attachments) · ~S–M · designed, the retriever already supports it
  - Absorbs `T:567` (unified DB), `T:751` (scopes), and the Visualizer↔Librarian selection-as-scope IPC
    (`T:424`, ~L)
- **Ligature repair** (`briefs/ligature-repair-brief.md`) — rebuild `fi`/`fl`/`ff` ligatures PDF extraction
  lost · recommendation: build the fixbib half (~S, the function plus a flag plus a report), not the indexer half
- **Spreadsheet ingestion** (`briefs/spreadsheet-ingestion-brief.md`) — `.xlsx`/`.ods` as Markdown tables for
  the docs DB and attachments · ~S–M · designed, not started; adds `openpyxl`. Absorbs `TD "Spreadsheets in the
  docs DB…"` and probably `T:479` (Excel import to BibTeX).
- **Bibliography dedup, residue** (`briefs/bibliography-dedup-brief.md`) — the tool shipped 2026-08-28; four
  small items stay in the brief on purpose: a shared HTML-entity decoder (S), an upstream `bibtexparser` writer
  report (S to report), the Visualizer importer learning the duplicate-field-key repair (~S), trimming
  `publisher_stopwords` after a real import (~S).

### Librarian

- **Librarian-extension sprint, 04–06** (`briefs/librarian-extension/`) — "open, and complete as a set …
  planned for scheduling in autumn 2026". Its only recorded ordering (05 first, on size) was set against the
  Night's deadline and no longer binds.
  - **04, MCP client** — external MCP tools join the built-ins in `perform_tool_calls`; `@tool` registry for
    built-ins, `raven.common.async_bridge`, transports, namespacing · ~M–L · gated on the Hindsight playground
    (06 steps 1–2). Absorbs `T:985` and the MCP toggle `T:956`.
  - **05, lorebook** — keyword-triggered injection from a watched directory, built on a new candidate-emitting
    context assembler; LM Studio plugin port as a second frontend · ~M native + ~M plugin · no upstream gate;
    defines the assembler interface 06 ranks on
  - **06, Hindsight** — stand it up in Docker, try it, go/no-go, then integrate two ways · ~L. Steps 1–3 touch
    no Raven code. Step 4 bundles the Nomic migration, assuming v1.5 — which the 2026-09-30 decision agrees with. Absorbs
    `T:765`, `T:767`, `T:769` (memory stores).
- **Block-level Markdown in the chat view** (`briefs/markdown-block-rendering-brief.md`) — fenced code,
  multi-line lists, paragraph gaps, later tables, by removing the per-line splitter · "next to be looked at after
  Yrityspäivä" · ~M; fenced code plus LaTeX would be "a multi-week build"
  - Step 2: remove the dead inline-`<think>` handling · ~S · ready (absorbs `TD "Remove the dead inline-`<think>`…"`)
  - Steps 3 + 7 together: stop splitting; a blank line becomes a paragraph gap · ~M · probe the stranded-`Pre`-box
    bug first
  - Step 4: dedent reasoning traces by common prefix · ~S
  - Step 8: tables · ~M · optional

### Robustness and the server

- **Surviving Raven-server going away** (`briefs/raven-server-availability-brief.md`) — groundwork landed
  2026-09-07 · ~M–L
  - Item 1: error handling at 37 unguarded call sites · ~M · two sites done
  - Item 2: Librarian boots before the server is up and connects when it appears · ~M · three server-side
    questions first
  - Item 3: the Visualizer declares its mode and survives losing the server · ~M
  - `mayberemote`: mode stays fixed at instantiation, declared loudly, "switch to local" a click · decided
- **Server autostart** (`briefs/server-autostart-brief.md`) — an app starts Raven-server if none answers; an
  app-started server exits when idle · ~M · **gated on availability item 2**; first check whether a
  venv-activated spawn gets `env.sh`'s CUDA paths

### Smaller briefs

- **FileDialog navigation history** (`briefs/filedialog-navigation-history-brief.md`) — Back/Forward, per-instance
  history, a FontAwesome toolbar · ~M · a one-minute probe for mouse X1/X2 first; absorbs two `TODO.md` fdialog items
- **Keyword pools** (`briefs/visualizer-keyword-pools-brief.md`) — rank cluster keywords by inverse
  cluster-frequency, guard the parser against prose replies, extract 15, add a dataset keyword dialog · ~M
  - IDF ranking with case unification, the parser guard, raising to 15 · ~S each · ready. That these need
    nothing from brief 13 is a reader's inference; the brief's gate names only the dialog.
  - The dataset keyword dialog · ~M · gated on brief 13. Absorbs `T:509` (common keywords in the GUI).
- **Image shapes in the xdot widget** (`briefs/xdot-image-shapes-brief.md`) — piece 3 landed; piece 1 (image
  store, about half a day) and piece 2 (parser `I` branch, S) open. Absorbs `T:787`.

### Design sketches (`briefs/design/`)

- **Interrogating a selection** — what Raven is for: map, select, run a first-pass reviewer, get keywords back to
  the map · ~L · needs 13, the per-document pass, and the constellation sketch's upload path. Partly overtaken:
  its scope key was superseded by 13's tags, its map stage now lives in the per-document pass.
- **How Raven's parts talk to each other** — the server as switchboard; clients pull; nobody listens on a port ·
  five open questions, two of which brief 13 and brief 04 §6 now bear on without saying so
- **The avatar as the interface** — an avatar-only mode for someone across the room · ~L · needs the
  constellation sketch and availability item 1. **The lab installation waits on it** (2026-10-01): an
  avatar-only toggle the operator can leave and re-enter with a key, and file objects hovering round the
  avatar as it reads
- **What kind of product Raven is** — a stance, no mechanism

### What gates what

Stated in the briefs unless marked *(inferred)*.

- **per-document pass** → corpus filter → AOKK numbers; → the interrogation sketch's map stage; → importer
  `_summarize` and `raven-pdf2bib` ports; → title-per-document (which also waits on 13)
- **the Nomic fork decision** (v1.5 image-text vs v2-moe multilingual) → brief 11 item 1 = brief 06 step 4 (the
  same migration, claimed by both; built during 13, 2026-10-01) → brief 12's vision-embedding producer → VLM reranking, images in the docs DB
  and the Visualizer, semantic grouping of the sidecar cleanup, re-measuring the `embeddings` VRAM *(the last
  four from `TD` items gated "post-Nomic")*
- **13** → keyword pools' dialog; → the corpus TOC and retiring the automatic search; → cross-corpus GC in 12; →
  the interrogation sketch; → backlog-as-dataset; ↔ server autostart (*Where the database lives*)
- **12** does not depend on 13; 12 → images in the DB
- **11 item 5** → the importer's clustering tests
- **availability item 2** → server autostart
- **the lab installation** (after 0.2.10, feature-gated) ← the avatar-only toggle and file-object polish (the
  avatar sketch), **04, the MCP client**, and the `cu130` move; the `cu130` move pulls in the CUDA half of
  easy install and, as a reader's inference, `pdm.lock`. Before or after 13: open
- **13** ← a release cut just before it, Raven being inoperable for the weeks it takes
- **06 steps 1–3** → 04 → 06's agentic path. 04 and 06 name each other as gates; 06's first three steps need no
  Raven code, which breaks the cycle *(the resolution is a reader's)*
- **05** (assembler interface) → 06 (assembler ranking)
- **Markdown step 2** → step 3; steps 3 and 7 together
- **the ooba upgrade** (`TD "Upgrade oobabooga…"`) → streaming thinking before `</think>`, Gemma's inline
  tool calls, re-testing ooba's continue (`T:191`), whether `chatutil.scrub`'s `<think>` repair is dead code
  (`T:687`). `T:158` wants this written up as a brief; it has not been.

Four decisions each unblock several of the above. Where they stand (maintainer, 2026-09-30):

- **The Nomic fork: multimodality wins for this year** if it must be one, so v1.5 — after checking whether
  anything now offers both (brief 11 item 1).
- **Brief 13's design session: wanted**, corpus scopes being a feature needed this year. Not yet held.
- **Brief 12's O1–O6: to be discussed in the near future.**
- **The ooba upgrade: not worth doing at the moment**; it and the four items behind it can safely wait.

Added 2026-10-01 (maintainer): **the lab installation comes after 0.2.10**, gated on features — the avatar-only
mode, the sci-fi file objects, the MCP client — and is the first install from zero, which dates the `cu130`
move. **Brief 13 is a drydock build** with a release cut just before it. Which of the two comes first is open.


## 2. Open items by theme

### 2.1 Corpus, retrieval and the document DB

*Librarian's document index*
- **Cite passages by printed page number** — carry PDF page labels through extraction and chunking to
  citations · M · gate: what `extract_text` returns · `TD "Cite a retrieved passage by the page number…"`
- **Ingest pool concurrency is nominal** — pypdf is pure Python, the GIL serializes 32 threads; use processes ·
  M · post-0.2.10 · `TD "The ingest pool's concurrency is nominal…"`
- **BM25 backend migration** — Tantivy (`TD "Hybridir: BM25 backend migration…"`) or ChromaDB FTS5 (`T:756`,
  Medium): one question, two candidate answers · ?
- **The document store is written whole, at the end, not atomically** · S–M · `TD "The RAG index's document
  store is written whole…"`
- **Full text and chunks both stored in the JSON** — about 2.6× source size; dissolves into brief 12's D1 plus a
  BM25 backend · ? · 0.2.10 · `TD "The docs DB stores each document's full text…"`
- **A crash during ingest loses the whole run** — cache extracted text content-addressed · ? · with the
  per-document pass · `TD "A crash during ingest loses the whole run…"`
- **Read-only index mode while an indexer writes**, then **hand the index lock over** · M, L · both tentative,
  may dissolve with the BM25 backend · `TD "Librarian should keep reading the document index…"`, `TD "Hand the
  index lock over…"`
- **Source code wants its own tokenizer** · ? · `TD "Source code in the document database…"`
- **HybridIR title field**, stored and weightable · ~M, costs a reindex · `T:753`
- **HybridIR pedigree field** · ~S · `T:979`
- **Adjustable similarity threshold** [High] · ~S · `T:706`. Brief 09 found no threshold carries across corpora,
  which undercuts it.
- **RAG PDF ingestion polish** — sanitize before indexing, OCR/captions · ~M · `T:704`
- **Are chunk full-IDs in the metadata for combined chunks?** [Verify] · ~S · `T:1022`
- **Document-level questions** ("which document is about X") · L · design · `T:718` — one mechanism with 13 and
  the per-document pass

*Formats (the `document-ingestion` cluster, gated post-0.2.10 together)*
- **Same formats in the docs DB and in attachments** — office done; images remain, Nomic-gated · `TD "Same file
  formats in the docs DB…"`
- **Text out of images** — SVG `<text>` first, raster via VLM · `TD "Text out of images…"`
- **Vector figures (`.svg`)** · `TD "Vector figures in the docs DB…"`
- **HTML pages whose content is produced by running them** — candidate for a brief · `TD "HTML pages whose
  content…"`
- **Read documents as page images** — a `read_pdf_page` tool · `TD "Read documents as page images…"`

*Visualizer's importer*
- **The importer reads the document DB, not just `.bib`** — the item says it deserves a brief · post-0.2.10,
  Nomic · `TD "Visualizer's importer should read the document database…"`
- **Import warnings visible in the GUI, and stored** · M · `TD "Import warnings should be visible…"`
- **HybridIR over BibTeX for full-text search** [High] · ~M · gate: full records saved in the dataset · `T:471`
- **Crashes silently on zero items** · ~S · `T:481`
- **Pre-filtering at import; incremental re-scan** · ~M · `T:483`
- **Configurable embedding fields, import plugin hook, stopwords** · ~M · `T:487`; then **general handling of
  missing fields** · ~M · `T:552`
- **More import sources** (Semantic Scholar, Scopus, ERIC) · ~M each · `T:492`
- **BibTeX export of the selection** · ~M · `T:477`
- **Report duplicate BibTeX keys** · ~S · `T:498`
- **De-duplicate words per abstract** before keyword extraction · ~S · `T:500`
- **LLM keyword detection refinements** — caching, progress, low-VRAM fallback · ~M · `T:565`
- **Configurable clustering hyperparameters** · ~S · `T:542`
- **Semantic grouping in the sidecar cleanup preview** · gate: Nomic · `TD "Semantic grouping in the sidecar
  cleanup preview…"`

*Papers tooling*
- **Reconcile a bibliography against the papers on disk** · L · `TD "Reconcile a bibliography…"`
- **Classifier pass over the arXiv stash** · ~S · `T:726` — may be what the corpus filter becomes
- **pdf2bib: canonical "no authors / no title"** · ~S · `T:1090`
- **pdf2bib overthinking / token-limit flag** · ~S · `T:1096` — the per-document pass's to solve, as is
  `TD "Batch tools: LLM reconnect mid-run"`
- **wosfile: vendor the fix or PR upstream** · ~S · `T:1124`

### 2.2 Librarian: backends, tools and context

*Backends and reasoning*
- **Upgrade oobabooga and re-check support** — gates four items (part 1) · ? · **decided: later** ·
  `TD "Upgrade oobabooga…"`; the cluster brief is `T:158`
- **Anthropic-style backends** [High] — top-level system field, tool blocks, SSE dialect · L · `T:936` (+`T:693`)
- **Containing the OpenAI wire shape** — a design sketch to write · ~S · `T:158`
- **Exact token count where neither tier reaches** · ~M · `T:205`
- **Resume an incomplete thinking trace on Continue** · ~M · probe LM Studio's parser first · `T:614` (+`T:848`).
  **Gates** `TD "Edit an AI reply's thinking trace…"`.
- **Streaming thinking shown as the answer until `</think>`** — mostly fixed, never observed live · M · ooba ·
  `TD "Streaming thinking is shown as the answer…"`
- **Idle prefill fires when the count is already exact** · ? · `TD "Idle prefill fires even when…"`
- **Abortable prefill; the turn-sequencing race** — an in-flight turn bleeding into a new chat · ? · `T:22`,
  `T:289`
- **Re-test `reasoning_effort` on Qwen 3.8** · S · needs the maintainer · `TD "Re-test whether `reasoning_effort`…"`
- **Qwen 3.6 `preserve_thinking` via LM Studio's per-model config** · ~S · `T:684`
- **Context meter counts reasoning on templates that keep it** (Gemma) · ~S · `T:667`
- **Model behaviour past its cutoff; deliberation cost per model** · ~S, probe exists · `T:1092`
- **Tell a wedged reply from a hard one** · M · tentative, may retire as models improve · `TD "Tell a wedged
  reply from a hard one…"`

*Tools*
- **`fetch_document` inlines up to 11× webfetch's threshold** — needs an addressing scheme · M · `TD "`fetch_document`
  inlines…"` (+`T:29`)
- **Read part of an attachment, and search a chat's attachments** — v1 of the fetched-page budget shipped in
  0.2.8 (verified 2026-10-01); this is its v2 · ? · brief 13 · `TD "Let the model read part of an attachment…"`
- **Tool-call budget for a multi-document read** [High] — re-run the phase F probe after 5 → 20 · ~S · `T:604`
- **webfetch allowlist: ship deny-by-default?** — a security-posture decision · 0.2.10 · `TD "Reconsider the
  webfetch allowlist default…"`
- **Relocate the "approve denied host" button**, then **batch-approve** · 0.2.10 · `TD "webfetch \"approve
  denied host\"…"`, `TD "webfetch: batch-approve…"`
- **webfetch per-site extraction** (Wikipedia, DOI, PubMed) · ~M · `T:962`
- **websearch: raw URLs [Verify], formatting, crawling to RAG** · ~M · `T:977`
- **webfetch local mode** · tentative, needs a clean-room driver · `TD "webfetch local (client-side) mode"`
- **Multi-step `calculate`** · M · `TD "The `calculate` tool takes one expression…"`
- **Weather and calendar tools** · ~S each · `T:958`, `T:960`
- **Attach from a URL** · 0.2.10 · `TD "No way for the user to attach a document from a URL"`
- **Attach a document already in the docs DB** — whose affordance · ~M · `T:708`
- **Agent skills** over the document DB · ? · post-0.2.10 · `TD "Agent skills for Librarian…"`
- **The AI drives the constellation's views** · post-0.2.10, after the shared corpus · `TD "Let the AI drive…"`
- **The automatic RAG search reads to the model as its own mistake** · research · `TD "The automatic RAG search
  reads…"` — may be mooted by 13 §4a

*Context, prompt and citations*
- **Context-window budgeting and compaction** [High] · L · `TD "Context-window budgeting…"` (+`T:758`)
- **Show the raw prompt** [High], then **save the full prompt per AI message** · ~M each · `T:778` (+`T:21`), `T:839`
- **The system-prompt trio** — per-turn placeholders, an optional greeting, a modernized card — "one question,
  three answers", brief-shaped · M each · post-0.2.10 · `TD "System prompt templating…"`, `TD "Make the canned
  AI greeting optional"`, `TD "Modernize the Librarian system prompt…"`
- **Proactive context engineering** (topic graph) · L · `T:749`
- **Expose the sources behind a reply's RAG citations** — tool side done 2026-09-29, auto-search side missing ·
  `TD "Expose the docs-DB source files…"` (+`T:981`)
- **Inline citations with validation** [High] · ~M · `T:983` (+`T:711`)
- **User persona sampling / prefill** [Parked] · `T:998`

### 2.3 Librarian: chat UI and navigation

*Chat graph*
- **Metrics readout** (Ctrl+Shift+M) · M · `TD "A metrics readout for the chat graph…"`
- **Bookmarks** · M · `TD "Bookmarks in the chat graph"`
- **Undo history for HEAD** · M · `TD "Nothing remembers which sibling…"` (+`T:773`)
- **Switch HEAD by node ID** · ~S · `T:833` (+`T:815`); copying an ID landed 2026-09-30
- **Only the configured character wears its own icon** · S · `TD "Only the configured character wears…"`
- **Let the system prompt be hidden** · M · `TD "Let the system prompt be hidden"`
- **Thin strokes go faint at some zooms** · S? · `TD "Thin strokes in the chat graph's labels…"`
- **A couple of blank frames before the first picture** · S? · tentative · `TD "The chat graph shows a couple of
  blank frames…"`

*Chat log and keyboard*
- **Nothing owns which pane has the keyboard** · ? · needs a design · `TD "Nothing owns \"which pane has the
  keyboard\"…"`
- **Ctrl+Left/Right cannot flick between siblings** · S to build, the design is the work · `TD "Ctrl+Left /
  Ctrl+Right cannot flick…"` (+`T:31`)
- **A sibling switch rebuilds on the callback thread, eating keys** · M · `TD "Switching a chat sibling
  rebuilds…"`
- **Wheel-up does not always release the end-latch** · ? · `TD "Scrolling up with the wheel…"`
- **A held scrollbar drifts while a reply streams** · ? · 0.2.10 · `TD "Holding the chat view's scrollbar…"`
- **Double-buffered rebuild; no auto-scroll after the user scrolls** · ~M · `T:835`
- **`replace_last_paragraph`'s mutex is disabled because it hangs** · ? · `TD "`replace_last_paragraph`'s
  `dpg.mutex()`…"`
- **Disable action buttons while the AI writes** — delete and edit already refuse · ~M · `T:846`
- **Per-message role backgrounds** · ~M · `T:859`

*Attachments and datastores*
- **Context meter reads ~1% before attachments are extracted** · S · `TD "The context-fill meter reads ~1%…"`
  (+`T:26`)
- **Attachment state is carried by colour and hover alone** · ? · `TD "Attachment state is carried by colour…"`
- **A clickable chip gives no hover cue** · ? · `TD "A clickable chip in the chat log…"`
- **Browse all attachments, not just the orphans** · post-0.2.10 · `TD "Browse *all* attachments…"`
- **Open a chat datastore other than the default** · post-0.2.10 · `TD "Librarian: open a chat datastore…"`
- **Recent chats list** · L · design · `T:808`
- **minichat: `null` retrieval results in `chat.json`** [Verify] · ~S · `T:864`

*Panel and character*
- **No-avatar mode, the chat tree in the vacated panel** — plus design notes for a speech-only mode · L ·
  `TD "A no-avatar mode…"` (+`T:1003`)
- **Switch the AI character and user profile at runtime** · M · pairs with server-down robustness · `TD "Switch
  the AI character…"`
- **Bilingual display / translating Finnish input** · ~L · UX prototypes · `T:841`

### 2.4 The Markdown renderer

Brief in part 1. The rest of the `markdown-renderer` cluster:

- **No "finished" signal** — so the chat view stops counting settle frames · S · 0.2.10 · `TD "The vendored
  Markdown renderer has no way to say…"`
- **Block constructs need a block container** — stranded code boxes, segmented blockquote bars · ? ·
  `TD "DearPyGui_Markdown block constructs…"`
- **Decorations placed by a premature measurement** · ? · `TD "Markdown decorations are placed by measuring…"`
- **Drops text** — mitigated by the startup atlas refresh, cause open · ? · `TD "The Markdown renderer drops
  text…"` (+`T:264`)
- **The atlas refresh flashes at startup** · S–M · `TD "The font atlas refresh flashes…"`
- **A wrapped line keeps the space it wrapped at** · ? · `TD "A wrapped line in the Markdown renderer…"`
- **LaTeX equations** · ? · 0.2.10 · `TD "Rendering LaTeX equations…"`
- **Emoji** · ? · `TD "Emoji support in the Markdown renderer…"`
- **A font with subscripts and symbol coverage** · M · `TD "Find a UI font that renders subscripts…"`
  (+`TD "Super/subscript font coverage in the GUI"`)
- **Promote the whole cluster to a brief** · 1–2 h · `T:270`. The block-rendering brief covers part of it.

### 2.5 Robustness: shutdown, server, data safety

Briefs in part 1 (server availability, autostart).

- **Error-reporting sweep, so failures reach the user** — what remains of the no-model item · L ·
  `TD "Librarian doesn't check that the LLM backend has a model loaded"`
- **Shared two-phase DPG shutdown helper, and an audit** — the `abnormal-exit` cluster's brief-to-be; its
  per-app table dates from June · ? · 0.2.10 · `TD "Fleet-wide: shared two-phase DPG shutdown helper…"`
- **`quitsignal.install` in the other five apps** · ~S · 0.2.10 · `TD "Librarian leaks its server-side avatar
  instance…"`
- **`is_dearpygui_running()` is not a guard against `destroy_context`** · S narrowing, M correct · `TD
  "`is_dearpygui_running()` is not a safe guard…"`
- **Version the chat datastore file** · 0.2.10 · `TD "Version the chat datastore file…"`
- **Datastore scaling** — one `chat.json` and a flat sidecar directory · post-0.2.10 · `TD "Datastore scaling…"`
- **Migrate the 38 bare `dpg.split_frame()` calls** · ? · `TD "Migrate the remaining `dpg.split_frame()`
  sites…"`
- **`recenter_window`'s degrade-instead-of-raise policy** · post-0.2.10 · `TD "Revisit `recenter_window`'s…"`
- **Don't crash when the `tts` module isn't running** · ~S · `T:1020`
- **The Visualizer's crash recovery file** · ~M · `T:583`
- **A `discard()` hook for dropped animations** · S · tentative, belt-and-braces · `TD "A `discard()` hook…"`
- **The suite hung once on Windows CI** · ? · waiting for a second occurrence · `TD "The test suite hung once
  on Windows CI…"`

### 2.6 Avatar and speech

*Avatar*
- **Hold the video until the answer is complete**, for one-GPU setups · S · `TD "An option to hold the
  avatar's video off…"`
- **Start synthesizing speech while the reply streams** · L · `TD "Start synthesizing speech while…"`
- **Is `target_fps = 20` still needed?** · S · `TD "Is Librarian's `target_fps = 20` still needed…"`
- **Stutter while receiving a response**, and at first speech · ~M · `T:1007` (+`T:330`)
- **The data eyes nearly vanish after the postprocessor** · ? · `TD "The data eyes are nearly invisible…"`
- **The subtitle translator drops `=`** · ? · `TD "The subtitle translator silently drops `=`…"`
- **TTS reads arXiv IDs digit by digit** · post-0.2.10 · `TD "TTS reads arXiv IDs…"`
- **Long subtitle splitter** · ~M · 0.2.10 · `T:916`
- **A better emotion classifier, judged at reading speed** · M · `TD "A better emotion classifier…"`
- **A transient effect cannot ease in or out** · ? · gate: where the envelope lives · `TD "A transient
  postprocessor effect…"`
- **The settings editor rewrites hand-ordered postprocessor chains** · L · `TD "Avatar settings editor: custom
  postprocessor chain ordering"`
- **The pose editor sometimes opens at its creation size** · ? · waiting for a specimen · `TD "The pose editor
  sometimes opens…"`
- **Backdrop onto `fit_cover`** · ? · 0.2.10, early · `TD "Move the avatar backdrop onto…"`
- **Isolate the DPG-specific parts of the renderer/controller** · ~M · `T:1009`
- **Re-measure VRAM with `crt` in the imagefx chain** · ~S · `T:299`
- **Client-local BSD animator** — a maintainer decision; pose-editor remote mode separable · `TD "Client-local
  avatar animator…"`
- **Art and distribution**: per-character chat icons (~S, `T:1011`), eye-waver cels (~M, `T:1075`), zipped
  characters (~M, `T:1061`), a JS client for SillyTavern (L, `T:1073`), vector emotions ([Parked], `T:1013`)

*Speech*
- **Wake-word trigger** [High] · L · `T:884`
- **STT language selector, and showing the detected language** · ~M · `T:873`
- **Proper names from NER into the STT prompt** · ~S · `T:912`
- **`raven-transcribe` CLI** · ~S · `T:910`
- **Voice commands** · ~S · `T:914`; **edit a spoken message before sending** · ? · `T:925`
- **STT: spurious text after long audio** · ~S · `T:929` (+`T:1059`)
- **Quantized whisper-large-v3-turbo** · ? · `T:927`
- **Translator upgrade to HPLT v2** [Parked] · `T:989`

`T:884`, `T:873` and `T:912` carry a note to design the STT items together.

### 2.7 Visualizer

*Search and the info panel*
- **Author search, with the full author list** [High] · ~M · `T:450` (+`T:511`)
- **DOI: importer, info panel, open button, export** [High] · ~M · `T:452` (+`T:525`, which adds a same-author
  search button)
- **Fragment search across several fields** · ~M · `T:454`
- **Semantic orienteering** — typed text as a virtual point · ~M · `T:456`
- **Select a cluster by number** · ~S · `T:458`
- **Word-boundary mark in search** · ~S · `T:462`; **nested highlight breaks** · ? · `T:464`
- **Configurable tooltip and info panel fields** · ~M · `T:519`; **panel left or right** · ~S · `T:521`;
  **show the BibTeX slug** · ~S · `T:523`; **a brighter "Search" heading** · ~S · `T:527`
- **Info panel is O(n²) through the Markdown renderer** · ~M · `T:592`

*Selection, views and analysis*
- **Colouring by cluster, year, source file** · ~M · `T:505`
- **A full report of the selection** · ~M · `T:507`; **AI summarize the selection** · ~M · `T:563`
- **Selection history; save and load a selection; filter ↔ selection; live filtering** · `T:536`, `T:538`,
  `T:550`, `T:546`
- **BibTeX entry types: show, count, filter** · ~M · `T:513`
- **Time granularity beyond years** · ~L · `T:489`
- **Comparative analysis** · ~L · `T:529`; **image support** · L · Nomic · `T:531`
- **Semantic map mouse interaction wants a UX pass** · ? · `TD "The semantic map's mouse interaction…"`
- **The annotation tooltip build makes pulsation choppy** · ? · `TD "Annotation tooltip build makes…"`

*Windows and settings*
- **Word cloud window** — resizable, 1:1, Lanczos, schemes · ~M · `T:515`; drawn under the toolbutton highlight ·
  `T:590`
- **A settings window over `gui_config`** · ~M · `T:517` (+`T:789`, the constellation's settings dialogs)
- **Import dialog: a multi-column file table** · ~S · `T:540`
- **What a dropped dataset means** · ~S · decision · `T:544`
- **All colours configurable** · ~L · `T:548`

*Structure and data*
- **Replace the `.pickle` format** · ~M · `T:494`
- **Finish the `app.py` refactor** · ~M · `T:439`; **FP refactor** · ~L · tentative · `T:441`
- **Publish a quick-start dataset** [High] · ~S · `T:469`

*Parked*: highlight as an outline (`T:554`), spaCy for Finnish (`T:556`), keyword detection alternatives
(`T:558`).

### 2.8 Shared GUI, keyboard, and the smaller apps

*Keyboard and accessibility*
- **Sliders have no keyboard story** · M · a design decision · `TD "Sliders have no keyboard story…"`
- **The main keyboard offers no zoom** · S once keys are chosen · `TD "The main keyboard offers no zoom…"`
- **Layout-aware positional hotkeys** · ? · `TD "Keyboard-layout-aware positional hotkeys…"`
- ~~**Every hotkey in a tooltip and on its card**~~ — **done**: every app read through and signed off by
  2026-09-14, the cards swept the same day. Both items removed 2026-10-01; `check_hotkey_tooltips.py` holds
  the line
- **Filter/search in a help card's hotkey list** · ~M · `T:460`
- **Colourblind-safe ok/error flashes** · 0.2.10 · `TD "Colorblind-safe status signaling…"`
- **Flash the search field when a hotkey focuses it** · ~S · `T:407` — the keyboard marks may have made it moot

*Styling constants* — one job across three entries: a DPG-free constants module, then the sweep.
- **Sweep the GUI styling constants into one module** · ~M · `T:416`, **gate lifted** (see part 4)
- **The 8/3 pass** · `TD "The 8/3 pass…"`; **hardcoded stand-ins for DPG theme values** · `TD "GUI: hardcoded
  stand-ins…"`; **the flash palette** · `TD "Consolidate the flash palette…"`
- **The global theme sets three of seven rounding vars** · S · `TD "Raven's global theme sets…"`
- **Toolbar buttons: WidgetFlash acknowledgement** · `TD "Audit toolbar buttons…"`

*Widgets*
- **A multiline text control of our own** — word wrap, Escape · L · nothing waits on it · `TD "A multiline text
  control…"`; subsumes **a resizable composer** · `TD "Make the Librarian chat composer…"`
- **The thumbnail grid materializes every tile** · M · `TD "The shared thumbnail grid materializes…"`; **its
  textures are dynamic** · S · `TD "The thumbnail grid's textures are dynamic…"`
- **VU meters blank for a frame or two** · S · not reproduced on demand · `TD "VU meters occasionally blank…"`

*FileDialog* (navigation history brief in part 1)
- **Split `raven/vendor/` into `vendor/` + `forks/`** · ~S mechanical · `T:409`, **gate lifted**
- **A recursive search mode** · ? · a UX decision · `TD "FileDialog: a recursive search mode"`
- **A file-type icon set of our own** · M · `TD "A file-type icon set of our own…"`
- **Is `fdialog` one unit?** · tentative · `TD "Is `fdialog` one logical unit…"`
- **Upstream PR of the fixes** · ~M · decide scope · `T:1107`

*Cherrypick* — six performance items that cross-reference each other and would want measuring together:
- **Low FPS with large images** · `TD "raven-cherrypick: low FPS with large images"`
- **16 MP preload optimization** · `TD "Preload cache: 16MP image optimization"`
- **Idle CPU/GPU load** · `TD "raven-cherrypick: further reduce idle CPU/GPU load"`
- **Faster PNG decoder** · `TD "Faster PNG decoder"`; **pillow-simd** (tentative) · `TD "pillow-simd…"`
- **Zoom-in doesn't upgrade cached neighbours** · ~S · `TD "Cherrypick: zoom-in doesn't upgrade…"`
- And, not performance: **crown the winner without leaving compare** · S · `TD "Cherrypick: crown the
  winner…"`; **export an image sequence (QOI→PNG)** · `TD "raven-cherrypick: export image sequence…"`;
  **smooth pan and zoom**, a call to make by trying it · L · tentative · `TD "Smooth pan and zoom in
  Cherrypick's…"`

*XDot viewer* (image shapes brief in part 1)
- **An actual-size button, as the chat graph has** · S · a decision · `TD "`raven-xdot-viewer` has no
  actual-size button…"`
- **A pan/zoom animation switch on `XDotWidget`** · ~S · `T:119`

### 2.9 Platform, packaging and release

- **Server config variants by VRAM tier**, with an auto-tiering or conservative default [High] · ~M · gate: the
  default-policy decision · `T:1031`
- **Uniform load-on-demand for server modules** · ? · measure PCIe load times first · `TD "Uniform load-on-demand…"`
- **A web status panel for long jobs** · post-0.2.10 · `TD "Web status panel…"`
- **Check for a local model before the HF Hub** · ~M · `T:1050`; **document the HF Hub env vars** [High] · ~S ·
  `T:398`; **model update UX** · `T:1057`
- **Move the torch trio to `cu130`, and with it to torch 2.14** — `cu128` stops at torch 2.11; the `[cuda]` extra
  goes to CUDA 13 too · M · before the lab installation · `TD "Move the torch trio to CUDA 13…"` (added 2026-10-01)
- **huggingface-hub 2.x** · S · waits on transformers, sentence-transformers and tokenizers lifting their caps ·
  `TD "`huggingface-hub` 2.x…"` (added 2026-10-01)
- **Replace torchaudio's resample, drop torchaudio** — low priority: torchaudio no longer pins torch (measured
  2026-10-01) · S · `TD "Replace
  `torchaudio.functional.resample`…"`
- **GPU-accelerated install on any OS and GPU, without editing `pyproject.toml`** (CPU only as the fallback for
  a machine with no compatible GPU; renamed 2026-10-01) · re-scope against the `cu130` move, which may settle
  the CUDA half · the lab installation · `TD "GPU-accelerated install on any OS and GPU…"`
- **`pdm.lock` is gitignored, against fleet policy** · M · `next`; the lab installation is the first install
  where it would pay · `TD "`pdm.lock` is gitignored…"`
- **Audit the wheel's contents** · S · `TD "Audit what the built wheel actually contains"`
- **The distribution rename to `raven-lab`**, and whether there is a PyPI upload at all · S · tentative ·
  `TD "Rename the distribution to `raven-lab`…"` (+`T:45`)
- **Licensing**: the top-level README and `LICENSE.md` still say bare BSD, and the FontAwesome header lacks its
  notice — the rest of both items is done · `TD "The licensing story is accurate only…"`, `TD "Two adopted
  directories ship…"`
- **Extract `raven.common` as `corvid-lab`** · post-0.2.10 · `TD "Extract `raven.common` into an upstream
  library…"`
- **GPU backends**: MPS device sync (`TD "MPS (Apple Silicon) device synchronization"`), a ROCm audit
  (`TD "AMD GPU (ROCm) support audit"`)
- **macOS**: Cmd for hotkeys, conflicts with builtins, one-button right-click, F-keys · `T:574`–`T:577`
- **`deviceinfo`'s bootup report says client or server** · ~S · `T:430`
- **The Kokoro/misaki Python cap** — probe CTC alignment before it is needed · Oct 2028 · `T:1111`
- **Split `nlptools` per backend** (tentative), **consolidate image conversions** (tentative) · `TD "Split
  `raven.common.nlptools`…"`, `TD "Consolidate remaining numpy/tensor/DPG image conversions"`

### 2.10 Tests, hygiene and docs

*Tests and checkers*
- **The flake8 → ruff move dropped indentation checks**, and **assert the linter runs what we rely on** — a pair
  · S each · 0.2.10 · `TD "The `flake8` → `ruff` migration…"`, `TD "Assert the linter actually runs…"`
- **`chat_controller` is not importable without spaCy** · S · 0.2.10 · `TD "`chat_controller` is not
  importable…"`
- **Documented command lines are unchecked** · M · `TD "Documented command lines are unchecked"`
- **Tests for the other `scripts/` checkers** · M · `TD "Tests for the rest of the `scripts/` checkers"`
- **Modules worth testing** — `pdf2bib` at 0% the easiest · L · `TD "Modules worth testing…"`
- **Negative controls for vacuous-pass assertions** · L · `TD "Give the assertions that could pass vacuously…"`
- **A robust public-API auditing tool**, general or none · `TD "Robust public API auditing tool"`
- **`systemd-coredump` truncates heavy cores** — machine setup, not a repo change · `TD "`systemd-coredump`
  truncates…"`

*The hygiene sweep* — tagged `hygiene-sweep` across both files, a ready-made dehydration pass:
- **Name the lambdas** · S · `TD "Anonymous lambdas where a named callable exists"` (+`TD "Audit unnamed
  lambdas"`)
- **Quotes onto double** · mechanical · `TD "String quotes: sweep the tree…"`. It suggests `ruff format`, which
  the fleet's no-formatter policy rules out; the tool needs choosing.
- **Startup `print()` → `logger.info()`** · `TD "Convert startup `print()`s…"`
- **Library modules stop configuring the logger; a detailed-debug level** [High] · ~M · `T:405`
- **`frozendict` constants**, **the typing audit**, **dotted imports** (tentative), **`raven/common/__init__`
  re-exports** (tentative) · `TD "Audit fleet for dict constants…"`, `TD "Audit typing…"`, `TD "Adopt dotted
  import style…"`, `TD "Should `raven/common/__init__.py` re-export…"`
- **Rename `vis_data` → `entries`** · ~S · `T:422`
- **Bundle `ai_turn`'s callbacks** · mechanical · `TD "scaffold: collect `ai_turn`'s callbacks…"`
- **Docstrings that describe a previous version** — about 121 markers · L · `TD "Docstrings that describe a
  previous version…"`
- **Large blobs stored separately from the datastores** · gate: blob support · `T:428`

*Docs*
- **A full README pass for stale claims** · M · `TD "A full README pass…"`; **the main README is a god
  document** · L · `TD "The main README is becoming a god document"`
- **Screenshots and clips left from 0.2.9** · M · a session where taking the keyboard is expected ·
  `TD "Screenshots and clips left over…"`
- **The dev-facing file dialog manual** · M · `TD "The dev-facing file dialog manual"`
- **An override cannot set a setting to `None`** — a bug, filed here for want of a better home · M ·
  `TD "An override cannot set a setting to `None`…"`
- **A Raven technical report on arXiv** · L · a CS endorser · `T:1126`
- **LM Studio honours `min_p`** — a note more than a task · `T:702`

*Process* — `TD "TODO.md goes stale…"` (a periodic visit, hooked into releases), `TD "Audit and slim down
project `CLAUDE.md`"`, `TD "Does `CLAUDE.md`'s DPG pitfall index…"`, `TD "Sweep `## Declined`…"`.


## 3. Small and ungated

A filter, not a ranking: open items costed S or ~S with no gate, from all of the above. Most costs here are
readers' estimates.

- **Corpus**: brief 11 item 5 (1–2 days, ready); Markdown brief step 2; the ligature fixbib half; the dedup
  entity decoder; keyword pools' IDF ranking, parser guard and count of 15; the importer's zero-item crash
  (`T:481`); duplicate-key report (`T:498`)
- **Librarian**: the ~1% context meter; the configured-character icon; thin strokes in graph labels; switch HEAD
  by ID (`T:833`); the tool-call budget re-run (`T:604`); weather and calendar tools
- **Robustness**: `quitsignal.install` in five apps; don't crash without the `tts` module (`T:1020`)
- **Avatar**: `target_fps`; the hold-video option; re-measuring VRAM with `crt`
- **GUI**: the rounding vars; static thumbnail textures; the `vendor/` → `forks/` split; the XDot pan/zoom
  switch; Cherrypick's crown-in-compare and zoom-in neighbours
- **Visualizer**: select a cluster by number; the BibTeX slug; the info panel side; the quick-start dataset
- **Release**: drop torchaudio; the wheel audit; the ruff indentation pair; `chat_controller` without spaCy;
  the HF env-var docs; the `raven-lab` rename, if an upload happens


## 4. Input for the TODO triage

Everything below was left out of parts 1–3. Each claim is a reader's unless marked otherwise, with its evidence
as the reader gave it. `verified` means the code was looked at; `unverified` means the text suggested it.

### Looks done

*`TODO_DEFERRED.md`*
- `"Librarian has no periodic autosave…"` — verified, `appstate.start_autosave`
- `"Two loose ends on what the data eyes mean"` — verified, `llmtools.EXTERNAL_SOURCE_TOOL_NAMES`
- `"raven.papers user manual"` — verified, `raven/papers/README.md`
- `"OS file drag-and-drop is not advertised anywhere"` — the README half verified; the in-app cue unchecked
- `"Read the chat graph's changelog block back…"` — unverified; the item calls itself "probably closeable"
- `"A HEAD change extracts every attached document synchronously…"` — the stall is gone; the remainder is brief 12's
- `"Widening the window feeds the chat log…"` — **per the Yrityspäivä README, 2026-09-30**: extra width now
  goes to the chat graph. Not checked in the code.
- `"The pose editor's list browsing does nothing…"` — **possibly**, by the avatar editors' keyboard work of
  2026-09-30 ("keys for every chooser, Esc out of one"). Unverified.
- `"Triage CLAUDE.md style conventions…"` — unverified; the fleet-wide rules appear to have moved

*`TODO.md`* — `continue_` on LM Studio (`T:691`, `T:187`); the thinking-trace UI (`T:290`, `T:853`); the context
fill meter (`T:771`); smooth scrolling (`T:1018`); `open_document_store` adoption (`T:724`); indexing progress
(`T:804`); the minichat deprecation note (`T:713`); `raven-docdb-import`, which is `raven-indexer` (`T:1088`);
csv2bib docs (`T:1098`); the avatar editors' help cards (`T:1071`); the per-module VRAM budget (`T:1029`); the
reranker, measured and rejected (`T:402`); RAG via a tool call (`T:248`, `T:972`); the calculator (`T:958`,
the weather half open); the scriptable scaffold, now `agent.py` (`T:728`); the graph transition animation
(`T:52`, `T:77`, `T:136`); search in both halves (`T:78`, `T:142`); message editing (`T:144`, `T:120`);
expression follows speech (`T:75`, `T:134`); the help card sweep (`T:82`, unverified); the glitch on branch
switch (`T:1005`); STT silence level and autostop (`T:869`); umlauts and braces (`T:485`, one remark about
unifying three implementations open).

*Briefs* — `bibliography-dedup-brief.md` and `visualizer-test-coverage-brief.md` are complete apart from
residue they hand elsewhere; candidates for `done/` once it has a home. Of the August triage artifacts,
`todo-triage-input-2026-08-12.md` and `triage-followup-edits.md` are fully applied, and
`todo-triage-output-2026-08.md` is applied except §5 (a held `CLAUDE.md` growth check) and §10 (about 107
confirmed items never ranked, which this triage would answer).

### Stale

*Gates naming Researchers' Night*, now past — re-gate each: the wheel-up end-latch; sibling flicking; the
file-type icon set; the webfetch allowlist's demo half; the clickable chip; Markdown decorations; the glyph
drop; the optional greeting; the no-avatar mode's "post-RN triage".

*Superseded by a brief* — **keep, as pointers**: a TODO scan is how open briefs get found (maintainer,
2026-09-30). At most trim each to a line naming its brief: `TD "Fenced code block…"`, `TD "Markdown tables don't
render…"`, `TD "Reasoning traces with indented bullets…"`, `TD "Remove the dead inline-`<think>`…"` (all
`markdown-block-rendering`); `TD "Ligature mojibake…"` (ligature repair); `TD "Batch tools: LLM reconnect
mid-run"` (per-document pass).

*No longer a task* — `TD "Remaining server modules without a MaybeRemote"` (navigational note) — **removed 2026-10-01**, the note moved into `mayberemote`'s docstring;
`TD "CLAUDE.md: rephrase DPG pitfall #5…"` (the numbering has moved); `TD "Two things a triage pass should
know"` (guidance, not an item — belongs in the file's header); `TD "Whether a short chat graph should sit at the
top…"` (a settled-by-looking note).

*`TODO.md`*
- **The session-plan blocks, `T:9`–`T:392`, are overtaken almost entirely.** Seven live items need moving out
  first: the ooba and wire-shape briefs (`T:158`), the exact token count (`T:205`), the Markdown cluster brief
  (`T:270`), re-measuring VRAM with `crt` (`T:299`), the `raven-lab` note (`T:45`), the hotkey-tooltip audit
  (`T:91`), the XDot pan/zoom switch (`T:119`).
- The Autumn section is mostly a measurement record (VRAM tables, MoE vs dense), which belongs in
  `investigations/` or `briefs/reference/model-lineup-autumn-2026.md`.
- The thinking-toggle item (`T:614`–`T:689`) is about 75 lines of probe findings around one open task.
- Individual items: `T:496` (user dir; overrides landed 2026-09-11); `T:585` (DPG 1.x regressions, now 2.3.1);
  `T:594` (fdialog Ctrl+F, since the keyboard brief closed); `T:747` (parse think blocks, since reasoning
  moved out of band); `T:1105` ("unit tests very sparse"); `T:578` (OS X 10.x).
- `T:908`, the Finnish demo path — the Night answered it with an English-only instruction, and Yrityspäivä also
  runs in English.

*Headings or premises out of date* — `TD "Librarian doesn't check that the LLM backend has a model loaded"`
(the titular case shipped); `TD "Librarian's help card: the room exists now…"` (**removed 2026-10-01**: every
card was swept on 2026-09-14, so the item was done rather than stale); `TD "Datastore scaling…"` (assumes
save-at-exit; **fixed 2026-10-01**); `TD "Documented command lines are unchecked"` (its slow-tool note is
fixed; **updated 2026-10-01**); `TD "Only the configured character wears…"` (the code moved to `chattextures`).

*Briefs out of date* — brief 12 says v0.2.9 in its body and v0.2.10 in its status, and does not record that its
chat-store renames shipped; `librarian-extension/README.md` still says "09 is the one in progress" and lists
residents that have moved; the per-document pass's closing "must settle" list
repeats questions its *Decided* sections answer.

### Duplicates to merge

Each pair is folded into one line in part 2; the triage decides which copy survives.

- `TD` ↔ `TODO.md`: context meter ~1% (`T:26`); `fetch_document` (`T:29`); sibling flick (`T:31`); compaction
  (`T:758`); citation sources (`T:981`); HEAD undo (`T:773`); the no-avatar mode (`T:1003`); the glyph drop
  (`T:264`); the hotkey audit (`T:91`)
- within `TODO.md`: Anthropic backend (`T:693` / `T:936`); author display (`T:450` / `T:511`); DOI
  (`T:452` / `T:525`); citations (`T:711` / `T:983`); memory stores (`T:765` / `T:767` / `T:769`); thinking resume
  (`T:614` / `T:848`); stutter (`T:330` / `T:1007`); STT (`T:929` / `T:1059`); raw prompt (`T:21` / `T:778`)
- within `TD`: the two lambda items; the two subscript-font items; the 8/3 pass with hardcoded stand-ins
- against a brief: `T:473` (11), `T:567` and `T:751` (13), `T:787` (image shapes), `T:985` (04), `T:479` and the
  spreadsheet `TD` item (spreadsheet ingestion)
- the Nomic migration, claimed by briefs 11, 06 and 12

### Gates that lifted unnoticed

- `T:409`, the `vendor/` → `forks/` split, and `T:416`, the styling-constants sweep, both waited on the
  FileDialog keyboard brief, which is in `briefs/done/`.

### Tier drift

Several [High] items have sat for months — the HF env-var docs, the quick-start dataset, author search, DOI,
the similarity threshold, config variants — while "unit tests [High]" is stale. The tiers look unread since the
Night.

### Clusters nobody has tagged

Each spans several items and would read better as one piece, several as a brief: **ooba** (part 1); **Nomic
and the shared corpus**; **docs-DB durability** (atomic save, bloat, the ingest cache, the BM25 backend — all
hinge on brief 12 and the backend choice); **Librarian responsiveness** (the ~1% meter, the HEAD-change
extraction, sibling switches eating keys, the ingest pool); **keyboard and accessibility**, spread across four
cluster names; **the system-prompt trio**; **styling constants**; **Cherrypick performance**; **chat-graph
follow-ups** (bookmarks, HEAD undo, the xdot 1:1, panel occupancy); **the tool surface** (page images, fetched
page budgeting, compaction, citation sources).

### Structural damage in `TODO_DEFERRED.md`

- `"FileDialog: a recursive search mode"` carries another item's body after its first paragraph (a 2026-08-18
  note about the path field's nav flash).
- `"The chat graph shows a couple of blank frames…"` ends with an unrelated paragraph (a trigger for mapping
  the backlog).
- `"Two adopted directories ship without their licence text"` has an orphaned cluster list dated 2026-07-27
  stranded inside it.
- A paragraph about pygame's `pkg_resources` warning sits between two items with no heading.
- Three items — `"A `discard()` hook…"`, `"A better emotion classifier…"`, `"Edit an AI reply's thinking
  trace…"` — are filed after `## Declined` and `## Waiting on upstream`.

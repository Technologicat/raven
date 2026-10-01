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

## B. `TODO_DEFERRED.md` items gated *into* 0.2.10

| item | proposal | why |
|---|---|---|
| Rename the distribution to `raven-lab` | gate → "the first PyPI upload" | whether there is one is open |
| Audit what the built wheel contains | gate → "the first PyPI upload" | same |
| The licensing story (`LICENSE.md`, README section) | keep 0.2.10, decoupled from PyPI | the docs are wrong today, upload or not *(inferred)* |
| flake8 → ruff dropped indentation checks | keep 0.2.10 | S |
| Assert the linter runs the rules we rely on | keep 0.2.10 | S, pairs with the above |
| Replace torchaudio's resample, drop torchaudio | ungate (was 0.2.10) | S; no longer unpins torch — measured 2026-10-01, so nothing forces it |
| `chat_controller` not importable without spaCy | keep 0.2.10 | S; brings its tests into CI |
| `quitsignal.install` in the other five apps (the leaked avatar item) | keep 0.2.10 | ~S |
| Shared two-phase DPG shutdown helper | keep 0.2.10 | pairs with the above |
| Version the chat datastore file | keep 0.2.10 | small, and cheaper before more migrations *(inferred)* |
| A clickable chip gives no hover cue | 0.2.10 | decided 2026-09-30 |
| The Markdown renderer has no "finished" signal | → with the Markdown renderer work | S, but that is when the code is open |
| Rendering LaTeX equations in the chat log | → post-0.2.10 | the block-rendering brief calls fenced code plus LaTeX multi-week |
| Holding the scrollbar drifts while a reply streams | → post-0.2.10 | cost unknown |
| Move the avatar backdrop onto `fit_cover` | → post-0.2.10 | nothing waits on it *(inferred)* |
| Consolidate image conversions | → post-0.2.10 | the item itself is tentative |
| Cherrypick: preload 16 MP, idle load, low FPS, zoom-in neighbours | → post-0.2.10, as one measured pass | a cluster; the items say measure first |
| Easy install with a chosen CUDA | → post-0.2.10 | "re-scope first" |
| Relocate webfetch's "approve denied host" button | → post-0.2.10 | cosmetic *(inferred)* |
| Colorblind-safe ok/error flashes | → with the styling-constants sweep | same code |
| Attach a document from a URL | → post-0.2.10 | a design question in the item |
| Docs DB stores full text *and* chunks in the JSON | gate → brief 12 | dissolves into its D1 and a BM25 backend |
| A fetched page is budgeted as a user attachment | **verify** whether v1 shipped in 0.2.8, then post-0.2.10 | |
| Nothing remembers which sibling (HEAD undo) | → `next` | M; pairs with the sibling-flick design |
| Re-test `reasoning_effort` on Qwen 3.8 | → anytime, with the 3.8 experiments | S; needs you at the keyboard |
| `chattree.get_all_root_nodes` is O(n) | leave as is | its gate is already conditional |

## C. `post-0.2.10` items that may be wanted sooner

The rest of the `post-0.2.10` gates just mean "later" and can stay. These four say more:

| item | proposal |
|---|---|
| Expose the docs-DB sources behind RAG citations | "wanted this year" — rank with inline citations |
| Let the AI drive the constellation's own views | "this year if possible" — rank with brief 13 |
| The document-ingestion cluster (formats, spreadsheets, text from images, `.svg`, page images) | write its one brief when the cluster is picked up, as the head item says |
| A crash during ingest loses the whole run | gate → the per-document LLM pass, as it says |

## D. No decision needed

Three headings or premises to correct: "Librarian's help card: the room exists now…" (Librarian's and the
Visualizer's cards are done; six remain), "Datastore scaling…" (still assumes the datastore is saved only
at exit), and "Documented command lines are unchecked" (its note about three slow CLI tools is fixed).

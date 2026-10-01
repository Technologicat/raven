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

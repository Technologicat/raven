# Researchers' Night sprint

**Deadline: 26 September 2026.** That is what this folder is named for, and it is the honest name — the
contents are not one scope. They are the briefs that were conceived during the Librarian run but scheduled
after it, and what they share is the deadline that orders them, not a subsystem.

Split out of `librarian-extension/` on 2026-08-07, at the 10/11 boundary. Worth knowing that the boundary is
chronological rather than topical: the numbers ran in the order the briefs were *written*, so 14 and 15 are
Librarian features and 11 is Visualizer, sitting side by side here because of when they were conceived.

## What's here

| Brief | What | Status |
|---|---|---|
| `15_headless-agent-driver-brief.md` | A scripting surface over the scaffold — build a turn's prompt, or run a turn and report what happened | **First in the queue.** v0.2.9. Ranked on *timing*, not closure: roughly seven weeks of investigation-heavy work sit ahead of it and every one of them wants a scriptable driver. It is also **a power multiplier for Claude specifically** — see below |
| `16_chat-graph-view-brief.md` | The chat tree as a graph, for the exhibit | Researchers' Night. Explanatory before navigational — the job is making "an LLM is a multiverse generator" visible. Step zero is that `XDotWidget.set_graph` has no callers and no tests |
| `crt-display.md` | Avatar postprocessor: CRT look | Researchers' Night |
| `atmospheric-dust.md` | Avatar postprocessor: dust | Researchers' Night, and **the schedule's slack** — lands only if time remains. Ranked behind 16 on 2026-08-05. Safe to drop: nothing depends on it, and it borrows its priority-band scheme *from* `crt-display.md` rather than the other way round |
| `12_derived-artifact-store-brief.md` | One keying and regeneration mechanism for everything computed *from* a source artifact | v0.2.9. Does not depend on 13 |
| `13_corpus-scopes-and-unified-db-brief.md` | Corpus scopes and the unified DB | **A draft, not a design** — it holds the 2026-08-01 session material with its `[D]`/`[N]`/`[P]`/`[X]` provenance markers intact, so a reader can tell settled from proposed. Realistically after Researchers' Night |
| `11_visualizer-importer-rework-brief.md` | Nomic migration, PCA preprocessing, cosine-to-medoid outlier assignment, Procrustes alignment | Its item 1 carries **a fork that needs deciding** — `nomic-embed-text-v1.5` buys a shared image-text space, `v2-moe` buys multilingual, and no v2-aligned vision encoder appears to exist. That decision reaches brief 12 |
| `14_chat-search-brief.md` | Search within the chat log | v0.2.9. The match unit is the **message**, which is what keeps v1 cheap — it sidesteps in-text highlighting, whose Visualizer implementation rebuilds the whole panel and so does not transfer to an incrementally-built chat log |

## Ordering

**15 first**, for the amortization argument above. After that the ordering is not settled, and the two
sensible axes disagree — closure rate (smallest first, so briefs shut faster than they open) against the
exhibit deadline. 16, `crt-display` and `atmospheric-dust` are the only ones the deadline actually binds;
everything else could slip past September without anything breaking.

### Decided 2026-08-07 — what Monday starts on

1. **Triage `TODO.md` and `TODO_DEFERRED.md` first** (Juha, partly via claude.ai). A two-part plan: a
   mechanical sweep for stale items against a brief Juha will supply, then a human-review pass over what
   survives. It goes first because the pile is what makes everything after it hard to see, and because the
   sweep's whole point is that a backlog nobody can read end to end is not a queue.
2. **Then OS-independent file-manager drag-and-drop, ASAP.** Not in this folder and not in any brief — it
   lives in `TODO_DEFERRED.md`, "OS drag-and-drop of files into DPG apps", with the probe and the measured
   result in `investigations/dpg-dnd/`. Two reasons, and **the exhibit is not one of them** — by the test
   above, a visitor in one sitting is not dragging files out of a file manager either:
   - **The 2026-08-07 probe collapsed its cost.** The platform work is already inside the GLFW that DPG
     links, so this is wiring rather than building.
   - **It is a power multiplier for our own testing**, which is where the gesture actually happens dozens of
     times a day: feeding corpora in, attaching a file to check a render, driving the GUI by hand. The
     `FileDialog` is currently the sole entry path for all of it, which is also why that picker has
     accumulated its own pile of deferred improvements.
3. **Then the exhibit briefs**: 16, then `crt-display`, with `atmospheric-dust` as slack.

### Two of these are power multipliers, and it is worth naming the category

Drag-and-drop and **15** are not features for end users at all. They are tooling that multiplies the
*builders'* throughput, and they are ranked accordingly rather than by user-visible value. Since review is
the binding constraint on this project (see the root `CLAUDE.md`, "Who develops Raven"), anything that
raises how much can be built and checked per session competes directly with feature work rather than
sitting beneath it.

They multiply different people, which is why both are wanted:

- **Drag-and-drop is for the human**: feeding corpora in, attaching a file to check a render — dozens of
  times a day, through a `FileDialog` that is currently the sole entry path. **The deferred `FileDialog`
  improvements belong to this same category** and are wanted for the same reason: smart-case find,
  thumbnail previews, the multi-extension filter, the per-use-site boilerplate. They are not user-facing
  polish so much as throughput for whoever is driving the apps all day.
- **15 is for Claude, and specifically for writing probes**, not for GUI testing — that is a separate
  problem which a scripting surface does nothing about, and which still needs a live session.

  The evidence is already in the tree: `../librarian-extension/manual_tests/` holds six scripts —
  `rag_live_corpus.py`, `rag_tool_rescue.py`, `webfetch_live_extractors.py`,
  `webfetch_tier2_escalation.py`, `gemma4_reasoning_roundtrip.py`, `vision_check.py` — each of which
  reaches into Librarian's agent machinery to exercise one feature, and each of which had to arrange that
  access for itself because the library does not offer it. That is a supported feature being asked for
  repeatedly and answered ad-hoc every time.

  So 15 is less "new capability" than "stop making every probe re-invent the entry point". Its part 0, lazy
  `api.initialize`, also removes `test_scaffold.py`'s `importorskip`, widening what CI can run.

**Not in the Researchers' Night run**, decided the same day: Hindsight memory (06) waits until after it —
because a visitor who talks to the system once cannot observe a feature that pays off over a long-running
relationship, at any level of completeness — and the MCP client (04) and lorebook (05) are question marks,
useful but not open-house-critical. See `../librarian-extension/README.md`, "After those three", which
carries the generalizable form of the memory argument: **a feature whose value accrues over time cannot be
demonstrated in an encounter that does not.**

**Ligature repair** (`../ligature-repair-brief.md`) also waits, *unless* the `raven-fixbib` half turns out
small enough to sneak in — which the brief argues it is, being the function plus a flag plus a report. The
indexer half is not a candidate under any reading.

**Brief 17 is reserved but unwritten** — a per-document LLM pass with retry, cache, resume and progress, cut
out of 15 because it has three users of its own and is a batch-execution primitive rather than a scripting
surface. If it stays unwritten, 17 may end up being something else, and 15's reference to it will need
chasing.

## Where the other sprint is

`../librarian-extension/` — the 01–10 run, mostly closed. Its `README.md` carries the ordering rationale for
what remains there (04, 05, 06) and the record of what 0.2.8 shipped.

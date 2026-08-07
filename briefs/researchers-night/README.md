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
| `15_headless-agent-driver-brief.md` | A scripting surface over the scaffold — build a turn's prompt, or run a turn and report what happened | **First in the queue.** v0.2.9. Ranked on *timing*, not closure: roughly seven weeks of investigation-heavy work sit ahead of it and every one of them wants a scriptable driver |
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

**Brief 17 is reserved but unwritten** — a per-document LLM pass with retry, cache, resume and progress, cut
out of 15 because it has three users of its own and is a batch-execution primitive rather than a scripting
surface. If it stays unwritten, 17 may end up being something else, and 15's reference to it will need
chasing.

## Where the other sprint is

`../librarian-extension/` — the 01–10 run, mostly closed. Its `README.md` carries the ordering rationale for
what remains there (04, 05, 06) and the record of what 0.2.8 shipped.

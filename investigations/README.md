# Investigations

Things we measured, profiled or reproduced, each as one self-contained directory: the write-up, the scripts
that produced it, and the data they emitted, together.

The reason for the layout is that a measurement whose apparatus lives elsewhere is not reproducible in
practice, however carefully it was written. Splitting a write-up from its scripts also loses the link both
ways — several of ours were only recoverable by asking git what landed in the same commit, because nobody
had written down which probe produced which table.

**So each bundle carries a `README.md` naming its own scripts and what each one answers.** That is the part
that stops the link decaying again, and it is worth keeping up even when the connection feels obvious today.

## What's here

| Directory | What it investigated |
|---|---|
| `context-injects/` | What shape Librarian's temporary context injects should take, measured across four local models |
| `retrieval/` | Retrieval quality against a known corpus — the evaluation set behind brief 09 |
| `tool_budget/` | Whether the tool-call round cap causes empty replies, and whether telling the model its budget is spent prevents them |
| `tool_refusal/` | Past that cap, whether refusing a call ends the turn or whether it takes withdrawing the tools — a follow-on to `tool_budget/` |
| `vram/` | The avatar's VRAM footprint, module by module |
| `tha3-performance/` | Where THA3 inference time goes, and whether the pipeline halves can overlap on the GPU |
| `anime4k-performance/` | Where the Anime4K upscaler's time goes |
| `dpg-focus/` | Which DPG predicate means "this text field holds the caret", and what `focus_item` does to a child window |
| `dpg-overlays/` | Why a floating overlay window must be sized to its content: it eats the mouse across its whole rect, and autosize has a silent 100 px floor |
| `dpg-dnd/` | Whether OS-level file drag-and-drop is reachable without writing a shim per platform (it is — via the GLFW DPG already links), and the render-thread constraint that comes with it |

## Shared instruments are pointed at, not copied

Some probes serve several investigations, or belong to a brief rather than to a study. Those stay where they
are — currently `briefs/summer_2026_librarian_extension/manual_tests/` — and the investigation names the path.
`tool_budget/README.md` does this with `rag_live_corpus.py`, whose phase F produced its samples. Copying a
shared instrument into every bundle that used it would trade one kind of drift for a worse one.

## Not investigations

Reference material — external requirements, lookup tables, archived snapshots — lives in `briefs/reference/`.
The first test is whether we ran something to find out.

Where a document is genuinely both, **file it by how it is read.** `briefs/reference/dpg-keycodes.md` was
produced by probing DPG and carries its own reproduction script, so the first test calls it an investigation;
but it is consulted as a lookup table several times a month and cited that way from `CLAUDE.md`, so it is
shelved where a reader will reach for it.

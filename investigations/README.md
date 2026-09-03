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
| `dpg-keyboard-chords/` | Which chords a global key handler still receives while a text field holds the caret, and whether modals stack — the groundwork for `FileDialog`'s keyboard operation |
| `dpg-input-text/` | What an `InputText(on_enter=True)` reports and when — that an item-edited handler still fires per keystroke, and that a global key handler sees Enter first and sees the field already deactivated, so no predicate can tell it whose Enter that was |
| `dpg-autosize/` | Why an autosize window renders one frame at the wrong size when its content changes, and which escapes are real. Mostly a lesson about instruments: the reported size is stale in every case, including the ones that never show it, so only screenshots answer the question |
| `filedialog-performance/` | Where FileDialog's open and close time goes. Building the listing was never the expensive part; the close path rebuilt it two or three times for nothing, and a table submits every row each frame unless it clips |
| `prompt-size-cache-relative/` | Whether a backend's reported `prompt_tokens` counts the whole prompt. On LM Studio it counts only what the KV cache did not already hold — which Raven's own prefill creates, so the context readout under-reported by an order of magnitude |
| `agent-batch-classification/` | Driving `librarian.agent` over ~1600 real papers to sort them by field. Mostly about what goes wrong at that scale — chiefly that the model answered *most* confidently about the filenames carrying the least information, which is the case an escalation rule based on its own confidence would never re-examine |
| `follow-tail-drift/` | Three episodes of the chat view's scrolling: why it intermittently stops following a streaming reply (diagnosed from an ordinary run's log rather than reproduced — the refusal reports its own numbers), why it opened part-way down its own content, and the 2026-07-30 work that made it follow at all |
| `absent-tool-behaviour/` | What a model does when asked the time with no clock tool — mostly it refuses; and what naming a tool that is not there does to it, which is send a third of the samples into a reasoning loop until the budget runs out |
| `thinking-toggle-cache/` | What flipping the thinking toggle mid-conversation costs. Nothing on Qwen, the whole KV cache on Gemma — decided by where each template puts its thinking marker, which the two models together prove and either alone would not |
| `chat-template-think-prefill/` | Whether a thinking model's chat template already opens the `<think>` block before generation starts, leaving only the close on the wire. Every Qwen we run does — so a backend that parses reasoning server-side is the only thing standing between Raven and that path |
| `todo-sweep-2026-08-10/` | Which `TODO_DEFERRED.md` items still hold against the code. Report only, and **complete** — all 130 carry a verdict (112 confirmed, 9 stale, 6 moved, 2 superseded, 1 left unchecked on purpose). Part C of `briefs/todo-sweep-2026-08-10/` |
| `abort-inflight-request/` | How to abandon a backend request from another thread. `Response.close()` — the obvious route — neither wakes the blocked reader nor returns to its caller, blocking it for the whole read timeout; only `socket.shutdown` does both. The expensive prompt-processing wait falls after the headers, so there is a socket to reach, and the backend does drop the abandoned work |
| `backend-fault-injection/` | A proxy that fails a turn on demand, so what Librarian does when the backend errors *while the user is elsewhere* is testable rather than waited for. Found a streaming message widget left on screen after its turn ended away from the view — visible for a second or two, which is below what a driven screenshot catches |
| `bibliography-dedup/` | What each of `raven-deduplicate`'s rules actually did to a given `.bib`, listed and grouped rather than totalled. The instrument that found both of its false merges, neither of which a cluster count could show — so it is re-run whenever a rule changes, not once. Also where `config.doi_title_floor` comes from, and why the guard it serves takes two conditions and not one |
| `highdim-clustering/` | Whether the Visualizer's clusters should be found in the embedding space rather than in the 2D map — yes, measurably, and that part is settled on two corpora in unrelated domains. Which algorithm, was decided by coverage rather than by any quality metric: HDBSCAN either labels a fifth of a corpus or silently collapses to two clusters (three representative corpora of four), so agglomerative average-linkage cut at a fixed resolution ships. Mostly a lesson about the yardstick — every interesting difference here was an artifact of coverage, cluster count, sliver clusters, or `float16` caches until a control removed it. Also kills two items of brief 11 on the data: PCA preprocessing hurts separation, and unconditional outlier assignment undoes the whole gain |
| `chatgraph-rebuild-cost/` | What one rebuild of the chat graph view costs, and what it scales with — the number of boxes drawn rather than the size of the forest, so the sibling window is bounded by legibility rather than by time. Also the reason the depth window can hide the very level the sibling window exists for |
| `aokk-corpus-scope/` | Flagging the records a boolean literature search pulled in that are not about its topic, so they can be reviewed out. Chiefly a lesson about which way to ask: every record has already passed the search, so a model asked to re-confirm it reads every silence as a rejection — *"AI focus, but educational level unstated"* dropping real studies at medium confidence. Asked instead for evidence of being *off* topic, the same records come back kept. Also the counter-example to `agent-batch-classification/`: on this corpus the model is honest about thin input rather than confident about it |

| `tool-round-shape/` | How many result nodes a tool round actually folds, which decides whether the chat graph's folding of them buys anything. On Qwen 3.6, 85% fold exactly one — a box to hide a box — so a designed way to unfold them was measured and deferred rather than built. Written as a script because the answer is a property of the *model's* habits, and "models are becoming more agentic" is a prediction where this is a measurement: re-run it when the model changes |

## Shared instruments are pointed at, not copied

Some probes serve several investigations, or belong to a brief rather than to a study. Those stay where they
are — currently `briefs/librarian-extension/manual_tests/` — and the investigation names the path.
`tool_budget/README.md` does this with `rag_live_corpus.py`, whose phase F produced its samples. Copying a
shared instrument into every bundle that used it would trade one kind of drift for a worse one.

## Not investigations

Reference material — external requirements, lookup tables, archived snapshots — lives in `briefs/reference/`.
The first test is whether we ran something to find out.

Where a document is genuinely both, **file it by how it is read.** `briefs/reference/dpg-keycodes.md` was
produced by probing DPG and carries its own reproduction script, so the first test calls it an investigation;
but it is consulted as a lookup table several times a month and cited that way from `CLAUDE.md`, so it is
shelved where a reader will reach for it.

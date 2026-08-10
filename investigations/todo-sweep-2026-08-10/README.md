# TODO_DEFERRED verification sweep, 2026-08-10

**Part C of `briefs/todo-sweep-2026-08-10/todo-mechanical-pass.md`. Report only — nothing here was applied
to `TODO_DEFERRED.md`.** The triage discussion consumes this file; heading text is verbatim, so it joins
against the deferred file and the cluster map.

**Status: partial. 16 of 130 items carry a verdict.** The remainder are unchecked, not confirmed — see
"Coverage" below, which says plainly what that means for the discussion.

It is a directory rather than the single file the brief named, because it came with a script, and this repo
keeps an artifact with what produced it.

## Scripts

| Script | What it answers |
|---|---|
| `check_references.py` | Which items name a file or symbol that no longer exists anywhere in the tree. Cheap evidence for MOVED and STALE, gathered for all 130 items at once. Re-runnable; run it again before the discussion pass. |

An earlier version of that script resolved bare basenames against the repo root and reported 43 items as
broken. Nearly all were false — `config.py` and `app.py` exist in plenty of places, just not at the root.
Resolving by basename across tracked files, and letting a filename satisfy a symbol reference, brought it to
11. **Treat a first-run number from a checker like this as a bug report about the checker.**

## Verdicts

Five verdicts, per the brief: CONFIRMED (claim holds), STALE (claim false — fixed, landed or removed), MOVED
(true, but the code named has relocated), SUPERSEDED (true, but a brief now owns it), UNCHECKABLE (needs a
running app, a live backend, or a human eye).

| Heading | Verdict | Evidence |
|---|---|---|
| OS drag-and-drop of files into DPG apps (cross-platform) | **STALE** | Shipped 2026-08-10 as `raven/common/gui/filedrop.py`, wired into all six GUI apps and live-tested in each. |
| "Internet" toggle: scope `tools_enabled` to a clear security boundary in the GUI | **STALE** | Landed in 0.2.9-dev: `scaffold.ai_turn` takes `internet_enabled`, and `llmclient.maybe_tool_names_for_turn` computes an allowed-tool-name list from `network_tool_names` / `document_tool_names`. The item asks for exactly this. Its "separate toggle for MCP tools later" aside belongs to brief 04. |
| Indexing a large corpus is silent for minutes, and reads as a hang | **STALE** | 0.2.8 added the INDEXING indicator with per-document progress (`librarian/app.py:848`). |
| Idle throttle for Librarian | **STALE** | 0.2.8 added it; `librarian/app.py:225`, `IDLE_SLEEP_S = 0.08` (~12 fps idle). |
| Headless scaffold mode for `ai_turn` (scriptable agent layer) | **SUPERSEDED** | This *is* brief 15, `briefs/researchers-night/15_headless-agent-driver-brief.md`. |
| Lazy `api.initialize` in `llmclient` and `hybridir` | **SUPERSEDED** | Brief 15 Part 0 owns it, and cites this entry by line number. |
| webfetch "approve denied host" button relocates in brief 03 | **CONFIRMED** | Brief 03 is closed, but `chat_controller.py:1090` still carries the `role == "tool"` branch and the approve button the item says to relocate. Its precondition landed without the follow-up, so it is *more* live than when filed. The 2026-08-05 ruling would have declined this conditionally; the condition is not met. |
| Same file formats in the docs DB and in chat attachments | **CONFIRMED** | Two halves, and the item says so. Office formats landed (`librarian_config.llm_docs_exts` now equals `docextract.supported_extensions()` exactly — 13 formats, verified equal). The images half is untouched and blocked on the Nomic embedder. |
| The DPG tests we have never run in CI | **CONFIRMED** | `.github/workflows/requirements-ci.txt` still has no `dearpygui`, so the `importorskip` fires on every CI run. |
| The docs DB stores each document's full text *and* its chunks, both in the JSON | **CONFIRMED** | `hybridir.py:410` still writes `fulldocs/data.json`. The `data.json` in this item is the *docs* store, untouched by the chat-datastore rename — the reference checker flagged it, and was wrong. |
| AMD GPU (ROCm) support audit | **MOVED** | Names `raven/common/lanczos.py`; it is `raven/common/image/lanczos.py`. |
| CLAUDE.md: rephrase DPG pitfall #5 to avoid Claude thinking loops | **MOVED** | Names `dpg-threading-notes.md`; it is `dpg-notes.md`. |
| Audit and slim down project CLAUDE.md | **MOVED** | Same rename. |
| Datastore scaling: a single `data.json` (+ flat sidecar dir) won't hold years of chats | **MOVED** | The concern stands; the names changed in 0.2.9-dev — `data.json` → `chat.json`, `<datastore>.images/` → `<datastore>.sidecars/`. |
| Librarian has no periodic autosave: an abnormal exit loses the whole session's chat | **MOVED** | Same rename; the item names `data.json` and `state.json`. |
| Reasoning traces with indented bullets mis-render | **MOVED** | Uses `data.json` as an example path; same rename. |

## What the reference checker flagged, and what it was worth

Eleven items name something that no longer resolves. Six became the MOVED verdicts above. The other five are
false positives worth recording so the next run does not re-litigate them:

- `librarian/cleanup.py`, `librarian/app.py` — real files, written repo-relative-ish rather than as full paths.
- `raven/common/gui/dpgstyle.py` — a file the item *proposes creating*, not one it claims exists.
- `icons.yml` — upstream FontAwesome's file, not ours.

## Coverage, stated plainly

**16 of 130 items have a verdict. 114 do not.** They were selected, not sampled: the ones today's and
0.2.8's work plausibly closed, plus everything the reference checker flagged. So the STALE rate here says
nothing about the rest of the file, and **an item's absence from this table is not evidence that it holds.**

The brief predicted a meaningful STALE rate and explained why it is expected rather than embarrassing: many
of these were filed during a period when the standing instruction was never to get sidetracked, so things
were filed rather than fixed, and some were then fixed without the item being closed. Four of the five STALE
findings above are exactly that shape.

What is left is the per-item work: read the claim, find the code it rests on, check it. The unchecked 114
skew old — the March–June items are where the yield should be, and none of those were touched here beyond
what the reference checker saw.

## Re-running

`python investigations/todo-sweep-2026-08-10/check_references.py` regenerates the mechanical half against
whatever the tree looks like then. The verdict table above is hand-made and does not regenerate; treat it as
of 2026-08-10 and re-check anything acted on much later.

# Brief: rebuild the temporary context injects

**Status: built, 2026-07-28.** All five changes landed in `scaffold._perform_injects` and `chatutil`, and
were verified end to end against a live backend on Qwen3.6-27B and Qwen3.6-35B-A3B
(`manual_tests/assembled_shape.py`). Four things went differently from the plan below, each for a reason
worth carrying forward:

- **No midnight watcher was needed** (§4). The plan assumed the date would be baked into the stored system
  prompt, which is what forces a rollover watch. Injecting it fresh each turn into a *copy* of the leading
  system message is always correct and needs no machinery. The date was removed from the system-prompt
  template in `librarian/config.py` at the same time, so exactly one place now states it — two would have
  been redundant on a good day and contradictory on a long one.
- **The focus reminder was retired, not moved** (§2). It was two instructions welded together: "reply to
  the user's most recent message" (the DeepSeek-R1 workaround, obsolete, and now protected structurally by
  the `before` placement) and a style nudge against report-shaped answers. Only the second survives, as
  `chatutil.format_reminder_to_write_conversationally`.
- **The search query rides in the synthetic call's arguments** (§3), which retires the old
  `# TODO: Should the RAG match notification show the query string, too?` — a tool call is the natural
  place for it, and the model can then see what was asked on its behalf.
- **A new failure was found and deliberately left unfixed:** asked something the documents do not answer,
  the model reaches for another search it cannot have (Q11 in the measurements). The obvious mitigation
  made things worse in the exact way Q4 warns about. The real fix is the queued RAG-tool work.

**What:** rework `scaffold._perform_injects` — the material Raven adds to every AI turn that the user
never typed: the current date and time, two behavioural reminders, and one message per RAG match. Change
the role each takes, where each sits, and the wording of one of them.

**Why now:** four of the six items in `TODO_DEFERRED.md`'s inject cluster turn on questions that were
recorded as unmeasured for months, and two of the "obvious" answers turned out to be wrong when measured.
Everything below is settled by experiment — the numbers, method and caveats are in
`../context-inject-shape-measurements.md`, which is the companion document to this brief and outlives it.

**Not a refactor.** The current code works; it is the *shapes* that are wrong. Expect a small diff in one
function plus a wording change in `chatutil`.

## What changes

### 1. Reword the "answer from context only" reminder, and fire it only when there is context

`chatutil.format_reminder_to_use_information_from_context_only` currently says *"Please answer based on
the information provided in the context only."* That is self-contradictory when no context was supplied,
and the models obey it literally rather than smoothing it over. Measured on every model tested it costs
5–37× the deliberation of sending nothing; Qwen3.5-9B never terminates at all (52796 characters of
reasoning against a 16000-token budget, `finish_reason=length`, no reply); Gemma 4 refuses outright.

Replace with:

> Base claims about the provided documents on those documents. Answer general questions normally.

This measured within noise of sending no reminder on three of four models, with no hedging, and still
declined correctly when asked about something absent from the documents (so it has not simply been
defanged — both halves were checked).

Separately, `_perform_injects` gates this on `if not speculate:` alone, with nothing tying it to whether
context exists. Gate it on context actually being present. "Context" is broader than docs matches:
attachments and material the user supplied earlier in the conversation count.

*Worth knowing before tuning the wording further:* with no reminder at all, every model still declined
correctly on the absent-fact question. The system prompt was carrying that behaviour by itself. Prefer
the mildest wording that does the job.

### 2. Both reminders move into the leading system message

The focus-on-latest-input and answer-from-context-only reminders are **instruction-like** — we want them
obeyed. Measured, `system_front` was the cheapest or near-cheapest shape on all four models and the only
one that never produced narration.

Two things make this affordable, and both matter:

- Their text is **constant**, so merging them into the leading block costs nothing in KV-cache terms. The
  usual objection to hoisting applies only to injects that vary per turn.
- The tool role is **wrong** for them. It measurably weakens how strongly an instruction binds — on
  Qwen3.6-35B-A3B the same reminder produced an outright refusal in the user role and was quietly ignored
  in the tool role. Reduced compulsion is right for data and wrong for directives.

Do **not** reach for the fold (appending inject text into the user's message). It was the shape this
project previously favoured, and it measured identically to the status quo on Qwen3.5-9B — it does not
fix the narration it was meant to fix.

**Hazard:** `chatutil.linearize_chat` returns `payload["message"]` — the stored dicts themselves, not
copies. Appending to the history list is safe; merging text into an existing message is not. Copy the
system message before modifying it, or the stored system prompt is corrupted in the datastore.

### 3. RAG matches become one tool message, before the user's latest message

Three separate changes, each measured:

- **Role → `tool`, with a synthetic `tool_calls` assistant message preceding it.** The synthetic call is
  required, not decoration: Gemma 4 ignores a bare `tool` message and confabulates a confident wrong
  answer in its place — across three packagings and two backend versions, so it is a model property.
- **One tool message carrying all matches**, not one per match. The OpenAI schema pairs a `tool` message
  with a single `tool_call_id`; the one-per-match form shares an id across many messages, which most
  models tolerate and Gemma4-E4B does not. Merged measured never worse and better on the weakest model.
  This also matches what `_perform_and_store_tool_calls` already does for real tool calls: one result
  message per call.
- **Position → immediately *before* the user's latest message**, replacing `history.insert(1, ...)`.

That last one is the non-obvious result. The front insert costs a full KV-cache prefix rebuild every
turn, and the constraint that motivated it (Qwen 3.0 ignoring late material) did not reproduce anywhere —
but moving the material to the *end*, after the user's question, breaks differently: with a tool result
as the last message, Qwen 3.6 sometimes answers by emitting **another** `search_documents` call instead
of replying. Placing it before the user's question keeps the prefix ahead of it stable *and* leaves the
last message the user's question. Measured 36/36 across three models and two corpus sizes, against 2/3
and 1/3 for the end placement on Qwen3.6-27B.

Note the interaction with `continue_`: when continuing the AI's last message, the always-on injects go
before that trailing assistant message. The new RAG placement has to respect the same rule — the history
must look as it did when generation was interrupted.

### 4. Split the datetime inject: date in the system block, time as tool output

The datetime was assumed to be unhoistable because it changes every turn. Only the *clock time* changes
every turn — the **date is good for a whole day**. Put the date in the leading system message and deliver
the time as tool output (same synthetic-call shape as the RAG matches), and the prefix is stable until
midnight.

This measured correct on every model tried and roughly **halved** the deliberation against putting the
whole datetime in the system block. Ungrounded, every model is wrong about the date and one of them
confidently computes from a date it never states — so this inject earns its place.

Requires a **midnight rollover watcher**: when the local date changes, patch the system message. That is
cheap against invalidating the cache on every turn, which is what the combined inject costs today.

### 5. No branch on model size

Worth stating because it was a live possibility: with Qwen3.5-4B in the small-model slot (see
`../model-lineup-autumn-2026.md`), the shapes above are uniform across every model in the lineup.
`_perform_injects` needs no capability check and no size fork.

## Verifying it

The probes in `manual_tests/` are the acceptance test — they measure exactly these shapes:

- `inject_shapes.py` — reminder wording (probes 5, 6, 7), narration and role (probes 2, 3)
- `rag_placement.py` — placement and packaging at realistic corpus sizes
- `datetime_inject.py` — the date/time shapes, including `split`

After the change, the wire format Raven produces should match what those probes send. A quick way to
check is to log one assembled history and compare it against the corresponding `build()` output.

## Out of scope

- **RAG reranking** (fewer, better matches). Its own item; it shrinks whatever this brief injects but
  does not change the shape.
- **Modernizing the system prompt / character card.** Adjacent — this brief adds text to the leading
  system message, so the two will collide in the same lines — but it is a separate judgement call.
- **Text watermarking / provenance.** See brief 07.

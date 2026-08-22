# A backend's `prompt_tokens` can count what it processed, not what you sent

**Measured 2026-08-22, LM Studio 0.3.x serving `qwen3.5-9b` Q4_K_XL at a loaded context of 131072.**

Librarian's context-fill readout is two-stage: a local character-ratio estimate shown as `~X%`, then the
backend's own figure shown as `X%`. The second comes from `llmclient.prefill`, which sends the real prompt
with `max_tokens=1` and reads `usage["prompt_tokens"]`.

**On LM Studio that figure can come back an order of magnitude short**, and nothing in the response says so.

**The mechanism is not identified, and the obvious explanation is wrong.** "It reports the tokens it had to
*process*, the rest being in the KV cache" fits the first three rows below and fails the fourth: appending a
~600-token nonce to the *end* of the history — verified on the wire, 5 messages and 325734 characters of
JSON — left the figure at exactly 8745, where processing 600 new tokens should have shown 600 more. The
number appears to be *insensitive to appended content* while being far below the truth. What is established
is the shape of the failure and the way to get a true figure, not why.

## The measurement

One chat branch, three attached PDF fulltexts, four requests:

| request | `prompt_tokens` |
|---|---|
| as-is (backend already asked about this branch) | 8745 |
| the same again | 8745 |
| attachments not resolved — the conversation alone | 2630 |
| **nonce prepended at the front** | **56365** |
| nonce appended at the end (~600 tokens, verified sent) | 8745 |
| large nonce appended at the end | 8745 |

The cache-busted figure moves by a few tokens between runs — the nonce is a fresh UUID each time and is
itself tokenized, so a re-run reading 56362 rather than 56365 has reproduced the result, not drifted from it.

Against a local estimate of **81158** tokens, and 301591 characters of message text actually sent — verified
at the HTTP boundary, where the request body was 323867 characters of JSON with `[Attached file:` present in
it. So Raven sent the whole thing and the backend answered about a tenth of it.

The response carries no `prompt_tokens_details.cached_tokens`, so the missing part cannot be added back:

```json
{"prompt_tokens": 8745, "completion_tokens": 1, "total_tokens": 8746,
 "completion_tokens_details": {"reasoning_tokens": 0}}
```

## Why it bites Raven specifically

Whatever the cause, it is *asking twice* that triggers it: the same branch measured 98257 tokens the day
before, from a backend that had not been asked about it, and 8745 once it had. `prefill` is also what warms
the KV cache for the next turn, so Raven necessarily asks about the same prompt more than once — the tool
degrades its own measurement in the course of doing its other job.

Symptom: a chat with three papers attached, genuinely 43% of the window, showing **7%** — and showing it
without the `~`, since the number was believed to be exact.

**What would identify the mechanism**, if it becomes worth knowing: ask LM Studio the same prompt from a
*restarted* backend (cold by construction) and then repeatedly; vary the prefix by one token rather than
prepending a whole nonce; and try a second model, since this may be one runtime's accounting rather than
LM Studio's. None of that is needed for the fix, which is why it was not done.

## What was ruled out on the way

Each of these looked likely and was wrong, so a re-investigation need not repeat them:

- **The model's context being smaller than advertised.** `GET /api/v0/models` reports
  `loaded_context_length: 131072`, the full window; nothing was truncated for lack of room.
- **The attachments not reaching the wire.** The fold runs for user-role attachments as well as tool-role
  ones (the pre-existing test covers only the latter), and `serialize_history_for_wire` produced 301591
  characters with the documents in them.
- **Raven mutating its own stored messages.** `serialize_history_for_wire` deep-copies its input first, so
  the in-place content rewrite in its second pass touches only the copy — the stored `text_file` parts
  survive, checked directly.
- **The extraction failing and caching a placeholder.** `count_branch_tokens` with extraction returns 75146
  tokens for the same branch in a fresh process, so the PDFs have text layers and are found.
- **A transient state in the running app.** The whole thing reproduces from a script with no GUI.

## The fix

`llmclient.prompt_size_report_looks_whole(reported, estimate)` — believe the backend only when its figure is
not far below the local estimate, and otherwise keep the estimate and its `~`. The two cases are an order of
magnitude apart here (8745 against 56365 true), so the bound only has to tell *far below* from *near*.

**A sanity check rather than a correction, deliberately.** Not knowing the mechanism means not knowing how
to reconstruct the true number from the short one — and the response carries nothing to reconstruct it from.
Refusing to believe an implausible figure needs no such knowledge, and stays right if the cause turns out to
be something else entirely.

**Appending a nonce to warm most of the cache and still get a true reading does not work** (Juha's
suggestion, measured): the figure does not move at all, so there is nothing to trade off. Only a *prefix*
change produces the true size, and that costs the full reprocess the prefill exists to avoid.

**The bound is loose on purpose: the local estimate ran 44% high** (81158 against 56365 true), so a tight
one would reject the true figure. Half the estimate is the midpoint of the gap between the two measured
cases; `prompt_size_report_looks_whole` logs whenever it disbelieves a backend, which is what would show the
choice being wrong.

## Files

- `probe_prompt_size.py` — the four requests above, plus the estimate and the wire size. Defaults to
  Librarian's configured datastore and HEAD; `--datastore` and `--head` point it at a scratch chat instead.
  **It sends the chat's contents to the backend you name.**

Re-run it after an LM Studio upgrade, or when adding support for a backend whose usage reporting is unknown:
the question "is this figure about the whole prompt" has no answer in the OpenAI schema, so it is per-backend
behaviour that can change under us.

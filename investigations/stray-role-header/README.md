# Why do some replies open with a bare `assistant`?

**Status: fixed 2026-09-15, and live-checked** — one `invoke[turn]` per send, and an empty send ignored. The
send is coalesced from when the first one *finished*, the gate counts a send from the moment it is accepted,
and an empty send is off by default (`llm_allow_empty_send`). The runaway below remains unexplained.

**Question.** Raven-librarian sometimes stores a reply whose content begins with the word `assistant` and
nothing else on that line — either the whole reply, or `assistant` followed by the real answer. Is it the
model, LM Studio, or something Raven sends?

## Answer so far: one send starts two turns

**Every leaked message is an AI message generated directly after another AI message**, with no user message
between. Each was created 2–68 ms after its parent, and both GUI occurrences with a log show **two
`llmclient.invoke[turn]` lines 6 and 9 ms apart** for a single send. So:

1. Turn A starts, creates its in-progress reply node, and moves HEAD onto it.
2. Turn B starts milliseconds later. Its history ends on A's still-empty *assistant* node, so the request
   ends on an assistant message, the template renders no generation prompt, and the model writes the header
   itself — `<|im_start|>assistant\n`, with the special token stripped. That is also why the leaked replies
   barely reason: the `<think>` opening that the generation prompt would have supplied is missing too.

In one case A's node was a tool-call message, and B's node became a sibling of A's tool result.

The probe below never leaked because it never sends twice. What remains is the double send, and three
things combine in it — read from the source, not measured:

- **The coalescing in `app._request_send` assumes the two key paths run microseconds apart.** Its window is
  one frame. But the first send clears the composer by parking focus and calling `dpg.split_frame()`, on the
  same callback thread the second path is queued behind, so the second can arrive two or more frames later —
  more likely with the app idle-throttled to about 12 fps.
- **The send gate cannot see an accepted send yet.** `_describe_send_gate` asks `is_generating()`, which
  counts only the AI-turn task manager. `chat_exchange` submits its task on the general task manager, and the
  AI turn is registered only once that task has run the user's turn, so a second send in between passes.
- **An empty send continues the chat**: `chat_exchange` with no text runs an AI turn from HEAD. By the time
  the second send runs, the composer has been cleared, so it is exactly that, from HEAD on A's node.

The third holds for a deliberate empty send too: pressing send with an empty composer while HEAD is an AI
message builds a request ending on an assistant message. **Measured the same day, five times**, from a
finished reply to `Hi!`: four of the five came back empty, one token generated, and the fifth was a generic
offer to help. No `assistant` leaked in that sample — the leak needs the *empty* in-progress node, where a
finished reply is simply ended. That is what put the feature behind a setting.

## What is established

**Three occurrences in the maintainer's datastore**, the first on 2026-09-10 and two on 2026-09-15, all on
`qwen3.6-35b-a3b` served by LM Studio. They share a shape:

- **The leaked word opens the reply.** In one the content is `assistant` alone; in another it is
  `assistant\n\nI'm here and ready to help…`.
- **Every one is early in a chat** — a reply to `Hi!`, or to `Hi!` followed by a line or two.
- **None of them thought**: zero or one reasoning token, against a model that normally reasons.

None were found before 2026-09-10. That says none were *stored* before then, not that none happened; the
backend had seen little use in the preceding weeks.

**What a fresh chat sends.** Captured from the probe, for a first message of `Hi!`:

1. `system` — the setup
2. `assistant` — the character's greeting
3. `assistant`, empty content, carrying a synthetic tool call
4. `tool` — `[System information: The local time now is …]`
5. `user` — `Juha: Hi!`

**What Librarian's idle prefill sends.** The branch up to HEAD, so `[system, assistant]` for a fresh chat,
with `max_tokens` 1. It ends in an assistant message by construction: the prefill runs before the user's next
message exists, and HEAD is the AI's last reply.

**Closing the connection stops LM Studio's generation.** Killing the probe mid-turn returned LM Studio to
READY at once.

## Measurements

| Arm | Turns | Leaks | Notes |
|---|---|---|---|
| No prefill | 30 | 0 | |
| Prefill, run 1 | 5 complete | 0 | **Turn 6 ran away**: past 50k generated tokens, per LM Studio's own counter, before the probe was stopped. Its text is lost — that version of the probe kept only the visible reply — so whether it was thinking or answering is unknown |
| Prefill, run 2 | 11 | 0 | Stopped by hand, no runaway |

**A zero here is weak evidence.** At a 5% leak rate, 30 clean turns happen 21% of the time, and 11 clean turns
57% of the time; at 10%, 30 clean turns still happen 4% of the time. Three leaks in the GUI over two sessions
of ordinary use suggest a rate in that range, so no arm has yet said anything about a cause. **Do not bisect
until some arm leaks**: an arm that never produces the effect cannot tell anything apart.

## Ruled out

- **`750f4cd9`** (2026-09-10 12:49), 24 minutes before the first leak. A rename of five private helpers; it
  changes nothing on the wire.

## Hypotheses considered before the logs were read

The first is now the established mechanism; the second and third are no longer needed to explain the leak.
The runaway in prefill run 1 remains unexplained.

- **A rendered prompt without the generation prompt.** Qwen's template ends a generation prompt with
  `<|im_start|>assistant\n<think>\n` (see `investigations/chat-template-think-prefill/`). If a prompt reaches
  the model without it, the model writes the header itself, and with the special token stripped only
  `assistant\n` remains — which would also explain the missing thinking, since the `<think>` opening would
  be missing too.
- **The idle prefill leaves the cache in that state.** A request ending in an `assistant` message may be
  rendered by LM Studio as a message to *continue*, with no turn close and no generation prompt. Whether it
  is, and whether the following turn's cache reuse can carry that over, is unknown. The runaway in prefill
  run 1 may be the same fault in a different form.
- **The setup changes of 2026-09-09**, which moved the framing notice into injected preamble and postamble
  messages (`2ec78fbb`) and changed the character and user cards around it.

## Next steps

1. **The runaway is a separate question.** Run the prefill arm again with the live stream, to catch one.

## Apparatus

| Script | What it answers |
|---|---|
| `probe_role_header.py` | Runs N fresh-chat `Hi!` turns against the configured backend, optionally prefilling the greeting branch first, and counts replies opening with `assistant`. Keeps every request body, the whole `TurnRecord` per turn, and a live stream of each turn |

`data/` is gitignored: the captured requests carry the configured system prompt and user card. The runs of
2026-09-15 are there on the machine that made them.

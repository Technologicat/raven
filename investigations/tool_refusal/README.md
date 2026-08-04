# Tool-call refusal (Librarian)

Past the tool-call round cap, does telling the model its budget is spent make it stop asking — or does it
take withdrawing the tools to end the turn?

The two paths cost very different amounts. Keeping the tools in the schema and refusing the call leaves the
backend's cached prompt prefix intact; withdrawing them changes the prompt where the tool schema sits and
invalidates the cache from there on. So the value of the refusal mechanism rests on how often the model
takes it.

Companion to `../tool_budget/`, which asked a different question about the same cap (does reaching it
produce empty replies — yes, p = 0.013).

## Setup

`refusal_probe.py`, which drives the real `scaffold.ai_turn` against a live backend. The retriever finds
nothing, ever, so the model rephrases a search that cannot succeed and walks into the escalation on
purpose. `max_tool_call_refusal_rounds = 1` throughout, i.e. one refusal round before the withdrawal.

Model: qwen3.6-35b-a3b (IQ4_NL_XL, 128 Ki context) via LM Studio.

## Result, 18 samples across four caps, 2026-08-04

**The cap's effect depends entirely on where it sits relative to the model's own stopping point, which for
this task is 9–10 rounds.**

| cap | samples | reached the cap | stopped when told | needed the withdrawal |
|---|---|---|---|---|
| 1 | 6 | 6 | 0 | 6 |
| 5 (shipping default) | 5 | 5 | 0 | 5 |
| 10 | 4 | 3 | 2 | 1 |
| 20 | 3 | **0** | — | — |

At cap 20 the model never reached the cap in any sample: it ran 9, 10 and 10 rounds and then answered on
its own. So the escalation is not a thing that happens to a model given room; it is what happens when the
harness stops a model mid-strategy.

Read down the table, the refusal goes from useless to reliable as the cap approaches that stopping point.
The inference — and it is an inference from four points, not a finding — is that a model interrupted early
has a plan it has not finished and reads the notice as an obstacle, while a model interrupted late has
already reached the conclusion the notice is asking for.

## What the traces say

`traces-cap10.json` and `traces-cap20.json` hold every sample's reasoning trace and final reply.
(The cap-1 and cap-5 runs predate the trace capture; their console output is in `samples-2026-08-04.txt`.)

**The refusal is read and understood, even when it does not prevent the call.** The cap-10 sample that
ignored the notice reasoned its way to the conclusion and then talked itself back out of it —

> None returned any results. I should inform the user that I searched thoroughly but found no information
> in the local document database. **Wait, let me try one more thing.** [...] Let's try searching for
> "Kelvin-3 electrolysis" one more time, just in case.

— and on the next round, having been refused:

> The tool call budget is spent. I have already tried multiple searches and found no results. I will now
> inform the user that the local document database contains no information [...]

That second trace is *not* clean evidence for the refusal, because the tools were withdrawn on the same
round: both signals were present, and the trace cannot say which it acted on. What it does show is that
the refusal text is legible to the model rather than being noise.

(Juha's note on the "wait, one more thing" turn: characteristic of the Qwen series, and possibly an
artifact of distillation — this is not the flagship model.)

**The cap-10 sample that heeded the notice needed no such prod.** Its last pre-notice trace ends "Let me
try one more search with just [...] as a full phrase"; its post-notice trace lists the eight queries it
tried and concludes "I should inform the user of this result."

## What this does not establish

- **The scenario is one shape, and the adversarial one.** A search that cannot succeed is the failure the
  round cap was originally added for (`briefs/summer_2026_librarian_extension/manual_tests/rag_tool_rescue.py`).
  The shape that motivated the budget work — reading a list of documents one fetch at a time, so the budget
  runs out with material already in hand — is **not measured here**, and there is no reason to assume a
  model's stopping point is the same for reading as for searching.
- **One model, small n.** Four caps, 18 samples, no repetition across temperatures. The 2-of-3 at cap 10 is
  three samples; treat the ordering as the signal and the fractions as noise.
  - Expect a **model-family effect** on top of that: the Qwen series is known for persistence (Juha), so
    where the stopping point falls, and how hard the notice has to push to move it, are both likely to
    differ on another family. Nothing here separates "how models behave" from "how Qwen behaves", and the
    shipping cap is currently set from one family's number.
- **The cache-burn premise is reasoned, not measured.** Withdrawing the tools is *believed* to invalidate
  the whole cached prefix, because the tool schema is serialized near the front of the prompt. Nothing here
  measured prompt-processing time on a withdrawal round against an ordinary one. If the refusal mechanism
  is ever argued for on cost grounds again, that is the missing measurement.
- **Rewording the notice was tried and did nothing.** On the hypothesis that the notice contradicted what
  the model could see — it says "no further tool calls are available" while the tools sit in the schema — a
  variant reading "the tools are still listed, but any further call will be refused" was run at cap 5.
  Result 0/5, unchanged, plus one empty reply. The wording is not what is doing the work here, and the
  shipped text was kept.

## Consequence for Raven

**`max_tool_call_rounds` was raised from 5 to 20 on the strength of this** (2026-08-04). A cap of 5 was
interrupting this model less than halfway through its own notion of due diligence, and a cap is supposed to
be a backstop: at 10 it was still binding in 3 of 4 samples, at 20 in none. The same conclusion arrives
from the document-*reading* side in the tool-budget item in `TODO.md`, by a different route.

**It also undercuts the premise of that item's two-budget design**, which argued against a single larger
cap on the grounds that "a `search_documents` can be rephrased forever against a corpus that has nothing".
Against a corpus that had *literally* nothing — the strongest form of that case — this model rephrased nine
or ten times and then stopped. One model and small n, so this is evidence rather than a refutation, but the
forever was doing a lot of work in that argument and it did not survive contact.


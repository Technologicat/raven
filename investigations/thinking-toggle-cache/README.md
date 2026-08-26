# Flipping the thinking toggle costs the whole KV cache on Gemma and nothing on Qwen

**Measured 2026-08-26, LM Studio 0.4.20 (build 1), against `qwen3.6-35b-a3b` and `gemma4-26b-a4b`.**

Raven's coming *Enable thinking* toggle is a request field — `reasoning_effort: "none"`, which LM Studio
implements by rendering the model's own non-thinking branch. The question here is what that costs when a
user flips it **mid-conversation**, which is what a demo operator does.

| model | cold prefill | warm cache | **the switch** | prompt tokens, on → off |
|---|---|---|---|---|
| `qwen3.6-35b-a3b` | 2.25 s | 0.20 s | **0.26 s** | 9957 → 9959 |
| `gemma4-26b-a4b` | 1.61 s | 0.20 s | **1.57 s** | 10282 → 10279 |

**On Qwen the cache survives** — the switch costs 0.06 s over a warm hit. **On Gemma it is discarded** — the
switch is indistinguishable from a cold prefill of the whole conversation. Same prompt, same backend, same
request; the only difference is which model is loaded.

## Why, and the alternative that had to be ruled out

Where each family's chat template puts its thinking marker predicts this exactly:

- **Qwen** emits it at the *generation prompt* — the tail. Toggling changes the last few tokens, so prefix
  matching still covers everything before them.
- **Gemma** puts `<|think|>` at the top of the **first system turn**. Toggling changes the prompt's opening,
  so prefix matching fails at token zero and there is nothing left to reuse.

**Gemma alone could not establish that**, and it is worth saying why the second run was necessary rather
than a formality. A simpler explanation fit the Gemma numbers perfectly — *LM Studio resets the cache
whenever any request parameter changes*, wherever the change lands in the prompt. That predicts a full
re-prefill on both models. Qwen's 0.26 s refutes it: the parameter changed there too, and the cache held. So
the cause is the marker's position, and the templates are what decide it.

## What it means for Raven

- **On Qwen — the model the exhibit runs — the toggle is effectively free**, so flipping it mid-conversation
  costs nothing a visitor would see.
- **On Gemma it costs a full re-prefill.** 1.6 s on a 10k-token chat, and it scales with the conversation:
  a long one would stall visibly. Worth knowing before a demo toggles it on a chat that has been running a
  while.
- Nothing here needs handling in code. It is a property of the model's template, it cannot be worked around
  from the client, and the direction the toggle needs (*off*) works on both.

## The instrument, and the one that failed first

**Time, not `prompt_tokens`.** The first attempt used the cache-relative reporting documented in
`../prompt-size-cache-relative/` — LM Studio counting only what the cache did *not* hold — which would have
answered this directly. **It does not engage at this scale**: a 682-token prompt reported 682 warm and cold
alike, so the readout was flat and said nothing. That refines the neighbouring investigation's finding with
a bound it did not have: whatever triggers cache-relative counting, ~682 tokens is below it, and ~10k did
not trigger it either (both models reported a stable full count throughout).

So the readout became how long the backend takes to answer with `max_tokens=1`, which times prompt handling
and not generation. The A-A-B-B-A sequence exists so the switch is read against a *measured* warm baseline
for each condition rather than against an assumption.

One thing observed and unexplained, not load-bearing: on Gemma, switching *back* (`A3`) took 0.42 s —
faster than cold, slower than warm. Possibly a second cache slot partially reused. On Qwen the same step was
0.16 s, fully warm.

## Apparatus

| Script | What it answers |
|---|---|
| `probe_toggle_cache.py` | Whether flipping `reasoning_effort` invalidates the KV cache, and how much of it, by timing `max_tokens=1` requests through an A-A-B-B-A sequence on a ~10k-token conversation. Takes the backend URL as its argument; run it once per loaded model |

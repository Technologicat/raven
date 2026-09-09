# What flipping a toggle mid-conversation costs the KV cache

Two of Librarian's switches change the *request* rather than the conversation, and a demo operator flips
both mid-chat. What each costs turns on **where in the rendered prompt the thing it changes lands** — a
property of the chat template, so it is measured per model rather than reasoned about. The directory is
named for the first of the two.

## The thinking toggle: the whole KV cache on Gemma, nothing on Qwen

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

### Why, and the alternative that had to be ruled out

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

### What it means for Raven

- **On Qwen — the model the exhibit runs — the toggle is effectively free**, so flipping it mid-conversation
  costs nothing a visitor would see.
- **On Gemma it costs a full re-prefill.** 1.6 s on a 10k-token chat, and it scales with the conversation:
  a long one would stall visibly. Worth knowing before a demo toggles it on a chat that has been running a
  while.
- Nothing here needs handling in code. It is a property of the model's template, it cannot be worked around
  from the client, and the direction the toggle needs (*off*) works on both.

## The tool toggles: a full re-prefill the first time, and not free afterwards

**Measured 2026-09-09, against `qwen3.6-35b-a3b`** — the model the exhibit runs, and the one for which the
thinking toggle was free. Librarian's **Internet** and **Documents** checkboxes decide which tools are
declared on the next turn, so the same question applies to them: seven tools declared against two.

| | run 1, tool sets unseen | run 2, both sets seen |
|---|---|---|
| `A2` all tools, warm baseline | 0.19 s | 0.22 s |
| `B1` **the switch**, to local-only | **1.84 s** | **0.51 s** |
| `B2` local only, warm baseline | 0.20 s | 0.20 s |
| `A3` **switching back** | **0.55 s** | **0.51 s** |

**The first time a tool set is used, the whole conversation is re-processed.** `B1`'s 1.84 s against a
0.20 s warm hit is ~9× — and the arithmetic says it is the whole prompt rather than the changed part: the
two prompts differ by 500 tokens of declarations (10862 against 10362), and at the prefill rate those
numbers imply, 500 tokens is under a tenth of a second. So the invalidation starts near the top of the
prompt, which is where a template renders tool declarations — inside the first system turn, though not at
character zero.

**Afterwards it is cheaper, and still not free.** Once both prompts have been seen, either switch costs
about 0.5 s against a 0.2 s warm hit — so the backend is holding more than one prefix cache and a flip
lands on the *other* one rather than on nothing.

That also **closes the loose end left by the thinking probe**, recorded below as unexplained: Gemma's
switch-back at 0.42 s, faster than cold and slower than warm. A second cache slot is what it looks like,
and run 2 here shows the effect symmetrically, in both directions, on a model whose template makes the
switch a full invalidation.

### The control that had to be discarded

**Run 1's `A1` (3.27 s) is not a cold-prefill baseline and is not used as one.** It was the day's first
call to that model (Juha's observation), so it carries whatever the backend does on first contact —
weights paged in, a slot allocated — on top of the prefill being measured. Every claim above rests on `B1`
instead, which is equally a first sight of its prompt but arrives after two requests have already been
served, so the warm-up is spent.

Reading `A1` as the baseline would have made the switch look *cheaper* than a cold prefill (1.84 against
3.27) and invited the conclusion that only part of the prompt was re-read — which is the opposite of what
the token arithmetic says.

### What it means for Raven

- **Flipping Internet or Documents mid-chat costs a re-prefill of the conversation**, on Qwen as much as on
  anything: unlike the thinking toggle, this one is not template-dependent in a way that spares any model,
  because every template puts the declarations in the system turn.
- **It scales with the conversation**, so the pause grows as a chat gets long. On the exhibit's chats it is
  a moment; on a long session it is visible.
- **Alternating between two settings is cheaper than the first flip each way**, the backend keeping both
  prompts. Nothing to act on, but it explains why the cost does not reappear at full size every time.
- Nothing here needs handling in code. Raven already avoids this cost *within* a turn, for this same
  reason: past the tool-call cap it keeps the tools declared and answers a further call with a refusal
  rather than withdrawing them (`raven/librarian/README.md`, on the refusal rounds). The across-turns
  equivalent would mean declaring tools the user has switched off, which is a different trade and not
  obviously the right one.

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

One thing observed and unexplained here, not load-bearing: on Gemma, switching *back* (`A3`) took 0.42 s —
faster than cold, slower than warm. Possibly a second cache slot partially reused. On Qwen the same step was
0.16 s, fully warm. **The tool-toggle run above answers it** — a second cache slot is exactly what it shows,
symmetrically and in both directions.

## Apparatus

| Script | What it answers |
|---|---|
| `probe_toggle_cache.py` | Whether flipping `reasoning_effort` invalidates the KV cache, and how much of it, by timing `max_tokens=1` requests through an A-A-B-B-A sequence on a ~10k-token conversation. Takes the backend URL as its argument; run it once per loaded model |
| `probe_tools_cache.py` | The same question for the tool set Librarian's **Internet** and **Documents** switches decide — every tool declared against only the ones reaching nothing outside the process. Same sequence, same instrument, same argument. **Run it twice**: the first run measures the first sight of each tool set, the second what alternating between two known ones costs, and those are different numbers |

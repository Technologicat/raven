# Temporary context injects: measured behaviour across four local models

Every AI turn, `scaffold._perform_injects` puts material on the wire that the user never typed: the
current date and time, two behavioural reminders, and one message per RAG match. *Which role* those
take and *where* they sit had been argued on paper for months without being measured — the deferred
items on the subject mostly recorded "neither option has been measured" as their state.

This document is the measurement. It is deliberately **not** a work brief: the briefs describing the
resulting implementation will be archived once that work lands, whereas these numbers stay useful
afterwards — they are the baseline any future change to the inject machinery gets compared against,
and re-running the sweep costs about an hour of GPU time.

Harness: `briefs/summer_2026_librarian_extension/manual_tests/inject_shapes.py`. It imports the inject
strings from `raven.librarian.chatutil` rather than copying them, so it always measures what Raven
actually sends.

```
python inject_shapes.py http://localhost:1234 <model>           # all probes
python inject_shapes.py http://localhost:1234 <model> 3,5,6     # a subset
```

## Environment

Measured 2026-07-28, all models at 131072 context, temperature 0, across two machines:

| | 16 GB machine | eGPU machine (24 GB) |
|---|---|---|
| LM Studio | 0.4.19 (Build 2) | 0.4.20 (Build 1) |
| covers | Q1, Q2, Q4, Q5, and Q3 for Qwen3.5-9B | Q3 at realistic scale, Q7, and Q1 re-check |
| Gemma build | `lmstudio-community` | `google/gemma-4-26b-a4b` |
| Qwen3.5-9B build | unsloth | unsloth |

Qwen3.5-9B ran on both, which makes it the cross-version control. The larger models could not keep
prompt processing in 16 GB — spilling to system RAM caps prefill near 300 tok/s against 2.7 ktok/s on
the eGPU machine, which also has the faster card — so their runs live on the second machine.

| model | VRAM (16 GB machine) |
|---|---|
| Qwen3.5-9B | 6.89 GB |
| Qwen3.6-35B-A3B | 20.40 GB |
| Qwen3.6-27B | 18.54 GB |
| Gemma4-26B-A4B | 17.99 GB |

**The Gemma build situation is now resolved: 0.4.20 fixed it.** Under 0.4.19 the unsloth (UD) Gemma
build was unusable — note that it *loaded* fine and failed at **generation** time, minja choking on
Gemma's template, so "it loads" was never sufficient evidence either way. Under 0.4.20 it both loads and
generates: the Q1 tool-role re-check and the whole Q9 probe below ran on it without complaint.

Three Gemma packagings have now been exercised — `lmstudio-community` (0.4.19), `google` (0.4.20) and
unsloth UD (0.4.20) — and Q1 behaves identically on all three: a bare `tool` message is ignored, a
`tool`+synthetic-call pair is read. So that failure is a property of Gemma 4 itself, not of a packaging
or a backend version.

Practical note for telling builds apart: LM Studio requires vendor subdirectories on disk but does not
show them in `lms ls`, so two 26B-A4B entries are distinguishable only by size (unsloth's UD quant is
18.21 GB against google's 17.99 GB — larger because a dynamic quant compresses sensitive layers less).

**Which of these numbers a backend upgrade can move.** Anything decided by the model reading prompt
text — the placement sweep, reasoning spend, refusal and termination behaviour, the constraint check —
should be robust: a different backend that builds the same prompt gets the same answer. Anything
decided by how a *role* is rendered is not, because that goes through the loader's template handling,
which is what changes between builds. That covers the bare-`tool`-versus-`tool+call` result (Q1),
`system_end` acceptance, and Gemma's dependence on an LM Studio-specific template workaround. Re-run
`backend_capabilities.py` after any upgrade before trusting the Q1 row — and note that the separate
finding that LM Studio ignores `chat_template_kwargs` (upstream `lmstudio-bug-tracker#1559`) is
precisely the kind of thing a release fixes, which would change the thinking-toggle design that rests
on it.

Two properties of the harness worth knowing before reading any number below:

- **Thinking is suppressed by a closed-thought prefill** in the probes that only want an answer, since
  LM Studio ignores `chat_template_kwargs` and prefill is the only toggle available on the
  OpenAI-compatible endpoint. Probes 3, 5 and 6 leave thinking on.
- **An empty reply is not a refusal.** Reasoning can exhaust the token budget, and the resulting empty
  answer looks identical to a decline. Probe 6 exists to tell the two apart via `finish_reason`;
  earlier readings of probe 5 were wrong until it did.

## The shapes

| shape | what it is |
|---|---|
| `user` | what Raven ships today: one user-role message per inject, appended after the user's turn |
| `tool` | same position, `role="tool"` |
| `tool+call` | as `tool`, preceded by an assistant message carrying the `tool_calls` entry it answers |
| `folded` | inject text appended into the user's own message; no extra turns at all |
| `system_front` | inject text merged into the leading system message |
| `system_end` | system-role messages appended at the end — the original design, kept as a control |

## Q1. Does the tool role work, and is the synthetic call optional?

Probes 1 and 4. A planted fact ("the Kuiper-7 sensor array reports a baseline drift of 4.2 millikelvin
per hour") that no model can recall, so using it proves the material was read.

| model | bare `tool` | `tool` + synthetic `tool_call` |
|---|---|---|
| Qwen3.5-9B | used | used |
| Qwen3.6-35B-A3B | used | used |
| Qwen3.6-27B | used | used |
| Gemma4-26B-A4B | **ignored, and confabulates** | used |

**The synthetic call is load-bearing, not schema decoration.** Gemma 4 does not merely skip a bare tool
message — asked about the planted fact it invented a different, confident-sounding value ("±0.04% per
24-hour cycle"), which is the worst available failure mode for a retrieval path. With the preceding
assistant `tool_calls` message supplying a `tool_call_id` it reads the material correctly, at both the
front and the end of the history.

This settles the standing open question about whether to synthesize the call. The argument against —
that it puts a call in the assistant's mouth that it never made — is real, but costs less than a
backend-dependent silent-confabulation mode, and it is also the honest shape, since Raven *did* run
that search.

## Q2. Does the model narrate the injects instead of answering?

Probe 3: all three always-on injects around the message "Testing 1 2 3". Reasoning characters spent;
✗ marks a reply that addressed the injects rather than the user.

| shape | Qwen3.5-9B | Qwen3.6-35B-A3B | Qwen3.6-27B | Gemma4-26B-A4B |
|---|---|---|---|---|
| `user` (current) | 10260 (no reply in budget) | 1227 | 4077 | 6333 ✗ |
| `folded` | 10572 (no reply in budget) | 1402 | 4178 | 584 |
| `tool` | 5299 | 993 | 4701 | 641 |
| `system_front` | **4823** | 1082 | **874** | **331** |
| `system_end` | HTTP 400 (strict template) | 703 | 731 | 831 |

- **Folding does not fix the narration.** On Qwen3.5-9B it measures indistinguishably from the status
  quo (10572 vs 10260). It does fix it on Gemma 4. So the fold is a model-dependent half-measure, not
  a shape to standardize on — which contradicts what the deferred item assumed when it called folding
  the "better shape".
- **Gemma 4 reproduces the self-reference bug verbatim.** In the `user` shape its reasoning opens
  *"User's most recent message: '[System information: NOTE: Please answer based on the information
  provided in the context only.]'"* and it replies *"Understood. I will only use the information
  provided in the context."* — answering the inject instead of the user. Second model family, same
  wart, so this is not a Qwen quirk.
- **`system_front` is cheapest or near-cheapest on all four, and never narrated.**
- **The reasoning-cost spread between shapes is mostly a small-model effect.** On Qwen3.5-9B the
  `user`/`folded` shapes cost roughly twice `tool`/`system_front`; on Qwen3.6-35B-A3B the whole spread
  is 703–1402, which is noise. Smaller models in this family spend far more reasoning than larger
  ones, plausibly a distillation artifact.

## Q3. Does retrieved material have to sit at the front?

Probe 4. The front placement (`history.insert(1, ...)`) dates from Qwen 3.0, which would not engage
with material injected late; it costs a full KV-cache prefix rebuild every turn.

Every model used the planted fact at **both** the front and the end, in the `user` role — and in the
`tool` role too, except Gemma, which needs `tool+call` (Q1). The Qwen-3.0-era constraint did not
reproduce anywhere.

**Caveat, and it is a real one:** this used one short needle in a nearly empty context (~2450 prompt
tokens), which is not the case that motivated the front placement. Treat it as "the old constraint is
not obviously still true" rather than "placement no longer matters". `tool+call` was measured at both
positions on Gemma only; the Qwens were measured bare.

The realistic-scale re-test is `rag_placement.py`, which models HybridIR's actual output rather than a
guess at it. Two things make "twenty chunks" the wrong mental picture:

- **A result is not a chunk.** `chunk_size = 1000` characters with `overlap_fraction = 0.25` gives a
  sliding window of stride 750, and `merge_contiguous_spans` seamlessly joins adjacent matched chunks
  from the same document before `k` is applied. So a result spanning *n* chunks runs about
  `1000 + (n-1)·750` characters, and `docs_num_results = 20` counts merged results, not chunks.
- **Chunk length is what matters, not the source.** In the ~10k-document case Raven is aimed at, one
  abstract is one *document*, often short enough that merging returns most of it.

Modelled that way, k=20 comes to ~29000 characters (~7-8k tokens) — an order of magnitude more than
the Q3 probe used, and the number that should be quoted for the configured case.

### Result at realistic scale, all four models

Twelve conditions per model per corpus size: needle first / middle / last, material in the `user` or
`tool+call` role, placed at the front or the end, each with a needle-absent control. Run on the machine
with the 24 GB eGPU (a newer LM Studio; exact build not captured in the log).

| role, placement | k=20 | k=100 |
|---|---|---|
| `user` @ front | 12/12 | 9/9 |
| `user` @ end | 12/12 | 9/9 |
| `tool+call` @ front | 12/12 | 9/9 |
| **`tool+call` @ end** | **11/12** | **7/9** |

81 of 84 found overall — and **81 of 81 excluding `tool+call` at the end**. No control invented the
figure anywhere, so every hit is genuine. There is no depth effect and no corpus-size effect: k=100
(~32k tokens) behaves like k=20 (~6.5k).

**The one broken cell is `tool+call` placed at the end, and it fails in a specific way.** The misses are
not the model overlooking the material; the model emitted *another* `search_documents` tool call instead
of answering. Six replies did this — three that scored as misses, three controls:

| model | k | where |
|---|---|---|
| Qwen3.6-27B | 20 | needle last, and the control |
| Qwen3.6-27B | 100 | needle first, needle middle, and the control |
| Qwen3.6-35B-A3B | 20 | the control |

This is the trained agent-loop pattern doing its job. When a tool result is the last thing in the
context, the model's next decision is "answer, or call again" — and the synthetic call we add to keep
Gemma honest is exactly what makes the history read as mid-loop. Qwen3.5-9B and Gemma 4 never did it;
Qwen 3.6 does, the dense 27B most.

Two practical notes. The 27B's emitted calls were *malformed* (`<function/search_documents>` rather than
`<function=...>`), so they would likely fail Raven's tool-call parsing rather than execute cleanly. And
in a real turn a well-formed one would simply run a second search — a wasted round trip, or a loop.

**Consequence for the design: the tool role and the end placement are individually safe and jointly
unsafe.** Three combinations are clean on all four models — `user` at either position, and `tool+call`
at the front — so the choice is between keeping the honest tool role (front placement, KV-cache rebuild
retained) and recovering the cache (user role at the end, narration wart retained).

### The fifth shape resolves it: place the material *before* the user's question

`system → conversation → material → question`, rather than after the question. The prefix ahead of the
material is still unchanged, so the cache benefit of a late insert survives; but the last message is the
user's question rather than a tool result, which should remove the "call again" temptation.

Measured on all three larger models, including Qwen3.6-27B — the worst offender, source of all three
misses and half the stray tool calls:

| model | `user` @ before | `tool+call` @ before |
|---|---|---|
| Qwen3.6-27B | 3/3 · 3/3 | 3/3 · 3/3 |
| Qwen3.6-35B-A3B | 3/3 · 3/3 | 3/3 · 3/3 |
| Gemma4-26B-A4B | 3/3 · 3/3 | 3/3 · 3/3 |

(each cell k=20 · k=100)

**36 of 36, every control clean, and not one `<tool_call>` leak.** Compare Qwen3.6-27B at `tool+call` @
end: 2/3 and 1/3, with stray calls in both, plus a stray call in its control at each size.

So the trade-off disappears. `before` keeps the honest tool role (and with it Gemma's requirement for a
synthetic call), keeps the KV-cache prefix stable, and avoids the agent-loop misfire. **This is the
recommended placement**, replacing both the current front insert and the end placement considered above.

### How the results are packaged: one tool message, not one per result

The OpenAI schema pairs a `tool` message with a single `tool_call_id`, so N results should arrive as one
message, not as N messages sharing an id. Measured at `before`, k=20:

| model | one message per result | **one merged message** |
|---|---|---|
| Qwen3.6-27B | 3/3 | 3/3 |
| Gemma4-26B-A4B | 3/3 | 3/3 |
| Gemma4-E4B | 1/3 | 2/3 |

Merged is never worse and is better on the weakest model, so **emit one tool message carrying all
matches**. (Gemma4-E4B is the only model where the tool role costs accuracy at all — `user` role gets
3/3 there. It is a 4B-effective model; every larger one is fine either way.)

### The end-placement failure, re-verified

The misses that ruled out the end placement were first measured with a 400-token cap and no
`finish_reason`, which is exactly the setup that has produced three false readings elsewhere in this
document. Re-run with a 1500-token budget and the finish reason surfaced, on Qwen3.6-27B:

| @ end | k=20 | k=100 |
|---|---|---|
| `tool+call` | 2/3 | 2/3 |
| `tool+call-merged` | **0/3** | **1/3** |

Every failure completed normally (`finish_reason=stop`) with a `<tool_call>` block as its answer, so
these are genuine, not truncations. And merging — which *helps* at `before` — makes matters worse here,
which fits the diagnosis: a single clean tool result immediately before the generation point is an even
stronger "your tool output has arrived, now what?" cue than a ragged pile of them.

## Q8. Does a backend upgrade change any of this? (LM Studio 0.4.20 Build 1)

`backend_capabilities.py` re-run on the newer build, against Qwen3.6-27B. Everything reproduced:

- **`chat_template_kwargs` is still ignored.** `enable_thinking=False` still produced 712 characters of
  reasoning where honouring it would produce zero. (Read the reasoning lengths, not the probe's
  "all three alike" summary line — that check compares three stochastic samples, since `chat()` does not
  pin temperature, so the raw numbers 679 / 712 / 916 are noise around "thinking happened regardless".)
  Upstream `lmstudio-bug-tracker#1559` is therefore still open, prefill remains the only thinking toggle
  on the OpenAI-compatible endpoint, and the design resting on that needs no rework.
- **`min_p` honoured** though undocumented; unknown parameters still accepted with HTTP 200, so a status
  code still proves nothing.
- **Anthropic-compat native thinking control works**: `disabled` → `['text']`, `enabled` →
  `['thinking', 'text']`. Prefill works on both endpoints. Streaming works.
- **Reasoning history does not feed back on Qwen3.6-27B by any route.** Prompt tokens were 38 / 38 / 38 /
  40 for baseline, a `reasoning_content` sibling, `+preserve_thinking=True`, and a native Anthropic
  thinking block — against ~700 tokens of reasoning offered. Nothing landed. `preserve_thinking` exists
  in the 3.6 template but rides on `chat_template_kwargs`, so it is unreachable in practice.

So the version-sensitivity caution above was right in principle and did not bite: no measurement in this
document changed under the upgrade.

### The same probe on Qwen3.5-9B, on the other machine

Twelve conditions per corpus size: needle first / middle / last, material in the `user` or `tool+call`
role, placed at the front or the end. Each with a needle-absent control.

| corpus | prompt size | needle found | controls |
|---|---|---|---|
| k=20 (configured) | ~6.5k tokens | **12 / 12** | 4 / 4 clean |
| k=100 (filled) | ~32.3k tokens | **12 / 12** | 4 / 4 clean |

No position effect, no depth effect, no role effect, and no control invented the figure — so no hit was
a lucky guess. **On this model the front placement buys nothing**, and it costs a full KV-cache prefix
rebuild every turn.

Qwen3.5-9B is the weakest model in the set, so this is not a "the small one coped, the big ones surely
will" argument — it is the model whose attention over 32k tokens should be least trustworthy, which is
what makes a clean sweep informative.

**This doubles as the cross-version control.** The 9B was run on both machines, on two different LM
Studio builds, and produced identical verdicts (12/12 at k=20 on each, same conditions). Q1 also
reproduced exactly across the two builds, including Gemma's bare-`tool` failure. So the
version-sensitivity worry recorded above did not materialize for these results — though it remains
correct in principle, and `backend_capabilities.py` has *not* yet been re-run on the newer build, so
the `chat_template_kwargs` question is still open.

## Q7. Does the replacement wording still refuse what it *should*?

Material about Kuiper-7 is supplied and the question asks about Kuiper-9 — neither in the documents nor
general knowledge, so the only correct answer is that we do not know. This is the half Q4 does not
measure, and a wording that never refuses anything would score perfectly there while being useless.

**Answered: all four models, all three wordings, declined correctly.** Not one invented a figure for
Kuiper-9. Representative reply under the proposed wording:

> The provided information only contains data for the **Kuiper-7** sensor array. There is no mention of
> a "Kuiper-9" sensor array or its baseline drift.

So the replacement keeps the anti-confabulation behaviour while dropping the over-refusal measured in
Q4. Taken together, Q4 and Q7 clear it on both halves, and it can be adopted.

Worth noting what this also shows: the *system prompt alone* ("using the provided knowledge-base
material") was enough for every model to decline correctly in the no-reminder control. The reminder is
not carrying the anti-confabulation behaviour by itself, which is an argument for the mildest wording
that does the job rather than the firmest.

**Two verdicts in this probe's first runs were the detector's fault, not the models'**, and are worth
recording so they are not mistaken for findings later: a correct decline reading "do **not** contain"
was missed by a phrase list containing only "does not contain", and the fabrication test — "the reply
contains a digit" — fired on the digits in *Kuiper-7* and *Kuiper-9* themselves. The detector now
matches inflections by regex and only counts a drift figure that differs from Kuiper-7's own. The
printed reply remains what actually decides; the verdict is a hint.

## Q4. The "answer from context only" reminder

Probes 5 and 6. Asked "What is 2+2?" with no context supplied — the general-knowledge question a live
audience asks against an empty or unrelated document database.

| wording | Qwen3.5-9B | Qwen3.6-35B-A3B | Qwen3.6-27B | Gemma4-26B-A4B |
|---|---|---|---|---|
| no reminder (control) | 1422 ✅ | 884 ✅ | 610 ✅ | 279 ✅ |
| **current** | **52796 ❌ never terminates** | 10808 ✅ | 5025 ⚠️ | 4704 ❌ refuses |
| "Prefer information from the context when it is relevant. Say so if you are drawing on general knowledge instead." | 3935 ⚠️ | 1025 ⚠️ | 2375 ⚠️ | 1813 ⚠️ |
| "Base claims about the provided documents on those documents. Answer general questions normally." | 1481 ✅ | 3743 ✅ | 988 ✅ | 759 ✅ |

✅ answers cleanly · ⚠️ answers but volunteers an unwanted "(drawing on general knowledge)" hedge, or
leads with "the provided context does not contain…" before answering · ❌ no usable answer.

The current wording — *"Please answer based on the information provided in the context only."* — is
bad on **every** model tested, costing 5–37× the control's deliberation. Qwen3.5-9B never finishes:
52796 characters of reasoning against a 16000-token budget, `finish_reason=length`, no reply at all.
Gemma 4 answers "The provided information does not contain the answer to this question."

**The models are right and the instruction is wrong.** "Answer based on the context only" does
prohibit answering from general knowledge, so a model that notices 2+2 is general knowledge, that no
context was supplied, and that it was told to use context only, is obeying a self-contradictory
instruction rather than misbehaving. It would be nicer if it reported the contradiction instead of
looping on it, but that is a lot of executive function to ask of a 9B in mid-2026.

The fourth wording is the recommended replacement: within noise of sending no reminder at all on three
of the four models, and no hedging.

**This only measures over-refusal, which is half the question.** The reminder's actual job is
anti-confabulation, and a wording that never refuses anything would score perfectly above while being
useless. Probe 7 measures the other half: material about Kuiper-7 is supplied and the question asks
about Kuiper-9 — neither in the documents nor general knowledge, so the only correct answer is that we
do not know. A replacement that answers 2+2 happily but also invents a figure for Kuiper-9 is not an
improvement, it is a different failure. Do not adopt the new wording on the strength of Q4 alone.

## Q5. Does the tool role weaken an instruction?

Probe 2, Qwen3.6-35B-A3B, thinking suppressed. With the injects in the `user` role the model refused a
trivial general-knowledge question outright — *"I cannot answer this question because the provided
context does not contain information about the capital of France"* — and in the `tool` role it answered
normally.

That is **not** the tool role fixing a bug. It is the model weighting the reminder less than it should,
landing on the outcome we want for the wrong reason. The consequence for design: anything we actually
want obeyed must not go in the tool role. Data-like injects, which only need to be *read*, are
unaffected — and that split (data-like to `tool+call`, instruction-like elsewhere) is the shape the
measurements support.

Note also that this refusal cost **zero** reasoning tokens. The model did not need to deliberate to
reach the conclusion; it simply followed the instruction as written.

## Template families, confirmed behaviourally

`system_end` returns HTTP 400 on Qwen3.5 (`System message must be at the beginning`, which is really a
*count* rule — the guard is `not loop.first`), and works on Qwen3.6, which dropped the guard, and on
Gemma 4. This is the strict/permissive split, previously read off the Jinja, now observed.

## Implementation note

`chatutil.linearize_chat` returns `payload["message"]` — the stored dicts themselves, not copies. The
list is fresh, so appending to it is safe, but merging inject text into an existing message must copy
that message first or it writes through to the datastore and corrupts the stored system prompt. This
is a quiet advantage for any shape that only *adds* messages.

## Q9. Telling the model what day it is

`datetime_inject.py`. The datetime inject is the odd one out: **data rather than instruction**, so it
only has to be read; and it **changes every turn**, so it can never sit in a stable cached prefix. Two
questions per shape, because reciting a date back is easier than using it — the second asks for a day
count to a date 60 days out, which a model reasoning from its training prior gets wildly wrong.

| shape | Qwen3.5-9B | Qwen3.6-27B | Qwen3.6-35B-A3B | Gemma4-26B-A4B | Gemma4-E4B |
|---|---|---|---|---|---|
| none | ✗ · ✗ | ✗ · ✗ **"33"** | ✗ · ✗ | ✗ · ✗ | ✗ · ✗ |
| `user` | ✓ · ✓ | ✓ · ✓ | ✓ · ✓ | ✓ · ✓ | ✓ · ✓ |
| bare `tool` | ✓ · (budget) | ✓ · ✓ | ✓ · ✓ | ✗ · ✗ | ✗ · ✗ |
| **`tool+call`** | ✓ · ✓ | ✓ · ✓ | ✓ · ✓ | ✓ · ✓ | ✓ · ✓ |
| `system_front` | ✓ · ✓ | ✓ · ✓ | ✓ · ✓ | ✓ · ✓ | ✓ · ✓ |

(recite · compute; ✓ compute means the 60-day count was correct)

**The synthetic-tool-call form works everywhere, and the model genuinely *uses* the date** rather than
reciting it — every ✓ in the compute column is a correct day count.

**Ungrounded, every model is wrong, and the interesting part is *how*.** Gemma states a confident
"Wednesday, May 22, 2024". Qwen3.6-27B refuses to give a date — "I don't have access to real-time
data" — and then, asked for a day count, silently assumes one anyway and answers **"33"**: a confident
wrong number with no hedge, which is the genuinely dangerous shape. Qwen3.6-35B-A3B instead never
terminates on that question; its reasoning tail reads

> "…Since I don't have it, I'll assume today is the date the user is asking. I'll just say 'I need
> today's date'…"

— oscillating between assuming and asking, and burning the budget without committing. So the larger MoE
notices the contradiction and stalls where the dense 27B doesn't notice and confabulates. Two data
points, so treat the ordering as an observation rather than a rule.

**Bare `tool` splits exactly as it does for RAG material**: Qwen reads it, Gemma does not — and on
Gemma the reasoning length under bare `tool` is *identical* to no inject at all (759 / 2169 chars on
E4B), so the message is not reaching the model rather than being read and discounted. One Qwen3.5-9B
cell is marked "(budget)" rather than ✗: it hit `finish=length` mid-reasoning, which is inconclusive,
not a failure.

Note the dispute markers survive into the reasoning even where the answer is right — Qwen3.6-35B-A3B
mentions 2024 or "future date" under `user`, bare `tool` and `tool+call` alike. `system_front` was the
only shape with none, and also the cheapest. Since the datetime inject cannot live in the system block
(it changes per turn, so it would invalidate the prefix), that deliberation is the price of the
placement.

## Still unmeasured

- Whether the two reminders keep steering from the leading system block. `system_front` loses recency,
  which is the one thing late placement buys. The reason late placement was chosen in the first place
  — DeepSeek-R1 distills needing it for multi-turn to work — dates from early 2025 and is **not worth
  designing around any more**: those models are two generations behind, and nothing in this sweep
  suggests current ones need the help. Worth confirming the reminders still bite from the front, but
  not worth preserving late placement on their account.
- oobabooga, which is not installed on this machine and stale elsewhere. Every number here is LM
  Studio.
- Whether any of this transfers to non-local backends.

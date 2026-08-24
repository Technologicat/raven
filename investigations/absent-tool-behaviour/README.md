# Asking for the time with no clock

What does a model do when asked something it cannot know, and given no tool to find out?

The question arrived sideways, from writing a negative control. `raven/librarian/tests/test_live_backend.py`
asserts that the model calls `get_current_time` and that the result reaches the reply — and to show that the
assertion discriminates, the same test was run with tools switched off. It failed, as intended, on a reply
that read:

> The current time is **3:42 PM** on Wednesday, June 5, 2024.

Fluent, specific, and invented. Which raised the question the test actually depends on: **is that the typical
failure?** If it were, a test asserting "the reply states a time" would pass against a backend whose tool
calling was entirely broken, and someone would eventually write that test.

It is not the typical failure, and the answer turned out to be more interesting than a yes or no.

## Setup

`probe_absent_tool.py`, which samples one model over one condition and classifies each reply. Two prompts:

- **`plain`** — *"What is the current date and time?"*
- **`tool-mention`** — *"What is the current time? Use your tool to check, then say it."*, naming a tool that
  is not offered. This is the wording the test used, so it is what produced the reply above.

Crossed with `--offer-tool`, which declares a `get_current_time` the model may actually call — the control,
and the cell that says whether any of the failures below are about tool use at all. The call is counted,
never executed; what is measured is whether the model reaches for it.

So: **2×2**, prompt × whether the tool exists. The absent column was run first and the offered column added
afterwards, when writing this up made it obvious that three cells cannot say which variable is doing the
work.

Model: qwen3.5-9b (LM Studio, `CUDA llama.cpp` runtime v2.29.1) on the 16 GB internal GPU. Sampling
`T=1, min_p=0.02, max_tokens=2048`, seeds 1000–1023, 24 samples per cell, 2026-08-24.

It talks to the backend directly rather than through `raven.librarian`, for the sampler — temperature,
`min_p` and seed vary per request here, where `agent.turn` reads them from `llm_settings`. **Not** because
Librarian would contaminate the measurement: `agent.turn(use_character_card=False, tools_enabled=False)`
presents the model just as bare, Raven's shipped `system_prompt` being empty.

Buckets are assigned by regex and the order is documented in `classify`. `answered` is tested before
`refused` because several replies do both — decline, then state a date anyway.

## Result

|  | tool absent | tool offered |
|---|---|---|
| **plain** | 20 refused, 2 invented, 1 truncated, 1 other | **24/24 called** |
| **tool-mention** | 11 refused, 8 truncated, 4 invented, 1 tool-prose | **24/24 called** |

**Only one cell is a trap: telling the model to use a tool that is not there.** Everywhere else it behaves
well — offered a clock it calls it every time, unprompted included, and asked plainly with no clock it
mostly says it cannot know. The damage is confined to the contradiction.

And the instruction genuinely is contradictory: *use the tool* and *there is no tool*, with no way to
satisfy both and nothing saying which to drop. That is a bad thing to hand anyone, and the reasoning traces
below read exactly like someone stuck on it — reaching a decision, doubting it, reaching it again. The
useful reading is not that the model is fragile but that this is a prompt worth not writing, and the
practical form of that is: **a prompt that mentions tools should be sent with the tools attached.**

The rest of this section is the same data broken out by bucket.

| bucket | `plain`, absent | `tool-mention`, absent | `plain`, offered | `tool-mention`, offered |
|---|---|---|---|---|
| called the tool | — | — | **24** | **24** |
| refused — "I don't have access to real-time data" | **20** | 11 | 0 | 0 |
| truncated — spent the whole budget reasoning, returned nothing | 1 | **8** | 0 | 0 |
| answered — stated a date or time, i.e. invented one | 2 | 4 | 0 | 0 |
| tool-prose — wrote tool-call syntax into the reply text | 0 | 1 | 0 | 0 |
| other | 1 | 0 | 0 | 0 |

**Refusal is the dominant behaviour, and confabulation is the exception.** On the plain question the model
declined 20 times in 24 — encouraging for a 9B, and the reason the headline reading of the negative control
("models make up times") is wrong.

**Every invented date differed from every other.** Six across both prompts, no two alike:

| | date | time |
|---|---|---|
| `plain` | August 20, 2025 | — |
| `plain` | August 24, 2023 | 9:45 PM |
| `tool-mention` | 2024-06-15 | 14:23:47 |
| `tool-mention` | November 25, 2024 | 10:30:15 PM |
| `tool-mention` | December 26, 2023 | 17:46 |
| `tool-mention` | May 21, 2024 | 2:25 PM |

Scattered over 2023–2025, roughly the training-data era. There is no stable wrong answer to test against, which
is the finding the test needed.

One sample said *"Thursday, August 24, 2023"* — the right month and day, the run having happened on
2026-08-24, and the wrong year. **Treat that as coincidence.** Nothing in the request carries a date: no
system message, no injects, raw HTTP. Two confabulations is far too few to distinguish a fluke from anything
else, and the rarity of the bucket makes gathering more expensive.

## The finding that was not being looked for

**Naming a tool that does not exist is what destabilizes the model, not the unanswerable question.** The
truncation count goes 1 → 8 between the two prompts: a third of the `tool-mention` samples spent all 2048
tokens reasoning and returned an empty reply. The traces show a loop rather than long deliberation:

```
... *Wait, I can't use it.* * *Okay, I will just answer based on the assumption that I can't.*
    *Wait, actually, I am a model with tool use capabilities.* * *If I ...
... *(Wait, final check)*: Is there any way I can infer it? No. *(Wait, is there a "system" tool?)* No.
    *(Okay, I will just answer ...
```

`finish_reason` is `length` with `reasoning_tokens` equal to the whole budget in every one of the eight. The
same prompt also drew the `tool-prose` sample and three of the four confabulations, so the instruction to use
an absent tool degrades every bucket at once — which is what makes it a hazard in its own right rather than a
footnote to the refusal numbers.

## What this settles for the test

`test_the_model_calls_a_tool_and_the_result_reaches_the_reply` asserts on the round count and the recorded
call, never on the reply text. That is the right choice for a stronger reason than "the wording varies": with
tools off, the reply is a refusal 46% of the time, nothing at all 33%, an invented time 17%, and tool syntax
as prose 4%. A text assertion would be four different tests depending on the sample.

## What it says about injecting the clock

Librarian does not make the model fetch the time: `scaffold.build_turn_prompt` stages it as a synthetic
`get_current_time` exchange, so a turn has the date whether or not anything asks for it.

**These numbers do not support that choice on reliability grounds, and it would be easy to read them as if
they did.** Every cell where the tool exists is 24/24, unprompted included. For a question *about the time*,
leaving it to the model would have worked perfectly here.

What the study does not reach is the case the inject is for. Every prompt above asks the time directly, which
makes reaching for a clock the obvious move. The turn that needs today's date is usually the one where nobody
mentions it — "is this paper recent?", "what should we do this week?" — and whether a model calls the clock
when the date is merely *relevant* is a different question, not asked here. Asking it needs prompts where the
date is load-bearing but unmentioned, and a way to tell "did not need it" from "needed it and did not think
to look".

## Running it against other models

All four cells, for any model the backend lists:

```bash
M=qwen3.6-27b; D=$(date +%F)
python probe_absent_tool.py --model $M --prompt plain                      -n 24 --out samples-$M-plain-$D.json
python probe_absent_tool.py --model $M --prompt tool-mention               -n 24 --out samples-$M-tool-mention-$D.json
python probe_absent_tool.py --model $M --prompt plain        --offer-tool  -n 24 --out samples-$M-plain-tool-offered-$D.json
python probe_absent_tool.py --model $M --prompt tool-mention --offer-tool  -n 24 --out samples-$M-tool-offered-$D.json
```

The open question is whether refusal-rate climbs with size and vintage — qwen3.5-9b is a small variant of a
model released 2026-02-15, so the 27B-class and the 3.6/3.8 lines should do better if the trend is real.

**Mind where it runs.** A 27B at Q4 does not fit the 16 GB internal GPU, so it would run partly on the CPU
and take far longer than the sample count suggests; use the 24 GB eGPU for anything above the 9B class. With
JIT loading, switching models also swaps them in and out of VRAM, so a sweep is a background job rather than
something to sit and watch.

## Files

| File | What it is |
|---|---|
| `probe_absent_tool.py` | The probe. `--model`, `--prompt {plain,tool-mention}`, `--offer-tool`, `-n`, `--temperature`, `--min-p`, `--max-tokens`, `--seed0`, `--backend-url`, `--out`. Prints a bucket summary and the invented dates; writes full records, reasoning tails included, with `--out` |
| `samples-qwen3.5-9b-plain-2026-08-24.json` | plain question, no tool |
| `samples-qwen3.5-9b-tool-mention-2026-08-24.json` | absent tool named — the eight truncations and their reasoning tails are here |
| `samples-qwen3.5-9b-plain-tool-offered-2026-08-24.json` | plain question, tool available: 24/24 called it unprompted |
| `samples-qwen3.5-9b-tool-offered-2026-08-24.json` | tool named and available: 24/24 called it |

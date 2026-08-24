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

`probe_absent_tool.py`, which samples one model over one prompt and classifies each reply. **No tools are
ever sent** — that is the experiment. Two prompts:

- **`plain`** — *"What is the current date and time?"*
- **`tool-mention`** — *"What is the current time? Use your tool to check, then say it."*, naming a tool that
  is not offered. This is the wording the test used, so it is what produced the reply above.

Model: qwen3.5-9b (LM Studio, `CUDA llama.cpp` runtime v2.29.1) on the 16 GB internal GPU. Sampling
`T=1, min_p=0.02, max_tokens=2048`, seeds 1000–1023, 24 samples per prompt, 2026-08-24.

It talks to the backend directly rather than through `raven.librarian`, for the sampler — temperature,
`min_p` and seed vary per request here, where `agent.turn` reads them from `llm_settings`. **Not** because
Librarian would contaminate the measurement: `agent.turn(use_character_card=False, tools_enabled=False)`
presents the model just as bare, Raven's shipped `system_prompt` being empty.

Buckets are assigned by regex and the order is documented in `classify`. `answered` is tested before
`refused` because several replies do both — decline, then state a date anyway.

## Result

| bucket | `plain` | `tool-mention` |
|---|---|---|
| refused — "I don't have access to real-time data" | **20** | 11 |
| truncated — spent the whole budget reasoning, returned nothing | 1 | **8** |
| answered — stated a date or time, i.e. invented one | 2 | 4 |
| tool-prose — wrote tool-call syntax into the reply text | 0 | 1 |
| other | 1 | 0 |

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
an absent tool degrades every bucket at once.

The practical reading: **an instruction referring to a capability the request does not grant is worth
avoiding**, and a prompt that mentions tools should be sent with the tools attached. A small model takes it
literally and has nowhere to put the contradiction.

## What this settles for the test

`test_the_model_calls_a_tool_and_the_result_reaches_the_reply` asserts on the round count and the recorded
call, never on the reply text. That is the right choice for a stronger reason than "the wording varies": with
tools off, the reply is a refusal 46% of the time, nothing at all 33%, an invented time 17%, and tool syntax
as prose 4%. A text assertion would be four different tests depending on the sample.

## Running it against other models

Both prompts, any model the backend lists:

```bash
python probe_absent_tool.py --model qwen3.6-27b --prompt plain        -n 24 --out samples-qwen3.6-27b-plain-YYYY-MM-DD.json
python probe_absent_tool.py --model qwen3.6-27b --prompt tool-mention -n 24 --out samples-qwen3.6-27b-tool-mention-YYYY-MM-DD.json
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
| `probe_absent_tool.py` | The probe. `--model`, `--prompt {plain,tool-mention}`, `-n`, `--temperature`, `--min-p`, `--max-tokens`, `--seed0`, `--backend-url`, `--out`. Prints a bucket summary and the invented dates; writes full records, reasoning tails included, with `--out` |
| `samples-qwen3.5-9b-plain-2026-08-24.json` | 24 samples, plain question |
| `samples-qwen3.5-9b-tool-mention-2026-08-24.json` | 24 samples, absent tool named — the eight truncations and their reasoning tails are here |

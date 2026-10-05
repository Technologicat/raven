# Temporary context injects — what shape should they take?

Every AI turn, Librarian puts material on the wire that the user never typed: the date and time, behavioural
reminders, and retrieved documents. This measured how that material should be shaped and placed, across four
local models.

**The write-up is [`context-inject-shape-measurements.md`](context-inject-shape-measurements.md).** It is the
result; the scripts below are how it was produced.

## The probes

All are **manual live probes, not pytest tests** — each needs a running backend with a model loaded, which is
why they were never part of the suite.

| Script | What it answers |
|---|---|
| `inject_shapes.py` | The main sweep: which inject shape (`user`, `system`, tool-message, folded) does a model handle best? |
| `assembled_shape.py` | Does Raven's *assembled* inject still behave the way the sweep predicted, once all the pieces are in place? |
| `datetime_inject.py` | Can we tell the model what day it is, and will it believe us over its own priors? |
| `absent_fact.py` | Asked something the retrieved documents do not answer, what does the model do? |
| `rag_placement.py` | At realistic corpus scale, does retrieved material still have to sit at the front of the history? |
| `backend_capabilities.py` | What does a given backend's HTTP API actually support, as opposed to advertise? |
| `reminder_placement.py` | The grounding reminder sent only with material (A), or always, worded conditionally (C′): does it change the answers, and what does A cost the KV cache? See below. |

`assembled_shape.py` is worth a note: the write-up does not name it, and it was recovered only by noticing it
landed in the same commit (`ef0ce0c`). It is listed here so that never has to be rediscovered.

## 2026-08-10: the two probes were measuring a stand-in prompt, and it hid a defect

`absent_fact.py` and `assembled_shape.py` are the two probes that build their history through Raven's own
code rather than reimplementing the shape — which is what the measurements here rest on. Both nevertheless
hand-assembled their `settings`: an `env` with seven of the twenty-one fields `llmclient.setup` returns, and
`system_prompt="You are a helpful assistant."` in place of Raven's real one. So the injects were real and
**the prompt they were injected into was not**, which is the failure this whole directory exists to avoid,
one layer up from where it was being watched. Neither docstring said so; both claimed to measure what Raven
actually sends.

Both now call `llmclient.configure`, which builds the genuine settings object without contacting a backend.
That is the durable gain here, and it holds regardless of what any individual run returns.

Re-running them once against qwen3.6-35b-a3b, two outputs differed from the forged-settings runs:

| check | with the stand-in prompt | with Raven's real prompt |
|---|---|---|
| `absent_fact`, as-shipped, T=0 | `finish=stop`, 4430 chars of reasoning | `finish=length`, 31726 chars — never produced a reply |
| `assembled_shape` [2], absent fact | declined cleanly | emitted literal `<tool_call> <function=search_documents>` text |

**Read that table as two anecdotes, not as a result.** It is one sample per cell, and the next section says
why one sample is worth nothing here.

**The measurements above stand as taken** — this section is the note that says what they were taken against.
Anything re-measured from here on is measured against the real settings.

## T=0 is not reproducible on this stack (2026-08-11)

Re-ran `absent_fact` with three samples per arm, and with the probe sending Raven's own request template
(`settings.request_data`) so that the samplers are Raven's too — `min_p=0.02` ahead of the temperature,
where before it sent a bare temperature and no `min_p`, i.e. a distribution nobody runs. Against
qwen3.6-35b-a3b, tools not declared:

| variant | T=0 (3 samples) | T=1 (3 samples) |
|---|---|---|
| as-shipped | reasoning **2484 / 30757 / 29684** chars — two of three hit the 8000-token cap with no reply | 2/3 asked to search again |
| closing-note | 2876 / 2492 / 2355 chars, all answered | 1/3 asked to search again |
| no-synthetic-call | 2849 / 1212 / 2472 chars, all answered | 0/3 asked to search again |

**The methodological finding comes first, because it invalidates the table above it.** Three identical
requests at T=0 produced 2484, 30757 and 29684 characters of reasoning. Greedy decoding is deterministic
only given identical numerics, and on a GPU it is not: kernel selection and float non-associativity can flip
a near-tie, after which the trajectory diverges completely. A generation this long is thousands of sampling
decisions, so it is the *least* reproducible thing here rather than the most. **Single-sample T=0 claims are
therefore worthless on this stack**, which is exactly what the one-sample table above was making.

What does survive, at 3 of 4 samples counting a fourth run made while timing the probe: **the as-shipped
wording is the one that runs away**, and the two alternatives do not, 0 of 3 each. The reasoning lengths
separate cleanly — roughly 2.5k when it answers, roughly 30k when it does not, with nothing in between.

**This inverts the reason `closing-note` was rejected — on a different model, which is the catch.** That
rejection was measured on Qwen3.6-27B, where `closing-note` was the variant burning 29000 characters at
T=0. On 35B-A3B it is clean and as-shipped is the one that burns. So the rejection rationale is
model-specific, and it was never re-checked against the models actually in service.

**Which is plural, and that is the real requirement here.** An inject wording ships to every model Raven
supports, so "which wording is best" is only answerable across the supported set — and a variant that is
clean on a 35B MoE and pathological on a 4B is the failure mode this whole directory exists to catch.

The arms are therefore the tiers in `../../briefs/reference/model-lineup-autumn-2026.md`, which is the
authority on what those are: Qwen3.5-4B, Qwen3.5-9B, and both 24 GB options, Qwen3.6-27B dense and
Qwen3.6-35B-A3B. **Set the arms by what a user may plausibly run, not by what is loaded here.** The two 24 GB
options are alternatives at the *same* tier, not one superseding the other — dense against MoE, at 18.54 and
20.40 GB — so choosing between them is a preference, and ours (35b-a3b, because it tested better) narrows
nothing. A model quietly dropped from the sweep is a model the shipped wording is no longer known to work
on. The 4B is also the cheapest arm, which makes skipping it the wrong economy twice over.

Do not read the two tables against each other for anything finer. Between the 27B nine-sample runs and
these, the model, the samplers and the surrounding prompt all changed; only the internal comparisons within
each table are controlled. What is warranted before the shipped wording is defended on the strength of the
old numbers: the full variant sweep, three or more samples per arm, across the fleet rather than on one
member of it.

Raw output: `absent_fact-2026-08-11.txt` (`.txt` rather than `.log`, which `.gitignore` excludes).

## 2026-10-05: the grounding reminder, sent only with material or always

`reminder_placement.py`, five models on LM Studio (Qwen 3.6 35B-A3B, Qwen 3.5 9B, Qwen 3.6 27B, Gemma 4
26B-A4B, Qwen 3.8 27B), 12 samples per scenario per arm (T=0 ×3, T=1 ×9). Arm **A** is what shipped: "Base
claims about the provided documents on those documents…" added to the system message only when the turn had
material. Arm **C′** sends "When documents, attachments or tool results are in the conversation, base claims
about them on them. Answer general questions normally." on every turn.

| Scenario | Wanted | A | C′ |
|---|---|---|---|
| *general*: "Who wrote Hamlet?", nothing retrieved | an answer | 59 / 60 | 59 / 60 |
| *absent*: asked about Kuiper-9, given the Kuiper-7 document | a decline | 58 / 60, 2 empty | 60 / 60 |
| *present*: asked about Kuiper-7, given that document | its figure | 60 / 60 | 60 / 60 |

- **No reply invented a figure**, in either arm. The script's regex left many declines "unclear"; every one
  was read, and every one says Kuiper-9 is not there and offers Kuiper-7's figure as Kuiper-7's.
- The two *general* misses are one runaway each on Qwen 3.6 27B, deliberating past the 16k cap over how to
  word "William Shakespeare". The two empty replies are Gemma under A at T=1.
- **C′ does not bring back Q4's over-deliberation.** Median reasoning length is close between the arms on
  every model, and on Gemma C′'s is about half of A's.

**The cache scenario does not measure what it was meant to**, and the reason is a finding of its own. Its
second turn's prompt parts from the first turn's at message 2 in *both* arms: the per-turn clock inject, a
synthetic tool exchange placed before the latest user message and not stored, sits where the next turn has
the previous user message. So without something between turns re-warming the cache, every turn reprocesses
the previous exchange. Librarian's GUI hides this with its context prefill between turns; a batch run
through `agent.turn` has no such step. Under A the prompt parts at message 0 instead, as expected.

A's cost therefore rests on its mechanism, which holds by construction: the reminder changes message 0 on
the first turn with material, and everything after message 0 is then reprocessed. In a long chat that is
the whole context. A live turn first taken as showing it (a PDF attached after a prefill, its whole prompt
reprocessed) turned out to show something else, found the same day: **the GUI's prefill sent the stored
system message without the turn's preamble and postamble**, so it warmed a prefix no turn ever sent, and
every turn reprocessed the whole conversation whatever the reminder did. The prefill now builds its prompt
with `scaffold.build_prefill_prompt`, which shares that step with `build_turn_prompt`.

**Decided** (maintainer, 2026-10-05): **C′, as standing text** in `prompts/interaction.md` rather than an
inject, since it no longer varies; the conditional inject is gone. Not measured: the same words as a bullet
in the card rather than as a line of their own at the head of the system message, which is where C′ sat
here. Still conditional in the system message, and so still a cache miss when it appears: the notice that
the tool budget is spent, on the round that spends it.

Data: `reminder_placement-summary.jsonl`, one row per sample — the reply, its verdict, reasoning *length*,
and the generation figures with their phases. The raw `reminder_placement.jsonl` the script writes keeps
every prompt and reasoning trace in full, which carry the local user profile card, so it stays local.

## Related

- `../tool_budget/` is a separate study that shares the same theme; it has its own apparatus.
- Model choices made on the strength of these measurements: `briefs/reference/model-lineup-autumn-2026.md`.
- The inject implementation itself: `raven.librarian.scaffold.build_turn_prompt` (called
  `_perform_injects` until 2026-08-10, when it was made public and stopped mutating its argument).

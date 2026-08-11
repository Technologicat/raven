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

## Related

- `../tool_budget/` is a separate study that shares the same theme; it has its own apparatus.
- Model choices made on the strength of these measurements: `briefs/reference/model-lineup-autumn-2026.md`.
- The inject implementation itself: `raven.librarian.scaffold.build_turn_prompt` (called
  `_perform_injects` until 2026-08-10, when it was made public and stopped mutating its argument).

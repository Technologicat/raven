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
Re-running them against qwen3.6-35b-a3b immediately changed two results, in the direction that matters:

| check | with the stand-in prompt | with Raven's real prompt |
|---|---|---|
| `absent_fact`, as-shipped, T=0 | `finish=stop`, 4430 chars of reasoning | **`finish=length`, 31726 chars** — never produced a reply |
| `assembled_shape` [2], absent fact | declined cleanly | emitted literal `<tool_call> <function=search_documents>` text |

Both are failures the probes were written to catch, and both were invisible while the prompt was a
placeholder. The second is precisely the behaviour `absent_fact`'s own docstring describes — with no tools
declared, the model writes the call out as text and the user gets that instead of an answer — so the probe
was reporting clean on the very defect it exists to find.

The runaway is the more serious of the two: `absent_fact`'s docstring records 29000 characters of reasoning
with no reply as the reason the `closing-note` wording was **rejected**. On the real system prompt, the
*shipped* wording does the same thing at T=0. That wants a fresh look before Researchers' Night, and it is
not what these older measurements say, because they were not taken against this prompt.

**The measurements above stand as taken** — this section is the note that says what they were taken against.
Anything re-measured from here on is measured against the real settings.

## Related

- `../tool_budget/` is a separate study that shares the same theme; it has its own apparatus.
- Model choices made on the strength of these measurements: `briefs/reference/model-lineup-autumn-2026.md`.
- The inject implementation itself: `raven.librarian.scaffold.build_turn_prompt` (called
  `_perform_injects` until 2026-08-10, when it was made public and stopped mutating its argument).

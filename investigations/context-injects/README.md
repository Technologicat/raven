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

## Related

- `../tool_budget/` is a separate study that shares the same theme; it has its own apparatus.
- Model choices made on the strength of these measurements: `briefs/reference/model-lineup-autumn-2026.md`.
- The inject implementation itself: `raven.librarian.scaffold._perform_injects`.

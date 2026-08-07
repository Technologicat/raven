# Tool-budget measurements (Librarian)

Does the tool-call round cap cause empty replies, and does telling the model its budget is spent prevent
them?

## Setup

`briefs/librarian-extension/manual_tests/rag_live_corpus.py`, phase F, against the real
document database (11974 Web of Science records on hydrogen production). Each **sample** is two AI turns:
one asking a corpus question, then a follow-up — *"Which of those documents said that, and what else does
it say?"* — which reliably provokes a multi-document read, since `list_consulted_documents` hands back a
list of IDs and invites reading them.

Arms alternate **within** the run rather than across runs, because earlier informal batches differed from
each other by more than the effect being measured. `notice` has
`chatutil.format_notice_that_tools_are_spent` active; `control` silences it.

Model: qwen3.6-35b-a3b (IQ4_NL_XL, 128 Ki context) via LM Studio. `max_tool_call_rounds = 5`.

## Files

- `spent_tools_notice.jsonl` — one line per sample, written as each finishes. This is also the resume
  ledger: re-running the same command continues from its length.
- `spent_tools_notice-NNN-{notice,control}.json` — the full chattree per sample, so a later analysis is not
  limited to the fields the probe thought to log. Contains the reasoning traces, which is where the
  diagnosis came from.
- `run-2026-07-29.log` — verbatim console output.

## Result, 24 samples (12 per arm), 2026-07-29

**Reaching the cap is what produces the empty reply.**

| | answered | empty |
|---|---|---|
| reached the cap (5 rounds) | 5 | 9 |
| did not (1–4 rounds) | 9 | 1 |

Fisher exact, two-sided: **p = 0.013**, odds ratio 0.06.

**The notice moved nothing.** 8 of 12 answered with it, 6 of 12 without (p = 0.68) — and restricted to
cap-reaching turns the sign reverses (1 of 5 with, 4 of 9 without), which is noise at that size but rules
out claiming even a directional benefit. It is kept because it costs one line and addresses a mechanism
observed directly in the reasoning traces, not because it was shown to work.

## What this does not establish

The cap-reaching turns and the others are not randomly assigned — a turn reaches the cap because the model
chose to fetch documents one at a time, and that choice may itself correlate with whatever makes a turn end
badly. The association is strong and the mechanism is legible in the traces, but this is observational, and
"raise the budget and the empty replies go away" remains a prediction rather than a finding until the
budget is actually raised and re-measured. The probe resumes and the arms are recorded, so that measurement
is a re-run rather than a rebuild.

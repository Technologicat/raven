# Sorting the AOKK corpus into in-scope and out-of-scope

The AOKK corpus was assembled by a boolean query — an AI-agent term AND a collaborative-learning term AND
`student*` AND a higher-education term — and the last three blocks are broad enough that a paper about
something else entirely can clear all four. This is the LLM pass that flags the ones that did, so they can
be reviewed and taken out.

The brief is `briefs/researchers-night/aokk-corpus-scope-classification-brief.md`; it carries the two
confirmed false-positive shapes and the argument for a two-pass design. This directory holds the
apparatus, and the measurements the brief left open.

Modelled on `investigations/agent-batch-classification/`, which did the same shape of job over ~1600
arXiv papers, and whose finding about escalation is the reason the escalation rule here has two triggers
rather than one.

## What is judged, and against what

The question is the broad one the search actually asked, not the narrower research questions:

> studies on different aspects of the use of AI agents in higher education

Judging against the research questions is a separate, later pass over a corpus that has already had its
obvious strays taken out — so "not about data-informed decision-making" stays a separate verdict asked
separately, rather than being folded in here where it would look like the same one.

**The source is the `.bib`, not the Visualizer dataset built from it.**
`00_stuff/rawdata/AOKK/multisource/tekoalyagentti_tutkimus_deduped.bib` has **5167 records with 5167
unique citekeys**, so there is a key to write the outputs against; `multisource.pickle` has no citekey, so
using it would mean matching 5007 records back to BibTeX by title in the critical path. The brief's
measurements reproduce on the `.bib` — median 13 title words, 5th percentile 7, 83% with abstracts.

The 160-record difference between the two is fully accounted for by the importer's own filter: the import
log records 158 skips for `no author` and one for `no title`. Sampling them, they are whole-proceedings
volume records — *"13th International Conference on the Future of Education (FoE 2023)"*. So judging the
`.bib` also puts 160 near-certain strays in front of the judge instead of inheriting a filter that
happened to remove them.

## Both halves ask for evidence of being *off* topic

This is the one design decision that is not obvious, and it was measured rather than reasoned (below).
Each record is judged on two booleans:

- **`no_ai`** — true only if the work is positively about something with no AI in it: human teaching
  staff, human undergraduate learning assistants, a non-AI technology.
- **`wrong_field`** — true only if the work is positively set somewhere other than higher education: a
  school, a hospital, industry, the general public.

Either one true drops the record; an unanswered half withholds the verdict rather than deciding it, so an
incomplete run under-filters rather than silently losing studies.

**The direction matters because silence is the common case and it is not evidence.** Every record here
already matched a higher-education search term, so a title that does not restate the setting says nothing
against it. Asked the other way round — *is this in higher education?* — the model answers **no** for any
title that merely fails to say, and those drops look exactly like real ones in the output.

## Findings

### The sibling run's failure mode does not reproduce here

`agent-batch-classification` found the model at its most confident exactly where the input carried least:
it classified `2006.05563.pdf` as AI with **high** confidence, explaining that the arXiv id was GPT-3's.
That is why the escalation rule there could not be driven by the model's confidence alone.

Measured here on all 35 records whose title runs to fewer than five informative words —
`judge_scope.py --thin`, the whole group rather than a sample — **28 of 35 came back `low`**, with reasons
that are accurate about their own poverty: *"Generic title, subject and setting unclear"*, *"Missing
title, cannot evaluate"*, *"Placeholder text"*. No confident invention anywhere in the group.

So on this corpus the model's self-report is roughly honest, and the structural criterion is mostly
redundant — the confidence field alone would have escalated 28 of the 35. It is kept because it costs
nothing and covers the other 7, and because "roughly honest on 35 records of one corpus" is not a property
to build a run on.

### Asked as a test of being on topic, the model reads an unstated setting as the wrong setting

The first version of the rubric asked `about_ai` and `higher_ed` directly. Over the same 35 records,
**9 came back `about_ai=yes, higher_ed=no`, and 5 of those at medium or high confidence** — so the
confidence criterion would not have escalated them. The model's own reasons name the problem:

| title | confidence | why |
|---|---|---|
| Teachable Agent | medium | AI educational concept, typically K-12 |
| Vibe Coding in Education | medium | AI trend in education, specific level |
| AI Mediated Learning Architectures | medium | Explicit AI focus, but educational lev… |
| Chatbot and Digital Communication | medium | Mentions chatbot AI, but educational s… |
| Generative AI in CALL | low | AI in language learning, specific leve… |

Every one of those is a study that would have been dropped for not restating what the search had already
established. Reframed to ask for evidence of being off topic, the same 35 records come back **33 keep / 2
drop**, the silent cases now sitting at `no/no` with low confidence, and both drops are correct — *A
theology rhizome*, and *ECTRIMS 2025 ePoster*, which the model recognized from the acronym as the European
multiple-sclerosis congress.

**The lesson generalizes past this corpus**: when every record has already passed a filter, asking a model
to re-confirm that filter turns every silence into a rejection. Ask for the contrary evidence instead.

### The corpus is noisier than the brief's two shapes suggested

Over a random 200 (`--pilot 200`, seed 42, titles only), pass 1 answered every record and **dropped 49 of
them — 24.5%**, at high confidence for 30. That is far more than a picture of a corpus with a few strays
in it, so the first thing to ask is whether the judge is over-dropping. Reading the 30 high-confidence
drops says it is not: link prediction, synthetic data generation, smart-grid security, crash analysis,
financial LLMs, computer vision and privacy-preserving neural networks, none of which mention education
at all — and a large contingent the higher-education block never excluded, on preschool, K-12, secondary
school, rural teachers, and children with neurodevelopmental disabilities.

The brief's own two shapes are in there and are caught: *Enhancing Well-being Through Food: A
Conversational Agent for Mindful Eating and Cooking* is the `"conversational agent"` shape exactly.

**The cut is roughly a quarter, then, and its bulk is not the two shapes the brief found by looking.**
Those are what a reader notices; what a corpus this size actually accumulates is off-domain AI research
and off-level education research, neither of which stands out until something asks about every record.

| | pass 1 over a random 200 |
|---|---|
| keep | 151 (high 70, medium 44, low 37) |
| drop | 49 (high 30, medium 18, low 1) |
| escalated to pass 2 | 38 (19%) |

### The full run is an overnight job, and pass 2 is the half worth batching

Measured at the pilot's own rate: pass 1 runs ~104 s per batch of 40, so 5167 records is **~3.8 h**. Pass 2
takes one call per record at **~20 s**, and 19% of the corpus is ~980 records, so **~5.4 h** — the larger
half, for a fifth of the records.

Abstracts here average 1362 characters, so ten of them in one call is around 4k tokens of prompt. Batching
pass 2 the way pass 1 is already batched would cut its half to well under an hour. Worth doing before the
full run, and not worth doing before the calibration says the design holds.

### A random sample cannot test the thin-title criterion

The thin titles are 35 of 5167, so a random sample of 200 is expected to contain **1.4 of them, and the
one actually drawn contained none.** The criterion exists to be tested, and a sample of any affordable
size cannot test it.

Hence `--thin`, which judges the whole group rather than a draw from it — few enough to read in full,
which is what the brief meant by calling them "a hand-checkable list". It is a separate calibration run
from `--pilot`, not a subset of one, so that neither contaminates the other's error rate.

## Files

| File | What it is |
|---|---|
| `judge_scope.py` | The classifier. `--pilot N` and `--thin` are the two calibration runs; a plain run does both passes and writes the outputs. Re-running resumes |

Generated at runtime and **not committed** — they list the contents of a corpus that lives under
`00_stuff/`, which is gitignored research data, and this repository is public:

| File | What it is |
|---|---|
| `pilot-<seed>-<n>.jsonl`, `.tsv` | the random-sample calibration run and its hand-check table |
| `pilot-thin-<n>.jsonl`, `.tsv` | the same over every record with a title under the informative bound |
| `judged.jsonl` | the full run's resumable state |
| `<corpus>_in_scope.bib` | the corpus with the strays taken out, importable into Visualizer |
| `dropped.tsv` | every dropped record with a one-line reason — the reviewable half |

## Reproducing

```bash
B=00_stuff/rawdata/AOKK/multisource/tekoalyagentti_tutkimus_deduped.bib

python investigations/aokk-corpus-scope/judge_scope.py --bib $B --thin        # the thin-title calibration
python investigations/aokk-corpus-scope/judge_scope.py --bib $B --pilot 200   # the random calibration
python investigations/aokk-corpus-scope/judge_scope.py --bib $B              # the full run
```

Needs an LLM backend; `--backend-url` and `--model` point it elsewhere. The calibration runs above were
made against `qwen3.6-35b-a3b`, which is the model the numbers in this file describe.

A hand-check table has an empty first column: put an `x` in it on every row you disagree with. The rows
are sorted so that the cells of verdict × confidence are contiguous, because the two kinds of error are
not equally bad — a false drop loses a study silently, a false keep costs a reader one line — and they can
only be counted separately if they can be read separately.

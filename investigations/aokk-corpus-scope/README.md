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

### Whether the model confabulates on a thin title depends on what the rubric invites

`agent-batch-classification` found the model at its most confident exactly where the input carried least:
it classified `2006.05563.pdf` as AI with **high** confidence, explaining that the arXiv id was GPT-3's.
That is why the escalation rule there could not be driven by the model's confidence alone.

Asked only *what is present* — the first two rubrics — the model is honest about thin input: **28 of the
35 short titles came back `low`**, with reasons accurate about their own poverty (*"Generic title,
subject and setting unclear"*, *"Placeholder text"*). Asked to reason from what is *absent*, the same
model on the same 35 titles invents subjects for names it does not know, at high confidence. Same model,
same inputs, opposite behaviour, and the rubric is the only variable.

So the finding to carry forward is not "this model does or does not confabulate" but that **the rubric
decides whether it has room to.** Which is also why the structural escalation criterion stays: on the
honest rubrics it is nearly redundant, and there is no version of this where its redundancy can be
verified in advance.

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

Over a random 200 (`--pilot 200`, seed 42, titles only), pass 1 answered every record and **dropped 55 of
them — 27.5%**, at high confidence for 36. That is far more than a picture of a corpus with a few strays
in it, so the first thing to ask is whether the judge is over-dropping. Reading the high-confidence drops
says it is not: link prediction, synthetic data generation, smart-grid security, crash analysis,
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
| keep | 145 (high 85, medium 20, low 40) |
| drop | 55 (high 36, medium 16, low 3) |
| escalated to pass 2 | 43 (22%) |

**Which test fires says the same thing quantitatively**, and is the argument for having split them:

| test | drops |
|---|---|
| `not_education` alone | 40 |
| `wrong_level` alone | 8 |
| `no_ai` alone | 4 |
| more than one | 3 |

So **three quarters of the cut is "this is not about education at all"** — the test that did not exist in
the first two rubrics, and whose absence let a link-prediction survey through. The two shapes the brief
was written around are the small remainder.

### The venue is the field the record already has and the title does not

The prompt sent title and abstract, and threw the journal away. It should not have: **85.3% of this
corpus carries a venue, and — the case that matters — so do 852 of the 853 records with no abstract at
all.** Exactly one record in 5167 has neither.

A venue is often the strongest evidence in the record. *Proceedings of the Learning Analytics and
Knowledge Conference*, *CEPS Journal: Center for Educational Policy Studies*, *Journal of English as a
Lingua Franca* — each settles the domain question outright, and none of it needs the model to recall
anything.

**The case that shows why this matters is `bedi_generative_2025`, whose whole title is "Generative AI in
CALL".** Without the venue the model answered correctly — CALL is Computer-Assisted Language Learning —
but it answered *from recall of an acronym*, which is indistinguishable from the confabulation two
sections above until you check. Its venue is *The Palgrave Encyclopedia of Computer-Assisted Language
Learning*. Given that, the same verdict arrives with the evidence attached, and the model says so.

So the improvement is not accuracy, it is **groundedness**: the same answer, reached from the record
instead of from the model. And it is free — the field was already parsed and discarded.

It does not make the model credulous, which was the risk: given *KI - Künstliche Intelligenz* as the venue
for `Project VoLL-KI`, it still declines to guess what the project is and answers `low`.

Two cautions are in the rubric, because a venue is weaker evidence than it looks in two specific ways. It
rarely states the *level* — an education journal covers schools and universities alike — so it says little
about `wrong_level`. And it describes where the work was published rather than what the work is: a book
review in an education journal is still a book review.

**The second caution is in the rubric and the model overrode it anyway**, which is worth knowing before
trusting this field. A monograph titled *Search as Learning*, published in a venue whose name is about
information retrieval, was dropped as `not_education` at medium confidence — and its abstract turns out to
review education research on learning objectives and strategies, self-regulated learning, learning
retention and transfer, and tools to support learning during search. It is not merely educational; that is
its subject. A technical venue name outweighed a title that says *learning* twice.

So the venue is worth adding — it grounds the judgement, and the alternative was the model recalling
acronyms — but it introduces a false-drop mode of its own, in a corpus where an education topic is
routinely published in the venue of whatever field it is applied to. That is an argument for escalating
uncertain drops rather than for withholding the field.

### A truncated abstract, and why the model perceives it but must not act on it

A tenth of this corpus's abstracts are publisher teasers that break off mid-sentence, and pass 2 escalates
precisely the uncertain records — so for one in ten of them the "extra evidence" stops before the methods.
Told about it in the prompt, the model concluded from the absence anyway, at **high** confidence, on a
511-character preview that never reaches the method it was asserting about.

So the policy is enforced in Python: an uncertain-by-truncation **drop** is withheld and recorded as
unknown, which keeps the record and flags it for a reader, and the confidence is forced to `low` whatever
the model claimed. The division that makes this work is *perception versus restraint* — the model is
reliable at seeing that a text breaks off and unreliable at declining to conclude from it, so it is asked
the first and never trusted with the second.

**Detection has two halves, and neither is sufficient.**

- **An ellipsis at the end**, which needs no model. Not an ellipsis *anywhere*: mid-text it is ordinary
  rhetoric and says nothing. Not length either — some abstracts are simply short, and a complete
  96-character abstract ("Public sector AI procurement checklists can help guide efforts to create
  regulatable AI systems.") is a real record rather than a stub.
- **The model's own `truncated` answer**, for the publisher who cuts silently. There is no text-level
  signal for that: "ends without terminal punctuation" was tried and rejected on the data, selecting 115
  records that are *complete* — ending in a URL, a DOI or a keyword list — with a median length longer
  than the corpus's.

**The visible half's false-positive rate is measured across five corpora**, since a rhetorical trailing
ellipsis would trip it: outside this corpus, across more than 15,000 abstracts, exactly **one** ends in
one, and it is a bullet list of questions trailing off. The cost of that case is a single withheld drop,
which a reader sees rather than loses.

### Unscreenable is not the same claim as off topic

`--require-abstract` sets aside the records with no abstract *before* judging, into `unscreenable.tsv`
with a reason of their own.

The distinction is the point. A record with a bare title is not off topic — nobody knows what it is — it
is **useless to a review study**, which has nothing to read. Those are different claims, and merging them
would put a sixth of the corpus into the topical drop list with an empty reason column, where a reviewer
could neither audit them nor chase them down by DOI.

Note this is unrelated to whether the *model* can judge them: with the venue in the prompt it usually can.
The filter is about what a human reviewer can screen, which a venue name does not help with at all.

**It probably belongs in its own tool rather than here** (Juha's question, 2026-09-02, and the answer is
provisional). It needs no LLM — it is a mechanical `.bib` → `.bib` question, the same shape as
`raven-fixbib` and `raven-deduplicate` — and running it *first* removes 853 records and takes roughly 16.5%
off the LLM run, about 35 minutes of pass 1. Bundled into the judge, that is time spent classifying records
already destined to be discarded. It lives here for now because extracting it would be building the
generalization before the prototype has settled.

### A drop and a keep do not deserve the same standard of proof

Escalation began as "the model said `low`, or the title is too thin to have been answerable". Both hand-
checked false verdicts found during calibration slipped through it: *A theology rhizome* and *Search as
Learning*, each dropped at **medium** confidence, each plainly in scope once its abstract is read. Both
escalated anyway — but only because their titles are short enough to trip the structural rule, which is
luck. The same answers on a normal-length title would have stood, and the record would be gone.

So an uncertain **drop** now escalates where an uncertain keep does not. This is the brief's own asymmetry
made operational rather than a second guess at the confidence field: a false keep costs a reader one line
of review, and a false drop removes a study from the review with nothing left behind to notice it by.

Measured on the 200-record sample, it takes escalation from **5% to 15%** — which is affordable, and is
the strongest argument for batching pass 2 before the full run.

### One predicted pass-2 failure, worth checking when the full run lands

Pass 1 correctly drops a monograph titled *Translation in the Wild* as having no educational dimension:
it asks why large language models can translate at all — incidental bilingualism in pre-training data,
context windows, batch training — and no student appears anywhere in it.

**Its abstract is nonetheless a trap for pass 2**, which is where it goes next. The word *learning*
occurs six times in it — "Local learning", "Global learning", "deep learning" — every one of them the
*model's* learning. The rubric tells the judge that teaching, learning, students and courses count as an
educational dimension, so a literal reading flips this record to keep.

That is the `"learning assistant"` collision one layer down, and it is not special to this record: any
machine-learning paper's abstract is dense in exactly the vocabulary the rubric uses to recognize
education. Worth checking specifically once pass 2 has run over the corpus, rather than trusting that the
title-level fix covers it.

### The full run is an overnight job, and pass 2 is the half worth batching

Measured at the pilot's own rate: pass 1 runs ~104 s per batch of 40, so 5167 records is **~3.8 h**. Pass 2
takes one call per record at **~20 s**, and 19% of the corpus is ~980 records, so **~5.4 h** — the larger
half, for a fifth of the records.

Abstracts here average 1362 characters, so ten of them in one call is around 4k tokens of prompt. Batching
pass 2 the way pass 1 is already batched would cut its half to well under an hour. Worth doing before the
full run, and not worth doing before the calibration says the design holds.

### A random sample cannot test the thin-title criterion, and the thin group became the regression fixture

The thin titles are 35 of 5167, so a random sample of 200 is expected to contain **1.4 of them, and the
one actually drawn contained none.** The criterion exists to be tested, and a sample of any affordable
size cannot test it.

Hence `--thin`, which judges the whole group rather than a draw from it — few enough to read in full,
which is what the brief meant by calling them "a hand-checkable list". It is a separate calibration run
from `--pilot`, not a subset of one, so that neither contaminates the other's error rate.

**What it turned into is more useful than what it was for.** One batch, ~85 s, 35 records that are all
worst case by construction — so it is the fixture a rubric change is checked against, and it caught two
defects that the 200-record sample could not have. Run it after any edit to the prompt, before spending
ten minutes on the larger one.

### Every version of this rubric has failed in the same direction, and each failure has a different door

Three iterations in one afternoon, each fixing the previous one's blind spot:

1. **`about_ai` / `higher_ed`, asked as tests of being on topic.** The model answered "not higher
   education" for any title that failed to state a level — 9 of 35, five at medium or high confidence,
   none of which the escalation rule would have caught.
2. **Reframed to `no_ai` / `wrong_field`, asking for evidence of being off topic.** Fixed that, and
   opened the mirror hole: a pure machine-learning paper is not set at the *wrong* level, it is not set
   anywhere, so both booleans came back false and a link-prediction survey was kept. Pass 1 and pass 2
   read the same rubric differently, which is how it surfaced — pass 2 flipped 8 of 30 high-confidence
   drops to keep while its stated reasons said *"zero educational content"*.
3. **Split into `no_ai` / `not_education` / `wrong_level`.** Fixed that, and the sentence that made it
   work — *"the absence IS the evidence"* — turned out to be a licence to assert on no evidence at all. A
   three-word title is nothing but absence, so `Reportronic` was dropped at **high** confidence as a
   "clinical reporting tool" (it is a Finnish research-project management system used by universities),
   and `A theology rhizome` likewise, a record whose abstract says outright that it is about teaching
   theology students with ChatGPT.

**So the sibling investigation's finding does reproduce here after all**, and the earlier note in this
file saying it did not was measured against a rubric that gave the model nothing to confabulate *with*.
Invited to reason from absence, the model invents what an unfamiliar proper noun refers to and says so at
high confidence — which is `2006.05563.pdf` exactly. The repair is that investigation's own, which should
have been carried across at the start: an identifier is not a description, and the model is told not to
claim recognition of a name it does not know. `Reportronic` and `Intelligent Communities` came back
`keep`/`low` immediately afterwards, quoting the instruction.

The generalizable shape, since a `raven.papers` version of this will meet it again: **a rubric that asks
about absence needs a body of text for the absence to be measured in.** So `not_education` is asked in two
ordered steps — *can you name the subject at all?* first, and *is that subject outside education?* only
if the first is yes. A test that skips the precondition is answered from the model's imagination.

## Where this is headed: a `raven.papers` corpus filter

Decided 2026-09-02, and deliberately **not** acted on yet — the AOKK framing is what the calibration is
measured against, and generalizing before it is settled would mean calibrating a moving target. Recorded
here because the seam is visible now, while the code is in front of us, and would have to be re-derived
later.

Nothing about the machinery is AOKK-specific. What is:

| AOKK-specific | Already general |
|---|---|
| `SCOPE_QUESTION` — the corpus's own topic | the two-pass structure: batched titles, then abstracts for the unsure |
| the rubric's three tests and their domain examples | the escalation rule, and its structural half being computed rather than asked |
| `MIN_INFORMATIVE_WORDS`, `TEASER_CHARS` — both measured *from this corpus* | resumable JSONL keyed on citekey; `.bib` in, filtered `.bib` plus reasons out |
| | the review TSV, and the verdict × confidence sort that makes it readable |

**The three tests are the interesting part of the generalization, not the boilerplate.** They are not a
fixed rubric that a second corpus would inherit — they are this corpus's answer to "which ways can a
record be off topic?", and a different boolean search has different ways. What generalizes is the
*shape*: one test per way, each asking for positive evidence of being off topic, with silence keeping. So
the tool's parameter is a list of named tests with their descriptions, and the AOKK three become its
first worked example rather than its schema.

The two corpus-measured constants want deriving from the corpus at run time rather than being carried
across, for the same reason — a corpus whose titles run short would need a different informative bound,
and inheriting 5 would silently escalate everything or nothing.

## The run plan, and why sifting comes second rather than first

Decided 2026-09-02. `raven-siftbib --require abstract` would remove 853 of the 5167 records and take
roughly 16.5% off the LLM run, so the tempting order is to sift first and judge less. **It is the wrong
order while the judge is still being calibrated**, and right once it is not:

1. **Calibrate against everything, abstract or no.** A sixth of this corpus is title-only, and that is
   the hardest input the judge will meet — the case where a rubric that reasons from absence invents a
   subject, which is how two of the three rubric defects above were found. Sifting first would take
   exactly those records out of the calibration and leave the judge untested on the inputs most likely
   to break it.
2. **Sift, then run.** The final scoping-review corpus does not want records that cannot be screened,
   so they go before the full pass rather than being judged and then discarded — which also buys back
   the 16.5%.

So the sift is a step in the pipeline and *not* a step in the calibration, and the two pilots deliberately
run against the unsifted corpus.

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

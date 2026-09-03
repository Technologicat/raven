# Sorting the AOKK corpus into in-scope and out-of-scope

The AOKK corpus was assembled by a boolean query — an AI-agent term AND a collaborative-learning term AND
`student*` AND a higher-education term — and the last three blocks are broad enough that a paper about
something else entirely can clear all four. This is the LLM pass that flags the ones that did, so they can
be reviewed and taken out.

The brief is `briefs/researchers-night/aokk-corpus-scope-classification-brief.md`; it carries the two
confirmed false-positive shapes and the argument for a two-pass design. This directory holds the
apparatus, and the measurements the brief left open.

Modelled on `investigations/agent-batch-classification/`, which did the same shape of job over ~1600
arXiv papers, and whose finding about escalation is the reason the escalation rule here is computed from the
input rather than taken from the model's own confidence.

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

## All three tests ask for evidence of being *off* topic

This is the design decision that is not obvious, and every part of it was measured rather than reasoned
(below). Each record is judged on three booleans:

- **`no_ai`** — true only if the work is positively about something with no AI in it: human teaching
  staff, human undergraduate learning assistants, a non-AI technology.
- **`not_education`** — true only if the work has no educational dimension at all: a machine-learning
  methods paper, a computer-vision study, a finance or energy application. Asked in two ordered steps —
  *can you name the subject?* first, and only then *is it outside education?*
- **`wrong_level`** — true only if the work is positively set at a level other than higher education: a
  school, a workplace, the general public.

Any one true drops the record; an unanswered half withholds the verdict rather than deciding it, so an
incomplete run under-filters rather than silently losing studies.

**The direction matters because silence is the common case and it is not evidence.** Every record here
already matched a higher-education search term, so a title that does not restate the setting says nothing
against it. Asked the other way round — *is this in higher education?* — the model answers **no** for any
title that merely fails to say, and those drops look exactly like real ones in the output.

**`not_education` is the exception, and the one place absence *is* evidence** — a machine-learning paper
is not set at the wrong level, it is not set anywhere — which is why it is a test of its own rather than
part of the level question, and why it needs the name-the-subject precondition to stop that licence
becoming a licence to invent.

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

## Reviewing the drops, and checking the reviewer before believing it

The full run dropped 1219 of 4314 records, which is far past what anyone will re-read. `review_drops.py`
is the second reader over that list, and it is asked the *opposite* question — *make the strongest case
that this record belongs* — because re-asking "was this right?" buys a rubber stamp. It is told nothing
about provenance: not that the records were judged, not what the verdict was, not that a model produced
it. Its output is a one-line case, which a person can check against the abstract in seconds; "I confirm
the drop" is not checkable at all.

**A reviewer asked whether a case exists can manufacture one for anything**, so kept records are mixed in
unlabelled and shuffled among the rest. If both groups come back at the same rate the instrument is
measuring nothing, and that has to be read before any of its findings are.

Four slices, **all 1219 dropped records**, each reviewed exactly once (`score_review.py --dropped` checks
that rather than trusting the offsets):

| the judge said | n | case found | rate |
|---|---:|---:|---:|
| `drop` / high / title | 749 | 29 | 3.9% |
| `drop` / medium / abstract | 175 | 7 | 4.0% |
| `drop` / high / abstract | 295 | 1 | 0.3% |
| **contested overall** | **1219** | **37** | **3.0%** |

Four independent controls, one per slice: 70.0%, 57.5%, 50.0%, 65.0%, for separations of 65.8, 53.9, 49.5
and 65.0 points against a 20-point floor. The instrument discriminates, and the spread across the four is
what a forty-record control costs in precision.

**A drop is clean when it is high-confidence *and* was reached from the abstract. Neither half does it
alone.** The three cells make the conjunction plain:

- read the abstract, and hedged → **4.0%**
- never read the abstract, but confident → **3.9%**
- read the abstract, and confident → **0.3%**

So confidence buys nothing on a title-only verdict, and reading the abstract buys nothing where the model
still hedges; together they drop the contested rate more than tenfold. The single record behind that 0.3%
is the reviewer reaching, not a real miss — a ChatGPT-on-a-medical-exam benchmark defended as "assessing
its potential for higher education testing" — so that cell is effectively clean.

**Which says pass 2 should run on everything, not only on the unsure.** Escalation today is triggered by
doubt, so a record dropped *confidently* from its title never gets a second look — and those are the ones
being contested, thirteen times as often as the ones that did. Confidence measured against a title is not
just uninformative here, it is anti-correlated with correctness relative to having read the abstract. The
fix is the cheap resource: reading the abstract for the other 749 costs about two thirds again on top of
the pass 2 already run, against a corpus where the rate it buys is 0.3%.

**This was measured in four slices, and the first one alone said something different — twice.** With only
`high/title` and `medium/abstract` in hand their rates matched, and what was written here was that the
judge's confidence predicts nothing on the drop side: a generalization to the whole from two of its three
cells, in a paragraph that noted the third was unmeasured. The same partial reading also said `no_ai` had
produced zero contested drops, which the full pass turned into three. Recorded because the shape is
available to anyone reading a partial table, and it is not random which cell is missing: the one left out
is the one that was hardest to reach, which is exactly why it is the one most likely to differ.

**The uniform control was easier than what it was compared against, and the stratified draw confirms
it.** A uniform draw from the kept pool came out 31 high / 8 medium / 1 low, where the dropped records
under review are 44% medium — so the two groups differed in difficulty as well as in verdict. Redrawing
20 high / 20 medium took the control from 70.0% to 50.0%, so the effect was real and worth roughly twenty
points. It changed no conclusion: the separation is 49.5 points, and the arithmetic said in advance that
even rejecting *every* hedged keep would leave 34.5. Worth having as a flag (`--control-strata`), not
worth a run of its own — which is why it rode along with the slice that covered the missing cell.

Note the reviewer scores the *hedged* keeps well below the confident ones (35–50% against 65–77%). That
is the instrument agreeing with the judge's hedge rather than failing, and a pooled control rate cannot
tell the two apart — which is what `score_review.py` splits out.

**Two error shapes among the 37 contested drops, and a rate alone cannot tell them apart.** This is a
reading of all 37 rather than a measurement:

- **In the title-only cell, real misses of a predictable kind**: the title names a *domain* and the
  abstract names a *university setting*. Agricultural planning that turns out to be taught to university
  agriculture students; a literary translation platform trialled on translation students; a voice
  framework deployed in a Master's thesis seminar. Roughly eight of the ten hits in that cell.
- **In the abstract cells, the reviewer reaching** — an HCI review that "covers educational applications",
  a computational-social-science paper that "simplifies learning", an anchoring-bias study whose only tie
  is a college-admissions *dataset*. The prompt says a case must rest on what the record says and that
  reaching means there is none; these are where it reached anyway.

Which is why the two ~4% cells are not the same 4%: the title-only one is mostly the judge's error, the
hedged-abstract one mostly the reviewer's. A single contested rate hides that, and the repair differs —
pass 2 on the title-only records; a stricter reviewer prompt for the rest.

**Which test fired barely matters; whether an abstract was read matters a great deal.** `not_education`
accounts for 31 of the 37 and `no_ai` for 3, but that tracks how often each test fires rather than how
reliable it is: those 3 come from 88 title-only `no_ai` drops, a rate of 3.4% against the 3.9% baseline
for title-only drops of any kind. So `no_ai` asserting absence from a title — the failure the rubric was
rewritten three times to stop — is no more common than every other title-only mistake. The guard in that
test works about as well as a guard can; the residue is the cost of judging a title.

**What the drop list holds, now counted rather than estimated.** 37 contested records in 1219, listed in
`contested.tsv` worst cell first. The two-slice extrapolation had predicted about 42, most of the excess
coming from the first slice's `high/title` rate (4.4%) sitting above the full cell's (3.9%).

**The sort's ordering matches contestedness, which the partial data had denied.** `least_defended` ranks
by confidence then source, ordering the cells medium/abstract, high/title, high/abstract — and measured
contestedness runs 4.0%, 3.9%, 0.3%, the same order. The gap between the first two is noise either way;
what the sort gets right, and all that the data supports, is putting the clean cell last.

## Escalating every drop, and what that cost

The review's finding — a drop is trustworthy only when confident *and* made after reading the abstract —
argues for one change: escalate every dropped record to pass 2 rather than only the ones the model
doubted. `needs_escalation`'s third trigger dropped its confidence qualifier, and the 749 title-only
drops went through pass 2.

| | before | after |
|---|---:|---:|
| kept | 3095 | 3230 |
| dropped | 1219 | 1084 |
| unknown, kept anyway | 73 | 138 |
| **records needing a person to look** | **37** | **14** |

135 records rescued, and **23 of the 37 hand-check items resolved themselves** — escalation kept them
with nobody reading a word.

**The two instruments agree, which neither could establish alone.** Of the 29 title-only drops the
reviewer had made a case for, pass 2 rescued 23 (79.3%); of the 720 it had not, 112 (15.6%). A 63.8-point
gap between two questions asked in opposite directions — *make a case for this* against the original
off-topic rubric — neither seeing the other's output. `check_escalation.py` is that comparison, and it
carries the negative control: rates within 20 points would mean the review's flags said nothing about
what a second look would find, and would not say which of the two was at fault.

**Escalating changed what pass 2's prompt had to say, and forgetting that was a bug.** Pass 2 opened by
telling the model that *"the titles of the records below said too little to judge them from"*. True while
escalation meant doubt; false for every record the new rule sends, which arrives with an informative title
that was confidently judged. The prompt was instructing the model to discount the evidence that had been
working. It was caught by one record — a paper whose title named its subject plainly, kept because its
abstract was "too brief to tell what the work was about" — and after the fix that record came back a
confident, correctly-reasoned drop.

The general shape is worth more than the instance: **a rule change that alters which inputs reach a
prompt can falsify the prompt**, and nothing links the two. The rule lives in Python and the claim lives
in a string several hundred lines away.

**What escalation costs, beyond the compute.** Pass 2 sees strictly more text than pass 1 — it is given
the title too — but that does not make its verdict strictly better. A thin abstract beside an informative
title can dilute rather than add, so the prompt now says to judge from whichever says more. And the
residual failures are *invisible afterwards*: a record wrongly rescued rejoins the kept corpus and appears
on no hand-check list ever again, where a wrong drop stays in `dropped.tsv` where somebody can find it.
Reading the 47 rescues the reviewer could make no case for, roughly ten are keeps on a stated inability to
tell the level — "level unspecified", "unspecified student level" — of which one has *School Level* in its
own title. That is the same failure the prompt fix addressed, on the axis it did not reach.

## Asking what a record says, once deciding stops working

**The judge's shape is right for discarding and wrong for keeping.** A drop needs a reason, so every drop
carries one and can be audited — which is what the whole review above is. A keep needs no reason at all: a
record survives whenever no test can be positively established. So a study plainly set in a university and
a study whose level is simply never stated are kept by the same rule, and afterwards nothing distinguishes
them. The corresponding audit cannot be run, because there is nothing recorded to audit.

`extract_fields.py` asks for fields instead of a verdict — who the work studies, what level it is set at,
whether a *person* is learning, what the AI does, and the words in the text that settle the level. Three
things follow:

- **Extraction is an easier task than judgement**, needing no confidence to assert a negative. `not_stated`
  is a first-class answer, which is exactly what `wrong_level` lacks: that test fails on silence, and
  silence is common.
- **A removal carries its reason.** `level: school, evidence: "TK-12 educators"` is checkable at a glance.
- **Re-filtering is free.** The fields are stored, so a cutoff can be changed and argued about without
  another model call — the opposite of the judge, where every adjustment cost a run.

A 40-record pilot earned the pilot's keep immediately: `level` is reliable and its `evidence` is quoted
from the text rather than composed, while `human_learning` looked wrong about a third of the time — a
knowledge-tracing paper and a study of educator communities of practice both came back `false` where a
person plainly is learning. So that field did not belong in a removal rule on its own, which is a thing
worth knowing before building one rather than after.

### What the full extraction says

1234 unsure keeps, no failed batches.

| `level` | n | | `human_learning` | n |
|---|---:|---|---|---:|
| not_stated | 581 | | true | 901 |
| higher_education | 296 | | false | 298 |
| not_applicable | 223 | | unclear | 35 |
| school | 96 | | | |
| mixed | 28 | | | |
| vocational | 10 | | | |

**The evidence field is what makes `level` trustworthy, and it held perfectly.** All 430 positive level
calls — `higher_education`, `school`, `vocational`, `mixed` — carry a phrase quoted from the record's own
text. All 581 `not_stated` carry none. The model never once named a level it could not quote, which is
exactly the instruction, and it is a structural check rather than a spot check: a claim with no quotation
beside it would have been visible without reading a single record.

The cross-tabs agree: `school`↔`school_pupils` 80 of 96, `higher_education`↔`university_students` 259 of
296, `not_applicable`↔`none` 108. And **the pilot's worry about `human_learning` did not survive the full
run** — it splits 181 false against 35 true under `not_applicable`, and 12 false against 279 true under
`higher_education`, so the field carries real signal and the pilot's third was small-sample noise. It is
still the softer of the two, which is why it is used only to corroborate.

### The filter, in three tiers

`filter_keeps.py` applies a rule to the stored fields. Tiers, because a single removal count would hide
that they are not equally reliable:

| tier | rule | n | |
|---|---|---:|---|
| **A** | `level` is `school` or `vocational` | 106 | removed |
| **B1** | `level` is `not_applicable` **and** `human_learning` false | 181 | removed |
| **B2** | `level` is `not_applicable`, uncorroborated | 42 | **held** |

A's removals quote their own evidence — *"secondary-school"*, *"sixth-grade students"*, *"secondary
English classrooms"*. B1's are machine-learning methods papers: explainable-AI dialogue, interactive
machine learning, thematic analysis performed with an LLM.

**B2 is held back because that is where the extraction's errors concentrate**, and its mistakes would be
false drops. Reading the 42, they are a coherent category the vocabulary has no word for: *learning
contexts that are not higher education* — professional coaching, psychotherapist and actor training,
elderly lifelong learning, cochlear-implant rehabilitation, a teachers' community of practice. So
`not_applicable` is the wrong label for most of them, since they plainly are education, while the removal
may still be right for a higher-education scope. That is a judgement about scope rather than an
extraction failure, and it wants a person. One in the tier is simply wrong — *"Exploring the Potential of
ChatGPT to improve experiential learning in Education"*, whose level is unstated rather than
inapplicable.

287 removed of 3230, leaving 2943. **Nothing existing was overwritten**: the filter writes a new
`_filtered.bib` beside the judge's output, so the corpus of record is unchanged until somebody says
otherwise.

**The vocabulary is the thing to revise first.** `not_applicable` is carrying two meanings — *not about
education* and *about education, but not at a level this asks about* — and every questionable record in
B2 sits on that seam. A `professional_training` or `informal` value would separate them, and it costs one
re-extraction rather than any new machinery.

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
| `review_drops.py` | The second reader over `dropped.tsv`, asked whether a case can be made *for* each record, with kept records mixed in unlabelled as the control. `--skip`/`--limit` take a slice, `--control-strata` draws the control per confidence level |
| `score_review.py` | Scores a review against the judge's own cells — which drops are contested, and whether the control was easier than what it was compared against. The table in *Reviewing the drops* above is its output. `--contested` also writes the hand-check list |
| `check_escalation.py` | Whether escalating the title-only drops rescued the records the review had independently flagged. Its negative control is that a small gap would mean one of the two instruments is not working, without saying which. `--rescues` lists where they disagree |
| `extract_fields.py` | Asks what a record *says* — population, level, whether a person is learning, what the AI does — rather than whether it belongs, so the keeps can be filtered on stored fields instead of re-judged. `--pilot N` reads a sample first; `--all-keeps` widens the selection beyond the ones kept on a hedge |
| `filter_keeps.py` | Applies a rule to those fields, in three tiers, removing two of them and holding the third for a person. Needs no model, so a cutoff can be changed and re-run for free. `-n` reports what would go; `--keep-uncorroborated` removes the held tier too, which is a decision rather than a flag |

Generated at runtime and **not committed** — they list the contents of a corpus that lives under
`00_stuff/`, which is gitignored research data, and this repository is public:

| File | What it is |
|---|---|
| `pilot-<seed>-<n>.jsonl`, `.tsv` | the random-sample calibration run and its hand-check table |
| `pilot-thin-<n>.jsonl`, `.tsv` | the same over every record with a title under the informative bound |
| `judged.jsonl` | the full run's resumable state |
| `<corpus>_in_scope.bib` | the corpus with the strays taken out, importable into Visualizer |
| `dropped.tsv` | every dropped record with a one-line reason — the reviewable half |
| `drop-review-<from>-<to>.tsv` | one slice of that list re-read by `review_drops.py`, with the control mixed in and labelled only here |
| `contested.tsv` | the hand-check list across every slice: each dropped record a case was made for, worst cell first, the judge's reason beside the reviewer's case, and an empty column to mark in |
| `dropped-before-escalating-titles.tsv` | the drop list as it stood before every drop was escalated to pass 2. Kept because it is what defines which records that run touched, and the review above was measured against it |
| `extracted.jsonl`, `-traces.jsonl` | the extracted fields per record, and the reasoning traces — one entry per model call, naming the keys that shared it, since a batched call yields one trace for the batch |
| `<corpus>_in_scope_filtered.bib` | the in-scope corpus with the ruled-out keeps taken out. Written beside the judge's output rather than over it, so the corpus of record does not move until somebody decides it should |
| `filtered-out.tsv` | every record the filter removed, with the fields and the quoted evidence that removed it |
| `uncorroborated.tsv` | the tier held back for a person: `not_applicable` with nothing corroborating it, which is where the extraction's errors are |

## Reproducing

```bash
B=00_stuff/rawdata/AOKK/multisource/tekoalyagentti_tutkimus_deduped.bib

python investigations/aokk-corpus-scope/judge_scope.py --bib $B --thin        # the thin-title calibration
python investigations/aokk-corpus-scope/judge_scope.py --bib $B --pilot 200   # the random calibration
python investigations/aokk-corpus-scope/judge_scope.py --bib $B              # the full run
```

A plain re-run of the last line replays from `judged.jsonl`, makes no model calls, and rewrites the
outputs — which is how to pick up a change to the filtering or the sort without paying for the run again.

Then the second reader over what it dropped, one slice at a time:

```bash
D=investigations/aokk-corpus-scope
python $D/review_drops.py --bib $B --dropped $D/dropped.tsv \
    --kept $D/<corpus>_in_scope.bib \
    --skip 0 --limit 400 --control-strata high=20,medium=20
```

`--skip` takes the next slice rather than re-asking about one already reviewed, and each run writes a file
named for its slice. The four that cover this corpus, in order, with the seeds they were run under — the
control draw is seeded, so these are needed to reproduce it and not only the dropped side:

| slice | records | control | seed |
|---|---|---|---|
| `--skip 0 --limit 400` | the hedged-abstract cell, and part of title-only | uniform, `--control 40` | 42 |
| `--skip 400 --limit 524` | the rest of title-only | `high=20,medium=20` | 43 |
| `--skip 924 --limit 200` | most of confident-abstract | `high=20,medium=20` | 42 |
| `--skip 1124 --limit 95` | the rest of it | `high=20,medium=20` | 44 |

The first ran before `--control-strata` existed, which is why its control is uniform, and why it is the
slice whose separation is flattered — see *Reviewing the drops*. Give a fresh `--seed` per slice:
repeating one re-asks about the same forty kept records instead of drawing forty more.

Then score them together, and get the list a person actually reads:

```bash
python $D/score_review.py $D/drop-review-*.tsv --dropped --contested
```

Cells first — a drop reached from the title alone and one reached from the abstract are contested for
different reasons, so their rates are reported apart and `contested.tsv` is ordered by which cell a record
came from. Read the control comparison before any of it.

Needs an LLM backend; `--backend-url` and `--model` point it elsewhere. The calibration runs above were
made against `qwen3.6-35b-a3b`, which is the model the numbers in this file describe.

A hand-check table has an empty first column: put an `x` in it on every row you disagree with. The rows
are sorted so that the cells of verdict × confidence are contiguous, because the two kinds of error are
not equally bad — a false drop loses a study silently, a false keep costs a reader one line — and they can
only be counted separately if they can be read separately.

# Sketch: interrogating a selection

**Status: a discussion sketch, not an implementation brief.** Written 2026-07-29 from a design conversation
that started as a scoping question about the RAG tool surface and turned into something larger. It is
recorded in this shape deliberately: the workflow is clear, the mechanism is not, and formalizing it into
the brief format now would freeze decisions that have not been made. Take it to a wider discussion first,
then write the brief.

## What the whole track is for

Recorded here because it is written down nowhere else, and everything below is a means to it. From the
line of thinking this started in (2024):

> Language processing tools have come a long way, and the quantity of new papers is becoming unmanageable.
> So build something to help a researcher cope with that. Take ten thousand papers and find the ones
> **actually worth reading** — specifically for what *you* are building or researching.

That is also where the name comes from — ravens collect shiny things. (The full etymology, including the
Korppi in-joke and the Corvus pun, is recorded under the naming item in `TODO_DEFERRED.md`. Note that it is
recorded *there*, filed with a decision that will be resolved and deleted: the naming question is transient
and the etymology is not, so the etymology needs a permanent home before that item is closed out.)

The stated form, from the ECCOMAS 2026 talk (`~/Documents/ECCOMAS 2026/presentation/eccomas-202607-jeronen.pdf`,
Munich, 21 July 2026), is sharper and is the version to quote:

> **One expert can rapidly screen tens of thousands of sources in a new topic, and find the relevant ones.**
> Useful for *agile research*, where the topic changes often.

**And the pipeline is not new design.** The same talk lists four stages, which this sketch converges on
rather than invents — worth knowing before treating anything below as a fresh idea:

1. **Semantic clustering** — build a map, similar items together, to ease navigation. *Built (Visualizer).*
2. **Screening** — narrow the reading set. *Partly built: the map does it by hand.*
3. **LLM as a first-pass reviewer and recommender.** *Not built. This is the map stage below.*
4. **LLM for more detailed, interactive analysis of the texts.** *Partly built: Librarian's chat, plus the
   document tools finished 2026-07-29.*

So what follows is the *how* for stages 3 and 4, not a new direction. "Screening" and "first-pass reviewer"
are also better vocabulary than "interrogation" — they say what the stage is *for*, and one of them is
already the word used in public.

Three constraints come with it, all from the same talk and none of them negotiable by a design decision
made later:

- **100% local, 100% privacy-first, 100% open source.** Not a preference — it is what makes the tool
  applicable to confidential data (internal company reports, patents), which is named as a target.
- **Not just papers.** Anything with author, year, title and optionally an abstract: company reports,
  patent databases, news articles. A design that only works for a WoS export has narrowed the tool.
- **Agile research, where the topic changes often.** "Often" here means annually, or every couple of
  years — which is slow in absolute terms and fast against how the field usually operates (pick a topic,
  build a research group around it, never change it). That working style is part of what the tool is for.
  - Note which rate matters for the cost fork below, because they are not the same rate. The **topic**
    changes on that yearly scale, and a topic change invalidates the selection and every summary with it.
    The **questions asked within one topic** change far faster — "which model is computationally
    lightweight", "what electrolyzer efficiencies get reported", "who else is doing this" are all the same
    two hundred papers. So question-independent summaries amortize across *questions within a topic*, which
    is where the saving actually is; the yearly topic change merely bounds how long a summary stays worth
    keeping, and a year is long enough that it is worth keeping.

**Each corpus is scoped to a project, and that is the sharpest design constraint here.** The team's
*methods* are the constant — numerics, PDEs, computational mechanics, and now AI. The *application domain*
follows the project: paper-machine runnability, then 3D printing of metals, then hydrogen production. Each
new domain brings its own literature, which means the tool is asked to make an unfamiliar field navigable
from a standing start, repeatedly, while the methods expertise carries across unchanged.

Three consequences, none of them obvious from the mission statement alone:

- **Any investment in a corpus has to pay back within the project it serves.** A design that gets good
  slowly — an index that improves with use, a curation step that rewards years of tending — assumes a
  single long-lived corpus, which is not the shape of project-based research.
- **The metric is time-to-competence in a new application domain**, not query quality in a familiar one.
  Note the word doing the work in the goal statement: screen tens of thousands of sources *in a new topic*.
  The expensive moment is the cold start.
- **Domain-agnosticism is structural, not a nice-to-have.** The next dataset will be in a domain the team
  is approaching fresh. Anything tuned to the vocabulary of one application area does not transfer to the
  next, and the presentation already states the goal as applicability to any field.

Two things follow from it that are easy to lose while designing mechanism. **"Worth reading" is indexed to
the reader's own work**, not to citation counts, recency, or topical similarity in the abstract — which is
why the human's selection on the semantic map is load-bearing rather than a convenience. And **the output
of the whole thing is a decision about where to spend attention**, so anything that costs more attention
than it saves has failed on its own terms, however good its retrieval was.

## One angle among several

Worth saying at the top, so that a later reader does not mistake this for the roadmap: it is one of the
directions the constellation is being pushed in, not the direction. Named siblings as of writing, each
pulling on different parts of the same tech:

- **Librarian as a digital lab assistant** — the STT / TTS / avatar HCI line, powered by MCP. Same LLM
  scaffold, entirely different interaction model. The talk's long-term vision names this one: a personal
  **co-researcher** (in the `substrate-independent` glossary sense — a dual to a researcher, different
  skills and different blind spots, a collaborator rather than a subordinate), with the avatar as an
  on-site natural-language interface, literature monitoring with novelty detection, and lab-equipment
  status by asking.
- **Trend and network tracking in research literature** — the Visualizer side, half-built.
- Whatever else the work turns up. Research as usual.

What follows is the first of those, worked out far enough to discuss. It is deliberately not worked out far
enough to build.

## The workflow this comes from

Stated as the user's own path through the constellation, because the design has to serve *that* rather than
a hypothetical:

1. Load ten thousand studies into Visualizer.
2. Look at the semantic map. Given an angle on a topic, one or two clusters are worth exploring.
3. Say the interest is hydrogen production, and the angle is photovoltaics. That is still **several hundred
   studies** — far past reading, well short of a corpus.
4. **This is where Librarian should become useful.** Take those few hundred studies and interrogate *all of
   them* against one specific question: *which is a computationally lightweight yet reasonably accurate
   model for energy production, for use in a value chain?*
5. What comes back should be a summary, plus specific keywords in case the answer is worth digging into.

## The shape this was originally imagined in, and half of it is built

The workflow predates the GUI apps. As first sketched, it was two lines:

    summaries = [summarize(x) for x in papers]   # or over abstracts
    overview  = synthesize(summaries)

— then read the overview, get on with writing the introduction section of your own paper, and dig deeper
where it turns out to matter.

**But do not infer the output format from that use.** The introduction is where this material ends up; it
is not the shape the tool should hand it back in. Introductions are written to satisfy the conventions of
publication, which is a different job from conveying findings quickly to the person who commissioned the
search — so reproducing that format would be a usability loss dressed as fidelity to the workflow. The format
is genuinely open: a query, a list, a table, prose, whatever presents the information readably. What is
*not* optional is that some **analysis** comes attached. A list of summaries is a smaller haystack, not a
found needle.

**And the first line already exists.** `raven.visualizer.importer.summarize` (`importer.py:873`) is one of
the importer's eight macrosteps, gated on `visualizer_config.summarize` (default `False`): per-entry
summaries of 1-3 sentences, with progress, ETA and caching, over a whole dataset at import time. So the map
half is shipped code that is currently switched off. What is missing is `synthesize` — and running the pair
over a *selection* rather than over everything at import.

That changes the size of the job from "build a map-reduce engine" to three smaller things: lift `summarize`
out of the importer into the library, add the reduce, and let both run against a scope rather than a whole
dataset.

**It also corrects the cost model below.** These summaries are *question-independent*: computed once per
document and reused for every question afterwards, so a rephrased question re-runs only the reduce, not the
two hundred. That is much cheaper than costing it as two hundred calls per question — but it buys the
saving by summarizing before knowing what will be asked, and a generic three-sentence summary may simply
not contain what a specific question needs ("which of these models is computationally lightweight?" is not
what a summary of an abstract is trying to preserve). So the real fork is:

- **Question-independent summaries** — amortized across every question, reusable, lossy in a way that
  cannot be predicted at summarize time. Already built.
- **Question-conditioned extraction** — precise, auditable against the question actually asked, paid for
  again on every rephrasing.

They are not exclusive: the cheap pass can run first and select the documents that get the expensive one.

## Three modes, not two

The screening pipeline is one of them, and it is easy to design as if it were the only one. The other two
are real, and they have different economics:

1. **Cold start in a new application domain.** A new project, a new field, ten thousand papers the team is
   coming to fresh. This is what the goal statement is about and where the four stages apply. The metric is
   time-to-competence.
2. **A corpus the reader already knows.** Less interesting as a workflow — and the most valuable thing
   available for *evaluation*, because only a reader who knows the field can say whether the right papers
   came back. See `evaluation/retrieval/README.md`.
3. **Co-analysis of a document against a background.** In a fast-moving literature — AI methods produce
   something worth attention every week or so — the bottleneck is not finding candidates but reading them.
   When a paper warrants more than its abstract, the useful thing is a collaborator to work through it
   *with*: fulltext, dialogue, and the document database supplying the surrounding sea of recent work that
   makes the paper interpretable. Retrieval is very much in play here, but doing a different job — it is
   **background for interpretation**, not selection. "Worth reading" was already settled by the human, so
   the screening apparatus does not apply, while the tools built for grounding and provenance do. This is
   the mode Librarian's chat, attachments and document tools already serve best, and the one where
   "co-researcher" is meant literally rather than as a slogan.

   Mode 3 spans a small range rather than a single document. The common case is one paper against the
   background; the demanding one is a handful compared against each other — *"here is a new method X, how
   does it compare to Y and Z?"*, with all three attached as fulltext. Small n, but the most context-hungry
   request in the whole system: three papers is on the order of 30k tokens before the conversation starts.

   **That makes the attachment fold budget load-bearing rather than defensive**, which is not how it was
   built. It landed on 2026-07-29 to stop a long PDF overflowing the window; this mode is a caller that
   needs it to be *good*, not merely safe. And it exposes a mismatch worth naming now: middle-truncation
   keeps the abstract and the conclusions and spends the omission on the middle, which is right for "what
   is this paper about" and wrong for "how do these three compare" — where methods and results are exactly
   what is being compared, and exactly what the middle holds. A comparison served three abstracts and three
   conclusions has been given the parts that agree and denied the parts that differ. Question-aware
   selection, or section-aware extraction, is the direction; neither exists.

Mode 3 is not a smaller version of mode 1, and building it as one would produce a worse version of both.
Screening triages many documents shallowly on a question the reader brings; co-analysis reads one document
deeply on questions that emerge while reading, with the corpus as context rather than as candidates.
Different unit, different depth, different failure mode — what breaks screening is missing something; what
breaks co-analysis is confabulating about the thing in front of it.

## Why this is not the retrieval tool surface

The document tools built in brief 10 answer a different question, and the difference is not one of scale.

- **The search agent** answers *"I have a question; go find what bears on it."* Retrieval's job there is to
  be **selective**, because the corpus is ten thousand documents and the model can read perhaps five.
- **This** is the inverse. The selection has already been made, by a human, on a semantic map. The question
  is not which documents are relevant — that was decided upstream — but what *all* of them say about one
  thing.

So top-k retrieval over a hand-picked set of two hundred would discard ninety percent of what the user
deliberately chose, undoing the work the map just did. **Retrieval here is not tuned. It is skipped.**

That has a consequence worth stating plainly, because it is what makes this a separate piece of work rather
than a setting: with nothing to rank, there is no agent loop. The tool-call round cap, the grounding
declaration, the provenance list — the whole apparatus of brief 10 — is aimed at a model deciding *what to
read next*. Here nothing decides that. The reading list is the input.

## The shape it wants: map, then reduce

Two hundred abstracts is on the order of eighty thousand tokens. It does not fit a 32k window, and in a
window where it nominally fits, the middle is read poorly. As two hundred *independent* small calls it is
embarrassingly parallel, cacheable per `(document, question)`, and resumable.

That puts it on the importer's side of the house rather than the chat turn's, and it should borrow what the
importer already has: a background task with progress and an ETA, a cancel button, and a cache keyed on
content. A chat turn is the wrong container for work that takes minutes and has a meaningful partial
result.

## The output is a handle, not a report

*Summary plus specific keywords* is the part to design around. The summary answers the question that was
asked; the keywords say what should have been asked. Visualizer already computes cluster keywords by the
same NLP machinery, so the loop closes:

    map -> select a cluster -> interrogate -> keywords -> back to the map with a sharper angle

Which suggests the result belongs in *both* apps, not only in whichever one launched it.

## Prerequisites already on the list

None of this is new construction from zero; three existing `TODO.md` items are this one item seen from
different sides.

- **Document scopes** — the load-bearing one. "These two hundred" has to be *nameable* before it can be
  interrogated, and the scope key for a Visualizer-derived selection is the dataset's file path. Every
  other part of this waits on that.
- **Visualizer↔Librarian integration** — the handoff itself, over the local network.
- **Save/load selection for reproducible reports** — the same selection, persisted, is what makes a result
  citable rather than a one-off.

## Open questions, which are the point of the discussion

Roughly in order of how much they change the design:

1. **What is the unit of analysis — abstract, or fulltext where available?** A Web of Science dataset has
   abstracts and nothing else, so abstracts are the floor. But the user's own paper stash is PDFs, and the
   answer to a question like the one above may only exist in a methods section.
2. **Is the per-document pass one-shot, or itself a small agent?** A librarian agent per document could do
   *situated* reading — "is this model computationally lightweight?" is partly a question about what else
   the corpus contains, which a document read in isolation cannot answer. Against that: an agentic pass over
   two hundred documents at five tool rounds each is a thousand calls for one question, and it reintroduces
   the round-budget problem per document rather than per turn. Current position, to be argued with: the map
   stage defaults to **one shot**, because its value is uniform coverage and one-shot is what makes two
   hundred documents affordable; tools are available but opt-in, with the cost visible when switched on. The
   agent earns its keep downstream — on the reduce, where cross-document questions live, and on the handful
   of documents the map flags as worth reading properly.
3. **One document per call, or a small batch per call?** Batching five or ten abstracts cuts the call count
   by that factor, at the cost of the per-document isolation that makes the cache and the audit trail work.
   This is the biggest lever on cost and the biggest threat to traceability, and it is genuinely open.
4. **What does a per-document answer look like?** Free text is flexible and makes the reduce stage hard.
   Something structured — relevant yes/no, plus an extraction — makes the reduce tractable and the result
   auditable, but constrains which questions can be asked at all.
5. **How does the reduce stage avoid being the same problem again?** Two hundred per-document answers is
   itself more than a window. Hierarchical reduction is the obvious answer and brings its own losses.
6. **Where does it run, and where does the result land?** A Librarian mode, a Visualizer action that hands
   off, or a CLI tool that both can invoke. Related: is the result a chat message the user can then discuss
   — which reconnects it to brief 10's tool surface, and is probably the point — or a separate artifact?
7. **What does it cost, and what is the cache key?** Two hundred calls per question is minutes on local
   hardware. That is affordable once and painful if a rephrased question re-runs the lot, so the caching
   granularity is a user-facing design decision, not an optimization.

## What this is not

Not a replacement for the chat. The interrogation produces a summary and a set of keywords; the *follow-up*
— "which of those studies said that, and what else does it say" — is exactly what brief 10's tools are for,
and it is the natural second act. The two modes are complementary, and the seam between them is where the
provenance list already lives.

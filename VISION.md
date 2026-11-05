# Raven: vision

## The problem it was built for

Our methods are the constant. Numerics, partial differential equations, computational mechanics, and now
applied AI — that expertise carries from project to project unchanged. The **application domain** is what
rotates: paper machine runnability, then 3D printing of metals, then hydrogen production. Each new project
arrives with its own literature, in a field we are approaching fresh.

That rotation is the thing worth designing for. The hard moment in a new project is the **cold start**: ten
thousand papers you have never read, in a vocabulary you have not yet learned, and a fixed number of weeks in
which to find the hundred that actually bear on what you are building.

Stated as the goal:

> One expert can rapidly screen tens of thousands of sources in a new topic, and find the relevant ones.

The metric that follows is **time-to-competence in an unfamiliar domain**.

The thing that has to survive the rotation is **the instrument**. Each project's corpus is assembled at the
start, used for the duration of the project, and becomes largely irrelevant once the topic moves. What carries
across is the tool that makes an unfamiliar literature navigable from a standing start.

Two consequences, and both constrain the design:

- **Any investment in a corpus has to pay back inside the project it serves.** A system that gets good slowly
  — an index that improves with use, a curation habit that rewards years of tending — assumes one long-lived
  collection. That is not the shape of project-based research.
- **Domain-agnosticism is structural, not a nice-to-have.** Anything tuned to the vocabulary of one
  application area does not survive the rotation.

"Worth reading" here means worth reading *for what you are building*. Not most cited, not most recent, not
most similar to the query — which is why a human judgement stays in the loop rather than being automated
away.

## What Raven is

An instrument for that job. In one sentence: **put your documents in a folder, get a map of them, then ask
questions about the parts you care about.**

Within one project, that means one document collection with several views onto it:

- **A semantic map** (Visualizer) — documents placed by meaning, clustered, with keywords per cluster. This
  is how ten thousand becomes navigable: you can see the shape of a literature before you have read any of it.
- **A conversation** (Librarian) — a *multiversal* LLM frontend: conversations branch, so a line of inquiry
  can be followed without losing the one it came from. It searches those same documents, reads them, cites
  what it used, and works through a paper with you.
- **A face** (the avatar) — speech in and out, an animated character, intended for the case where you are
  across the room with your hands busy rather than at the keyboard.

The name comes from the job: ravens collect shiny things.

The important word above is *same*. Today the map and the conversation read two different collections that
happen to share a machine — Visualizer imports bibliography files, Librarian reads a document database. That
split is a historical accident, and closing it is the main architectural work ahead. Under one collection,
"load ten thousand studies into Visualizer" stops being an import and becomes a **view**: drop the files in a
folder, and both the map and the conversation already see them.

## What comes out: a review, for one problem

What a session with Raven produces is a review of the literature for the problem in front of you. That is
what a review paper offers too, but a review paper was written for the field as its author saw it, often
years before you arrived with your question. So Raven competes with the review paper as a genre.

**And the review it produces is only partly prose.** A literature has a shape — what it covers, how much
there is on each topic, which topics sit near each other, where nothing has been done — and prose can give
that shape only one item at a time: Smith did this, Jones extended it to that. By the end of the paragraph
the shape is gone, the way the reader of a long German sentence has lost the thread by the time the verb
arrives ([Mark Twain, *The Awful German Language*](https://en.wikipedia.org/wiki/The_Awful_German_Language),
1880). It is a picture in two dimensions read out as a line.

There is a ladder here. A recorded talk can only be played through. Its transcript is still a line, but it
can be skimmed and searched. A map is the next rung: what it shows was never put into a line, so nothing is
lost to the ordering.

So the review has three parts, each carrying what it carries best:

- **The map carries the structure**: the topics, their sizes, their neighbours, the gaps.
- **Prose carries the argument**: why a cluster matters for your problem, what is settled and what is
  contested, what nobody has tried. That is judgement, and a map cannot make it.
- **The conversation answers the follow-up.** A static review answers the questions its author thought of.
  Here you can open a cluster and ask the next one.

The map is part of the review, not a step before it.

## What it asks you to accept

A review made this way will contain mistakes. At one wrong decision in a hundred, five thousand records hold about fifty, and the
chance of none is about 10⁻²². At this scale no person checks them all, so some will never be caught.

**That is no new loss, because the alternative never had the property it is credited with.** A review done by hand cannot read
the first ten thousand matches, and it applies its own criteria inconsistently. People do, and rarely measure or admit it. An
exact, exhaustive review was always a fiction; what changes at this scale is that it can no longer pass for anything else.

**The trade is to see the whole picture, with artifacts in it, in place of being sure of a few pixels.** Engineering made a
similar move when [uncertainty quantification](https://en.wikipedia.org/wiki/Uncertainty_quantification) replaced point values
with distributions, following the lead of [ensemble weather forecasting](https://en.wikipedia.org/wiki/Ensemble_forecasting).
Scientific knowledge outside mathematics has always worked this way, tentative and updated on evidence. A hundred-percent
standard is a strange one to hold a method to, when nothing else in science meets it.

**What made that move a gain was measuring the error, and tolerating it was only the price.** So the trade comes with obligations:

- **Measure the rate, and report it with the result.** People rate a sample without seeing the tool's answers, and that gives a
  posterior on the error rate rather than a pass or a fail. The precision wanted sets the size of the sample: with no mistakes among
  the rated records, 15 of them bound the error rate in what was sampled at 20%, 30 at 10%, 60 at 5%, 150 at 2% and 300 at 1%, with
  one-sided 95% confidence ([Hanley & Lippman-Hand 1983](https://doi.org/10.1001/jama.1983.03330370053031)). Then convert it to the share of
  relevant work missed, which is what a review's reader needs. *How to read such a number*, below, spells this out. This much is
  already a strict upgrade: a review by hand has an error rate too, and does not know it.
- **Look for errors with a structure.** Random mistakes blur the picture. A misread criterion removes a whole kind of work at once,
  and that looks like a real gap in the literature. So the check is stratified by what made each decision.
- **Put the error bars on the claim, not the record.** "Most of this work is on X" survives a few percent of errors. "Nobody has
  studied Y" does not, since one wrongly removed record refutes it, so a claim of absence gets a check by hand. Careful qualifiers
  were always part of scientific writing; here they have numbers behind them.
- **Say what is stable.** A grouping run several times shows what holds and what moves, as an ensemble forecast does.

**In return, it can be audited as no manual review can be.** Every decision keeps its reason, the model and the prompt that made it,
and it can be run again. A person's reasoning on the four-hundredth abstract of the afternoon cannot be replayed. That is a
different kind of rigour from a fixed protocol such as [PRISMA](https://en.wikipedia.org/wiki/Preferred_Reporting_Items_for_Systematic_Reviews_and_Meta-Analyses),
and we think it is the kind that scales.

### How to read such a number

Written for scientists who are not statisticians. The example: 30 removed records rated by hand, and no mistakes
found. In plain words:

> Unlikely (95%) that more than about 10% of the removals are wrong. The likeliest figure is a few percent.

**Where the 95% comes from.** Call the true error rate *p*. Each record is then right with probability 1 − *p*, so a clean sample of 30 has
probability (1 − *p*)³⁰. That falls as *p* rises: about 74% at *p* = 1%, 4% at 10%, and 0.1% at 20%. The bound is the rate at which it reaches
5%, by solving (1 − *p*)³⁰ = 0.05, which gives *p* = 9.5%. So the "95%" means: above that rate, a sample this clean would turn up less than 5%
of the time. Note this is a statement about the sample, for each possible *p*. And it is one-sided, because the question is only how high the
rate can be. A clean sample of this size cannot tell 1% from 0%, and no finite sample can establish 0%.

**What people want to say instead** is "the rate is probably under 10%", a statement about *p*, and that takes Bayes's formula. *p* is the unknown,
so the prior is on *p*: before looking, take every rate from 0 to 1 as equally plausible. Bayes's formula is the joint probability written both ways,
f(*p*, clean) = f(*p* | clean) P(clean) = P(clean | *p*) f(*p*), and knowing three of the terms gives the fourth:

- **f(*p*) = 1**, the prior.
- **P(clean | *p*) = (1 − *p*)³⁰**, the likelihood.
- **P(clean) = ∫₀¹ P(clean | *p*) f(*p*) d*p* = ∫₀¹ (1 − *p*)³⁰ d*p* = 1/31**, the evidence: the likelihood marginalized over *p*, which is its average under the prior.
- **f(*p* | clean) = P(clean | *p*) f(*p*) / P(clean) = 31 (1 − *p*)³⁰**, the posterior.

Integrating it, the chance that the rate is below *x* is 1 − (1 − *x*)³¹. At *x* = 9.5% that is
95.5%. The two readings nearly agree because the only difference is an exponent of 31 in place of 30. The same posterior gives the best single guesses:
a mean of 1/32, about 3%, and a median of about 2%.

**What the 10% says**, with four cautions:

1. **It is a ceiling, not an estimate.** After a clean sample the best single guess is about 2–3%, as above. So: probably a few percent, and very
   likely under 10%.
2. **It covers only what was sampled.** Rate 30 removed records, and the bound is on the removals: at most 10% of what was removed is
   actually in scope. It says nothing about the kept records. Each group needs a check of its own, which is why the sample is stratified.
3. **It is the case of no mistakes found.** Finding some moves the ceiling quickly:

   | Mistakes in 30 | 95% upper bound |
   |---|---|
   | 0 | 9.5% |
   | 1 | 14.9% |
   | 2 | 19.5% |
   | 3 | 23.9% |

4. **It rests on two assumptions.** The sample is random, and the person rating it is right. Where people disagree among themselves, the
   "error rate" partly measures that, which is what rating some records twice is for: two human opinions, preferably from different
   people. Failing that, the same person after enough time for a fresh look, which shows how consistent they are rather than how far
   people agree.

**And the trap: 10% of removals is not 10% of the relevant work.** Say the tool removed 2,000 records and kept 1,000 relevant ones. At
the ceiling, 200 of the removed ones are relevant, so 200 of 1,200 relevant studies are missed: 17%. Removal is the larger pile here, as it
often is in a search that casts a wide net, and then a small rate on it weighs more than it sounds. With 500 removed, the same rate misses
50 of 1,050: 5%. So convert with the actual counts: the share missed is *rR* / (*rR* + *K*), for an error rate *r* on *R* removed records and
*K* relevant ones kept. That share is the number for a review to report.

## The pipeline, and where it actually stands

The pipeline has four stages. Their honest status:

| Stage | What it does | Status |
|---|---|---|
| 1. Semantic clustering | Build the map; similar items near each other | **Built** |
| 2. Screening | Narrow to a reading set | **Partly** — you do it yourself, using the map |
| 3. LLM as first-pass reviewer | Read all of the selection against one question | **Not built** |
| 4. LLM for detailed analysis | Work through the texts interactively | **Partly** — chat and document tools work |

Stage 3 is the gap, and it is a specific one. This four-stage plan is also, near enough, the problem-and-solution
pair Raven was started from in 2024 — the middle of it is what has not been built yet.

The workflow stage 3 serves: you look at the map, zoom into the cluster or two that look promising, and select
everything visible. That is still several hundred studies — far past reading, well short of a corpus. Then you
ask them all one question — *which of these describes a computationally lightweight yet reasonably accurate
model I could use in a value chain?* — against **all of them** rather than against the five a search would
return.

That is not a retrieval problem. The selection was already made, by a person, on the map. Ranking two hundred
hand-picked documents and keeping the top five would discard ninety percent of what the user deliberately
chose. So the shape is map-and-reduce: a small independent pass per document, then a synthesis. That makes it
cacheable and resumable, and it is also why stage 3 is not a button someone forgot to add — the work takes
minutes, has a meaningful partial result, and needs progress reporting, cancellation and resumption. A chat
turn is the wrong container for it.

Half of it already exists: per-document summarization is shipped code, currently switched off because it was
built to run over an entire dataset at import time rather than over a selection.

What comes back should be a **summary plus keywords** — the summary answers what was asked, the keywords say
what should have been asked, and they lead back to the map with a sharper angle. The loop can close by voice
as well, once the map is something the assistant can address: *"could you highlight the studies you just
recommended?"*

The output of the whole system is a decision about where to spend attention, so anything that costs more
attention than it saves has failed on its own terms however good its retrieval was.

## Three commitments

**Local, private, open source.** Local and private are what make the tool applicable to material that cannot
leave the building — internal company reports, unpublished results, patent work. A cloud tool is simply
disqualified for that class of data.

Local does not mean that nothing ever touches the network. Raven downloads its AI models from HuggingFace the
first time each is needed, and runs them on your machine from then on. Beyond that, the network is reached
only by features that exist to reach it, each a tool you run or a switch you can turn off: the arXiv tools,
the chat's web search and page fetching behind its *Internet* toggle (on by default), and filling abstracts
from Crossref (off unless asked for). The [README's privacy section](README.md#privacy) says what each one
sends, and to whom. Your own LLM backend and Raven-server are wherever you put them.

Open source belongs to the same argument. Everything in Raven is open source. A claim about what a tool sends
where is an assertion in a closed product and a checkable fact in an open one, so openness is what makes the
privacy commitment credible to the people it is meant to serve. Beyond that, spreading what
works is one of the functions of research, and open source is what publication looks like for software — the
fuller version of that argument is at the end of this document.

**One corpus, several views.** Each project brings its own collection. The unification described above is what
makes the handoff between the map and the conversation a change of view rather than a data transfer, and it is
what lets the same machinery be pointed at different subsets — which in turn is what makes *"show me the map"*
and *"search my documents"* into things you can simply say.

**Traceable.** Two separate obligations, which the project names separately so they do not get confused:

- **Disclosure** — this text was AI-generated. An EU AI Act Article 50 obligation, implemented. It applies to
  any system supplied for use in the EU, whether free of charge or for payment — the Act's open-source
  exemption does not reach Article 50.
- **Provenance** — this claim came from that source. The one researchers care about. A first-pass reviewer
  whose output cannot be traced back to specific documents is useless for a review you have to defend to a
  referee, so this is a value proposition rather than compliance overhead.

## What is actually running

How much the forward-looking half of this document is worth depends on what has already been built: it is the
evidence that the work gets done, and it is the base the rest would stand on.

**The map works at the scale it was built for.** The hydrogen production dataset is roughly twelve thousand
publications, mapped, clustered and searchable. That is the demonstration Raven usually opens with, because
sorting ten thousand studies into a navigable shape is the part that was hard.

**And it has been applied outside our own field, with the results published.** Two bibliometric and content
analyses in education sciences — sustainability education in biology, and in geography — both peer-reviewed
and published in 2026, as joint work with a domain expert in that field. Education sciences is nowhere near
computational mechanics, which is what makes these the real test of the domain-agnosticism claim: the tool
transferred to a literature the team has no methods expertise in, in collaboration with someone who does, and
the output survived review. Raven also catalogues a personal collection of arXiv preprints, which is a third
domain again.

> E. Jeronen and J. Jeronen (2026). *Bibliometric and Content Analysis of Sustainable Education in Biology…*
> Education Sciences 16(2), 201. https://doi.org/10.3390/educsci16020201
>
> E. Jeronen and J. Jeronen (2026). *Bibliometric and Content Analysis of Sustainability Education in
> Geography…* International Journal of Educational Sciences 9(2). https://doi.org/10.53935/2641-533x.v9i2.1156

**The conversational half is real, if younger.** Librarian runs a local model against that same document
collection with hybrid retrieval, tool calling, web search and fetch, branching chat history, and the animated
speaking avatar with speech in and out. What it cannot yet do is stage 3.

**"Not just papers" became literal this summer.** The document database now ingests Word and OpenDocument
files, saved web pages, PDFs and plain text, which covers the company reports, patents and news articles named
as targets — most of which never arrive as bibliography exports.

**The hardware floor is one workstation.** At the desk: the language model alone on a 24 GB external card, and
all nine server modules including the avatar on the internal 8 GB one.

## Can this run on local models?

The question behind the whole design, and worth answering directly, because "local" is the constraint that
everything else is built around.

The answer is yes, at three hardware tiers, with the model chosen per tier on measured performance rather than
on reputation. **The table below is current as of August 2026** — open-weight models turn over every few
months, so treat a stale copy of it as a description of what was true rather than as a recommendation:

| Hardware | Model | Use |
|---|---|---|
| 8 GB GPU, mobile | Qwen3.5-4B | Working away from the desk |
| 16 GB GPU, mobile | Qwen3.5-9B *(provisional)* | The better mobile option where the card allows it |
| 24 GB external + 8 GB internal | Qwen3.6-27B, or 35B-A3B | Serious single-workstation use |

Gemma is kept installed as the multilingual alternative.

Three things this has taught us that are worth knowing before planning around local models:

- **The small models are better than expected at the jobs that matter here.** The 4B scored perfectly on
  retrieval across every corpus size tested, and reads fine detail out of a screenshot — which is what makes
  the mobile tier a real working configuration rather than a demo.
- **They are literal**: they do what the prompt says rather than what a charitable reader would guess it
  meant. Given a self-contradictory instruction, one model spent fifty thousand characters of reasoning trying
  to resolve it and never answered — a loud failure, where the alternative would have been a plausible answer
  resting on a silently repaired instruction. That is the behaviour to want when the same prompt is about to
  run over twelve thousand documents, since a small rate of quiet reinterpretation is a large number of wrong
  readings with nothing marking them. The cost is that the burden moves to whoever writes the prompt:
  ambiguity a person would have resolved charitably becomes an error to fix.
- **The backend is not wired in.** Model choice is configuration, so the lineup moves as the open-weight field
  does, and it has moved twice already during development.

## Where it is going

The long-term direction is a **co-researcher**, with the
*[co-](https://github.com/Technologicat/substrate-independent/blob/main/glossary.md#co-)* read as in *cosine*
or *cohomology*: something complementary, filling a different role in the same space, with different skills
and different blind spots. A collaborator rather than a subordinate — which is the reading the prefix usually
collapses to.

Three lines lead there:

- **Memory.** What separates a chat frontend from a collaborator that accumulates context about the work.
- **Literature monitoring with novelty detection.** A new paper that fits no existing cluster is, plausibly,
  one worth looking at. The mechanism is already implied by the map: distance from the existing structure.
- **The lab assistant.** The avatar as an interface you speak to, able to ask instruments for their status and
  an experiment controller for the results of a run, over standard tool protocols. Speech in and out works
  today, but starts with a key combination; the across-the-room version waits on wake-word listening.

Two bounds on that, stated deliberately because this design space contains plenty of things nobody should
build. It **assists someone doing their own work** rather than doing the work instead of them — the
researcher stays the researcher. And it **reports rather than acting on its own initiative**: its job is to
tell you things so that you can decide.

### When the corpus framing would stop fitting

"One corpus, several views" has been deciding what gets built next, so it is worth knowing in advance what
would retire it.

*"Find me papers about X"* is corpus-shaped. *"What is the reactor doing right now?"* is not — that is state
in a controlled system, and no amount of document indexing addresses it. If the lab-assistant line grows,
the unifying idea stops being the collection and becomes something closer to *the set of things the assistant
can address*, with the corpus as one addressable thing among several.

We are not there yet, and the corpus framing is the right one for the immediate future. But it is a
description, not a principle, and it should be allowed to lose.

## Why free and open

A local-first research assistant with a face, running on a consumer GPU and phoning nowhere, is currently an
argument most people would tell you loses — to scale, to the cloud, to the platforms. It stops being an
argument the moment someone can run it. **Demonstrated feasibility is a different object from asserted
feasibility.**

That effect works through three channels:

1. **Existence proof.** Needs only the artifact to exist. Settles feasibility, permanently.
2. **The ideas spread and get reimplemented.** Needs the work to be legible and borrowable — public design
   documents, comments that explain *why*, and licensing that lets a piece be taken without adopting the
   whole. It does **not** require Raven to win.
3. **Raven itself is adopted.** Needs a user base.

Most open-source projects are pitched on channel 3. Ours is built not to need it — which is also why the
design discussions, the measurements and the failures are in the public repository rather than in a drawer.

## A note on how it looks

Raven is built to a deliberate aesthetic: cyberpunk in register — a 1980s–90s genre, which is why it reads as
retrofuturistic today. On the avatar this reaches the lighting, where the video postprocessor implements the
digital bloom of early-2000s anime: real code in the render path rather than decoration applied afterwards.

On the working surfaces — the map, the chat log, dialogs — it shows instead as palette, typography, and the
character of a transition. Often at no cost at all: the Viridis colormap on the map was chosen because it is
perceptually uniform, and looks the part anyway.

One rule governs the rest: **when the register and usefulness conflict, usefulness wins.** The aesthetic is
what the product is written in, not something it serves, and on the surfaces where people read and think,
readability wins outright.

---

*Further detail lives in the repository: `briefs/design/` for the design discussions this document
synthesizes, `investigations/` for the measurements, `briefs/` for the implementation work.*

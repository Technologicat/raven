# Sketch: interrogating a selection

**Status: a discussion sketch, not an implementation brief.** Written 2026-07-29 from a design conversation
that started as a scoping question about the RAG tool surface and turned into something larger. It is
recorded in this shape deliberately: the workflow is clear, the mechanism is not, and formalizing it into
the brief format now would freeze decisions that have not been made. Take it to a wider discussion first,
then write the brief.

## One angle among several

Worth saying at the top, so that a later reader does not mistake this for the roadmap: it is one of the
directions the constellation is being pushed in, not the direction. Named siblings as of writing, each
pulling on different parts of the same tech:

- **Librarian as a digital lab assistant** — the STT / TTS / avatar HCI line, powered by MCP. Same LLM
  scaffold, entirely different interaction model.
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
2. **One document per call, or a small batch per call?** Batching five or ten abstracts cuts the call count
   by that factor, at the cost of the per-document isolation that makes the cache and the audit trail work.
   This is the biggest lever on cost and the biggest threat to traceability, and it is genuinely open.
3. **What does a per-document answer look like?** Free text is flexible and makes the reduce stage hard.
   Something structured — relevant yes/no, plus an extraction — makes the reduce tractable and the result
   auditable, but constrains which questions can be asked at all.
4. **How does the reduce stage avoid being the same problem again?** Two hundred per-document answers is
   itself more than a window. Hierarchical reduction is the obvious answer and brings its own losses.
5. **Where does it run, and where does the result land?** A Librarian mode, a Visualizer action that hands
   off, or a CLI tool that both can invoke. Related: is the result a chat message the user can then discuss
   — which reconnects it to brief 10's tool surface, and is probably the point — or a separate artifact?
6. **What does it cost, and what is the cache key?** Two hundred calls per question is minutes on local
   hardware. That is affordable once and painful if a rephrased question re-runs the lot, so the caching
   granularity is a user-facing design decision, not an optimization.

## What this is not

Not a replacement for the chat. The interrogation produces a summary and a set of keywords; the *follow-up*
— "which of those studies said that, and what else does it say" — is exactly what brief 10's tools are for,
and it is the natural second act. The two modes are complementary, and the seam between them is where the
provenance list already lives.

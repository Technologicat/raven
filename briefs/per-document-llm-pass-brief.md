# Brief: the per-document LLM pass

**Unnumbered**, following the convention adopted after brief 15 — `markdown-block-rendering`,
`wake-word-voice-input` and `ligature-repair` are all unnumbered, and numbering was abandoned at 16.

> **Previously "brief 17".** Scoped out of brief 15 on 2026-08-04, given a reserved number, and never
> written. The reservation is now retired: refer to this by name. The dangling reference in
> `briefs/researchers-night/done/15_headless-agent-driver-brief.md` (`:362`, `:946`) stays as it stands —
> `done/` is a historical record and is not retconned — but the live content is here.

> **Line numbers are as of 2026-08-12** and want verifying.

**Gate: `next`.** Not exhibit work. But see the user count below: this is the most-pointed-at unwritten
thing in the project, and two of its users now carry measured costs.

## What it is

A **per-document LLM pass**: run the same question over every item in a set, with **retry, cache, resume and
progress**.

It sits one level above `raven.librarian.agent` (brief 15). That surface answers "run one turn and tell me
what happened"; this answers "run one turn per document over two thousand documents, and survive the
afternoon."

## Why now: six users, none of which knew about the others

Each surfaced from a different direction. That is the pattern that says a shared primitive is missing,
rather than six features being wanted.

1. **`raven-pdf2bib`** — eight `perform_throwaway_task` call sites, each wrapped in its own hand-written
   retry loop: the same six lines, eight times, in one 1058-line file. No caching and no resume, so a crash
   at document 2400 restarts from zero.
2. **`rag_live_corpus`'s persistence layer** — a `PersistentForest` per sample plus a JSONL ledger, worth
   lifting wholesale. Those runs take an hour and the machines reboot.
3. **`briefs/design/corpus-interrogation-sketch.md`'s map stage** — `summaries = map(summarize, docs)`. Note
   `summarize` is *shipped code that is currently switched off*, sitting in the importer with progress, ETA
   and caching already written; what is missing is `synthesize` and a place for both to live.
4. **Mid-run LLM backend recovery for batch tools** — the model-loaded work made `raven-pdf2bib` and
   `raven-importer` stop at *start time* on a backend that is unreachable or has no model loaded, and
   explicitly deferred the mid-run case. CC's three deferred questions are this brief's scope exactly: how
   long to wait before giving up, whether to resume or restart, and what to do with the documents already
   written.
5. **A crash during ingest loses the whole run** — the measured version. The delayed-commit coalescer defers
   a commit one second after each finished read, so on a large corpus it never fires until the reads *stop*:
   **~40 minutes on the 1268-PDF fulltext corpus (2026-08-06)**, with every extracted document pending in
   memory and nothing on disk.
6. **Two shapes found while implementing brief 15** — a VLM pass over page images, and "here is a fulltext
   PDF, what does it say about X?" over a set. Brief 15 names the batch mechanics of both as this work.

Corpus sizes make several of these concrete rather than prospective: ~12k hydrogen abstracts already
ingested, ~2500 one-page ECCOMAS 2024 conference abstracts, an arXiv AI fulltext set of 1200+ full papers.

## What it is not

**Not an orchestration framework**, and it must not become one — no agent-role DSL, no supervisor
abstraction, no declarative pipeline. The scope is a loop with durability. Brief 15's scope note applies
here, with the correction recorded in the triage decisions: what is ruled out is a *framework*, not
scripting, and the scripting language is Python.

**Not the map-reduce engine either.** The corpus-interrogation sketch is explicit that `summarize` already
exists and what changes the size of that job is *"lift `summarize` out of the importer into the library, add
the reduce, and let both run against a scope"*. This brief is the lifting-and-durability half. `synthesize`
belongs to that sketch.

## Design starting points

### Resume is the load-bearing feature

Everything else here is convenience; resume is what makes an hour-long run survivable. Two of the six users
exist *only* because it is missing.

The shape follows from what already works: `rag_live_corpus` keeps a JSONL ledger beside a
`PersistentForest`, and that pairing is worth lifting rather than redesigning. A ledger of completed items,
appended per item, gives resume, progress and the caching story at once — resume is "skip what the ledger
already has".

### Reset between documents, not one shared context

Settled 2026-08-11 while discussing the multi-agent question. A map stage processes documents independently,
so **a fresh `Forest` per item** gives isolation as well as bounded memory — no chance of one document's
context leaking into the next. The memory bound falls out of correct semantics rather than being arranged
for. This is what `pdf2bib` already did.

Persist per item where the run is worth keeping (`PersistentForest`), reset where it is not.

### The failure taxonomy is the interesting part

Not all failures are the same and the item's three deferred questions are really about telling them apart:

- **A bad document** — one item fails, the rest are fine. Record and continue.
- **A backend that has gone away** — every remaining item will fail. Stopping is right; the questions are
  how long to wait first, and whether to resume or restart afterwards.
- **A crash of the run itself** — nothing gets to decide anything, which is why the ledger has to be on disk
  before it is needed rather than written at the end.

Conflating the first two is the current failure mode: a batch run against a dead backend produces a
thousand "failed" documents that were never tried properly.

## What this brief must settle before implementation

1. **Where it lives, and its name.** Beside `agent` as a sibling module, or as a layer in the same file.
   Brief 15 left its own naming open for the same reason and the answer came from what the module did; do
   the same here.
2. **The ledger format and location.** JSONL beside the datastore is the existing precedent. Whether it is
   the same file for progress and for results, or two.
3. **Cache key.** What makes two runs "the same item" — content hash, path, or a caller-supplied id. The
   sidecar store is already content-addressed, which argues for the first, but a caller re-asking a
   *different question* about the same document must not hit the cache.
4. **The mid-run backend policy**: how long to wait, resume or restart, and what happens to documents
   already written. CC deferred these deliberately; they are the reason this brief exists rather than a
   detail of it.
5. **Progress reporting shape.** `summarize` in the importer already has progress, ETA and caching — read it
   before designing, since lifting may be most of the work.
6. **Whether `raven-pdf2bib` is converted as part of this or after.** It is the loudest user (eight
   hand-rolled retry loops) and the best test that the API is right; it is also a 1058-line file that
   nothing else depends on this brief to fix.

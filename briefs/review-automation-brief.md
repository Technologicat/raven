# Automating a literature review: what the two current studies need next

*Status: collecting (2026-10-29). Nothing here is scheduled yet; the order below is a first proposal.*

Two studies are using Raven's paper tools end to end, and between them they have turned up the next layer
of work:

- **AOKK** — AI agents in higher education: a multi-database search of a few thousand records, screened with
  `raven-scopebib`, with a team screening by hand in parallel.
- **Health education** — sustainability education in health education, the third study in a series after
  biology and geography: a Scopus search of about a hundred records, screened with `raven-scopebib` and by
  hand on 2026-10-28/29, with the final set due early in the week of 2026-11-02.

The detail of each item lives in the entry named, in `TODO.md` or `TODO_DEFERRED.md`; this brief says how
they fit together and in what order. Settled decisions that left no diff go in the log at the end.

## Screening

1. **Carry a person's decisions from one run to the next.** Done by hand twice on 2026-10-29: a decisions
   file keyed by record, the next run's check-by-hand list pre-filled from it, and a list of the decided
   records the new run settled the other way — which `finish` cannot reach, since they never reach the
   worklist. The same thing is what the AOKK team needs when their hand labels arrive: a new list with
   their decisions already filled in, so nothing is redone. Not filed yet; this brief is its first entry.
2. **`finish` overturning a removal** — `TODO_DEFERRED.md`, *`raven-scopebib finish` cannot overturn a
   removal*. The other half of item 1: a decision against a run's outright keep or removal, applied by the
   tool and recorded in the hand audit, rather than edited into the final bib by hand.
3. **Preparing a definition, and drafting one from notes** — `TODO_DEFERRED.md`, *Preparing a
   `raven-scopebib` definition file, as a manual section*, which also carries the `raven-scopebib draft`
   idea. Both definitions so far were drafted by an agent from the researchers' observations.
4. **Stratified samples for replicates** — `TODO_DEFERRED.md`, *Stratified samples for `raven-scopebib`
   replicates*. For the test-retest figure a methods section reports; gated on a full run's verdicts, which
   AOKK's full run supplies.

## Preparing the corpus

5. **More abstract sources than Crossref** — `TODO.md`, *`raven-fillbib` v2*. PubMed is named there for the
   health-education corpus in particular.
6. **`raven-deduplicate --judge` onto the shared sweep** — `TODO_DEFERRED.md`, of that name.
7. **Words broken mid-line** — `TODO_DEFERRED.md`, *Dehyphenate a word broken mid-line…*. Two cases in the
   health-education corpus so far.

## After screening

8. **Per-study attributes from the fulltexts**: school levels and countries, the table the two earlier
   studies built by hand — `TODO.md`, *Per-study attributes from the fulltext*. The health-education set,
   counted by hand as well, is its calibration test. The same pass can flag a screening decision that the
   fulltext disagrees with.
9. **Per-cluster cohesion in the Visualizer** — `TODO_DEFERRED.md`, under *Visualizer: quality flags…*. For
   reading the accepted set's map: on the health-education corpus the 2D map misrepresented which cluster
   was the real one.

## To be added

The maintainer has one or two more items for this set (2026-10-29).

## Decision log

- **2026-10-29**: this brief created, to hold the review-automation items of both studies as one set.

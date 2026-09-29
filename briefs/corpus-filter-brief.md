# Brief: the corpus filter

**Filed 2026-09-29. Unnumbered.** Generalizes `investigations/aokk-corpus-scope/` into a `raven.papers`
tool: given a `.bib` from a literature search and the question the search was meant to answer, flag the
records that are not about it, with a reason for each, so a person can review them out.

**Gate: `per-document-llm-pass-brief.md` lands first** (Juha, 2026-09-29). Of the prototype's three
scripts that call a model, two carry their own copy of a resumable per-record loop and the third is
driven slice by slice with `--skip`. This tool should be built on the shared primitive rather than lift
a third copy. That brief lists this as its eighth user, and carries the two
requirements the prototype adds to it: an instrument stamp in the cache key, and batching.

## The specification is the investigation README

`investigations/aokk-corpus-scope/README.md`, from *"Where this is headed: a `raven.papers` corpus
filter"* to the end. It was written on 2026-09-02 and brought up to date on 2026-09-04, while the
apparatus was in front of us. This brief does not restate it. What it adds is the scope of v1, and the
questions the README leaves open.

What that section settles, in one line each:

- **Four instruments, not one judge.** Sift (is it screenable), judge (is there evidence it is off topic),
  reviewer (can a case be made that a drop belongs), extractor (what does a kept record say). The sift has
  shipped already, as `raven-siftbib`.
- **The judge's parameter is a list of named tests**, each asking for positive evidence of one way to be
  off topic, with silence keeping. The prototype's three tests are the first worked example rather than
  the schema.
- **Constants measured from the prototype's corpus are derived at run time** from the corpus in hand:
  `MIN_INFORMATIVE_WORDS` and `TEASER_CHARS`.
- **Five lessons a generalized tool should start with**: stamp the instrument, tiers that separate
  wrong from out of scope, escalate every drop, a `not_stated` value with a quotation rule, no mixing of
  JSON literals with strings in one field.
- **The reviewer is the piece most worth lifting.**
- **Sift after calibrating, not before** (*"The run plan"*): the title-only records are the inputs most
  likely to break a new rubric, so calibration runs on the unsifted corpus.

## v1 takes a scope question, not research questions

Juha, 2026-09-29: *"scope questions are good enough for v1. RQs are for later."*

A **scope question** is what the search actually asked — for the prototype, *studies on different aspects
of the use of AI agents in higher education*. A research question is narrower, and the 2026-09-01
decision in `researchers-night/done/aokk-corpus-scope-classification-brief.md` already put those in a
separate, later pass, after the corpus has been ingested into Librarian. So the filter is the coarse cut
that makes a corpus worth ingesting, and nothing in v1 asks about research questions.

## What this brief must settle before implementation

1. **How a user writes the tests.** The scope question is one string; the named tests are not, and
   each wants a description good enough to be a prompt. A file beside the corpus is the obvious shape.
   Whether the model could *propose* a test list from the scope question, for the user to edit, is an
   open idea rather than a decision.
2. **How the extractor generalizes, which the README does not say.** Its fields — population, level,
   whether a person is learning — are this corpus's, as much as the judge's tests are, and so are
   `filter_keeps.py`'s tiers over them. The same move probably applies (the vocabularies become a
   parameter), but it is untested, and a field turned out unreliable in the prototype's own pilot
   (`human_learning`, wrong about a third of the time), so a user-supplied vocabulary would need a pilot
   of its own.
3. **Calibration as a first-class step.** The prototype calibrated with `--pilot N` and `--thin` before
   any full run, which was step 0 of the original brief, and the README's findings come largely from
   those runs. What the pilot hands the user to read, and in what order, wants designing — the review TSV's verdict ×
   confidence sort is the starting point.
4. **One console script or several.** `raven-siftbib` is already its own tool, and the README's case for
   that (no LLM, same shape as `raven-fixbib` and `raven-deduplicate`) holds. Whether judge, reviewer
   and extractor are one tool with modes, or three, is open.
5. **Where the outputs go, and what they are called.** The prototype writes a filtered `.bib`, a
   dropped-with-reasons TSV, a held-for-review TSV, and JSONL answers — the extractor's named after
   its instrument fingerprint. How much of that the per-document primitive owns is decided by that brief, not this one.

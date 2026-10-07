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

## A lead from the dedup judge: show a judge what the rules already computed

Untested, carried over from `done/bibliography-dedup-brief.md`, where it was written up and never tried
(moved here 2026-09-30, the maintainer's call).

The standing rule from that work is Juha's, 2026-08-28: **do algorithmically what we can, and invoke the
judge only as needed.** Its corollary for priming splits in two:

- **Priming with a domain convention does not generalize.** Telling a model about one convention fixes
  that case and buys nothing for the next one nobody anticipated.
- **Priming with what the deterministic layer already computed does**, since it is in hand and costs
  nothing to include. `raven-deduplicate`'s judge is shown the raw fields but not the title similarity, the
  author disagreement or the year gap its own rules computed, nor which question raised the pair — those
  are used only to veto its answer afterwards.

For the filter that means: whatever the sift and the derived constants have measured about a record — how
informative its title is, whether its abstract is a teaser — goes into the judge's prompt, not only into a
check on its verdict. The dedup corpus offered 14 pairs, too few to tell whether it helps; the filter's
corpus is large enough to measure it. A related precedent that did work is priming with a *stance*:
`investigations/agent-batch-classification/`.

## v1 takes a scope question, not research questions

Juha, 2026-09-29: *"scope questions are good enough for v1. RQs are for later."*

A **scope question** is what the search actually asked — for the prototype, *studies on different aspects
of the use of AI agents in higher education*. A research question is narrower, and the 2026-09-01
decision in `done/researchers-night/done/aokk-corpus-scope-classification-brief.md` already put those in a
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
   - **Decided (Juha, 2026-09-30): reports are named to be read as a sequence, numbered by stage.** A
     user should be able to list the output directory and see the pipeline in order — which file each
     stage wrote, and which one is current. The prototype's directory is the case against: twelve TSVs
     with nothing in their names saying which stage wrote them or in what order, and
     `dropped-before-escalating-titles.tsv` sitting beside `dropped.tsv` with nothing to mark it
     superseded. That was fine while prototyping, and it is what a user of the tool must not meet.
   - **A superseded report is either removed or visibly marked** — in its name, or by moving it aside.
     Which of the two is open; what is not open is leaving it looking like a peer of the current one.
   - **Reports can optionally be written as a spreadsheet, `.ods` or `.xlsx`** (Juha, 2026-09-30), so
     that opening one does not depend on the import dialog's settings. `.ods` costs no new dependency:
     `odfpy` is already one, backing `docextract`'s `.odt`/`.odp`. `.xlsx` needs `openpyxl`, to be
     installed when this is built.
   - **The TSV default is already safe**: `raven.papers.utils.write_tsv` (2026-09-30) quotes a cell
     containing a `"`, and the dedup and sift audits use it. Use it here too. Unquoted, the dedup audit
     lost rows in LibreOffice when space was ticked as a separator alongside tab — the import dialog's
     setting, which has to be unticked by hand; quoted, it kept every row either way (checked headless).
6. **Replaying existing decisions** (Juha, 2026-09-30). The prototype's judgements should be loadable
   into the new tools, so that the finished toolset can be run over the prototype's corpus and produce
   clean, stage-numbered reports without paying for the LLM passes again. The motive is the methodology
   section, which needs exact numbers from a pipeline that can be named.
   - **A run whose answers are all saved connects to nothing** (maintainer, 2026-10-07). The backend is
     contacted at the first question actually asked, never up front, so a replay works with no LLM
     running. `raven-deduplicate --judge` is the pattern: `_apply_judge` hands the passes a memoized
     connect-on-first-use function, which `_ask_judge` resolves, and a test with a control pins it. The
     prototype's `judge_scope.py` and `extract_fields.py` connect up front, so their replays need a
     backend; left as they are, being prototypes this brief replaces. `regenerate_reports.py` sidesteps
     it by replaying through `write_outputs`.
   - **Replayed answers carry the prototype's instrument, not the new tool's.** A report built from them
     must say which instrument made each decision; stamping them with the new tool's fingerprint would
     present the prototype's judge as the new one. So the import names the source instrument, and the
     ledger key (id plus fingerprint, per the per-document brief) keeps the two apart for free.
   - **The prototype stamped only the extractor.** `judged.jsonl` carries no fingerprint, so the import
     assigns one — a hash of `judge_scope.py`'s prompts at the commit that produced the file is the honest
     choice, since that is what decided the answers.
   - **The field names differ**, and the mapping from the prototype's JSON to the new schema is the real
     work: the three named tests are `no_ai`, `not_education` and `wrong_level` there, and become a
     user-supplied list here. Cheap if the new schema keeps a per-test boolean-or-unknown plus a reason;
     costly if it diverges, so the schema should be chosen with this import in view.
   - **The same holds for the dedup stage's LLM judge**, whose answers sit in `dedup_judge.jsonl` beside
     the corpus. The rest of the dedup is deterministic and simply re-runs: checked 2026-09-30,
     today's `raven-deduplicate --judge`, reusing a copy of that file, reproduces the 2026-08-31 run group
     for group — 1296 clusters, 1767 records removed, and every row's recorded differences unchanged. (A run
     without `--judge` merges fewer, which is the difference to watch for, not drift.) The output `.bib`
     differs only in how `month` is written, `{apr}` then and the bare macro `apr` now: the `bibtexparser`
     upgrade from 2.0.0b9 to 2.0.1 (confirmed 2026-09-30 by running today's code under 2.0.0b9, which
     reproduces the old output byte for byte). So the library version belongs in the methods too.
7. **A `sample` option on every model-driven tool** (Juha, 2026-09-30): run a stage over a seeded random
   sample of its input instead of all of it. Three uses, and the first is why it exists:
   - **Test-retest reliability.** Run the final judge twice over the same sample — a few hundred records —
     and report how often the two runs agree. That is a reliability figure for an LLM judge, which is one
     of the methodology questions the project is asking. The two runs must draw the *same* records, so
     the sample is seeded, and both the seed and the sample size go into the report header (Juha,
     2026-09-30) — the same seed with a different size draws a different set. The second run must also re-ask
     rather than read the first run's ledger, so the replicate needs a ledger of its own, or a
     replicate number in the instrument stamp; which of the two is open.
   - **The agreed sanity checks** (`investigations/aokk-corpus-scope/README.md`, *"Open: is the off-topic
     rate too low to believe?"*) sample each stage's output for hand-checking.
   - **Calibration**, which is the prototype's `--pilot N --seed S` under a general name.

## The final study's numbers

**Recommended, not yet decided: re-run the judge with the finished tools** for the published numbers,
rather than replaying the prototype's answers. A day or two of GPU time, unattended. The prototype's
instrument changed while it ran — the escalation rule widened, and pass 2's prompt was fixed partway
(`judged-before-prompt-fix.jsonl` holds the answers from before) — so its answers describe a
development history rather than one procedure a methods section can state and a reviewer can repeat.
The replay (question 6) stays useful for building and checking the reports cheaply. The sanity checks
then run on the final pipeline's output, not on the prototype's.

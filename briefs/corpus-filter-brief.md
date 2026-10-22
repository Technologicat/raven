# Brief: the corpus filter

**Filed 2026-09-29. Unnumbered.** Generalizes `investigations/aokk-corpus-scope/` into a `raven.papers`
tool: given a `.bib` from a literature search and the question the search was meant to answer, flag the
records that are not about it, with a reason for each, so a person can review them out.

**Gate, now passed: the per-document pass lands first** (Juha, 2026-09-29). It landed on 2026-10-21 as
`raven.librarian.sweep`; the brief is `briefs/done/per-document-llm-pass-brief.md`. Of the prototype's three
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
   **Decided (maintainer, 2026-10-22):**
   - **`run --pilot N`, default 50**, runs the model stages — the judge, the reviewer and the extractor —
     over a seeded sample of N records: question 7's `sample` machinery. Its answers go into the same
     ledgers the full run uses, so the full run asks only the rest; after an edit to the definition the
     instrument changes and the pilot is simply asked again. Its reports go to a `pilot/` subdirectory, so
     neither set removes the other.
   - **One pilot report, a spreadsheet with a section column, pre-sorted in reading order**, which is
     ordered by what a mistake costs:
     1. **The drops** (true at high confidence), each with the test that fired and its reason. A false drop
        loses a record silently, so these are read first, all of them.
     2. **The records sent to review** (true at medium or low). Mostly obvious drops means the tests are
        too timid; mostly obvious keeps, too eager.
     3. **A sample of the keeps with their extracted fields**, the rules' removals and holds marked.
        Anything here that plainly should have gone is a test that is missing.
     4. **The title-only records**, and how each stage did on them: the inputs most likely to break a
        rubric. **The set may be empty**, where a sift ran before the pilot, and the section then says so
        rather than vanishing.

     The constants derived from the corpus (`MIN_INFORMATIVE_WORDS`, `TEASER_CHARS`) head the report as
     context, the counts go to the console as well, and the run says how many title-only records the
     sample happened to include.
4. **One console script or several.** `raven-siftbib` is already its own tool, and the README's case for
   that (no LLM, same shape as `raven-fixbib` and `raven-deduplicate`) holds. **Decided (maintainer,
   2026-10-21): one console script, with the stages as subcommands** — `judge`, `review`, `extract`,
   `report`, and `run`, which does every automatable stage in order, so a user need not write a script to
   chain them.
   - **Calibration stays a human checkpoint**: `run --pilot N` and `run --until STAGE`. The ledgers make
     running again from where it stopped free.
   - **`run` switches models between stages by itself where the backend allows**, which LM Studio's REST API
     does (`/api/v1/models/load` and `/unload`, used on 2026-10-21 for the register investigation). Starting
     the run is the permission: the switch is logged and never waits for a confirmation. The documentation
     warns about it in so many words. The sweep's stop on a model swap still holds within a stage — the
     switch happens between stages, and each stage's ledger is named for its own model.
   - **Where each stage's model is named (decided, maintainer, 2026-10-22): not in the definition file**, which
     says what to ask of a corpus, where the model says what a machine has. On the command line,
     `--model STAGE=KEY`, repeatable; **or from a file, `--models FILE`**, naming a model per stage or one
     for all, so a rerun needs no typing, remembering or digging through ledgers for what each stage used.
     A data file rather than a `run.sh`, because it is platform-independent: a researcher on Linux can hand
     it to a colleague on Windows, who reproduces the run without translating a shell script. `--model`
     wins over the file, stage by stage. A stage named nowhere uses the loaded model, so with neither option
     `run` switches nothing.
     - **And `run` writes that file itself** (agreed, maintainer, 2026-10-22), naming the model each stage
       actually used, beside its reports. A finished run then carries its own recipe for repeating it. The
       information is already in each stage's ledger header.
   - **A backend that cannot switch (decided, maintainer, 2026-10-22): fail fast.** Before any work, `run`
     works out which stages still have something to ask — a stage whose answers are all saved needs no
     model at all — and if one of those wants a model other than the loaded one, it refuses at once,
     naming the model each stage wants. Softened: where the stages that can run come first, it says so and
     offers `--until` to run those. LM Studio is the only backend treated as able to switch; ooba has a
     loading API, untested here.
   - `raven-siftbib` stays separate, having no LLM; `run` can call the same library function as stage 0,
     after calibration, as the run plan orders it.
5. **Where the outputs go, and what they are called.** The prototype writes a filtered `.bib`, a
   dropped-with-reasons TSV, a held-for-review TSV, and JSONL answers — the extractor's named after
   its instrument fingerprint. How much of that the per-document primitive owns is decided by that brief, not this one.
   - **Decided (Juha, 2026-09-30): reports are named to be read as a sequence, numbered by stage.** A
     user should be able to list the output directory and see the pipeline in order — which file each
     stage wrote, and which one is current. The prototype's directory is the case against: twelve TSVs
     with nothing in their names saying which stage wrote them or in what order, and
     `dropped-before-escalating-titles.tsv` sitting beside `dropped.tsv` with nothing to mark it
     superseded. That was fine while prototyping, and it is what a user of the tool must not meet.
   - **A superseded report is removed (decided, maintainer, 2026-10-22)**, never left looking like a peer
     of the current one. Reports are derived data, regenerated from the ledgers, so deleting a stale one
     loses nothing.
     - **`report` owns its output directory**: on every write it removes the files matching its own naming
       pattern that it did not just write, and touches nothing else, so a user's own notes beside them are
       safe. Dated subfolders per run were rejected: a "current" pointer would need a symlink, which
       Windows handles poorly.
     - **A hand-marked copy, such as the team's marked sheet, must be saved under a name of its own**, or the
       next write removes it. The documentation says so in a sentence, and the tool says so when it writes.
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
   - **A run whose answers are all saved connects to nothing** (maintainer, 2026-10-07), so a replay works
     with no LLM running.
   - **Decided (maintainer, 2026-10-21): a replay names its instrument explicitly**, as
     `extract_fields.py --seed-from` does. `raven.librarian.sweep` names a ledger by a fingerprint that
     includes the *loaded model's* id, so finding a ledger by fingerprint would need a backend — exactly
     what the rule above forbids. Naming the fingerprint lets the to-do be read off the ledger first, and a
     backend is connected only if something is left to ask.
   - **And whether a run will ask anything is decided before its work starts** (maintainer, 2026-10-07).
     Compute what is still unanswered first: if anything is, connect and check the backend then, failing
     at once as before; if nothing is, never connect. Connecting lazily at the first question is not
     enough on its own, because in a long replay the first uncached question can arrive an hour in, when
     the user has gone for coffee and the backend has gone away — the failure the up-front check exists to
     prevent, moved to the worst moment. Where a later pass's questions depend on an earlier pass's
     answers, the earlier pass's to-do decides; a replayed earlier pass is cheap, so the later one's to-do
     can then be known within seconds.
   - **`raven-deduplicate --judge` connects on first use**: `_apply_judge` hands the passes a memoized
     connect function, which `_ask_judge` resolves, and a test with a control pins that a replay never
     connects. It meets the second rule by its structure rather than by design: each pass asks its first
     batch as soon as it starts, and all that precedes the second pass is the first pass and a
     re-clustering. The new tools should meet it by design.
   - The prototype's `judge_scope.py` and `extract_fields.py` connect up front, so their replays need a
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
     rather than read the first run's ledger. **Decided (maintainer, 2026-10-21): a replicate number in
     the instrument's material**, which gives a replicate its own ledger with nothing added to the sweep.
   - **The agreed sanity checks** (`investigations/aokk-corpus-scope/README.md`, *"Open: is the off-topic
     rate too low to believe?"*) sample each stage's output for hand-checking.
   - **Calibration**, which is the prototype's `--pilot N --seed S` under a general name.
8. **Whether confidence decides a drop. Decided (maintainer, 2026-10-21): a drop needs high confidence;
   any test answered true at medium or low confidence sends the record to review instead.** The record that
   showed the disagreement below goes to review under that rule. The question as it was raised (maintainer,
   2026-10-07: investigate and fix when generalizing):
   The prototype's `verdict_of` decides from the three tests alone: any test answered `True` drops, and
   confidence plays no part. `write_outputs`' comment says otherwise — that a low-confidence answer is an
   `unknown` verdict and so is kept. The two disagree, and one record shows it: `shen_directions_2024`, a
   high-confidence title drop whose abstract pass agreed at **low** confidence, and the only low-confidence
   drop of the 1084. Both halves are suspect until it is settled which behaviour is wanted; the README's
   *A drop and a keep do not deserve the same standard of proof* is where to start. The record was flagged
   for the team's hand check in the 2026-10-07 report set (`00-pipeline.xlsx`, its notes).

## Draft: the filter definition (questions 1 and 2)

**A draft to iterate on, not a decision** (maintainer, 2026-10-21: "I'd like to see the details so we can
iterate if needed before we build this"). One file holds everything corpus-specific — the scope question,
the judge's tests, the extractor's fields and the rules that act on them — and the tool supplies
everything that is not. Here is the AOKK prototype written in it, with the long descriptions cut short:

```toml
# aokk.filter.toml — what the corpus filter asks of this corpus, and what it does with the answers.

scope = "studies on different aspects of the use of AI agents in higher education"

# Domain cautions for the judge, beyond what each test says. Optional.
hints = """
"Learning assistant" names an AI tool, and is ALSO the term in STEM education research for a HUMAN
undergraduate who helps teach a course. "Teaching", "training" and "learning" are also how machine
learning describes what is done to a model; they count as education only when a PERSON is learning.
"""

# ---- The judge: one test per way of being off topic. True means positive evidence; silence keeps.
#
# For each record the judge answers every test true or false, and gives one confidence for the whole
# record - high, medium or low - with a sentence saying why. A test's description may say what confidence
# to give, as `not_education` does below. Only a true at high confidence drops a record; a true at medium
# or low sends it to review.

[[test]]
name = "no_ai"
asks = """true ONLY if the work is positively about something with no artificial intelligence in it -
human teaching staff, human tutors, or a non-AI technology. False when AI is present in any way, AND
false when the work does not say what it used."""

[[test]]
name = "not_education"
asks = """answered in two steps. FIRST: can you name what this work is about? If not, answer false with
low confidence, and stop. SECOND: is that subject entirely outside education? ..."""

[[test]]
name = "wrong_level"
asks = """true ONLY if the work is positively set at a level other than higher education - ... False
when the work does not say what level it is set at."""

# ---- The extractor: what a kept record says. Closed vocabularies, so the rules can act on them.
#
# Every closed vocabulary also has `not_stated`, added by the tool rather than listed here: the answer
# when the record does not say. It is never quoted, there being nothing to quote.

[[field]]
name = "level"
asks = "the educational level the work is set at"
quote = true            # any value but the implicit `not_stated` must quote the words that settle it
[field.values]
higher_education = "a university, college, polytechnic or their students and faculty - degree study"
school = "primary or secondary schooling, K-12, pupils"
vocational = "formal vocational or trade education leading to a qualification"
professional_training = "practitioners already qualified, being coached or trained on the job"
informal = "learning outside any institution - self-directed learners, the general public"
not_applicable = "the work is not set in education at all"

[[field]]
name = "human_learning"
asks = "whether a PERSON is taught or is learning in this work, as against a model being trained"
[field.values]
yes = "a person is taught or learning"
no = "the only thing being taught or trained is software"

[[field]]
name = "ai_role"
asks = "what the AI does in this work, or 'none'"
max_words = 8           # no `values`: free text, kept for reading, never acted on by a rule

# ---- The rules over the fields. First match wins; a `when` is AND across fields, OR within a list.

[[rule]]
label = "school or vocational level, stated in the text"
when = { level = ["school", "vocational"] }
action = "remove"

[[rule]]
label = "not about education: no educational level, and no one learning"
when = { level = ["not_applicable"], human_learning = ["no"] }
action = "remove"

[[rule]]
label = "possibly not about education: no educational level, but someone may be learning"
when = { level = ["not_applicable"] }
action = "hold"

[[rule]]
label = "learning outside an institution, e.g. workplace training"
when = { level = ["professional_training", "informal"] }
action = "hold"
```

What the draft decides, each open to change:

- **TOML.** `tomllib` is in the standard library, it takes comments, and its multi-line strings hold
  prose. JSON has no comments and mangles prose; a Python module would execute a file that arrives with a
  corpus; Markdown is friendlier for prose but needs conventions of our own for the structure.
- **The tool supplies the frame, the file supplies the domain.** "Find positive evidence of being off
  topic; silence keeps; a generic title is `low`; use the venue; judge the work, not the phrase that
  matched" holds for any corpus, and is the tool's prompt. What is about *this* corpus is the file's. The
  prototype's rubric splits along that line without much argument; `hints` takes the domain cautions that
  belong to no single test.
- **The judge's answer is what the prototype's was**: per test `true`/`false`, one `confidence` of
  `high`/`medium`/`low`, a `why`, and which pass answered (title or abstract). So importing the prototype's
  answers is a rename at most, and none where the test names are kept, which question 6 asks for. A drop
  needs a `true` at `high` (question 8).
- **Every closed vocabulary gets `not_stated` from the tool**, so a user cannot forget it, and the quotation
  rule is per field (`quote = true`). Both are lessons from the prototype.
- **No booleans in the extractor.** `human_learning` becomes a vocabulary, `yes`/`no`, plus the
  `not_stated` every vocabulary gets — the prototype's `true`/`false`/`"unclear"` mix is what lost a batch
  to an unquoted `unclear`.
- **Rules are data**, and the prototype's four tiers fit them unchanged. `hold` is the review list, and the
  label is what a report shows. A record matching no rule is kept.
- **The instrument is the parsed file**, normalized, so editing a comment or rewrapping prose does not
  start a fresh ledger, while changing a test, a field or a value does. Rules are applied when the
  reports are written rather than asked of the model, so changing one re-asks nothing — as with
  `raven-pdf2bib`'s conference options.

Settled on the draft (maintainer, 2026-10-21), besides agreeing the points above:

- **Where the file lives: `--definition FILE`**, defaulting to `<bib stem>.filter.toml` beside the `.bib`.
  Good for v1.
- **A `remove` resting on a quoted field whose quote is not in the text becomes a `hold`**, saying why. The
  prototype's tier A was trustworthy because every such call carried a quote that checked out.
- **`hints` stays**: it is the one way to give an instruction that holds across several tests.

- **The tool's frame is worded for any published work**, never for studies (maintainer, 2026-10-21). Its
  core — positive evidence of being off topic, silence keeps, a generic title is no evidence, judge the
  work rather than the phrase that matched — holds for any search result. The rest is bibliographic,
  since the input is a `.bib`: a title, an abstract, where it was published. A `.bib` can hold books,
  reports, standards or news as well as studies, so "study" and "method" stay out of it.
- **The file may replace the frame**, as `frame = """..."""`: the escape hatch for a corpus the default
  does not fit. It is part of the instrument either way.
- **No built-in frames per kind of corpus**: nothing asks for one yet.

**Where this is likely to go next** (maintainer, 2026-10-21): once the drydock merges the Visualizer's and
Librarian's databases (brief 13), items of other kinds may want the same treatment. Fulltexts fit the
bibliographic frame; news articles, patent databases, server logs and company-internal documents do not.
Then the input stops being a `.bib`, and what a record *is* — which fields it has, which of them the
judge is shown — becomes a parameter too. The `frame` key is the part of that which v1 already has.

## The final study's numbers

**Recommended, not yet decided: re-run the judge with the finished tools** for the published numbers,
rather than replaying the prototype's answers. A day or two of GPU time, unattended. The prototype's
instrument changed while it ran — the escalation rule widened, and pass 2's prompt was fixed partway
(`judged-before-prompt-fix.jsonl` holds the answers from before) — so its answers describe a
development history rather than one procedure a methods section can state and a reviewer can repeat.
The replay (question 6) stays useful for building and checking the reports cheaply. The sanity checks
then run on the final pipeline's output, not on the prototype's.

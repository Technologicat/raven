# Deduplicating a multi-database bibliography (`raven-deduplicate`)

**Status: built and shipped**, all on 2026-08-28. Named `raven-deduplicate` rather than the
`raven-dedupbib` this brief was written under — Juha's preference, and the module is
`raven/papers/deduplicate.py`.

| landed | what |
|---|---|
| `5fd6c4eb` | `raven-fixbib` repairs duplicate field keys, and every report names its fault |
| `63bedd51` | `raven.common.text.boilerplate`, wired into the Visualizer's importer |
| `04610fdd` | the stripper's two-tier rework, after it ate prose about copyright |
| `51e547b1` | this brief |
| `99583940` | `bibtex.write_string`, and a reader that can be written back |
| `51ae63af` | `&amp;` and `&nbsp;` decoded wherever Raven reads bibliography text |
| `76f46be4` | **`raven-deduplicate` itself**, with its tests |
| `4801b834` | `raven-fixbib` decodes the HTML a database leaves in the field values |

**The design below stood, and four things in it were corrected by contact with the corpus.** Each is
marked *Corrected* where it appears, and the short version is:

- The degenerate-title guard became a **pairwise** rule rather than a filter on the key, and the length
  threshold it was going to use was measured, found to catch nothing, and dropped.
- A **second false merge** turned up that the design had not anticipated: a serial's recurring section
  heading, which needed a guard of its own.
- The fuzzy pass offers **10 pairs, not ~370**. The design's figure was measured before the guards.
- The judge **must not be asked about Springer chapter versions**, where the design predicted it would
  merely agree with the rule. It does not agree; it is confidently wrong.

What it does on the corpus it was built for, as shipped: **6934 records read** — all of them, see
*Reading, and the last two records* below — **5166 unique, 1768 merged away from 1300 clusters**, in about
seven seconds. With `--judge`, 5 more merges and 2 refusals.

Sits outside the sprint folders because it is `raven.papers` work, like `ligature-repair-brief.md` is
`docextract` work: a bibliography tool whose consumers are whoever has a `.bib`, not a feature of any one
app.

## The problem

A literature search run across several databases — Scopus, Web of Science, ProQuest, Springer, arXiv — and
concatenated into one file. The same paper is exported by each database that indexes it, in each one's own
dialect, so the file holds every record several times over with different fields filled in.

The measurements below are from the file that motivated this, a 6934-record export. They are here to size
the problem, not because the tool is for that file.

## What had to be fixed first, and why it is not a footnote

**1598 of the 6934 records did not parse at all**, and nothing said so more specifically than
"unparseable". Every one was a ProQuest record carrying several `annote` fields — one for the copyright
statement, one for the last-updated date, one for the subject terms — which `bibtexparser` rejects as a
duplicate field key, discarding the entry whole.

This is worth stating plainly because it inverts the obvious order of work: **you cannot deduplicate what
you cannot read**, and a quarter of the corpus was invisible to every count anyone had made of it.

The Visualizer importer would have reported all 1598 as suspected unbalanced braces —
`bibtex.repair_record` rescued 0 of a 50-record sample, the braces being perfectly balanced. So the
existing diagnosis was not merely unhelpful, it pointed the wrong way.

Landed in `raven-fixbib` as `bibtex.repair_duplicate_field_keys`. 1596 of the 1598 repaired; the two
holdouts carry `author = {Surname, MSc, RN, Given}`, which BibTeX cannot express — it allows a name two
commas and that uses three — so they are reported by key and line instead of guessed at.

## Duplication, measured

All 6934 records, after the repair:

| pass | unique | removed |
|---|---|---|
| exact normalized DOI | 5307 | 1625 |
| exact normalized title | 5191 | 1741 |
| DOI + title, transitive | 5161 | 1771 |

Cluster sizes: 947 pairs, 260 triples, 67 quads, 17 fives, 7 sixes. 6417 records carry a DOI, 515 do not.

Three findings shaped the design:

- **DOI equality is evidence; DOI inequality is not.** 13 title clusters carry more than one distinct DOI,
  and none of them are false merges. One paper about GPT feedback carries an *Astronomy & Astrophysics*
  DOI. Another pair differs only by an en-dash where the other has a double hyphen. A third has a Zenodo
  DOI on one copy and a journal DOI on the other.
- **Keeping "the richest record" loses data in 622 clusters.** The ProQuest twins are field-rich
  (15–17 fields, long abstracts) but some are author-less; the Scopus and Springer twins have the author
  and no abstract. Only a field-level union keeps both.
- **Title-only merging needs a guard.** Four clusters key on a degenerate title; `Editorial` merges
  Bandyopadhyay 2022 with McNally 2024. Few, but silent.

**These figures predate the guards, and the guards are what the numbers above are missing.** They are kept
because they size the problem, which is what they were for. As shipped the same corpus gives 5166 unique
and 1768 removed — a little less merging, because three false merges stopped happening, and a little more
reading, because two more records now parse.

## Design

`raven/papers/dedupbib.py`, console script `raven-dedupbib`. A shipped tool rather than a study-local
script, because a scoping review's method section can cite a versioned tool and cannot cite somebody's
`/tmp`.

```
raven-deduplicate input.bib -o deduped.bib --audit audit.tsv [--judge]
```

**Reads through the repair.** `fixbib`'s own docstring sets the precedent — Raven's readers repair what
they read, without writing back — so this gets all 6934 records without the user running two tools in
sequence. `raven-fixbib` remains the way to fix the file itself.

**Normalization is for match keys only.** Nothing normalized is ever written to the output. Output field
values keep their original bytes: `Ä` stays `Ä`, LaTeX braces stay as the source had them. The merge only
ever *adds* a field from a twin that had one.

- DOI: lowercase, strip `https://doi.org/` and `doi:`, fold the seven Unicode dashes to `-`.
- Title: NFKD, drop combining marks, reduce to `[a-z0-9]`.

**Deterministic clustering**, union-find over two keys, no model involved: exact normalized DOI, then
exact normalized title subject to the degenerate-title guard.

> *Corrected.* **The guard is applied per pair, not to the key.** Withholding a degenerate title as a key
> altogether also refuses the merges that are right — the two Bandyopadhyay copies really are one
> editorial — so what ships weighs each *pair* of records sharing a title, and a group of three comes
> apart into the two that are one paper and the one that is not. `_title_edge_holds` is the rule, and it
> is three cases: a generic title needs the records to positively agree about author and year; two
> authorless records are refused if their DOIs disagree (below); anything else is refused only if the
> records contradict each other on both author *and* year.
>
> **The length threshold in the design was measured and dropped.** Over the corpus it caught nothing the
> curated genre-label list did not already catch, while rejecting `Reportronic` — a real and thoroughly
> distinctive title — for being eleven characters. Distinctiveness is not length, and the cautionary
> example is `Generative AI`: longer than the title the rule rejected, and far likelier to head two
> unrelated editorials.

> *Corrected — a second false merge the design did not anticipate.* **A serial's recurring section
> heading.** `II Political Science: Method and Theory` and `Abstracts Abstracts` head an item in every
> issue of their journals: authorless, same year, different DOIs, and an ordinary title by every other
> test. The title alone merged four issues into one record. Where *neither* record names an author the
> title is the only evidence there is, so a DOI disagreement now overrules it. Of the 36 authorless
> merges in the corpus, the three where the DOIs disagreed were the three that were wrong and the 33
> where they agreed were all right — which is what makes this a rule rather than a list of titles.

**`--judge` is opt-in**, because it needs an LLM backend. It sees only what the deterministic pass could
not settle: roughly 370 fuzzy residuals, blocked by (first-author surname, year ± 1) and filtered by a
cheap string similarity, plus the 13 DOI-conflict clusters.

> *Corrected.* **10 fuzzy pairs, not ~370**, and 8 DOI-disagreement pairs after the exclusion below. The
> design's figure was measured before the guards and before the title normalization learned to resolve
> HTML entities, so most of those 370 are now merged deterministically. The run is two batches and about
> a minute, which changes what the flag is for: not a bulk pass to be endured but a cheap last sweep.
>
> It is worth saying what the blocking *cannot* see, since the number came out so much smaller than
> expected: a record with no author or no year is in no block at all, and two databases spelling the
> first author differently put one paper in two blocks. Both are missed merges, which is the direction to
> fail in — the deterministic keys have already found everything that agrees exactly.

`agent.turn(llm_settings, prompt, use_character_card=False, tools_enabled=False)` is the idiom, as
`raven/papers/pdf2bib.py` already uses it.

**Follow `investigations/agent-batch-classification/` closely.** Its finding transfers almost verbatim:
index-keyed batches, answers whose index does not resolve are dropped rather than trusted, resumable
JSONL, and — the load-bearing one — **escalation conditions computed in Python, never the model's own
confidence.** Two near-identical titles differing only in a subtitle are exactly the input where a model
invents a distinction and rates itself sure of it.

**Merge is base + fill.** The record with the most fields is the base; fields it lacks are filled from its
twins. Every *differing* value that lost is written to the audit, so nothing disappears without a trace.

### Abstracts: strip the publisher's notice before comparing, then take the longest

> *Corrected — "before comparing" is the whole of it.* An early cut wrote the *stripped* text into the
> output, which left the file carrying two kinds of abstract: trimmed where a record happened to have a
> twin, untouched where it did not. It also made the tool a content editor, which is the thing Juha had
> just ruled out for `&amp;` — cleaning up what a database wrote is `raven-fixbib`'s job. So stripping
> decides *which* abstract and never edits the one that wins, and every value the tool writes is one of
> the copies it read.
>
> The audit still compares abstracts stripped, so 162 pairs on the corpus that differ only by whose
> copyright line is attached are not reported as differences. That is the single exemption in "nothing
> disappears without a trace", it is stated in `AuditRow`, and a check over the corpus confirms nothing
> else falls through: 0 values neither kept nor recorded for any other reason.

"Longest wins" on its own is wrong here, and wrong in the common case rather than a corner. Of the 601
clusters holding two or more different abstracts, **592 differ only by an appended rights notice** —
`© The Author(s), under exclusive license to Springer Nature Singapore Pte Ltd. 2025.` — so the rule would
have grafted boilerplate onto the majority of merged records.

Stripping first collapses **537 of those 601 to exact agreement**, leaving no choice to make. What remains:

| after stripping | clusters | what it is |
|---|---|---|
| agree exactly | 1006 | nothing to decide |
| one contains the other | 69 | database truncation — 254 chars against 1494. Longest is right. |
| genuinely different | 76 | 56 differ by under 50 characters, 11 by over 300 |

The last row takes the longest too, and the audit records the value that lost — the 56 small ones are
whitespace and encoding variants where either choice is the same abstract, and the 11 large ones are worth
a human glance in the audit rather than a rule. Nothing here goes to the judge.

So the deterministic rule covers it and the judge is not involved. Recorded because it was considered:
*rejected — send disagreeing abstracts to the judge.* It would have been ~600 model calls to reproduce
what a regex settles, and on the 76 residuals the question ("which of two near-identical abstracts is
better?") has no answer a model is better placed to give than a length comparison.

**The stripper must not be loose, and the check for that is a measurement.** A first version matching a
bare `copyright` cut 284 characters of real content from an abstract whose closing sentences discuss
*copyright concerns* in AI-generated work — and separately mangled 5783 abstracts by a character each,
having stripped every trailing full stop. Both were invisible in the collapse statistics, which looked
excellent throughout. What surfaced them was listing the largest cuts and reading them.

**The hard case is not an abstract with a notice, it is an abstract about copyright** — and a corpus on AI
in education has those. Raised by Juha, 2026-08-28, with the sentence *"Copyright remains a widely debated
field of law, and further research into the topic is encouraged."* That one survived; eight of nine
sibling probes did not. `Creative Commons Attribution licences are increasingly common in open education`
lost 81 characters, `We show the phrase All rights reserved has no legal effect` lost 64.

The rule that survives has two tiers, and the split is the design:

- **Unmistakable, trusted anywhere in the tail window**: the copyright sign and `(c) YYYY`, in the
  spellings that survive markup conversion. Not English — nobody writes them mid-argument. Measured over
  the corpus, this tier alone accounts for 1587 of 1658 detections.
- **Ordinary English, trusted only where it opens a sentence**: `copyright` qualified by a year or
  `held by`, `all rights reserved`, a licence-grant clause, `licensee <Name>`. Appended boilerplate starts
  its own sentence; a clause inside an argument does not. That is what separates
  `Copyright 2024, Society of Petroleum Engineers.` from `The Copyright 1976 settlement still governs
  derivative works.`

`creative commons attribution` is in **neither** tier. It is a proper noun that opens sentences in prose,
so the sentence test cannot save it, and it was the sole evidence for 2 detections out of 1658. Widening
the licence-grant clause reaches both of those instead.

Both were found by *adversarial probes rather than by the corpus*, which is the transferable part: the
corpus contains the notices, so measuring against it says how many are caught and nothing about what else
would be. The probes are now `TestPapersAboutCopyright` in the test module, ten prose endings that must
survive whole.

Two further corrections came out of chasing them, and both were real notices being missed: a bracketed
aside between the full stop and the notice (`... domain. (CC BY-NC 4.0) This article is licensed to you
under ...`) defeated the sentence test, and a full Creative Commons grant with its URL and permissions
sentence runs past 400 characters, so the window was cutting off the longest notices — which are the ones
carrying the most junk. The budget is 600, and the aside is now taken with the notice rather than left
dangling.

A third failure, caught by the unit tests rather than by the corpus: trimming trailing punctuation after a
cut also removed the abstract's own closing full stop. What dangles at a cut is the *separator* that joined
the notice on — a comma, a dash — and a full stop is not one. The corpus statistics could not show this
either; only an assertion that an abstract ends the way it was written.

**It lives in `raven.common.text.boilerplate`**, not in the deduplicator, and the Visualizer's importer runs
every abstract through it at import time (decided by Juha, 2026-08-28). See below for why that was more than
a tidiness preference.

**Audit TSV**, stamped with `raven.__version__`. One row per cluster: surviving key, merged-away keys,
which rule fired, the model's reason where one was consulted. A scoping review has to report duplicates
removed, and this is what that number is computed from.

## Decisions, and what was rejected

Decided by Juha, 2026-08-28. Both produced no diff at the time, which is why they are written down here.

**A preprint and its published version are one record, and the published version is preferred.** They are
one study, counted once, cited by its version of record. The merge keeps the journal DOI and venue and
fills gaps from the preprint, which often carries the longer abstract. The audit row names both.

- *Rejected: keep both, flag the pair.* Safest, and correct for a protocol that counts them separately —
  but it leaves ~200 pairs of manual work and this protocol does not count them separately.
- *Rejected: prefer whichever record is richer.* Simpler, and usually picks the ProQuest or Scopus record,
  but it can leave a Zenodo DOI on a paper that has a journal one, which is wrong in a bibliography.
  Publication status decides the base, not field count.

**Springer living-reference-work chapter versions (`..._12-1`, `..._12-2`) are one work; keep the highest
version.** Consistent with `raven.papers.utils.deduplicate_arxiv_ids`, which already resolves arXiv
versions this way, so the toolkit behaves the same way about versions wherever they turn up.

- *Rejected: send them to the judge first.* Considered because the same chapter title appearing under two
  different *book* DOIs is odd. Cheap — 13 clusters — but the version suffix is a documented Springer
  convention, so a rule reads it correctly and a model would only agree.

  > *Corrected, and the correction matters more than the decision did.* **It does not only agree — it
  > actively disagrees, fluently.** Before `settled_by_rule` existed, the judge was shown four version
  > pairs and refused all four, reasoning *"different DOI suffixes indicate separate chapters"* and
  > *"different DOIs and ISBNs indicate separate chapters in different volumes."* Both sentences are
  > true about the strings and wrong about the works, and acting on them would have split four papers
  > the project had decided are one. Measured 2026-08-28 against Qwen3.6-35B.
  >
  > So the pairs are now withheld rather than asked about, and the general form is worth carrying
  > forward: **a rule that encodes a convention is not a rule a model can rediscover from the data.** A
  > documented suffix is invisible to a reader seeing the string cold, and what fills the gap is a
  > confident inference from surface form. Do not ask a question a rule already answers — not because
  > the answer is redundant, but because it may come back different.
- *Rejected: treat versioned chapters as distinct works.* Conservative, and it leaves visible
  near-duplicates that a reviewer will notice.

### Why the stripper is shared rather than the deduplicator's own

The copyright line that ends an abstract was already a known nuisance elsewhere in Raven, patched
downstream instead of removed. `nlptools.default_stopwords` documents that "Elsevier" lemmatizes to
"elsevi" *because* it sits in that line, where spaCy's tagger reads it as an adjective and strips what looks
like a comparative `-er`; and the Visualizer carries a `publisher_stopwords` list to keep publisher names out
of the word cloud, with a docstring saying it needs re-checking after every spaCy model bump.

That is a workaround for text that should not have reached the NLP stage. Measured over the 6934-record
export, running the importer's own sequence: occurrences of `©` reaching spaCy fall from 1650 to 1,
*All rights reserved* from 241 to 0, *Springer* from 337 to 11, *Elsevier* from 59 to 2. *MDPI* stays at 4,
which is the reassuring number — those are genuine mentions inside an abstract's body, and nothing touched
them.

It runs last in the importer's abstract pipeline, after `unicodize_basic_markup`, so the notice is seen in
the form everything downstream sees. Checked rather than assumed: markup conversion does *not* turn
`\copyright` or `&copy;` into `©`, so the stripper matches those spellings itself.

## Reading, and the last two records

The design says the tool reads through `fixbib`'s repair, and it does. Two things came out of building
that which the design did not foresee, and both closed the "two unrepairable records" item:

**Raven's shared reader chain cannot be written back.** `bibtex.parse_string` splits `author` into a list
of `NameParts`, and handing such a library to `bibtexparser.write_string` renders that list with `repr()`
— so the file comes back out carrying `author = {[NameParts(first=['Jane'], ...)]}`. Still valid BibTeX,
with every author field destroyed, and nothing logged. Nothing in tree had done this; the deduplicator is
the first tool that reads a bibliography and writes one back, and it would have. So the reader takes
`split_names`, and `write_string` undoes the split when the library carries it — deciding from the data
rather than from an argument, because merging an unsplit library raises and stops while *not* merging a
split one writes the mangled file and exits 0.

**The repair's oracle was asking the wrong question**, and that is why two records looked unrepairable.
Both repairs judge a candidate by parsing it, which is right; they were parsing it with the full chain,
which asks "is this record now free of *all* faults" where the question is "did my edit produce an entry".
The two `gosak_picot_2025` records name `annote` three times *and* carry `Gosak, MSc, RN, Lucija` — three
commas where BibTeX allows two. The merge was perfect and was thrown away for the name.

Worse, the report named the wrong fault: `duplicate field keys (repeats annote)`, which is a fault the
tool repairs, so the message read as a contradiction and sent the reader to look at fields that were fine.
Now the oracle is structural, the merge is kept, and `raven-fixbib` re-checks the repaired record against
the full chain to report what is actually left: *Cannot split the following name `Gosak, MSc, RN, Lucija`
into parts: Too many commas.*

A third fault surfaced while fixing that: `_diagnose` consulted a line-by-line brace heuristic before the
parser's own words, and the merge it had just performed — which joins values with newlines — makes any
multi-line value look unbalanced line by line. The heuristic's own docstring says it is a shortlist of
suspects rather than a verdict; it is now asked last, after a middleware error, which by construction
means the syntax was fine.

**Result: the deduplicator reads 6934 of 6934.** `raven-fixbib` still reports the two, correctly — Raven's
standard chain still cannot read them, and the name needs a human to decide which commas separate name
parts and which separate credentials. But nothing is lost to it any more.

## Open

- **Whether the Visualizer importer should learn the same repair.** It still loses these records with a
  misleading reason. Deliberately left out to keep the blast radius small; raised, not filed.

- **Whether `publisher_stopwords` can now be trimmed.** It exists to keep publisher names out of the word
  cloud, and with the notice removed at import there is much less for it to catch. Stale entries are
  harmless — `nlptools.default_stopwords` says so — so this is tidying rather than a fix, and it wants doing
  after a real import has been eyeballed, not before.

## What was learned, and what to carry to the next thing like this

**The habit that earned its place, three times over:** *a corpus tells you what a rule catches and nothing
about what else it catches.* It was written down here after the boilerplate stripper ate prose about
copyright and trimmed every abstract's final full stop, both invisible in statistics that looked excellent.
It then caught two more, and neither would have been found by looking at a cluster count:

- The `Editorial` merge, which the design had anticipated.
- The serial's section heading, which it had not — found by listing every authorless merge and reading
  them, having noticed the `noauthor_` keys in an unrelated printout.

So the check that works is not a number. It is **listing what a rule did and reading it**, grouped by
whatever the rule keys on. Both defects were three rows in a list of thirty-six.

**The corresponding test habit:** every fixture written to pin a guard was run against the code *without*
that guard, and required to fail. Two of them needed the stronger version — the serial fixture passes with
the generic-title guard alone, so it had to be checked against *that* too, or it would have been pinning
the wrong clause. A test written after a fix asserts what the fixed code does, which is not the same as
asserting what was wrong.

**And the one that is specific to having a model in the loop:** a rule that encodes a convention is not a
rule a model can rediscover from the data. See the Springer correction above. The judge is a proposer here
and never a decider, which is what made that survivable — a verdict contradicted by the records themselves
is dropped in Python, so the worst a confident wrong answer costs is a missed merge.

## Where this leaves the corpus

`00_stuff/rawdata/AOKK/multisource/tekoalyagentti_tutkimus.bib`, 6934 records. Not committed and not to be
— it is a search export, and this repository is public.

```
raven-fixbib      tekoalyagentti_tutkimus.bib          # 1596 records repaired, 142 entities decoded
raven-deduplicate tekoalyagentti_tutkimus_fixed.bib \
                  -o deduped.bib -a audit.tsv [--judge]
```

6934 read, **5162 unique and 1772 merged away** with the judge, 5166 and 1768 without it. Two records still
need a human to fix a name; they are read and deduplicated regardless.

The audit is the artifact the review's method section is built from, and it carries `raven.__version__`.

### A second session runs alongside this one

Two agents on this tree: this work, and `researchers-night/` item 11, avatar and TTS integration. **Do not
assume the commits this brief names are anywhere near `HEAD`** — fetch and look. On 2026-08-28 the other
session landed a dozen commits between one push here and the next.

The subsystems really are independent, and that is what makes the pairing safe: `raven/papers/`,
`raven/common/text/` and the Visualizer importer on this side; `raven/avatar/`, `raven/common/audio/`, the
TTS layers and Librarian's avatar integration on the other. Nothing in the design above reaches into any of
those.

**But subsystem independence is not what actually collided.** What did was a *tree-wide sweep* — the other
session put every `__all__` in file order and made never-API names private, which touched
`raven/papers/bibtex.py` and renamed `importer.parse_input_files` to `_parse_input_files` underneath work in
progress here. It was harmless, because it was committed and pushed promptly and picked up by a fetch. The
lesson is that a sweep crosses every boundary by definition, so the protection is *cadence*, not scope:

- **Push at each seam** rather than at the end, so the other session's next fetch has your work in it.
- **Fetch before asserting anything** about the tree, and treat a "changed on disk" notice as real.
- **Stage by name.** Never `git add -A`, `-u`, `.` or `raven/` — that would have swept up the other
  session's in-flight edits to `recorder.py` and `audio_input_panel.py` along with the three `config.py`
  overrides this repo always carries.
- **`CHANGELOG.md` is the one file both sessions certainly touch.** Read it immediately before editing, and
  check `git diff --stat` afterwards for hunks that are not yours. The component headers keep the entries
  apart — *Raven-visualizer*, *Raven-fixbib* and *Raven-deduplicate* here, *Raven-avatar* and
  *Raven-librarian* there — so the conflict risk is low and the review risk of not looking is not.
- **`scripts/check_exports.py`** fails a public name missing from `__all__`, so anything added here is
  checked against the convention the other session established. Run it before pushing.

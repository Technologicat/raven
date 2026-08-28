# Deduplicating a multi-database bibliography (`raven-dedupbib`)

**Status: stage 0 built, the rest designed.** `raven-fixbib` learned the duplicate-field-key repair on
2026-08-28 (`5fd6c4eb`). `raven/papers/dedupbib.py` is designed below and not written.

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

## Design

`raven/papers/dedupbib.py`, console script `raven-dedupbib`. A shipped tool rather than a study-local
script, because a scoping review's method section can cite a versioned tool and cannot cite somebody's
`/tmp`.

```
raven-dedupbib input.bib -o deduped.bib --audit audit.tsv [--judge]
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

**`--judge` is opt-in**, because it needs an LLM backend. It sees only what the deterministic pass could
not settle: roughly 370 fuzzy residuals, blocked by (first-author surname, year ± 1) and filtered by a
cheap string similarity, plus the 13 DOI-conflict clusters.

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

So the deterministic rule covers it and the judge is not involved. Recorded because it was considered:
*rejected — send disagreeing abstracts to the judge.* It would have been ~600 model calls to reproduce
what a regex settles, and on the 76 residuals the question ("which of two near-identical abstracts is
better?") has no answer a model is better placed to give than a length comparison.

**The stripper must not be loose, and the check for that is a measurement.** A first version matching a
bare `copyright` cut 284 characters of real content from an abstract whose closing sentences discuss
*copyright concerns* in AI-generated work — and separately mangled 5783 abstracts by a character each,
having stripped every trailing full stop. Both were invisible in the collapse statistics, which looked
excellent throughout. What surfaced them was listing the largest cuts and reading them.

The rule that survives: a notice must *look like a notice* — `©`, `(c) YYYY`, `copyright` followed by a
year or symbol or `held by`, `all rights reserved`, `this work is published/licensed/distributed under`,
`licensee <Name>`, `creative commons attribution` — and it counts only in the last 400 characters, since a
rights notice sits at the end and a discussion of rights may be anywhere.

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

## Open

- **The two unrepairable records** (`Surname, MSc, RN, Given`) need a human edit deciding which commas
  separate name parts and which separate credentials. They are themselves a duplicate pair, so this is one
  record's worth of work.
- **Whether the Visualizer importer should learn the same repair.** It currently loses these records with a
  misleading reason. Deliberately left out of the `fixbib` commit to keep the blast radius small; raised,
  not filed.

- **Whether `publisher_stopwords` can now be trimmed.** It exists to keep publisher names out of the word
  cloud, and with the notice removed at import there is much less for it to catch. Stale entries are
  harmless — `nlptools.default_stopwords` says so — so this is tidying rather than a fix, and it wants doing
  after a real import has been eyeballed, not before.

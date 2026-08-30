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
| `e8020aa7` | `raven-fixbib` moves a rights notice into a `copyright` field; `deduplicate` unions it |

Reviewed 2026-08-29, file by file. Two correctness fixes came out of it — `_first_surname` disagreeing
with itself across the two BibTeX name orders (`b4860466`), and the generic-title guard admitting merges
that author-and-year agreement cannot settle (`15d56208`) — along with `raven/papers/config.py` and the
knobs moved into it (`605461f0`, `532ec33a`).

**The design below stood, and five things in it were corrected by contact with the corpus.** Each is
marked *Corrected* where it appears, and the short version is:

- The degenerate-title guard became a **pairwise** rule rather than a filter on the key, and the length
  threshold it was going to use was measured, found to catch nothing, and dropped.
- A **second false merge** turned up that the design had not anticipated: a serial's recurring section
  heading, which needed a guard of its own.
- A **third and fourth**, both found in review rather than by counting: the same person writes several
  book reviews in a year and every one is titled `Book Review`, and multi-volume conference proceedings
  share a title across genuinely different books. Author and year agreement settles neither; what does is
  refusing a pair that contradicts itself on DOI, pages, volume or issue.
- The fuzzy pass offers **10 pairs, not ~370**. The design's figure was measured before the guards.
- The judge **must not be asked about Springer chapter versions**, where the design predicted it would
  merely agree with the rule. It does not agree; it is confidently wrong.

What it does on the corpus it was built for, as shipped: **6934 records read** — all of them, see
*Reading, and the last two records* below — **5171 unique, 1763 merged away from 1295 clusters**, in about
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
because they size the problem, which is what they were for. As shipped the same corpus gives 5171 unique
and 1763 removed — less merging, because eight false merges stopped happening (three serial section
headings, five pairs of multi-volume conference proceedings), and more reading, because two more records
now parse.

## Design

`raven/papers/deduplicate.py`, console script `raven-deduplicate`. A shipped tool rather than a
study-local script, because a scoping review's method section can cite a versioned tool and cannot cite
somebody's `/tmp`.

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
> is three cases: a generic title needs the records to positively agree about author and year *and* not to
> contradict each other on `config.identifying_fields`; two authorless records are refused on that same
> contradiction (below); anything else is refused only if the records disagree on both author *and* year.
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

Kept here rather than in `TODO_DEFERRED.md` (Juha, 2026-08-28): these three are one another's context and
belong with the work that raised them, where a flat backlog would charge attention rent from everyone who
scans it and hand the eventual reader three entries that only make sense together.

### The entity decoder in `unicodize_basic_markup` has far more gaps than the ampersand

`&amp;` and `&nbsp;` were fixed today because they were what the corpus contained. They were not the only
gaps, and the list is not close to complete: measured 2026-08-28, `common.utils.unicodize_basic_markup`
resolves **13 of the 2125 named HTML5 entities and none of the numeric ones**. `&#8217;` — a right single
quote, which database exports use constantly — passes straight through. The thirteen are a hand-maintained
run of `str.replace` calls that grew one entity at a time as someone hit one.

*Cost: S. Gate: none.* Juha's framing is the 4D principle — **dinky, dirty, dynamic data** — from a visiting
lecturer at JYU: real inputs are small, messy and always moving, so a reader has to cope with the whole
messy space rather than the part someone happened to meet.

**On where the table should live**, since the worry is that one would bloat the module: **there is no table
to ship.** `html.entities.html5` is in the standard library, all 2125 of them, and
`bibtex.decode_html_entities` already uses it — that function costs a `import html.entities` and nothing
else. So a JSON file or a data module would be carrying weight Python already carries.

**A shared module is still worth it, for the rules rather than the data.** Decoding correctly is not a
lookup: `&amp;` must resolve *after* every other entity or `&amp;lt;` silently becomes `<`; an entity naming
an invisible character must not decode to one; a separator or control character must not change the line
count. Those live in `bibtex.py` today and would have to be written a second time in `utils.py`. One
`raven.common.text.entities` holding the decode-one-entity rule, with `bibtex` adding its BibTeX escaping
on top and `unicodize_basic_markup` taking the plain-text form, is the shape — and it puts the entity
handling beside `boilerplate.py`, which is the same kind of thing.

Watch the ordering `unicodize_basic_markup` already depends on: it unescapes `\&` near the top, and resolves
`&lt;`/`&gt;` *after* the `<sub>`/`<b>` passes so that markup written as entities is not turned into
markup. A table-driven pass has to keep both.

### `bibtexparser`'s writer destroys a split-name library, and upstream would take a patch

`bibtexparser.write_string` renders an unrecognized value with `repr()`, so a library read through the
name-splitting middleware comes back out carrying `author = {[NameParts(first=['Jane'], ...)]}` — valid
BibTeX with every author gone, and nothing logged. `bibtex.write_string` works around it locally and pins
the behaviour with a test.

*Cost: S to report, ? to land. Gate: none. Separate session.* Juha's read is that the maintainers are
responsive, so this is worth a minimal reproduction and a PR rather than a permanent local workaround. Two
candidate fixes to offer: raise on a value the writer cannot serialize, or apply the inverse middleware
automatically. The first is the smaller ask and fixes the silence, which is the actual harm.

### Sanitization symmetry: which importers clean, and which demand a clean file

**Raven has no uniform stance, and the asymmetry is now visible.** The Visualizer's importer sanitizes the
`.bib` it reads — dehyphenation, markup conversion, and as of today boilerplate stripping. The arXiv chain
does its own cleaning. `raven-deduplicate` deliberately does none, on the principle settled today that
repair belongs in `raven-fixbib`. Three importers, three answers.

**Decided by Juha, 2026-08-28, and the framing is the decision.** The question above is posed wrongly, and
so was the reading of it offered here first — which proposed that a tool writing a `.bib` should be
faithful to the bytes it read, while a tool building a dataset could clean freely. That is an asymmetry,
which is almost as bad as the one it replaced: two tools writing bibliographies would produce different
qualities of the same record, and the reader would have to know which had touched theirs.

**The framing that resolves it: the input is not a document, and the output is.**

> A `.bib` concatenated from several databases, with duplicates and broken records mixed in, is unusable
> as it stands. There is no useful provenance to record about it — except which items were merged, which
> is what the audit is for. The point of deduplicating and repairing is to get *something citable* out.
> The tools only need to give the user a useful copy of the metadata, as a `.bib` file.

That answers the policy question outright: **every tool that produces a `.bib` produces the most usable
metadata it can**, and no tool needs to defend the byte-level shape of an input nobody could cite from.
The asymmetry goes because there is nothing on the other side of it.

Worth noting what was *not* in question, since it is the reason one might have hesitated: whether a study
is sound is orthogonal to any of this, and is Retraction Watch's job rather than a bibliography tool's. We
are handling metadata, not adjudicating literature.

**What that changed in the code**, same day: `raven-deduplicate` reads through the whole of `fixbib`, the
entity decoding as much as the structural rescue. Half of it was arbitrary — the structural half already
rewrote records that were never merged, so nothing about the output had been byte-for-byte anyway, and the
half-measure bought an asymmetry against `raven-arxiv2bib` and nothing else.

**And where the line falls for a rights notice. Decided by Juha, 2026-08-28:**

> It doesn't [count as useful metadata]. Humans would also strip it off before analyzing the abstracts.
> Arguably doesn't even belong in the *abstract* field — it's separate metadata.

So `fixbib` should strip it from the abstract, and the second sentence is the part that shapes the
implementation: the notice is **metadata in the wrong field**, not noise. That argues for *relocating* it
rather than deleting it — which also keeps `fixbib` honest, since unlike the Visualizer's importer it
rewrites the user's own file, where dropping content outright is a heavier act than dropping it from an
in-memory dataset that a re-import regenerates.

Note this reverses a stated non-goal in `common/text/boilerplate.py`, whose docstring says it does not
record what the notice said and points a caller at "the record's own fields, where an exporter that knows
it puts it". Measured over the corpus, that pointer is half-right and half-circular: **1598 records do
carry the rights statement in a field of its own — `annote`** — and 1658 abstracts carry one anyway, often
the same record doing both. So the exporters that know do put it somewhere, and it is still in the abstract
as well.

**Decided: it moves to `copyright`** (Juha, 2026-08-28). Absent from real exports so it collides with
nothing, and not typeset by standard BibTeX styles so it cannot reach a reference list. `annote` had the
precedent here but is already three kinds of thing merged into one value; `note` is typeset.

**And the reason the notice is worth keeping at all is narrower than "it is metadata"**, which is Juha's
correction and the thing that sizes the feature: nobody is going to redistribute a `.bib` pulled out of a
paywalled aggregator — that runs into the rights in a *collection* rather than in the items — and personal
use is fine under the same laws. So the notice is not being preserved for legal reasons. Its one real use
is **provenance: telling the user which of their own exports an item came from.**

Which is why it is *plural*. A merged record came from several exports, so `raven-deduplicate` unions
`copyright` across a cluster rather than choosing one, joined with newlines the way
`repair_duplicate_field_keys` joins repeated fields — same reason, and the two read alike. Picking one
would throw away the only thing saying where the other copy came from, and a union means nothing was
dropped, so no audit row is needed for it.

Shipped the same day: `boilerplate.split_rights_notice` returns both halves, `bibtex.relocate_rights_notices`
does the move at text level so records with no notice stay byte-identical and the diff shows exactly which
abstracts changed, and `raven-fixbib` runs it by default with `--keep-notices` to opt out.

**Superseded below.** The either/or that follows is kept for the record only — it was the shape of the
question before the framing above dissolved it, and nothing is gated on it any more.

*Cost: was M, now closed.* For the record, the two ends that were being weighed:

- **Every importer sanitizes.** A user drops in whatever a database gave them and it works. The cost is
  that no tool's output is a faithful record of its input, which is exactly what the deduplicator's audit
  depends on being able to claim.
- **No importer sanitizes; `raven-fixbib` is a required first step.** Honest, auditable, and one more
  command. Juha's observation is that this implies a small GUI for people who will not touch a CLI —
  which is a real piece of work, not a footnote, and it should be counted as part of the cost of choosing
  this branch rather than discovered afterwards.

Worth noting the two are not symmetric in reversibility: sanitizing at read time leaves the user's file
alone and can be revisited, while a required `fixbib` step rewrites their bibliography once and for good.

### Smaller, and not blocking anything

- **The Visualizer importer should learn the duplicate-field-key repair** (Juha, 2026-08-28). It still
  loses those records with a misleading reason. Deliberately left out of the `fixbib` commit to keep the
  blast radius small. Gated on the sanitization question only in the weak sense that it is the same
  subject — under the repair-to-read reading above it is a gap in a stance the importer already holds,
  and does not wait on anything.

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

Juha's generalization, 2026-08-28, and it is the standing rule for this kind of work:

> **Do algorithmically what we can, and invoke the judge only as needed.**

Which is both a cost argument and a correctness one, and the second is the surprising half. The cheap
reading is that model calls are slow and a rule is free. The Springer case says something stronger: on the
inputs a rule *can* settle, asking is not merely wasteful but actively worse, because the model answers
anyway and its answer can be wrong in a direction the rule never is.

**His second thought — "prime the judge with relevant information" — splits into a part that generalizes
and a part that does not**, and the split is worth keeping straight:

- **Priming with a domain convention does not generalize.** Telling the model that `_12-1` is a Springer
  living-reference version suffix would have fixed that case, and buys nothing for the next convention
  nobody anticipated. It is the same problem as trying to enumerate generic titles: you are back to
  needing the list in advance, which is what the rule already gives you more reliably.
- **Priming with what your own deterministic layer computed does generalize**, because you already have
  it and it costs nothing to include. Here `_judge_admits` computes an author disagreement and a year gap
  and uses them only to *veto* an answer after the fact; `_describe_for_judge` shows the model the raw
  fields and not the title similarity, nor which question the pair came from (a near-miss, or a DOI
  disagreement). Showing those is a concrete, untried improvement. **Untested** — the corpus offers 14
  pairs, far too few to tell whether it helps, so it is recorded as a design rather than a finding.

There is precedent for the second working. `investigations/agent-batch-classification/` found that adding
*"an identifier is not a description — do not claim to recognize which paper an arXiv id refers to"* turned
confident fabrications into correct low-confidence answers. That is priming with a **stance** rather than a
fact, and it is available whenever you can articulate the failure mode, which you generally can once you
have seen it once.

## Where this leaves the corpus

`00_stuff/rawdata/AOKK/multisource/tekoalyagentti_tutkimus.bib`, 6934 records. Not committed and not to be
— it is a search export, and this repository is public.

```
raven-fixbib      tekoalyagentti_tutkimus.bib          # 1596 records repaired, 142 entities decoded
raven-deduplicate tekoalyagentti_tutkimus_fixed.bib \
                  -o deduped.bib -a audit.tsv [--judge]
```

6934 read, **5167 unique and 1767 merged away** with the judge, 5171 and 1763 without it. Two records still
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

<p align="center">
<img src="../../img/logo.png" alt="Logo of Raven" height="200"/> <br/>
</p>

-----

# Raven's paper tools

Eleven command-line tools for the part of a literature review that happens before anyone reads anything: finding the papers, fetching them, and repairing what the databases hand you.

**This manual documents the rules the tools apply** — what counts as the same paper, when a record is dropped, what a merge keeps. For the pipeline itself, in the order you would run it, see [*Command-line tools*](../../README.md#command-line-tools) in the main README.

<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Contents**

- [`raven-fixbib` — repairing a database export](#raven-fixbib--repairing-a-database-export)
    - [What it repairs](#what-it-repairs)
    - [HTML that a database left in the values](#html-that-a-database-left-in-the-values)
    - [The publisher's rights notice](#the-publishers-rights-notice)
    - [What it reports](#what-it-reports)
- [`raven-deduplicate` — merging a multi-database search](#raven-deduplicate--merging-a-multi-database-search)
    - [What counts as the same paper](#what-counts-as-the-same-paper)
    - [What a DOI decides, and what it does not](#what-a-doi-decides-and-what-it-does-not)
    - [What a merge keeps](#what-a-merge-keeps)
    - [The audit](#the-audit)
    - [`--judge`: asking an LLM about the near misses](#--judge-asking-an-llm-about-the-near-misses)
    - [Why it errs toward leaving duplicates](#why-it-errs-toward-leaving-duplicates)
- [`raven-siftbib` — removing what a review cannot screen](#raven-siftbib--removing-what-a-review-cannot-screen)
    - [Naming the criterion](#naming-the-criterion)
    - [What comes out, and the record of it](#what-comes-out-and-the-record-of-it)

<!-- markdown-toc end -->

## `raven-fixbib` — repairing a database export

```bash
raven-fixbib myreferences.bib          # writes myreferences_fixed.bib
raven-fixbib -i myreferences.bib       # in place
raven-fixbib -n myreferences.bib       # report only, write nothing
```

### What it repairs

**Braces a parser refuses.** A record whose field values carry unbalanced braces — which is how mathematics in a title often arrives — cannot be read at all, so it is invisible to everything downstream rather than merely awkward.

**Records naming the same field twice**, which is how a database export arrives: a ProQuest record carries a separate `annote` for its copyright statement, its last-updated date and its subject terms. BibTeX has no way to say that, so the parser rejects the entry whole — title, authors and all. This can account for a large share of a file.

The repeats are **merged, not chosen between**: the values are kept and joined by newlines, and everything else in the record is left character for character as it was. Each repeat holds something different, so keeping one would be deleting your data rather than repairing it.

### HTML that a database left in the values

Decoded by default. This is the one fault here that afflicts records a parser is perfectly happy with, so nothing reports it and it survives all the way into a reference list.

**The result is BibTeX, not plain text.** A decoded character that BibTeX reserves is escaped on the way out — `&amp;` becomes `\&` — so the file stays as readable to the next tool as it was.

**An entity naming something invisible is not decoded to it.** A zero-width joiner or a directional mark is dropped; a control character or line separator becomes a space, since a `&#10;` landing mid-record would move every line after it. Real spaces are kept as themselves, no-break spaces included — not breaking the line there is what the source asked for.

Everything outside an entity is left byte for byte, and an entity naming nothing is left alone.

`--keep-entities` switches the whole thing off.

### The publisher's rights notice

Moved out of the `abstract` into a `copyright` field of its own. It is not what the paper says, and anything reading abstracts — a screening pass, a keyword extractor, a similarity measure — is reading the publisher's boilerplate along with them.

**Moved, not deleted.** In a bibliography merged from several database exports the notice is often the only thing saying which export a record came from.

**`copyright` because it collides with nothing real exports emit**, and standard BibTeX styles do not typeset it, so it cannot turn up in a reference list. A record that already has one keeps what is there and gains the moved notice below it — both name a source the record came from — and a notice already recorded there is not moved twice.

A record whose braces would not survive the split is left alone. Everything outside a moved notice is byte for byte as it was, so a diff shows exactly which abstracts changed.

`--keep-notices` leaves abstracts as they are.

### What it reports

Every report names the fault, so an unreadable `.bib` says what is wrong with it and not merely how much. Each line carries the record's key, its line number in *your* file, which fault it is, and the specifics — which fields repeat, which look unbalanced, or the parser's own complaint where it is neither. A record whose author reads `Bloggs, PhD, MSc, Joan` is reported as *too many commas* rather than as a suspected brace problem.

**A record broken twice over is repaired as far as it goes, and reported for what is left.** One naming `annote` three times *and* carrying an author BibTeX cannot express keeps the merge — so you do not have to redo it by hand after fixing the name — and the report names the name, rather than the repeated fields the tool has just fixed and would send you looking for.

`--list` names every record that was repaired rather than only counting them. Off by default, since a database export can need repairing a thousand times over.

## `raven-deduplicate` — merging a multi-database search

Search Scopus, Web of Science, ProQuest, Springer and arXiv for the same question, and your bibliography holds every paper once per database that indexes it — each copy in that database's dialect, with a different subset of the fields filled in. This finds those copies and merges them into one record.

```bash
raven-deduplicate scopus.bib wos.bib proquest.bib springer.bib arxiv.bib -o deduped.bib
```

Several files are read as one corpus, so there is nothing to concatenate first. Without `-o` the run reports what it would do and writes nothing; your inputs are never modified either way.

**The repair `raven-fixbib` performs is applied on the way through**, so records the parser would otherwise refuse are still counted and HTML left in the field values is still decoded. You get an honest number and a citable file from one command, and you do not need to run `raven-fixbib` first.

### What counts as the same paper

**Two keys, and neither is a guess:** the DOI, and the title reduced until two databases' spellings of one title agree.

**Matching is transitive.** A record sharing a DOI with one twin and a title with another brings all three together. The two keys are complementary rather than redundant — neither is present on every record, and a corpus from five databases has records carrying only one of them.

### What a DOI decides, and what it does not

**DOI equality carries a merge. DOI *inequality* does not refuse one.** A paper carrying two different DOIs usually means something ordinary — a preprint beside its published version, a repository deposit beside the journal's own, a hyphen typed as an en-dash. So a title match is not overruled by a DOI mismatch; the disagreement goes into the audit for you to look at.

**A wrong DOI does refuse a merge**, and wrong DOIs happen: a database mismatches an identifier, or an author pastes the wrong journal reference into a preprint submission, and a record ends up carrying the DOI of a different paper.

Two records sharing a DOI are kept apart only when they contradict it **twice over** — unalike titles *and* different first authors. The tool names such a pair in the audit, since that is a fault in your data rather than a fact about the merge.

Requiring both signals is what keeps the rule quiet. Either alone would split real papers: a title may change between a preprint and its publication, and two databases may disagree about which author comes first.

### What a merge keeps

**Everything, rather than the best record.** The surviving copy is the most complete one — preferring the version of record over a preprint, and the higher version of a Springer living-reference chapter. Every field that copy lacks is filled in from a twin that has one, and where two copies disagree about a field, the audit records the value that lost.

**Normalized and stripped forms decide *which* value to keep and never reach the output.** The record that wins is written as it stands, not composed from the ones that lost.

**`copyright` is the one field kept from every copy.** Each notice names one of the exports the record came from, so a merged record says where all of it came from.

**An abstract has the publisher's rights notice removed before it is compared** — and only for the comparison. What gets written is the abstract as its database wrote it. Two copies of one abstract often differ *only* by that notice, so without stripping first, "keep the longest" would be choosing a record for the size of its copyright line.

### The audit

**Written by default**, beside the deduplicated file as `<output>_audit.tsv`.

The `.bib` is what you came for; the audit is what lets you stand behind it. A review has to report how many duplicates it removed and answer for the number, and the audit is what that number is computed from: one row per merge, naming what was kept, what was merged away, which key matched, and every value that differed from the one kept.

It carries Raven's version, so a method section can cite a published tool.

- `--audit PATH` puts it elsewhere.
- `--no-audit` declines it, at the cost of the only record of what the merge did. A merge cannot be read back out of the merged file.

### `--judge`: asking an LLM about the near misses

Off by default, since it needs a backend. It asks about near-miss titles that no exact key joined, and about merges whose records disagree about the DOI.

**The model proposes and Raven disposes.** A "same work" verdict contradicted by the records themselves is dropped, so a confident wrong answer cannot create a merge the ordinary rules would have refused.

**The run is resumable.** Answers are kept in a JSONL file beside the audit (`--judge-state PATH` to place it), and a re-run skips what is already there.

**It also decides which DOI a merged work keeps.** Where the records of one work disagree about the identifier, one of them can simply be wrong, and the merge would otherwise pick by completeness and write that into your bibliography. Each candidate is checked against the venue named by the record carrying it — an astronomy journal on a paper about classroom assessment is visible, where the identifier alone is not — and a rejected DOI is dropped along with that venue, the audit saying which and why.

**The judge is asked about the venue and never about the identifier.** A DOI says nothing to a reader who does not already know it, including an LLM, while the venue beside it is something the record actually states.

**The bias is toward keeping.** A venue that is general, interdisciplinary or merely unfamiliar fits; an unparseable answer fits; and a work whose every identifier is rejected keeps them all. Those cases are treated as the model recognizing nothing rather than as a bibliography where nothing is right.

### Why it errs toward leaving duplicates

A missed merge leaves a visible duplicate that a reviewer can act on. A false merge deletes a paper from the review, and nothing downstream can notice.

That asymmetry decides the edge cases. Two records carrying the same genre label as their title — `Editorial`, `Book Review` — are merged only if they agree about the author and the year. Two authorless records carrying a serial's recurring section heading are not merged when their DOIs disagree.

## `raven-siftbib` — removing what a review cannot screen

A search export carries records of wildly uneven completeness. A record holding nothing but a title is not off topic — nobody can tell what it is — it simply has no text to screen on, and a screening pass has to account for it rather than quietly carry it into the count.

```bash
raven-siftbib corpus.bib --require abstract          # corpus_sifted.bib + corpus_removed.tsv
raven-siftbib corpus.bib --min-chars abstract=200
raven-siftbib -n corpus.bib --require abstract       # report only
```

### Naming the criterion

**The criterion is yours, not the tool's.** `--require FIELD` keeps the records carrying that field. `--min-chars FIELD=N` keeps those whose field reaches a given length. Either may be given more than once, and a record must satisfy all of them.

**A run naming no criteria is refused rather than defaulted.** A default would be the tool holding an opinion about what a usable record is, which is the caller's to hold.

**`--min-chars` is for the field that is present and useless.** Publishers routinely export a truncated teaser in place of the abstract — a sentence or two ending mid-word — which satisfies `--require abstract` and tells a screener nothing.

**Deterministic and offline.** No model, no network: the same bibliography and the same flags produce the same two files on any machine. Whether a record is *about* the right thing is a judgement for something else; this answers only whether there is enough of it to judge.

### What comes out, and the record of it

Two files, named from the input: `<name>_sifted.bib` and `<name>_removed.tsv`.

The audit has one row per removed record, naming the record, where it was published, and which criterion it failed, under a header stamping the tool version, the inputs and the tests applied. The venue is in there because it is what tells you whether a dropped record is worth chasing up by hand.

- `-o SUFFIX` / `--audit-suffix SUFFIX` rename the two outputs.
- `--out-dir DIR` puts them elsewhere.
- `--no-audit` declines the audit.
- `-n` / `--dry-run` reports what would go and writes nothing. The input file is never modified either way.

<p align="center">
<img src="../../img/logo.png" alt="Logo of Raven" height="200"/> <br/>
</p>

-----

# Raven's paper tools

Eleven command-line tools for the part of a literature review that happens before anyone reads anything: finding the papers, fetching them, and repairing what the databases hand you.

**This manual documents the rules the tools apply** — what counts as the same paper, when a record is dropped, what a merge keeps. For the pipeline itself, in the order you would run it, see [*Command-line tools*](../../README.md#command-line-tools) in the main README.

<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Contents**

- [`raven-deduplicate` — merging a multi-database search](#raven-deduplicate--merging-a-multi-database-search)
    - [What counts as the same paper](#what-counts-as-the-same-paper)
    - [What a DOI decides, and what it does not](#what-a-doi-decides-and-what-it-does-not)
    - [What a merge keeps](#what-a-merge-keeps)
    - [The audit](#the-audit)
    - [`--judge`: asking an LLM about the near misses](#--judge-asking-an-llm-about-the-near-misses)
    - [Why it errs toward leaving duplicates](#why-it-errs-toward-leaving-duplicates)

<!-- markdown-toc end -->

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

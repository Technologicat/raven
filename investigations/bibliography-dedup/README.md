# Auditing what a deduplication rule actually did

The instrument behind `raven-deduplicate`, and the reason it is here rather than in `/tmp`: **a cluster
count says how much a rule caught and nothing about what else it caught.** Both false merges found while
building the tool were invisible in statistics that looked correct, and both were found by listing what a
rule did, grouped by whatever the rule keys on, and reading it.

That is a check that has to be re-run every time a rule changes, so it is a script rather than an
afternoon.

## Files

| File | What it answers |
|---|---|
| `audit_rules.py` | What each clustering, merging and stripping rule did to a given `.bib`, in six sections meant to be read rather than totalled. `python audit_rules.py search.bib [--limit N]`; writes nothing. |
| `doi_calibration.py` | Where `config.doi_title_floor` can go, and why the guard it serves needs a second condition beside it. `python doi_calibration.py search.bib`; writes nothing. |

```bash
python audit_rules.py path/to/search.bib --limit 20
python doi_calibration.py path/to/search.bib
```

Both read through `raven-fixbib`'s repair exactly as the tool does, so their counts are the tool's counts.
Point them at any bibliography; nothing here is specific to the corpus this was built against, and that
corpus is not committed — it is a search export and this repository is public.

## Why the calibration is a script and not a note

`doi_calibration.py` produced a number that is now in `raven.papers.config`, and the temptation was to
write the number down and throw the script away. What the script holds that a note cannot is the *shape*
of the evidence: a distribution, a counter-example that no corpus of correct records can supply, and the
seven hundredths between them that make a title threshold alone impossible to place. Anyone moving
`doi_title_floor` — for a corpus in another language, or one drawing on venues this one did not — needs
that shape rather than the number, and needs it for *their* corpus.

It carries its own negative control, built in the file: two papers that really do claim one DOI, one of
them from arXiv metadata that names an astronomy journal on an education paper. The last thing it prints
is whether `_doi_edge_holds` still refuses them, so a change that quietly stops the guard firing on the
case it was written for says so.

## The six sections, and which bug each would have caught

- **Shape of the run.** The aggregate. Framing for the rest, and the part that is *not* the check.
- **Merges with no author anywhere.** Where a title carried the whole decision. This is the section that
  found the serial-heading bug: `II Political Science: Method and Theory` heads an item in every issue of
  its journal, and four issues had merged into one record. Of 36 authorless merges, the three whose DOIs
  disagreed were the three that were wrong — which is what turned a list of titles into a rule.
- **Title matches refused by the guards.** The other direction, and the only place it is visible: every
  refusal is a merge that did not happen, so a wrong one leaves a duplicate in the output.
- **Merges whose records disagree about the DOI.** Ordinary in most cases — a preprint beside its
  published version — and the likeliest place for a wrong merge, since a title match was made over a
  contradiction the records were able to state.
- **The nothing-disappears invariant**, checked exhaustively rather than on a fixture. Counts the one
  designed exemption separately, so a change to abstract handling shows up as a number moving rather than
  as silence.
- **Largest boilerplate cuts.** Reading the top of this list is what caught `strip_boilerplate` eating an
  abstract that *discussed* copyright. A collapse statistic cannot show that: a wrong cut and a right cut
  are both cuts.

## What is deliberately not here

The one-off probes that produced the design's figures — how many records carry a DOI, what the entity
census was, whether `bibtexparser`'s writer round-trips. Those discovered an answer rather than asserting
one, the answers are in `briefs/bibliography-dedup-brief.md`, and the ones that turned out to be
invariants are in the test suite (`raven/papers/tests/test_deduplicate.py`) where they run on every push.

The write-up is the brief rather than a document here, because the measurements were made *while*
designing rather than to settle a question of their own.

# Ligature repair, and the lost hyphens that come with it

**Status: designed, not started. The recommendation is to build half of it.**

**Scheduled 2026-08-07 (Juha): waits, unless the `raven-fixbib` half is small enough to sneak in.** The
Researchers' Night work comes first. This brief argues that half *is* small — the function plus a flag plus
a report, with the collection already in hand — so it is a candidate for a gap rather than a slot. The
indexer half is not a candidate under any reading.

Sits outside the numbered Librarian run because it is not Librarian work: it touches `raven.papers`
(`raven-fixbib`) and `raven.common` (`docextract`), and its consumers are the Visualizer importer and the
Librarian indexer equally. Corpus hygiene, not a feature of any one app.

Discussed 2026-08-07 (Juha and Claude); the defect and its measurements are below, the design that came out
of them follows.

## The defect

A PDF extractor can emit a font's ligature glyphs as raw control codes rather than as characters. The
ECCOMAS 2024 bibliography carries six: five `U+001C` and one `U+001D`, and the surrounding text says plainly
what they were meant to be — `phase-`␜`eld`, `of `␜`nite`, `the in`␝`uence`. So in that file `U+001C` is
**fi** and `U+001D` is **fl**, which makes them the two most common ligatures in English prose rather than
anything exotic.

Prototyped 2026-08-07 against that corpus, and it separates cleanly — **every ligature site resolves to
exactly one candidate, and every non-ligature site to none:**

| codepoint | context | resolves to | corpus frequency |
|---|---|---|---|
| `U+000E` | `coe`␍`cients` | **ffi** → coefficients | 99 |
| `U+001B` | `a`␍`ected`, `e`␍`ects` | **ff** → affected, effects | 60, 397 |
| `U+001C` | `phase-`␍`eld`, ␍`nite` | **fi** → field, finite | 1230, 1932 |
| `U+001D` | `in`␍`uence` | **fl** → influence | 235 |
| `U+001A`, `U+0001` | after `considered`, `system`, `holds` | — | correctly rejected |

Discovered while bringing up the corpus, from `raven-fixbib` mis-splicing a repair — the same six characters
make `str.splitlines()` and a newline count disagree.

### Why `normalize` must not be wired into `docextract`

**The trap is that stripping the control codes looks like hygiene and is data loss.**
`raven.common.text.normalize` deletes C0/C1 controls, and its docstring invites exactly this use ("any
handler of untrusted retrieved text ... future PDF ingestion"). Wiring it into `docextract.extract_text` was
tried and reverted the same hour: deletion turns *phase-field* into *phase-eld*, *finite* into *nite*,
*influence* into *inuence* — silently, across every affected document, in text nobody will re-read.

Replacing them with a space is no better. It breaks the word visibly instead of invisibly, and leaves a
trailing hyphen that a later dehyphenation pass would then close up into the same wrong token.

**So until the repair exists, `docextract` leaves the control characters alone.** They are invisible in an
editor and they survive into the index, and that is the *better* of the available wrong answers, because the
text is still recoverable from them. Only 3 of ECCOMAS's 2520 records are affected, so the cost of waiting is
small — which is also the measurement behind "build the fixbib half, specify the indexer half" below.

## The shape

**The mapping is a property of the font, not of Unicode.** `U+001C` means "file separator" in general and
"fi" only in this document's encoding, so a fixed table is a guess dressed as a standard. But the guess is
checkable the same way the BibTeX brace repair is: propose each known ligature, and accept the one that turns
the surrounding letters into a word **the corpus itself uses**. No external dictionary, no locale assumption,
and the evidence comes from the same data being repaired.

**The vocabulary has to come from the whole collection, and that is the constraint everything else follows
from.** Measured against ECCOMAS 2024: built per-document it resolves 0 of 12 sites; built from the whole
5.6 MB file, 12 of 12. A single conference abstract never happens to spell the damaged word correctly
somewhere else in its own 1600 characters.

So the API is plural, and unavoidably so:

```python
repair_ligatures(texts: Sequence[str]) -> list[str]   # or an iterable-in, iterable-out variant
```

A per-document signature cannot be made to work, which is worth stating in the docstring — it is exactly
the shape a later reader will try to "simplify" it into.

## Two products, one function

Decided (Juha): **the fixer and the reader are doing different jobs, and the difference is who owns the
bytes.**

- **`raven-fixbib` repairs files.** The user invoked a tool whose whole purpose is to fix their
  bibliography; rewriting it is the thing they asked for. Same gesture as the brace repair already there.
- **The `docextract` path repairs *extracted text*, never the source.** It is trying to hand something
  readable to an LLM and to a human out of a possibly-corrupt source, and changing that source is off the
  table — the user did not ask for it and may not want it.

Both come from the one function; what differs is whether the caller writes the result back to disk.

## Decisions

| | Decision | Why |
|---|---|---|
| **Ambiguity** | Repair only when exactly one candidate yields a corpus word. Otherwise leave the site alone and report it. | Every ECCOMAS site resolved uniquely, but that is one corpus. A silently wrong repair is unrecoverable; a surviving control character is invisible but the text is still *there*, which the deferred item already argues is the better of the available wrong answers. |
| **Vocabulary source** | A surface-form scan of the collection's text, with a cheap regex. **Not** the BM25 token vocabulary. | `hybridir` tokenizes through spaCy with lemmatization and stopword removal, so its vocabulary is in lemma space — matching a candidate against it needs a lemmatization round trip per candidate, and loses the surface forms the repair is actually reconstructing. A regex scan needs no ML, no server, and no index, which is also what lets `raven-fixbib` use the identical code path with no retriever anywhere in sight. |
| **Hyphens** | Same pass. `strainstiffening` → *strain-stiffening* is one character's damage between two word fragments, checkable the same way. | Guard it on **the joined form being absent from the corpus** — otherwise a genuine closed compound gets split. This guard has no counterpart in the ligature case, where the damaged form is never a word. |
| **`raven-fixbib` gating** | Behind a flag, not on by default, and reporting each site it changed. | Brace repair fixes a *parse failure* the user can see. This changes text content the user cannot see either before or after, which is a bigger promise and should be asked for. |

## Scope: build the fixbib half, specify the indexer half

**The effort is asymmetric, and only one half carries the complexity.**

`raven-fixbib` already reads the whole file, so the collection is simply *in hand*. No scan to arrange, no
ordering question, no second pass, no new state. It is the function plus a flag plus a report.

The indexer half needs the repair to **mutate stored text** and **re-tokenize** the documents it changed
(a server round trip), inside a subsystem that today only ever appends. That is new machinery in a hot
path, and it buys a defect rate of **3 documents in 2520** on the one corpus where it has been measured.

So: **build the fixbib half now; leave the indexer half designed and unbuilt.** If a corpus ever turns up
where the damage rate is high enough to matter, the design below is ready.

### The indexer half, for when it is wanted

Two findings from working it through, both worth keeping because neither is obvious:

**The "second pass" problem solves itself, for a reason that is easy to miss.** The worry is real —
document 1 is repaired before document N exists, so it never saw N's vocabulary. But repair only ever
fires on a *unique* resolution, so a site that could not be resolved **still carries its control
character**. Re-running the same whole-corpus pass at the next commit *is* the second pass: already-repaired
text has nothing left to repair, and unresolved damage gets another attempt against a larger vocabulary.
There is no bookkeeping of which documents were done, and no separate mechanism — just the same pass, run
again. It is cheap because the guard is "does this text contain any of the ligature codepoints", false for
almost every document.

**A whole-corpus pass at commit is already what the indexer does.** `HybridIR._rebuild_keyword_search_index`
rebuilds the entire BM25 index from `self.documents` on every commit, and the TODO above it explains why:
*"The new document may have added new tokens so that the token vocabulary must be updated."* Every chunk's
text is in hand at that moment.

**The coupling to write down before it bites:** that same TODO wants indexing to become incremental. If it
does, the corpus-wide guarantee this design leans on disappears silently — repair would start seeing only
the documents in the current commit, i.e. the per-document case that resolves 0 of 12. Whoever makes
indexing incremental has to give ligature repair its own corpus-wide pass, or drop it.

## Next actions

1. `repair_ligatures` in `raven.common.text`, with the corpus-as-dictionary requirement stated in the
   docstring, plus the hyphen case.
2. Wire it into `raven-fixbib` behind a flag, with a per-site report.
3. Tests from the ECCOMAS evidence already gathered: the four codepoints that resolve (`U+000E` → ffi,
   `U+001B` → ff, `U+001C` → fi, `U+001D` → fl), the two that correctly resolve to nothing (`U+001A`,
   `U+0001`), the per-document-vs-whole-collection difference (0 of 12 against 12 of 12, which is the
   measurement that justifies the plural signature and should fail loudly if someone narrows it), and an
   ambiguous site that must be left alone.

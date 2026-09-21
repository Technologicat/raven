<p align="center">
<img src="../../img/logo.png" alt="Logo of Raven" height="200"/> <br/>
</p>

-----

# Raven's paper tools

Eleven command-line tools for the part of a literature review that happens before anyone reads anything: finding the papers, fetching them, and repairing what the databases hand you.

**This manual documents the rules the tools apply** — what counts as the same paper, when a record is dropped, what a merge keeps. For the pipeline itself, in the order you would run it, see [*Command-line tools*](../../README.md#command-line-tools) in the main README.

<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Contents**

- [The arXiv tools](#the-arxiv-tools)
    - [Identifiers and versions](#identifiers-and-versions)
    - [arXiv's rate limit, and why batching matters](#arxivs-rate-limit-and-why-batching-matters)
    - [When something goes wrong](#when-something-goes-wrong)
    - [What a rerun does](#what-a-rerun-does)
    - [Downloading](#downloading)
- [The converters — getting an export into BibTeX](#the-converters--getting-an-export-into-bibtex)
    - [`raven-wos2bib` — Web of Science](#raven-wos2bib--web-of-science)
    - [`raven-csv2bib` — a spreadsheet](#raven-csv2bib--a-spreadsheet)
    - [`raven-pdf2bib` — conference abstracts as PDFs](#raven-pdf2bib--conference-abstracts-as-pdfs)
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
- [`raven-burstbib` — one file per record](#raven-burstbib--one-file-per-record)

<!-- markdown-toc end -->

## The arXiv tools

`raven-arxiv-search` finds papers, `raven-arxiv2id` reads identifiers out of the filenames you already have, `raven-arxiv2bib` turns identifiers into BibTeX, and `raven-arxiv-download` fetches the PDFs. The main README says what each one does; what follows is what they decide.

### Identifiers and versions

**A bare identifier means "whatever is current".** `2410.07866` asks for the newest version; `2410.07866v5` pins that one.

**What gets recorded is the version arXiv answered with**, however you spelled the request. Ask for the bare form and the BibTeX entry names the version you actually got, so a bibliography says which text the claims in it came from.

**That is what makes a collection refreshable.** `raven-arxiv2id --strip-versions` prints each identifier without its version suffix, and re-fetching those gets you whatever is current now. Scanning filenames without it keeps the newest version of each paper that you already have.

### arXiv's rate limit, and why batching matters

**Three seconds between requests is arXiv's requirement, not a tuning knob.** Raising it is fine; lowering it is a violation of their terms rather than a faster setting. It lives in `raven.papers.config.arxiv_request_delay` where a user can find it.

**The delay is charged per request, not per paper**, which is the whole reason metadata is fetched in batches — up to `arxiv_id_batch_size` (100) identifiers per request. A download of several hundred papers spends its politeness budget on a handful of requests rather than on one per paper.

**A search pages at 200 results and stops at 30,000**, the latter being arXiv's own ceiling however you page through it. A query expecting more than that needs splitting into narrower ones.

### When something goes wrong

**An identifier arXiv returns nothing for is reported at the end, not thrown.** One withdrawn or mistyped entry cannot cost the several hundred that worked.

**HTTP 429 is retried**, up to three attempts total, with backoff that honours a `Retry-After` header when the response carries one. Transport failures — dropped connections, read timeouts — are retried the same way.

**A malformed identifier fails in one line.** A typoed month like `2614.19062` reports "no arXiv entry for ID …" rather than a traceback, that being an expected outcome rather than a bug.

### What a rerun does

**Almost nothing, which is the point.** The summary is counted by outcome — downloaded, already present, duplicate identifier, no PDF available, failed — so running the same list again tells you it is already there rather than fetching it again.

**A repeated identifier is dropped before any metadata is fetched**, rather than being carried to the download step and skipped there. Both arXiv tools treat duplicates the same way.

### Downloading

**The citation is printed before the PDF is fetched** — `Authors (Year) - Title` — so it is on screen during the rate-limit wait, and a mistyped identifier that resolved to the wrong paper is visible before the file lands.

**Filenames keep clause boundaries.** A `:`, `?`, `!` or `;` followed by a space becomes ` - ` rather than being dropped, so a long title does not come out as a run-on sentence.

**`--save-bib file.bib` writes the metadata that was already fetched**, so downloading a set of papers and building its bibliography costs one set of politeness delays rather than two.

## The converters — getting an export into BibTeX

All three write BibTeX to stdout, so redirect them, and all three read several inputs as one corpus.

### `raven-wos2bib` — Web of Science

Web of Science indexes the engineering sciences well, and exports plain text.

```bash
raven-wos2bib input1.txt ... inputn.txt 1>output.bib 2>log.txt
```

The inputs are `.txt` files exported from Web of Science; the BibTeX goes to stdout and the log — including warnings about broken input data — to stderr, so redirect both. The resulting `.bib` imports into Raven-visualizer.

**Two publication types are mapped**: `J` becomes `@article` and `B` becomes `@book`. Series and patents are not handled yet, so an export holding them needs a look.

**Author affiliations are normalized to one string.** Web of Science's `C1` field arrives in more than one shape depending on the export, and BibTeX has one field to put it in.

### `raven-csv2bib` — a spreadsheet

**The header row is the schema.** Each column name becomes a BibTeX field name verbatim, and each row becomes an `@article` — so a sheet with columns `author`, `year`, `title`, `abstract` converts without any mapping to configure, and a sheet with a column called `Notes on screening` produces a field of that name.

**The delimiter is autodetected** — tab or semicolon — and `-d` forces one. Values are BibTeX-escaped and braced on the way out.

**Entry keys are generated**, one UUID per row, since a spreadsheet rarely carries one. They are unique rather than memorable; if you want citable keys, set them afterwards.

### `raven-pdf2bib` — conference abstracts as PDFs

Abstract submissions to scientific conferences sometimes arrive as free-form, human-readable PDF files. If you are a conference organizer who would like to semantically visualize the set of abstracts sent to you, this is the way in. It extracts each PDF's text with [`raven.common.docextract`](../common/docextract.py) and has an LLM turn it into a BibTeX entry.

**An LLM is required here rather than optional.** `raven-deduplicate`'s `--judge` can simply be left off; this tool does not work without one, and its output is worth reading before you trust it.

```bash
raven-pdf2bib -i some_input_directory --slug myconf --year 2026 \
    -s success.bib -f failed.bib -l log.txt -o done_success -of done_failed
```

`--slug` and `--year` are required: the conference they name is injected into every entry, since it is not reliably in the files, and `--booktitle`, `--note` and `--url` are optional additions to the same. The LLM is the configured backend; `--backend-url` points it at another one serving an OpenAI-compatible API.

#### What it expects of a PDF

- **An extractable text layer** — a born-digital PDF, not a scan.
- **One abstract per file.** Several abstracts are fed in as separate PDFs.
- **A human-recognizable title, authors and main text.** Exact formatting does not matter.
- No length limit is enforced, but the intended case is a typical conference abstract of one or two pages.

The extracted text is sanitized with the dehyphenator before it is written to the `abstract` field. If the abstract carries a line beginning *"keywords:"* or *"key words:"*, keywords are extracted too.

#### Running a directory

Every PDF under `-i` is converted, descending into subdirectories, one directory at a time in Unicode lexicographical order by filename. Without `-i`, the working directory is used. The directories named by `-o` and `-of` are skipped while descending, so `-o some_input_directory/done_success` is safe.

**The output buffers are flushed after each entry**, so a text editor that notices changed files can be used to watch progress.

**A long run is resumable, and that is what the directory moving is for.** A PDF is moved to `-o` only if it was processed successfully, and only *after* its entry has been written; `-of` moves the failures likewise. Stop the job and run the same command again — the success bib, the failure bib and the log are all appended to — and it picks up what is left. `-r` sets how many attempts one paper gets, three by default.

#### How it works, and why you should check its output

**Anything after a *"References"* section title is discarded** before processing, which stops the reference list contaminating the title and the author list.

**Fields are extracted one at a time**, which is more reliable than asking for the whole record at once. The system prompt and each field's prompt have been engineered by hand; no automatic prompt optimization has been tried.

**The extracted data is double-checked by heuristics** where that is reasonably possible, and anything suspicious is flagged with a warning. **It is very strongly recommended to check a flagged entry against the original PDF**, because a flagged entry is very likely to be wrong in one or more ways.

Failures go to `-f`, are logged to `-l`, and their full LLM traces are written into the *input* directory beside the file — `myfile.pdf` leaves `myfile_errors.txt`. LLMs being stochastic, simply moving the PDF back and retrying sometimes works.

People do submit abstracts with no author list, or no title at all. The converter tries to catch that and does not always manage it. And an LLM may confabulate, answer incorrectly, or fail to follow an instruction — which is what the heuristics and the flagging are for, and why a flagged entry is worth the minute it takes to check.

#### Which LLM to use

**Only ever tested with a locally hosted backend**, meaning on the same LAN as Raven, possibly the same machine. A cloud LLM with an OpenAI-compatible API *might* work if you put a key in `~/.config/raven/librarian/api_key.txt` and set the URL in `raven.librarian.config`; it is not a development priority.

For hosting one, we recommend [oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui), which Raven is tested with, through its OpenAI-compatible API. Others such as [AnythingLLM](https://anythingllm.com/) might work and have not been tested — *OAI compatible* covers several dialects, so some features may not.

- **2024**: originally tested on a local Llama 3.1 8B on Oobabooga. Fits a laptop's 8 GB VRAM at 4 bits (Q4_K_M) with room for 24k tokens of context. Accuracy about 80% — eight abstracts in ten convert without warnings and look right on inspection.
- **February 2025**: thinking models supported, first tested on a Q4_K_M [DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B) at 64k context on a 24 GB eGPU. A couple of percentage points more accurate, and much slower. [QwQ 32B](https://huggingface.co/Qwen/QwQ-32B) runs on similar specs.
- **December 2025**: [Qwen3 2507 30B A3B Thinking](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507), on a 24 GB eGPU at 4 bits. A MoE with 3B active parameters per token, so much faster than the earlier models; loads with 128k context and works at least to ~50k. Noticeably smarter, and informally about 88% accurate. **This is the most recent model the converter has been measured against.**

**For what to run now, see the [main README](../../README.md#raven-librarian-multiversal-llm-frontend)**, which carries the current recommendations by VRAM for the whole constellation. They move with each model generation, so they are kept in one place rather than repeated per tool — and the accuracy figures above belong to the models named beside them rather than to whatever you end up running.

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

## `raven-burstbib` — one file per record

Splits a multi-entry `.bib` into one file per entry, which is what makes a bibliography usable as a **document database**: Raven-librarian indexes a folder of files, and a bibliography is one file holding hundreds of records until this is run over it. Records carrying abstracts are the ones this pays off for.

```bash
raven-burstbib references.bib -o records/
```

**Each file is named from the record's own BibTeX key**, sanitized so it can be a filename. That matters more than it sounds: a `.bib` written by somebody not steeped in BibTeX often carries a DOI or a URL where the key goes, and those contain characters a filesystem will not take.

`-V` prints progress, which is worth having on a file with thousands of records in it.

**It splits rather than parses.** The implementation finds record boundaries by looking for header lines, so it does not read the BibTeX as a grammar and cannot be relied on for a file that is malformed in the middle of a record. Run [`raven-fixbib`](#raven-fixbib--repairing-a-database-export) first if the file came out of a database export.

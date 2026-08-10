# Spreadsheets in the docs DB and attachments (`.xlsx`, `.ods`)

**Status: designed, not started.** The shape is decided; nothing is built.

Sits outside the numbered runs for the same reason `ligature-repair-brief.md` does: it is `docextract` work,
and its consumers are the Librarian indexer and the chat attach path equally. Corpus ingestion, not a feature
of any one app.

This brief exists because the design was decided inside a `TODO_DEFERRED.md` item that has since been closed
— office-format support landed 2026-07-29 (`093c400`) for word processor documents, presentations and saved
web pages, leaving spreadsheets as the one format still out. Deferred items are deleted when they close, so
the reasoning moved here rather than going with it. Wording below is carried over as it was written.

## Why it was left out of the office-format work

Spreadsheets are the awkward ones, which is why they were left. A sheet is not a linear document, so "the
text of a spreadsheet" is a design decision before it is an `openpyxl` call.

Put the other way round: a spreadsheet is a different problem class wearing the same file picker. Its content is tabular, so "the text
of this file" is not well defined — reading a sheet row-major into a paragraph produces something that chunks
badly for retrieval and reads poorly when folded into a chat message. Getting it right means deciding how a
table becomes prose (or whether it should become Markdown table syntax instead, which the model can actually
read), and how a multi-sheet workbook maps onto one document.

Worth doing eventually — research data does arrive as spreadsheets — but as its own design question, not as
three more lines in the extractor's dispatch.

## The agreed first approximation

Decided by Juha, 2026-07-30: **emit Markdown tables.** One table per detected table region, regions delimited
by at least one fully blank row or column, taken in Western reading order (left to right, then top to bottom).
Markdown is the right target — the models are steeped in it, and `docextract`'s other formats already produce
prose that the chat view renders as Markdown, so it needs no new convention.

One substitution on that sketch: **separate sheets with a heading carrying the sheet name** (`## Sheet: Q3
Budget`) rather than a bare `-----`. A horizontal rule says "something else starts here" and throws away the
name, which is often the single most informative string in the file — "Assumptions" versus "Raw data" tells a
reader, and a model, what it is looking at. Same cost, strictly more information.

## The details that decide whether the output is useful or merely plausible

- **Values, not formulas.** `openpyxl`'s `data_only=True` yields the *cached* result, which is present only if
  a real spreadsheet application last saved the file.

  The empty-cell case needs **two** things true together: the file contains formulas *and* was never saved by
  an application that computes them. Most inputs fail one of those, which is why the expected sources look
  safe — a report downloaded from a web dashboard is usually pure values with no formulas at all (so
  `data_only` is moot), and a human-authored workbook has been through Excel or LibreOffice (so the cache is
  populated). The gap is narrow: a formula-bearing file written by a library (`openpyxl`, `xlsxwriter`,
  `pandas`) and never opened.

  What makes it worth handling anyway is that the failure is **silent** — blank cells, not an exception — so
  it surfaces as a confidently empty table rather than an error. Cheap insurance for a narrow case, not a
  workaround for a common one: fall back to the formula text, and never emit a table that is entirely blank
  without saying why. Confirm the behaviour against a file written by `openpyxl` itself before relying on any
  of this.
- **Merged cells.** Markdown cannot express a merge. `openpyxl` reports the value in the top-left cell and
  `None` for the rest of the range; repeating the value across the merged span usually retrieves better than
  leaving blanks, since a row then still reads as a complete record.
- **The used range lies.** One stray cell far out to the right makes a sheet nominally enormous. Bound the
  emitted region by actual content, and cap total output — a 50k-row sheet rendered in full is a wall of text
  that crowds out the question being asked about it.
- **Charts, images and pivot caches: skip.** No text to extract, and a placeholder line invites the model to
  comment on something it cannot see.
- **`.ods` may be nearly free.** `odfpy` is already a dependency (it backs `_extract_odf` for `.odt`/`.odp`)
  and handles spreadsheets too, so the second format is likely a different reader over the same
  region-detection and Markdown-emission logic. Worth structuring the code that way from the start.

## Out of scope

The legacy binary formats (`.doc`, `.ppt`) are deliberately out of scope: reading them means shelling out to
a separate converter.

## Where it is tracked

`TODO_DEFERRED.md`, "Spreadsheets in the docs DB and attachments (`.xlsx`, `.ods`)" — kept as the queue entry,
pointing here for the design.

# Sketch: what an ingestible document is

**Status: a discussion sketch, not an implementation brief.** Written 2026-10-01 to gather the
`document-ingestion` cluster of `TODO_DEFERRED.md` into one picture. The subject is too large for one brief,
so this names the briefs it splits into, each sized to close on its own (`briefs/README.md`, *Size a brief so
that it can close*). The decisions below are the ones the items already record; the split and its order are
the proposal.

## What it covers

The cluster's items, all in `TODO_DEFERRED.md`:

- *Same file formats in the docs DB and in chat attachments* — the head item: a user who can attach a file
  expects to be able to index it, and vice versa.
- *Spreadsheets in the docs DB and attachments* — with its own brief already,
  `briefs/spreadsheet-ingestion-brief.md`.
- *Vector figures in the docs DB and attachments (`.svg`)*.
- *Text out of images, so figures work without a vision model* — OCR, and SVG `<text>`.
- *Read documents as page images, for figure- and math-heavy sources*.
- *HTML pages whose content is produced by running them*.
- *Cite a retrieved passage by the page number printed on the page*.

Two neighbours outside the cluster touch the same code: `briefs/ligature-repair-brief.md` (`docextract`'s
text, repaired), and OCR for scanned PDFs in `TODO.md` ("RAG PDF ingestion — polish").

## The shape, already decided

**Two columns, split by what the caller wants out** (decided 2026-07-30, in the SVG item):

|              | → text                   | → pixels                          |
|--------------|--------------------------|-----------------------------------|
| **document** | `docextract` (exists)    | PDF/docx page images (wanted)     |
| **image**    | OCR; SVG `<text>`        | `image.codec` (exists)            |

Grow "give me the text of this file" from `docextract`, and "give me pixels for this file" from `codec`.
Retrieval wants text whatever the file was; a vision model wants pixels whatever the file was. An
`imageextract` module would split along the rows instead, and a PDF rendered to page images belongs in both.

**And the constraints every brief below inherits:**

- **One chokepoint for both surfaces.** The indexer and the attach path both call `docextract.extract_text`,
  so a format added there serves both, and a format added at one call site breaks the symmetry the head item
  asks for.
- **Nothing runs a file to read it, and nothing reaches the network.** Rendering an HTML page in a browser is
  ruled out for the automatic paths: dropping a file into a watched folder must never execute its scripts. An
  SVG rasterizer must have external entities and remote fetching off, and that wants a test rather than an
  assumption.
- **Expensive per-document work runs at import, offline** (`briefs/design/offline-and-remote-processing-sketch.md`),
  and what it produces — OCR text, page images, descriptions — are brief 12's derived artifacts, which already
  names `ocr_text` and `page_image` as kinds.
- **Large material is made fetchable, not smaller.** A page image is a few thousand tokens; the model asks for
  the pages it wants rather than receiving every one.

## The briefs it splits into

Each is meant to land and close by itself. Sizes are the items' own where they have one.

1. **Spreadsheets** — the existing brief, designed: Markdown tables, one per detected region. Probably M
   (maintainer, 2026-10-01). No gate.
2. **SVG** — rasterize in `image.codec` at the declared size, keep the vector original as the archival
   sidecar, and extract `<text>` elements as the figure's text. Picks a backend (`cairosvg`, `svglib`, …) and
   tests that nothing is fetched. No gate.
3. **Self-contained HTML apps** — the data in a page that builds its DOM from an inline `<script>`. Read
   declared data (`application/json`, `ld+json`), and perhaps fall back to script text when readability found
   nothing and the script is small. The item calls this the highest-value of the reading gaps. No gate.
4. **Page-anchored text** — `extract_text` keeps page boundaries, with each page's physical index and its
   printed label (`pypdf`'s `page_labels`). Citations can then name the printed page, *"p. 32 (PDF 5)"*. M, and
   the item's gate is exactly this decision about what `extract_text` returns.
5. **Page images on request** — a `read_pdf_page(document, pages)` tool, a per-call budget guard, rendered pages
   cached as derived artifacts, and a retention policy, since an injected page costs its tokens on every later
   turn. Needs 4, and its retention policy needs a word with *Context-window budgeting and compaction*.
6. **Text out of raster images** — OCR for text-bearing images (scans, screenshots), a VLM transcription at
   ingest for figures, stored as searchable text. Needs:
   - **brief 09's tokenizer fix**, or the extracted text is mangled — figure text is mostly the proper nouns,
     symbols and digits today's tokenizer lowercases, lemmatizes or drops;
   - brief 12's producers, and the offline processing above, since it is per-image work over a whole corpus.
7. **Images as documents in the DB** — the image half of the head item. Needs the multimodal embedder, which
   goes in during brief 13's build, so it probably belongs to that work rather than here. The Visualizer
   showing images is its expensive half.

Independent of each other and ungated: 1, 2 and 3. A chain: 4 → 5. Behind other work: 6 (the tokenizer fix,
brief 12, offline processing) and 7 (brief 13).

## Open

- Whether 7 is this sketch's at all, or brief 13's from the start.
- Whether OCR for scanned PDFs joins 6, being the same engine, or stays with the PDF polish in `TODO.md`.
- In 3, the size budget for the script-text fallback, which is a threshold to derive rather than tune.

# Summer 2026 Librarian extension

Implementation briefs for the Librarian work of summer 2026. Completed briefs move to `done/`; live
probes shared across briefs live in `manual_tests/`.

## Order of work, decided 2026-07-29

The aim is to close briefs faster than new ones open, so the order is chosen for *closure* rather than
for value alone.

1. **07 — export provenance** (`07_export-provenance-brief.md`). Small, self-contained, and on a clock:
   the EU AI Act Article 50(2) marking obligation bites 2026-12-02 for systems already on the market.
   Next.
2. **03 §D — GC UX & navigation** (`03_librarian-content-parts-brief.md`, the one remaining checkpoint).
   GUI work: manual "Clean up & save" with dry-run preview, and bidirectional tool-call↔response
   navigation links. Closing it finishes the 01–06 run — the first half of the marathon — and wires the
   sidecar sweep, which currently has no non-test caller, so attachments accumulate on disk.
3. **09 — retrieval query side** (`09_retrieval-query-side-brief.md`). Deliberately after 10 rather than
   before it: 10 built the infrastructure that makes retrieval quality visible and actionable, and 09's
   lever 1 plus the confidence signal are what the grounding marker has been blocked on since it shipped.

Not in this list and still open: the tool-budget split (`TODO.md`, Librarian / urgent), which is a
follow-up to 10 rather than a brief of its own.

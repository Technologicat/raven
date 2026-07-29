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

## After those three

Not ordered, and the ordering is a real question rather than a formality — they are wanted for three
different reasons:

- **04 — MCP client** (`04_librarian-mcp-client-brief.md`). The most practical going forward: it is how
  external tools arrive, which is what an agentic per-document pass wants, and it is tool *supply* rather
  than a plugin system (see the scriptable-scaffold item in `TODO.md`).
- **05 — lorebook** (`05_librarian-lorebook-brief.md`). A UX improvement.
- **06 — memory / Hindsight standup** (`06_hindsight-standup-brief.md`). The most interesting from a
  research standpoint, and the one closest to the co-researcher line in the ECCOMAS talk — memory is what
  separates a chat frontend from a collaborator that accumulates context about the work. Adjacent reading:
  Seth Herd on language model cognitive architectures.

Recording the reasons rather than a rank, because "most interesting" and "most useful next" are different
orderings and both are legitimate. The three above are ordered for closure; these are not yet ordered at
all.

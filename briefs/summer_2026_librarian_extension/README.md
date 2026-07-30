# Summer 2026 Librarian extension

Implementation briefs for the Librarian work of summer 2026. Completed briefs move to `done/`; live
probes shared across briefs live in `manual_tests/`.

## Order of work, decided 2026-07-29

The aim is to close briefs faster than new ones open, so the order is chosen for *closure* rather than
for value alone.

1. **07 — export provenance** (`07_export-provenance-brief.md`). Small, self-contained, and on a clock:
   the EU AI Act Article 50(2) marking obligation bites 2026-12-02 for systems already on the market.
   **Done 2026-07-29** (archived to `done/`).
2. **03 §D — GC UX & navigation** (`03_librarian-content-parts-brief.md`, the one remaining checkpoint).
   GUI work: manual "Clean up & save" with dry-run preview, and bidirectional tool-call↔response
   navigation links. Closing it finishes the 01–06 run — the first half of the marathon — and wires the
   sidecar sweep, which currently has no non-test caller, so attachments accumulate on disk.
   **Done 2026-07-30** (archived to `done/`). The sidecar sweep now has its caller, so the release blocker
   below is cleared; 09 is what 0.2.8 is still waiting on.
3. **09 — retrieval query side** (`09_retrieval-query-side-brief.md`). Deliberately after 10 rather than
   before it: 10 built the infrastructure that makes retrieval quality visible and actionable, and 09's
   lever 1 plus the confidence signal are what the grounding marker has been blocked on since it shipped.

Not in this list and still open: the tool-budget split (`TODO.md`, Librarian / urgent), which is a
follow-up to 10 rather than a brief of its own.

## Cut 0.2.8 after 09 — and nothing else before it

**Decided 2026-07-29.** The release goes out once 03 §D and 09 are done, and no feature lands between
09 and the tag.

The reason to wait rather than ship at 07: both remaining items leave a user-visible half-feature.
Sidecar attachments have no way to remove strays until §D wires the sweep — `prune_unreferenced_sidecars`
still has no non-test caller, so a released 0.2.8 would grow attachment files with no way to reclaim
them. And the document database needs 09's retrieval improvements to be worth the name. Shipping either
half-done means a release whose headline features come with a caveat.

The reason not to wait *longer*: 0.2.8 has already outgrown 0.2.7 by half (4400 words of changelog
against 2974, 58 entries) while 0.2.7 shipped in April. The wall-of-text changelog is the symptom; the
cadence is the cause. So §D and 09 are the boundary, not a moving one.

Release prep, when the time comes (see also the `release` and `changelog` skills):

- **Audit the `Fixed` entries.** The house rule is *"was the broken behavior present in the most recent
  tagged release?"* — a bug born and buried inside the 0.2.8 window never reached a user and does not
  belong in the changelog. With this much development volume, a few almost certainly slipped through
  review.
- **Regroup the changelog by component** within Added / Changed / Fixed. Every entry currently opens with
  `*Raven-<app>*:` — 24 of 58 with `*Raven-librarian*:` alone — which is a heading doing prose duty. This
  is a Raven-local rule, not a fleet one: Raven is the only project in the fleet shipping many separate
  user-facing apps, and elsewhere the prefix would be noise.

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
orderings and both are legitimate.

**05 goes first of the three** (decided 2026-07-29), on size: it is likely smaller than either 04 or 06,
and the point of this whole ordering is to close briefs faster than new ones open. 04 and 06 remain
unordered against each other.

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

**Amended 2026-07-30: one feature does land** — large `webfetch` results become attachments rather than
being dumped inline (`TODO_DEFERRED.md`). Admitted under the same argument the rule was written to serve
rather than against it: 0.2.8's headline is attachments, and a release that stores a user's PDF as a tidy
chip while the AI's own fetches bury the chat log under dozens of screens ships the feature half-applied.
That is the "user-visible half-feature" the freeze exists to prevent, so excluding it would honour the
letter and lose the point. The chat-view scrolling work (keys, smooth scrolling, end-of-scroll feedback,
and the streaming autoscroll fix) is the sibling of this and is a *defect* set rather than a feature, so it
was never covered by the freeze.

**Also admitted 2026-07-30, as defects rather than features** (the freeze covers features): the chat-view
scrolling set — keys, smooth scrolling, end-of-scroll feedback, and the streaming autoscroll that dragged a
reader back to the bottom on every chunk — plus two "crying wolf" log-noise items, the context-prefill
strict-template warning and `setup_font_ranges`' DeprecationWarnings. The last two are one defect in two
costumes: output that looks alarming, means nothing, and fires every run, which teaches the reader to skim
past logs that will one day matter.

**Considered and left out**, with reasons, so they are not re-litigated: the Markdown renderer set (four
separate defects inside vendored code — a project, not a fix), the dropped-character render bug
(unreproducible on demand, so there is nothing to test a fix against), TTS reading arXiv IDs digit by digit
(tedious rather than broken), and FileDialog's OS drag-and-drop and image previews. The FileDialog pair are
the ones that bite hardest in practice (Juha) but they are *enhancements*, and the demo they would serve can
ride a later release — 0.2.9 or beyond.

The rule still holds for everything else: nothing further is admitted without an argument of the same
shape, made explicitly here.

### What is left, as of end of 2026-07-30

Done and confirmed: **03 §D** (archived), and the **streaming autoscroll** half of the scrolling set — the view
now follows a reply for a reader at the end and leaves a scrolled-away reader alone, verified live over a
thinking block and a multi-screenful `webfetch` answer. Details and the three faults it took are in
`TODO_DEFERRED.md`, "Chat view scroll position jumps back down while the model is writing".

Still open, in the order they are expected to be done:

1. **Chat view scrolling, the rest** — smooth scrolling, the end-of-scroll flasher, Home/End/PageUp/PageDown,
   and the jump-to-latest pill. Traps already scouted; see `TODO_DEFERRED.md`, "Chat view scrolling: keys,
   smoothness, and end-of-scroll feedback".
2. **The two log-noise fixes** — the context-prefill strict-template warning, and `setup_font_ranges`'
   DeprecationWarnings.
3. **Large `webfetch` results become attachments** — the one feature admitted past the freeze.
4. **Brief 09** — the retrieval query side, and the last blocker.

Then the release-prep checklist below.

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

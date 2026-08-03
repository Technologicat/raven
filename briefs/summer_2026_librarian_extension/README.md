# Summer 2026 Librarian extension

Implementation briefs for the Librarian work of summer 2026. Completed briefs move to `done/`; live
probes shared across briefs live in `manual_tests/`.

**Closed so far**, in `done/`: 01 (webfetch), 02 (LM Studio compat), 03 (content parts, including §D),
07 (export disclosure), 08 (context injects), 10 (RAG tool surface). **09 is the one in progress**, and the
last thing v0.2.8 is waiting on.

## Order of work, decided 2026-07-29

The aim is to close briefs faster than new ones open, so the order is chosen for *closure* rather than
for value alone.

1. **07 — export disclosure** (`07_export-disclosure-brief.md`). Small, self-contained, and on a clock:
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

### What is left, as of 2026-08-03

**This section is the runway to the tag.** If something has to be done before 0.2.8 ships, it is listed here,
including the items that arrived without a brief of their own. Anything not listed does not gate the release.

Done and confirmed: **03 §D** (archived), and the **streaming autoscroll** half of the scrolling set — the view
now follows a reply for a reader at the end and leaves a scrolled-away reader alone, verified live over a
thinking block and a multi-screenful `webfetch` answer. Details and the three faults it took are in
`TODO_DEFERRED.md`, "Chat view scroll position jumps back down while the model is writing".

Still open, in the order they are expected to be done:

1. ~~**Chat view scrolling, the rest**~~ — **done 2026-08-03**, live-tested. Smooth scrolling, the
   reader-driven keys (Home/End/PageUp/PageDown, and arrows by five lines), the end-of-scroll flasher and
   the jump-to-latest pill all landed, with the `SmoothScrolling.start()` retarget fix first as planned.
   The CHANGELOG entry covers the feature as a whole.
   - Three bugs surfaced *by* the live testing rather than by the suite, each needing its own diagnosis: the
     follow-tail decision reading the animation's current position rather than its destination (an arrow key
     "sometimes" worked); a sample-then-act race that discarded roughly one keypress in fifteen; and the
     same race one level deeper, inside `scroll_view`'s settle wait, which the first fix moved rather than
     closed. Worth knowing before touching this code again — the window between deciding and scrolling is
     ~100 ms and the reader's keyboard is live throughout.
   - Two loose ends, both filed in `TODO_DEFERRED.md` and to be **done together**, since they land in the
     same file: the scrollbar-drag creep (ImGui holds a *fraction* of the content, so holding the thumb
     while a reply streams slides you down), and `SmoothScrolling` committing during construction.
2. **The three log-noise fixes** — the context-prefill strict-template warning, `setup_font_ranges`'
   DeprecationWarnings, and the avatar's emotion autoreset, which logs at INFO every three seconds forever
   and reports a change to neutral when the emotion is already neutral. The third was added 2026-08-03, on
   noticing it while reading a debug log during the chat-scrolling live test; all three are the same defect
   in different costumes, so they are one job.
3. **Large `webfetch` results become attachments** — the one feature admitted past the freeze.
4. **Tool budget: error out informatively instead of withdrawing the tools.** Admitted 2026-08-03, and the
   freeze rule wants the argument stated rather than assumed. It is *defect-shaped*: withdrawing a tool
   mid-turn burns the KV cache, and a history referencing a tool no longer in the schema is off-distribution
   for the model, whereas tool *errors* are well represented in training. It is also a fix to a feature this
   very release ships (brief 10's round cap), so shipping the v1 shape would mean revisiting it in 0.2.9
   immediately. Measured basis in `investigations/tool_budget/`. A further measurement — whether it actually
   stops Qwen going into unasked deep research — is worth doing but does not gate the fix, because the
   cache-burn argument carries it alone.
5. **Brief 09** — the retrieval query side, and the last blocker. Absorbs the inject-shape decisions that were
   filed separately: document-inject offset/length, the consulted-docs list gaining offsets and a
   "previously consulted" marker, the "no sources consulted" marker, and whether the Speculation toggle still
   carries information once Docs-ON implies marking. These are all §4-shaped — 09's implementation has to
   settle them anyway, so they are scoped into it rather than tracked apart.

Then the release-prep checklist below.

**Queued immediately after 09, before anything else starts:** the turn/round terminology sweep. Deferred only
because it renames things 09 is about to edit, and doing it first would mean rewriting the brief. Confused
terminology makes for confused developers, so it does not wait longer than that. The decision is settled —
*turn* = one participant's contribution including the whole tool loop, *round* = one iteration of the agent
loop within a turn, *exchange* = user turn + assistant turn — and it is the code that moves, not the briefs.

**Not gating this release, recorded so they are not mistaken for gates:**

- The **8/3 DPG margin audit** and the **`dpg-notes.md` skill**. Both in `TODO_DEFERRED.md`; both small; both
  fair game whenever there is a gap, but neither is user-visible.
- **Visualizer defects surfaced by the two studies now using it.** Those studies are live dogfooding of a
  component with zero tests, so they will find things. They go to `TODO_DEFERRED.md`. 0.2.8 is a Librarian
  release, and a Visualizer bug is not an argument for delaying it.
- **Brief 11** (importer rework), **brief 12** (derived artifact store) and **brief 14** (chat search) are
  0.2.9 work.
- **Brief 13** — corpus scopes and the unified DB, plus the corpus TOC that is blocked on them. Drafted
  2026-08-03 as a rough draft rather than a design: it carries the decisions already made, the proposals not
  yet agreed, and the retractions with their reasons, so the design session starts from an agenda instead of
  a pile. The design comes back from a claude.ai session as a filled-in version of it. Realistically after
  Researchers' Night (26 September 2026).

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

## Also in this folder

11 and 12 are in the same numbered sequence as the rest; what they sit outside is the *ordering* decided for
04/05/06 above, which was a ranking of the Librarian run against itself. The last two carry no number at all.
Listed so the folder's contents are legible from its README rather than from `ls`.

- **11 — Visualizer importer rework** (`11_visualizer-importer-rework-brief.md`). Nomic migration, PCA
  preprocessing, cosine-to-medoid outlier assignment, Procrustes alignment. Predates the sprint and had been
  forgotten at the top level; moved in and numbered 2026-08-03. Its item 1 carries a **fork that needs
  deciding** — `nomic-embed-text-v1.5` buys a shared image-text space, `v2-moe` buys multilingual, and no
  v2-aligned vision encoder appears to exist. That decision reaches brief 12.
- **12 — derived artifact store** (`12_derived-artifact-store-brief.md`). One keying and regeneration
  mechanism for everything computed *from* a source artifact — extracted text, OCR, thumbnails, burst `.bib`
  records, embeddings — with separate stores for the chat and document-DB lifecycles. **v0.2.9 work; it does
  not gate v0.2.8**, and the webfetch attachment work deliberately lands ahead of it rather than waiting.
- **14 — search within the chat log** (`14_chat-search-brief.md`). **v0.2.9**, and the freeze is why: it is a
  feature, and unlike the webfetch attachment work it is not half of anything 0.2.8 already ships. The match
  unit is the **message**, which is what keeps v1 cheap — it sidesteps in-text highlighting, whose Visualizer
  implementation rebuilds the whole panel and therefore does not transfer to an incrementally-built chat log.
  v1 reuses the scroll-and-flash the tool-call navigation links already do. v2 adds in-text highlighting in
  completed messages only, with the message still the unit that next/previous jumps to.
- **13 — corpus scopes and unified DB** (`13_corpus-scopes-and-unified-db-brief.md`). A **draft**, unlike the
  rest: it holds the 2026-08-01 design-session material with its `[D]`/`[N]`/`[P]`/`[X]` provenance markers
  intact, so a later reader can tell settled from proposed. Prerequisite for the corpus TOC and for most of
  what the corpus-interrogation sketch wants. Brief 12 deliberately does not depend on it.
- **`atmospheric-dust.md`** and **`crt-display.md`**. Avatar postprocessor work, queued for Researchers'
  Night. Moved in 2026-08-03; neither is done.

**The folder name still mostly holds.** The avatar subsystem exists for Librarian — it is Librarian's UX — so
the two avatar briefs are Librarian work under a wider reading rather than strays. 12 is cross-cutting, and
touches Librarian at the chat-sidecar end. Only **11 sits genuinely outside**, being Visualizer. Not enough to
force a rename; noted so that a later reader wondering why an importer brief lives here finds the answer
instead of re-deriving the question.

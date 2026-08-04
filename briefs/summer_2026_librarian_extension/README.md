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

**Admitted 2026-08-04, riding on the webfetch item rather than on their own weight** — two small pieces of the
same surface, each needing its own argument since neither is the half-feature the webfetch item was:

- **Attachments open on click**, not only from the button below them. This one is *discoverability polish*,
  and the weaker case of the two: the button already works, is labelled and has a tooltip. What carries it is
  that a thumbnail plainly looks clickable, so clicking it and getting nothing is a papercut paid every time,
  on the release's headline feature — and that it was verified in a GUI session the webfetch work required
  anyway, so it cost no extra round of live testing. The hover cue a *text* chip needs is not in it; that is
  filed.
- **A `fetch_document` result gets the same two handles** (open the file, reveal the documents folder). The
  argument is uniformity of a category the release itself creates: after the webfetch change, "a document the
  AI fetched" becomes a thing the chat log shows with handles on it, and having that be true of web fetches
  but not knowledge-base fetches invents a distinction the user has no reason to expect. It is *not* an
  attachment and deliberately does not become one — the file is already the user's, in the documents folder,
  and copying it into the sidecar store would archive a second copy of something that cannot go away. So the
  affordance matches while the backing store does not, which is the whole design.

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
2. ~~**The three log-noise fixes**~~ — **done 2026-08-04.** The context-prefill strict-template warning, and
   the avatar's emotion autoreset, both became conditional on there being something to report: the template
   check now *describes* rather than logs, and its result is emitted only if the backend actually refuses the
   request; the autoreset speaks only when it returns the avatar from an expression. `setup_font_ranges` was
   deleted outright rather than gated — probing the wheels showed `add_font_range` live through DPG 2.2 and a
   no-op from 2.3, so the floor moved to `dearpygui>=2.3` and Raven declares no ranges anywhere. Two
   surprises worth knowing: the deprecation also fired from the *icon* font paths in `guiutils` and
   `cherrypick`, which the item had not counted, so "four call sites" was really seven; and the fix is
   pinned by `raven/common/gui/tests/test_fontsetup.py`, which asserts the *absence* of warnings — the
   failure mode here is silent, since an app that warns on every start still works.
   - **The template diagnosis was wired to the wrong refusal path**, and only a live test found it. It went
     on the HTTP-error branch, which is the obvious one and the one a mocked test reaches for — but LM
     Studio answers a template rejection with **HTTP 200 and an SSE error event mid-stream**, so the
     diagnosis never fired in the single case it exists for. Verified against LM Studio serving Qwen3.5,
     whose template raises `No user query found in messages.` on the `[system, greeting]` shape Raven's own
     idle prefill builds. Both paths now report, and `TestRefusalCarriesTheTemplateDiagnosis` covers both —
     the lesson being that "the backend refused" is not one code path, and a test that mocks only the
     tidy one certifies nothing.
3. ~~**Large `webfetch` results become attachments**~~ — **done 2026-08-04**, live-tested. A tool result that
   *declares itself a document* goes to a content-addressed sidecar and leaves an excerpt plus a chip;
   declared rather than matched on the tool's name, which is what keeps `websearch` inline at any length.
   Verified end to end against a running backend: 34079 characters stored, 800 shown, context readout
   8% → 15% confirming the whole document is still counted and sent.
   - **It pulled a polish cascade behind it, and that was the right call each time**, but it is why this was
     a whole day. The chat log needed one *uniform* handle for "a document the AI fetched", so a
     `fetch_document` result got the same two buttons (pointing at the user's own file, not a copy of it);
     attachments became click-to-open; the controls moved into a left gutter, which is also where the
     jump-back link belongs, since it is where the view scrolls to when its counterpart is clicked. Each was
     small; together they were most of the day. Recorded because the *next* release will do this too.
   - **Two bugs the suite could not have caught**, both from the same reasoning error — asking what a
     category contains rather than looking. `websearch` emits one text part per result, and the gutter path
     would have rendered the first and dropped the rest; and Ctrl+Enter cannot be caught by a global key
     handler at all, because ImGui commits *and unfocuses* the field first. The second took three attempts,
     two of them confident and wrong, before reading `add_input_text`'s parameter list showed
     `ctrl_enter_for_new_line` sitting there — the toolkit had owned this the whole time.
   - **Two consequences of shipping it are filed rather than fixed**, and neither gates the tag: a fetched
     page is now budgeted as a *user attachment* (no per-fetch ceiling) though the config says a speculative
     fetch should have one, and a long `fetch_document` result still fills the log, wanting a collapsible
     rendering rather than the attachment treatment.
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

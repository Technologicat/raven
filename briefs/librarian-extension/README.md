# Summer 2026 Librarian extension

*A sprint README is the sprint's decision log, not an introduction to it: what was scheduled and in which
order, what was cut and on what argument, what was learned while building each item. Most of it is here
because it produced no diff and would otherwise be re-decided from scratch. Read it when taking stock or
picking the next item; it is not a document anyone reads through.*

Implementation briefs for the Librarian work of summer 2026. Completed briefs move to `done/`; live
probes shared across briefs live in `manual_tests/`.

**Closed so far**, in `done/`: 01 (webfetch), 02 (LM Studio compat), 03 (content parts, including §D),
07 (export disclosure), 08 (context injects), 10 (RAG tool surface), and the unnumbered
`self-sizing-tooltip` — a `raven/common/gui` component so a tooltip whose caption changes stops rendering
one frame at the wrong size. It was filed here because Librarian was the first consumer, not because it
belonged to this sprint's scope; Visualizer took it too. Measurements in `investigations/dpg-autosize/`.
**09 is the one in progress**, and the last thing v0.2.8 is waiting on.

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
3. ~~**09 — retrieval query side**~~ — **closed as an experiment set 2026-08-06, archived to `done/`.**
   It was deliberately scheduled after 10, since 10 built the infrastructure that makes retrieval quality
   visible and actionable and 09's lever 1 was what the grounding marker had been blocked on. That
   consumer is now served: the marker keys on whether documents are in play, and the Speculation toggle
   that used to gate it is gone. The brief's own status block carries what shipped, what was refuted, and
   what moved after it was written.

Not in this list, and **largely overtaken rather than open**: the tool-budget split (`TODO.md`, Librarian /
urgent), a follow-up to 10 rather than a brief of its own. What made it urgent was that
`max_tool_call_rounds = 5` was exhausted by gathering alone, so nine of fourteen sampled turns that reached
the cap ended with an empty assistant message (p = 0.013). The cap went to **20** on 2026-08-04, set from
where models actually stop — the same model rephrases nine or ten times and then gives up unprompted — so
the failure that gave the item its priority should no longer be reachable. The *split* itself (a small cap
for searches, a larger allowance for fetches) is still unbuilt and now looks unjustified: the runaway it
was designed against did not survive measurement. `TODO.md` carries both halves.

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
`investigations/follow-tail-drift/README.md`, under "Prior episode".

**Next session starts here** (agreed 2026-08-04, item 4 done that evening): settle whether item 3's two
defect-shaped consequences are release blockers — the note under item 3 has the argument and the one
non-obvious cost — then straight into brief 09, which is now the only long task left on the runway and
therefore where the whole schedule risk sits.

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
   - **Two consequences were filed alongside it. Settled 2026-08-05: one of them never existed.**
     - *A long `fetch_document` result fills the log* — **already fixed when the note was written**, by the
       polish cascade above. A result past `tool_result_attachment_threshold` renders collapsed to an excerpt
       with a toggle, and the knowledge-base document gets its own chip with open-file and open-folder
       actions. The collapsible landed at 15:47 and the note claiming it was missing at 15:55; it was carried
       forward from the earlier filing without re-reading what had just landed. Worth knowing as a failure
       mode: a note written from a note, rather than from the tree, and the eight-minute gap made it
       invisible. Deferred item removed.
     - *A fetched page is budgeted as a user attachment* — **real, and fixed before the tag.** `webfetch`
       applies no budget at fetch time and inherits `fit_attachments_to_context`, which deliberately carries
       no per-document ceiling because an attachment is the user saying *read this*. A fetch is the opposite
       case, and `docs_fetch_max_fraction_of_context`'s own comment says so. Not a regression — `webfetch`
       does not exist in v0.2.7 at all — but the policy is now stated in one place and contradicted in
       another. The v1 shape and the v2 it defers are in `TODO_DEFERRED.md`, "A fetched web page is budgeted
       as a user attachment, not as a speculative fetch".
4. ~~**Tool budget: error out informatively instead of withdrawing the tools.**~~ — **done 2026-08-04**, as
   scoped and without surprises. Past `max_tool_call_rounds` the tool schema stays put and a call is answered
   with an error result saying the budget is spent; the "no more calls" system notice now fires on the same
   round, one invocation *before* the doomed call rather than after it. Admitted on the argument that it is
   *defect-shaped* — withdrawing a tool mid-turn burns the KV cache, a history referencing a tool no longer
   in the schema is off-distribution for the model, and it is a fix to a feature this very release ships
   (brief 10's round cap), so the v1 shape would have been revisited in 0.2.9 immediately.
   - **Withdrawal did not go away; it demoted.** It is the terminator of last resort, after
     `max_tool_call_refusal_rounds` (default 1). This is not a compromise but the shape of the problem: a
     refusal cannot guarantee the model stops asking, and the alternative terminator — breaking out of the
     loop — leaves the turn ending on a tool result, which reads as a paused agent loop and draws yet another
     call. Setting the knob to 0 reproduces the v1 behavior exactly, which is what makes it a knob.
   - **It does not address the empty replies, and must not be recorded as having done so.** The model still
     runs out of budget at the same round; the measurement in `investigations/tool_budget/` stands, and the
     larger-budget item in `TODO.md` is still the fix with the evidence behind it. Whether this stops Qwen
     going into unasked deep research is unmeasured.
   - Adjacent defect found and fixed on the way, in its own commit: every malformed-tool-call-request path in
     `perform_tool_calls` built its error report and then raised `TypeError` delivering it, so none of the
     five had ever reached a model. Shipped that way since 0.2.7 at least — the paths are reached only when a
     backend garbles a `tool_calls` entry, which ours do not.
5. ~~**Brief 09** — the retrieval query side, and the last blocker.~~ **Closed 2026-08-06 as an experiment
   set**, which is a different thing from finished: it shipped `docs_num_results = 50` (+10 points of
   known-item recall on a 12k corpus), a cap on merged result length, query-time fusion parameters and a
   handful of fixes the harness surfaced — and it refuted rather more than it shipped, reranking included.
   The brief's own status section says what landed; `investigations/retrieval/REPORT.md` carries the
   findings and the ready-to-run experiments. The remaining levers want clustering and a summary layer, so
   they are post-Researchers' Night.
   - **The absorbed inject-shape decisions did not close with it.** Document-inject offset/length, the
     consulted-docs list gaining offsets and a "previously consulted" marker, the "no sources consulted"
     marker, and whether the Speculation toggle still carries information once Docs-ON implies marking were
     scoped into 09 on the reasoning that its implementation would have to settle them. 09 stopped being
     that implementation, so they were orphaned.
     - **Settled 2026-08-07, and recorded in `design-session-2026-08-03.md` §4 rather than here** —
       in place, against the items themselves, so a reader following either route lands on the decision.
       In short: the Speculation toggle goes and one Documents toggle remains, with the marker made honest
       about *why* it is silent rather than a second marker being added; the offsets defer into brief 12,
       whose D1 is the coordinate space they would be expressed in; and the "previously consulted" wording
       and inject ordering split off as the small half that waits for nothing.

Then the release-prep checklist below.

~~**Queued immediately after 09, before anything else starts:** the turn/round terminology sweep.~~
**Done 2026-08-07**, in that slot, immediately after the 0.2.8 tag. *turn* = one participant's contribution
including the whole tool loop, *round* = one iteration of the agent loop within a turn, *exchange* = user turn
+ assistant turn; the code moved and the briefs did not. The convention is now stated in
`raven/librarian/CLAUDE.md`, which outlives this folder; the sweep's own account is in
`design-session-2026-08-03.md` §"Turn / round terminology".

**Not gating this release, recorded so they are not mistaken for gates:**

- The **8/3 DPG margin audit**, still in `TODO_DEFERRED.md`: small, fair game whenever there is a gap, and not
  user-visible. Its companion here, the **`dpg-notes.md` skill**, shipped 2026-08-13 as
  `.claude/skills/dpg/` — a router into the notes rather than a copy of them, with a checker beside it.
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

- ~~**Audit the `Fixed` entries.**~~ **Done 2026-08-07.** The house rule is *"was the broken behavior
  present in the most recent tagged release?"* Two entries failed it and were removed, both for the same
  reason — they described features that did not exist in 0.2.7, so no user upgrading could have met the
  bug. The **UTF-16 surrogate repair** fixes text from pypdf, and 0.2.7 ingested only `.txt`/`.md`/`.rst`/
  `.org`/`.bib`/`.tex` with no PDF path and no `docextract` module. The **export think-marker fix** repairs
  a boundary that 0.2.7 got for free: it spoke only to oobabooga, which sent the whole reply in one channel
  with the tags in it, and the June 2026 `reasoning_content` migration is what removed them.
- ~~**Regroup the changelog by component**~~ — **already done**; the note described the pre-regrouping
  state. Verified 2026-08-07: canonical component order in all three sections, `*Constellation-wide*` last
  in each. One duplicated `*Raven-arxiv-download*` header was folded on the way past.

### 0.2.8 shipped, 2026-08-07

Tagged `v0.2.8` on `77a6d1c`, with CI green on that exact commit, and released on GitHub as *Raven 0.2.8*
with the changelog section as its body. `main` is back on `0.2.9-dev` with a fresh changelog stub.

Landed on release day and worth knowing when reading the diff: the Speculation toggle removal, the
`get_current_time` tool, the tool-registry extraction to module level, the export think-markers, the
server-side logging sweep, `raven-fixbib`, and the BibTeX brace recovery in the importer.

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

**Superseded 2026-08-07 (Juha), by a deadline rather than by a better argument.** With Researchers' Night
on 26 September, the question stopped being "which of these closes fastest" and became "which of these the
open house needs":

- **06 — Hindsight memory: after Researchers' Night, and the reason is the venue rather than the
  schedule.** Memory pays off over a long-running relationship with the system, and an open house is the
  opposite of that: visitors arrive, talk to it once, and leave. Its effects are invisible to that audience
  *however finished it is*, so shipping it early buys nothing the exhibit can show. This is not a demotion on
  merit — it remains the most interesting of the three, and it is what the **team** wants in the autumn, when
  the interaction actually is long-term and accumulated context about the work is the whole point.

  Worth generalizing, because it will decide other calls the same way: **a feature whose value accrues over
  time cannot be demonstrated in an encounter that does not.** Ask what a visitor could observe in one
  sitting before scheduling anything into the exhibit run.
- **04 and 05 — MCP client and lorebook: question marks.** Both genuinely useful, neither
  open-house-critical. They go in if the Researchers' Night work lands with room to spare, which is not the
  way to bet.

So all three of these are now *behind* `../researchers-night/`, and this sprint's remaining briefs are not
what to pick up next. Read that folder's README instead.

**Amended 2026-08-05, and it bends the closure-first rule rather than fitting it.** A deferred-TODO triage
session opened two more briefs (15 and 16) and moved one of them ahead of 05. Recorded here because the
ordering above is the thing being changed, and because the argument is not the one this section was built on.

**15 — scripting surface over the scaffold** (`15_headless-agent-driver-brief.md`) **goes ahead of 05.** The
ordering above ranks by *closure* — smallest first, so briefs shut faster than they open. 15 does not win on
that axis and does not claim to. It wins on **timing**: roughly seven weeks of investigation-heavy work sit
ahead of it (the markdown renderer set, the turn-sequencing race, the auto-RAG-as-mistake bug, table layout,
equation scoping), and every one of them wants a scriptable driver. Landing it first amortizes it across all
of them, whereas landing it in October amortizes it across nothing. It also sits immediately before three
consecutive agent-loop features — 05, 04, 06 — whose behaviour is what one would most want to script.

That is a different quantity from closure rate, and both are legitimate; this is the case where they point
opposite ways and timing wins. Noted rather than smoothed over, because the closure-first principle is
otherwise still the rule.

**15 supersedes three items in `TODO_DEFERRED.md`** — the headless-mode item, lazy `api.initialize`, and the
`ai_turn` callback bundle — and consolidates the design already written under `TODO.md`'s Librarian → Core
features. The callback bundle turns out **not** to be a prerequisite: the scripting surface takes no
callbacks at all, so the bundle stays an independent GUI-side cleanup.

**16 — chat graph view** (`16_chat-graph-view-brief.md`) **is added, for Researchers' Night, and takes
precedence over `atmospheric-dust.md`** (decided 2026-08-05: the graph adds more value; dust lands only if
time remains). The dependency runs dust → crt, so dropping dust leaves both `crt` and 16 intact.

**Brief 17 is reserved but unwritten** — a per-document LLM pass with retry, cache, resume and progress, cut
out of 15 because it has three users of its own (`raven-pdf2bib`, `rag_live_corpus`'s persistence layer, and
the corpus-interrogation sketch's map stage) and is a batch-execution primitive rather than a scripting
surface. If it stays unwritten, 17 may end up being something else and 15's reference to it will need
chasing.

**Honest accounting on the folder's own principle**: this session closed no briefs and opened two. The
dehydration pass is not failing so much as being outvoted — "close briefs faster than new ones open" is a
count target, and count targets lose to depth processes, since every brief closed at the quality actually
wanted spawns the observations that become the next brief. The count was never the quantity under control.

## Also in this folder

11 and 12 are in the same numbered sequence as the rest; what they sit outside is the *ordering* decided for
04/05/06 above, which was a ranking of the Librarian run against itself. The last two carry no number at all.
**15 is the exception**: it is numbered *and* ranked into that ordering, ahead of 05, for the reasons in the
amendment above. Listed so the folder's contents are legible from its README rather than from `ls`.

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
- **15 — scripting surface over the scaffold** (`15_headless-agent-driver-brief.md`). **Landed 2026-08-12**;
  closed into `researchers-night/done/`. Was first in that queue, ahead of 05 — see the amendment above for
  why it ranked on timing rather than closure. Two
  entry points, not one: build the turn's prompt and hand it back (no backend), and run the turn and report
  what happened (a result record, not a node id). Part 0 is lazy `api.initialize`, which is what makes
  `scaffold` importable without the full dep stack and removes `test_scaffold.py`'s `importorskip`. The
  scripted backend and the per-document pass are both explicitly cut.
- **16 — chat graph view** (`16_chat-graph-view-brief.md`). **Researchers' Night**, and ahead of
  `atmospheric-dust.md`. Framed explanatory before navigational: the exhibit's job is making "an LLM is a
  multiverse generator" visible, so rerolling from the graph and exploring existing branches are the two
  carrying interactions. Built via `XDotWidget.set_graph` in memory rather than by emitting xdot — which
  means step zero is that `set_graph` currently has **no callers and no tests**, a widget defect owed
  regardless of this brief. Fragment search is v2, scoped with 14.
- **`atmospheric-dust.md`** and **`crt-display.md`**. Avatar postprocessor work, queued for Researchers'
  Night. Moved in 2026-08-03; neither is done. **Dust is now behind brief 16** (2026-08-05) and lands only
  if time remains — it is the schedule's slack, and safely so: `atmospheric-dust.md` takes its priority-band
  scheme from `crt-display.md` §0, so nothing points back at dust and dropping it leaves `crt` intact.

**The folder name still mostly holds.** The avatar subsystem exists for Librarian — it is Librarian's UX — so
the two avatar briefs are Librarian work under a wider reading rather than strays. 12 is cross-cutting, and
touches Librarian at the chat-sidecar end. Only **11 sits genuinely outside**, being Visualizer. Not enough to
force a rename; noted so that a later reader wondering why an importer brief lives here finds the answer
instead of re-deriving the question.

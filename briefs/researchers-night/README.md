# Researchers' Night sprint

*A sprint README is the sprint's decision log, not an introduction to it: what was scheduled and in which
order, what was cut and on what argument, what was learned while building each item. Most of it is here
because it produced no diff and would otherwise be re-decided from scratch. Read it when taking stock or
picking the next item; it is not a document anyone reads through.*

**Deadline: 26 September 2026.** That is what this folder is named for, and it is the honest name — the
contents are not one scope. They are the briefs that were conceived during the Librarian run but scheduled
after it, and what they share is the deadline that orders them, not a subsystem.

Split out of `librarian-extension/` on 2026-08-07, at the 10/11 boundary. Worth knowing that the boundary is
chronological rather than topical: the numbers ran in the order the briefs were *written*, so 14 and 15 are
Librarian features and 11 is Visualizer, sitting side by side here because of when they were conceived.

## What's here

| Brief | What | Status |
|---|---|---|
| `markdown-block-rendering-brief.md` | Band-2 item 12: block-level Markdown in the chat view | Researchers' Night. Steps 1 and 5 landed 2026-08-25; what remains is behind the single-newline split, which is also what fenced code and multi-line lists are waiting on |
| `16_chat-graph-view-brief.md` | The chat tree as a graph, for the exhibit | Researchers' Night, and **next**. Explanatory before navigational — the job is making "an LLM is a multiverse generator" visible. **Step zero is done** (2026-08-25): a hand-built chat-shaped `Graph` renders through `XDotWidget.set_graph` with no GraphViz and no xdot in the path, and `test_widget.py` covers it. So this starts from the feature work, not from proving the door opens |
| `12_derived-artifact-store-brief.md` | One keying and regeneration mechanism for everything computed *from* a source artifact | v0.2.9. Does not depend on 13 |
| `13_corpus-scopes-and-unified-db-brief.md` | Corpus scopes and the unified DB | **A draft, not a design** — it holds the 2026-08-01 session material with its `[D]`/`[N]`/`[P]`/`[X]` provenance markers intact, so a reader can tell settled from proposed. Realistically after Researchers' Night |
| `11_visualizer-importer-rework-brief.md` | Nomic migration, ~~PCA preprocessing~~, ~~cosine-to-medoid outlier assignment~~, Procrustes alignment, and **clustering in high-D**. **Items 2, 3 and 5 were measured on 2026-09-01 and the brief now points at `investigations/highdim-clustering/README.md`, which is the specification** — 2 and 3 are measured harmful and must not be built as written, 5 is settled and ready to implement (agglomerative, not HDBSCAN) | Its item 1 carries **a fork that needs deciding** — `nomic-embed-text-v1.5` buys a shared image-text space, `v2-moe` buys multilingual, and no v2-aligned vision encoder appears to exist. That decision reaches brief 12 |
| `aokk-corpus-scope-classification-brief.md` | An LLM batch pass over the AOKK corpus, flagging records the boolean search pulled in that are not about AI agents in higher education | Filed 2026-08-31. **Next up, and the only near-term item with a date**: needed within two weeks of 2026-09-01, and its two open questions were settled that day, so it starts from a decided spec. The problem is confirmed rather than suspected: `"conversational agent"` caught a child-helpline paper, and `"learning assistant"` matches 40 records across *two senses* — the AI tool, and the STEM term of art for a human undergraduate Learning Assistant. Carries a warning about reusing `agent-batch-classification`'s confidence-based escalation, whose known failure mode applies directly |
| `14_chat-search-brief.md` | Search within the chat log | v0.2.9. The match unit is the **message**, which is what keeps v1 cheap — it sidesteps in-text highlighting, whose Visualizer implementation rebuilds the whole panel and so does not transfer to an incrementally-built chat log |

## Closed

| Brief | What | Landed |
|---|---|---|
| `done/15_headless-agent-driver-brief.md` | A scripting surface over the scaffold — `raven.librarian.agent`, plus the backend-status work and the per-variety system prompt storage that came out of it | 2026-08-12, v0.2.9 |
| `done/filedialog-thumbnails-brief.md` | Image previews in the file dialog, as a toggled grid view | 2026-08-14 |
| `done/filedialog-keyboard-brief.md` | Operating the file dialog without a pointing device, and saying where the keyboard is | 2026-08-21 |
| `done/in-progress-reply-as-a-node.md` | Band-2 item 9: the reply is a node in the datastore while it streams, so an interrupted one survives | 2026-08-28 |
| `done/stt-audio-input-panel.md` | Band-2 item 10: the microphone set up where it will be used — level, automatic stop, peak hold and the device itself, as a panel rather than as config knobs | 2026-08-28. Written and closed the same day. Its closing section is worth reading for the eight faults live testing found that no desk testing would have, and for what it took to make the test stub worth anything |
| `done/crt-display.md` | Avatar postprocessor: the character drawn by a scanning electron beam | 2026-08-31, in about ninety minutes. Its §7 is the one to read: the brief was right about *what* to build and wrong about four *whys*, including one that shipped as a rendering bug. It also records the half-precision hazard that any filter drawing structure at the pixel pitch will hit |
| `done/atmospheric-dust.md` | Avatar postprocessor: light-catching motes drifting in the air | 2026-08-31, the same session as `crt`. Its closing section records a brightness formula the brief got wrong — and, more usefully, a test that passed against the very mistake it was written to reject, found by deliberately breaking the source rather than by reading the test |

The keyboard brief's last item became a constellation-wide component — `raven.common.gui.keyboardmark`, the
blue pulse that says *the keyboard is here* — so its closing section is worth reading outside the file
dialog's context. It also ends with an unbuilt piece re-homed rather than dropped:
`briefs/filedialog-navigation-history-brief.md`, unscheduled.

It raised one question it did not settle, and the analysis is worth finding: **the character card carries
character-independent text** (`setup_interaction_style` — deployment facts, conversational manner, and the
two backend facts, which are three different things sharing one block). The live queue entry for it is
`TODO_DEFERRED.md`, "Modernize the Librarian system prompt / character card"; the argument for why the block
cannot move as a unit is in the closed brief's last section.

## Ordering

**15 is done** (2026-08-12); the amortization argument below is why it went first. After that the ordering is
not settled, and the two sensible axes disagree — closure rate (smallest first, so briefs shut faster than
they open) against the exhibit deadline. 16, `crt-display` and `atmospheric-dust` are the only ones the
deadline actually binds; everything else could slip past September without anything breaking.

### Loose ends, filed 2026-09-02 — the chat graph's palette, and a first-press bug

Two loose ends from the chat-graph run, neither of them a brief. They are here rather than in
`TODO_DEFERRED.md` because that file has passed 150 items and anything short goes missing in it — and
the dehydration pass that would drain it is itself a post-sprint job.

**The chat graph's palette is settled** (2026-09-03), and what settled is better described as
**informatively colourful** than as a palette (Juha): hue carries the *role*, saturation the *branch*,
lightness the *speaker*, so the branch a reader is on is coloured and everything else greys out, and the
user/AI pair stripes it the way ruled paper stripes a page.

Two things about that, and the second is the one not to overstate:

- **Blue, green and orange are what Juha typically reaches for**, here and before, so their going
  together is familiarity rather than accident — but they are a habit and not a canon. Other triples work
  (yellow, magenta and a deep pink, say), and nothing about this view required these three. What is
  particular to it is that they are *also* doing work.
- **Colour carrying information is not a rule, it is a fit.** Colour may have a role or may simply look
  good, and the two are not in opposition — there is nothing to read into a colour that means nothing.
  Raven is a constellation of information tools, which is what makes colour-coding pay here specifically.
  So a fourth thing wanting one — bookmarks are already promised "a colour of their own" — is worth
  asking the question about, not holding to a test.

What it came out as: SYSTEM keeps the chat log's green and TOOL its orange; the conversation itself is
**blue**, hue 210, green and orange being spoken for and amber having come out brown five degrees from
the TOOL orange. The fills stay dark throughout, since the graph draws light text.

Four things came out of settling it that were not in the plan:

- **`_fill_for` had no system case at all.** SYSTEM was green only because the *branch* hue was green, so
  the moment that moved, the system prompt moved with it and this note's own claim quietly became false.
  A role→hue table now, which cannot drift that way again.
- **A tool *call* is coloured as a tool**, matching the log, where the call renders in the tool colour
  inside an otherwise ordinary assistant message. Here a box has one fill and its label *is* the call.
- **Orange needs more lightness and saturation than the others to read as orange** — dark, muted orange is
  *brown*, the one hue with its own name at that position. Measured: at the others' values it came out
  `(116, 85, 47)`.
- **And the washed-out tool box keeps the darker lightness anyway**, which is the one place a role's
  lightness depends on the branch. Brightening it with the focused box turned a warm grey that reads like
  some metal off the end of the periodic table into a paler nothing. Juha's call, and the principle with
  it: what looks good wins over the systematic derivation.

**Matched to the chat log by *vibe*, not by value** (Juha) — the same three families, chosen to read in
this medium. The log carries its colour on thin glyphs where a lightness step reads hard; this carries it
on large fills where the same step disappears. Copying the log's numbers reproduced its values and not its
effect, which is why the user/AI zebra here is wider than the log's own ratio.

The lasting lesson is about *mode* rather than colour. The structural half — which channel carries what —
behaved like ordinary building work and turned up a real bug. Only the numbers needed the other head, and
they defeated every attempt to reason about them: green read oddly, amber went brown, the first blue did
not land. There is no probe that can fail for "is this the right blue", which is exactly what the building
mode runs on. Worth splitting the look pass (item 10) the same way: settle the channels with the building
head, and come out with a short list of numbers to judge on screen in one sitting.

**The first Ctrl+Shift+O opens nothing** (2026-09-02, Juha). In `raven-librarian`: the first press flashes
the toolbar button as though the hotkey landed, and no dialog appears; every press after that works. So the
key reaches the handler and the handler runs far enough to flash — what fails is downstream, on the first
call only. The shape suggests something built lazily on first use and returned before it is ready; the
constellation has a standing pattern of deferring first-time GUI work to a frame callback for exactly that
reason. Not investigated.

### Starting tomorrow, filed 2026-09-01 — the AOKK judge pass, then two shared-GUI items

The Visualizer thread of the three below has closed: `done/visualizer-app-py-extraction.md` landed both
its extractions, `app.py` went 1763 → 1344 lines, and the package went from 4 of its 9 test modules
running in CI to 8 of 9.

**Take the AOKK judge pass first.** It is the only one of these three with a date on it — Juha needs the
result **within two weeks of 2026-09-01**, against a fortnight that already has other work in it — and it
is the one that unblocks other things rather than tidying something. The other two came out of closing
the extraction brief, are bound by nothing, and wait on the Librarian tree besides.

1. **The AOKK on-topic judge pass** — `aokk-corpus-scope-classification-brief.md`, in this folder.
   **Do it in a fresh session**: the brief is self-contained and carries its own measurements, and the
   work is a batch script over a dataset rather than a continuation of anything.

   Both of its open questions were settled on 2026-09-01, so it starts from a decided spec: judge
   against the **broad** question (the research questions get their own more detailed passes later, after
   the corpus is in Librarian), and emit **both** a filtered `.bib` — importable into Visualizer — and a
   list of what was dropped with a one-line reason each. The source corpus is not modified.

   Run the cluster screen the brief describes before the LLM pass; it puts 4 clusters covering 107
   records in front of a reader instead of 5007, and it is already measured.

   **Then begin with the brief's step 0**, the calibration pilot — pass 1 over a sample, hand-checked —
   which is the one measurement the two-pass design rests on and the only part of the brief that is still
   argument rather than evidence. Measuring before building is the habit; this closes the last place
   where the brief reasons about the model's behaviour instead of having looked at it.

2. **Move the Markdown font warm-up into `raven.common.gui`.** `setup_markdown`'s own docstring already
   says an app "may still" want a dummy element that forces `DearPyGui_Markdown` to load its fonts at
   startup, tells the reader not to add more than one during startup, and points at
   `raven.visualizer.app` as the example. That is the shared layer describing a job and handing it to
   every caller — and **only the Visualizer does it.** `librarian/app.py`, both avatar editors,
   `xdot_viewer/app.py` and the shared `common/gui/helpcard.py` all render Markdown and none warms up,
   so they are all exposed to the race the Visualizer's hack exists to avoid.

   Measured 2026-09-01, so it need not be re-derived: a Markdown element **can** be created between
   `create_context` and `create_viewport`, and it survives `setup_dearpygui`. But **`show=False` will not
   do** — DPG does not render a hidden window, and the warm-up only fires when the element renders. The
   Visualizer's version works because its container is *visible* and clips the text away, which is the
   `park_offscreen` shape `setup_markdown`'s docstring is really describing. Whether the font load
   actually fires cannot be confirmed headless, since it needs rendered frames, so this one ends with a
   live check.

3. **Test what each app's `FileDialog`s are configured for.** `importer_gui`'s two got this on
   2026-09-01 — filters, `multi_selection`, `save_mode` — because a wrong filter fails quietly, showing
   the user an empty folder rather than an error. Juha's call the same day: if we test those, the rest
   should be tested too, and there are a lot of them across the apps. `vendor/file_dialog` is already
   the most heavily tested widget in the tree, so the work is per-app configuration rather than the
   dialog itself.

**Why both wait**, and it is the lesson directly below rather than a guess: these are shared-GUI changes,
so they reach `common/gui/utils.py`, `common/gui/helpcard.py` and every app's tests at once — including
Librarian's. Dividing by module would not divide the test files, which is exactly how the 2026-09-01
collision happened.

### Learned 2026-09-01 — dividing threads by module does not divide their test files

The three threads below ran in parallel and collided anyway, in the one place the division did not
cover. One session was asked to stay clear of `importer.py` and did; both then wrote into
`test_importer.py`, because a thread that changes a module needs that module's tests, and "stay out of
X" does not imply "stay out of X's tests". The result was a working tree where three tests were red and
a lint error sat in a file two sessions were both editing — so neither could commit without pushing the
other's half-finished work.

**So divide by test file as explicitly as by module**, and when it is already tangled, the cheap way out
is to extract the staged blob and run it alone:

    git show :path/to/test_file.py > /tmp/check.py   # run pytest on this, then delete it

That proves what the *commit* contains passes, independently of what the working tree holds, and it took
one command. Juha staged the trailing hunk in Emacs and the commit went in green while the other
session's work sat untouched beside it.

### Starting tomorrow, decided 2026-08-31 — three threads, in parallel

Juha's call at the end of the day. These touch three different subsystems, so none is waiting on
another's tree, and they are different enough to review side by side — which is the actual constraint,
review bandwidth rather than machine time. **Only the first is bound by the deadline**; the other two are
here because they are ready to start, not because they are urgent.

1. **The graph view** — `16_chat-graph-view-brief.md`, which is item 2 of the ordering below and the
   thread that matters for 26 September. **Step zero is already done** (2026-08-25): a hand-built
   chat-shaped `Graph` renders through `XDotWidget.set_graph` with no GraphViz and no xdot in the path,
   and `test_widget.py` covers it. So this starts from the feature work rather than from proving the door
   opens.
2. **Visualizer tests** — ***done, 2026-09-01***, and it grew an extraction brief of its own on the way;
   see the subsection above for what closed and what came out of it.
   `briefs/visualizer-test-coverage-brief.md`, **which is in the parent folder,
   not this one**, being maintenance rather than sprint work. Start at `entry_renderer.py`: 114 SLOC and
   no `dpg.` calls at all, which the brief's difficulty table settles. The package went from zero tests
   to three on 2026-08-31, and that module records what it cost — the `importorskip` guard and `ml`
   marker `importer` needs to stay out of CI's way — so the second module starts from there.
   - **One thing not to test yet**: `_cluster_*` and `_reduce_dimension` are about to be rewritten
     (brief 11 item 5, and `investigations/highdim-clustering/`). Tests for those belong to that change;
     written now they would pin a defect.
3. **The AOKK on-topic detector** — `aokk-corpus-scope-classification-brief.md`, filed the same day.
   Buildable in parallel with the other two: it is a batch script over a dataset, touching neither GUI.
   The brief carries the two confirmed false-positive shapes and a warning about reusing
   `agent-batch-classification`'s confidence-based escalation.
   - *Not started that day, and now item 1 of the subsection above — it acquired a date.*

### Reordered 2026-08-31 — the effects first, then the graph view

Juha's call, 26 days out. **This subsection supersedes the band order below wherever the two disagree**, and
the thing it disagrees with most is the one constraint the 08-25 pass placed on itself: the graph view no
longer has to stay ahead of `crt`.

1. **`crt-display` and `atmospheric-dust`**, taken as one piece of work — *"or let's at least try for a day
   and see how far we get"*. Estimated **1–3 days in total**: self-contained, and straightforward once we
   iterate a bit. They go first because they are an important part of the wow factor, and because the past
   week has been unglamorous-but-necessary features. Time to switch gears.
   - ~~**Both done 2026-08-31**~~, in one session — well inside the 1–3 days, so the estimate was generous
     rather than wrong. `done/crt-display.md` §7 and `done/atmospheric-dust.md`'s closing section are what
     to read. The two share a lesson worth the reading: each brief was right about *what* to build and
     wrong about a *why*, and in both cases the wrong why was a formula that would have shipped. What
     remains is a look decision rather than a build: `crt` is in the default chain in place of
     `scanlines`, the dust is not in it at all, and the rastered dust field its placement implies has
     still not been seen in motion.
2. **The graph view** (brief 16). Estimated **one week, perhaps two**. It is the last major piece of
   *functionality* Juha needs on hand to explain the multiversal nature of LLMs, should the conversation go
   that way — and it has been promised since the first version of Librarian a year ago. The multiversal
   storage is what sets Librarian apart, and this is its missing GUI.
3. **The avatar's expression follows the spoken words** (band-2 item 11). Missing polish, and less
   noticeable than the effects — nobody cared last year. Straightforward and small, so it can probably be
   crammed in before the deadline.
   - **One complication risk, and it cannot be settled before it is tested**: the avatar holding neutral
     while the final text streams in may look uncanny, and if it does, what to do about it is not obvious.
4. **Block-level Markdown, the remaining steps** — after the event. Wanted, definitely, but last year did
   not even have the thinking trace streaming correctly and nobody cared; we are in much better shape now,
   and nobody is going to mind that it ships without the latest bells and whistles. Estimated **one week,
   perhaps two**, and it is **the first item to cut** if the time runs out.

That inverts the cut order recorded on 08-26, which named the Markdown remainder and the dust as the two to
give up. The dust is at the front with `crt` now rather than being the slack, and the Markdown remainder is
the declared slack in its place.

### Sized 2026-08-28 — what the last four weeks hold

Juha's sizing of what is left, with 29 days to the deadline. Recorded because it is a scheduling
judgement that produced no diff, and because two of the three are large enough that discovering their
size in week four would be discovering it too late:

| | |
|---|---|
| **11 — avatar expression follows the spoken words** | Monday's, and the design is already written below. Exhibit-facing: TTS is on whenever the avatar speaks |
| **12 — block-level Markdown, remaining steps** | **may eat a week** |
| **13 — the graph view** (brief 16) | **may eat a week** |
| **14 `crt-display` + 15 `atmospheric-dust`** | self-contained, and semi-quick. They are postprocessor filters over machinery that exists |

**Two week-eaters and a quick one, against four weeks.** That fits, and it fits without slack, which is
the thing to know now rather than in September. `atmospheric-dust` remains the declared slack (see the
2026-08-05 ranking below) — it is the one item nothing depends on, so it is what gives if a week runs
long.

The ordering that follows from this is 11, then 12 and 13 in either order, with 14 and 15 last because
they are the ones that can be compressed. Note this reverses nothing: it is the band order already
recorded, now with sizes against it.

**Superseded on 2026-08-31** (see above): 14 and 15 go first and 12 goes last. The sizes themselves stand —
they are what makes the new order fit.

### Decided 2026-08-07 — what Monday starts on

1. **Triage `TODO.md` and `TODO_DEFERRED.md` first** (Juha, partly via claude.ai). A two-part plan: a
   mechanical sweep for stale items against a brief Juha will supply, then a human-review pass over what
   survives. It goes first because the pile is what makes everything after it hard to see, and because the
   sweep's whole point is that a backlog nobody can read end to end is not a queue.
2. ~~**Then OS-independent file-manager drag-and-drop, ASAP.**~~ **Shipped 2026-08-10**, in all six GUI
   apps, as `raven.common.gui.filedrop`. Measurements and the two probes are in `investigations/dpg-dnd/`.
   It went in for two reasons, and **the exhibit was not one of them** — by the test above, a visitor in
   one sitting is not dragging files out of a file manager either:
   - **The 2026-08-07 probe collapsed its cost.** The platform work was already inside the GLFW that DPG
     links, so this was wiring rather than building.
   - **It is a power multiplier for our own testing**, which is where the gesture actually happens dozens of
     times a day: feeding corpora in, attaching a file to check a render, driving the GUI by hand. The
     `FileDialog` was the sole entry path for all of it, which is also why that picker has accumulated its
     own pile of deferred improvements.
     - **The part that only shows up in use** (Juha, on the day it shipped): the file manager *holds its
       place across app restarts*, and the dialog does not. Restarting an app twenty times to check one
       change means re-navigating to the same folder twenty times — so the saving is not one gesture per
       open, it is the whole navigation, every restart. That also means the `FileDialog` improvements stay
       worth doing rather than being made redundant by this: the dialog is still the path whenever the
       file is *not* already on screen somewhere.
3. **Then the exhibit briefs**: 16, then `crt-display`, with `atmospheric-dust` as slack.

### Where the FileDialog work stands, 2026-08-14

Ordering decided with Juha: **the grid view next, then keyboard access**, the latter expected to spill past
the weekend, which is fine — nothing else is waiting on it.

**The grid view is done and its brief is closed** (`done/filedialog-thumbnails-brief.md`, 2026-08-14).
**Only keyboard access is left.** In order, what landed: the shared `ThumbnailGrid` and Cherrypick's port
onto it (08-13), then in one day (08-14) smooth scrolling and the scroll-end flasher,
`raven.common.filelisting` with the dialog rewired onto it, and the grid view itself.

All of it is live-tested — the dialog in Visualizer and Librarian, the grid in both Librarian and
Cherrypick. Most of what the day cost was found by *using* it rather than by writing it: a coordinate bug in
`guiutils.get_widget_pos` that made tiles unclickable, a dead strip under every filename, a thread joining
itself, a rebuild per click, a re-decode per keystroke, and a missing multi-selection path — none of which a
test suite would have raised on its own.

### Four weeks out, 2026-08-25

Raised by Juha at the end of the 08-24/25 session, and recorded rather than acted on — the scheduling
itself is the next session's first job, not this note's.

**A prioritization pass is owed before any of it starts.** The concrete list under `TODO.md` → *Autumn 2026*
is dated 2026-07-28, sixty days out; a month of work has landed against it since, and nobody has asked
which of its entries are now done, stale, or newly urgent. Re-deciding item by item as each comes up is the
expensive way to answer that.

What was named as still open, at the point of naming:

- **`crt-display` and `atmospheric-dust`.** Unchanged in scope; both still briefed and unstarted, and the
  dust-as-slack ranking from 2026-08-05 still stands.
- **The turn-sequencing race**, "at least" — with **the abortable prefill**, which it was paired with the
  same evening. Both are in `TODO_DEFERRED.md` and each entry says what it takes from the other. This is the
  one piece of demo *correctness* that got named by hand tonight, which is not the same as it being the
  worst one; that is what the pass is for.

#### The pass, run 2026-08-25

Decided with Juha the following morning, over the `Gate: RN2026` items in `TODO_DEFERRED.md`, the briefs in
this folder, and the 2026-07-28 list in `TODO.md`. **This subsection supersedes that list where they
disagree**; the July one is left in place because its *reasoning* is still the reasoning.

**What changed underneath it.** Phase 1 was framed around "RAG access via tool-call", called there the
largest item and the only new construction — it shipped on **2026-07-29** (`TODO.md:704`). Reranking was
measured and rejected on 08-06. So phase 1 is now repair work throughout, and it is smaller than the July
framing reads.

**The build order: cheapest first, in three bands** (Juha, 2026-08-25). The reasoning is not that the small
items matter more — it is that finishing them *shortens the list*, and a shorter list is one whose remainder
can actually be seen. It also maximizes what is done if the four weeks run out early, since the yield
question at an exhibit is "what at least landed", not "what was started".

**One constraint on it, and it is the only one:** the graph view must not fall behind `crt`. It is the
largest item and the second-most demo-visible, and its own step zero sits in band 1 precisely so the risk is
known before band 3 begins.

> **Dropped 2026-08-31.** The effects go first after all. Half of what the constraint was for has been
> discharged in the meantime: step zero ran on 08-25, so the graph view's risk is already known rather than
> waiting on band 3 to reveal it.

**Band 1 — quick wins.** Hours each, and every one of them removes a line from this list.

1. ~~**Block-level Markdown, step 1**~~ — **done 2026-08-25.** Headings render, streaming included, and
   the `TODO_DEFERRED.md` item for them is retired. Step 6 (the `line_attributes` rename) landed with it.
   The remaining steps are band 2. Two things the brief had not foreseen are recorded there: the attribute
   rebuild that `LineEntity.append` performs had to carry the colour explicitly, and the colour parsing is
   now shared with the `<font>` attribute so the two cannot drift.
2. ~~**Brief 16's step zero**~~ — **done 2026-08-25.** A hand-built chat-shaped `Graph` goes into
   `set_graph` and comes out drawn, with no GraphViz and no xdot text in the path, so band 3 can be costed
   as the brief writes it. Every hook the brief leans on — `pan_to_node`, `set_highlighted_nodes`,
   `search`, hit-testing — works against a constructed graph. Seven tests, which is the coverage debt the
   step says is owed regardless of the feature. One measurement carried into the brief: culling is by
   viewport, so a freshly-set graph is only partly drawn until `zoom_to_fit` establishes the view.
3. ~~**Make `chattree`'s `save` atomic**~~ — **done 2026-08-25.** Sibling temp file, `fsync`, `os.replace`.
   The *cadence* half of the autosave item stays open, as planned.
4. ~~**The help card's missing `wrap=`**~~ — **done 2026-08-25**, on all three cards. Wrapping turned each
   clipped row into two or three real ones, so Librarian's card grew thirty pixels to hold them — and at
   1030 against a 1040 viewport it is now out of room. The next addition of any size needs the shape
   decision rather than more height.
5. ~~**`--qr`**, the "Get Raven" overlay.~~ **Done 2026-08-25**, in all six GUI apps.
6. ~~**The calculator tool**~~ — **done 2026-08-25**, and it was as small as claimed.
7. ~~**The simulated glitch on branch switch**~~ — **done 2026-08-25**, over four discontinuities rather
   than the one in its name. See below for what it needed and what is left to tune.

**Band 1 is closed as of 2026-08-25**, all seven items. **Item 8 closed on 2026-08-26 and item 9 on
2026-08-28**, so the work resumes at item 10 — with the two large builds, 12 and 13, still the ones carrying
unknowns. Seven days in, of four weeks.

Item 9 is worth reading as a schedule fact rather than only as a feature: it was costed as "a session each"
and took three, because returning to a branch as if you never left turned out to require the in-progress
reply to be a node, and making it one surfaced six faults that only a live run could find. Nothing about it
was mis-scoped — the requirement it grew was Juha's, added once the first version was on screen and was
merely *safe*. What that says for 12 and 13 is that the estimate covers the item as written, and a demo item
is not fully specified until someone has watched it run.

**Band 2 — a session each.**

8. ~~**The thinking trace: collapsed by default, with the cloud pulsating while the model thinks.**~~ —
   `TODO.md:598`. Demo *correctness* by the argument already recorded there, and Juha's reason for keeping
   it in: hidden thinking is what people now expect from an LLM system. The token/time readout in the same
   item is in, not optional (Juha, 2026-08-26).

   **Done 2026-08-26**, all eight sub-steps. Built as a sequence, data first so most of it is verifiable
   without a window. Sub-step 8's entry says what was deliberately left out of the sprint rather than
   built.

   1. ~~The local-tokenizer fallback counted only the answer, never the reasoning.~~ Done.
   2. ~~`invoke` records where the turn's wall time went — `generation_metadata["phases"]`, holding
      `prefill` and `thinking`; the answer is the remainder, so no third number can disagree.~~ Done.
   3. ~~`StreamParser` recognizes reasoning that arrives with no opening tag (`reasoning_retcon`).~~ Done.
   4. ~~The trace grows in its bubble from the first word, cloud pulsating while it fills.~~ Done.
   5. ~~The `show_thinking` preference, applying to what arrives next.~~ Done.
   6. ~~**The readouts.**~~ Done. `[Thought for 759t, 8.79s, 86.36t/s]` on the cloud row, `Thinking…
      4.8s, ~428t` counting up while streaming, and a four-column breakdown in a tooltip on the message's
      own line (label / time / tokens / speed, units in the header, blanks where a quantity does not
      apply). The message line's meaning is unchanged, since an old node cannot be recomputed.
   7. ~~**`Enable thinking`.**~~ Done, and it shipped as *Thinking*, first in the row: a per-call
      `thinking_enabled` on `invoke` and `ai_turn`, an app-state flag, and one checkbox. The wire mapping
      is `llmclient.thinking_request_fields`, the single place where Raven's vocabulary becomes the
      backend's. The live-backend group gained the assertion a mock cannot make — that the field is
      honoured — with the on-case as its control, so a non-thinking model skips instead of passing
      vacuously.
      - **The agent loop survives the toggle**, which is the part that matters for the exhibit and was not
        obvious in advance. With reasoning off, `qwen3.6-35b-a3b` still reached for the calculator when
        asked for 1234 × 5678 rather than answering from its head (Juha, 2026-08-26). So on this model
        tool use does not ride on the thinking channel, and switching thinking off buys speed without
        costing the tools. Recorded here rather than in `TODO.md`, whose items are deleted when done.
   8. ~~**The live display when the opening tag never arrives.**~~ Done 2026-08-26, though **not the way
      this step was written**, and the difference is worth reading before anyone reopens it.

      The plan was to start the parser in `_PS_THINK` when step 7 says thinking is on. That does not work:
      it needs three facts and the toggle supplies one — thinking is on, the backend does not split
      reasoning into its own channel, and *this model's template pre-fills the open tag*. Without the
      third, a thinking-enabled model that simply does not reason this turn puts its whole answer in the
      thought bubble with no close to end it, which is worse than the defect being fixed.

      What landed instead is the correction, forwarded live: `invoke` already computed it and declined to
      pass it on, and the renderer's half had become a re-render rather than surgery — step 4 gave every
      paragraph its own `is_thought`, and `_render_text` reads it per paragraph. So it needs no signal, no
      flavor gate, and cannot swallow an answer. `TODO_DEFERRED.md` carries what is left: the interval
      *before* the close tag, and the fact that none of it has been watched running, because the only
      backend on hand cannot produce the event.

      **Out of the sprint, deliberately:** the `_PS_THINK` start itself, whose only honest caller is the
      mid-thought resume; and that resume, which is behind an unmeasured probe (what a backend's own
      reasoning parser does with a prompt that ends inside a block) and is a Qwen mechanism with no Gemma
      mirror. Neither is four-weeks work, and neither is exhibit-visible — the exhibit runs LM Studio,
      which never reaches this path at all.

   Two renderer bugs were found and fixed while verifying step 4, both of which had been stranding
   decorations on any reflow: list markers and blockquote bars. What is left of that is one queued item —
   block constructs need a block container, which is also what gives a quote a single full-height bar.
9. **The turn-sequencing race with the abortable prefill.** ✅ **Done 2026-08-26.** The mechanism the two share is measured in
   `investigations/abort-inflight-request/` — and it is not the one the deferred items assumed. Closing the
   `requests` response from outside neither wakes the blocked reader nor returns to its caller, so a Cancel
   button wired to it would freeze the GUI for the length of the read timeout; `socket.shutdown` is what
   works. That an abort frees the *backend*, not merely the client thread, is measured there too.

   **What each control does while a reply is in flight** — decided 2026-08-26, and the four answers are not
   the same because the four actions do not mean the same thing:

   - **Switching branches lets the turn finish**, quietly, on the branch it started from. This is the
     Librarian-shaped answer: the tree keeps growing while you browse it, and the reply is waiting when you
     come back. It needs no cancellation, only a guard stopping a finished turn from writing into a view
     the user has since navigated away from. **New chat is a branch switch in disguise** — the docs already
     say so — so it behaves the same way.
   - **Cancel (Ctrl+G) is the stop-what-you-are-doing control**, and the only one that abandons work. It
     exists already; what this item adds is that it starts working *during prompt processing*, where today
     the cooperative flag cannot be read until tokens flow.
   - **Send and Reroll refuse** until the reply is complete. Both are ungated today — `app.py` still
     carries the `# TODO: disable this button while AI is writing` — and both currently start a second
     concurrent generation against the same backend, with two turns writing HEAD.

   The alternative considered and rejected for branch switching was cancelling the turn, which is what the
   deferred item originally proposed. Once the guard exists the bleed is gone either way, so cancelling
   would buy only a freed backend, at the cost of discarding a reply the user may well come back for.

   **Queueing a message sent during a reply** was raised as a possible replacement for Send's refusal, and
   is *not* part of this item. It needs its own cancellation, a way to amend a queued message in place
   before it goes, and invalidation whenever the context it was written against changes — reroll, new chat
   or a branch switch.

   **Landed 2026-08-26.** A turn and a prefill are cancelled on opposite policies, and the reason is what
   each leaves behind: a turn's product is a reply, which keeps its value on the branch it was written for,
   so a branch switch lets it finish. A prefill's product is a warm cache *for one branch* and is worth
   nothing off it, so a branch switch abandons it — flick through branches, stop to read one, and the
   prefill that starts is dropped the moment you move again, leaving the backend free for the question you
   then type (Juha, 2026-08-26).

   What the deferred item called the abortable prefill turned out to be the smaller half of the fix: the
   same handle is what finally makes **Cancel work during prompt processing**, which is where most of the
   wait is on a long chat and where the button had always been inert. The mechanism sits in
   `raven.common.netutil` rather than in Librarian, because abandoning a blocked streaming read is a
   networking problem; `bgtask` grew an `on_cancel` hook so that a task blocked in a library call can be
   cancelled at all, which its co-operative flag could never reach.

   **Returning to a branch acts as if you never left** (Juha's requirement, 2026-08-26, and the second thing
   this item grew). The first attempt let a rebuilt view drop an in-progress reply until it finished, which
   was safe and wrong; the second re-rendered it from the message's own paragraph records, which worked and
   left every *other* walker of the tree unable to see the reply at all.

   **So the reply became a node** (2026-08-27, `done/in-progress-reply-as-a-node.md`). The
   assistant node is created before the request and filled in as the text arrives, which makes an
   in-progress reply an ordinary member of the branch: `descend_to_latest` lands on it, the sibling counter
   counts it, and `DPGLinearizedChatView.build` renders it like any other node — so a resize, a branch
   flick and a reader scrolled away all come back to the same place, and no navigation path has to know a
   turn is running. That deleted the publish/reattach machinery the paragraph-records version needed.

   **Verified live 2026-08-26 and again 2026-08-27**, against LM Studio with the avatar and TTS running.
   The second run drove the node-first rework end to end: a reply streaming, a resize mid-reply, a flick to
   another branch and back with the reply still writing, a reader scrolled away while a rebuild landed, and
   Cancel mid-generation keeping the partial reply. Three faults surfaced and were fixed there that no test
   could have caught — the finished reply deleted by the turn's own cleanup, the thinking counter's cadence
   following the model's newline pattern instead of the clock, and a DPG container-stack error inside a
   repaint being recorded as a backend failure over the model's text.

   **An incomplete stored message now says so on screen** (2026-08-27). A reply that was cancelled, or that
   the backend failed partway through, is kept — that is Cancel's promise — and it used to look exactly like
   a reply the model chose to end there. The marker was already in the node's `generation_metadata`, so this
   is a render-time footer added by the GUI build and never stored, which is what keeps the continue
   machinery from having to strip it again: `[Interrupted — the reply was stopped here]` for one the user
   stopped, `[Incomplete — Raven exited while this reply was being written]` for a chat closed mid-reply.

   **And the last red is green** (2026-08-28). Continuing a message replaced it with the continuation alone,
   both channels, because `invoke`'s accumulators start empty and the backend sends only the new text.
   Predates this work and was found by it. Fixed by seeding the accumulators from the message being
   continued — held apart from them, so every count and every reset stays about the current call — and
   verified live in both shapes, with the stored revisions read back off disk. **One thing it opens: ooba is
   unmeasured and is now a regression risk**, since it continues through a request field of its own where LM
   Studio continues by prefill; `TODO_DEFERRED.md`'s ooba item carries the question. **Item 9 is closed**,
   and the brief is in `done/`.

   **Cancel during prompt processing was the last uncovered case, and it works** (2026-08-27): a new chat
   with a fulltext PDF attached gives a prompt long enough to have a window, and Ctrl+G inside it took the
   abort path — `abandoning the backend request`, and the blocked read returned **two milliseconds** later,
   against the read timeout it would have waited out before. That is the whole `netutil` mechanism
   confirmed against a real backend rather than against a probe.
10. ~~**The STT silence level / autostop GUI**~~ — **done 2026-08-28**, designed and closed in a day,
    as `done/stt-audio-input-panel.md`. The reason it was on the path held up: speech input is new
    this year, and the number it exists to set is one only the room can supply. What the day actually
    demonstrated is narrower and worth keeping — **the eight faults that mattered were all invisible
    from the desk**, and every one of them came out of driving the thing with a real microphone. Read
    the brief's closing section before assuming any other GUI item is finished when its tests pass.
11. **The avatar's expression follows the spoken words rather than the streaming ones** — ranked in on
    2026-08-26, out of the three items raised that day. It is the one of them the exhibit's own hardware
    does not excuse: TTS is on every time the avatar speaks, so a face reacting to a sentence the voice has
    not reached is live all evening. The section below carries the design and says why the other two stayed
    out.
12. **Block-level Markdown, the remaining steps** — the single-newline split, which is the barrier fenced
    code and multi-line lists are behind. **Step 5 came out of this band and landed on 08-25** with step 1:
    the white bullets were visible the moment headings started rendering, and step 1 had already built the
    fallback the marker colours needed.

**Band 3 — the large ones.** *Reordered 2026-08-31 to 14, 15, 13; item 12 moves behind all three.*

13. **The graph view** (brief 16).
14. ~~**`crt-display`.**~~ **Done 2026-08-31**; see `done/crt-display.md`.
15. **`atmospheric-dust`.**

And the two whose detail belongs with the ordering rather than in it:

- **The simulated glitch on branch switch** (`TODO.md` → *Avatar (Librarian-side)*, band 1) — **landed
  2026-08-25.** It covers four discontinuities rather than the one it is named for, all of them the
  conversation on screen being replaced by a different one: switching to a sibling, jumping to where a
  branch continues, starting a new chat, and rerolling a reply. The reroll's alternative is generated
  rather than already there, which is the only thing that sets it apart from a sibling switch.
  - **It was bookkeeping rather than construction, as expected** — the filter exists
    (`postprocessor.digital_glitches`) and the chain is reconfigurable on the fly, so the job was to
    overlay one filter and put the chain back. Putting it back is the part that needed building: the server
    offers no getter for animator settings, so nothing could restore what it had changed. Settings now go
    out through `DPGAvatarController.load_animator_settings`, which remembers them per instance.
  - **The duration is the effect's own, merely *triggered* by the switch** (Juha, 2026-08-25): a **floor**,
    because a switch too fast to see would otherwise flash something nobody registers, and a **ceiling**,
    because glitching for too long stops reading as artistic and starts reading as broken. The restore runs
    off a timer, and repeated calls extend rather than restart, so flicking through siblings reads as one
    glitch instead of a stutter of them.
  - **What is left is tuning by eye.** The parameters start from
    `raven/avatar/assets/settings/glitchyholo.json`, which runs this same filter as a continuous ambient
    effect; a switch wants it *more* prominent than that. The knob runs backwards — `unboost` is the
    probability profile, and **higher makes glitches rarer and fewer** — so the flourish sits *below* both
    that file's `10.0` and the filter's own default of `4.0`.
    - **The ceiling was chosen against a case nobody can currently perform.** It caps a *sustained* run of
      switches, and sustained switching is exactly what the sibling-flicking item exists to make possible,
      so it wants re-checking when flicking lands. Recorded on that item in `TODO_DEFERRED.md`, which is
      where someone will be looking.
- **`atmospheric-dust`** (band 3, last) — still the slack in the schedule, but **wanted rather than merely
  tolerated** (Juha): a significant wow factor, and self-contained enough to be a safe last item. The 08-05
  ordering behind 16 stands.
  - **Not any more, on either count** (2026-08-31): it goes first, alongside `crt`. The half of this that
    survives is the appraisal — a significant wow factor, self-contained — which is now the argument for
    its position rather than for its safety as a last item.

Notes on four of the band-1 entries, each of which is smaller than the thing it is a piece of:

- **The calculator tool** (`Gate: calculator RN2026`) — in *if it is as small as the item says* (~a page
  with `simpleeval`). A model doing arithmetic badly in front of an audience is a visible failure and this
  is the one built-in that removes it. If it turns out not to be a page, it drops out rather than moving
  bands.
- **`--qr`**, the "Get Raven" overlay. Cost S, and the only item on the list whose whole purpose is the
  event.
- **Atomic `save`** is deliberately *half* of the autosave item. The **cadence** is not being opened here —
  it couples to datastore scaling, and that is a design question, not a four-week one. Writing to a temp
  file, `fsync`, then `os.replace` is independent of whatever cadence is chosen later, and it reduces the
  blast radius of the current once-per-session write on its own.
- **The help card's missing `wrap=`** is likewise half of its item: the horizontal clipping and nothing
  else. The card's *shape* decision stays off the path.

**Accepted risk, with a workaround rather than a fix: the renderer dropping text.** The `Gate: RN2026` item
stays open and is *not* scheduled — there is no repro, the diagnosis points at the font atlas, and Juha's
reading is that it is a pharaoh's curse on the session rather than on the run: a session that renders
correctly goes on rendering correctly. **So the operational answer on the night is to restart Librarian
until it comes up clean**, which is cheap and needs nobody to have fixed anything. Look at it if the atlas
work happens for another reason.

**Cut, each for a stated reason** — this is the half that would otherwise be re-decided from scratch:

- **Wake-word input** (`TODO.md:625`) — pushed past the event. Wanted in the lab, and later this year is
  fine. It needs continuous capture fanned out to three consumers plus two interaction styles tested against
  real strangers, which is not a four-week item alongside the rest.
- **The STT input-language selector** (`TODO.md:617`) — off the path. Raven is English-only for now and
  language selection is future expansion. Note the mixed-audience argument in that item is *true*, and is
  answered on the night by the operator asking visitors who want to speak to the system to ask in English —
  by instruction rather than by a control, which is why cutting the control costs nothing this year.
- **A file-type icon set of our own** (Cost M) — deferred. The file dialog barely appears on stage.
- **Advertising drag-and-drop**, and **the help card's shape decision** — deferred. Both are discoverability,
  and an open house has an operator standing next to the machine.

**The prompt viewer sits after band 3, and only if time remains** (Juha, 2026-08-25 — asked and answered
during the pass). The transparency argument for it is real, but **the graph view answers the same question
better**: asked to talk about LLMs with an audience, that is what Juha reaches for first. And its other
half is that it is a debugging feature for the maintainers — which is a good reason to build it and a poor
reason to build it *now*, two weeks after a run of power-multiplier work. Time to switch gears.

It is pointed at here rather than merely left in place because the risk to it is real but misidentified: it
lives in `TODO.md` → *Librarian → Chat UI* ("Show the raw prompt"), `[High]`, with its decisions taken —
**not in `TODO_DEFERRED.md`**, so what threatens it is not the hydra but the other failure mode, that
nothing in the workflow makes anyone open `TODO.md`. This pointer is the fix. It does not need a brief; the
sizing that kept it out of one was re-checked on 2026-08-25 and holds.

**Two premises from the July list that no longer bind:**

- **VRAM is not a constraint** (Juha). The system ran last year on 24 + 8 GB; this year it is 24 + 16 GB.
  The instruction to re-run `investigations/vram/avatar_footprint.py` after `crt` and `dust` land is
  therefore a curiosity, not a gate.
- **Phase 3 does not need a week.** This is an open house rather than a demo presentation, so **one day of
  checking that everything works is enough, and even that is generous.** The July note that phase 3 "always
  gets eaten" was written for a schedule that no longer has that shape.

**The STT silence level / autostop GUI is on the path** (`TODO.md:614`; raised as unsettled during the pass,
decided by Juha the same day). **Speech input is new this year** — last year the operator typed the
visitors' questions in — so a visitor talking to Aria directly is a first outing in a room whose noise floor
nobody can predict, and that is exactly the case the item was filed for: the threshold has to be tunable
*in the room, on the day*, from a control rather than from a config file.

**And this is what settles the language question rather than the selector doing it.** The operator asks
visitors who want to speak to the system directly to ask in English. That covers the mixed-audience case
the cut selector was for, costs nothing, and is available on the night whatever else does or does not
land.

**And one item was misread during the pass, worth recording so it is not misread again.** The "avatar
branch-switch glitch" in `TODO.md:45` is not a defect awaiting a filing — it is `TODO.md:737`, a *wanted*
digital-glitch effect on branch switch, scoped there as a scripting task over postprocessor filters that
already exist, and treated in `briefs/design/product-identity-sketch.md:241` as the first concrete claim of
the aesthetic direction. It belongs beside `crt` and `dust` as impressiveness work, and is plausibly the
cheapest of the three.

### Raised 2026-08-26 — what the avatar does while a reply is being generated

Three items from Juha, arriving after the 08-25 pass. They are written up together because they are one
subject — what the avatar, the emotion detector and the TTS each do *during* a reply rather than after it —
and because two of them want the same piece of machinery, the sentence split that already exists inside
`DPGAvatarController.preprocess_task`.

**Ranked 2026-08-26, and the exhibit's hardware decides two of the three.** The exhibit runs the full rig
with the eGPU attached, so items 1 and 3 below — both filed for the single-GPU, low-VRAM, CPU-TTS
configuration — buy little there and **stay out of the sprint**. Item 2 went into band 2 as its item 11:
speech is on whenever the avatar talks, so it is live all evening regardless of what the machine can spare.

**Both are now in `TODO_DEFERRED.md`** (2026-09-02), as *An option to hold the avatar's video off until the
answer is complete* and *Start synthesizing speech while the reply is still streaming* — they outlive the
sprint, and this folder moves to `briefs/done/` when it closes. Everything below is carried over there;
what stays here is the ranking itself, which is a decision about this exhibit.

**How the rig is budgeted is what makes both of those firm** (Juha, 2026-08-26): the 24 GB card is
dedicated to the LLM, and everything else — avatar, TTS, and the rest of the server's models — runs on the
internal dGPU. So the two premises the items were filed on are both absent here. There is no contention
between the avatar and the LLM to relieve, because they are not on the same card; and the TTS is not on the
CPU, so there is no first-sentence latency worth overlapping generation to hide.

- **Item 1 remains wanted for the road**, where it was aimed in the first place; it is simply not
  exhibit work.
  - **It has never bitten anyone**, and that is a fact about the development environment rather than
    about the item: the eGPU is attached at the desk whenever work happens, so the configuration the
    item is *for* is the one nobody develops in (Juha, 2026-08-26). So it is untried rather than
    unimportant, and whoever builds it has to manufacture the condition to see it at all, by hiding
    the eGPU from the process and letting the avatar and the LLM land on one card.
  - **The road configuration is supported, not exceptional** (Juha, 2026-08-26): the laptop away from the
    desk, wherever there is a power outlet. It is an ordinary way to run Raven and is expected to work
    well, which is why this is an option worth having rather than an accommodation for a rare case.
  - **Within it, the occasion to design for is the unplanned demo** — *"I have my laptop with me, want to
    see it?"* It is the least forgiving use of the mode: the audience is standing there, the attention
    lasts about a minute, and a sluggish avatar spends it. It is also unplanned by definition, so the item
    cannot be built when it is needed — the moment it is wanted is already too late.
- **Item 3 would come back only on a machine that runs TTS on the CPU**, which this one does not. Its
  measurement stands either way: the first sentence is the whole latency, and everything after it renders
  faster than it can be spoken.

Their common shape is that each currently acts on the wrong unit of work: the whole turn where it wants the
sentence, or the streamed text where it wants the spoken text.

**The band-2 order after this ranking is 9, 10, 11, 12, then band 3.** Correctness first, then the item
that can only be tuned in the room, then the avatar's face, then the Markdown polish. The pass's one
constraint still holds: the graph view stays ahead of `crt`. If the four weeks tighten, the two to give up
are the Markdown remainder and the dust — the dust was declared slack from the start, and the Markdown
remainder is the only band-2 item nobody in the room will notice missing.

**Superseded 2026-08-31**, with 9 and 10 done: the order is now the effects, the graph view, then 11, and 12
is the only thing left to give up. What still holds is the *reason* item 11 is in at all, which is the
paragraph above it — the exhibit's hardware does not excuse this one.

1. **An option to hold the avatar's video off until the answer is complete** — GPU anti-congestion.

   Today the avatar wakes as soon as the user's message is sent: `DPGLinearizedChatView.build` calls
   `avatar_controller.ping`, and `ai_turn` then wraps the whole turn in `idle_override`, so the video runs
   for the entire generation. **That is the right behavior and stays the default** — it acknowledges
   receipt, which is what a user needs from those seconds, and on a machine with headroom there is nothing
   to fix.

   It is wrong on a **single-GPU, low-VRAM setup**, the on-the-road configuration, where the avatar and the
   LLM contend for the same cores and the acknowledgement is bought with generation speed. So: an option
   that defers the wake to the point the full answer completes.

2. **With TTS on, the emotion should follow the spoken words, not the streaming ones.**

   `_update_avatar_emotion_from_incoming_text` fires per paragraph as text arrives, thoughts included, and
   `on_done` updates once more from the final message. With speech off that is correct and should be kept:
   the streaming text is what the user is reading, so it is what the face should answer to.

   With speech on it is wrong, because the user is *listening*, and the face is already reacting to a
   sentence the voice has not reached. The reply should instead settle to a neutral expression while it
   streams — or, in thinking mode, arguably to the majority of the emotion updates seen so far, which is an
   open question rather than a decision — and then take its updates from each sentence **as it is spoken**.

   The hook exists: `send_text_to_tts` splits the batch into sentences and already emits `on_start_sentence`
   per sentence.

   **The stabilization logic is the reusable part.** The streaming path keeps a deque of recent paragraphs
   with 75% overlap between updates, precisely because per-chunk detection is unstable; the spoken path
   needs the same treatment for the same reason. That is a utility waiting to be extracted rather than a
   second copy waiting to be written.

3. **Start synthesizing speech while the reply is still streaming**, to cut time-to-first-spoken-word.

   This matters most when **TTS runs on the CPU**, and there it is specifically the **first sentence** that
   is the whole latency. Kokoro on CPU is slightly faster than realtime (Juha, 2026-08-26), so once speech
   has started, every later sentence renders comfortably within the time the previous ones take to speak.
   The pipeline only ever stalls at its head.

   **That bounds the work**, and suggests a variant worth weighing rather than a decided design: getting
   the first sentence submitted the moment it completes captures nearly all of the win, so the rest of the
   reply could still go over as one batch. That would be a smaller change than "submit each sentence as it
   arrives", and would leave one extra batch boundary per reply rather than one per sentence — but it buys
   that by making the reply's first sentence a special case, which per-sentence submission does not. Which
   of the two is cleaner is for whoever builds it.

   The preprocessor already precomputes audio per sentence as early as it can — its docstring says
   `on_audio_ready` "may trigger long before the sentence is actually spoken out loud" — so the machinery
   is there. What blocks it is *when the work is handed over*: `on_done` submits the finished reply as one
   batch, so nothing can start until the last token has landed. Submitting each sentence as it completes
   would let synthesis overlap generation.

   **It also ends the assumption that nothing is spoken until the reply is whole**, which something else
   relies on. On a backend that leaves reasoning-tag parsing to us, a thinking model whose template opened
   the block streams its reasoning indistinguishably from an answer until the closing tag lands; Raven
   corrects that when it lands (`reasoning_retcon`, 2026-08-26) by moving the text into the thought bubble.
   That correction is safe today only because the TTS batch is submitted after the whole reply — so
   whatever was mis-shown was never spoken. Hand a sentence over as it completes, and a correction arriving
   afterwards means the avatar has already said a "sentence" that was reasoning, with nothing able to
   unsay it.

   **Decided (Juha, 2026-08-26): hold the first submission until the text is known to be the answer.**
   Speech is the one output that cannot be taken back, so it is the one that has to wait for certainty.

   **"Known" has an exact meaning here**, and it is worth taking from the parser rather than re-deriving:
   a retcon fires at most once per stream and only while `StreamParser._may_retcon` is set, which it clears
   as soon as reasoning has been identified by any other route — a native `reasoning_content` channel, a
   properly opened block, or a retcon already spent. So the condition is "that flag is clear", and the
   natural implementation is for the parser to say when the window shuts rather than for a consumer to
   infer it.

   **The case it costs is worth knowing before building:** a model that does not reason at all, on a
   backend that does not split the reasoning channel. Nothing ever closes the window there, because nothing
   ever proves a close tag is not still coming — so that combination gets no speedup. It is also the
   combination this whole hazard is confined to; on the demo backend the reasoning arrives on its own
   channel, which shuts the window on the first thinking token and long before any sentence completes.

   **The division of concerns is what needs deciding**, and it is the reason this is not simply a smaller
   `send_text_to_tts` call. The batch is currently the unit that `on_start_speaking` / `on_stop_speaking`
   describe, and ordering across batches is preserved by queueing each one whole. A reply split into many
   batches keeps its order but changes what those events mean, and anything keyed on "the reply started
   being spoken" has to be found and re-anchored. Worth checking what subtitling and the recording hooks
   assume before costing this.

### Two of these are power multipliers, and it is worth naming the category

**All of it has landed** — drag-and-drop 2026-08-10, brief 15 on 08-12, the `FileDialog` keyboard brief on
08-21 — so this section is now the *argument* rather than a queue. It is kept because the category outlives
its first three members: the next time something is proposed that no visitor will ever see, this is the case
for ranking it against feature work rather than beneath it.

Drag-and-drop and **15** are not features for end users at all. They are tooling that
multiplies the *builders'* throughput, and they are ranked accordingly rather than by user-visible value. Since review is
the binding constraint on this project (see the root `CLAUDE.md`, "Who develops Raven"), anything that
raises how much can be built and checked per session competes directly with feature work rather than
sitting beneath it.

They multiply different people, which is why both are wanted:

- **Drag-and-drop is for the human**: feeding corpora in, attaching a file to check a render — dozens of
  times a day, through what was then the sole entry path, the `FileDialog`. **The `FileDialog` improvements
  belong to this same category** and were wanted for the same reason: smart-case find, thumbnail previews,
  the multi-extension filter, the per-use-site boilerplate, and keyboard access. They are not user-facing
  polish so much as throughput for whoever is driving the apps all day. Whatever remains of that list is in
  `TODO_DEFERRED.md` and in the re-homed `../filedialog-navigation-history-brief.md`.
- **15 is for Claude, and specifically for writing probes**, not for GUI testing — that is a separate
  problem which a scripting surface does nothing about, and which still needs a live session.

  The evidence is already in the tree: `../librarian-extension/manual_tests/` holds six scripts —
  `rag_live_corpus.py`, `rag_tool_rescue.py`, `webfetch_live_extractors.py`,
  `webfetch_tier2_escalation.py`, `gemma4_reasoning_roundtrip.py`, `vision_check.py` — each of which
  reaches into Librarian's agent machinery to exercise one feature, and each of which had to arrange that
  access for itself because the library did not offer it. That was a supported feature being asked for
  repeatedly and answered ad-hoc every time; `raven.librarian.agent` is the answer, and the surface is
  documented for outside users under *Scripting* in `raven/librarian/README.md`.

  So 15 is less "new capability" than "stop making every probe re-invent the entry point". Its part 0, lazy
  `api.initialize`, also removes `test_scaffold.py`'s `importorskip`, widening what CI can run.

**Not in the Researchers' Night run**, decided the same day: Hindsight memory (06) waits until after it —
because a visitor who talks to the system once cannot observe a feature that pays off over a long-running
relationship, at any level of completeness — and the MCP client (04) and lorebook (05) are question marks,
useful but not open-house-critical. See `../librarian-extension/README.md`, "After those three", which
carries the generalizable form of the memory argument: **a feature whose value accrues over time cannot be
demonstrated in an encounter that does not.**

**Ligature repair** (`../ligature-repair-brief.md`) also waits, *unless* the `raven-fixbib` half turns out
small enough to sneak in — which the brief argues it is, being the function plus a flag plus a report. The
indexer half is not a candidate under any reading.

**Brief 17 is reserved but unwritten** — a per-document LLM pass with retry, cache, resume and progress, cut
out of 15 because it has three users of its own and is a batch-execution primitive rather than a scripting
surface. If it stays unwritten, 17 may end up being something else, and 15's reference to it will need
chasing.

## Where the other sprint is

`../librarian-extension/` — the 01–10 run, mostly closed. Its `README.md` carries the ordering rationale for
what remains there (04, 05, 06) and the record of what 0.2.8 shipped.

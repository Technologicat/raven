# Researchers' Night sprint

**Deadline: 26 September 2026.** That is what this folder is named for, and it is the honest name — the
contents are not one scope. They are the briefs that were conceived during the Librarian run but scheduled
after it, and what they share is the deadline that orders them, not a subsystem.

Split out of `librarian-extension/` on 2026-08-07, at the 10/11 boundary. Worth knowing that the boundary is
chronological rather than topical: the numbers ran in the order the briefs were *written*, so 14 and 15 are
Librarian features and 11 is Visualizer, sitting side by side here because of when they were conceived.

## What's here

| Brief | What | Status |
|---|---|---|
| `16_chat-graph-view-brief.md` | The chat tree as a graph, for the exhibit | Researchers' Night. Explanatory before navigational — the job is making "an LLM is a multiverse generator" visible. Step zero is that `XDotWidget.set_graph` has no callers and no tests |
| `crt-display.md` | Avatar postprocessor: CRT look | Researchers' Night |
| `atmospheric-dust.md` | Avatar postprocessor: dust | Researchers' Night, and **the schedule's slack** — lands only if time remains. Ranked behind 16 on 2026-08-05. Safe to drop: nothing depends on it, and it borrows its priority-band scheme *from* `crt-display.md` rather than the other way round |
| `12_derived-artifact-store-brief.md` | One keying and regeneration mechanism for everything computed *from* a source artifact | v0.2.9. Does not depend on 13 |
| `13_corpus-scopes-and-unified-db-brief.md` | Corpus scopes and the unified DB | **A draft, not a design** — it holds the 2026-08-01 session material with its `[D]`/`[N]`/`[P]`/`[X]` provenance markers intact, so a reader can tell settled from proposed. Realistically after Researchers' Night |
| `11_visualizer-importer-rework-brief.md` | Nomic migration, PCA preprocessing, cosine-to-medoid outlier assignment, Procrustes alignment | Its item 1 carries **a fork that needs deciding** — `nomic-embed-text-v1.5` buys a shared image-text space, `v2-moe` buys multilingual, and no v2-aligned vision encoder appears to exist. That decision reaches brief 12 |
| `14_chat-search-brief.md` | Search within the chat log | v0.2.9. The match unit is the **message**, which is what keeps v1 cheap — it sidesteps in-text highlighting, whose Visualizer implementation rebuilds the whole panel and so does not transfer to an incrementally-built chat log |

## Closed

| Brief | What | Landed |
|---|---|---|
| `done/15_headless-agent-driver-brief.md` | A scripting surface over the scaffold — `raven.librarian.agent`, plus the backend-status work and the per-variety system prompt storage that came out of it | 2026-08-12, v0.2.9 |
| `done/filedialog-thumbnails-brief.md` | Image previews in the file dialog, as a toggled grid view | 2026-08-14 |
| `done/filedialog-keyboard-brief.md` | Operating the file dialog without a pointing device, and saying where the keyboard is | 2026-08-21 |

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

**Band 1 is closed as of 2026-08-25**, all seven items. Band 2 is where the work resumes.

**Band 2 — a session each.**

8. **The thinking trace: collapsed by default, with the cloud pulsating while the model thinks** —
   `TODO.md:598`. Demo *correctness* by the argument already recorded there, and Juha's reason for keeping
   it in: hidden thinking is what people now expect from an LLM system. The token/time readout in the same
   item is in, not optional (Juha, 2026-08-26).

   **In progress, 2026-08-26.** Built as a sequence, data first so most of it is verifiable without a
   window. Six of eight steps are done and committed; what remains is below.

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
   7. **`Enable thinking`** — `TODO.md`'s "Thinking toggle", pulled into this sprint and much smaller than
      its July design: `reasoning_effort: "none"`, measured working across Qwen 3.6, 3.8 and Gemma 4. No
      prefill, no per-family marker table.
   8. **Then the parser can start in `_PS_THINK`** when step 7 says thinking is on and the backend is
      single-channel — which closes `TODO_DEFERRED.md`'s "Streaming thinking shows as gray" live half.
      Headless-testable; end-to-end only against a single-channel backend, i.e. not before ooba.
      - **It gained a second consumer on 2026-08-26, so do not cost it as a one-line parser tweak.**
        Resuming a reply that was interrupted *mid-thought* needs the same signal — something has to say
        "you are starting inside a think block". See `TODO.md`'s thinking-toggle item for the mechanism,
        what is already measured, and the catch: it is a Qwen trick, and Gemma has no mirror for it.

   Two renderer bugs were found and fixed while verifying step 4, both of which had been stranding
   decorations on any reflow: list markers and blockquote bars. What is left of that is one queued item —
   block constructs need a block container, which is also what gives a quote a single full-height bar.
9. **The turn-sequencing race with the abortable prefill.**
10. **The STT silence level / autostop GUI** — see below for why it is on the path.
11. **Block-level Markdown, the remaining steps** — the single-newline split, which is the barrier fenced
    code and multi-line lists are behind. **Step 5 came out of this band and landed on 08-25** with step 1:
    the white bullets were visible the moment headings started rendering, and step 1 had already built the
    fallback the marker colours needed.

**Band 3 — the large ones.**

12. **The graph view** (brief 16).
13. **`crt-display`.**
14. **`atmospheric-dust`.**

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

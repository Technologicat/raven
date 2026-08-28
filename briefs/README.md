# Briefs and project documents

Working documents that are not user-facing docs and not source. Four kinds live here, and the distinction is
what stops "done" from being applied to something that was never going to be finished.

## The categories

- **`design/`** — **sketches.** A direction where the *workflow* is clear and the *mechanism* is not. Writing
  one as a brief would freeze decisions nobody has made yet. A sketch graduates by producing a brief, not by
  becoming one. Each carries a status line saying which parts are decided. See `design/README.md`.

- **One folder per sprint**, named for its scope — currently `librarian-extension/` and
  `researchers-night/`. Implementation briefs for work that has been decided on, each folder with
  its own `README.md` for ordering and its own `done/` for the ones that have closed. A sprint folder is a
  working set, so it also holds unnumbered briefs and its session records.

  **A sprint `README.md` is the sprint's decision log**, which is a different document from the folder
  introduction the name suggests: the schedule and the argument for it, what was cut and why, what each
  item turned out to need. It reads as an odd sort of README because most of what is in it produced no
  diff — a decision that produced a commit is recorded by that commit's message, and a decision that did
  not is recorded here or nowhere. Consulted when taking stock or picking the next item, rather than read
  through. Both sprint READMEs say so at the top.

  **Numbering is discontinued.** The 01–16 run is historical and keeps its numbers; new briefs are named for
  what they are. The numbers recorded the order briefs were *written*, which is not the order they are read,
  built or closed in — `researchers-night/` inherited 11–16 from that run and its own contents are ordered by
  the README's table, not by the filenames. A name that says what a brief covers survives being reordered,
  rescheduled, or moved between sprint folders; a number has to be defended against all three.

- **One folder per one-off pass** — the worked example is `done/todo-sweep-2026-08-10/`. It is a folder because the pass
  arrived as several files that belong together, which is the same reason anything else here gets one. It
  is not a sprint: no numbered implementation briefs, and the work is a single bounded pass over something
  that already exists rather than a run of features. Name it for the pass and the date it was scoped, and
  close it into `done/` whole, as a sprint folder does.

- **`done/`** — **closed briefs.** Implementation plans whose work has landed, plus session reports and code
  reviews of completed work. The test for this folder is that the document described something to *do*, and
  it got done. **A finished sprint moves here whole**, as its own directory, alongside the individual
  closed items.

### Why the grouping unit is the sprint, and not a state

Briefly tried and rejected on 2026-08-07: a single `active/` folder, emptying into a single `done/`. It is
tempting because a state name can never go stale the way `summer_2026_librarian_extension` did — the season
was wrong by August and the scope was wrong as soon as a Visualizer brief landed in it.

But it erases the sprint boundary permanently, and sprints are real: the 01–10 set is one run of
mostly-Librarian work, and losing that grouping when it closes loses the only thing that says those briefs
belong together. Naming a folder for its *scope* is not the same trap as naming it for a *season* — a
finished run's name is a historical fact, and stays accurate forever. It was the open-ended name that rotted,
because the run kept growing past its own description.

So: the sprint folder is the unit for its whole life, and closing a sprint is one `git mv` into `done/`
rather than a scatter.

**The numbers group; they do not order.** They ran 01–16 in the order the briefs were *conceived*, which
stopped matching the order they get *done* a while ago — 15 is ahead of 05 on timing rather than closure, and
the two orderings genuinely disagree. Read a number as "which run, roughly when conceived", not as a
position in a queue; each sprint's `README.md` carries the actual ordering and the argument for it. New
briefs need not take a number: `ligature-repair-brief.md` sits at the top level with none.

- **`reference/`** — **durable knowledge.** Documents that were never going to be "finished", because they
  describe how something *is* rather than what to build: external requirements (the EU AI Act summary),
  lookup tables (the DPG keycode mismatch), research notes on OS APIs, a decision record naming which models
  we develop against, and an archived style snapshot. Consulted, not completed.

  Note what is *not* here. Our own measurements went to `investigations/`, with the scripts that produced
  them. And a document that reads as a spec may still be a brief: `cherrypick-spec.md` is in `done/`, because
  it is the thing that produced the app.

The `reference/` split exists because filing the AI Act summary under `done/` was a category error: it is a
description of a regulation, and regulations do not get finished.

## Citing code and TODO items: by name, not by line number

**Cite a heading, a symbol or a function name. Not `file.md:1825`.** Audited across the open briefs on
2026-08-10, after brief 15 was found carrying three `TODO_DEFERRED.md` line numbers that had all drifted.

The failure is not that a pointer breaks. A broken pointer announces itself. **The dangerous case is the one
that still resolves**: brief 15's `TODO_DEFERRED.md:1825` had come to rest on a different item's real
heading, so it read as valid and nothing would ever have prompted a re-check. Brief 16 had four citations to
`TODO.md:480` for an item that now sits at `:504`, on a line that is blank.

`TODO_DEFERRED.md` is the worst host for a line number — it is edited constantly and is about to get
substantially shorter, so any number written into it is already wrong. But source files drift too, and a
symbol name (`scaffold.build_turn_prompt`, `chat_controller._render_text`) is both stabler and more
informative than a coordinate: it says *what* is being pointed at, so a reader who has to go looking can.

Where a line number genuinely helps — the exact line of a subtle expression — write it *alongside* the name,
so the name still works when the number stops.

### The exception: dated session records are not maintained, they are closed

A document like `design-session-2026-08-03.md` records what was true on a date. Its citations were correct
when written, and **repointing them falsifies the record** — the same reason `investigations/` keeps
measurements as they were made. Two consequences:

- Do not repoint them. A stale line number in a dated record is not a defect; it is the record aging, which
  is what records do.
- **They belong in `done/`**, which this README already says holds session reports of completed work. Leaving
  one in a sprint's working set invites exactly the maintenance it must not receive.

So an audit of citations should skip them, and the audit script should be pointed at the live briefs only.

A folder here may be a **bundle** rather than a single file, when a document has apparatus: `done/` holds
`dpg-markdown-bullet/` (the write-up plus the script that reproduces the bug) and `visualizer-refactoring/`
(the notes plus the one-shot rewriter that performed part of it). Same principle as `investigations/` below —
keep an artifact with what produced it — applied wherever the artifact happens to live.

## Related, elsewhere — and deliberately not here

- **`investigations/`** (repo root) holds things we measured, profiled or reproduced — **one directory per
  investigation, with its write-up, its scripts and its data together.** A measurement write-up does not live
  in `briefs/`, because separating it from the apparatus that produced it makes it unreproducible in practice
  however carefully it was written. See `investigations/README.md`.

  This is why `reference/` holds no measurements of ours: the audits and the context-inject sweep moved out to
  their bundles on 2026-08-03. What stays in `reference/` is what gets *consulted* rather than completed.

  **Authorship is not the test, and reading it as one gets `reference/` wrong.** *Every* document in there is
  ours: the AI Act summary written from the EU's own materials, the keycode table out of our own
  investigation, the keyboard-layout and model-lineup notes likewise, and the `unpythonic` style extract made
  from another of our own repositories. Not one of them is a foreign document we filed.

  The test is what the document *is*: a lookup table stays useful and unfinished forever, while the
  measurement that produced it belongs with its scripts. Where both exist, both are kept, in their own places
  — `dpg-keycodes.md` is the table you consult; the probes that established it are not filed beside it.
- **`TODO.md`** for planned work and **`TODO_DEFERRED.md`** for things noticed mid-task and set aside. Where
  a brief depends on one, it names it.
- **`dpg-notes.md`** (repo root) is the DearPyGui reference, kept at the root because it is consulted
  constantly rather than occasionally.

## Housekeeping

The loose `tools_*.py` scripts at this level are one-off helpers kept with the work that produced them;
`dpg_markdown_bullet_verify.py` is a reproduction script retained for a possible upstream bug report.

When a brief's work lands, move it to `done/` — and when moving anything, grep for its path first. Several
documents cite each other and are cited from source comments, so a move without a sweep leaves dead
references that read as real ones.

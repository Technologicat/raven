# Briefs and project documents

Working documents that are not user-facing docs and not source. Four kinds live here, and the distinction is
what stops "done" from being applied to something that was never going to be finished.

## The categories

- **`design/`** — **sketches.** A direction where the *workflow* is clear and the *mechanism* is not. Writing
  one as a brief would freeze decisions nobody has made yet. A sketch graduates by producing a brief, not by
  becoming one. Each carries a status line saying which parts are decided. See `design/README.md`.

- **`summer_2026_librarian_extension/`** — **the current sprint.** Numbered implementation briefs for work
  that has been decided on, with its own `README.md` for ordering and its own `done/` for closed ones. A
  sprint folder is a working set, so it also holds a couple of unnumbered briefs and the Monday checklist.

- **`done/`** — **closed briefs.** Implementation plans whose work has landed, plus session reports and code
  reviews of completed work. The test for this folder is that the document described something to *do*, and
  it got done.

- **`reference/`** — **durable knowledge.** Documents that were never going to be "finished", because they
  describe how something *is* rather than what to build: external requirements (the EU AI Act summary),
  measured behaviour (the performance audits, the context-inject measurements), reference tables (the DPG
  keycode mismatch), specs of shipped tools, and archived snapshots. Consulted, not completed.

The `reference/` split exists because filing the AI Act summary under `done/` was a category error: it is a
description of a regulation, and regulations do not get finished. The same argument covers the audits and the
measurement write-ups, which is why they moved together.

## Related, elsewhere — and deliberately not here

- **`evaluation/`** (repo root) is the *apparatus* for the measurements that `reference/` writes up:
  `evaluate.py`, `make_questions.py`, `avatar_footprint.py`, and the captured transcripts, results and logs.
  It stays at the root because it is runnable code with data, meant to be re-run against new models and new
  retrieval work — live tooling rather than a record. The pairing is worth knowing: `reference/` holds the
  conclusions, `evaluation/` holds what produced them and can produce more.
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

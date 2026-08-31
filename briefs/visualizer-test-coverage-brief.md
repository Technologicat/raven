# Brief: getting the Visualizer under test

**Filed 2026-08-31.** The package went from zero tests to three that day, all on
`importer._parse_input_files`. This is the plan for the rest, ordered so that a session can start at the
top and stop anywhere.

Not sprint work and not deadline-bound; it is the standing gap named in `raven/visualizer/CLAUDE.md` and
in the root `CLAUDE.md`'s coverage section.

**Intended as one of two parallel sessions** (Juha, 2026-08-31): this alongside a session starting
`16_chat-graph-view-brief.md`, which the sprint README has as next. The pairing is deliberate rather than
opportunistic — the two touch different subsystems, so neither is waiting on the other's tree, and they
are different enough to review side by side, which is the actual constraint.

## Why now, and why it is easier than it looks

The refactor that motivated wanting tests has landed, so what they would pin is no longer a rewrite in
flight but **the module boundaries the split created**, before feature work starts leaning on them. The
importer rework (brief 11, and `investigations/highdim-clustering/`) is about to change the clustering
stage, which makes the timing concrete rather than aspirational.

**The first test module already answered the questions that were keeping this at zero**, and they were
cheaper than expected — see `raven/visualizer/tests/test_importer.py`:

- `importer` imports cleanly under pytest, because everything expensive in it is lazy: the LLM
  connection is set up at import time only when the config asks for cluster keywords or summaries, and
  the NLP and embedding models load on first use.
- It does reach sklearn, torch and spaCy, which CI does not install, so the module carries
  `pytest.importorskip("raven.visualizer.importer")` and the `ml` marker. **Without the guard a
  module-level import failure is a *collection* error and reddens the matrix rather than skipping.**
  `python scripts/check_ci_imports.py` reports exactly this, and did.
- Config is monkeypatched per test (`monkeypatch.setattr(visualizer_config, "dehyphenate", False)`),
  which works because the module reads it at call time.

**DPG is not the obstacle** — see `dpg-notes.md`, "Testing DPG code", and `common/gui/tests/`, which
drives a real DPG context with an unmapped viewport. `vendor/file_dialog` has 175 tests over a 2900-line
DPG widget. Whatever has kept this package at zero, it is not the toolkit.

## Ordering, by measured difficulty

Measured 2026-08-31: SLOC excluding blanks and comments, and the number of `dpg.` calls, which is the
better proxy for how much a test has to stand up before it can assert anything.

| module | SLOC | `dpg.` calls | note |
|---|---|---|---|
| `entry_renderer.py` | 114 | **0** | **start here** — per-entry rendering shared by panel and tooltip |
| `app_state.py` | 46 | 1 | state containers; small enough to be a warm-up |
| `importer.py` | 874 | 0 | partly covered; the rest of the pipeline is plain functions |
| `word_cloud.py` | 185 | 11 | |
| `selection.py` | 179 | 18 | selection algebra (replace/add/subtract/intersect) is testable apart from the widgets |
| `plotter.py` | 183 | 22 | |
| `annotation.py` | 298 | 79 | |
| `info_panel.py` | 1078 | 172 | |
| `app.py` | 1350 | 448 | last, and possibly never — an entry point is wiring |

`config.py` needs no tests (configuration-as-code, and it carries local overrides). `importer_cli.py`
looks easy on this table but **check whether it calls `parse_args` at module scope before trying** —
that pattern makes a module unimportable under pytest, and is why `raven/cherrypick/preload.py` exists
apart from its `app.py`.

## What to aim the first few at

Prefer assertions about **what Raven decides**, not what a library does — the house style visible in
`test_docextract.py`. For this package that means:

- `entry_renderer`: which fields appear, in what order, and what happens to a record missing an
  abstract or an author. The panel and the tooltip share this vocabulary, so pinning it is what makes
  the two safe to change independently.
- `selection`: the four combination modes, and that undo restores exactly what was there. This is set
  algebra over index arrays and needs no GUI at all.
- `importer`: the remaining pipeline stages. `_cluster_*` and `_reduce_dimension` are about to be
  rewritten (brief 11 item 5), so **write those tests as part of that change, not before it** — a test
  of the current two-stage clustering pins a defect.

## The rule that matters most here

**Check every new test against the code without the fix or the behaviour it claims to pin** —
`git stash push <file>`, run, confirm it fails *with the reported symptom*, `git stash pop`. Two of the
three existing tests were verified this way, and the third is their negative control. That discipline
caught a real fixture bug while the module was being written: a predicate matched on
`"Alpha" in author_field` where bibtexparser hands over a parsed name list rather than a string, so it
silently matched nothing and the test passed against the unfixed code.

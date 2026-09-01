# Brief: getting the Visualizer under test

**Filed 2026-08-31.** The package went from zero tests to three that day, all on
`importer._parse_input_files`. This is the plan for the rest, ordered so that a session can start at the
top and stop anywhere.

Not sprint work and not deadline-bound; it is the standing gap named in `raven/visualizer/CLAUDE.md` and
in the root `CLAUDE.md`'s coverage section.

**Intended as one of three parallel threads** starting the day after 2026-08-31 (Juha's call): this,
`briefs/researchers-night/16_chat-graph-view-brief.md`, and
`briefs/researchers-night/aokk-corpus-scope-classification-brief.md`. The split is deliberate rather than
opportunistic — the three touch different subsystems, so none waits on another's tree, and they are
different enough to review side by side, which is the actual constraint. Of the three, only the graph
view is bound by the Researchers' Night deadline. `briefs/researchers-night/README.md` under *Ordering*
is the copy of this that a reader will find first.

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
| `entry_renderer.py` | 114 | **0** | **done 2026-09-01**, 27 tests — per-entry rendering shared by panel and tooltip |
| ~~`app_state.py`~~ | ~~46~~ | ~~1~~ | **nothing to test.** Three lines of code (`app_state = env()`) under 46 of docstring; the SLOC figure counted the prose |
| `importer.py` | 874 | 0 | partly covered; the rest of the pipeline is plain functions |
| `word_cloud.py` | 185 | 11 | **done 2026-09-01**, 28 tests — the two render guards, and the save dialog |
| `selection.py` | 179 | 18 | **done 2026-09-01**, 36 tests — the selection algebra and the undo history |
| `plotter.py` | 183 | 22 | **done 2026-09-01**, 26 tests — the cluster sort, and the plotter-space queries |
| `annotation.py` | 298 | 79 | **done 2026-09-01**, 17 tests — the guards; the content build is not covered |
| `info_panel.py` | 1078 | 172 | **done 2026-09-01**, 31 tests — the decisions; the geometry is not covered |
| ~~`app.py`~~ | ~~1350~~ | ~~448~~ | **out of scope** (Juha, 2026-09-01) — see below |

**`app.py` is not on the list, and that is settled rather than deferred.** It lays out the GUI, wires the
animations and the event handlers, and boots the app up; the general Raven rule is that an `app.py`
should not be doing anything else, so anything found in there worth testing is a sign that it belongs in
another module, not that the entry point needs a test. Testing it would also mean testing the wiring
against itself. So `info_panel.py` is the last module in this plan, not the second to last.

**What was still in `app.py` that should not have been was a separate document**,
`done/visualizer-app-py-extraction.md`, written from a survey done while these tests were, and closed the
same day. Its argument was this one turned around: `app.py` cannot be imported under pytest, so everything
left in it is untestable by construction, and each extraction converts code that cannot be tested into code
that can. It moved the BibTeX importer's GUI and the info panel's header and navigation bar out, taking
`app.py` from 1763 lines to 1344 and this package from 4 of 9 test modules running in CI to 8 of 9.

**The `dpg.` count is the right proxy, but it overstates the cost where the calls are few and simple.**
`selection`'s eighteen are four distinct calls — `enable_item`, `disable_item`, `set_value`,
`is_key_down` — so `test_selection.py` monkeypatches a recording stand-in over the module's `dpg`
binding, delegating everything else to the real toolkit so the key constants stay DPG's own. Assertions
are then about what the module asked the GUI to do. That works because those calls are *commands*, not
queries: nothing in `selection` reads geometry back. A module that measures a widget wants a real
context instead, which per-module is fine (`dpg-notes.md`, "Testing DPG code") — so read the count as
"how many distinct things does this ask DPG for", not as a line total.

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
  - **`_collect_cluster_keywords`'s `llm` branch is under repair as of 2026-09-01** and joins that list
    for the same reason. It joins every entry of a cluster into one prompt with no cap, which on a
    crowded corpus is a prompt the backend either refuses or silently truncates — so the largest
    cluster, the one most worth labelling, gets keywords that are wrong or absent. The fix caps the
    prompt. Anything asserted about the current shape would pin the unbounded one.

## The rule that matters most here

**Check every new test against the code without the fix or the behaviour it claims to pin** —
`git stash push <file>`, run, confirm it fails *with the reported symptom*, `git stash pop`. Two of the
three existing tests were verified this way, and the third is their negative control. That discipline
caught a real fixture bug while the module was being written: a predicate matched on
`"Alpha" in author_field` where bibtexparser hands over a parsed name list rather than a string, so it
silently matched nothing and the test passed against the unfixed code.

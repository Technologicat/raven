# Where FileDialog's open and close time actually goes

Measured 2026-08-13, against `raven/vendor/file_dialog/fdialog.py`.

Two symptoms had been filed together since 2026-07-18: opening the dialog on a directory of thousands of
files took "a couple of seconds", and immediately after *closing* it the opener button appeared dead for a
similar interval — not even its own acknowledgement flash fired. Both were attributed to the listing being
fully materialized as DPG widgets, with virtualization and listing reuse proposed as the fixes.

**Building the listing was never the expensive part.** The close-side symptom had a different cause than
the one filed, and the open-side "couple of seconds" is not reproducible at all.

## What was measured

All numbers from this machine, `dearpygui` 2.3.1.

**Building and deleting rows** (`bench_fdialog.py`, headless — item creation only, no rendering):

| files | stat only | `reset_dir` | delete | DPG items | per row |
|---|---|---|---|---|---|
| 100 | 0.000 s | 0.006 s | 0.000 s | 469 | 4.7 |
| 500 | 0.002 s | 0.030 s | 0.002 s | 3504 | 7.0 |
| 1000 | 0.003 s | 0.059 s | 0.006 s | 7004 | 7.0 |
| 2000 | 0.007 s | 0.119 s | 0.022 s | 14004 | 7.0 |
| 4000 | 0.013 s | 0.239 s | 0.085 s | 28004 | 7.0 |

Linear, about 60 µs per row, seven DPG items each. The filesystem part is 5% of it.

**The real directory** (`bench_real.py`), 2520 entries of actual documents rather than synthetic 64-byte
files: **0.174 / 0.190 / 0.192 s** over three trials, and `FileDialog` construction — every icon texture,
theme and handler — is 0.006 s. So the synthetic measurement was not flattering; a real directory behaves
the same.

**Re-applying the sort** (`bench_sort.py`). `reset_dir` ends in `reapply_latest_sort`, which re-runs the
table sort callback whenever the user has ever clicked a column header, reading four DPG items back out of
the tree per row. It adds about 13% — 0.051 s at 4000 files — and only after the first header click, which
is why a fresh-session measurement never sees it.

**Rendered frame cost** (`bench_render.py`, real window, vsync off, median over 200 frames). This is the
one that found something:

| rows | `clipper=False` | `clipper=True` |
|---|---|---|
| 0 | 0.74 ms | 0.77 ms |
| 500 | 1.10 ms | 1.01 ms |
| 2500 | **3.76 ms** | **0.68 ms** |

ImGui submits every row of a table each frame unless the table clips to the visible range. At 2500 rows a
clipped table costs what an empty one costs.

## What was wrong, and what was fixed

**The close path rebuilt the listing two or three times over.** `ok` called `reset_dir` twice — once via
`_update_search`, once directly — and `cancel` once, for rows that are already hidden and that the next
`show_file_dialog` rebuilds regardless. At 0.19 s a rebuild, closing with OK spent ~0.4 s producing
nothing.

**And that is the dead-button symptom, by a different mechanism than the one filed.** Input was not being
swallowed by a modal tearing down its children. DPG runs Python callbacks one at a time on a single
callback thread (`dpg-notes.md` → "How callbacks are dispatched"), so the opener's callback was *queued
behind* the close that was still rebuilding — which is exactly why the button's own flash did not fire
either: the flash is inside the callback that had not started.

Fixed by `_forget_listing`, which clears the two Python lists and leaves the widgets alone. A hidden window
renders nothing, and `reset_dir` deletes them as its first act on the next open, so nothing is gained by
deleting them earlier.

**The table now sets `clipper=True`**, removing the per-frame cost above. Its one requirement is uniform
row height, which holds because every cell is created with `height=self.selec_height`. Sorting, scroll
extent and row alignment were checked by screenshot afterwards and are unaffected.

## What is still unexplained

**Nothing measured here reaches "a couple of seconds" on the open path.** Opening a 2520-entry directory
costs 0.19 s to build plus a few milliseconds of first frame. Either the original report over-estimated a
sub-second delay — plausible, since the close-side stall was real and adjacent — or something in the live
app is absent from this apparatus. The next step is a live re-test now that the close path is cheap; if the
open still feels slow, the cause is not in the listing build and this data says where not to look.

One factor genuinely unmeasured: a **cold page cache** on the first open of a session. Dropping caches
needs root, so every measurement here is warm.

## A DPG hazard found on the way

Comparing two configurations in one process does not work: with a `FileDialog` in the picture, destroying
and recreating the DPG context segfaults **3/8 to 8/8 of runs**, nondeterministically. Bare context cycles
are clean even with 60 rendered frames on a shown viewport (`probe_minimal.py`), so this is not context
recreation as such.

`probe_bisect.py` runs one configuration per subprocess and takes the variant name as an argument; run each
several times, because at these rates a single trial says nothing — the first bisection attempt produced a
table implying that *removing* ingredients caused the crash, which was the 1-in-8 survival.

The mechanism is not identified, and the write-up in `dpg-notes.md` → "Context recreation is not reliably
safe once real widgets have rendered" says so. It costs nothing at runtime (an app holds one context for
its life) and nothing in the test suite (one module-scoped context, no rendered frames). It costs
benchmarks one subprocess per configuration.

## The scripts

| Script | What it answers |
|---|---|
| `bench_fdialog.py` | What do building and deleting the listing cost, per row, headless? |
| `bench_real.py` | Does the real documents directory behave like the synthetic one? |
| `bench_sort.py` | What does re-applying the column sort add on top of a rebuild? |
| `bench_render.py` | What does a listing cost per rendered frame, with and without the table clipper? Takes `clipper` or `noclipper`; needs a mapped window, so it takes keyboard focus while it runs. |
| `probe_minimal.py` | Does bare context recreation survive rendered frames? Takes `frames` or `noframes`. |
| `probe_bisect.py` | Which ingredient turns a `FileDialog` context cycle into a segfault? (Answer: none in isolation.) Takes a variant name, or runs all of them one subprocess each. |

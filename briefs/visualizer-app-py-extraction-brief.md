# Brief: finishing the Visualizer's `app.py` split

**Filed 2026-09-01**, from a survey done while writing the package's tests. Not deadline-bound, and not a
bug list: nothing here is broken. It is the remainder of the refactor that produced `info_panel`,
`selection`, `plotter`, `annotation`, `word_cloud`, `entry_renderer`, `app_state` and `search`.

## What the file is for, and how far off it is

The project's rule is that an `app.py` is an OS entry point and a wiring skeleton: parse the command
line, build the GUI, instantiate the objects, run the DPG render loop. Anything worth calling from
elsewhere — or worth testing — belongs in another module, beside the thing it operates on.

Measured before this session's extractions, `app.py` was ~1925 lines, of which **roughly 56-60% was
legitimately wiring and ~38% had a better home**. Four of those items have since moved:

| moved to | what |
|---|---|
| `search.py` (new) | the search scan, its two boxes, the field colouring |
| `plotter.py` | `compute_highlight_alpha`, the select-radius brush indicator |

`app.py` is now ~1763 lines. **The two big items are still there**, and they are most of what remains.

## The remaining work, in the order worth doing it

### 1. `importer_gui.py` — the largest, and the one with a knot in it

~315 lines, and a whole second application UI: its own state, two file dialogs, a per-frame status
poller, start/stop lifecycle with a button-disabling protocol against double-clicks. It belongs beside
`importer.py`, which is the pipeline and has no GUI.

Two distant regions of `app.py`:

- **the logic**, currently under the `# BibTeX importer integration` banner: `importer_input_files_box`
  and `importer_output_file_box`, the `importer_action_start` / `importer_action_stop` symbols,
  `show_importer_window`, the file-dialog callbacks, `import_bibtex_files`, `update_importer_status`,
  the `started`/`done` task callbacks, `start_importer`, `stop_importer`, `start_or_stop_importer`. It
  ends where the `# Animations, live updates` banner begins.
- **the layout**, the `with dpg.window(..., tag="importer_window")` block, ending just before the
  `Done in ...` log line that closes the GUI build.

**The move is the easy part. The knot is this:** `update_open_import_gui_table` and
`update_save_import_gui_table` are *defined inside* the `with dpg.window(tag="importer_window")` layout
block, and *called from* module-level callbacks defined hundreds of lines earlier. Those are forward
references that resolve only because the callbacks fire after the layout has run — so the two halves
cannot simply be concatenated in either order without deciding what those two functions are.

They are almost certainly a component's methods in disguise. The same pattern appears three more times
in the file (`select_search_results`, `select_visible_all`, `toggle_fullscreen` are each defined inside a
layout block and called from a hotkey handler), so whatever shape is chosen here is worth choosing
deliberately: it is the pattern for the rest.

**Do this one in a fresh session.** Untangling the definition order needs the whole component in view.

### 2. The info panel's header and navigation controls (~126 lines)

`info_panel.build_window()` builds the *content* group, but the header child window
(`item_information_header`) and the whole navigation control bar
(`item_information_navigation_controls` — top / page up / page down / bottom / prev match / next match,
and their `[x/x]` indicators) are built in `app.py`. The comments there admit the split ownership:

> `# The callback function is bound in 'info_panel.build_window()'.`
> `# The callback functions for all buttons in this group are defined (and bound) later when we define the info panel.`

So `app.py` creates widgets whose behaviour lives in `info_panel.py`, which then reaches back for them by
tag. **This is a judgement call rather than a defect** — "all layout in one place" is a defensible rule
too — which is why it is second rather than first. Lowest priority of the three.

The same argument applies, much more weakly, to the word cloud window's layout (~14 lines).

### 3. Help-card content (~84 lines)

`hotkey_info` is 40 lines of pure static data — `env(key=..., action=...)` rows — that computes nothing
and touches nothing. `render_help_extras` is 41 lines of prose *about how search works*, including the
matching semantics ("a **lowercase** fragment matches case-insensitively"), which will fall out of sync
with `search.find_matches` the moment either changes. That prose now has a module to sit beside.

`config.py` is already configuration-as-code and already holds GUI settings, so `hotkey_info` could go
there; or both could go to a `help_content.py`. `render_help_extras` calls `dpg_markdown.add_text`, so
it is a move rather than a purification. The `HelpWindow` construction stays in `app.py` as wiring.

**Not in scope: the hotkey strings duplicated by hand into ~20 tooltips.** `app.py` states that as a
decided design call — no shared keymap, the surfaces that make hotkeys discoverable mirror them by hand
(KISS) — so it is a decision, not drift.

## Two smaller items, and one deliberate non-item

- **The right-click jump-target computation** (~30 lines in `mouse_click_callback`) is a three-way set
  intersection over `annotation.data_idxs`, `info_panel.entry_title_widgets` and the search results,
  holding two modules' content locks in a fixed order. That lock ordering is invisible from anywhere
  else. **Decision 2026-09-01: it stays in `app.py`** — it is the app coordinating three subsystems,
  which is what an app is for. Recorded so it is not re-litigated.
- **Sixteen `# TODO: DRY duplicate definitions for labels`** are scattered through the file, on strings
  that appear in two or three places each. Still open, mechanical, and unrelated to any extraction.
- **`reset_app_state` reads either way.** Every line of it is a call into another module, which is
  wiring — but it also encodes a policy (which animations survive a dataset reload versus an app exit;
  that the search is cleared while the highlight series is rebuilt) that is not obviously app-level. Left
  alone, and flagged here so the next reader knows it was considered rather than missed.

## What good looks like

After item 1 alone, `app.py` is ~1450 lines and very nearly all wiring. After all three, ~1250 —
at which point the guideline conversation is about `info_panel.py` instead, which is the package's
largest module and the next split candidate on its own merits.

**None of this is urgent, and none of it is a bug.** The reason to do it is that `app.py` cannot be
imported under pytest — it runs `parse_args` at module scope — so everything left in it is untestable by
construction. That is the whole argument: each extraction converts code that cannot be tested into code
that can, which is exactly what happened to the search scan and the brush geometry, both of which got
their first assertions the day they moved.

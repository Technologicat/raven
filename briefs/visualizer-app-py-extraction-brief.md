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
| `importer_gui.py` (new) | the whole BibTeX importer UI — both regions, and its two file dialogs |

`app.py` is now ~1440 lines. **One big item is left**, plus the help-card content.

**The knot in the importer extraction dissolved once the two halves were in one file**, and the shape it
settled on is the package's existing one rather than a new one: *the module is the component*. The two
table updaters — `update_open_import_gui_table` and `update_save_import_gui_table`, formerly defined
inside the layout block and called from callbacks hundreds of lines earlier — are now module-level
privates defined before both, closing over nothing. That works because `word_cloud`, `search`,
`annotation`, `selection` and `plotter` are all singletons-as-modules already: there is one importer
window, so there is nothing for an instance to distinguish.

**So the pattern for the remaining three sites** (`select_search_results`, `select_visible_all`,
`toggle_fullscreen`, each defined inside a layout block and called from a hotkey handler) is: hoist to
module level in the module that owns the widget, and have `build_window` call it. No class needed.

## The remaining work, in the order worth doing it

### 1. The info panel's header and navigation controls (~126 lines)

`info_panel.build_window()` builds the *content* group, but the header child window
(`item_information_header`) and the whole navigation control bar
(`item_information_navigation_controls` — top / page up / page down / bottom / prev match / next match,
and their `[x/x]` indicators) are built in `app.py`. The comments there admit the split ownership:

> `# The callback function is bound in 'info_panel.build_window()'.`
> `# The callback functions for all buttons in this group are defined (and bound) later when we define the info panel.`

So `app.py` creates widgets whose behaviour lives in `info_panel.py`, which then reaches back for them by
tag — seven `set_item_callback` calls, and a `build_window` docstring that is a precondition list nothing
enforces. **This was a judgement call rather than a defect** — "all layout in one place" is a defensible
rule too — which is why the importer went first.

**Decided 2026-09-01: move it, as two functions called from inside `app.py`'s layout tree** —
`info_panel.build_header()` and `info_panel.build_navigation_controls()`, each where its child window sits
now. What the "one place" rule is really defending is layout *order*, and that stays visible in `app.py`;
only the contents of two boxes move. Same shape as `importer_gui.build_window()`.

Two details settled with it:

- **`markdown_font_loader_trigger_dummy` stays in `app.py`.** It is a startup workaround for a DPG
  Markdown font-loading race, in the header only because it needed a parent that existed at that moment.
  It parents into `item_information_header` by tag after `build_header()` returns.
- **`app_state.copy_report_tooltip` becomes module-local to `info_panel`**, which is its only reader. The
  `app_state` table loses a row, which is what its own docstring anticipates for state that finds a more
  natural home.

The same argument applies, much more weakly, to the word cloud window's layout (~14 lines).

### ~~2. Help-card content (~84 lines)~~ — decided 2026-09-01: it stays in `app.py`

`hotkey_info` (40 lines of `env(key=..., action=...)` rows) and `render_help_extras` (41 lines of prose,
about half of it describing how search matches) were proposed for a `help_content.py`. **Juha's call: no
move.** The help card is GUI wiring, and it is specific to this one app, which is what `app.py` is for.

The concern that motivated the proposal was drift — the prose states `search.find_matches`'s rules, and
would go stale the moment either changed. That is now handled where it belongs instead: `test_search.py`
pins case-insensitivity for a lowercase fragment, the uppercase escape hatch, order-independence and
subscript normalization, so the claims the card makes are checked whether or not the prose sits next to
them. Relocating the text would not have checked anything.

Checked at the same time, since a stale claim would have changed the answer: every assertion the card
makes about search is currently true, including both worked examples (`"hydrogen"` matching
*"Hydrogen"*, `"TiO"` matching titanium oxide but not *"bastion"*).

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

The importer extraction landed `app.py` at 1439 lines, close to the ~1450 predicted, and very nearly all
wiring. After the remaining two, ~1250 — at which point the guideline conversation is about
`info_panel.py` instead, which is the package's largest module and the next split candidate on its own
merits.

**None of this is urgent, and none of it is a bug.** The reason to do it is that `app.py` cannot be
imported under pytest — it runs `parse_args` at module scope — so everything left in it is untestable by
construction. That is the whole argument: each extraction converts code that cannot be tested into code
that can, which is exactly what happened to the search scan and the brush geometry, both of which got
their first assertions the day they moved.

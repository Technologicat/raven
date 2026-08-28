---
name: dpg
description: Raven's DearPyGui reference — threading and callback dispatch, `split_frame` deadlocks, texture upload ordering, widget/tag lifecycle, font atlas limits, keyboard input traps (stale `mvKey_*` codes, focus vs. caret), scrolling, window sizing, and testing DPG headless. Use BEFORE writing or editing anything that imports `dearpygui` — the render loop, event/key/mouse handlers, texture or `split_frame` work, widget creation from background threads, hotkeys, fonts, themes, or DPG tests — and before asserting anything about how DPG behaves.
---

# DearPyGui in Raven

**`dpg-notes.md` in the repo root is the reference. This skill is the index into it.**

The notes are ~760 lines across 44 sections of lore that mostly cost a debugging session each to learn, and
they are what a person reads in an IDE. Nothing here restates them — routing you to the right section is the
whole job. Read the section rather than guessing from its heading: several of these are counter-intuitive,
which is why they are written down.

`CLAUDE.md` carries a numbered index of the eight worst pitfalls, as a safety net for when this skill does
not fire. That index is a *warning list*; the table below is a *map*.

## The standing instruction

**Measure a DPG claim before writing it down.** Not before *acting* on one — before committing it to a code
comment, a docstring, a note or a commit message. A wrong "why" attached to working code is worse than no
why, because it reads as checked and gets believed by whoever arrives next.

This is cheap in a way that makes refusing it hard to justify: a headless context (`create_context` /
`create_viewport` / `setup_dearpygui`, no `show_viewport`) answers most behavioural questions in one short
probe and a few seconds — does this callback fire, does that setter take effect, what does this getter
return before a frame renders.

Live cases, all from a single day: "`set_value` fires no callback" was believed from two code comments and
had never been checked (true, as it turns out); "`configure_item(default_value=...)` is creation-time only"
was *invented* to justify a change and is false; "`setup_dearpygui` segfaults on a second context" was a
plausible reading of a crash whose real cause was a stale class-level texture cache. Three claims, three
one-line probes, and only the first survived.

**Before writing a probe, check whether the question is already answered.** `investigations/README.md` is
seventeen lines, one per bundle saying which question it answers; `investigations/dpg-focus/` alone holds
six probes and a write-up covering focus versus caret, what `focus_item` does to a child window, and which
predicate an Enter handler may be gated on. Re-measuring costs an hour; the worse outcome is a *second* answer to a settled
question, phrased differently, for the next reader to reconcile.

If a probe captures an invariant worth keeping, promote it to a test rather than leaving it in `/tmp` — see
`raven/common/gui/tests/` for the shape. A probe that needs a click, a keystroke or a drag stays a probe,
and belongs in the closest existing bundle rather than a new one.

**A probe asking "how big can this be" is the exception, and it must climb.** The cheapness above holds for
behavioural questions — does this fire, what does that return. It does not hold for limits: a probe that
jumps straight to the size the feature wants can wedge the machine rather than answer. One that asked
whether a drawlist could be 60800 px tall took the X session down, recoverable only from a text terminal;
every API-level signal had said it was fine. Start an order of magnitude below what you need and increase
until something gives. A probe is allowed to fail; it is not allowed to take the user's desktop with it.

## Where the answer lives

Each target below names a top-level section as `*Section*`, optionally a subsection after `→`, and any
commentary in a trailing `(…)`. `·` separates targets. That shape is not decoration —
`.claude/skills/dpg/check_router.py` parses it and fails if a name no longer resolves against
`dpg-notes.md`'s headings, which is what stops this table from rotting the next time a heading is reworded.

| If you are about to… | Read, in `dpg-notes.md` |
|---|---|
| create widgets, set values or make textures off the main thread | *Threading* → Thread architecture · *Raven DPG app structure* → Background work and thread safety |
| call `split_frame` anywhere at all | *Threading* → `split_frame()` mechanism · *Threading* → Use `guiutils.split_frame`, not `dpg.split_frame` |
| register a frame callback | *Threading* → `set_frame_callback` holds one callback per frame number |
| wire a callback that has to know *which* item it came from | *Threading* → A callback is passed as many arguments as it declares (the loop-variable lambda is shadowed by the sender; use `user_data`) |
| handle a file drop, or write any GLFW-side callback | *Threading* → GLFW callbacks are the exception: they run *on* the render thread |
| diagnose a hang with no traceback, or a background-task race | *Threading* → Three-way deadlock pattern · *Threading* → Diagnosing background-task races |
| work out why hotkeys stopped responding, while the app is plainly still alive | *Threading* → Hotkeys dead while the mouse wheel still scrolls means the *callback thread* is blocked |
| reach for `dpg.mutex()` | *Threading* → `dpg.mutex()` — the atomicity tool that Raven cannot currently use |
| upload, replace or delete a texture | *Threading* → Texture upload ordering · *Raven DPG app structure* → Textures |
| build a widget that will later be rebuilt or swapped | *Raven DPG app structure* → DPG item management (version-counted tags; alias rebinding) |
| set a widget's value from code | *Raven DPG app structure* → DPG item management (no callback fires; `configure_item` vs `set_value`) |
| add or change a hotkey | *Keyboard input* → `mvKey_*` constants vs. runtime codes (the 517/518 trap) · *Keyboard input* → Same-frame dispatch is by keycode, not press order |
| gate a hotkey on a text field having the caret | *Keyboard input* → Focus is not the same as the caret: gate hotkeys on `is_item_active` (mind the commit-chord exception) |
| define what Tab does, or keep your own "which control has the keys" state | *Keyboard input* → Tab reaches a global handler and still moves ImGui's nav, after a programmatic focus (the activate/deactivate pair nobody asked for) |
| park focus on a panel or child window | *Keyboard input* → `focus_item` cannot focus a child window — and does harm when asked to |
| scroll programmatically, or follow a growing log | *Scrolling* → Three input paths move a scroll position, and DPG surfaces them differently · *Scrolling* → `max_y_scroll` moves when content is added |
| set a window's size, or a tooltip's padding, or fight z-order | *Window sizing* (all of it) |
| park a window offscreen to measure it before placing it | *Window sizing* → An offscreen park lasts exactly one frame — ImGui pulls the window back (re-park per frame; a modal comes back *fully* on screen) |
| give a widget a tooltip whose text changes, or wonder why one flickers when it does | *Window sizing* → An autosize window is one frame behind its content — and whether that shows depends on the window's age (the answer is `raven.common.gui.tooltip.Tooltip`; the section is why) |
| touch fonts, icons, super/subscripts or `dpg_markdown` | *Font atlas limits* (all of it) |
| size a drawlist, or build anything that scrolls over a lot of content | *Drawlists* → Never size a drawlist to a scroll extent — it will take the X session down |
| place a drawn shape at a particular spot inside a window | *Drawlists* → A drawlist ignores `pos`, and reports back the position it was asked for (there is no floating drawlist; reach for a theme or an overlay window) |
| divide a toolbar or a row of controls into sections | *Drawlists* → There is no vertical separator, and `add_separator` inside a horizontal group is not one (`guiutils.add_toolbar_separator`; its `horizontal=` names the toolbar, not the line) |
| draw a decoration beside a widget — a bullet, a rule, a background box — or defer work until a position is readable | *Drawlists* → Needing `get_item_pos` to place a decoration means it is in the wrong container (metrics are available at build time; positions are not, and wanting one is the signal to change containers) |
| ask what a drawlist actually drew — counting draw items in a test or a probe | *Drawlists* → A drawlist's children are in slot 2, and slot 1 answers "nothing was drawn" |
| highlight a widget, or give one an indicator it can be shown and hidden | *Themes* → A theme bound to a container reaches its children, and a child's own theme still applies · *Themes* → A border is drawn inside the item rect, so turning one on moves nothing (`raven.common.gui.keyboardmark`) |
| build a table, or wonder why a long listing costs frame time | *Tables* → Rows are submitted every frame unless the table clips |
| fill rows lazily — thumbnails, previews, anything per-row and expensive | *Tables* → To find which rows are on screen, ask a cell — never the row |
| ask where on the screen a widget is — to place something beside it, or to aim a driven click at it | *Testing DPG code* → Windows and child windows have no `rect_min`, and `get_item_pos` answers a different question (`guiutils.get_widget_pos`; `CLAUDE.md` → "Live GUI testing on a shared desktop" carries the screen-coordinate half) |
| write a test that drives DPG | *Testing DPG code* → DPG runs without a mapped window, so GUI code is unit-testable (mind the headless ceiling: no `render_dearpygui_frame`, so no layout) · *Testing DPG code* → Context recreation is not reliably safe once real widgets have rendered (one module-scoped context per module, and the `--run-gui` group shares the session-scoped one in the root `conftest.py`) |
| measure a widget in pixels — sizing a window to its content, checking something fits | *Testing DPG code* → A probe that measures anything must load the app's fonts first · *Testing DPG code* → Windows and child windows have no `rect_min`, and `get_item_pos` answers a different question (the stale-subtraction trap) · *Tables* → A table's column widths take a second frame to settle, and text wrapping follows them |
| drive a running app with synthetic key input, or wonder why a chord the app should answer does nothing | *Keyboard input* → `is_key_down` is sampled when the callback runs, not when the key was pressed (hold modifiers across frames, and mind how briefly a synthetic key is held; `CLAUDE.md` → "Live GUI testing on a shared desktop" carries the focus-and-etiquette half) |
| benchmark two GUI configurations, or otherwise build a second context in one process | *Testing DPG code* → Context recreation is not reliably safe once real widgets have rendered |
| start a new app, or wire its startup sequence | *Raven DPG app structure* → Startup sequence · *Raven DPG app structure* → Layout and GUI |

## Adjacent material, not in the notes

- `briefs/reference/dpg-keycodes.md` — the full stale-keycode table and its reproduction.
- `investigations/dpg-focus/` — probes behind the focus-vs-caret rules.
- `investigations/dpg-overlays/`, `investigations/dpg-dnd/` — overlay stacking, and the drag-and-drop mechanism.
- `raven/common/gui/` — Raven's own widgets and the `guiutils` wrappers the notes tell you to prefer.
- `CLAUDE.md` → "Live GUI testing on a shared desktop" — driving a *running* app steals the user's keyboard;
  read it before launching anything or injecting input.

## When you learn something new

Record it in `dpg-notes.md`, in the section it belongs to — that file is authoritative and this skill must
not grow a second copy. Add a row here only if it introduces a *new question* a reader might arrive with;
a new fact inside an existing section needs no new row.

Then run `python .claude/skills/dpg/check_router.py`, which fails on any citation or path that no longer
resolves. Cheap insurance against the one way this file rots: a reworded heading leaves the row reading
perfectly while sending the reader nowhere.

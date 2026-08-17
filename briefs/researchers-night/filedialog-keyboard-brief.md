# FileDialog: keyboard accessibility

**Status: designed in full, unbuilt.** The design below was settled on 2026-08-13 (Juha and Claude) and is
agreed; what remains is building it, plus three live checks listed at the end. Moved out of
`TODO_DEFERRED.md` on 2026-08-13, where a 164-line settled specification was the largest thing in a file
meant for items noticed and parked.

One of the two final FileDialog pieces for Researchers' Night, with `filedialog-thumbnails-brief.md`.
They touch the same widget and should be built with each other in view — the grid view needs the same cursor
and selection machinery this brief defines, and the type-filter hotkeys here are what switch the grid on.

## Why

`FileDialog` cannot be operated from the keyboard. The listing has no cursor at all — selection happens only
through mouse clicks on per-cell selectables, with a hand-rolled double-click timer — so there is no way to
move through a directory, descend into a folder, or toggle a selection without a pointing device. The five
hotkeys that exist (`fdialog_hotkeys_callback`: Enter, Esc, F5, Ctrl+Home, Ctrl+F) cover the frame around the
listing and not the listing itself, and the source has carried a standing `# TODO: Add hotkeys to navigate
up/down in the table, descend into folder, ...` since adoption.

**This is an equality consideration first.** It is also a straightforward usability one: DPG is mouse-centric
by default, Raven supports keyboard operation wherever reasonably possible, and this dialog is the part that
has not. The bar to clear is bash's filename completion, or a file manager with find-as-you-type and arrow
navigation.

## The design

**Focus lives in the find field**, which doubles as the filename field in save mode. Being a single-line
entry it leaves *up and down* free for the listing; left and right stay with the text caret, so the design
must not want horizontal navigation, and does not. Nothing Tab-cycles. Two other widgets are reachable, each
by explicit key rather than by a focus order.

**The governing rule for Enter: *Enter goes as deep as it can; Ctrl+Enter stops here.*** Uniform across all
modes. In open-file mode Enter on a file has nothing deeper, so it accepts — which is why the rule reads as
one sentence rather than a table of special cases.

| key | action |
|---|---|
| Up / Down | move the cursor one row |
| Page Up / Page Down | move *most of* a visible page, as in the Librarian chat log and the Visualizer info panel |
| Home / End | first / last row |
| Enter | descend into the cursor directory; on a file (open mode) accept it; on `..` go up |
| Ctrl+Enter | commit here — same as the OK button. In `dirs_only`, accepts the cursor directory |
| Esc | cancel |
| Ctrl+Space | toggle the cursor row's selection (multi-selection mode only) |
| Alt+Up | up one level |
| Ctrl+Up | up one level, one-handed alias — see below |
| Ctrl+Home | back to the default directory (exists) |
| F5 | refresh (exists) |
| Ctrl+F | focus the find field (exists) |
| Ctrl+L | focus the path field; Enter navigates, Esc returns to find |
| Ctrl+1 … Ctrl+9 | select the Nth offered type filter |
| Ctrl+T | focus the type filter combo; Up / Down / Home / End then cycle it — see below |
| Ctrl+B | focus the places panel — the folder shortcuts and drives; Up / Down / Home / End then move within it, Enter goes there — see below |
| Ctrl+Shift+1 … Ctrl+Shift+4 | sort by the Nth criterion — Name, Date, Type, Size — see below |
| F1 | open the dialog's help card — see below |
| Tab | complete in the field — see below |

**Why Ctrl+Up exists alongside the standard Alt+Up.** On a Nordic layout Alt sits only to the *left* of
space — the right-hand key is AltGr, which is a different key — so Alt+Up needs two hands. Ctrl is mirrored
on both sides, so right Ctrl and the arrow cluster are both under the right hand. Alt+Up was the only
two-handed chord in the table: Ctrl+Enter, Ctrl+Home and Ctrl+End already fall under the right hand, and
Ctrl+F / Ctrl+L / Ctrl+Space / Ctrl+1…9 are all reachable with the left alone. One alias fixes the set.

**The type filter follows the Raven combo idiom, not only Ctrl+1…9.** DPG combos have no keyboard operation
at all, so Raven supplies its own: a hotkey focuses the combo, and while it is focused Up / Down / Home / End
cycle the choices. `raven-avatar-settings-editor` is the reference implementation — a `combobox_choice_map`
of `{widget: (choices, callback)}`, and a bare-key branch that dispatches on `dpg.get_focused_item()`. Copy
that shape rather than inventing one. Ctrl+1…9 stays as the direct-jump shortcut, useful because the filter
list here is short and labelled.

This means bare Up / Down mean different things depending on what holds focus — the listing cursor from the
find field, the filter choices from the combo — which is the idiom's normal behaviour and is why the handler
dispatches on the focused item rather than on a mode flag.

**Esc means "give me back the find field" wherever focus has been parked, and cancels the dialog only when
it is already there.** One rule covering the path field, the combo and the places panel alike, and it is
what keeps Esc from being a mode-dependent surprise. Note that in save mode that widget is the **filename
entry** — the same field under its other name, so Esc lands you back where you were typing the name, and
"returns to the find field" below should be read that way throughout.

What does *not* generalize is restoring on the way out. The path field holds a **draft** — typing there does
nothing until Enter — so Esc discards it and puts the current directory back. Combo cycling **applies on
every keypress**, and watching the listing re-filter as you cycle is the useful part of it; there is no
uncommitted state to discard, and reverting would undo a change the user just watched happen and kept
cycling past. So: same focus rule, no restore. Worth stating because "same rule for both" is the natural
assumption and is half right.

**Ctrl+Enter rather than a double-Enter in `dirs_only` mode.** A timing window penalizes exactly the users
this brief exists to serve. The existing overwrite confirmation survives that objection because it is a
*confirmation* — press #1 is inert, so missing the window merely re-arms it — whereas in `dirs_only` press #1
would *descend*, and a slow second press descends again rather than accepting, losing your place in the tree.
Deferring the descent by the confirm window would restore the benign-failure property, at the cost of lag on
the most common keystroke. Ctrl+Enter costs neither.

### Sorting

Added 2026-08-17 (Juha). The sort row above the listing is mouse-only, and it is the one control the
original design of this brief left out — an oversight rather than a decision, since sorting a directory by
date is how you find what you just downloaded.

**Ctrl+Shift+1 … Ctrl+Shift+4 pick the Nth criterion**, parallel to Ctrl+1…9 for the type filter: *Ctrl and
a number picks the Nth filter; add Shift and it picks the Nth sort criterion.* Bare Ctrl+numbers are already
spoken for, which is what makes the Shift necessary rather than decorative. Numbers also dodge a collision
the letter form walks into — `T` is the type-filter combo, while *Type* is also a sort criterion, and the
two have nothing to do with each other.

Each chord does exactly what clicking that button does, by calling the same `sort_by(sort_key)`: asking for
the criterion already in force reverses it, any other starts ascending. No separate key for direction.

### Discoverability

A hotkey nobody knows about has not been added. Three tiers, in descending order of how much they buy:

- **Tooltips wherever a widget exists** — sort buttons, type filter, thumbnails checkbox, places entries.
  Nearly free, and already the house pattern here (`"Refresh the current folder listing [F5]"`). Covers
  roughly half the table.
- **F1 opens the dialog's help card**, built on `raven.common.gui.helpcard.HelpWindow` like the six apps
  that already have one. This is the load-bearing tier, because the listing keys — Up/Down, Enter,
  Ctrl+Space, Tab — have no widget to hang a tooltip on, and they are what this brief is for. It is also
  the only place the governing rule can be stated as a sentence rather than inferred from a table.
- **Not a permanent hint bar under the listing.** Vertical space in a file dialog is what shows you files.

Two mechanics to get right:

- **DPG does not stack modals**, measured 2026-08-17: `show_item` on a second modal while one is up
  succeeds, raises nothing, and never becomes visible. Since both the dialog and `HelpWindow` are modal,
  the dialog must hide itself while its card is up and restore itself when the card closes. The card should
  say whose keys it is listing, since it appears with the dialog gone.
- **`FileDialog.is_visible()` must keep returning `True` across that gap.** Apps ask it to suppress their
  own hotkeys and file drops while a picker is up; if hiding the dialog window flips it to `False`, the app
  wakes up underneath the help card. `HelpWindow`'s `on_show`/`on_hide` are *not* the place for this — those
  are the app's hooks for suspending its own viewport-level things (Visualizer's overlay) and should not
  fire for a card belonging to a dialog that is itself already modal.

Write the card's hotkey text as "Up / Down", never "↑↓": Raven's UI font is OpenSans, which has no arrow
glyphs, so the arrows render as missing-glyph boxes. It is why the sort indicators are drawn as triangles
rather than written.

### Tab completion, and how it coexists with the fragment search

One rule with a fallback:

> Among the shown items, take those whose name *starts with* the field content. If none do, take **all**
> shown items. Extend the field to that group's longest common prefix.

Read as a sentence: Tab answers *"what do the things I am looking at have in common?"*, and prefix-matching
is a refinement for when the field happens to be a prefix. With `readme.txt, readme.md, headers.h`, `re`
gives `readme.` by prefix; `eadm` gives `readme.` through the fallback, the fragment search having already
narrowed it; `ead` gives nothing and flashes, since `headers.h` is still shown. Two riders, both bash's:
complete fully when unique, and append the separator when the unique result is a directory — which gives
Tab-descent for free.

**The path field completes too, by the same rule against a different candidate set.** It already exists and
is already typable — `ex_path_input_*`, an `InputText` with `on_enter=True` whose callback `chdir`s to what
was typed — so Ctrl+L has a real target rather than needing one built. What it lacks is completion, and
without that a Ctrl+L that lands you in a field where you must type an absolute path by hand is worse than
no Ctrl+L.

Tab there completes the **last path component** against the directory named by everything before it, using
the same prefix-preferred-else-all rule and the same smart-case matcher as the find field. Two differences
from the find field, both falling out of what the field is for:

- **Only directories are candidates.** The callback `chdir`s, so a file path can only produce the "not a
  directory" message box. Completing to one would be completing to a dead end.
- **There is no listing to fall back on**, since typing here filters nothing. The candidate set is read from
  the filesystem for the directory named by the typed prefix, on each Tab.

Append the separator after a completed component, as bash does, so repeated Tab walks down the tree.

**Esc in the path field returns to the find field** rather than cancelling the dialog, restoring the field to
the current directory on the way — the browser's Ctrl+L behaviour. Cancelling then takes a second Esc, which
is a consequence of the rule rather than a timing window: focus is back in the find field, where Esc cancels
as always.

**Nothing but Tab writes the find field**, which is what keeps save mode honest: `ok()` already takes the field
verbatim, so `readm` saves as `readm` with `readme.txt` sitting right there. Implicit completion is the only
thing that could break that, so there is none.

**In save mode only**, arrow navigation also fills the field from the cursor row — *unless the user has typed
since the last programmatic write*. One boolean, cleared on any character, set when we write. Browse and the
name follows you; type one character and the field is yours. The mechanical catch: those writes must **not**
re-run the filter, or the listing collapses to one row and the cursor has nowhere left to move. The existing
click path calls `_update_search()` explicitly after `set_value`, which is right for a click and wrong for
arrows.

### Cursor and selection

**Cursor and selection are different things** once Ctrl+Space exists, so they need different looks. Selection
keeps the selectable's `True` highlight; the proposal for the cursor is a bound theme painting
`mvThemeCol_HeaderHovered`, on the reasoning that "hovered" is the affordance a cursor wants. **Whether the
two are actually distinguishable at a glance has to be looked at**, not argued — they are adjacent greys in
the default theme, and a cursor that reads as a weak selection is worse than either. Render a row in each
state side by side and pick from that; if the hovered colour is too close, the fallback is a border or a
distinct text colour rather than a third fill. The cursor is an index into `shown_items`,
re-anchored **by path** after every rebuild and clamped if that path is gone; every keystroke in the find
field rebuilds the listing, so this is the common case rather than an edge one.

### The places panel

Added 2026-08-14 (Juha). The side panel holding **Home, Desktop, Downloads, Images, Documents, Music,
Videos**, a separator, and then one entry per mount point, is reachable only by mouse. That makes the whole
of "start somewhere else entirely" a pointer-only operation, which is the gap this brief exists to close —
and it is not a small one, since jumping to a mount point is how you reach anything outside your home
directory without typing its path.

It takes the same shape as the type filter, so it adds an entry point rather than a mechanism: **Ctrl+B
focuses the panel; bare Up / Down / Home / End then move within it; Enter goes to the highlighted place;
Esc returns to the find field**, per the universal rule. `B` for *bookmarks*, which is what a browser and
GTK's own file chooser call this panel's key.

Four things to settle while building it:

- **One list, not two.** The folder shortcuts and the drives are visually separated, but a cursor should run
  through them as a single sequence and treat the separator as scenery. Two sub-lists would need a key to
  cross between them and would buy nothing.
- **The entries are `dpg.add_menu_item`, and whether `dpg.focus_item` works on one is unverified.** DPG's
  focus model has already produced two surprises here — a child window cannot be focused, while a button
  can — so a menu item is a third case and needs a probe, not an assumption. If it cannot hold focus, the
  cursor is drawn rather than real, exactly as the listing's cursor already is; the panel's *own* focus can
  then rest on any focusable widget inside it.
- **Enter should return focus to the find field** after changing directory. The reason to jump to a place is
  to look at what is in it, so leaving focus in the panel would cost a Ctrl+F on every use.
- **The mount list is built once, when the dialog is built** (`_get_all_drives`, from
  `psutil.disk_partitions()`). So unlike the listing — which rebuilds on every keystroke in the find field
  and needs its cursor re-anchored by path — this list is static for the dialog's lifetime and its cursor is
  a plain index. A USB stick inserted while the dialog is open will not appear, which is pre-existing
  behaviour and out of scope here.

Raven passes neither `user_style` nor `show_shortcuts_menu`, so it always gets style 0 with the panel
shown; the compact style-1 variant is unused and need not be designed for.

### Odds and ends

**Constructor parameters**: `smooth_scrolling` and `smooth_scrolling_step_parameter`, so each app passes its
own config setting (Librarian already has both).

**Save mode keeps its double-Enter overwrite confirmation**, unchanged.

## What still needs checking, and what no longer does

**`dpg.set_value` does not fire the widget's callback** — measured 2026-08-13 on a combo and on an
`InputText`, both silent while the value did change. So the save-mode arrow-fill can write the field without
re-running the filter, which is what that rule needs, and the combo idiom can set a choice and call the
callback itself. Two places in the tree already relied on this in comments; it is now measured.

**Every chord in the table reaches a key handler while the find field holds the caret** — measured
2026-08-17, probe write-up in `investigations/dpg-keyboard-chords/`. Ctrl+Enter, Alt+Up, Ctrl+Up,
Ctrl+Space, Ctrl+Home, Ctrl+Shift+1 and the bare navigation keys all arrive with their modifiers intact,
and nothing intercepts Alt+Up under Cinnamon. So Ctrl+Up stays as the one-handed alias it was proposed as,
rather than becoming a fallback for a broken Alt+Up — though since both dev machines run Cinnamon, whether
some other desktop claims Alt+Up is for users elsewhere to report, and the alias covers that too.

**Tab arrives and ImGui does not spend it** — focus does not move, no character is inserted. Tab completion
is therefore free to be built as designed. One wrinkle to code around: Tab pressed while the field is
focused but *inactive* re-activates it, so the handler can see the field become active under it.

**Page Up / Page Down are `517` / `518`**, confirmed against the live enum: Tab=512, Up=515, Down=516,
**517**, **518**, Home=519, End=520 — exactly where the sequence says they belong, while `mvKey_Prior` and
`mvKey_Next` still read 266 and 267. Compare against the literals.

**Ctrl+Enter deactivates the find field**, since it commits the edit exactly as bare Enter does on a
single-line field. Two consequences: gate the Enter and Ctrl+Enter handlers on `is_item_focused` rather
than `is_item_active`, per the standing rule; and where the dialog *stays open* afterwards — Enter
descending into a directory — the field has silently lost the caret, so it must be reactivated or the next
thing typed goes nowhere.

Still open:

- **Whether the cursor and the selection are visually distinguishable** — see above; a looking question, not
  an arguing one, and it needs the cursor built before it can be asked.
- **Whether `dpg.focus_item` can focus a `menu_item`**, which the places panel is built from. A child window
  cannot be focused and a button can, so this is a third case and neither answer is safe to assume. One
  headless probe settles it; it blocks only the places panel, so it can wait until that step.

**Out of scope, worth recording**: an audio cue for the overwrite warning, and for whatever other warnings the
dialog grows. Visual-only feedback is a gap for the same audience this brief is about.

## Where the dialog stands

Four sibling items closed on 2026-08-13 and changed what this brief builds on: smart-case find, grouped
multi-extension type filters, reduced per-call-site boilerplate, and the performance work
(`investigations/filedialog-performance/`) that made the listing cheap to rebuild — which matters here,
because every keystroke in the find field rebuilds it and the cursor has to survive that.

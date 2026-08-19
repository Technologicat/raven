# FileDialog: keyboard accessibility

**Status: mostly built.** The design below was settled on 2026-08-13 (Juha and Claude); the listing cursor,
Enter's rules, Tab and its completion and fill, the sort and filter chords, Alt+Up / Ctrl+Up and the
"Will pick" line are in and live-tested, as is the F1 help card. Still to build: Ctrl+L, Ctrl+Shift+F,
Ctrl+B, the places-panel migration, and the navigation history added on 2026-08-18.
**See "What is built" near the end for the current state** — including the
several places where the design below was overtaken by what building it taught, each noted there rather
than edited into the design, so the reasoning stays legible.

Moved out of `TODO_DEFERRED.md` on 2026-08-13, where a 164-line settled specification was the largest thing
in a file meant for items noticed and parked.

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
entry it leaves *up and down* free for the listing; left and right stay with the text caret.

**Which costs nothing in the table and something in the grid**, noticed 2026-08-17 on the first build:
a table row is a whole entry, so horizontal movement means nothing there, but a grid row holds several
tiles and stepping between them is the obvious gesture. So the keys are not unwanted, they are *occupied* —
and only while the field has the caret. Once Tab hands focus to the listing they are free, and
`navigate_next` / `navigate_prev` (a single entry either way) are already waiting for them.

**So grid view is not merely missing a convenience until Tab lands — it is incomplete.** Up and Down move a
whole row of tiles, so with no horizontal step every column but the first is unreachable from the keyboard
altogether. In a table that would be nothing; in a grid it is most of the listing. Worth knowing when
judging how finished the keyboard is: the table is usable now, the grid is not. Nothing Tab-cycles. Two other widgets are reachable, each
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
| Alt+Left / Alt+Right | back / forward through where this dialog has been — see below |
| Ctrl+Left / Ctrl+Right | the same, but only while the caret is in the listing — see below |
| mouse back / forward | the same, for the mice that have the buttons — see below |
| Ctrl+Home | back to the default directory (exists) |
| F5 | refresh (exists) |
| Ctrl+F | focus the find field (exists) |
| Ctrl+L | focus the path field; Enter navigates, Esc returns to find |
| Ctrl+1 … Ctrl+9 | select the Nth offered type filter |
| Ctrl+Shift+F | focus the type filter combo; Up / Down / Home / End then cycle it — see below |
| Ctrl+H | toggle showing hidden files and folders — see below |
| Ctrl+T | toggle the thumbnail grid — see below |
| Ctrl+B | focus the places panel — the folder shortcuts and drives; Up / Down / Home / End then move within it, Enter goes there — see below |
| Ctrl+Shift+1 … Ctrl+Shift+4 | sort by the Nth criterion — Name, Date, Type, Size — see below |
| F1 | open the dialog's help card — see below |
| Tab | move the caret between the find field and the listing; completion is applied on the way out — see below |

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

**Ctrl+Shift+F focuses it, not Ctrl+T.** The combo's label is `Show`, so it reads as "Show [Documents and
images]" — which leaves *Type* naming only the sort criterion, and a Ctrl+T pointing at the filter would name
the wrong control. Ctrl+Shift+F carries the mnemonic instead of a label: Ctrl+F filters the listing by name
fragment, Ctrl+Shift+F filters it by type. Same listing, two filters, and nothing to re-learn if the label
changes again. Ctrl+S is out for Save, which this dialog has a mode for; Ctrl+H is reserved below; and Ctrl+0
was rejected so that thumbnail zoom can have it (`ThumbnailGrid.set_tile_size` exists, so Ctrl+plus / minus /
0 is a plausible addition to the grid view).

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
spoken for, which is what makes the Shift necessary rather than decorative. Numbers also avoid the letters
the criteria would otherwise want: Ctrl+Shift+S for *Size* reads as a Save variant in a dialog that saves,
and Ctrl+Shift+N is *new folder* in every file manager. One indexed rule for both rows costs no mnemonics
at all.

Each chord does exactly what clicking that button does, by calling the same `sort_by(sort_key)`: asking for
the criterion already in force reverses it, any other starts ascending. No separate key for direction.

### The thumbnail toggle

Added 2026-08-17 (Juha), and it was missing rather than deferred: the grid view arrived from
`filedialog-thumbnails-brief.md` after this table was written, so its checkbox is the one control in the
dialog that no key reached.

**Ctrl+T**, which is available precisely because this brief took it *away* from the type filter — the combo's
label is `Show`, so `T` named nothing there, while the control it does name is labelled "Thumbnails". The
letter freed by one decision turns out to be the right one for the control the same decision left stranded.

**Silently ignored where the grid is not on offer.** A `"dir"` picker lists no files and so has nothing to
make tiles of; its checkbox is hidden, and a key that acts where its control is invisible is worse than one
that does nothing. Same predicate the checkbox uses, so the two cannot drift apart.

### Hidden files

Added 2026-08-17 (Juha). `show_hidden_files` is already a constructor parameter (`fdialog.py:182`), passed
through to the listing as `show_hidden`, but it is fixed for the dialog's lifetime and has no control at
all — so a user who needs to reach a dotfile cannot, by keyboard or by mouse.

**Ctrl+H toggles it, and a checkbox on the sort row does the same**, next to Thumbnails. Ctrl+H is what
every GTK and GNOME file chooser binds this to, which is the whole argument for it; the checkbox is what
makes it discoverable, and the two share one callback exactly as the thumbnails pair will.

The toggle re-lists the current directory, so it goes through the same rebuild path as a sort or a filter
change — meaning the cursor is re-anchored by path and clamped like any other rebuild, with no special
case. If the cursor was sitting on a hidden entry when it is switched off, the clamp handles it.

### Navigation history

Added 2026-08-18 (Juha), for building the next day. The dialog can go *up*, and that is all: there is no
back, so a wrong turn costs you the way you came rather than one keystroke. Up and back are different
questions — up is a fact about the filesystem, back is a fact about this session — and only one of them is
answerable today.

**Alt+Left and Alt+Right**, which is what a browser and every file manager bind. They pair with Alt+Up
already being the up key, so one modifier covers the whole navigation cluster.

**Ctrl+Left / Ctrl+Right join them, but only while the caret is in the listing.** Settled 2026-08-18
(Juha). They cannot be unconditional aliases the way Ctrl+Up is: in a text field those chords mean *move
the caret by one word*, and the find field is where the caret sits by default. But the boundary that would
gate them is not a new one — Tab already makes bare Left and Right mean the text caret on one side and the
listing cursor on the other, so a pair that also depends on which side the caret is on is the established
rule reaching one step further, not an exception carved for it.

The implementation consequence is the same one Ctrl+Up had: the listing's bare Left / Right branch does not
currently test for Ctrl, so it would swallow the chord. The modified form has to be checked first, exactly
as `Alt+Up` / `Ctrl+Up` is checked ahead of bare Up.

**The mouse's back and forward buttons, if they arrive.** DPG exposes them — `mvMouseButton_X1` (3) and
`mvMouseButton_X2` (4), alongside Left/Right/Middle, with `add_mouse_click_handler` taking a button — so
the API surface is there. Whether the events actually reach a DPG app from a thumb-button mouse on this
desktop is **unmeasured**; a click handler that logs the button answers it in a minute, and that probe is
the first step of building this. If they do not arrive, the keys stand alone and nothing else changes.

**The history persists across openings, because an instance is a task.** Raven apps build one `FileDialog`
per job rather than one per app — 15 construction sites across the constellation, each with its own title,
filter list and default path, so "open character image" and "open backdrop image" are separate objects in
the same editor. An instance therefore reopens onto the *same kind of task*, and a history carried between
its openings is a history of that task: reopening the character-image picker where you were last time is
what a user would want, not a surprise. A constructor option can turn it off for a caller who disagrees.

*The first version of this section said the opposite*, reasoning from the thumbnail override being
per-opening — and that reset exists for a reason which does not transfer. The grid override competes with
an **automatic** rule, so an override that outlived one opening would disable the automation for the
session: a one-way door. Nothing automatic is competing for the navigation history, so there is no door to
leave open. Worth stating because the two look like the same shape of state, and only one of them is.

**What persistence costs is staleness**, and it is worth handling rather than discovering: the longer a
history lives, the likelier it holds a directory that has since been deleted or unmounted. `chdir` today
catches `PermissionError` and `NotADirectoryError` — not `FileNotFoundError`, which is what a vanished
directory raises, and which only `on_path_enter` guards against at its own call site. Back into a deleted
directory would therefore raise out of a key handler.

**Back skips over what is gone** (Juha, 2026-08-18) rather than stopping there or reporting it. A history
step is a request to be somewhere you have been, and a directory that no longer exists cannot satisfy it;
stopping at a dead entry would make Back appear to do nothing, which is worse than going one step further.
Validity is re-tested on each traversal rather than pruned, so a path that comes back — a remount, a
recreated directory — is reachable again.

**And that is the interesting thing to have found**, because it lands on the shared-stack question above.
A state can go invalid between being pushed and being popped, and the mechanism cannot know: validity is
domain knowledge, the same way equality is. So it takes the same treatment — **a predicate the caller
supplies, which the traversal walks until it finds a state the caller accepts, or runs out**. That keeps
the stack opaque, so this is not evidence against a generic shape.

What it *is* evidence about is how much such a shape would be carrying. Visualizer needs the equality
predicate (its no-op-commit test is a set comparison) and has no use for validity — `reset_undo_history`
fires on dataset load, so its stack never outlives what its states index. The dialog needs both. So the
shared thing is a list, a cursor, and two injected policies, and the honest test tomorrow is whether
`selection.py` comes out *simpler* for using it. If it comes out merely different, the answer is no.

Two mechanics that fall out of skipping, both of which have to be right or Back lies about where you are:

- **The cursor moves only if the navigation did.** A skipped entry is one the predicate rejected, but a
  *permission* failure is not visible to `os.path.isdir` and surfaces inside `chdir`, which handles it with
  a message box and stays put. If the history had already moved its cursor, it would then disagree with the
  working directory. `chdir` returns nothing today and swallows its errors into message boxes, so it needs
  to report success for this to be checkable at all.
- **Skipping has to be symmetric.** Forward must skip the same dead entries Back did, or the pair stops
  being inverse and a user cannot get back to where they pressed Back from.

**Which changes what "no more history" means for the buttons.** Raven's convention, set by Visualizer long
ago and worth keeping: *the button is disabled exactly when there is nothing left in that direction*. With
skipping, that is no longer "the cursor is at the end" — it is **"there is a valid entry in that
direction"**, so the enabled state is computed by running the same validity predicate over the remaining
entries. No third policy is needed; the enabling falls out of the predicate that was already required.

Two consequences of computing it that way:

- **The answer goes stale on its own.** A directory deleted from another window changes whether Back has
  anywhere to go, with nothing in the history having moved. Re-evaluate at the moments the dialog re-reads
  the filesystem *deliberately* — on opening, on F5, and after each history step — and not on every listing
  rebuild, which happens per keystroke in the find field and concerns only the current directory. A stat
  per entry is nothing on a history this size; doing it per keystroke would be.
- **A press that finds nothing disables the button**, which makes the staleness self-healing rather than
  something to chase: the worst case is one press that goes nowhere and leaves the button correct.

**The icon needs care, because the dialog already has a back arrow that does not mean back.** The toolbar's
`img_back` button is *"Go back to the default path [Ctrl+Home]"* — a left arrow, sitting exactly where a
user would look for history-back. Putting a real Back beside it without distinguishing the two would make
both misleading.

This is not a new problem and should not be solved twice: `TODO.md`'s fdialog list already carries *"change
the 'go to default directory' icon to something less confusing"* and *"add a 'go up to parent directory'
button"* — which is the third arrow in this cluster, and now has keys (Alt+Up / Ctrl+Up) but still no
widget. Up, back and default-path as three arrow buttons is precisely the confusion that item was filed
about, so all of it wants deciding at once rather than a button at a time.

**Use FontAwesome glyphs for them, not drawn assets** (Juha, 2026-08-18), which takes these buttons out of
the icon-set problem entirely. Two separate things wear icons here, and only one of them is in trouble: the
**file-type** icons are icons8 "3D Fluency" third-party assets, which `TODO_DEFERRED.md`'s "A file-type icon
set of our own" replaces; the **toolbar** is UI chrome, and chrome everywhere else in Raven is an icon-font
glyph on a plain button. Doing the same here matches the look and needs nothing drawn.

Licensing comes out clean rather than merely deferred: Font Awesome's split terms were settled on
2026-08-03 (`TODO_DEFERRED.md`) — icons CC-BY-4.0, fonts OFL-1.1 — and the webfonts already ship. Every
glyph this needs exists: `ICON_ARROW_LEFT`, `ICON_ARROW_RIGHT`, `ICON_ARROW_UP`, and
`ICON_ARROW_ROTATE_RIGHT` / `ICON_ARROWS_ROTATE` for refresh. Converting the two existing image buttons at
the same time is what makes the row one set instead of a mixture.

**Default-path is the one that must not be a house** (Juha, 2026-08-18, who spotted it colliding with the
places panel's Home). A second house on screen meaning a different destination would replace one confusing
arrow with a confusing house — but the collision is the smaller half of it. The larger half is that
`default_path` is *not* the user's home: of the 15 call sites it is an avatar assets directory six times,
the current working directory five, an app-supplied path four, and `~` exactly **once**. A house would be
wrong about the destination in fourteen cases out of fifteen, which is worse than ambiguous.

What the button means is **"the default directory, as the app asked for it"** — the caller's designation
for this dialog's task, not a mark the user left and not where this particular opening happened to begin.
`self.default_path` is assigned once in `__init__` and never reassigned, so it is constructor state: the
same destination every time, for the life of the instance.

That narrows the glyph. Anything implying the *user* put it there — a dropped pin, a planted flag — is
telling the wrong story about who chose it, so `ICON_BULLSEYE` (the designated target) fits the meaning
better than `ICON_FLAG` does, with `ICON_LOCATION_DOT` and `ICON_ANCHOR` as the remaining candidates.
Settle it by looking at the row rendered rather than from this list: four glyphs side by side is a thing to
judge by eye, and rendering the candidates together is a two-minute probe.

The one piece of wiring to expect: the house pattern is `add_button(label=fa.ICON_X)` followed by
`bind_item_font(tag, themes_and_fonts.icon_font_solid)`, and `fdialog.py` currently touches neither the
icon font nor `themes_and_fonts` — it has no reference to reach. So the font handle has to get into the
dialog somehow, and *how* is the only real design question in this part.

**Whether the stack itself is shared with Visualizer is the open design question.** Visualizer's selection
undo (`raven/visualizer/selection.py`) is the reference and is the only undo stack in the tree: a module-
level `_undo_stack` list plus an `_undo_pos` cursor, where committing truncates everything after the cursor
and a commit equal to the current state is skipped. That core — a stack of opaque states, a cursor, commit
/ undo / redo, and `can_undo` / `can_redo` for enabling buttons — is what both users want, and neither
half of it knows anything about what the states *are*.

What is entangled in the Visualizer copy, and would have to stay behind: DPG button enabling by tag, numpy
set-comparison as the equality test, and the `app_state` side effects each step fires. So the extraction is
the stack and the cursor, with equality passed in as a predicate and the side effects left to the caller.
Worth doing if it comes out that clean, and worth abandoning if it does not — one of the two consumers is
being written from scratch, so this is the cheapest moment to find out, and also the moment where forcing
a shared shape would cost the most.

### Discoverability

A hotkey nobody knows about has not been added. Three tiers, in descending order of how much they buy:

- **Tooltips wherever a widget exists** — sort buttons, type filter, thumbnails checkbox, places entries,
  OK/Cancel. Nearly free, and already the house pattern here (`"Refresh the current folder listing [F5]"`).
  Covers roughly half the table. **Done 2026-08-17** for every key that exists; the rest arrive with theirs.
  - OK and Cancel were not on the original list and should have been. Bare Enter used to press OK; now it
    descends into the cursor's folder and Ctrl+Enter commits — a change that silently invalidates what a
    returning user knows, which is exactly the case a tooltip is for.
- **F1 opens the dialog's help card**, built on `raven.common.gui.helpcard.HelpWindow` like the six apps
  that already have one. This is the load-bearing tier, because the listing keys — Up/Down, Enter,
  Ctrl+Space, Tab — have no widget to hang a tooltip on, and they are what this brief is for. It is also
  the only place the governing rule can be stated as a sentence rather than inferred from a table.
- **Not a permanent hint bar under the listing.** Vertical space in a file dialog is what shows you files —
  and the dialog is **fixed-size**, so there is no getting it back. `no_resize=True` is the constructor
  default and no caller overrides it, the layout computes three spacers from `self.width` at build time, and
  there is no resize handler in the file. A row spent is spent for good, which is the argument against a
  permanent one and *for* the card that appears only when asked.

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

Write the card's hotkey text as "Up / Down" rather than "↑↓", and check the UI font before assuming
otherwise. Arrows are not safe by default: OpenSans, the current default, has no arrow or triangle glyphs
at all, so they render as missing-glyph boxes — which is why the sort indicators are drawn rather than
written. InterTight *does* carry them (`raven/common/gui/utils.py`), so if the UI font moves the constraint
lifts; the spelled-out words are correct either way, which is the reason to prefer them here.

### Tab completion, and how it coexists with the fragment search

> **Buildable, but only with the caret out of the field.** Measured 2026-08-17: ImGui's edit buffer owns an
> *active* `InputText`, so `set_value` is reverted on the next frame and `configure_item(default_value=...)`
> never lands at all. There is no spelling of the write that survives. On an **inactive** field both work
> normally — which is what "Tab is the mode-switch key" below is built on: Tab moves the caret out of the
> field first, and the completion is written into a field nobody is typing in.
>
> So the rule below stands as written. What changes is *when* it runs.

One rule with a fallback:

> Among the shown items, take those whose name *starts with* the field content. If none do, take **all**
> shown items. Extend the field to that group's longest common prefix.

Read as a sentence: Tab answers *"what do the things I am looking at have in common?"*, and prefix-matching
is a refinement for when the field happens to be a prefix. With `readme.txt, readme.md, headers.h`, `re`
gives `readme.` by prefix; `eadm` gives `readme.` through the fallback, the fragment search having already
narrowed it; `ead` gives nothing and flashes, since `headers.h` is still shown. Two riders, both bash's:
complete fully when unique, and append the separator when the unique result is a directory — which gives
Tab-descent for free.

**Completing a component re-lists, and a path that resolves to nothing says so** (Juha, 2026-08-18). Tab
that lands on an existing directory and appends the separator should show that directory's contents, which
is what makes Ctrl+L a way of *going* somewhere rather than a way of typing a string blind. Where the typed
path does not exist, tint the field, following what Visualizer already does with its search: it drives the
field's text colour from a live theme handle — white with nothing typed, `(180, 255, 180)` green when the
search finds something, `(255, 128, 128)` red when it does not. Three states, steady rather than flashed,
and the same three fit here exactly.

*This collides with the draft rule two paragraphs up, and the collision has to be resolved rather than
noticed halfway through.* The path field was specified as a draft — typing does nothing until Enter, which
is what lets Esc discard it and put the current directory back. Re-listing on completion makes it partly
live. Two ways out, and they are not equivalent:

- **Completion navigates.** Tab `chdir`s to the completed component, so the listing follows for free and
  `reset_dir` keeps reading the working directory from the process — which it does deliberately, so that
  the dialog cannot show you A and hand back B. Esc then means *return to where I pressed Ctrl+L*, which
  needs one remembered path and is the browser behaviour this section already cites. **Recommended**, and
  it costs the draft semantics rather than the invariant.
- **Completion previews.** The listing shows the completed directory without moving the process, so the
  draft rule survives. But then the listing and `os.getcwd()` disagree by construction, which is precisely
  what `reset_dir`'s "read from the process" comment exists to prevent. Choosing this means changing that
  contract knowingly.

**Either way, an ambiguous completion has nowhere to show its candidates** (Juha, 2026-08-18) — Tab extends
to the common prefix and the user is left guessing what else matched. Navigating narrows the problem rather
than solving it: after a successful completion the listing *is* the candidate list for the next component,
since the process is now in that directory, so what remains uncovered is the first ambiguous component of a
path typed somewhere far from the current directory. Under the draft branch nothing is covered at all,
which is a third cost to weigh against keeping the draft.

So it wants a candidate list of its own — a programmatic popup under the field, tooltip-like but not
hover-driven. Prototype before committing to a shape, and read `investigations/dpg-overlays/` first, which
already measured two traps this walks straight into:

- **A floating overlay is opaque to the mouse across its whole rect**, `no_background=True` included. A
  candidate list hanging over the listing is a dead zone over whatever it covers — wheel and clicks both
  swallowed — which is why `ScrollEndFlasher` is two windows rather than one.
- **An autosize window is silently ~100 px tall unless `min_size` says otherwise**, and the clamp applies
  to an explicit size too. A three-item list comes out with phantom blank space under it.

What that bundle does *not* answer is the one thing this needs: whether a **non-modal** window renders above
a **modal** one. The measured result is that a second *modal* never becomes visible, which is a different
question, and it is the question the help card had to work around by hiding the dialog — an answer unusable
here, since the whole point is to see the field while the list is up. One probe, before any of it is built.

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

### Tab is the mode-switch key

Settled 2026-08-17 (Juha), and it is what makes completion possible at all rather than a consolation for
losing it. **Tab moves the caret between the find field and the listing.** Everything else follows from the
field being inactive while the listing holds focus:

- **The field becomes writable.** Programmatic writes land on an inactive field, so this is where the
  completion above is applied, and where save mode fills the field from the cursor row.
- **The focused item is the cursor.** In table view the rows are selectables, which are ordinary focusable
  items, so Tab focuses the cursor row's own selectable — no stand-in widget, and `get_focused_item`
  dispatch says something true rather than approximating.
- **Grid view needs the stand-in**, being a child window and drawlists with nothing focusable in it. Same
  rule, two implementations; the mode-flag fallback lives there.

**The return Tab is for editing the text, not for accepting.** Enter acts directly from the listing, per the
governing rule, so the common path is *type → Tab → arrow → Enter* and never comes back to the field. Tab
back is what you press when you want the name in the field to work on — which in practice means save mode,
amending an existing name into a variant.

**Returning arms ImGui's select-all**, and there is no API to undo it. The next character typed replaces the
field instead of extending it. Three cases, and only one bites:

- *type → Tab → arrow → Enter*: structurally exempt. Focus never returns to the field, so nothing arms the
  selection in the first place — this is not a case that survives the hazard, it is one that never meets it.
- *amending a name in save mode*: to append you press End first anyway, which collapses the selection. Other
  save dialogs in the wild behave the same way, so the motion is already in the fingers.
- *Tab back and immediately type, meaning to extend*: the completion vanishes, silently. Documented rather
  than fixed — one line on the help card, along the lines of "Tab returns to the field with the text
  selected; End to keep it".

### Cursor and selection

**Cursor and selection are different things** once Ctrl+Space exists, so they need different looks — and
**different *axes*, not different shades of the same one.** That is settled rather than open, because the
thumbnail grid solved it already and has been shipping the answer (`thumbnailgrid.py`, `_draw_tile`):

| | grid | table |
|---|---|---|
| selection | fill — a white wash over the tile at alpha 40 | the selectable's own `True` highlight |
| cursor | inner border — inset 3 px, 2 px thick, `(80, 160, 255)` | **distinct text colour** |

The grid's two marks are different *kinds*, so they compose: a tile that is both selected and current reads
as both at once, with neither competing for the other's channel. An earlier draft of this brief proposed
painting the table cursor with `mvThemeCol_HeaderHovered` — a *second fill* — and that is the mistake the
grid's design rules out. Two adjacent greys is not a problem to be solved by finding a better grey.

A table row cannot take the grid's answer literally: its fill is spoken for by selection, and selectables do
not draw borders. What is left is text colour, which was this brief's fallback and is now the choice, for
the reason above rather than for want of anything else. Across all three panels, then: **fill means
selection, text colour means cursor** — which also makes the places panel's lack of a selection concept
invisible instead of a special case.

Still a looking question, but a narrower one: whether the chosen cursor colour reads as a cursor against
both a selected and an unselected row. Render the four combinations together and pick from that.

The cursor is an index into `shown_items`, re-anchored **by path** after every rebuild and clamped if that
path is gone; every keystroke in the find field rebuilds the listing, so this is the common case rather than
an edge one.

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

**The entries move from `menu_item` to selectables, in a table like the listing's.** Decided 2026-08-17
(Juha), after measuring that a `menu_item` cannot hold focus and cannot be asked to. The colour-a-menu-item
route was the alternative and loses on cleanliness even where it works: it would bind and unbind per-item
themes on every cursor move to produce a weaker affordance than the listing's, on a widget that then *still*
needs a mode flag because it cannot be focused.

The migration is close to a rename — a listing row is `[image][selectable spanning columns]` and a places
entry is `[image][menu_item]` — and on the far side the panel stops needing anything of its own: the same
cursor machinery as the listing, a real selection highlight so the cursor *looks* the same in both panels,
and Ctrl+B working by the rule Ctrl+Shift+F works by. The one wrinkle is that a selectable is a toggle where
a menu item is an action, so a click has to clear the value afterwards — which the listing already does, so
it is an established pattern here rather than a new one.

**The panel has a cursor and no selection at all** (Juha, 2026-08-17), which is what makes the toggle-vs-
action wrinkle go away rather than needing to be managed. You never *select* a place, you jump to one, so a
highlight left behind afterwards would assert a state that does not exist. Concretely:

- **Clicking** applies — go there — clears the highlight, and puts focus back in the find field.
- **Keyboard** moves a cursor through the panel; **Enter** does exactly what a click does, including the
  return to the find field. No special case needed for it: the governing rule is *Enter goes as deep as it
  can*, and for a place, going there is the deepest thing there is.
- So **both exits lead home** — Esc without applying, Enter with — which is the universal Esc rule and the
  universal Enter rule meeting, rather than two panel-specific behaviours.

**Focus now has three homes**: the find field, the listing, and this panel. Tab bounces between the first
two; Ctrl+B reaches the third and it hands focus straight back on use, so it is a visit rather than a place
to be.

*Which leaves the panel's cursor with nothing special to do*: it is a text colour, like the listing's. The
tempting shortcut was to use the selectable's own highlight here — the panel has no selection to confuse it
with, so it is free — but that would make one mark mean "selected" in the listing and "cursor" in the panel,
side by side on screen. The uniform rule costs nothing and says the same thing everywhere.

**Build it with the cursor rather than before or after it.** The cursor is what both panels need, and a
component's second consumer is what finds the gaps the first one hid — `ThumbnailGrid` gained three missing
pieces within an hour of being wired into a second app. **With GUI tests** (Juha, 2026-08-17): this is the
first cursor-bearing widget in the dialog, and the behaviour worth pinning — re-anchoring by path across a
rebuild, clamping when the path is gone, the focus hand-off — is invisible in a screenshot and easy to
regress.

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

**Should cancelling take two presses of Esc?** Raised 2026-08-18 (Juha), open. The dialog already has the
pattern — save mode's overwrite confirmation, where the OK button must be pressed twice and animates to say
so — and this brief's own argument for it transfers cleanly: a confirmation survives the timing-window
objection *because press #1 is inert*, so missing the window merely re-arms it. That is the case here too.

Three things pull the other way, which is why it is a question rather than a decision:

- **Esc-to-dismiss is the most universal dialog reflex there is**, and it is one users have already built.
  Breaking it costs everyone a little to protect against a rare accident.
- **The accident is cheap.** A cancelled picker loses the browsing you did, not any data — reopen it and
  the default path is where it always was. The overwrite confirmation guards a *file*, which is a different
  order of loss, so the two are not the same bet.
- **Save mode is where it would actually bite**, a typed filename being the one thing in this dialog worth
  losing. But making it two presses in save mode and one elsewhere is exactly the mode dependence the Tab
  discussion rejected for the user's mental model. Uniform, or not at all.

**What to measure is which annoyance is larger**, and stating it that way is what makes it measurable at
all. The case for two presses is not safety in the abstract: it is that Esc would then *never* close the
dialog by accident, and an accidentally dismissed picker is a genuinely irritating thing to happen. The
case against is that every deliberate dismissal costs a second press, forever. Both are annoyances, one
rare and sharp and the other constant and small, so the question is their product rather than either alone
— which nobody can compute from here.

**If it goes in, the Cancel button flashes and says so** (Juha, 2026-08-18), exactly as the overwrite
confirmation does — an inert first press that gives no sign is not a confirmation, it is a key that stopped
working. Same call, `gui_animation.WidgetFlash` with `target=self.btn_cancel` and the message routed to
`target_text=self.text_notification`, which is where "Press again to overwrite file" already appears.

**Green, and that falls out of the defaults**: `WidgetFlash` flashes `(96, 128, 96)` with `(180, 255, 180)`
text unless told otherwise, and the overwrite is the one that overrides — to `(255, 32, 32)` — because it
is confirming something destructive. Cancelling destroys nothing, so it takes the ordinary colour, and the
difference between the two reads at a glance rather than needing to be learned. Those greens are also the
ones Visualizer's search field uses for a match, so the constellation already means one thing by them.

Cheap to test both ways once the caret has more than two homes, since Esc's behaviour is being rewritten
then anyway. Worth trying rather than arguing: the reflex either survives the second press or it does not.

**`dpg.set_value` does not fire the widget's callback** — measured 2026-08-13 on a combo and on an
`InputText`, both silent while the value did change. So the save-mode arrow-fill can write the field without
re-running the filter, which is what that rule needs, and the combo idiom can set a choice and call the
callback itself. Two places in the tree already relied on this in comments; it is now measured.

**…but only while the field is inactive, and both features here write it while the user is typing in it.**
Measured 2026-08-17 (`investigations/dpg-keyboard-chords/`). ImGui's own edit buffer owns an active
`InputText`: `set_value` looks like it worked — `get_value` immediately after reports the new string — and
the next frame writes the old buffer back *and fires the edit callback while doing so*. Typing `abc`,
writing `SETVALUE`, then typing `Z` leaves `abcZ`.

This lands squarely on **Tab completion** and on the **save-mode arrow-fill**, both of which write the field
with the caret in it, and it is not a detail to code around later — it decides whether they are buildable as
specified. What is known so far:

- The unfocus → write → refocus dance does work, but the caret is not released on the calling frame — with
  `focus_item` on another widget, `is_item_active` read `[1, 0, 0, 0, 0, 0]` per frame. That is one sample
  on an idle app and **not a frame count to build on**; how many rendered frames the queued focus change
  costs depends on what else is in flight. So the dance polls `is_item_active` with `split_frame` until it
  goes false, bounded and logged, rather than waiting a fixed number of frames.
- **Refocusing then arms ImGui's select-all**, so the next character the user types replaces the whole
  field instead of extending it — which is precisely the wrong behaviour for a completion. DPG exposes no
  caret or selection API to undo it.

So the open question is not "how do we write the field" but "how do we hand it back with the caret at the
end". Tab answered it, by making a moment where the field is inactive and there is no caret to hand back
— see the build-state section at the end. What follows was written before that landed: Tab completion was
designed but not buildable, and the save-mode arrow-fill
needs the same answer — with the mitigating detail that arrow navigation happens when the user is browsing
rather than mid-word, so a select-all there is less destructive.

Worth re-reading the *"nothing but Tab writes the find field"* rule above in this light: it was written to
keep save mode honest, and it now also bounds how much of the dialog this problem touches.

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
**A `menu_item` cannot hold focus, and asking is a no-op** — measured 2026-08-17. It is a stronger no than
expected: `get_item_state` on one has no `"focused"` key at all, so `is_item_focused` raises rather than
answering False. `focus_item` on one changes nothing (focus stayed on the text field, still active), which
at least makes it the harmless case rather than the child-window one.

**Resolved by moving the panel off `menu_item` entirely** — see "The places panel" above. Selectables hold
focus, so Ctrl+B needs no special case and the panel gains the listing's cursor rather than a second kind.
The mode-flag fallback survives for grid view, which has nothing focusable in it either.

**Out of scope, worth recording**: an audio cue for the overwrite warning, and for whatever other warnings the
dialog grows. Visual-only feedback is a gap for the same audience this brief is about.

## What is built, as of 2026-08-19

The table above is the design. This is the state of it, so the next session does not have to reconstruct
that from the code.

**Working, live-tested:**

- **The listing has a cursor**, in both views, drawn as blue text in the table and as the grid's own inner
  border — one `CURSOR_COLOR`, so the mark means the same thing in either. Up / Down / Page Up / Page Down /
  Home / End move it.
- **Enter goes as deep as it can**, acting on the cursor: `..` and directories descend, a choosable file is
  accepted, a file in a folder picker is scenery and Enter declines it. **Ctrl+Enter** commits without
  descending.
- **The cursor rests on `..` while nothing is typed** — that is what Ctrl+Enter returns, so the resting
  place is load-bearing — and **jumps to the first match once a filter is typed**, a filter being a search.
- **Ctrl+1…9** type filter, **Ctrl+Shift+1…4** sort, **Ctrl+T** thumbnails, plus the five that already
  existed (Enter, Esc, F5, Ctrl+Home, Ctrl+F).
- **A "Will pick:" line** names the path OK would return, live, from the same `_effective_target` that
  `ok` uses — so what is promised and what is delivered cannot drift. It took three corrections to get
  right, each of them a rule firing outside the conditions it was written for: it said "open" for a dialog
  that saves, it answered from the listing in save mode where `ok` answers from the name field, and its
  unique-match shortcut fired with nothing typed, offering a lone subfolder instead of the folder being
  browsed.
- **The cursor survives a rebuild** by the rules in `gridnav.reanchor_cursor`, shared with the grid: follow
  the entry you chose, else hold position, else start at the top somewhere new. The *chosen* entry is
  remembered apart from the displayed one, so a filtered-out file gets its cursor back when it returns.

- **Tooltips name every key that exists** — the sort buttons (with the fact that pressing again reverses),
  the thumbnails checkbox, refresh, back-to-default, and OK/Cancel. The last of those matter most: bare
  Enter used to press OK, and now it descends, so a user who learned the old behaviour needs telling.
  - The tier is therefore *complete rather than finished*: every remaining tooltip belongs to a key that
    does not exist yet, so each lands with its key rather than as a separate pass. Nothing to come back for.

- **Tab moves the caret between the find field and the listing**, and Left / Right come free with it —
  `navigate_prev` / `navigate_next`, which both views had implemented and neither could be given a key.
  **Grid view is therefore crossable at last**: its rows hold eight tiles, and until Left and Right were
  freed every column but the first was unreachable. Live-tested 2026-08-18 in Cherrypick's thumbnail view.
  - **Focus parks on the refresh button, not on the cursor row**, which is a departure from what this brief
    proposed. A table row is a selectable and could hold focus, but then focus would have to chase the
    cursor on every move, and grid view has nothing to chase — a drawlist has no focusable items, so that
    view needed a stand-in whatever the table did. One target for both is what keeps the two views
    answering to the same code. A button is safe to park on: DPG leaves ImGui's keyboard-nav activation
    off, so it ignores Space and Enter rather than pressing itself.
  - **Which button is a correctness constraint, not a preference**, and it cost most of an afternoon to
    find out. `focus_item` is refused when focus sits on an item at *window level* and the target is inside
    a child window — the only refused direction, measured across every source/target position on DPG 2.3.1.
    The find field lives in the listing's child window and the OK button does not, so parking there made
    every later Ctrl+F and Tab-back a window-to-child request, ignored in silence: the caret never came
    back and typing went nowhere. Refresh shares the child window with the field. Recorded in `dpg-notes.md`,
    with the probe in `investigations/dpg-focus/`.
  - **The caret's home is a flag, not a reading of `is_item_active`.** The field is inactive whenever
    anything at all has been clicked, and that must not silently rebind the arrow keys.

- **Tab completes the find field on the way out**, to the longest common prefix of the entries on screen.
  The prefix-preference this brief specified is gone: `candidates` is what the listing shows, and the
  fragment search has already narrowed that, so preferring the entries that *start with* the query applies
  a second rule to a set the first one chose. It answered `datasets` for `data` against `rawdata`,
  `datasets`, `tempdatasets` — which then filtered `rawdata` off the screen. Same fault in the casing:
  returning an entry's own spelling made a lowercase query case-sensitive and dropped what differed.
  - **The invariant that replaced both: a completion may narrow the listing to fewer things, never to
    different ones.** Every example this brief works through gives the same answer under the simpler rule,
    checked against `make_search_matcher` rather than assumed.

- **Tab back fills the field from the cursor entry**, the same in both modes. In save mode that is how an
  existing name becomes the template for a variant — the keyboard route this dialog did not have; in open
  mode it collapses the listing to what was picked. Unconditional, because the only reason to press Tab is
  to go and navigate, so coming back means "give me the one I navigated to". The query is not preserved,
  which is the trade: returning arms ImGui's select-all, so it was a keystroke from gone regardless.
  - **Kept after live testing** (Juha, 2026-08-18). Whether it beat keeping the query in open mode was an
    empirical question rather than an arguable one, so it went in as one commit of pure additions
    (`40ea291`) that `git revert` could undo without touching the completion work, and was driven before
    being believed.
  - **Tab rewrites the query; Ctrl+F does not. That pairing is the design, not an accident of which key
    got the feature.** Both return the caret to the field, and the dialog needs both: one for "I picked
    this, give me its name to work on", one for "let me back to what I was typing". Since Ctrl+F already
    means *focus the find field*, it is the natural home for the non-destructive return, which leaves Tab
    free to be the one that carries something.
    - The alternative — Tab as the pure focus cycle it is in most software, with a new chord for "rewrite
      the query with this" — was considered and rejected. Tab is *already* not a pure cycle here: on the
      way out it completes, which is bash's convention and the strongest one in play for a file dialog. A
      Tab that completes outbound and does nothing inbound is the asymmetric option, not the tidy one.
    - What this costs is discoverability, since neither key announces which it is. That is the help card's
      job, and the pair should be described there together rather than as two unrelated entries.

- **The cursor's placements are told from its choices.** A search shows its first hit — always, including
  after arrowing somewhere, typing being a fresh intent. Erasing the query returns the cursor to `..` if
  nobody moved it, and to the entry that *was* arrowed to if somebody did. `is_anchored` on `TableCursor`
  and `FileGrid` is the one question that decides it, spelled the same on both.

- **`..` answers a search like any other name**, so typing `..` puts the cursor on it and Enter goes up.
  Navigate-by-search covers leaving a directory as well as entering one, rather than requiring a separate
  key for the one direction. Being *searchable* and being *filterable* are kept apart deliberately: the
  listing always shows `..` whatever is typed, because it has to remain the way out when a query matches
  nothing — and that case now lands the cursor on it too, it being the one row left to act on. So the
  first-hit rule is "the first entry that matches" rather than "the row after the parent".

- **Navigating clears the find field**, in `chdir`, which is the one place this dialog navigates. The mouse
  path had always cleared it and the keyboard path had not. Arriving also returns the caret to the field —
  unless the listing had it, arriving being no reason to change modes.

- **Up one level answers to Alt+Up and to Ctrl+Up**, both sides of each modifier, and the bare key still
  moves the cursor one row. The alias is not redundancy: it is what makes the chord one-handed on a layout
  where Alt has no right-hand twin.

- **Hidden files toggle from Ctrl+H and from a Hidden checkbox**, as specified — and the checkbox is offered
  in every mode, including the directory picker whose Thumbnails box is hidden. A folder picker lists no
  files to make tiles of, but hidden *folders* are the case the toggle exists for.
  - **The row it joined has a measured floor, and this moved it**: `min_size` was 900 px because below
    roughly that the rightmost checkbox is clipped off the edge, the sort buttons being fixed-width and
    unable to reflow. Re-measured at `font_size=20`: the row now needs 945 px of window width, so the floor
    is 960. `test_the_sort_row_fits_the_minimum_width` re-takes that measurement whenever the row grows
    another control — which is what the next one added here would otherwise break silently, a clipped
    checkbox looking exactly like one that was never there.

- **Ctrl+Space marks the cursor entry**, in either view, and declines on anything the dialog would not
  return — `..`, and files in a folder picker. The bookkeeping half is shared with Ctrl+click rather than
  written twice, which closed a gap in the mouse path on the way: a Ctrl+click on a folder had never
  refreshed the promised-target line, so in a directory picker the line could name the folder the user
  had just stopped choosing.

- **F1 opens a card of this dialog's keys**, built on `helpcard.HelpWindow` like the six apps that have
  one, and built **per dialog rather than once**: Ctrl+Space appears only where more than one file can be
  taken, Ctrl+T only where files are listed, and the text field is described as finding or as naming
  depending on the mode. The dialog's shape is fixed at construction, so the question is answered once and
  the card never offers a key that does nothing here.
  - **The card carries three openings that `HelpWindow` did not have**, all of them the same fact in
    different places: a card can belong to something other than the app. `handle_own_hotkeys=False` stops
    the shared handler claiming Escape, `label` says whose keys are listed once the owner is off the
    screen, and `show` now reports whether the card came up — which a caller that hid something behind it
    needs and one that merely opens a card can ignore.
  - **Hiding the dialog is not enough to make room; a frame has to pass.** A window leaves ImGui's popup
    stack only once a frame has drawn without it, and the second modal is refused until then — after which
    DPG treats it as *closed* and fires its close handler, so the card undid itself 80 ms after F1 and put
    the dialog back. The symptom reads exactly like F1 being delivered twice, and the thing that tells them
    apart is that the card measures `0x0` and never became visible. Recorded in `dpg-notes.md`.
  - **Escape closes the card; the next Escape cancels the dialog.** One handler sees the key, which is the
    whole reason for the opt-out above.
    - **And the dialog comes back only once Escape is released**, which needs a key-release handler beside
      the press one. ImGui dismisses the topmost modal popup on Escape by itself, and this dialog's close
      handler is `cancel` — so a dialog restored under a still-held key was dismissed the frame it appeared
      and the picker returned nothing. `is_visible` answers yes across that wait too, there being nothing on
      the screen while it lasts.
    - **The driven test passed and the real press failed**, which is worth remembering rather than fixing
      once: `xdotool key Escape` holds the key about 12 ms and a finger holds it for a hundred and
      something, so anything that depends on a key still being down is invisible to a tapped test. Drive it
      as `keydown` / `sleep` / `keyup`, with the sleep chosen against the machine's repeat delay — under it
      for one press, over it for auto-repeat as well. Recorded in `dpg-notes.md`.
  - The card is a fixed size and `HelpWindow` gives it no scrollbar, so **a row added past what fits is
    clipped in silence.** 1250×640 was measured against the longest column at `font_size=20`, with room for
    the one row a multi-selection dialog adds.

- **Ctrl+Shift+F hands the arrows to the type filter**, and Esc hands them back to the find field. The
  combo idiom copied from `raven-avatar-settings-editor`, with one departure: **the combo never gets DPG's
  focus.** Which home has the keys is `_caret_home`, and DPG's focus stays parked where every other home
  parks it — one answer to "where do the arrows go" instead of that answer plus whatever
  `get_focused_item` says. A DPG combo has no keyboard operation of its own to lose by it.
  - **This was a hard constraint when it was written and is a design choice now**, which is worth knowing
    before anyone "simplifies" it. The combo then sat at window level, the one position `focus_item` cannot
    move focus *out of* into a child window, so focus put on it could never have come back. Wrapping it (see
    the click-trap entry below) moved it child-side, where focusing it would work. It would also gain
    nothing — see item 7 — so the choice stands on its own reasons.
  - Escape is now the general rule the design asked for, over a set of homes: hand the caret back, and
    cancel only once it is already there. Nothing is restored on the way out, the combo having applied
    every step as it was made.

- **Both cursors breathe**, at one period shared from `thumbnailgrid.CURSOR_PULSE_SECONDS`. The table's is
  a theme colour, so one `PulsatingColor` drives all six variants the mark is drawn in; the grid's is a
  drawn rectangle that no theme reaches, so `ThumbnailGrid` recolours it once a frame from the same
  `pulsating_alpha`. Two marks meaning one thing have to be the same shade at the same moment.
  - **It costs no frame rate.** The animator now separates *ambient* animations from ones that report
    activity, so an idle throttle keeps throttling — measured live at ~10 FPS with the pulse running, from
    DPG's own metrics window. That also retired the startup `_AMBIENT_ANIMATOR_COUNT` snapshot two apps
    carried, which was only ever correct for animations that existed before the line ran.

- **Nothing clickable sits directly in the dialog window any more**, and that turned out to be a
  correctness constraint rather than tidiness. `focus_item` is refused when focus sits on an item at
  *window level* and the target is inside a child window, so one click on such a control strands the caret
  for the rest of that dialog's life — Ctrl+F, Tab-back and Escape-to-the-field all fire, return, and
  arrive nowhere, with nothing reporting it.
  - Found live on 2026-08-19: clicking the type-filter combo to read its options killed Ctrl+F. The
    `Show` label and the combo now share a borderless child window, which costs nothing on screen (the row
    is pixel-identical) and says what was true anyway — they are one control.
  - **OK and Cancel were the same trap**, reasoned to rather than stumbled on. They look harmless because
    clicking one closes the dialog, leaving no rest-of-the-dialog for a stranded caret to spoil — except
    at the overwrite confirmation, where the first click on OK deliberately leaves it open, which is
    exactly when a user reaches for the filename field to change the name instead.
  - `test_nothing_clickable_sits_directly_in_the_dialog_window` walks the widget tree and fails on any
    clickable item no child window encloses. Worth asserting rather than remembering: the two offenders
    were missable for opposite reasons, one looking inert and the other looking self-closing.

**Open, and worth settling before the places panel:** whether Escape from the *listing* should also hand
the caret back rather than cancel. The rule as built applies to parked controls — the type filter today,
the path field and the places panel to come — and the listing is deliberately excluded, since Esc-closes-a-
dialog is the strongest convention in play and Tab and Ctrl+F already return from there. But that leaves
one home answering Escape differently from the others, which a user has to learn rather than derive. The
help card currently states the simple rule ("Esc — Cancel"), so changing this means changing that too.

**Not built:** Ctrl+L, Ctrl+B, the places-panel migration, and the navigation history —
Alt+Left / Alt+Right and the mouse's back and forward buttons, which is the one item here that arrived
after the dialog started being built rather than with the original design.

**The save-mode arrow-fill is superseded rather than pending.** This brief specified that arrowing fills
the field in save mode, gated on a flag tracking whether the user had typed since the last programmatic
write. Tab-back fills it instead: explicit, so there is no flag to keep truthful, and the same in both
modes rather than one more mode-dependent rule. The flag is exactly the kind of stashed state that goes
stale, which is the argument that settled it.

**Writing the find field is a solved problem now**, and the recipe is worth stating once. A write lands on
an *inactive* field and is reverted on an active one, so either take the caret away first — `_focus_listing`
then `_write_find_field`, which polls `is_item_active` rather than counting frames — or write at a moment
when the caret is already gone. Enter provides one of those for free: committing a single-line `InputText`
deactivates it, which is why `chdir` can clear the field with a plain `set_value`.

The cheap keys were spent on 2026-08-18 — Alt+Up, Ctrl+Up, Ctrl+H and Ctrl+Space are in. **Ctrl+B and the
places-panel migration are the big one left**, and `self._places` is its groundwork: the panel is menu items
today, which have no focus state at all, so a keyboard cursor over it needs the places as data first.

**When testing type filters, use Librarian's attach dialog rather than Cherrypick**, and the reason is now
twofold. Cherrypick passes no `filter_list`, so it gets the catch-all and Ctrl+1 selects the filter already
active — which looks like the key doing nothing. And since 2026-08-19 its dialog has no type filter at all:
it is `pick="dir-with-contents"`, and a dialog that returns a directory does not offer one.

### Added 2026-08-19 (afternoon)

**Item 8 is built and live-tested**: the find field is white with nothing typed, green when the folder
holds a match, red when it does not, recoloured in `reset_dir` beside `_refresh_target_notification` on the
argument that every route into the listing changes both. Two rulings the brief did not anticipate, both
pinned by tests:

- **`..` counts as a hit**, though it is never in `shown_items`. It answers a query like any other name, so
  red would deny a row that is on screen and about to work.
- **A save dialog is left uncoloured.** There the field names the file to be written, so matching nothing
  is the ordinary case; and the inversion that suggests itself — colouring by whether the name is *taken*,
  as an early overwrite warning — lies in both directions. Red reads as "there is something wrong with this
  name" when the name is fine and it is the write that wants a second look; green reads as approval of the
  one outcome the dialog stops to ask about twice.

**The type filter is gone from both directory modes**, with Ctrl+1…9 inert, Ctrl+Shift+F falling through
to plain Ctrl+F, and the two rows off the help card. A type filter applies to files only, and must, or it
would hide the folders you navigate through — so in `"dir"` it narrows an empty set and in
`"dir-with-contents"` it narrows scenery. The target line takes the whole row where no combo shares it.

**The dialog can report a problem at all now**, which it could not: DPG stacks no modal over a modal, so
`message_box` logged and returned, and every error in a picker — permission denied, not-a-directory, a
listing that failed — reached only a log. Reports go on the **target line**, not the notification line
above the buttons: that one is short, and is where the eye already is when clicking OK. Each names what it
failed on, since the report outlives the moment it describes.

This needed three additions to `WidgetFlash`, all of which are the general shape rather than dialog-local:
a text `target` can carry a `message` (that branch used to fade a colour and drop the message); a
`message_duration` separate from the fade, because a report wants the flash quick and the text readable
while an acknowledgement wants them as one gesture; and `set_text_under_flash`, so a derived status line
can be updated *through* a flash — writing plainly wipes the message mid-sentence, and letting the flash
restore what it captured puts back a value from two navigations ago.

**A text target needs a colour of its own**, or the fade runs to black: DPG reports an unset colour as the
`r = -1` sentinel rather than the theme's, which it does not expose.

Four bugs fixed in passing, one of which is worth knowing beyond this dialog: **a DPG callback is passed as
many arguments as it declares**, so the `lambda label=label:` idiom for binding a loop variable gets the
*sender* in that parameter. The places panel carried that for a sprint before a use turned it into
`KeyError: 152`. Recorded in `dpg-notes.md`.

**Item 8's convention is therefore settled**, which is what item 3 was waiting on. The colours are
`_TEXT_NEUTRAL` / `_TEXT_GOOD` / `_TEXT_BAD` in `fdialog.py`, plus `_ALARM_RED` for a report — louder than
`_TEXT_BAD`, deliberately, since a soft red is right for "nothing matched" and wrong for "that failed".

## What to build next, and in what order

Written 2026-08-18 at the end of the day, for the session that picks this up.

**Everything left needs the caret to have more than two homes, and that is one change rather than three.**
The dialog tracked where the caret was in `_caret_in_listing`, a bool, there being exactly two places it
could be. Ctrl+L parks it on the path field, Ctrl+Shift+F on the type filter combo, Ctrl+B on the places
panel: three more homes, and each of the three keys is small *once the flag can name them*. Widening it
first is therefore the cheap order, and doing the keys one at a time means widening it three times, each
time touching every branch that reads the flag. (Done on 2026-08-19 — the flag is `CaretHome` now, and the
argument held: Ctrl+Shift+F cost an afternoon of live-testing rather than a redesign.)

Two existing rules become general at the same moment, which is the other reason to do it once:

- **Esc means "give me back the find field" from wherever the caret was parked**, cancelling only when it
  is already there. One rule over whatever set of homes exists — worth writing against the general shape
  rather than growing a branch per home.
- **Bare Up / Down / Home / End mean different things per home** — the listing cursor from the find field,
  the choices from the combo. That is the Raven combo idiom (`raven-avatar-settings-editor` is the
  reference implementation, a `combobox_choice_map` plus dispatch on `dpg.get_focused_item()`), and it is
  already how Tab makes Left and Right mean two things. A third and fourth home is more of the same.

**Agreed 2026-08-19, for the session after this one:** start with the **find field's colouring** (item 8).
It is independent of everything else, wanted in its own right, and it is where the convention gets settled
against a working reference — so the path field inherits a decided thing rather than deciding it. Then
Ctrl+L with the path colouring, which the completion question turned out to *shrink* rather than block:
without completion the key is focus, Escape, and colour, and its value is paste and teleport rather than
typing.

Suggested order, with what each actually costs:

1. ~~**Widen the caret home**~~ — **built 2026-08-19** as `CaretHome`. Went as advertised: no user-visible
   change, no live drive needed.
2. ~~**Ctrl+Shift+F, the type filter**~~ — **built 2026-08-19**, and small as predicted once (1) was in.
   The surprise was a *click trap*: a control sitting directly in the dialog window strands the caret,
   `focus_item` being refused from window level into a child window. Both offenders are in child windows
   now. That fix left the combo able to hold focus after all — which buys nothing, DPG drawing nothing on
   a focused combo (see item 7). See "What is built".
3. **Ctrl+L, the path field** — small, once the completion question is settled below.

   **Tab completion in it is not buildable in the shape the find field uses** (2026-08-19). Writing the
   completion is the solved half: take the caret away, write, as `_write_find_field` does. Handing the
   field *back* is the half that has no answer — refocusing an `InputText` arms ImGui's select-all, DPG
   exposes no caret or selection API to clear it (`dpg-notes.md`), and the next character therefore
   replaces the whole content. The find field accepts that trade because a lost query is a few characters
   retyped; a completed path is the entire thing the user was building, and losing it to one keystroke is
   worse than never having completed at all.

   **And without completion, typing a path by hand is useless** (Juha) — nobody hand-types
   `…/briefs/researchers-night/`. Which raises the right question: whether this key should exist at all,
   given that **the dialog already completes paths, better, through the find field.** Type a fragment,
   the cursor lands on the first match, Enter descends, repeat. That is incremental completion, and being
   fragment-based and smart-case it beats prefix-based Tab completion at its own job.

   **Agreed 2026-08-19: the simplified field is worth having, on the paste case.** It should exist for the
   two things the find field cannot do, and neither of them involves typing:

   - **Paste.** A path arrives from a terminal, a browser, a message. Ctrl+L, Ctrl+V, Enter. Completion is
     irrelevant to a path that is already complete, and nothing else in the dialog accepts one.
   - **Teleport, and specifically to a short root.** The find field goes *down* from here, and `..` up one
     level, so reaching an unrelated branch means walking up and back down. In practice what gets typed is
     `/mnt` or `/tmp` — "a couple of characters, and I'll rather navigate via the file search" (Juha) — and
     that is exactly the gap the places panel leaves, since it covers home and the standard folders and the
     drives cover `/`, but not a mount point someone happens to use.

   **Which is the measure of how much this field has to be**: somewhere to paste a whole path, and
   somewhere to type four characters. Neither wants completion, and a future session finding the typing
   awkward should reach for the places panel or a bookmark before reaching for a completer.

   **So the feature is "give the address bar the keyboard", and that is cheap.** `on_enter=True` already
   navigates, and DPG's `InputText` already pastes — what is missing is only the focus key, Escape giving
   the caret back, and the colouring. This item moves out of "the only one with real new machinery": the
   machinery was the completion, and the completion is neither buildable nor wanted.

   The colouring earns more here than it would have for typing, too: a pasted path that is stale, mistyped
   at the far end, or from another machine goes red on arrival rather than after Enter and a dismissed
   modal.

   ### Recolouring as you type

   Sketched 2026-08-19 (Juha). The field is coloured by what the typed text *is*, and the three states are
   not the search field's three — a path being typed has a middle state that a query does not:

   | state | colour | meaning |
   |---|---|---|
   | names an existing directory | green | Enter goes there |
   | a valid prefix, not yet a directory | neutral | nothing is wrong; keep typing |
   | cannot lead anywhere | red | Enter would fail |

   **Green means Enter works**, which is the whole value of having a positive state: `/home/jje/Doc` is on
   its way to `Documents` and perfectly fine, but it is not somewhere you can go, and colouring it green
   would promise something the key does not deliver. Two checks decide it, one rule wearing two hats —
   *the text so far cannot be completed to an existing directory*:

   - **Up to the last separator**, that directory must exist on disk. One `isdir`, cheap.
   - **After the last separator**, the fragment must be a prefix of at least one subdirectory of it. That
     needs a listing, cached per parent — it is read once when the parent is checked and reused for every
     further keystroke in the same component.

   **What red predicts is a modal.** `on_path_enter` currently answers a bad path with a message box —
   *"Invalid path / No such file or directory"* — that has to be dismissed. So this is not decoration and
   not an analogy to a search box: it is that same information, moved to before the commit and made free.
   The message box stays as the backstop, since a directory can vanish between the typing and the Enter.

   Four cases the rule has to answer explicitly, because each one can make the colour lie:

   - **An empty fragment needs no rule of its own**, which is a point in the three-state model's favour.
     Typing `/some/dir/` leaves nothing after the separator, and asking whether that prefixes any
     subdirectory is the wrong question — it is a prefix of all of them, and of none when the directory is
     empty. The first check already answers it: `/some/dir/` names an existing directory, so it is green,
     and Enter takes it whether or not anything lives inside.
   - **Hidden directories should count, whatever the Hidden checkbox says.** Typing `.conf` when `.config`
     exists must not go red because a toggle elsewhere is off. A dot typed into a path field is an
     intention, not a browsing preference.
   - **Match exactly, not smart-case.** The find field's smart-case rule is right for *searching* and wrong
     here: on a case-sensitive filesystem a case-insensitive match would show neutral for something Tab
     cannot complete, which is the one thing the colour must never do. Worth a comment at the code, since
     the two fields sitting one above the other now differ deliberately.
   - **`~` expands, and the colour validates the expansion** (Juha, 2026-08-19). The backend takes `~` to
     *mean* the home directory, so a path carrying one is resolved before anything is asked of it — which
     is what makes the colour honest: green says Enter will go there, and it can only say that about the
     path that will actually be opened.

     **The field itself expands only on commit.** What stands in it while typing is what was typed, so a
     `~` the user entered does not silently become eight characters of somewhere else under the caret,
     and the field stops fighting the person editing it. The two halves are the same decision seen from
     the two sides of the commit: the *meaning* is the expansion from the first keystroke, the *text* is
     the literal until Enter.

   **The find field gets the same treatment** (Juha, 2026-08-19): a query matching nothing leaves a listing
   holding only `..`, and today the field says nothing about it. So this is one dialog-wide convention, not
   a detail of Ctrl+L.

   **Copy `raven-visualizer`'s search field**, which has had it all along — `app.py`, "Color the search
   field". Three states rather than two, and the third is the one worth having:

   | state | colour | |
   |---|---|---|
   | nothing typed | `(255, 255, 255)` | white — no search active |
   | matches | `(180, 255, 180)` | green |
   | matches nothing | `(255, 128, 128)` | red |

   The positive state is what makes it a readout instead of a warning: green says the thing you are typing
   *works*, which in the path field means Enter will go there. Note the red is `(255, 128, 128)` here and
   not the `(255, 96, 96)` of `guiutils`' `disablable_red_widget_theme` — that one is for dangerous buttons,
   and reusing it would say the wrong thing.

   The mechanism is worth copying too: one theme bound once to the field, holding a single
   `add_theme_color`, and the colour changed with `dpg.set_value` on that colour item. No rebinding, and no
   theme per state — the same technique `PulsatingColor` uses to breathe a colour.

   **The cost to watch is the first keystroke after a separator**, which is the one that reads a directory.
   Navigation already does that synchronously, so it is not a new kind of cost — but it lands on every
   component of a typed path rather than on an explicit move, so a slow or network mount is worth trying
   before deciding whether the listing needs to go off-thread.
4. ~~**F1, the help card**~~ — **built 2026-08-19**, out of order: it went first rather than last. The
   argument for last was that the card is the only place the whole set is stated as prose, so writing it
   early means writing it twice — which held, and cost about four lines. What it bought was that every key
   already built became findable, and the two mechanics that turned out to be the real work — the modal
   swap and its two frame-timing traps — are independent of which keys exist. Adding Ctrl+L or Ctrl+B to
   the card later is one entry each.
5. **The navigation history** — the largest, fully designed above, and the one Juha asked for by name.
   Independent of 1–4, so it can go first if the appetite is for building rather than refactoring.
6. **Ctrl+B and the places-panel migration** — its own day, not a tail end of this one.
7. **A mark saying which home has the caret** — small, and raised on 2026-08-19 (Juha) once the homes
   became a set rather than a pair: nothing on the screen says where the arrow keys will land.

   **The mark goes on the widget, not on the cursor.** The tempting version is to paint the listing cursor
   blue only while the listing has the keys, as a text box shows its caret only when focused — and it is
   wrong here, because the cursor is not this dialog's focus indicator. It is *what Enter acts on*, from
   either home: the main flow is to type a fragment, watch the cursor jump to the first match, and press
   Enter. Dimming it while the find field has the caret hides the answer to "what am I about to open?"
   exactly when it is being asked, and Ctrl+Enter's resting-on-`..` behaviour needs it visible too.

   So: tint the listing's child window border, and the grid's outer edge, while the caret is there. One
   indicator rather than a per-row state, and the same mechanism serves the places panel — where ImGui
   draws nothing of its own, unlike the combo and the path field, which highlight themselves.

   It also stays clear of the table cursor's theme product, which `_initialize_class` warns about by name:
   three alignments times cursor-or-not is affordable, and a third axis is six more themes and the moment
   to stop binding whole themes per cell.

   **The type filter is no exception to it, and there is no shortcut for that one** (asked and answered
   2026-08-19). The click-trap fix left the combo able to hold DPG's focus, which briefly looked like a
   free mark — it is not: **DPG draws nothing on a focused combo.** So focusing it would buy no highlight,
   remove no code (DPG does not browse a combo by itself; that is custom in every Raven app that has one),
   and change nothing a user could see. The mark has to be drawn either way.

   **Which makes this item fleet-wide, and it has to land in one session** (Juha, 2026-08-19).
   Keyboard-browsable combos are in `raven-librarian`, `raven-xdot-viewer` and both avatar editors, and
   every one has the same invisible focus for the same reason. So the mark belongs in `raven/common/gui/`,
   as something a combo opts into with one call — the `filedrop.install(...)` shape.

   **Splitting it is the tempting mistake.** The listing's mark is local and small, and doing that half
   first would work — and would leave the constellation half-marked, with the other half's fix living as a
   TODO among a hundred others, which is where such things stop being found. An inconsistency introduced
   on purpose is a debt that goes invisible the moment the session ends. So: one session, all of it, or
   not yet.

   That also sets its size honestly. This is not a tail end of the file dialog's keyboard; it is its own
   day, alongside Ctrl+B and the places panel.

   **Nor should the dispatch move onto focus.** `raven-avatar-settings-editor` routes its combo browsing by
   asking whether the focused item is one of two named combos, and that works there — it has a text entry
   too, and arrows in it simply go to the text caret, because nothing claims them. The shape does not
   transfer, for a reason that has nothing to do with combos: **this dialog's arrows have to drive the
   listing, which cannot hold focus at all.** A table's rows and a grid's drawlist have nothing focusable
   in them, which is why focus parks on the refresh button in the first place. "The arrows belong to
   whatever is focused" cannot express "the arrows drive the listing", and `_caret_home` can.

8. ~~**The find field says whether it found anything**~~ — **built 2026-08-19**, see "Added 2026-08-19"
   above. Doing it first did what it was meant to: the convention is settled against a working reference,
   so item 3 inherits it rather than deciding it.
   - The two fields will disagree about *matching*, deliberately: the find field is smart-case because it
     searches, and the path field must be exact because it addresses. Same colours, different rule, and
     worth a comment at each so the difference reads as chosen.

**Items 1–3 and item 5 are each about a day's work, so both in one day is unlikely.** Which to drop is a
real choice rather than a scheduling detail: 1–3 finish the keyboard, and 5 adds a capability the dialog has
never had. Worth deciding at the start of the session rather than discovering at the end of it.

**Standing 2026-08-19, end of day.** Built: 1, 2, 4, 8. Left: **3** (Ctrl+L, small — the next thing to
build), then **5**, **6** and **7**, each its own day. Ctrl+L finishes the *keyboard* in the sense items
1–3 meant; the other three are capability and polish rather than the key set.

**Nothing is open inside item 3 any more.** The last question — what `~` does — was settled on 2026-08-19:
it expands in the backend, the field shows the literal until Enter, and the colour validates the expansion.
That and the other three cases (empty fragment, hidden directories, exact-not-smart-case) are decided
above, so the colouring can be written rather than designed.

## Where the dialog stands

Four sibling items closed on 2026-08-13 and changed what this brief builds on: smart-case find, grouped
multi-extension type filters, reduced per-call-site boilerplate, and the performance work
(`investigations/filedialog-performance/`) that made the listing cheap to rebuild — which matters here,
because every keystroke in the find field rebuilds it and the cursor has to survive that.

# FileDialog: navigation history

**Status: designed, not started, unscheduled.** Split out of
`briefs/researchers-night/done/filedialog-keyboard-brief.md` on 2026-08-21 (Juha) so that brief can close: the
keyboard work has run a week, every key the design named is built, and this is a *new capability* rather
than a gap in the keyboard. It sits outside the numbered runs for the same reason
`ligature-repair-brief.md` and `spreadsheet-ingestion-brief.md` do.

Nothing here is stale — the design below was settled 2026-08-18 and nothing built since has touched it. It
was item 5 of the keyboard brief, described there as "the largest, fully designed, and the one Juha asked
for by name", and independent of the rest.

**The keys it adds**, from the keyboard brief's table:

| key | action |
|---|---|
| Alt+Left / Alt+Right | back / forward through where this dialog has been |
| Ctrl+Left / Ctrl+Right | the same, but only while the caret is in the listing |
| mouse back / forward | the same, for the mice that have the buttons |

**One thing to re-check before building**: `raven-cherrypick` and `raven-xdot-viewer` bind main-row `+`/`-`
for zoom, and both are broken on a non-US layout — see `TODO_DEFERRED.md`, "Main-row `+` and `-` both zoom
out". Alt+Left / Alt+Right are letters-free and so unaffected, but it is the same class of question and
worth asking once of any new chord: does this key exist, unshifted, on a Nordic keyboard?

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


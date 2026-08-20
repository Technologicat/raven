# A tooltip that resizes without a glitch frame

**Status:** designed, measured, not yet built. Started 2026-08-20.

## Why

A `dpg.tooltip` whose caption changes while it is showing renders one frame at its old size — clipped when
the caption grows, skirted when it shrinks. Raven-librarian does this on every toolbutton flash, where the
resting caption is three lines and the acknowledgment is one, so the jump is large and draws the eye. Raven's
apps otherwise go out of their way to avoid visible glitches, and software that *looks* glitchy is read as
unreliable, so a half-fix is not worth having here.

`investigations/dpg-autosize/` has the measurements. The short version: nothing that changes *when or how the
content changes* helps — not mutation, not swapping pre-built widgets, not hiding, not rebuilding the
tooltip (clean when the content shrinks, clipped when it grows, and a flash does one of each). An explicitly
sized window has no lag at all, and `dpg.tooltip` cannot be sized: `configure_item(tooltip, width=…)` raises
`width keyword does not exist`.

What does work is what `raven.visualizer.annotation` and the XDot viewer already do without either having
recorded why: **a tooltip is a window with no title bar, and an app-owned window can be positioned** — so it
can be parked offscreen, shown there, allowed to autosize on a frame nobody sees, and only then moved to the
cursor.

## What to build

`raven/common/gui/tooltip.py`, class `Tooltip`. This is `raven/common/`, so it is held to the foundation
bar: the awkward shape here is paid by every future caller.

- **The window.** `no_title_bar=True, autosize=True, show=False, no_collapse=True, no_scrollbar=True,
  no_focus_on_appearing=True`, and `min_size=[1, 1]` — without which autosize will not shrink below roughly
  100×100 and the empty skirt eats mouse events across its whole rect, as `dpg-notes.md` records.
- **Hover in, hover out.** A `dpg.add_item_hover_handler` on the target shows it; there is no un-hover
  handler, so un-hover needs a per-frame sweep calling `is_item_hovered`. `DearPyGui_Markdown`'s
  `attribute_types._check_hovered_items` is the in-tree pattern, including the `does_item_exist` guard for a
  target deleted mid-hover — which Librarian does constantly when the chat view rebuilds.
- **Placement is not "at the cursor", and `guiutils.compute_tooltip_position_scalar` already does it** —
  per-axis, with three algorithms and an `offset` defaulting to 20 px. The offset is load-bearing rather
  than decorative: a tooltip under the cursor is a separate window, so it takes the hover away from the
  widget beneath and suppresses the very events keeping it open. Its docstring recommends `"snap"` for x and
  `"smooth"` for y, which is what the component should default to, and both should be parameters.
- **`set_text(text)`** is the whole point, and owns the invariant *this window is always correctly sized for
  its content*: park offscreen, show there, set the value, `guiutils.wait_for_resize`, then place and show
  or hide again depending on whether it should be visible. It must do this even when hidden, since the next
  hover of a stale-sized window is itself a glitch frame.
- **Every piece of this already exists and is assembled by hand in two apps** — `wait_for_resize`,
  `compute_tooltip_position_scalar`, `min_size=[1, 1]`, the offscreen park. That is the argument that the
  component is the right shape rather than a new abstraction: it is the assembly that is missing, not the
  parts.
- **Not callable from the render thread.** `wait_for_resize` raises there rather than hanging. Flash
  messages arrive on the DPG callback thread, so the normal path is fine; the constraint belongs in the
  docstring.

## How `WidgetFlash` reaches it

Decided 2026-08-20 (Juha): **duck-typing.** `message_target` may be a DPG widget id *or* anything exposing
`text` for get/set, and `WidgetFlash` routes through two small helpers instead of calling `dpg.set_value`
directly. `set_text_under_flash` takes the same treatment.

Rejected: a separate `message_tooltip` parameter. It is more honest at the signature and costs one more
parameter on a class just narrowed from three target roles to one paint list plus a message — and the
polymorphism here is real rather than a special case, since any future self-sizing widget works unchanged.

## Order

1. ~~`Tooltip` plus its tests~~ — **done 2026-08-20.**
2. ~~The `WidgetFlash` duck-typing~~ — **done 2026-08-20.**
3. Migrate Librarian's flashing tooltips onto it. **Not started.**

Each step leaves the tree green and is committable on its own.

### What building 1 and 2 changed about the design

**The settle is a two-frame state machine, not a wait.** The brief assumed `wait_for_resize`, on the
grounds that a flash's message arrives on the callback thread where waiting is legal. It does — but the
*restore* comes from `WidgetFlash.finish`, which the animator calls **on the render thread**, and nothing
ticked by the animator may wait for a frame. Handing that off to a worker thread was tried and is worse: it
leaves the worker inside DPG at teardown, which is the shutdown fault already logged against the vendored
markdown renderer, and it dumped core in the test suite. So the sweeper — which ticks every frame anyway —
carries a queued change one step per tick: apply-and-park-offscreen, then place. No wait, no thread, and
the interesting half becomes testable, since `animator.render_frame()` drives it in the suite exactly as
the render loop does.

### Step 3, as far as it has been thought through

`flash_button` should accept a `Tooltip` in its `tooltip=` parameter and adapt, rather than every call site
knowing how a `Tooltip` is put together: `also_flash=(tip.window, tip.caption)` for the paint list, and
`message_target=tip` so the text goes through the staged path. That keeps `text=` for the plain
`dpg.tooltip` case, which the apps still use everywhere a caption never changes.

Then the call sites: each `with dpg.tooltip(button): dpg.add_text(caption)` whose caption is replaced by a
flash message becomes a `Tooltip(button, caption)`. Only the ones that *flash* need it — a caption written
once is better off as a `dpg.tooltip`, and the component's docstring says so.

**A modal cannot spawn a window, so a modal's tooltips cannot be migrated** (Juha, 2026-08-20). Being a
window is what makes this one placeable, so there is no version of it that works there. `FileDialog` — the
file picker the constellation's apps share — is a modal, which rules it out on top of the reason below; the
messagebox is a modal too and has no tooltips. The app windows that host them are not modal, so everything
migrated here is unaffected.

**Visualizer is in scope too** (Juha, 2026-08-20), not only Librarian. Counting `flash_button` sites that
carry a caption: Librarian about 13 (`app.py` 10, `chat_controller.py` 3) and Visualizer about 4
(`word_cloud.py` 2, `info_panel.py` 2), plus the two sites that build a `WidgetFlash` directly and have
tooltips — Librarian's delete-subtree confirmation and its send gate. Cherrypick's one flash has no
tooltip, and the file dialog's flashes write to a notification *line* rather than to a caption, so neither
of those apps is affected.

The second consumer is the point rather than a bonus: `CLAUDE.md` observes that wiring a component into a
second app is what surfaces the API gaps the first one hid, and doing Librarian alone would leave this one
shaped by exactly one caller.

**This step wants live testing**, since it changes what a real tooltip does on screen and the whole point
is how it looks. Librarian's copy-chatlog button is the sharpest case: a three-line caption replaced by a
one-line acknowledgment, which is the jump that started all this.

## Afterwards

`raven.visualizer.annotation` and the XDot viewer's tooltip are hand-rolled instances of this. Folding them
in is the obvious follow-on and explicitly *not* part of this: both carry app-specific behaviour (the
annotation double-buffers a whole content group and tracks which entries it lists) and the component should
earn its shape on the simple case first. Note it as a candidate once the component has a second consumer.

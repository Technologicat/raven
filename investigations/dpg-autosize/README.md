# The one-frame autosize lag, and which escapes from it are real

A DPG window with `autosize=True` fits itself to the content it measured on the **previous** frame. Change
the content and there is a frame where the two disagree. Whether anyone sees that frame turns out to depend
on things the geometry API cannot report, which is why this took five probes.

Investigated 2026-08-20, on DPG 2.3.1, prompted by Raven-librarian's toolbutton tooltips: a caption
replaced by a flash message renders once at the old size — clipped when it grows, skirted when it shrinks.
The distilled result is in `dpg-notes.md`, *An autosize window is one frame behind its content*; this
directory holds the apparatus and the reasoning.

## The trap, first

`get_item_rect_size` reports the stale size on frame +1 in **every** case — mutated, hidden and reshown,
even a window created that instant. Read on its own it says every route glitches equally, and three
conclusions were drawn from it before anyone checked the screen. Two were wrong.

**What is drawn is not what is reported**, and only a screenshot distinguishes them. Juha caught this by
knowing something no measurement had: a tooltip has never once been seen to glitch on its first hover,
which the reported sizes say it should. Every claim here past that point is from pixels.

## What was measured

| script | question | answer |
|---|---|---|
| `probe_reported_size.py` | does *how* the content changed matter — `set_value`, a widget swap, an explicit `width=`? | reported size is stale for all but an explicitly sized window; and `configure_item(tooltip, width=…)` raises `width keyword does not exist`, so a tooltip cannot be sized |
| `probe_hidden_metrics.py` | does hiding something across the change skip the stale frame? | no — a hidden item is not laid out and keeps its old metrics, so the first visible frame is the stale one. This is why `reposition_subtitle` parks the subtitle *offscreen* rather than hiding it |
| `probe_drawn_fresh_vs_mutated.py` | is the stale frame actually drawn? | **a window ImGui has not laid out before is not drawn at all** on that frame, then appears fitted. An existing, mutated one is drawn clipped |
| `probe_drawn_reshown.py` | does hiding and re-showing an existing window get that same treatment? | no — drawn clipped. Only genuine first layout is withheld |
| `probe_tooltip_offset.py` | where does DPG place a tooltip, relative to the cursor? | **(25, 10)** — and no API will tell you: `rect_min` raises for a window, `get_item_pos` reports `(0, 0)`. Diff two captures of the *same* hovered button at two cursor positions, so the hover highlight cancels and only the tooltip is left. Re-measure on a DPG upgrade |
| `probe_tooltip_rebuild.py` | so: delete the tooltip and build a new one holding the message? | clean when the content **shrinks**, clipped when it **grows**. Entry to a flash is a shrink, the restore is a grow, so this fixes half of it |
| `probe_zorder.py` | may a tooltip window be built *during* the render loop, as a chat view rebuilds? | yes, where the app sets a primary window: one created 60 frames in draws in front of it. `dpg-notes.md`'s standing warning about lazy creation is about two ordinary windows |
| `probe_many_tooltips.py` | can a chat view afford one window per tooltip — 14 buttons per message, several hundred on screen? | yes: 400 hidden root windows, 400 `dpg.tooltip`s and 400 bare buttons all cost about 1 ms/frame, and which wins changes per run. **Pass `vsync=False`**, or all three report 16.666 ms and answer nothing |

`read_screenshot_colors.py` is the shared reader — brightest/dominant colours out of a capture, used to
tell a rendered colour from a coverage-weighted blend.

## How to run one

Each probe maps a real window; DPG cannot lay out without one (`render_dearpygui_frame` aborts on a GLFW
assertion when the viewport was never shown), so **these steal keyboard focus** — say so before running one
on a shared desktop. The pixel probes render exactly one frame and then sleep, so whatever is on screen *is*
that frame and can be captured from another process:

```bash
python probe_drawn_fresh_vs_mutated.py &          # writes PIXEL_STAGE naming the frame it is holding
import -window "$(xdotool search --onlyvisible --name RAVEN_PIXEL_PROBE | head -1)" shot.png
```

`probe_tooltip_rebuild.py` additionally wants the mouse parked over its button, since a real `dpg.tooltip`
only exists while hovered:

```bash
eval $(xwininfo -id "$WID" | awk '/Absolute upper-left X/{print "X="$4} /Absolute upper-left Y/{print "Y="$4}')
xdotool mousemove $((X + 120)) $((Y + 22))
```

## Where it leaves the fix

Nothing that operates on *when the content changes* works, and the one thing that does — an explicit size —
is precisely what `dpg.tooltip` withholds. What works is what `raven.visualizer.annotation` and the XDot
viewer already do without either having written down why: **do not use `dpg.tooltip` for anything whose
contents change.** A tooltip is a window with no title bar, and an app-owned window can be positioned — so
it can be parked offscreen, shown there, allowed to autosize on a frame nobody sees, and only then moved to
the cursor. `guiutils.wait_for_resize` is the wait, and it raises rather than hanging if called from the
render thread, so the settle belongs on a callback or background thread.

The remaining alternative is to give autosize nothing to react to — a fixed-size child, a spacer, a padded
message. All the same bargain: the size stops changing because it is always the largest state's size, so a
one-line message sits in a three-line box.

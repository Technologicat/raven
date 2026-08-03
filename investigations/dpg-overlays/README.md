# Floating overlay windows in DearPyGui: size them, or they bite

Raven builds several affordances as borderless, backgroundless windows floating over a panel — the
Visualizer's and Librarian's scroll-end flashers, Librarian's jump-to-latest pill, the XDot viewer's
tooltip. Two DPG behaviours make that pattern sharper than it looks, and both were found the same way: a
window that looked right and misbehaved.

## An overlay is opaque to the mouse across its whole rect

Not just where its widgets are, and `no_background=True` does not change it. Measured 2026-08-03 with
`mouse_capture_probe.py`: an overlay window whose only widget is a button at the top, with ~200 px of empty
window below it, laid over a scrollable panel.

| wheel aimed at | panel `y_scroll` |
|---|---|
| the overlay's **empty** area | 0.0 → **0.0** — swallowed |
| clear of the overlay | 0.0 → **195.0** — scrolled |

So an oversized overlay is an invisible dead zone over whatever it covers. This is the measurement behind
`ScrollEndFlasher`'s design, which splits its overlay into two windows — one per end — rather than covering
the panel with one; its comment asserts the behaviour, and this is the check.

## …and an autosize window is silently ~100 px tall unless told otherwise

`dpg.add_window`'s `min_size` defaults to about `[100, 100]`, and the theme style
`mvStyleVar_WindowMinSize` does not override it. Recorded in `dpg-notes.md`, "Window sizing", where it was
first met as phantom blank space under a tooltip's content.

**And it clamps an explicit size, not only an autosize one** — which is the part that had gone unnoticed,
and the reason this bundle exists. Measured 2026-08-03:

| window | asked for | actual rect |
|---|---|---|
| explicit, no `min_size` | 400×48 | **400×100** |
| explicit, `min_size=[1, 1]` | 400×48 | 400×48 |
| autosize, no `min_size` | autosize, one small button | **100×100** |

Combined with the capture result above, the two turn a cosmetic-looking default into a functional bug, and
it had shipped in three places:

- **`ScrollEndFlasher`'s two bands** are created `width=w, height=48`. They were really 100 px tall, so the
  top band laid 52 px of dead zone over the panel and the bottom band 52 px past its lower edge, for as long
  as a flash lasted — defeating the very split into two windows that was meant to avoid capturing the wheel.
  Found because Juha asked whether the flasher had the same problem as the pill. It did.
- **The Visualizer's annotation tooltip** is autosize, so any annotation shorter than 100 px carried a skirt
  of empty window over the plot.
- **Librarian's jump-to-latest pill** holds one small button; its window ran ~100 px tall and hung past the
  chat panel's bottom edge over the composer. Its placement arithmetic also has to measure from the
  *button* rather than the window, since the window adds a padding ring around it.

The XDot viewer's tooltip already passed `min_size=[1, 1]`, having met this first.

**The rule both give: size a floating overlay to its content, and where the content cannot fill the rect,
use several windows rather than one large one.**

## Files

- `mouse_capture_probe.py` — the wheel test above. Self-driving; needs `xdotool` and a real X session, and
  takes keyboard focus for about seven seconds.

The `min_size` half needs no script: set it wrong and the blank space is visible immediately.

## Where this ended up

- `dpg-notes.md`, "Window sizing" — the `min_size` default.
- `raven/librarian/chat_controller.py` — the pill's window passes `min_size=[1, 1]`, with the reasoning at
  the call site.
- `raven/common/gui/xdotwidget/widget.py` — the tooltip window, which met the `min_size` default first.
- `raven/common/gui/animation.py` — `ScrollEndFlasher`'s two-window split.

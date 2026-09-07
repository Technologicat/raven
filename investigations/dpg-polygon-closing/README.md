# How to close a stroked polygon outline in DearPyGui

**The question:** the xdot renderer draws an unfilled polygon by appending the first point to a
`draw_polyline`, with a comment saying `draw_polygon` "doesn't close automatically for stroke". Is that
true, and is appending the point the right way to close one?

**The answer, measured 2026-09-08 against dearpygui 2.x:** the comment is right and the spelling is wrong.

| spelling | result |
|---|---|
| `draw_polygon(points, color=…, thickness=…)` | strokes an **open** path — the edge from the last vertex back to the first is not drawn at all, so a four-point rectangle comes out as three sides. It fills correctly; only the stroke is open. |
| `draw_polyline(points + [points[0]], …)` | closes the outline, but **not the join at the seam**: repeating the vertex ends one stroke and starts another, so two butt caps meet there instead of a mitre and the outer corner is left unfilled. |
| `draw_polyline(points, closed=True, …)` | closes it **with a join**. The one to use. |
| `draw_polyline(points + [points[0], points[1]], …)` | also joins — the extra segment overlaps the first and buries both caps. Works, and there is no reason to prefer it over the flag. |

## Why it hides

The notch is as wide as the stroke, and a renderer that scales line width by zoom draws a hairline at 1:1
and a bite out of the corner once a reader has zoomed in. So it surfaces long after the code was written,
in the view nobody tests in, and it reads as "something is wrong with that corner" rather than as a
drawing-API question.

**It lands on the outline's first vertex**, wherever whoever built the outline put that. In Raven this
showed up twice on 2026-09-08, on a chat graph message box and on a pointer pill, at the top left and the
leftmost point respectively — one defect, two shapes. GraphViz, by contrast, writes a box starting at its
*top right*: parsing `raven/xdot_viewer/testdata/test_callgraph.dot` gives
`[(805, 225), (590, 225), (590, 261), (805, 261)]`.

## Dashes are the exception

A dash pattern is drawn as one open polyline per mark, and each mark is *meant* to have caps, so `closed`
has nothing to do there. The closing edge still has to be walked or the outline is dashed on three sides
and bare on the fourth, so the dash walk keeps `points + [points[0]]`.

## The scripts

Both map a window briefly, render, and read the frame buffer back through `dpg.output_frame_buffer`.
`output_frame_buffer` needs a shown viewport — an unmapped one aborts in GLFW with
`Assertion 'window != NULL' failed` — which is why these take focus for a couple of seconds.

| file | what it renders |
|---|---|
| `probe_polygon_vs_polyline.py` → `polygon_vs_polyline.png` | left: the appended-vertex polyline, notch visible at the top left. Middle and right: `draw_polygon` with a thickness, with and without an explicit transparent fill — both missing their left edge. |
| `probe_closing_spellings.py` → `closing_spellings.png` | left: the appended vertex again. Middle: `closed=True`. Right: the two-point overlap. The last two are clean. |

The conclusion is written up for daily use in `dpg-notes.md`, "Closing a stroked outline"; what lives here
is the apparatus and the pictures.

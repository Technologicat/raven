"""Does a DPG window swallow the mouse wheel over a part of itself that holds no widget?

It does, and that governs how any floating overlay in Raven has to be built: an overlay window is opaque to
the mouse across its whole rect, not just where its widgets are, and `no_background=True` does not change
that. An oversized overlay is therefore an invisible dead zone over whatever it covers.

    python investigations/dpg-overlays/mouse_capture_probe.py

Needs `xdotool` and a real X session; drives itself, and takes keyboard focus for about seven seconds.

An overlay window (no background, one button at the very top, ~200 px of empty window below it) sits over a
scrollable child window. Wheel events are aimed at two points and the panel's scroll position is read after
each. Measured output:

    before wheel, over overlay empty area      y_scroll=0.0
    after wheel over overlay EMPTY area        y_scroll=0.0
    before wheel, clear of overlay             y_scroll=0.0
    after wheel clear of overlay               y_scroll=195.0

The panel scrolls only where the overlay is not. Two consequences already load-bearing in Raven:

- `ScrollEndFlasher` splits its overlay into two windows, one per end, rather than covering the panel with
  one. Its own comment says this is to avoid capturing the wheel; this is the measurement behind that.
- The jump-to-latest pill passes `min_size=[1, 1]`. Without it the window is silently ~100 px tall
  regardless of content (see `dpg-notes.md`, "Window sizing"), and the surplus would hang past the chat
  panel's bottom edge as a dead zone over the composer.

So: size a floating overlay to its content, and where the content cannot fill the rect, use several windows
rather than one large one.
"""

import subprocess

import dearpygui.dearpygui as dpg

TITLE = "raven overlay mouse capture probe"

dpg.create_context()
dpg.create_viewport(title=TITLE, width=480, height=420)
dpg.setup_dearpygui()

with dpg.window(tag="main"):
    with dpg.child_window(tag="panel", width=440, height=360):
        for i in range(120):
            dpg.add_text(f"line {i}")
dpg.set_primary_window("main", True)

# The overlay: no background, one button at the very top, and ~200 px of empty window below it. The empty
# part is the region under test.
with dpg.window(tag="overlay", show=True, no_title_bar=True, no_background=True,
                no_move=True, no_resize=True, no_collapse=True, no_scrollbar=True,
                no_scroll_with_mouse=True, no_focus_on_appearing=True,
                width=200, height=200, pos=[60, 60]):
    dpg.add_button(tag="ovbtn", label="overlay button")

dpg.show_viewport()


def x(*args: str) -> None:
    subprocess.run(["xdotool", *args], check=False, capture_output=True)


results = []
wid = None
for frame in range(420):
    dpg.render_dearpygui_frame()

    if frame == 30:
        out = subprocess.run(["xdotool", "search", "--name", TITLE],
                             capture_output=True, text=True).stdout.split()
        wid = out[-1] if out else None
        x("windowactivate", "--sync", wid)

    # Point 1: inside the overlay's rect, well below its only widget.
    elif frame == 60:
        x("mousemove", "--window", wid, "120", "200")
    elif frame == 80:
        results.append(("before wheel, over overlay empty area", dpg.get_y_scroll("panel")))
        for _ in range(3):
            x("click", "--window", wid, "5")  # button 5 = wheel down
    elif frame == 140:
        results.append(("after wheel over overlay EMPTY area", dpg.get_y_scroll("panel")))

    # Point 2: the control — over the panel, clear of the overlay entirely.
    elif frame == 200:
        x("mousemove", "--window", wid, "350", "300")
    elif frame == 220:
        results.append(("before wheel, clear of overlay", dpg.get_y_scroll("panel")))
        for _ in range(3):
            x("click", "--window", wid, "5")
    elif frame == 300:
        results.append(("after wheel clear of overlay", dpg.get_y_scroll("panel")))

dpg.destroy_context()

for label, y_scroll in results:
    print(f"{label:42} y_scroll={y_scroll}")

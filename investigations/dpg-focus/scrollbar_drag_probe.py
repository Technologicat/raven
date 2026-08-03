"""Is a scrollbar drag on a child window detectable at all?

ImGui handles the drag internally and raises no event Raven can hook — `should_follow_tail`'s docstring
names this as the reason it compares positions rather than listening for scroll events. So any fix for the
drag-creep described in `README.md` depends on some predicate being true for the drag's duration. This finds
out which, by holding the mouse button down on a panel's scrollbar and printing the candidates.

    python investigations/dpg-focus/scrollbar_drag_probe.py

Needs `xdotool` and a real X session; drives itself, and takes keyboard focus for about eight seconds.

Measured output, stable across runs:

    label                         active  focused  hovered  mousedn
    idle                           False    False    False    False
    hovering scrollbar             False    False     True    False
    button down on scrollbar       False     True    False     True
    dragged, still held            False     True    False     True
    released                       False     True     True    False

Three things to read off it:

- **`is_item_active` on the panel is the wrong signal** — the obvious candidate, and False throughout.
- **`is_mouse_button_down(mvMouseButton_Left) and is_item_focused(panel)`** is true for exactly the duration
  of the press. That is the detector a compensator would key on. It is not specific to the scrollbar — a
  click-and-hold in the panel body satisfies it too — which is harmless for the intended use, since holding
  the reader's position while they hold the button is wanted either way.
- **A click focuses a child window.** `dpg.focus_item` cannot (see `README.md`), but a click can, scrollbar
  included, and the focus persists after release. So child windows are unaskable, not unfocusable.

**What this does not reproduce is the creep itself**, and the reason is worth recording so nobody re-runs it
expecting to see one. Pressing at this coordinate lands on the scrollbar *track* rather than on the thumb,
and ImGui responds with auto-repeat paging: the position jumps by exactly one panel height at a time while
the button is held, which is a different behaviour from a thumb drag. Hitting the thumb reliably means
computing its extent from the content ratio, and there is little reason to: the creep is already established
from a real session's logs, in numbers no synthetic drag would improve on. See `README.md`.
"""

import subprocess

import dearpygui.dearpygui as dpg

TITLE = "raven scrollbar drag probe"

dpg.create_context()
dpg.create_viewport(title=TITLE, width=440, height=340)
dpg.setup_dearpygui()

with dpg.window(tag="main"):
    with dpg.child_window(tag="panel", width=400, height=260):
        for i in range(80):
            dpg.add_text(f"line {i}")
dpg.set_primary_window("main", True)
dpg.show_viewport()

detection = []

# The scrollbar sits at the panel's right edge; the panel starts at roughly (8, 30) in client coordinates.
SB_X, SB_Y = 400, 60


def snap(label: str) -> None:
    detection.append((label,
                      dpg.is_item_active("panel"), dpg.is_item_focused("panel"),
                      dpg.is_item_hovered("panel"),
                      dpg.is_mouse_button_down(dpg.mvMouseButton_Left)))


def x(*args: str) -> None:
    subprocess.run(["xdotool", *args], check=False, capture_output=True)


wid = None
for frame in range(420):
    dpg.render_dearpygui_frame()

    if frame == 30:
        out = subprocess.run(["xdotool", "search", "--name", TITLE],
                             capture_output=True, text=True).stdout.split()
        wid = out[-1] if out else None
        x("windowactivate", "--sync", wid)
    elif frame == 45:
        # Park the pointer clear of the panel first. Without this the `idle` row reports whatever the mouse
        # happened to be resting on, so the run is not reproducible — caught exactly that way.
        x("mousemove", "--window", wid, "10", "320")
    elif frame == 60:
        snap("idle")
    elif frame == 90:
        x("mousemove", "--window", wid, str(SB_X), str(SB_Y))
    elif frame == 110:
        snap("hovering scrollbar")
    elif frame == 120:
        x("mousedown", "1")
    elif frame == 150:
        snap("button down on scrollbar")
    elif frame == 200:
        x("mousemove", "--window", wid, str(SB_X), str(SB_Y + 60))
    elif frame == 260:
        snap("dragged, still held")
    elif frame == 320:
        x("mouseup", "1")
    elif frame == 360:
        snap("released")

dpg.destroy_context()

print(f"{'label':<28} {'active':>7} {'focused':>8} {'hovered':>8} {'mousedn':>8}")
for label, active, focused, hovered, mousedown in detection:
    print(f"{label:<28} {str(active):>7} {str(focused):>8} {str(hovered):>8} {str(mousedown):>8}")

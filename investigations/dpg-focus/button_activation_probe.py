"""If keyboard focus rests on a button, do Space and Enter fire its callback?

The question decides whether a button is a safe place to park keyboard focus. Both Librarian and the
Visualizer need somewhere to put focus after a text field is done with — Librarian after sending a message,
the Visualizer after accepting a search — and a child window cannot take focus at all, so the nearest
button is the obvious candidate. It is only obvious if pressing Space while reading cannot re-send the
message.

Drives itself with `xdotool`, so it needs a real X session and takes keyboard focus for about nine seconds.

    python investigations/dpg-focus/button_activation_probe.py

Expected output, and the result the two apps rely on:

    label                 btn.foc  fld.foc  fld.act  callback fired at frames
    focus on btn             True    False    False  []
    after Space              True    False    False  []
    after Enter              True    False    False  []
    after Up                 True    False    False  []

The callback never fires: DPG does not enable ImGui's keyboard-nav activation, so a focused button is inert
and parking focus on one is safe. Note the second column too — focusing the button really does take focus
off the text field, which is what makes it usable as a parking spot in the first place.

Re-run this before trusting that conclusion again after a DPG upgrade; the send path in
`raven/librarian/app.py` and the search-accept path in `raven/visualizer/app.py` both depend on it.
"""

import subprocess

import dearpygui.dearpygui as dpg

TITLE = "raven dpg-focus button activation probe"
fired = []

dpg.create_context()
dpg.create_viewport(title=TITLE, width=420, height=260)
dpg.setup_dearpygui()

with dpg.window(tag="main"):
    dpg.add_child_window(tag="panel", width=400, height=100)
    dpg.add_input_text(tag="field", multiline=True, width=400, height=60)
    dpg.add_button(tag="btn", label="send",
                   callback=lambda: fired.append(dpg.get_frame_count()))
dpg.set_primary_window("main", True)

dpg.show_viewport()

log = []


def snap(label: str) -> None:
    log.append((label, dpg.is_item_focused("btn"), dpg.is_item_focused("field"),
                dpg.is_item_active("field"), list(fired)))


def x(*args: str) -> None:
    subprocess.run(["xdotool", *args], check=False, capture_output=True)


wid = None
for frame in range(560):
    dpg.render_dearpygui_frame()
    if frame == 30:
        out = subprocess.run(["xdotool", "search", "--name", TITLE],
                             capture_output=True, text=True).stdout.split()
        wid = out[-1] if out else None
        x("windowactivate", "--sync", wid)
    elif frame == 90:
        dpg.focus_item("btn")
    elif frame == 120:
        snap("focus on btn")
    elif frame == 180:
        x("key", "space")
    elif frame == 240:
        snap("after Space")
    elif frame == 300:
        x("key", "Return")
    elif frame == 360:
        snap("after Enter")
    elif frame == 420:
        x("key", "Up")
    elif frame == 480:
        snap("after Up")

dpg.destroy_context()

print(f"{'label':<20} {'btn.foc':>8} {'fld.foc':>8} {'fld.act':>8}  callback fired at frames")
for label, button_focused, field_focused, field_active, frames in log:
    print(f"{label:<20} {str(button_focused):>8} {str(field_focused):>8} {str(field_active):>8}  {frames}")

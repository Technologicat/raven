"""Which of `is_item_focused` / `is_item_active` tracks "this text field owns the caret"?

Answers it by driving a real DPG window through the four states a text field passes through — untouched,
clicked into, typed in, escaped out of — and printing both predicates at each. Synthesizes the input with
`xdotool`, so it needs a real X session and it takes keyboard focus for about ten seconds. Nothing else on
the desktop receives the keystrokes: every one is aimed at this window.

    python investigations/dpg-focus/focus_states_probe.py

Expected output, and the result recorded in `dpg-notes.md` under "Keyboard input":

    label                             fld.foc  fld.act  value                  focused_item
    window active, no click              True    False  ''                     field
    after click in field f120            True     True  ''                     field
    while typed-in f240                  True     True  'hello'                field
    after Escape f330                    True    False  ''                     field
    after click on panel f440           False    False  ''                     field

Two things to read off it. `is_item_focused` is already True on the first row, where nobody has touched the
field — ImGui gives nav focus to the first navigable item of a newly focused window by itself — so gating a
global hotkey on it makes that hotkey dead from app start. `is_item_active` is False there and True only
while the field is genuinely being edited, which is the distinction the hotkey handlers need. Note also that
`get_focused_item()` keeps naming the field on the last row, where `is_item_focused` on that same field is
False, so it is not a usable cross-check.

The rows that need no synthetic input are asserted continuously by
`raven/common/gui/tests/test_focus_semantics.py` (`pytest --run-gui`). This script covers the rest — the
click, the typing and the Escape — which the test suite cannot reach without driving the keyboard. Re-run it
when a DPG upgrade might have changed focus handling.
"""

import subprocess

import dearpygui.dearpygui as dpg

TITLE = "raven dpg-focus states probe"

dpg.create_context()
dpg.create_viewport(title=TITLE, width=420, height=320)
dpg.setup_dearpygui()

with dpg.window(tag="main"):
    dpg.add_child_window(tag="panel", width=400, height=140)
    dpg.add_input_text(tag="field", multiline=True, width=400, height=70)
    dpg.add_button(tag="btn", label="a button")
dpg.set_primary_window("main", True)

dpg.show_viewport()

report = []
wid = None


def snap(label: str) -> None:
    report.append((label,
                   dpg.is_item_focused("field"), dpg.is_item_active("field"),
                   repr(dpg.get_value("field"))[:20], dpg.get_focused_item()))


def x(*args: str) -> None:
    subprocess.run(["xdotool", *args], check=False, capture_output=True)


# Frame schedule. At ~60 fps, 30 frames is about half a second — loose enough that the exact frame rate
# does not matter, since every step is separated from the next by far more than one frame of slack.
for frame in range(600):
    dpg.render_dearpygui_frame()

    if frame == 30:
        out = subprocess.run(["xdotool", "search", "--name", TITLE],
                             capture_output=True, text=True).stdout.split()
        wid = out[-1] if out else None
        x("windowactivate", "--sync", wid)
    elif frame == 60:
        snap("window active, no click")
    elif frame == 90:
        # Click into the multiline field: it sits below the 140 px child window.
        x("mousemove", "--window", wid, "200", "200", "click", "1")
    elif frame in (120, 150):
        snap(f"after click in field f{frame}")
    elif frame == 180:
        x("type", "--window", wid, "--delay", "30", "hello")
    elif frame in (240, 270):
        snap(f"while typed-in f{frame}")
    elif frame == 300:
        x("key", "Escape")
    elif frame in (330, 360):
        snap(f"after Escape f{frame}")
    elif frame == 400:
        # Click on the child window (the scrollable panel), which is what a reader does to leave the field.
        x("mousemove", "--window", wid, "200", "60", "click", "1")
    elif frame in (440, 480):
        snap(f"after click on panel f{frame}")
    elif frame == 520:
        x("key", "Up")
    elif frame in (550, 580):
        snap(f"after Up f{frame}")

dpg.destroy_context()

print(f"{'label':<32} {'fld.foc':>8} {'fld.act':>8}  {'value':<22} focused_item")
for label, focused, active, value, focused_item in report:
    print(f"{label:<32} {str(focused):>8} {str(active):>8}  {value:<22} {focused_item}")

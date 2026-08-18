"""When does focusing a text field select its contents?

Written 2026-08-18 to explain an intermittency: `FileDialog`'s Ctrl+F sometimes replaced what
was in the find field and sometimes did not. The answer is that *every* effective `focus_item`
selects the contents — including one aimed at a field that already had focus — so the
intermittency was really about whether the focus request succeeded at all.

Synthesizes keystrokes, so it takes the keyboard for a couple of seconds. Measured on DPG 2.3.1.
"""
import shutil
import subprocess
import time
import dearpygui.dearpygui as dpg

TITLE = "raven select-all probe"

def render(n=12):
    for _ in range(n):
        dpg.render_dearpygui_frame()

def type_text(text):
    found = subprocess.run(["xdotool", "search", "--name", TITLE], capture_output=True, text=True)
    ids = found.stdout.split()
    if not ids:
        return False
    subprocess.run(["xdotool", "windowactivate", "--sync", ids[-1]], check=False)
    time.sleep(0.3)
    subprocess.run(["xdotool", "type", "--delay", "40", text], check=False)
    time.sleep(0.4)
    return True

if not shutil.which("xdotool"):
    raise SystemExit("xdotool required")

dpg.create_context()
dpg.create_viewport(title=TITLE, width=400, height=220)
dpg.setup_dearpygui()
with dpg.window(tag="w", width=360, height=180):
    with dpg.child_window(tag="panel", width=340, height=120):
        dpg.add_input_text(tag="field", width=300)
        dpg.add_button(tag="park", label="park (same child)")
dpg.set_primary_window("w", True)
dpg.show_viewport()
render()

def scenario(label, steps):
    dpg.set_value("field", "")
    dpg.focus_item("park"); render()
    dpg.set_value("field", "SEED"); render()
    for step in steps:
        step(); render()
    type_text("x")
    render()
    value = dpg.get_value("field")
    verdict = "SELECTED ALL (seed replaced)" if value == "x" else "kept the seed"
    print(f"  {label:44s} -> {value!r:<12} {verdict}")

scenario("focus moves park -> field, then type",
         [lambda: dpg.focus_item("field")])
scenario("field already focused, focus_item again",
         [lambda: dpg.focus_item("field"), lambda: dpg.focus_item("field")])
scenario("focus never moved to the field at all",
         [])

dpg.destroy_context()

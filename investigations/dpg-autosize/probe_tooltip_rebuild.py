"""The proposed fix, on a real `dpg.tooltip`, photographed frame by frame.

A: mutate the caption of the live tooltip            (today's behaviour)
B: delete the tooltip, build a new one with the message
C: ...then delete that and rebuild the original      (the way back)
"""
import json, time
import dearpygui.dearpygui as dpg
from raven.common.gui import utils as guiutils

SHORT = "Copy to clipboard [F8]\n    no modifier: as-is\n    with Shift: include node IDs"
MSG = "Copied to clipboard!"
PAUSE = 1.7

dpg.create_context()
dpg.create_viewport(title="RAVEN_TIP_PROBE", width=900, height=320)
guiutils.bootup(font_size=20)
dpg.setup_dearpygui()
with dpg.window(no_title_bar=True, no_scrollbar=True, width=900, height=320, pos=(0, 0)):
    button = dpg.add_button(label="hover me", width=240, height=44)
    with dpg.tooltip(button) as tip:
        caption = dpg.add_text(SHORT)
dpg.show_viewport()

state = {"tip": tip, "caption": caption, "n": 0}
log = []
def step(tag):
    dpg.render_dearpygui_frame()
    with guiutils.nonexistent_ok():
        w, h = dpg.get_item_rect_size(state["tip"])
        log.append({"tag": tag, "w": w, "h": h})
    with open("TIP_STAGE", "w") as f:
        f.write(tag)
    time.sleep(PAUSE)

def rebuild(text):
    """Delete the tooltip and build a new one holding `text`."""
    state["n"] += 1
    dpg.delete_item(state["tip"])
    with dpg.tooltip(button, tag=f"probe_tip_{state['n']}") as new_tip:
        state["caption"] = dpg.add_text(text)
    state["tip"] = new_tip

for _ in range(120):        # long enough for the shell to park the mouse and the tooltip to appear
    dpg.render_dearpygui_frame()
step("0-hovered, resting caption")

dpg.set_value(state["caption"], MSG)          # A
step("A-mutated+1")
step("A-mutated+2")

dpg.set_value(state["caption"], SHORT)
for _ in range(30):
    dpg.render_dearpygui_frame()

rebuild(MSG)                                   # B
step("B-rebuilt+1")
step("B-rebuilt+2")

for _ in range(60):   # let the message tooltip settle, as a real ~1 s flash would
    dpg.render_dearpygui_frame()
rebuild(SHORT)                                 # C
step("C-restored+1")
step("C-restored+2")

with open("TIP_LOG", "w") as f:
    json.dump(log, f)
with open("TIP_STAGE", "w") as f:
    f.write("done")
t0 = time.monotonic()
while dpg.is_dearpygui_running() and time.monotonic() - t0 < 2.0:
    dpg.render_dearpygui_frame()
dpg.destroy_context()

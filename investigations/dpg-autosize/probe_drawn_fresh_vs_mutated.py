"""Is the stale frame ever *drawn*, or only reported?

Renders exactly one frame at a time and then stops, so whatever is on screen is that frame and can be
photographed. `get_item_rect_size` is recorded alongside, so the two can be compared directly.
"""
import json, os, time
import dearpygui.dearpygui as dpg
from raven.common.gui import utils as guiutils

LONG = "a considerably longer caption than the one before it"
SHORT = "short"
PAUSE = 1.6   # long enough for the screenshotter to catch this frame

dpg.create_context()
dpg.create_viewport(title="RAVEN_PIXEL_PROBE", width=1000, height=300)
guiutils.bootup(font_size=20)
dpg.setup_dearpygui()
with dpg.window(no_title_bar=True, no_scrollbar=True, width=1000, height=300, pos=(0, 0)):
    dpg.add_text("probe")
dpg.show_viewport()
for _ in range(60):
    dpg.render_dearpygui_frame()

log = []
def step(tag, win):
    """Render exactly one frame, record the reported size, then hold that frame on screen."""
    dpg.render_dearpygui_frame()
    w, h = dpg.get_item_rect_size(win)
    log.append({"tag": tag, "reported_w": w, "reported_h": h})
    with open("PIXEL_STAGE", "w") as f:
        f.write(tag)
    time.sleep(PAUSE)

# Case 1: a window created fresh, already holding the long text.
with dpg.window(label="fresh", autosize=True, pos=(40, 60), no_move=True) as fresh:
    dpg.add_text(LONG)
step("fresh+1", fresh)
step("fresh+2", fresh)
dpg.delete_item(fresh)
for _ in range(10):
    dpg.render_dearpygui_frame()

# Case 2: an existing settled window whose text changes.
with dpg.window(label="mutated", autosize=True, pos=(40, 60), no_move=True) as mutated:
    t = dpg.add_text(SHORT)
for _ in range(30):
    dpg.render_dearpygui_frame()
dpg.set_value(t, LONG)
step("mutated+1", mutated)
step("mutated+2", mutated)

with open("PIXEL_LOG", "w") as f:
    json.dump(log, f)
with open("PIXEL_STAGE", "w") as f:
    f.write("done")
t0 = time.monotonic()
while dpg.is_dearpygui_running() and time.monotonic() - t0 < 2.0:
    dpg.render_dearpygui_frame()
dpg.destroy_context()

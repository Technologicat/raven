"""Does hiding and re-showing an existing window get the not-drawn-until-measured treatment too?

If yes, the fix is `hide -> set_value -> show`. If no, the tooltip has to be built fresh.
"""
import json, time
import dearpygui.dearpygui as dpg
from raven.common.gui import utils as guiutils

LONG = "a considerably longer caption than the one before it"
SHORT = "short"
PAUSE = 1.6

dpg.create_context()
dpg.create_viewport(title="RAVEN_PIXEL_PROBE2", width=1000, height=300)
guiutils.bootup(font_size=20)
dpg.setup_dearpygui()
with dpg.window(no_title_bar=True, no_scrollbar=True, width=1000, height=300, pos=(0, 0)):
    dpg.add_text("probe")
dpg.show_viewport()
for _ in range(60):
    dpg.render_dearpygui_frame()

log = []
def step(tag, win):
    dpg.render_dearpygui_frame()
    w, h = dpg.get_item_rect_size(win)
    log.append({"tag": tag, "reported_w": w, "reported_h": h})
    with open("PIXEL2_STAGE", "w") as f:
        f.write(tag)
    time.sleep(PAUSE)

with dpg.window(label="reshown", autosize=True, pos=(40, 60), no_move=True) as win:
    t = dpg.add_text(SHORT)
for _ in range(30):
    dpg.render_dearpygui_frame()

# hide, change the text while hidden, show again
dpg.hide_item(win)
dpg.set_value(t, LONG)
for _ in range(3):
    dpg.render_dearpygui_frame()
dpg.show_item(win)
step("reshown+1", win)
step("reshown+2", win)

with open("PIXEL2_LOG", "w") as f:
    json.dump(log, f)
with open("PIXEL2_STAGE", "w") as f:
    f.write("done")
t0 = time.monotonic()
while dpg.is_dearpygui_running() and time.monotonic() - t0 < 2.0:
    dpg.render_dearpygui_frame()
dpg.destroy_context()

"""Does hiding across the stale frame skip it, or only defer it?

(a) Does a hidden text item report metrics at all?
(b) Hide the window, change the text, show it again — is the first visible frame correct?
"""
import json
import time
import dearpygui.dearpygui as dpg
from raven.common.gui import utils as guiutils

SHORT, LONG = "short", "a considerably longer caption than the one before it"
dpg.create_context()
dpg.create_viewport(title="RAVEN_HIDE_PROBE", width=900, height=420)
guiutils.bootup(font_size=20)
dpg.setup_dearpygui()

with dpg.window(no_title_bar=True, no_scrollbar=True, width=900, height=420, pos=(0, 0)):
    dpg.add_text("probe")
with dpg.window(label="autosize", autosize=True, pos=(60, 120), no_move=True) as auto:
    caption = dpg.add_text(SHORT)

dpg.show_viewport()
log = []
def note(tag):
    log.append({"frame": dpg.get_frame_count(), "tag": tag,
                "win_w": dpg.get_item_rect_size(auto)[0],
                "text_w": dpg.get_item_rect_size(caption)[0]})
def frames(n):
    for _ in range(n):
        dpg.render_dearpygui_frame()

frames(60)
note("visible, short")

# (a) metrics of a hidden item
dpg.hide_item(caption)
frames(3)
note("caption hidden")
dpg.set_value(caption, LONG)
frames(3)
note("hidden, text changed to long")
dpg.show_item(caption)
for i in range(4):
    frames(1)
    note(f"caption reshown +{i + 1}")

# reset
dpg.set_value(caption, SHORT)
frames(30)
note("reset to short")

# (b) hide the whole window across the change
dpg.hide_item(auto)
dpg.set_value(caption, LONG)
frames(3)
note("window hidden, text changed")
dpg.show_item(auto)
for i in range(4):
    frames(1)
    note(f"window reshown +{i + 1}")

with open("HIDE_LOG", "w") as f:
    json.dump(log, f)
t0 = time.monotonic()
while dpg.is_dearpygui_running() and time.monotonic() - t0 < 3.0:
    dpg.render_dearpygui_frame()
dpg.destroy_context()

"""How many frames does an autosize window take to catch up with changed content?

Three ways of changing it, measured the same way:
  A  set_value on the one text widget           (what `WidgetFlash` does today)
  B  hide one text widget, show a longer one    (Juha's proposal)
  C  set_value, plus an explicit size the same frame
"""
import json, time
import dearpygui.dearpygui as dpg
from raven.common.gui import utils as guiutils

SHORT = "short"
LONG = "a considerably longer caption than the one before it"

dpg.create_context()
dpg.create_viewport(title="RAVEN_RESIZE_PROBE", width=900, height=420)
guiutils.bootup(font_size=20)
dpg.setup_dearpygui()

with dpg.window(no_title_bar=True, no_scrollbar=True, no_move=True, no_resize=True,
                width=900, height=420, pos=(0, 0)):
    button = dpg.add_button(label="hover me for a tooltip", width=300, height=40)
    with dpg.tooltip(button) as tooltip:
        tip_a = dpg.add_text(SHORT)
        tip_b = dpg.add_text(LONG, show=False)

# An autosize window is the same auto-fit machinery, and is measurable whether or not the tooltip renders.
with dpg.window(label="autosize", autosize=True, pos=(60, 120), no_move=True) as auto:
    auto_a = dpg.add_text(SHORT)
    auto_b = dpg.add_text(LONG, show=False)

dpg.show_viewport()
log = []

def measure(tag):
    log.append({"frame": dpg.get_frame_count(), "tag": tag,
                "auto_w": dpg.get_item_rect_size(auto)[0],
                "tip_w": dpg.get_item_rect_size(tooltip)[0]})

def settle(n=12, tag="settle"):
    for _ in range(n):
        dpg.render_dearpygui_frame()
    measure(tag)

def watch(tag, n=6):
    for i in range(n):
        dpg.render_dearpygui_frame()
        measure(f"{tag}+{i + 1}")

settle(60, "baseline-short")

# A: set_value
dpg.set_value(auto_a, LONG); dpg.set_value(tip_a, LONG)
measure("A-set_value(before any frame)")
watch("A")

# back to short
dpg.set_value(auto_a, SHORT); dpg.set_value(tip_a, SHORT); settle(30, "reset")

# B: show/hide swap
dpg.hide_item(auto_a); dpg.show_item(auto_b)
dpg.hide_item(tip_a); dpg.show_item(tip_b)
measure("B-swap(before any frame)")
watch("B")

# back to short
dpg.show_item(auto_a); dpg.hide_item(auto_b)
dpg.show_item(tip_a); dpg.hide_item(tip_b); settle(30, "reset")

# C: set_value plus an explicit width, computed without rendering
text_w = dpg.get_text_size(LONG)[0]
padded = int(text_w + 2 * guiutils.DPG_WINDOW_PADDING)
dpg.set_value(auto_a, LONG); dpg.set_value(tip_a, LONG)
dpg.configure_item(auto, autosize=False, width=padded)
measure(f"C-set_value+width({padded}, text {text_w:.0f})")
watch("C")

with open("RESIZE_LOG", "w") as f:
    json.dump(log, f, indent=1)
t0 = time.monotonic()
while dpg.is_dearpygui_running() and time.monotonic() - t0 < 4.0:
    dpg.render_dearpygui_frame()
dpg.destroy_context()

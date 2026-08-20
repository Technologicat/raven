"""When the tooltip is placed, is the size it is placed by the new one or the old one?

The settle parks the window offscreen for one frame so autosize can fit it. `_place` then reads
`get_item_rect_size` — and the reported size is known to lag the content by a frame, which would mean the
placement is computed from the size the tooltip *used* to be. Near a viewport edge that decides whether the
tooltip goes below the cursor or above it, so a wrong answer is a visibly misplaced frame.

The button sits near the bottom, where a three-line caption cannot fit below the cursor and a one-line one can.
"""
import time

import dearpygui.dearpygui as dpg

from raven.common.gui import animation as gui_animation
from raven.common.gui import tooltip as gui_tooltip
from raven.common.gui import utils as guiutils

THREE_LINE = "Copy this conversation to clipboard [F8]\n    no modifier: as-is\n    with Shift: include message node IDs"
ONE_LINE = "Copied to clipboard! (as-is)"

dpg.create_context()
dpg.create_viewport(title="RAVEN_SETTLE_SIZE", width=600, height=300, vsync=False)
dpg.setup_dearpygui()
guiutils.setup_themes()
with dpg.window(tag="main") as main:
    button = dpg.add_button(label="C", width=26, height=26, pos=(30, 240))  # near the bottom edge
dpg.set_primary_window(main, True)
tip = gui_tooltip.Tooltip(button, THREE_LINE)
dpg.show_viewport()

print("PROBE_READY", flush=True)
deadline = time.perf_counter() + 25.0
while not tip._shown and time.perf_counter() < deadline:
    dpg.render_dearpygui_frame()
    gui_animation.animator.render_frame()
    time.sleep(0.005)
for _ in range(30):
    dpg.render_dearpygui_frame()
    gui_animation.animator.render_frame()
    time.sleep(0.005)

def show(tag, n=10):
    for i in range(n):
        dpg.render_dearpygui_frame()
        gui_animation.animator.render_frame()
        print(f"  {tag} +{i}: pos={[round(v) for v in dpg.get_item_pos(tip.window)]} "
              f"reported_size={[round(v) for v in dpg.get_item_rect_size(tip.window)]} "
              f"text={dpg.get_value(tip.caption)[:22]!r}", flush=True)
        time.sleep(0.02)

print(f"at rest: pos={[round(v) for v in dpg.get_item_pos(tip.window)]} size={[round(v) for v in dpg.get_item_rect_size(tip.window)]}", flush=True)
print("--- shrink to the one-line message ---", flush=True)
tip.text = ONE_LINE
show("shrink", 6)
print("--- restore the three-line caption (the reported case) ---", flush=True)
tip.text = THREE_LINE
show("restore", 6)
print("DONE", flush=True)
dpg.destroy_context()

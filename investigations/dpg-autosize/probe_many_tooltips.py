"""What does a hidden root window cost per frame, against the `dpg.tooltip` it would replace?

Librarian builds 14 tooltips per chat message, so a long view holds several hundred. `Tooltip` makes each
one a root window; the question is whether that is free the way a hidden `dpg.tooltip` is.
"""
import sys
import time

import dearpygui.dearpygui as dpg

MODE = sys.argv[1]  # "none" | "tooltip" | "window"
N = int(sys.argv[2])

dpg.create_context()
dpg.create_viewport(title="RAVEN_TOOLTIP_COST", width=500, height=400, vsync=False)
dpg.setup_dearpygui()

with dpg.window(tag="main") as main:
    for i in range(N):
        b = dpg.add_button(label=f"b{i}", width=20, height=20, pos=(10 + (i % 20) * 22, 10 + (i // 20) * 22))
        if MODE == "tooltip":
            with dpg.tooltip(b):
                dpg.add_text(f"caption for button {i}\nsecond line")
        elif MODE == "window":
            with dpg.window(show=False, no_title_bar=True, autosize=True, min_size=[1, 1],
                            no_focus_on_appearing=True):
                dpg.add_text(f"caption for button {i}\nsecond line", wrap=-1)
            with dpg.item_handler_registry() as reg:
                dpg.add_item_hover_handler(callback=lambda: None)
            dpg.bind_item_handler_registry(b, reg)
dpg.set_primary_window(main, True)
dpg.show_viewport()

for _ in range(120):  # warm up
    dpg.render_dearpygui_frame()
t0 = time.perf_counter()
FRAMES = 600
for _ in range(FRAMES):
    dpg.render_dearpygui_frame()
dt = time.perf_counter() - t0
print(f"RESULT {MODE} n={N}: {1000 * dt / FRAMES:.3f} ms/frame ({FRAMES / dt:.1f} fps)", flush=True)
dpg.destroy_context()

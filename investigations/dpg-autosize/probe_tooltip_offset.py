"""Where does DPG put a tooltip, relative to the cursor? Measured from the screen, because no API says.

`get_item_rect_min` raises for a window, `get_item_pos` returns (0, 0) for a tooltip, and
`guiutils.get_widget_pos` inherits that. So: hover one button, capture at two cursor positions, and diff.
The button's hover highlight is identical in both and cancels; what is left is the tooltip, twice.

Re-measure on a DPG upgrade - `raven.common.gui.tooltip.Tooltip` defaults to the answer, so that a tooltip
built on it lands where a plain `dpg.tooltip` beside it would. Answer on DPG 2.3.1, 2026-08-20: (25, 10).

See this directory's README for how to run a probe and capture the frames.
"""
import time
import dearpygui.dearpygui as dpg
from raven.common.gui import utils as guiutils

dpg.create_context()
dpg.create_viewport(title="RAVEN_OFFSET_PIXELS", width=800, height=400)
guiutils.bootup(font_size=20)
dpg.setup_dearpygui()
with dpg.window(no_title_bar=True, no_scrollbar=True, width=800, height=400, pos=(0, 0)):
    dpg.add_button(label="hover me", width=300, height=340, tag="probe_button")
    with dpg.tooltip("probe_button"):
        dpg.add_text("a tooltip")
dpg.show_viewport()
t0 = time.monotonic()
while dpg.is_dearpygui_running() and time.monotonic() - t0 < 40.0:
    dpg.render_dearpygui_frame()
dpg.destroy_context()

"""Is a window created during the render loop drawn in front of the primary window?

`dpg-notes.md` says lazy creation once put a tooltip *behind* it. That was 2026-04-03 and the note does
not say whether a primary window was set. It decides whether per-message tooltips can be built as the
chat view rebuilds.
"""
import dearpygui.dearpygui as dpg

dpg.create_context()
dpg.create_viewport(title="RAVEN_ZORDER_PROBE", width=400, height=300)
dpg.setup_dearpygui()

with dpg.window(tag="main") as main:
    with dpg.drawlist(width=380, height=260):
        dpg.draw_rectangle((0, 0), (380, 260), fill=(255, 0, 0, 255), color=(255, 0, 0, 255))
dpg.set_primary_window(main, True)
dpg.show_viewport()

for _ in range(60):
    dpg.render_dearpygui_frame()

# ...now, mid-loop, the way a chat-view rebuild would.
with dpg.window(no_title_bar=True, autosize=True, min_size=[1, 1], no_focus_on_appearing=True, show=False) as late:
    dpg.add_text("LATE WINDOW")
dpg.set_item_pos(late, [60, 60])
dpg.show_item(late)

for _ in range(30):
    dpg.render_dearpygui_frame()
print("PROBE_READY", flush=True)
for _ in range(600):
    dpg.render_dearpygui_frame()
dpg.destroy_context()

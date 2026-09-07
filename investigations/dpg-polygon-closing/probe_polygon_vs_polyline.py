"""Does a thick closed polyline join at its seam, and does draw_polygon close a stroke itself?

Three rectangles, same points, drawn three ways, rendered offscreen and saved.
"""
import dearpygui.dearpygui as dpg

dpg.create_context()
dpg.create_viewport(width=640, height=260)
dpg.setup_dearpygui()

pts = lambda ox: [(ox + 20.0, 40.0), (ox + 160.0, 40.0), (ox + 160.0, 180.0), (ox + 20.0, 180.0)]
T = 24.0

with dpg.window(tag="w", no_title_bar=True, no_move=True, no_resize=True):
    with dpg.drawlist(width=640, height=260, tag="dl"):
        # 1. what the renderer does now: polyline with the first point appended
        p = pts(0)
        dpg.draw_polyline(p + [p[0]], color=(200, 200, 200, 255), thickness=T)
        # 2. draw_polygon with a thickness
        p = pts(200)
        dpg.draw_polygon(p, color=(200, 200, 200, 255), thickness=T)
        # 3. draw_polygon, does it also fill by default?
        p = pts(400)
        dpg.draw_polygon(p, color=(200, 200, 200, 255), fill=(0, 0, 0, 0), thickness=T)

dpg.set_primary_window("w", True)
dpg.show_viewport()
for _ in range(4):
    dpg.render_dearpygui_frame()
dpg.output_frame_buffer("/tmp/claude-1000/-home-jje-Documents-koodit-raven/2d343187-294b-4b44-b201-e72321271d1e/scratchpad/join.png")
for _ in range(3):
    dpg.render_dearpygui_frame()
print("saved")
dpg.destroy_context()

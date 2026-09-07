"""Does draw_polyline(closed=True) join the seam that appending the first point leaves open?"""
import dearpygui.dearpygui as dpg

dpg.create_context()
dpg.create_viewport(width=640, height=260)
dpg.setup_dearpygui()

def pts(ox):
    return [(ox + 20.0, 40.0), (ox + 160.0, 40.0), (ox + 160.0, 180.0), (ox + 20.0, 180.0)]

T = 24.0
C = (200, 200, 200, 255)

with dpg.window(tag="w", no_title_bar=True, no_move=True, no_resize=True):
    with dpg.drawlist(width=640, height=260, tag="dl"):
        p = pts(0)
        dpg.draw_polyline(p + [p[0]], color=C, thickness=T)          # now
        p = pts(200)
        dpg.draw_polyline(p, closed=True, color=C, thickness=T)      # the flag
        p = pts(400)
        dpg.draw_polyline(p + [p[0], p[1]], color=C, thickness=T)    # the overlap trick

dpg.set_primary_window("w", True)
dpg.show_viewport()
for _ in range(4):
    dpg.render_dearpygui_frame()
dpg.output_frame_buffer("/tmp/claude-1000/-home-jje-Documents-koodit-raven/2d343187-294b-4b44-b201-e72321271d1e/scratchpad/join2.png")
for _ in range(3):
    dpg.render_dearpygui_frame()
print("saved")
dpg.destroy_context()

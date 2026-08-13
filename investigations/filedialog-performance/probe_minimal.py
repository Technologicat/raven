"""The minimal shape of the context-recreation crash: does it need anything beyond rendered frames?

`frames` renders on a shown viewport, then destroys and recreates. `noframes` is identical but never
renders. Nothing else is in either — no FileDialog, no textures, no widgets.
"""
import sys
import dearpygui.dearpygui as dpg

n_frames = 60 if sys.argv[1] == "frames" else 0
for run in range(2):
    dpg.create_context()
    dpg.create_viewport(title=f"minimal {run}", width=400, height=300)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    for _ in range(n_frames):
        dpg.render_dearpygui_frame()
    dpg.destroy_context()
    print(f"run {run} ok", flush=True)

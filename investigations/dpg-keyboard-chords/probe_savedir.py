"""Does the "Will pick:" line follow the name field live, in a save-mode directory picker?

That is the pose editor's configuration — `pick="dir", save_mode=True` — where `ok` answers from the typed
name rather than from the listing, and the usual gesture is type-type-type-Enter. If the line only refreshed
on a listing change it would lag the field, promising a path the user has already moved past.
"""

import os
import time

import dearpygui.dearpygui as dpg

from raven.common.gui import utils as guiutils
from raven.vendor.file_dialog.fdialog import FileDialog

dpg.create_context()
guiutils.bootup(font_size=20)
dpg.create_viewport(title="probe_savedir", width=1600, height=900, x_pos=40, y_pos=40)
dpg.setup_dearpygui()

fd = FileDialog(title="save-mode directory picker", tag="fd_savedir",  # tag
                pick="dir", save_mode=True, filter_list=[""],
                default_path=os.path.expanduser("~"),
                callback=lambda paths: print(f"CALLBACK {paths}", flush=True))
dpg.show_viewport()

shown = False
t0 = time.monotonic()
last = 0.0
while dpg.is_dearpygui_running():
    now = time.monotonic()
    if not shown and now - t0 > 1.0:
        shown = True
        fd.show_file_dialog()
    if shown and now - last > 0.5:
        last = now
        with guiutils.nonexistent_ok():
            print(f"field={dpg.get_value(f'ex_search_{fd.instance_tag}')!r} "
                  f"line={dpg.get_value(fd.text_target)!r}", flush=True)
    dpg.render_dearpygui_frame()

dpg.destroy_context()

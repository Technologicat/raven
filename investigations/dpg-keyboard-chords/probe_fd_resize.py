"""Probe: does `FileDialog(no_resize=False)` actually work in Raven's fork?

The parameter exists and defaults to True, and nothing in Raven passes it — so whether the layout
survives a drag of the resize grip has never been exercised. The layout computes several spacers from
`self.width` at build time, which is the reason to doubt it.

Environment:
    PROBE_FONT=1     call `guiutils.bootup` first, as the real apps do. Without it everything renders in
                     DPG's built-in ProggyClean, which is ASCII-only, and the thumbnail grid's "…"
                     truncation marker shows as a missing-glyph box that reads as a bug in the grid.
    PROBE_PATH=...   directory to open (default: home).
    PROBE_THUMBS=1   start in thumbnail grid mode, to check that the grid reflows on resize.
"""

import os
import time

import dearpygui.dearpygui as dpg

from raven.vendor.file_dialog.fdialog import FileDialog

dpg.create_context()
if os.environ.get("PROBE_FONT") == "1":
    from raven.common.gui import utils as guiutils
    # 20 is Raven's global standard — every app passes it, so a layout floor measured at any other size
    # is an underestimate of the one that ships.
    guiutils.bootup(font_size=20)
dpg.create_viewport(title="probe_fd_resize", width=1600, height=900, x_pos=40, y_pos=40)
dpg.setup_dearpygui()


def log(msg):
    print(f"{time.time():.3f} {msg}", flush=True)


fd = FileDialog(title="resize test",
                tag="fd_resize_test",  # tag
                no_resize=False,
                modal=True,
                default_path=os.path.expanduser(os.environ.get("PROBE_PATH", "~")),
                show_thumbnails=(os.environ.get("PROBE_THUMBS") == "1"),
                callback=lambda paths: log(f"callback: {paths}"))

dpg.show_viewport()

shown = False
t0 = time.monotonic()
last = 0.0
while dpg.is_dearpygui_running():
    now = time.monotonic()
    if not shown and now - t0 > 1.0:
        shown = True
        # Called from the render loop, so the dialog's own `split_frame` for button alignment cannot
        # wait and says so. Harmless here; the real apps open it from a callback.
        fd.show_file_dialog()
        log("dialog shown")
    if shown and now - last > 1.0:
        last = now
        log(f"window size={dpg.get_item_width('fd_resize_test')}x"  # tag
            f"{dpg.get_item_height('fd_resize_test')}")  # tag
    dpg.render_dearpygui_frame()

dpg.destroy_context()

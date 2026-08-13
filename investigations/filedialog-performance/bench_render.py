"""How much does a big FileDialog listing cost *per rendered frame*?

The headless benchmarks showed item creation and deletion are cheap (~60 us/row), which leaves rendering
as the only remaining candidate for the seconds-long open. ImGui submits every row of a table each frame
unless the table has a clipper, so the hypothesis is that per-frame cost scales with row count and that
`clipper=True` flattens it.

Needs a mapped window: `render_dearpygui_frame` requires one. vsync off, or every measurement is 16.7 ms.
The dialog window must actually be *shown* — it is created with `show=False`, and a hidden window renders
nothing, which is how the first version of this script measured vsync and nothing else.

One configuration per process: creating a second context after a shown viewport core-dumps.

Run: python bench_render.py {noclipper|clipper}
"""

import os
import pathlib
import shutil
import statistics
import sys
import tempfile
import time

import dearpygui.dearpygui as dpg

import raven.vendor.file_dialog.fdialog as fdialog_module
from raven.vendor.file_dialog.fdialog import FileDialog

SIZES = [0, 500, 2500]
WARMUP_FRAMES = 60
MEASURE_FRAMES = 200


def make_dir(root: pathlib.Path, n: int) -> pathlib.Path:
    d = root / f"n{n}"
    d.mkdir()
    for i in range(n):
        ext = [".pdf", ".png", ".txt", ".py", ".bin"][i % 5]
        (d / f"file_{i:05d}{ext}").write_bytes(b"x" * 64)
    return d


def render_frames(n: int) -> list[float]:
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        dpg.render_dearpygui_frame()
        times.append(time.perf_counter() - t0)
    return times


def main() -> None:
    use_clipper = (len(sys.argv) > 1 and sys.argv[1] == "clipper")

    root = pathlib.Path(tempfile.mkdtemp(prefix="fdialog_renderbench_"))
    try:
        dirs = {n: make_dir(root, n) for n in SIZES}

        dpg.create_context()
        dpg.create_viewport(title=f"fdialog bench (clipper={use_clipper})", width=1200, height=800, vsync=False)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_viewport_vsync(False)

        original_table = fdialog_module.dpg.table
        if use_clipper:
            def table_with_clipper(*args, **kwargs):
                kwargs["clipper"] = True
                return original_table(*args, **kwargs)
            fdialog_module.dpg.table = table_with_clipper

        dialog = FileDialog(tag="bench_dialog", title="bench", modal=False, width=1100, height=750)
        fdialog_module.dpg.table = original_table
        # `show_file_dialog` calls `split_frame`, which cannot be called from the render thread — and this
        # script's main thread is it. Show the window directly; only the OK/Cancel alignment is skipped.
        dpg.show_item(dialog.tag)
        render_frames(WARMUP_FRAMES)

        for n in SIZES:
            os.chdir(dirs[n])
            t0 = time.perf_counter()
            dialog.reset_dir(default_path=str(dirs[n]))
            t_build = time.perf_counter() - t0

            first = render_frames(1)[0]
            render_frames(WARMUP_FRAMES)
            times = render_frames(MEASURE_FRAMES)
            print(f"clipper={str(use_clipper):<5} {n:>5} rows  build {t_build:6.3f}s  "
                  f"first frame {1000 * first:7.1f}ms  median {1000 * statistics.median(times):6.2f}ms  "
                  f"worst {1000 * max(times):6.2f}ms", flush=True)

        os.chdir("/")
        dpg.destroy_context()
    finally:
        os.chdir("/")
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
    sys.exit(0)

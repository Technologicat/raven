"""What does the sort re-application cost, on top of building the listing?

`reset_dir` ends in `reapply_latest_sort`, which re-runs the table sort callback whenever the user has
ever clicked a column header. That callback reads four DPG items per row back out of the tree
(`get_item_children` twice, `get_item_user_data` twice) — a per-row cost that the plain build does not
have, and one that only appears after the first header click, i.e. not in a fresh-session measurement.
"""

import os
import pathlib
import shutil
import tempfile
import time

import dearpygui.dearpygui as dpg

from raven.vendor.file_dialog.fdialog import FileDialog

SIZES = [500, 1000, 2000, 4000]


def make_dir(root: pathlib.Path, n: int) -> pathlib.Path:
    d = root / f"n{n}"
    d.mkdir()
    for i in range(n):
        ext = [".pdf", ".png", ".txt", ".py", ".bin"][i % 5]
        (d / f"file_{i:05d}{ext}").write_bytes(b"x" * 64)
    return d


def main() -> None:
    dpg.create_context()
    dpg.create_viewport(width=100, height=100)
    dpg.setup_dearpygui()

    root = pathlib.Path(tempfile.mkdtemp(prefix="fdialog_sortbench_"))
    try:
        dirs = {n: make_dir(root, n) for n in SIZES}
        dialog = FileDialog(tag="bench_dialog", title="bench", modal=False)
        table = f"explorer_{dialog.instance_tag}"
        sort_callback = dpg.get_item_callback(table)
        name_column = dpg.get_item_children(table, 0)[0]

        print(f"{'files':>6} {'build':>9} {'sort':>9} {'build+sort':>11}")
        for n in SIZES:
            d = dirs[n]
            os.chdir(d)

            t0 = time.perf_counter()
            dialog.reset_dir(default_path=str(d))
            t_build = time.perf_counter() - t0

            # One header click, as the user would.
            t0 = time.perf_counter()
            sort_callback(table, [[name_column, 1]])
            t_sort = time.perf_counter() - t0

            # Now the sort is remembered, so every later rebuild pays for it too.
            t0 = time.perf_counter()
            dialog.reset_dir(default_path=str(d))
            t_both = time.perf_counter() - t0

            print(f"{n:>6} {t_build:>8.3f}s {t_sort:>8.3f}s {t_both:>10.3f}s")

        os.chdir("/")
    finally:
        shutil.rmtree(root, ignore_errors=True)
        dpg.destroy_context()


if __name__ == "__main__":
    main()

"""Where does FileDialog's open time actually go?

Headless: no render loop, so this measures item *creation* and *deletion* only, not layout or draw.
That is the half the deferred item blames ("the listing is fully materialized as DPG widgets"), so it is
the half worth measuring first — if creation alone is already seconds, the diagnosis holds without
needing a mapped window.

Run: python bench_fdialog.py
"""

import os
import pathlib
import shutil
import tempfile
import time

import dearpygui.dearpygui as dpg

from raven.vendor.file_dialog.fdialog import FileDialog

SIZES = [100, 500, 1000, 2000, 4000]


def make_dir(root: pathlib.Path, n: int) -> pathlib.Path:
    d = root / f"n{n}"
    d.mkdir()
    for i in range(n):
        # A realistic-ish mix: the icon lookup in `_makefile` walks a dict of extension tuples.
        ext = [".pdf", ".png", ".txt", ".py", ".bin"][i % 5]
        (d / f"file_{i:05d}{ext}").write_bytes(b"x" * 64)
    return d


def count_items() -> int:
    return len(dpg.get_all_items())


def main() -> None:
    dpg.create_context()
    dpg.create_viewport(width=100, height=100)
    dpg.setup_dearpygui()

    root = pathlib.Path(tempfile.mkdtemp(prefix="fdialog_bench_"))
    try:
        dirs = {n: make_dir(root, n) for n in SIZES}

        dialog = FileDialog(tag="bench_dialog", title="bench", modal=False, show_hidden_files=False)
        table = f"explorer_{dialog.instance_tag}"

        print(f"{'files':>6} {'stat only':>10} {'reset_dir':>10} {'delete':>10} {'items':>8} {'per row':>9}")
        for n in SIZES:
            d = dirs[n]
            os.chdir(d)

            # The pure-filesystem part: what the listing costs before any DPG call.
            t0 = time.perf_counter()
            names = os.listdir(d)
            for name in names:
                os.path.getmtime(name)
                os.path.isdir(name)
                os.path.getsize(name)
            t_stat = time.perf_counter() - t0

            before = count_items()
            t0 = time.perf_counter()
            dialog.reset_dir(default_path=str(d))
            t_build = time.perf_counter() - t0
            created = count_items() - before

            t0 = time.perf_counter()
            for child in dpg.get_item_children(table, 1):
                dpg.delete_item(child)
            t_delete = time.perf_counter() - t0

            print(f"{n:>6} {t_stat:>9.3f}s {t_build:>9.3f}s {t_delete:>9.3f}s {created:>8} {created / n:>9.1f}")

        os.chdir("/")
    finally:
        shutil.rmtree(root, ignore_errors=True)
        dpg.destroy_context()


if __name__ == "__main__":
    main()

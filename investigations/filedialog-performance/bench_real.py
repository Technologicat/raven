"""Same measurement, against the real documents directory rather than synthetic files.

Synthetic files are 64 bytes, freshly written, and named uniformly — every stat is a page-cache hit and
every filename is short. A real directory of PDFs is the case the deferred item was filed about, so it is
the one that has to be measured before concluding where the seconds go.

Run: python bench_real.py [directory]
"""

import os
import pathlib
import sys
import time

import dearpygui.dearpygui as dpg

from raven.librarian import config as librarian_config
from raven.vendor.file_dialog.fdialog import FileDialog


def main() -> None:
    d = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(librarian_config.llm_docs_dir)
    entries = os.listdir(d)
    print(f"{d}: {len(entries)} entries")

    dpg.create_context()
    dpg.create_viewport(width=100, height=100)
    dpg.setup_dearpygui()
    try:
        t0 = time.perf_counter()
        dialog = FileDialog(tag="bench_dialog", title="bench", modal=False)
        t_construct = time.perf_counter() - t0
        print(f"  FileDialog construction (icon textures etc.): {t_construct:.3f}s")

        os.chdir(d)
        for trial in range(3):
            t0 = time.perf_counter()
            dialog.reset_dir(default_path=str(d))
            print(f"  reset_dir trial {trial}: {time.perf_counter() - t0:.3f}s, "
                  f"{len(dialog.shown_items)} rows shown")
        os.chdir("/")
    finally:
        os.chdir("/")
        dpg.destroy_context()


if __name__ == "__main__":
    main()

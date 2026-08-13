"""Which single ingredient turns a survivable DPG context cycle into a segfault?

Reproduced: two contexts in one process, each showing a viewport, constructing a `FileDialog` under the
*same tag*, showing its window, populating the listing, and rendering 60 frames — second context dies with
SIGSEGV. Each ingredient in isolation was already measured harmless, so the fault is in a combination.

Variants drop one ingredient each from the reproducing configuration. A variant that *survives* names the
ingredient that mattered.

Run: python probe_bisect.py           (parent: runs every variant)
     python probe_bisect.py <variant> (child: runs one)
"""

import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import dearpygui.dearpygui as dpg

from raven.vendor.file_dialog.fdialog import FileDialog

VARIANTS = [
    "full",             # the reproducing configuration
    "unique_tags",      # ...but a different dialog tag per context
    "no_show_dialog",   # ...but the dialog window is never shown
    "no_listing",       # ...but the listing is never populated
    "few_frames",       # ...but 5 frames instead of 60
    "vsync_on",         # ...but vsync left alone
]


def cycle(variant: str, run: int, d: pathlib.Path) -> None:
    dpg.create_context()
    dpg.create_viewport(title=f"bisect {variant} {run}", width=1200, height=800,
                        vsync=(variant == "vsync_on"))
    dpg.setup_dearpygui()
    dpg.show_viewport()
    if variant != "vsync_on":
        dpg.set_viewport_vsync(False)

    tag = f"bisect_dialog_{run}" if variant == "unique_tags" else "bisect_dialog"
    dialog = FileDialog(tag=tag, title="bisect", modal=False, width=1100, height=750)
    if variant != "no_show_dialog":
        dpg.show_item(dialog.tag)
    if variant != "no_listing":
        os.chdir(d)
        dialog.reset_dir(default_path=str(d))
        os.chdir("/")
    for _ in range(5 if variant == "few_frames" else 60):
        dpg.render_dearpygui_frame()
    dpg.destroy_context()


def main() -> None:
    if len(sys.argv) > 1:
        variant = sys.argv[1]
        root = pathlib.Path(tempfile.mkdtemp(prefix="bisect_"))
        d = root / "files"
        d.mkdir()
        for i in range(500):
            (d / f"f_{i:05d}.txt").write_bytes(b"x")
        try:
            for run in range(2):
                cycle(variant, run, d)
                print(f"  run {run} ok", flush=True)
        finally:
            os.chdir("/")
            shutil.rmtree(root, ignore_errors=True)
        return

    for variant in VARIANTS:
        result = subprocess.run([sys.executable, __file__, variant],
                                capture_output=True, text=True, timeout=200)
        status = {0: "survived", -11: "SIGSEGV", -6: "SIGABRT"}.get(result.returncode,
                                                                    f"exit {result.returncode}")
        runs = sum(1 for line in result.stdout.splitlines() if "ok" in line)
        print(f"{variant:<16} {runs} run(s) completed  -> {status}", flush=True)


if __name__ == "__main__":
    main()

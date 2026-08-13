"""Under a table clipper, does `dpg.is_item_visible` report which rows are on screen?

This decides the FileDialog thumbnail architecture. Thumbnails must be decoded only for rows the user can
actually see, so something has to answer "which rows are those". `is_item_visible` reports what was rendered
in the last frame, and a clipper submits only the on-screen range — so the two *should* line up. Should is
not measured.

The fallback if they don't: compute the range from `get_y_scroll` and a known row height, which needs the
row height to be right and stay right.

Also asked, because both answers are needed either way: does the same predicate work on a cell *inside* a
row (thumbnails live in a cell, not on the row), and what does an unclipped table report (if it says
everything is visible, that confirms the clipper is what makes the predicate informative).

Run: python probe_row_visibility.py {clipper|noclipper}
"""

import os
import pathlib
import shutil
import sys
import tempfile

import dearpygui.dearpygui as dpg

import raven.vendor.file_dialog.fdialog as fdialog_module
from raven.vendor.file_dialog.fdialog import FileDialog

N_FILES = 400


def render(n: int) -> None:
    for _ in range(n):
        dpg.render_dearpygui_frame()


def visible_rows(table: str) -> list[int]:
    """Indices of the table's rows that DPG reports as visible, in row order."""
    rows = dpg.get_item_children(table, 1)
    return [i for i, row in enumerate(rows) if dpg.is_item_visible(row)]


def visible_cells(table: str) -> list[int]:
    """Same, but asking the first cell of each row rather than the row itself."""
    rows = dpg.get_item_children(table, 1)
    out = []
    for i, row in enumerate(rows):
        cells = dpg.get_item_children(row, 1)
        if cells and dpg.is_item_visible(cells[0]):
            out.append(i)
    return out


def summarize(label: str, indices: list[int], total: int) -> None:
    if not indices:
        print(f"  {label:<28} none of {total}")
        return
    contiguous = (indices == list(range(indices[0], indices[-1] + 1)))
    print(f"  {label:<28} {len(indices):>4} of {total}, rows {indices[0]}..{indices[-1]}, "
          f"{'contiguous' if contiguous else 'NOT CONTIGUOUS'}")


def main() -> None:
    use_clipper = (sys.argv[1] == "clipper") if len(sys.argv) > 1 else True

    root = pathlib.Path(tempfile.mkdtemp(prefix="rowvis_"))
    d = root / "files"
    d.mkdir()
    for i in range(N_FILES):
        (d / f"file_{i:04d}.txt").write_bytes(b"x")

    try:
        dpg.create_context()
        dpg.create_viewport(title=f"row visibility ({'clipper' if use_clipper else 'no clipper'})",
                            width=1000, height=700)
        dpg.setup_dearpygui()
        dpg.show_viewport()

        original_table = fdialog_module.dpg.table
        if not use_clipper:
            def table_without_clipper(*args, **kwargs):
                kwargs["clipper"] = False
                return original_table(*args, **kwargs)
            fdialog_module.dpg.table = table_without_clipper

        dialog = FileDialog(tag="rowvis_dialog", title="row visibility", modal=False,
                            width=900, height=600)
        fdialog_module.dpg.table = original_table
        table = f"explorer_{dialog.instance_tag}"

        dpg.show_item(dialog.tag)
        os.chdir(d)
        dialog.reset_dir(default_path=str(d))
        render(30)

        total = len(dpg.get_item_children(table, 1))
        print(f"clipper={use_clipper}, {total} rows in the table")

        print("at the top:")
        summarize("is_item_visible(row)", visible_rows(table), total)
        summarize("is_item_visible(first cell)", visible_cells(table), total)

        # Scroll to the middle and again to the end; a predicate that tracks the viewport must move with it.
        max_scroll = dpg.get_y_scroll_max(table)
        for label, position in (("middle", max_scroll / 2), ("bottom", max_scroll)):
            dpg.set_y_scroll(table, position)
            render(10)
            print(f"scrolled to {label} (y_scroll={dpg.get_y_scroll(table):.0f} of {max_scroll:.0f}):")
            summarize("is_item_visible(row)", visible_rows(table), total)
            summarize("is_item_visible(first cell)", visible_cells(table), total)

        os.chdir("/")
        dpg.destroy_context()
    finally:
        os.chdir("/")
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()

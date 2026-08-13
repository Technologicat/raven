"""Minimal `FileDialog` demo.

Run it with:

    python -m raven.vendor.file_dialog.example

from the repository root. It has to be run as a module rather than as a script, because `fdialog` reaches
Raven's own packages for its button-flash animations and its search matching.
"""

import dearpygui.dearpygui as dpg

from .fdialog import FileDialog

dpg.create_context()

def pr(selected_files):  # file_dialog calls the callback with as argument a list containing the selected files
    dpg.delete_item("txt_child", children_only=True)
    if not selected_files:
        dpg.add_text("(cancelled)", parent="txt_child")
    for file in selected_files:
        dpg.add_text(file, parent="txt_child")

# A filter item is either a bare extension, which is its own label, or a (label, extensions) pair for when
# the set is too long to read as one — "every image Pillow opens" is 67 extensions.
fd = FileDialog(callback=pr,
                default_path="..",
                filter_list=[("Images", [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"]),
                             ("Text", [".txt", ".md", ".rst"]),
                             ".py",
                             ".*"],
                multi_selection=True)  # off by default; Ctrl+click to pick several

with dpg.window(label="hi", height=480, width=600):
    dpg.add_button(label="Show file dialog", callback=fd.show_file_dialog)
    dpg.add_text("Find is smart-case and matches fragments: 'ead' finds README, 'py test' finds test_x.py")
    dpg.add_child_window(width=-1, height=-1, tag="txt_child")


dpg.create_viewport(title='file_dialog example')
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()

import dearpygui.dearpygui as dpg
from fdialog import FileDialog

dpg.create_context()

def pr(selected_files): # file_dialog calls the callback with as argument a list containing the selected files
    dpg.delete_item("txt_child", children_only=True)
    for file in selected_files:
        dpg.add_text(file, parent="txt_child")

fd = FileDialog(callback=pr, default_path="..")

with dpg.window(label="hi", height=480, width=600):
    dpg.add_button(label="Show file dialog", callback=fd.show_file_dialog)
    dpg.add_child_window(width=-1, height=-1, tag="txt_child")


dpg.create_viewport(title='file_dialog example')
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
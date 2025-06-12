"""A simple modal messagebox for DPG.

Supports a title, content text, and customizable texts on any number of buttons (but in a single row).

One button may be configured as "OK", and pressing Enter (while the message box is open) will click it.
Similarly one may be configured as "Cancel", and pressing Esc (while the message box is open) will click it.

This module is licensed under the 2-clause BSD license, to facilitate integration anywhere.
"""

__all__ = ["modal_dialog",
           "modal_dialog_window_exists"]

from typing import Callable, List, Optional, Union

import dearpygui.dearpygui as dpg

from . import utils as guiutils

_modal_dialog_initialized = False
def _init():
    """Initialize this module. Only call after `setup_dearpygui`."""
    global _modal_dialog_initialized
    if _modal_dialog_initialized:
        return
    with dpg.window(label="Modal dialog title", autosize=True, modal=True, show=False, tag="modal_dialog_window"):
        dpg.add_text("Modal dialog message", wrap=600, tag="modal_dialog_message")
        dpg.add_separator()
        dpg.add_group(horizontal=True, tag="modal_dialog_button_group")
    with dpg.handler_registry(tag="modal_dialog_handler_registry"):  # global (whole viewport)
        dpg.add_key_press_handler(tag="modal_dialog_hotkeys_handler", callback=modal_dialog_hotkeys_callback)
    _modal_dialog_initialized = True

def modal_dialog_window_exists():
    # Sanity check. Just try to call *some* DPG function with the modal dialog window to check that the handle is valid (it isn't before `_init` has been called).
    try:
        dpg.get_item_alias("modal_dialog_window")
    except Exception:
        return False
    return True

def modal_dialog_hotkeys_callback(sender, app_data):
    if not modal_dialog_window_exists():
        return
    if not dpg.is_item_visible("modal_dialog_window"):
        return
    key = app_data
    if current_on_close is not None:
        if key == dpg.mvKey_Return:
            current_on_close(sender, app_data, user_data=current_ok_button)
        elif key == dpg.mvKey_Escape:
            current_on_close(sender, app_data, user_data=current_cancel_button)

current_on_close = None
current_ok_button = None
current_cancel_button = None
def modal_dialog(window_title: str,
                 message: str,
                 buttons: List[str],
                 ok_button: str,
                 cancel_button: str,
                 callback: Optional[Callable] = None,
                 centering_reference_window: Union[str, int] = None) -> None:
    """A simple modal dialog.

    `buttons`: Texts on buttons. These play a double role as return values.
    `ok_button`: When Enter is pressed, this value is returned.
    `cancel_button`: When Esc is pressed, or the window is closed by clicking on the "X", this value is returned.
    `callback`: CPS due to how DPG works. `modal_dialog` itself returns immediately; put the stuff you want to run
                (if any) after the modal closes into your `callback`.
    `centering_reference_window`: DPG tag or ID of parent window to center the dialog on.
    """
    _init()

    # Remove old buttons, if any
    for child in dpg.get_item_children("modal_dialog_button_group", slot=1):
        dpg.delete_item(child)

    def modal_dialog_callback(sender, app_data, user_data):
        global current_on_close
        global current_cancel_button
        current_on_close = None
        current_cancel_button = None
        dpg.hide_item("modal_dialog_window")
        if callback:
            callback(user_data)  # send the label of the clicked button
    global current_on_close
    global current_ok_button
    global current_cancel_button
    current_on_close = modal_dialog_callback
    current_ok_button = ok_button
    current_cancel_button = cancel_button

    dpg.configure_item("modal_dialog_window", label=window_title, on_close=modal_dialog_callback, user_data=cancel_button)
    dpg.set_value("modal_dialog_message", message)
    for label in buttons:
        dpg.add_button(label=label, width=75, callback=modal_dialog_callback, user_data=label, parent="modal_dialog_button_group")

    dpg.split_frame()  # We might be called when another modal (e.g. `FileDialog`) closes. Give it a chance to close first, to make DPG happy. (Otherwise this modal won't always show.)
    if centering_reference_window:
        guiutils.recenter_window("modal_dialog_window", reference_window=centering_reference_window)
    else:
        dpg.show_item("modal_dialog_window")

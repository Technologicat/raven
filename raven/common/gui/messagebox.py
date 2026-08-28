"""A simple modal messagebox for DPG.

Supports a title, content text, and customizable texts on any number of buttons (but in a single row).

One button may be configured as "OK", and pressing Enter (while the message box is open) will click it.
Similarly one may be configured as "Cancel", and pressing Esc (while the message box is open) will click it.

This module is licensed under the 2-clause BSD license, to facilitate integration anywhere.
"""

__all__ = ["modal_dialog_window_exists", "is_visible", "modal_dialog"]

from typing import Callable, List, Optional, Union

import dearpygui.dearpygui as dpg

from . import utils as guiutils

_modal_dialog_initialized = False
def _init():
    """Initialize this module. Only call after `setup_dearpygui`."""
    global _modal_dialog_initialized
    if _modal_dialog_initialized:
        return
    # Explicit parents, no `with`: this runs on first use rather than at app start, so the render loop is
    # already going and something else may be building widgets. DPG's container stack is one process-wide
    # global. See `dpg-notes.md`, "DPG parent management".
    window = dpg.add_window(label="Modal dialog title", autosize=True, modal=True, show=False, tag="modal_dialog_window")  # tag
    dpg.add_text("Modal dialog message", wrap=600, parent=window, tag="modal_dialog_message")  # tag
    dpg.add_separator(parent=window)
    dpg.add_group(horizontal=True, parent=window, tag="modal_dialog_button_group")  # tag
    registry = dpg.add_handler_registry(tag="modal_dialog_handler_registry")  # tag  # global (whole viewport)
    dpg.add_key_press_handler(parent=registry, tag="modal_dialog_hotkeys_handler", callback=modal_dialog_hotkeys_callback)  # tag
    _modal_dialog_initialized = True

def modal_dialog_window_exists():
    """Return whether the modal dialog window has been created yet (it has not, before `_init` runs)."""
    # `dpg.does_item_exist` is the direct question. The previous implementation called `get_item_alias` inside
    # a try/except, on the theory that any DPG call would raise for an invalid handle — but that call *returns
    # `None`* for an unknown tag instead of raising, so the except never fired and this always answered `True`.
    # Callers then went on to `dpg.is_item_visible("modal_dialog_window")`, where the tag resolves to `0` and
    # `get_item_state(0)` does raise, once per call.
    return dpg.does_item_exist("modal_dialog_window")  # tag

def is_visible() -> bool:
    """Return whether a modal dialog is on screen right now.

    Widgets that register *global* mouse handlers need this. DPG's `handler_registry` handlers fire
    regardless of what is under the cursor, and a geometric "is the mouse over me" test knows nothing
    about being covered — so without consulting this, a click on a dialog's button also lands on
    whatever the dialog is floating over.
    """
    return modal_dialog_window_exists() and dpg.is_item_visible("modal_dialog_window")  # tag

def modal_dialog_hotkeys_callback(sender, app_data):
    if not is_visible():
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
                The callback is expected to take one argument: the "return value" from the modal dialog,
                i.e. the label of the button that was chosen.
                The return value of the callback itself is ignored.
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

    # We might be called when another modal (e.g. `FileDialog`) closes. Give it a chance to close first,
    # to make DPG happy. (Otherwise this modal won't always show.)
    #
    # Not required: this is often the app's *error reporting* path, so raising here would replace the
    # error being reported with one about the reporting. A dialog that fails to appear is the lesser
    # evil, and the warning says which call site to defer to a frame callback.
    guiutils.split_frame(operation="modal_dialog: let a previously open modal close first",
                         required=False)
    if centering_reference_window:
        guiutils.recenter_window("modal_dialog_window", reference_window=centering_reference_window)
    else:
        dpg.show_item("modal_dialog_window")

"""DPG GUI utilities."""

__all__ = ["modal_dialog", "recenter_window"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import Callable, List, Optional, Union

import dearpygui.dearpygui as dpg

from .. import utils as raven_utils

# --------------------------------------------------------------------------------
# Simple modal dialog for OK/cancel

_modal_dialog_initialized = False
def init():
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
    # Sanity check. Just try to call *some* DPG function with the modal dialog window to check that the handle is valid (it isn't before `init` has been called).
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
    if key == dpg.mvKey_Escape and current_on_close is not None:
        current_on_close(sender, app_data, user_data=current_cancel_button)

current_on_close = None
current_cancel_button = None
def modal_dialog(window_title: str,
                 message: str,
                 buttons: List[str],
                 cancel_button: str,
                 callback: Optional[Callable] = None,
                 centering_reference_window: Union[str, int] = None) -> None:
    """A simple modal dialog.

    `buttons`: Texts on buttons. These play a double role as return values.
    `cancel_button`: When Esc is pressed, or the window is closed by clicking on the "X", this value is returned.
    `callback`: CPS due to how DPG works. `modal_dialog` itself returns immediately; put the stuff you want to run
                (if any) after the modal closes into your `callback`.
    `centering_reference_window`: Parent window to center the dialog on.
    """
    init()

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
    global current_cancel_button
    current_on_close = modal_dialog_callback
    current_cancel_button = cancel_button

    dpg.configure_item("modal_dialog_window", label=window_title, on_close=modal_dialog_callback, user_data=cancel_button)
    dpg.set_value("modal_dialog_message", message)
    for label in buttons:
        dpg.add_button(label=label, width=75, callback=modal_dialog_callback, user_data=label, parent="modal_dialog_button_group")

    dpg.split_frame()  # We might be called when another modal (e.g. `FileDialog`) closes. Give it a chance to close first, to make DPG happy. (Otherwise this modal won't always show.)
    if centering_reference_window:
        recenter_window("modal_dialog_window", reference_window=centering_reference_window)
    else:
        dpg.show_item("modal_dialog_window")

# --------------------------------------------------------------------------------
# Utilities

def recenter_window(thewindow: Union[str, int], *, reference_window: Union[str, int]):
    """Reposition `thewindow` (DPG ID or tag), if visible, so that it is centered on `reference_window`.

    To center on viewport, pass your maximized main window as `reference_window`.
    """
    if reference_window is None:
        return
    if thewindow is None:
        return
    # Sanity check. Just try to call *some* DPG function with `thewindow` to check that the handle is valid.
    try:
        dpg.get_item_alias(thewindow)
    except Exception:
        logger.debug(f"recenter_window: {thewindow} does not exist, skipping.")
        return

    main_window_w, main_window_h = raven_utils.get_widget_size(reference_window)  # Get the size of the main window, and hence also the viewport, in pixels.
    logger.debug(f"recenter_window: Main window size is {main_window_w}x{main_window_h}.")

    # Render offscreen so we get the final size. Only needed if the size can change.
    dpg.set_item_pos(thewindow,
                     (main_window_w,
                      main_window_h))
    dpg.show_item(thewindow)
    logger.debug(f"recenter_window: After show command: Window is visible? {dpg.is_item_visible(thewindow)}.")
    dpg.split_frame()  # wait for render
    logger.debug(f"recenter_window: After wait for render: Window is visible? {dpg.is_item_visible(thewindow)}.")

    w, h = raven_utils.get_widget_size(thewindow)
    logger.debug(f"recenter_window: window {thewindow} (tag '{dpg.get_item_alias(thewindow)}', type {dpg.get_item_type(thewindow)}) size is {w}x{h}.")

    # Center the window in the viewport
    dpg.set_item_pos(thewindow,
                     (max(0, (main_window_w - w) // 2),
                      max(0, (main_window_h - h) // 2)))

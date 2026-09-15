"""Run by `test_focus_semantics.py` in a process of its own; not a test module itself.

Builds a window whose text field has never held the caret, in a fresh DPG context before its first frame —
the state an app is in at launch — then asks for the caret while a `Tooltip` rewrites its text, and prints
whether the field ended up active.

The fresh context is the point. A window built inside the shared test context, which has been rendering for a
while, does not lose the request, so the in-process fixture cannot tell the two ways of asking apart; and a
context cannot be recreated in one process once real widgets have rendered.

Usage: python focus_request_subprocess.py plain|give_caret
"""

import sys

import dearpygui.dearpygui as dpg

from raven.common.gui import animation as gui_animation
from raven.common.gui import tooltip


def render(n: int) -> None:
    for _ in range(n):
        gui_animation.animator.render_frame()
        dpg.render_dearpygui_frame()


def main(how: str) -> None:
    dpg.create_context()
    dpg.create_viewport(title="raven gui tests: focus request", width=420, height=200)
    with dpg.window(tag="main") as main_window:
        dpg.add_input_text(tag="field", multiline=True, width=300, height=60)
        dpg.add_button(tag="park", label="park")
        target = dpg.add_button(label="has a tooltip")
    tip = tooltip.Tooltip(target, "a caption")
    dpg.set_primary_window(main_window, True)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    try:
        render(20)
        dpg.focus_item("park")
        render(6)
        if how == "plain":
            dpg.focus_item("field")
        else:
            gui_animation.give_caret("field")
        tip.text = "a different caption"
        render(10)
        print(f"RESULT active={dpg.is_item_active('field')}")
    finally:
        tip.destroy()
        gui_animation.animator.clear()
        dpg.destroy_context()


if __name__ == "__main__":
    main(sys.argv[1])

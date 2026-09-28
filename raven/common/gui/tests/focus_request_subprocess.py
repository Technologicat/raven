"""Run by `test_focus_semantics.py` in a process of its own; not a test module itself.

Builds a window whose text field has never held the caret, in a fresh DPG context before its first frame —
the state an app is in at launch — then rewrites an unhovered `Tooltip`'s text while doing one of three
things, and prints the outcome:

- `plain`, `give_caret`: asks for the caret in the field that way, and prints whether the field ended up
  active.
- `modal`: opens a modal window, and prints whether it is still open.

The fresh context is the point. A window built inside the shared test context, which has been rendering for a
while, does not lose the request, so the in-process fixture cannot tell the ways of asking apart; and a
context cannot be recreated in one process once real widgets have rendered.

With `hovered`, the tooltip is on screen when its text changes — the pointer resting on its target, faked,
since nothing here moves the real one.

With `bystander`, a hidden window built like a tooltip's is shown offscreen instead of the tooltip being
rewritten — which is exactly what a `Tooltip` must not do with a text change nobody is looking at. That is the
control: the failure a test built on this script must be able to see.

Usage: python focus_request_subprocess.py plain|give_caret|modal [hovered] [bystander]
"""

import sys

import dearpygui.dearpygui as dpg

from raven.common.gui import animation as gui_animation
from raven.common.gui import tooltip
from raven.common.gui import utils as guiutils


def render(n: int) -> None:
    for _ in range(n):
        gui_animation.animator.render_frame()
        dpg.render_dearpygui_frame()


def main(how: str, hovered: bool, bystander: bool) -> None:
    dpg.create_context()
    dpg.create_viewport(title="raven gui tests: focus request", width=420, height=200)
    with dpg.window(tag="main") as main_window:
        dpg.add_input_text(tag="field", multiline=True, width=300, height=60)
        dpg.add_button(tag="park", label="park")
        target = dpg.add_button(label="has a tooltip")
    with dpg.window(tag="modal", modal=True, show=False, width=200, height=100):
        dpg.add_input_text(tag="modal_field")
    with dpg.window(tag="bystander", show=False, no_title_bar=True, no_focus_on_appearing=True, autosize=True, min_size=[1, 1]):
        dpg.add_text("shown offscreen")
    tip = tooltip.Tooltip(target, "a caption")
    dpg.set_primary_window(main_window, True)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    try:
        render(20)
        dpg.focus_item("park")
        render(6)
        if hovered:  # the pointer rests on the tooltip's target, so the tooltip is on screen when the text changes
            real_is_item_hovered = dpg.is_item_hovered
            dpg.is_item_hovered = lambda item: item == target or real_is_item_hovered(item)
            tip._on_hover(None, None, None)
            render(6)
            print(f"HOVERED tooltip_shown={dpg.is_item_shown(tip.window)}")
        if how == "plain":
            dpg.focus_item("field")
        elif how == "give_caret":
            gui_animation.give_caret("field")
        else:
            dpg.show_item("modal")
            dpg.focus_item("modal_field")
        if bystander:
            guiutils.park_offscreen("bystander")  # tag
            dpg.show_item("bystander")  # tag
        else:
            tip.text = "a different caption"
        render(10)
        if how == "modal":
            print(f"RESULT shown={dpg.is_item_shown('modal')}")
        else:
            print(f"RESULT active={dpg.is_item_active('field')}")
    finally:
        tip.destroy()
        gui_animation.animator.clear()
        dpg.destroy_context()


if __name__ == "__main__":
    main(sys.argv[1], hovered="hovered" in sys.argv[2:], bystander="bystander" in sys.argv[2:])

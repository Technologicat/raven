"""Which item does DearPyGui auto-focus in a child window, and does it activate it?

Written 2026-08-18 while chasing a blue flash in `FileDialog`: on every Tab the *path* field became
active for 25-100 ms before focus settled on its intended target, and active means the caret, which means
select-all, which paints the text blue.

The tempting explanation was ImGui's documented fallback — nav landing on the container's first navigable
item — with image buttons skipped in nav order, so that the first *reachable* item is a text field. This
probe rules that out, which is why it is kept: it records what the answer is not.

  - Image buttons are not skipped. Auto-focus lands on whichever item is first, image button or plain.
  - Auto-focus does not *activate* what it focuses; the flash shows an active field.

So the mechanism behind that flash is still unknown. Anyone picking it up again can start by not
re-deriving these two.
"""

import dearpygui.dearpygui as dpg
import numpy as np

SETTLE_FRAMES = 10
WATCHED = ("image_button", "plain_button", "first_field", "second_field")


def render(n: int = SETTLE_FRAMES) -> None:
    for _ in range(n):
        dpg.render_dearpygui_frame()


def auto_focus_target(build_contents) -> tuple[list[str], list[str]]:
    """Build a child window with `build_contents`, let it settle, and report what took focus."""
    dpg.create_context()
    dpg.create_viewport(width=520, height=300)
    dpg.setup_dearpygui()
    with dpg.texture_registry():
        dpg.add_static_texture(16, 16, np.ones(16 * 16 * 4, dtype=np.float32), tag="tex")
    with dpg.window(tag="w", width=480, height=260):
        with dpg.child_window(tag="panel", width=460, height=200):
            build_contents()
    dpg.set_primary_window("w", True)
    dpg.show_viewport()
    render()

    focused = [t for t in WATCHED if dpg.does_item_exist(t) and dpg.is_item_focused(t)]
    active = [t for t in WATCHED if dpg.does_item_exist(t) and dpg.is_item_active(t)]
    dpg.destroy_context()
    return focused, active


def image_button_first() -> None:
    dpg.add_image_button("tex", tag="image_button")
    dpg.add_input_text(tag="first_field", width=250)
    dpg.add_input_text(tag="second_field", width=250)


def plain_button_first() -> None:
    dpg.add_button(tag="plain_button", label="R")
    dpg.add_input_text(tag="first_field", width=250)
    dpg.add_input_text(tag="second_field", width=250)


def fields_only() -> None:
    dpg.add_input_text(tag="first_field", width=250)
    dpg.add_input_text(tag="second_field", width=250)


def main() -> None:
    for label, build in (("image button, then two fields", image_button_first),
                         ("plain button, then two fields", plain_button_first),
                         ("two fields, no button", fields_only)):
        focused, active = auto_focus_target(build)
        print(f"  {label:34s} focused={focused or ['-']}  active={active or ['-']}")


if __name__ == "__main__":
    main()

"""Is a pending `focus_item` on a text field lost when a `Tooltip` measures itself in the same frames?

Seen in Raven-librarian on 2026-09-15: at a fresh launch, where the composer holds ImGui's navigation focus
without being active, New chat's `focus_item` on the composer never took, because `flash_button` changed a
tooltip's text and the tooltip's offscreen measurement showed its window for two frames. From a state where
focus was parked on a button, the same sequence worked.

This builds that fresh-launch state in a context of its own — a window whose first navigable item is a text
field, rendered untouched for a few frames — and runs four arms: `focus_item` alone, and `focus_item` with a
tooltip text change, each from the fresh state and from focus parked on a button.

Maps a small window, so it takes keyboard focus while it runs. Each arm runs in a process of its own, a DPG
context not being reliably recreatable once real widgets have rendered.
"""

import os
import subprocess
import sys

import dearpygui.dearpygui as dpg

from raven.common.gui import animation as gui_animation
from raven.common.gui import tooltip as gui_tooltip


def render(n: int) -> None:
    for _ in range(n):
        gui_animation.animator.render_frame()
        dpg.render_dearpygui_frame()


def run_arm(label: str, park_first: bool, change_tooltip: bool, reassert: bool = False) -> None:
    dpg.create_context()
    dpg.create_viewport(title="focus request probe", width=420, height=200)
    with dpg.window(tag="main") as main:
        if os.environ.get("PROBE_CHILD_PANEL"):  # a child window ahead of the field, as the test fixture has
            dpg.add_child_window(width=300, height=40)
        dpg.add_input_text(tag="field", multiline=True, width=300, height=60)
        dpg.add_button(label="park", tag="park")
        other = dpg.add_button(label="has a tooltip", tag="other")
    tip = gui_tooltip.Tooltip(other, "a caption")
    dpg.set_primary_window(main, True)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    render(20)
    fresh = f"active {dpg.is_item_active('field')} focused {dpg.is_item_focused('field')}"
    if park_first:
        dpg.focus_item("other" if os.environ.get("PROBE_PARK_ON_TOOLTIP_TARGET") else "park")
        render(6)
    dpg.focus_item("field")
    if change_tooltip:
        tip.text = os.environ.get("PROBE_NEW_TEXT", "a different caption")
    if reassert:  # ask again every frame until the field has the caret
        frames = 0
        while not dpg.is_item_active("field") and frames < 10:
            dpg.focus_item("field")
            render(1)
            frames += 1
        print(f"PROBE   re-asserted over {frames} frames")
    render(10)
    print(f"PROBE {label}: startup state [{fresh}] -> active {dpg.is_item_active('field')} focused {dpg.is_item_focused('field')}")
    tip.destroy()
    gui_animation.animator.clear()
    dpg.destroy_context()


ARMS = {"fresh-alone": ("fresh, focus_item alone        ", False, False),
        "fresh-tooltip": ("fresh, focus_item + tooltip    ", False, True),
        "parked-alone": ("parked, focus_item alone       ", True, False),
        "parked-tooltip": ("parked, focus_item + tooltip   ", True, True),
        "fresh-tooltip-reassert": ("fresh, + tooltip, re-asserted  ", False, True, True),
        "parked-tooltip-reassert": ("parked, + tooltip, re-asserted ", True, True, True)}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        label, park_first, change_tooltip, *rest = ARMS[sys.argv[1]]
        run_arm(label, park_first=park_first, change_tooltip=change_tooltip, reassert=bool(rest and rest[0]))
    else:
        for arm in ARMS:
            subprocess.run([sys.executable, __file__, arm], check=False)

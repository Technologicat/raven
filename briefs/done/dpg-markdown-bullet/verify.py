"""Verify the fix: bullets inside an initially-hidden container land at
distinct positions once the container is shown.

Proxy setup: we use `dpg.window(show=False)` rather than `dpg.tooltip`
because DPG's tooltip visibility is hover-tied — `configure_item(show=True)`
does not override it, so a headless script can't force a tooltip visible.
A hidden window reproduces the exact same root cause (DPG doesn't lay out
children of an invisible container; `get_item_pos` returns (0, 0)), but is
controllable without a live mouse.
"""
import time

import dearpygui.dearpygui as dpg

from raven.vendor import DearPyGui_Markdown as dpg_markdown
from raven.common.gui import fontsetup

bullet_positions = []

# Capture bullet drawlist_group positions by wrapping dpg.add_group
orig_add_group = dpg.add_group

def spy_add_group(*args, **kwargs):
    item = orig_add_group(*args, **kwargs)
    # Capture explicit-pos groups created inside attributes_group (bullet overlays)
    if "pos" in kwargs and kwargs["pos"]:
        bullet_positions.append(("add_group", item, tuple(kwargs["pos"])))
    return item

dpg.add_group = spy_add_group

FONT = "raven/fonts/OpenSans-Regular.ttf"
MD = """Note that in a real lens:

- Axial CA is typical at long focal lengths.
- Axial CA increases at high F-stops.
- Transverse CA is typical at short focal lengths.

After the list.
"""


def main():
    dpg.create_context()
    with dpg.font_registry() as registry:
        dpg_markdown.set_font_registry(registry)
        dpg_markdown.set_add_font_function(fontsetup.markdown_add_font_callback)
        dpg_markdown.set_font(font_size=20, default=FONT, bold=FONT, italic=FONT, italic_bold=FONT)

    dpg.create_viewport(title="bullet verify", width=900, height=700)
    dpg.setup_dearpygui()

    with dpg.window(label="main", width=800, height=600, pos=(20, 20)):
        dpg.add_text("Inline (should be fine):")
        dpg_markdown.add_text(MD)

    # Initially-hidden window: same root cause as an initially-hidden tooltip,
    # but controllable without a mouse.
    with dpg.window(label="hidden", width=600, height=400, pos=(40, 40),
                    show=False, tag="hidden_win"):  # tag
        dpg_markdown.add_text(MD)

    dpg.show_viewport()

    print("== Frames with hidden window ==")
    for _ in range(40):
        dpg.render_dearpygui_frame()
        time.sleep(0.02)

    print(f"Bullet positions so far (inline only expected): {bullet_positions}")
    before_count = len(bullet_positions)

    print("== Showing previously-hidden window ==")
    dpg.configure_item("hidden_win", show=True)  # tag

    for _ in range(40):
        dpg.render_dearpygui_frame()
        time.sleep(0.02)

    new_positions = bullet_positions[before_count:]
    print(f"Bullet positions added after window shown: {new_positions}")

    assert len(new_positions) >= 3, f"expected >= 3 new bullets, got {len(new_positions)}"
    ys = sorted({round(p[2][1], 1) for p in new_positions})
    assert len(ys) >= 3, f"expected >= 3 distinct Y positions, got {ys}"
    print(f"OK: {len(ys)} distinct y-positions for hidden-window bullets: {ys}")

    dpg.destroy_context()


if __name__ == "__main__":
    main()

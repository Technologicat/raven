"""Does `min_size` clamp a window that was given an explicit size, or only an autosize one?

The distinction is the one that let a bug ship. `dpg-notes.md` recorded the floor as an autosize
phenomenon, so `width=w, height=48` read like an escape from it. It is not: an explicit size is clamped
too, and `ScrollEndFlasher`'s 48 px bands had always been 100 px.

    python investigations/dpg-overlays/min_size_probe.py

Needs a real X session — the window has to be mapped for DPG to lay anything out — but synthesizes no
input, so it only takes focus for the second or so it is on screen.

Measured output:

    window                        asked for    actual rect
    explicit_no_minsize              400x48        400x100
    explicit_with_minsize            400x48         400x48
    autosize_no_minsize            autosize        100x100

`min_size` defaults to about [100, 100], the theme style `mvStyleVar_WindowMinSize` does not override it,
and neither does asking for a size directly. Pair this with `mouse_capture_probe.py`, which shows a window
eats the mouse across its whole rect: the surplus is not blank space, it is a dead zone.
"""

import dearpygui.dearpygui as dpg

dpg.create_context()
dpg.create_viewport(title="raven min_size probe", width=520, height=420)
dpg.setup_dearpygui()

with dpg.window(tag="main"):
    dpg.add_text("background")
dpg.set_primary_window("main", True)

# The shape ScrollEndFlasher uses: explicit size, no min_size.
dpg.add_window(tag="explicit_no_minsize", show=True, no_title_bar=True, no_background=True,
               no_collapse=True, no_focus_on_appearing=True, no_resize=True, no_move=True,
               no_scrollbar=True, no_scroll_with_mouse=True,
               pos=[10, 40], width=400, height=48)

# The same, with the floor lifted.
dpg.add_window(tag="explicit_with_minsize", show=True, no_title_bar=True, no_background=True,
               no_collapse=True, no_focus_on_appearing=True, no_resize=True, no_move=True,
               no_scrollbar=True, no_scroll_with_mouse=True,
               pos=[10, 160], width=400, height=48, min_size=[1, 1])

# An autosize window holding one small widget, for reference — the case the notes already covered.
with dpg.window(tag="autosize_no_minsize", show=True, no_title_bar=True, no_background=True,
                autosize=True, no_collapse=True, no_focus_on_appearing=True, no_resize=True,
                no_move=True, no_scrollbar=True, no_scroll_with_mouse=True, pos=[10, 280]):
    dpg.add_button(label="tiny")

dpg.show_viewport()
for _ in range(30):  # let the layout settle; the rect is not meaningful before anything has rendered
    dpg.render_dearpygui_frame()

print(f"{'window':26} {'asked for':>12}   {'actual rect':>12}")
for tag, asked in (("explicit_no_minsize", "400x48"),
                   ("explicit_with_minsize", "400x48"),
                   ("autosize_no_minsize", "autosize")):
    w, h = dpg.get_item_rect_size(tag)
    print(f"{tag:26} {asked:>12}   {f'{w}x{h}':>12}")

dpg.destroy_context()

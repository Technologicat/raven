"""Does `focus_item` still reach a field after its container has been moved with `move_item`?

Answers the question Raven-visualizer ran into on 2026-10-29: with the info panel switched to the other side
at runtime (the columns reordered with `dpg.move_item`), Ctrl+F stopped returning the caret to the search
field after a click on a toolbar button. Booted with the panel already on that side, the same keys worked.

Two child windows side by side in a horizontal group, a text field in A and a button in B. Focus is put on
the button (or nowhere), then the field is asked for, under four histories: never moved; A moved after B
before the first frame; A moved after B once frames have rendered; and, built B first, B moved after A once
frames have rendered, so that the field's window, the newer of the two, comes first. Only the last is
refused, whatever had the focus. Run it and read the table.

The second table (2026-10-30) asks what gets the refused case out of trouble, and whether building the
field's window first avoids it: a reset after the move (hiding and showing either window), the same order
reached by moving the field's window `before=` its sibling, and, built field-first, the sibling moved to
the front and the two swapped there and back. The third replays the app's startup, the first frames drawn
visibly, transparently, or with the field's window hidden.
"""

import dearpygui.dearpygui as dpg

SETTLE_FRAMES = 12


def render(n: int = SETTLE_FRAMES) -> None:
    for _ in range(n):
        dpg.render_dearpygui_frame()


def trial(move: str, focus_source: str) -> tuple[bool, bool]:
    """`move`: "never", "before first frame", "after frames" or "field to front, after frames".

    Returns (focused, active) for the field.
    """
    dpg.create_context()
    dpg.create_viewport(width=520, height=200)
    dpg.setup_dearpygui()
    with dpg.window(tag="w", width=480, height=160):
        with dpg.group(horizontal=True, tag="row"):
            if move == "field to front, after frames":  # built button first, as the app builds the plot last
                with dpg.child_window(tag="B", width=220, height=100):
                    dpg.add_button(tag="button", label="button")
            with dpg.child_window(tag="A", width=220, height=100):
                dpg.add_input_text(tag="field", width=180)
            if move != "field to front, after frames":
                with dpg.child_window(tag="B", width=220, height=100):
                    dpg.add_button(tag="button", label="button")
    dpg.set_primary_window("w", True)
    if move == "before first frame":
        dpg.move_item("A", parent="row")  # to the end: B, A
    dpg.show_viewport()
    render()
    if move == "after frames":
        dpg.move_item("A", parent="row")
        render()
    if move == "field to front, after frames":
        dpg.move_item("B", parent="row")  # to the end: A, B — the field's column now comes first
        render()

    if focus_source == "button":
        dpg.focus_item("button")
        render()
    dpg.focus_item("field")
    render()
    answer = (dpg.is_item_focused("field"), dpg.is_item_active("field"))
    dpg.destroy_context()
    return answer


def remedy(variant: str) -> tuple[bool, bool]:
    """A history ending with the field's window in front, then the field asked for after focus was on the button.

    Built "B first" (the field's window newer, as the app builds its plot column last) or "A first" (older).
    """
    dpg.create_context()
    dpg.create_viewport(width=520, height=200)
    dpg.setup_dearpygui()
    b_first = variant.startswith("B first")
    with dpg.window(tag="w", width=480, height=160):
        with dpg.group(horizontal=True, tag="row"):
            if b_first:
                with dpg.child_window(tag="B", width=220, height=100):
                    dpg.add_button(tag="button", label="button")
            with dpg.child_window(tag="A", width=220, height=100):
                dpg.add_input_text(tag="field", width=180)
            if not b_first:
                with dpg.child_window(tag="B", width=220, height=100):
                    dpg.add_button(tag="button", label="button")
    dpg.set_primary_window("w", True)
    if variant == "A first, behind B at the first frame: A to the front":
        dpg.move_item("A", parent="row")                         # B, A when the first frame is drawn
    dpg.show_viewport()
    render()
    if variant == "A first, behind B at the first frame: A to the front":
        dpg.move_item("A", parent="row", before="B")
    elif variant == "B first: B to the end":                      # the refused case, as a reference
        dpg.move_item("B", parent="row")
    elif variant == "B first: B to the end, A hidden and shown":
        dpg.move_item("B", parent="row")
        render()
        dpg.hide_item("A")
        render(2)
        dpg.show_item("A")
    elif variant == "B first: B to the end, B hidden and shown":
        dpg.move_item("B", parent="row")
        render()
        dpg.hide_item("B")
        render(2)
        dpg.show_item("B")
    elif variant == "B first: A moved before B":
        dpg.move_item("A", parent="row", before="B")
    elif variant == "A first: B moved before A":                # the field's window older, its sibling to the front
        dpg.move_item("B", parent="row", before="A")
    elif variant == "A first: B before A, then back":           # a swap and a swap back
        dpg.move_item("B", parent="row", before="A")
        render()
        dpg.move_item("B", parent="row")
    elif variant == "A first: never moved":
        pass
    render()
    dpg.focus_item("button")
    render()
    dpg.focus_item("field")
    render()
    answer = (dpg.is_item_focused("field"), dpg.is_item_active("field"))
    dpg.destroy_context()
    return answer


def startup(first_frames: str) -> tuple[bool, bool]:
    """The app's startup, then its swap: built panel (B) first and field (A) second, arranged field-first for
    the first frames, the panel moved in front, then swapped back so that the field's window is in front.

    `first_frames`: "visible", "transparent" (an alpha-0 theme on the window), or "field hidden" — the
    negative control, a window that is not drawn at all and so is not dated.
    """
    dpg.create_context()
    dpg.create_viewport(width=520, height=200)
    dpg.setup_dearpygui()
    with dpg.theme() as invisible:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_style(dpg.mvStyleVar_Alpha, 0.0, category=dpg.mvThemeCat_Core)
    with dpg.window(tag="w", width=480, height=160):
        with dpg.group(horizontal=True, tag="row"):
            with dpg.child_window(tag="B", width=220, height=100):
                dpg.add_button(tag="button", label="button")
            with dpg.child_window(tag="A", width=220, height=100):
                dpg.add_input_text(tag="field", width=180)
    dpg.set_primary_window("w", True)
    dpg.move_item("B", parent="row")                             # A, B: the field's window drawn first
    if first_frames == "transparent":
        dpg.bind_item_theme("w", invisible)
    elif first_frames == "field hidden":
        dpg.hide_item("A")
    dpg.show_viewport()
    render(3)
    dpg.move_item("B", parent="row", before="A")                 # the panel into place on the left
    dpg.bind_item_theme("w", 0)
    dpg.show_item("A")
    render()
    dpg.move_item("B", parent="row")                             # the runtime swap: the field's window in front
    render()
    dpg.focus_item("button")
    render()
    dpg.focus_item("field")
    render()
    answer = (dpg.is_item_focused("field"), dpg.is_item_active("field"))
    dpg.destroy_context()
    return answer


def main() -> None:
    print("  A moved ...                      focus was on ...   field focused  active")
    for move in ("never", "before first frame", "after frames", "field to front, after frames"):
        for source in ("nothing", "button"):
            focused, active = trial(move, source)
            refused = "   <-- refused" if not focused else ""
            print(f"  {move:<32} {source:<18} {str(focused):<14} {active}{refused}")
    print()
    print("  remedy                                                 field focused  active")
    for variant in ("B first: B to the end", "B first: B to the end, A hidden and shown",
                    "B first: B to the end, B hidden and shown", "B first: A moved before B",
                    "A first: never moved", "A first: B moved before A", "A first: B before A, then back",
                    "A first, behind B at the first frame: A to the front"):
        focused, active = remedy(variant)
        refused = "   <-- refused" if not focused else ""
        print(f"  {variant:<54} {str(focused):<14} {active}{refused}")
    print()
    print("  the app's startup, first frames ...   field focused  active")
    for first_frames in ("visible", "transparent", "field hidden"):
        focused, active = startup(first_frames)
        refused = "   <-- refused" if not focused else ""
        print(f"  {first_frames:<37} {str(focused):<14} {active}{refused}")


if __name__ == "__main__":
    main()

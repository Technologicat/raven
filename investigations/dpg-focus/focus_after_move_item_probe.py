"""Does `focus_item` still reach a field after its container has been moved with `move_item`?

Answers the question Raven-visualizer ran into on 2026-10-29: with the info panel switched to the other side
at runtime (the columns reordered with `dpg.move_item`), Ctrl+F stopped returning the caret to the search
field after a click on a toolbar button. Booted with the panel already on that side, the same keys worked.

Two child windows side by side in a horizontal group, a text field in A and a button in B. Focus is put on
the button (or nowhere), then the field is asked for, under four histories: never moved; A moved after B
before the first frame; A moved after B once frames have rendered; and, built B first, B moved after A once
frames have rendered, so that the field's window, the newer of the two, comes first. Only the last is
refused, whatever had the focus. Run it and read the table.
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


def main() -> None:
    print("  A moved ...                      focus was on ...   field focused  active")
    for move in ("never", "before first frame", "after frames", "field to front, after frames"):
        for source in ("nothing", "button"):
            focused, active = trial(move, source)
            refused = "   <-- refused" if not focused else ""
            print(f"  {move:<32} {source:<18} {str(focused):<14} {active}{refused}")


if __name__ == "__main__":
    main()

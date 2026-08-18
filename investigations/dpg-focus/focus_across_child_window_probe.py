"""When does `focus_item` refuse to move focus?

Answers the question `FileDialog` ran into on 2026-08-18: Ctrl+F and Tab-back stopped returning the caret
to the find field, while the same call had always worked before, and still worked in Librarian and
Visualizer. The variable is neither the widget kind nor the modality of the window, but *where focus
already is* relative to the child windows.

The obvious theory — that a child window is a boundary `focus_item` cannot cross — is wrong, and fits most
of the evidence, which is why this probe enumerates positions rather than testing the theory. A field in
one child window can be focused perfectly well from another. What cannot reach into a child window is an
item sitting directly in the enclosing window.

Run it and read the table: only window-level to child is refused.
"""

import dearpygui.dearpygui as dpg

SETTLE_FRAMES = 12

# One field per position: directly in the window, in child A, in child B.
POSITIONS = {"window": "w_field", "child A": "a_field", "child B": "b_field"}


def render(n: int = SETTLE_FRAMES) -> None:
    for _ in range(n):
        dpg.render_dearpygui_frame()


def move_focus(source: str, target: str) -> tuple[bool, bool]:
    """Focus the field at `source`, then ask for the one at `target`. Returns what became of the target."""
    dpg.create_context()
    dpg.create_viewport(width=520, height=380)
    dpg.setup_dearpygui()
    with dpg.window(tag="w", width=480, height=340):
        with dpg.child_window(tag="A", width=460, height=100):
            dpg.add_input_text(tag="a_field", width=300)
        with dpg.child_window(tag="B", width=460, height=100):
            dpg.add_input_text(tag="b_field", width=300)
        dpg.add_input_text(tag="w_field", width=300)
    dpg.set_primary_window("w", True)
    dpg.show_viewport()
    render()

    dpg.focus_item(POSITIONS[source])
    render()
    dpg.focus_item(POSITIONS[target])
    render()
    answer = (dpg.is_item_focused(POSITIONS[target]), dpg.is_item_active(POSITIONS[target]))
    dpg.destroy_context()
    return answer


def main() -> None:
    print("  focus is in ...   ask to focus ...   target focused  active")
    for source in POSITIONS:
        for target in POSITIONS:
            if source == target:
                continue
            focused, active = move_focus(source, target)
            refused = "   <-- refused" if not focused else ""
            print(f"  {source:<17} {target:<18} {focused!s:<15} {active!s}{refused}")


if __name__ == "__main__":
    main()

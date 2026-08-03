"""Does Enter deactivate a text field? Single-line and multiline answer differently.

This is the exception to "gate hotkeys on `is_item_active`". A *single-line* `InputText` deactivates itself
on Enter, because the key commits the edit — so a handler gated on `is_item_active` can never fire on Enter:
by the time it runs, the field is already inactive. A *multiline* field does not, because there Enter
inserts a newline.

The distinction is not academic. Switching `raven-visualizer`'s Enter gate from `is_item_focused` to
`is_item_active` — correct for its bare-key branch, and correct wholesale in `raven-librarian` — silently
killed its search: Ctrl+F, type, Enter, and nothing happened. Caught in live testing, not by the suite.

    python investigations/dpg-focus/enter_key_probe.py

Needs `xdotool` and a real X session; drives itself, and takes keyboard focus for about twelve seconds.

Measured output:

    label                             focused  active  value
    single: after focus_item             True    True  ''
    single: typed, before Enter          True    True  'cat'
    single: after Enter f210             True   False  'cat'
    single: after Enter f260             True   False  'cat'
    multi: typed, before Enter           True    True  'dog'
    multi: after Enter f470              True    True  'dog\n'
    multi: after Enter f520              True    True  'dog\n'

So an app whose text field is single-line gates its Enter handler on `is_item_focused` while still gating
its bare-key branch on `is_item_active` — two different questions about one widget, each chosen for the
state the key actually arrives in. Raven's two GUI apps differ from each other for exactly this reason: the
Visualizer's search field is single-line, Librarian's composer is multiline.

Incidentally confirms that `dpg.focus_item` on a text field *activates* it — the first row — which is what
makes Ctrl+F work in the Visualizer. Contrast a child window, which `focus_item` cannot focus at all; see
`README.md`.

Each field is entered with `focus_item` rather than a click, both because that follows the app's real Ctrl+F
path and because click coordinates on a reparented window are unreliable — the first version of this probe
guessed them and put both typings into the wrong field.
"""

import subprocess

import dearpygui.dearpygui as dpg

TITLE = "raven enter key probe"

dpg.create_context()
dpg.create_viewport(title=TITLE, width=420, height=260)
dpg.setup_dearpygui()

with dpg.window(tag="main"):
    dpg.add_input_text(tag="single", width=380)
    dpg.add_input_text(tag="multi", multiline=True, width=380, height=60)
dpg.set_primary_window("main", True)
dpg.show_viewport()

log = []


def snap(label: str, which: str) -> None:
    log.append((label,
                dpg.is_item_focused(which), dpg.is_item_active(which),
                repr(dpg.get_value(which))[:16]))


def x(*args: str) -> None:
    subprocess.run(["xdotool", *args], check=False, capture_output=True)


wid = None
for frame in range(700):
    dpg.render_dearpygui_frame()

    if frame == 30:
        out = subprocess.run(["xdotool", "search", "--name", TITLE],
                             capture_output=True, text=True).stdout.split()
        wid = out[-1] if out else None
        x("windowactivate", "--sync", wid)

    # Single-line: the shape of raven-visualizer's search field.
    elif frame == 60:
        dpg.focus_item("single")
    elif frame == 90:
        snap("single: after focus_item", "single")
    elif frame == 100:
        x("type", "--window", wid, "--delay", "30", "cat")
    elif frame == 160:
        snap("single: typed, before Enter", "single")
    elif frame == 180:
        x("key", "Return")
    elif frame in (210, 260):
        snap(f"single: after Enter f{frame}", "single")

    # Multiline: the shape of raven-librarian's composer.
    elif frame == 320:
        dpg.focus_item("multi")
    elif frame == 360:
        x("type", "--window", wid, "--delay", "30", "dog")
    elif frame == 420:
        snap("multi: typed, before Enter", "multi")
    elif frame == 440:
        x("key", "Return")
    elif frame in (470, 520):
        snap(f"multi: after Enter f{frame}", "multi")

dpg.destroy_context()

print(f"{'label':<32} {'focused':>8} {'active':>7}  value")
for label, focused, active, value in log:
    print(f"{label:<32} {str(focused):>8} {str(active):>7}  {value}")

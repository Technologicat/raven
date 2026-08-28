"""When a text field commits on its chord, does a global key handler see that keypress too?

The question a second send path turns on. `raven-librarian` sends by the composer's *own* commit callback —
ImGui owns Enter and Ctrl+Enter while a multiline `InputText` is active, so the field decides what "commit"
means and `on_enter` reports it. That leaves the chord dead whenever the composer is *not* focused, which is
a real state: after a send, focus rests on the send button, and a user who wants to send an empty message
(Librarian's "take another turn" gesture) has nothing to press.

Wiring the same chord into the global hotkey handler fixes that — if the two paths can coexist. They cannot
if the global handler also fires on the commit, because then a focused send runs the callback twice, and the
second one is a second turn.

`app.py`'s own comment says the global handler *does* see the chord, having "failed silently in both
`is_item_active` and `is_item_focused` forms" — the gate was false by the time it ran, which says the
handler ran. That is a recollection of a failure rather than a measurement, so it is worth one.

    python investigations/dpg-focus/commit_chord_dispatch_probe.py

Needs `xdotool` and a real X session; drives itself, and takes keyboard focus for about fifteen seconds.

Two fields, because the send chord is configurable (`config.send_message_key`) and the two settings put the
commit on different keys — and `ctrl_enter_for_new_line` names what *Ctrl+Enter* does rather than what
sends, so the flag reads backwards from the setting:

    field  ctrl_enter_for_new_line  commits on
    A      False (ImGui default)    Ctrl+Enter   <- `send_message_key = "ctrl+enter"`, Raven's default
    B      True                     Enter        <- `send_message_key = "enter"`

Each is driven twice: once with the caret in it, once with focus parked on a button, which is the state the
second send path exists for.

The `source` column is what the answer is read from. `field` is the `InputText`'s own commit callback,
`global` a keyless `add_key_press_handler` — the shape `librarian_hotkeys_callback` is registered as. Both
report `dpg.get_frame_count()` at the moment they run, so a double dispatch shows as two rows, and whether
it can be told apart by frame number shows as whether those numbers differ.
"""

import subprocess

import dearpygui.dearpygui as dpg

TITLE = "raven commit chord dispatch probe"

dpg.create_context()
dpg.create_viewport(title=TITLE, width=460, height=320)
dpg.setup_dearpygui()

log = []  # (phase, source, frame, detail)
phase = "startup"


def note(source: str, detail: str) -> None:
    log.append((phase, source, dpg.get_frame_count(), detail))


with dpg.window(tag="main"):
    # A: Ctrl+Enter commits. Librarian's composer, at the default setting.
    dpg.add_input_text(tag="A", multiline=True, width=420, height=50,
                       on_enter=True, ctrl_enter_for_new_line=False,
                       callback=lambda: note("field", "A committed"))
    # B: Enter commits. The same composer under `send_message_key = "enter"`.
    dpg.add_input_text(tag="B", multiline=True, width=420, height=50,
                       on_enter=True, ctrl_enter_for_new_line=True,
                       callback=lambda: note("field", "B committed"))
    dpg.add_button(tag="park", label="park focus here")
dpg.set_primary_window("main", True)


def on_key(sender, app_data):
    """Every keypress the global handler sees, with what the fields say about themselves at that moment."""
    if app_data != dpg.mvKey_Return:
        return
    ctrl = dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)
    note("global", f"Return, ctrl={ctrl}, "
                   f"A(focused={dpg.is_item_focused('A')}, active={dpg.is_item_active('A')}), "  # tag
                   f"B(focused={dpg.is_item_focused('B')}, active={dpg.is_item_active('B')})")  # tag


with dpg.handler_registry():
    dpg.add_key_press_handler(callback=on_key)

dpg.show_viewport()


def x(*args: str) -> None:
    subprocess.run(["xdotool", *args], check=False, capture_output=True)


def chord(wid: str, ctrl: bool) -> None:
    """Press Return, holding Ctrl across frames if asked.

    Held rather than sent as a combined chord because `is_key_down` is sampled when the callback runs, not
    when the key went down — a modifier released too quickly is gone before anything reads it.
    """
    if ctrl:
        x("keydown", "--window", wid, "ctrl")
    x("key", "--window", wid, "Return")


def release_ctrl(wid: str) -> None:
    x("keyup", "--window", wid, "ctrl")


wid = None
for frame in range(1000):
    dpg.render_dearpygui_frame()

    if frame == 30:
        out = subprocess.run(["xdotool", "search", "--name", TITLE],
                             capture_output=True, text=True).stdout.split()
        wid = out[-1] if out else None
        x("windowactivate", "--sync", wid)

    # ---- A auto-focused and never touched: the startup state, and the one that decides the gate.
    #
    # Runs first, before anything is focused by hand, because ImGui gives nav focus to the first navigable
    # item of a newly focused window on its own. At the global handler's moment that reads identically to a
    # field that has just committed — focused, not active — so whether the two can be told apart at all
    # turns on whether an untouched field commits.
    elif frame == 45:
        phase = "A auto-focused, ctrl+Return"
        chord(wid, ctrl=True)
    elif frame == 95:
        release_ctrl(wid)

    # ---- A, caret in the field: the case where both paths could fire.
    elif frame == 140:
        phase = "A active, ctrl+Return"
        dpg.focus_item("A")
    elif frame == 170:
        x("type", "--window", wid, "--delay", "30", "abc")
    elif frame == 230:
        chord(wid, ctrl=True)
    elif frame == 280:
        release_ctrl(wid)

    # ---- A, focus parked on a button: the case the second send path exists for.
    elif frame == 340:
        phase = "A inactive, ctrl+Return"
        dpg.focus_item("park")
    elif frame == 400:
        chord(wid, ctrl=True)
    elif frame == 450:
        release_ctrl(wid)

    # ---- B, caret in the field: same question for the bare-Enter setting.
    elif frame == 510:
        phase = "B active, Return"
        dpg.focus_item("B")
    elif frame == 540:
        x("type", "--window", wid, "--delay", "30", "xyz")
    elif frame == 600:
        chord(wid, ctrl=False)

    # ---- B, focus parked: bare Enter with no field holding the caret.
    elif frame == 680:
        phase = "B inactive, Return"
        dpg.focus_item("park")
    elif frame == 740:
        chord(wid, ctrl=False)

    elif frame == 830:
        phase = "done"

dpg.destroy_context()

print(f"{'phase':<26} {'source':<7} {'frame':>6}  detail")
for phase_, source, frame_, detail in log:
    print(f"{phase_:<26} {source:<7} {frame_:>6}  {detail}")

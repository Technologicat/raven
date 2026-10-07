"""What does a multiline text field do with Shift held on its commit chord?

The question a second save action in Librarian's message editor turns on. The send key saves an edit, and
"save as a new branch" wants the same chord with Shift added: Ctrl+Shift+Enter under the default
`send_message_key = "ctrl+enter"`, Shift+Enter under `"enter"`. Three things ImGui could do with that, and
each needs a different handler:

- **commit** — the field deactivates, as on the plain chord, and the global handler sees the key with the
  field focused and not active;
- **insert a newline** — the field stays active, and the draft gains a line break wherever the caret was,
  which is not something a `strip` can undo once the caret is mid-text;
- **nothing** — the field stays active, and the text is unchanged.

`app.py` says that Shift+Enter "does nothing" in the composer, which is the third reading for one of the two
settings, and a recollection rather than a measurement.

    python investigations/dpg-focus/shift_commit_chord_probe.py

Needs `xdotool` and a real X session; drives itself, and takes keyboard focus for about fifteen seconds.

Two fields, configured as the editor's is under each setting (no `on_enter`, which the editor does not use):

    field  ctrl_enter_for_new_line  commits on
    A      False (ImGui default)    Ctrl+Enter   <- `send_message_key = "ctrl+enter"`
    B      True                     Enter        <- `send_message_key = "enter"`

Each is driven with the plain chord first, as the control: it must come out *committed*, or the probe cannot
tell a commit from anything else. Then with Shift added. Every run types `abcd` and moves the caret two
places left first, so an inserted newline lands mid-text and shows in the value.
"""

import subprocess

import dearpygui.dearpygui as dpg

TITLE = "raven shift commit chord probe"
TYPED = "abcd"

dpg.create_context()
dpg.create_viewport(title=TITLE, width=460, height=320)
dpg.setup_dearpygui()

log = []  # (phase, source, frame, detail)
phase = "startup"


def note(source: str, detail: str) -> None:
    log.append((phase, source, dpg.get_frame_count(), detail))


with dpg.window(tag="main"):
    dpg.add_input_text(tag="A", multiline=True, width=420, height=50, ctrl_enter_for_new_line=False)
    dpg.add_input_text(tag="B", multiline=True, width=420, height=50, ctrl_enter_for_new_line=True)
    dpg.add_button(tag="park", label="park focus here")
dpg.set_primary_window("main", True)


def down(*keys: str) -> bool:
    return any(dpg.is_key_down(k) for k in keys)


def on_key(sender, app_data):
    """Each Return the global handler sees, with the modifiers and the field's state at that moment."""
    if app_data != dpg.mvKey_Return:
        return
    field = "A" if phase.startswith("A") else "B"
    note("global", f"Return, ctrl={down(dpg.mvKey_LControl, dpg.mvKey_RControl)}, "
                   f"shift={down(dpg.mvKey_LShift, dpg.mvKey_RShift)}, "
                   f"{field}(focused={dpg.is_item_focused(field)}, active={dpg.is_item_active(field)})")


with dpg.handler_registry():
    dpg.add_key_press_handler(callback=on_key)

dpg.show_viewport()


def x(*args: str) -> None:
    subprocess.run(["xdotool", *args], check=False, capture_output=True)


#: (phase, field, modifiers held across the Return). The plain chord of each field is its control.
RUNS = [("A control, ctrl+Return", "A", ["ctrl"]),
        ("A ctrl+shift+Return", "A", ["ctrl", "shift"]),
        ("B control, Return", "B", []),
        ("B shift+Return", "B", ["shift"])]

#: Frames from a run's start at which each step happens. Generous, as the window renders unthrottled.
STEP_FOCUS, STEP_TYPE, STEP_LEFT, STEP_CHORD, STEP_RELEASE, STEP_READ, RUN_LENGTH = 0, 30, 80, 110, 150, 180, 200
FIRST_RUN = 60

wid = None
for frame in range(FIRST_RUN + RUN_LENGTH * len(RUNS) + 30):
    dpg.render_dearpygui_frame()

    if frame == 30:
        out = subprocess.run(["xdotool", "search", "--name", TITLE],
                             capture_output=True, text=True).stdout.split()
        wid = out[-1] if out else None
        x("windowactivate", "--sync", wid)
        continue
    if frame < FIRST_RUN:
        continue

    k, step = divmod(frame - FIRST_RUN, RUN_LENGTH)
    if k >= len(RUNS):
        phase = "done"
        continue
    phase, field, mods = RUNS[k]
    if step == STEP_FOCUS:
        dpg.focus_item("park")
        dpg.set_value(field, "")
    elif step == STEP_FOCUS + 10:
        dpg.focus_item(field)
    elif step == STEP_TYPE:
        x("type", "--window", wid, "--delay", "30", TYPED)
    elif step == STEP_LEFT:
        x("key", "--window", wid, "Left", "Left")
    elif step == STEP_CHORD:
        for mod in mods:
            x("keydown", "--window", wid, mod)
        x("key", "--window", wid, "Return")
    elif step == STEP_RELEASE:
        for mod in reversed(mods):
            x("keyup", "--window", wid, mod)
    elif step == STEP_READ:
        note("after", f"value={dpg.get_value(field)!r}, focused={dpg.is_item_focused(field)}, "
                      f"active={dpg.is_item_active(field)}")

dpg.destroy_context()

print(f"{'phase':<24} {'source':<7} {'frame':>6}  detail")
for phase_, source, frame_, detail in log:
    print(f"{phase_:<24} {source:<7} {frame_:>6}  {detail}")

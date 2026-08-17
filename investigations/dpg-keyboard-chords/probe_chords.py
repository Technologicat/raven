"""Probe: what reaches a DPG key handler while a single-line InputText holds the caret,
and whether a modal window stacks over another modal window.

Two questions, one run, because both need a mapped window and rendered frames.

Phase 1 (manual/xdotool driven): every key press is logged with its modifier state and
with whether the InputText is focused / active. Answers whether Ctrl+Enter, Alt+Up,
Ctrl+Up, Tab and Ctrl+Space survive ImGui's own handling of a text field, and confirms
the runtime codes for Page Up / Page Down.

Phase 2 (automatic, at T_STACK seconds): a second modal window is shown on top of the
first. A screenshot then says whether DPG stacks modals.
"""

import sys
import time

import dearpygui.dearpygui as dpg

T_STACK = 22.0  # seconds until phase 2 fires

NAMED = {}  # keycode -> name, filled below from the dpg.mvKey_* constants


def log(msg):
    print(f"{time.time():.3f} {msg}", flush=True)


dpg.create_context()
dpg.create_viewport(title="probe_fdialog_keys", width=900, height=520, x_pos=60, y_pos=60)
dpg.setup_dearpygui()

for name in dir(dpg):
    if name.startswith("mvKey_"):
        value = getattr(dpg, name)
        if isinstance(value, int):
            NAMED.setdefault(value, name)


def key_handler(sender, app_data):
    key = app_data
    ctrl = dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)
    alt = dpg.is_key_down(dpg.mvKey_LAlt) or dpg.is_key_down(dpg.mvKey_RAlt)
    shift = dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift)
    mods = "".join([("C" if ctrl else "-"), ("A" if alt else "-"), ("S" if shift else "-")])
    focused = dpg.is_item_focused("probe_input")  # tag
    active = dpg.is_item_active("probe_input")  # tag
    log(f"KEY {key:5d} {NAMED.get(key, '?'):24s} mods={mods} "
        f"focused={int(focused)} active={int(active)} "
        f"focused_item={dpg.get_focused_item()}")


with dpg.handler_registry():
    dpg.add_key_press_handler(callback=key_handler)

with dpg.window(tag="modal_a", label="Modal A (stand-in for FileDialog)",  # tag
                modal=True, no_title_bar=False, width=820, height=380, pos=(20, 20)):
    dpg.add_text("Click the field below, then send chords.")
    dpg.add_input_text(tag="probe_input", hint="single-line, like the find field",  # tag
                       width=600)
    dpg.add_text("")
    dpg.add_button(label="a button in modal A", width=240)

with dpg.window(tag="modal_b", label="Modal B (stand-in for the help card)",  # tag
                modal=True, show=False, width=460, height=200, pos=(240, 180)):
    dpg.add_text("If you can read this, DPG stacked a modal over a modal.")
    dpg.add_input_text(tag="probe_input_b", hint="can you type in here?", width=380)  # tag
    dpg.add_button(label="a button in modal B", width=240)

dpg.show_viewport()

t0 = time.monotonic()
stacked = False
last_beat = 0.0
while dpg.is_dearpygui_running():
    now = time.monotonic()
    if not stacked and now - t0 > T_STACK:
        stacked = True
        log("PHASE2 showing modal B over modal A")
        dpg.show_item("modal_b")  # tag
    if now - last_beat > 0.5:  # heartbeat, so the caret state is readable at any moment
        last_beat = now
        log(f"BEAT active={int(dpg.is_item_active('probe_input'))} "  # tag
            f"focused_item={dpg.get_focused_item()} "
            f"a_vis={int(dpg.is_item_visible('modal_a'))} "  # tag
            f"b_vis={int(dpg.is_item_visible('modal_b'))}")  # tag
    dpg.render_dearpygui_frame()

dpg.destroy_context()
log("probe exited")
sys.exit(0)

"""Probe: does `set_value` on an InputText stick while the field holds the caret?

ImGui keeps an internal edit buffer for an active InputText, so a programmatic write may be
ignored, may be visible but discarded on commit, or may take. Tab completion and the save-mode
arrow-fill both write the field while the user is typing in it, so which of those it is decides
whether either can be built at all.

F2 writes "SETVALUE" into the field. The heartbeat logs what `get_value` reports, so the
sequence type -> F2 -> type -> Enter is readable afterwards.
"""

import time

import dearpygui.dearpygui as dpg

dpg.create_context()
dpg.create_viewport(title="probe_setvalue", width=900, height=300, x_pos=60, y_pos=60)
dpg.setup_dearpygui()


def log(msg):
    print(f"{time.time():.3f} {msg}", flush=True)


def key_handler(sender, app_data):
    if app_data == dpg.mvKey_F2:
        before = dpg.get_value("probe_input")  # tag
        dpg.set_value("probe_input", "SETVALUE")  # tag
        after = dpg.get_value("probe_input")  # tag
        log(f"F2 set_value: before={before!r} after={after!r} "
            f"active={int(dpg.is_item_active('probe_input'))}")  # tag
    elif app_data == dpg.mvKey_F3:
        # The unfocus -> modify -> refocus dance, in case a plain set_value does not take.
        before = dpg.get_value("probe_input")  # tag
        dpg.focus_item("parking_button")  # tag
        dpg.set_value("probe_input", "DANCED")  # tag
        dpg.focus_item("probe_input")  # tag
        log(f"F3 dance: before={before!r} after={dpg.get_value('probe_input')!r} "  # tag
            f"active={int(dpg.is_item_active('probe_input'))}")
    elif app_data == dpg.mvKey_F4:
        # Same dance, but spanning frames: `focus_item` only takes effect on the next frame, so
        # the F3 version never actually unfocused anything. Legal here because a key handler is
        # dispatched off the render thread.
        before = dpg.get_value("probe_input")  # tag
        dpg.focus_item("parking_button")  # tag
        dpg.split_frame()
        mid_active = dpg.is_item_active("probe_input")  # tag
        dpg.set_value("probe_input", "DANCED2")  # tag
        dpg.split_frame()
        dpg.focus_item("probe_input")  # tag
        dpg.split_frame()
        log(f"F4 dance: before={before!r} mid_active={int(mid_active)} "
            f"after={dpg.get_value('probe_input')!r} "  # tag
            f"active={int(dpg.is_item_active('probe_input'))}")  # tag
    elif app_data == dpg.mvKey_F5:
        # How many frames does it take for `focus_item` elsewhere to release the caret — if it ever does?
        before = dpg.get_value("probe_input")  # tag
        dpg.focus_item("parking_button")  # tag
        states = []
        for _ in range(6):
            dpg.split_frame()
            states.append(int(dpg.is_item_active("probe_input")))  # tag
        dpg.set_value("probe_input", "DANCED3")  # tag
        dpg.split_frame()
        wrote = dpg.get_value("probe_input")  # tag
        dpg.focus_item("probe_input")  # tag
        dpg.split_frame()
        log(f"F5 dance: before={before!r} active_per_frame={states} wrote={wrote!r} "
            f"after={dpg.get_value('probe_input')!r} "  # tag
            f"active={int(dpg.is_item_active('probe_input'))}")  # tag
    elif app_data == dpg.mvKey_F6:
        # `configure_item(default_value=...)` is a different spelling of the same write, and `dpg-notes`
        # records it changing the live value where the name says it should not. That was measured on an
        # *inactive* field, so whether it survives ImGui's edit buffer is a separate question — and the one
        # that decides whether Tab completion is buildable at all.
        before = dpg.get_value("probe_input")  # tag
        dpg.configure_item("probe_input", default_value="CONFIGURED")  # tag
        dpg.split_frame()
        log(f"F6 configure_item: before={before!r} after={dpg.get_value('probe_input')!r} "  # tag
            f"active={int(dpg.is_item_active('probe_input'))}")  # tag
    elif app_data == dpg.mvKey_F7:
        # Can a `menu_item` hold focus? The places panel is built from free-floating ones, and DPG's focus
        # model has already produced two surprises here (a child window cannot be focused, a button can).
        before_focus = dpg.get_focused_item()
        dpg.focus_item("probe_menu_item")  # tag
        dpg.split_frame()
        # Asked separately and defensively: a `menu_item` turns out to have no `focused` state at all, so
        # the query raises rather than answering False — and where focus actually *went* is the half that
        # decides whether asking for it is merely useless or harmful.
        try:
            focused_flag = dpg.is_item_focused("probe_menu_item")  # tag
        except Exception as exc:  # noqa: BLE001 -- the point is to report whatever it does
            focused_flag = f"<{type(exc).__name__}: {exc}>"
        log(f"F7 focus menu_item: before={before_focus} after={dpg.get_focused_item()} "
            f"is_item_focused={focused_flag} "
            f"input_active={int(dpg.is_item_active('probe_input'))}")  # tag
    elif app_data == dpg.mvKey_Return:
        log(f"RETURN value={dpg.get_value('probe_input')!r}")  # tag


def on_edit(sender, app_data):
    log(f"CALLBACK app_data={app_data!r} get_value={dpg.get_value('probe_input')!r}")  # tag


with dpg.handler_registry():
    dpg.add_key_press_handler(callback=key_handler)

with dpg.window(tag="win", width=860, height=240, pos=(10, 10)):  # tag
    dpg.add_text("Click the field, type, press F2, type again, press Enter.")
    dpg.add_input_text(tag="probe_input", width=600, callback=on_edit)  # tag
    dpg.add_button(tag="parking_button", label="parking spot for the unfocus dance", width=300)  # tag
    # Free-floating, the way `fdialog`'s places panel builds them — not inside a menu bar.
    with dpg.group(horizontal=True):
        dpg.add_menu_item(tag="probe_menu_item", label="a menu item, as the places panel uses")  # tag

dpg.show_viewport()

last = 0.0
while dpg.is_dearpygui_running():
    now = time.monotonic()
    if now - last > 0.5:
        last = now
        log(f"BEAT value={dpg.get_value('probe_input')!r} "  # tag
            f"active={int(dpg.is_item_active('probe_input'))}")  # tag
    dpg.render_dearpygui_frame()

dpg.destroy_context()

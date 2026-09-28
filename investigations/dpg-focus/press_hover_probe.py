"""What a mouse press does to hover and active state: the pressed item, and the window it is in.

Builds a child window holding a drawlist, clicks twice on the drawlist with synthetic input, and prints
every change in the button, the child window's hover and active state, and the drawlist's. Self-driving:
run it and read the table. It maps a window and moves the pointer, so it takes the keyboard for about ten
seconds. Needs `xdotool`, `xwininfo` and `wmctrl` (X11).

Measured 2026-09-28 on DPG 2.3.1: the drawlist is active from the press frame to the release, the child
window is never active, and the child window's hover drops from the frame after the press until one frame
after the release — while the drawlist's own hover stays True throughout. See `dpg-notes.md`, "Is this
mouse event mine?".

Usage: python press_hover_probe.py
"""

import subprocess
import threading
import time

import dearpygui.dearpygui as dpg

TITLE = "raven probe: press and hover"


def click_twice() -> None:
    """Wait for the window, then press and release twice on the drawlist, held across several frames."""
    wid = ""
    for _ in range(50):
        listing = subprocess.run(["wmctrl", "-l"], capture_output=True, text=True).stdout
        wid = next((line.split()[0] for line in listing.splitlines() if TITLE in line), "")
        if wid:
            break
        time.sleep(0.2)
    if not wid:
        print("no window found; nothing was clicked")
        return
    time.sleep(1.0)
    info = subprocess.run(["xwininfo", "-id", wid], capture_output=True, text=True).stdout
    x0 = int(next(line.split()[-1] for line in info.splitlines() if "Absolute upper-left X" in line))
    y0 = int(next(line.split()[-1] for line in info.splitlines() if "Absolute upper-left Y" in line))
    for x, y in ((80, 80), (250, 150)):  # on the drawn rectangle, then on the drawlist's empty part
        subprocess.run(["xdotool", "mousemove", "--sync", str(x0 + x), str(y0 + y)])
        time.sleep(0.5)  # let the app read the new pointer position before the press
        subprocess.run(["xdotool", "mousedown", "1"])
        time.sleep(0.3)  # held across frames, as a finger would
        subprocess.run(["xdotool", "mouseup", "1"])
        time.sleep(0.6)


def main() -> None:
    dpg.create_context()
    dpg.create_viewport(title=TITLE, width=400, height=300)
    with dpg.window(tag="main") as main_window:
        with dpg.child_window(tag="panel", width=300, height=200, pos=(20, 20)):
            dpg.add_drawlist(width=280, height=180, tag="drawlist")
            dpg.draw_rectangle((10, 10), (100, 100), parent="drawlist", fill=(80, 80, 200))
    dpg.set_primary_window(main_window, True)  # pins the window to the viewport origin, so the click offsets hold
    dpg.setup_dearpygui()
    dpg.show_viewport()

    clicker = threading.Thread(target=click_twice, daemon=True)
    clicker.start()
    frame, previous = 0, None
    t_end = time.monotonic() + 12
    while dpg.is_dearpygui_running() and time.monotonic() < t_end and (clicker.is_alive() or frame < 10):
        frame += 1
        state = (dpg.is_mouse_button_down(dpg.mvMouseButton_Left),
                 dpg.is_item_hovered("panel"), dpg.is_item_active("panel"),
                 dpg.is_item_hovered("drawlist"), dpg.is_item_active("drawlist"))
        if state != previous:
            print(f"frame {frame:4d}: button_down={state[0]!s:5}  panel hovered={state[1]!s:5} active={state[2]!s:5}  "
                  f"drawlist hovered={state[3]!s:5} active={state[4]!s:5}")
            previous = state
        dpg.render_dearpygui_frame()
    dpg.destroy_context()


if __name__ == "__main__":
    main()

"""Which ways of ending a DPG render loop run the exit callback (`dpg.set_exit_callback`)?

Usage: python probe_exit_callback.py MODE

  MODE = stop     the render loop calls `dpg.stop_dearpygui()` itself after a few frames, on the main thread
         sigterm  a SIGTERM arrives, with a handler calling `dpg.stop_dearpygui()` — what `quitsignal` installs
         wmclose  wait for the window manager to close the window; the caller runs `wmctrl -i -c`

Maps a small window titled `exit-callback-probe`, which takes keyboard focus. Prints one line per event, each
with the time since start and the calling thread, and ends with a `RESULT:` line saying whether the callback
ran and whether that was before or after the loop exited.
"""

import os
import signal
import sys
import threading
import time

import dearpygui.dearpygui as dpg

mode = sys.argv[1] if len(sys.argv) > 1 else "stop"
if mode not in ("stop", "sigterm", "wmclose"):
    sys.exit(f"unknown mode {mode!r}")

t0 = time.monotonic()
def say(msg: str) -> None:
    print(f"{time.monotonic() - t0:7.3f} [{threading.current_thread().name}] {msg}", flush=True)

state = {"callback_ran": False, "loop_exited": False, "callback_after_loop": None}

def on_exit(*args) -> None:
    state["callback_ran"] = True
    state["callback_after_loop"] = state["loop_exited"]
    say(f"exit callback ran (loop exited already: {state['loop_exited']})")

dpg.create_context()
dpg.create_viewport(title="exit-callback-probe", width=320, height=120)
dpg.setup_dearpygui()
with dpg.window(tag="w"):
    dpg.add_text(f"exit callback probe, mode {mode}")
dpg.set_primary_window("w", True)
dpg.set_exit_callback(on_exit)
dpg.show_viewport()

if mode == "sigterm":
    def handle(signum, frame):
        say("SIGTERM received; calling stop_dearpygui")
        dpg.stop_dearpygui()
    signal.signal(signal.SIGTERM, handle)
    threading.Timer(1.0, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()

say(f"render loop starting, pid {os.getpid()}")
frame = 0
try:
    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()
        frame += 1
        if mode == "stop" and frame == 30:
            say("calling stop_dearpygui from the render loop")
            dpg.stop_dearpygui()
        if mode == "wmclose" and time.monotonic() - t0 > 15:
            say("no close arrived in 15 s; stopping (inconclusive)")
            dpg.stop_dearpygui()
finally:
    state["loop_exited"] = True
    say(f"render loop exited after {frame} frames")

# The exit callback is dispatched on DPG's callback thread, so give it time to arrive before deciding it did not.
time.sleep(float(os.environ.get("PROBE_DELAY", "1.0")))
dpg.destroy_context()
say(f"RESULT: mode={mode} callback_ran={state['callback_ran']} "
    f"callback_after_loop_exit={state['callback_after_loop']}")

"""Probe: at what point in DPG startup does `glfwGetCurrentContext()` hand back DPG's window?

This is the question that decides *where* `filedrop.install` may be called, and it is not answerable by
reading anything: GLFW's "current context" is per-thread state that DPG sets as a side effect of its own
startup, and nothing documents which call does it.

Answered on X11 / `dearpygui` 2.3.1 (see this directory's `README.md`): `show_viewport()` makes it current,
not the first rendered frame — and it is NULL on any other thread at every point. Re-run this on a new
platform or after a DPG upgrade; the answers are printed rather than asserted, because the point is to find
out what a given build does. The invariant the shipped code depends on is asserted separately, in
`raven/common/gui/tests/test_filedrop.py`.

Note it maps a window, briefly, and therefore takes keyboard focus for about a second. It needs no human
input otherwise.

Usage: `python investigations/dpg-dnd/context_timing_probe.py`
"""

import ctypes
import pathlib
import threading

import dearpygui.dearpygui as dpg

glfw = ctypes.CDLL(str(pathlib.Path(dpg.__file__).parent / "_dearpygui.so"))
glfw.glfwGetCurrentContext.restype = ctypes.c_void_p


def report(label):
    """Print this thread's current GLFW context at this moment in the startup sequence."""
    window = glfw.glfwGetCurrentContext()
    print(f"{label:<34} -> {hex(window) if window else 'NULL'}", flush=True)


def report_from_a_background_thread(label):
    """Same question, asked off the main thread — the reason `install` refuses to run there."""
    answer = {}
    thread = threading.Thread(target=lambda: answer.update(window=glfw.glfwGetCurrentContext()))
    thread.start()
    thread.join()
    window = answer["window"]
    print(f"{label:<34} -> {hex(window) if window else 'NULL'}", flush=True)


def main():
    report("before create_context")
    dpg.create_context()
    report("after create_context")
    dpg.create_viewport(title="context timing probe", width=320, height=200)
    report("after create_viewport")
    dpg.setup_dearpygui()
    report("after setup_dearpygui")

    # Everything above is NULL, so the handle is not reachable during app construction — only from here on.
    dpg.show_viewport()
    report("after show_viewport")

    # And check whether a frame is additionally required, which is the natural assumption given that most
    # other DPG state only settles once the loop is turning.
    dpg.render_dearpygui_frame()
    report("after 1st rendered frame")

    report_from_a_background_thread("from a background thread")

    dpg.destroy_context()


if __name__ == "__main__":
    main()

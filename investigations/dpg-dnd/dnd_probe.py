"""Probe: can we get OS-level file drop into a DPG app by calling the GLFW that DPG already links?

Yes, on X11 — see this directory's `README.md` for what came back. Four questions, and the answers are
printed rather than asserted, because the point is to find out what a given platform does:

  1. Does `glfwGetCurrentContext()` hand back DPG's window? (GLFW's current context is per-thread, and
     DPG's render loop is what makes it current, so this needs observing rather than reasoning about.)
  2. Does our `glfwSetDropCallback` survive, or does DPG overwrite it later?
  3. Does a drop actually fire on this platform?
  4. Which thread does the callback arrive on? If it is the render thread, the callback must not do
     anything that waits for a frame (`split_frame`, and therefore the modal messagebox).

**Kept because it is re-runnable, not as a record.** Question 3 is still open on Wayland, and this is what
answers it there: run it, drag a file, read the terminal. It needs a human to do the dragging, which is why
it is a probe and not a test — the feature will get tests of its own.

Usage: `python investigations/dpg-dnd/dnd_probe.py`, then drag a file onto the window that appears. Note it
maps a window, so it takes keyboard focus.
"""

import ctypes
import pathlib
import threading

import dearpygui.dearpygui as dpg

# GLFW is statically linked into the DPG extension, and its symbols are exported (checked with
# `nm -D --defined-only`), so we can bind to the already-loaded library rather than to a system GLFW —
# which matters, because it has to be the *same* GLFW instance that owns DPG's window.
_dpg_so = pathlib.Path(dpg.__file__).parent / "_dearpygui.so"
glfw = ctypes.CDLL(str(_dpg_so))

glfw.glfwGetCurrentContext.restype = ctypes.c_void_p
glfw.glfwGetX11Window.restype = ctypes.c_ulong
glfw.glfwGetX11Window.argtypes = [ctypes.c_void_p]

# void (*GLFWdropfun)(GLFWwindow*, int path_count, const char* paths[])
DROPFUN = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p))
glfw.glfwSetDropCallback.restype = DROPFUN
glfw.glfwSetDropCallback.argtypes = [ctypes.c_void_p, DROPFUN]

render_thread_id = None
observations = []


def on_drop(window, count, paths):
    """The thing under test. Deliberately does nothing that could wait for a frame."""
    files = [paths[i].decode("utf-8", errors="replace") for i in range(count)]
    this_thread = threading.get_ident()
    observations.append(files)
    print("\n=== DROP RECEIVED ===")
    print(f"  paths ({count}): {files}")
    print(f"  callback thread : {threading.current_thread().name} (id {this_thread})")
    print(f"  render thread   : id {render_thread_id}")
    print(f"  ON RENDER THREAD: {this_thread == render_thread_id}"
          "   <- if True, this callback must never call split_frame() or show a modal messagebox")
    print(f"  window arg      : {hex(window) if window else window}")
    print("=====================\n")


# Keep a reference: ctypes callbacks are garbage-collected like any other object, and GLFW holds only the
# raw pointer. Losing this is a segfault at drop time, not an exception.
drop_callback = DROPFUN(on_drop)


def main():
    global render_thread_id

    dpg.create_context()
    dpg.create_viewport(title="drag a file onto me", width=640, height=260)
    dpg.setup_dearpygui()

    with dpg.window(tag="main"):  # tag
        dpg.add_text("Drag a file from the file manager onto this window.")
        dpg.add_text("Watch the terminal. Close the window when done.")
        dpg.add_text("", tag="status")  # tag

    dpg.set_primary_window("main", True)  # tag
    dpg.show_viewport()

    # Render one frame first: the context has to be current on this thread before asking for it, and it is
    # the render call that makes it so.
    dpg.render_dearpygui_frame()
    render_thread_id = threading.get_ident()

    window = glfw.glfwGetCurrentContext()
    print(f"glfwGetCurrentContext() -> {hex(window) if window else window}")
    if not window:
        print("NULL: DPG's window is not this thread's current context. The whole approach needs another "
              "way to reach the handle.")
        dpg.destroy_context()
        return
    try:
        print(f"glfwGetX11Window()      -> {glfw.glfwGetX11Window(window)}  (0 would mean not an X11 window)")
    except Exception as exc:  # noqa: BLE001 -- probe; a Wayland session legitimately has no X11 window
        print(f"glfwGetX11Window()      -> raised {type(exc).__name__}: {exc}")

    previous = glfw.glfwSetDropCallback(window, drop_callback)
    print(f"glfwSetDropCallback()   -> previous callback was {'set' if previous else 'NULL'}"
          f" ({'DPG wires one after all' if previous else 'DPG never wired one, as expected'})")
    print("\nReady. Drag a file onto the window.\n")

    frames = 0
    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()
        frames += 1
        # Question 2: does DPG replace our callback at some later point (a viewport reconfigure, say)?
        # Sampling costs nothing and a silent replacement would otherwise read as "drop does not work".
        if frames % 600 == 0:
            still_ours = glfw.glfwSetDropCallback(window, drop_callback)
            if not still_ours:
                print(f"[frame {frames}] our callback had been cleared — DPG or GLFW replaced it")

    dpg.destroy_context()
    print(f"\nExited after {frames} frames. Drops received: {len(observations)}")
    for files in observations:
        print(f"  {files}")


if __name__ == "__main__":
    main()

# OS-level file drag-and-drop into a DPG app

**Question.** DPG apps cannot receive files dragged in from the OS file manager, so the in-app `FileDialog`
is the only way in — which is why the picker has to be good, and why "attach a file" costs more gestures
than it should. The only existing extension for this is Windows-only. Do we have to write a shim per
platform?

**Answer: no. The platform work is already in the binary, one layer below DPG.** DPG statically links
GLFW; GLFW has had cross-platform file drop since 3.1; and the symbols are exported from
`_dearpygui.so`, so the callback can be installed from Python with `ctypes`. Measured 2026-08-07 on X11,
`dearpygui` 2.3.1.

## Scripts

| Script | What it answers |
|---|---|
| `dnd_probe.py` | Whether a file dropped on a DPG viewport reaches Python, which window handle to use, whether DPG competes for the callback, and which thread it arrives on. Needs a human to do the dragging; run it, drag a file, read the terminal. |

**It is kept because it is re-runnable, not as a record of the run below.** Wayland is still unanswered and
this is what answers it there.

## What came back

```
glfwGetCurrentContext() -> 0x2f1c11a0
glfwGetX11Window()      -> 67108871
glfwSetDropCallback()   -> previous callback was NULL

=== DROP RECEIVED ===
  paths (1): ['/home/jje/Downloads/papers/08_Plasticity_02_Stress_Analysis.pdf']
  callback thread : MainThread (id 135675734098048)
  render thread   : id 135675734098048
  ON RENDER THREAD: True
```

- **The window handle is reachable.** `glfwGetCurrentContext()`, called on the render thread after the
  first `render_dearpygui_frame()`, returns DPG's window — the same pointer the callback later receives.
  This needed observing: GLFW's current context is per-thread, and nothing documents that DPG's render
  loop leaves it current on the caller's thread.
- **Nothing competes for the callback.** The previous one came back NULL, so DPG never installs one, and
  ours survived ~7000 frames of resampling.
- **The drop delivers the absolute path**, correctly decoded.

## The constraint this imposes

**The callback runs on the render thread.** GLFW dispatches from `glfwPollEvents()`, which DPG calls inside
`render_dearpygui_frame()`, so the callback executes synchronously within frame processing — the callback
thread id came back identical to the render loop's.

That is *not* a restriction on touching DPG state, which DPG permits from any thread. It is the
`split_frame` restriction, reached by a route that does not look like the render loop: the handler must not
call anything that waits for a frame, which rules out showing a modal messagebox from it. So the obvious
implementation of "you dropped a file type this app cannot use" — an error dialog written straight into the
handler — deadlocks. Capture the paths, hand off to a background task or a frame callback.

Recorded in `dpg-notes.md` under "GLFW callbacks are the exception: they run *on* the render thread",
because it is a general fact about GLFW callbacks rather than about drag-and-drop.

## What is still open

- **Wayland.** GLFW implements it, but that is inference from GLFW's feature set, not something observed
  here. **Decided 2026-08-07 (Juha) not to gate the feature on it**: this machine has no Wayland session to
  test against, X11 / macOS / Windows are already three platforms, and if it turns out not to work someone
  can file an issue. Re-run `dnd_probe.py` under Wayland to close it.
- **Multi-file drops.** The signature is `(window, count, paths)` and only `count == 1` has been seen.

## Where the feature itself is tracked

`TODO_DEFERRED.md`, "OS drag-and-drop of files into DPG apps (cross-platform)".

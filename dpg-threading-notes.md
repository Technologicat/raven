# DearPyGui Threading Model

Reference notes on DPG's internal threading, derived from the C++ source
(`hoffstadt/DearPyGui`, confirmed against DPG 2.0).

## Thread architecture

DPG uses **three kinds of threads**:

1. **Main thread** — the thread that calls `render_dearpygui_frame()`.
   Runs the ImGui render pass, detects input events, signals frame completion.

2. **Callback thread** — a single dedicated thread, launched by
   `setup_dearpygui()` via `std::async`. Executes all Python user callbacks
   (event handlers, frame callbacks). There is exactly one globally.

3. **User background threads** — any threads the application creates.
   DPG is largely thread-safe for item creation/deletion, `set_value`,
   and texture operations from these threads.

## How callbacks are dispatched

Detection and execution happen on **different threads**:

1. **Main thread** (inside `Render()`): iterates handler registries,
   e.g. `mvKeyPressHandler::draw()` checks `ImGui::IsKeyPressed(_key)`.
   On match, calls `mvSubmitCallback()` which pushes a lambda onto a
   thread-safe queue (`GContext->callbackRegistry->calls`).

2. **Callback thread** (`mvRunCallbacks()`): blocks on the queue
   (releasing the GIL while waiting), pops each lambda, acquires the GIL,
   and executes the Python callback.

This means:
- Event handlers **do not block rendering** (they run on a different thread).
- Event handlers **block each other** (single callback thread, serial execution).
- Heavy work in a callback delays all subsequent callbacks.

## `split_frame()` mechanism

`split_frame()` waits for `frameEndedEvent`, which `Render()` signals at the
end of each frame. This is why:

- **Safe** from the callback thread and user background threads — they're
  waiting for the main thread to complete a frame, which it can do independently.
- **Deadlocks** from the main thread — it's the thread that needs to signal
  the event, so it can't also wait for it.

## The two internal queues

- **`calls`** (thread-safe queue): Python user callbacks. Consumed by the
  callback thread via `mvRunCallbacks()`.
- **`tasks`** / **`earlyTasks`**: Internal DPG operations. Consumed by the
  main thread via `mvRunTasks()` during `Render()`.

## `manualCallbacks` exception

If `configure_app(manual_callback_management=True)` is set, `mvAddCallback`
pushes to `GContext->callbackRegistry->jobs` (a plain vector) instead of the
queue. The user must poll and execute callbacks manually. This does not apply
to normal DPG usage.

## Three-way deadlock pattern

When a callback holds a lock that the main loop also needs:

1. **Callback thread**: holds lock L, calls `TaskManager.clear(wait=True)`,
   blocking until a background task finishes.
2. **Background task thread**: stuck in `split_frame()`, waiting for
   `Render()` to signal `frameEndedEvent`.
3. **Main thread**: in the render loop body (before `render_dearpygui_frame()`),
   tries to acquire lock L — blocked.

Circular wait: callback -> task -> main -> callback.

**Prevention**: never call blocking waits on `split_frame`-using tasks while
holding locks that the main loop needs. Defer heavy work (image loading, task
cancellation) to the main loop body via a pending flag.

## Texture upload ordering

`set_value` on a dynamic texture and `add_dynamic_texture` are both deferred —
they update DPG's internal state but the actual OpenGL texture upload happens
during `render_dearpygui_frame()`. Empirically, DPG does **not** guarantee that
pending texture uploads complete before draw items referencing those textures
are rendered within the same `render_dearpygui_frame()` call.

**Consequence**: a `draw_image` referencing a texture whose data was just
changed via `set_value` (pool reuse) or just created via `add_dynamic_texture`
may render stale or uninitialized data for one frame.

**Workaround**: background threads must call `split_frame()` **twice** after
`_acquire_texture` — once to trigger the upload, once to ensure it's complete
before the texture is inserted into the live mip set and rendered.

This was discovered in raven-cherrypick's image viewer: preloaded images
flashed stale data from same-sized cached images (pool reuse via `set_value`),
and freshly created textures showed uninitialized data. Single `split_frame`
reduced but didn't eliminate the flashes; double `split_frame` fixed them
completely.

Note: code running on the **main thread** (inside the render loop body) cannot
use `split_frame` at all (deadlock — see above). Such code must delegate
texture creation to a background thread via `split_frame`, using the old-mips
bridge for display continuity during the one-frame upload delay.

## Source references

- `mvRunCallbacks()`: `src/mvCallbackRegistry.cpp`
- `mvSubmitCallback()`: `src/mvCallbackRegistry.h`
- `Render()` / `mvRenderFrame()`: `src/mvContext.cpp`
- Handler draw methods: `src/mvGlobalHandlers.cpp`
- Thread launch: `setup_dearpygui` in `src/dearpygui_commands.h:2527`

## Investigation history

- 2026-03-28: Investigated by grepping DPG C++ source on GitHub, prompted by
  a three-way deadlock in raven-cherrypick's image loader. Confirmed empirical
  observation that event handlers don't block rendering but do block each other.
- 2026-03-28: Discovered texture upload ordering issue during flash/corruption
  debugging. DPG doesn't guarantee upload-before-render within a single
  render_dearpygui_frame(). Double split_frame() is the reliable workaround.

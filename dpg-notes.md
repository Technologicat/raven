# DearPyGui Notes

Reference notes on DPG gotchas, internal mechanisms, and workarounds.
Derived from experience with Raven apps and confirmed against DPG 2.0 / ImGui source.

---

# Threading

Derived from the C++ source (`hoffstadt/DearPyGui`).

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

**Possible exception: `raw_texture`**. The avatar renderer uses `set_value` on
a `raw_texture` with a single `split_frame` and has never exhibited upload
ordering glitches despite heavy use. Hypothesis: raw textures are zero-copy —
DPG reads directly from the user-provided buffer during rendering, so there's
no deferred upload step to race against. If confirmed, switching from
`dynamic_texture` to `raw_texture` could eliminate the need for double
`split_frame` in cherrypick's mip pipeline. Needs investigation.

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

---

# Window sizing

## `min_size` vs `mvStyleVar_WindowMinSize`

`dpg.add_window()` has an explicit `min_size` parameter (default ~`[100, 100]`).
The theme style `mvStyleVar_WindowMinSize` does **not** override it — the
window parameter takes precedence. This means an autosize window won't shrink
below its `min_size` even when the content is smaller.

**Symptom**: autosize window appears to have phantom blank space below the
content. Looks like padding or an extra text line, but is actually the window
being clamped to its minimum height.

**Fix**: set `min_size=[1, 1]` explicitly on the window:

```python
dpg.add_window(autosize=True, no_title_bar=True, min_size=[1, 1], ...)
```

## Asymmetric vertical padding for tooltip-style windows

`WindowPadding` applies symmetrically to top and bottom. However, text items
have built-in ascender space above the first line (from the font metrics),
adding natural top padding. Setting `WindowPadding` y=0 gives a good top
appearance, but the bottom then has zero padding.

**Workaround**: use `WindowPadding` y=0 and add a trailing `dpg.add_spacer(height=N)`
to the content group for bottom padding. Typically N=2 balances well against
the font's natural ascender space.

## Window z-order

DPG renders windows in creation order. The primary window (set via
`set_primary_window`) is always at the back. Windows created later render on
top. There is no runtime z-order control — `focus_item` brings a window to
front but also steals keyboard focus.

**Implication for tooltips**: create the tooltip window during app
initialization (before the render loop), not lazily during hover. Windows
created mid-render-loop may end up behind earlier windows.

## Investigation history

- 2026-04-03: Discovered `min_size` default causing phantom padding in
  xdot-viewer tooltip. The theme style `WindowMinSize(1, 1)` had no effect;
  only the window parameter `min_size=[1, 1]` fixed it.
- 2026-04-03: Tooltip z-order issue — lazy window creation during render loop
  placed the tooltip behind the primary window. Fixed by creating the window
  during `__init__` (before the render loop starts).

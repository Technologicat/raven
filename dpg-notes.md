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

---

# Font atlas limits

DPG (via ImGui/stb_truetype) rasterizes every glyph in a font's character range
into a single texture atlas. The atlas has a finite size, and exceeding it causes
**silent glyph loss** — no error, no warning, just missing characters and wrong
`get_item_rect_size` measurements.

## `setup_font_ranges` and extended Unicode

`raven.common.gui.fontsetup.setup_font_ranges()` adds `dpg.add_font_range(0x100, 0x2FFF)`
— approximately 11,500 codepoints. This is fine at normal font sizes (20px), but at
large sizes (600px+) the atlas overflows. Each glyph at 600px is roughly 350×600 pixels;
11,500 of them need ~2.5 billion pixels of atlas space.

**Workaround**: for apps that only need digits/ASCII at large sizes, skip
`setup_font_ranges()` — load fonts directly with `dpg.font()`, which includes
the default Latin-1 range (~224 codepoints).

## Maximum font size

Even with only the default Latin-1 range (~224 codepoints), the atlas overflows
at font sizes above ~1200px. Empirically tested limits (2026-04-04, RTX 4090 /
RTX 3070 Ti):

| Font size | Latin-1 (224 chars) | Status |
|-----------|---------------------|--------|
| 600px     | ~20M pixels         | Works  |
| 1000px    | ~56M pixels         | Works  |
| 1200px    | ~80M pixels         | Works  |
| 1400px    | ~134M pixels        | Fails (missing glyphs) |

The conference timer caps at 1000px (`config.MAX_COUNTDOWN_FONT_SIZE`).

## `add_font_chars` does not reduce the range

`dpg.add_font_chars([...])` **adds** characters on top of the default Latin-1
range — it cannot remove characters. The default range is always loaded.
There is no way to load fewer than ~224 glyphs per font in DPG.

## Failure mode

When the atlas overflows:
- Some glyphs silently fail to rasterize.
- `get_item_rect_size` returns the size of whatever *did* render (clipped/wrapped).
- `bind_item_font` appears to succeed but the text renders with wrong/missing glyphs.

## Atlas rebuild flash

When a new font is added to the registry, DPG rebuilds the atlas texture. During
the rebuild, text briefly renders with the bound default font. This is orthogonal
to overflow — it happens on any valid font switch. At normal sizes the rebuild is
fast enough to be invisible; at large sizes (~1000px) it takes a couple of frames,
producing a visible flash.

**Workaround**: hide the text (e.g. position offscreen) before switching fonts,
reveal it after the new font has rendered.

## `guiutils.bootup` and atlas space

`guiutils.bootup()` loads multiple fonts (Regular, Bold, Italic, BoldItalic,
FontAwesome) with extended Unicode ranges at the standard 20px size. This is
correct for scientific apps but wastes atlas space when large countdown-style
fonts are also needed.

`bootup` is composed of four lower-level functions that can be called individually:
- `setup_default_font(font_size, font_basename)` — font registry + default font
- `setup_icon_fonts(font_registry, font_size)` — FontAwesome into existing registry
- `setup_markdown(font_registry, font_size, font_basename)` — `dpg_markdown` configuration
- `setup_themes()` — global rounded theme, disabled-control themes

Apps with non-standard font needs (e.g. the conference timer, which skips the
default font to keep the atlas lean) can call just the functions they need.

## `dpg_markdown` during app init

**Do not** call `dpg_markdown.add_text` more than once before the first frame
renders — this segfaults DPG (at least 1.11), likely a race condition in font
loading.

The render also appears asynchronous: if you populate other content into the same
container while `dpg_markdown` is loading its fonts, the rendering engine can lose
its place — some content is omitted, and the rest injected mid-Markdown-render.
This may also interact with DPG's global container stack.

**Workaround**: trigger Markdown font loading once at startup with a single dummy
element (`dpg_markdown.add_text("hello, *hello*, **hello**, ***hello***")`) that
exercises all four font families. Place it in a throwaway group. Do not add any
other Markdown elements until after the first frame. See `raven.visualizer.app`
(the `markdown_font_loader_trigger_dummy` group) for an example.

If your app creates Markdown content only on demand (e.g. a help card opened by
F1), this isn't an issue — by the time the user presses F1, the render loop has
been running for many frames.

## `bind_item_font` is queued, not immediate

`bind_item_font` from a frame callback takes effect **after the callback returns**
— it's queued as an internal DPG task. Calling `split_frame()` within the same
callback does **not** force the font change; the next render still uses the old font.

**Workaround**: use two separate frame callbacks (e.g. frames 10 and 12) — the
first loads and binds the font, the second measures the text with the new font
applied.

## `get_item_rect_size` and text overflow

When text overflows the primary window's content width, DPG wraps it. The
`get_item_rect_size` for the text widget then returns the **wrapped** dimensions
(width of the longest line, total height of all lines), not the full unwrapped
text extent.

**Workaround**: ensure the viewport is wide enough that the text doesn't wrap
before calling `get_item_rect_size`. For large fonts where this isn't practical,
measure a reference text at a smaller font size and use linear scaling (font
metrics scale ~linearly with size, within ~1%).

## `no_scrollbar=True`

Without `no_scrollbar=True`, DPG reserves ~14px (`mvStyleVar_ScrollbarSize`)
on the right side of the window for a potential scrollbar, even when no scrollbar
is shown. This causes asymmetric margins. Adding `no_scrollbar=True` to the
window eliminates the reservation.

## Investigation history

- 2026-04-04: Discovered during conference timer `--size` implementation.
  Silent atlas overflow at 1711px caused text to render at default font size
  with no error. Extended Unicode ranges (11k codepoints) overflow at ~600px.
  `add_font_chars` confirmed to add, not replace.
- 2026-04-05: Confirmed `bind_item_font` queuing behavior — `split_frame()`
  in a frame callback cannot force font changes within the same callback.
  Two-callback pattern (frames 10/12) is the reliable workaround.

---

# Raven DPG app structure

Reference patterns for building DearPyGui apps in Raven (Librarian as primary reference).

## Layout and GUI

- **Layout**: App-specific. Both Librarian and Visualizer use two-column layouts, but this isn't a general requirement. All in a single `main_window`.
- **Resize**: `resize_gui()` callback recalculates sizes. Debounced via background task for expensive updates.
- **Themes**: Named themes for button variants, pulsating indicators. Created at module level.
- **Fonts**: Default + icon fonts (FontAwesome), loaded at startup.
- **Animations**: `PulsatingColor` (cyclic) and `ButtonFlash` (one-shot) via `raven.common.gui.animation` global `animator` singleton.
- **Hotkeys**: Registered via `dpg.add_key_*_handler` in a handler registry.
- **Help card (F1)**: Every GUI app should have a help card (built with `raven.common.gui.helpcard`). Apps that skip `bootup` can pass a `gui_font` parameter to `HelpWindow` for the correct text size. Currently present in Librarian, Visualizer, Cherrypick, XDot Viewer, Conference Timer, and Avatar Settings Editor. The Avatar Pose Editor is still missing its help card.
- **Fullscreen (F11)**: Toggle via `dpg.toggle_viewport_fullscreen()` + `resize_gui()`. Standard pattern: `_toggle_fullscreen` calls both, `resize_gui` waits for size to settle via `wait_for_resize`, then calls `_resize_gui` to relayout.

## Background work and thread safety

- **Background work**: All async ops (LLM, avatar, RAG) run in background threads via `raven.common.bgtask`. `TaskManager` represents a set of related tasks sharing a `ThreadPoolExecutor`; the whole set can be cancelled via `.clear()`. Several task managers can share one executor. Debouncing via `ManagedTask` (OOP) or `make_managed_task` (functional) — use whichever is clearer.
- **Thread safety**: All components must be thread-safe. When every component has proper locking, thread-safety bugs are eliminated and there's no need to orchestrate main-thread-only operations. The price is lock contention; the advantage is erring on the side of safety and correctness. Any approach is valid as long as the end result is thread-safe: `threading.Lock` or `RLock` (choose based on whether re-entry is needed), lock-free atomic access (with a comment stating it's intentional), or other mechanisms. Prefer lock-free where possible — it's simpler and faster.
- **DPG threading** (unintuitive — unlike most GUI toolkits): DPG allows most operations from background threads, including creating/deleting items, setting values, and creating textures. `dpg.split_frame()` is safe **only** from background threads — it waits for the main thread's render loop to complete one frame. Calling it from the main thread **deadlocks** (the render loop can't proceed). Use `split_frame()` after creating textures in a background thread to ensure DPG has processed them before the render loop tries to use them (eliminates flicker from half-uploaded textures).

## DPG item management

- **DPG parent management**: Never depend on the state of the DPG container stack. Don't use `dpg.last_container()` or rely on implicit parenting. Always pass `parent=` explicitly (using tags or saved IDs). This is a thread safety concern: component `__init__` methods create handler registries and other items that pollute the container stack, and some parts of Raven create GUI controls from background threads. Explicit parents are the only safe approach. **Exception**: when initially building the app's main GUI in `main()`, using the `with` context managers (`with dpg.window(...):`, `with dpg.group(...):`) is fine — the main loop hasn't started yet, so no background tasks are running, and the stack is predictable.
- **DPG group size attributes**: `width`/`height` on `dpg.add_group()` are unreliable as of DPG 2.0 — the data may not actually constrain layout, and reading the values back may not reflect reality. Don't depend on them. For grid/tile layouts, let groups auto-size to their content and use DPG's `item_spacing` (default 8px horizontal, 4px vertical) for inter-element gaps.
- **DPG error handling**: DPG raises either `SystemError` (older versions) or `Exception` (newer) for "item not found" errors, with no proper exception subclass. The `nonexistent_ok()` context manager in `raven.common.gui.utils` suppresses these via string matching on the exception chain (EAFP pattern, avoids TOCTTOU). Has `.errored` attribute to check whether the block errored out.

## Textures

- **DPG texture buffer sizes**: When a pipeline produces textures asynchronously and the expected size changes (e.g. tile size switch), stale pipeline output can arrive with wrong dimensions. `dpg.add_dynamic_texture(w, h, data)` with undersized data causes a buffer overread → heap corruption → segfault or "double free" later. Guard with a size check before creating/updating textures. This bug is insidious because the crash often manifests far from the overflow (during an unrelated texture delete or render call).
- **DPG texture operations — defensive patterns**: Delete textures from DPG callbacks (inside `render_dearpygui_frame`) where the OpenGL context is active. Avoid synchronous CUDA work during callbacks; defer it to outside `render_dearpygui_frame` via a pending flag, or to a background thread. Use `dynamic_texture` for anything that may be deleted at runtime; `static_texture` is for truly permanent assets.

## Startup sequence

1. DPG init (context, fonts, themes, viewport)
2. Connect to raven-server (`raven.client.api`)
3. Connect to LLM backend, if needed by this app (`llmclient.setup()`)
4. Load persistent state (app-specific `appstate` implementation)
5. Load domain-specific backends (e.g. RAG: `hybridir.setup()`)
6. Build GUI layout
7. Create controller(s)
8. Initial view render
9. Start DPG event loop

## Idle framerate throttling

DPG's render loop runs at full GPU frame rate (typically 60 fps with vsync, or uncapped without). For apps with a mostly static GUI — where the user spends most time looking at results rather than interacting — this wastes CPU and GPU cycles, heats the machine, and drains laptop batteries.

The pattern: detect whether anything actually needs updating, and `time.sleep()` in the render loop when idle. This drops the effective frame rate to ~12 fps when nothing is happening, then instantly returns to full speed on user input or animation.

### Components

**1. Configuration** (`config.py`):

```python
IDLE_SLEEP_S = 0.08    # ~12 fps when idle (1 / 0.08 ≈ 12.5)
INPUT_ACTIVE_S = 0.5   # stay at full fps for this long after last user input
```

**2. Input timestamp tracking** — a module-level `_last_input_ns` updated by all input handlers:

```python
_last_input_ns: int = 0

def _on_any_input(*_args):
    global _last_input_ns
    _last_input_ns = time.monotonic_ns()

with dpg.handler_registry():
    dpg.add_mouse_move_handler(callback=_on_any_input)
    dpg.add_mouse_click_handler(callback=_on_any_input)
    dpg.add_mouse_wheel_handler(callback=_on_any_input)
    # Key handler also updates _last_input_ns (at the top of the handler body).
```

**3. Activity detector** — `_is_busy()` returns `True` when any of these hold:

- Recent user input (within `INPUT_ACTIVE_S`)
- GUI animations running (`gui_animation.animator.active_count > 0`)
- Background pipeline producing results (app-specific: thumbnail loading, mip loading, etc.)
- Visual effects in progress (resize flash, scroll countdown, etc.)

Minimal version (xdot-viewer):

```python
def _is_busy() -> bool:
    if (time.monotonic_ns() - _last_input_ns) < config.INPUT_ACTIVE_S * 1e9:
        return True
    if gui_animation.animator.active_count > 0:
        return True
    widget = _app_state["widget"]
    if widget is not None and widget.is_animating():
        return True
    return False
```

**4. Render loop** — the sleep goes *after* `render_dearpygui_frame()`:

```python
while dpg.is_dearpygui_running():
    # ... poll pipelines, update components ...
    gui_animation.animator.render_frame()
    dpg.render_dearpygui_frame()

    if not _is_busy():
        time.sleep(config.IDLE_SLEEP_S)
```

### Design notes

- **Sleep after render, not before.** This way the last input event still gets a full-speed frame immediately, and the sleep only affects the *next* frame if still idle.
- **`INPUT_ACTIVE_S = 0.5`** provides a grace period after the last input. This keeps tooltips, combo dropdowns, and hover highlights responsive — DPG needs a few frames after mouse-move to settle these. Too short and the UI feels sluggish; too long and the power savings are lost.
- **`IDLE_SLEEP_S = 0.08`** (~12 fps) is a sweet spot: fast enough that the GUI doesn't feel frozen (cursor changes, repaints still happen reasonably quickly), slow enough to cut idle CPU/GPU usage dramatically.
- **`time.sleep` precision**: on Linux, actual sleep granularity is ~1–4 ms (timer slack), so 80 ms sleeps are accurate enough. On Windows, default timer resolution is ~15.6 ms, which is still fine at this scale.
- **Animations self-wake**: since `_is_busy()` checks `animator.active_count`, starting an animation (e.g. a fade or smooth scroll) automatically returns to full frame rate for the animation's duration.
- **No explicit target FPS**: the pattern doesn't set a target frame rate. Full-speed mode runs at whatever vsync or the GPU provides; idle mode is governed by the sleep duration. This is simpler and more robust than trying to maintain a precise low FPS.

### When to use

Good candidates: apps with static content display (image viewers, graph viewers, document readers). Poor candidates: apps with continuous animation (real-time video, particle systems) — they're always busy anyway.

Currently used in: `raven-cherrypick`, `raven-xdot-viewer`.

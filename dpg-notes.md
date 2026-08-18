# DearPyGui Notes

Reference notes on DPG gotchas, internal mechanisms, and workarounds.
Derived from experience with Raven apps and confirmed against DPG 2.0 / ImGui source.

**This file is indexed by `.claude/skills/dpg/`**, a router that maps "if you are about to do X, read section
Y" so an agent lands on the right section instead of the whole file. Two consequences for anyone editing here:

- **Adding a section that answers a *new question* means adding a row to that router.** A question nobody can
  look up is one that gets re-derived, which is what the router exists to stop. A new fact inside an existing
  section needs nothing — the row already points at it.
- **Renaming a heading breaks a row silently**, because the router cites section names verbatim and prose
  never errors. So it is checked instead of remembered: run `python .claude/skills/dpg/check_router.py` after
  editing either file, and it reports every citation and path that no longer resolves.

This file stays authoritative. The router deliberately holds no content of its own, because two copies of a
fact means one of them is stale and nobody knows which.

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

## GLFW callbacks are the exception: they run *on* the render thread

Everything above describes DPG's own callback registry. A callback registered directly with **GLFW** does
not go through it, and the difference is the one that matters for deadlocks.

DPG statically links GLFW and exports its symbols, so a GLFW callback can be installed from Python with
`ctypes` against the already-loaded `_dearpygui.so`. GLFW dispatches such callbacks from
`glfwPollEvents()`, which DPG calls **inside** `render_dearpygui_frame()` — so the callback executes
synchronously within frame processing, on the render thread.

Consequences, measured 2026-08-07 by installing `glfwSetDropCallback` and dropping a file (the callback
thread id came back identical to the render loop's):

- It may touch DPG state freely, like any other thread (see "Thread architecture").
- It must **not** call anything that waits for a frame — `split_frame()`, and therefore the modal
  messagebox. This is the render-thread case from the `split_frame` rules below, reached by a route that
  does not look like the render loop at all. `guiutils.split_frame` detects it and reports rather than
  hanging, which is exactly the situation that guard exists for.
- The fix is the usual one: capture what arrived, hand off to a background task or a frame callback.

The live case is **OS-level file drag-and-drop**, which DPG does not implement at any version (checked
against the binary, not the docs: no public API takes a dropped file, the handler registry has no drop
handler, and DPG's own `drop`-named symbols are all ImGui's widget-to-widget DragDrop). GLFW does implement
it, cross-platform, and installing the callback works. It ships as `raven.common.gui.filedrop`, which does
exactly the hand-off above — the C callback copies the paths and queues them, and a worker runs the app's
handler, so handlers may show dialogs.

Two further facts about that route, both measured (`investigations/dpg-dnd/`):

- **`show_viewport()` is what makes DPG's window the calling thread's current GLFW context** — not the
  first rendered frame, as one might assume from needing the render loop for everything else. It is NULL
  before that call and on every other thread, so the callback can only be installed from the render thread
  after `show_viewport()`, and that is the uniform install site.
- **GLFW has no drag-*hover* event** — no drag-enter, drag-over or drag-leave, only the drop itself on
  release. So no drop target can highlight while a drag is in flight, and a UI cannot ask the user to aim
  at one of several targets. An app with more than one destination has to decide from the dropped file.

## `split_frame()` mechanism

`split_frame()` waits for `frameEndedEvent`, which `Render()` signals at the
end of each frame. This is why:

- **Safe** from the callback thread and user background threads — they're
  waiting for the main thread to complete a frame, which it can do independently.
- **Deadlocks** from the main thread — it's the thread that needs to signal
  the event, so it can't also wait for it.

### Use `guiutils.split_frame`, not `dpg.split_frame`

The hazard above used to be documented and hoped for. It is now enforced —
enforced, not solved. `raven.common.gui.utils.split_frame(operation=...,
required=...)` cannot wait on the render loop thread any more than the bare DPG
call can; the constraint is in the mechanism, not the wrapper. What it adds is
detection: it checks the calling thread and converts the hang into something you
can read.

- `required=True` (default) raises `RuntimeError` naming the operation. For code
  where waiting *is* the job — `wait_for_resize`, a double-buffer swap, settling
  a scroll maximum — there is nothing to degrade to, so fail fast and loudly.
- `required=False` logs a warning and returns `False`, letting the caller
  continue on whatever geometry it has. For waits that only *improve* a result:
  `recenter_window`'s offscreen measure (an off-center window beats a dead app),
  and `messagebox.modal_dialog` (which is often the error-reporting path itself,
  so raising there would replace the reported error with one about reporting).

**One check covers two separately documented hazards.** Startup code runs on the
main thread as well, so `is_render_thread()` also catches "called before the
render loop exists" — the pitfall behind the "defer startup work that may show
an error dialog to a frame callback" rule. There is no second predicate to
remember.

Why bother, when a docstring already said so: a deadlock is the only DPG failure
that produces *nothing* — no traceback, no log line, no exit code. It is
indistinguishable from a slow model, a wedged GPU, or a hung network call, so it
costs an hour before you even suspect the right subsystem. Any named error beats
it.

When the guard sends you to a frame callback, read the next section first —
`set_frame_callback` has a footgun of its own.

## `set_frame_callback` holds one callback per frame number

A second `dpg.set_frame_callback(N, ...)` for the same `N` **silently replaces**
the first. No error, no warning; the earlier callback simply never runs. Learned
the hard way, and easy to reintroduce, because the two registrations are usually
written months apart in different parts of a startup sequence and neither looks
wrong on its own.

Combine the actions into a single callback, or give each a frame number nothing
else uses. Frame 10 is the de facto Raven convention for "the GUI has settled",
so it is the number most likely to already be taken in a given app.

When the actions are not known statically — one per widget in a list that is
built at runtime, say — the combining has to happen at runtime too: accumulate
them in a dict under a lock and register *one* master callback that drains it.
Raven has no live instance of this, but it is the general form of "combine into
a single callback", and it is what a per-widget deferred action needs.

Audited 2026-07-30 across the constellation: no app registers a frame number
twice. `conference_timer` registers 10 and 12 in each of two branches, but the
branches are mutually exclusive (`if font_size <= reference_size: ... else: ...`),
so only one set is ever installed. No library-level code registers a frame
callback at all, so an app cannot collide with a shared helper — worth preserving,
since such a collision would be invisible from the app's own source.

Related: for order-sensitive *input* handling, defer in the main loop rather than
via `set_frame_callback` — rapid input would overwrite the pending callback. See
"Mitigation: defer the order-sensitive action" under Keyboard input.

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

## `dpg.mutex()` — the atomicity tool that Raven cannot currently use

`with dpg.mutex():` is what makes a multi-call sequence land in one frame, so that
the render loop never observes an intermediate state — the documented use is
exactly a delete-then-add widget swap (DearPyGui discussion #1002).

In Raven it is **disabled where it is most wanted.**
`chat_controller.replace_last_paragraph` has it commented out with a TODO: holding
it there hangs the app at random during `on_llm_progress`. The live consequence is
that a streaming paragraph swap is *not* atomic, so a reader of panel geometry can
catch the content while the old paragraph is gone and the new one is not yet
rendered. Anything that measures the panel during streaming has to tolerate that;
`DPGLinearizedChatView.scroll_view` does, by re-issuing its scroll.

Only one live use remains in the tree (vendored `file_dialog/fdialog.py`).

**When investigating, start at issue #2366** ("Deadlock when holding dpg.mutex() a
long time in a frame callback") — same primitive held across slow work, and the
Markdown render inside that mutex is precisely a long hold. Treat it as a lead
rather than the cause: #2366 is reported from a *frame callback*, and its reporter
found keyboard and mouse handlers unaffected, while `on_llm_progress` runs on a
background task thread. Not the same dispatch context, so the match is
suggestive and unconfirmed.

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

## Diagnosing background-task races

Raven's DPG apps push decode/mip/texture work onto background threads, guarded by a monotonic generation counter (bumped on each image switch) and cooperative cancellation (`bgtask.TaskManager` sequential mode cancels the prior task when a new one is submitted). Two failure modes recur, and both are **silent** by default:

- **A discarded stale result looks identical to work that never ran.** A task that completes but finds `e.generation != current_generation` (or `e.cancelled`) must drop its result — correct, but invisible. When the symptom is "the image never loaded", you can't tell a discarded result from a task that was never submitted. Log the discard at debug level at *each* generation/cancel guard, including both generation numbers; it collapses a multi-round hunt into a single repro.
- **Cooperative cancellation only bites where the task checks the flag.** A superseded task keeps running until its next `e.cancelled` check. Check the flag *before* expensive or IO steps (decode, GPU work), not only after — otherwise a cancelled task burns a full decode, or faults on a path that a concurrent triage/file-move relocated out from under it (a `FileNotFoundError` whose traceback points at the *old* image, not the one you're loading). Treat a post-cancellation failure as expected: log it quietly, don't surface a traceback.

Meta-pattern: a single fast user gesture (e.g. cherrypick's `C`+`Right`) exercises several of these seams at once, so one observed symptom often has multiple *independent* causes. You can only see one per repro — fixing cause #1 unmasks #2 — which is why DPG concurrency bugs feel like layered detective work. Instrumenting the silent points up front is what makes the layers visible.

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
window parameter takes precedence.

**It clamps an explicitly sized window, not only an autosize one** (measured
2026-08-03): a window created with `width=400, height=48` and no `min_size`
reports a rect of `400x100`. So "I set the size myself" is not an escape.

**Symptom**: the window appears to have phantom blank space below the content.
Looks like padding or an extra text line, but is actually the window being
clamped to its minimum height.

**Why it is not merely cosmetic**: a DPG window swallows the mouse across its
whole rect, background or not (see `investigations/dpg-overlays/`), so the
phantom area is a dead zone over whatever it covers. Every floating overlay in
Raven passes `min_size=[1, 1]` for this reason.

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

## A modal window does not stack over another modal window

`show_item` on a second modal while one is already up does nothing visible. The call succeeds, no error
is raised, and the window simply never appears — `is_item_visible` on it stays `False` for as long as the
first modal is up. Measured 2026-08-17 with two `modal=True` windows, the second shown from the render
loop several seconds after the first, and still absent eight seconds later.

So a modal that wants a modal of its own — a file dialog offering its own help card, say — has to hide
itself first and restore itself when the inner one closes. Whatever the *app* keys on to mean "a picker
is up" must keep answering yes across that gap, or the app un-suppresses its own hotkeys and file drops
exactly while the inner window is on screen.

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

## Character ranges are automatic; don't declare them

From **DPG 2.3**, `dpg.add_font_range` and `dpg.add_font_range_hint` are no-ops that emit a
`DeprecationWarning` — "character ranges are now automatic". Raven requires `dearpygui>=2.3` for
exactly this reason and declares no ranges at all: `dpg.add_font(file, size)` is the whole of it, and
a font carries every codepoint its TTF has glyphs for. A character that renders as a box means the
*font* lacks the glyph, so the fix is a different TTF, never a range call.

The history matters only if you meet an older version: through DPG 2.2 a font loaded plain covered
Latin-1 (~224 codepoints) and anything beyond it — Greek, the math symbols Raven's BibTeX importer
emits — had to be requested explicitly. Raven asked for `0x100`–`0x2FFF`, ~11,500 codepoints, which
was cheap at 20 px and ruinous at 600 px (each glyph ~350×600 px, so ~2.5 billion pixels of atlas),
hence a standing workaround for apps wanting only digits at large sizes. Automatic ranging retires
both the request and the workaround.

Whether it also retires the *large-size* atlas hazard is not something we have measured on 2.3 — the
overflow ceiling below was established on the old explicit-range behaviour. Treat it as still live
until someone re-measures.

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
- **Setting a value programmatically does not fire the widget's callback.** `dpg.set_value(item, x)` changes the value and calls nothing; a callback runs only for a change the *user* made. Measured 2026-08-13 on `add_combo` and `add_input_text` — both silent, both with the value genuinely changed afterwards.
  - This cuts both ways, and both matter. Where a write should have consequences, invoke the callback yourself — `raven-avatar-settings-editor`'s combo-cycling does exactly that, and `fdialog`'s save-mode click populates the filename field and then calls `_update_search()` by hand. Where a write should *not* have consequences, this is what makes that free: `fdialog` fills the filename field from the cursor row during arrow navigation without re-running the filter, which would otherwise collapse the listing to one row and strand the cursor.
  - So a "why is my callback not firing?" and a "why did my programmatic write cascade?" are the same fact seen from two sides. Neither is a bug.
  - **All of which holds only while the field is *inactive*. On a field that holds the caret, the write is undone.** ImGui keeps its own edit buffer for an active `InputText` and that buffer is authoritative: `set_value` appears to work — `get_value` immediately afterwards reports the new string — and then the next frame writes the old buffer back **and fires the edit callback while doing it**. So on an active field both rules above invert: the write does not take, and a callback fires anyway. Measured 2026-08-17; typing `abc`, writing `SETVALUE`, then typing `Z` yields `abcZ`.
    - **`focus_item` elsewhere is what releases the caret, and it does not happen on the calling frame.** Polling `is_item_active` once per frame after focusing a button gave `[1, 0, 0, 0, 0, 0]` — still active for the first frame, released from the second. **Treat that as one observation on an idle app, not as a frame count to code against**: `focus_item` queues a change that ImGui applies on its next NewFrame, so how many *rendered* frames that costs depends on what else is queued and where the vsyncs fall. A dance built on "wait one frame" is a race that passes on a quiet app and fails on a busy one.
      - So **poll the predicate, don't count frames**: `split_frame()` until `is_item_active` goes false, with a bounded number of tries and a log line if it runs out. Correct whatever the real number is on the day, and it degrades into a visible complaint rather than a silent wrong result.
    - **But refocusing the field afterwards arms ImGui's select-all**, so the user's next character replaces the whole content rather than appending to it. After the full dance the field genuinely held `DANCED3`; typing `Y` left it holding `Y`. Any "complete the text and hand the field back" feature has to answer this before it can work, and DPG exposes no caret or selection API to answer it with.
    - **`configure_item(default_value=...)` does not get around it either**, and fails harder: where `set_value` at least reports the new string before the revert, this one reads back unchanged on the very next line. Measured 2026-08-17 — `before='abc' after='abc'` with the caret in the field. So there is no spelling of "write this field while the user is typing in it"; a feature that needs one has to be redesigned rather than re-spelled.
    - Writing a field the user is *not* typing in is unaffected, which is why `fdialog`'s save-mode click path works: clicking a listing row is itself what deactivates the field.
- **`configure_item(item, default_value=...)` *does* change the live value** — on `add_input_text` and `add_combo` alike, measured the same day. The name is the trap: "default" reads as creation-time-only, and reasoning from the name gives exactly the wrong answer. `fdialog`'s path field has always relied on this to track the current directory. Prefer `set_value` for a pure value write anyway, since it says what it does; use `configure_item` when the value changes *along with* other configuration, as when a combo's `items` are replaced.

- **Swapping content in: build hidden, then hide the old one *before* showing the new one.** Three widgets
  do this — Visualizer's annotation tooltip and info panel, and `raven.common.gui.thumbnailgrid` — and the
  order is not arbitrary. Deleting the old content first leaves every frame until the new content exists
  rendering an empty panel, which on a few hundred items is a visible blank-and-repopulate; so the
  replacement is built into a fresh hidden container and swapped for the old one.
  - Given that, showing the new before hiding the old *looks* safer: a frame caught between the two calls
    then renders both, and since the old content comes first the viewport shows it unchanged. **That
    reasoning holds only while the two contents are the same**, which is never the case a rebuild exists
    for. Tried in the grid on 2026-08-14 and reverted the same day: switching filters showed a few frames of
    the *previous* listing. A frame of nothing beats a frame of the wrong thing.
  - Two further reasons, each from one of the other two call sites. Showing both means **laying out both**,
    and neither the info panel's 400 abstracts nor a grid of unclipped tiles is free for a frame. And with
    both shown the new container is *displaced*, so any position measured in it is wrong — which the info
    panel depends on, since it restores scroll anchors by measuring widgets after the swap.
  - Deleting the old container is a third step, after a `split_frame` (Visualizer, which always runs off the
    render thread) or after a tick (the grid, which cannot assume that — Cherrypick drives its `update`
    *from* the render loop, where waiting for a frame can never succeed).

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

# Keyboard input

## `mvKey_*` constants vs. runtime codes (the 517/518 trap)

A key-press handler receives the live **ImGuiKey code** in `app_data`. For most keys, `dpg.mvKey_*` equals that code, but a handful of constants are **stale 1.x values** that no longer match what's delivered — most notably **Page Up = 517** (`mvKey_Prior` is still 266) and **Page Down = 518** (`mvKey_Next` is still 267). Comparing against the constant silently never matches; compare against the literal code instead. DPG 1.x reported Windows-VK-style codes that the constants matched; DPG 2.0 rebased delivered codes onto the ImGuiKey enum but forgot to update these specific constants.

Also trapped (constant kept its 1.x value, real code is the gap): LWin 343→530, RWin 347→534, Quote 39→596, Colon 59→601, Plus 61→602, Tilde 96→606. Everything else (letters, digits, F1–F24, arrows, Tab, Home/End, modifiers as L/R pairs, numpad) matches.

## Same-frame dispatch is by keycode, not press order

A keyless key-press handler is dispatched once per key pressed *that frame*, in **ascending keycode order** — ImGui's per-frame edge detection discards the sub-frame order in which keys were physically struck. So when a lower-keycode handler mutates state that a higher-keycode handler reads, two keys struck within the same frame interact as if the lower-keycode one came first, regardless of the real press order. (In raven-cherrypick: triage letters all sort after the arrows — `C`=548 vs `Right`=514 — so a fast two-handed `C`+`Right` is dispatched `Right`+`C`; navigation moves the current image synchronously, so the triage key then tags the *next* image.) If correctness depends on the order of two near-simultaneous keys, you cannot rely on dispatch order; see *Mitigation* below.

**Full code↔name table, the trap details, and the reproduction script:** `briefs/reference/dpg-keycodes.md`.

## Mitigation: defer the order-sensitive action

You can't reorder DPG's same-frame dispatch, so make correctness independent of it: keep the higher-priority action synchronous (so it acts on current state) and **defer the lower-priority one by a frame**. In raven-cherrypick the triage keys (`C`/`V`/`X`) stay synchronous, while the navigation keys store a pending thunk (`_request_nav` → `_pending_nav`) that the main loop applies once per frame, before the component-update step consumes the change. A same-frame triage key then runs against the pre-navigation image.

Cost is one frame (~16 ms) of latency on the deferred action — imperceptible, and held-key repeat just gains a constant one-frame offset (no cumulative lag). Apply the deferral in the main loop, **not** via `set_frame_callback`: only one callback can be registered per frame number, so rapid input would silently overwrite it.

## Focus is not the same as the caret: gate hotkeys on `is_item_active`

When a global key handler needs to know "should this key go to the text field instead of the app", **ask `dpg.is_item_active`, not `dpg.is_item_focused`.**

ImGui gives nav focus to the first navigable item of a newly focused window all by itself. So a text field reports **focused within the first few frames with no user having touched it**, and a handler gated on `is_item_focused` silently swallows every key it hands to the field — from app start until something else is clicked. *Active* is the state that means the field owns the caret. Measured on a multiline `add_input_text`:

| composer state | `is_item_focused` | `is_item_active` |
|---|---|---|
| startup, no interaction | **True** | False |
| clicked in / typing | True | **True** |
| after Escape (InputText's own cancel) | True | False |
| after clicking another widget | False | False |

Corollary: `dpg.get_focused_item()` is not a cross-check — it kept naming the field even in the last row, where `is_item_focused` on that same field was `False`.

**But *Enter* is the exception, and which predicate is right depends on the field's kind.** A **single-line** `InputText` deactivates itself on Enter — the key commits the edit — so a hotkey handler gated on `is_item_active` can never fire on Enter: by the time it runs, the field is inactive. A **multiline** field does not, because there Enter inserts a newline.

| after pressing Enter | `is_item_focused` | `is_item_active` |
|---|---|---|
| single-line `add_input_text` | True | **False** |
| multiline `add_input_text` | True | **True** |
| multiline `add_input_text`, **Ctrl+Enter** | True | **False** |

So an app whose text field is single-line must gate its Enter handler on `is_item_focused` while still gating its *bare-key* branch on `is_item_active` — two different questions about the same widget, each chosen for the state the key actually arrives in. Both Raven GUI apps do this, and they differ from each other because their fields differ in kind: `raven-visualizer`'s search field is single-line, `raven-librarian`'s composer is multiline. Learned by regression — switching the Visualizer's Enter gate to `is_item_active` silently killed its search.

**The rule is really about the chord that *commits*, not about the kind of field** — the third row is what makes that visible. Ctrl+Enter commits and deactivates a **multiline** field too, so a send handler gated on `is_item_active` can never fire on it either, exactly as for single-line bare Enter. Found 2026-08-04 when `raven-librarian` made Ctrl+Enter its default send chord: the chord unfocused the composer and sent nothing, silently, because the branch guarding it tested a state the commit had already cleared.

Stated generally, so it survives the next variation: **gate on `is_item_focused` any key that ends the edit, and on `is_item_active` any key that happens during it.** Ask which of the two a chord is before choosing the predicate; "it is a multiline field, so Enter keeps it active" is a fact about one chord, not about the widget.

**Escape is not a second exception.** It deactivates either kind, so a bare-key branch gated on `is_item_active` is live again on the next press and needs no handler of its own to "restore" focus. Measured on a multiline field; confirmed behaviourally on a single-line one (the Visualizer's navigation keys reach the info panel after `Ctrl+F`, `Esc`).

## `focus_item` cannot focus a child window — and does harm when asked to

`dpg.focus_item` works on ordinary items (measured on a button: focus moves on the *next* frame, not the same one). On a **child window** it does not merely fail: focus lands on the first navigable item of the enclosing window and is **activated** — so if that item is a text field, the call *hands it the caret*.

This makes "park focus on the scroll panel so the navigation keys are live" — the natural thing to write for a reading-first app — the one instruction that reliably does the opposite. There is also no need for it, since ImGui's default leaves the auto-focused item *inactive*, which is exactly the state a caret-gated handler wants. To move focus out of a text field deliberately, focus a real widget (a button works).

**A child window is not unfocusable, though — only unaskable.** Clicking one focuses it, scrollbar included: `is_item_focused` on the panel goes true on mouse-down and stays true after release. What has no working spelling is *requesting* it.

**`focus_item` is refused in exactly one situation: focus currently sits on an item at *window level*, and the target is inside a child window.** Measured 2026-08-18 on **DearPyGui 2.3.1**, across every source/target position — window level, child A, child B. The version is worth carrying with the result: unlike the other entries here this reads as a bug rather than a design choice, so it is the one most likely to change under upgrade. `investigations/dpg-focus/focus_across_child_window_probe.py` re-runs it in under a second. Window→child is refused both ways round, leaving the target neither focused nor active; **child→child works, including between siblings**, as do child→window and window→window. Nothing else is a factor: not modality, not whether the target is a text field or a button, not how many frames pass afterwards.

The asymmetry is worth stating plainly, because the obvious theory — "a child window is a boundary `focus_item` cannot cross" — is wrong and fits most of the evidence. A field in child A can be focused from child B perfectly well. What cannot reach into a child is an item sitting directly in the enclosing window.

This is why the failure is so rarely met. Focus is usually already inside some child window, or has not been placed anywhere yet, so `focus_item` on a nested field just works — Librarian focuses its composer, Visualizer its search field, and `FileDialog`'s Ctrl+F has always worked after a row click, the row and the field sharing a child window. Librarian's deliberate park is child→child too: `chat_send_button` lives in `chat_controls` beside the composer.

`FileDialog` met it the moment Tab began parking on the OK button, which sits directly in the dialog window rather than in the listing's child window. Every later Ctrl+F and Tab-back was a window→child request: the handler fired, the call returned, the caret never arrived, and typing went nowhere.

**So a parking spot is not free to choose. Park inside a child window — the same one is simplest — whenever something in a child window must be focusable afterwards.** Any focusable item there will do; the constraint is position, not kind. From window level there is one escape hatch: `focus_item` on the *child window itself* focuses its first navigable item, which is the "does harm" behaviour above put to work deliberately, and only helps when the item you want happens to be first.

**A focused button is a safe parking spot:** DPG does not enable ImGui's keyboard-nav activation, so a focused button ignores Space and Enter and cannot fire its callback. Verified, because parking focus on a *send* button is not something to assume about.

Since 2026-08-18 it is also *pinned*, by `test_a_focused_button_ignores_the_keys_that_would_press_it` in `raven/common/gui/tests/test_focus_semantics.py` — the one test in the suite that synthesizes key presses, with a companion control asserting that a synthesized key reaches the app at all, so that the button's silence cannot be confused with a keystroke that never arrived. Nothing in DPG's API reports the flag (`configure_app` takes bare `**kwargs`; the only nav-related names are theme colours), so behaviour is the only place it can be read. It became worth the intrusion when `FileDialog` started parking there on every Tab. Note which key the pin actually protects: not the arrows the parking is done *for* — a button has nothing to do with those either way — but **Enter**, which in that dialog descends into the folder under the cursor. A focused OK button that answered Enter would commit and close the dialog instead, and in save mode that writes a file.

**A `menu_item` has no focus state at all, and `focus_item` on one is a no-op.** Not "returns False" — `get_item_state` on a `menu_item` returns a dict with no `"focused"` key, so `dpg.is_item_focused` raises `KeyError: 'focused'` rather than answering. Asking `focus_item` to focus one changes nothing: measured 2026-08-17 with the caret in a text field, focus was on that field before the call and still on it after, still active.

So this is the third distinct case in DPG's focus model, and the mildest: a child window *cannot* be focused and does harm when asked, a button focuses normally, and a menu item quietly declines. Anything built from menu items — `fdialog`'s places panel is — cannot use the focus-dispatch idiom, and needs either a drawn cursor with a mode flag, or a real focusable widget inside it to hold the focus on the panel's behalf.

## What still reaches a global handler while a single-line field holds the caret

Nearly everything, which is what makes a keyboard-operable dialog possible at all with focus parked in a
text field. Measured 2026-08-17 on `add_input_text` with the caret in it, on this desktop: Ctrl+Enter,
Alt+Up, Ctrl+Up, Ctrl+Space, Ctrl+Home, Ctrl+Shift+1, and bare Up / Down / Home / End / Page Up / Page
Down all arrive at a `add_key_press_handler`, each carrying its modifiers.

**Tab arrives too, and ImGui does not spend it** — focus does not move and no character is inserted, so an
app is free to give Tab its own meaning while a field is being typed into. One wrinkle: Tab pressed while
the field is *focused but inactive* re-activates it (ImGui's tab-into-field behaviour), so a Tab handler
sees the field become active under it.

The one to plan around is **Ctrl+Enter, which deactivates a single-line field** the same way bare Enter
does — it commits the edit. That is the commit-chord case above, so gate it on `is_item_focused`; and if
the dialog *stays open* afterwards (Enter descending into a directory, rather than accepting), the field
has silently lost the caret and later typing goes nowhere until something reactivates it.

Alt is the modifier that varies by desktop: nothing intercepted Alt+Up under Cinnamon, but window managers
commonly bind Alt chords, and that is a statement about the desktop rather than about DPG. Both dev
machines here run Cinnamon, so this one cannot be settled in-house — it is for users on other desktops to
report. Which is the argument for Ctrl+Up existing as an alias regardless of what Alt does.

## `is_key_down` is sampled when the callback runs, not when the key was pressed

Modifier state is read *inside* the handler, and handlers are dispatched per frame — so the answer
describes the modifier at dispatch time rather than at press time. A chord whose modifier is released
before the next frame arrives reporting no modifier at all.

Human typing never does this; a modifier is held for far longer than a frame. **Synthetic input does.**
`xdotool key ctrl+Up` presses and releases both keys in well under a millisecond, and the `Up` is
dispatched a frame later with `is_key_down(LControl)` already false. Sent as `xdotool keydown ctrl` /
`xdotool key Up` / `xdotool keyup ctrl`, the same chord reports its modifier correctly.

This is a hazard for probes and driven-GUI tests, not for apps — but it fails in the direction that
invents a finding: the chord looks like it does not carry its modifier, and that reads as a DPG
limitation rather than as an artifact of the harness. Hold synthetic modifiers across frames.

Related, and visible in any such log: **held modifier keys arrive as repeated key presses** (~50 ms
apart, LControl / LShift / LAlt alike), alongside a companion pseudo-key — 663 for Ctrl, 664 for Shift,
665 for Alt — that no `mvKey_*` constant names. A handler that acts on a bare modifier keycode therefore
fires over and over while the key is down.

## Investigation history

- 2026-08-17: Surveyed which chords survive a single-line `InputText` holding the caret, ahead of building
  `FileDialog`'s keyboard operation. All of them do, Tab included. Confirmed the 517/518 codes from the
  live enum in the same run — Tab=512, Up=515, Down=516, **517**, **518**, Home=519, End=520, so Page
  Up/Down sit exactly where the sequence says they should while `mvKey_Prior`/`mvKey_Next` still read
  266/267. Two harness traps found on the way: synthetic chords lose their modifier unless it is held
  across frames, and held modifiers auto-repeat as press events.
- 2026-06-06: Traced a raven-cherrypick mis-tag (fast `C`+`Right` tagging the next image instead of the current one) to same-frame keycode-order dispatch; confirmed empirically that every triage letter outranks every arrow, so navigation always fires first. Fixed by deferring keyboard navigation one frame (`_request_nav`). Resolved the long-standing "mysterious 517/518" in the same pass — the `mvKey_Prior`/`mvKey_Next` constants are stale DPG-1.x values; the live codes are 517/518.
- 2026-08-03: Librarian's arrow keys were dead at app start until the chat log was clicked. Four standalone DPG probes (auto-focus baseline; `focus_item` on a child window vs. a button; a self-driving `xdotool` run that clicks, types and presses Escape; and a Space/Enter test on a focused button) produced the two sections above. The Visualizer had the same pair — an `is_item_focused("search_field")` gate on its bare-key branch, and `focus_item("item_information_panel")` on Enter and Escape, `item_information_panel` being a child window — and was fixed in the same pass: the gate now reads `is_item_active`, Enter parks on `clear_search_button`, and the Escape branch is gone, since `InputText` deactivates itself and deactivated is what the bare-key branch tests for. The input-free half of the probes is now `raven/common/gui/tests/test_focus_semantics.py`, marked `gui` and run with `pytest --run-gui`.

# Scrolling

## Three input paths move a scroll position, and DPG surfaces them differently

A child window's scroll position can be changed by **dragging the scrollbar**, by the **mouse wheel**, or by **hotkeys** you implement yourself. These are not equivalent from the code's point of view: the scrollbar drag is handled inside ImGui, so there is no DPG-level event for it in the way there is for a key press. (Cost real time while building Visualizer's smooth scrolling.)

Background-thread GUI updates interact with this. Raven updates the GUI from worker threads deliberately — DPG permits it, which is one of its real advantages: the GUI behaves like any other data structure instead of a special place you have to marshal into, and that removes a whole category of plumbing. The price is the ordinary price of concurrency, and it shows up here. A sequence that looks atomic in the source — delete a widget, add its replacement — is not, because the render loop is on the main thread and can lay out the container between the two. So a swap briefly shrinks the content, DPG clamps the scroll position to the smaller maximum, and a reader who was below that point is moved. Appending has no such window; only replacing does.

The consequence is a design rule: **decide "has the user scrolled away?" from the scroll position, never from scroll events.** Position is where all three paths end up, so `dpg.get_y_scroll` needs no per-path handling and cannot silently miss one. Watching for the *act* of scrolling means enumerating the paths, and the one that is hardest to hook is the one users reach for most on a long document.

**But position compared against what?** Not against `max_y_scroll` — that is the trap, and it is easy to walk into because it looks like the same thing. Two independent endpoints move: the user moves the *position*, arriving content moves the *maximum*. Comparing position to maximum collapses both into one number, so "content grew" is indistinguishable from "the user scrolled up", and follow-the-tail logic reads the arrival of its own content as a reason to stop following.

Compare the position against **the position you last commanded** instead. Content arrival cannot change that relationship — it moves the maximum and leaves the position alone — while a user scroll is precisely a change to the position that you did not ask for. One remembered integer, no events, and the two causes separate cleanly. Two details make it hold:

- **Clamp the remembered value by the current maximum before comparing**, since DPG pulls the position down by itself when content shrinks, and that is your doing rather than the user's.
- **Remember the concrete number that gets applied.** The comparison is only meaningful if the remembered value and the panel's reported value denote the same position, so record the actual offset you are setting. Compute the maximum yourself — you need it for clamping anyway — and record that, rather than asking to be moved somewhere and remembering the request. Raven's chat view keeps its scroll setter honest with a non-negative precondition, since a content offset below zero is out of range whatever the toolkit does with it.
- **`get_y_scroll` does not reflect your `set_y_scroll` for more than one frame.** One `split_frame` after commanding a scroll still reads the *previous* position. Measured across a session of streaming replies: an extra frame was needed 114 times out of 115, and two extra frames once. So wait for the panel to report the position you asked for, with a bounded number of rounds, and re-issue the (recomputed) target each round.

  **`raven.common.gui.animation.SmoothScrolling` is the reference implementation, and it already encodes this** — it was found there first, during that class's development, and rediscovered later the expensive way. Its per-frame guard is literally the pattern: it keeps `prev_frame_new_y_scroll` (the last value it wrote) and refuses to advance until `dpg.get_y_scroll` reports that value back, *"Only proceed if DPG has actually applied our previous update. This prevents stuttering, as well as keeps our subpixel calculations correct."* The give-up bound is `update_pending_frames` counted against `update_pending_threshold = 4`, whose comment names the trade-off directly: *"Smaller threshold looks better, but may fire prematurely if a GUI update takes too many frames."*

  Two consequences worth carrying: read that class before writing anything that sets a scroll position, because the lag is not something you can find by reading the DPG docs; and note that "remember the value you wrote, wait for it to be reported back" is the *same* device whether you are animating a scroll or deciding whether a reader has scrolled away. Raven's chat view calls it `_commanded_y_scroll`; `SmoothScrolling` calls it `prev_frame_new_y_scroll`. Convergent, which is a good sign for composing the two.

  This is worth handling rather than shrugging at, because anything comparing the position against what it commanded — follow-the-tail logic does — reads its own in-flight command as a discrepancy, which is indistinguishable from the user having scrolled away. Diagnosed the hard way: a `NEAR MISS` reporting `gap=52.0px ... drifted 52.0px from the 533.0 we last commanded` was simply the position not having caught up, and the earlier hypothesis that DPG had clamped the command to a momentarily smaller content height did not survive the log — the very first retry in the session read a position of `0.0` against a maximum of `692.0`, where nothing had shrunk.

  Recomputing the target each round covers two further cases cheaply, so there is no reason not to: the target genuinely moving (content keeps arriving during a stream), and a real clamp if one occurs. A clamp window does exist in Raven — `replace_last_paragraph` swaps a paragraph by delete-then-add, and the `dpg.mutex()` that would confine the pair to one frame is disabled because holding it hangs the app — it just isn't what the measured cases were.

  **Bound the loop by a round count, not a pixel threshold.** A shortfall can be as large as whatever left the view, and where the text comes from an LLM, a "paragraph" is however much the model chose to emit between newlines — a line, or a screenful. Tool output is the case to worry about rather than the model's prose: it is machine-generated and under nobody's stylistic control. So no tolerance can cover the shortfall in principle, while waiting for the reported position is magnitude-independent. (In Raven a tool result arrives as a complete message rather than as streamed text, so it takes the swap path where the whole streaming widget is replaced — a bigger shrink again, and why that path restores the reader's offset explicitly.)

  The knock-on is what makes the check worth doing at all rather than accepting a slightly-off view. If anything else compares the position against what you commanded — follow-the-tail logic does — then a silently clamped command looks precisely like the user having scrolled away.

Get this wrong and it **latches**, which is why it is worth the care. The verdict is sampled once per arriving chunk, before that chunk renders; a single transient displacement makes the answer "not following", after which every later sample is taken from a view that has fallen one more chunk behind. The gap grows monotonically and never recovers, so a two-line hiccup freezes the view for the whole turn. Measured on a live reply before the fix: 52 → 68 → 120 → 146 → 172 → 198 → 224 px.

## `max_y_scroll` moves when content is added

`dpg.get_y_scroll_max` is a function of the current content height, so appending to a container changes it immediately. Anything that asks "is the view at the bottom?" in order to decide whether to *keep* it at the bottom must sample that **before** adding the content, and act **after**. Sampling afterwards reports "not at the bottom" every time — the content grew, the position did not — so follow-the-tail logic written that way never engages, and the view sticks wherever the stream started.

Raven's chat view carries this as an explicit split: `DPGLinearizedChatView.should_follow_tail()` samples, and `follow_tail(was_following)` takes the sampled answer as an argument rather than asking for itself, precisely so the ordering cannot be lost at a call site. Every scroll the view performs goes through one private setter that records what was asked for, so the "compare against what you commanded" test above has something to compare against — a bare `dpg.set_y_scroll` anywhere else would be indistinguishable from a user scroll and would silently stop the view following.

Note the hazard is not confined to streaming: a user can scroll *while* content is arriving, so a "user-initiated" scroll is not automatically a quiet moment either. What differs is the consequence — a scroll-end flasher that guesses wrong shows one wrong flash, while follow-the-tail that guesses wrong stops working for the rest of the turn.

# Testing DPG code

## DPG runs without a mapped window, so GUI code is unit-testable

`dpg.create_context()` + `dpg.create_viewport()` + `dpg.setup_dearpygui()` gives a fully working DPG — real widgets, real themes, working getters — **without** `dpg.show_viewport()`. Nothing is mapped, so nothing takes focus and nothing appears on screen. This matters on a shared desktop, where the alternative (launching the app) interrupts whoever is using it.

That makes a large class of GUI code testable against real DPG state rather than mocks:

```python
dpg = pytest.importorskip("dearpygui.dearpygui", reason="dearpygui not installed (GUI toolkit absent in CI)")

@pytest.fixture(scope="module")
def dpg_context():
    dpg.create_context()
    dpg.create_viewport(width=100, height=100)  # never shown
    dpg.setup_dearpygui()
    yield
    dpg.destroy_context()
```

Create the context once per module: it is not cheap, and DPG holds global state, so per-test contexts are both slow and a good way to find out which of your other tests leaked a widget.

**A shared context means widget tags have to be unique across the module**, not merely within a test — a duplicate ID takes the process down rather than raising (pitfall 5). Parameterize the tag on `request.node.name` where a fixture builds the same widget for every test.

**And watch for class-level caches of DPG items, which outlive the context they were created in.** DPG itself is well-behaved here — the failure surfaces as `SystemError: Texture not found` from the first `add_image`, not as silent corruption — but the *cache* is application code, and a "have I initialized yet?" boolean is not the same question as "do my items still exist?". Nothing resets such a flag on `destroy_context`, so the second context finds it set and every cached tag dangling. An app never meets this, holding one context for its whole life; a test suite meets it on the second context it builds. (Live case: the adopted `file_dialog` cached its icon textures, themes and handler registry behind `_class_initialized`. Fixed by having the guard ask the context — `if cls._class_initialized and dpg.does_item_exist("ico_home")` — rather than trusting the flag. Bare `create_context` / `destroy_context` cycles with no such cache were measured to be fine, so this is about the cache.)

**These run in CI, on all three platforms** (since 2026-08-12; `dearpygui` and `mistletoe` are in `.github/workflows/requirements-ci.txt`). The open question was whether GLFW could get a context on a runner with no display server, and it can — ubuntu, macOS and Windows alike, 2090 → 2147 tests passing per platform. Keep the `importorskip` anyway: it costs nothing and it is what lets the suite run in an environment that genuinely lacks the toolkit.

Tests that *map* a window are the separate case and stay out: they carry the `gui` marker and need `pytest --run-gui`.

**Know the ceiling before writing one: "DPG runs headless" is narrower than it sounds** (measured 2026-08-03, on a machine *with* a display). Contexts, widgets, themes and item state all work with an unshown viewport. But `dpg.render_dearpygui_frame()` **aborts the process** — `SIGABRT` on the GLFW assertion `window != NULL` in `glfwWindowShouldClose`, not a catchable exception — so nothing that needs *layout* is reachable: no real scroll extents, no `get_y_scroll_max`, no hit-testing, no measured text sizes.

That is why the existing tests step `animation.animator.render_frame()` — Raven's own animator, pure Python — rather than DPG's frame, and why `test_animation.py`'s `SmoothScrolling` tests assert against state transitions instead of against pixels. The tier this buys is "widget and state logic", not "the GUI works"; someone who reads it the cheap way will write a layout-dependent test and get a core dump rather than a failure. Whether a software GL stack (`xvfb-run`, or Mesa's llvmpipe) lifts the `render_dearpygui_frame` restriction as well as initialization is a separate unknown, and unmeasured.

**Animations need no wall-clock waiting.** `animator.render_frame()` is what Raven's render loop calls; a test can call it directly and step the animation as fast as the CPU allows, with a wall-clock deadline so a bug fails the test instead of hanging the suite.

Wait on a condition specific to the animation under test, *not* on `animator.active_count` reaching zero. `animator` is a process-wide singleton, so an empty-animator condition also waits on anything else running — and if something ambient never ends (a cyclic `PulsatingColor`, say), the test times out blaming the wrong animation.

Two different registrations are in play, and only the first is universal:

- **Every** animation deregisters from the *animator* by returning `action_finish` / `action_cancel` from `render_frame`. `Animator.add` returns the animation, so a test can keep that reference and wait on it — this works for any animation type.
- `WidgetFlash` *additionally* keeps `WidgetFlash.instances`, a per-widget registry for its own de-duplication (at most one flash per widget). That is specific to this animation, and it happens to give a test a convenient public per-widget signal: the widget's key disappears exactly when its flash finishes.

**What this is good for, and what it isn't.** Worth testing: state machines, "restore what you borrowed" contracts, teardown ordering, de-duplication logic, anything holding a lock — behavior that is invisible to the eye and breaks silently. Not worth testing: whether it *looks* right. Layout, spacing and color are a screenshot's job (see "Live GUI testing on a shared desktop" in `CLAUDE.md`), not an assertion's.

First use: `raven/common/gui/tests/test_animation.py`, covering `WidgetFlash`'s color/theme restoration and its ghost-vs-reified de-duplication — both of which had latent bugs that no amount of looking at the screen would have revealed.

## Context recreation is not reliably safe once real widgets have rendered

The cache paragraph above says bare cycles are fine, and they are — including 60 rendered frames on a shown viewport, clean over 8 trials. **That result does not extend to a cycle with an application's widgets in it**, which is the shape a benchmark reaches for when comparing two configurations in one process.

Measured 2026-08-13, two contexts per process, a `FileDialog` built in each, 8 trials per configuration: **3/8 to 8/8 of runs died with `SIGSEGV`** on the second context. Nondeterministic, and not monotonic in anything tried — one configuration crashed 5/8 in one script and 0/8 in another that differed only in leaving vsync alone. Bisecting by dropping one ingredient at a time is therefore useless at this sample size, and the first attempt produced a table that read as if *removing* ingredients caused the crash.

**The mechanism is not identified.** Waiting half a second before `destroy_context`, and calling `stop_dearpygui` first, both changed nothing — but only in a configuration that was not crashing anyway, so neither is evidence. Candidates not ruled out: work still queued on the callback thread, pending texture uploads, driver-side teardown.

Consequences, which are small:

- **An app never meets this**, holding one context for its whole life.
- **The default test suite does not either**, using one module-scoped context per module and never rendering a frame (see the ceiling above).
- **The `--run-gui` group does do this cycle, and is one module away from dying of it.** `test_focus_semantics.py`'s `mapped_viewport` fixture is *function*-scoped: a context created, shown, rendered and destroyed once per test, five times over. Adding any further module that maps a context and renders frames segfaults the group — measured 2026-08-13 at 3/3 with a table in it, 1/3 without, and *only* when `test_focus_semantics` runs first, which alphabetical collection order decides. The group passes today because the one other `gui` test (`test_filedrop`) sorts before it and renders no frames. See the deferred item "The `--run-gui` group segfaults if a second module maps a context".
  - The same shapes do **not** crash outside pytest: five focus-like cycles then a table-building cycle, six table cycles, twelve plain cycles, all clean over three runs each. So something about the pytest process is part of it, and an earlier guess here — that a *heavier subsystem* would be what tipped it — was wrong; 400 plain buttons did it too.
- **A benchmark or probe must use one process per context.** Run configurations as subprocesses and compare their printed output. This is cheap, and it is the only reason the constraint matters at all.

## Introspection gaps to expect

`dpg.get_item_configuration(item)["color"]` reports color as **normalized floats** while `dpg.configure_item(item, color=...)` takes **0–255**, so a read-modify-write round trip has to scale. An unset color reads back as the sentinel `[-1.0, 0.0, 0.0, 1.0]`, and writing that sentinel back (scaled) correctly restores "unset".

`dpg.get_item_theme(item)` returns `None` for an unbound widget, and `dpg.bind_item_theme(item, None)` unbinds — so capture-and-restore of a theme is symmetric with no special case. (`0` also unbinds; prefer `None`.)

**A `mvTable` has no `rect_size` in its item state, so `guiutils.get_widget_size` answers with its *configured* size — which for an autosizing table is `-1`.** That is a faithful answer to a different question: `-1` is the layout directive "fill the available space", and the helper is reporting what the item was told, not what it became. The caller asked for pixels and got a directive, and nothing says so — a guard like `if not height` does not catch it, `-1` being truthy, and arithmetic against "the view's height" then computes against −1. Measured 2026-08-17 while giving the file dialog's table a keyboard cursor, where it scrolled the listing on the third arrow keypress.

Note the fallback is not what rescues the child window beside it: `listing_area` is *also* configured `(-1, -1)`, and reports a real size because `get_item_rect_size` succeeds for it. The fallback runs only for items that have no `rect_size` at all — which is precisely where a configured `-1` cannot be resolved into anything. **So treat a negative from `get_widget_size` as "unknown", not as a size.**

Measure the **enclosing child window** instead — it reports a real size — and keep scrolling the table, which is what `y_scroll` and `max_y_scroll` work on. Both halves were measured on the same widget pair: table `get_widget_size=(-1, -1)` with `max_y_scroll=588.0`, its `listing_area` child window `(1159, 581)` with `max_y_scroll=0.0`.

**Row pitch is not the height you asked for, either.** Cells created with `height=16` came out 18 px tall at a 22 px pitch, below a header occupying the first 26 px. So a row's position is worth reading off the row (`get_item_pos` on a cell, which answers in content coordinates, exactly what `set_y_scroll` wants) rather than computed from the height that was requested.

There is **no getter for theme contents** — you cannot ask a theme for its colors or spacings. Code that needs to restore a themed value therefore tends to hardcode a measured literal instead; see the audit item in `TODO_DEFERRED.md`. Where a *per-widget* getter exists (as for a text widget's own color), use it — the gap is theme state specifically, not all of DPG.

## Windows and child windows have no `rect_min`, and `get_item_pos` answers a different question

Measured 2026-08-14, with a mapped viewport and thirty rendered frames, so this is a property of those item
types rather than of a missing frame:

- `dpg.get_item_rect_min(item)` raises `KeyError: 'rect_min'` for a window or a child window. Ordinary items
  — buttons, text, drawlists — and **groups** do have it, and for them it is a true viewport position.
- `dpg.get_item_pos(item)` works for everything, but it is the position **relative to the parent
  container**, which is not the same thing and is not interchangeable.

**The trap is that the two agree in the easy case.** One level below a window sitting at the origin, a
child window's parent-relative position *is* its viewport position, so code that reaches for `get_item_pos`
when `rect_min` is missing looks correct for as long as every layout is shallow. In a modal dialog three
levels down it reported `(0, 0)` for a widget at `(46, 149)` — and since the value feeds hit testing, the
symptom was a grid of thumbnails that could not be clicked, with nothing logged and nothing raised.

**A group is not a coordinate space, and this is the part that bites.** An item inside a group reports its
position relative to the enclosing *window or child window*, skipping the group entirely — while the group
also reports a position of its own. So accumulating positions naively counts every group twice. Measured
16 × 35 px of overshoot on one dialog's layout, which is enough to put a panel's believed origin below the
content the user is clicking.

**Use `guiutils.get_widget_pos`, which accumulates `get_item_pos` up the parent chain, skips groups, and
subtracts each ancestor's scroll.** Verified against `rect_min` — a true viewport position — at two depths
of the same tree, and at three scroll offsets. A layout position knows nothing of scroll, so a widget inside
a scrolled container would otherwise report where it sits at scroll zero; the *widget's own* scroll is not
subtracted, since that moves its contents rather than the widget. `get_x_scroll` / `get_y_scroll` raise a
bare `Exception` on an item that cannot scroll (a group, a button), so asking is cheap but must be guarded.

**The first-position trap, worth knowing before writing a probe for anything like this.** A group that is
the *first* item in its parent has offset `(0, 0)`, so double-counting it adds nothing and the bug is
invisible. The first probe written for this had groups only in first position and pronounced the sum
correct. A tree that can discriminate needs a group with a sibling before it.

Reproduction: build `window(pos=(30, 70)) -> child_window -> child_window -> group -> child_window ->
button`, render ten frames, and compare `get_widget_pos` of the innermost child window against
`get_item_rect_min(button) - get_item_pos(button)`. It is the `gui`-marked
`test_get_widget_pos_is_viewport_coordinates_however_deeply_nested` in
`raven/common/gui/tests/test_utils.py`.

# Drawlists

## Never size a drawlist to a scroll extent — it will take the X session down

**Measured 2026-08-13, and it cost a desktop.** A drawlist of 860 × 60800 px inside a scrollable child window
— the natural way to build a windowed grid, one canvas covering the whole content height, drawing only the
tiles on screen — rendered the session unusable. Recovery took logging in on a text terminal and sending
`SIGTERM` to the process; nothing on the graphical side responded.

**Every API-level signal said it was fine**, which is what makes this worth writing down rather than filing
under "don't do daft things":

| asked | answered |
|---|---|
| `get_item_height("canvas")` | `60800` — exactly as requested |
| `get_y_scroll_max("scroller")` | `60296` — a correct scroll extent |
| `set_y_scroll(...)` to the bottom, then read back | `60296` — scrolling worked |

Creation raised nothing, the geometry was right, and the numbers were the ones a windowing implementation
would want. The cost appeared only in rendering.

**The mechanism is not identified** and was not chased — the fix does not depend on it, and the experiment
that would settle it is the experiment that killed the session.

**So: size a drawlist to the viewport, never to the content.** A grid that scrolls needs its scroll extent
established by something cheap (a spacer, or the layout of the tiles themselves) and its drawing done in a
viewport-sized canvas, or in per-tile drawlists as `raven/cherrypick/grid.py` does. What it must not do is
ask one drawlist to *be* the scrollable area.

**And when probing DPG for a limit, climb toward it.** The standing instruction is that a headless probe
answers most behavioural questions in seconds, and that is true — but a probe testing *how big* something can
be is a different animal, and the honest version starts small and increases until something gives, rather
than jumping to the size the feature would want. A probe is allowed to fail; it is not allowed to take the
user's desktop with it.

# Tables

## Rows are submitted every frame unless the table clips

`dpg.add_table` takes `clipper` (default `False`). Without it, ImGui walks and submits *every* row each frame regardless of how few are on screen, so per-frame cost grows with the row count rather than with the viewport.

Measured 2026-08-13 on the adopted `file_dialog`'s listing, vsync off, median over 200 frames:

| rows | `clipper=False` | `clipper=True` |
|---|---|---|
| 0 | 0.74 ms | 0.77 ms |
| 500 | 1.10 ms | 1.01 ms |
| 2500 | 3.76 ms | 0.68 ms |

At 2500 rows the clipped table costs what an *empty* one costs: the row count stops appearing in the frame time at all. Sorting, scroll extent and row alignment were checked by screenshot afterwards and are unaffected — the clipper changes what is submitted, not what the table contains.

**Its one requirement is uniform row height**, which is why it cannot simply be switched on everywhere: a table whose rows vary in height needs each cell created with an explicit matching height first. The file dialog qualifies because every cell already passes `height=self.selec_height`.

Worth knowing what this does *not* fix: building the rows still costs what it costs (~60 µs/row there, so ~0.19 s for 2500), and deleting them likewise. The clipper is about the frames after the build, not the build.

**A clipped-away row has no geometry, and says so as `0` rather than as an error.** ImGui never submits it, so `get_item_pos` and `get_item_rect_size` on one of its cells return zeros — indistinguishable from a row that is genuinely at the top, and from a row that has simply not been laid out yet (which is what every row reads as during the build that creates it). Measured 2026-08-17 giving the file dialog's table a keyboard cursor: Page Down asked where row 28 was, got `0`, concluded it was already on screen, and did not scroll.

The sting is that **the row you cannot measure is always the row you want**: anything scrolling *to* a position is by definition aimed off screen. So do not ask the destination where it is. Measure the *pitch* instead — two adjacent rows, while they happen to be visible — and compute from it. That is sound precisely because the clipper already requires uniform row height, and both the pitch and the header's origin are constants of the table's styling rather than of its contents, so one measurement serves for the widget's life. In `fdialog` this reads `origin=26, pitch=22` for cells created with `height=16`.

The bug it produces is a memorable shape: the scroll appears broken until you happen to scroll the target row into view by mouse, after which it works — because now the row has geometry to report.

## To find which rows are on screen, ask a cell — never the row

Any "fill this in only for what the user can see" feature needs to know which rows are visible. `dpg.is_item_visible` is the right call and the `table_row` is the wrong thing to call it on, which is the trap: the row is what the clipper is clipping, so it is what a reader reaches for.

Measured 2026-08-13 on a 400-row scrolled table (`investigations/filedialog-performance/probe_row_visibility.py`), asking at the top, the middle and the bottom:

| asked of | `clipper=True` | `clipper=False` |
|---|---|---|
| the `table_row` | the on-screen run **plus row 0**, at every scroll position | **every row**, always |
| a widget *inside* the row | the on-screen run, contiguous, correct | the on-screen run, contiguous, correct |

So the cell answers the question and the row does not, under either configuration. Two consequences worth having:

- **The predicate does not depend on the clipper.** A table can be lazily filled whether or not it clips; the clipper is a separate decision about frame cost.
- **The row's answer is wrong in two different ways**, so neither is a usable approximation. Unclipped it is uselessly permissive; clipped it is nearly right, which is worse — a lazy-fill built on it would work in casual testing and decode one extra row forever.

Row 0's appearance in the clipped answer is unexplained; it was not chased, because the cell-side answer is what the feature needs.

**Rendered frames are required before any of this means anything** — visibility is a property of the last frame drawn, so it is unavailable headless (see "Testing DPG code"), and unavailable for a window shown microseconds ago.

## What a sort callback receives

A `sortable=True` table's `callback` is called as `callback(sender, sort_specs)`, where `sender` is the
table itself and `sort_specs` is one of:

| `sort_specs` | meaning |
|---|---|
| `None` | no sorting — the header's third state, reached by cycling past descending |
| `[[column_id, direction]]` | sorted by one column |
| `[[column_id, direction], …]` | sorted by several, when the table has `sort_multi=True` |

`direction` is **`1` for ascending and `-1` for descending**. `column_id` is the column's DPG ID, so
`dpg.get_item_alias(column_id)` recovers the tag it was created with — which is what to key on, since a
`reorderable` table lets the user move columns and any position-based mapping then sorts by the wrong one.

Multi-column sorting only ever arrives if the table asked for it: `sort_multi` defaults to `False`, and
`dpg.get_item_configuration(table)` reports it along with `sortable` and `sort_tristate`.

None of this is discoverable from a DPG traceback — a callback that ignores the `None` case simply
misbehaves in the state the user reaches by clicking one header three times.

## A column's sort flags are settable after creation; whether the header arrow follows is not known

Measured 2026-08-14, headless. `dpg.configure_item` accepts `default_sort`, `prefer_sort_ascending`,
`prefer_sort_descending` and `no_sort` on a column that already exists, and `get_item_configuration` reads
each back changed. So a table's sortability, and its nominal sort column, can be driven from code rather
than only from a header click.

**What was not established is whether ImGui redraws the header's sort arrow to match.** The sort state
proper lives in ImGui's own table state, and `default_sort` is a flag consulted when a table first
establishes its sort specs — so a later change may be inert as far as the drawn arrow is concerned. Seeing
that needs a rendered frame; do not assume either answer.

The reason this matters, and the reason `no_sort` is the useful one: an app that sorts its own data (a
listing shared between a table view and something else, say) can end up with the header's arrow asserting
an order the data no longer has. Making the columns `no_sort=True` and supplying one's own sort control
removes the second source of truth entirely, which is a guarantee rather than a hope — and it has a second
payoff, since ImGui's header sorting has no keyboard operation at all, exactly like its combos.

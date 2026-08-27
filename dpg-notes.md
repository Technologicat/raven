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

**An exception in a callback prints its traceback and is then dropped.** Measured
2026-08-20 on DPG 2.3.1: a `ValueError` raised inside a frame callback wrote a
full Python traceback to stderr, the render loop carried on, and the process
exited 0. Nothing is swallowed, and nothing stops either.

That is the budget a deliberate `raise` in GUI code is working with, and it is
worth knowing which half you are relying on. As a *report* it is good — the
control visibly does nothing and a traceback says why, which beats a warning in
a log nobody is tailing. As a *guard* it is nothing: the next callback runs as
if it had not happened, so a raise cannot protect an invariant the rest of the
app depends on. Reproduce with `set_frame_callback(30, lambda s, a, u: 1 / 0)`
around a mapped viewport.

## Hotkeys dead while the mouse wheel still scrolls means the *callback thread* is blocked

A useful three-way fingerprint, because the two threads above fail separately and the symptoms name which:

| symptom | what is stuck |
|---|---|
| keys dead, wheel scrolls, animations still run | the **callback thread** — something long is running in a callback, and every later key is *queued behind it*, not lost |
| keys dead, wheel dead, nothing animates | the **render loop** |
| both alive but the app does the wrong thing | neither; a state bug |

The asymmetry is not a quirk. A key press reaches Python through the callback queue, so a callback that takes
three seconds delays every key pressed in those three seconds. A **mouse wheel scroll is applied by ImGui
itself, inside the frame** — it never enters the queue, which is the same fact that forces Raven to *poll*
for reader scrolling (`DPGLinearizedChatView.update_jump_to_latest_pill`) rather than hooking an event.

So "the keyboard went dead for a while and the wheel was fine" is not a keyboard bug to chase. Find what ran
long in a callback. (Live case 2026-08-21: switching a chat sibling near the top of a long Librarian chat
rebuilds the view on the callback thread; measured 38 ms, 289 ms and **3038 ms** for three switches, and
during the last of those every hotkey looked dead. The same fault is recorded from the other end in
`raven/vendor/file_dialog/fdialog.py`'s `_forget_listing`, where a rebuild on close left the *opening*
button looking dead afterwards.)

## A callback is passed as many arguments as it declares

DPG fills a callback's parameters positionally from `(sender, app_data, user_data)`, taking **as many as
the signature declares** and no more. `dearpygui.run_callbacks` is that rule written out:

```python
sig = inspect.signature(job[0])
args = []
for arg in range(len(sig.parameters)):
    args.append(job[arg + 1])
job[0](*args)
```

So a zero-argument callback is called with nothing, and a one-argument callback receives the **sender** —
which is what makes the loop-variable idiom unsafe here:

```python
for label, icon in places:
    dpg.add_menu_item(label=label, callback=lambda label=label: chdir(places[label]))  # broken
```

That lambda declares one parameter, so DPG passes the sender into it and the default never applies. The
binding is silently replaced by a widget id. **Pass the value in `user_data` instead**, with a callback
taking all three parameters — a signature wide enough for the full triple cannot be shadowed.

The failure is invisible until the parameter is *read*. Raven's file dialog carried these lambdas for a
sprint while the bound label went unused; the day one of them became a dict key, clicking a shortcut
raised `KeyError: 152`. Nothing about the traceback points at the dispatch rule.

`len(sig.parameters)` counts every kind of parameter, `*args` and keyword-only included — so `lambda *_:`
declares one and receives only the sender, and the arity is not a count of what the function can usefully
absorb.

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

## `set_exit_callback` holds one callback, process-wide — a second call silently replaces the first

Measured 2026-08-21 on DPG 2.3.1: register two, and only the **second** runs. Same shape as
`set_frame_callback`, and the same failure — no error, no warning, just a callback that never fires.

**Which makes it unusable from shared or library code.** Every Raven app registers one for its own teardown
(`_gui_cancel_tasks`, `gui_shutdown`, …), so a widget that registers its own would silently disable the
teardown of whichever app embeds it — a far worse bug than whatever it was trying to fix. The slot belongs
to the application, and a shared widget wanting to know about shutdown needs the app to tell it: either the
app calls the widget's own teardown from its cancel/drain phases, or there is a process-wide Python flag
the widget can read.

Worth stating because reaching for it is a natural move: "let a background thread know the render loop is
stopping" is exactly what an exit callback is for, and it is exactly what a library must not use it for.

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

### The two-party version: a per-frame reader that takes a lock

The same wait, with the callback thread left out, and easier to build by accident because each half looks
correct on its own. A background task holds lock L while doing DPG work that needs frames to finish; the
render loop, once per frame, wants L to read the structure L protects. The render thread then waits for a
lock whose holder is waiting for a frame the render thread was going to produce. Two parties, same circle.

It presents as **an app that never finishes starting** — panels blank, the window manager reporting it not
responding — because the deadlock catches the *first* rebuild. `py-spy dump` names it immediately: MainThread
sits on the `with` line, and another thread holds the GIL somewhere in `dearpygui`.

**A reader that must not block cannot be made safe by blocking it.** Where a per-frame reader needs a list
that background threads mutate, copy it without the lock: `tuple(the_list)` is a single C-level pass that
never releases the GIL, so no other thread can mutate the list part-way through, and the reader gets a
coherent snapshot for free. Being an instant out of date is almost always fine — a frame later it is right
again.

Iterating the live list instead does not crash (the list iterator bounds-checks, so a concurrent `clear`
ends the loop early rather than raising), but it can be read *torn*: half from before a rebuild, half from
after. The copy costs one allocation and removes that.

The tempting third option — publishing a snapshot the writers rebuild after every change — is worse than
both, and worth naming because it looks the most rigorous: every future mutation site has to remember to
republish, and the one that forgets freezes the reader's view silently.

Live case 2026-08-27: `chat_controller.get_current_message`, called from `update_animations`, against
`DPGLinearizedChatView.build` running on the debounced resize task. The lock was added to fix a genuine
race — the same function was iterating a list that `build` clears — so the hazard was real and only the
remedy was wrong.

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

## An autosize window is one frame behind its content — and whether that shows depends on the window's age

**If you came here about a tooltip, the answer is `raven.common.gui.tooltip.Tooltip` and the rest of this
section is why.** It is the packaged form of everything below, and Raven's tooltips that change their text
already use it. Read on if you are building something else that resizes itself, or if you need to know what
the component is protecting you from.

**Except inside a modal, which cannot spawn a window — and being a window is exactly what makes that class
work.** A modal's tooltips stay `dpg.tooltip` and keep the glitch, and there is currently nothing to be done
about that. So the rest of this section is the live reference for anything inside `FileDialog` or a
messagebox, rather than history.

`get_item_rect_size` on an autosize window reports the size auto-fit computed from the content it measured
on the *previous* frame. So after any content change there is one frame where the reported size describes
the content that was there before. Measured 2026-08-20 on DPG 2.3.1, an autosize window holding one text
item, width in pixels:

| frame | `set_value` | hide/show two widgets | hidden window, reshown | window created fresh | explicit `width=` |
|---|---|---|---|---|---|
| +1 | 100 | 100 | 100 | 100 | **371** |
| +2 | 371 | 371 | 371 | 371 | 371 |

**That table is nearly useless on its own, and the trap is worth stating before the finding.** The reported
size is stale in every case except an explicitly sized window, which suggests every case glitches equally.
Photographs of those same frames say otherwise:

| | frame +1, on screen |
|---|---|
| existing window, `set_value` | **drawn, clipped** — the new text in the old window |
| existing window, hidden and reshown | **drawn, clipped** — identically |
| window created fresh, already holding the content | **not drawn at all** |

A window ImGui has not laid out before is measured and *withheld* for that frame, then appears fitted. So a
freshly built window never shows a stale frame, while an existing one always does — and `get_item_rect_size`
cannot tell the two apart, because it reports the same 100 either way. **Measure this with pixels, not with
the geometry API.** (Live case: three conclusions were drawn from the reported sizes and two were wrong,
until Juha objected that a tooltip has never once been seen to glitch on first hover — which the metric
says it should.)

**The two halves catch up on different frames, which is what bites anything that *positions* the window.**
The window is drawn at its new size on +1; `get_item_rect_size` only says so on +2. So code that lays the
window out as soon as it looks right is computing from the size it used to be. Hiding it across the change
does not help — a hidden item is not laid out at all, so the clock has not started.

Where the position is cursor-plus-offset this is invisible, the answer being the same for any size. Near a
viewport edge it is not: whether a window goes below the cursor or flips above it depends on the height, so
the wrong height puts it below, overflowing, for exactly one frame, and something on the next frame moves
it. **A window whose placement depends on its own size therefore needs two settle frames, not one.**
`raven.common.gui.tooltip`'s `_SETTLE_FRAMES` is that count, and
`investigations/dpg-autosize/probe_settle_size.py` prints the frame-by-frame sizes it came from. (Live case:
Librarian's copy-conversation tooltip, whose three-line caption cannot fit below a cursor near the bottom of
the window while the one-line acknowledgment it replaces can.)

**For a tooltip whose caption changes, none of this needs solving again: use
`raven.common.gui.tooltip.Tooltip`.** It is a window rather than a `dpg.tooltip`, so it can be parked
offscreen, settled there for `_SETTLE_FRAMES`, and only then placed — and it keeps DPG's own `(25, 10)`
cursor offset and follows the mouse, so one sitting beside a plain `dpg.tooltip` is indistinguishable from
it. A caption that is written once and never changes is still better off as a `dpg.tooltip`; the class
docstring says so, and says the one thing it cannot do (a modal cannot spawn a window, so a modal's
tooltips keep the glitch).

Two dead ends worth knowing before reaching for them. **Rebuilding the tooltip fixes half of it** — clean
when the caption shrinks, clipped when it grows, so a flash gets one of each. And **a tooltip cannot be
sized out of the problem**: `dpg.configure_item(tooltip, width=...)` raises `width keyword does not exist`.

**And a hidden item is not laid out at all**, keeping whatever metrics it last had — its own width stays at
the *old* text's 37 while hidden, however long it stays hidden. Which is why
`raven.client.avatar_controller.reposition_subtitle` parks the subtitle offscreen at
`(main_window_w, main_window_h)` rather than hiding it: it needs the thing drawn in order to measure it.
It then `split_frame()`s so layout catches up, reads the now-correct size, positions the widget, and
`split_frame()`s again — the standard shape for needing a size before you can place something, subject to
`split_frame` not being callable from the render thread (see *Threading*).

**Raven's own answer, arrived at twice independently, is not to use `dpg.tooltip` for anything whose
contents change.** A tooltip is a window with no title bar, and an app-owned window can be positioned —
which is the whole difference, because it makes the offscreen settle available. `raven.visualizer.annotation`
and the XDot viewer both build theirs that way. The annotation's swap is the full pattern:

```python
dpg.set_item_pos("annotation_tooltip_window", [w, h])  # offscreen, but not hidden -> rendered -> autosize runs
dpg.show_item("annotation_tooltip_window")
guiutils.wait_for_resize("annotation_tooltip_window")  # ...then move it where the user will look
```

It double-buffers the *content* on top of that — build a new group hidden, show it, `split_frame`, delete
the old one — but that part is about the churn of many widgets, not about the size. The no-glitch property
comes from the offscreen settle.

Both windows also pass `min_size=[1, 1]`, without which autosize will not shrink below roughly 100×100 and
a short annotation carries a skirt of empty window — which matters beyond looks, since a DPG window takes
the mouse across its whole rect and the skirt becomes a dead zone.

Failing all of that, give auto-fit nothing to react to: a fixed-size child window as the content, or a
spacer sized to the largest state. Worth naming what that costs — the size stops changing *because it is
always the largest state's size*, so a one-line message sits in a three-line box.

## An offscreen park lasts exactly one frame — ImGui pulls the window back

The standard way to measure a window before placing it is to park it outside the viewport, render, and read
its size (see the autosize section above, and `guiutils.recenter_window`). **That park survives one frame.**
ImGui clamps a window back inside the viewport on every frame whose position did not come through the API,
and only the frame immediately after `set_item_pos` is exempt.

Measured 2026-08-21 on DPG 2.3.1, a 600×400 window parked at the corner of a 1200×800 viewport, reading the
position its content was actually drawn at:

| window | parked once, frames 1–4 | position re-set every frame |
|---|---|---|
| modal, with title bar | offscreen, then **fully on screen** for every frame after | stays offscreen |
| plain window | offscreen, then **19 px shy of the corner** | stays offscreen |
| `no_title_bar` + `no_move` | offscreen, then 19 px shy of the corner | stays offscreen |

`no_move` does not exempt a window, and a *modal* is the worst case by far: ImGui drags it fully into view
rather than leaving a corner, so a multi-frame settle draws the whole window in the middle of the screen.

**So a settle that spans more than one frame must re-park before each frame.**
`helpcard.HelpWindow.settle_offscreen` is the packaged form — set the position, show, wait one frame — and
`fdialog._fit_help_card_to_content` calls it once per measuring pass for this reason.

**How exposed the rest of the tree is, measured rather than assumed.** The rule bites a park only if it
spends a *second* frame, and most do not: `wait_for_resize` returns after **one** frame in the ordinary case
(measured 2026-08-21, logged by the function itself), so the annotation tooltip and `reposition_subtitle` —
one `split_frame` each — are not reached by this. `tooltip.Tooltip` is the one that deliberately spends two,
`_SETTLE_FRAMES` being 2, so its second frame is exposed in principle. In practice nobody has seen a
tooltip glitch, and no probe written here reproduced one; the parks are renewed now because a park that is
correct only by accident of timing is worth two lines to make correct outright, which is a weaker claim
than a bug fixed.

**`get_item_pos` is no use for catching any of this**: it reports the position that was *set*, so it says
`(1500, 1000)` for a window ImGui is drawing at `(247, 360)`. Read the drawn position off a child item's
`rect_min`, or take a screenshot. Held by the `gui`-marked
`test_a_park_has_to_be_renewed_every_frame_to_hold` in `raven/common/gui/tests/test_utils.py`, which
measures the renewed and abandoned parks in the same run so the fixture cannot stop discriminating.

## A hidden root window costs nothing per frame

Measured 2026-08-20 on DPG 2.3.1, with vsync off: 400 buttons alone, 400 with a
`dpg.tooltip` each, and 400 with a hidden root window plus a hover handler each
all render in about 1 ms/frame, and which one comes out fastest changes between
runs. So an app-owned tooltip window per widget is as free as the `dpg.tooltip`
it replaces — which is what makes `raven.common.gui.tooltip.Tooltip` usable in
Librarian's chat view, where 14 buttons per message put several hundred of them
on screen at once.

**Measure this with `vsync=False`.** With vsync on, all three variants report
exactly 16.666 ms and the question goes unanswered while looking answered.
`investigations/dpg-autosize/probe_many_tooltips.py` re-runs it.

## Window z-order

DPG renders windows in creation order. The primary window (set via
`set_primary_window`) is always at the back. Windows created later render on
top. There is no runtime z-order control — `focus_item` brings a window to
front but also steals keyboard focus.

**Implication for tooltips**: create the tooltip window during app
initialization (before the render loop), not lazily during hover. Windows
created mid-render-loop may end up behind earlier windows.

**But a primary window is the exception, and it is the one that matters here.**
Measured 2026-08-20 on DPG 2.3.1: a window created 60 frames into the render
loop draws in front of the primary window, which is what "the primary window is
always at the back" says and what the 2026-04-03 entry below reads as denying.
So a tooltip window built as a chat view rebuilds is fine, as long as the app
sets a primary window — Librarian and Visualizer both do. The rule still binds
between two ordinary windows.
`investigations/dpg-autosize/probe_zorder.py` re-runs it in a few seconds.

## A modal window does not stack over another modal window

`show_item` on a second modal while one is already up does nothing visible. The call succeeds, no error
is raised, and the window simply never appears — `is_item_visible` on it stays `False` for as long as the
first modal is up. Measured 2026-08-17 with two `modal=True` windows, the second shown from the render
loop several seconds after the first, and still absent eight seconds later.

So a modal that wants a modal of its own — a file dialog offering its own help card, say — has to hide
itself first and restore itself when the inner one closes. Whatever the *app* keys on to mean "a picker
is up" must keep answering yes across that gap, or the app un-suppresses its own hotkeys and file drops
exactly while the inner window is on screen.

**And hiding the outer one is not enough: a frame has to be drawn without it first.** `hide_item` takes
effect at the next frame, so a modal shown in the same callback still meets the first one on ImGui's popup
stack and is refused. What happens then is worse than nothing appearing: DPG concludes the second window
is closed and **fires its `on_close` callback**, so a card that hides its owner on show and restores it on
close undoes itself — the dialog comes back, the card was never seen, and the log says only that something
was shown and then hidden 80 ms later. Wait for a frame between the hide and the show
(`guiutils.split_frame`, from a callback thread — never the render thread) and both windows behave.

Measured 2026-08-19, driving `raven-xdot-viewer`'s open dialog with `xdotool`. The tell that it is this and
not a key being delivered twice: the inner window's measured size is `0x0` and `is_item_visible` is still
`False` after a rendered frame, meaning it never drew at all.

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
  - **A container context manager whose anchor no longer exists reports a *stack* error, not an item error.** `with dpg.tooltip(<deleted item>):` fails to push, then still pops on exit, and DPG raises **`[1009] No container to pop`** — from whichever code happened to be building next, which can be a different widget in a different subsystem on a different thread. So the message names neither the dead item nor the place that referenced it. `dpg.add_tooltip(<deleted item>)` raises at the call site instead, which is the whole diagnosis.
    - **But not the error you would expect, and this cost a second investigation.** A dead item is reported two different ways depending on what you did with it. *Operate* on one — `set_value`, `delete_item`, `show_item` — and DPG says **`[1005] Item not found: <id>`**. Hand one to an `add_*` as the **parent**, and it says **`[1011] Parent could not be deduced`**, naming the *new* item and never the parent, so the message cannot say which widget went away. Measured 2026-08-27; pinned by `TestNonexistentOkAndWhatDPGSaysAboutDeadItems` in `raven/common/gui/tests/test_utils.py`.
      - So **`guiutils.nonexistent_ok` does not catch a dead parent by default**: it matches the string "Item not found", which [1011] never contains. Pass `parent_gone_ok=True` where the parent is one the caller was handed and a background rebuild may delete — the streaming chat renderer is the case it was added for.
      - It is opt-in because `[1011]` is *also* what `add_*(parent=0)` with an empty container stack produces, which is an ordinary mistake in the calling code, and the two are indistinguishable from the outside. A guard swallowing it everywhere would silence "you forgot the parent" permanently, and that failure is a widget which simply never appears.
      - Either way the guard now logs what it suppressed, at DEBUG, with the deepest frame **outside** `dearpygui`'s own wrapper — which is the only thing that answers *whose* parent it was. For a `[1011]` the parent is read out of that frame's `parent` local, since DPG's message does not carry it, and described by `guiutils.describe_item`.
- **A deleted item still answers `get_item_alias` with its tag.** DPG frees items lazily, so a description built from the alias reads exactly the same before and after a delete — which is backwards, the usual reason to be describing a widget being that it has just gone. `dpg.does_item_exist` is the only authority, and `guiutils.describe_item` asks it separately and marks the answer `[deleted]`. Measured 2026-08-27.
    - Worth knowing even where the stack is otherwise safe, because it converts a local mistake into a report from somewhere else entirely. It cost an investigation on 2026-08-27: the error surfaced as a *backend* failure in a chat message, because the raising callback ran inside `llmclient.invoke` and the turn's error path materialized it as one.
    - The tempting wrong reading is that concurrent `with` blocks corrupt the stack. **Measured 2026-08-27: they do not** — four threads, a forced GIL yield inside each container block, no errors in any run. The stack is global, but that alone is not what breaks; a dead anchor is.
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

**Single-line is the default** (`add_input_text(multiline=False)`), so the Enter-deactivates behaviour is what an unmarked field does, and the multiline case is the one that needs remembering.

**Which key commits a multiline field is a flag, and the default is the one Raven wants.** `add_input_text(ctrl_enter_for_new_line=...)` selects it; DPG's own docstring is unusually clear here — *"In multi-line mode, unfocus with Enter, add new line with Ctrl+Enter (default is opposite: unfocus with Ctrl+Enter, add line with Enter)"*. It defaults to `False`, so Enter inserts a newline and **Ctrl+Enter commits**, which is what the table below measured and what Librarian's composer relies on. Worth knowing it is switchable rather than inherent: an app wanting Enter-to-send in a multiline box sets the flag instead of intercepting the key.

**Every effective `focus_item` on a text field selects its whole contents**, so the next character typed replaces the text rather than extending it. Measured 2026-08-18: it happens when focus arrives from elsewhere *and* when the field already had it, so this is not an on-arrival effect that a caller can dodge by checking first. DPG exposes no caret or selection API to undo it. `auto_select_all` is a different thing — an `add_input_text` flag governing *mouse* focus — and leaving it `False` does not prevent this.

**A focus change is not instantaneous, and something else holds focus while it is in flight** — which turns the paragraph above into a visible artifact. Measured 2026-08-18 in `FileDialog`: on every Tab, focus lands on the *path* field, **active**, for 25–100 ms before settling on the intended target. Active means the caret, which means select-all, which paints the field's text blue. It reads as a one-or-two-frame flash in a field nobody touched.

Finding it took three failed reproductions and one that worked, and the difference is worth copying. A headless probe of the same widget shapes never showed it, because the artifact needs a *queued* focus change to be in flight; adding the listing rebuild did not reproduce it either. What settled it was instrumenting the real app — logging `is_item_focused` / `is_item_active` for the specific widgets from the grid's tick thread, which runs every frame — plus `ffmpeg -f x11grab -framerate 60` to record the window and a per-frame scan for blue pixels. Screenshots cannot do this: `import` costs 50–200 ms a frame and samples perhaps one frame in five. 18 frames out of 360 carried the flash.

**The mechanism is not established, and the obvious explanation is ruled out.** The tempting reading is the fallback above — nav landing on the container's first navigable item — with image buttons skipped so that a text field is first. Measured: they are not skipped. Auto-focus lands on whichever item is first, image button or plain, and it lands there *focused but not active*, where the flash shows the path field **active**. So whatever activates it is something else, and it is not the two image buttons ahead of it being invisible to nav.

Recorded as an open question rather than a theory, because three plausible mechanisms have already been falsified here and a fourth written down as fact would be worth less than nothing.

The practical consequence is that a programmatic focus is never neutral: returning the caret to a field the user was editing arms a replacement of what they typed. `FileDialog` accepts that on Tab-back and tells the user to press End first. It also explains a puzzling intermittency — whether the selection happens tracks whether the focus request *succeeded*, so a refused window-to-child request (see above) leaves the text alone and looks like inconsistent behaviour from the same key.

**Enter de-activates without de-focusing**, which is worth separating because the two are easy to run together. Re-measured 2026-08-18 on DPG 2.3.1: after Enter a single-line field reports `focused=True, active=False`, and the text it held survives. Focus does not move; only the edit ends.

That has a use beyond gating. Since a write lands on an *inactive* field and is reverted on an active one, **Enter is itself a licence to write the field** — no focus dance required, because the commit already released it. `FileDialog.chdir` relies on exactly this: Enter on a directory clears the find field on the way in, and the `set_value` sticks precisely because Enter had deactivated the field a moment earlier. On a multiline field it would not: Enter inserts a newline there and leaves it active, so the same write would be reverted on the next frame.

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

**The parking spot is the half you control; the other half is where the user can click, and it bites harder.** A click focuses whatever it lands on, so any interactive control sitting *directly in the window* — a combo, a checkbox, a button that does not close the dialog — is a one-click trap: from that moment `focus_item` cannot reach anything inside a child window, and every hotkey that returns the caret fires and arrives nowhere. Nothing reports it. The cure is to put such controls inside a child window of their own, borderless and unpadded, which costs no screen space and is worth doing for grouping anyway; a click then lands child-side and child→child works. (Live case 2026-08-19: `FileDialog`'s type-filter combo sat at window level. Clicking it killed Ctrl+F, Tab-back and Escape-to-the-field for the rest of the dialog's life. Wrapping the `Show` label and the combo in one child window fixed it, and the row is pixel-identical before and after.)

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

**A global handler sees Enter before the field's own `on_enter` callback does, and sees the field already
deactivated.** Measured 2026-08-20 on DPG 2.3.1, both fired within the same frame: the key-press handler
runs first, reporting `is_item_active` False and `is_item_focused` True, and the field's callback follows.
So a dialog that binds Enter globally *and* has an `on_enter=True` field will run both on one press, in
that order — and `is_item_active` cannot tell it that the Enter belonged to the field, because ImGui has
already cleared the active id by then. Track which control the caret is in and dispatch on that; the file
dialog's `CaretHome` is the worked example, and `test_fdialog.py` pins the case in
`test_enter_in_the_path_field_does_not_also_open_what_the_cursor_is_on`.

**This bites only where both exist, and two of the three Raven fields deliberately have one.**
`raven-visualizer`'s search field takes a plain per-keystroke `callback` and leaves Enter entirely to the
global handler, gated on `is_item_focused`. `raven-librarian`'s composer does the opposite — the commit is
wired at the widget with no global branch at all, because a *multiline* field is unfocused as well as
deactivated by its commit chord, so no gate at that level can catch it. The file dialog's path field is the
one place a field's own Enter and a global Enter both fire, and it is there because the dialog binds Enter
for the listing behind the field.

**`on_enter=True` buys Enter and costs every other keystroke**, the field's `callback` then firing only on
the commit. An `add_item_edited_handler` on the same field still fires per keystroke — one frame after the
key, carrying the new value — so a field that has to react as it is typed into (recolouring, validating)
can have both. Measured the same day; the probe is `investigations/dpg-input-text/probe_input_text_enter.py`.

Alt is the modifier that varies by desktop: nothing intercepted Alt+Up under Cinnamon, but window managers
commonly bind Alt chords, and that is a statement about the desktop rather than about DPG. Both dev
machines here run Cinnamon, so this one cannot be settled in-house — it is for users on other desktops to
report. Which is the argument for Ctrl+Up existing as an alias regardless of what Alt does.

## A punctuation `mvKey_*` is a US-layout assumption, and it fails silently elsewhere

DPG reports **keycodes, not characters** — `add_key_press_handler` and its siblings are the whole input API,
and there is no typed-character handler anywhere in it (checked 2026-08-21; `add_char_remap` is font-atlas
machinery). Letters are safe: a letter key yields the same code shifted or not, on every Latin layout. **A
punctuation constant is not.**

Measured on a Finnish (`fi`, pc105) layout, DearPyGui 2.3.1:

| pressed | reported | note |
|---|---|---|
| `/` — which is **Shift+7** here | `mvKey_7` **+ Shift** | `mvKey_Slash` (600) never fires at all |
| `+` — the key right of `0` | **`mvKey_Minus` (598)** | *not* `mvKey_Plus`, which is 61 — a stale pre-2.0 code |
| `,` and `.` | `mvKey_Comma`, `mvKey_Period` | these do agree with US |
| numpad `/`, `+`, `-` | `mvKey_Divide`, `mvKey_Add`, `mvKey_Subtract` | separate codes, layout-independent |

**So a hotkey on `mvKey_Plus` is dead on this keyboard, and worse than dead where a neighbouring constant
catches the code instead.** `raven-cherrypick` and `raven-xdot-viewer` both zoom in on
`(mvKey_Plus, mvKey_Add)` and out on `(mvKey_Minus, mvKey_Subtract)`. **Both main-row keys report 598**,
which *is* `mvKey_Minus` — so on this layout `+` and `-` alike zoom **out**, and the main keyboard has no
zoom-in at all. The numpad is unaffected and is the only *keyboard* way in; the mouse zooms as it always
did, which together with the numpad is how this survived unreported.

Confirmed twice by different routes, which matters because the first route was suspect: the synthetic
measurement was open to being an `xdotool` artifact (it may bind an unmapped keysym to a scratch keycode),
and Juha then reported the same behaviour from ordinary use of both apps. Two physical keys really do arrive
as one code here.

Two consequences worth carrying:

- **Prefer letters, function keys and the named editing keys for hotkeys.** They are the layout-stable set.
  Punctuation needs either a numpad alias or a deliberate decision to be US-only.
- **Anything matching *text* from key events cannot be done with punctuation.** A type-ahead over a list, for
  instance, matches letter keys only — and that restriction is also what leaves Shift free to mean a
  direction, since Shift is never needed to produce a letter.

Not established: *why* the mapping lands where it does. "The constant follows the physical US position" fits
`+` → `mvKey_Minus` and fails for `-`, which by that argument should report `mvKey_Slash` and does not — and
both are now confirmed from real keypresses, so neither is a synthesis artifact. (One candidate worth
checking before theorising further: ImGui's key enum has `Minus` and `Equal` but **no `Plus`**, so a `+` key
may have nowhere of its own to land.) The table above is measured and repeatable; the story behind it is not
settled, and a hotkey design should lean on the table rather than on any account of it.

## Tab reaches a global handler and still moves ImGui's nav, after a programmatic focus

An `InputText` never inserts a Tab, which makes it easy to conclude that Tab is the app's to define. It is,
but ImGui is not finished with it: when the field holds a caret it got from `dpg.focus_item` rather than
from a click, Tab **also** moves keyboard-navigation focus to the next item. That item then reports itself
**activated**, and **deactivated** again a frame later once whatever the app's own handler focused has
landed.

So an app that keeps its own "where do the keys go" state, and updates it from `activated` / `deactivated`
handlers, gets that state written twice by a Tab it thought only its own handler had seen — and the writes
straddle its own, so its decision is the one that loses. The failure is silent and reads as a dead key: the
Tab appears to do nothing, the arrow keys stop working, and pressing Tab again fails the same way, because
each press is undone before the next arrives.

Two things follow, and the second is the general one:

- **A handler for `deactivated` must put the state back where it *was*, not where it assumes.** Restoring a
  remembered previous value is stable under a spurious activate/deactivate pair; naming a fixed destination
  is not.
- **The click and the `focus_item` paths into a text field are not the same state.** A mouse click leaves
  ImGui with no nav position, so Tab moves nothing; `focus_item` sets one, so Tab moves from it. Anything
  measured about Tab after clicking into a field says nothing about Tab after Ctrl+F.

Measured 2026-08-21 on DearPyGui 2.3.1, from `FileDialog`: Ctrl+B into the places panel, Enter to go there
(which focuses the find field via `chdir`), then Tab. Reproduced identically from Ctrl+F, so it is the
programmatic focus and not the route that caused it. Traced by logging every write to the state together
with a stack and a thread name — worth reaching for early here, since the two writes come from DPG's own
handlers and appear nowhere in the app's key path.

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

**The other half of driving a live app is in `CLAUDE.md` → "Live GUI testing on a shared desktop"** —
stealing and restoring focus, keeping the whole sequence in one command, finding the window, closing the
app afterwards. The two halves are one activity split across two documents by topic, so each points at the
other: arriving at the etiquette without the fidelity traps is how they get rediscovered.

Related, and visible in any such log: **held modifier keys arrive as repeated key presses** (~50 ms
apart, LControl / LShift / LAlt alike), alongside a companion pseudo-key — 663 for Ctrl, 664 for Shift,
665 for Alt — that no `mvKey_*` constant names. A handler that acts on a bare modifier keycode therefore
fires over and over while the key is down.

**And a synthetic tap is far shorter than a human press, which hides anything that depends on how long a
key is held.** `xdotool key Escape` holds it about 12 ms; a finger holds it for a hundred and something.
Where the app does something *while* the key is down — or where ImGui does, as it does on Escape,
dismissing the topmost modal popup by itself — a driven test passes and a real press fails, which is the
worst direction for a check to be wrong in.

Drive such keys as `keydown` / `sleep` / `keyup`, and **pick the sleep against the machine's keyboard
repeat delay** (250 ms on both dev machines here, and a per-machine setting rather than a constant to hard-code):
below it for *one press*, above it to additionally exercise auto-repeat. The two are different tests — a
handler that fires twice is not the same fault as one that runs while a key is down — so a 600 ms hold
that reproduces a bug has not said which of them it found.

(Live case 2026-08-19: Escape over `FileDialog`'s help card. Tapped, the card closed and the dialog
returned; held, the dialog was put back under the still-down key and ImGui dismissed it, so the picker
cancelled itself.)

## Investigation history

- 2026-08-21: `FileDialog`'s Tab went dead after arriving anywhere from the places panel — the caret left
  the find field, the arrow keys did nothing, and Tab could not get back. Direct calls to the key handler
  could not reproduce it, headless or in a live app, which is what said the fault was between the X key and
  the handler rather than in the handler. Logging every write to the "where do the keys go" state, with a
  stack and a thread name, showed two writes arriving from DPG's own item handlers on either side of the
  one Tab made — producing the section above. The 2026-08-17 harness traps both fired again on the way:
  `xdotool key ctrl+b` lost its modifier and had to be sent as `keydown`/`key`/`keyup`, and the 663
  pseudo-key filled the log. Both were already written down here, and reading this section first would have
  saved a run.
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

## A probe that measures anything must load the app's fonts first

`guiutils.bootup(font_size=...)` — or at least `setup_default_font(...)` — before `create_viewport`. A probe that goes straight from `setup_dearpygui` to building widgets draws in ImGui's built-in font, which is both narrower and shorter than OpenSans, so **every pixel it reports is a measurement of an app nobody runs**. Sizing the help card that way reported 190 px of slack in a card that in fact had none to spare.

The built-in font also has no glyph outside ASCII, so any text carrying an em-dash comes back as `?`. That reads as a bug in the app, and was investigated as one before the font was suspected — both failures from the same missing call, on 2026-08-21.

The `gui`-marked tests do the same thing one call lower: `guiutils.setup_default_font(20)` at the top, `dpg.bind_font(0)` in the `finally`, since 20 is what every app in the constellation uses and the module's other tests are not measuring.

## Context recreation is not reliably safe once real widgets have rendered

The cache paragraph above says bare cycles are fine, and they are — including 60 rendered frames on a shown viewport, clean over 8 trials. **That result does not extend to a cycle with an application's widgets in it**, which is the shape a benchmark reaches for when comparing two configurations in one process.

Measured 2026-08-13, two contexts per process, a `FileDialog` built in each, 8 trials per configuration: **3/8 to 8/8 of runs died with `SIGSEGV`** on the second context. Nondeterministic, and not monotonic in anything tried — one configuration crashed 5/8 in one script and 0/8 in another that differed only in leaving vsync alone. Bisecting by dropping one ingredient at a time is therefore useless at this sample size, and the first attempt produced a table that read as if *removing* ingredients caused the crash.

**The mechanism is not identified.** Waiting half a second before `destroy_context`, and calling `stop_dearpygui` first, both changed nothing — but only in a configuration that was not crashing anyway, so neither is evidence. Candidates not ruled out: work still queued on the callback thread, pending texture uploads, driver-side teardown.

Consequences, which are small:

- **An app never meets this**, holding one context for its whole life.
- **The default test suite does not either**, using one module-scoped context per module and never rendering a frame (see the ceiling above).
- **The `--run-gui` group used to do this cycle, and died of it for eleven days.** `test_focus_semantics.py`'s `mapped_viewport` fixture was *function*-scoped: a context created, shown, rendered and destroyed once per test, eight times over. Any further module that mapped a context and rendered frames segfaulted the group — measured 2026-08-13 at 3/3 with a table in it, 1/3 without, and *only* when `test_focus_semantics` ran first, which alphabetical collection order decides. From **2026-08-21** `pytest -m "not ml" --run-gui` segfaulted outright, in `ImGui_ImplGlfw_WindowFocusCallback` reached from GLFW's `processEvent` — a focus event delivered to a backend whose context is gone.
  - **Fixed 2026-08-24 by removing the cycle rather than reordering it**: `mapped_gui_context` in the root `conftest.py` is one session-scoped mapped context, and every `gui` test that just needs a window to render into takes it. They sort last, so it comes up as late as possible and is never destroyed before the run ends. Tests sharing it use per-test tags and delete their widgets, a duplicate widget ID being fatal rather than an exception.
  - **What the cycle costs is not ordering-dependent, which is why there is no position for a test that owns its contexts.** `test_filedrop.py`'s two lifecycle tests create, show and destroy contexts because that is what they assert. Run *before* the rest they take down the next context created anywhere in the process — including unmapped ones, `test_fontsetup` dying inside `setup_dearpygui`. Run *after*, they die in their own `setup_dearpygui`, the shared context still being up. So `conftest.py` skips them whenever a shared-context test is also collected, with a skip reason naming the command that runs them alone. Both states are green: the group at 2811 passed / 2 skipped, and `pytest --run-gui raven/common/gui/tests/test_filedrop.py` at 18 passed.
  - The same shapes do **not** crash outside pytest: five focus-like cycles then a table-building cycle, six table cycles, twelve plain cycles, all clean over three runs each. So something about the pytest process is part of it, and an earlier guess here — that a *heavier subsystem* would be what tipped it — was wrong; 400 plain buttons did it too.
- **A benchmark or probe must use one process per context.** Run configurations as subprocesses and compare their printed output. This is cheap, and it is the only reason the constraint matters at all.

## Introspection gaps to expect

`dpg.get_item_configuration(item)["color"]` reports color as **normalized floats** while `dpg.configure_item(item, color=...)` takes **0–255**, so a read-modify-write round trip has to scale. An unset color reads back as the sentinel `[-1.0, 0.0, 0.0, 1.0]`, and writing that sentinel back (scaled) correctly restores "unset".

`dpg.get_item_theme(item)` returns `None` for an unbound widget, and `dpg.bind_item_theme(item, None)` unbinds — so capture-and-restore of a theme is symmetric with no special case. (`0` also unbinds; prefer `None`.)

**An unshown viewport does not have the client size it was created with.** Measured
2026-08-20 on DPG 2.3.1: after `create_viewport(width=400, height=300)` and
`setup_dearpygui()`, `get_viewport_width()`/`get_viewport_height()` report `400`/`300`
as asked, while `get_viewport_client_width()`/`_client_height()` report **1280×800** —
DPG's built-in default, untouched until `show_viewport`. `get_viewport_configuration`
shows both pairs side by side, which is the quickest way to see it.

This matters in the default test suite, which never shows a viewport. Anything placing
a widget relative to the viewport — a tooltip, an overlay, a centred dialog — is
computed against 1280×800 there, so a test that reasons from the numbers it passed to
`create_viewport` is reasoning about the wrong rectangle. **Ask for the client size
rather than assuming it.** The failure is quiet: the position is merely *elsewhere*, and
a fixture placed near an edge by accident can make two different placement rules agree,
which is a test that passes against the behaviour it was written to reject.

**A `mvTable` has no `rect_size` in its item state, so `guiutils.get_widget_size` answers with its *configured* size — which for an autosizing table is `-1`.** That is a faithful answer to a different question: `-1` is the layout directive "fill the available space", and the helper is reporting what the item was told, not what it became. The caller asked for pixels and got a directive, and nothing says so — a guard like `if not height` does not catch it, `-1` being truthy, and arithmetic against "the view's height" then computes against −1. Measured 2026-08-17 while giving the file dialog's table a keyboard cursor, where it scrolled the listing on the third arrow keypress.

Note the fallback is not what rescues the child window beside it: `listing_area` is *also* configured `(-1, -1)`, and reports a real size because `get_item_rect_size` succeeds for it. The fallback runs only for items that have no `rect_size` at all — which is precisely where a configured `-1` cannot be resolved into anything. **So treat a negative from `get_widget_size` as "unknown", not as a size.**

Measure the **enclosing child window** instead — it reports a real size — and keep scrolling the table, which is what `y_scroll` and `max_y_scroll` work on. Both halves were measured on the same widget pair: table `get_widget_size=(-1, -1)` with `max_y_scroll=588.0`, its `listing_area` child window `(1159, 581)` with `max_y_scroll=0.0`.

**Row pitch is not the height you asked for, either.** Cells created with `height=16` came out 18 px tall at a 22 px pitch, below a header occupying the first 26 px. So a row's position is worth reading off the row (`get_item_pos` on a cell, which answers in content coordinates, exactly what `set_y_scroll` wants) rather than computed from the height that was requested.

**`dpg.get_focused_item` answers with an *alias* for an item that has one, and with an ID for one that does not.** Measured 2026-08-21: a combo created with `tag="filter_combo"` came back as the string `'filter_combo'`, while the window above it came back as a bare `71`. Same convention as `add_*` returning the tag it was given, but here it is a *return value about somebody else's widget*, so the caller has no say in which form arrives.

**So an identity test against one name is right for some widgets and silently wrong for the rest**, and this failure is worse than most: never matching is indistinguishable from nothing ever being focused, so the feature does nothing, reports nothing, and looks like a key that is not being delivered. (Live case: `keyboardmark.install_focus_follower` keyed its marks by ID and lit none of them; the app's combo browsing worked throughout, which made it look like a drawing problem.) Compare against both names — `guiutils.item_identifiers` returns the set.

The apps that route arrow keys to a combo normalize this by hand instead, with `dpg.get_item_alias(dpg.get_focused_item())` (`raven-xdot-viewer`, and both avatar editors). That works while every browsable combo carries a tag, and stops working the moment one does not: `get_item_alias` answers `""` for an untagged item, which then matches nothing — or, worse, matches another untagged one.

**A theme's contents are readable after all — by walking the item tree, since a theme is made of items like everything else.** Measured 2026-08-21: `dpg.get_item_theme(widget)` gives the theme, `dpg.get_item_children(theme, slot=1)` its theme components, and their slot-1 children are the individual color and style items. Each of those reports **which** slot or style variable it sets as `dpg.get_item_configuration(item)["target"]` — an `mvThemeCol_*` / `mvStyleVar_*` constant — and **what** it sets it to via `dpg.get_value(item)`: a 4-element RGBA in 0–255 for a color, a 2-element vector for a style (the second component is `-1` for a scalar style var).

`raven/common/gui/tests/test_keyboardmark.py` reads a mark's theme this way rather than the component's private attributes, which is the shape to copy.

What is still missing is a getter for the **resolved** value — "what color will this widget actually draw with" — which would have to walk the parent chain and fall back to DPG's built-in default theme, and there is no getter for that default. So code restoring a themed value still tends to hardcode a measured literal; see the audit item in `TODO_DEFERRED.md`, whose premise is narrower than it was written. Where a *per-widget* getter exists (as for a text widget's own color), it remains the direct answer.

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

**`rect_min` is where the item was *drawn*, so it lags a `set_item_pos` by a frame while `get_item_pos`
does not.** Measured 2026-08-21: a window moved from `(100, 100)` to `(700, 300)` reports the new position
from `get_item_pos` immediately and from its child group's `rect_min` only after the next frame; the
group's own `get_item_pos` stays `(8, 34)` throughout, being parent-relative and so indifferent to where
the window is.

**The trap is a subtraction whose two halves come from different frames.** "How far below the window's top
does the content start" reads naturally as `get_item_rect_min(child) - get_item_pos(window)`, and that is
one number from the last frame drawn and one from now — correct only while nothing has moved. Where the
question is an *offset within* a container, ask the child's own `get_item_pos`, which answers it directly
and cannot go stale. (Live case: `helpcard.HelpWindow.measure_content_height`, called from `on_show` —
which `show` reaches immediately after repositioning the window, so the two halves were always a frame
apart.)

**A group reports the full extent of its content, whether or not the window clips it.** Same session, a
1221×606 group in a window forced to 400, 640 and 900 px tall: `get_item_rect_size` said 606 every time.
So "how tall does my content want to be" is answerable from a window that is currently too short for it —
which is what lets a measurement *grow* a fixed-size window and not only shrink one.

## A tooltip's position is not readable, and its offset from the cursor is (25, 10)

Nothing DPG exposes reports where a tooltip window actually is. `get_item_rect_min` raises
`KeyError: 'rect_min'` (windows have none, as the previous section covers), `get_item_pos` returns
`(0, 0)`, and `guiutils.get_widget_pos` inherits that, since it accumulates `get_item_pos` up a parent
chain a tooltip does not have. So a question as ordinary as "where does DPG put a tooltip" has to be
answered from pixels.

Measured 2026-08-20 on DPG 2.3.1, by screenshotting the same hovered button with the cursor at two
positions and diffing — which cancels the button's hover highlight and leaves only the tooltip:

**A tooltip's top-left sits at the cursor plus (25, 10)**, identical at both positions. Note it is not
square; DPG offsets further horizontally than vertically, presumably to clear the mouse pointer glyph,
which is taller than it is wide. `raven.common.gui.tooltip.Tooltip` defaults to this pair so that a
tooltip migrated to it lands where the plain `dpg.tooltip` beside it would.

`investigations/dpg-autosize/probe_tooltip_offset.py` re-measures it, which a DPG upgrade is reason to do.

# Themes

## A theme bound to a container reaches its children, and a child's own theme still applies

Measured 2026-08-21. Binding a theme to a group, and to a child window, applies it to everything inside —
and where a descendant carries a theme of its own, the two **compose per property** rather than the inner
one replacing the outer. A button wearing a red-background theme, inside a group bound to a blue-border
theme, comes out red with a blue border; a combo wearing a green-text theme, likewise.

That is what makes an opt-in highlight possible at all. DPG binds **one theme per item**, so a component
that marked a widget by binding a theme to it would silently drop whatever theme the widget already had —
and there is no getter for a theme's contents (see "Introspection gaps to expect"), so it could not merge
the two either. Marking the *enclosing group* sidesteps the whole problem: the mark supplies its property,
the widget's own theme keeps supplying the rest, and unbinding restores exactly what was there.

`raven.common.gui.keyboardmark.Mark` is built on this, and the consequence to know at a call site is that
marking a container marks **every** matching descendant. For a row of buttons that is the intent; for a
panel with buttons in it, choose the style variable that only the panel answers to (below).

## A border is drawn inside the item rect, so turning one on moves nothing

Same session. `mvThemeCol_Border` plus a border-size style gives a widget a visible outline:

| style var | what it borders |
|---|---|
| `mvStyleVar_FrameBorderSize` | framed widgets — buttons, combos, input fields |
| `mvStyleVar_ChildBorderSize` | child windows |

Switching either from 0 to 2 left every tracked widget's `rect_min` **and** `rect_size` unchanged, across a
2×2 button group, an 11-button row, and the text following each. So a border can be turned on and off as an
indicator without the layout jumping, which is the property that decides whether this is usable as a mark
at all.

The two are separate style vars rather than one because the distinction is load-bearing: a theme carrying
`FrameBorderSize` bound to a *panel* would border every button inside it, and one carrying
`ChildBorderSize` bound to a *group of buttons* would do nothing.

# Drawlists

## A drawlist's children are in slot 2, and slot 1 answers "nothing was drawn"

Measured 2026-08-25, checking whether `XDotWidget.set_graph` renders a hand-built graph.
`dpg.get_item_children(drawlist, 1)` returns an empty list for a drawlist holding fifteen draw items;
`slot=2` returns all fifteen. Slots 0 and 3 are empty too.

**The trap is that the wrong slot is indistinguishable from a correct negative.** An empty list reads as
"the renderer emitted nothing", which is a plausible finding rather than an obvious error — and one worth
acting on, since it would say a public entry point does not work. It cost a round of diagnosis here, aimed
at viewport culling and font availability, before the slots were simply enumerated.

So when a count of drawn items comes back zero, **enumerate the slots before believing it**:

```python
for slot in range(4):
    print(slot, len(dpg.get_item_children(item, slot) or []))
```

`raven/common/gui/xdotwidget/tests/test_widget.py` names the constant rather than passing a literal, for
the same reason.

**The full mapping is upstream's**, at
[Container Slots](https://dearpygui.readthedocs.io/en/latest/documentation/container-slots.html) — read
2026-08-25 against DPG 2.3.1. That URL is the `latest` build and therefore a moving target: if the table
below ever disagrees with the page, the page wins and this one wants re-reading.

| slot | holds |
|---|---|
| 0 | `mvFileExtension`, `mvFontRangeHint`, `mvNodeLink`, `mvAnnotation`, `mvDragLine`, `mvDragRect`, `mvDragPoint`, `mvLegend`, `mvTableColumn` |
| 1 | most items |
| 2 | draw items |
| 3 | `mvDragPayload` |

So slot 1 is the right guess for almost everything, which is exactly why the drawlist case is worth
knowing: it is one of the few places the reflex is wrong, and being wrong there is silent. Slot 0 is a
collection of special cases rather than a category — table columns and font range hints share it.

## A drawlist ignores `pos`, and reports back the position it was asked for

Measured 2026-08-21, while looking for a way to draw a mark around an arbitrary widget. `dpg.add_drawlist`
accepts `pos=`, and a drawlist created with one **is laid out in the normal flow anyway** — taking its
place in the sequence and displacing everything after it by its full height. True in a window and in a
child window alike, and `dpg.set_item_pos` afterwards changes nothing either.

**The trap is that asking looks like confirmation.** `get_item_pos` returns the `pos` that was passed,
whatever the item did with it, so the only witness is `get_item_rect_min` — which reports where the item
was actually drawn. A drawlist created with `pos=(4, 46)` at the top of a child window's content reported
`get_item_pos=[4, 46]` and `rect_min=[16, 16]`, and pushed the button below it down by 68 px. An ordinary
item in the same probe honoured its `pos` exactly.

So there is no "float a drawn overlay inside this window" — a drawn shape either goes in a drawlist that is
already there (`thumbnailgrid` draws its cursor into the grid's own canvas), or in a floating overlay
*window*, which is opaque to the mouse across its whole rect and stacks by window z-order rather than
clipping to a panel (`investigations/dpg-overlays/`). Where the thing to be drawn is a highlight, prefer a
theme — see "Themes" above, which is the mechanism `keyboardmark` ended up using for exactly this reason.

## There is no vertical separator, and `add_separator` inside a horizontal group is not one

Measured 2026-08-26, dividing a row of checkboxes into groups. DPG offers `add_separator` and nothing else —
no `add_vseparator`, no orientation argument — and putting one inside a `dpg.group(horizontal=True)` does
**not** turn it on its side. It draws its horizontal rule anyway, inside the row, and forces the row's
height, squeezing everything beside it.

The answer is **`raven.common.gui.utils.add_toolbar_separator`**, which draws the line into a drawlist of
the orientation you ask for. `horizontal=True` means *a horizontal toolbar*, so the line it draws is
vertical — the argument names the toolbar, not the line. `toolbar_extent` is the row's cross-axis size (its
height, here), `size` the gap's width, and `line=False` gives spacing with no rule, which is what every call
site in Librarian's bottom toolbar and Visualizer's side toolbar actually uses.

Worth knowing before reaching for `add_separator`: this helper predates the question by a long way and
answers it in both orientations, so a row divider is a call rather than a small drawing job.

## Needing `get_item_pos` to place a decoration means it is in the wrong container

The tell is a decoration that has to be told where its text is — a bullet, a rule, a background box. That
requirement never comes from the *drawing*; it comes from having put the decoration somewhere the text is
not, so that nothing lays the two out together. Everything unpleasant then follows from it: the position
only becomes real after a frame, so the work has to be deferred; it is (0, 0) inside a container that has
never been shown, so the deferral has to wait on *visibility* rather than on a frame count; and the answer
goes stale the moment anything above the text changes height, stranding the decoration beside whatever
moved into its place.

Put it in the same row as its text instead, as an ordinary child, and every one of those disappears at
once — DPG lays it out, later, along with everything else. Nobody computes a coordinate, so no coordinate
can be early, absent or stale.

**The distinction the wrong version blurs is metrics versus positions.** Font metrics — `get_text_size` for
a cell's width, a line's height — are available at build time with no frame rendered, and reserving space
from them is fine. A laid-out *position* is not available then, and wanting one is the signal to change
containers rather than to wait harder.

Worked example, 2026-08-26: `raven/vendor/DearPyGui_Markdown/line_attributes.py`. List markers and
blockquote bars each reserved their width with a spacer in the flow and then drew the mark itself into a
separate `attributes_group` at `dpg.get_item_pos(...)` coordinates. Expanding a chat message's thinking
trace pushed the text down and left a column of orphaned numbers and bars in the margin. Both now fill a
slot in their own row; the `CallInNextFrame` deferral and a `_run_when_laid_out` helper (89 lines, one
item-visible handler registry per marker) were deleted rather than fixed, because nothing was left for them
to wait for.

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

## A table's column widths take a second frame to settle, and text wrapping follows them

Measured 2026-08-21, sizing the file dialog's help card to its content. The card is a two-column table of
hotkeys in a 1250 px window; several of its cells hold text with `wrap=0`, which wraps at the column edge.
On the **first** frame the table drew, the content group measured 584 px; from the frame after, 606 — a
wrapped line's worth. Column widths are computed from what was submitted, and which cells need two lines
follows from the widths, so the first frame is measuring a layout still on its way somewhere.

So **a measurement of anything containing a table needs at least two rendered frames**, and the honest form
is to re-ask until the answer stops moving rather than to pick a frame count. `fdialog`'s
`_fit_help_card_to_content` is the worked example: split a frame, measure, apply, repeat, stop when the
measurement equals what the window already is.

The failure it prevents is quiet and one-directional: a card sized from the first frame comes out *short*,
and a fixed-size DPG window with `no_scrollbar=True` clips the overflow away without a mark.

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

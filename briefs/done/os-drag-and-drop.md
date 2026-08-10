# OS file drag-and-drop across the constellation

**Built and shipped 2026-08-10**, in one session, into all six GUI apps. Written afterwards: this never had a
brief, going straight from a `TODO_DEFERRED.md` item to code once a probe showed what it would cost. So the
*decisions* existed only in a conversation and a git log, which is what this document is for. The
*measurements* live in `investigations/dpg-dnd/` and are not repeated here.

## Why it was built, and what it is not for

**It is tooling for the builders, not a feature for end users** — ranked as a power multiplier alongside brief
15, and against feature work rather than beneath it. Two arguments, and the exhibit is not one of them: a
Researchers' Night visitor in a single sitting is not dragging files out of a file manager.

- **The 2026-08-07 probe collapsed the estimate.** The platform work was already inside the GLFW that DPG
  links, so this turned out to be wiring rather than building. Before that probe it looked like writing a
  shim per platform, which is why it had sat in the deferred file since 2026-07-17.
- **It is where our own gesture volume is**: feeding corpora in, attaching a file to check a render, driving
  a GUI by hand — dozens of times a day, through a `FileDialog` that was the sole entry path.
  - **The part that only showed up in use** (Juha, the day it shipped): a file manager *holds its place
    across app restarts*; a file dialog does not. So the saving is not one gesture per open, it is the whole
    navigation, on every restart — which is the loop when checking one change twenty times. This also settles
    that it does **not** make the deferred `FileDialog` improvements redundant: the dialog is still the path
    whenever the file is not already on screen.

## What it is

`raven/common/gui/filedrop.py`. DPG has no OS drag-and-drop — its drag-and-drop is ImGui-internal, between
widgets — but DPG statically links GLFW, GLFW has had `glfwSetDropCallback` since 3.1, and the symbols are
exported from the C extension, so `ctypes` reaches them. No per-platform code: GLFW's X11, Cocoa and Win32
backends each implement the drop, and we bind to whichever was compiled in.

Two constraints shaped the API, both measured rather than reasoned:

- **`install` must run on the render thread, after `dpg.show_viewport()`.** That call is what makes DPG's
  window the calling thread's current GLFW context — NULL before it, and NULL on every other thread. No
  rendered frame is needed, which is what makes `show_viewport()` a uniform install site for all six apps
  rather than "somewhere in the render loop". Pinned by a `gui`-marked test, because a DPG upgrade moving
  that point would break drops everywhere at once, silently.
- **Handlers must not run where GLFW delivers them.** The drop arrives from `glfwPollEvents()` inside
  `render_dearpygui_frame()`, i.e. on the render thread, where nothing may wait for a frame — so an error
  dialog written straight into the handler would deadlock. The C callback therefore only copies the paths and
  queues them; a worker runs handlers, which may then show dialogs freely.

Handlers receive `list[str]` of absolute paths, matching what `FileDialog` sends its callback, so an app can
route a drop into the callback it already has.

## The router, and the decisions inside it

`make_router` covers the shape all six apps need, because all six were otherwise going to write it.

- **First matching rule wins.** Load-bearing rather than incidental: it is how overlapping predicates express
  "an image with transparency is a character, any other image is a backdrop" without a branch inside a
  handler. Reordering those two rules silently sends every character to the backdrop slot, so the ordering has
  a test of its own.
- **A drop that matches nothing, straddles two rules, or brings several files to a single-file rule is
  rejected whole**, with a dialog naming what was dropped and what would have worked. Rejecting the whole
  drop rather than its usable part is deliberate: a partial action on an ambiguous gesture is harder to undo
  than no action, and the file dialogs are still there.
- **A drop arriving while a modal is open is ignored, not reported.** The OS drop targets the window, not
  whatever DPG is drawing inside it, so a file can land while a dialog has the app's attention; acting on it
  would answer a question the user is still being asked, and reporting would stack a second modal on the
  first.
- **Predicates compose (`all_of`)**, cheapest first. Without it the avatar rules were lopsided — one content-
  based, one extension-based — so a file the imaging library merely happened to open would have routed as a
  character image.

## What each app does with a drop

| App | Drop | Rationale worth keeping |
|---|---|---|
| Librarian | images and documents attach to the next message | **One rule, not one per kind.** A drop mixing an image and a document is a supported attach, and the router rejects drops that straddle rules. Routing between kinds is `_attach_callback`'s job anyway — it already does it for the browser, including the gate refusing images on a text-only model. |
| Visualizer | `.pickle` opens a dataset; `.bib` opens the importer, pre-filled | Two kinds, two destinations. Only one dataset can be open, so several is an error rather than a choice; BibTeX is *input* to a tool that takes any number. It stops short of starting the import, since choosing the output dataset is the user's next step. |
| Cherrypick | a folder opens it | A folder is the only thing the app opens. |
| XDot viewer | `.dot`, `.xdot`, `.gv` opens it | Matches what its dialog offers. |
| Avatar pose editor | image with an alpha channel loads as the character; `.json` loads emotion templates | The rule tests for the alpha channel because *that is what the loader requires* — `torch_load_rgba_image` rejects an image without one. A photo therefore meets a dialog naming what the app takes, rather than a traceback. |
| Avatar settings editor | image *with transparency* → character; any other image → backdrop; `.json` → animator settings | See below. |

### The settings editor is the interesting one

It has two image slots and **the gesture cannot distinguish them**: GLFW's drop callback fires only on
release, with no drag-enter, drag-over or drag-leave event. So nothing can light up a drop zone while a drag
is in flight, and there is no zone to aim at — on every platform, not just X11. The drop-target overlay this
would otherwise have wanted is not implementable through this route at all.

So the image decides. The discriminator is **transparency, not the presence of an alpha channel**, and that
distinction is the whole design: a character is a cutout, while a backdrop is a full frame — and a backdrop
exported as RGBA carries an alpha channel with every pixel opaque. Testing for the channel would have
swallowed every such backdrop into the character slot. `codec.has_alpha_channel` and `codec.has_transparency`
exist as separate predicates for exactly this reason, and their difference is what their tests assert.

A consequence worth stating because it looks like a bug and is not: an opaque character *render* (a raw
generation, before the background is cut out) loads as a backdrop. It is not a usable character — the loader
would reject it — so the routing is right.

## What it cost, and what it surfaced

Wiring, as the probe predicted. What was not predicted is that live-testing it found a **live bug unrelated to
drag-and-drop**: clicking OK on an error dialog also acted on the graph behind it, because `XDotWidget`'s
mouse handlers are global and its "is the mouse over me" test is geometric, so it cannot see occlusion.

Auditing that across the constellation found two more, both of the same kind and both predating this work:
`raven-avatar-pose-editor`'s modal guard was written before the app had a messagebox and never revisited, so
hotkeys fired behind every load-failure dialog; and `raven-avatar-settings-editor`'s listed four of its five
file dialogs. The fix was one `is_any_modal_window_visible` per app — six now, uniformly — and `XDotWidget`
taking it as an `input_blocked` callback rather than importing `messagebox` itself, since what counts as "on
top" is the app's business.

## Still open

**Wayland.** GLFW implements it, but that is inference from GLFW's feature set rather than something observed.
Decided 2026-08-07 not to gate the feature on it — X11, macOS and Windows are three platforms, and this
machine has no Wayland session to test against. `investigations/dpg-dnd/dnd_probe.py` is what answers it
there.

## Where things are

- Implementation: `raven/common/gui/filedrop.py`; predicates in `raven/common/image/codec.py`.
- Tests: `raven/common/gui/tests/test_filedrop.py` (routing is pure and fully tested; the install-timing
  invariant is the one `gui`-marked test).
- Measurements and both probes: `investigations/dpg-dnd/`.
- The general GLFW-callback facts: `dpg-notes.md`, "GLFW callbacks are the exception".

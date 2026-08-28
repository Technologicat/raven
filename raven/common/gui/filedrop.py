"""OS-level file drag-and-drop for DPG apps: files dragged in from the desktop file manager.

DPG has no API for this — its drag-and-drop is ImGui-internal, between widgets in the same app. But the
capability is already in the binary one layer down. DPG statically links GLFW; GLFW has had
`glfwSetDropCallback` since 3.1; and the GLFW symbols are exported from DPG's C extension, so the callback
can be installed from Python with `ctypes`. That is what this module does, and it is why there is no
per-platform code here: GLFW's X11, Cocoa and Win32 backends each implement the drop, and we bind to
whichever one was compiled in.

Measured on X11 with `dearpygui` 2.3.1. Wayland is unverified; if it does not work there, `is_available`
and `install` degrade to returning `False` with a log message rather than breaking the app. The apparatus
is in `investigations/dpg-dnd/`.

Two constraints shape the API, and both are load-bearing:

  - **`install` must be called on the render thread, after `dpg.show_viewport()`.** GLFW's "current
    context" is per-thread, and `show_viewport` is what makes DPG's window current on the calling thread.
    Measured: the handle is NULL before that call, non-NULL after it (rendering a frame first is not
    needed), and NULL on every other thread. There is no other route to the handle, so a background task
    or a frame callback cannot do this.

  - **Handlers do not run on the render thread, deliberately.** GLFW dispatches the drop from
    `glfwPollEvents()`, which DPG calls inside `render_dearpygui_frame()`, so the raw C callback arrives
    *on* the render thread — where nothing may wait for a frame, ruling out `split_frame` and therefore
    the modal messagebox. Since "you dropped something this app cannot use" wants a dialog, the raw
    callback does nothing but copy the paths and hand them to a worker thread. Handlers may then use the
    full GUI vocabulary, dialogs included.

Handlers receive a `list` of `str`, each an absolute path — the same shape `FileDialog` sends its callback,
so an app can usually route a drop straight into the callback it already has.

Drops are dispatched one at a time, in arrival order. The modal messagebox is a singleton window, so two
handlers reporting an error concurrently would race over it.

This module is licensed under the 2-clause BSD license, to facilitate integration anywhere.
"""

__all__ = ["is_available", "install", "uninstall", "by_extension", "is_directory", "all_of", "DropRule", "make_router"]

import logging
logger = logging.getLogger(__name__)

import ctypes
import os
import queue
import threading
from typing import Callable, NamedTuple, Optional, Sequence, Union

import dearpygui.dearpygui as dpg

from . import messagebox
from . import utils as guiutils

# ---------------------------------------------------------------------------
# Binding to the GLFW inside DPG
# ---------------------------------------------------------------------------

# void (*GLFWdropfun)(GLFWwindow* window, int path_count, const char* paths[])
_DROPFUN = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p))

_lib = None  # the bound C extension, or None if binding failed
_bind_attempted = False

def _bind() -> Optional[ctypes.CDLL]:
    """Bind to the GLFW functions exported by DPG's C extension. Idempotent; returns None if unavailable."""
    global _lib
    global _bind_attempted
    if _bind_attempted:
        return _lib
    _bind_attempted = True
    try:
        # Load the *already-loaded* extension by its own `__file__` rather than by a constructed filename:
        # it has to be the same GLFW instance that owns DPG's window, and the extension suffix is
        # platform-specific (`.so`, `.pyd`, and ABI tags).
        import dearpygui._dearpygui as dpg_core
        lib = ctypes.CDLL(dpg_core.__file__)
        lib.glfwGetCurrentContext.restype = ctypes.c_void_p
        lib.glfwSetDropCallback.restype = _DROPFUN
        lib.glfwSetDropCallback.argtypes = [ctypes.c_void_p, _DROPFUN]
    except (ImportError, OSError, AttributeError) as exc:
        # A build that hides its GLFW symbols, or a platform where the extension cannot be opened as a
        # plain shared library. Not fatal: the app keeps its file dialogs.
        logger.warning(f"_bind: OS file drop unavailable, {type(exc)}: {exc}")
        return None
    _lib = lib
    return _lib

def is_available() -> bool:
    """Whether OS file drop can be installed in this build. Cheap; safe to call any time."""
    return _bind() is not None

# ---------------------------------------------------------------------------
# Dispatch off the render thread
# ---------------------------------------------------------------------------

_handler: Optional[Callable[[list], None]] = None
_drops: Optional[queue.Queue] = None
_worker: Optional[threading.Thread] = None

# GLFW stores only the raw function pointer, so the `ctypes` callback object must outlive it. Losing this
# reference is a segfault at drop time, not an exception — hence a module-level binding that is never cleared.
_drop_callback = None

def _dispatch_loop() -> None:
    """Worker: run handlers off the render thread, one drop at a time."""
    while True:
        paths = _drops.get()
        handler = _handler
        if handler is None:  # uninstalled between the drop and now
            continue
        if not dpg.is_dearpygui_running():
            # Teardown, or a drop that landed before the render loop started. Calling into DPG after
            # `destroy_context` segfaults, so drop the drop and say so.
            logger.info(f"_dispatch_loop: render loop not running, discarding {len(paths)} dropped path(s)")
            continue
        try:
            handler(paths)
        except Exception as exc:
            # A handler that raises must not kill the worker; the next drop should still work.
            logger.exception(f"_dispatch_loop: handler raised, {type(exc)}: {exc}")

def _on_drop(window, count, paths) -> None:  # noqa: ARG001 -- `window` is part of the GLFW callback signature
    """The GLFW drop callback. **Runs on the render thread**, inside `render_dearpygui_frame`.

    Does the minimum and returns: copies the paths out of GLFW's buffer (valid only for the duration of
    this call) and queues them for the worker. Anything that waits for a frame would deadlock here.
    """
    # An exception escaping a `ctypes` callback unwinds into C, so contain everything.
    try:
        files = [paths[i].decode("utf-8", errors="replace") for i in range(count)]
        logger.info(f"_on_drop: {len(files)} path(s) dropped")
        _drops.put(files)
    except Exception as exc:
        logger.exception(f"_on_drop: {type(exc)}: {exc}")

def install(handler: Callable[[list], None]) -> bool:
    """Start delivering OS file drops to `handler`. Returns whether it was installed.

    Call **on the render thread, after `dpg.show_viewport()`** — that call is what makes DPG's window the
    calling thread's current GLFW context, and there is no other way to reach the handle. Called too early,
    or from another thread, this logs an error and returns `False`. "Too early" is judged against DPG's own
    viewport state rather than GLFW's, which cannot tell a live window from a destroyed one.

    `handler` receives a `list` of `str`, each an absolute path, and runs on a worker thread, so it may
    show dialogs and wait for frames. Exceptions from it are logged, not propagated.

    Installing again replaces the previous handler. A `False` return is not an error to handle: the app
    simply has no drag-and-drop, and its file dialogs still work.
    """
    global _handler
    global _drops
    global _worker
    global _drop_callback

    if not guiutils.is_render_thread():
        logger.error("install: must be called on the render thread; GLFW's current context is per-thread, "
                     "so the window handle is not reachable from here. Not installing.")
        return False

    lib = _bind()
    if lib is None:
        return False

    # Ask DPG whether a viewport is up before trusting GLFW's answer. `glfwGetCurrentContext` is not
    # cleared by `dpg.destroy_context()` — measured: it keeps returning the destroyed window's handle
    # until the next `show_viewport` replaces it — so on its own it answers "has a window ever been
    # current on this thread", and a second context would install against freed memory.
    if not dpg.is_viewport_ok():
        logger.error("install: no viewport is up — call this after `dpg.show_viewport()`. Not installing.")
        return False

    window = lib.glfwGetCurrentContext()
    if not window:
        logger.error("install: no current GLFW context — call this after `dpg.show_viewport()`. Not installing.")
        return False

    _handler = handler
    if _drops is None:
        _drops = queue.Queue()
    if _worker is None:
        _worker = threading.Thread(target=_dispatch_loop, name="filedrop", daemon=True)
        _worker.start()
    if _drop_callback is None:
        _drop_callback = _DROPFUN(_on_drop)
        previous = lib.glfwSetDropCallback(window, _drop_callback)
        if previous:
            # Nothing in DPG installs one (checked), so this would mean another extension is also using
            # GLFW's single drop slot — and we have just displaced it.
            logger.warning("install: replaced an existing GLFW drop callback")
    logger.info("install: OS file drop installed")
    return True

def uninstall() -> None:
    """Stop delivering drops to the handler.

    The GLFW-level callback stays installed and becomes a no-op. Clearing it would buy nothing — the
    `ctypes` callback object has to stay alive regardless, since GLFW holds a raw pointer to it.
    """
    global _handler
    _handler = None
    logger.info("uninstall: OS file drop handler removed")

# ---------------------------------------------------------------------------
# Routing a drop to the right handler
# ---------------------------------------------------------------------------

def by_extension(*extensions: str) -> Callable[[str], bool]:
    """Predicate: a file (not a directory) whose extension is one of `extensions`, compared case-insensitively.

    Extensions may be given with or without the leading dot: `by_extension(".bib")` and `by_extension("bib")`
    are the same thing.
    """
    wanted = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    def matches(path: str) -> bool:
        return os.path.splitext(path)[1].lower() in wanted and os.path.isfile(path)
    return matches

def is_directory(path: str) -> bool:
    """Predicate: the dropped path is a directory."""
    return os.path.isdir(path)

def all_of(*predicates: Callable[[str], bool]) -> Callable[[str], bool]:
    """Predicate combinator: match paths satisfying every one of `predicates`.

    Evaluated left to right and short-circuiting, so order them cheapest first — `by_extension` before
    anything that opens the file, which spares every unrelated dropped path a decode.

    Its usual job is pairing a name test with a content test, so that "an image with transparency" cannot
    be satisfied by some unrelated file that the imaging library merely happens to be able to open.
    """
    def matches(path: str) -> bool:
        return all(predicate(path) for predicate in predicates)
    return matches

class DropRule(NamedTuple):
    """One branch of a drop router: which dropped paths it claims, what to do with them, what to call them.

    `matches`: Predicate on one absolute path. Rules are tried in order and the **first** match wins, so a
               narrow rule must be listed before a broader one that would also match. That ordering does
               real work — it is how "an image with transparent pixels is a character, any other image is a
               backdrop" is expressed as two rules rather than as a branch inside one handler.

    `handler`: Called with the matched paths (a non-empty `list` of `str`) when this rule wins the drop.

    `label`: What this rule accepts, in the user's words and plural — "BibTeX files", "a dataset file".
             Appears in the dialog shown when a drop matches nothing, so it should read as a list item.

    `multiple`: Whether this rule accepts more than one file at a time. When `False`, a drop of several
                matching files is rejected rather than silently using the first.
    """
    matches: Callable[[str], bool]
    handler: Callable[[list], None]
    label: str
    multiple: bool = True

def _describe(paths: Sequence[str]) -> str:
    """Format dropped paths for a dialog: basenames, one per line, abridged if there are many."""
    names = [os.path.basename(p.rstrip(os.sep)) or p for p in paths]
    shown = names[:8]
    lines = "\n".join(f"  - {name}" for name in shown)
    if len(names) > len(shown):
        lines += f"\n  … and {len(names) - len(shown)} more"
    return lines

def make_router(rules: Sequence[DropRule],
                *,
                reference_window: Union[str, int],
                what: str = "This window",
                blocked: Optional[Callable[[], bool]] = None,
                on_rejected: Optional[Callable[[str, str], None]] = None) -> Callable[[list], None]:
    """Build a drop handler that dispatches to the first matching rule, and explains itself when it cannot.

    Pass the result to `install`.

    A drop is accepted when every dropped path matches the same rule. It is rejected — with a dialog naming
    what was dropped and listing what the rules accept — when some path matches nothing, when the drop
    straddles two rules, or when several files match a rule declared `multiple=False`. Rejecting the whole
    drop rather than the unusable part of it is deliberate: a partial action on an ambiguous gesture is
    harder to undo than no action, and the user still has the file dialogs.

    `reference_window`: DPG tag or ID to center the rejection dialog on, normally the app's main window.

    `what`: How the rejection dialog refers to the app, e.g. `"Raven-visualizer"`. Reads as
            "<what> accepts:".

    `blocked`: Optional predicate; while it answers `True`, drops are ignored. Raven's apps pass their
               `is_any_modal_window_visible`. The OS drop targets the window, not whatever the app happens
               to be showing inside it, so a file can land while a dialog is up — where acting on it would
               mean answering a question the user is still in the middle of being asked. Ignored rather
               than reported, since reporting means stacking a second modal on the first.

    `on_rejected`: Override the rejection reporting; receives `(title, message)`. Defaults to a modal
                   dialog. Mainly for tests, which have no viewport to center on.
    """
    accepted = "\n".join(f"  - {rule.label}" for rule in rules)

    def reject(title: str, message: str) -> None:
        if on_rejected is not None:
            on_rejected(title, message)
            return
        messagebox.modal_dialog(window_title=title,
                                message=message,
                                buttons=["OK"], ok_button="OK", cancel_button="OK",
                                centering_reference_window=reference_window)

    def route(paths: list) -> None:
        if not paths:
            return
        if blocked is not None and blocked():
            logger.info(f"make_router.route: a modal window is open, ignoring {len(paths)} dropped path(s)")
            return
        claimed = {}  # rule index -> matched paths, in rule order (dicts preserve insertion order)
        unmatched = []
        for path in paths:
            for index, rule in enumerate(rules):
                if rule.matches(path):
                    claimed.setdefault(index, []).append(path)
                    break
            else:
                unmatched.append(path)

        if unmatched:
            logger.info(f"make_router.route: {len(unmatched)} dropped path(s) match no rule")
            reject("Cannot open this",
                   f"Dropped:\n{_describe(unmatched)}\n\n{what} accepts:\n{accepted}")
            return
        if len(claimed) > 1:
            logger.info(f"make_router.route: drop straddles {len(claimed)} rules")
            reject("Cannot open these together",
                   f"Dropped:\n{_describe(paths)}\n\n"
                   f"These are different kinds of file, and {what.lower()} handles them differently. "
                   "Drop one kind at a time.")
            return

        index, matched = next(iter(claimed.items()))
        rule = rules[index]
        if len(matched) > 1 and not rule.multiple:
            logger.info(f"make_router.route: {len(matched)} paths for single-file rule '{rule.label}'")
            reject("Cannot open several at once",
                   f"Dropped:\n{_describe(matched)}\n\n"
                   f"{what} opens one at a time here ({rule.label}). Drop a single one.")
            return
        logger.info(f"make_router.route: routing {len(matched)} path(s) to rule '{rule.label}'")
        rule.handler(matched)

    return route

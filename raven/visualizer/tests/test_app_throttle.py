"""Pins the wiring of the Visualizer's idle framerate throttle.

The throttle drops the render loop to ~12 fps when nothing is happening, and back to full rate for half a
second after any user input. That second half is what makes it invisible, and it is spread across seven
callbacks: each registered input handler notes the input by calling `_on_any_input`, because the app owns
those callbacks and noticing the input is as much its business as acting on it.

Spread across seven places, it rots in the way that is hardest to catch. Miss one and the app is sluggish to
exactly one kind of input — the mouse wheel, say — and normal to everything else, which reads as a graphics
problem rather than as a missing line. Add an eighth handler later and the same thing happens to whatever it
handles.

**These read `app.py` as source rather than importing it**, because importing a Raven `app.py` *runs that
app*: the module body is the program — it parses the command line, builds the GUI and enters the render
loop, all at import time. Failing on pytest's own argv is merely the first thing that goes wrong, and a
conftest that fed it a plausible argv would get further into starting an app, not closer to a test.

Reading the source is what is left, and here it is enough: which callback is registered for which handler,
and whether each one makes the call, are both syntax. It also catches what an import-based test could not,
since a handler added in a future edit is visible to the parse whether or not anything ever calls it.
"""

import ast
import pathlib

import pytest

_APP = pathlib.Path(__file__).resolve().parent.parent / "app.py"

#: What the input callbacks must call, once, to keep the app at full frame rate while the user is working.
_BOOKKEEPING = "_on_any_input"

#: The DPG calls that register a callback for user input. A handler not on this list is one this test does
#: not know about; adding one to DPG's API means adding it here, which is the point at which somebody
#: decides whether its callback owes the bookkeeping too.
_INPUT_HANDLER_ADDERS = frozenset({"add_mouse_move_handler", "add_mouse_click_handler",
                                   "add_mouse_release_handler", "add_mouse_down_handler",
                                   "add_mouse_wheel_handler", "add_mouse_drag_handler",
                                   "add_mouse_double_click_handler",
                                   "add_key_press_handler", "add_key_down_handler", "add_key_release_handler",
                                   "add_item_clicked_handler"})


@pytest.fixture(scope="module")
def app_tree() -> ast.AST:
    return ast.parse(_APP.read_text(encoding="utf-8"))


def _registered_callback_names(tree: ast.AST) -> set:
    """Return the name of every function registered as a DPG input-handler callback."""
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr in _INPUT_HANDLER_ADDERS):
            continue
        for kw in node.keywords:
            if kw.arg == "callback" and isinstance(kw.value, ast.Name):
                names.add(kw.value.id)
    return names


def _functions_calling(tree: ast.AST, callee: str) -> set:
    """Return the name of every top-level function whose body calls `callee`."""
    calling = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name) and inner.func.id == callee:
                calling.add(node.name)
    return calling


class TestInputBookkeeping:
    def test_the_app_registers_input_handlers_at_all(self, app_tree):
        """The negative control: the rest of this class is vacuous if nothing is found to check.

        A rename in DPG's API, or a move of the registrations out of `app.py`, would leave the search
        finding nothing — and an empty set satisfies "every one of them calls it" perfectly.
        """
        registered = _registered_callback_names(app_tree)
        assert len(registered) >= 5, (f"found only {sorted(registered)}; this test cannot tell a wired-up "
                                      f"app from an unwired one unless it can find the handlers")

    def test_every_registered_input_callback_notes_the_input(self, app_tree):
        registered = _registered_callback_names(app_tree)
        calling = _functions_calling(app_tree, _BOOKKEEPING)
        missing = sorted(registered - calling)
        assert not missing, (f"{missing} are registered as input callbacks but never call {_BOOKKEEPING}(), "
                             f"so the app will stay at the idle frame rate while the user drives it that way")

    def test_the_bookkeeping_is_not_called_from_the_render_loop(self, app_tree):
        """It marks *user* input; called from the loop it would mean the app is never idle.

        Cheap to get wrong in the plausible direction — the loop is where the throttle lives, so it is where
        a reader looking to "make sure the timestamp is fresh" would put it.
        """
        calling = _functions_calling(app_tree, _BOOKKEEPING)
        registered = _registered_callback_names(app_tree)
        unexpected = sorted(name for name in calling if name not in registered and name != _BOOKKEEPING)
        assert not unexpected, (f"{unexpected} call {_BOOKKEEPING}() without being input callbacks; if that "
                                f"is deliberate, say why here, and check it is not the render loop")

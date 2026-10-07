"""Pins that every way of sending a chat message in `raven-librarian` sends the staged attachments with it.

There is more than one send path — typing and the microphone, so far — and only one of them knew about the
attachment strip, so a spoken message went out bare and left its attachments to be sent with whatever came
next. The remedy was a single function that snapshots the strip, sends and clears it, and this test holds the
app to it: `chat_controller.chat_exchange` is called from that function and nowhere else.

These read `app.py` as source rather than importing it, because importing a Raven `app.py` runs that app —
the module body is the program. A send path added later is visible to the parse whether or not anything
calls it, which is the case this exists for.
"""

import ast
import pathlib

_APP = pathlib.Path(__file__).resolve().parent.parent / "app.py"

#: The one function allowed to start an exchange.
_SENDER = "_send_with_staged_attachments"


def _enclosing_functions_of_chat_exchange_calls(source: str) -> list[str]:
    """The name of the innermost function around each `<anything>.chat_exchange(...)` call in `source`."""
    found = []

    def visit(node: ast.AST, maybe_function: str | None) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            maybe_function = node.name
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and
                node.func.attr == "chat_exchange"):
            found.append(maybe_function or "<module>")
        for child in ast.iter_child_nodes(node):
            visit(child, maybe_function)

    visit(ast.parse(source), None)
    return found


class TestSendPaths:
    def test_the_parse_finds_a_call_in_a_named_function(self):
        source = "def f():\n    controller.chat_exchange('hi')\ncontroller.chat_exchange('bare')\n"
        assert _enclosing_functions_of_chat_exchange_calls(source) == ["f", "<module>"]

    def test_only_the_shared_sender_starts_an_exchange(self):
        callers = _enclosing_functions_of_chat_exchange_calls(_APP.read_text(encoding="utf-8"))
        assert callers, "no `chat_exchange` call found at all, so this test is no longer looking at the send path"
        strays = [name for name in callers if name != _SENDER]
        assert not strays, (f"`chat_exchange` is called from {strays}; send through `{_SENDER}` instead, "
                            f"or the staged attachments are left behind")

"""Every GUI app's render loop logs, on the way out, that DPG stopped running.

Spans every `app.py` with a render loop rather than testing one module. Read from the source: importing a
Raven `app.py` runs the app.
"""

import ast
from pathlib import Path

import pytest

RAVEN_ROOT = Path(__file__).resolve().parents[3]

def _app_modules() -> list[Path]:
    return sorted(path for path in RAVEN_ROOT.rglob("app.py")
                  if "render_dearpygui_frame" in path.read_text(encoding="utf-8"))

def _is_dpg_running_loop(node: ast.AST) -> bool:
    return (isinstance(node, ast.While) and
            isinstance(node.test, ast.Call) and
            ast.unparse(node.test.func) == "dpg.is_dearpygui_running")

def _statement_after_render_loop(source: str) -> ast.stmt | None:
    """The statement following the `while dpg.is_dearpygui_running():` loop in `source`, or `None`."""
    for node in ast.walk(ast.parse(source)):
        for field in ("body", "orelse", "finalbody"):
            body = getattr(node, field, None)
            if not isinstance(body, list):
                continue
            for k, stmt in enumerate(body):
                if _is_dpg_running_loop(stmt):
                    return body[k + 1] if k + 1 < len(body) else None
    return None

def _logs_loop_stopped(stmt: ast.stmt | None) -> bool:
    return (isinstance(stmt, ast.Expr) and
            isinstance(stmt.value, ast.Call) and
            ast.unparse(stmt.value.func) == "guiutils.log_render_loop_stopped")

def test_the_app_list_is_not_empty():
    assert len(_app_modules()) >= 7, f"found only {[str(p) for p in _app_modules()]}; the glob has gone stale"

def test_a_loop_without_the_call_is_detected():
    source = ("while dpg.is_dearpygui_running():\n"
              "    dpg.render_dearpygui_frame()\n"
              "logger.info('done')\n")
    assert _statement_after_render_loop(source) is not None, "the loop was not found, so the check below is vacuous"
    assert not _logs_loop_stopped(_statement_after_render_loop(source))

@pytest.mark.parametrize("path", _app_modules(), ids=lambda path: str(path.relative_to(RAVEN_ROOT)))
def test_the_render_loop_logs_why_it_ended(path):
    stmt = _statement_after_render_loop(path.read_text(encoding="utf-8"))
    assert _logs_loop_stopped(stmt), (f"{path.relative_to(RAVEN_ROOT)}: the statement after the render loop "
                                      f"should be `guiutils.log_render_loop_stopped(logger)`, "
                                      f"got {ast.unparse(stmt) if stmt is not None else 'nothing'}")

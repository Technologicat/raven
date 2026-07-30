"""Tests for `raven.common.gui.messagebox`'s existence check.

`modal_dialog_window_exists` is a guard: every app's "is a modal open?" hotkey check calls it, and the call
that follows (`dpg.is_item_visible("modal_dialog_window")`) raises if the window has not been created. So the
guard answering `True` too eagerly is not a cosmetic fault — it turns every keystroke into a traceback in
every app that guards its hotkeys. It did exactly that, because the original implementation probed with
`dpg.get_item_alias`, which returns `None` for an unknown tag rather than raising.

DPG runs without a mapped window, so this drives the real thing; see `dpg-notes.md`, "Testing DPG code".
"""

import pytest

dpg = pytest.importorskip("dearpygui.dearpygui", reason="dearpygui not installed (GUI toolkit absent in CI)")

from raven.common.gui import messagebox  # noqa: E402 -- after importorskip by design


@pytest.fixture
def dpg_context():
    """A DPG context with an unmapped viewport, fresh per test so the item registry starts empty."""
    dpg.create_context()
    dpg.create_viewport(width=100, height=100)  # never shown: tests must not steal focus
    dpg.setup_dearpygui()
    yield
    dpg.destroy_context()


def test_reports_absent_before_the_window_is_created(dpg_context):
    """The whole point of the guard: no modal dialog has been built, so there is nothing to ask about."""
    assert messagebox.modal_dialog_window_exists() is False


def test_the_follow_up_call_is_what_makes_a_wrong_answer_expensive(dpg_context):
    """Documents why `True` here is not a harmless over-report: callers immediately do this, and it raises."""
    assert not dpg.does_item_exist("modal_dialog_window")
    with pytest.raises(Exception):
        dpg.is_item_visible("modal_dialog_window")


def test_reports_present_once_a_window_carries_the_tag(dpg_context):
    """And the guard must still say yes when there really is one, or hotkeys would never be suppressed."""
    with dpg.window(tag="modal_dialog_window"):  # tag
        dpg.add_text("hello")
    assert messagebox.modal_dialog_window_exists() is True

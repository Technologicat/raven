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


def test_is_visible_answers_false_rather_than_raising_when_there_is_no_dialog(dpg_context):
    """`is_visible` is called from global mouse handlers, i.e. potentially every frame before any dialog exists.

    It must therefore answer the no-dialog case itself. Asking `dpg.is_item_visible` for a tag that does not
    resolve raises (see above), so a version that skipped the existence check would turn every mouse move in
    a freshly started app into an exception.
    """
    assert messagebox.is_visible() is False


def test_is_visible_is_false_for_a_dialog_that_exists_but_is_hidden(dpg_context):
    """Existence is not visibility: the modal window is created once and then shown and hidden repeatedly.

    `modal_dialog` builds the window on first use and `hide_item`s it on close, so after the first dialog in
    a session the window exists forever. A check that only asked whether it exists would suppress input
    permanently from then on.
    """
    with dpg.window(tag="modal_dialog_window", show=False):  # tag
        dpg.add_text("hello")
    assert messagebox.modal_dialog_window_exists() is True
    assert messagebox.is_visible() is False

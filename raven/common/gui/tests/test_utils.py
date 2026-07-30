"""Tests for `raven.common.gui.utils`' frame-waiting guard.

`dpg.split_frame()` waits for the render loop to complete a frame. Called *from* that loop — or from app
startup, which runs on the same thread before the loop begins — the wait can never be satisfied, and the app
hangs with no traceback, no log line, and nothing to bisect. `guiutils.split_frame` converts that into either
a `RuntimeError` naming the operation (when waiting is the whole job) or a warning and a stale-geometry
fallback (when it is only an improvement).

**The successful wait is deliberately not tested**, and that is not an oversight: with no render loop running,
a real `dpg.split_frame()` would hang the test suite in exactly the way this guard exists to prevent. What is
asserted is the guard, and that the two library functions with opposite policies are really wired to it —
`wait_for_resize` raising and `recenter_window` degrading. pytest runs on the main thread, so every test here
is already standing on the hazardous thread.
"""

import threading

import pytest

dpg = pytest.importorskip("dearpygui.dearpygui", reason="dearpygui not installed (GUI toolkit absent in CI)")

from raven.common.gui import utils as guiutils  # noqa: E402 -- after importorskip by design


@pytest.fixture
def dpg_context():
    """A DPG context with an unmapped viewport, fresh per test so the item registry starts empty."""
    dpg.create_context()
    dpg.create_viewport(width=100, height=100)  # never shown: tests must not steal focus
    dpg.setup_dearpygui()
    yield
    dpg.destroy_context()


def test_the_test_runner_itself_is_on_the_render_thread():
    """Establishes the premise the rest of the module depends on."""
    assert guiutils.is_render_thread() is True


def test_a_worker_thread_is_not_the_render_thread():
    """The distinction has to be real, or the guard would refuse every caller including the legitimate ones."""
    answers = []
    worker = threading.Thread(target=lambda: answers.append(guiutils.is_render_thread()))
    worker.start()
    worker.join()
    assert answers == [False]


def test_a_required_wait_raises_instead_of_hanging():
    with pytest.raises(RuntimeError):
        guiutils.split_frame(operation="unit test: a wait that cannot be skipped")


def test_the_error_names_the_operation_so_the_call_site_is_findable():
    """A hang tells you nothing; the point of raising is that the message has to be worth reading."""
    with pytest.raises(RuntimeError, match="a distinctive operation name"):
        guiutils.split_frame(operation="a distinctive operation name")


def test_an_optional_wait_reports_that_it_did_not_happen(caplog):
    """`False` is what lets a caller adapt, and the warning is what makes the skip discoverable."""
    with caplog.at_level("WARNING", logger="raven.common.gui.utils"):
        waited = guiutils.split_frame(operation="unit test: a wait we can live without", required=False)
    assert waited is False
    assert "unit test: a wait we can live without" in caplog.text


def test_wait_for_resize_raises_rather_than_hanging(dpg_context):
    """Waiting *is* the operation, so there is nothing to degrade to."""
    with dpg.window() as window:
        dpg.add_text("x")
    with pytest.raises(RuntimeError):
        guiutils.wait_for_resize(window)


def test_recenter_window_degrades_instead_of_raising(dpg_context):
    """The opposite policy: an off-center window beats both a hang and an exception."""
    with dpg.window(width=100, height=100) as reference_window:
        dpg.add_text("reference")
    with dpg.window(autosize=True) as thewindow:
        dpg.add_text("centered on the reference")
    guiutils.recenter_window(thewindow, reference_window=reference_window)  # must not raise

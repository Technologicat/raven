"""`WaitUntilShown`: render work that needs a laid-out widget, held while the widget is hidden.

Nothing here renders a frame: whether an item is shown is its configuration, readable headless. The worker's
queue is stubbed with a recorder, so no worker thread starts — one would outlive this module's context.
"""

import pytest

dpg = pytest.importorskip("dearpygui.dearpygui", reason="dearpygui not installed")

from raven.vendor import DearPyGui_Markdown as dpg_markdown  # noqa: E402 -- after importorskip by design


@pytest.fixture(scope="module")
def dpg_context():
    dpg.create_context()
    dpg.create_viewport(width=100, height=100)  # never shown: tests must not steal focus
    dpg.setup_dearpygui()
    yield
    dpg.destroy_context()


@pytest.fixture
def queued(dpg_context, monkeypatch):
    """The calls that reach the worker's queue, recorded instead of queued."""
    calls = []
    monkeypatch.setattr(dpg_markdown.CallInNextFrame, "append",
                        classmethod(lambda cls, func, *args, **kwargs: calls.append((func, args))))
    dpg_markdown.WaitUntilShown._by_blocker.clear()
    yield calls
    dpg_markdown.WaitUntilShown._by_blocker.clear()


def work(label):
    def run():
        pass
    run.label = label
    return run


def labels(calls):
    return [func.label for func, _ in calls]


def test_work_for_a_shown_widget_is_queued_at_once(queued):
    with dpg.window():
        text = dpg.add_text("x")
    dpg_markdown.WaitUntilShown.call_when_shown(text, work("a"))
    assert labels(queued) == ["a"]
    assert not dpg_markdown.WaitUntilShown.is_waiting()


def test_work_under_a_hidden_container_waits_for_it_and_no_longer(queued):
    with dpg.window():
        with dpg.group(show=False) as collapsed:
            text = dpg.add_text("x")
    dpg_markdown.WaitUntilShown.call_when_shown(text, work("a"))
    dpg_markdown.WaitUntilShown.release_shown()
    assert queued == [], "released while its container was still hidden"
    assert dpg_markdown.WaitUntilShown.is_waiting()

    dpg.show_item(collapsed)
    dpg_markdown.WaitUntilShown.release_shown()
    assert labels(queued) == ["a"]
    assert not dpg_markdown.WaitUntilShown.is_waiting()


def test_many_pieces_of_work_under_one_container_share_one_blocker(queued):
    with dpg.window():
        with dpg.group(show=False):
            texts = [dpg.add_text(str(i)) for i in range(5)]
    for i, text in enumerate(texts):
        dpg_markdown.WaitUntilShown.call_when_shown(text, work(i))
    assert len(dpg_markdown.WaitUntilShown._by_blocker) == 1


def test_work_whose_blocker_is_deleted_is_dropped(queued):
    with dpg.window():
        with dpg.group(show=False) as collapsed:
            text = dpg.add_text("x")
    dpg_markdown.WaitUntilShown.call_when_shown(text, work("a"))
    assert dpg_markdown.WaitUntilShown.is_waiting(), "the control: it was waiting before the delete"

    dpg.delete_item(collapsed)
    dpg_markdown.WaitUntilShown.release_shown()
    assert queued == []
    assert not dpg_markdown.WaitUntilShown.is_waiting()


def test_work_for_a_widget_that_is_gone_is_dropped_at_once(queued):
    with dpg.window():
        text = dpg.add_text("x")
    dpg.delete_item(text)
    dpg_markdown.WaitUntilShown.call_when_shown(text, work("a"))
    assert queued == []
    assert not dpg_markdown.WaitUntilShown.is_waiting()


def test_the_blocker_is_the_innermost_hidden_container(queued):
    """Released when that one is shown, the work re-checks and finds the next — so waiting is per level, not lost."""
    with dpg.window():
        with dpg.group(show=False) as outer:
            with dpg.group(show=False) as inner:
                text = dpg.add_text("x")
    dpg_markdown.WaitUntilShown.call_when_shown(text, work("a"))
    assert list(dpg_markdown.WaitUntilShown._by_blocker) == [inner]

    dpg.show_item(outer)  # the outer one alone changes nothing for work blocked by the inner one
    dpg_markdown.WaitUntilShown.release_shown()
    assert queued == []

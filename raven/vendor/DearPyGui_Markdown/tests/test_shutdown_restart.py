"""The renderer can be shut down for one DPG context and used again in the next.

`shutdown` stops the worker threads before `dpg.destroy_context()`, and keeps them stopped for the rest of
that context's life. `restart` — which `raven.common.gui.utils.setup_markdown` calls, and so `bootup` —
brings them back for a new context. Without it, one module calling `teardown` would silently disable
Markdown's deferred work for every module after it in the same pytest process.

Nothing here maps a window. With no render loop running, a started worker waits in its first
`split_frame` for good, so these tests look at whether a worker was *started* and what was *queued*, not at
rendering — and a worker left waiting like that is the case `restart` has to retire.
"""

import threading
import time

import pytest

dpg = pytest.importorskip("dearpygui.dearpygui", reason="dearpygui not installed")

from raven.common.gui import utils as guiutils  # noqa: E402 -- after importorskip by design
from raven.vendor import DearPyGui_Markdown as dpg_markdown  # noqa: E402 -- after importorskip by design


def _stand_in_worker(generation: int) -> None:
    """Lives and retires as `CallInNextFrame._worker` does, without ever calling into DPG."""
    while not dpg_markdown._retired(generation):
        time.sleep(0.005)


# Every worker these tests start is a stand-in, never the real `_worker`. The real one, with no render loop
# running, enters `dpg.split_frame` and stays there: `shutdown` can retire it but not pull it out of that C
# call, and the module's `dpg.destroy_context()` then frees the context under it. On the Windows CI runner
# that is an access violation, now and then, killing the whole pytest process — so the suite went red on
# pushes that had changed nothing near here. The tests ask whether a worker was *started* and what was
# *queued*, never whether work ran, so a stand-in that keeps the retirement rule answers them just as well.
#
# Autouse and module-scoped, so it is in place before `dpg_context` boots anything. `setattr` raises if
# `_worker` is renamed, rather than quietly patching nothing.
@pytest.fixture(scope="module", autouse=True)
def no_real_worker():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(dpg_markdown.CallInNextFrame, "_worker", staticmethod(_stand_in_worker))
        yield


@pytest.fixture(scope="module")
def dpg_context():
    """One DPG context for the module, with an unmapped viewport."""
    dpg.create_context()
    guiutils.bootup(font_size=20)
    dpg.create_viewport(width=100, height=100)  # never shown: tests must not steal focus
    dpg.setup_dearpygui()
    yield
    guiutils.teardown()
    dpg.destroy_context()


@pytest.fixture
def stopped(dpg_context):
    """The renderer as `shutdown` leaves it, with no worker left running."""
    dpg_markdown.shutdown()
    yield
    dpg_markdown.shutdown()  # leave it stopped, as the module fixture's teardown expects


@pytest.fixture
def parked_worker():
    """A live stand-in for a worker thread, released at the end of the test. The test installs it."""
    release = threading.Event()
    thread = threading.Thread(target=release.wait, daemon=True, name="stand-in worker")
    thread.start()
    yield thread
    release.set()
    thread.join()
    dpg_markdown.CallInNextFrame.worker_thread = None


def _noop():
    pass


class TestShutdown:
    def test_work_is_refused_after_shutdown(self, stopped):
        """The negative control for everything below: a stopped renderer queues nothing and starts nothing."""
        dpg_markdown.CallInNextFrame.append(_noop)
        assert dpg_markdown.CallInNextFrame.worker_thread is None or \
            not dpg_markdown.CallInNextFrame.worker_thread.is_alive()
        assert dpg_markdown.CallInNextFrame.now_frame_queue == []


class TestRestart:
    def test_restart_starts_a_worker_on_next_use(self, stopped):
        before = dpg_markdown.CallInNextFrame.worker_thread
        dpg_markdown.restart()
        dpg_markdown.CallInNextFrame.append(_noop)
        worker = dpg_markdown.CallInNextFrame.worker_thread
        assert worker is not None and worker is not before, "restart left the worker unstartable"
        assert worker.ident is not None, "the worker was created but never started"
        assert [_noop, (), {}] in dpg_markdown.CallInNextFrame.now_frame_queue, "the work was refused"

    def test_setup_markdown_restarts(self, stopped):
        """What makes `bootup` after `teardown` work, in an app or a test module."""
        guiutils.setup_markdown(dpg.add_font_registry(), font_size=20)
        dpg_markdown.CallInNextFrame.append(_noop)
        # The queue, not the worker: a worker left over from an earlier test would satisfy "a worker exists".
        assert [_noop, (), {}] in dpg_markdown.CallInNextFrame.now_frame_queue, "the work was refused"

    def test_restart_drops_work_queued_for_the_old_context(self, stopped):
        dpg_markdown.CallInNextFrame.now_frame_queue.append([_noop, (), {}])
        dpg_markdown.restart()
        assert dpg_markdown.CallInNextFrame.now_frame_queue == []

    def test_restart_retires_a_worker_that_outlived_its_shutdown(self, stopped, parked_worker, caplog):
        """The unmapped-context case: a worker parked in a frame wait that will never end."""
        old_generation = dpg_markdown._generation
        dpg_markdown.CallInNextFrame.worker_thread = parked_worker
        dpg_markdown.restart()
        assert "retiring it" in caplog.text
        assert dpg_markdown._retired(old_generation), "the old worker would go on serving the new context"
        dpg_markdown.CallInNextFrame.append(_noop)
        assert dpg_markdown.CallInNextFrame.worker_thread is not parked_worker, "no fresh worker was started"

    def test_restart_leaves_a_running_renderer_alone(self, dpg_context, parked_worker):
        dpg_markdown.restart()  # a stopped renderer, from whatever ran before, comes back first
        dpg_markdown.CallInNextFrame.worker_thread = parked_worker
        queued = [_noop, (), {}]
        dpg_markdown.CallInNextFrame.now_frame_queue.append(queued)
        try:
            dpg_markdown.restart()
            assert queued in dpg_markdown.CallInNextFrame.now_frame_queue
            assert dpg_markdown.CallInNextFrame.worker_thread is parked_worker
        finally:
            dpg_markdown.CallInNextFrame.now_frame_queue.clear()

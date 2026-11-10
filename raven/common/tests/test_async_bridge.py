"""Tests for `raven.common.async_bridge`."""

import asyncio
import threading
import time

import pytest

from raven.common.async_bridge import AsyncBridge


@pytest.fixture
def bridge():
    the_bridge = AsyncBridge(name="test_bridge")
    yield the_bridge
    the_bridge.shutdown(timeout=2.0)


async def _answer(value):
    await asyncio.sleep(0)
    return value


async def _boom():
    raise ValueError("boom")


class TestSubmit:
    def test_returns_the_result(self, bridge):
        assert bridge.submit(_answer(42)) == 42

    def test_raises_the_coroutines_exception(self, bridge):
        with pytest.raises(ValueError, match="boom"):
            bridge.submit(_boom())

    def test_runs_on_the_bridge_thread_and_not_the_callers(self, bridge):
        async def thread_name():
            return threading.current_thread().name
        assert bridge.submit(thread_name()) == "test_bridge"

    def test_a_coroutine_raising_timeout_error_is_not_mistaken_for_the_wait_expiring(self, bridge):
        # In 3.11+, `concurrent.futures.TimeoutError` *is* the builtin, so catching it around `.result()` would
        # swallow the coroutine's own.
        async def own_timeout():
            raise TimeoutError("the coroutine's own")
        with pytest.raises(TimeoutError, match="the coroutine's own"):
            bridge.submit(own_timeout(), timeout=5.0)

    def test_expiry_raises_and_cancels_the_coroutine(self, bridge):
        cancelled = threading.Event()
        async def forever():
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                cancelled.set()
                raise
        with pytest.raises(TimeoutError):
            bridge.submit(forever(), timeout=0.05)
        assert cancelled.wait(timeout=2.0)

    def test_refuses_to_wait_for_itself(self, bridge):
        async def reenter():
            bridge.submit(_answer(1))
        with pytest.raises(RuntimeError, match="deadlock"):
            bridge.submit(reenter(), timeout=2.0)

    def test_callable_from_many_threads_at_once(self, bridge):
        results = [None] * 16
        def work(i):
            results[i] = bridge.submit(_answer(i))
        threads = [threading.Thread(target=work, args=(i,)) for i in range(16)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert results == list(range(16))


class TestSpawn:
    def test_returns_before_the_coroutine_finishes(self, bridge):
        release = threading.Event()
        async def wait_for_release():
            while not release.is_set():
                await asyncio.sleep(0.01)
            return "released"
        future = bridge.spawn(wait_for_release())
        assert not future.done()
        release.set()
        assert future.result(timeout=2.0) == "released"

    def test_cancelling_the_future_cancels_the_coroutine(self, bridge):
        started, cancelled = threading.Event(), threading.Event()
        async def forever():
            started.set()
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                cancelled.set()
                raise
        future = bridge.spawn(forever())
        assert started.wait(timeout=2.0)
        future.cancel()
        assert cancelled.wait(timeout=2.0)


class TestShutdown:
    def test_cancels_what_is_running(self):
        bridge = AsyncBridge()
        started, cancelled = threading.Event(), threading.Event()
        async def forever():
            started.set()
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                cancelled.set()
                raise
        bridge.spawn(forever())
        assert started.wait(timeout=2.0)
        assert bridge.shutdown(timeout=2.0) is True
        assert cancelled.is_set()

    def test_stops_the_thread(self):
        bridge = AsyncBridge(name="test_bridge_stops")
        assert any(thread.name == "test_bridge_stops" for thread in threading.enumerate())
        bridge.shutdown(timeout=2.0)
        assert not any(thread.name == "test_bridge_stops" for thread in threading.enumerate())

    def test_is_idempotent(self):
        bridge = AsyncBridge()
        assert bridge.shutdown(timeout=2.0) is True
        assert bridge.shutdown(timeout=2.0) is True

    @pytest.mark.filterwarnings("error::RuntimeWarning")  # "coroutine was never awaited"
    def test_afterwards_spawn_refuses_and_closes_the_coroutine(self):
        bridge = AsyncBridge()
        bridge.shutdown(timeout=2.0)
        coro = _answer(1)
        with pytest.raises(RuntimeError, match="shut down"):
            bridge.spawn(coro)
        assert coro.cr_frame is None  # closed, not merely dropped

    def test_a_coroutine_that_ignores_cancellation_cannot_hold_shutdown_past_its_allowance(self):
        bridge = AsyncBridge()
        started = threading.Event()
        async def stubborn():
            started.set()
            while True:
                try:
                    await asyncio.sleep(0.01)
                except asyncio.CancelledError:
                    pass
        bridge.spawn(stubborn())
        assert started.wait(timeout=2.0)
        t0 = time.monotonic()
        bridge.shutdown(timeout=0.1)
        assert time.monotonic() - t0 < 2.0

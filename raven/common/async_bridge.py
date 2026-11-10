"""Call asyncio code from a program that is not asyncio.

Raven's concurrency is threads, and some libraries speak only `asyncio`. `AsyncBridge` is the boundary
between the two: a background thread running an event loop, which any other thread can hand a coroutine to.
Nothing upstream of it has to become async.
"""

__all__ = ["AsyncBridge"]

import asyncio
import concurrent.futures
import logging
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

class AsyncBridge:
    """An event loop on a thread of its own, reachable from any other thread.

    The loop runs from construction until `shutdown`.

    `name`: names the thread, for debuggers and `py-spy`.
    """
    def __init__(self, name: str = "async_bridge") -> None:
        self.name = name
        self._lock = threading.Lock()
        self._closed = False
        self._loop = asyncio.new_event_loop()
        # A daemon, so that a coroutine which ignores cancellation cannot keep the process from exiting.
        #
        # Not a `bgtask` task: a `run_forever` never completes and has no use for `env.cancelled`, so it
        # would only pin a pool worker for the life of the app. The units of work here are the coroutines.
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            self._loop.close()

    def spawn(self, coro: Coroutine[Any, Any, T]) -> "concurrent.futures.Future[T]":
        """Start `coro` on the loop and return immediately.

        Returns a `concurrent.futures.Future`; cancelling it cancels the coroutine.

        Raises `RuntimeError` if the bridge has been shut down, in which case `coro` is closed unstarted.
        """
        with self._lock:  # against `shutdown`, which would otherwise close the loop between the check and the call
            if self._closed:
                coro.close()  # or Python warns that it was never awaited, about a caller that did nothing wrong
                raise RuntimeError(f"AsyncBridge.spawn: instance '{self.name}' has been shut down")
            return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def submit(self, coro: Coroutine[Any, Any, T], timeout: float | None = None) -> T:
        """Run `coro` on the loop and block until it finishes. Returns its result, or raises its exception.

        `timeout`: seconds to wait. On expiry the coroutine is cancelled and `TimeoutError` raised.

        Raises `RuntimeError` if called on the bridge's own thread, where waiting would be waiting for itself,
        or if the bridge has been shut down.
        """
        if threading.current_thread() is self._thread:
            coro.close()
            raise RuntimeError(f"AsyncBridge.submit: called on instance '{self.name}'s own thread, which would deadlock; `await` it instead")
        future = self.spawn(coro)
        done, _not_done = concurrent.futures.wait([future], timeout=timeout)
        if not done:
            future.cancel()
            raise TimeoutError(f"AsyncBridge.submit: instance '{self.name}': no result in {timeout} seconds")
        return future.result()

    def shutdown(self, timeout: float = 5.0) -> bool:
        """Cancel everything still running, stop the loop and join its thread.

        Safe to call more than once, and from any thread but the bridge's own.

        `timeout`: seconds to wait for the coroutines to finish cancelling, and for the thread to exit.

        Returns whether the loop stopped in time. On `False` the daemon thread is abandoned, which cannot
        keep the process from exiting.
        """
        with self._lock:
            if self._closed:
                return not self._thread.is_alive()
            self._closed = True

            async def cancel_everything() -> None:
                me = asyncio.current_task()
                tasks = [task for task in asyncio.all_tasks() if task is not me]
                for task in tasks:
                    task.cancel()
                if tasks:
                    _done, pending = await asyncio.wait(tasks, timeout=timeout)
                    if pending:
                        plural_s = "s" if len(pending) != 1 else ""
                        logger.warning(f"AsyncBridge.shutdown: instance '{self.name}': {len(pending)} task{plural_s} ignored cancellation for {timeout} seconds; abandoning")

            finished = asyncio.run_coroutine_threadsafe(cancel_everything(), self._loop)
            finished.add_done_callback(lambda _future: self._loop.call_soon_threadsafe(self._loop.stop))

        # Twice the allowance: the coroutines get `timeout` to cancel, and stopping the loop comes after that.
        self._thread.join(timeout=2 * timeout)
        stopped = not self._thread.is_alive()
        if not stopped:
            logger.warning(f"AsyncBridge.shutdown: instance '{self.name}': the loop did not stop in time; abandoning its thread")
        return stopped

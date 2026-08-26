"""Tests for raven.common.bgtask — background task management.

Focused on the `ManagedTask` pending-wait invariant: a task superseded during its debounce must release its
worker promptly (within ~one `running_poll_interval`), not pin it for the whole `pending_wait_duration`. This
matters for long debounces with rapid resubmission, where the old "single `time.sleep`" would stack one pinned
pool worker per cancelled task.
"""

import concurrent.futures
import threading
import time

from unpythonic import box
from unpythonic.env import env

from raven.common import bgtask


def _make_env(**kwargs) -> env:
    """Build a task environment as `TaskManager` would, for calling `make_managed_task`'s result directly."""
    e = env(**kwargs)
    e.task_name = "test_task"
    e.cancelled = False
    return e


class TestManagedTaskPendingWait:
    def test_cancel_during_pending_releases_worker_promptly(self):
        """A task cancelled mid-debounce returns within ~one poll interval, not after the full pending wait."""
        ran = []
        fn = bgtask.make_managed_task(status_box=box(bgtask.status_stopped),
                                      lock=threading.Lock(),
                                      entrypoint=lambda task_env: ran.append(True),
                                      running_poll_interval=0.02,
                                      pending_wait_duration=10.0)  # long, so a single-sleep impl would block ~10 s
        task_env = _make_env(wait=True)
        thread = threading.Thread(target=fn, args=(task_env,))
        t0 = time.monotonic()
        thread.start()
        time.sleep(0.1)               # let it enter the pending state
        task_env.cancelled = True     # supersede it (as a newer submission would)
        thread.join(timeout=2.0)
        elapsed = time.monotonic() - t0

        assert not thread.is_alive()  # returned, didn't hang for the full 10 s pending wait
        assert elapsed < 1.0          # ... and promptly (chunked wait), nowhere near pending_wait_duration
        assert not ran                # entrypoint never ran (cancelled before the wait elapsed)

    def test_uncancelled_pending_runs_the_entrypoint(self):
        """The happy path still works: a task left alone through the (short) pending wait runs its entrypoint."""
        ran = []
        fn = bgtask.make_managed_task(status_box=box(bgtask.status_stopped),
                                      lock=threading.Lock(),
                                      entrypoint=lambda task_env: ran.append(True),
                                      running_poll_interval=0.02,
                                      pending_wait_duration=0.1)
        task_env = _make_env(wait=True)
        thread = threading.Thread(target=fn, args=(task_env,))
        thread.start()
        thread.join(timeout=2.0)

        assert not thread.is_alive()
        assert ran  # entrypoint ran after the pending wait elapsed without cancellation


class TestOnCancelHook:
    """Cancelling work that cannot poll the flag, because it is blocked inside a library call.

    The flag alone is only readable between steps, so a task blocked in a socket read stays blocked until
    the read returns on its own. `on_cancel` is where such a task supplies the operation that ends the wait
    from outside; the task then observes `cancelled` and unwinds as it always did.
    """

    @staticmethod
    def _manager():
        return bgtask.TaskManager(name="test_on_cancel", mode="concurrent",
                                  executor=concurrent.futures.ThreadPoolExecutor(max_workers=2))

    def test_on_cancel_runs_and_sees_the_flag_already_set(self):
        """Order matters: the hook only unblocks the task, so what the task wakes to must already say cancelled."""
        seen = {}
        manager = self._manager()
        blocked = threading.Event()
        release = threading.Event()

        def task(task_env):
            blocked.set()
            release.wait(timeout=5.0)  # stands in for a blocking library call

        task_env = env(on_cancel=lambda e: (seen.__setitem__("cancelled_when_called", e.cancelled),
                                            release.set()))
        manager.submit(task, task_env)
        assert blocked.wait(timeout=5.0)
        manager.clear(wait=True)

        assert seen["cancelled_when_called"] is True, "`on_cancel` ran before the flag was set"

    def test_the_hook_is_what_ends_the_wait(self):
        """The negative control: without a hook, the same task is still blocked when cancellation returns.

        Otherwise this fixture could not tell a hook that unblocked the task from a task that was never
        blocked in the first place.
        """
        manager = self._manager()
        blocked = threading.Event()
        release = threading.Event()
        finished = threading.Event()

        def task(task_env):
            blocked.set()
            release.wait(timeout=5.0)
            finished.set()

        manager.submit(task, env())  # no `on_cancel`
        assert blocked.wait(timeout=5.0)
        manager.clear(wait=False)
        assert not finished.wait(timeout=0.3), ("the task ended without anything unblocking it, so this "
                                                "fixture cannot tell the hook's effect from its absence")
        release.set()  # let it go, so the pool worker is not left pinned

    def _blocked_task(self):
        """A task that stays in flight until released, so there is something left to cancel.

        A task that returns immediately is popped from the manager by its own done-callback, and cancelling
        it is then correctly a no-op — which would make either test below pass for the wrong reason.
        """
        blocked = threading.Event()
        release = threading.Event()

        def task(task_env):
            blocked.set()
            release.wait(timeout=5.0)

        return task, blocked, release

    def test_a_raising_hook_does_not_break_cancellation(self):
        manager = self._manager()
        task, blocked, release = self._blocked_task()

        def boom(task_env):
            raise RuntimeError("hook failure")

        task_env = env(on_cancel=boom)
        manager.submit(task, task_env)
        assert blocked.wait(timeout=5.0)
        manager.clear(wait=False)  # must not propagate the hook's exception
        assert task_env.cancelled  # cancellation completed regardless
        release.set()

    def test_a_task_without_the_hook_is_unaffected(self):
        """`on_cancel` is optional, and its absence must not be an AttributeError during cancellation."""
        manager = self._manager()
        task, blocked, release = self._blocked_task()
        task_env = env()
        manager.submit(task, task_env)
        assert blocked.wait(timeout=5.0)
        manager.clear(wait=False)
        assert task_env.cancelled
        release.set()

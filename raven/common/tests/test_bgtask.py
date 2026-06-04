"""Tests for raven.common.bgtask — background task management.

Focused on the `ManagedTask` pending-wait invariant: a task superseded during its debounce must release its
worker promptly (within ~one `running_poll_interval`), not pin it for the whole `pending_wait_duration`. This
matters for long debounces with rapid resubmission, where the old "single `time.sleep`" would stack one pinned
pool worker per cancelled task.
"""

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

__all__ = ["SequentialTaskManager",
           "make_managed_task",
           "status_stopped", "status_pending", "status_running"]

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

import threading
import time
import traceback

from unpythonic import gensym, sym, unbox

# --------------------------------------------------------------------------------
# Background task manager

status_stopped = sym("stopped")
status_pending = sym("pending")
status_running = sym("running")

class SequentialTaskManager:
    def __init__(self, name, executor):
        """Simple manager for there-can-be-only-one (of each kind) background tasks, for GUI updates.

        Tasks can be submitted concurrently, but when a new one is submitted, all previous tasks are automatically cancelled.
        If you have several kinds of background tasks, create one `SequentialTaskManager` instance for each kind.
        You can use the same `ThreadPoolExecutor` in all of them.

        This is useful when the underlying data changes so that completing the old, running GUI update task
        no longer makes sense. Raven uses this for handling annotation tooltip and info panel updates.

        In practice this is a rather thin wrapper over `ThreadPoolExecutor`, adding the auto-cancel mechanism.

        See also our friend function `make_managed_task`.

        `name`: Name for this task manager, lowercase with underscores recommended.
                This is included in the automatically generated names for the tasks,
                to easily distinguish tasks from different managers in log messages.
        `executor`: A `ThreadPoolExecutor` that actually manages the concurrent execution.
        """
        self.name = name  # used in generated task names
        self.executor = executor
        self.tasks = {}  # task name (unique) -> (future, env)
        self.lock = threading.RLock()

    def submit(self, function, env):
        """Submit a new task.

        `function`: callable, must take one positional argument.
        `env`: `unpythonic.env.env`, passed to `function` as the only argument.

               When `submit` returns, `env` will contain two new attributes:

                   `task_name`: str, unique name of the task, for use in log messages.

                   `cancelled`: bool. This flag signals task cancellation.

                                The `function` must monitor this flag, and terminate
                                as soon as conveniently possible if the flag ever
                                becomes `True`.

                                Note that a task may become cancelled before it even starts,
                                if there are not enough resources (threads in the pool) to
                                attempt to start them all simultaneously. This happens
                                especially often if `function` needs to wait internally
                                before it can actually start doing its job.

                                Then both the task that is actively waiting, and any tasks
                                added later to the queue (waiting for a free thread in the pool)
                                may have their `cancelled` flags set to `True`.

        Returns an `unpythonic.gsym` representing the task name. Task names are unique.
        """
        with self.lock:
            self.clear()
            env.task_name = gensym(f"{self.name}_task")
            env.cancelled = False
            future = self.executor.submit(function, env)
            self.tasks[env.task_name] = (future, env)  # store a reference to `env` so we have access to the `cancelled` flag
            future.add_done_callback(self._done_callback)  # autoremove the task when it exits (we don't need its return value)
            return env.task_name

    def _find_task_by_future(self, future):
        """Internal method. Find the `task_name` for a given `future`. Return `task_name`, or `None` if not found."""
        with self.lock:
            for task_name, (f, e) in self.tasks.items():
                if f is future:
                    return task_name
            return None

    def _done_callback(self, future):
        """Internal method. Remove a completed task, by a reference to its `future` (that we get from `ThreadPoolExecutor`)."""
        with self.lock:
            task_name = self._find_task_by_future(future)
            if task_name is not None:  # not removed already? (`cancel` might have removed it)
                self.tasks.pop(task_name)

    def cancel(self, task_name, pop=True):
        """Cancel a specific task, by name.

        Usually there is no need to call this manually.

        `task_name`: `unpythonic.gsym`, the return value from `submit`.
        `pop`: bool, whether to remove the task from `self.tasks`.
               Default is `True`, which is almost always the right thing to do.
               The option is provided mainly for internal use by `clear`.
        """
        with self.lock:
            if task_name not in self.tasks:
                raise ValueError(f"SequentialTaskManager.cancel_task: instance '{self.name}': no such task '{task_name}'")
            if pop:
                future, e = self.tasks.pop(task_name)
            else:
                future, e = self.tasks[task_name]
            future.cancel()  # in case it's still queued in the executor (don't start it)
            e.cancelled = True  # in case it's running (pythonic co-operative cancellation)

    def clear(self, wait=False):
        """Cancel all tasks.

        During normal operation, usually there is no need to call this manually, but can be useful during app shutdown.

        `wait`: Whether to wait for all tasks to exit before returning.
        """
        with self.lock:
            for task_name in list(self.tasks.keys()):
                self.cancel(task_name, pop=False)
            if wait:
                while not all(future.done() for future, e in self.tasks.values()):
                    time.sleep(0.01)
            self.tasks.clear()

def make_managed_task(*, status_box, lock, entrypoint, running_poll_interval, pending_wait_duration):
    """Create a background task that makes double-sure that only one instance is running at a time.

    This works together with `SequentialTaskManager`, adding on top of it mechanisms to track the status for
    the same kind of tasks (for displaying spinners in the GUI), as well as a lock to ensure that only one call
    to `entrypoint` can be running at a time.

    `status_box`: An `unpythonic.box` that stores the task status flag. At any one time, the flag itself is one of:
                  `status_stopped`: No task (of this kind) in progress.
                  `status_pending`: A task has entered the pending state (see below).
                  `status_running`: A task is currently running.
    `lock`: A `threading.Lock`, to control that at most one call to `entrypoint` is running at a time.
    `entrypoint`: callable, the task itself. It must accept a kwarg `task_env`, used for sending in the task environment
                  (`unpythonic.env.env`), see below.
    `running_poll_interval`: float, when the task starts, it polls for the task status flag until there is no previous task running.
                             This parameter sets the poll interval.
    `pending_wait_duration`: float, once the polling finishes, the task enters the pending state.

                             This parameter sets how long the task waits in the pending state before it starts running.

                             This mechanism is used by e.g. the search field to wait for more keyboard input before starting to update
                             the info panel. Each keystroke in the search field submits a new managed task to the relevant
                             `SequentialTaskManager`. If more input occurs during the pending state, the manager cancels any
                             previous tasks, so that only the most recently submitted one remains.

                             Only if the pending step completes successfully (task not cancelled during `pending_wait_duration`),
                             the manager proceeds to acquire `lock`, changes the task status to `running`, and calls the task's
                             `entrypoint`.

                             Once `entrypoint` exits:

                                 If the `cancelled` flag is not set in the task environment (i.e. the task ran to completion),
                                 the manager changes the task status to `stopped`.

                                 If the `cancelled` flag is set in the task environment, the manager changes the task status
                                 to pending (because in Raven, cancellation only occurs when replaced by a new task of the same kind).

    Returns a 1-argument function, which can be submitted to a `SequentialTaskManager`.
    The function's argument is an `unpythonic.env.env`, representing the task environment.

    The environment has one mandatory attribute, `wait` (bool), that MUST be filled in by the task submitter:
        If `wait=True`, the task uses the pending state mechanism as described above.
        If `wait=False`, the pending state is skipped, and the task starts running as soon as all previous tasks
                         of the same kind have exited.

    When `entrypoint` is entered, the task environment (sent in as the kwarg `task_env`) will contain two more attributes,
    filled in by `SequentialTaskManager`:
        `task_name`: `unpythonic.gsym`, unique task name. This is the return value from `SequentialTaskManager.submit`,
                      thereby made visible for the task itself, for use in log messages.
        `cancelled`: bool, co-operative cancellation flag. The task must monitor this flag, and if it ever becomes `True`,
                     exit as soon as reasonably possible.

    Currently, the task submitter is allowed to create and use any other attributes to pass custom data into the task.
    """
    # TODO: This function is too spammy even for debug logging, needs a "detailed debug" log level.
    def _managed_task(env):
        # logger.debug(f"_managed_task: {env.task_name}: setup")
        # The task might be cancelled before this function is even entered. This happens if there are many tasks
        # (more than processing threads) in the queue, since submitting a new task to a `SequentialTaskManager` cancels all previous ones.
        if env.cancelled:
            # logger.debug(f"_managed_task: {env.task_name}: cancelled (from task queue)")
            return
        while unbox(status_box) is status_running:  # wait for the previous task of the same kind to finish, if already running
            time.sleep(running_poll_interval)
        if env.cancelled:
            # logger.debug(f"_managed_task: {env.task_name}: cancelled (from initial wait state)")
            return
        if env.wait:  # The pending mechanism is optional, so it can be disabled in use cases where "wait for more input" doesn't make sense.
            # logger.debug(f"_managed_task: {env.task_name}: pending")
            status_box << status_pending
            time.sleep(pending_wait_duration)  # [s], cancellation period
            if env.cancelled:  # Note we only cancel a task when it has been obsoleted by a newer one. So the task status is still pending.
                # logger.debug(f"_managed_task: {env.task_name}: cancelled (from pending state)")
                return
        # logger.debug(f"_managed_task: {env.task_name}: acquiring lock for this kind of task")
        with lock:
            try:
                # logger.debug(f"_managed_task: {env.task_name}: starting")
                status_box << status_running
                entrypoint(task_env=env)
            except Exception as exc:  # VERY IMPORTANT, to not silently swallow uncaught exceptions from background tasks
                logger.warning(f"_managed_task: {env.task_name}: exited with exception {type(exc)}: {exc}")
                traceback.print_exc()  # DEBUG
                raise
            # else:
            #     logger.debug(f"_managed_task: {env.task_name}: exited with status {'OK' if not env.cancelled else 'CANCELLED (from running state)'}")
            finally:
                if not env.cancelled:
                    status_box << status_stopped
                else:
                    status_box << status_pending
        # logger.debug(f"_managed_task: {env.task_name}: all done")
    return _managed_task

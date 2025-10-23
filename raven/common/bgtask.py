__all__ = ["TaskManager",
           "make_managed_task",
           "status_stopped", "status_pending", "status_running"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

import concurrent.futures
import threading
import time
import traceback
from typing import Callable, Union

from unpythonic import box, gensym, gsym, sym, unbox
from unpythonic.env import env

# --------------------------------------------------------------------------------
# Background task manager

status_stopped = sym("stopped")
status_pending = sym("pending")
status_running = sym("running")

class TaskManager:
    def __init__(self, name: str, mode: str, executor: concurrent.futures.Executor):
        """Simple background task manager.

        In practice this is a rather thin wrapper over `ThreadPoolExecutor`, adding task tracking
        and an optional auto-cancel mechanism.

        When using the auto-cancel mechanism, tasks can be submitted concurrently, but when a new one
        is submitted, all previous tasks are automatically cancelled. If you have several kinds of
        background tasks, create one `TaskManager` instance for each kind. You can use the same
        `ThreadPoolExecutor` in all of them.

        This is useful when the underlying data changes so that completing an old, running GUI update task
        no longer makes sense. Raven-visualizer uses this for handling annotation tooltip and info panel updates.

        See also our friend function `make_managed_task`.

        `name`: Name for this task manager, lowercase with underscores recommended.
                This is included in the automatically generated names for the tasks,
                to easily distinguish tasks from different managers in log messages.
        `mode`: str, one of:
                    "concurrent": Plain task manager, just add task tracking on top of the executor.
                    "sequential": There Can Be Only One background task. Useful for GUI updates,
                                  where submitting a new task invalidates (and should cancel)
                                  any earlier ones. In this mode, `TaskManager` takes care of
                                  cancelling the old tasks.

        `executor`: A `ThreadPoolExecutor` that actually manages the concurrent execution.
        """
        if mode not in ("concurrent", "sequential"):
            raise ValueError(f"Unknown mode '{mode}'; valid values: 'concurrent', 'sequential'.")

        self.name = name  # used in generated task names
        self.mode = mode
        self.executor = executor
        self.tasks = {}  # task name (unique) -> (future, env)
        self.lock = threading.RLock()

    def submit(self, function: Callable, env: env) -> gsym:
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

               When `function` exits or the task is cancelled (i.e. when the future becomes done),
               if `env` contains an attribute `done_callback` at that time, that function
               will be called with `env` as its only argument. This mechanism has two primary uses:

                 1) Returning a value from the task asynchronously, with convenient instance tracking
                    (each activation of `function` has its own `env`).

                 2) Division of responsibilities. For example, `function` may live in a backend module,
                    while a GUI module that calls into that backend needs to perform its own cleanup
                    (e.g. reset the state of some GUI widgets) when the background task exits.

               Note that any cleanup of resources used by the task internally is better done
               in the task body (as usual, using a `try/finally` or a `with`).

               To return a value from the background task, provide an `env.done_callback`
               when calling `submit`. Then, in `function`, stash the return value as,
               for example, `env.ret`. The done callback will have access to `env`,
               so it can get the return value from there, and do what it wants with it.

               To check in your `done_callback` whether the task completed or was cancelled,
               check the `env.cancelled` flag. If you need more fine-grained status (e.g.
               exception info if the task exited due to an exception), you can stash that
               in `env` manually, from the task body.

               The return value of `done_callback` itself is ignored.

               For the purpose of tracking tasks in this manager, `done_callback` is considered
               to be part of the dynamic extent of the task. That is, for a task that has a
               `done_callback`, upon the completion or cancellation of that task, the reference
               to that task is only removed from the manager *after* the `done_callback` exits.

        Returns an `unpythonic.gsym` representing the task name. Task names are unique.
        """
        with self.lock:
            if self.mode == "sequential":  # There Can Be Only One background task in this manager
                self.clear()
            env.task_name = gensym(f"{self.name}_task")
            env.cancelled = False
            future = self.executor.submit(function, env)
            self.tasks[env.task_name] = (future, env)  # store a reference to `env` so we have access to the `cancelled` flag and the custom `done_callback`, if any
            future.add_done_callback(self._done_callback)  # autoremove the task when it exits (and let the user handle its return value, if any)
            logger.info(f"TaskManager.submit: instance '{self.name}': task '{env.task_name}' submitted.")
            return env.task_name

    def has_tasks(self) -> bool:
        """Return whether this task manager is currently tracking any tasks."""
        with self.lock:
            return len(self.tasks)

    def _find_task_by_future(self, future: concurrent.futures.Future) -> gsym:
        """Internal method. Find the `task_name` for a given `future`. Return `task_name`, or `None` if not found."""
        with self.lock:
            for task_name, (f, e) in self.tasks.items():
                if f is future:
                    return task_name
            return None

    def _done_callback(self, future: concurrent.futures.Future) -> None:
        """Internal method. Remove a completed task, by a reference to its `future` (that we get from `ThreadPoolExecutor`).

        Before removing the task, automatically call the custom `done_callback`, if it was provided.
        Note that this triggers also when the task is cancelled, triggering the custom callback also in that case.
        """
        logger.debug(f"TaskManager._done_callback: instance '{self.name}': called for future '{future}'.")
        logger.debug(f"TaskManager._done_callback: instance '{self.name}': task list now: {self.tasks}.")

        # Avoid silently swallowing exceptions from background tasks
        try:
            exc = future.exception()  # the future exited already, so we don't need to set a timeout
        except concurrent.futures.CancelledError:
            pass
        else:
            if exc is not None:
                logger.error(f"TaskManager._done_callback: instance '{self.name}': future '{future}' exited with exception {type(exc)}: {exc}")
                traceback.print_exc()

        with self.lock:
            task_name = self._find_task_by_future(future)
            logger.debug(f"TaskManager._done_callback: instance '{self.name}': task lookup for future '{future}' returned '{task_name}'.")
            if task_name is not None:  # not removed already? (`cancel` might have removed it)
                logger.info(f"TaskManager._done_callback: instance '{self.name}': '{task_name}' finalizing.")

                # Call the custom done callback if provided.
                #
                # NOTE: We remove the task *after* calling the custom `done_callback`, so that the task still shows as running
                # in our status until the `done_callback` has exited. The `done_callback` is considered to be part of the task.
                # Strictly speaking, it is - it's a CPS continuation.
                try:
                    future, e = self.tasks[task_name]
                    if "done_callback" in e and e.done_callback is not None:
                        logger.info(f"TaskManager._done_callback: instance '{self.name}': {task_name}: custom `done_callback` exists, calling it now.")
                        e.done_callback(e)
                finally:
                    self.tasks.pop(task_name)

    def cancel(self, task_name: gsym, pop: bool = True) -> None:
        """Cancel a specific task, by name.

        Usually there is no need to call this manually.

        `task_name`: `unpythonic.gsym`, the return value from `submit`.
        `pop`: bool, whether to remove the task from `self.tasks`.
               Default is `True`, which is almost always the right thing to do.
               The option is provided mainly for internal use by `clear`.

        Raises `ValueError` if no task with `task_name` was found.
        """
        logger.info(f"TaskManager.cancel: instance '{self.name}': cancelling task '{task_name}'.")
        with self.lock:
            if task_name not in self.tasks:
                raise ValueError(f"TaskManager.cancel_task: instance '{self.name}': no such task '{task_name}'")
            if pop:
                future, e = self.tasks.pop(task_name)
            else:
                future, e = self.tasks[task_name]
            e.cancelled = True  # In case it's running (pythonic co-operative cancellation). Do this first, so the custom `done_callback` (if any) sees the `cancelled` flag when we cancel the future.
            future.cancel()  # In case it's still queued in the executor, don't start it.

    def clear(self, wait: bool = False) -> None:
        """Cancel all tasks.

        During normal operation, usually there is no need to call this manually, but can be useful during app shutdown.

        `wait`: Whether to wait for all tasks to exit before returning.
        """
        logger.info(f"TaskManager.clear: instance '{self.name}': cancelling all tasks.")
        with self.lock:
            for task_name in list(self.tasks.keys()):
                self.cancel(task_name, pop=False)
        # Release the lock while we wait so that `_done_callback` can access the task list before we clear it.
        # TODO: We assume that `Future` will call the `done_callback` before marking itself as `done`. Should check the docs or source code for whether this is the case.
        if wait:
            logger.info(f"TaskManager.clear: instance '{self.name}': waiting for tasks to exit.")
            while not all(future.done() for future, e in self.tasks.values()):
                time.sleep(0.01)
        with self.lock:
            logger.info(f"TaskManager.clear: instance '{self.name}': clearing task list.")
            self.tasks.clear()

def make_managed_task(*,
                      status_box: box,
                      lock: Union[threading.Lock, threading.RLock],
                      entrypoint: Callable,
                      running_poll_interval: float,
                      pending_wait_duration: float) -> Callable:
    """Create a background task that makes double-sure that only one instance is running at a time.

    This works together with `TaskManager`, adding on top of it mechanisms to track the status for
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
                             `TaskManager`. If more input occurs during the pending state, the manager cancels any
                             previous tasks, so that only the most recently submitted one remains.

                             Only if the pending step completes successfully (task not cancelled during `pending_wait_duration`),
                             the manager proceeds to acquire `lock`, changes the task status to `running`, and calls the task's
                             `entrypoint`.

                             Once `entrypoint` exits:

                                 If the `cancelled` flag is not set in the task environment (i.e. the task ran to completion),
                                 the manager changes the task status to `stopped`.

                                 If the `cancelled` flag is set in the task environment, the manager changes the task status
                                 to pending (because in Raven, cancellation only occurs when replaced by a new task of the same kind).

    Returns a 1-argument function, which can be submitted to a `TaskManager`.
    The function's argument is an `unpythonic.env.env`, representing the task environment.

    The environment has one mandatory attribute, `wait` (bool), that MUST be filled in by the task submitter:
        If `wait=True`, the task uses the pending state mechanism as described above.
        If `wait=False`, the pending state is skipped, and the task starts running as soon as all previous tasks
                         of the same kind have exited.

    When `entrypoint` is entered, the task environment (sent in as the kwarg `task_env`) will contain two more attributes,
    filled in by `TaskManager`:
        `task_name`: `unpythonic.gsym`, unique task name. This is the return value from `TaskManager.submit`,
                      thereby made visible for the task itself, for use in log messages.
        `cancelled`: bool, co-operative cancellation flag. The task must monitor this flag, and if it ever becomes `True`,
                     exit as soon as reasonably possible.

    Currently, the task submitter is allowed to create and use any other attributes to pass custom data into the task.
    """
    # TODO: This function is too spammy even for debug logging, needs a "detailed debug" log level.
    def _managed_task(env):
        # logger.debug(f"_managed_task: {env.task_name}: setup")
        # The task might be cancelled before this function is even entered. This happens if there are many tasks
        # (more than processing threads) in the queue, since submitting a new task to a `TaskManager` in sequential mode
        # cancels all previous ones.
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
            # # Used to be VERY IMPORTANT, to not silently swallow uncaught exceptions from background task.
            # # But now `TaskManager._done_callback` does this.
            # except Exception as exc:
            #     logger.warning(f"_managed_task: {env.task_name}: exited with exception {type(exc)}: {exc}")
            #     traceback.print_exc()  # DEBUG; `TaskManager._done_callback` now does this.
            #     raise
            # else:
            #     logger.debug(f"_managed_task: {env.task_name}: exited with status {'OK' if not env.cancelled else 'CANCELLED (from running state)'}")
            finally:
                if not env.cancelled:
                    status_box << status_stopped
                else:
                    status_box << status_pending
        # logger.debug(f"_managed_task: {env.task_name}: all done")
    return _managed_task

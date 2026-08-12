"""One writer at a time for a file two apps can both open.

Raven-librarian and Raven-minichat share one chat datastore, and each holds the whole thing in memory and
writes it back at exit. Run both and the one that exits last wins: everything the other did is gone, with
nothing to indicate it happened. Two Librarians do the same. The lock makes the second one refuse to start
rather than quietly destroy the first one's session.

The lock is *advisory between Raven apps* and nothing more. It does not protect the file from an editor, a
backup tool or a script, and it is not a substitute for the in-process `threading.RLock` a `Forest` already
takes — that one serializes threads within an app, this one serializes apps.
"""

__all__ = ["DatastoreBusyError", "lock_path_for", "acquire"]

import hashlib
import logging
import os
import pathlib
import tempfile
from typing import Union

import filelock

logger = logging.getLogger(__name__)


class DatastoreBusyError(RuntimeError):
    """Another process already has the datastore open."""


def lock_path_for(target: Union[str, pathlib.Path]) -> pathlib.Path:
    """Return the path of the lock file guarding `target`.

    In the system temp directory rather than beside `target`, for two reasons. The lock is runtime state
    about which process is using a file, not part of the user's data, and Librarian offers a button that
    opens the datastore folder in a file manager — a stray `.lock` there is litter in a place the user
    looks. And on the Linux machines this is developed on, the temp directory is a ramdisk, so a lock cannot
    outlive the boot that created it.

    The name is derived from `target`'s resolved absolute path, so two different datastores get two
    different locks, and the same datastore reached by two different relative paths gets one.
    """
    resolved = str(pathlib.Path(target).expanduser().resolve())
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]
    return pathlib.Path(tempfile.gettempdir()) / f"raven-datastore-{digest}.lock"


def acquire(target: Union[str, pathlib.Path], what: str) -> filelock.FileLock:
    """Take the lock guarding `target`, or raise `DatastoreBusyError` if another process has it.

    Returns the held lock. **Keep the returned object alive for as long as the datastore is open** — a lock
    that gets garbage collected releases. Callers that hold it for the whole run can simply bind it to a
    module-level name; there is no need to release it explicitly, since the operating system drops the lock
    when the process exits, including when it crashes. That is the whole reason for locking the file rather
    than writing a PID into it: there is no such thing as a stale lock to detect, clean up, or override.

    `target`: the datastore file to guard. It need not exist yet — a first run creates it, and two first
              runs racing each other are exactly what this prevents.

    `what`: what is being guarded, for the error message: "the chat datastore", "the dataset". Named from
            the user's vocabulary rather than the code's, since this is what they will read.
    """
    lock_path = lock_path_for(target)
    lock = filelock.FileLock(str(lock_path), timeout=0)
    try:
        lock.acquire()
    except filelock.Timeout:
        logger.error(f"acquire: {what} at '{target}' is already open in another Raven app (lock file '{lock_path}').")
        raise DatastoreBusyError(f"{what} at '{target}' is already open in another Raven app. "
                                 f"Close the other one (Raven-librarian or Raven-minichat) and try again.")
    logger.info(f"acquire: locked {what} at '{target}' (lock file '{lock_path}', PID {os.getpid()}).")
    return lock

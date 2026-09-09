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
import sys
import tempfile
from typing import Union

import filelock

logger = logging.getLogger(__name__)


class DatastoreBusyError(RuntimeError):
    """Another process already has the datastore open."""


#: What a refusal says when the lock file does not name its holder. See `_read_holder` for when that happens.
_UNKNOWN_HOLDER = "another Raven app"


def _describe_self() -> str:
    """Name this process the way its user would: the command they typed, and a PID to `kill` if need be."""
    # `sys.argv[0]` is the console script — `raven-indexer`, `raven-librarian` — so this stays correct as
    # apps are added, where a hand-maintained list in this module would not. It was one: the refusal used
    # to name "Raven-librarian or Raven-minichat", which went stale the moment the document index became
    # lockable too, since `raven-indexer` holds that one.
    #
    # It is not always a filename, though, and the PID is worth having even when the name is not: `argv[0]`
    # is `''` in a REPL, `-c` for `python -c`, and `-` for a script on stdin. Anything starting with a dash
    # would be printed to the user as if it were a program's name, so those become the generic phrase.
    name = pathlib.Path(sys.argv[0]).name
    if not name or name.startswith("-"):
        name = _UNKNOWN_HOLDER
    return f"{name} (PID {os.getpid()})"


def _write_holder(lock_path: pathlib.Path) -> None:
    """Record who holds the lock, for the benefit of whoever is refused next. Call this holding it.

    Safe to write into the lock file itself: the lock is the `flock`, not the file's contents. `filelock`
    opens without `O_TRUNC` and truncates only after winning, so a process that loses the race cannot
    blank out the winner's identity on its way past.
    """
    try:
        lock_path.write_text(_describe_self(), encoding="utf-8")
    except OSError:  # noqa: S110 -- see below
        # Never fatal. This is a nicety for an error message that may never be printed, and the lock is
        # held and valid whether or not it succeeds. A caller that failed to start because it could not
        # write a *comment* would be an absurd way to lose a session.
        logger.debug(f"_write_holder: could not record holder in '{lock_path}'; refusals will be vaguer.",
                     exc_info=True)


def _read_holder(lock_path: pathlib.Path) -> str:
    """Return a description of the process holding `lock_path`, or `_UNKNOWN_HOLDER` if it does not say.

    Blank is a normal answer rather than an error: the holder truncates the file when it wins and writes
    its name a moment later, so a refusal landing in that window reads nothing. An older `filelock` that
    truncates on open gets the same result for a different reason.
    """
    try:
        holder = lock_path.read_text(encoding="utf-8").strip()
    except OSError:
        return _UNKNOWN_HOLDER
    return holder or _UNKNOWN_HOLDER


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
        holder = _read_holder(lock_path)
        # Refusing *ourselves* is a different situation and deserves a different sentence. `flock` is per
        # open file description rather than per process, so a second `acquire` on the same target in one
        # process is refused exactly as another process would be — and "close the other one" is then advice
        # nobody can act on, since the other one is you. It means something opened the store twice, which
        # is a bug in the caller rather than a busy resource.
        if holder.endswith(f"(PID {os.getpid()})"):
            logger.error(f"acquire: {what} at '{target}' is already open in this same process ({holder}); "
                         f"it has been opened twice (lock file '{lock_path}').")
            raise DatastoreBusyError(f"{what} at '{target}' is already open in this same process "
                                     f"({holder}) — it has been opened twice.")
        logger.error(f"acquire: {what} at '{target}' is already open in {holder} (lock file '{lock_path}').")
        raise DatastoreBusyError(f"{what} at '{target}' is already open in {holder}. "
                                 f"Close that one and try again.")
    _write_holder(lock_path)
    logger.info(f"acquire: locked {what} at '{target}' (lock file '{lock_path}', PID {os.getpid()}).")
    return lock

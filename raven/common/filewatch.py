"""Follow the files in a directory: what changed while we were away, and what changes while we watch.

`DirectoryWatcher` reports each change as a *net* effect on one file — created, modified or deleted — by
its path relative to the watched directory, however the change was made. A save that writes a temporary
file and renames it over the original is a modification of the original, and the temporary file is never
reported. A symlinked file is followed to its target, so editing the target is a modification of the link.

This module is licensed under the 2-clause BSD license, to facilitate integration anywhere.
"""

__all__ = ["CREATED", "MODIFIED", "DELETED",
           "Signature", "signature_of",
           "DirectoryWatcher", "watching"]

import contextlib
import logging
import os
import pathlib
import stat
import threading
import time
from typing import Callable, Iterator, Mapping

import watchdog.events
import watchdog.observers

from . import utils as common_utils

logger = logging.getLogger(__name__)

CREATED = "created"
MODIFIED = "modified"
DELETED = "deleted"

Signature = tuple[float, int]  # (st_mtime, st_size), read through symlinks

def signature_of(path: str | pathlib.Path) -> Signature | None:
    """`(mtime, size)` of the regular file at `path`, following symlinks; `None` if there is none."""
    try:
        info = os.stat(path)  # follows symlinks: a link reports its target
    except OSError:  # absent, a broken link, or unreadable
        return None
    if not stat.S_ISREG(info.st_mode):
        return None
    return (info.st_mtime, info.st_size)


class DirectoryWatcher:
    """Report created, modified and deleted files in `directory`, live and against a reference snapshot.

    `directory`: what to watch. May itself be reached through a symlink, resolved when the watcher starts.

    `on_changed`: called as `on_changed(relative_path, kind)`, `relative_path` being a `pathlib.Path`
                  relative to `directory` and `kind` one of `CREATED`, `MODIFIED`, `DELETED`. Changes found
                  by `start` are reported on the thread calling it; live ones on the watcher's own thread,
                  one at a time.

    `accept`: which files count, called with the file's full path. Default: every file. Applied to the
              path in the watched directory, not to a symlink's target.

    `recursive`: also watch subdirectories. A subdirectory that is a **symlink is not followed**, neither
                 when scanning nor while watching: only symlinked *files* are. (Linking the watched
                 directory itself somewhere is fine.)

    `quiet_period`: seconds without new events before pending changes are reported. A save made of
                    several filesystem operations then arrives as one change, and a temporary file created
                    and renamed away within it is not reported at all.

    `max_delay`: the most seconds a pending change waits, however busy the directory is.

    `observer_timeout`: watchdog's own polling interval, in seconds. Its Windows backend delivers events only
                        after this long without new ones, and stopping waits up to this long, so a smaller
                        value is faster there at the price of more idle wakeups. Other backends barely notice.

    A file is modified when its mtime or size differs from the last one seen, either way: a file restored
    from a backup, with an older mtime, is a modification too.
    """
    # How it works, in one paragraph: watchdog's events are used only as hints, marking a path dirty. Once
    # the directory goes quiet, each dirty path is stat'ed and compared with the snapshot, and the difference
    # is what gets reported. So the event sequence an editor or a sync tool happens to produce — create,
    # modify, move, delete, in any order and on any name — does not have to be interpreted, only noticed.
    # A symlinked file's target lives elsewhere, and inotify on the directory sees nothing when it changes,
    # so each target's directory gets a watch of its own, with a map from target back to link.
    def __init__(self,
                 directory: str | pathlib.Path,
                 *,
                 on_changed: Callable[[pathlib.Path, str], None],
                 accept: Callable[[pathlib.Path], bool] | None = None,
                 recursive: bool = False,
                 quiet_period: float = 0.3,
                 max_delay: float = 5.0,
                 observer_timeout: float = 1.0) -> None:
        self.directory = common_utils.canonical_path(directory)  # not symlink-resolved: the name we were given
        self.on_changed = on_changed
        self.accept = accept if accept is not None else (lambda path: True)
        self.recursive = recursive
        self.quiet_period = quiet_period
        self.max_delay = max_delay
        self.observer_timeout = observer_timeout

        self._resolved_directory = self.directory.resolve()  # where events can also arrive from
        self._lock = threading.Lock()
        self._wakeup = threading.Condition(self._lock)
        self._snapshot: dict[pathlib.Path, Signature] = {}
        self._dirty: set[pathlib.Path] = set()
        self._first_dirty_time: float | None = None
        self._last_event_time = 0.0
        self._links: dict[pathlib.Path, set[pathlib.Path]] = {}  # resolved target -> relative link paths
        self._target_watches: dict[pathlib.Path, watchdog.observers.api.ObservedWatch] = {}  # target dir -> watch
        self._maybe_observer: watchdog.observers.Observer | None = None
        self._maybe_handler: _Handler | None = None
        self._maybe_settler: threading.Thread | None = None
        self._stopping = False

    def _get_snapshot(self) -> dict[pathlib.Path, Signature]:
        """Copy, so that a caller cannot edit the watcher's record."""
        with self._lock:
            return dict(self._snapshot)
    snapshot = property(fget=_get_snapshot, doc="The files currently known, as `relative path -> Signature`.")

    def start(self, *, known: Mapping[str | pathlib.Path, Signature] | None = None) -> None:
        """Scan the directory, report what differs from `known`, and start watching.

        `known`: the files as they were last seen — typically saved by the caller at the end of an earlier
                 session — as `relative path -> (mtime, size)`. Each difference from the directory as it is
                 now is reported before this returns. `None` reports nothing, and only starts watching.
        """
        if self._maybe_observer is not None:
            raise RuntimeError("DirectoryWatcher.start: already started")
        self._resolved_directory = self.directory.resolve()
        handler = _Handler(self)
        observer = watchdog.observers.Observer(timeout=self.observer_timeout)
        # The resolved directory, because watchdog's inotify backend does not follow a symlinked watch path,
        # and reports nothing at all for it. Resolved here, at start, so a link repointed later is not followed.
        observer.schedule(handler, path=str(self._resolved_directory), recursive=self.recursive)
        # Watching starts before the scan, so that nothing done during the scan is missed: an event arriving
        # now only marks a path dirty, and the settle that follows compares against what the scan saw.
        observer.start()
        self._maybe_observer = observer
        self._maybe_handler = handler

        current = self._scan()
        with self._lock:
            self._snapshot = dict(current)
            self._update_links()
        if known is not None:
            previous = {pathlib.Path(path): signature for path, signature in known.items()}
            for relative_path in sorted(previous.keys() | current.keys()):
                kind = _difference(previous.get(relative_path), current.get(relative_path))
                if kind is not None:
                    self._notify(relative_path, kind)

        self._maybe_settler = threading.Thread(target=self._settle_loop,
                                               name=f"DirectoryWatcher_{self.directory.name}",
                                               daemon=True)
        self._maybe_settler.start()
        plural_s = "s" if len(current) != 1 else ""
        logger.info(f"DirectoryWatcher.start: instance '{self.directory}': watching {len(current)} file{plural_s}.")

    def stop(self) -> None:
        """Stop watching. Pending changes not yet reported are dropped. Safe to call more than once."""
        with self._lock:
            self._stopping = True
            self._wakeup.notify_all()
        if self._maybe_settler is not None:  # first the reporting, so nothing is reported after `stop`
            if self._maybe_settler is not threading.current_thread():  # `stop` from inside `on_changed`
                self._maybe_settler.join()
            self._maybe_settler = None
        if self._maybe_observer is not None:
            self._maybe_observer.stop()
            self._maybe_observer.join()
            self._maybe_observer = None

    # ---- events -> dirty paths

    def _mark(self, path: str) -> None:
        """Mark what `path` (absolute, as an event reported it) means for the files we follow."""
        absolute = pathlib.Path(path)
        # A link's target is keyed by its resolved path, and an event inside the watched directory may name it
        # by the unresolved one, so look it up under both.
        spellings = {absolute}
        for base in (self.directory, self._resolved_directory):
            try:
                spellings.add(self._resolved_directory / absolute.relative_to(base))
                break
            except ValueError:
                continue
        links = self._links  # replaced whole under the lock and never mutated, so reading it needs none
        relatives = set().union(*(links.get(spelling, ()) for spelling in spellings))
        maybe_relative = self._relative(absolute)
        if maybe_relative is not None:
            relatives.add(maybe_relative)
        if not relatives:
            return
        now = time.monotonic()
        with self._lock:
            self._dirty |= relatives
            self._last_event_time = now
            if self._first_dirty_time is None:
                self._first_dirty_time = now
            self._wakeup.notify_all()

    def _relative(self, absolute: pathlib.Path) -> pathlib.Path | None:
        # An event names the path as the watch was scheduled, which for a symlinked directory may be either
        # spelling depending on the platform's backend, so accept both.
        for base in (self.directory, self._resolved_directory):
            try:
                relative = absolute.relative_to(base)
            except ValueError:
                continue
            if relative == pathlib.Path("."):
                return None
            if not self.recursive and len(relative.parts) > 1:
                return None
            return relative
        return None

    # ---- settling: dirty paths -> reported changes

    def _settle_loop(self) -> None:
        while True:
            with self._lock:
                while not self._stopping:
                    if self._dirty:
                        now = time.monotonic()
                        quiet_until = self._last_event_time + self.quiet_period
                        deadline = self._first_dirty_time + self.max_delay
                        if now >= min(quiet_until, deadline):
                            break
                        self._wakeup.wait(timeout=min(quiet_until, deadline) - now)
                    else:
                        self._wakeup.wait()
                if self._stopping:
                    return
                dirty, self._dirty = self._dirty, set()
                self._first_dirty_time = None
            try:
                self._settle(dirty)
            except Exception:  # noqa: BLE001 -- one bad settle must not stop the watching
                logger.exception(f"DirectoryWatcher._settle_loop: instance '{self.directory}': settling failed")

    def _settle(self, dirty: set[pathlib.Path]) -> None:
        with self._lock:
            known = dict(self._snapshot)
        # A dirty path can be a directory, and then it stands for everything under it: what we knew there, and
        # what is there now. A directory moved out of the tree arrives as one event naming only the directory.
        # For one moved *in*, watchdog's inotify backend also synthesizes an event per file inside (observed on
        # Linux, 2026-11-13); the rescan keeps that from depending on the backend, the others being untested.
        candidates = set()
        for relative_path in dirty:
            candidates.add(relative_path)
            candidates.update(path for path in known if relative_path in path.parents)
            full = self.directory / relative_path
            if self.recursive and full.is_dir() and not full.is_symlink():
                candidates.update(self._scan(full).keys())
        changes = []
        for relative_path in sorted(candidates):
            new = self._signature_if_accepted(relative_path)
            kind = _difference(known.get(relative_path), new)
            if kind is None:
                continue
            changes.append((relative_path, kind))
            with self._lock:
                if new is None:
                    self._snapshot.pop(relative_path, None)
                else:
                    self._snapshot[relative_path] = new
        if changes:
            with self._lock:
                self._update_links()
        for relative_path, kind in changes:
            if self._stopping:
                return
            self._notify(relative_path, kind)

    def _notify(self, relative_path: pathlib.Path, kind: str) -> None:
        logger.debug(f"DirectoryWatcher._notify: instance '{self.directory}': '{relative_path}' {kind}.")
        try:
            self.on_changed(relative_path, kind)
        except Exception:  # noqa: BLE001 -- the caller's handler must not stop the watching
            logger.exception(f"DirectoryWatcher._notify: instance '{self.directory}': ignoring exception from `on_changed`")

    # ---- the directory as it is now

    def _scan(self, top: pathlib.Path | None = None) -> dict[pathlib.Path, Signature]:
        """The accepted files at or under `top` (default: the watched directory), with their signatures."""
        top = top if top is not None else self.directory
        found = {}
        for root, dirs, files in os.walk(top):  # does not descend into symlinked directories
            if not self.recursive:
                dirs.clear()
            for filename in files:
                full = pathlib.Path(root) / filename
                relative_path = full.relative_to(self.directory)
                signature = self._signature_if_accepted(relative_path)
                if signature is not None:
                    found[relative_path] = signature
        return found

    def _signature_if_accepted(self, relative_path: pathlib.Path) -> Signature | None:
        full = self.directory / relative_path
        if not self.accept(full):
            return None
        return signature_of(full)

    def _update_links(self) -> None:
        """Bring the symlink-target watches up to date with the snapshot. Call with `self._lock` held."""
        links: dict[pathlib.Path, set[pathlib.Path]] = {}
        for relative_path in self._snapshot:
            full = self.directory / relative_path
            if full.is_symlink():
                links.setdefault(full.resolve(), set()).add(relative_path)
        self._links = links
        if self._maybe_observer is None:
            return
        # A target in a directory we already watch needs no watch of its own; its events arrive anyway, and
        # `_mark` maps them back to the link.
        wanted = {target.parent for target in links
                  if not (target.parent == self._resolved_directory or
                          (self.recursive and self._resolved_directory in target.parent.parents))}
        for target_dir in wanted - self._target_watches.keys():
            try:
                self._target_watches[target_dir] = self._maybe_observer.schedule(self._maybe_handler, path=str(target_dir),
                                                                                 recursive=False)
            except OSError as exc:  # gone again, or unreadable: the link reads as whatever `stat` says
                logger.warning(f"DirectoryWatcher._update_links: instance '{self.directory}': cannot watch '{target_dir}': {type(exc)}: {exc}")
        for target_dir in self._target_watches.keys() - wanted:
            self._maybe_observer.unschedule(self._target_watches.pop(target_dir))


def _difference(old: Signature | None, new: Signature | None) -> str | None:
    if old is None and new is None:
        return None
    if old is None:
        return CREATED
    if new is None:
        return DELETED
    return MODIFIED if old != new else None


class _Handler(watchdog.events.FileSystemEventHandler):
    def __init__(self, watcher: DirectoryWatcher) -> None:
        super().__init__()
        self.watcher = watcher

    def on_any_event(self, event: watchdog.events.FileSystemEvent) -> None:
        # An open, a read-only close, or a directory's own mtime changing says nothing about any file. The last
        # one fires for every entry added or removed, and marking the directory dirty for it would rescan the
        # whole subtree each time; the entry's own event already names it.
        if event.event_type in (watchdog.events.EVENT_TYPE_OPENED, watchdog.events.EVENT_TYPE_CLOSED_NO_WRITE):
            return
        if event.is_directory and event.event_type == watchdog.events.EVENT_TYPE_MODIFIED:
            return
        self.watcher._mark(os.fsdecode(event.src_path))
        if event.event_type == watchdog.events.EVENT_TYPE_MOVED:
            self.watcher._mark(os.fsdecode(event.dest_path))


@contextlib.contextmanager
def watching(directory: str | pathlib.Path,
             *,
             known: Mapping[str | pathlib.Path, Signature] | None = None,
             **kwargs) -> Iterator[DirectoryWatcher]:
    """`DirectoryWatcher(directory, **kwargs)`, started with `known` and stopped on leaving the block."""
    watcher = DirectoryWatcher(directory, **kwargs)
    watcher.start(known=known)
    try:
        yield watcher
    finally:
        watcher.stop()

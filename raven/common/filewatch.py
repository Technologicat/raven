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

    `recursive`: also watch subdirectories, symlinked ones included, their files reported under the
                 link's name. A link that leads back to a directory it is inside is not followed.

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
    # A symlink's target lives elsewhere, and inotify on the directory sees nothing when it changes, so each
    # target gets a watch of its own — a linked file its parent directory, a linked directory itself — and a
    # map from target back to link turns an event there into the path under the link.
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
        # (directory, recursive) -> watch. Touched only by `start`, before the settler exists, and then by the
        # settler alone, so it needs no lock.
        self._target_watches: dict[tuple[pathlib.Path, bool], watchdog.observers.api.ObservedWatch] = {}
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

        current, links = self._walk()
        with self._lock:
            self._snapshot = dict(current)
        self._update_links(links)
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
        # by the unresolved one, so look it up under both. A target is a file, or a directory the event is
        # somewhere inside, so every ancestor of the event's path is a candidate.
        spellings = {absolute}
        for base in (self.directory, self._resolved_directory):
            try:
                spellings.add(self._resolved_directory / absolute.relative_to(base))
                break
            except ValueError:
                continue
        links = self._links  # replaced whole and never mutated, so reading it needs no lock
        relatives = set()
        for spelling in spellings:
            for target in (spelling, *spelling.parents):
                for link in links.get(target, ()):
                    relatives.add(link / spelling.relative_to(target))
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
            if self.recursive and full.is_dir():
                candidates.update(self._walk(full)[0].keys())
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
        # A link can come or go without any file changing — a linked directory that holds nothing we accept
        # yet — so a dirty directory, or a dirty path that was a link, also brings the links up to date.
        was_link = {link for links in self._links.values() for link in links}
        if changes or any((self.directory / path).is_dir() or path in was_link for path in dirty):
            self._update_links(self._walk(signatures=False)[1])
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

    def _walk(self,
              top: pathlib.Path | None = None,
              *,
              signatures: bool = True) -> tuple[dict[pathlib.Path, Signature], dict[pathlib.Path, set[pathlib.Path]]]:
        """The accepted files at or under `top` (default: the watched directory), and the symlinks on the way.

        Returns `(files, links)`: `files` maps relative paths to signatures (empty unless `signatures`), and
        `links` maps each link's resolved target to the relative paths of the links leading to it — linked
        accepted files, and when recursive, linked directories.
        """
        top = top if top is not None else self.directory
        files: dict[pathlib.Path, Signature] = {}
        links: dict[pathlib.Path, set[pathlib.Path]] = {}
        for root, dirs, filenames in os.walk(top, followlinks=self.recursive):
            root_path = pathlib.Path(root)
            if not self.recursive:
                dirs.clear()
            else:
                # The cycle guard. A link to a directory that the walk is already inside — itself, or any
                # directory on the way down to it — is not followed. Followed, it is walked again at every
                # level until the OS refuses to resolve the path, and its files are reported once per level. What
                # counts is where the walk *is*, link targets included, so the chain is resolved prefix by
                # prefix. A link to anywhere else is followed, even into a directory reached another way:
                # its files are then reported under both names, as a linked file is beside its target.
                relative_root = root_path.relative_to(self.directory)
                chain = {self._resolved_directory}
                chain.update((self.directory / prefix).resolve()
                             for prefix in (relative_root, *relative_root.parents) if prefix != pathlib.Path("."))
                kept = []
                for name in sorted(dirs):
                    subdirectory = root_path / name
                    if subdirectory.is_symlink():
                        target = subdirectory.resolve()
                        if target in chain:
                            logger.debug(f"DirectoryWatcher._walk: instance '{self.directory}': not following '{subdirectory}', which leads back to '{target}'.")
                            continue
                        links.setdefault(target, set()).add(subdirectory.relative_to(self.directory))
                    kept.append(name)
                dirs[:] = kept
            for filename in filenames:
                full = root_path / filename
                relative_path = full.relative_to(self.directory)
                if not self.accept(full):
                    continue
                if full.is_symlink():
                    links.setdefault(full.resolve(), set()).add(relative_path)
                if signatures:
                    signature = signature_of(full)
                    if signature is not None:
                        files[relative_path] = signature
        return files, links

    def _signature_if_accepted(self, relative_path: pathlib.Path) -> Signature | None:
        full = self.directory / relative_path
        if not self.accept(full):
            return None
        return signature_of(full)

    def _update_links(self, links: dict[pathlib.Path, set[pathlib.Path]]) -> None:
        """Adopt `links` (from `_walk`) and bring the target watches up to date with it.

        Never call with `self._lock` held. Watchdog holds its own lock while it calls the event handler, which
        takes ours, and `schedule` and `unschedule` take watchdog's — so holding ours here can deadlock.
        """
        self._links = links
        if self._maybe_observer is None:
            return
        # A target already inside a watched tree needs no watch of its own: its events arrive anyway, and
        # `_mark` maps them back to the link. Directories first, the shallowest first, so that one linked
        # directory's watch can cover another's.
        trees = [(self._resolved_directory, self.recursive)]
        wanted: set[tuple[pathlib.Path, bool]] = set()
        def covered(directory: pathlib.Path, *, recursive: bool) -> bool:
            return any(directory == tree or (tree_recursive and tree in directory.parents)
                       for tree, tree_recursive in trees
                       if tree_recursive or not recursive)
        directory_targets = sorted((target for target in links if target.is_dir()), key=lambda t: len(t.parts))
        for target in directory_targets:
            if not covered(target, recursive=True):
                wanted.add((target, True))
                trees.append((target, True))
        for target in links:
            if not target.is_dir() and not covered(target.parent, recursive=False):
                wanted.add((target.parent, False))
                trees.append((target.parent, False))
        for key in wanted - self._target_watches.keys():
            directory, recursive = key
            try:
                self._target_watches[key] = self._maybe_observer.schedule(self._maybe_handler, path=str(directory),
                                                                          recursive=recursive)
            except OSError as exc:  # gone again, or unreadable: the link reads as whatever `stat` says
                logger.warning(f"DirectoryWatcher._update_links: instance '{self.directory}': cannot watch '{directory}': {type(exc)}: {exc}")
        for key in self._target_watches.keys() - wanted:
            self._maybe_observer.unschedule(self._target_watches.pop(key))


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

"""Tests for `raven.common.filewatch`: real files, real renames, real symlinks, a real watchdog observer."""

import os
import pathlib
import queue
import time

import pytest

from raven.common import filewatch

QUIET = 0.1
FAST = {"quiet_period": QUIET, "observer_timeout": QUIET}  # the observer's default second, on Windows, per event and per stop
TIMEOUT = 10.0  # generous: macOS's FSEvents delivers late, and an absent event costs this much only on failure


class Recorder:
    """Collects `on_changed` calls, and waits for them."""
    def __init__(self) -> None:
        self.events: queue.Queue = queue.Queue()
        self.seen: list[tuple[str, str]] = []

    def __call__(self, relative_path: pathlib.Path, kind: str) -> None:
        self.events.put((relative_path.as_posix(), kind))

    def wait_for(self, expected: tuple[str, str]) -> list[tuple[str, str]]:
        """Wait until `expected` has been reported, then a little longer for stragglers; return everything seen."""
        deadline = time.monotonic() + TIMEOUT
        while expected not in self.seen:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError(f"never reported {expected}; reported {self.seen}")
            try:
                self.seen.append(self.events.get(timeout=remaining))
            except queue.Empty:
                pass
        time.sleep(5 * QUIET)
        while not self.events.empty():
            self.seen.append(self.events.get())
        return self.seen


@pytest.fixture
def docs(tmp_path):
    directory = tmp_path / "docs"
    directory.mkdir()
    (directory / "a.md").write_text("version one")
    return directory


@pytest.fixture
def recorder():
    return Recorder()


def symlink_or_skip(link: pathlib.Path, target: pathlib.Path) -> None:
    try:
        link.symlink_to(target)
    except OSError as exc:  # Windows without the privilege
        pytest.skip(f"cannot create symlinks here: {exc}")


class TestLiveChanges:
    def test_an_in_place_write_is_a_modification(self, docs, recorder):
        with filewatch.watching(docs, on_changed=recorder, **FAST):
            (docs / "a.md").write_text("version two, longer")
            assert recorder.wait_for(("a.md", filewatch.MODIFIED)) == [("a.md", filewatch.MODIFIED)]

    def test_a_new_file_is_created_and_a_removed_one_deleted(self, docs, recorder):
        with filewatch.watching(docs, on_changed=recorder, **FAST):
            (docs / "b.md").write_text("new")
            recorder.wait_for(("b.md", filewatch.CREATED))
            (docs / "b.md").unlink()
            assert recorder.wait_for(("b.md", filewatch.DELETED)) == [("b.md", filewatch.CREATED),
                                                                       ("b.md", filewatch.DELETED)]

    @pytest.mark.parametrize("temporary_name", ["tmpk3j2x9", ".a.md.swp.md"])
    def test_a_save_by_rename_is_a_modification_of_the_original_only(self, docs, recorder, temporary_name):
        # The temporary file exists for a moment under a name `accept` may well take (the second case keeps
        # the extension), and is renamed away within the quiet period; only the original's change is real.
        with filewatch.watching(docs, on_changed=recorder, **FAST):
            temporary = docs / temporary_name
            temporary.write_text("version two, longer")
            os.replace(temporary, docs / "a.md")
            assert recorder.wait_for(("a.md", filewatch.MODIFIED)) == [("a.md", filewatch.MODIFIED)]

    def test_a_backup_then_a_new_file_is_a_modification(self, docs, recorder):
        # The original renamed to a backup, then written afresh: Emacs' default way to save.
        accept = lambda path: path.suffix == ".md"  # noqa: E731 -- a one-line filter, named for the call below
        with filewatch.watching(docs, on_changed=recorder, accept=accept, **FAST):
            os.replace(docs / "a.md", docs / "a.md~")
            (docs / "a.md").write_text("version two, longer")
            assert recorder.wait_for(("a.md", filewatch.MODIFIED)) == [("a.md", filewatch.MODIFIED)]

    def test_files_accept_turns_away_are_never_reported(self, docs, recorder):
        accept = lambda path: path.suffix == ".md"  # noqa: E731 -- as above
        with filewatch.watching(docs, on_changed=recorder, accept=accept, **FAST):
            (docs / "notes.txt").write_text("not ours")
            (docs / "b.md").write_text("ours")
            assert recorder.wait_for(("b.md", filewatch.CREATED)) == [("b.md", filewatch.CREATED)]


class TestSymlinks:
    @pytest.fixture
    def linked(self, docs, tmp_path):
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        target = elsewhere / "target.md"
        target.write_text("version one")
        symlink_or_skip(docs / "link.md", target)
        return target

    def test_editing_the_target_in_place_modifies_the_link(self, docs, recorder, linked):
        with filewatch.watching(docs, on_changed=recorder, **FAST):
            linked.write_text("version two, longer")
            assert recorder.wait_for(("link.md", filewatch.MODIFIED)) == [("link.md", filewatch.MODIFIED)]

    def test_replacing_the_target_by_rename_modifies_the_link(self, docs, recorder, linked):
        with filewatch.watching(docs, on_changed=recorder, **FAST):
            temporary = linked.parent / "tmpq8"
            temporary.write_text("version two, longer")
            os.replace(temporary, linked)
            assert recorder.wait_for(("link.md", filewatch.MODIFIED)) == [("link.md", filewatch.MODIFIED)]

    def test_a_link_to_a_file_beside_it_follows_that_file(self, docs, recorder):
        symlink_or_skip(docs / "alias.md", docs / "a.md")
        with filewatch.watching(docs, on_changed=recorder, **FAST):
            (docs / "a.md").write_text("version two, longer")
            seen = recorder.wait_for(("alias.md", filewatch.MODIFIED))
            assert sorted(seen) == [("a.md", filewatch.MODIFIED), ("alias.md", filewatch.MODIFIED)]

    def test_a_link_made_while_watching_is_followed_from_then_on(self, docs, recorder, tmp_path):
        target = tmp_path / "later.md"
        target.write_text("version one")
        with filewatch.watching(docs, on_changed=recorder, **FAST):
            symlink_or_skip(docs / "later.md", target)
            recorder.wait_for(("later.md", filewatch.CREATED))
            target.write_text("version two, longer")
            assert recorder.wait_for(("later.md", filewatch.MODIFIED)) == [("later.md", filewatch.CREATED),
                                                                            ("later.md", filewatch.MODIFIED)]

    def test_the_watched_directory_may_itself_be_a_link(self, docs, recorder, tmp_path):
        # The commonest use: a documents folder that is a link to a collection kept elsewhere. Paths are
        # relative, so they do not depend on which spelling of the directory an event used.
        slot = tmp_path / "slot"
        symlink_or_skip(slot, docs)
        with filewatch.watching(slot, on_changed=recorder, **FAST):
            (docs / "a.md").write_text("version two, longer")
            assert recorder.wait_for(("a.md", filewatch.MODIFIED)) == [("a.md", filewatch.MODIFIED)]


class TestSymlinkedDirectories:
    """A linked subdirectory, as an agent skill kept in a repository and linked into the skills folder."""

    @pytest.fixture
    def skill(self, docs, tmp_path):
        """`docs/skill` -> `elsewhere/skill`, holding `SKILL.md`."""
        target = tmp_path / "elsewhere" / "skill"
        target.mkdir(parents=True)
        (target / "SKILL.md").write_text("version one")
        symlink_or_skip(docs / "skill", target)
        return target

    def test_its_files_are_found_under_the_links_name(self, docs, recorder, skill):
        watcher = filewatch.DirectoryWatcher(docs, on_changed=recorder, recursive=True, **FAST)
        watcher.start()
        watcher.stop()
        assert set(watcher.snapshot) == {pathlib.Path("a.md"), pathlib.Path("skill/SKILL.md")}

    def test_editing_a_file_in_the_target_modifies_it_under_the_link(self, docs, recorder, skill):
        with filewatch.watching(docs, on_changed=recorder, recursive=True, **FAST):
            (skill / "SKILL.md").write_text("version two, longer")
            assert recorder.wait_for(("skill/SKILL.md", filewatch.MODIFIED)) == [("skill/SKILL.md", filewatch.MODIFIED)]

    def test_a_file_new_in_a_target_that_held_nothing_accepted_is_created(self, docs, recorder, skill):
        # Nothing in the target passes `accept` at start, so no file stands for the link; it must be watched
        # anyway, or the first accepted file to appear there is never seen.
        accept = lambda path: path.suffix == ".txt"  # noqa: E731 -- a one-line filter, named for the call below
        with filewatch.watching(docs, on_changed=recorder, recursive=True, accept=accept, **FAST):
            (skill / "notes.txt").write_text("new")
            assert recorder.wait_for(("skill/notes.txt", filewatch.CREATED)) == [("skill/notes.txt", filewatch.CREATED)]

    def test_a_link_made_and_removed_while_watching(self, docs, recorder, skill, tmp_path):
        other = tmp_path / "other"
        other.mkdir()
        (other / "SKILL.md").write_text("version one")
        with filewatch.watching(docs, on_changed=recorder, recursive=True, **FAST):
            symlink_or_skip(docs / "other", other)
            recorder.wait_for(("other/SKILL.md", filewatch.CREATED))
            (other / "SKILL.md").write_text("version two, longer")  # followed from then on
            recorder.wait_for(("other/SKILL.md", filewatch.MODIFIED))
            (docs / "other").unlink()
            assert recorder.wait_for(("other/SKILL.md", filewatch.DELETED)) == [("other/SKILL.md", filewatch.CREATED),
                                                                                ("other/SKILL.md", filewatch.MODIFIED),
                                                                                ("other/SKILL.md", filewatch.DELETED)]

    def test_a_link_made_to_a_directory_holding_nothing_accepted_is_followed(self, docs, recorder, tmp_path):
        # The link's arrival changes no file we report, so nothing but the link itself says to watch it.
        empty = tmp_path / "empty"
        empty.mkdir()
        (empty / "readme.md").write_text("not accepted")
        accept = lambda path: path.suffix == ".txt"  # noqa: E731 -- as above
        with filewatch.watching(docs, on_changed=recorder, recursive=True, accept=accept, **FAST):
            symlink_or_skip(docs / "later", empty)
            time.sleep(10 * QUIET)  # the link's own event settles, reporting nothing
            (empty / "notes.txt").write_text("new")
            assert recorder.wait_for(("later/notes.txt", filewatch.CREATED)) == [("later/notes.txt", filewatch.CREATED)]

    def test_a_link_back_to_the_watched_directory_is_not_followed(self, docs, recorder):
        symlink_or_skip(docs / "loop", docs)
        watcher = filewatch.DirectoryWatcher(docs, on_changed=recorder, recursive=True, **FAST)
        watcher.start()  # returning at all is half the test
        watcher.stop()
        assert set(watcher.snapshot) == {pathlib.Path("a.md")}

    def test_two_links_leading_into_each_other_are_walked_once_each(self, docs, recorder, tmp_path):
        # `docs/out` -> `elsewhere`, and `elsewhere/back` -> `docs`. Neither is a link to its own ancestor
        # as spelled, so only the resolved chain of where the walk is can see the cycle.
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "b.md").write_text("b")
        symlink_or_skip(docs / "out", elsewhere)
        symlink_or_skip(elsewhere / "back", docs)
        watcher = filewatch.DirectoryWatcher(docs, on_changed=recorder, recursive=True, **FAST)
        watcher.start()
        watcher.stop()
        assert set(watcher.snapshot) == {pathlib.Path("a.md"), pathlib.Path("out/b.md")}

    def test_without_recursive_a_linked_directory_is_ignored(self, docs, recorder, skill):
        watcher = filewatch.DirectoryWatcher(docs, on_changed=recorder, **FAST)
        watcher.start()
        watcher.stop()
        assert set(watcher.snapshot) == {pathlib.Path("a.md")}


class TestRecursive:
    def test_a_directory_moved_in_brings_its_files(self, docs, recorder, tmp_path):
        # The kernel reports only the directory's move; whether the files inside are reported depends on the
        # watchdog backend, and the watcher must not depend on it.
        outside = tmp_path / "batch"
        outside.mkdir()
        (outside / "one.md").write_text("1")
        (outside / "two.md").write_text("2")
        with filewatch.watching(docs, on_changed=recorder, recursive=True, **FAST):
            os.replace(outside, docs / "batch")
            seen = recorder.wait_for(("batch/two.md", filewatch.CREATED))
            assert sorted(seen) == [("batch/one.md", filewatch.CREATED), ("batch/two.md", filewatch.CREATED)]

    def test_a_directory_moved_out_takes_its_files(self, docs, recorder, tmp_path):
        (docs / "batch").mkdir()
        (docs / "batch" / "one.md").write_text("1")
        with filewatch.watching(docs, on_changed=recorder, recursive=True, **FAST):
            os.replace(docs / "batch", tmp_path / "gone")
            assert recorder.wait_for(("batch/one.md", filewatch.DELETED)) == [("batch/one.md", filewatch.DELETED)]

    def test_without_recursive_a_subdirectory_is_ignored(self, docs, recorder):
        (docs / "sub").mkdir()
        with filewatch.watching(docs, on_changed=recorder, **FAST) as watcher:
            (docs / "sub" / "deep.md").write_text("not followed")
            (docs / "b.md").write_text("followed")
            assert recorder.wait_for(("b.md", filewatch.CREATED)) == [("b.md", filewatch.CREATED)]
            assert pathlib.Path("sub/deep.md") not in watcher.snapshot


class TestOfflineChanges:
    def test_differences_from_the_known_snapshot_are_reported_at_start(self, docs, recorder):
        (docs / "b.md").write_text("unchanged")
        (docs / "new.md").write_text("added while away")
        unchanged = filewatch.signature_of(docs / "b.md")
        known = {"a.md": (0.0, 1),          # a different signature: modified
                 "b.md": unchanged,          # the same: nothing
                 "gone.md": (0.0, 1)}        # absent now: deleted
        watcher = filewatch.DirectoryWatcher(docs, on_changed=recorder, **FAST)
        watcher.start(known=known)
        try:
            # Reported before `start` returned, on this thread, so nothing needs waiting for.
            reported = []
            while not recorder.events.empty():
                reported.append(recorder.events.get())
            assert reported == [("a.md", filewatch.MODIFIED), ("gone.md", filewatch.DELETED),
                                ("new.md", filewatch.CREATED)]
        finally:
            watcher.stop()

    def test_an_older_mtime_is_a_modification_too(self, docs, recorder):
        # A file restored from a backup goes back in time; "mtime increased" would miss it.
        current = filewatch.signature_of(docs / "a.md")
        known = {"a.md": (current[0] + 3600.0, current[1])}
        watcher = filewatch.DirectoryWatcher(docs, on_changed=recorder, **FAST)
        watcher.start(known=known)
        watcher.stop()
        assert recorder.events.get_nowait() == ("a.md", filewatch.MODIFIED)

    def test_without_known_nothing_is_reported_at_start(self, docs, recorder):
        watcher = filewatch.DirectoryWatcher(docs, on_changed=recorder, **FAST)
        watcher.start()
        watcher.stop()
        assert recorder.events.empty()
        assert watcher.snapshot == {pathlib.Path("a.md"): filewatch.signature_of(docs / "a.md")}


class TestLifecycle:
    def test_nothing_is_reported_after_stop(self, docs, recorder):
        watcher = filewatch.DirectoryWatcher(docs, on_changed=recorder, **FAST)
        watcher.start()
        (docs / "a.md").write_text("version two, longer")
        recorder.wait_for(("a.md", filewatch.MODIFIED))  # the apparatus reports while running...
        watcher.stop()
        watcher.stop()  # ...and a second stop is harmless
        (docs / "a.md").write_text("version three, longer still")
        time.sleep(10 * QUIET)
        assert recorder.events.empty()

    def test_on_changed_may_stop_the_watcher(self, docs):
        stopped = []

        def stop_on_first(relative_path, kind):
            watcher.stop()  # from the watcher's own thread: must not wait for itself
            stopped.append(kind)
        watcher = filewatch.DirectoryWatcher(docs, on_changed=stop_on_first, **FAST)
        watcher.start()
        (docs / "a.md").write_text("version two, longer")
        deadline = time.monotonic() + TIMEOUT
        while not stopped and time.monotonic() < deadline:
            time.sleep(QUIET)
        assert stopped == [filewatch.MODIFIED]
        watcher.stop()

    def test_a_raising_handler_does_not_stop_the_watching(self, docs, recorder):
        def flaky(relative_path, kind):
            recorder(relative_path, kind)
            raise RuntimeError("a handler with a bug")
        with filewatch.watching(docs, on_changed=flaky, **FAST):
            (docs / "a.md").write_text("version two, longer")
            recorder.wait_for(("a.md", filewatch.MODIFIED))
            (docs / "b.md").write_text("still watching")
            recorder.wait_for(("b.md", filewatch.CREATED))

    def test_a_path_may_be_str_or_path(self, docs, recorder):
        for spelling in (str(docs), docs):
            watcher = filewatch.DirectoryWatcher(spelling, on_changed=recorder)
            assert watcher.directory == docs

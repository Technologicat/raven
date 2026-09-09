"""Unit tests for raven.common.datastorelock.

What is being pinned is the property the lock exists for: the second opener is refused, so that two apps
sharing one chat datastore cannot each write back a whole in-memory copy and have the later exit win.
"""

import os
import pathlib
import tempfile

import pytest

from raven.common import datastorelock


@pytest.fixture
def datastore(tmp_path):
    """A datastore path. Deliberately not created — a lock has to work before the first run makes the file."""
    return tmp_path / "chat.json"


@pytest.fixture
def released(datastore):
    """Release anything this test left locked, so a failure cannot leak a lock into the next test."""
    held = []
    yield held
    for lock in held:
        lock.release()


class TestLockPath:
    def test_the_same_datastore_gets_the_same_lock_by_any_route(self, tmp_path):
        # Two spellings of one path must not become two locks, or the whole guard is bypassed by launching
        # one app from a different working directory than the other.
        direct = tmp_path / "chat.json"
        roundabout = tmp_path / "sub" / ".." / "chat.json"
        (tmp_path / "sub").mkdir()
        assert datastorelock.lock_path_for(direct) == datastorelock.lock_path_for(roundabout)

    def test_two_datastores_get_two_locks(self, tmp_path):
        # Otherwise a second datastore -- a separate corpus, a test fixture -- would be refused for no
        # reason by whatever holds the first.
        assert datastorelock.lock_path_for(tmp_path / "a.json") != datastorelock.lock_path_for(tmp_path / "b.json")

    def test_the_lock_lives_outside_the_datastore_folder(self, datastore):
        # Librarian has a button that opens the datastore folder in a file manager, so a lock file there
        # would be litter in a place the user looks.
        lock_path = datastorelock.lock_path_for(datastore)
        assert lock_path.parent == pathlib.Path(tempfile.gettempdir())
        assert datastore.parent not in lock_path.parents


class TestExclusion:
    def test_the_second_opener_is_refused(self, datastore, released):
        first = datastorelock.acquire(datastore, what="The chat datastore")
        released.append(first)
        with pytest.raises(datastorelock.DatastoreBusyError):
            datastorelock.acquire(datastore, what="The chat datastore")

    def test_the_refusal_names_the_datastore_and_what_to_do(self, datastore, released):
        # This message is the entire user interface of the feature: it is printed and the app exits.
        released.append(datastorelock.acquire(datastore, what="The chat datastore"))
        with pytest.raises(datastorelock.DatastoreBusyError) as excinfo:
            datastorelock.acquire(datastore, what="The chat datastore")
        message = str(excinfo.value)
        assert str(datastore) in message
        assert "chat datastore" in message.lower()

    def test_a_different_datastore_is_not_blocked(self, tmp_path, released):
        released.append(datastorelock.acquire(tmp_path / "a.json", what="The chat datastore"))
        released.append(datastorelock.acquire(tmp_path / "b.json", what="The chat datastore"))

    def test_releasing_lets_the_next_app_in(self, datastore):
        # The ordinary case: close one frontend, open the other. Nothing in between should have to clean up
        # after the first -- which is why this locks the file rather than writing a PID into it.
        first = datastorelock.acquire(datastore, what="The chat datastore")
        first.release()
        second = datastorelock.acquire(datastore, what="The chat datastore")
        second.release()


class TestNamingTheHolder:
    """A refusal says who to close, because the process holding the lock is the one that knows.

    Acquiring twice in one process is how a refusal is provoked here — `flock` is per open file
    description, so that is refused exactly as another process would be. But it reaches the *same-process*
    message, not the cross-process one, so a test about the cross-process wording has to make the lock file
    name somebody else first.
    """

    def _hold_as_another_process(self, target, holder="raven-indexer (PID 999999)"):
        """Take the lock, then overwrite what it says about its holder with somebody else's name.

        Simpler than faking `getpid` around the acquire, and it exercises the same thing: what the
        refusing process reads out of the lock file is what decides which message it gets.
        """
        held = datastorelock.acquire(target, what="The document index")
        datastorelock.lock_path_for(target).write_text(holder, encoding="utf-8")
        return held

    def test_the_refusal_names_the_holding_process(self, tmp_path, monkeypatch):
        monkeypatch.setattr(datastorelock.sys, "argv", ["/usr/bin/raven-indexer", "somedir"])
        target = tmp_path / "index"
        held = datastorelock.acquire(target, what="The document index")
        try:
            with pytest.raises(datastorelock.DatastoreBusyError) as excinfo:
                datastorelock.acquire(target, what="The document index")
            message = str(excinfo.value)
            assert "raven-indexer" in message, f"the refusal must name the holder; got: {message}"
            assert str(os.getpid()) in message, "and its PID, so it can be found without guessing"
            # The control: this is what the message said before the holder recorded itself, and a test
            # that only checked for the generic phrase would pass against that older behaviour.
            assert message.count(datastorelock._UNKNOWN_HOLDER) == 0
        finally:
            held.release()

    @pytest.mark.parametrize("argv0", ["", "-c", "-"],
                             ids=["REPL", "python -c", "script on stdin"])
    def test_a_non_script_holder_degrades_to_the_generic_phrase(self, tmp_path, monkeypatch, argv0):
        """`argv[0]` is not always a program name, and printing `-c` at a user would be nonsense."""
        monkeypatch.setattr(datastorelock.sys, "argv", [argv0])
        target = tmp_path / "index"
        held = datastorelock.acquire(target, what="The document index")
        try:
            with pytest.raises(datastorelock.DatastoreBusyError) as excinfo:
                datastorelock.acquire(target, what="The document index")
            message = str(excinfo.value)
            assert datastorelock._UNKNOWN_HOLDER in message
            assert str(os.getpid()) in message, "the PID is worth having even when the name is not"
        finally:
            held.release()

    def test_a_foreign_holder_is_told_to_be_closed(self, tmp_path):
        """The ordinary case: somebody else has it, and closing them is the thing to do."""
        target = tmp_path / "index"
        held = self._hold_as_another_process(target)
        try:
            with pytest.raises(datastorelock.DatastoreBusyError) as excinfo:
                datastorelock.acquire(target, what="The document index")
            message = str(excinfo.value)
            assert "raven-indexer" in message and "999999" in message
            assert "Close that one" in message
            # The control: if the fixture failed to fake a foreign PID, this would be the same-process
            # message instead, and the assertions above would still pass on the name and the number.
            assert "same process" not in message, "the fixture is not reaching the cross-process path"
        finally:
            held.release()

    def test_opening_the_same_store_twice_in_one_process_says_so(self, tmp_path, monkeypatch):
        """Advice to close the other app is useless when the other app is you; that is a caller bug."""
        monkeypatch.setattr(datastorelock.sys, "argv", ["/usr/bin/raven-librarian"])
        target = tmp_path / "index"
        held = datastorelock.acquire(target, what="The document index")
        try:
            with pytest.raises(datastorelock.DatastoreBusyError) as excinfo:
                datastorelock.acquire(target, what="The document index")
            message = str(excinfo.value)
            assert "same process" in message and "opened twice" in message
            assert "Close that one" not in message, "nobody can act on that when the holder is themselves"
        finally:
            held.release()

    def test_a_lock_file_that_says_nothing_is_not_an_error(self, tmp_path):
        """The holder truncates on winning and writes its name a moment later; a refusal can land between."""
        lock_path = datastorelock.lock_path_for(tmp_path / "index")
        lock_path.write_text("", encoding="utf-8")
        assert datastorelock._read_holder(lock_path) == datastorelock._UNKNOWN_HOLDER
        assert datastorelock._read_holder(tmp_path / "no-such-lock") == datastorelock._UNKNOWN_HOLDER

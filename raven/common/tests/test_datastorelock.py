"""Unit tests for raven.common.datastorelock.

What is being pinned is the property the lock exists for: the second opener is refused, so that two apps
sharing one chat datastore cannot each write back a whole in-memory copy and have the later exit win.
"""

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

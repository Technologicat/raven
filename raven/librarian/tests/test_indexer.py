"""Unit tests for raven.librarian.indexer.

What is under test is the *waiting*, not the indexing: `hybridir` already covers building an index, and
`open_document_store` is a thin argument-defaulting shim over `hybridir.setup` that cannot be exercised
without a real embedding model and server.

`wait_for_indexing` is the part with logic of its own, and the logic is easy to get subtly wrong in a way
that only shows up as a headless run exiting before it has indexed anything. There is no
"indexing finished" event to await — the apps that use `hybridir` never need one, because they keep
running — so the wait polls `is_indexing()`, and a *single* quiet sample is ambiguous: it can mean the work
is done, or that the background rescan has not started yet.
"""

import types

import pytest

pytest.importorskip("raven.librarian.indexer", reason="full dependency stack not installed")

from raven.librarian import indexer


def _fake_retriever(busy_sequence, progress_sequence=()):
    """A stand-in exposing just what `wait_for_indexing` touches, driven by scripted samples.

    `busy_sequence`: what successive `is_indexing()` calls return. Exhausting it yields False forever, so a
                     test only has to script the interesting prefix.
    `progress_sequence`: what successive `get_indexing_progress_text()` calls return, same convention
                         (exhausted -> "").
    """
    busy = iter(busy_sequence)
    progress = iter(progress_sequence)
    fake = types.SimpleNamespace()
    fake.polls = 0

    def is_indexing():
        fake.polls += 1
        return next(busy, False)

    fake.is_indexing = is_indexing
    fake.get_indexing_progress_text = lambda: next(progress, "")
    return fake


@pytest.fixture
def instant_polls(monkeypatch):
    """Make the poll interval free, so the tests measure sample *counts* rather than wall-clock."""
    monkeypatch.setattr(indexer, "POLL_SECONDS", 0.0)


def test_an_already_quiet_retriever_still_takes_the_full_settle_window(instant_polls):
    """The startup window: quiet does not mean finished, so one quiet sample must not end the wait."""
    fake = _fake_retriever([])
    indexer.wait_for_indexing(fake)
    assert fake.polls == indexer.SETTLED_POLLS


def test_the_wait_covers_work_that_starts_after_several_quiet_samples(instant_polls):
    """The failure this prevents: exiting during the gap before the background rescan gets going.

    Three quiet samples, then real work, then quiet again. A wait that stopped at the first run of quiet
    samples would return with the index half-built and the process would exit mid-commit.
    """
    fake = _fake_retriever([False, False, False] + [True] * 4)
    indexer.wait_for_indexing(fake)
    # Three wasted, four busy, then a full settle window — nothing may be counted twice.
    assert fake.polls == 3 + 4 + indexer.SETTLED_POLLS


def test_one_busy_sample_resets_the_settle_counter(instant_polls):
    """A flicker of activity partway through the window restarts it rather than shortening it."""
    busy_at = indexer.SETTLED_POLLS - 2
    fake = _fake_retriever([False] * busy_at + [True])
    indexer.wait_for_indexing(fake)
    assert fake.polls == busy_at + 1 + indexer.SETTLED_POLLS


def test_progress_is_reported_only_when_the_text_changes(instant_polls):
    """Per-chunk progress repeats the same string for many polls; a caller redrawing each time flickers."""
    fake = _fake_retriever([True] * 4, ["a", "a", "b", "b"])
    seen = []
    indexer.wait_for_indexing(fake, on_progress=seen.append)
    assert seen == ["a", "b"]


def test_empty_progress_text_is_not_reported(instant_polls):
    """`get_indexing_progress_text` returns "" between documents; that is not a progress update."""
    fake = _fake_retriever([True] * 3, ["", "", ""])
    seen = []
    indexer.wait_for_indexing(fake, on_progress=seen.append)
    assert seen == []


def test_no_progress_callback_is_allowed(instant_polls):
    """`--quiet` passes None, and that must not become an attribute error partway through a long run."""
    fake = _fake_retriever([True] * 3, ["a", "b", "c"])
    indexer.wait_for_indexing(fake, on_progress=None)
    assert fake.polls == 3 + indexer.SETTLED_POLLS

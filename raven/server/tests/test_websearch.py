"""Unit tests for raven.server.modules.websearch: what `search` caches, and how it stops, with the search engines faked."""

import threading

import pytest

pytest.importorskip("selenium", reason="websearch imports selenium at module top (not in the CI minimal dep subset)")
websearch = pytest.importorskip("raven.server.modules.websearch",
                                reason="websearch needs colorama (not in the CI minimal dep subset)")
from raven.server.modules import webcommon  # noqa: E402 -- after the guard, which covers it too


@pytest.fixture
def fake_engine(monkeypatch):
    """Replace the DuckDuckGo scraper with a scripted one; returns the list of queries that reached it.

    Each call pops its outcome off `script`: an exception to raise, or a result to return.
    """
    calls = []
    script = []

    def _search(query, max_links, is_cancelled, on_progress):
        calls.append(query)
        outcome = script.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setitem(websearch._engines, "duckduckgo", _search)
    monkeypatch.setattr(websearch, "_results_cache", {})
    return calls, script


class TestSearchCache:
    def test_an_answer_is_cached(self, fake_engine):
        calls, script = fake_engine
        script.append(("text", [{"text": "a result"}]))
        first = websearch.search("q")
        second = websearch.search("q")
        assert first == second
        assert calls == ["q"], "the second search should have come from the cache"

    def test_a_failure_is_not_cached(self, fake_engine):
        # The engine is asked again, and its answer this time is what the caller gets.
        calls, script = fake_engine
        script.append(websearch.EngineUnavailable("no results container"))
        script.append(("text", [{"text": "a result"}]))
        with pytest.raises(websearch.EngineUnavailable):
            websearch.search("q")
        assert websearch.search("q") == ("text", [{"text": "a result"}])
        assert calls == ["q", "q"]


class TestSearchCancellation:
    def test_a_cancelled_search_never_reaches_the_engine(self, fake_engine):
        calls, script = fake_engine
        script.append(("text", [{"text": "a result"}]))
        with pytest.raises(webcommon.Cancelled):
            websearch.search("q", is_cancelled=lambda: True)
        assert calls == []
        assert websearch.search("q") == ("text", [{"text": "a result"}]), "a cancel should leave nothing cached"

    def test_a_search_waiting_its_turn_gives_up_when_cancelled(self, fake_engine):
        # Another search holds the browser; this one is abandoned while it waits.
        # On a thread of its own with a bounded join, so that a search that waits without asking fails this
        # test rather than hanging it.
        calls, script = fake_engine
        cancel = threading.Event()
        outcome = {}
        def searcher():
            try:
                websearch.search("q", is_cancelled=cancel.is_set)
            except webcommon.Cancelled:
                outcome["cancelled"] = True
        with websearch._search_lock:
            worker = threading.Thread(target=searcher, daemon=True)
            worker.start()
            cancel.set()
            worker.join(timeout=2.0)
            assert not worker.is_alive(), "the waiting search did not give up when cancelled"
        assert outcome == {"cancelled": True}
        assert calls == []

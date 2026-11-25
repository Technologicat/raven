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


class TestDeadBrowser:
    """The one browser can stop answering (a crashed tab); a search then starts a new one, once."""

    @pytest.fixture
    def browsing_engine(self, monkeypatch):
        """A DuckDuckGo scraper that navigates the module's current browser, as the real one does."""
        def _search(query, max_links, is_cancelled, on_progress):
            websearch.driver.get(f"https://duckduckgo.com/?q={query}")
            return ("text", [{"text": "a result"}])
        monkeypatch.setitem(websearch._engines, "duckduckgo", _search)
        monkeypatch.setattr(websearch, "_results_cache", {})

    def test_a_dead_browser_is_replaced_and_the_search_answers(self, browsing_engine, fake_browser, monkeypatch):
        dead, fresh = fake_browser(answers=False), fake_browser()
        monkeypatch.setattr(websearch, "driver", dead)
        monkeypatch.setattr(webcommon, "get_driver", lambda page_load_timeout=None: fresh)
        assert websearch.search("q") == ("text", [{"text": "a result"}])
        assert dead.quit_called
        assert websearch.driver is fresh
        assert fresh.visited, "the search should have run again, on the new browser"

    def test_an_ordinary_failure_does_not_restart_the_browser(self, browsing_engine, fake_browser, monkeypatch):
        # The control for the one above: the navigation fails the same way, but the browser still answers.
        failing = fake_browser(navigation_fails=True)
        monkeypatch.setattr(websearch, "driver", failing)
        monkeypatch.setattr(webcommon, "get_driver", lambda page_load_timeout=None: pytest.fail("restarted a browser that answers"))
        with pytest.raises(Exception, match="ERR_NAME_NOT_RESOLVED"):
            websearch.search("q")
        assert websearch.driver is failing
        assert len(failing.visited) == 1

    def test_a_browser_that_cannot_be_restarted_reports_the_engine_unavailable(self, browsing_engine, fake_browser, monkeypatch):
        monkeypatch.setattr(websearch, "driver", fake_browser(answers=False))
        monkeypatch.setattr(webcommon, "get_driver", lambda page_load_timeout=None: None)
        with pytest.raises(websearch.EngineUnavailable, match="could not be started"):
            websearch.search("q")

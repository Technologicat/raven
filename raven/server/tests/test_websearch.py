"""Unit tests for raven.server.modules.websearch: what `search` caches, with the search engines faked."""

import pytest

pytest.importorskip("selenium", reason="websearch imports selenium at module top (not in the CI minimal dep subset)")
websearch = pytest.importorskip("raven.server.modules.websearch",
                                reason="websearch needs colorama (not in the CI minimal dep subset)")


@pytest.fixture
def fake_engine(monkeypatch):
    """Replace the DuckDuckGo scraper with a scripted one; returns the list of queries that reached it.

    Each call pops its outcome off `script`: an exception to raise, or a result to return.
    """
    calls = []
    script = []

    def _search(query, max_links):
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

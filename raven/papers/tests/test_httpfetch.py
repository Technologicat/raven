"""Tests for `raven.papers.httpfetch` — User-Agent and 429 retry-with-backoff."""

from __future__ import annotations

from unittest.mock import patch

from raven.papers import httpfetch


class _FakeResponse:
    """Minimal stand-in for ``requests.Response`` covering what `arxiv_get` inspects."""

    def __init__(self, status_code: int, headers: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text


class _ScriptedSession:
    """Returns canned responses in order; records the request kwargs each call."""

    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params, "headers": headers, "timeout": timeout})
        return self._responses.pop(0)


def test_useragent_identifies_raven():
    """User-Agent string mentions raven-papers and contains a contact handle."""
    assert "raven-papers/" in httpfetch.USER_AGENT
    assert "Technologicat/raven" in httpfetch.USER_AGENT


def test_arxiv_get_passes_useragent_header():
    """Every outgoing request carries the identifying User-Agent."""
    session = _ScriptedSession([_FakeResponse(200)])
    with patch.object(httpfetch.requests, "get", side_effect=session):
        httpfetch.arxiv_get("https://example.test/api")
    assert session.calls[0]["headers"]["User-Agent"] == httpfetch.USER_AGENT


def test_arxiv_get_returns_immediately_on_2xx():
    """Happy path — single request, no sleeps."""
    session = _ScriptedSession([_FakeResponse(200)])
    with patch.object(httpfetch.requests, "get", side_effect=session), \
         patch.object(httpfetch.time, "sleep") as sleep_mock:
        resp = httpfetch.arxiv_get("https://example.test/api")
    assert resp.status_code == 200
    assert len(session.calls) == 1
    sleep_mock.assert_not_called()


def test_arxiv_get_retries_on_429_then_succeeds():
    """A 429 triggers a retry; the second 200 is returned."""
    session = _ScriptedSession([_FakeResponse(429), _FakeResponse(200)])
    with patch.object(httpfetch.requests, "get", side_effect=session), \
         patch.object(httpfetch.time, "sleep") as sleep_mock:
        resp = httpfetch.arxiv_get("https://example.test/api", base_backoff=1.0)
    assert resp.status_code == 200
    assert len(session.calls) == 2
    sleep_mock.assert_called_once()
    # First backoff is base_backoff * 2**0 = 1.0
    assert sleep_mock.call_args.args[0] == 1.0


def test_arxiv_get_honors_retry_after_header():
    """When the server sets Retry-After (in seconds), `arxiv_get` waits exactly that long."""
    session = _ScriptedSession([
        _FakeResponse(429, headers={"Retry-After": "7"}),
        _FakeResponse(200),
    ])
    with patch.object(httpfetch.requests, "get", side_effect=session), \
         patch.object(httpfetch.time, "sleep") as sleep_mock:
        httpfetch.arxiv_get("https://example.test/api", base_backoff=1.0)
    assert sleep_mock.call_args.args[0] == 7.0


def test_arxiv_get_falls_back_to_exponential_on_unparseable_retry_after():
    """Non-numeric Retry-After (HTTP-date form) is ignored; exponential backoff kicks in."""
    session = _ScriptedSession([
        _FakeResponse(429, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}),
        _FakeResponse(200),
    ])
    with patch.object(httpfetch.requests, "get", side_effect=session), \
         patch.object(httpfetch.time, "sleep") as sleep_mock:
        httpfetch.arxiv_get("https://example.test/api", base_backoff=2.5)
    assert sleep_mock.call_args.args[0] == 2.5


def test_arxiv_get_returns_final_429_after_max_attempts():
    """After exhausting retries, the last 429 is returned for the caller to handle."""
    session = _ScriptedSession([_FakeResponse(429), _FakeResponse(429), _FakeResponse(429)])
    with patch.object(httpfetch.requests, "get", side_effect=session), \
         patch.object(httpfetch.time, "sleep"):
        resp = httpfetch.arxiv_get("https://example.test/api",
                                   max_attempts=3, base_backoff=1.0)
    assert resp.status_code == 429
    assert len(session.calls) == 3


def test_arxiv_get_does_not_retry_non_429():
    """5xx and other failures are returned immediately — the caller decides."""
    session = _ScriptedSession([_FakeResponse(500)])
    with patch.object(httpfetch.requests, "get", side_effect=session), \
         patch.object(httpfetch.time, "sleep") as sleep_mock:
        resp = httpfetch.arxiv_get("https://example.test/api")
    assert resp.status_code == 500
    assert len(session.calls) == 1
    sleep_mock.assert_not_called()

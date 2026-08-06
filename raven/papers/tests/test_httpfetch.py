"""Tests for `raven.papers.httpfetch` — User-Agent and 429 retry-with-backoff."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import requests

from raven.papers import httpfetch


class _FakeResponse:
    """Minimal stand-in for ``requests.Response`` covering what `arxiv_get` inspects."""

    def __init__(self, status_code: int, headers: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text


class _ScriptedSession:
    """Returns canned responses in order; records the request kwargs each call.

    An entry that is an `Exception` instance is *raised* instead of returned, which is how `requests`
    reports a transport failure — a dropped connection or a read timeout never becomes a response.
    """

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params, "headers": headers, "timeout": timeout})
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


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
    """5xx and other failures are returned immediately — the caller decides.

    A status is an *answer*: the server was reached and said something. Only 429 carries a documented
    "try again later", so the rest go straight back to the caller, which knows what a 404 means for the
    request it made. Contrast the transport-error tests below, where there is no answer at all.
    """
    session = _ScriptedSession([_FakeResponse(500)])
    with patch.object(httpfetch.requests, "get", side_effect=session), \
         patch.object(httpfetch.time, "sleep") as sleep_mock:
        resp = httpfetch.arxiv_get("https://example.test/api")
    assert resp.status_code == 500
    assert len(session.calls) == 1
    sleep_mock.assert_not_called()


def test_arxiv_get_retries_transport_error_then_succeeds():
    """A dropped connection is retried with backoff, not surfaced.

    This is what `download.get_papers_metadata` depends on: one request now carries up to 100
    identifiers, so a transient failure that used to cost a single paper would cost a hundred.
    """
    session = _ScriptedSession([requests.exceptions.ConnectionError("connection reset"),
                                _FakeResponse(200)])
    with patch.object(httpfetch.requests, "get", side_effect=session), \
         patch.object(httpfetch.time, "sleep") as sleep_mock:
        resp = httpfetch.arxiv_get("https://example.test/api", base_backoff=1.0)
    assert resp.status_code == 200
    assert len(session.calls) == 2
    assert sleep_mock.call_args.args[0] == 1.0  # base_backoff * 2**0


def test_arxiv_get_retries_read_timeout():
    """The failure actually observed against arXiv's `id_list` endpoint (2026-08-06)."""
    session = _ScriptedSession([requests.exceptions.ReadTimeout("timed out"),
                                requests.exceptions.ReadTimeout("timed out"),
                                _FakeResponse(200)])
    with patch.object(httpfetch.requests, "get", side_effect=session), \
         patch.object(httpfetch.time, "sleep") as sleep_mock:
        resp = httpfetch.arxiv_get("https://example.test/api", base_backoff=1.0)
    assert resp.status_code == 200
    assert [c.args[0] for c in sleep_mock.call_args_list] == [1.0, 2.0]  # exponential


def test_arxiv_get_reraises_transport_error_after_max_attempts():
    """A real outage still surfaces — retrying must not turn "down" into "silently no data"."""
    session = _ScriptedSession([requests.exceptions.ConnectionError("down")] * 3)
    with patch.object(httpfetch.requests, "get", side_effect=session), \
         patch.object(httpfetch.time, "sleep"), \
         pytest.raises(requests.exceptions.ConnectionError):
        httpfetch.arxiv_get("https://example.test/api", max_attempts=3, base_backoff=1.0)
    assert len(session.calls) == 3

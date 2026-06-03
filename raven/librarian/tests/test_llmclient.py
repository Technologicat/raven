"""Unit tests for raven.librarian.llmclient.

Currently focused on the client-side webfetch allowlist gating in `webfetch_wrapper` —
the security-critical decision of whether a URL the model wants to fetch is permitted.
The actual fetch (`api.webfetch_fetch`, HTTP to the server) is monkeypatched.
"""

import pytest

# llmclient transitively imports `raven.client.api`, which pulls the heavy client dep stack
# (qoi, spaCy, the TTS stack) that the CI test job's minimal subset omits. Skip this whole module
# when the import can't be satisfied. Mirrors test_scaffold.py's importorskip on scaffold.
llmclient = pytest.importorskip("raven.librarian.llmclient",
                                reason="llmclient transitively needs the full raven-client dep stack")

from unpythonic import dyn  # noqa: E402 -- after importorskip by design
from unpythonic.env import env  # noqa: E402 -- after importorskip by design


@pytest.fixture
def fake_fetch(monkeypatch):
    """Replace the HTTP fetch with a recorder; returns the list of URLs that reached the server."""
    fetched_urls = []

    def _fake(url, output_format="markdown"):
        fetched_urls.append(url)
        return {"content": f"CONTENT of {url}", "url": url, "spaSuspected": False}

    monkeypatch.setattr(llmclient.api, "webfetch_fetch", _fake)
    return fetched_urls


def _set_allowlist(monkeypatch, allowlist):
    monkeypatch.setattr(llmclient.librarian_config, "webfetch_allowlist", allowlist)


class TestWebfetchWrapperGating:
    def test_no_allowlist_fetches_anything(self, monkeypatch, fake_fetch):
        _set_allowlist(monkeypatch, None)
        result = llmclient.webfetch_wrapper("https://random-site.com/x")
        assert "CONTENT of" in result
        assert fake_fetch == ["https://random-site.com/x"]

    def test_allowlisted_host_fetches(self, monkeypatch, fake_fetch):
        _set_allowlist(monkeypatch, ["*.arxiv.org"])
        with dyn.let(tool_context=env(webfetch_allowed_hosts=frozenset())):
            result = llmclient.webfetch_wrapper("https://arxiv.org/html/2301.1")
        assert "CONTENT of" in result
        assert fake_fetch == ["https://arxiv.org/html/2301.1"]

    def test_non_allowlisted_host_refused(self, monkeypatch, fake_fetch):
        _set_allowlist(monkeypatch, ["*.arxiv.org"])
        with dyn.let(tool_context=env(webfetch_allowed_hosts=frozenset())):
            result = llmclient.webfetch_wrapper("https://evil.com/x")
        assert "not on the configured allowlist" in result
        assert fake_fetch == []  # the request never reached the server

    def test_auto_allowed_host_fetches(self, monkeypatch, fake_fetch):
        # Host not on the configured list, but auto-allowed this turn (user typed it).
        _set_allowlist(monkeypatch, ["*.arxiv.org"])
        with dyn.let(tool_context=env(webfetch_allowed_hosts=frozenset({"user-typed.com"}))):
            result = llmclient.webfetch_wrapper("https://user-typed.com/x")
        assert "CONTENT of" in result
        assert fake_fetch == ["https://user-typed.com/x"]

    def test_fail_closed_without_context(self, monkeypatch, fake_fetch):
        # No dyn.let binding at all -> the process-wide empty-env default -> no auto-allow.
        # A non-listed host must be refused (fail closed), not fetched.
        _set_allowlist(monkeypatch, ["*.arxiv.org"])
        result = llmclient.webfetch_wrapper("https://surprise.com/x")
        assert "not on the configured allowlist" in result
        assert fake_fetch == []

    def test_canonical_refusal_names_the_host(self, monkeypatch, fake_fetch):
        _set_allowlist(monkeypatch, ["doi.org"])
        with dyn.let(tool_context=env(webfetch_allowed_hosts=frozenset())):
            result = llmclient.webfetch_wrapper("https://blocked.example/path")
        assert "blocked.example" in result


@pytest.fixture
def clean_session_approvals():
    """Isolate the module-level session-approved-hosts set across tests."""
    llmclient._session_approved_hosts.clear()
    yield
    llmclient._session_approved_hosts.clear()


class TestSessionApprovedHosts:
    def test_approve_lets_non_allowlisted_host_through(self, monkeypatch, fake_fetch, clean_session_approvals):
        _set_allowlist(monkeypatch, ["*.arxiv.org"])
        with dyn.let(tool_context=env(webfetch_allowed_hosts=frozenset())):
            denied = llmclient.webfetch_wrapper("https://blog.example/post")
            assert "not on the configured allowlist" in denied
            assert fake_fetch == []  # denied before approval

            llmclient.approve_host_for_session("blog.example")
            allowed = llmclient.webfetch_wrapper("https://blog.example/post")
            assert "CONTENT of" in allowed
            assert fake_fetch == ["https://blog.example/post"]  # goes through after approval

    def test_approve_is_case_insensitive(self, monkeypatch, fake_fetch, clean_session_approvals):
        _set_allowlist(monkeypatch, ["doi.org"])
        llmclient.approve_host_for_session("Example.COM")
        with dyn.let(tool_context=env(webfetch_allowed_hosts=frozenset())):
            assert "CONTENT of" in llmclient.webfetch_wrapper("https://example.com/x")

    def test_approval_does_not_apply_when_allowlist_is_none(self, monkeypatch, fake_fetch, clean_session_approvals):
        # With no allowlist there is no gate anyway; approval is simply irrelevant (everything passes).
        _set_allowlist(monkeypatch, None)
        assert "CONTENT of" in llmclient.webfetch_wrapper("https://anything.example/x")
        assert fake_fetch == ["https://anything.example/x"]

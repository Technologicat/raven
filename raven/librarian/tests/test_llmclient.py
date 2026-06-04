"""Unit tests for raven.librarian.llmclient.

Currently focused on the client-side webfetch allowlist gating in `webfetch_wrapper` —
the security-critical decision of whether a URL the model wants to fetch is permitted.
The actual fetch (`api.webfetch_fetch`, HTTP to the server) is monkeypatched.
"""

import json

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
            text, metadata = llmclient.webfetch_wrapper("https://evil.com/x")  # denial returns (text, metadata)
        assert "not on the configured allowlist" in text
        assert metadata == {"webfetch_denied_host": "evil.com"}  # structured marker for the GUI override
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
        text, metadata = llmclient.webfetch_wrapper("https://surprise.com/x")
        assert "not on the configured allowlist" in text
        assert metadata == {"webfetch_denied_host": "surprise.com"}
        assert fake_fetch == []

    def test_canonical_refusal_names_the_host(self, monkeypatch, fake_fetch):
        _set_allowlist(monkeypatch, ["doi.org"])
        with dyn.let(tool_context=env(webfetch_allowed_hosts=frozenset())):
            text, metadata = llmclient.webfetch_wrapper("https://blocked.example/path")
        assert "blocked.example" in text
        assert metadata["webfetch_denied_host"] == "blocked.example"


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
            denied_text, denied_metadata = llmclient.webfetch_wrapper("https://blog.example/post")
            assert "not on the configured allowlist" in denied_text
            assert denied_metadata == {"webfetch_denied_host": "blog.example"}
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


class TestPerformToolCallsMetadata:
    """A tool entrypoint may return `(text, metadata)`; perform_tool_calls threads the metadata
    onto the tool-response record (which scaffold then stores in generation_metadata)."""

    @staticmethod
    def _settings(entrypoint):
        return env(personas={"tool": None, "assistant": "AI", "user": "U", "system": None},
                   tool_entrypoints={"mytool": entrypoint})

    @staticmethod
    def _message():
        return {"role": "assistant", "content": "",
                "tool_calls": [{"type": "function", "function": {"name": "mytool", "arguments": "{}"},
                                "id": "call_1", "index": "0"}]}

    def test_tuple_return_attaches_metadata(self):
        settings = self._settings(lambda: ("the result text", {"webfetch_denied_host": "example.com"}))
        records = llmclient.perform_tool_calls(settings, self._message(), on_call_start=None, on_call_done=None)
        assert len(records) == 1
        assert records[0].data["content"] == "the result text"
        assert records[0].tool_metadata == {"webfetch_denied_host": "example.com"}

    def test_plain_string_return_has_no_metadata(self):
        settings = self._settings(lambda: "just text")
        records = llmclient.perform_tool_calls(settings, self._message(), on_call_start=None, on_call_done=None)
        assert records[0].data["content"] == "just text"
        assert "tool_metadata" not in records[0]


# ---------------------------------------------------------------------------
# Streaming tool-call accumulator (brief 02 §2) — pure helpers
# ---------------------------------------------------------------------------

class TestToolCallAccumulator:
    def test_incremental_fragments_concatenate_arguments(self):
        # LM Studio / OpenAI shape: first fragment carries id/type/name + empty args; later fragments
        # carry only `function.arguments` pieces to concatenate.
        acc = {}
        llmclient._accumulate_tool_call_delta(acc, [{"index": 0, "id": "call_1", "type": "function",
                                                     "function": {"name": "get_weather", "arguments": ""}}])
        llmclient._accumulate_tool_call_delta(acc, [{"index": 0, "type": "function",
                                                     "function": {"arguments": '{"location":'}}])
        llmclient._accumulate_tool_call_delta(acc, [{"index": 0, "function": {"arguments": '"Tokyo"}'}}])
        assert llmclient._materialize_tool_calls(acc) == [
            {"type": "function", "function": {"name": "get_weather", "arguments": '{"location":"Tokyo"}'},
             "id": "call_1", "index": "0"}]

    def test_whole_object_in_one_delta_ooba(self):
        acc = {}
        llmclient._accumulate_tool_call_delta(acc, [{"index": 0, "id": "call_x", "type": "function",
                                                     "function": {"name": "websearch", "arguments": '{"query":"raven"}'}}])
        assert llmclient._materialize_tool_calls(acc) == [
            {"type": "function", "function": {"name": "websearch", "arguments": '{"query":"raven"}'},
             "id": "call_x", "index": "0"}]

    def test_parallel_calls_keyed_by_index(self):
        acc = {}
        llmclient._accumulate_tool_call_delta(acc, [{"index": 0, "id": "a", "type": "function", "function": {"name": "get_weather", "arguments": ""}}])
        llmclient._accumulate_tool_call_delta(acc, [{"index": 1, "id": "b", "type": "function", "function": {"name": "get_weather", "arguments": ""}}])
        llmclient._accumulate_tool_call_delta(acc, [{"index": 0, "function": {"arguments": '{"location":"Tokyo"}'}}])
        llmclient._accumulate_tool_call_delta(acc, [{"index": 1, "function": {"arguments": '{"location":"Paris"}'}}])
        out = llmclient._materialize_tool_calls(acc)
        assert [c["id"] for c in out] == ["a", "b"]
        assert [c["index"] for c in out] == ["0", "1"]
        assert out[0]["function"]["arguments"] == '{"location":"Tokyo"}'
        assert out[1]["function"]["arguments"] == '{"location":"Paris"}'

    def test_empty_accumulator_is_none(self):
        assert llmclient._materialize_tool_calls({}) is None


# ---------------------------------------------------------------------------
# invoke stream robustness (brief 02 §1) — [DONE] sentinel, error events, null content
# ---------------------------------------------------------------------------

class _FakeEvent:
    def __init__(self, data):
        self.data = data

class _FakeSSEClient:
    def __init__(self, datas):
        self._datas = datas
    def events(self):
        for d in self._datas:
            yield _FakeEvent(d)
    def close(self):
        pass

class _FakeResponse:
    status_code = 200


@pytest.fixture
def invoke_settings(llm_settings):
    """Augment the shared `llm_settings` with the fields `invoke` reads off the wire path."""
    llm_settings.request_data = {"stream": True, "messages": [], "tools": []}
    llm_settings.stopping_strings = []
    llm_settings.backend_url = "http://test-backend"
    return llm_settings


def _fake_stream(monkeypatch, payloads):
    """Make `invoke` read `payloads` as its SSE stream. Each item is a dict (JSON-encoded) or a raw
    string like '[DONE]'. Patches the HTTP POST and the SSE client."""
    datas = [p if isinstance(p, str) else json.dumps(p) for p in payloads]
    monkeypatch.setattr(llmclient.requests, "post", lambda *a, **k: _FakeResponse())
    monkeypatch.setattr(llmclient.sseclient, "SSEClient", lambda resp: _FakeSSEClient(datas))


class TestInvokeStreamRobustness:
    def test_done_sentinel_null_content_and_usage(self, monkeypatch, invoke_settings):
        # `content: null` on the priming delta (must not crash io.write), a usage-only final chunk
        # (empty `choices`), and a `[DONE]` sentinel (not JSON — must be skipped).
        _fake_stream(monkeypatch, [
            {"choices": [{"delta": {"role": "assistant", "content": None}}]},
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": " world"}}]},
            {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}},
            "[DONE]",
        ])
        out = llmclient.invoke(invoke_settings, [{"role": "user", "content": "hi"}], tools_enabled=False)
        assert "Hello world" in out.data["content"]
        assert out.n_tokens == 2  # from real usage, not the n_chunks-2 heuristic
        assert out.usage["prompt_tokens"] == 10
        assert not out.data["tool_calls"]  # create_chat_message normalizes "no tool calls" to []

    def test_error_event_raises_runtimeerror(self, monkeypatch, invoke_settings):
        # LM Studio reports backend errors as HTTP 200 + an SSE error payload with no `choices`.
        _fake_stream(monkeypatch, [
            {"choices": [{"delta": {"content": "partial"}}]},
            {"error": {"message": "Error rendering prompt with jinja template"}},
        ])
        with pytest.raises(RuntimeError, match="jinja template"):
            llmclient.invoke(invoke_settings, [{"role": "user", "content": "hi"}], tools_enabled=False)

    def test_streamed_tool_call_materialized(self, monkeypatch, invoke_settings):
        _fake_stream(monkeypatch, [
            {"choices": [{"delta": {"role": "assistant", "content": None,
                                    "tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                                    "function": {"name": "get_weather", "arguments": ""}}]}}]},
            {"choices": [{"delta": {"content": None,
                                    "tool_calls": [{"index": 0, "type": "function",
                                                    "function": {"arguments": '{"location":"Tokyo"}'}}]}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
             "usage": {"prompt_tokens": 291, "completion_tokens": 27, "total_tokens": 318}},
            "[DONE]",
        ])
        out = llmclient.invoke(invoke_settings, [{"role": "user", "content": "weather?"}], tools_enabled=True)
        assert out.data["tool_calls"] == [
            {"type": "function", "function": {"name": "get_weather", "arguments": '{"location":"Tokyo"}'},
             "id": "call_1", "index": "0"}]
        assert out.n_tokens == 27

    def test_token_count_falls_back_to_chunks_without_usage(self, monkeypatch, invoke_settings):
        # A backend that reports no usage (e.g. ignores stream_options, or an interrupt closed the stream
        # early): n_tokens estimates from the chunk count minus the single priming/overhead delta.
        _fake_stream(monkeypatch, [
            {"choices": [{"delta": {"role": "assistant", "content": None}}]},  # priming overhead
            {"choices": [{"delta": {"content": "one"}}]},
            {"choices": [{"delta": {"content": " two"}}]},
        ])
        out = llmclient.invoke(invoke_settings, [{"role": "user", "content": "hi"}], tools_enabled=False)
        assert out.usage is None
        assert out.n_tokens == 2  # 3 chunks - 1 overhead

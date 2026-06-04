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
    llm_settings.backend_flavor = "lmstudio"
    llm_settings.tokenizer = None
    llm_settings.char_to_token_ratio = 0.27
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


# ---------------------------------------------------------------------------
# Backend detection + model identity (brief 02 §0/§3/§4/§5)
# ---------------------------------------------------------------------------

class _FakeGetResponse:
    def __init__(self, payload):
        self._payload = payload
    def json(self):
        return self._payload


def _route_get(routes):
    """A fake `requests.get` mapping URL-substring -> payload. Unmatched URLs return LM Studio's
    200-with-error-body shape (so they look like an unknown endpoint, not a discriminating field)."""
    def _get(url, *args, **kwargs):
        for key, payload in routes.items():
            if key in url:
                return _FakeGetResponse(payload)
        return _FakeGetResponse({"error": {"message": "Unexpected endpoint or method."}})
    return _get


class TestDetectBackendFlavor:
    def test_lmstudio_via_native_endpoint(self, monkeypatch):
        monkeypatch.setattr(llmclient.requests, "get", _route_get({
            "/api/v0/models": {"data": [{"id": "qwen3.5-4b", "state": "loaded", "arch": "qwen35"}]}}))
        assert llmclient.detect_backend_flavor("http://x") == "lmstudio"

    def test_oobabooga_when_native_endpoint_absent(self, monkeypatch):
        # /api/v0/models hits the default error-body (not LM Studio); /v1/internal/model/info has model_name.
        monkeypatch.setattr(llmclient.requests, "get", _route_get({
            "/v1/internal/model/info": {"model_name": "Qwen3-4B.gguf"}}))
        assert llmclient.detect_backend_flavor("http://x") == "oobabooga"

    def test_generic_when_neither_field_present(self, monkeypatch):
        # Both probes return the error-body default — neither `data` list nor `model_name`.
        monkeypatch.setattr(llmclient.requests, "get", _route_get({}))
        assert llmclient.detect_backend_flavor("http://x") == "generic"

    def test_status_200_with_error_body_is_not_oobabooga(self, monkeypatch):
        # The real gotcha: LM Studio returns 200 + {"error": ...} for /v1/internal/model/info. Detection keys
        # on the `model_name` field, not the status, so this must NOT be misread as ooba.
        monkeypatch.setattr(llmclient.requests, "get", _route_get({
            "/api/v0/models": {"data": [{"id": "m", "state": "loaded"}]},
            "/v1/internal/model/info": {"error": {"message": "Unexpected endpoint"}}}))
        assert llmclient.detect_backend_flavor("http://x") == "lmstudio"


class TestModelInfoResolution:
    def test_lmstudio_rich_label_and_context(self, monkeypatch):
        monkeypatch.setattr(llmclient.requests, "get", _route_get({
            "/api/v0/models": {"data": [
                {"id": "other", "state": "not-loaded", "quantization": "Q8", "max_context_length": 262144},
                {"id": "qwen3.5-4b", "state": "loaded", "quantization": "Q4_K_XL", "loaded_context_length": 131072}]}}))
        info = llmclient._resolve_model_info("http://x", "lmstudio")
        assert info.label == "qwen3.5-4b, Q4_K_XL, 128 Ki context"
        assert info.model_id == "qwen3.5-4b"
        assert info.context_length == 131072

    def test_oobabooga_filename_label_no_context(self, monkeypatch):
        monkeypatch.setattr(llmclient.requests, "get", _route_get({
            "/v1/internal/model/info": {"model_name": "Qwen3-4B-Thinking.gguf"}}))
        info = llmclient._resolve_model_info("http://x", "oobabooga")
        assert info.label == "Qwen3-4B-Thinking.gguf"
        assert info.model_id == "Qwen3-4B-Thinking.gguf"
        assert info.context_length is None

    def test_generic_single_model_named(self, monkeypatch):
        monkeypatch.setattr(llmclient.requests, "get", _route_get({
            "/v1/models": {"data": [{"id": "the-only-model"}]}}))
        info = llmclient._resolve_model_info("http://x", "generic")
        assert info.label == "the-only-model"

    def test_generic_ambiguous_never_guesses(self, monkeypatch):
        monkeypatch.setattr(llmclient.requests, "get", _route_get({
            "/v1/models": {"data": [{"id": "a"}, {"id": "b"}]}}))
        info = llmclient._resolve_model_info("http://x", "generic")
        assert info.label == llmclient.NO_MODEL_INFO

    def test_lmstudio_jit_idle_nothing_loaded(self, monkeypatch):
        # No model resident (all not-loaded) and no configured llm_model -> honest "no info", not a guess.
        monkeypatch.setattr(llmclient.requests, "get", _route_get({
            "/api/v0/models": {"data": [{"id": "a", "state": "not-loaded"}, {"id": "b", "state": "not-loaded"}]}}))
        monkeypatch.setattr(llmclient.librarian_config, "llm_model", None)
        info = llmclient._resolve_model_info("http://x", "lmstudio")
        assert info.label == llmclient.NO_MODEL_INFO
        assert info.context_length is None


# ---------------------------------------------------------------------------
# Token counting tiers + usage calibration (brief 02 §7)
# ---------------------------------------------------------------------------

class _FakeTokenizer:
    """Deterministic stand-in: one 'token' per character, so counts are easy to assert."""
    def encode(self, text):
        return list(text)


class TestCountTokens:
    def test_tier1_local_tokenizer_is_exact(self, invoke_settings):
        invoke_settings.tokenizer = _FakeTokenizer()
        count, is_exact = llmclient.count_tokens(invoke_settings, "hello")
        assert (count, is_exact) == (5, True)  # 5 chars -> 5 fake tokens, exact

    def test_tier2_oobabooga_endpoint_is_exact(self, monkeypatch, invoke_settings):
        invoke_settings.tokenizer = None
        invoke_settings.backend_flavor = "oobabooga"
        monkeypatch.setattr(llmclient.requests, "post", lambda *a, **k: _FakeGetResponse({"length": 42}))
        assert llmclient.count_tokens(invoke_settings, "whatever") == (42, True)

    def test_tier3_calibrated_estimate_is_not_exact(self, invoke_settings):
        invoke_settings.tokenizer = None
        invoke_settings.backend_flavor = "lmstudio"
        invoke_settings.char_to_token_ratio = 0.25
        count, is_exact = llmclient.count_tokens(invoke_settings, "x" * 40)
        assert (count, is_exact) == (10, False)  # round(40 * 0.25) = 10, estimate


class TestUsageCalibration:
    def test_ratio_refined_from_prompt_usage(self, monkeypatch, invoke_settings):
        # Calibration divides prompt_tokens by the chars actually sent. invoke scrubs the history (which adds
        # the "User: " persona prefix), so compute the expected ratio from what `on_prompt_ready` reports.
        sent = {}
        def capture(history):
            sent["chars"] = sum(len(m.get("content") or "") for m in history)
        _fake_stream(monkeypatch, [
            {"choices": [{"delta": {"content": "ok"}}]},
            {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}},
            "[DONE]",
        ])
        llmclient.invoke(invoke_settings, [{"role": "user", "content": "x" * 40}], tools_enabled=False, on_prompt_ready=capture)
        assert invoke_settings.char_to_token_ratio == pytest.approx(10 / sent["chars"])

    def test_mismatched_tokenizer_warns(self, monkeypatch, caplog, invoke_settings):
        # Tokenizer counts 100 tokens for the content alone; backend reports only 50 for the full prompt ->
        # the tokenizer over-counts (wrong vocab) and must warn.
        invoke_settings.tokenizer = _FakeTokenizer()  # one token per char -> 100 for a 100-char prompt
        _fake_stream(monkeypatch, [
            {"choices": [{"delta": {"content": "ok"}}]},
            {"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 1, "total_tokens": 51}},
            "[DONE]",
        ])
        import logging
        caplog.set_level(logging.WARNING, logger="raven.librarian.llmclient")
        llmclient.invoke(invoke_settings, [{"role": "user", "content": "x" * 100}], tools_enabled=False)
        assert any("does not match the served model" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Idle context-prefill (brief 02 §7, tier 3)
# ---------------------------------------------------------------------------

class TestPrefill:
    def test_prefill_caps_at_one_token_and_returns_usage(self, monkeypatch, invoke_settings):
        # prefill must minimize generation (one token) while reading back the exact prompt size.
        sent = {}
        def capturing_post(url, **kwargs):
            sent["data"] = kwargs["json"]
            return _FakeResponse()
        monkeypatch.setattr(llmclient.requests, "post", capturing_post)
        monkeypatch.setattr(llmclient.sseclient, "SSEClient",
                            lambda resp: _FakeSSEClient([json.dumps(
                                {"choices": [], "usage": {"prompt_tokens": 123, "completion_tokens": 1, "total_tokens": 124}})]))
        out = llmclient.prefill(invoke_settings, [{"role": "user", "content": "hi"}], tools_enabled=False)
        assert sent["data"]["max_tokens"] == 1  # overrides the configured cap for this call
        assert out.usage["prompt_tokens"] == 123

    def test_prefill_returns_none_on_failure(self, monkeypatch, invoke_settings):
        # Backend down / template render error: prefill swallows it and returns None (caller keeps the estimate).
        def boom(*a, **k):
            raise RuntimeError("backend down")
        monkeypatch.setattr(llmclient.requests, "post", boom)
        assert llmclient.prefill(invoke_settings, [{"role": "user", "content": "hi"}]) is None


class TestSetupOutputCap:
    """The per-turn `max_tokens` cap merge in `setup` (a `None` value means 'no cap')."""

    def _patch_generic_backend(self, monkeypatch):
        # Minimal generic backend: one model listed, neither the LM-Studio-native nor the ooba endpoint present.
        monkeypatch.setattr(llmclient.requests, "get", _route_get({
            "/v1/models": {"data": [{"id": "test-model"}]},
        }))
        monkeypatch.setattr(llmclient.librarian_config, "llm_backend_flavor", None)
        monkeypatch.setattr(llmclient.librarian_config, "llm_tokenizer_path", None)

    def test_none_max_tokens_becomes_context_length(self, monkeypatch):
        self._patch_generic_backend(monkeypatch)
        monkeypatch.setattr(llmclient.librarian_config, "llm_sampler_config", {"max_tokens": None, "temperature": 1})
        settings = llmclient.setup("http://test-backend", quiet=True)
        assert settings.request_data["max_tokens"] == settings.context_length

    def test_real_max_tokens_is_preserved(self, monkeypatch):
        self._patch_generic_backend(monkeypatch)
        monkeypatch.setattr(llmclient.librarian_config, "llm_sampler_config", {"max_tokens": 1234})
        settings = llmclient.setup("http://test-backend", quiet=True)
        assert settings.request_data["max_tokens"] == 1234

    def test_none_valued_sampler_key_is_dropped(self, monkeypatch):
        self._patch_generic_backend(monkeypatch)
        monkeypatch.setattr(llmclient.librarian_config, "llm_sampler_config", {"max_tokens": 800, "min_p": None})
        settings = llmclient.setup("http://test-backend", quiet=True)
        assert "min_p" not in settings.request_data  # None -> field omitted (use backend default)

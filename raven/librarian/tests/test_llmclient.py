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

from raven.librarian import chatutil  # noqa: E402 -- after importorskip by design


def _history(text):
    """A one-message user history in content-parts shape — what `invoke` receives in production."""
    return [{"role": "user", "content": [chatutil.text_content_part(text)]}]


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
        assert chatutil.content_to_text(records[0].data["content"]) == "the result text"
        assert records[0].tool_metadata == {"webfetch_denied_host": "example.com"}

    def test_plain_string_return_has_no_metadata(self):
        settings = self._settings(lambda: "just text")
        records = llmclient.perform_tool_calls(settings, self._message(), on_call_start=None, on_call_done=None)
        assert chatutil.content_to_text(records[0].data["content"]) == "just text"
        assert "tool_metadata" not in records[0]

    def test_parts_return_becomes_multipart_content(self):
        # brief 03 §4: an entrypoint may return a content-parts list (one part per result); it is used verbatim
        # as the tool message's content (not collapsed into a single part).
        parts = [chatutil.text_content_part("result 1\n"), chatutil.text_content_part("result 2\n")]
        settings = self._settings(lambda: parts)
        records = llmclient.perform_tool_calls(settings, self._message(), on_call_start=None, on_call_done=None)
        assert records[0].data["content"] == parts
        assert chatutil.content_to_text(records[0].data["content"]) == "result 1\nresult 2\n"

    def test_parts_return_with_metadata_tuple(self):
        # The `(output, metadata)` tuple form composes with a parts-list output, not just a string.
        parts = [chatutil.text_content_part("x")]
        settings = self._settings(lambda: (parts, {"webfetch_denied_host": "example.com"}))
        records = llmclient.perform_tool_calls(settings, self._message(), on_call_start=None, on_call_done=None)
        assert records[0].data["content"] == parts
        assert records[0].tool_metadata == {"webfetch_denied_host": "example.com"}


class TestWebsearchWrapper:
    """brief 03 §4: websearch returns one text content-part per result, with each field normalized."""

    @staticmethod
    def _patch_search(monkeypatch, data):
        monkeypatch.setattr(llmclient.api, "websearch_search", lambda *a, **k: {"data": data})

    def test_one_text_part_per_result_with_markdown_links(self, monkeypatch):
        self._patch_search(monkeypatch, [
            {"title": "First", "link": "https://example.com/1", "text": "snippet one"},
            {"title": "Second", "link": "https://example.com/2", "text": "snippet two"},
        ])
        parts = llmclient.websearch_wrapper("query")
        assert len(parts) == 2
        assert all(p["type"] == "text" for p in parts)
        assert "[First](https://example.com/1)" in parts[0]["text"]
        assert "snippet one" in parts[0]["text"]
        assert "[Second](https://example.com/2)" in parts[1]["text"]

    def test_fields_are_normalized(self, monkeypatch):
        # Invisible-injection glyphs in scraped SERP content must be stripped (hostile input).
        zwsp = "\u200b"  # zero-width space — a classic injection glyph that normalize removes
        self._patch_search(monkeypatch, [
            {"title": f"Ti{zwsp}tle", "link": f"https://e.com/{zwsp}x", "text": f"bo{zwsp}dy"},
        ])
        text = llmclient.websearch_wrapper("q")[0]["text"]
        assert zwsp not in text  # removed from title, link, and body
        assert "Title" in text and "body" in text

    def test_result_without_title_falls_back_to_bare_url(self, monkeypatch):
        self._patch_search(monkeypatch, [{"link": "https://e.com/x", "text": "body"}])
        text = llmclient.websearch_wrapper("q")[0]["text"]
        assert "<https://e.com/x>" in text

    @staticmethod
    def _patch_capture_engine(monkeypatch):
        """Patch `api.websearch_search` to record the engine it was called with; return the capture dict."""
        captured = {}
        def fake_search(query, engine, num):
            captured["engine"] = engine
            return {"data": []}
        monkeypatch.setattr(llmclient.api, "websearch_search", fake_search)
        return captured

    def test_uses_configured_engine_by_default(self, monkeypatch):
        # The LLM tool passes only a query; the engine comes from config (host choice, not model choice).
        captured = self._patch_capture_engine(monkeypatch)
        monkeypatch.setattr(llmclient.librarian_config, "websearch_engine", "google")
        llmclient.websearch_wrapper("q")
        assert captured["engine"] == "google"

    def test_explicit_engine_overrides_config(self, monkeypatch):
        captured = self._patch_capture_engine(monkeypatch)
        monkeypatch.setattr(llmclient.librarian_config, "websearch_engine", "google")
        llmclient.websearch_wrapper("q", engine="duckduckgo")
        assert captured["engine"] == "duckduckgo"


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
        out = llmclient.invoke(invoke_settings, _history("hi"), tools_enabled=False)
        assert "Hello world" in chatutil.content_to_text(out.data["content"])
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
            llmclient.invoke(invoke_settings, _history("hi"), tools_enabled=False)

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
        out = llmclient.invoke(invoke_settings, _history("weather?"), tools_enabled=True)
        assert out.data["tool_calls"] == [
            {"type": "function", "function": {"name": "get_weather", "arguments": '{"location":"Tokyo"}'},
             "id": "call_1", "index": "0"}]
        assert out.n_tokens == 27

    def test_token_count_falls_back_to_chunks_without_usage(self, monkeypatch, invoke_settings):
        # A backend that reports no usage (e.g. ignores stream_options, or an interrupt closed the stream
        # early): n_tokens estimates from the count of text-bearing deltas (the empty priming delta is not counted).
        _fake_stream(monkeypatch, [
            {"choices": [{"delta": {"role": "assistant", "content": None}}]},  # priming overhead (empty -> not counted)
            {"choices": [{"delta": {"content": "one"}}]},
            {"choices": [{"delta": {"content": " two"}}]},
        ])
        out = llmclient.invoke(invoke_settings, _history("hi"), tools_enabled=False)
        assert out.usage is None
        assert out.n_tokens == 2  # two text-bearing deltas


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


class TestImageTokenCost:
    """Per-image context-fill estimate: family match against the config table (first match wins, None fallback)."""

    @staticmethod
    def _settings(model=None, model_id=None):
        return env(model=model, model_id=model_id)

    def test_flat_family_ignores_dimensions(self):
        assert llmclient.image_token_cost(self._settings(model_id="gemma4"), 512, 512) == 1120
        assert llmclient.image_token_cost(self._settings(model_id="gemma4"), 64, 64) == 1120  # flat: dims don't matter

    def test_callable_family_scales_with_resolution(self):
        # Qwen-VL: ceil(h/28) * ceil(w/28), capped at 16384. 1024x1024 -> 37*37 = 1369.
        assert llmclient.image_token_cost(self._settings(model="Qwen3-VL-4B"), 1024, 1024) == 1369

    def test_more_specific_key_wins_over_prefix(self):
        # "llava-1.5" must match before the plainer "llava" (table order is specific-first).
        assert llmclient.image_token_cost(self._settings(model_id="llava-1.5-7b"), 100, 100) == 576
        assert llmclient.image_token_cost(self._settings(model_id="llava-v1.6-mistral"), 100, 100) == 2880

    def test_unknown_family_uses_none_fallback(self):
        assert llmclient.image_token_cost(self._settings(model="some-mystery-model"), 100, 100) == 1000

    def test_matches_across_model_and_model_id(self):
        # The haystack is model + model_id, so a family named in either field is found.
        assert llmclient.image_token_cost(self._settings(model="gemma4-27b", model_id=None), 1, 1) == 1120
        assert llmclient.image_token_cost(self._settings(model=None, model_id="google/gemma4"), 1, 1) == 1120


class TestUsageCalibration:
    def test_ratio_refined_from_prompt_usage(self, monkeypatch, invoke_settings):
        # Calibration divides prompt_tokens by the chars actually sent. invoke scrubs the history (which adds
        # the "User: " persona prefix), so compute the expected ratio from what `on_prompt_ready` reports.
        sent = {}
        def capture(history):
            sent["chars"] = sum(len(chatutil.content_to_text(m.get("content"))) for m in history)
        _fake_stream(monkeypatch, [
            {"choices": [{"delta": {"content": "ok"}}]},
            {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}},
            "[DONE]",
        ])
        llmclient.invoke(invoke_settings, _history("x" * 40), tools_enabled=False, on_prompt_ready=capture)
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
        llmclient.invoke(invoke_settings, _history("x" * 100), tools_enabled=False)
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
        out = llmclient.prefill(invoke_settings, _history("hi"), tools_enabled=False)
        assert sent["data"]["max_tokens"] == 1  # overrides the configured cap for this call
        assert out.usage["prompt_tokens"] == 123

    def test_prefill_returns_none_on_failure(self, monkeypatch, invoke_settings):
        # Backend down / template render error: prefill swallows it and returns None (caller keeps the estimate).
        def boom(*a, **k):
            raise RuntimeError("backend down")
        monkeypatch.setattr(llmclient.requests, "post", boom)
        assert llmclient.prefill(invoke_settings, _history("hi")) is None


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


class TestStreamParser:
    """The `StreamParser` — `invoke`'s single parser of the response stream (brief 02 §9)."""

    @staticmethod
    def _run(deltas, native_tool_calls=None):
        """Feed `(content, reasoning)` tuples through a parser and finalize; return the flat event list."""
        parser = llmclient.StreamParser()
        events = []
        for content, reasoning in deltas:
            events.extend(parser.feed(content, reasoning))
        events.extend(parser.finalize(native_tool_calls))
        return events

    @staticmethod
    def _texts(events, etype):
        return "".join(e["text"] for e in events if e["type"] == etype)

    def test_plain_content_passes_through(self):
        events = self._run([("Hello", ""), (" world", "")])
        assert self._texts(events, "content") == "Hello world"
        assert not any(e["type"] == "reasoning" for e in events)

    def test_native_reasoning_channel(self):
        # reasoning_content deltas (LM Studio / llama.cpp) become reasoning events, content stays separate.
        events = self._run([("", "thinking"), ("", " hard"), ("answer", "")])
        assert self._texts(events, "reasoning") == "thinking hard"
        assert self._texts(events, "content") == "answer"

    def test_inline_think_extracted_from_content(self):
        # ooba-style: <think> arrives inline in the content stream; the parser routes it to reasoning and strips the tags.
        events = self._run([("<think>pondering</think>the answer", "")])
        assert self._texts(events, "reasoning") == "pondering"
        assert self._texts(events, "content") == "the answer"

    def test_inline_think_split_across_chunks(self):
        # Tags split at chunk boundaries (`</thi` | `nk>`) must still be recognized; no tag text leaks to content.
        events = self._run([("<thi", ""), ("nk>secret th", ""), ("oughts</thi", ""), ("nk>visible", "")])
        assert self._texts(events, "reasoning") == "secret thoughts"
        assert self._texts(events, "content") == "visible"
        assert "<think>" not in self._texts(events, "content")
        assert "think" not in self._texts(events, "content")

    def test_inline_gemma_channel_extracted_from_content(self):
        # Gemma 3/4 spell the reasoning channel as `<|channel>thought ... <channel|>`. A passthrough backend
        # (ooba/generic) delivers it inline; the parser must route it to reasoning, same as `<think>`.
        events = self._run([("<|channel>thought\nThe user wants Paris weather.\n<channel|>The answer.", "")])
        assert self._texts(events, "reasoning").strip() == "The user wants Paris weather."
        assert self._texts(events, "content") == "The answer."
        assert "channel" not in self._texts(events, "content")  # no marker text leaks

    def test_inline_gemma_channel_split_across_chunks(self):
        # Both the asymmetric open (`<|channel>thought`) and close (`<channel|>`) markers may straddle a chunk
        # boundary; the look-ahead must still recognize them without leaking marker text to content.
        events = self._run([("<|chan", ""), ("nel>thoughtsecret th", ""), ("oughts<chan", ""), ("nel|>visible", "")])
        assert self._texts(events, "reasoning") == "secret thoughts"
        assert self._texts(events, "content") == "visible"
        assert "channel" not in self._texts(events, "content")

    def test_inline_gemma_unterminated_channel_flushed_at_finalize(self):
        # Stream cut off mid-thought (interrupt): the buffered Gemma reasoning flushes, not lost.
        events = self._run([("<|channel>thought\ncut off mid-thou", "")])
        assert self._texts(events, "reasoning").strip() == "cut off mid-thou"

    def test_less_than_that_is_not_a_tag_passes_through(self):
        # A bare '<' (e.g. an inequality) is not a tag prefix worth holding forever; it streams as content.
        events = self._run([("if a < b then", "")])
        assert self._texts(events, "content") == "if a < b then"

    def test_inline_tool_call_emits_event(self):
        events = self._run([('<tool_call>{"name": "websearch", "arguments": {"query": "ravens"}}</tool_call>', "")])
        calls = [e for e in events if e["type"] == "tool_call"]
        assert len(calls) == 1
        assert calls[0]["name"] == "websearch"
        assert json.loads(calls[0]["arguments"]) == {"query": "ravens"}
        assert calls[0]["id"].startswith("inline_")  # inline calls get a synthetic id
        assert not self._texts(events, "content")  # the tag span is fully consumed

    def test_inline_tool_call_split_across_chunks(self):
        events = self._run([("<tool_call>{\"name\": \"web", ""), ("search\", \"arguments\": {}}</tool", ""), ("_call>", "")])
        calls = [e for e in events if e["type"] == "tool_call"]
        assert len(calls) == 1
        assert calls[0]["name"] == "websearch"

    def test_native_tool_call_emitted_at_finalize(self):
        native = [{"id": "call_9", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"Tokyo"}'}}]
        events = self._run([("", "")], native_tool_calls=native)
        calls = [e for e in events if e["type"] == "tool_call"]
        assert len(calls) == 1
        assert calls[0]["id"] == "call_9"
        assert calls[0]["name"] == "get_weather"

    def test_dedup_inline_and_native_same_call(self):
        # Some backends emit a call BOTH inline and in the structured field — exactly one event must result.
        native = [{"id": "call_x", "type": "function",
                   "function": {"name": "websearch", "arguments": '{"query": "ravens"}'}}]
        events = self._run(
            [('<tool_call>{"name": "websearch", "arguments": {"query": "ravens"}}</tool_call>', "")],
            native_tool_calls=native)
        calls = [e for e in events if e["type"] == "tool_call"]
        assert len(calls) == 1  # the structured duplicate is suppressed
        assert calls[0]["id"].startswith("inline_")  # the inline one (emitted first) wins

    def test_two_genuinely_distinct_native_calls_both_survive(self):
        # No inline duplication: two native calls (even same name) are both real and must both emit.
        native = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"x": 1}'}},
                  {"id": "c2", "type": "function", "function": {"name": "f", "arguments": '{"x": 2}'}}]
        events = self._run([("", "")], native_tool_calls=native)
        assert len([e for e in events if e["type"] == "tool_call"]) == 2

    def test_unterminated_think_flushed_at_finalize(self):
        # Stream ends mid-think (interrupt): the buffered thinking is not lost — it flushes as a reasoning event.
        events = self._run([("<think>cut off mid-thou", "")])
        assert self._texts(events, "reasoning") == "cut off mid-thou"

    def test_think_then_content_ordering(self):
        events = self._run([("", "step one"), ("", "step two"), ("Final.", "")])
        types = [e["type"] for e in events if e["text"]]
        assert types == ["reasoning", "reasoning", "content"]


class TestInvokeTypedEvents:
    """`invoke` end-to-end: typed events to `on_progress`, reasoning into `message["reasoning_content"]`."""

    @staticmethod
    def _collect(monkeypatch, invoke_settings, payloads, native_in_message_check=False):
        _fake_stream(monkeypatch, payloads)
        events = []
        out = llmclient.invoke(invoke_settings, _history("hi"),
                               on_progress=lambda ev: events.append(ev) or llmclient.action_ack,
                               tools_enabled=False)
        return out, events

    def test_native_reasoning_lands_in_reasoning_content(self, monkeypatch, invoke_settings):
        # LM Studio streams thinking via delta.reasoning_content — it must surface as reasoning events AND be
        # stored in message["reasoning_content"], never in content. This is the headline brief-02 §9 driver.
        out, events = self._collect(monkeypatch, invoke_settings, [
            {"choices": [{"delta": {"role": "assistant", "reasoning_content": "let me think"}}]},
            {"choices": [{"delta": {"reasoning_content": " about it"}}]},
            {"choices": [{"delta": {"content": "The answer is 42."}}]},
            {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}},
            "[DONE]",
        ])
        assert out.data["reasoning_content"] == "let me think about it"
        assert chatutil.content_to_text(out.data["content"]) == "The answer is 42."
        assert "think" not in chatutil.content_to_text(out.data["content"])
        reasoning_events = [e for e in events if e["type"] == "reasoning"]
        assert "".join(e["text"] for e in reasoning_events) == "let me think about it"
        assert all("n_chunks" in e for e in events if e["type"] in ("content", "reasoning"))

    def test_inline_think_routed_to_reasoning_content(self, monkeypatch, invoke_settings):
        # ooba-style inline <think> in content: same destination as the native channel, content left clean.
        out, events = self._collect(monkeypatch, invoke_settings, [
            {"choices": [{"delta": {"role": "assistant", "content": "<think>hmm</think>Done."}}]},
            "[DONE]",
        ])
        assert out.data["reasoning_content"] == "hmm"
        assert chatutil.content_to_text(out.data["content"]) == "Done."

    def test_no_reasoning_means_no_field(self, monkeypatch, invoke_settings):
        # A plain answer with no thinking: reasoning_content is omitted entirely (not stored as "").
        out, events = self._collect(monkeypatch, invoke_settings, [
            {"choices": [{"delta": {"role": "assistant", "content": "Just answering."}}]},
            "[DONE]",
        ])
        assert "reasoning_content" not in out.data

    def test_inline_tool_call_in_message(self, monkeypatch, invoke_settings):
        out, events = self._collect(monkeypatch, invoke_settings, [
            {"choices": [{"delta": {"role": "assistant",
                                    "content": '<tool_call>{"name": "websearch", "arguments": {"query": "x"}}</tool_call>'}}]},
            "[DONE]",
        ])
        assert len(out.data["tool_calls"]) == 1
        assert out.data["tool_calls"][0]["function"]["name"] == "websearch"
        assert not chatutil.content_to_text(out.data["content"])  # the tag span never leaks into content
        assert any(e["type"] == "tool_call" for e in events)


# ---------------------------------------------------------------------------
# _serialize_history_for_wire: text scrub + image-part preservation + sidecar resolution
# ---------------------------------------------------------------------------

class TestSerializeHistoryForWire:
    settings = env(personas={"user": "U", "assistant": "AI", "system": None, "tool": None})

    def test_text_only_message_scrubbed_to_single_text_part(self):
        history = _history("hello there")
        out = llmclient._serialize_history_for_wire(self.settings, history, continue_=False)
        assert out[0]["content"] == [chatutil.text_content_part("U: hello there")]  # persona-prefixed, one text part

    def test_input_history_not_mutated(self):
        history = _history("hello")
        llmclient._serialize_history_for_wire(self.settings, history, continue_=False)
        assert history[0]["content"] == [chatutil.text_content_part("hello")]  # deep-copied; original untouched

    def test_image_part_preserved_and_sidecar_resolved(self, tmp_path):
        import base64
        from raven.librarian import chattree
        ds = chattree.PersistentForest(tmp_path / "chat.json", autosave=False)
        raw = b"\x89PNG\r\n\x1a\n" + b"fake-png-bytes"
        filename = ds.store_sidecar(raw, "png")

        history = [{"role": "user", "content": [chatutil.text_content_part("what is this?"),
                                                chatutil.image_content_part(f"sidecar:{filename}")]}]
        out = llmclient._serialize_history_for_wire(self.settings, history, continue_=False, datastore=ds)

        parts = out[0]["content"]
        assert parts[0] == chatutil.text_content_part("U: what is this?")
        assert parts[1]["type"] == "image_url"
        url = parts[1]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        assert base64.b64decode(url.split(",", 1)[1]) == raw  # the model receives the actual bytes
        # stored history still references the sidecar (only the wire copy was substituted)
        assert history[0]["content"][1]["image_url"]["url"] == f"sidecar:{filename}"

    def test_image_part_passes_through_without_datastore(self):
        history = [{"role": "user", "content": [chatutil.text_content_part("x"),
                                                {"type": "image_url", "image_url": {"url": "sidecar:abc.png"}}]}]
        out = llmclient._serialize_history_for_wire(self.settings, history, continue_=False, datastore=None)
        assert out[0]["content"][1]["image_url"]["url"] == "sidecar:abc.png"  # unresolved, but preserved

    def test_continue_leaves_last_message_untouched(self):
        history = [{"role": "user", "content": [chatutil.text_content_part("q")]},
                   {"role": "assistant", "content": [chatutil.text_content_part("partial ans")]}]
        out = llmclient._serialize_history_for_wire(self.settings, history, continue_=True)
        assert out[0]["content"] == [chatutil.text_content_part("U: q")]  # scrubbed
        assert out[1]["content"] == [chatutil.text_content_part("partial ans")]  # last message untouched

"""Unit tests for raven.librarian.llmclient.

Currently focused on the client-side webfetch allowlist gating in `webfetch` —
the security-critical decision of whether a URL the model wants to fetch is permitted.
The actual fetch (`api.webfetch_fetch`, HTTP to the server) is monkeypatched.
"""

import http.server
import json
import logging
import pathlib
import threading
import time

import pytest  # noqa: F401 -- fixtures and marks below
import requests

from unpythonic import dyn
from unpythonic.env import env

from raven.common import netutil
from raven.librarian import chatutil
from raven.librarian import llmclient
from raven.librarian import llmtools

# This module used to open with an `importorskip` on `llmclient`, because importing it reached
# `raven.client.api` and so required `spacy` and `av`, neither of which CI installs. That import is now
# deferred to the two network tool wrappers — which is also why the tests below patch
# `llmtools._client_api` rather than `llmclient.api`. Patching through the seam is what lets this file
# exercise the wrappers without the real client module being importable at all.


def _history(text):
    """A one-message user history in content-parts shape — what `invoke` receives in production."""
    return [{"role": "user", "content": [chatutil.text_content_part(text)]}]


class _StubClientAPI:
    """Stands in for `raven.client.api` in the network-tool tests.

    Patched over `llmtools._client_api`, which is the seam the tool wrappers reach the real module through.
    Going through the seam rather than through the real module is what keeps this test file importable
    without `spacy` and `av` — the two the client stack pulls in and CI does not install.
    """

    def __init__(self, **entrypoints):
        for name, fn in entrypoints.items():
            setattr(self, name, fn)


@pytest.fixture
def fake_fetch(monkeypatch):
    """Replace the HTTP fetch with a recorder; returns the list of URLs that reached the server."""
    fetched_urls = []

    def _fake(url, output_format="markdown"):
        fetched_urls.append(url)
        return {"content": f"CONTENT of {url}", "url": url, "spaSuspected": False, "title": f"TITLE of {url}"}

    monkeypatch.setattr(llmtools, "_client_api", lambda: _StubClientAPI(webfetch_fetch=_fake))
    return fetched_urls


def _set_allowlist(monkeypatch, allowlist):
    monkeypatch.setattr(llmclient.librarian_config, "webfetch_allowlist", allowlist)


class TestToolRegistry:
    """The three halves of the registry have to agree, and nothing checks that at runtime.

    A tool advertised without an entrypoint is a promise the model will try to collect: `perform_tool_calls`
    looks the name up, misses, and spends a round returning an error. One registered without a spec is
    unreachable — the model is never told it exists. Neither shows up as an exception anywhere.
    """

    def test_every_advertised_tool_has_an_entrypoint(self):
        advertised = {t["function"]["name"] for t in llmclient.TOOLS}
        assert advertised <= set(llmclient.TOOL_ENTRYPOINTS), (
            f"advertised with no implementation: {sorted(advertised - set(llmclient.TOOL_ENTRYPOINTS))}")

    def test_every_entrypoint_is_advertised(self):
        advertised = {t["function"]["name"] for t in llmclient.TOOLS}
        assert set(llmclient.TOOL_ENTRYPOINTS) <= advertised, (
            f"implemented but never offered: {sorted(set(llmclient.TOOL_ENTRYPOINTS) - advertised)}")

    def test_the_document_tools_are_a_subset_of_the_registry(self):
        """`maybe_tool_names_for_turn` computes the non-document group by subtracting this set, so a name
        in it that is not a real tool would silently shrink nothing while looking like it gated something."""
        assert llmclient.DOCUMENT_TOOL_NAMES <= set(llmclient.TOOL_ENTRYPOINTS)

    def test_the_gated_groups_are_disjoint(self):
        """A tool answering to two switches would make one of them a lie, whichever way they were set."""
        assert not (llmclient.DOCUMENT_TOOL_NAMES & llmclient.NETWORK_TOOL_NAMES)

    def test_every_gated_name_is_a_registered_tool(self):
        """A typo in either group is silent: the name gates nothing, and the tool it meant stays ungated —
        i.e. permanently on offer, which is the failure direction that does not announce itself."""
        registered = set(llmclient.TOOL_ENTRYPOINTS)
        assert llmclient.DOCUMENT_TOOL_NAMES <= registered
        assert llmclient.NETWORK_TOOL_NAMES <= registered

    def test_the_entrypoints_are_callable(self):
        """The registry is module-level and shared, so a name bound to `None` by accident would surface only
        when a model happened to call that tool."""
        for name, function in llmclient.TOOL_ENTRYPOINTS.items():
            assert callable(function), f"entrypoint for '{name}' is not callable: {function!r}"


class TestWebfetchWrapperGating:
    def test_no_allowlist_fetches_anything(self, monkeypatch, fake_fetch):
        _set_allowlist(monkeypatch, None)
        text, metadata = llmclient.webfetch("https://random-site.com/x")  # success returns (text, metadata)
        assert "CONTENT of" in text
        assert fake_fetch == ["https://random-site.com/x"]

    def test_allowlisted_host_fetches(self, monkeypatch, fake_fetch):
        _set_allowlist(monkeypatch, ["*.arxiv.org"])
        with dyn.let(tool_context=env(webfetch_allowed_hosts=frozenset())):
            text, metadata = llmclient.webfetch("https://arxiv.org/html/2301.1")
        assert "CONTENT of" in text
        assert fake_fetch == ["https://arxiv.org/html/2301.1"]

    def test_non_allowlisted_host_refused(self, monkeypatch, fake_fetch):
        _set_allowlist(monkeypatch, ["*.arxiv.org"])
        with dyn.let(tool_context=env(webfetch_allowed_hosts=frozenset())):
            text, metadata = llmclient.webfetch("https://evil.com/x")  # denial returns (text, metadata)
        assert "not on the configured allowlist" in text
        assert metadata == {"webfetch_denied_host": "evil.com"}  # structured marker for the GUI override
        assert fake_fetch == []  # the request never reached the server

    def test_auto_allowed_host_fetches(self, monkeypatch, fake_fetch):
        # Host not on the configured list, but auto-allowed this turn (user typed it).
        _set_allowlist(monkeypatch, ["*.arxiv.org"])
        with dyn.let(tool_context=env(webfetch_allowed_hosts=frozenset({"user-typed.com"}))):
            text, metadata = llmclient.webfetch("https://user-typed.com/x")
        assert "CONTENT of" in text
        assert fake_fetch == ["https://user-typed.com/x"]

    def test_fail_closed_without_context(self, monkeypatch, fake_fetch):
        # No dyn.let binding at all -> the process-wide empty-env default -> no auto-allow.
        # A non-listed host must be refused (fail closed), not fetched.
        _set_allowlist(monkeypatch, ["*.arxiv.org"])
        text, metadata = llmclient.webfetch("https://surprise.com/x")
        assert "not on the configured allowlist" in text
        assert metadata == {"webfetch_denied_host": "surprise.com"}
        assert fake_fetch == []

    def test_canonical_refusal_names_the_host(self, monkeypatch, fake_fetch):
        _set_allowlist(monkeypatch, ["doi.org"])
        with dyn.let(tool_context=env(webfetch_allowed_hosts=frozenset())):
            text, metadata = llmclient.webfetch("https://blocked.example/path")
        assert "blocked.example" in text
        assert metadata["webfetch_denied_host"] == "blocked.example"


@pytest.fixture
def clean_session_approvals():
    """Isolate the module-level session-approved-hosts set across tests."""
    llmtools._session_approved_hosts.clear()
    yield
    llmtools._session_approved_hosts.clear()


class TestSessionApprovedHosts:
    def test_approve_lets_non_allowlisted_host_through(self, monkeypatch, fake_fetch, clean_session_approvals):
        _set_allowlist(monkeypatch, ["*.arxiv.org"])
        with dyn.let(tool_context=env(webfetch_allowed_hosts=frozenset())):
            denied_text, denied_metadata = llmclient.webfetch("https://blog.example/post")
            assert "not on the configured allowlist" in denied_text
            assert denied_metadata == {"webfetch_denied_host": "blog.example"}
            assert fake_fetch == []  # denied before approval

            llmclient.approve_host_for_session("blog.example")
            allowed_text, _ = llmclient.webfetch("https://blog.example/post")
            assert "CONTENT of" in allowed_text
            assert fake_fetch == ["https://blog.example/post"]  # goes through after approval

    def test_approve_is_case_insensitive(self, monkeypatch, fake_fetch, clean_session_approvals):
        _set_allowlist(monkeypatch, ["doi.org"])
        llmclient.approve_host_for_session("Example.COM")
        with dyn.let(tool_context=env(webfetch_allowed_hosts=frozenset())):
            text, _ = llmclient.webfetch("https://example.com/x")
        assert "CONTENT of" in text

    def test_approval_does_not_apply_when_allowlist_is_none(self, monkeypatch, fake_fetch, clean_session_approvals):
        # With no allowlist there is no gate anyway; approval is simply irrelevant (everything passes).
        _set_allowlist(monkeypatch, None)
        text, _ = llmclient.webfetch("https://anything.example/x")
        assert "CONTENT of" in text
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


class TestPerformToolCallsRefusal:
    """`maybe_refusal_text` answers a whole round of calls without calling anything."""

    @staticmethod
    def _settings(entrypoint):
        return env(personas={"tool": None, "assistant": "AI", "user": "U", "system": None},
                   tool_entrypoints={"mytool": entrypoint})

    @staticmethod
    def _message(*call_ids):
        return {"role": "assistant", "content": "",
                "tool_calls": [{"type": "function", "function": {"name": "mytool", "arguments": "{}"},
                                "id": call_id, "index": str(i)}
                               for i, call_id in enumerate(call_ids)]}

    def test_the_entrypoint_never_runs(self):
        called = []
        settings = self._settings(lambda: called.append(1) or "should not happen")
        records = llmclient.perform_tool_calls(settings, self._message("call_1"),
                                               on_call_start=None, on_call_done=None,
                                               maybe_refusal_text="not now")
        assert called == []
        assert records[0].status == "error"
        assert chatutil.content_to_text(records[0].data["content"]) == "not now"

    def test_every_call_in_the_round_is_answered(self):
        # The OAI protocol wants one tool result per requested call; a partial round leaves the model
        # waiting on an ID that never comes back.
        settings = self._settings(lambda: "x")
        records = llmclient.perform_tool_calls(settings, self._message("call_1", "call_2", "call_3"),
                                               on_call_start=None, on_call_done=None,
                                               maybe_refusal_text="not now")
        assert [record.tool_call_id for record in records] == ["call_1", "call_2", "call_3"]
        assert all(record.function_name == "mytool" for record in records)

    def test_no_call_is_reported_as_started(self):
        # Nothing started, so the GUI's "this tool is running" indicators must not light up.
        settings = self._settings(lambda: "x")
        started, done = [], []
        llmclient.perform_tool_calls(settings, self._message("call_1"),
                                     on_call_start=lambda *a: started.append(a),
                                     on_call_done=lambda *a: done.append(a),
                                     maybe_refusal_text="not now")
        assert started == []
        assert [(a[1], a[2]) for a in done] == [("mytool", "error")]


class TestMalformedToolCallRequests:
    """A request the backend garbled becomes an error result the model can read, not an exception.

    Each of these paths builds a report and hands it back; before, they raised `TypeError` on the way out,
    so a single garbled `tool_calls` entry took down the whole turn.
    """

    @staticmethod
    def _settings():
        return env(personas={"tool": None, "assistant": "AI", "user": "U", "system": None},
                   tool_entrypoints={"mytool": lambda: "x"})

    @pytest.mark.parametrize("tool_call, expected", [
        ({"id": "c", "function": {"name": "mytool", "arguments": "{}"}}, "missing the 'type' field"),
        ({"id": "c", "type": "banana", "function": {"name": "mytool", "arguments": "{}"}}, "Unknown request type"),
        ({"id": "c", "type": "function"}, "missing the 'function' field"),
        ({"id": "c", "type": "function", "function": {"arguments": "{}"}}, "missing the 'name' field"),
        ({"id": "c", "type": "function", "function": {"name": "nosuchtool", "arguments": "{}"}}, "Function not found"),
        ({"id": "c", "type": "function", "function": {"name": "mytool", "arguments": "{oops"}}, "failed to parse"),
    ])
    def test_a_garbled_request_becomes_an_error_result(self, tool_call, expected):
        records = llmclient.perform_tool_calls(self._settings(),
                                               {"role": "assistant", "content": "", "tool_calls": [tool_call]},
                                               on_call_start=None, on_call_done=None)
        assert len(records) == 1
        assert records[0].status == "error"
        assert expected in chatutil.content_to_text(records[0].data["content"])


class TestWebsearchWrapper:
    """brief 03 §4: websearch returns one text content-part per result, with each field normalized."""

    @staticmethod
    def _patch_search(monkeypatch, data):
        monkeypatch.setattr(llmtools, "_client_api",
                            lambda: _StubClientAPI(websearch_search=lambda *a, **k: {"data": data}))

    def test_one_text_part_per_result_with_markdown_links(self, monkeypatch):
        self._patch_search(monkeypatch, [
            {"title": "First", "link": "https://example.com/1", "text": "snippet one"},
            {"title": "Second", "link": "https://example.com/2", "text": "snippet two"},
        ])
        parts = llmclient.websearch("query")
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
        text = llmclient.websearch("q")[0]["text"]
        assert zwsp not in text  # removed from title, link, and body
        assert "Title" in text and "body" in text

    def test_result_without_title_falls_back_to_bare_url(self, monkeypatch):
        self._patch_search(monkeypatch, [{"link": "https://e.com/x", "text": "body"}])
        text = llmclient.websearch("q")[0]["text"]
        assert "<https://e.com/x>" in text

    @staticmethod
    def _patch_capture_engine(monkeypatch):
        """Patch `api.websearch_search` to record the engine it was called with; return the capture dict."""
        captured = {}
        def fake_search(query, engine, num):
            captured["engine"] = engine
            return {"data": []}
        monkeypatch.setattr(llmtools, "_client_api", lambda: _StubClientAPI(websearch_search=fake_search))
        return captured

    def test_uses_configured_engine_by_default(self, monkeypatch):
        # The LLM tool passes only a query; the engine comes from config (host choice, not model choice).
        captured = self._patch_capture_engine(monkeypatch)
        monkeypatch.setattr(llmclient.librarian_config, "websearch_engine", "google")
        llmclient.websearch("q")
        assert captured["engine"] == "google"

    def test_explicit_engine_overrides_config(self, monkeypatch):
        captured = self._patch_capture_engine(monkeypatch)
        monkeypatch.setattr(llmclient.librarian_config, "websearch_engine", "google")
        llmclient.websearch("q", engine="duckduckgo")
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
    llm_settings.tokens_per_character = 0.27
    return llm_settings


def _fake_stream(monkeypatch, payloads):
    """Make `invoke` read `payloads` as its SSE stream. Each item is a dict (JSON-encoded) or a raw
    string like '[DONE]'. Patches the HTTP POST and the SSE client."""
    datas = [p if isinstance(p, str) else json.dumps(p) for p in payloads]
    monkeypatch.setattr(llmclient.requests, "post", lambda *a, **k: _FakeResponse())
    monkeypatch.setattr(llmclient.sseclient, "SSEClient", lambda resp: _FakeSSEClient(datas))


def _capture_request(monkeypatch, payloads):
    """Like `_fake_stream`, but also keep the request body. Returns the dict it is written into."""
    sent = {}
    datas = [p if isinstance(p, str) else json.dumps(p) for p in payloads]

    def fake_post(*args, **kwargs):
        sent["json"] = kwargs["json"]
        return _FakeResponse()

    monkeypatch.setattr(llmclient.requests, "post", fake_post)
    monkeypatch.setattr(llmclient.sseclient, "SSEClient", lambda resp: _FakeSSEClient(datas))
    return sent


class TestToolNamesFiltersTheRequest:
    """`None` and an empty collection are different `tool_names` values, and both are reachable.

    `None` is the permissive one — every registered tool — and an empty collection the restrictive one.
    `invoke` keeps them apart with an `is None` test rather than a truthiness test, which is what makes an
    empty tuple fall through to the filtering branch and reduce the spec list to nothing. Testing
    truthiness there would invert the meaning of the restrictive value into the permissive one, silently
    handing the model every tool on a turn where the caller asked for none.

    `maybe_tool_names_for_turn` returns both values, so this is a live distinction: it answers `None` when
    both switches are on, and a proper subset otherwise.
    """

    _SPECS = [{"type": "function", "function": {"name": "websearch"}},
              {"type": "function", "function": {"name": "search_documents"}}]

    def _send(self, monkeypatch, invoke_settings, tool_names):
        invoke_settings.request_data = dict(invoke_settings.request_data, tools=list(self._SPECS))
        sent = _capture_request(monkeypatch, [{"choices": [{"delta": {"content": "ok"}}]}, "[DONE]"])
        llmclient.invoke(invoke_settings, _history("hi"), tools_enabled=True, tool_names=tool_names)
        return sent["json"]

    def test_none_offers_every_tool(self, monkeypatch, invoke_settings):
        data = self._send(monkeypatch, invoke_settings, None)
        assert [t["function"]["name"] for t in data["tools"]] == ["websearch", "search_documents"]

    def test_an_empty_collection_offers_none(self, monkeypatch, invoke_settings):
        data = self._send(monkeypatch, invoke_settings, ())
        # Dropped rather than sent empty: some backends reject an empty `tools` list.
        assert "tools" not in data

    def test_a_subset_offers_exactly_that_subset(self, monkeypatch, invoke_settings):
        data = self._send(monkeypatch, invoke_settings, ("search_documents",))
        assert [t["function"]["name"] for t in data["tools"]] == ["search_documents"]

    def test_tools_disabled_wins_over_any_tool_names(self, monkeypatch, invoke_settings):
        invoke_settings.request_data = dict(invoke_settings.request_data, tools=list(self._SPECS))
        sent = _capture_request(monkeypatch, [{"choices": [{"delta": {"content": "ok"}}]}, "[DONE]"])
        llmclient.invoke(invoke_settings, _history("hi"), tools_enabled=False, tool_names=None)
        assert "tools" not in sent["json"]


class TestThinkingPreferenceReachesTheWire:
    """Asking a reasoning model not to reason, and the asymmetry that "on" is silence.

    There is no value meaning "think": every accepted `reasoning_effort` other than `"none"` was measured
    indistinguishable from omitting the field, so the on-case sends nothing and the model's own default is
    what carries it. That makes the two branches easy to conflate — a helper that returned `{}` for both
    would look right in every on-case test — so each test here is paired against its opposite.
    """

    def test_thinking_on_sends_nothing(self):
        assert llmclient.thinking_request_fields(thinking_enabled=True) == {}

    def test_thinking_off_asks_for_no_reasoning_effort(self):
        assert llmclient.thinking_request_fields(thinking_enabled=False) == {"reasoning_effort": "none"}

    def test_the_field_is_absent_by_default(self, monkeypatch, invoke_settings):
        sent = _capture_request(monkeypatch, [{"choices": [{"delta": {"content": "ok"}}]}, "[DONE]"])
        llmclient.invoke(invoke_settings, _history("hi"), tools_enabled=False)
        assert "reasoning_effort" not in sent["json"]

    def test_the_field_is_present_when_thinking_is_off(self, monkeypatch, invoke_settings):
        sent = _capture_request(monkeypatch, [{"choices": [{"delta": {"content": "ok"}}]}, "[DONE]"])
        llmclient.invoke(invoke_settings, _history("hi"), tools_enabled=False, thinking_enabled=False)
        assert sent["json"]["reasoning_effort"] == "none"

    def test_it_does_not_persist_into_the_shared_settings(self, monkeypatch, invoke_settings):
        """A per-call parameter that leaked into `settings.request_data` would be a session-wide setting.

        `settings` is shared with every other consumer of the same backend — `perform_throwaway_task` and
        `agent.turn` route through this same `invoke` — so a leak here would silently apply one chat turn's
        preference to unrelated calls. Nothing would report it: a keyword extraction that has started
        reasoning first still returns the right keywords, just slowly, and "the backend feels sluggish
        today" is where it would surface.
        """
        sent = _capture_request(monkeypatch, [{"choices": [{"delta": {"content": "ok"}}]}, "[DONE]"])
        llmclient.invoke(invoke_settings, _history("hi"), tools_enabled=False, thinking_enabled=False)
        assert sent["json"]["reasoning_effort"] == "none"  # it did reach the wire on the call that asked
        assert "reasoning_effort" not in invoke_settings.request_data

        llmclient.invoke(invoke_settings, _history("hi"), tools_enabled=False)
        assert "reasoning_effort" not in sent["json"]  # ...and the next call is unaffected


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

    def test_local_tokenizer_fallback_counts_reasoning_too(self, monkeypatch, invoke_settings):
        # No usage (an interrupt closed the stream early), but a local tokenizer is configured, so the count
        # comes from encoding the generated text. Reasoning is generated text: `dt` covers the time spent on
        # it, so a count that skips it makes the speed readout understate a thinking model's turn by however
        # much of it was thinking.
        invoke_settings.tokenizer = _FakeTokenizer()  # one 'token' per character
        _fake_stream(monkeypatch, [
            {"choices": [{"delta": {"role": "assistant", "content": None}}]},
            {"choices": [{"delta": {"reasoning_content": "hmm"}}]},      # 3
            {"choices": [{"delta": {"reasoning_content": " well"}}]},    # 5
            {"choices": [{"delta": {"content": "yes"}}]},                # 3
        ])
        out = llmclient.invoke(invoke_settings, _history("hi"), tools_enabled=False)
        assert out.usage is None
        assert out.data["reasoning_content"], ("this fixture produced no reasoning, so it cannot tell "
                                               "counting both channels from counting content alone")
        assert out.n_tokens == 11  # "hmm" + " well" + "yes"


# ---------------------------------------------------------------------------
# Where an invocation's wall time went
# ---------------------------------------------------------------------------

class TestThinkingTokenCount:
    _COMMON = dict(n_tokens=100, n_chunks=100, n_chunks_at_first_content=80, tokenizer=None, usage=None)

    def test_no_reasoning_means_nothing_to_report(self):
        assert llmclient.thinking_token_count(**dict(self._COMMON, reasoning_content="")) == (None, False)

    def test_the_backends_own_split_wins(self):
        usage = {"completion_tokens": 100, "completion_tokens_details": {"reasoning_tokens": 77}}
        result = llmclient.thinking_token_count(**dict(self._COMMON, reasoning_content="mm", usage=usage,
                                                       tokenizer=_FakeTokenizer()))
        assert result == (77, True)  # ...over the tokenizer, which would have said 2

    def test_the_local_tokenizer_is_next(self):
        result = llmclient.thinking_token_count(**dict(self._COMMON, reasoning_content="0123456789",
                                                       tokenizer=_FakeTokenizer()))
        assert result == (10, True)  # one fake token per character

    def test_apportioned_by_chunk_when_neither_is_available(self):
        # 80 of 100 text-bearing deltas had arrived when the answer began, so 80 of the 100 tokens.
        assert llmclient.thinking_token_count(**dict(self._COMMON, reasoning_content="mm")) == (80, False)

    def test_an_answer_that_never_began_was_all_thinking(self):
        # A round that thought and then asked for a tool. Still inexact: the tool call's own tokens are in
        # `n_tokens` and not in the trace.
        result = llmclient.thinking_token_count(**dict(self._COMMON, reasoning_content="mm",
                                                       n_chunks_at_first_content=None))
        assert result == (100, False)


class TestPhaseReport:
    def test_no_generated_text_reports_nothing(self):
        assert llmclient.phase_report(dt=2.0, t0=1000.0, t_first_token=None, t_first_content=None,
                                      maybe_thinking_tokens=None, thinking_tokens_exact=False) is None

    def test_prefill_only_when_the_model_did_not_think(self):
        phases = llmclient.phase_report(dt=10.0, t0=1000.0, t_first_token=1002.0, t_first_content=1002.0,
                                        maybe_thinking_tokens=None, thinking_tokens_exact=False)
        assert phases == {"prefill": {"dt": 2.0}}

    def test_the_phases_never_sum_past_the_whole(self):
        phases = llmclient.phase_report(dt=10.0, t0=1000.0, t_first_token=1002.0, t_first_content=1007.0,
                                        maybe_thinking_tokens=300, thinking_tokens_exact=True)
        assert phases["prefill"]["dt"] == pytest.approx(2.0)
        assert phases["thinking"] == {"dt": pytest.approx(5.0), "n_tokens": 300, "tokens_exact": True}
        # The answer is the remainder, and there being one is what says the phases did not eat the turn.
        assert phases["prefill"]["dt"] + phases["thinking"]["dt"] < 10.0

    def test_a_sample_taken_after_the_timer_stopped_is_pulled_back_in(self):
        # The parser flushes its buffer at stream end, so the first content event can be timestamped after
        # `timer` has already recorded `dt`. Without the clamp the phases would sum past the whole.
        phases = llmclient.phase_report(dt=10.0, t0=1000.0, t_first_token=1002.0, t_first_content=1010.5,
                                        maybe_thinking_tokens=300, thinking_tokens_exact=True)
        assert phases["prefill"]["dt"] + phases["thinking"]["dt"] == pytest.approx(10.0)

    def test_thinking_that_never_ended_runs_to_the_end_of_the_call(self):
        # No content event at all: the model thought, then asked for a tool.
        phases = llmclient.phase_report(dt=10.0, t0=1000.0, t_first_token=1002.0, t_first_content=None,
                                        maybe_thinking_tokens=300, thinking_tokens_exact=False)
        assert phases["thinking"]["dt"] == pytest.approx(8.0)


class TestInvokeReportsPhases:
    def test_a_thinking_reply_splits_into_prefill_thinking_and_answer(self, monkeypatch, invoke_settings):
        invoke_settings.tokenizer = _FakeTokenizer()  # one 'token' per character, so the split is assertable
        _fake_stream(monkeypatch, [
            {"choices": [{"delta": {"role": "assistant", "content": None}}]},
            {"choices": [{"delta": {"reasoning_content": "let me see"}}]},   # 10 chars
            {"choices": [{"delta": {"content": "forty-two"}}]},
        ])
        out = llmclient.invoke(invoke_settings, _history("hi"), tools_enabled=False)
        assert out.phases["thinking"]["n_tokens"] == 10
        assert out.phases["thinking"]["tokens_exact"] is True
        assert out.phases["prefill"]["dt"] + out.phases["thinking"]["dt"] <= out.dt

    def test_a_reply_with_no_reasoning_reports_no_thinking_phase(self, monkeypatch, invoke_settings):
        _fake_stream(monkeypatch, [
            {"choices": [{"delta": {"role": "assistant", "content": None}}]},
            {"choices": [{"delta": {"content": "forty-two"}}]},
        ])
        out = llmclient.invoke(invoke_settings, _history("hi"), tools_enabled=False)
        assert "thinking" not in out.phases
        assert "prefill" in out.phases

    def test_a_round_that_only_asks_for_a_tool_reports_nothing(self, monkeypatch, invoke_settings):
        # The tool-call deltas are not text: nothing was generated on either channel, so there are no
        # phases to describe — and calling the whole call "prompt processing" would be a wrong answer
        # rather than a missing one.
        _fake_stream(monkeypatch, [
            {"choices": [{"delta": {"role": "assistant", "content": None,
                                    "tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                                    "function": {"name": "get_current_time",
                                                                 "arguments": "{}"}}]}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        ])
        out = llmclient.invoke(invoke_settings, _history("time?"), tools_enabled=True)
        assert out.data["tool_calls"]  # the fixture did produce a call, so this is not a vacuous pass
        assert out.phases is None


class TestPartialMessages:
    """The reply so far, offered to a caller that stores it as it arrives.

    Its purpose is that something reading the store *during* a reply sees the words that have arrived, so
    what matters is that a partial says exactly what the finished message would say at that point. A
    partial that lags is worse than none: a consumer that renders it and then resumes streaming appends
    what arrives next onto stale text, and the words in between are lost from what it shows.
    """

    @staticmethod
    def _partials(monkeypatch, invoke_settings, payloads):
        seen = []
        _fake_stream(monkeypatch, payloads)
        out = llmclient.invoke(invoke_settings, _history("hi"), tools_enabled=False,
                               on_partial_message=seen.append)
        return seen, out

    def test_a_partial_arrives_for_every_chunk(self, monkeypatch, invoke_settings):
        """Not per paragraph, which would leave the paragraph being written out of the store.

        The cost of that is a *wrong* message rather than a short one. A consumer that renders the store
        and then resumes streaming draws the text up to the last boundary and appends what arrives from
        there on, welding the gap shut — so the words written in between vanish from a reply that reads as
        though they were never there, and stay vanished until something re-renders from the finished
        message. With a model that writes long paragraphs, that span is most of the answer.
        """
        seen, unused_out = self._partials(monkeypatch, invoke_settings, [
            {"choices": [{"delta": {"content": "one "}}]},
            {"choices": [{"delta": {"content": "two "}}]},
            {"choices": [{"delta": {"content": "three\n"}}]},
            {"choices": [{"delta": {"content": "four "}}]},   # mid-paragraph, and must still be offered
        ])
        assert len(seen) == 4, f"expected one partial per chunk, got {len(seen)} for 4 chunks"
        assert chatutil.content_to_text(seen[0]["content"]) == "one "
        assert chatutil.content_to_text(seen[-1]["content"]) == "one two three\nfour "

    def test_a_partial_says_what_the_finished_message_would_say(self, monkeypatch, invoke_settings):
        """The two are assembled by the same code, so the last partial is a prefix of the final message."""
        seen, out = self._partials(monkeypatch, invoke_settings, [
            {"choices": [{"delta": {"content": "done\n"}}]},
        ])
        assert seen[-1]["role"] == out.data["role"]
        assert chatutil.content_to_text(seen[-1]["content"]) == chatutil.content_to_text(out.data["content"])

    def test_a_retcon_produces_a_partial_of_its_own(self, monkeypatch, invoke_settings):
        """It changes what the text so far *is*, so the store must hear about it as its own event.

        The reason it needs saying separately, now that every chunk fires one: a `reasoning_retcon`
        deliberately carries no `text`, so anything keyed on text having arrived would skip it — and the
        store would go on calling the reasoning an answer until the next chunk happened along, which is
        precisely what the retcon exists to correct.
        """
        seen, unused_out = self._partials(monkeypatch, invoke_settings, [
            {"choices": [{"delta": {"content": "thinking hard"}}]},
            {"choices": [{"delta": {"content": "</think>"}}]},   # carries no text of its own
        ])
        assert seen[-1]["reasoning_content"] == "thinking hard"
        assert chatutil.content_to_text(seen[-1]["content"]) == ""
        # The control: the partial *before* the retcon read the same text as an answer. Without it, a
        # fixture whose reasoning was already in the right place would pass without the retcon firing.
        assert chatutil.content_to_text(seen[0]["content"]) == "thinking hard"
        assert seen[0].get("reasoning_content") is None  # absent, not `None`: no reasoning had been seen yet

    def test_no_callback_means_no_partials_are_built(self, monkeypatch, invoke_settings):
        """The default costs nothing: `build_message` is not called at all for a caller that did not ask."""
        _fake_stream(monkeypatch, [{"choices": [{"delta": {"content": "a\nb\n"}}]}, "[DONE]"])
        out = llmclient.invoke(invoke_settings, _history("hi"), tools_enabled=False)
        assert chatutil.content_to_text(out.data["content"]) == "a\nb\n"  # and the reply is unaffected


class TestOrphanThinkClose:
    """Reasoning that arrives with no opening tag, because the chat template supplied it.

    Every Qwen in the local archive does this (`investigations/chat-template-think-prefill/`), so on any
    backend that does not split reasoning into its own channel, the close tag is the only marker the stream
    ever carries.
    """
    def test_the_close_tag_reclassifies_everything_before_it(self, monkeypatch, invoke_settings):
        _fake_stream(monkeypatch, [
            {"choices": [{"delta": {"role": "assistant", "content": None}}]},
            {"choices": [{"delta": {"content": "Let me work it out."}}]},
            {"choices": [{"delta": {"content": "</think>"}}]},
            {"choices": [{"delta": {"content": "Forty-two."}}]},
        ])
        out = llmclient.invoke(invoke_settings, _history("what is it?"), tools_enabled=False)
        assert out.data["reasoning_content"] == "Let me work it out."
        assert chatutil.content_to_text(out.data["content"]) == "Forty-two."

    def test_a_close_split_across_deltas_is_still_recognized(self, monkeypatch, invoke_settings):
        _fake_stream(monkeypatch, [
            {"choices": [{"delta": {"content": "mulling"}}]},
            {"choices": [{"delta": {"content": "</thi"}}]},
            {"choices": [{"delta": {"content": "nk>done"}}]},
        ])
        out = llmclient.invoke(invoke_settings, _history("hi"), tools_enabled=False)
        assert out.data["reasoning_content"] == "mulling"
        assert chatutil.content_to_text(out.data["content"]) == "done"

    def test_a_properly_opened_block_is_not_reclassified(self, monkeypatch, invoke_settings):
        # The control: with both tags present the parser was never in doubt, and the answer *before* the
        # block must stay in the answer. Without this the retcon would swallow it.
        _fake_stream(monkeypatch, [
            {"choices": [{"delta": {"content": "Sure. "}}]},
            {"choices": [{"delta": {"content": "<think>hmm</think>"}}]},
            {"choices": [{"delta": {"content": "Forty-two."}}]},
        ])
        out = llmclient.invoke(invoke_settings, _history("hi"), tools_enabled=False)
        assert out.data["reasoning_content"] == "hmm"
        assert chatutil.content_to_text(out.data["content"]) == "Sure. Forty-two."

    def test_a_stray_close_after_native_reasoning_is_not_a_signal(self, monkeypatch, invoke_settings):
        # The thinking already came on its own channel, so nothing about the answer needs inferring and a
        # close tag in it says nothing about what preceded it.
        _fake_stream(monkeypatch, [
            {"choices": [{"delta": {"reasoning_content": "thought hard"}}]},
            {"choices": [{"delta": {"content": "a </think> tag looks like this"}}]},
        ])
        out = llmclient.invoke(invoke_settings, _history("hi"), tools_enabled=False)
        assert out.data["reasoning_content"] == "thought hard"  # not joined by the answer
        assert "tag looks like this" in chatutil.content_to_text(out.data["content"])

    def test_it_fires_at_most_once(self, monkeypatch, invoke_settings):
        _fake_stream(monkeypatch, [
            {"choices": [{"delta": {"content": "first"}}]},
            {"choices": [{"delta": {"content": "</think>"}}]},
            {"choices": [{"delta": {"content": "answer"}}]},
            {"choices": [{"delta": {"content": "</think>"}}]},
            {"choices": [{"delta": {"content": "more"}}]},
        ])
        out = llmclient.invoke(invoke_settings, _history("hi"), tools_enabled=False)
        assert out.data["reasoning_content"] == "first"  # the second close did not move "answer" as well
        assert "answer" in chatutil.content_to_text(out.data["content"])

    def test_the_correction_reaches_the_consumer_that_showed_the_text(self, monkeypatch, invoke_settings):
        """A renderer has already drawn the reasoning as the answer, so it has to be told.

        The event is what lets it move what it drew. Position matters as much as presence: it must arrive
        *after* the text it reclassifies and *before* the real answer, since that is the only ordering a
        consumer can act on without buffering.
        """
        _fake_stream(monkeypatch, [
            {"choices": [{"delta": {"content": "mulling"}}]},
            {"choices": [{"delta": {"content": "</think>"}}]},
            {"choices": [{"delta": {"content": "Forty-two."}}]},
        ])
        seen = []
        llmclient.invoke(invoke_settings, _history("hi"), tools_enabled=False,
                         on_progress=lambda event: (seen.append((event["type"], event.get("text"))),
                                                    llmclient.action_ack)[1])
        assert seen == [("content", "mulling"),
                        ("reasoning_retcon", None),
                        ("content", "Forty-two.")]

    def test_no_correction_is_sent_when_there_is_nothing_to_correct(self, monkeypatch, invoke_settings):
        # The control for the test above: an ordinary reply must not make a consumer move anything. Without
        # this, a parser that emitted the event unconditionally would satisfy the assertion above.
        _fake_stream(monkeypatch, [
            {"choices": [{"delta": {"reasoning_content": "thought hard"}}]},
            {"choices": [{"delta": {"content": "Forty-two."}}]},
        ])
        seen = []
        llmclient.invoke(invoke_settings, _history("hi"), tools_enabled=False,
                         on_progress=lambda event: (seen.append(event["type"]), llmclient.action_ack)[1])
        assert seen == ["reasoning", "content"]

    def test_the_thinking_phase_is_reported_for_a_reclassified_turn(self, monkeypatch, invoke_settings):
        # The point of the whole exercise: the stored numbers come out right on a backend that leaves the
        # tag inference to us.
        invoke_settings.tokenizer = _FakeTokenizer()  # one 'token' per character
        _fake_stream(monkeypatch, [
            {"choices": [{"delta": {"content": "0123456789"}}]},
            {"choices": [{"delta": {"content": "</think>"}}]},
            {"choices": [{"delta": {"content": "ok"}}]},
        ])
        out = llmclient.invoke(invoke_settings, _history("hi"), tools_enabled=False)
        assert out.phases["thinking"]["n_tokens"] == 10
        # The answer began at the reclassification, not at the first delta — which is what the reset of the
        # phase samples buys. Had it not been reset, thinking would have measured as nothing at all.
        assert out.phases["thinking"]["dt"] > 0.0


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

    def test_whether_a_model_is_loaded_is_a_tristate(self, monkeypatch):
        # The three answers a frontend has to tell apart. `None` is "cannot tell" and must not be shown as a
        # fault: ooba and generic backends report nothing to go on, and calling that "no model loaded" would
        # put a permanent warning in front of every user of either.
        monkeypatch.setattr(llmclient.librarian_config, "llm_model", None)

        monkeypatch.setattr(llmclient.requests, "get", _route_get({
            "/api/v0/models": {"data": [{"id": "a", "state": "loaded"}]}}))
        assert llmclient._resolve_model_info("http://x", "lmstudio").loaded is True

        monkeypatch.setattr(llmclient.requests, "get", _route_get({
            "/api/v0/models": {"data": [{"id": "a", "state": "not-loaded"}]}}))
        assert llmclient._resolve_model_info("http://x", "lmstudio").loaded is False

        monkeypatch.setattr(llmclient.requests, "get", _route_get({
            "/v1/internal/model/info": {"model_name": "Qwen3-4B.gguf"}}))
        assert llmclient._resolve_model_info("http://x", "oobabooga").loaded is True

        # A generic backend's model list says what it *has*, which is not what is resident -- LM Studio's
        # own list would read as "loaded" by that test with nothing running at all.
        monkeypatch.setattr(llmclient.requests, "get", _route_get({
            "/v1/models": {"data": [{"id": "the-only-model"}]}}))
        assert llmclient._resolve_model_info("http://x", "generic").loaded is None

    def test_oobabooga_says_nothing_is_loaded_with_the_string_None(self, monkeypatch):
        # ooba's own `list_models_openai_format` tests `model_name != 'None'` before deciding it has a model
        # to list, so the sentinel is a string rather than a null. Taken from its source; there is no ooba
        # instance here to confirm it against, which is also true of the whole ooba path.
        monkeypatch.setattr(llmclient.requests, "get", _route_get({
            "/v1/internal/model/info": {"model_name": "None"}}))
        info = llmclient._resolve_model_info("http://x", "oobabooga")
        assert info.loaded is False
        assert info.label == llmclient.NO_MODEL_INFO  # ...and it is not reported as a model called "None"
        assert info.model_id is None

    def test_a_missing_capability_field_is_cannot_tell_not_cannot_see(self, monkeypatch):
        # `is_vlm=False` hard-refuses image attachment, so it must mean the backend said so. Every record
        # LM Studio returns carries `type`; a record without one has not answered the question.
        monkeypatch.setattr(llmclient.requests, "get", _route_get({
            "/api/v0/models": {"data": [{"id": "a", "state": "loaded"}]}}))
        assert llmclient._resolve_model_info("http://x", "lmstudio").is_vlm is None

        monkeypatch.setattr(llmclient.requests, "get", _route_get({
            "/api/v0/models": {"data": [{"id": "a", "state": "loaded", "type": "llm"}]}}))
        assert llmclient._resolve_model_info("http://x", "lmstudio").is_vlm is False

    def test_naming_a_model_does_not_make_it_resident(self, monkeypatch):
        # LM Studio's JIT loads on demand, so a configured model name is a request the backend will *try* to
        # honor - and it fails outright when the model does not fit in the free VRAM. So the name settles
        # what a request would ask for, not whether the backend can answer now.
        monkeypatch.setattr(llmclient.requests, "get", _route_get({
            "/api/v0/models": {"data": [{"id": "a", "state": "not-loaded"}]}}))
        monkeypatch.setattr(llmclient.librarian_config, "llm_model", "qwen3.6-35b-a3b")
        info = llmclient._resolve_model_info("http://x", "lmstudio")
        assert info.label == "qwen3.6-35b-a3b"
        assert info.loaded is False


class TestBackendStatus:
    """The fold from two backend facts to the one question a frontend asks: can this thing answer?"""

    def test_the_three_states_a_frontend_has_to_tell_apart(self):
        # Distinct rather than collapsed into "not working", because the user meets all three at the same
        # moment -- the first message of a session -- and what they should do about each differs.
        unreachable = env(backend_is_reachable=False, model_is_loaded=False)
        assert llmclient.backend_status(unreachable) is llmclient.backend_unreachable

        empty = env(backend_is_reachable=True, model_is_loaded=False)
        assert llmclient.backend_status(empty) is llmclient.backend_has_no_model

        ready = env(backend_is_reachable=True, model_is_loaded=True)
        assert llmclient.backend_status(ready) is llmclient.backend_ready

    def test_each_state_is_described_distinctly_and_names_the_backend(self):
        # Four frontends print these -- two batch tools, a console client and a GUI tooltip -- so the
        # wording is shared rather than copied. What each caller relies on: the three differ, the address is
        # in the headline (the user may have several backends, or a typo in one), and only a state the user
        # can act on carries advice.
        described = {status: llmclient.describe_backend_status(status, "http://x:1234")
                     for status in (llmclient.backend_unreachable,
                                    llmclient.backend_has_no_model,
                                    llmclient.backend_ready)}
        headlines = [headline for headline, _advice in described.values()]
        assert len(set(headlines)) == 3
        assert all("http://x:1234" in headline for headline in headlines)
        assert described[llmclient.backend_ready][1] == ""
        assert all(advice for status, (_headline, advice) in described.items() if status is not llmclient.backend_ready)

    def test_cannot_tell_is_not_a_fault_to_report(self):
        # ooba and generic backends have nothing to report residency with, so `model_is_loaded is None`.
        # Reading that as "no model loaded" would park a permanent warning in front of every user of either.
        cannot_tell = env(backend_is_reachable=True, model_is_loaded=None)
        assert llmclient.backend_status(cannot_tell) is llmclient.backend_ready


class TestConnectAndReconnect:
    """`connect` is the interactive frontends' `setup`: it reports a dead backend rather than raising."""

    def _good_settings(self):
        return llmclient.configure(model_info=env(label="a-model", model_id="a-model", context_length=4096,
                                                  is_vlm=True, loaded=True),
                                   backend_flavor="lmstudio",
                                   backend_url="http://x",
                                   quiet=True)

    def test_an_unreachable_backend_yields_usable_settings(self, monkeypatch):
        # The whole point: a window opens anyway, so what comes back has to be a real settings object -- the
        # chat view, the cleanup dialog and the settings all read it -- and merely say that nothing answered.
        def _refuse(backend_url, quiet=False):
            raise llmclient.requests.exceptions.ConnectionError("nothing listening")
        monkeypatch.setattr(llmclient, "setup", _refuse)

        settings = llmclient.connect("http://x", quiet=True)
        assert llmclient.backend_status(settings) is llmclient.backend_unreachable
        assert settings.backend_url == "http://x"
        assert settings.context_length == 64 * 1024  # the default, standing in for one nobody could ask about
        assert settings.model == llmclient.NO_MODEL_INFO

    def test_reconnect_updates_the_settings_object_the_caller_is_holding(self, monkeypatch):
        # Every consumer -- the chat controller, the app state, a script -- already holds this object, so a
        # replacement returned to one caller would leave all the others on the stale one.
        def _refuse(backend_url, quiet=False):
            raise llmclient.requests.exceptions.ConnectionError("nothing listening")
        monkeypatch.setattr(llmclient, "setup", _refuse)
        settings = llmclient.connect("http://x", quiet=True)

        good = self._good_settings()
        monkeypatch.setattr(llmclient, "setup", lambda backend_url, quiet=False: good)

        status = llmclient.reconnect(settings, quiet=True)
        assert status is llmclient.backend_ready
        assert settings.model == "a-model"
        assert settings.context_length == 4096
        assert settings.model_is_vlm is True
        assert settings.backend_is_reachable is True

    def test_a_backend_that_answers_with_nothing_loaded_is_its_own_state(self, monkeypatch):
        empty = llmclient.configure(model_info=env(label=llmclient.NO_MODEL_INFO, model_id=None,
                                                   context_length=None, is_vlm=None, loaded=False),
                                    backend_flavor="lmstudio",
                                    backend_url="http://x",
                                    quiet=True)
        monkeypatch.setattr(llmclient, "setup", lambda backend_url, quiet=False: empty)
        settings = llmclient.connect("http://x", quiet=True)
        assert llmclient.backend_status(settings) is llmclient.backend_has_no_model


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
        invoke_settings.tokens_per_character = 0.25
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
        assert invoke_settings.tokens_per_character == pytest.approx(10 / sent["chars"])

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


class TestConfigureMatchesSetup:
    """`setup` is `configure` plus the two queries that discover its arguments — and nothing else.

    The point of the split is that a caller holding the backend's facts can build the *real* settings object
    without a backend. That is only worth anything if the object is genuinely the same one, so this pins the
    equality rather than the plumbing: given the same `model_info` and flavor, `configure` must reproduce
    `setup` field for field.
    """

    def _patch_generic_backend(self, monkeypatch):
        monkeypatch.setattr(llmclient.requests, "get", _route_get({
            "/v1/models": {"data": [{"id": "test-model"}]},
        }))
        monkeypatch.setattr(llmclient.librarian_config, "llm_backend_flavor", None)
        monkeypatch.setattr(llmclient.librarian_config, "llm_tokenizer_path", None)

    def test_configure_reproduces_setup_given_the_same_facts(self, monkeypatch):
        self._patch_generic_backend(monkeypatch)
        from_setup = llmclient.setup("http://test-backend", quiet=True)

        flavor = llmclient.detect_backend_flavor("http://test-backend")
        model_info = llmclient._resolve_model_info("http://test-backend", flavor)

        # From here on, any HTTP at all is a bug: `configure` must not contact the backend. Breaking
        # `requests.get` outright is what distinguishes "does not need it" from "happens not to have used it".
        def _no_http(*args, **kwargs):
            raise AssertionError("configure() contacted the backend")
        monkeypatch.setattr(llmclient.requests, "get", _no_http)
        monkeypatch.setattr(llmclient.requests, "post", _no_http)

        from_configure = llmclient.configure(model_info=model_info,
                                             backend_flavor=flavor,
                                             backend_url="http://test-backend",
                                             quiet=True)

        assert set(from_configure.keys()) == set(from_setup.keys())
        # Compare the values, not their reprs. `repr(env)` leads with the object's own address, so an
        # env-valued field (`settings.formatters`) never matches itself across two calls that way — while
        # `env.__eq__` compares contents and gets it right.
        differing = [k for k in sorted(from_setup.keys())
                     if from_setup[k] != from_configure[k]]
        assert not differing, f"configure() diverged from setup() on: {differing}"

    def test_missing_context_length_defaults_the_same_way(self, monkeypatch):
        # A backend that reports no context window is the case the 64k default exists for, and the default
        # has to live in `configure` — a caller synthesizing a `model_info` gets it too, or the split leaks.
        self._patch_generic_backend(monkeypatch)
        model_info = llmclient._resolve_model_info("http://test-backend", "generic")
        assert model_info.context_length is None, "precondition: generic backend reports no context length"

        settings = llmclient.configure(model_info=model_info,
                                       backend_flavor="generic",
                                       backend_url="http://test-backend",
                                       quiet=True)
        assert settings.context_length == 64 * 1024


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
# serialize_history_for_wire: text scrub + image-part preservation + sidecar resolution
# ---------------------------------------------------------------------------

class TestSerializeHistoryForWire:
    # The token-accounting fields are what the attachment fold sizes itself against: a 10000-token window
    # at 0.25 tokens/character is 40000 characters, of which the 25% reserve leaves 30000 for attachments.
    settings = env(personas={"user": "U", "assistant": "AI", "system": None, "tool": None},
                   context_length=10000,
                   tokens_per_character=0.25)

    def test_text_only_message_scrubbed_to_single_text_part(self):
        history = _history("hello there")
        out = llmclient.serialize_history_for_wire(self.settings, history, continue_=False)
        assert out[0]["content"] == [chatutil.text_content_part("U: hello there")]  # persona-prefixed, one text part

    def test_input_history_not_mutated(self):
        history = _history("hello")
        llmclient.serialize_history_for_wire(self.settings, history, continue_=False)
        assert history[0]["content"] == [chatutil.text_content_part("hello")]  # deep-copied; original untouched

    def test_a_fetched_page_is_ceilinged_where_an_attachment_is_not(self, tmp_path, monkeypatch):
        # The whole point of putting `source` on the part: the wire builder gets bare messages, so without
        # it these two documents are indistinguishable here and both go whole.
        from raven.librarian import chattree, textfilestore
        monkeypatch.setattr("raven.librarian.config.context_reserve_fraction", 0.25)
        monkeypatch.setattr("raven.librarian.config.docs_fetch_max_fraction_of_context", 0.10)
        ds = chattree.PersistentForest(tmp_path / "chat.json", autosave=False)
        body = ("word " * 4000).encode("utf-8")  # 20000 characters, well over the 4000-character ceiling

        fetched = textfilestore.store_file_as_sidecar(ds, body, name="page.md",
                                                      provenance_url="https://example.com/page",
                                                      provenance_source="tool_result")
        attached = textfilestore.store_file_as_sidecar(ds, body, name="paper.txt",
                                                       provenance_url="file:///tmp/paper.txt",
                                                       provenance_source="user_attachment")
        history = [{"role": "tool", "content": [chatutil.text_content_part("(excerpt)"), fetched.part]},
                   {"role": "user", "content": [chatutil.text_content_part("and read this"), attached.part]}]
        out = llmclient.serialize_history_for_wire(self.settings, history, continue_=False, datastore=ds)

        fetched_block, attached_block = out[0]["content"][0]["text"], out[1]["content"][0]["text"]
        assert "characters omitted" in fetched_block  # ceilinged: a hunch does not get the whole window
        assert "characters omitted" not in attached_block  # the user said read this
        assert len(fetched_block) < len(attached_block)

    def test_the_wire_fold_and_the_context_readout_size_an_attachment_alike(self, tmp_path, monkeypatch):
        # The invariant the `source` field exists to protect. These two walk the same attachments from
        # opposite ends -- the readout up from the head, the fold down from the root -- and a disagreement
        # shows up as a context-fill percentage that describes a request Raven did not send.
        from raven.librarian import chattree, textfilestore
        monkeypatch.setattr("raven.librarian.config.context_reserve_fraction", 0.25)
        monkeypatch.setattr("raven.librarian.config.docs_fetch_max_fraction_of_context", 0.10)
        ds = chattree.PersistentForest(tmp_path / "chat.json", autosave=False)
        stored = textfilestore.store_file_as_sidecar(ds, ("word " * 4000).encode("utf-8"), name="page.md",
                                                     provenance_url="https://example.com/page",
                                                     provenance_source="tool_result")
        message = {"role": "tool", "content": [chatutil.text_content_part("(excerpt)"), stored.part]}
        node_id = ds.create_node(payload={"message": message,
                                          "general_metadata": {"sidecars": {stored.filename: stored.sidecar_metadata}}},
                                 parent_id=None)

        # `count_branch_tokens` reaches `count_tokens`, which tiers through a local tokenizer and ooba's
        # endpoint before the ratio; neither is available here, so pin it to the estimate path.
        settings = env(**self.settings, tokenizer=None, backend_flavor="lmstudio")
        counted, _is_exact = llmclient.count_branch_tokens(settings, ds, node_id)
        wire = llmclient.serialize_history_for_wire(settings, [message], continue_=False, datastore=ds)
        wire_characters = sum(len(part["text"]) for part in wire[0]["content"] if part["type"] == "text")

        # The readout counts the same fitted attachment text the wire carries; the small residual is the
        # persona prefix and the `[Attached file: ...]` framing the fold adds, not a budget disagreement.
        assert counted == pytest.approx(wire_characters * self.settings.tokens_per_character, rel=0.05)

    def test_a_cache_relative_prompt_size_is_not_mistaken_for_the_whole_prompt(self):
        """A backend may report the tokens it *processed*, not the size of the prompt.

        The numbers are the ones measured against LM Studio on 2026-08-22, in
        `investigations/prompt-size-cache-relative/`: 56365 for the whole prompt, 8745 for the same prompt
        with its prefix already cached, against a local estimate of 81158. The estimate runs high, which is
        why the bound is loose — a tight one would reject the true figure.
        """
        assert llmclient.prompt_size_report_looks_whole(56365, 81158) is True  # the real count, estimate 44% high
        assert llmclient.prompt_size_report_looks_whole(8745, 81158) is False  # the cache-relative one
        assert llmclient.prompt_size_report_looks_whole(81158, 81158) is True  # estimate dead on
        assert llmclient.prompt_size_report_looks_whole(200000, 81158) is True  # estimate low; believe the backend
        assert llmclient.prompt_size_report_looks_whole(0, 0) is True  # nothing to compare against

    def test_the_readout_can_decline_to_extract_an_attachment_it_has_not_seen(self, tmp_path, monkeypatch):
        """`extract_attachments=False` must skip an un-extracted document rather than pay for it.

        What is being bought is not accuracy but *latency*: this count runs on every HEAD change, from a DPG
        callback, and extracting a large PDF there holds the callback thread and freezes the keyboard. So
        the assertions are about what is *not* counted, and about the figure admitting it.
        """
        from raven.librarian import chattree, textfilestore
        monkeypatch.setattr("raven.librarian.config.context_reserve_fraction", 0.25)
        monkeypatch.setattr("raven.librarian.config.docs_fetch_max_fraction_of_context", 0.10)
        # A cache shared across the test session would decide the answer here, this being memoized on the
        # content-addressed name and other tests storing documents of their own.
        monkeypatch.setattr("raven.librarian.textfilestore._extracted_text_cache", {})
        ds = chattree.PersistentForest(tmp_path / "chat.json", autosave=False)
        stored = textfilestore.store_file_as_sidecar(ds, ("word " * 4000).encode("utf-8"), name="page.md",
                                                     provenance_url="https://example.com/page",
                                                     provenance_source="tool_result")
        message = {"role": "tool", "content": [chatutil.text_content_part("(excerpt)"), stored.part]}
        node_id = ds.create_node(payload={"message": message,
                                          "general_metadata": {"sidecars": {stored.filename: stored.sidecar_metadata}}},
                                 parent_id=None)
        settings = env(**self.settings, tokenizer=None, backend_flavor="lmstudio")

        # `store_file_as_sidecar` remembers the text it was handed, so extraction has already happened for
        # this document — clearing the cache above is what makes it genuinely unseen.
        skipped, skipped_is_exact = llmclient.count_branch_tokens(settings, ds, node_id, extract_attachments=False)
        # States the contract, but note it cannot fail here: this fixture has no tokenizer, so the estimate
        # path forces `is_exact` False whatever the attachments did. The size comparison below is what
        # actually discriminates — checked by making the skip path extract anyway, which fails it.
        assert skipped_is_exact is False

        # And the default still pays for it, which is what the fetch budget needs.
        extracted, _is_exact = llmclient.count_branch_tokens(settings, ds, node_id)
        assert extracted > skipped

        # Once extracted, the cheap path counts it too: the saving is per document per process, not forever.
        again, _is_exact = llmclient.count_branch_tokens(settings, ds, node_id, extract_attachments=False)
        assert again == extracted

    def test_image_part_preserved_and_sidecar_resolved(self, tmp_path):
        import base64
        from raven.librarian import chattree
        ds = chattree.PersistentForest(tmp_path / "chat.json", autosave=False)
        raw = b"\x89PNG\r\n\x1a\n" + b"fake-png-bytes"
        filename = ds.store_sidecar(raw, "png")

        history = [{"role": "user", "content": [chatutil.text_content_part("what is this?"),
                                                chatutil.image_content_part(f"sidecar:{filename}")]}]
        out = llmclient.serialize_history_for_wire(self.settings, history, continue_=False, datastore=ds)

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
        out = llmclient.serialize_history_for_wire(self.settings, history, continue_=False, datastore=None)
        assert out[0]["content"][1]["image_url"]["url"] == "sidecar:abc.png"  # unresolved, but preserved

    def test_continue_leaves_last_message_untouched(self):
        history = [{"role": "user", "content": [chatutil.text_content_part("q")]},
                   {"role": "assistant", "content": [chatutil.text_content_part("partial ans")]}]
        out = llmclient.serialize_history_for_wire(self.settings, history, continue_=True)
        assert out[0]["content"] == [chatutil.text_content_part("U: q")]  # scrubbed
        assert out[1]["content"] == [chatutil.text_content_part("partial ans")]  # last message untouched

    # --- attached documents (text_file parts) fold into the message text (P2-A) ---

    def _datastore_with_file(self, tmp_path, body, name="spec.txt"):
        from raven.librarian import chattree, textfilestore
        datastore = chattree.PersistentForest(tmp_path / "chat.json", autosave=False,
                                              sidecar_extractor=textfilestore.sidecar_refs_in_payload)
        stored = textfilestore.store_file_as_sidecar(datastore, body, name=name,
                                                     provenance_url=f"file:///{name}",
                                                     provenance_source="user_attachment")
        return datastore, stored.part

    def test_text_file_folded_into_message_text(self, tmp_path):
        datastore, file_part = self._datastore_with_file(tmp_path, b"the attached document body")
        history = [{"role": "user",
                    "content": [chatutil.text_content_part("What does the spec say?"), file_part]}]
        wire = llmclient.serialize_history_for_wire(self.settings, history,
                                                    continue_=False, datastore=datastore)
        assert len(wire) == 1
        parts = wire[0]["content"]
        assert all(p["type"] == "text" for p in parts)  # no text_file part survives onto the wire
        text = "".join(p["text"] for p in parts)
        assert "What does the spec say?" in text
        assert "[Attached file: spec.txt]" in text
        assert "the attached document body" in text

    def test_text_file_not_folded_without_datastore(self, tmp_path):
        datastore, file_part = self._datastore_with_file(tmp_path, b"secret body")
        history = [{"role": "user",
                    "content": [chatutil.text_content_part("hello"), file_part]}]
        # No datastore -> the sidecar can't be resolved, so the document is not folded (the throwaway/prefill
        # callers that pass no datastore carry no attachments in practice).
        wire = llmclient.serialize_history_for_wire(self.settings, history,
                                                    continue_=False, datastore=None)
        text = "".join(p["text"] for p in wire[0]["content"])
        assert "hello" in text
        assert "secret body" not in text

    def test_an_oversized_attachment_is_truncated_rather_than_overflowing(self, tmp_path):
        # Before the budget existed this folded wholesale and blew the window.
        datastore, file_part = self._datastore_with_file(tmp_path, b"A" * 200000, name="huge.txt")
        history = [{"role": "user",
                    "content": [chatutil.text_content_part("summarize this"), file_part]}]
        wire = llmclient.serialize_history_for_wire(self.settings, history,
                                                    continue_=False, datastore=datastore)
        text = "".join(p["text"] for p in wire[0]["content"])
        assert len(text) < 40000  # fits the window it is going into
        assert "characters omitted" in text  # and says where the missing part was
        assert "[Attached file: huge.txt]" in text

    def test_an_attachment_with_no_room_left_is_named_not_dropped(self, tmp_path):
        # A silently vanished attachment leaves the model reading a message that refers to a document it
        # cannot see, which it will resolve by guessing.
        datastore, file_part = self._datastore_with_file(tmp_path, b"B" * 1000, name="late.txt")
        history = [{"role": "user", "content": [chatutil.text_content_part("C" * 40000)]},
                   {"role": "user",
                    "content": [chatutil.text_content_part("and this too"), file_part]}]
        wire = llmclient.serialize_history_for_wire(self.settings, history,
                                                    continue_=False, datastore=datastore)
        text = "".join(p["text"] for p in wire[1]["content"])
        assert "late.txt" in text
        assert "no room left" in text
        assert "BBB" not in text


# ---------------------------------------------------------------------------
# Strict-chat-template shape warnings
# ---------------------------------------------------------------------------

def _roles(*roles):
    """A history carrying only the roles — all these checks look at is the role sequence."""
    return [{"role": role, "content": [chatutil.text_content_part(f"({role})")]} for role in roles]


class TestStrictTemplateWarnings:
    """`_describe_strict_template_violations` makes a backend template rejection legible.

    A strict template (Qwen3.5's) answers a bad message shape with a 400 that names its own parser,
    not the conversation — so this description is what points at the real cause. These tests pin
    that the two rejected shapes are described and good ones yield nothing; a describer that cried
    wolf would be worse than none, since the caller logs what it returns on a refusal and a reader
    who sees it beside every failure learns to skip it.
    """

    def test_missing_user_message_is_described(self):
        violations = llmclient._describe_strict_template_violations(_roles("system", "assistant"))
        assert len(violations) == 1
        assert "no user message" in violations[0]
        assert "system, assistant" in violations[0]  # the actual sequence, for diagnosis

    def test_late_system_message_is_described(self):
        # The shape Raven itself sent before the injects moved to the user role.
        violations = llmclient._describe_strict_template_violations(_roles("system", "assistant", "user", "system", "system"))
        assert len(violations) == 1
        assert "system message that is not the first message" in violations[0]
        assert "system, assistant, user, system, system" in violations[0]

    def test_consecutive_leading_system_messages_are_described(self):
        # The RAG-inject shape: matches inserted at index 1, each as its own system message. Every
        # system message is ahead of the conversation, and Qwen3.5's template rejects it anyway —
        # its guard is `not loop.first`, so only the message at index 0 may carry the system role.
        violations = llmclient._describe_strict_template_violations(_roles("system", "system", "system", "assistant", "user"))
        assert any("system message that is not the first message" in violation for violation in violations)

    def test_both_violations_described_separately(self):
        violations = llmclient._describe_strict_template_violations(_roles("system", "assistant", "system"))
        assert len(violations) == 2
        assert any("no user message" in violation for violation in violations)
        assert any("system message that is not the first message" in violation for violation in violations)

    @pytest.mark.parametrize("roles", [("system", "user"),
                                       ("system", "assistant", "user"),
                                       ("system", "user", "assistant", "user"),
                                       ("system", "user", "assistant", "tool", "assistant", "user"),  # tool results mid-conversation
                                       ("user",)])  # no system prompt at all
    def test_accepted_shapes_describe_nothing(self, roles):
        assert llmclient._describe_strict_template_violations(_roles(*roles)) == []

    def test_empty_history_describes_nothing(self):
        # A degenerate history is someone else's error to report; don't add noise to it.
        assert llmclient._describe_strict_template_violations([]) == []

    def test_nothing_is_logged_when_describing(self, caplog):
        # The point of the change: a bad shape is *held*, not announced. Raven's own idle context
        # prefill sends `[system, greeting]` on every new chat, and that must stay silent.
        with caplog.at_level(logging.DEBUG):
            llmclient._describe_strict_template_violations(_roles("system", "assistant"))
        assert caplog.text == ""


class TestRefusalCarriesTheTemplateDiagnosis:
    """Every path that reports a refused request must attach the held description — and there are two.

    Backends disagree on how to refuse. An HTTP error status is the obvious one; LM Studio answers 200
    and puts the error in an SSE event mid-stream, which is the one a template rejection actually takes
    (verified against LM Studio serving Qwen3.5, whose template calls `raise_exception('No user query
    found in messages.')`). Wiring the diagnosis to only the first path passes every test that mocks an
    HTTP failure while never firing in the case it was written for.
    """

    def test_in_stream_error_carries_it(self, monkeypatch, invoke_settings, caplog):
        _fake_stream(monkeypatch, [{"error": {"message": "template refused the conversation"}}])
        with caplog.at_level(logging.ERROR):
            with pytest.raises(RuntimeError, match="template refused"):
                llmclient.invoke(invoke_settings, _roles("system", "assistant"))
        assert "no user message" in caplog.text

    def test_http_error_status_carries_it(self, monkeypatch, invoke_settings, caplog):
        class _Refused:
            status_code = 400
            reason = "Bad Request"
            text = "nope"
        monkeypatch.setattr(llmclient.requests, "post", lambda *a, **k: _Refused())
        with caplog.at_level(logging.ERROR):
            with pytest.raises(RuntimeError, match="HTTP 400"):
                llmclient.invoke(invoke_settings, _roles("system", "assistant"))
        assert "no user message" in caplog.text

    def test_a_good_shape_adds_nothing_to_an_unrelated_failure(self, monkeypatch, invoke_settings, caplog):
        # A refusal with nothing wrong with the shape must not acquire a spurious explanation.
        _fake_stream(monkeypatch, [{"error": {"message": "out of memory"}}])
        with caplog.at_level(logging.ERROR):
            with pytest.raises(RuntimeError, match="out of memory"):
                llmclient.invoke(invoke_settings, _roles("system", "user"))
        assert "Strict chat templates" not in caplog.text


# ---------------------------------------------------------------------------
# Budgeting text fetched into the context
# ---------------------------------------------------------------------------

class TestFetchBudget:
    """How much of a document may be pulled into a conversation that is already partly full."""

    def settings(self, context_length=10000):
        return env(context_length=context_length, tokens_per_character=0.25)

    def test_empty_conversation_gets_the_per_fetch_ceiling(self, monkeypatch):
        # The per-fetch ceiling is the limit that normally binds: one document must not crowd out the
        # conversation it is meant to inform, however much room happens to be free.
        monkeypatch.setattr("raven.librarian.config.docs_fetch_max_fraction_of_context", 0.10)
        monkeypatch.setattr("raven.librarian.config.context_reserve_fraction", 0.25)
        assert llmclient.budget_for_fetched_text(self.settings(), used_tokens=0) == 1000

    def test_a_full_conversation_refuses_rather_than_serving_a_sliver(self, monkeypatch):
        monkeypatch.setattr("raven.librarian.config.docs_fetch_max_fraction_of_context", 0.10)
        monkeypatch.setattr("raven.librarian.config.context_reserve_fraction", 0.25)
        assert llmclient.budget_for_fetched_text(self.settings(), used_tokens=8000) <= 0

    def test_the_reserve_shrinks_the_budget_before_it_refuses(self, monkeypatch):
        # Two regimes, and this is the middle one. The reserve is not slack -- it is what the model's own
        # reasoning after the fetch will consume, which the size estimate cannot see -- so it starts eating
        # into the budget well before the window is exhausted, and only then refuses outright.
        monkeypatch.setattr("raven.librarian.config.docs_fetch_max_fraction_of_context", 0.10)
        monkeypatch.setattr("raven.librarian.config.context_reserve_fraction", 0.25)
        budget = llmclient.budget_for_fetched_text(self.settings(), used_tokens=7000)  # 30% still free
        assert 0 < budget < 1000  # below the per-fetch ceiling, but still worth serving

    def test_a_nonsensical_fraction_is_clamped_and_logged(self, monkeypatch, caplog):
        monkeypatch.setattr("raven.librarian.config.docs_fetch_max_fraction_of_context", -0.5)
        monkeypatch.setattr("raven.librarian.config.context_reserve_fraction", 0.25)
        caplog.set_level(logging.WARNING, logger="raven.librarian.llmclient")
        assert llmclient.budget_for_fetched_text(self.settings(), used_tokens=0) == 0
        assert "docs_fetch_max_fraction_of_context" in caplog.text


class TestTruncateMiddle:
    def test_short_text_is_returned_unchanged(self):
        assert llmclient.truncate_middle("hello", 100) == "hello"

    def test_result_fits_the_budget(self):
        out = llmclient.truncate_middle("x" * 5000, 500)
        assert len(out) <= 500

    def test_both_ends_survive(self):
        # The ends are what carry: for a paper, the abstract at one end and the conclusions at the other.
        text = "BEGINNING" + "." * 5000 + "ENDING"
        out = llmclient.truncate_middle(text, 500)
        assert out.startswith("BEGINNING")
        assert out.endswith("ENDING")

    def test_the_omission_is_stated(self):
        # Silently truncated text is indistinguishable from a document that simply ends there, and a model
        # will summarize the fragment as though it were the whole.
        out = llmclient.truncate_middle("x" * 5000, 500)
        assert "characters omitted" in out

    def test_a_budget_too_small_to_explain_itself_yields_nothing(self):
        assert llmclient.truncate_middle("x" * 5000, 5) == ""

    def test_token_budget_is_converted_not_used_as_characters(self):
        # The failure this guards: treating a token budget as a character budget, wrong by ~4x.
        settings = env(context_length=10000, tokens_per_character=0.25)
        out = llmclient.fit_text_to_token_budget(settings, "x" * 100000, budget_tokens=100)
        assert 300 < len(out) <= 400  # 100 tokens / 0.25 tokens-per-char = 400 characters

    def test_no_budget_yields_nothing(self):
        settings = env(context_length=10000, tokens_per_character=0.25)
        assert llmclient.fit_text_to_token_budget(settings, "x" * 1000, budget_tokens=0) == ""


class TestShareCharacters:
    """Max-min fair allocation of one budget over several attachments."""

    def test_everyone_gets_what_they_asked_for_when_it_fits(self):
        assert llmclient._share_characters([10, 20, 30], budget=100) == [10, 20, 30]

    def test_equal_appetites_split_evenly(self):
        assert llmclient._share_characters([500, 500], budget=100) == [50, 50]

    def test_a_modest_item_is_served_in_full_and_its_leftovers_raise_the_rest(self):
        # The point of max-min fairness over equal shares: cutting the 10-character item to 50 would free
        # characters nobody was asking for, and the 500-character one would be no better off for it.
        assert llmclient._share_characters([10, 500], budget=100) == [10, 90]

    def test_order_does_not_matter(self):
        # Two callers walk the same attachments in opposite directions and must agree.
        forwards = llmclient._share_characters([10, 500, 60], budget=100)
        backwards = llmclient._share_characters([60, 500, 10], budget=100)
        assert forwards == list(reversed(backwards))

    def test_no_budget_gives_nobody_anything(self):
        assert llmclient._share_characters([100, 200], budget=0) == [0, 0]
        assert llmclient._share_characters([100, 200], budget=-500) == [0, 0]


class TestFitAttachmentsToContext:
    """Sizing the user's attached documents against what the window has left."""

    # 10000 tokens at 0.25 tokens/character = 40000 characters; the 25% reserve leaves 30000.
    def settings(self):
        return env(context_length=10000, tokens_per_character=0.25)

    def requested(self, *texts):
        """`(text, kind)` pairs for documents the user handed over — the no-ceiling case."""
        return [(text, llmclient.ATTACHMENT_REQUESTED) for text in texts]

    def test_attachments_that_fit_are_returned_unchanged(self, monkeypatch):
        # The ordinary case, and it must be byte-identical to an unbudgeted fold: anything else would
        # rewrite the prompt prefix for no reason.
        monkeypatch.setattr("raven.librarian.config.context_reserve_fraction", 0.25)
        texts = ["a" * 1000, "b" * 2000]
        assert llmclient.fit_attachments_to_context(self.settings(), 500, self.requested(*texts)) == texts

    def test_an_oversized_attachment_is_cut_to_the_budget(self, monkeypatch):
        monkeypatch.setattr("raven.librarian.config.context_reserve_fraction", 0.25)
        out = llmclient.fit_attachments_to_context(self.settings(), 1000, self.requested("a" * 200000))
        assert len(out[0]) <= 29000  # 30000 minus the conversation, then quantized down
        assert "characters omitted" in out[0]

    def test_a_small_attachment_survives_beside_a_large_one(self, monkeypatch):
        # An attachment the user is still discussing must not be shredded merely because a book arrived
        # alongside it.
        monkeypatch.setattr("raven.librarian.config.context_reserve_fraction", 0.25)
        small = "the short note"
        out = llmclient.fit_attachments_to_context(self.settings(), 0, self.requested(small, "a" * 200000))
        assert out[0] == small
        assert len(out[1]) < 200000

    def test_no_per_attachment_ceiling_on_a_requested_document(self, monkeypatch):
        # An attachment is an instruction to read this, so a lone one may occupy everything the reserve
        # leaves. This is the half of the policy that must NOT change when the ceiling is applied to fetches.
        monkeypatch.setattr("raven.librarian.config.context_reserve_fraction", 0.25)
        monkeypatch.setattr("raven.librarian.config.docs_fetch_max_fraction_of_context", 0.10)
        out = llmclient.fit_attachments_to_context(self.settings(), 0, self.requested("a" * 200000))
        assert len(out[0]) > 4000  # 10% of the window would have been 4000 characters

    def test_a_speculative_document_is_ceilinged_even_when_it_would_fit(self, monkeypatch):
        # The page the model fetched on a hunch gets the per-document ceiling `fetch_document` already
        # applies. "Even when it would fit" is the point: the reserve alone would have let this through,
        # and before the ceiling was wired in here, it did.
        monkeypatch.setattr("raven.librarian.config.context_reserve_fraction", 0.25)
        monkeypatch.setattr("raven.librarian.config.docs_fetch_max_fraction_of_context", 0.10)
        out = llmclient.fit_attachments_to_context(self.settings(), 0,
                                                   [("a" * 20000, llmclient.ATTACHMENT_SPECULATIVE)])
        assert len(out[0]) <= 4000  # 10% of the 40000-character window
        assert "characters omitted" in out[0]

    def test_a_ceilinged_fetch_releases_its_leftovers_to_the_others(self, monkeypatch):
        # The ceiling clamps what a fetch *asks for*, before the fair split rather than after it, so the
        # characters it does not get are available to the attachment beside it. Clamping the allowance
        # instead would leave them unused.
        monkeypatch.setattr("raven.librarian.config.context_reserve_fraction", 0.25)
        monkeypatch.setattr("raven.librarian.config.docs_fetch_max_fraction_of_context", 0.10)
        attachment = "u" * 40000
        out = llmclient.fit_attachments_to_context(self.settings(), 0,
                                                   [("f" * 40000, llmclient.ATTACHMENT_SPECULATIVE),
                                                    (attachment, llmclient.ATTACHMENT_REQUESTED)])
        assert len(out[0]) <= 4000
        # 30000 of budget, of which the fetch takes at most 4000 -- the attachment gets the rest, which is
        # far more than the even split it would have received had both wanted 40000.
        assert len(out[1]) > 20000

    def test_kinds_are_sized_together_against_one_budget(self, monkeypatch):
        # A fetch and an attachment in the same conversation compete for the same room; the kinds change
        # each one's ceiling, not whether they share.
        monkeypatch.setattr("raven.librarian.config.context_reserve_fraction", 0.25)
        monkeypatch.setattr("raven.librarian.config.docs_fetch_max_fraction_of_context", 0.10)
        out = llmclient.fit_attachments_to_context(self.settings(), 0,
                                                   [("f" * 200000, llmclient.ATTACHMENT_SPECULATIVE),
                                                    ("u" * 200000, llmclient.ATTACHMENT_REQUESTED)])
        assert sum(len(text) for text in out) <= 30000

    def test_a_full_conversation_leaves_nothing(self, monkeypatch):
        monkeypatch.setattr("raven.librarian.config.context_reserve_fraction", 0.25)
        assert llmclient.fit_attachments_to_context(self.settings(), 40000, self.requested("a" * 1000)) == [""]

    def test_the_budget_holds_still_as_the_conversation_grows(self, monkeypatch):
        # Folded attachment text is part of the prompt prefix, so a budget that drifted turn by turn would
        # force a full prompt reprocess every turn, exactly where the prompt is already enormous.
        monkeypatch.setattr("raven.librarian.config.context_reserve_fraction", 0.25)
        text = "a" * 200000
        first = llmclient.fit_attachments_to_context(self.settings(), 1000, self.requested(text))
        later = llmclient.fit_attachments_to_context(self.settings(), 1600, self.requested(text))
        assert first == later


class TestAttachmentBudgetKind:
    """Which budget a stored `text_file` part falls under. One classifier, so the two readers cannot diverge."""

    def test_a_user_attachment_is_requested(self):
        part = chatutil.text_file_content_part("sidecar:abc.txt", "paper.pdf", "user_attachment")
        assert llmclient.attachment_budget_kind(part) == llmclient.ATTACHMENT_REQUESTED

    def test_a_tool_result_is_speculative(self):
        part = chatutil.text_file_content_part("sidecar:abc.md", "example.com - Title", "tool_result")
        assert llmclient.attachment_budget_kind(part) == llmclient.ATTACHMENT_SPECULATIVE

    def test_a_pasted_url_is_requested(self):
        # Reserved vocabulary, not emitted yet. The user typed the URL, so it is an instruction to read this
        # however the bytes arrived -- the budget axis is who asked, not where it came from.
        part = chatutil.text_file_content_part("sidecar:abc.html", "a page", "paste_url")
        assert llmclient.attachment_budget_kind(part) == llmclient.ATTACHMENT_REQUESTED

    def test_an_mcp_tool_result_is_speculative_whatever_the_server(self):
        part = chatutil.text_file_content_part("sidecar:abc.txt", "a doc", "mcp:some-server")
        assert llmclient.attachment_budget_kind(part) == llmclient.ATTACHMENT_SPECULATIVE

    def test_an_unknown_source_is_treated_as_requested(self):
        # The conservative direction: send it whole rather than silently truncate something that may have
        # been asked for.
        part = chatutil.text_file_content_part("sidecar:abc.txt", "a doc", "something_new")
        assert llmclient.attachment_budget_kind(part) == llmclient.ATTACHMENT_REQUESTED

    def test_a_part_predating_the_field_is_treated_as_requested(self):
        # Belt and braces: `upgrade_datastore` backfills `source` at load, so a part reaching the budget
        # without one means the migration was skipped. Degrade to the safe kind rather than raising.
        assert llmclient.attachment_budget_kind({"type": "text_file",
                                                 "text_file": {"url": "sidecar:abc.txt", "name": "old"}}) == llmclient.ATTACHMENT_REQUESTED


# ---------------------------------------------------------------------------
# Labelling a document for a list the model has to make a decision from
# ---------------------------------------------------------------------------

class TestDocumentLabel:
    """A search can return twenty documents; the label is what lets the model skip nineteen of them."""

    def test_a_bibtex_record_is_read_from_its_own_fields(self):
        # Web of Science output capitalizes the field keys, which is why the reader normalizes them
        # rather than matching on `title = {`.
        record = ("@article{WOS:000000000000001,\n"
                  "\tAuthor = {Kataoka, N and Miya, A and Kiriyama, K},\n"
                  "\tYear = {1997},\n"
                  "\tTitle = {Studies on hydrogen production},\n"
                  "\tJournal = {WATER SCIENCE AND TECHNOLOGY}\n}\n")
        assert chatutil.document_label(record) == '"Studies on hydrogen production" (Kataoka et al. 1997)'

    def test_author_names_survive_their_own_awkwardness(self):
        # Names are their own small horror, so they go through the same machinery as Visualizer citations
        # rather than a split on the first comma.
        record = "@book{x, title={Trio}, author={Ludwig van Beethoven and Beeblebrox, IV, Zaphod}, year={1808}}"
        assert chatutil.document_label(record) == '"Trio" (van Beethoven and Beeblebrox IV 1808)'

    def test_a_whole_reference_database_is_described_as_one(self):
        # A `.bib` the user dropped in whole is one document as far as the retriever is concerned, however
        # many works it lists -- and the decision it needs to drive is *not* to fetch it.
        record = "@article{a, title={One}}\n\n@article{b, title={Two}}\n"
        assert chatutil.document_label(record) == "BibTeX database of 2 records"  # the count says "do not fetch this"

    def test_a_plain_document_falls_back_to_its_first_line(self):
        assert chatutil.document_label("Meeting minutes, 4 March\n\nPresent: ...") == "Meeting minutes, 4 March"

    def test_a_document_that_describes_itself_with_nothing_gets_no_label(self):
        # `""` reads correctly as "the ID is all there is" -- the caller shows the ID anyway.
        assert chatutil.document_label("...\n\n") == ""

    def test_the_label_does_not_run_away_with_the_list(self):
        assert len(chatutil.document_label("T" * 5000)) < 250


class TestFormatConsultedDocuments:
    def test_an_entry_carries_id_label_and_query(self):
        out = chatutil.format_consulted_documents([{"document_id": "a.bib", "label": '"Paper"', "query": "hydrogen"}])
        assert "a.bib" in out
        assert '"Paper"' in out
        assert "hydrogen" in out

    def test_an_unlabelled_entry_still_names_its_id(self):
        out = chatutil.format_consulted_documents([{"document_id": "a.txt"}])
        assert "a.txt" in out

    def test_an_essay_length_query_is_shortened(self):
        # The auto-search query is the user's whole message, so it can be an essay. It is shown to say
        # *why* a document is on the list, which the first line of it does.
        out = chatutil.format_consulted_documents([{"document_id": "a.txt", "query": "q" * 5000}])
        assert len(out) < 1000

    def test_the_header_does_not_claim_the_text_is_gone(self):
        # True of the automatic search, whose matches are never persisted; false of a document the model
        # fetched, which is a stored node still written out where the window reaches.
        out = chatutil.format_consulted_documents([{"document_id": "a.txt"}])
        assert "no longer written out above" in out

    def test_a_deleted_document_says_so(self):
        # The user can delete a document mid-conversation. It stays listed - the conversation did read it -
        # but the model must not have to spend a `fetch_document` round finding out it is gone.
        out = chatutil.format_consulted_documents([{"document_id": "a.txt", "present": False}])
        assert "[no longer in the database]" in out

    def test_an_unlabelled_document_is_not_mistaken_for_a_deleted_one(self):
        # The reason `present` exists rather than testing the label: an empty label means *either*, and the
        # two want opposite responses from the model.
        out = chatutil.format_consulted_documents([{"document_id": "a.txt", "label": "", "present": True}])
        assert "no longer in the database" not in out

    def test_an_entry_without_the_flag_reads_as_present(self):
        # Entries assembled without `llmclient.label_documents` must render exactly as before.
        assert "no longer in the database" not in chatutil.format_consulted_documents([{"document_id": "a.txt"}])

    def test_a_title_survives_a_record_that_will_not_parse(self):
        # Real corpora contain records that are not quite valid BibTeX: an abstract with unbalanced braces
        # aborts the parse of the whole record, title and all (one in ~12000 on the hydrogen corpus).
        record = ("@article{WOS:000000000000002,\n"
                  "\tAuthor = {Afgan, Nain H. and Carvalho, Maria G.},\n"
                  "\tTitle = {Sustainability assessment of a hybrid energy system},\n"
                  "\tAbstract = {The Object structure is defined as: Hybrid Energy System {a, b},\n"
                  "\tYear = {2008}\n}\n")
        assert chatutil.document_label(record) == '"Sustainability assessment of a hybrid energy system"'

    def test_a_broken_record_never_labels_itself_with_a_field_name(self):
        # Falling through to "first substantial line" would surface `Author = {...}`, which looks like a bug
        # even though nothing of ours went wrong. No label is the honest answer there.
        record = "@article{k,\n\tAuthor = {Nobody, A.},\n\tAbstract = {unbalanced {\n}\n"
        assert chatutil.document_label(record) == ""


class TestTokenizerProbe:
    """The probes that check a local tokenizer against the backend must not disturb what they measure."""

    def test_the_probe_asks_for_no_recalibration(self, monkeypatch):
        # `invoke` refines `settings.tokens_per_character` from each call's usage, and
        # `fit_attachments_to_context` turns a token budget back into characters with that ratio. A probe is
        # short and mostly chat-template framing, so the ratio it implies is far too high — and calibrating on
        # it silently truncates the attachments of the branch being measured. Observed as a branch total
        # dropping from 86655 tokens to 51956 with nothing else changed.
        seen = {}

        def fake_prefill(settings, history, **kwargs):
            seen.update(kwargs)
            return None

        monkeypatch.setattr(llmclient, "prefill", fake_prefill)
        settings = env(backend_flavor="lmstudio", backend_url="http://localhost:1", tokens_per_character=0.27)
        llmclient._make_backend_token_counter(settings)("a probe")
        assert seen.get("calibrate") is False, "a probe that recalibrates changes the number it was sent to check"

    def test_an_ordinary_prefill_still_calibrates(self, monkeypatch):
        # The context prefill sends a real conversation, which is exactly what the ratio should learn from.
        seen = {}

        def fake_invoke(settings, history, **kwargs):
            seen.update(kwargs)
            return None

        monkeypatch.setattr(llmclient, "invoke", fake_invoke)
        llmclient.prefill(env(), _history("a real conversation"))
        assert seen.get("calibrate") is True, ("the default changed, so the probe's `calibrate=False` no longer "
                                               "distinguishes it from an ordinary call")


class TestAbortingAnInFlightRequest:
    """Abandoning a request from another thread, which is what Cancel and a superseded prefill both need.

    The mechanism is measured in `investigations/abort-inflight-request/`, and the finding that shapes these
    tests is that the obvious route does not work: closing the response leaves the reader blocked *and*
    blocks the thread that asked. So the assertion that matters is not "an exception was raised" but "the
    blocked call came back promptly" — which is worth a real socket, because a mocked stream cannot be
    blocked and so cannot show that unblocking happened.
    """

    WATCH = 2.0  # an abort that works lands in milliseconds; slower than this is the mechanism not working
    CEILING = 30.0  # so that neither the server nor a client timeout can be what ends a wait

    @classmethod
    def _stall_server(cls):
        """Serve 200 plus SSE headers, then never send a body. Returns `(url, shutdown)`."""
        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                self.wfile.flush()
                time.sleep(cls.CEILING)

            def log_message(self, *args):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{server.server_address[1]}", server.shutdown

    def _blocked_invoke(self, invoke_settings, maybe_abort):
        """Start `invoke` against a stall server on its own thread. Returns `(thread, outcome, shutdown)`."""
        url, shutdown = self._stall_server()
        invoke_settings.backend_url = url
        outcome = {}

        def run():
            try:
                llmclient.invoke(invoke_settings, _history("hi"), tools_enabled=False, maybe_abort=maybe_abort)
                outcome["result"] = "returned"
            except BaseException as exc:  # noqa: BLE001 -- which exception arrived *is* the measurement
                outcome["result"] = type(exc)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread, outcome, shutdown

    def test_the_socket_under_a_streaming_response_is_reachable(self, invoke_settings):
        """The private-attribute path to the socket resolves on this platform and this `requests`.

        Separate from the abort tests because it isolates the half that fails silently. `Abort.abort` no-ops
        when the socket cannot be found, so "the read was not abandoned" has two possible causes — the path
        not resolving, or the closers not waking the reader — and they need different fixes. This one says
        which. (It is also the tripwire for a `requests`/`urllib3` version that moves the internals.)
        """
        url, shutdown = self._stall_server()
        response = requests.post(url, json={"x": 1}, stream=True, timeout=self.CEILING)
        try:
            assert netutil._maybe_socket_of(response) is not None
        finally:
            response.close()
            shutdown()

    def test_a_blocked_read_is_abandoned_promptly(self, invoke_settings):
        abort = netutil.Abort()
        thread, outcome, shutdown = self._blocked_invoke(invoke_settings, abort)
        try:
            time.sleep(0.5)  # let it reach the blocking read
            assert thread.is_alive(), ("the request finished on its own, so this fixture cannot tell an "
                                       "abandoned read from a completed one")  # <- the negative control
            abort.abort()
            thread.join(timeout=self.WATCH)
            assert not thread.is_alive(), "the blocked read was not abandoned"
            assert outcome["result"] is netutil.Aborted
        finally:
            shutdown()

    def test_aborting_returns_at_once_to_its_own_caller(self, invoke_settings):
        """The caller is usually a GUI callback, and the `close()` route would block it for the read timeout.

        The abort runs on its own thread rather than this one, so that a closer which blocks fails this
        assertion instead of wedging the test run — which is what the `close()` variants actually do, and a
        hung suite says far less than a red one.
        """
        abort = netutil.Abort()
        thread, unused_outcome, shutdown = self._blocked_invoke(invoke_settings, abort)
        returned = threading.Event()
        try:
            time.sleep(0.5)
            threading.Thread(target=lambda: (abort.abort(), returned.set()), daemon=True).start()
            assert returned.wait(timeout=self.WATCH), "abort() did not return to its caller"
        finally:
            thread.join(timeout=self.WATCH)
            shutdown()

    def test_aborting_before_the_call_stops_the_request_being_sent(self, monkeypatch, invoke_settings):
        """A turn cancelled while still queued in the executor should never reach the backend."""
        posted = []

        def fake_post(*args, **kwargs):
            posted.append(True)
            return _FakeResponse()

        monkeypatch.setattr(llmclient.requests, "post", fake_post)
        abort = netutil.Abort()
        abort.abort()
        with pytest.raises(netutil.Aborted):
            llmclient.invoke(invoke_settings, _history("hi"), tools_enabled=False, maybe_abort=abort)
        assert not posted

    def test_a_connection_failure_we_did_not_ask_for_is_not_an_abort(self, monkeypatch, invoke_settings):
        """The discriminating case: the same broken connection, with nobody having aborted.

        Both arrive at the same `except`, and only the handle can say which is which — so a branch that
        reported every dropped connection as an abort would pass every other test in this class.
        """
        _fake_stream(monkeypatch, [])

        def die(self_):
            raise llmclient.requests.exceptions.ChunkedEncodingError("the backend went away")

        monkeypatch.setattr(_FakeSSEClient, "events", die)
        with pytest.raises(llmclient.requests.exceptions.ChunkedEncodingError):
            llmclient.invoke(invoke_settings, _history("hi"), tools_enabled=False, maybe_abort=netutil.Abort())

    def test_the_handle_is_disarmed_once_the_call_is_over(self, monkeypatch, invoke_settings):
        """Aborting a spent handle must not reach into whatever owns that socket now."""
        abort = netutil.Abort()
        _fake_stream(monkeypatch, [{"choices": [{"delta": {"content": "ok"}}]}, "[DONE]"])
        llmclient.invoke(invoke_settings, _history("hi"), tools_enabled=False, maybe_abort=abort)
        abort.abort()  # a no-op, not an AttributeError on a dead response
        assert abort.aborted


class TestCapturedStrayThinkReply:
    """A real reply in which the model put a second `<think>` block on the *content* channel.

    Qwen3.6-35B-A3B, captured 2026-08-27 (`data/qwen36_stray_think_reply.json`). It had already delivered
    10.9k characters of reasoning on the native `reasoning_content` channel; it then emitted, as content,
    `Aria: <think>` … `</think>` … the answer … and a final `</think>` that was never opened. One
    occurrence in a day's use, and not reproducible on demand — which is why the stream is in the tree
    rather than described in a comment.

    **The capture also outruns what is understood about it.** The stored node holds that content raw, tags
    and all, and the only writer of stored content is the parser's `content` events — yet the parser below
    routes the inner block to `reasoning` at every chunk size from one character up. So the live run and
    the parser disagree about the same bytes, and the artifact does not say why. These tests pin what the
    parser does today, so that whoever resolves it can see immediately whether the answer moved.
    """

    @staticmethod
    def _reply():
        path = pathlib.Path(__file__).parent / "data" / "qwen36_stray_think_reply.json"
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _replay(content: str, chunk: int) -> dict[str, str]:
        """Feed `content` through a parser that has already seen native reasoning, `chunk` characters at a time."""
        parser = llmclient.StreamParser()
        out = {"content": "", "reasoning": ""}
        for event in parser.feed("", "reasoning on the native channel\n"):
            out[event["type"]] += event["text"]
        for i in range(0, len(content), chunk):
            for event in parser.feed(content[i:i + chunk], ""):
                if event["type"] in out:
                    out[event["type"]] += event.get("text", "")
        return out

    def test_the_fixture_still_holds_the_shape_it_was_captured_for(self):
        """A guard on the data, not on the parser: these are the three tags the rest of the class is about."""
        content = self._reply()["content"]
        assert content.count("<think>") == 1
        assert content.count("</think>") == 2, "the unmatched close is the whole point of this capture"

    @pytest.mark.parametrize("chunk", [1, 2, 3, 7, 8, 37, 100000])
    def test_an_opening_tag_is_routed_however_the_stream_is_chopped(self, chunk):
        """The open tag may straddle a chunk boundary, and the parser holds back a partial tag for that."""
        out = self._replay(self._reply()["content"], chunk)
        assert "<think>" not in out["content"]
        assert "1: The multiplicative identity." in out["reasoning"], "the block's text was not routed"

    def test_an_unmatched_close_is_left_in_the_answer(self):
        """Deliberate: with reasoning arriving on its own channel, an orphan close is no longer evidence.

        `_may_retcon` is off by then — the earlier text demonstrably was *not* unmarked reasoning — so
        there is nothing to infer and nothing to move. The renderer shows the tag rather than dropping it,
        which is what let this reply be recognized as malformed at all.
        """
        out = self._replay(self._reply()["content"], 37)
        assert out["content"].rstrip().endswith("</think>")

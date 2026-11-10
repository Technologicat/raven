"""Tests for `raven.librarian.mcpclient`, against a real MCP server over real transports."""

import json
import pathlib
import socket
import subprocess
import sys
import threading
import time

import pytest
from unpythonic import dyn
from unpythonic.env import env

from raven.common import netutil
from raven.librarian import llmclient
from raven.librarian import llmtools
from raven.librarian import mcpclient

_FIXTURE_SERVER = str(pathlib.Path(__file__).parent / "mcp_fixture_server.py")


def _stdio_entry(**extra) -> dict:
    return {"command": sys.executable, "args": [_FIXTURE_SERVER, "stdio"], **extra}

def _write_config(path: pathlib.Path, servers: dict) -> pathlib.Path:
    path.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")
    return path

def _make_settings() -> env:
    """Real settings, built with no backend in the picture."""
    model_info = env(label="test-model", model_id="test-model", context_length=4096, is_vlm=False, loaded=True)
    return llmclient.configure(model_info=model_info, backend_flavor="generic", backend_url="http://localhost:1", quiet=True)

def _offered_names(settings: env) -> set[str]:
    return {spec["function"]["name"] for spec in llmtools.offered_tools(settings)}

def _call(settings: env, tool: str, /, **arguments):
    return settings.tool_entrypoints[tool](**arguments)


@pytest.fixture
def settings():
    return _make_settings()

@pytest.fixture(scope="module")
def module_config(tmp_path_factory):
    return _write_config(tmp_path_factory.mktemp("mcp") / "mcp.json", {"fixture": _stdio_entry(env={"RAVEN_MCP_TEST": "from-config"})})

@pytest.fixture(scope="module")
def connected_settings(module_config):
    """One server for the whole module; starting one costs about a second."""
    the_settings = _make_settings()
    with mcpclient.connected(the_settings, config_file=module_config) as client:
        yield the_settings, client


class TestLoadConfig:
    def test_a_missing_file_means_no_servers_and_no_problems(self, tmp_path):
        assert mcpclient.load_config(tmp_path / "nope.json") == ([], {})

    def test_accepts_a_string_path_too(self, tmp_path):
        path = _write_config(tmp_path / "mcp.json", {"a": {"command": "x"}})
        servers, _problems = mcpclient.load_config(str(path))
        assert [server.name for server in servers] == ["a"]

    @pytest.mark.parametrize("entry, transport", [({"command": "x"}, "stdio"),
                                                  ({"url": "http://h/mcp"}, "http"),
                                                  ({"url": "http://h/sse", "type": "sse"}, "sse"),
                                                  ({"url": "http://h/mcp", "type": "streamable-http"}, "http"),
                                                  ({"url": "http://h/mcp", "type": "streamableHttp"}, "http"),
                                                  ({"url": "http://h/mcp", "type": "HTTP"}, "http"),
                                                  ({"command": "x", "url": "http://h/mcp", "type": "http"}, "http")])
    def test_the_transport_is_inferred_and_type_overrides_it(self, tmp_path, entry, transport):
        servers, problems = mcpclient.load_config(_write_config(tmp_path / "mcp.json", {"a": entry}))
        assert problems == {}
        assert servers[0].transport == transport

    def test_a_disabled_server_is_skipped_silently(self, tmp_path):
        servers, problems = mcpclient.load_config(_write_config(tmp_path / "mcp.json", {"a": {"command": "x", "disabled": True},
                                                                                        "b": {"command": "y"}}))
        assert [server.name for server in servers] == ["b"]
        assert problems == {}

    @pytest.mark.parametrize("entry, message", [({}, "neither"),
                                                ({"url": "http://h", "type": "carrier-pigeon"}, "carrier-pigeon"),
                                                ({"type": "stdio"}, "command"),
                                                ({"type": "sse"}, "url"),
                                                ("a string", "object"),
                                                ({"command": "x", "allowed_tools": "echo"}, "allowed_tools")])
    def test_a_bad_entry_is_reported_by_name_and_the_others_survive(self, tmp_path, entry, message):
        servers, problems = mcpclient.load_config(_write_config(tmp_path / "mcp.json", {"bad": entry, "good": {"command": "x"}}))
        assert [server.name for server in servers] == ["good"]
        assert message in problems["bad"]

    @pytest.mark.parametrize("content", ["{not json", "[1, 2]", '{"mcpServers": 3}'])
    def test_a_bad_file_is_a_problem_and_not_an_exception(self, tmp_path, content):
        path = tmp_path / "mcp.json"
        path.write_text(content, encoding="utf-8")
        servers, problems = mcpclient.load_config(path)
        assert servers == []
        assert "" in problems

    def test_the_lm_studio_shape_loads_unchanged(self, tmp_path):
        servers, problems = mcpclient.load_config(_write_config(tmp_path / "mcp.json", {"open-meteo": {"command": "bash", "args": ["-lc", "npx -y open-meteo-mcp-server"]}}))
        assert problems == {}
        assert servers[0].args == ("-lc", "npx -y open-meteo-mcp-server")


class TestPublicName:
    def test_is_namespaced_by_server(self):
        assert mcpclient._public_name("open-meteo", "forecast") == "open-meteo__forecast"

    def test_replaces_what_a_function_name_may_not_contain(self):
        assert mcpclient._public_name("my server", "a.b") == "my_server__a_b"

    def test_a_long_name_is_cut_to_the_limit_and_stays_distinct(self):
        one = mcpclient._public_name("s" * 40, "t" * 40 + "one")
        two = mcpclient._public_name("s" * 40, "t" * 40 + "two")
        assert len(one) == len(two) == 64
        assert one != two
        assert one == mcpclient._public_name("s" * 40, "t" * 40 + "one")


class TestRegistration:
    def test_tools_are_offered_beside_the_built_ins(self, connected_settings):
        the_settings, _client = connected_settings
        offered = _offered_names(the_settings)
        assert "fixture__echo" in offered
        assert "websearch" in offered

    def test_the_spec_carries_the_servers_schema(self, connected_settings):
        the_settings, _client = connected_settings
        spec = next(spec for spec in the_settings.request_data["tools"] if spec["function"]["name"] == "fixture__add")
        assert spec["type"] == "function"
        assert spec["function"]["description"] == "Add two integers."
        assert set(spec["function"]["parameters"]["properties"]) == {"a", "b"}
        assert set(spec["function"]["parameters"]["required"]) == {"a", "b"}

    def test_tools_are_listed_in_a_stable_order(self, connected_settings):
        the_settings, _client = connected_settings
        names = [spec["function"]["name"] for spec in the_settings.tools if spec["function"]["name"].startswith("fixture__")]
        assert names == sorted(names)

    def test_the_gate_withholds_them_and_only_them(self, connected_settings):
        the_settings, client = connected_settings
        without = llmtools.maybe_tool_names_for_turn(the_settings, documents_available=True, internet_available=True, mcp_available=False)
        assert not (set(without) & client.tool_names)
        assert "websearch" in without
        with_all = llmtools.maybe_tool_names_for_turn(the_settings, documents_available=True, internet_available=True, mcp_available=True)
        assert with_all is None  # every tool on offer

    def test_the_module_level_tables_are_left_alone(self, connected_settings):
        _the_settings, client = connected_settings
        assert not (client.tool_names & set(llmtools.TOOL_ENTRYPOINTS))
        assert not (client.tool_names & {spec["function"]["name"] for spec in llmtools.TOOLS})

    def test_another_settings_object_gets_none_of_them(self, connected_settings):
        _the_settings, client = connected_settings
        assert not (client.tool_names & _offered_names(_make_settings()))

    def test_they_survive_a_reconnect(self, connected_settings, monkeypatch):
        the_settings, client = connected_settings
        monkeypatch.setattr(llmclient, "connect", lambda backend_url, quiet=True: _make_settings())
        llmclient.reconnect(the_settings)
        assert client.tool_names <= _offered_names(the_settings)
        assert the_settings.mcp_tool_names == client.tool_names

    def test_allowed_tools_restricts_what_is_offered(self, tmp_path, settings):
        config = _write_config(tmp_path / "mcp.json", {"fixture": _stdio_entry(allowed_tools=["echo", "add"])})
        with mcpclient.connected(settings, config_file=config) as client:
            assert client.tool_names == {"fixture__echo", "fixture__add"}

    def test_a_name_in_allowed_tools_the_server_does_not_publish_is_reported(self, tmp_path, settings, caplog):
        config = _write_config(tmp_path / "mcp.json", {"fixture": _stdio_entry(allowed_tools=["echo", "ecoh"])})
        with mcpclient.connected(settings, config_file=config) as client:
            assert client.tool_names == {"fixture__echo"}
        assert "ecoh" in caplog.text


class TestCallingATool:
    def test_returns_the_text_and_says_which_server(self, connected_settings):
        the_settings, _client = connected_settings
        text, metadata = _call(the_settings, "fixture__echo", text="hello")
        assert text == "hello"
        assert metadata["mcp_server"] == "fixture"

    def test_the_configured_environment_reaches_the_server(self, connected_settings):
        the_settings, _client = connected_settings
        text, _metadata = _call(the_settings, "fixture__getenv", name="RAVEN_MCP_TEST")
        assert text == "from-config"

    def test_a_successful_result_offers_itself_as_a_document(self, connected_settings):
        the_settings, _client = connected_settings
        _text, metadata = _call(the_settings, "fixture__big", n_characters=10000)
        assert metadata["fetched_document"]["source"] == "mcp:fixture"
        assert netutil.url_host(metadata["fetched_document"]["url"]) == "fixture"

    def test_an_error_inside_the_tool_is_reported_to_the_model(self, connected_settings):
        the_settings, _client = connected_settings
        text, metadata = _call(the_settings, "fixture__fail")
        assert "deliberate failure" in text
        assert text.startswith("The tool fail reported an error")
        assert metadata["mcp_error"] is True
        assert "fetched_document" not in metadata

    def test_through_the_real_dispatch(self, connected_settings):
        the_settings, _client = connected_settings
        message = {"role": "assistant",
                   "tool_calls": [{"type": "function", "id": "call_1",
                                   "function": {"name": "fixture__add", "arguments": '{"a": 2, "b": 3}'}}]}
        [record] = llmtools.perform_tool_calls(the_settings, message, on_call_start=None, on_call_done=None)
        assert record.status == "success"
        assert record.data["content"][0]["text"] == "5"
        assert record.tool_metadata["mcp_server"] == "fixture"

    def test_many_calls_at_once(self, connected_settings):
        the_settings, _client = connected_settings
        results = [None] * 8
        def work(i):
            results[i], _metadata = _call(the_settings, "fixture__echo", text=str(i))
        threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert results == [str(i) for i in range(8)]

    def test_an_abort_ends_a_slow_call_promptly(self, connected_settings):
        the_settings, _client = connected_settings
        abort = netutil.Abort()
        threading.Timer(0.2, abort.abort).start()
        t0 = time.monotonic()
        with dyn.let(tool_context=env(maybe_abort=abort)):
            with pytest.raises(netutil.Aborted):
                _call(the_settings, "fixture__slow", seconds=30.0)
        assert time.monotonic() - t0 < 5.0

    def test_a_call_past_the_timeout_is_reported_to_the_model(self, connected_settings, monkeypatch):
        the_settings, _client = connected_settings
        monkeypatch.setattr(mcpclient.librarian_config, "mcp_call_timeout", 0.2)
        text, _metadata = _call(the_settings, "fixture__slow", seconds=3.0)
        assert text.startswith("The call to slow failed")


class TestFailures:
    def test_no_config_file_is_quietly_nothing(self, tmp_path, settings):
        before = list(settings.tools)
        with mcpclient.connected(settings, config_file=tmp_path / "nope.json") as client:
            assert client.tool_names == frozenset()
            assert client.problems == {}
            assert settings.tools == before

    def test_a_server_that_cannot_start_does_not_stop_the_others(self, tmp_path, settings):
        config = _write_config(tmp_path / "mcp.json", {"broken": {"command": "/nonexistent/mcp-server"},
                                                       "fixture": _stdio_entry()})
        with mcpclient.connected(settings, config_file=config) as client:
            assert "broken" in client.problems
            assert "fixture__echo" in client.tool_names
            assert _call(settings, "fixture__echo", text="still here")[0] == "still here"

    def test_a_server_that_never_answers_times_out(self, tmp_path, settings):
        # `sleep` starts and says nothing, which is what a wedged server looks like.
        config = _write_config(tmp_path / "mcp.json", {"mute": {"command": sys.executable, "args": ["-c", "import time; time.sleep(60)"]}})
        client = mcpclient.MCPClient(settings, config_file=config)
        t0 = time.monotonic()
        try:
            client.start(timeout=0.5)
            assert "mute" in client.problems
            assert "no answer" in client.problems["mute"]
        finally:
            client.shutdown(timeout=1.0)
        assert time.monotonic() - t0 < 10.0

    def test_a_server_that_dies_mid_call_answers_the_model_and_the_next_call_too(self, tmp_path, settings):
        config = _write_config(tmp_path / "mcp.json", {"fixture": _stdio_entry()})
        with mcpclient.connected(settings, config_file=config):
            text, _metadata = _call(settings, "fixture__die")
            assert text.startswith("The call to die failed")
            text, _metadata = _call(settings, "fixture__echo", text="anyone there?")
            assert "failed" in text or "not available" in text


class TestLifecycle:
    def test_leaving_the_block_removes_the_tools(self, tmp_path, settings):
        before_tools, before_entrypoints = list(settings.tools), dict(settings.tool_entrypoints)
        config = _write_config(tmp_path / "mcp.json", {"fixture": _stdio_entry()})
        with mcpclient.connected(settings, config_file=config):
            assert settings.tools != before_tools
        assert settings.tools == before_tools
        assert settings.request_data["tools"] == before_tools
        assert settings.tool_entrypoints == before_entrypoints
        assert settings.mcp_tool_names == frozenset()

    def test_the_server_process_is_gone_afterwards(self, tmp_path, settings):
        psutil = pytest.importorskip("psutil")
        config = _write_config(tmp_path / "mcp.json", {"fixture": _stdio_entry()})
        with mcpclient.connected(settings, config_file=config):
            pid = int(_call(settings, "fixture__pid")[0])
            assert psutil.pid_exists(pid), "the server was not running, so this fixture cannot tell a stopped one from a missing one"
        assert not psutil.pid_exists(pid)

    def test_an_entrypoint_held_past_shutdown_says_so(self, tmp_path, settings):
        config = _write_config(tmp_path / "mcp.json", {"fixture": _stdio_entry()})
        with mcpclient.connected(settings, config_file=config):
            stale = settings.tool_entrypoints["fixture__echo"]
        text, _metadata = stale(text="too late")
        assert "not available" in text

    def test_shutdown_is_idempotent_and_safe_before_start(self, tmp_path, settings):
        client = mcpclient.MCPClient(settings, config_file=tmp_path / "nope.json")
        client.shutdown()
        client.shutdown()

    def test_starting_twice_is_refused(self, tmp_path, settings):
        config = _write_config(tmp_path / "mcp.json", {"fixture": _stdio_entry()})
        client = mcpclient.MCPClient(settings, config_file=config)
        client.start()
        try:
            with pytest.raises(RuntimeError, match="already started"):
                client.start()
        finally:
            client.shutdown()

    def test_a_slow_call_in_flight_cannot_hold_up_shutdown(self, tmp_path, settings):
        config = _write_config(tmp_path / "mcp.json", {"fixture": _stdio_entry()})
        client = mcpclient.MCPClient(settings, config_file=config)
        client.start()
        slow = settings.tool_entrypoints["fixture__slow"]
        threading.Thread(target=lambda: slow(seconds=60.0), daemon=True).start()
        time.sleep(0.3)
        t0 = time.monotonic()
        client.shutdown(timeout=2.0)
        assert time.monotonic() - t0 < 10.0


@pytest.fixture(scope="module")
def http_server():
    """The fixture server over streamable HTTP, on a free port."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    process = subprocess.Popen([sys.executable, _FIXTURE_SERVER, "http", str(port)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 30.0
        while True:
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                break
            except OSError:
                if time.monotonic() > deadline or process.poll() is not None:
                    raise RuntimeError("the fixture server did not start listening") from None
                time.sleep(0.05)
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        process.terminate()
        process.wait(timeout=10)


class TestHTTPTransport:
    def test_a_tool_call_completes(self, tmp_path, settings, http_server):
        config = _write_config(tmp_path / "mcp.json", {"web": {"url": http_server}})
        with mcpclient.connected(settings, config_file=config) as client:
            assert client.problems == {}
            assert _call(settings, "web__echo", text="over http")[0] == "over http"

    def test_configured_headers_reach_the_server(self, tmp_path, settings, http_server):
        config = _write_config(tmp_path / "mcp.json", {"web": {"url": http_server, "headers": {"X-Api-Key": "sesame"}}})
        with mcpclient.connected(settings, config_file=config):
            assert _call(settings, "web__request_header", name="x-api-key")[0] == "sesame"

    def test_without_headers_the_control_sees_none(self, tmp_path, settings, http_server):
        config = _write_config(tmp_path / "mcp.json", {"web": {"url": http_server}})
        with mcpclient.connected(settings, config_file=config):
            assert _call(settings, "web__request_header", name="x-api-key")[0] == "<unset>"

    def test_an_address_nothing_answers_at_is_a_problem(self, tmp_path, settings):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        config = _write_config(tmp_path / "mcp.json", {"web": {"url": f"http://127.0.0.1:{port}/mcp"}})
        with mcpclient.connected(settings, config_file=config) as client:
            assert "web" in client.problems

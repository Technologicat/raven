"""Tests for `raven.librarian.mcpclient`, against a real MCP server over real transports."""

import json
import pathlib
import socket
import subprocess
import sys
import threading
import time
import uuid

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

def _mute_entry() -> tuple[str, dict]:
    """A server that starts and says nothing, as a wedged one does, and a marker that finds its process.

    The marker is unique so that it matches no other process, such as a shell whose command line quotes this file.
    """
    marker = f"mute_{uuid.uuid4().hex}"
    return marker, {"command": sys.executable, "args": ["-c", f"import time; time.sleep(60)  # {marker}"]}

def _processes_with(marker: str) -> list[int]:
    psutil = pytest.importorskip("psutil")
    return [process.pid for process in psutil.process_iter(["cmdline"]) if marker in " ".join(process.info["cmdline"] or [])]

def _make_settings() -> env:
    """Real settings, built with no backend in the picture."""
    model_info = env(label="test-model", model_id="test-model", context_length=4096, is_vlm=False, loaded=True)
    return llmclient.configure(model_info=model_info, backend_flavor="generic", backend_url="http://localhost:1", quiet=True)

def _offered_names(settings: env) -> set[str]:
    return {spec["function"]["name"] for spec in llmtools.offered_tools(settings)}

def _call(settings: env, tool: str, /, **arguments):
    return settings.tool_entrypoints[tool](**arguments)


@pytest.fixture(scope="module", autouse=True)
def _logs_stay_out_of_the_users_config(tmp_path_factory):
    original = mcpclient.librarian_config.mcp_log_dir
    mcpclient.librarian_config.mcp_log_dir = tmp_path_factory.mktemp("mcp_logs")
    yield
    mcpclient.librarian_config.mcp_log_dir = original

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
        monkeypatch.setattr(llmclient, "detect_backend_flavor", lambda backend_url: "generic")
        monkeypatch.setattr(llmclient, "resolve_model_info", lambda backend_url, backend_flavor: _make_settings().model_record)
        llmclient.reconnect(the_settings)
        assert client.tool_names <= _offered_names(the_settings)
        assert set(the_settings.mcp_tool_servers) == client.tool_names

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
        marker, entry = _mute_entry()
        config = _write_config(tmp_path / "mcp.json", {"mute": entry})
        client = mcpclient.MCPClient(settings, config_file=config)
        t0 = time.monotonic()
        try:
            client.start(timeout=0.5)
            assert "mute" in client.problems
            assert "no answer" in client.problems["mute"]
        finally:
            client.shutdown()
        assert time.monotonic() - t0 < 10.0
        assert not _processes_with(marker)

    def test_a_server_that_dies_mid_call_answers_the_model_and_the_next_call_too(self, tmp_path, settings):
        config = _write_config(tmp_path / "mcp.json", {"fixture": _stdio_entry()})
        with mcpclient.connected(settings, config_file=config):
            text, _metadata = _call(settings, "fixture__die")
            assert text.startswith("The call to die failed")
            assert _call(settings, "fixture__echo", text="anyone there?")[0] == "anyone there?"


class TestServerOutput:
    def test_stderr_goes_to_the_servers_log_file(self, tmp_path, settings):
        config = _write_config(tmp_path / "mcp.json", {"my server": _stdio_entry()})
        with mcpclient.connected(settings, config_file=config):
            _call(settings, "my_server__complain", text="grumble")
        assert "grumble" in mcpclient._stderr_log_path("my server").read_text(encoding="utf-8")

    def test_a_server_that_exits_at_once_says_where_to_read_why(self, tmp_path, settings):
        script = "import sys; print('missing licence file', file=sys.stderr); sys.exit(3)"
        config = _write_config(tmp_path / "mcp.json", {"quitter": {"command": sys.executable, "args": ["-c", script]}})
        with mcpclient.connected(settings, config_file=config) as client:
            log_path = mcpclient._stderr_log_path("quitter")
            assert str(log_path) in client.problems["quitter"]
        assert "missing licence file" in log_path.read_text(encoding="utf-8")


class TestConcurrency:
    def _timed_round(self, settings, *tool_names) -> float:
        calls = [{"type": "function", "id": f"call_{i}", "function": {"name": name, "arguments": '{"seconds": 1.0}'}}
                 for i, name in enumerate(tool_names)]
        t0 = time.monotonic()
        records = llmtools.perform_tool_calls(settings, {"role": "assistant", "tool_calls": calls}, on_call_start=None, on_call_done=None)
        assert [record.status for record in records] == ["success"] * len(calls)
        return time.monotonic() - t0

    def test_each_server_is_its_own_group(self, tmp_path, settings):
        config = _write_config(tmp_path / "mcp.json", {"one": _stdio_entry(), "two": _stdio_entry()})
        with mcpclient.connected(settings, config_file=config):
            assert self._timed_round(settings, "one__slow", "one__slow") >= 2.0, "two calls to one server overlapped"
            assert self._timed_round(settings, "one__slow", "two__slow") < 1.8, "two servers ran one after the other"


class TestRestart:
    @pytest.fixture
    def quick_restarts(self, monkeypatch):
        monkeypatch.setattr(mcpclient.librarian_config, "mcp_restart_delay", 0.05)
        monkeypatch.setattr(mcpclient.librarian_config, "mcp_liveness_interval", 0.2)

    def test_a_server_that_dies_mid_call_comes_back(self, tmp_path, settings, quick_restarts, caplog):
        config = _write_config(tmp_path / "mcp.json", {"fixture": _stdio_entry()})
        with mcpclient.connected(settings, config_file=config):
            first_pid = int(_call(settings, "fixture__pid")[0])
            assert _call(settings, "fixture__die")[0].startswith("The call to die failed")
            assert int(_call(settings, "fixture__pid")[0]) != first_pid  # `int`, or a failure's text would pass too
        assert "restarting" in caplog.text

    def test_a_server_killed_between_calls_is_noticed(self, tmp_path, settings, quick_restarts):
        psutil = pytest.importorskip("psutil")
        config = _write_config(tmp_path / "mcp.json", {"fixture": _stdio_entry()})
        with mcpclient.connected(settings, config_file=config) as client:
            first_pid = int(_call(settings, "fixture__pid")[0])
            psutil.Process(first_pid).kill()
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline and not client._servers[0].deaths:
                time.sleep(0.05)
            assert client._servers[0].deaths, "no restart was noticed without a call failing"
            assert int(_call(settings, "fixture__pid")[0]) != first_pid

    def test_a_restart_leaves_what_is_offered_unchanged(self, tmp_path, settings, quick_restarts):
        config = _write_config(tmp_path / "mcp.json", {"fixture": _stdio_entry()})
        with mcpclient.connected(settings, config_file=config):
            before = json.dumps(settings.request_data["tools"], sort_keys=True)
            _call(settings, "fixture__die")
            assert _call(settings, "fixture__echo", text="back")[0] == "back"
            assert json.dumps(settings.request_data["tools"], sort_keys=True) == before

    def test_the_log_of_the_run_that_died_is_kept(self, tmp_path, settings, quick_restarts):
        config = _write_config(tmp_path / "mcp.json", {"logged": _stdio_entry()})
        with mcpclient.connected(settings, config_file=config):
            _call(settings, "logged__complain", text="famous last words")
            _call(settings, "logged__die")
            assert _call(settings, "logged__echo", text="back")[0] == "back"
        assert "famous last words" in mcpclient._stderr_log_path("logged").read_text(encoding="utf-8")

    def test_a_server_that_keeps_dying_is_left_stopped(self, tmp_path, settings, quick_restarts, monkeypatch, caplog):
        monkeypatch.setattr(mcpclient.librarian_config, "mcp_max_restarts", 1)
        config = _write_config(tmp_path / "mcp.json", {"fixture": _stdio_entry()})
        with mcpclient.connected(settings, config_file=config) as client:
            _call(settings, "fixture__die")
            assert _call(settings, "fixture__echo", text="back")[0] == "back", "the first restart did not happen"
            _call(settings, "fixture__die")
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline and "fixture" not in client.problems:
                time.sleep(0.05)
            assert "left stopped" in client.problems["fixture"]
            assert _call(settings, "fixture__echo", text="anyone?")[0] == "The tool echo is not available right now, because its server fixture is not running."
            assert "fixture__echo" in client.tool_names
        assert "left stopped" in caplog.text

    def test_a_slow_call_does_not_count_as_a_death(self, tmp_path, settings, quick_restarts, monkeypatch):
        monkeypatch.setattr(mcpclient.librarian_config, "mcp_call_timeout", 0.2)
        config = _write_config(tmp_path / "mcp.json", {"fixture": _stdio_entry()})
        with mcpclient.connected(settings, config_file=config) as client:
            assert _call(settings, "fixture__slow", seconds=1.0)[0].startswith("The call to slow failed")
            time.sleep(1.5)
            assert client._servers[0].deaths == []

    def test_a_server_that_never_started_is_not_retried(self, tmp_path, settings, quick_restarts):
        config = _write_config(tmp_path / "mcp.json", {"broken": {"command": "/nonexistent/mcp-server"}})
        with mcpclient.connected(settings, config_file=config) as client:
            time.sleep(0.5)
            assert client._servers[0].deaths == []


class TestLiveReload:
    @pytest.fixture
    def quick_polls(self, monkeypatch):
        monkeypatch.setattr(mcpclient.librarian_config, "mcp_config_poll_interval", 0.05)

    @staticmethod
    def _wait_until(predicate, timeout=15.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.05)
        return False

    @pytest.fixture
    def watching(self, settings, tmp_path, quick_polls):
        def start(servers=None):
            config = tmp_path / "mcp.json"
            if servers is not None:
                _write_config(config, servers)
            client = mcpclient.MCPClient(settings, config_file=config)
            client.start(watch=True)
            started.append(client)
            return client, config
        started = []
        yield start
        for client in started:
            client.shutdown()

    def test_a_server_added_to_the_file_appears(self, watching, settings):
        client, config = watching({"one": _stdio_entry()})
        _write_config(config, {"one": _stdio_entry(), "two": _stdio_entry()})
        assert self._wait_until(lambda: "two__echo" in client.tool_names)
        assert _call(settings, "two__echo", text="new")[0] == "new"
        assert "two__echo" in _offered_names(settings)

    def test_a_server_removed_from_the_file_goes(self, watching, settings):
        client, config = watching({"one": _stdio_entry(), "two": _stdio_entry()})
        _write_config(config, {"one": _stdio_entry()})
        assert self._wait_until(lambda: "two__echo" not in client.tool_names)
        assert "two__echo" not in _offered_names(settings)
        assert "one__echo" in _offered_names(settings)

    def test_an_unchanged_server_keeps_running(self, watching, settings):
        client, config = watching({"one": _stdio_entry(), "two": _stdio_entry()})
        pid = _call(settings, "one__pid")[0]
        _write_config(config, {"one": _stdio_entry()})
        assert self._wait_until(lambda: "two__echo" not in client.tool_names), "the reload did not happen, so this fixture cannot tell a kept server from an untouched one"
        assert _call(settings, "one__pid")[0] == pid

    def test_a_changed_entry_is_restarted_with_its_new_settings(self, watching, settings):
        client, config = watching({"one": _stdio_entry(env={"RAVEN_MCP_TEST": "old"})})
        assert _call(settings, "one__getenv", name="RAVEN_MCP_TEST")[0] == "old"
        _write_config(config, {"one": _stdio_entry(env={"RAVEN_MCP_TEST": "new"})})
        assert self._wait_until(lambda: _call(settings, "one__getenv", name="RAVEN_MCP_TEST")[0] == "new")

    def test_the_file_appearing_starts_servers(self, watching, settings):
        client, config = watching(None)
        assert client.tool_names == frozenset()
        _write_config(config, {"one": _stdio_entry()})
        assert self._wait_until(lambda: "one__echo" in client.tool_names)

    def test_the_file_going_away_stops_them(self, watching, settings):
        client, config = watching({"one": _stdio_entry()})
        config.unlink()
        assert self._wait_until(lambda: client.tool_names == frozenset())

    def test_a_half_written_file_is_ignored(self, watching, settings):
        client, config = watching({"one": _stdio_entry()})
        config.write_text("{not json", encoding="utf-8")
        assert self._wait_until(lambda: "" in client.problems)
        assert _call(settings, "one__echo", text="still here")[0] == "still here"

    def test_on_tools_changed_is_called(self, tmp_path, settings, quick_polls):
        config = _write_config(tmp_path / "mcp.json", {"one": _stdio_entry()})
        calls = []
        client = mcpclient.MCPClient(settings, config_file=config, on_tools_changed=lambda: calls.append(1))
        client.start(watch=True)
        try:
            assert len(calls) == 1
            _write_config(config, {"one": _stdio_entry(), "two": _stdio_entry()})
            assert self._wait_until(lambda: len(calls) == 2)
        finally:
            client.shutdown()

    def test_a_script_does_not_follow_the_file(self, tmp_path, settings, quick_polls):
        config = _write_config(tmp_path / "mcp.json", {"one": _stdio_entry()})
        with mcpclient.connected(settings, config_file=config) as client:
            _write_config(config, {"one": _stdio_entry(), "two": _stdio_entry()})
            time.sleep(0.5)
            assert "two__echo" not in client.tool_names


class TestLifecycle:
    def test_leaving_the_block_removes_the_tools(self, tmp_path, settings):
        before_tools, before_entrypoints = list(settings.tools), dict(settings.tool_entrypoints)
        config = _write_config(tmp_path / "mcp.json", {"fixture": _stdio_entry()})
        with mcpclient.connected(settings, config_file=config):
            assert settings.tools != before_tools
        assert settings.tools == before_tools
        assert settings.request_data["tools"] == before_tools
        assert settings.tool_entrypoints == before_entrypoints
        assert settings.mcp_tool_servers == {}

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

    def test_shutdown_does_not_wait_out_its_allowance_when_the_server_exits_promptly(self, tmp_path, settings):
        config = _write_config(tmp_path / "mcp.json", {"fixture": _stdio_entry()})
        client = mcpclient.MCPClient(settings, config_file=config)
        client.start()
        t0 = time.monotonic()
        client.shutdown(timeout=10.0)
        assert time.monotonic() - t0 < 8.0

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

    def test_shutdown_during_start_ends_the_start_and_offers_nothing(self, tmp_path, settings):
        before = list(settings.tools)
        marker, entry = _mute_entry()
        config = _write_config(tmp_path / "mcp.json", {"mute": entry})
        client = mcpclient.MCPClient(settings, config_file=config)
        thread = threading.Thread(target=client.start, kwargs={"timeout": 60.0}, daemon=True)
        thread.start()
        time.sleep(0.5)
        assert thread.is_alive(), "the start finished by itself, so this fixture cannot tell an interrupted one from a normal one"
        assert _processes_with(marker), "the server is not running, so this fixture cannot tell a stopped one from a missing one"
        client.shutdown()
        thread.join(timeout=10.0)
        assert not thread.is_alive()
        assert settings.tools == before
        assert "no answer" not in client.problems.get("mute", ""), "a start cut short by shutdown is not the server's failure"
        assert not _processes_with(marker)

    def test_a_slow_call_in_flight_cannot_hold_up_shutdown(self, tmp_path, settings):
        config = _write_config(tmp_path / "mcp.json", {"fixture": _stdio_entry()})
        psutil = pytest.importorskip("psutil")
        client = mcpclient.MCPClient(settings, config_file=config)
        client.start()
        pid = int(_call(settings, "fixture__pid")[0])
        slow = settings.tool_entrypoints["fixture__slow"]
        threading.Thread(target=lambda: slow(seconds=60.0), daemon=True).start()
        time.sleep(0.3)
        t0 = time.monotonic()
        client.shutdown()
        assert time.monotonic() - t0 < 10.0
        assert not psutil.pid_exists(pid), "a server stuck in a call was left running"


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

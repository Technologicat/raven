"""Tools from external MCP servers, offered to the LLM beside the built-in ones.

An MCP server is a separate program that publishes tools. This module connects to the servers named in an
`mcp.json` file and registers what they publish in the LLM settings, so that `llmtools.perform_tool_calls`
finds an MCP tool the same way it finds a built-in one.

Whoever wants MCP tools starts them, through one lifecycle: a GUI app, `minichat`, or a script. In a script::

    llm_settings = llmclient.setup(backend_url)
    with mcpclient.connected(llm_settings):
        record = agent.turn(llm_settings, "What's the weather in Jyväskylä?", mcp_enabled=True)

A long-lived frontend calls `MCPClient.start` and `MCPClient.shutdown` instead, since its lifetime does not
fit in a `with` block.

The format of `mcp.json` is the de facto one, so that an entry can be lifted from another MCP client as it
is: see `load_config`.
"""

__all__ = ["CANONICAL_MCP_SERVER_UNAVAILABLE", "CANONICAL_MCP_CALL_FAILED",
           "CANONICAL_MCP_TOOL_ERROR", "CANONICAL_MCP_EMPTY_RESULT",

           "ServerConfig", "load_config",

           "MCPClient", "connected"]

import asyncio
import concurrent.futures
import contextlib
import dataclasses
import hashlib
import json
import logging
import math
import pathlib
import re
import sys
import time
from collections.abc import Callable, Iterator
from typing import Any, TextIO

import anyio
import mcp
from mcp import types as mcp_types
from mcp.shared.exceptions import MCPError
from mcpyrate import colorizer
from unpythonic import dyn
from unpythonic.env import env

from ..common import netutil
from ..common.async_bridge import AsyncBridge
from . import config as librarian_config

logger = logging.getLogger(__name__)

# What the model is told when a call does not produce a result. Statements of fact, like the built-in tools'
# canned replies, and returned rather than raised for the same reason theirs are: the model can act on them.
CANONICAL_MCP_SERVER_UNAVAILABLE = "The tool {tool} is not available right now, because its server {server} is not running."
CANONICAL_MCP_CALL_FAILED = "The call to {tool} failed: {reason}. Its server {server} may have stopped."
CANONICAL_MCP_TOOL_ERROR = "The tool {tool} reported an error: {text}"
CANONICAL_MCP_EMPTY_RESULT = "The tool {tool} finished and returned no output."


# --------------------------------------------------------------------------------
# Configuration

@dataclasses.dataclass(frozen=True)
class ServerConfig:
    """One entry of `mcp.json`. See `load_config`."""
    name: str
    transport: str  # "stdio", "http" or "sse"
    command: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] | None = None
    cwd: str | None = None
    url: str | None = None
    headers: dict[str, str] | None = None
    allowed_tools: frozenset[str] | None = None

# The spellings other MCP clients use for a transport, compared with case and separators removed.
_TRANSPORT_SPELLINGS = {"stdio": "stdio",
                        "http": "http",
                        "streamablehttp": "http",
                        "sse": "sse"}

def _parse_server(name: str, entry: Any) -> ServerConfig:
    """Validate one `mcpServers` entry. Raises `ValueError` saying what is wrong with it."""
    if not isinstance(entry, dict):
        raise ValueError("the entry is not a JSON object")

    if "type" in entry:
        spelled = str(entry["type"])
        try:
            transport = _TRANSPORT_SPELLINGS[re.sub(r"[-_]", "", spelled).lower()]
        except KeyError:
            raise ValueError(f"unknown transport type '{spelled}'; known types are stdio, http, streamable-http and sse") from None
    elif "command" in entry:
        transport = "stdio"
    elif "url" in entry:
        transport = "http"
    else:
        raise ValueError("the entry has neither a 'command' nor a 'url'")

    if transport == "stdio" and not entry.get("command"):
        raise ValueError("a stdio server needs a 'command'")
    if transport != "stdio" and not entry.get("url"):
        raise ValueError(f"an {transport} server needs a 'url'")

    allowed_tools = entry.get("allowed_tools")
    if allowed_tools is not None and not (isinstance(allowed_tools, list) and all(isinstance(tool, str) for tool in allowed_tools)):
        raise ValueError("'allowed_tools' must be a list of tool names")

    return ServerConfig(name=name,
                        transport=transport,
                        command=entry.get("command"),
                        args=tuple(str(arg) for arg in entry.get("args", ())),
                        env=entry.get("env"),
                        cwd=entry.get("cwd"),
                        url=entry.get("url"),
                        headers=entry.get("headers"),
                        allowed_tools=None if allowed_tools is None else frozenset(allowed_tools))

def load_config(path: str | pathlib.Path) -> tuple[list[ServerConfig], dict[str, str]]:
    """Read the MCP servers configured in the file at `path`.

    The file holds a JSON object whose `mcpServers` key maps a server's name to its settings:

    `command`, `args`, `env`, `cwd`: start the server as a subprocess, and talk to it over its stdin and stdout.

    `url`, `headers`: talk to a server that is already running, over streamable HTTP. A per-service API key
                      is an entry in `headers`.

    `type`: the transport, when it cannot be told from the above: `"stdio"`, `"http"` or `"sse"`. The other
            spellings in use elsewhere for streamable HTTP are accepted too.

    `disabled`: `true` skips the server, without removing its entry.

    `allowed_tools`: a list of tool names; the server's other tools are not offered to the LLM.

    Returns `(servers, problems)`. `servers` holds the entries that are valid and not disabled, in file order.
    `problems` maps a server's name to what is wrong with its entry; a problem with the file as a whole is
    under the key `""`.

    A missing file means no servers and no problems. Nothing here raises for a bad file: the app that reads
    it is useful without its MCP servers.
    """
    path = pathlib.Path(path).expanduser()
    if not path.exists():
        return [], {}

    try:
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, ValueError) as exc:
        return [], {"": f"could not read '{path}': {type(exc)}: {exc}"}
    if not isinstance(data, dict) or not isinstance(data.get("mcpServers", {}), dict):
        return [], {"": f"'{path}' should hold a JSON object with an 'mcpServers' object in it"}

    servers, problems = [], {}
    for name, entry in data.get("mcpServers", {}).items():
        if isinstance(entry, dict) and entry.get("disabled", False):
            logger.info(f"load_config: server '{name}' is disabled; skipping it.")
            continue
        try:
            servers.append(_parse_server(name, entry))
        except ValueError as exc:
            problems[name] = str(exc)
    return servers, problems


# --------------------------------------------------------------------------------
# From MCP's shapes to Raven's

# The limits OpenAI's API puts on a function name, which local backends' parsers tend to assume as well.
_MAX_TOOL_NAME_LENGTH = 64

def _public_name(server_name: str, tool_name: str) -> str:
    """The name the LLM knows a tool by: `<server>__<tool>`, which keeps two servers' tools apart.

    Deterministic, since this name goes into the prompt prefix and into stored chats.
    """
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", f"{server_name}__{tool_name}")
    if len(name) > _MAX_TOOL_NAME_LENGTH:
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
        name = f"{name[:_MAX_TOOL_NAME_LENGTH - len(digest) - 1]}_{digest}"
    return name

def _tool_spec(public_name: str, tool: mcp_types.Tool) -> dict[str, Any]:
    """Translate an MCP tool listing into an OpenAI function specification, as in `llmtools.TOOLS`."""
    return {"type": "function",
            "function": {"name": public_name,
                         "description": tool.description or tool.title or "",
                         "parameters": tool.input_schema}}

def _describe_block(block: Any) -> str:
    """The text of one content block of a tool's result."""
    if isinstance(block, mcp_types.TextContent):
        return block.text
    if isinstance(block, mcp_types.EmbeddedResource) and isinstance(block.resource, mcp_types.TextResourceContents):
        return block.resource.text
    if isinstance(block, mcp_types.ResourceLink):
        return f"[A resource: {block.uri}]"
    return f"[Output of type '{block.type}', which cannot be shown here.]"

def _result_text(result: mcp_types.CallToolResult) -> str:
    """The text of a tool's result, or `""` if it has none."""
    text = "\n\n".join(_describe_block(block) for block in result.content)
    if not text and result.structured_content is not None:
        text = json.dumps(result.structured_content, ensure_ascii=False)
    return text

def _describe_exception(exc: BaseException) -> str:
    """A readable account of `exc`. The SDK's task groups wrap the actual failure in layers of `ExceptionGroup`."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return f"{type(exc)}: {exc}"

def _stderr_log_path(server_name: str) -> pathlib.Path:
    """Where the stderr of the stdio server called `server_name` goes."""
    return pathlib.Path(librarian_config.mcp_log_dir).expanduser() / f"{re.sub(r'[^a-zA-Z0-9_-]', '_', server_name)}.log"

@contextlib.contextmanager
def _stderr_log(config: ServerConfig, *, append: bool) -> Iterator[TextIO | None]:
    """Open the file receiving a stdio server's stderr, which would otherwise interleave with a terminal app's output.

    `append`: keep what is in the file. A restarted server's log is how to find out why it died.

    Yields `None` for a server with no stderr of ours to capture, and `sys.stderr` if the file cannot be opened.
    """
    if config.transport != "stdio":
        yield None
        return
    path = _stderr_log_path(config.name)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        file = open(path, "a" if append else "w", encoding="utf-8")  # noqa: SIM115 -- closed by the `with` below, which must not span the handler
    except OSError as exc:
        logger.warning(f"MCPClient: server '{config.name}': cannot write '{path}', so its output goes to stderr. Reason {type(exc)}: {exc}")
        yield sys.stderr
        return
    with file:
        yield file

def _open_session(config: ServerConfig, errlog: TextIO | None) -> mcp.Client:
    """An unopened SDK client for the server `config` describes. `errlog` receives a stdio server's stderr."""
    if config.transport == "stdio":
        from mcp.client.stdio import stdio_client
        return mcp.Client(stdio_client(mcp.StdioServerParameters(command=config.command,
                                                                 args=list(config.args),
                                                                 env=config.env,
                                                                 cwd=config.cwd),
                                       errlog=errlog))
    if config.transport == "sse":
        from mcp.client.sse import sse_client
        return mcp.Client(sse_client(config.url, headers=config.headers))
    if config.headers:
        from mcp.client.streamable_http import streamable_http_client
        from mcp.shared._httpx_utils import create_mcp_http_client  # the transports' own timeouts
        return mcp.Client(streamable_http_client(config.url, http_client=create_mcp_http_client(headers=config.headers)))
    return mcp.Client(config.url)


# --------------------------------------------------------------------------------
# The client

# A server that died gets a moment before it is started again, and a caller meanwhile waits this long for it.
_RESTART_WAIT = 15.0

def _is_connection_failure(exc: BaseException) -> bool:
    """Whether `exc` says the connection to a server is gone, as opposed to a server answering badly or slowly."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    if isinstance(exc, MCPError):
        return exc.code == mcp_types.CONNECTION_CLOSED
    return not isinstance(exc, TimeoutError)

class _Server:
    """One configured server, and what became of it."""
    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.maybe_session: mcp.Client | None = None
        self.tools: dict[str, mcp_types.Tool] = {}  # by public name
        self.maybe_failure: str | None = None
        self.maybe_task: asyncio.Task | None = None
        self.maybe_scope: anyio.CancelScope | None = None
        self.connection_lost = asyncio.Event()
        self.connected_this_attempt = False
        self.restarting = False
        self.stopping = False
        self.left_stopped = False
        self.deaths: list[float] = []  # `time.monotonic` of each failed run; see `MCPClient._run_server`

class MCPClient:
    """Connections to the configured MCP servers, and the tools they publish.

    A server that dies is started again, as long as it does not keep dying; see `start`.

    `settings`: from `llmclient.setup` or `llmclient.connect`. `start` adds the tools to it, and `shutdown`
                removes them.

    `config_file`: the `mcp.json` to read. Default is `librarian_config.mcp_config_file`.

    `on_status_changed`: 0-argument callable, called on a background thread when `tool_names`, `problems` or
                         `stopped_servers` may have changed.

    `on_reloaded`: 0-argument callable, called on a background thread after following a change to the config file.
    """
    def __init__(self,
                 settings: env,
                 *,
                 config_file: str | pathlib.Path | None = None,
                 on_status_changed: Callable[[], None] | None = None,
                 on_reloaded: Callable[[], None] | None = None) -> None:
        self.settings = settings
        self.config_file = pathlib.Path(config_file if config_file is not None else librarian_config.mcp_config_file)
        self.on_status_changed = on_status_changed
        self.on_reloaded = on_reloaded
        self._maybe_bridge: AsyncBridge | None = None
        self._maybe_watcher: asyncio.Task | None = None
        self._servers: list[_Server] = []
        self._problems: dict[str, str] = {}
        self._entrypoints: dict[str, Callable] = {}

    def _get_tool_names(self) -> frozenset[str]:
        """The names of the MCP tools now offered."""
        return frozenset(self._entrypoints)
    tool_names = property(fget=_get_tool_names, doc="The names the LLM knows the installed MCP tools by.")

    def _get_problems(self) -> dict[str, str]:
        """Copy, so that a caller cannot edit the client's record."""
        return dict(self._problems)
    problems = property(fget=_get_problems, doc="What went wrong, by server name: a bad entry, or a server that did not start or keeps dying. "
                                                "A problem with the file as a whole is under the key `\"\"`.")

    def _get_stopped_servers(self) -> list[str]:
        """The servers that failed too often, in configuration order."""
        return [server.config.name for server in self._servers if server.left_stopped]
    stopped_servers = property(fget=_get_stopped_servers, doc="The names of the servers left stopped after failing too often. `retry_stopped` tries them again.")

    def _get_all_running(self) -> bool:
        """Whether every server has an open session; `False` if there are no servers."""
        return bool(self._servers) and all(server.maybe_session is not None for server in self._servers)
    all_running = property(fget=_get_all_running, doc="Whether there are servers, and every one of them is connected right now.")

    def report_to_console(self) -> None:
        """Print how the start went, for a frontend with a terminal. Prints nothing if no servers are configured."""
        if self.tool_names:
            plural_s = "s" if len(self.tool_names) != 1 else ""
            print(colorizer.colorize(f"MCP: {len(self.tool_names)} tool{plural_s} available.", colorizer.Style.BRIGHT, colorizer.Fore.GREEN))
        for name, problem in self.problems.items():
            print(colorizer.colorize(f"MCP: server '{name}': {problem}" if name else f"MCP: {problem}", colorizer.Style.BRIGHT, colorizer.Fore.RED))

    # ----------------------------------------
    # Lifecycle

    def start(self, *, timeout: float | None = None, watch: bool = False) -> None:
        """Connect to the configured servers, and add their tools to the settings.

        Blocks until every server has connected or failed; a GUI app calls this from a background task. A
        server that fails is logged and recorded in `problems`, and the others carry on.

        A server that dies or cannot be started is tried again and the error logged. Past
        `librarian_config.mcp_max_restarts` retries it is left stopped for the rest of the session, and its
        tools answer that it is not running. For a server that had connected, only deaths within
        `librarian_config.mcp_restart_window` seconds count.

        `timeout`: seconds to allow each server to connect. Default is `librarian_config.mcp_connect_timeout`.

        `watch`: follow changes to the config file, stopping and starting servers to match. For a long-lived
                 frontend. A script leaves this off, so that what it offers cannot change during a run.
        """
        if self._maybe_bridge is not None:
            raise RuntimeError("MCPClient.start: already started")
        timeout = timeout if timeout is not None else librarian_config.mcp_connect_timeout

        signature = self._config_signature()  # before reading, so that an edit right after the read is seen
        configs, self._problems = load_config(self.config_file)
        for name, problem in self._problems.items():
            logger.error(f"MCPClient.start: server '{name}': {problem}")
        if not configs:
            logger.info(f"MCPClient.start: no MCP servers configured in '{self.config_file}'.")
            if not watch:
                return

        self._servers = [_Server(config) for config in configs]
        self._maybe_bridge = AsyncBridge(name="mcp_client")
        try:
            self._maybe_bridge.submit(self._begin(timeout, signature if watch else None))
        except (concurrent.futures.CancelledError, RuntimeError):  # `shutdown` got here first
            logger.info("MCPClient.start: interrupted by shutdown.")

    async def _begin(self, timeout: float, maybe_signature: bytes | None) -> None:
        await self._connect_servers(self._servers, timeout)
        self._refresh_tools()
        if maybe_signature is not None:
            self._maybe_watcher = asyncio.get_running_loop().create_task(self._watch_config(maybe_signature))

    async def _connect_servers(self, servers: list[_Server], timeout: float) -> None:
        await asyncio.gather(*(self._connect(server, timeout) for server in servers))

    async def _connect(self, server: _Server, timeout: float) -> None:
        loop = asyncio.get_running_loop()
        ready = loop.create_future()
        server.maybe_task = loop.create_task(self._run_server(server, timeout, ready))
        await ready

    async def _run_server(self, server: _Server, timeout: float, ready: asyncio.Future) -> None:
        """Keep the server's session open until told to stop, starting the server again if it dies."""
        name = server.config.name
        try:
            while await self._hold_session(server, timeout, ready):
                if not ready.done():  # `start` waits for the first attempt only
                    ready.set_result(None)
                now = time.monotonic()
                # Attempts that never connected all count, however far apart: one that takes the whole connect
                # timeout to fail would otherwise never reach the limit inside the window.
                if server.connected_this_attempt:
                    server.deaths = [t for t in server.deaths if now - t < librarian_config.mcp_restart_window]
                server.deaths.append(now)
                n_deaths = len(server.deaths)
                if n_deaths > librarian_config.mcp_max_restarts:
                    plural_s = "s" if n_deaths != 1 else ""
                    server.restarting = False
                    server.left_stopped = True
                    server.maybe_failure = f"failed {n_deaths} time{plural_s}, so it was left stopped for the rest of this session"
                    self._problems[name] = server.maybe_failure
                    logger.error(f"MCPClient: server '{name}': {server.maybe_failure}.")
                    self._call_handler(self.on_status_changed)
                    return
                server.restarting = True
                happened = "the connection was lost" if server.connected_this_attempt else "it could not be started"
                logger.error(f"MCPClient: server '{name}': {happened}; trying again ({n_deaths} of at most {librarian_config.mcp_max_restarts}).")
                await asyncio.sleep(librarian_config.mcp_restart_delay)
        finally:
            server.restarting = False
            if not ready.done():
                ready.set_result(None)

    async def _hold_session(self, server: _Server, timeout: float, ready: asyncio.Future) -> bool:
        """Open the server's session, and keep it open until told to stop or the connection is lost.

        One long-lived task per server, because the SDK's sessions are anyio cancel scopes, which must be
        exited by the task that entered them. Calls from other tasks are fine.

        Stopped by cancelling this scope, which ends the wait at once. The SDK shields its bounded teardown of
        a server from anyio's cancellation; its source warns that a native `Task.cancel` can defeat that.

        Returns whether to try again: that is, unless told to stop.
        """
        name = server.config.name
        server.connection_lost = asyncio.Event()
        server.connected_this_attempt = False
        server.maybe_failure = None
        try:
            with anyio.CancelScope(deadline=anyio.current_time() + timeout) as scope:
                server.maybe_scope = scope
                with _stderr_log(server.config, append=bool(server.deaths)) as errlog:
                    async with _open_session(server.config, errlog) as session:
                        listed = await self._list_tools(session)
                        server.tools = self._allowed(server, listed)
                        server.maybe_session = session
                        scope.deadline = math.inf  # the deadline was the connection's allowance, and this is connected
                        plural_s = "s" if len(server.tools) != 1 else ""
                        logger.info(f"MCPClient: server '{name}': connected, {len(server.tools)} tool{plural_s}.")
                        server.connected_this_attempt = True
                        self._problems.pop(name, None)
                        if ready.done():  # this is a retry, as `start` is no longer waiting
                            server.restarting = False
                            self._refresh_tools()
                        if not ready.done():
                            ready.set_result(None)
                        await self._watch_connection(server, session)
            if scope.cancelled_caught and not server.connected_this_attempt and not server.stopping:
                server.maybe_failure = f"no answer in {timeout:g} seconds"
                logger.error(f"MCPClient: server '{name}': {server.maybe_failure}")
        except Exception as exc:  # noqa: BLE001 -- one server's failure stops that server, and not the app
            server.maybe_failure = _describe_exception(exc)
            if server.config.transport == "stdio":
                server.maybe_failure += f" (the server's own output is in '{_stderr_log_path(name)}')"
            logger.error(f"MCPClient: server '{name}': {server.maybe_failure}")
        finally:
            server.maybe_session = None
        if server.stopping:
            return False
        if server.maybe_failure is not None:
            self._problems[name] = server.maybe_failure
        return True

    @staticmethod
    async def _watch_connection(server: _Server, session: mcp.Client) -> None:
        """Return when the connection to the server has gone.

        A failed tool call reports that at once. Asking for the tool list finds a server that dies between
        calls; it is the cheapest request every server answers, the protocol's `ping` having been removed as of
        its 2026-07-28 revision. A question that goes unanswered proves nothing, as a server may be busy.
        """
        interval = librarian_config.mcp_liveness_interval
        while True:
            with anyio.move_on_after(interval):
                await server.connection_lost.wait()
                return
            try:
                with anyio.move_on_after(interval):
                    await session.list_tools()
            except Exception as exc:  # noqa: BLE001 -- inspected right below
                if _is_connection_failure(exc):
                    return

    @staticmethod
    async def _list_tools(session: mcp.Client) -> list[mcp_types.Tool]:
        """Every tool the server publishes, across pages, sorted by name.

        Sorted so that the prompt prefix does not depend on the order a server happens to list them in.
        """
        tools: list[mcp_types.Tool] = []
        cursor = None
        while True:
            page = await session.list_tools(cursor=cursor)
            tools.extend(page.tools)
            cursor = page.next_cursor
            if cursor is None:
                return sorted(tools, key=lambda tool: tool.name)

    def _allowed(self, server: _Server, listed: list[mcp_types.Tool]) -> dict[str, mcp_types.Tool]:
        """Those of the `listed` tools that the server's configuration allows, by public name."""
        allowed_tools = server.config.allowed_tools
        if allowed_tools is not None:
            unknown = allowed_tools - {tool.name for tool in listed}
            if unknown:  # a typo here would silently switch a tool off, so say so
                logger.warning(f"MCPClient: server '{server.config.name}': 'allowed_tools' names {sorted(unknown)}, which it does not publish. "
                               f"It publishes {sorted(tool.name for tool in listed)}.")
        return {_public_name(server.config.name, tool.name): tool
                for tool in listed
                if allowed_tools is None or tool.name in allowed_tools}

    def _refresh_tools(self) -> None:
        """Rebuild the tool tables from what the servers publish now, and put them in the settings.

        Runs on the bridge's thread, or before it has anything else to do, so these never overlap.
        """
        specs: list[dict[str, Any]] = []
        entrypoints: dict[str, Callable] = {}
        tool_servers: dict[str, str] = {}
        taken = set(self.settings.tool_entrypoints) - set(self._entrypoints)
        for server in self._servers:
            for public_name, tool in server.tools.items():
                if public_name in entrypoints or public_name in taken:
                    logger.warning(f"MCPClient: server '{server.config.name}': the name '{public_name}' is already taken; not offering this tool.")
                    continue
                specs.append(_tool_spec(public_name, tool))
                entrypoints[public_name] = self._make_entrypoint(server, tool.name)
                tool_servers[public_name] = server.config.name
        self._install(specs, entrypoints, tool_servers)
        self._call_handler(self.on_status_changed)

    @staticmethod
    def _call_handler(maybe_handler: Callable[[], None] | None) -> None:
        if maybe_handler is None:
            return
        try:
            maybe_handler()
        except Exception as exc:  # noqa: BLE001 -- a frontend's handler must not stop the servers
            logger.warning(f"MCPClient: ignoring exception from an event handler: {_describe_exception(exc)}")

    def _install(self, specs: list[dict[str, Any]], entrypoints: dict[str, Callable], tool_servers: dict[str, str]) -> None:
        """Replace the MCP tools in the settings with these.

        New objects throughout, as `tools` and `tool_entrypoints` start out as `llmtools`' module-level
        tables, which other settings in this process share.
        """
        settings = self.settings
        previous = self._entrypoints
        # Gated first and ungated last, for a turn starting meanwhile: a name in `mcp_tool_servers` that is
        # not registered yet is harmless, whereas one registered and not yet gated would be offered with the
        # toggle off.
        settings.mcp_tool_servers = {**settings.mcp_tool_servers, **tool_servers}
        settings.tool_entrypoints = {**{name: function for name, function in settings.tool_entrypoints.items() if name not in previous}, **entrypoints}
        all_specs = [*(spec for spec in settings.tools if spec["function"]["name"] not in previous), *specs]
        settings.tools = all_specs
        settings.request_data["tools"] = all_specs
        settings.mcp_tool_servers = tool_servers
        self._entrypoints = entrypoints

    def shutdown(self, *, timeout: float | None = None) -> None:
        """Remove the tools from the settings and close the connections. Safe to call more than once.

        A server that does not stop in time is abandoned, so a misbehaving one cannot keep the app from exiting.

        `timeout`: seconds to allow. Default is `librarian_config.mcp_shutdown_timeout`.
        """
        timeout = timeout if timeout is not None else librarian_config.mcp_shutdown_timeout
        bridge, self._maybe_bridge = self._maybe_bridge, None
        if bridge is not None:
            try:
                bridge.submit(self._stop_all(timeout), timeout=timeout + 1.0)
            except Exception as exc:  # noqa: BLE001 -- at exit, there is nothing more to do about it
                logger.warning(f"MCPClient.shutdown: {_describe_exception(exc)}")
            bridge.shutdown(timeout=timeout)
            logger.info("MCPClient.shutdown: done.")
        self._install([], {}, {})

    async def _stop_all(self, timeout: float) -> None:
        if self._maybe_watcher is not None:
            self._maybe_watcher.cancel()
        await self._stop_servers(self._servers, timeout)

    @staticmethod
    async def _stop_servers(servers: list[_Server], timeout: float) -> None:
        tasks = []
        for server in servers:
            server.stopping = True
            if server.maybe_scope is not None:
                server.maybe_scope.cancel()
            if server.maybe_task is not None:
                tasks.append(server.maybe_task)
        if tasks:
            await asyncio.wait(tasks, timeout=timeout)

    # ----------------------------------------
    # Following the config file

    def _config_signature(self) -> bytes:
        """The content of the config file; empty if it does not exist."""
        try:
            return self.config_file.expanduser().read_bytes()
        except OSError:
            return b""

    async def _watch_config(self, signature: bytes) -> None:
        while True:
            await asyncio.sleep(librarian_config.mcp_config_poll_interval)
            current = self._config_signature()
            if current != signature:
                signature = current
                try:
                    logger.info(f"MCPClient: '{self.config_file}' changed; reloading the MCP servers.")
                    if await self._sync_with_config():
                        self._call_handler(self.on_reloaded)
                except Exception:  # noqa: BLE001 -- the next change to the file gets another try
                    logger.exception("MCPClient: reloading the MCP servers failed")

    def retry_stopped(self) -> None:
        """Try the servers that were left stopped again, in the background, with a fresh allowance of retries.

        Does nothing before `start` or after `shutdown`. `on_status_changed` reports the outcome.
        """
        async def retry() -> None:
            try:
                logger.info("MCPClient.retry_stopped: trying the stopped MCP servers again.")
                await self._sync_with_config()
            except Exception:  # noqa: BLE001 -- runs detached, so nothing else would hear of it
                logger.exception("MCPClient.retry_stopped: failed")
        bridge = self._maybe_bridge
        if bridge is not None:
            with contextlib.suppress(RuntimeError):  # the bridge has been shut down, and the question with it
                bridge.spawn(retry())

    async def _sync_with_config(self) -> bool:
        """Bring the servers in line with the config file. Returns whether the file could be used.

        A server whose entry did not change, and which is running, is left alone. An unreadable file is
        reported and otherwise ignored: an editor's half-finished save should not stop everything.
        """
        configs, problems = load_config(self.config_file)
        if "" in problems:
            self._problems[""] = problems[""]
            logger.error(f"MCPClient: the config file cannot be used; keeping the servers as they are. {problems['']}")
            self._call_handler(self.on_status_changed)
            return False
        for name, problem in problems.items():
            logger.error(f"MCPClient: server '{name}': {problem}")

        wanted = {config.name: config for config in configs}
        kept = [server for server in self._servers
                if wanted.get(server.config.name) == server.config and server.maybe_session is not None]
        kept_names = {server.config.name for server in kept}
        await self._stop_servers([server for server in self._servers if server.config.name not in kept_names],
                                 librarian_config.mcp_shutdown_timeout)
        started = [_Server(config) for name, config in wanted.items() if name not in kept_names]
        self._servers = kept + started
        self._problems = problems
        self._call_handler(self.on_status_changed)
        await self._connect_servers(started, librarian_config.mcp_connect_timeout)
        self._refresh_tools()
        return True

    # ----------------------------------------
    # Calling a tool

    def _make_entrypoint(self, server: _Server, tool_name: str) -> Callable:
        def entrypoint(**arguments: Any) -> tuple[str, dict[str, Any]]:
            return self._call(server, tool_name, arguments)
        return entrypoint

    def _call(self, server: _Server, tool_name: str, arguments: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Call one tool, and return its result in the form `perform_tool_calls` takes."""
        name = server.config.name
        metadata: dict[str, Any] = {"mcp_server": name}
        maybe_abort = getattr(dyn.tool_context, "maybe_abort", None)

        deadline = time.monotonic() + _RESTART_WAIT
        while server.restarting and server.maybe_session is None and time.monotonic() < deadline:
            if maybe_abort is not None and maybe_abort.aborted:
                raise netutil.Aborted
            time.sleep(0.1)

        bridge = self._maybe_bridge
        maybe_session = server.maybe_session
        if bridge is None or maybe_session is None:
            return CANONICAL_MCP_SERVER_UNAVAILABLE.format(tool=tool_name, server=name), metadata

        future = bridge.spawn(maybe_session.call_tool(tool_name, arguments,
                                                      read_timeout_seconds=librarian_config.mcp_call_timeout))
        # Polled, as there is no single wait for both the call finishing and the handle firing.
        while not concurrent.futures.wait([future], timeout=0.1).done:
            if maybe_abort is not None and maybe_abort.aborted:
                future.cancel()
                raise netutil.Aborted

        try:
            result = future.result()
        except concurrent.futures.CancelledError:
            raise netutil.Aborted from None
        except Exception as exc:  # noqa: BLE001 -- reported to the model, which can tell the user
            reason = _describe_exception(exc)
            logger.warning(f"MCPClient: server '{name}': tool '{tool_name}': {reason}")
            if _is_connection_failure(exc):
                # Right here, so that a call right after this one waits for the restart instead of using
                # this session. Only if the session is still this one, as a restart may have finished by now.
                if server.maybe_session is maybe_session:
                    server.restarting = True
                    server.maybe_session = None
                async def report_lost() -> None:
                    server.connection_lost.set()
                with contextlib.suppress(RuntimeError):  # the bridge has been shut down, and the question with it
                    bridge.spawn(report_lost())
            return CANONICAL_MCP_CALL_FAILED.format(tool=tool_name, server=name, reason=reason), metadata

        text = _result_text(result)
        if result.is_error:
            metadata["mcp_error"] = True
            return CANONICAL_MCP_TOOL_ERROR.format(tool=tool_name, text=text), metadata
        if not text:
            return CANONICAL_MCP_EMPTY_RESULT.format(tool=tool_name), metadata

        # A long result goes to an attachment, as a fetched web page does. See `scaffold._attachmentify_tool_result`.
        metadata["fetched_document"] = {"url": f"mcp://{name}/{tool_name}",
                                        "name": tool_name,
                                        "source": f"mcp:{name}"}
        return text, metadata


@contextlib.contextmanager
def connected(settings: env, *, config_file: str | pathlib.Path | None = None) -> Iterator[MCPClient]:
    """Have the configured MCP servers' tools in `settings` for the duration of a `with` block.

    Arguments as for `MCPClient`. Yields the client, whose `problems` say which servers failed to start.
    """
    client = MCPClient(settings, config_file=config_file)
    try:
        client.start()
        yield client
    finally:
        client.shutdown()

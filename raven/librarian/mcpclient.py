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
import pathlib
import re
from collections.abc import Callable, Iterator
from typing import Any

import mcp
import mcp_types
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

def _open_session(config: ServerConfig) -> mcp.Client:
    """An unopened SDK client for the server `config` describes."""
    if config.transport == "stdio":
        return mcp.Client(mcp.StdioServerParameters(command=config.command,
                                                    args=list(config.args),
                                                    env=config.env,
                                                    cwd=config.cwd))
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

class _Server:
    """One configured server, and what became of it."""
    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.maybe_session: mcp.Client | None = None
        self.tools: dict[str, mcp_types.Tool] = {}  # by public name
        self.maybe_failure: str | None = None
        self.maybe_task: asyncio.Task | None = None
        self.stop: asyncio.Event | None = None  # created on the bridge's loop

class MCPClient:
    """Connections to the configured MCP servers, and the tools they publish.

    `settings`: from `llmclient.setup` or `llmclient.connect`. `start` adds the tools to it, and `shutdown`
                removes them.

    `config_file`: the `mcp.json` to read. Default is `librarian_config.mcp_config_file`.
    """
    def __init__(self, settings: env, *, config_file: str | pathlib.Path | None = None) -> None:
        self.settings = settings
        self.config_file = pathlib.Path(config_file if config_file is not None else librarian_config.mcp_config_file)
        self._maybe_bridge: AsyncBridge | None = None
        self._servers: list[_Server] = []
        self._problems: dict[str, str] = {}
        self._specs: list[dict[str, Any]] = []
        self._entrypoints: dict[str, Callable] = {}

    def _get_tool_names(self) -> frozenset[str]:
        """The names of the MCP tools now offered."""
        return frozenset(self._entrypoints)
    tool_names = property(fget=_get_tool_names, doc="The names the LLM knows the installed MCP tools by.")

    def _get_problems(self) -> dict[str, str]:
        """Copy, so that a caller cannot edit the client's record."""
        return dict(self._problems)
    problems = property(fget=_get_problems, doc="What went wrong, by server name: a bad entry, or a server that did not start. "
                                                "A problem with the file as a whole is under the key `\"\"`.")

    # ----------------------------------------
    # Lifecycle

    def start(self, *, timeout: float | None = None) -> None:
        """Connect to the configured servers, and add their tools to the settings.

        Blocks until every server has connected or failed; a GUI app calls this from a background task. A
        server that fails is logged and recorded in `problems`, and the others carry on.

        `timeout`: seconds to allow each server to connect. Default is `librarian_config.mcp_connect_timeout`.
        """
        if self._maybe_bridge is not None:
            raise RuntimeError("MCPClient.start: already started")
        timeout = timeout if timeout is not None else librarian_config.mcp_connect_timeout

        configs, self._problems = load_config(self.config_file)
        for name, problem in self._problems.items():
            logger.error(f"MCPClient.start: server '{name}': {problem}")
        if not configs:
            logger.info(f"MCPClient.start: no MCP servers configured in '{self.config_file}'.")
            return

        self._servers = [_Server(config) for config in configs]
        self._maybe_bridge = AsyncBridge(name="mcp_client")
        self._maybe_bridge.submit(self._connect_all(timeout))
        self._register_tools()

    async def _connect_all(self, timeout: float) -> None:
        await asyncio.gather(*(self._connect(server, timeout) for server in self._servers))

    async def _connect(self, server: _Server, timeout: float) -> None:
        loop = asyncio.get_running_loop()
        server.stop = asyncio.Event()
        ready = loop.create_future()
        server.maybe_task = loop.create_task(self._hold_session(server, ready))
        try:
            await asyncio.wait_for(asyncio.shield(ready), timeout)
        except TimeoutError:
            server.maybe_failure = f"no answer in {timeout:0.0f} seconds"
            logger.error(f"MCPClient: server '{server.config.name}': {server.maybe_failure}")
            server.maybe_task.cancel()

    async def _hold_session(self, server: _Server, ready: asyncio.Future) -> None:
        """Open the server's session, and keep it open until told to stop.

        One long-lived task per server, because the SDK's sessions are anyio cancel scopes, which must be
        exited by the task that entered them. Calls from other tasks are fine.
        """
        name = server.config.name
        try:
            async with _open_session(server.config) as session:
                listed = await self._list_tools(session)
                server.tools = self._allowed(server, listed)
                server.maybe_session = session
                plural_s = "s" if len(server.tools) != 1 else ""
                logger.info(f"MCPClient: server '{name}': connected, {len(server.tools)} tool{plural_s}.")
                if not ready.done():
                    ready.set_result(None)
                await server.stop.wait()
        except Exception as exc:  # noqa: BLE001 -- one server's failure stops that server, and not the app
            server.maybe_failure = _describe_exception(exc)
            logger.error(f"MCPClient: server '{name}': {server.maybe_failure}")
        finally:
            server.maybe_session = None
            if not ready.done():
                ready.set_result(None)

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

    def _register_tools(self) -> None:
        """Build the tool tables from what the servers published, and add them to the settings."""
        for server in self._servers:
            if server.maybe_failure is not None:
                self._problems[server.config.name] = server.maybe_failure
            for public_name, tool in server.tools.items():
                if public_name in self._entrypoints or public_name in self.settings.tool_entrypoints:
                    logger.warning(f"MCPClient: server '{server.config.name}': the name '{public_name}' is already taken; not offering this tool.")
                    continue
                self._specs.append(_tool_spec(public_name, tool))
                self._entrypoints[public_name] = self._make_entrypoint(server, tool.name)
        self._install()

    def _install(self) -> None:
        settings = self.settings
        # In this order, for a turn starting during `start`: a name in `mcp_tool_names` that is not yet
        # registered is harmless, whereas one registered and not yet gated would be offered with the
        # toggle off.
        #
        # New objects throughout, as `tools` and `tool_entrypoints` start out as `llmtools`' module-level
        # tables and other settings in this process share them.
        settings.mcp_tool_names = frozenset(self._entrypoints)
        settings.tool_entrypoints = {**settings.tool_entrypoints, **self._entrypoints}
        specs = [*settings.tools, *self._specs]
        settings.tools = specs
        settings.request_data["tools"] = specs

    def _uninstall(self) -> None:
        settings = self.settings
        def ours(spec: dict[str, Any]) -> bool:
            return spec["function"]["name"] in self._entrypoints
        specs = [spec for spec in settings.tools if not ours(spec)]
        settings.tools = specs
        settings.request_data["tools"] = specs
        settings.tool_entrypoints = {name: function for name, function in settings.tool_entrypoints.items()
                                     if name not in self._entrypoints}
        settings.mcp_tool_names = frozenset()

    def shutdown(self, *, timeout: float | None = None) -> None:
        """Remove the tools from the settings and close the connections. Safe to call more than once.

        A server that does not stop in time is abandoned, so a misbehaving one cannot keep the app from exiting.

        `timeout`: seconds to allow. Default is `librarian_config.mcp_shutdown_timeout`.
        """
        timeout = timeout if timeout is not None else librarian_config.mcp_shutdown_timeout
        self._uninstall()
        self._specs, self._entrypoints = [], {}
        bridge, self._maybe_bridge = self._maybe_bridge, None
        if bridge is None:
            return
        try:
            bridge.submit(self._stop_all(timeout), timeout=timeout + 1.0)
        except Exception as exc:  # noqa: BLE001 -- at exit, there is nothing more to do about it
            logger.warning(f"MCPClient.shutdown: {_describe_exception(exc)}")
        bridge.shutdown(timeout=timeout)
        logger.info("MCPClient.shutdown: done.")

    async def _stop_all(self, timeout: float) -> None:
        tasks = []
        for server in self._servers:
            if server.stop is not None:
                server.stop.set()
            if server.maybe_task is not None:
                tasks.append(server.maybe_task)
        if tasks:
            await asyncio.wait(tasks, timeout=timeout)

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
        bridge = self._maybe_bridge
        if bridge is None or server.maybe_session is None:
            return CANONICAL_MCP_SERVER_UNAVAILABLE.format(tool=tool_name, server=name), metadata

        maybe_abort = getattr(dyn.tool_context, "maybe_abort", None)
        future = bridge.spawn(server.maybe_session.call_tool(tool_name, arguments,
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

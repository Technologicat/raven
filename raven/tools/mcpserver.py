"""`raven-mcp`: Raven's tools for other LLM frontends, over the Model Context Protocol.

Serves the web search and page fetch of Raven-server, and the lorebook's lookup, to any MCP client — Open
WebUI, Claude Code, LM Studio. The tools are Raven-librarian's own: the same specifications its LLM is shown,
and the same functions answering, so the two cannot come to describe or behave differently.

Over Streamable HTTP by default, which is what Open WebUI speaks, at `http://127.0.0.1:8130/mcp`; over stdio
with `--stdio`, for a client that starts the server itself.
"""

__all__ = ["tool_specs", "call_tool", "main"]

import argparse
import logging
from typing import Any

from .. import __version__

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8130


def _lorebook_entries() -> dict:
    """Read the lorebook afresh, so that an edit is seen at the next call without a file watcher. A few small files."""
    from ..librarian import config as librarian_config
    from ..librarian import lorebookcore
    entries, problems = lorebookcore.load(librarian_config.lorebook_dir, entry_levels=librarian_config.lorebook_entry_levels)
    for problem in problems.values():
        logger.warning(f"_lorebook_entries: {problem}")
    return entries


def tool_specs(*, lorebook: bool = True) -> list[dict[str, Any]]:
    """The tools offered, as OpenAI-style function specifications, the form Raven-librarian's LLM is shown them in.

    `lorebook`: whether to offer the lorebook's lookup. When offered, it is listed only while the lorebook has
                entries, as in Raven-librarian.
    """
    from ..librarian import llmtools
    from ..librarian import lorebookcore
    by_name = {spec["function"]["name"]: spec for spec in llmtools.TOOLS}
    specs = [by_name["websearch"], by_name["webfetch"]]
    if lorebook and _lorebook_entries():
        specs.append(lorebookcore.tool_spec())
    return specs


def call_tool(name: str, arguments: dict[str, Any], *, lorebook: bool = True) -> list[str]:
    """Run the tool `name` with `arguments`, and return its answer as a list of texts.

    The answer is what Raven-librarian's LLM would read, failures included: a search that found nothing, or
    a host the allowlist refuses, answers with a sentence saying so. Raises `ValueError` for a tool not offered.

    `lorebook`: as for `tool_specs`.

    Blocking: it waits for Raven-server.
    """
    from ..librarian import llmtools
    from ..librarian import lorebookcore
    if name == "websearch":
        output, _ = llmtools.websearch(str(arguments.get("query", "")))
    elif name == "webfetch":
        output, _ = llmtools.webfetch(str(arguments.get("url", "")))
    elif name == lorebookcore.TOOL_NAME and lorebook:
        output = lorebookcore.answer(_lorebook_entries(), str(arguments.get("term", "")))
    else:
        raise ValueError(f"call_tool: no tool named '{name}'")
    if isinstance(output, str):
        return [output]
    return [part["text"] for part in output]  # a search answers with one text part per result


def _make_server(*, lorebook: bool):
    """Return the MCP server, its handlers calling `tool_specs` and `call_tool`, with `lorebook` passed to both."""
    import anyio
    from mcp import types
    from mcp.server.lowlevel import Server

    async def on_list_tools(context, params) -> types.ListToolsResult:
        specs = await anyio.to_thread.run_sync(lambda: tool_specs(lorebook=lorebook))  # reads the lorebook folder
        return types.ListToolsResult(tools=[types.Tool(name=spec["function"]["name"],
                                                       description=spec["function"]["description"],
                                                       input_schema=spec["function"]["parameters"])
                                            for spec in specs])

    async def on_call_tool(context, params: types.CallToolRequestParams) -> types.CallToolResult:
        logger.info(f"on_call_tool: {params.name}({params.arguments})")
        try:
            texts = await anyio.to_thread.run_sync(lambda: call_tool(params.name, params.arguments or {}, lorebook=lorebook))
        except Exception as exc:
            logger.error(f"on_call_tool: {params.name} failed: {type(exc)}: {exc}")
            return types.CallToolResult(content=[types.TextContent(type="text", text=f"The tool failed: {exc}")],
                                        is_error=True)
        return types.CallToolResult(content=[types.TextContent(type="text", text=text) for text in texts])

    return Server("raven-mcp", version=__version__, on_list_tools=on_list_tools, on_call_tool=on_call_tool)


def main() -> None:
    parser = argparse.ArgumentParser(prog="raven-mcp",
                                     description="""Serve Raven's tools to other LLM frontends over MCP (the Model Context Protocol): web search and page fetch through Raven-server, and the lorebook's lookup. Over HTTP by default, at http://127.0.0.1:PORT/mcp, which is the address to give the frontend; over stdin and stdout with --stdio.""",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--stdio", action="store_true",
                        help="Serve over stdin and stdout instead of HTTP, for a client that starts the server itself.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="Port to serve HTTP on.")
    parser.add_argument("--listen", action="store_true",
                        help="Serve HTTP on all interfaces, not just this machine. Anyone who can reach the port can then search and fetch through Raven-server, so use this on a trusted network only.")
    parser.add_argument("--no-lorebook", dest="lorebook", action="store_false",
                        help="Do not offer the lorebook's lookup: for a frontend that has it already, from grimoire, so that the LLM is not handed the tool twice.")
    parser.add_argument("--server-url", metavar="URL", default=None,
                        help="Raven-server to talk to, overriding the configured one.")
    parser.add_argument("--log", metavar="PATH", default=None,
                        help="Mirror the log to this file (overwritten each run).")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Root logger level.")
    opts = parser.parse_args()

    # Heavy imports after parsing, so that `--help` answers at once.
    from ..common import logsetup
    logsetup.configure(level=getattr(logging, opts.log_level), logfile=opts.log)

    from ..client import api
    from ..client import config as client_config
    raven_server_url = opts.server_url if opts.server_url is not None else client_config.raven_server_url
    if opts.server_url is not None:
        logger.info(f"main: using Raven-server '{raven_server_url}' from --server-url, overriding the configured '{client_config.raven_server_url}'.")
    # First settings win, so this is what the tools use. A tool call is part of somebody's chat turn, hence
    # the interactive retry, short enough not to stall it.
    api.initialize(raven_server_url=raven_server_url,
                   maybe_network_retry=client_config.network_retry_interactive,
                   raven_api_key_file=client_config.raven_api_key_file)

    server = _make_server(lorebook=opts.lorebook)
    if opts.stdio:
        import anyio
        from mcp.server.stdio import stdio_server

        async def serve_stdio() -> None:
            async with stdio_server() as (read_stream, write_stream):
                await server.run(read_stream, write_stream, server.create_initialization_options())

        logger.info("main: serving over stdio")
        anyio.run(serve_stdio)
    else:
        import uvicorn
        host = "0.0.0.0" if opts.listen else "127.0.0.1"
        app = server.streamable_http_app(host=host)
        logger.info(f"main: serving at http://{host}:{opts.port}/mcp, for Raven-server at {raven_server_url}")
        uvicorn.run(app, host=host, port=opts.port, log_level=opts.log_level.lower())


if __name__ == "__main__":
    main()

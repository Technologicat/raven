"""An MCP server for `test_mcpclient`, run as a subprocess so that the tests exercise a real transport.

Usage: `python mcp_fixture_server.py stdio`, or `python mcp_fixture_server.py http PORT`.
"""

import os
import sys
import time

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

server = MCPServer("fixture")

@server.tool()
def echo(text: str) -> str:
    """Return the text as it was given."""
    return text

@server.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

@server.tool()
def fail() -> str:
    """Raise an error."""
    raise ToolError("deliberate failure")

@server.tool()
def slow(seconds: float) -> str:
    """Take a while to answer."""
    time.sleep(seconds)
    return "finished"

@server.tool()
def die() -> str:
    """End the server's process in the middle of a call."""
    os._exit(1)

@server.tool()
def big(n_characters: int) -> str:
    """Return a long text."""
    return "x" * n_characters

@server.tool()
def getenv(name: str) -> str:
    """Return an environment variable of the server's process."""
    return os.environ.get(name, "<unset>")

@server.tool()
def complain(text: str) -> str:
    """Write the text to the server's stderr."""
    print(text, file=sys.stderr, flush=True)
    return "complained"

@server.tool()
def pid() -> int:
    """Return the server's process ID."""
    return os.getpid()

@server.tool()
def request_header(name: str, ctx: Context) -> str:
    """Return a header of the HTTP request carrying this call."""
    return ctx.request_context.request.headers.get(name, "<unset>")

if __name__ == "__main__":
    if sys.argv[1] == "http":
        server.run("streamable-http", host="127.0.0.1", port=int(sys.argv[2]))
    else:
        server.run("stdio")

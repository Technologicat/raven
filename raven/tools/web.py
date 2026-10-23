"""Search the web, and fetch web pages as clean text, from the command line, via Raven-server.

This is the `raven-web` console script. Two verbs:

    raven-web search "unladen swallow airspeed"       # numbered title / link / snippet
    raven-web fetch https://example.com/article       # the page's main content, as Markdown

Both are thin frontends to Raven-server's `websearch` and `webfetch` modules — the same ones Librarian's
tools use — so the search engine interaction, the two-tier fetch (plain HTTP, then a headless browser for
pages that need JavaScript), and the refusal of private-network addresses all happen on the server.
Librarian's domain allowlist does not apply here: it guards what an LLM may fetch, and here whoever runs
the command chose the URL.

`--json` prints the server's result record instead, for scripts and agents. Progress lines go to stderr,
so stdout carries only the result.

Exit status: 0 on success; 1 when there is no result — the search engine did not answer, the page was
refused or could not be read, or the job failed on the server; 2 for a usage error; 3 when Raven-server
could not be reached, or refused the request (the module not running, a missing API key).

To reach a Raven-server on another machine, that server must be started with `--listen`, and this tool
pointed at it with `--server-url`. If the server runs with `--secure`, the API key is read from the
configured key file (`raven_api_key_file` in `raven.client.config`), or from `--api-key-file`.
"""

__all__ = ["main"]

import argparse
import json
import logging
import sys

from .. import __version__

# `..client.api` and `..client.config` are imported in `main`, after parsing, so that `--help` and a
# rejected option answer at once.

EXIT_OK = 0
EXIT_NO_RESULT = 1
EXIT_USAGE = 2  # argparse's own
EXIT_SERVER = 3


def _print_search_results(result: dict) -> None:
    """Print a `websearch2` result as a numbered list of title, link and snippet."""
    for k, item in enumerate(result["data"], start=1):
        maybe_title = item.get("title")  # not every engine returns one
        print(f"{k}. {maybe_title if maybe_title else item['link']}")
        if maybe_title:
            print(f"   {item['link']}")
        text = item.get("text", "").strip()
        if text:
            print(f"   {text}")
        print()


def _print_fetched_page(result: dict) -> None:
    """Print a `webfetch` result: title and effective URL, then the content under a rule."""
    if result.get("title"):
        print(result["title"])
    print(result["url"])
    print("---")
    print(result["content"])


def main() -> None:
    parser = argparse.ArgumentParser(description="""Search the web, or fetch a web page's main content, via Raven-server.""",
                                     epilog="""Exit status: 0 on success; 1 when there is no result (the engine did not answer, the page was refused or unreadable, or the job failed on the server); 2 for a usage error; 3 when Raven-server could not be reached or refused the request. A Raven-server on another machine must be started with --listen.""",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version=('%(prog)s ' + __version__))
    parser.add_argument('--server-url', metavar='URL', default=None,
                        help='Raven server to talk to, overriding the configured one; e.g. http://localhost:5100.')
    parser.add_argument('--api-key-file', metavar='PATH', default=None,
                        help='file holding the API key of a server started with --secure, overriding the configured one.')
    parser.add_argument('--timeout', metavar='SECONDS', type=float, default=None,
                        help='give up on the whole job after this long (default: the configured network timeout).')
    parser.add_argument('--json', action='store_true',
                        help="print the server's result record as JSON, instead of the human-readable form.")
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='do not print progress lines to stderr.')
    parser.add_argument('--log', metavar='PATH', default=None,
                        help='mirror stderr log to this file (overwritten each run)')
    parser.add_argument('--log-level', default='WARNING',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='root logger level (default: WARNING, so that only the result reaches the terminal)')
    subparsers = parser.add_subparsers(dest="verb", required=True, metavar="{search,fetch}")

    search_parser = subparsers.add_parser("search", help="search the web",
                                          description="Search the web. Prints a numbered list of title, link and snippet.")
    search_parser.add_argument(dest="query", type=str, metavar="QUERY", help="what to search for; quote it if it has spaces.")
    search_parser.add_argument('--engine', choices=["duckduckgo", "google"], default="duckduckgo",
                               help='search engine (default: duckduckgo)')
    search_parser.add_argument('--max-links', metavar='N', type=int, default=10,
                               help='how many results to ask for (default: 10). A hint: the engine may return more.')

    fetch_parser = subparsers.add_parser("fetch", help="fetch a web page's main content",
                                         description="Fetch a web page's main content. Prints its title and effective URL, then the content under a rule.")
    fetch_parser.add_argument(dest="url", type=str, metavar="URL", help="the page to fetch.")
    fetch_parser.add_argument('--format', dest="output_format", choices=["markdown", "text"], default="markdown",
                              help='output format (default: markdown)')

    opts = parser.parse_args()
    if opts.verb == "search" and not opts.query.strip():
        search_parser.error("the query is empty")
    if opts.verb == "search" and opts.max_links < 1:
        search_parser.error("--max-links must be at least 1")

    from ..common import logsetup  # noqa: PLC0415 -- not needed to parse
    logsetup.configure(level=getattr(logging, opts.log_level),
                       logfile=opts.log)
    logger = logging.getLogger(__name__)

    import requests  # noqa: PLC0415 -- not needed to parse
    from ..client import api  # noqa: PLC0415 -- not needed to parse
    from ..client import config as client_config  # noqa: PLC0415 -- not needed to parse
    from ..client import util as client_util  # noqa: PLC0415 -- not needed to parse

    raven_server_url = opts.server_url if opts.server_url is not None else client_config.raven_server_url
    if opts.server_url is not None:
        logger.info(f"main: using Raven server '{raven_server_url}' from --server-url, overriding the configured '{client_config.raven_server_url}'.")
    api.initialize(raven_server_url=raven_server_url,
                   raven_api_key_file=(opts.api_key_file if opts.api_key_file is not None else client_config.raven_api_key_file))

    timeout = None
    if opts.timeout is not None:
        timeout = client_config.Timeout(connect=client_config.network_timeout.connect, read=opts.timeout)

    def on_progress(line: str) -> None:
        if not opts.quiet:
            print(line, file=sys.stderr, flush=True)

    try:
        if opts.verb == "search":
            result = api.websearch_search(opts.query, engine=opts.engine, max_links=opts.max_links,
                                          timeout=timeout, on_progress=on_progress)
        else:
            result = api.webfetch_fetch(opts.url, output_format=opts.output_format,
                                        timeout=timeout, on_progress=on_progress)
    except KeyboardInterrupt:
        # Exiting closes the connection, which is also what tells the server to stop the job.
        sys.exit(130)
    except requests.ConnectionError as exc:
        print(f"Could not reach Raven-server at {raven_server_url}: {type(exc)}: {exc}", file=sys.stderr)
        print("Is it running? A server on another machine must be started with --listen.", file=sys.stderr)
        sys.exit(EXIT_SERVER)
    except requests.Timeout as exc:
        print(f"No result in time: {exc}", file=sys.stderr)
        sys.exit(EXIT_NO_RESULT)
    except client_util.ServerRefused as exc:  # the module not running, a missing API key; the server's message is in the log
        print(f"{exc}", file=sys.stderr)
        sys.exit(EXIT_SERVER)
    except client_util.JobFailed as exc:
        print(f"{exc}", file=sys.stderr)
        sys.exit(EXIT_NO_RESULT)

    if opts.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))

    if opts.verb == "search":
        if not result["engineAnswered"]:
            print(f"The search engine '{opts.engine}' did not answer. Try again, or try another --engine.", file=sys.stderr)
            sys.exit(EXIT_NO_RESULT)
        if not opts.json:
            if result["data"]:
                _print_search_results(result)
            else:
                print("No results.", file=sys.stderr)
    else:
        if not result["fetched"]:
            print(result["content"], file=sys.stderr)  # the server's canonical message for a refusal or a limit
            sys.exit(EXIT_NO_RESULT)
        if result.get("spaSuspected"):
            print("Note: the page may need JavaScript that neither fetch tier could run; the content may be incomplete.", file=sys.stderr)
        if not opts.json:
            _print_fetched_page(result)
    sys.exit(EXIT_OK)


if __name__ == "__main__":
    main()

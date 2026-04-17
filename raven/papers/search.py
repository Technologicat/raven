"""arXiv API search — paginated fetching and CLI.

Searches the arXiv API with a boolean expression (parsed by `raven.papers.query`),
fetches results with automatic pagination and rate limiting, and exports as BibTeX.
"""

from __future__ import annotations

__all__ = [
    "ARXIV_API_URL",
    "PAGE_SIZE",
    "MAX_ARXIV_RESULTS",
    "load_query",
    "determine_output_path",
    "clamp_max_results",
    "search",
    "main",
]

import argparse
import sys
from pathlib import Path

import feedparser
import requests

from .. import __version__
from .bibtex import entries_to_bibtex
from .query import node_to_query, parse_query
from .ratelimit import RateLimiter

ARXIV_API_URL = "https://export.arxiv.org/api/query"
PAGE_SIZE = 200
MAX_ARXIV_RESULTS = 30_000


def load_query(query_file: Path | None, query: str | None) -> str:
    """Resolve the query text from the mutually-exclusive CLI inputs.

    Exactly one of *query_file* (read from disk) or *query* (inline string)
    should be provided.  Returns the stripped query text.

    Raises ``ValueError`` if both are ``None`` or the resolved text is empty.
    """
    if query is not None:
        text = query.strip()
    elif query_file is not None:
        text = query_file.read_text().strip()
    else:
        raise ValueError("No query source provided (need either query_file or query).")
    if not text:
        raise ValueError("Query is empty.")
    return text


def determine_output_path(output: Path | None, query_file: Path | None) -> Path:
    """Pick the output ``.bib`` path, cascading through CLI defaults.

    - If *output* is given, use it as-is.
    - Else if *query_file* is given, replace its extension with ``.bib``.
    - Else fall back to ``results.bib`` in the current directory.
    """
    if output is not None:
        return output
    if query_file is not None:
        return query_file.with_suffix(".bib")
    return Path("results.bib")


def clamp_max_results(max_results: int | None) -> int | None:
    """Clamp *max_results* to arXiv's hard limit (``MAX_ARXIV_RESULTS``).

    ``None`` (fetch all) passes through unchanged.  Values within the limit
    pass through unchanged.  Values above the limit are reduced to it.
    """
    if max_results is None:
        return None
    return min(max_results, MAX_ARXIV_RESULTS)


def search(query: str, max_results: int | None = None) -> list[dict]:
    """Search arXiv, paginating as needed. Returns feedparser entries.

    Respects the arXiv rate limit of one request per 3 seconds.
    """
    rate_limiter = RateLimiter()
    results: list[dict] = []
    start = 0
    total: int | None = None

    while True:
        # How many to request this page
        page_size = PAGE_SIZE
        if max_results is not None:
            remaining = max_results - len(results)
            if remaining <= 0:
                break
            page_size = min(page_size, remaining)

        params = {
            "search_query": query,
            "start": start,
            "max_results": page_size,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        rate_limiter.wait()
        resp = requests.get(ARXIV_API_URL, params=params, timeout=30)
        resp.raise_for_status()

        feed = feedparser.parse(resp.text)

        if total is None:
            total = int(feed.feed.get("opensearch_totalresults", 0))
            effective = min(total, max_results) if max_results else total
            print(f"Total matches: {total}; fetching up to {effective}.", file=sys.stderr)

        # arXiv signals errors as entries with id "http://arxiv.org/api/errors"
        if feed.entries and "api/errors" in feed.entries[0].get("id", ""):
            msg = feed.entries[0].get("summary", "Unknown API error")
            raise RuntimeError(f"arXiv API error: {msg}")

        if not feed.entries:
            break

        results.extend(feed.entries)
        start += len(feed.entries)

        if start >= total:
            break
        if max_results is not None and len(results) >= max_results:
            break

    return results


# ---- CLI -------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(
        description="Search arXiv with a boolean expression and export results as BibTeX.",
    )
    ap.add_argument(
        "-V", "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "query_file",
        nargs="?",
        type=Path,
        default=None,
        help="Query file containing a boolean search expression. Alternative with -q.",
    )
    source.add_argument(
        "-q", "--query",
        type=str,
        default=None,
        help='Boolean search expression, e.g. \'"LLM" AND "AI"\'. Alternative with query file.',
    )
    ap.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output BibTeX file (default: <query_file>.bib, or results.bib with -q)",
    )
    ap.add_argument(
        "--max-results",
        type=int,
        default=None,
        help=f"Maximum results to fetch (arXiv hard limit: {MAX_ARXIV_RESULTS})",
    )
    args = ap.parse_args(argv)

    try:
        query_text = load_query(args.query_file, args.query)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        ast = parse_query(query_text)
    except ValueError as exc:
        print(f"Error parsing query: {exc}", file=sys.stderr)
        sys.exit(1)

    arxiv_query = node_to_query(ast)
    print(f"Query: {arxiv_query}", file=sys.stderr)

    if args.max_results is not None and args.max_results > MAX_ARXIV_RESULTS:
        print(f"Warning: arXiv caps at {MAX_ARXIV_RESULTS} results, clamping.", file=sys.stderr)
    max_results = clamp_max_results(args.max_results)

    entries = search(arxiv_query, max_results=max_results)
    if not entries:
        print("No results found.", file=sys.stderr)
        sys.exit(0)

    bibtex = entries_to_bibtex(entries)
    output_path = determine_output_path(args.output, args.query_file)
    output_path.write_text(bibtex, encoding="utf-8")
    print(f"Wrote {len(entries)} entries to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

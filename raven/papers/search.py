"""arXiv API search — paginated fetching and CLI.

Searches the arXiv API with a boolean expression (parsed by `raven.papers.query`),
fetches results with automatic pagination and rate limiting, and exports as BibTeX.
"""

from __future__ import annotations

__all__ = ["ARXIV_API_URL", "PAGE_SIZE", "search", "main"]

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

def main(argv: list[str] | None = None) -> None:
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

    # --- Read and parse the query ---
    if args.query is not None:
        query_text = args.query.strip()
    else:
        query_text = args.query_file.read_text().strip()
    if not query_text:
        print("Error: query is empty.", file=sys.stderr)
        sys.exit(1)

    try:
        ast = parse_query(query_text)
    except ValueError as exc:
        print(f"Error parsing query: {exc}", file=sys.stderr)
        sys.exit(1)

    arxiv_query = node_to_query(ast)
    print(f"Query: {arxiv_query}", file=sys.stderr)

    # --- Clamp max_results ---
    max_results = args.max_results
    if max_results is not None and max_results > MAX_ARXIV_RESULTS:
        print(
            f"Warning: arXiv caps at {MAX_ARXIV_RESULTS} results, clamping.",
            file=sys.stderr,
        )
        max_results = MAX_ARXIV_RESULTS

    # --- Fetch ---
    entries = search(arxiv_query, max_results=max_results)
    if not entries:
        print("No results found.", file=sys.stderr)
        sys.exit(0)

    # --- Write BibTeX ---
    bibtex = entries_to_bibtex(entries)
    if args.output:
        output_path = args.output
    elif args.query_file:
        output_path = args.query_file.with_suffix(".bib")
    else:
        output_path = Path("results.bib")
    output_path.write_text(bibtex, encoding="utf-8")
    print(f"Wrote {len(entries)} entries to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

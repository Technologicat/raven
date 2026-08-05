"""Fetch arXiv metadata for a list of identifiers and export it as BibTeX.

Completes the path that `raven-arxiv2id` starts. That tool reads a directory of PDFs and prints the arXiv
identifiers it finds in the filenames; this one turns those identifiers into a bibliography, so a folder of
downloaded papers becomes a searchable BibTeX database in two piped commands:

    raven-arxiv2id -i ~/papers | raven-arxiv2bib -o papers.bib

Reading identifiers from stdin is what makes that pipe work, and is the default when none are given as
arguments.

**Why this is not the `arxiv2bib` package on PyPI**, which does the same job: that one sends its requests
with no rate limiting at all, and this workflow's natural scale is a personal paper collection — hundreds
to thousands of identifiers. Raven's other arXiv tools already wait arXiv's documented three seconds
between requests (`raven.papers.ratelimit`), so routing the middle step through an external tool made the
workflow polite at both ends and rude in the middle. Everything needed to close it was already here: the
same API endpoint `raven.papers.search` calls takes an `id_list` alongside its `search_query`.

**Versions are preserved**, unlike in a search. An identifier list often carries them — `raven-arxiv2id`
emits `2410.07866v5`, having deduplicated a collection down to the newest version of each paper — and there
the version is part of what was asked for rather than an incidental fact about when the query ran. Asking
for two versions of one paper therefore yields two entries rather than one.

**Do not replace `id_list` with `search_query=id:...`**, however tempting it looks when arXiv is
misbehaving. The two are not interchangeable, and arXiv's API manual is explicit about which to use and
why: *"The `id_list` parameter should be used rather than `search_query=id:xxx` to properly handle article
versions."* Version handling is the whole reason this module exists in preference to the `arxiv2bib`
package, so a switch made to route around an outage would quietly cost the feature it is built on. Observed
on 2026-08-06: `id_list` requests hung while `search_query` answered in under a second, from the same
machine, minutes apart — an arXiv-side transient, not an argument about the API. `httpfetch.arxiv_get`
already retries with backoff; a stubborn outage is a reason to wait, not to change endpoints.
"""

from __future__ import annotations

__all__ = [
    "ARXIV_API_URL",
    "BATCH_SIZE",
    "read_identifiers",
    "fetch_metadata",
    "main",
]

import argparse
import sys
from pathlib import Path

import feedparser

from .. import __version__
from . import httpfetch
from .bibtex import entries_to_bibtex
from .identifiers import strip_version
from .ratelimit import RateLimiter

ARXIV_API_URL = "https://export.arxiv.org/api/query"

# Identifiers per request. The API accepts a long `id_list`, but a URL is not a good place to discover
# your limits: an over-long one fails as an opaque HTTP error partway through a batch job. 100 keeps the
# query string well inside any sane bound while still amortizing the three-second wait over real work.
BATCH_SIZE = 100


def read_identifiers(maybe_paths: list[str], stream=None) -> list[str]:
    """Collect arXiv identifiers from CLI arguments, files, or a stream.

    `maybe_paths`: what the user typed. Each item is treated as a file to read if such a file exists, and
                   as a literal identifier otherwise — so a list can be given directly, by file, or mixed,
                   without a flag to say which.
    `stream`: read from here when `maybe_paths` is empty (defaults to stdin). One identifier per line;
              anything after the first whitespace on a line is ignored, so `raven-arxiv2id --verbose`
              output, which appends the filename, pipes in unchanged.

    Blank lines and `#` comments are skipped. Order is preserved and duplicates are dropped, because the
    request is a set and a repeated identifier would otherwise buy a duplicate entry.
    """
    raw: list[str] = []
    if maybe_paths:
        for item in maybe_paths:
            path = Path(item)
            if path.is_file():
                raw.extend(path.read_text(encoding="utf-8").splitlines())
            else:
                raw.append(item)
    else:
        raw.extend((stream if stream is not None else sys.stdin).read().splitlines())

    seen: set[str] = set()
    identifiers: list[str] = []
    for line in raw:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        identifier = line.split()[0]
        if identifier not in seen:
            seen.add(identifier)
            identifiers.append(identifier)
    return identifiers


def fetch_metadata(identifiers: list[str], batch_size: int = BATCH_SIZE) -> list:
    """Fetch arXiv metadata for `identifiers`, in batches, respecting the rate limit.

    Returns feedparser entries in arXiv's order, ready for `bibtex.entries_to_bibtex`.

    Identifiers arXiv does not return are silently absent from the result rather than raising — a
    mistyped or withdrawn identifier should not discard the several hundred that worked. The caller is
    expected to diff the request against the result and report the gap; `main` does.
    """
    rate_limiter = RateLimiter()
    entries: list = []

    for start in range(0, len(identifiers), batch_size):
        batch = identifiers[start:start + batch_size]
        rate_limiter.wait()
        response = httpfetch.arxiv_get(ARXIV_API_URL,
                                       params={"id_list": ",".join(batch), "max_results": len(batch)},
                                       timeout=30)
        response.raise_for_status()
        feed = feedparser.parse(response.text)

        # arXiv reports errors as a single entry rather than an HTTP status, so a malformed request looks
        # like a successful fetch of one useless record unless this is checked.
        if feed.entries and "api/errors" in feed.entries[0].get("id", ""):
            raise RuntimeError(f"arXiv API error: {feed.entries[0].get('summary', 'unknown')}")

        entries.extend(feed.entries)
        print(f"  fetched {len(entries)} / {len(identifiers)}", file=sys.stderr)

    return entries


def _returned_identifiers(entries: list) -> set[str]:
    """Base identifiers present in `entries`, for diffing against what was requested."""
    return {strip_version(entry.id.split("/abs/")[-1]) for entry in entries}


# ---- CLI -------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(
        description="Fetch arXiv metadata for a list of identifiers and write it as BibTeX. "
                    "Reads identifiers from the command line, from files, or from stdin — so it pipes "
                    "directly from raven-arxiv2id.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="example:  raven-arxiv2id -i ~/papers | raven-arxiv2bib -o papers.bib",
    )
    ap.add_argument("identifiers", nargs="*", metavar="id_or_file", default=[],
                    help="arXiv identifiers (e.g. 2103.12345v2), or files listing one per line. "
                         "Default is to read them from stdin.")
    ap.add_argument("-o", "--output", type=Path, default=Path("results.bib"), metavar="out.bib",
                    help="Output BibTeX file (default: results.bib). Use - for stdout.")
    ap.add_argument("--strip-versions", action="store_true", default=False,
                    help="Record papers without their version suffix. Off by default: an identifier list "
                         "usually names specific versions, and that is part of what was asked for.")
    ap.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")
    args = ap.parse_args(argv)

    identifiers = read_identifiers(args.identifiers)
    if not identifiers:
        print("No identifiers given. Pass them as arguments, in a file, or on stdin.", file=sys.stderr)
        sys.exit(1)
    print(f"Fetching metadata for {len(identifiers)} identifiers.", file=sys.stderr)

    try:
        entries = fetch_metadata(identifiers)
    except Exception as exc:  # noqa: BLE001 -- the CLI reports, it does not add a traceback
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)

    if not entries:
        print("arXiv returned no records.", file=sys.stderr)
        sys.exit(1)

    # Report what did not come back. A silently shorter bibliography is the failure mode worth guarding
    # against here: it looks like success, and the gap only surfaces much later as a missing document.
    missing = sorted({strip_version(i) for i in identifiers} - _returned_identifiers(entries))
    if missing:
        print(f"Warning: {len(missing)} identifier(s) returned nothing: {', '.join(missing[:10])}"
              + (" ..." if len(missing) > 10 else ""), file=sys.stderr)

    bibtex = entries_to_bibtex(entries, keep_versions=not args.strip_versions)
    if str(args.output) == "-":
        print(bibtex)
    else:
        args.output.write_text(bibtex, encoding="utf-8")
        print(f"Wrote {len(entries)} entries to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()

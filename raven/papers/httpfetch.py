"""HTTP wrapper for the arXiv API.

The arXiv API TOU asks callers to identify themselves with a descriptive
`User-Agent` header, and the service occasionally returns HTTP 429 ("Too
Many Requests") even when the caller is within the published 3 s rate
limit — typically when a request misses the Varnish/Fastly cache and the
origin is briefly busy. `arxiv_get` papers over both: it attaches an
identifying `User-Agent` and retries on 429 with backoff, honoring the
`Retry-After` header when present.
"""

from __future__ import annotations

__all__ = [
    "arxiv_get",
]

import logging
import time
from typing import Any, Optional

import requests

from . import config as papers_config

logger = logging.getLogger(__name__)



def arxiv_get(url: str,
              params: Optional[dict[str, Any]] = None,
              timeout: float = 30,
              max_attempts: int = 3,
              base_backoff: float = 3.0) -> requests.Response:
    """GET `url`, with arXiv-identifying `User-Agent` and retry-with-backoff.

    Retried, up to `max_attempts` total attempts:

    - **HTTP 429** (rate limited). Wait time comes from the `Retry-After` response header (treated as
      seconds) when present, falling back to `base_backoff * 2**attempt` (3 s, 6 s, 12 s, ...).
    - **Transport errors** — connection failures, read timeouts, DNS trouble: anything
      `requests` raises rather than answers. Same backoff. The exception is re-raised if the last
      attempt also fails, so a genuine outage still surfaces rather than being swallowed.

    Returns the `requests.Response` from the final attempt — the caller is responsible for
    `raise_for_status()` and body parsing. HTTP error *statuses* other than 429 (including 5xx) are
    returned immediately without retry; those are answers, and the caller is better placed to decide
    what a 404 means than this function is.

    **Why transport errors are retried at all**, since an earlier version deliberately did not: the
    argument then was that the caller's own loop already survives unrelated failures, which held while
    callers fetched one paper per request. `download.get_papers_metadata` batches up to 100 identifiers
    into a single request, so one dropped connection now costs a hundred papers instead of one, and the
    per-item loop can no longer absorb it. Observed on 2026-08-06: an `id_list` request to arXiv timed
    out while a `search_query` from the same machine answered in 0.087 s — transient, and exactly the
    shape a retry fixes.
    """
    headers = {"User-Agent": papers_config.http_user_agent}
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
        except requests.exceptions.RequestException as exc:
            if attempt + 1 >= max_attempts:
                raise
            wait_s = base_backoff * (2 ** attempt)
            logger.warning(
                f"arxiv_get: {type(exc).__name__} from {url} "
                f"(attempt {attempt + 1}/{max_attempts}); retrying in {wait_s:.1f} s: {exc}"
            )
            time.sleep(wait_s)
            continue
        if response.status_code != 429:
            return response
        if attempt + 1 >= max_attempts:
            return response
        retry_after = response.headers.get("Retry-After")
        wait_s: float
        if retry_after is not None:
            try:
                wait_s = float(retry_after)
            except ValueError:
                wait_s = base_backoff * (2 ** attempt)
        else:
            wait_s = base_backoff * (2 ** attempt)
        logger.warning(
            f"arxiv_get: HTTP 429 from {url} "
            f"(attempt {attempt + 1}/{max_attempts}); retrying in {wait_s:.1f} s"
        )
        time.sleep(wait_s)
    # Unreachable: the loop returns or sleeps then continues. Kept for the type checker.
    return response  # noqa

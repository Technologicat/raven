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
    "USER_AGENT",
    "arxiv_get",
]

import logging
import time
from typing import Any, Optional

import requests

from .. import __version__

logger = logging.getLogger(__name__)

USER_AGENT = (
    f"raven-papers/{__version__} "
    f"(+https://github.com/Technologicat/raven; mailto:juha.jeronen@jamk.fi)"
)


def arxiv_get(url: str,
              params: Optional[dict[str, Any]] = None,
              timeout: float = 30,
              max_attempts: int = 3,
              base_backoff: float = 3.0) -> requests.Response:
    """GET `url`, with arXiv-identifying `User-Agent` and retry-with-backoff on HTTP 429.

    On a 429 response, retries up to `max_attempts` total attempts. Wait time
    between attempts is taken from the `Retry-After` response header (treated
    as seconds) when present, falling back to `base_backoff * 2**attempt`
    (3 s, 6 s, 12 s, ...) otherwise.

    Returns the `requests.Response` from the final attempt — the caller is
    responsible for `raise_for_status()` and body parsing. Non-429 statuses
    (including 5xx) are returned immediately without retry; arXiv's TOU
    only motivates 429 handling, and the caller's higher-level loop already
    survives unrelated failures.
    """
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(max_attempts):
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
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

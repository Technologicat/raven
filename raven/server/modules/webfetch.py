"""Web page fetch — retrieve a single page's main content as clean text/markdown.

The natural companion to `websearch`: `websearch` returns ranked links, `webfetch`
retrieves the chosen one and hands the model clean content instead of raw HTML.

Two-tier fetch, cheapest path first:

- **Tier 1** (default): `requests` GET + `trafilatura` readability extraction. Fast,
  no browser process, handles the overwhelming majority of pages.
- **Tier 2** (fallback): Selenium (a real headless browser) for JS-rendered pages.
  Slow, so used only when Tier 1 extracts below a usefulness threshold.

Escalation is decided by *result length*, not by framework-sniffing: sniffing for
SPA markers (`<div id="root">`, `__NEXT_DATA__`, script-to-text ratios) is a losing
arms race against new frameworks. If Tier 1 yields below threshold, escalate; if
Tier 2 is also below threshold, flag `spaSuspected` so the model can give up
gracefully rather than hallucinate around thin content.

Two cross-cutting concerns handled here, on the server because this is the machine
that actually makes the outbound request:

- **SSRF defense** (always on): refuse URLs whose host resolves to a private-network
  address (LAN admin pages, loopback, link-local / cloud-metadata), and refuse
  non-HTTP(S) schemes. The AI chooses the URLs, so this constrains what an
  AI-chosen (or prompt-injected) URL can reach. Opt out with
  `server_config.webfetch_allow_private_networks`.
- **Content normalization**: extracted text is run through `raven.common.text.normalize`
  to strip invisible-injection glyphs before it reaches the model.

The domain *allowlist* and *auto-allow-user-typed-URLs* policy is NOT here — that
constrains the AI's initiative and needs the conversation context, so it lives
client-side in `raven.librarian.llmclient`. This module enforces the network-level
safety (SSRF, scheme) that only the fetching machine can.

Every refusal / limit case returns a canonical user-facing string the model is meant
to copy verbatim, so user-facing messaging stays consistent across sessions and
models instead of each model improvising its own (often inaccurate) explanation.
"""

__all__ = ["init_module", "is_available", "fetch"]

import logging
logger = logging.getLogger(__name__)

import atexit
import ipaddress
import re
import socket
import threading
from typing import Callable, Dict, Optional, Tuple
import urllib.parse

import requests

from colorama import Fore, Style

from ...common import text as common_text

# `trafilatura`, `youtube_transcript_api`, and `selenium` (via `websearch`) are imported lazily
# at their use sites, so the pure helpers (URL rewriting, SSRF/scheme checks) — and their tests —
# import without dragging in the heavy fetch/extraction stack.

# See `navigator.userAgent` in a browser's JS console. Some sites serve thin content to unknown agents.
_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"

# --------------------------------------------------------------------------------
# Canonical user-facing strings.
#
# Pre-templated so the model copies them verbatim instead of fabricating variations.
# `{host}`, `{scheme}`, `{url}`, `{status}` are filled per case. The allowlist-refusal
# string lives client-side (`raven.librarian.llmclient`), since the allowlist is enforced there.

CANONICAL_PRIVATE_NETWORK = ("This URL points to a private-network address ({host}). Fetching it would risk "
                             "accessing local services unintentionally. Blocked for safety.")
CANONICAL_BAD_SCHEME = "Only HTTP and HTTPS URLs can be fetched. {scheme} URLs are not supported."
CANONICAL_DNS_FAILURE = ("Could not resolve the host name {host}. The site may be down, or the URL may be wrong.")
CANONICAL_SPA_SUSPECTED = "This site doesn't render its content as static HTML and can't be fetched as text."
CANONICAL_HTTP_ERROR = ("The server returned HTTP {status} for {url}. The page may be unavailable or require "
                        "authentication.")

# --------------------------------------------------------------------------------
# Bootup

_initialized = False
_driver = None  # lazily created on first Tier 2 use
_driver_lock = threading.Lock()  # serializes Tier 2 navigations (a single Selenium driver is not concurrency-safe)

def init_module(config_module_name: str) -> None:
    """Initialize the webfetch module.

    Tier 1 (`requests` + `trafilatura`) needs no setup; the Selenium driver for Tier 2
    is created lazily on first use (and only if a browser is available), so enabling
    this module is cheap.

    `config_module_name` is accepted for signature parity with the other server modules;
    webfetch reads its settings from `raven.server.config` directly.
    """
    global _initialized
    print(f"Initializing {Fore.GREEN}{Style.BRIGHT}webfetch{Style.RESET_ALL}...")
    _initialized = True

def is_available() -> bool:
    """Return whether this module is up and running.

    True once `init_module` has run; Tier 1 always works. Tier 2 availability depends on
    a browser being installed, which is checked lazily when a fetch actually needs it.
    """
    return _initialized

# --------------------------------------------------------------------------------
# Pre-fetch URL rewriting / special-case extractors.
#
# A small URL-pattern layer routes known-difficult sites to better extraction paths,
# applied before the two-tier fetch. Each special extractor is an isolated function
# `(url) -> str` that returns clean text/markdown, bypassing the standard path. Absent
# patterns fall through to the standard two-tier fetch unchanged.

_ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf|html)/(?P<arxiv_id>[^?#]+?)(?:\.pdf)?/?$", re.IGNORECASE)
_YOUTUBE_WATCH_RE = re.compile(r"(?:youtube\.com/watch\?(?:.*&)?v=|youtu\.be/)(?P<video_id>[A-Za-z0-9_-]{11})", re.IGNORECASE)

def _rewrite_url(url: str) -> Tuple[str, Optional[Callable[[str], str]]]:
    """Apply pre-fetch URL rewriting / special-case routing.

    Returns `(effective_url, maybe_special_extractor)`:

    - `effective_url`: the URL the standard two-tier fetch should hit (possibly rewritten).
    - `maybe_special_extractor`: if not `None`, a function `(effective_url) -> str` that
      produces the content directly, bypassing the standard fetch (used when the content
      doesn't come from the page's static HTML, e.g. a YouTube transcript, or needs a
      custom fallback, e.g. arXiv HTML→abstract).

    Pure: no network access here. Rewriting and extraction are separated so the rewrite
    decision is unit-testable without hitting the network.
    """
    parts = urllib.parse.urlsplit(url)
    host = (parts.hostname or "").lower()

    # arXiv: prefer the clean HTML rendering (`arxiv.org/html/ID`) over PDF / abstract-only.
    if host.endswith("arxiv.org"):
        if (m := _ARXIV_ID_RE.search(url)) is not None:
            arxiv_id = m.group("arxiv_id")
            return f"https://arxiv.org/html/{arxiv_id}", _extract_arxiv

    # Reddit: the new design is JS-heavy; `old.reddit.com` is plain HTML that extracts cleanly via Tier 1.
    if host == "reddit.com" or host.endswith(".reddit.com"):
        if host != "old.reddit.com":
            rewritten = parts._replace(netloc="old.reddit.com").geturl()
            return rewritten, None

    # YouTube: the page is almost pure player chrome; fetch the transcript instead.
    if host.endswith("youtube.com") or host == "youtu.be" or host.endswith(".youtu.be"):
        if (m := _YOUTUBE_WATCH_RE.search(url)) is not None:
            return url, _extract_youtube_transcript

    return url, None

def _extract_arxiv(url: str) -> str:
    """Fetch an arXiv paper, preferring the HTML rendering, falling back to the abstract page.

    `url` is the `arxiv.org/html/ID` form produced by `_rewrite_url`. Older papers have no
    HTML rendering (404); for those, fall back to `arxiv.org/abs/ID` (the abstract page).
    """
    html, status = _http_get(url)
    if status == 404 and (m := _ARXIV_ID_RE.search(url)) is not None:
        arxiv_id = m.group("arxiv_id")
        logger.info(f"_extract_arxiv: HTML form 404 for arXiv {arxiv_id}, falling back to abstract page.")
        html, status = _http_get(f"https://arxiv.org/abs/{arxiv_id}")
    return _extract_clean_text(html, url=url) if html else ""

def _extract_youtube_transcript(url: str) -> str:
    """Return the transcript text of a YouTube video, or "" if no transcript is available."""
    m = _YOUTUBE_WATCH_RE.search(url)
    if m is None:
        return ""
    video_id = m.group("video_id")
    from youtube_transcript_api import YouTubeTranscriptApi  # noqa: PLC0415 -- deferred: keep yt-api out of pure-helper imports
    try:
        fetched = YouTubeTranscriptApi().fetch(video_id)
    except Exception as exc:
        logger.info(f"_extract_youtube_transcript: no transcript for video '{video_id}', reason {type(exc)}: {exc}")
        return ""
    return " ".join(snippet.text for snippet in fetched)

# --------------------------------------------------------------------------------
# SSRF defense and scheme check (pure helpers + a thin DNS-resolving wrapper).

def _check_scheme(url: str) -> Optional[str]:
    """Return a canonical refusal string if `url`'s scheme is not HTTP(S), else `None`. Pure."""
    scheme = (urllib.parse.urlsplit(url).scheme or "").lower()
    if scheme not in ("http", "https"):
        return CANONICAL_BAD_SCHEME.format(scheme=(scheme or "(none)"))
    return None

def _is_blocked_ip(ip_str: str) -> bool:
    """Return whether `ip_str` is a non-public address that SSRF defense should refuse. Pure.

    Blocks loopback, private (RFC 1918 / ULA), link-local (incl. the 169.254 cloud-metadata
    range), multicast, unspecified, and reserved addresses — anything not a globally-routable
    public host.
    """
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True  # unparseable -> fail closed
    return (ip.is_private or ip.is_loopback or ip.is_link_local or
            ip.is_multicast or ip.is_unspecified or ip.is_reserved)

def _classify_url_network_safety(url: str, *, allow_private: bool) -> Optional[str]:
    """Return a canonical refusal string if `url` is unsafe to fetch, else `None`.

    Checks scheme (pure) and then resolves the host and checks every resolved IP against
    `_is_blocked_ip` (unless `allow_private`). DNS failure yields the DNS canonical string.
    This is the one network-touching gate; the building blocks (`_check_scheme`,
    `_is_blocked_ip`) are pure and unit-tested directly.
    """
    if (refusal := _check_scheme(url)) is not None:
        return refusal

    host = (urllib.parse.urlsplit(url).hostname or "").lower()
    if not host:
        return CANONICAL_DNS_FAILURE.format(host="(none)")

    try:
        resolved = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return CANONICAL_DNS_FAILURE.format(host=host)

    if not allow_private:
        for family, _type, _proto, _canonname, sockaddr in resolved:
            ip_str = sockaddr[0]
            if _is_blocked_ip(ip_str):
                logger.warning(f"_classify_url_network_safety: refusing '{url}': host '{host}' resolves to blocked address {ip_str}.")
                return CANONICAL_PRIVATE_NETWORK.format(host=host)

    return None

# --------------------------------------------------------------------------------
# The two-tier fetch.

def _http_get(url: str) -> Tuple[Optional[str], Optional[int]]:
    """Tier 1 transport: plain `requests` GET. Returns `(html_or_None, status_code_or_None)`.

    On a network-level failure (timeout, connection error), returns `(None, None)`; the
    caller escalates to Tier 2.
    """
    from .. import config as server_config
    try:
        response = requests.get(url,
                                headers={"User-Agent": _USER_AGENT},
                                timeout=server_config.webfetch_request_timeout,
                                allow_redirects=True)
    except requests.RequestException as exc:
        logger.info(f"_http_get: Tier 1 GET failed for '{url}', reason {type(exc)}: {exc}")
        return None, None
    return response.text, response.status_code

def _extract_clean_text(html: Optional[str], *, url: str, output_format: str = "markdown") -> str:
    """Run `trafilatura` readability extraction over `html`. Returns "" if nothing extractable."""
    if not html:
        return ""
    import trafilatura  # noqa: PLC0415 -- deferred: keep trafilatura out of pure-helper imports
    extracted = trafilatura.extract(html,
                                    url=url,
                                    output_format=output_format,
                                    include_tables=True,
                                    favor_recall=True)
    return extracted or ""

def _fetch_tier2(url: str, *, output_format: str) -> str:
    """Tier 2: render `url` in a real headless browser, then extract. Returns "" if unavailable.

    Lazily creates a Selenium driver (reusing `websearch`'s factory) on first use; serialized
    by a lock since a single driver can't navigate concurrently. If no browser is installed,
    returns "" and the caller treats the page as unfetchable.
    """
    global _driver
    from . import websearch  # noqa: PLC0415 -- deferred: reuse its Selenium driver factory only when Tier 2 runs
    with _driver_lock:
        if _driver is None:
            _driver = websearch.get_driver()
            if _driver is not None:
                atexit.register(lambda: _driver.quit())
        if _driver is None:
            logger.info("_fetch_tier2: no browser available; cannot render JS-heavy page.")
            return ""
        try:
            _driver.get(url)
            html = _driver.page_source
        except Exception as exc:
            logger.warning(f"_fetch_tier2: browser navigation failed for '{url}', reason {type(exc)}: {exc}")
            return ""
    return _extract_clean_text(html, url=url, output_format=output_format)

def _make_result(content: str, *, url: str, spa_suspected: bool = False) -> Dict:
    """Build the structured result dict returned by `fetch`.

    `content` is the LLM-facing text (extracted content, or a canonical message for
    refusals / limits). `spaSuspected` flags a page neither tier could extract.
    """
    return {"content": content, "url": url, "spaSuspected": spa_suspected}

def fetch(url: str, output_format: str = "markdown") -> Dict:
    """Retrieve a web page's main content as clean text/markdown.

    Returns a dict `{"content": str, "url": str, "spaSuspected": bool}`. `content` is the
    text the model reads; for any refusal or limit case it is the canonical user-facing
    string. `url` is the effective URL after rewriting. `spaSuspected` is True when neither
    fetch tier could extract usable content (heavy SPA, login wall, captcha).

    `output_format` is "markdown" (default, preserves headings/lists/links) or "text".

    Order of operations: scheme + SSRF gate (on the effective URL) → URL rewriting →
    special extractor or two-tier fetch → content normalization.
    """
    from .. import config as server_config
    trafilatura_format = "markdown" if output_format == "markdown" else "txt"

    effective_url, special_extractor = _rewrite_url(url)

    # SSRF / scheme gate on the *effective* URL (a rewrite can change the host, e.g. reddit -> old.reddit.com).
    refusal = _classify_url_network_safety(effective_url, allow_private=server_config.webfetch_allow_private_networks)
    if refusal is not None:
        return _make_result(refusal, url=effective_url)

    if special_extractor is not None:
        content = special_extractor(effective_url)
        if len(content) >= server_config.webfetch_min_content_chars:
            return _make_result(common_text.normalize(content), url=effective_url)
        # A special extractor that came up short (e.g. no YouTube transcript) falls through
        # to the standard two-tier fetch on the same URL, which may still find something.

    html, status = _http_get(effective_url)
    content = _extract_clean_text(html, url=effective_url, output_format=trafilatura_format)

    if len(content) < server_config.webfetch_min_content_chars:
        tier2_content = _fetch_tier2(effective_url, output_format=trafilatura_format)
        if len(tier2_content) >= len(content):
            content = tier2_content

    if len(content) < server_config.webfetch_min_content_chars:
        # Both tiers came up short. Distinguish a definitive HTTP error from a JS-only page.
        if status is not None and status >= 400 and not content:
            return _make_result(CANONICAL_HTTP_ERROR.format(status=status, url=effective_url), url=effective_url)
        return _make_result(CANONICAL_SPA_SUSPECTED, url=effective_url, spa_suspected=True)

    return _make_result(common_text.normalize(content), url=effective_url)

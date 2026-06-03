#!/usr/bin/env python3
"""Manual live smoke test for the webfetch two-tier escalation (Tier 1 -> Selenium Tier 2).

NOT a pytest test — it hits the live internet AND spawns a headless Chrome (Selenium), so it
lives here under `briefs/` rather than in the suite. The deterministic logic of the escalation
(thresholding, spaSuspected) is covered structurally in `raven/server/tests/test_webfetch.py`;
this proves the real browser path works end-to-end.

Default target: `quotes.toscrape.com/js/` — a scraping sandbox whose `/js/` variant injects its
content via JavaScript. Tier 1 (`requests` + readability) therefore comes up thin, which forces
escalation to Tier 2; Chrome renders the JS and the quotes appear, so the fetch is rescued.

This script *proves* the escalation actually fired (by wrapping `_fetch_tier2`), rather than
merely observing that content appeared — a static page would pass the latter check trivially.

Needs network and a working Selenium/Chrome (the same one `websearch` Tier 2 uses).

Usage:
    python webfetch_tier2_escalation.py            # default JS-rendered target
    python webfetch_tier2_escalation.py <url>      # try your own JS-heavy / SPA URL
"""

import sys

from raven.server.modules import webfetch

DEFAULT_URL = "http://quotes.toscrape.com/js/"
# A phrase that exists only in the JS-rendered DOM of the default target (absent from static HTML).
DEFAULT_RENDERED_MARKER = "Einstein"


def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    rendered_marker = DEFAULT_RENDERED_MARKER if url == DEFAULT_URL else None

    # Show what Tier 1 alone extracts, to confirm escalation is genuinely warranted.
    html, status = webfetch._http_get(url)
    tier1 = webfetch._extract_clean_text(html, url=url, output_format="markdown")
    print(f"Tier 1 alone: status={status}, extracted_len={len(tier1)} "
          f"(threshold is server_config.webfetch_min_content_chars)")

    # Wrap Tier 2 to prove it actually fired.
    original_tier2 = webfetch._fetch_tier2
    tier2_calls = []

    def traced_tier2(target_url, *, output_format):
        tier2_calls.append(target_url)
        print(f">>> Tier 2 (Selenium/Chrome) escalation triggered for {target_url}")
        return original_tier2(target_url, output_format=output_format)

    webfetch._fetch_tier2 = traced_tier2
    try:
        result = webfetch.fetch(url)
    finally:
        webfetch._fetch_tier2 = original_tier2

    content = result["content"]
    print(f"Tier 2 invoked: {bool(tier2_calls)}")
    print(f"final spaSuspected={result['spaSuspected']}  content_len={len(content)}")
    if rendered_marker is not None:
        print(f"JS-rendered marker ('{rendered_marker}') present: {rendered_marker in content}")
    print(f"preview: {content[:200].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()

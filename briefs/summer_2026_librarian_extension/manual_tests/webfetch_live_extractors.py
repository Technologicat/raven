#!/usr/bin/env python3
"""Manual live smoke test for `raven.server.modules.webfetch` extraction paths.

NOT a pytest test — it hits the live internet and depends on external sites, so it is
inherently non-deterministic and lives here under `briefs/` rather than in the test suite
(the deterministic structural tests are in `raven/server/tests/test_webfetch.py` etc.).

Exercises the extraction surface that unit tests can't reach: generic readability extraction
plus the arXiv / Reddit / YouTube special routers, and confirms content normalization ran.

Needs no LLM and no running Raven-server — it calls `webfetch.fetch` in-process. Requires
network access and a working Selenium/Chrome only if a page falls back to Tier 2 (none of the
defaults below should).

Usage:
    python webfetch_live_extractors.py            # run the default probe set
    python webfetch_live_extractors.py <url> ...  # probe specific URLs instead
"""

import sys

from raven.server.modules import webfetch

# Codepoints that `raven.common.text.normalize` strips; used to confirm normalization ran.
_INJECTION_CODEPOINTS = set(range(0xE0000, 0xE0080)) | {0x200B, 0x200C, 0x200D, 0x2060, 0x180E}

DEFAULT_PROBES = [
    ("generic (Wikipedia)", "https://en.wikipedia.org/wiki/Common_raven"),
    ("arXiv abs -> HTML form", "https://arxiv.org/abs/1706.03762"),
    ("YouTube transcript", "https://www.youtube.com/watch?v=aircAruvnKk"),
    ("Reddit -> old.reddit", "https://www.reddit.com/r/MachineLearning/"),
]


def probe(label: str, url: str) -> None:
    print(f"=== {label} ===")
    print(f"    requested: {url}")
    try:
        result = webfetch.fetch(url, output_format="markdown")
    except Exception as exc:  # noqa: BLE001 -- this is a smoke test; report anything that goes wrong
        print(f"    ERROR {type(exc).__name__}: {exc}\n")
        return
    content = result["content"]
    leftover_injection = sum(1 for ch in content if ord(ch) in _INJECTION_CODEPOINTS)
    print(f"    effective url:  {result['url']}")
    print(f"    spaSuspected:   {result['spaSuspected']}")
    print(f"    content length: {len(content)}")
    print(f"    injection chars remaining: {leftover_injection}")
    print(f"    preview: {content[:160].replace(chr(10), ' ')}")
    print()


def main() -> None:
    probes = [(url, url) for url in sys.argv[1:]] if len(sys.argv) > 1 else DEFAULT_PROBES
    for label, url in probes:
        probe(label, url)


if __name__ == "__main__":
    main()

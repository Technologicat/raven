"""How many of a `.bib`'s missing and truncated abstracts can Crossref supply?

Probe for `raven-fillbib`. For each record carrying a DOI, of three kinds:

- **missing**: no abstract at all;
- **truncated**: an abstract that ends in an ellipsis, the shape a vendor export leaves when it cuts the
  text at a fixed length;
- **control**: a random sample of the rest, whose abstracts we take to be complete. These are what tell
  whether the comparison works at all: a complete abstract should come back *equal* to Crossref's, and if
  the controls mostly come back *different*, the normalization is wrong and every other number is suspect.

fetch `https://api.crossref.org/works/{doi}` and classify what comes back. The responses are cached in a
JSONL keyed on DOI, so a rerun with a changed comparison asks Crossref nothing.

Usage::

    python investigations/bib-abstract-fill/probe_crossref.py INPUT.bib --cache CACHE.jsonl [--controls N]
"""

import argparse
import collections
import json
import pathlib
import random
import re
import sys
import time
import urllib.parse

import requests

from raven.common import utils as common_utils
from raven.papers import bibtex
from raven.papers import config as papers_config

ELLIPSIS = re.compile(r"\s*(\.\.\.|…)\s*$")
TAG = re.compile(r"<(/?)([a-zA-Z][\w:.-]*)[^>]*>")


def normalize(text: str) -> str:
    """Strip markup, collapse whitespace. What the prefix test compares."""
    text = TAG.sub(" ", text)
    text = common_utils.unicodize_basic_markup(text)
    return " ".join(text.split())


def loose(text: str) -> str:
    """Letters and digits only, casefolded. A second opinion on near-misses."""
    return "".join(c for c in normalize(text).casefold() if c.isalnum())


def strip_leading_heading(text: str) -> str:
    """Crossref abstracts often open with a title saying "Abstract"."""
    return re.sub(r"^(abstract|summary)[\s:.]*", "", text, flags=re.IGNORECASE)


def fetch(doi: str, session: requests.Session) -> dict:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="/")
    for attempt in range(4):
        try:
            r = session.get(url, timeout=30)
        except requests.RequestException as exc:
            err = f"{type(exc)}: {exc}"
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 429:
            time.sleep(float(r.headers.get("Retry-After", 2 ** (attempt + 1))))
            continue
        if r.status_code == 404:
            return {"status": 404}
        if r.status_code != 200:
            return {"status": r.status_code}
        msg = r.json()["message"]
        return {"status": 200,
                "abstract": msg.get("abstract"),
                "publisher": msg.get("publisher"),
                "pool": r.headers.get("x-api-pool"),
                "rate": (r.headers.get("x-rate-limit-limit"), r.headers.get("x-rate-limit-interval"))}
    return {"status": "failed", "error": err if "err" in locals() else "429 retries exhausted"}


def classify(ours: str, theirs: str | None) -> str:
    if not theirs:
        return "crossref has none"
    t = strip_leading_heading(normalize(theirs))
    if not ours:
        return "fill (ours missing)"
    o = normalize(ELLIPSIS.sub("", ours))
    if o == t:
        return "equal"
    if t.startswith(o) and len(t) > len(o):
        return "repair (ours is a prefix)"
    lo, lt = loose(ELLIPSIS.sub("", ours)), loose(strip_leading_heading(normalize(theirs)))
    if lo == lt:
        return "equal only loosely"
    if lt.startswith(lo):
        return "prefix only loosely"
    return "different"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("--cache", type=pathlib.Path, required=True)
    parser.add_argument("--controls", type=int, default=150)
    parser.add_argument("--delay", type=float, default=0.35, help="seconds between requests")
    parser.add_argument("--examples", type=int, default=0, help="print this many examples per class")
    args = parser.parse_args()

    library = bibtex.parse_file(args.input)
    groups = collections.defaultdict(list)
    for entry in library.entries:
        doi = bibtex.entry_field(entry, "doi").strip()
        if not doi:
            continue
        abstract = bibtex.entry_field(entry, "abstract")
        kind = "missing" if not abstract else "truncated" if ELLIPSIS.search(abstract) else "control"
        groups[kind].append((entry, doi.lower(), abstract))
    random.Random(42).shuffle(groups["control"])
    groups["control"] = groups["control"][:args.controls]
    print({k: len(v) for k, v in groups.items()}, flush=True)

    cache = {}
    if args.cache.exists():
        for line in args.cache.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            cache[rec["doi"]] = rec
    session = requests.Session()
    session.headers["User-Agent"] = papers_config.http_user_agent
    todo = sorted({doi for g in groups.values() for _, doi, _ in g} - set(cache))
    print(f"fetching {len(todo)}, {len(cache)} cached", flush=True)
    with args.cache.open("a", encoding="utf-8") as f:
        for i, doi in enumerate(todo):
            rec = {"doi": doi, **fetch(doi, session)}
            cache[doi] = rec
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 100 == 0:
                print(f"  {i}/{len(todo)} pool={rec.get('pool')} rate={rec.get('rate')}", flush=True)
            time.sleep(args.delay)

    tags = collections.Counter()
    for rec in cache.values():
        for close, name in TAG.findall(rec.get("abstract") or ""):
            if not close:
                tags[name] += 1
    for kind, items in groups.items():
        results = collections.Counter()
        by_publisher = collections.defaultdict(collections.Counter)
        examples = collections.defaultdict(list)
        for entry, doi, ours in items:
            rec = cache[doi]
            c = classify(ours, rec.get("abstract")) if rec["status"] == 200 else f"http {rec['status']}"
            results[c] += 1
            by_publisher[rec.get("publisher") or "?"][c] += 1
            examples[c].append((doi, ours, rec.get("abstract")))
        print(f"\n== {kind}: {len(items)}")
        for c, n in results.most_common():
            print(f"  {n:5d}  {c}")
        print("  by publisher (top 10):")
        for pub, cnt in sorted(by_publisher.items(), key=lambda kv: -sum(kv[1].values()))[:10]:
            print(f"    {sum(cnt.values()):4d}  {pub[:40]:40s}  {dict(cnt)}")
        for c, exs in examples.items():
            for doi, ours, theirs in exs[:args.examples]:
                print(f"  -- {c}: {doi}\n     ours:   {ours[-150:]!r}\n     theirs: {(theirs or '')[:300]!r}")
    print("\nJATS/markup tags seen in Crossref abstracts:")
    for name, n in tags.most_common():
        print(f"  {n:6d}  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

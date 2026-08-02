# `raven.papers` — consolidating paper/bibliography tools

**Date:** 2026-04-10
**Status:** Implemented
**Trigger:** Integrating the standalone `arxiv-api-search` CLI tool into Raven

## Motivation

Raven has accumulated several tools for finding, fetching, converting, and preparing
academic papers — currently scattered across `raven/tools/` and an external standalone
project. Creating `raven.papers` groups them by domain and enables code sharing.

## Source inventory

### External (to integrate)

| Module | Lines | What it does |
|--------|-------|--------------|
| `arxiv-api-search/parser.py` | 191 | Boolean expression tokenizer + recursive-descent parser → AST |
| `arxiv-api-search/query.py` | 95 | AST → arXiv API `search_query` string; paginated fetch via `feedparser` |
| `arxiv-api-search/bibtex.py` | 80 | Feedparser entries → BibTeX via `bibtexparser` |
| `arxiv-api-search/__main__.py` | 103 | CLI glue (argparse) |
| Tests: `test_parser.py`, `test_query.py`, `test_bibtex.py` | 302 | Good coverage of parser, query builder, BibTeX output |

License: MIT (same author). Relicensed to BSD for integration — no extra license file needed.

### Existing in `raven/tools/` (to relocate)

| Module | Lines | What it does |
|--------|-------|--------------|
| `arxiv2id.py` | 110 | Extract arXiv IDs from PDF filenames in a directory |
| `arxiv_download.py` | 250 | Fetch metadata + download PDFs by arXiv ID |
| `burstbib.py` | 116 | Split a `.bib` file into individual entries (for Librarian RAG ingest) |
| `csv2bib.py` | 88 | CSV → BibTeX |
| `pdf2bib.py` | ~400 | PDF conference abstracts → BibTeX via LLM (uses librarian's llmclient) |
| `wos2bib.py` | 187 | Web of Science export → BibTeX |

### Staying in `raven/tools/` (not paper-related)

| Module | What it does |
|--------|--------------|
| `check_cuda.py` | CUDA hardware diagnostic |
| `check_audio_devices.py` | Audio device listing |
| `qoi2png.py` | QOI → PNG image conversion (raven-avatar-settings-editor video recordings) |
| `dehyphenate.py` | ML-based text dehyphenation (server-backed) |

## Proposed structure

```
raven/papers/
    __init__.py          # __all__, package docstring
    identifiers.py       # arXiv ID regex, filename scanning, version parsing (from arxiv2id.py)
    ratelimit.py         # RateLimiter class, extracted (from arxiv_download.py)
    query.py             # Boolean parser + arXiv query builder (from arxiv-api-search parser+query)
    bibtex.py            # arXiv feed entries → BibTeX (from arxiv-api-search bibtex)
    search.py            # arXiv paginated search + CLI main() (from arxiv-api-search)
    download.py          # arXiv metadata fetch, paper download + CLI main() (from arxiv_download.py)
    burstbib.py          # Split BibTeX into entries + CLI main()
    csv2bib.py           # CSV → BibTeX + CLI main()
    pdf2bib.py           # PDF abstracts → BibTeX via LLM + CLI main()
    wos2bib.py           # Web of Science → BibTeX + CLI main()
    tests/
        __init__.py
        test_query.py    # tokenizer, parser, query builder
        test_bibtex.py   # BibTeX converter
```

11 source modules. Each has a clear single responsibility.

## Synergies to exploit

### RateLimiter extraction

`arxiv_download.py` has a solid thread-safe `RateLimiter` with tqdm progress.
The search tool uses bare `time.sleep(3)`. Extract to `ratelimit.py`, share between
`search.py` and `download.py`. The search tool gets a progress bar during long
paginated fetches for free.

### arXiv ID version-stripping

Three separate implementations today:
- `arxiv2id.py`: `split_arxiv_identifier()` → `(base, version_int)`
- `arxiv_download.py`: `clean_arxiv_id()` → strips `vN` suffix
- `arxiv-api-search/bibtex.py`: inline `re.sub(r"v\d+$", "", arxiv_id)`

Consolidate into `identifiers.py` with a canonical `strip_version()`.

### `bibtex_escape()` duplication

`csv2bib.py` and `wos2bib.py` define identical `bibtex_escape()` functions.
Extract to a shared location (e.g. a `utils.py` or into `__init__.py`).
Fix during implementation.

Note: `wos2bib` has a known escaping bug — some WoS export files break the import
(see `TODO_DEFERRED.md`). The shared `bibtex_escape()` is the right place to fix it,
but the fix itself is a separate task from the extraction.

### API base URL

Both `search.py` and `download.py` hardcode `export.arxiv.org/api/query`.
Define once (e.g. in `__init__.py` or a shared constant).

## Merge decisions

### `parser.py` + `query.py` → single `query.py`

The parser's AST types (`Term`, `BinOp`) are the exact types `node_to_query()` walks.
They're coupled at the type level. One ~280-line module with section headers
(AST / Tokenizer / Parser / Query Builder) is cleaner than two ~100-line files
that share every type.

### `bibtex.py` stays separate from `search.py`

Despite being used only by the search workflow, the BibTeX formatting is a distinct
concern with its own test suite. Keeps `search.py` focused on API interaction + CLI.
The name is unambiguous in context — the `*2bib.py` modules follow a different naming
pattern (named by source format).

## Entry point changes (`pyproject.toml`)

```toml
# Existing (path changes only)
raven-arxiv2id = "raven.papers.identifiers:main"
raven-arxiv-download = "raven.papers.download:main"
raven-burstbib = "raven.papers.burstbib:main"
raven-csv2bib = "raven.papers.csv2bib:main"
raven-pdf2bib = "raven.papers.pdf2bib:main"
raven-wos2bib = "raven.papers.wos2bib:main"

# New
raven-arxiv-search = "raven.papers.search:main"
```

## Dependency changes (`pyproject.toml`)

```toml
"feedparser>=6.0",    # papers: arXiv search
```

Already fixed in this session (were missing, now added):
```toml
"requests>=2.31",     # client, librarian, papers
"tqdm>=4.0",          # server.stt, papers
```

## Style adaptations for integrated code

- Version: `importlib.metadata.version(...)` → `from raven import __version__`
- CLI prog name: remove the hardcoded `prog="arxiv-api-search"` — all other Raven CLI
  tools let argparse auto-detect from `sys.argv[0]`, which gives the correct entry point
  name (e.g. `raven-arxiv-search`) automatically
- Imports: collapse internal cross-refs to intra-package imports
- `__all__` on every module
- Type hints on all functions
- Log message prefixes per style guide

## Other affected files

- `raven/common/stringmaps.py:84` — comment references `raven-arxiv2id` / `raven-arxiv-download`;
  update to new module paths.

## Not in scope

- Feature integration (e.g. search → download pipeline) — future opportunity.
- Tests for `identifiers.py`, `download.py`, or the `*2bib` converters — they have none
  today; adding them is orthogonal.
- Unifying `fetch_metadata()` (XML-based, download-specific) with `search()` (feedparser-based)
  — different parsers for different response shapes.

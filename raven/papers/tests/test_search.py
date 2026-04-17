"""Tests for arXiv search — query loading, output-path logic, paginated fetch."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from raven.papers import search as search_module
from raven.papers.search import (
    MAX_ARXIV_RESULTS,
    PAGE_SIZE,
    clamp_max_results,
    determine_output_path,
    load_query,
    search,
)


# ---- load_query -------------------------------------------------------------

class TestLoadQuery:
    """Resolve query text from the mutually-exclusive CLI inputs."""

    def test_inline_string(self):
        assert load_query(None, '"LLM" AND "AI"') == '"LLM" AND "AI"'

    def test_inline_string_stripped(self):
        assert load_query(None, "  foo  \n") == "foo"

    def test_from_file(self, tmp_path):
        f = tmp_path / "query.txt"
        f.write_text("  bar  \n")
        assert load_query(f, None) == "bar"

    def test_inline_wins_when_both_given(self):
        """If caller passes both (shouldn't happen — argparse prevents it), inline wins."""
        assert load_query(Path("/nonexistent"), "inline") == "inline"

    def test_neither_raises(self):
        with pytest.raises(ValueError, match="No query source"):
            load_query(None, None)

    def test_empty_inline_raises(self):
        with pytest.raises(ValueError, match="empty"):
            load_query(None, "   ")

    def test_empty_file_raises(self, tmp_path):
        f = tmp_path / "q.txt"
        f.write_text("")
        with pytest.raises(ValueError, match="empty"):
            load_query(f, None)


# ---- determine_output_path --------------------------------------------------

class TestDetermineOutputPath:
    """Cascading defaults for the ``.bib`` output path."""

    def test_explicit_output(self):
        assert determine_output_path(Path("/tmp/out.bib"), None) == Path("/tmp/out.bib")

    def test_explicit_output_wins_over_query_file(self):
        """``-o`` takes precedence over the ``<query_file>.bib`` default."""
        result = determine_output_path(Path("/tmp/out.bib"), Path("/tmp/q.txt"))
        assert result == Path("/tmp/out.bib")

    def test_query_file_derives_output(self):
        assert determine_output_path(None, Path("/tmp/q.txt")) == Path("/tmp/q.bib")

    def test_both_missing_default_name(self):
        assert determine_output_path(None, None) == Path("results.bib")


# ---- clamp_max_results ------------------------------------------------------

class TestClampMaxResults:
    def test_none_passes_through(self):
        assert clamp_max_results(None) is None

    def test_below_limit_unchanged(self):
        assert clamp_max_results(100) == 100

    def test_at_limit_unchanged(self):
        assert clamp_max_results(MAX_ARXIV_RESULTS) == MAX_ARXIV_RESULTS

    def test_above_limit_clamped(self):
        assert clamp_max_results(MAX_ARXIV_RESULTS + 5000) == MAX_ARXIV_RESULTS


# ---- Synthetic arXiv Atom feed ---------------------------------------------
#
# Minimal Atom feed mimicking what arXiv returns.  `feedparser` is forgiving;
# this is enough to exercise the pagination and error-detection branches.
def _atom_feed(total: int, entries: list[tuple[str, str]]) -> str:
    """Render an Atom feed with *total* opensearch totalResults and *entries*.

    Each entry is ``(arxiv_id, title)``.
    """
    entries_xml = "\n".join(
        textwrap.dedent(f"""\
            <entry>
              <id>http://arxiv.org/abs/{arxiv_id}</id>
              <title>{title}</title>
              <summary>Abstract for {title}.</summary>
              <author><name>Smith, A.</name></author>
              <published>2023-01-01T00:00:00Z</published>
            </entry>""")
        for arxiv_id, title in entries
    )
    return textwrap.dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom"
              xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
          <opensearch:totalResults>{total}</opensearch:totalResults>
        {entries_xml}
        </feed>""")


def _error_feed(message: str) -> str:
    """Render an arXiv API error feed."""
    return textwrap.dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <id>http://arxiv.org/api/errors</id>
            <summary>{message}</summary>
          </entry>
        </feed>""")


class _FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


def _no_wait_rate_limiter():
    """A RateLimiter substitute that never sleeps — keeps tests fast."""
    class _NoWait:
        def wait(self, show_progress=True):
            pass
    return _NoWait()


# ---- search() — mocked HTTP -------------------------------------------------

class TestSearchFunction:
    """Exercise ``search()`` against a canned ``requests.get``."""

    def _patch(self, pages, rate_limiter=None):
        """Context manager: patch ``requests.get`` to return *pages* sequentially.

        Each call to ``requests.get`` pops the next item from *pages*.  Also
        bypasses the arXiv rate limit by default so tests stay fast.
        """
        if rate_limiter is None:
            rate_limiter = _no_wait_rate_limiter
        responses = iter(pages)
        return patch.multiple(
            search_module,
            requests=_FakeRequests(lambda: next(responses)),
            RateLimiter=rate_limiter,
        )

    def test_single_page(self, capsys):
        feed = _atom_feed(total=2, entries=[("2301.00001", "First"),
                                            ("2301.00002", "Second")])
        with self._patch([_FakeResponse(feed)]):
            results = search("foo")
        assert len(results) == 2
        assert results[0].title == "First"
        assert results[1].title == "Second"

    def test_empty_feed(self):
        feed = _atom_feed(total=0, entries=[])
        with self._patch([_FakeResponse(feed)]):
            assert search("foo") == []

    def test_arxiv_api_error_raises(self):
        feed = _error_feed("Invalid query syntax.")
        with self._patch([_FakeResponse(feed)]):
            with pytest.raises(RuntimeError, match="Invalid query syntax"):
                search("foo")

    def test_pagination(self, monkeypatch):
        """When total > one page, ``search`` keeps fetching until exhausted."""
        monkeypatch.setattr(search_module, "PAGE_SIZE", 2)
        page1 = _atom_feed(total=3, entries=[("2301.00001", "A"), ("2301.00002", "B")])
        page2 = _atom_feed(total=3, entries=[("2301.00003", "C")])
        with self._patch([_FakeResponse(page1), _FakeResponse(page2)]):
            results = search("foo")
        assert [r.title for r in results] == ["A", "B", "C"]

    def test_max_results_caps_fetch(self, monkeypatch):
        """``max_results`` stops paging even when more are available."""
        monkeypatch.setattr(search_module, "PAGE_SIZE", 2)
        page1 = _atom_feed(total=10, entries=[("2301.00001", "A"), ("2301.00002", "B")])
        with self._patch([_FakeResponse(page1)]):
            results = search("foo", max_results=2)
        assert len(results) == 2

    def test_max_results_smaller_than_page_size(self, monkeypatch):
        """A small ``max_results`` shrinks the per-page request."""
        monkeypatch.setattr(search_module, "PAGE_SIZE", 100)
        captured = {}

        def capture(url, params, timeout):
            captured.update(params)
            return _FakeResponse(_atom_feed(total=50, entries=[("2301.00001", "A")]))

        with patch.multiple(
            search_module,
            requests=_FakeRequests(None, get_fn=capture),
            RateLimiter=_no_wait_rate_limiter,
        ):
            search("foo", max_results=1)
        assert captured["max_results"] == 1


class _FakeRequests:
    """Stand-in for the ``requests`` module inside ``search.py``."""

    def __init__(self, response_fn, get_fn=None):
        self._response_fn = response_fn
        self._get_fn = get_fn

    def get(self, url, params, timeout):
        if self._get_fn is not None:
            return self._get_fn(url, params, timeout)
        return self._response_fn()


# Assert a sanity check on the module-level constants that the tests reason about.
def test_constants_sane():
    assert PAGE_SIZE > 0
    assert MAX_ARXIV_RESULTS >= PAGE_SIZE

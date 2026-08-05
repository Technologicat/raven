"""Tests for arXiv identifier → BibTeX: identifier collection, batched fetch, version handling."""

from __future__ import annotations

import io
import textwrap
from unittest.mock import patch

import pytest

from raven.papers import arxiv2bib as arxiv2bib_module
from raven.papers.arxiv2bib import (
    BATCH_SIZE,
    _returned_identifiers,
    fetch_metadata,
    read_identifiers,
)
from raven.papers.bibtex import entries_to_bibtex


# ---- read_identifiers -------------------------------------------------------

class TestReadIdentifiers:
    """Collect identifiers from arguments, files or a stream, in that order of preference."""

    def test_literal_arguments(self):
        assert read_identifiers(["2301.00001", "2301.00002"]) == ["2301.00001", "2301.00002"]

    def test_versions_are_preserved(self):
        """The version is part of the identifier here — stripping belongs to the caller, not the reader."""
        assert read_identifiers(["2410.07866v5"]) == ["2410.07866v5"]

    def test_from_stream_when_no_arguments(self):
        stream = io.StringIO("2301.00001\n2301.00002\n")
        assert read_identifiers([], stream=stream) == ["2301.00001", "2301.00002"]

    def test_from_file(self, tmp_path):
        path = tmp_path / "ids.txt"
        path.write_text("2301.00001\n2301.00002\n")
        assert read_identifiers([str(path)]) == ["2301.00001", "2301.00002"]

    def test_files_and_literals_can_be_mixed(self, tmp_path):
        """An argument that names an existing file is read; anything else is taken literally."""
        path = tmp_path / "ids.txt"
        path.write_text("2301.00001\n")
        assert read_identifiers([str(path), "2301.00002"]) == ["2301.00001", "2301.00002"]

    def test_verbose_arxiv2id_output_pipes_in_unchanged(self):
        """`raven-arxiv2id --verbose` appends the filename; only the first field is the identifier."""
        stream = io.StringIO("2301.00001 Some Paper Title.pdf\n2301.00002 Another.pdf\n")
        assert read_identifiers([], stream=stream) == ["2301.00001", "2301.00002"]

    def test_blank_lines_and_comments_are_skipped(self):
        stream = io.StringIO("# a note\n\n2301.00001\n\n  # indented note\n2301.00002\n")
        assert read_identifiers([], stream=stream) == ["2301.00001", "2301.00002"]

    def test_duplicates_are_dropped_preserving_order(self):
        """A repeated identifier would otherwise buy a duplicate BibTeX entry."""
        stream = io.StringIO("2301.00002\n2301.00001\n2301.00002\n")
        assert read_identifiers([], stream=stream) == ["2301.00002", "2301.00001"]

    def test_empty_input_gives_empty_list(self):
        assert read_identifiers([], stream=io.StringIO("")) == []


# ---- fetch_metadata — mocked HTTP -------------------------------------------

def _atom_feed(entries: list[tuple[str, str]]) -> str:
    """Render an arXiv Atom feed. Each entry is ``(arxiv_id, title)``.

    The author is written first-name-first, as arXiv's Atom feed actually gives it ("Hsin-Ling Hsu"),
    because `bibtex._make_key` takes the last whitespace-separated word as the surname. A "Surname, I."
    fixture would key the entry on the initial and quietly make key assertions meaningless.
    """
    entries_xml = "\n".join(
        textwrap.dedent(f"""\
            <entry>
              <id>http://arxiv.org/abs/{arxiv_id}</id>
              <title>{title}</title>
              <summary>Abstract for {title}.</summary>
              <author><name>Ada Smith</name></author>
              <published>2023-01-01T00:00:00Z</published>
            </entry>""")
        for arxiv_id, title in entries
    )
    return textwrap.dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
        {entries_xml}
        </feed>""")


def _error_feed(message: str) -> str:
    return textwrap.dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <id>http://arxiv.org/api/errors</id>
            <summary>{message}</summary>
          </entry>
        </feed>""")


class _FakeResponse:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


class _FakeHttpfetch:
    """Records the params of every call, so batching can be asserted on rather than inferred."""

    def __init__(self, next_response):
        self._next_response = next_response
        self.calls: list[dict] = []

    def arxiv_get(self, url, params=None, timeout=None):
        self.calls.append(params or {})
        return self._next_response()


def _no_wait_rate_limiter():
    class _NoWait:
        def wait(self, show_progress=True):
            pass
    return _NoWait()


class TestFetchMetadata:
    """Exercise the batched fetch against a canned API."""

    def _patch(self, pages):
        responses = iter(pages)
        fake = _FakeHttpfetch(lambda: next(responses))
        return patch.multiple(arxiv2bib_module,
                              httpfetch=fake,
                              RateLimiter=_no_wait_rate_limiter), fake

    def test_single_batch(self, capsys):
        feed = _atom_feed([("2301.00001", "First"), ("2301.00002", "Second")])
        ctx, fake = self._patch([_FakeResponse(feed)])
        with ctx:
            entries = fetch_metadata(["2301.00001", "2301.00002"])
        assert [e.title for e in entries] == ["First", "Second"]
        assert len(fake.calls) == 1
        assert fake.calls[0]["id_list"] == "2301.00001,2301.00002"

    def test_batching_splits_the_id_list(self):
        """More identifiers than `batch_size` means several requests, each with its own slice."""
        page1 = _atom_feed([("2301.00001", "A"), ("2301.00002", "B")])
        page2 = _atom_feed([("2301.00003", "C")])
        ctx, fake = self._patch([_FakeResponse(page1), _FakeResponse(page2)])
        with ctx:
            entries = fetch_metadata(["2301.00001", "2301.00002", "2301.00003"], batch_size=2)
        assert [e.title for e in entries] == ["A", "B", "C"]
        assert [c["id_list"] for c in fake.calls] == ["2301.00001,2301.00002", "2301.00003"]

    def test_api_errors_are_raised_not_returned(self):
        """arXiv reports errors as a normal-looking entry, so an unchecked fetch 'succeeds' with junk."""
        ctx, _fake = self._patch([_FakeResponse(_error_feed("incorrect id format"))])
        with ctx, pytest.raises(RuntimeError, match="incorrect id format"):
            fetch_metadata(["not-an-id"])

    def test_missing_identifiers_do_not_discard_the_rest(self):
        """A withdrawn or mistyped identifier must not cost the several hundred that worked."""
        feed = _atom_feed([("2301.00001", "Only this one")])
        ctx, _fake = self._patch([_FakeResponse(feed)])
        with ctx:
            entries = fetch_metadata(["2301.00001", "2301.99999"])
        assert len(entries) == 1
        missing = {"2301.00001", "2301.99999"} - _returned_identifiers(entries)
        assert missing == {"2301.99999"}

    def test_default_batch_size_is_used_when_unspecified(self):
        feed = _atom_feed([("2301.00001", "A")])
        ctx, fake = self._patch([_FakeResponse(feed)])
        with ctx:
            fetch_metadata(["2301.00001"])
        assert len(fake.calls) == 1
        assert fake.calls[0]["max_results"] == 1
        assert BATCH_SIZE >= 1


# ---- version handling in the emitted BibTeX ---------------------------------

class TestVersionHandling:
    """`keep_versions` is what distinguishes this tool's output from a search's."""

    def _entries(self, ids_and_titles):
        import feedparser
        return feedparser.parse(_atom_feed(ids_and_titles)).entries

    def test_versions_are_kept_when_requested(self):
        bibtex = entries_to_bibtex(self._entries([("2410.07866v5", "Paper")]), keep_versions=True)
        assert "2410.07866v5" in bibtex

    def test_versions_are_stripped_by_default(self):
        """Unchanged behaviour for `raven-arxiv-search`, where the version is incidental."""
        bibtex = entries_to_bibtex(self._entries([("2410.07866v5", "Paper")]))
        assert "2410.07866v5" not in bibtex
        assert "2410.07866" in bibtex

    def test_two_versions_of_one_paper_stay_distinct(self):
        """The failure this prevents: a deduplicated collection collapsing back into one entry."""
        entries = self._entries([("2410.07866v1", "Draft"), ("2410.07866v5", "Final")])
        bibtex = entries_to_bibtex(entries, keep_versions=True)
        assert "2410.07866v1" in bibtex
        assert "2410.07866v5" in bibtex

    def test_stripping_versions_collapses_two_versions_into_the_newest(self):
        """One entry per paper is what stripping means; the surviving record is the higher version's.

        The failure this prevents is not a duplicate entry but a lost bibliography: `bibtexparser` turns a
        repeated key into a failed block and then raises `AttributeError` while writing it, so emitting
        both would take the whole file down with an error from inside the library.
        """
        entries = self._entries([("2410.07866v1", "Draft"), ("2410.07866v5", "Final")])
        bibtex = entries_to_bibtex(entries)
        assert bibtex.count("Smith_2023_2410.07866") == 1
        assert "Final" in bibtex
        assert "Draft" not in bibtex

    def test_the_newest_version_wins_regardless_of_input_order(self):
        entries = self._entries([("2410.07866v5", "Final"), ("2410.07866v1", "Draft")])
        bibtex = entries_to_bibtex(entries)
        assert "Final" in bibtex
        assert "Draft" not in bibtex

    def test_deduplication_preserves_the_order_of_the_rest(self):
        """A collapsed pair keeps its original slot rather than migrating to the end."""
        entries = self._entries([("2301.00001", "AlphaPaper"),
                                 ("2410.07866v1", "BetaDraft"),
                                 ("2301.00009", "GammaPaper"),
                                 ("2410.07866v5", "BetaFinal")])
        bibtex = entries_to_bibtex(entries)
        assert [t for t in ("AlphaPaper", "BetaFinal", "GammaPaper") if t in bibtex] == \
               sorted(("AlphaPaper", "BetaFinal", "GammaPaper"), key=bibtex.index)

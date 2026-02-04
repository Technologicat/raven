"""Unit tests for raven.common.utils."""

import os
import pathlib
import re
import time
import types

import numpy as np
import pytest

from raven.common import utils


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_name_parts(*, last, von=None, first=None, jr=None):
    """Create a mock bibtexparser NameParts-like object."""
    return types.SimpleNamespace(last=last,
                                 von=von or [],
                                 first=first or [],
                                 jr=jr or [])


# ---------------------------------------------------------------------------
# File utilities
# ---------------------------------------------------------------------------

class TestAbsolutizeFilename:
    def test_relative_becomes_absolute(self):
        result = utils.absolutize_filename("foo/bar.txt")
        assert os.path.isabs(result)

    def test_tilde_expanded(self):
        result = utils.absolutize_filename("~/foo.txt")
        assert "~" not in result

    def test_already_absolute(self):
        result = utils.absolutize_filename("/tmp/foo.txt")
        assert result == "/tmp/foo.txt"


class TestStripExt:
    def test_basic(self):
        assert utils.strip_ext("/foo/bar.bib") == "/foo/bar"

    def test_no_extension(self):
        assert utils.strip_ext("/foo/bar") == "/foo/bar"

    def test_multiple_dots(self):
        assert utils.strip_ext("/foo/bar.tar.gz") == "/foo/bar.tar"


class TestMakeCacheFilename:
    def test_basic(self):
        result = utils.make_cache_filename("data/papers.bib", "vectors", "npy")
        assert result == "data/papers_vectors.npy"

    def test_pathlib_input(self):
        result = utils.make_cache_filename(pathlib.Path("data/papers.bib"), "cache", "pkl")
        assert result == "data/papers_cache.pkl"

    def test_no_directory(self):
        result = utils.make_cache_filename("papers.bib", "vectors", "npy")
        assert result == "papers_vectors.npy"


class TestValidateCacheMtime:
    def test_cache_newer_than_original(self, tmp_path):
        orig = tmp_path / "orig.txt"
        orig.write_text("original")
        time.sleep(0.05)
        cache = tmp_path / "cache.txt"
        cache.write_text("cached")
        assert utils.validate_cache_mtime(str(cache), str(orig)) is True

    def test_cache_older_than_original(self, tmp_path):
        cache = tmp_path / "cache.txt"
        cache.write_text("cached")
        time.sleep(0.05)
        orig = tmp_path / "orig.txt"
        orig.write_text("original")
        assert utils.validate_cache_mtime(str(cache), str(orig)) is False


# ---------------------------------------------------------------------------
# Misc utilities
# ---------------------------------------------------------------------------

class TestMakeBlankIndexArray:
    def test_empty_int64(self):
        result = utils.make_blank_index_array()
        assert isinstance(result, np.ndarray)
        assert len(result) == 0
        assert result.dtype == np.int64


class TestEnvironOverride:
    def test_sets_and_restores_existing_var(self):
        os.environ["_RAVEN_TEST_VAR"] = "old"
        with utils.environ_override(_RAVEN_TEST_VAR="new"):
            assert os.environ["_RAVEN_TEST_VAR"] == "new"
        assert os.environ["_RAVEN_TEST_VAR"] == "old"
        del os.environ["_RAVEN_TEST_VAR"]

    def test_new_var_removed_after(self):
        key = "_RAVEN_TEST_NEWVAR"
        assert key not in os.environ
        with utils.environ_override(**{key: "value"}):
            assert os.environ[key] == "value"
        assert key not in os.environ

    def test_multiple_vars(self):
        with utils.environ_override(_RAVEN_A="1", _RAVEN_B="2"):
            assert os.environ["_RAVEN_A"] == "1"
            assert os.environ["_RAVEN_B"] == "2"
        assert "_RAVEN_A" not in os.environ
        assert "_RAVEN_B" not in os.environ


class TestMaybeOpen:
    def test_with_filename(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        with utils.maybe_open(str(f), "r", fallback=None) as fh:
            assert fh.read() == "hello"

    def test_with_none_uses_fallback(self):
        import io
        fallback = io.StringIO("fallback content")
        with utils.maybe_open(None, "r", fallback=fallback) as fh:
            assert fh.read() == "fallback content"


class TestUnionFilter:
    def test_matches_any_filter(self):
        import logging
        f = utils.UnionFilter(logging.Filter("raven.common"),
                              logging.Filter("raven.librarian"))
        record_common = logging.LogRecord("raven.common.utils", logging.INFO,
                                          "", 0, "msg", (), None)
        record_librarian = logging.LogRecord("raven.librarian.chat", logging.INFO,
                                             "", 0, "msg", (), None)
        record_other = logging.LogRecord("some.other.module", logging.INFO,
                                         "", 0, "msg", (), None)
        assert f.filter(record_common) is True
        assert f.filter(record_librarian) is True
        assert not f.filter(record_other)


# ---------------------------------------------------------------------------
# BibTeX author formatting
# ---------------------------------------------------------------------------

class TestFormatBibtexAuthor:
    def test_simple_last_name(self):
        author = _make_name_parts(last=["Knuth"])
        assert utils.format_bibtex_author(author) == "Knuth"

    def test_multi_word_last_name(self):
        author = _make_name_parts(last=["Brinch", "Hansen"])
        assert utils.format_bibtex_author(author) == "Brinch Hansen"

    def test_von_part(self):
        author = _make_name_parts(von=["van"], last=["Beethoven"])
        assert utils.format_bibtex_author(author) == "van Beethoven"

    def test_jr_part(self):
        author = _make_name_parts(last=["Beeblebrox"], jr=["IV"])
        assert utils.format_bibtex_author(author) == "Beeblebrox IV"

    def test_all_parts(self):
        author = _make_name_parts(von=["de", "la"], last=["Cruz"], jr=["III"], first=["Juan"])
        assert utils.format_bibtex_author(author) == "de la Cruz III"

    def test_empty_last_raises(self):
        author = _make_name_parts(last=[])
        with pytest.raises(ValueError, match="missing last name"):
            utils.format_bibtex_author(author)


class TestFormatBibtexAuthors:
    def test_single_author(self):
        authors = [_make_name_parts(last=["Knuth"])]
        assert utils.format_bibtex_authors(authors) == "Knuth"

    def test_two_authors(self):
        authors = [_make_name_parts(last=["Knuth"]),
                    _make_name_parts(last=["Lamport"])]
        assert utils.format_bibtex_authors(authors) == "Knuth and Lamport"

    def test_three_authors_et_al(self):
        authors = [_make_name_parts(last=["Knuth"]),
                    _make_name_parts(last=["Lamport"]),
                    _make_name_parts(last=["Dijkstra"])]
        assert utils.format_bibtex_authors(authors) == "Knuth et al."

    def test_empty_list(self):
        assert utils.format_bibtex_authors([]) == ""

    def test_invalid_author_returns_empty(self):
        # An author with no last name should cause a warning and return ""
        authors = [_make_name_parts(last=[])]
        assert utils.format_bibtex_authors(authors) == ""


# ---------------------------------------------------------------------------
# String normalization
# ---------------------------------------------------------------------------

class TestNormalizeWhitespace:
    def test_multiple_spaces(self):
        assert utils.normalize_whitespace("hello   world") == "hello world"

    def test_tabs_and_newlines(self):
        assert utils.normalize_whitespace("hello\t\nworld") == "hello world"

    def test_leading_trailing(self):
        assert utils.normalize_whitespace("  hello  ") == "hello"

    def test_empty_string(self):
        assert utils.normalize_whitespace("") == ""

    def test_already_normal(self):
        assert utils.normalize_whitespace("hello world") == "hello world"


class TestNormalizeUnicode:
    def test_nfkc_normalization(self):
        # Fullwidth "Ａ" (U+FF21) → regular "A"
        assert utils.normalize_unicode("\uff21") == "A"

    def test_regular_ascii_unchanged(self):
        assert utils.normalize_unicode("hello") == "hello"

    def test_compatibility_superscript(self):
        # Unicode superscript 2 (U+00B2) stays as ² in NFKC
        # But ﬁ (U+FB01, fi ligature) → "fi"
        assert utils.normalize_unicode("\ufb01") == "fi"


class TestUnicodizeBasicMarkup:
    def test_html_subscript(self):
        assert utils.unicodize_basic_markup("CO<sub>2</sub>") == "CO₂"

    def test_html_superscript(self):
        assert utils.unicodize_basic_markup("x<sup>2</sup>") == "x²"

    def test_latex_percent(self):
        assert utils.unicodize_basic_markup(r"100\%") == "100%"

    def test_latex_dollar(self):
        assert utils.unicodize_basic_markup(r"\$5") == "$5"

    def test_html_entity_le(self):
        assert utils.unicodize_basic_markup("x &le; y") == "x ≤ y"

    def test_html_entity_ge(self):
        assert utils.unicodize_basic_markup("x &ge; y") == "x ≥ y"

    def test_html_entity_auml(self):
        assert utils.unicodize_basic_markup("&auml;") == "ä"

    def test_html_entity_ouml(self):
        assert utils.unicodize_basic_markup("&Ouml;") == "Ö"

    def test_html_bold(self):
        assert utils.unicodize_basic_markup("<b>bold</b>") == "*bold*"

    def test_html_italic(self):
        assert utils.unicodize_basic_markup("<i>italic</i>") == "/italic/"

    def test_html_underline(self):
        assert utils.unicodize_basic_markup("<u>underline</u>") == "_underline_"

    def test_lt_gt_entities_last(self):
        # &lt; and &gt; should be replaced after HTML tags are processed
        assert utils.unicodize_basic_markup("a &lt; b &gt; c") == "a < b > c"

    def test_combined(self):
        result = utils.unicodize_basic_markup("H<sub>2</sub>O at 100&le;T")
        assert result == "H₂O at 100≤T"


# ---------------------------------------------------------------------------
# Search utilities
# ---------------------------------------------------------------------------

class TestNormalizeSearchString:
    def test_whitespace_and_unicode(self):
        assert utils.normalize_search_string("  hello   world  ") == "hello world"

    def test_subscript_to_regular(self):
        assert utils.normalize_search_string("O₂") == "O2"

    def test_superscript_to_regular(self):
        assert utils.normalize_search_string("x²") == "x2"

    def test_mixed(self):
        assert utils.normalize_search_string("CO₂  and  x²") == "CO2 and x2"


class TestSearchStringToFragments:
    def test_all_lowercase_is_case_insensitive(self):
        cs, ci = utils.search_string_to_fragments("cat photo", sort=False)
        assert cs == []
        assert ci == ["cat", "photo"]

    def test_uppercase_is_case_sensitive(self):
        cs, ci = utils.search_string_to_fragments("Cat photo", sort=False)
        assert cs == ["Cat"]
        assert ci == ["photo"]

    def test_sort_longest_first(self):
        cs, ci = utils.search_string_to_fragments("a bb ccc", sort=True)
        assert ci == ["ccc", "bb", "a"]

    def test_unsorted_preserves_order(self):
        cs, ci = utils.search_string_to_fragments("a bb ccc", sort=False)
        assert ci == ["a", "bb", "ccc"]

    def test_empty_string(self):
        cs, ci = utils.search_string_to_fragments("", sort=False)
        assert cs == []
        assert ci == []

    def test_single_fragment(self):
        cs, ci = utils.search_string_to_fragments("photocatalytic", sort=False)
        assert ci == ["photocatalytic"]

    def test_subscripts_normalized(self):
        cs, ci = utils.search_string_to_fragments("CO₂", sort=False)
        # "CO2" has uppercase letters, so it's case-sensitive
        assert cs == ["CO2"]
        assert ci == []


class TestSearchFragmentToHighlightRegex:
    def test_parens_escaped(self):
        result = utils.search_fragment_to_highlight_regex_fragment("f(x)")
        assert r"\(" in result
        assert r"\)" in result

    def test_dot_escaped(self):
        result = utils.search_fragment_to_highlight_regex_fragment("e.g.")
        assert r"\." in result

    def test_brackets_escaped(self):
        result = utils.search_fragment_to_highlight_regex_fragment("[a]")
        assert r"\[" in result
        assert r"\]" in result

    def test_digits_expanded(self):
        result = utils.search_fragment_to_highlight_regex_fragment("H2O")
        # "2" should be expanded to include subscript and superscript variants
        assert "₂" in result
        assert "²" in result

    def test_result_is_valid_regex(self):
        result = utils.search_fragment_to_highlight_regex_fragment("CO2")
        # Should compile without error
        re.compile(result)


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

class TestChunkifyText:
    def test_short_text_single_chunk(self):
        chunks = utils.chunkify_text("hello", chunk_size=100, overlap=0, extra=0.4)
        assert len(chunks) == 1
        assert chunks[0]["text"] == "hello"
        assert chunks[0]["chunk_id"] == 0
        assert chunks[0]["offset"] == 0

    def test_exact_chunk_size_single_chunk(self):
        text = "a" * 100
        chunks = utils.chunkify_text(text, chunk_size=100, overlap=0, extra=0.4)
        assert len(chunks) == 1

    def test_multiple_chunks(self):
        text = "a" * 300
        chunks = utils.chunkify_text(text, chunk_size=100, overlap=0, extra=0.0)
        assert len(chunks) == 3
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_id"] == i

    def test_overlap_shared_content(self):
        text = "abcdefghij" * 10  # 100 chars
        chunks = utils.chunkify_text(text, chunk_size=40, overlap=10, extra=0.0)
        assert len(chunks) > 1
        # The end of chunk[0] should appear at the start of chunk[1]
        overlap_from_first = chunks[0]["text"][-10:]
        assert chunks[1]["text"].startswith(overlap_from_first)

    def test_zero_overlap(self):
        text = "a" * 200
        chunks = utils.chunkify_text(text, chunk_size=100, overlap=0, extra=0.0)
        total_len = sum(len(c["text"]) for c in chunks)
        assert total_len == 200

    def test_orphan_control_folds_remainder(self):
        # 130 chars with chunk_size=100 and extra=0.4:
        # remainder is 30, which is < 0.4*100=40, so it should fold into one chunk
        text = "a" * 130
        chunks = utils.chunkify_text(text, chunk_size=100, overlap=0, extra=0.4)
        assert len(chunks) == 1
        assert len(chunks[0]["text"]) == 130

    def test_orphan_control_separate_chunk(self):
        # 160 chars with chunk_size=100 and extra=0.1:
        # remainder is 60, which is > 0.1*100=10, so it becomes a separate chunk
        text = "a" * 160
        chunks = utils.chunkify_text(text, chunk_size=100, overlap=0, extra=0.1)
        assert len(chunks) == 2

    def test_offsets_correct(self):
        text = "a" * 300
        chunks = utils.chunkify_text(text, chunk_size=100, overlap=0, extra=0.0)
        for chunk in chunks:
            assert text[chunk["offset"]:chunk["offset"] + len(chunk["text"])] == chunk["text"]

    def test_chunk_ids_sequential(self):
        text = "a" * 500
        chunks = utils.chunkify_text(text, chunk_size=100, overlap=0, extra=0.0)
        ids = [c["chunk_id"] for c in chunks]
        assert ids == list(range(len(chunks)))

    def test_with_trimmer(self):
        # Trimmer that strips 5 chars from the beginning (except first chunk) and 5 from the end (except last chunk)
        def trimmer(overlap, mode, text):
            if mode == "first":
                return text[:-5], 0
            elif mode == "last":
                return text[5:], 5
            else:  # middle
                return text[5:-5], 5
        text = "a" * 300
        chunks = utils.chunkify_text(text, chunk_size=100, overlap=10, extra=0.0, trimmer=trimmer)
        assert len(chunks) >= 2
        # First chunk should have trimmed end
        assert len(chunks[0]["text"]) == 95

    def test_empty_text(self):
        chunks = utils.chunkify_text("", chunk_size=100, overlap=0, extra=0.4)
        assert len(chunks) == 1
        assert chunks[0]["text"] == ""

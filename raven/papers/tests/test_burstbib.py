"""Tests for BibTeX bursting — header detection and slug extraction."""

import pytest

from raven.papers.burstbib import is_headerline, get_slug


class TestIsHeaderline:
    """Verify BibTeX record header detection."""

    def test_article_header(self):
        assert is_headerline("@article{Smith_2021,") == "@article{Smith_2021,"

    def test_inproceedings_header(self):
        assert is_headerline("@inproceedings{key123,") == "@inproceedings{key123,"

    def test_book_header(self):
        assert is_headerline("@book{Knuth_1997,") == "@book{Knuth_1997,"

    def test_misc_header(self):
        assert is_headerline("@misc{rfc2616,") == "@misc{rfc2616,"

    def test_strips_whitespace(self):
        assert is_headerline("  @article{key,  ") == "@article{key,"

    def test_field_line_not_header(self):
        assert is_headerline("  author = {Smith},") is False

    def test_closing_brace_not_header(self):
        assert is_headerline("}") is False

    def test_empty_line_not_header(self):
        assert is_headerline("") is False

    def test_blank_line_not_header(self):
        assert is_headerline("   ") is False

    def test_at_without_trailing_comma(self):
        """A line starting with @ but not ending with , is not a header."""
        assert is_headerline("@article{key}") is False

    def test_comment_not_header(self):
        assert is_headerline("% @article{key,") is False


class TestGetSlug:
    """Verify BibTeX key extraction and filename sanitization."""

    def test_basic_slug(self):
        assert get_slug("@article{Smith_2021,") == "Smith_2021"

    def test_slug_with_numbers(self):
        assert get_slug("@article{Smith_2021_2103.12345,") == "Smith_2021_2103.12345"

    def test_slug_with_hyphen(self):
        assert get_slug("@article{Smith-Jones_2021,") == "Smith-Jones_2021"

    def test_doi_slug_sanitized(self):
        """DOIs used as keys have slashes and colons — those get stripped."""
        result = get_slug("@article{10.1234/journal:2021,")
        assert "/" not in result
        assert ":" not in result

    def test_url_slug_sanitized(self):
        """URLs used as keys have slashes and colons — those get stripped."""
        result = get_slug("@article{https://example.com/paper,")
        assert "/" not in result
        assert ":" not in result

    def test_preserves_safe_chars(self):
        """Alphanumerics, spaces, hyphens, underscores, commas, apostrophes, dots are kept."""
        result = get_slug("@article{Author's Paper-Title_v2.1,")
        assert result == "Author's Paper-Title_v2.1"

    def test_missing_brace_asserts(self):
        with pytest.raises(AssertionError):
            get_slug("@article Smith_2021,")

    def test_missing_comma_asserts(self):
        with pytest.raises(AssertionError):
            get_slug("@article{Smith_2021}")

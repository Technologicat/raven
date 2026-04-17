"""Tests for BibTeX bursting — header detection, slug extraction, splitting."""

import textwrap

import pytest

from raven.papers.burstbib import is_headerline, get_slug, burst_bibtex


# ---- Synthetic BibTeX fixture -----------------------------------------------

SAMPLE_BIBTEX = textwrap.dedent("""\
    % Hand-crafted synthetic BibTeX sample for burstbib tests.
    @article{Smith_2021,
        author = {Smith, Alice},
        title = {A Paper},
        year = {2021},
    }
    @book{Knuth_1997,
        author = {Knuth, Donald E.},
        title = {The Art of Computer Programming},
        year = {1997},
    }
    @misc{rfc2616,
        author = {Fielding, Roy T.},
        title = {Hypertext Transfer Protocol},
        year = {1999},
    }
""")


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


class TestBurstBibtex:
    """Verify splitting a multi-record BibTeX source into individual entries."""

    def test_empty_source(self):
        assert burst_bibtex("") == []

    def test_whitespace_only(self):
        assert burst_bibtex("\n\n   \n") == []

    def test_no_headers(self):
        """Leading junk without any ``@type{key,`` header yields nothing."""
        assert burst_bibtex("% just a comment\nsome text\n") == []

    def test_three_records(self):
        result = burst_bibtex(SAMPLE_BIBTEX)
        assert [slug for slug, _ in result] == ["Smith_2021", "Knuth_1997", "rfc2616"]

    def test_record_content_preserved(self):
        """Each record text spans from its header to just before the next header."""
        result = burst_bibtex(SAMPLE_BIBTEX)
        slugs = dict(result)
        assert slugs["Smith_2021"].startswith("@article{Smith_2021,")
        assert "author = {Smith, Alice}" in slugs["Smith_2021"]
        assert "Knuth" not in slugs["Smith_2021"]  # doesn't bleed into next
        assert slugs["Knuth_1997"].startswith("@book{Knuth_1997,")
        assert slugs["rfc2616"].startswith("@misc{rfc2616,")

    def test_single_record(self):
        source = "@article{only_one,\n  year = {2021},\n}\n"
        result = burst_bibtex(source)
        assert len(result) == 1
        slug, text = result[0]
        assert slug == "only_one"
        assert text == source

    def test_leading_noise_skipped(self):
        """Any pre-record content (comments, @string, blank lines) is discarded."""
        source = textwrap.dedent("""\
            % Generated on some day
            @string{journal = "X"}
            @article{real_entry,
              year = {2021},
            }
        """)
        result = burst_bibtex(source)
        assert len(result) == 1
        assert result[0][0] == "real_entry"
        assert "@string" not in result[0][1]

    def test_order_preserved(self):
        """Records are emitted in source order."""
        source = "@article{a,\n}\n@article{b,\n}\n@article{c,\n}\n"
        assert [slug for slug, _ in burst_bibtex(source)] == ["a", "b", "c"]

    def test_sanitized_slug(self):
        """Slugs with non-filesystem-safe characters come back sanitized."""
        source = "@article{10.1234/journal:2021,\n  year = {2021},\n}\n"
        result = burst_bibtex(source)
        assert len(result) == 1
        slug = result[0][0]
        assert "/" not in slug and ":" not in slug
        assert "10.1234" in slug and "journal" in slug and "2021" in slug

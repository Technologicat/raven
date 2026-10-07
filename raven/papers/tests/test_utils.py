"""Tests for shared bibliography utilities."""

import bibtexparser
from bibtexparser.model import Entry, Field
from bibtexparser import Library

import pytest

from raven.papers.utils import bibtex_escape, bibtex_unescape, normalize_doi, paper_url


class TestBibtexEscape:
    """Verify that bibtex_escape produces valid BibTeX field values."""

    def _roundtrip(self, text: str) -> str:
        """Write text as a BibTeX field, parse it back, return the parsed value."""
        lib = Library()
        escaped = bibtex_escape(text)
        lib.add(Entry("article", "test", fields=[
            Field("abstract", f"{{{escaped}}}"),
        ]))
        bib_str = bibtexparser.write_string(lib)
        parsed = bibtexparser.parse_string(bib_str)
        assert not parsed.failed_blocks, f"BibTeX parse failed for input {text!r}"
        return parsed.entries[0].fields_dict["abstract"].value

    def test_plain_text_unchanged(self):
        assert bibtex_escape("hello world") == "hello world"

    def test_backslash(self):
        assert bibtex_escape("a \\ b") == "a \\\\ b"

    def test_braces(self):
        assert bibtex_escape("{text}") == r"\{text\}"

    def test_percent(self):
        assert bibtex_escape("20% more") == r"20\% more"

    def test_ampersand(self):
        assert bibtex_escape("A & B") == r"A \& B"

    def test_hash(self):
        assert bibtex_escape("sample #1") == r"sample \#1"

    def test_dollar(self):
        assert bibtex_escape("costs $5") == r"costs \$5"

    def test_brackets(self):
        assert bibtex_escape("[note]") == "{[}note{]}"

    # -- Round-trip tests: write to BibTeX, parse back -----------------------

    def test_roundtrip_plain(self):
        val = self._roundtrip("plain text")
        assert "plain text" in val

    def test_roundtrip_lone_opening_brace(self):
        """The original bug: a lone { in source text broke bibtexparser parsing."""
        val = self._roundtrip("text { more")
        assert val  # parsed without error

    def test_roundtrip_lone_closing_brace(self):
        val = self._roundtrip("text } more")
        assert val

    def test_roundtrip_matched_braces(self):
        val = self._roundtrip("hydrogen {H2} storage")
        assert val

    def test_roundtrip_hash_in_text(self):
        val = self._roundtrip("sample #1 result")
        assert val

    def test_roundtrip_percent(self):
        val = self._roundtrip("20% increase in yield")
        assert val

    def test_roundtrip_multiple_specials(self):
        val = self._roundtrip("H{2} costs $5 & is 20% of #1")
        assert val


class TestBibtexUnescape:
    """Verify that bibtex_unescape reverses bibtex_escape."""

    def test_roundtrip_plain(self):
        assert bibtex_unescape(bibtex_escape("hello")) == "hello"

    def test_roundtrip_all_specials(self):
        original = r"a \ b { c } d & e % f # g $ h [ i"
        assert bibtex_unescape(bibtex_escape(original)) == original

    def test_roundtrip_backslash_then_brace(self):
        """Tricky: \\{ in source — backslash escapes first, then brace."""
        original = r"\{"
        assert bibtex_unescape(bibtex_escape(original)) == original

    def test_individual_unescapes(self):
        assert bibtex_unescape(r"\%") == "%"
        assert bibtex_unescape(r"\$") == "$"
        assert bibtex_unescape(r"\#") == "#"
        assert bibtex_unescape(r"\&") == "&"
        assert bibtex_unescape(r"\{") == "{"
        assert bibtex_unescape(r"\}") == "}"
        assert bibtex_unescape("\\\\") == "\\"
        assert bibtex_unescape("{[}") == "["
        assert bibtex_unescape("{]}") == "]"


class TestNormalizeDoi:
    def test_case_is_folded(self):
        assert normalize_doi("10.1234/ABC-def") == "10.1234/abc-def"

    @pytest.mark.parametrize("prefix", ["https://doi.org/", "http://doi.org/", "https://dx.doi.org/",
                                        "http://dx.doi.org/", "doi:", "doi: ", "info:doi/"])
    def test_resolver_prefixes_are_stripped(self, prefix):
        assert normalize_doi(f"{prefix}10.1234/abc") == "10.1234/abc"

    @pytest.mark.parametrize("dash", ["‐", "‑", "‒", "–", "—", "―", "−"])
    def test_every_unicode_dash_folds_to_ascii(self, dash):
        """Two databases exporting one DOI disagree about which dash it contains; the paper is one paper."""
        assert normalize_doi(f"10.1234/abc{dash}def") == "10.1234/abc-def"

    def test_a_line_wrapped_doi_loses_its_whitespace(self):
        assert normalize_doi("10.1234/abc\n  def") == "10.1234/abcdef"

    def test_trailing_sentence_punctuation_is_dropped(self):
        assert normalize_doi("10.1234/abc.") == "10.1234/abc"

    def test_enclosing_braces_are_dropped(self):
        assert normalize_doi("{10.1234/abc}") == "10.1234/abc"

    @pytest.mark.parametrize("value", ["", None, "   ", "n/a", "N/A", "not available",
                                       "https://example.com/article/123", "10.1234", "10.1234/",
                                       "10.12/x", "doi", "-"])
    def test_what_is_not_a_doi_is_refused(self, value):
        """A `doi` field regularly holds something that is not one, and those must not become a key.

        Two records both saying `n/a` are equal to each other, so admitting the value would merge papers
        with nothing whatsoever in common — the worst failure this tool has, since a merged record is
        gone from the review and nothing downstream can notice.
        """
        assert normalize_doi(value) is None

    def test_a_suffix_full_of_punctuation_is_still_a_doi(self):
        # Real DOIs carry slashes, parentheses and dots in the suffix; the shape check must not be
        # so tight that it starts rejecting the thing it is checking for.
        assert normalize_doi("10.1002/(SICI)1097-0258(19980815)17:15<1661::AID-SIM968>3.0.CO;2-2") \
            == "10.1002/(sici)1097-0258(19980815)17:15<1661::aid-sim968>3.0.co;2-2"


class TestPaperUrl:
    """The link a reviewer clicks: by DOI when the record has one, by its `url` otherwise, else nothing."""

    def test_a_doi_wins_over_a_url(self):
        assert paper_url("10.1234/ABC", "https://publisher.example/abc") == "https://doi.org/10.1234/abc"

    def test_a_doi_written_as_a_resolver_url_is_not_doubled(self):
        assert paper_url("https://doi.org/10.1234/abc", None) == "https://doi.org/10.1234/abc"

    def test_without_a_doi_the_url_is_used(self):
        assert paper_url(None, "https://publisher.example/abc") == "https://publisher.example/abc"

    def test_a_doi_field_holding_something_else_falls_through_to_the_url(self):
        assert paper_url("n/a", " {https://publisher.example/abc} ") == "https://publisher.example/abc"

    @pytest.mark.parametrize("doi, url", [(None, None), ("", ""), ("n/a", "not a link"), (None, "ftp://x/y")])
    def test_with_neither_the_link_is_empty(self, doi, url):
        assert paper_url(doi, url) == ""

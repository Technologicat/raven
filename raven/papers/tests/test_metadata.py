"""Tests for arXiv metadata fetching — XML parsing and filename generation."""

from unittest.mock import patch, MagicMock

from raven.papers.download import get_paper_metadata

# Minimal arXiv API response for a new-style paper
ARXIV_RESPONSE_NEW = b"""\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2301.12345v2</id>
    <title>Attention Is All You Need (Maybe)</title>
    <summary>We investigate whether attention is indeed all you need.</summary>
    <published>2023-01-15T00:00:00Z</published>
    <updated>2023-06-20T00:00:00Z</updated>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Jones</name></author>
    <link title="pdf" rel="related" type="application/pdf"
          href="http://arxiv.org/pdf/2301.12345v2"/>
  </entry>
</feed>
"""

# Old-style arXiv ID
ARXIV_RESPONSE_OLD = b"""\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/hep-th/0601001v1</id>
    <title>String Theory and the Real World</title>
    <summary>A review of string theory phenomenology.</summary>
    <published>2006-01-02T00:00:00Z</published>
    <updated>2006-01-02T00:00:00Z</updated>
    <author><name>Carol Williams</name></author>
    <link title="pdf" rel="related" type="application/pdf"
          href="http://arxiv.org/pdf/hep-th/0601001v1"/>
  </entry>
</feed>
"""

# Three authors (triggers "et al.")
ARXIV_RESPONSE_THREE_AUTHORS = b"""\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2301.00001v1</id>
    <title>Collaborative Research</title>
    <summary>A paper with three authors.</summary>
    <published>2023-01-01T00:00:00Z</published>
    <updated>2023-01-01T00:00:00Z</updated>
    <author><name>Alice</name></author>
    <author><name>Bob</name></author>
    <author><name>Carol</name></author>
    <link title="pdf" rel="related" type="application/pdf"
          href="http://arxiv.org/pdf/2301.00001v1"/>
  </entry>
</feed>
"""

# No PDF link
ARXIV_RESPONSE_NO_PDF = b"""\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2301.99999v1</id>
    <title>Invisible Paper</title>
    <summary>No PDF available.</summary>
    <published>2023-01-01T00:00:00Z</published>
    <updated>2023-01-01T00:00:00Z</updated>
    <author><name>Nobody</name></author>
  </entry>
</feed>
"""


def _mock_response(content: bytes) -> MagicMock:
    """Create a mock requests.Response with the given XML content."""
    resp = MagicMock()
    resp.content = content
    resp.raise_for_status = MagicMock()
    return resp


class TestGetPaperMetadata:
    """Verify metadata extraction from arXiv API XML responses."""

    def _fetch(self, arxiv_id: str, content: bytes, **kwargs) -> dict:
        with patch("raven.papers.download.requests.get", return_value=_mock_response(content)):
            return get_paper_metadata(arxiv_id, **kwargs)

    # -- Basic field extraction --

    def test_title(self):
        md = self._fetch("2301.12345", ARXIV_RESPONSE_NEW)
        assert md["title"] == "Attention Is All You Need (Maybe)"

    def test_abstract(self):
        md = self._fetch("2301.12345", ARXIV_RESPONSE_NEW)
        assert "attention" in md["abstract"].lower()

    def test_authors_two(self):
        md = self._fetch("2301.12345", ARXIV_RESPONSE_NEW)
        assert md["authors"] == "Alice Smith and Bob Jones"

    def test_authors_three_et_al(self):
        md = self._fetch("2301.00001", ARXIV_RESPONSE_THREE_AUTHORS)
        assert md["authors"] == "Alice and Bob et al."

    def test_original_year(self):
        md = self._fetch("2301.12345", ARXIV_RESPONSE_NEW)
        assert md["original_year"] == "2023"

    def test_version_year(self):
        md = self._fetch("2301.12345", ARXIV_RESPONSE_NEW)
        assert md["version_year"] == "2023"

    def test_version_extracted(self):
        md = self._fetch("2301.12345", ARXIV_RESPONSE_NEW)
        assert md["version"] == "v2"

    def test_pdf_url(self):
        md = self._fetch("2301.12345", ARXIV_RESPONSE_NEW)
        assert md["pdf_url"] == "http://arxiv.org/pdf/2301.12345v2"

    def test_no_pdf_url(self):
        md = self._fetch("2301.99999", ARXIV_RESPONSE_NO_PDF)
        assert md["pdf_url"] is None

    # -- ID resolution --

    def test_resolved_id_includes_version(self):
        md = self._fetch("2301.12345", ARXIV_RESPONSE_NEW)
        assert md["resolved_id"] == "2301.12345v2"

    def test_original_id_preserved(self):
        md = self._fetch("2301.12345", ARXIV_RESPONSE_NEW)
        assert md["original_id"] == "2301.12345"

    # -- Old-style IDs --

    def test_old_style_resolved_id(self):
        md = self._fetch("hep-th/0601001", ARXIV_RESPONSE_OLD)
        assert md["resolved_id"] == "hep-th/0601001v1"

    def test_old_style_year(self):
        md = self._fetch("hep-th/0601001", ARXIV_RESPONSE_OLD)
        assert md["original_year"] == "2006"

    # -- Filename generation --

    def test_filename_contains_authors(self):
        md = self._fetch("2301.12345", ARXIV_RESPONSE_NEW)
        assert "Alice Smith and Bob Jones" in md["filename"]

    def test_filename_contains_year(self):
        md = self._fetch("2301.12345", ARXIV_RESPONSE_NEW)
        assert "(2023" in md["filename"]

    def test_filename_contains_id(self):
        md = self._fetch("2301.12345", ARXIV_RESPONSE_NEW)
        assert "2301.12345v2" in md["filename"]

    def test_filename_ends_with_pdf(self):
        md = self._fetch("2301.12345", ARXIV_RESPONSE_NEW)
        assert md["filename"].endswith(".pdf")

    def test_filename_colon_replaced(self):
        """Colons in titles are replaced with hyphens for filesystem safety."""
        md = self._fetch("2301.12345", ARXIV_RESPONSE_NEW)
        assert ": " not in md["filename"]

    def test_filename_revised_year_shown(self):
        """When version year differs from original, 'revised YYYY' appears."""
        # In this fixture both years are 2023, so no "revised" — let's build
        # a response with different years.
        xml = ARXIV_RESPONSE_NEW.replace(
            b"<updated>2023-06-20T00:00:00Z</updated>",
            b"<updated>2024-03-01T00:00:00Z</updated>",
        )
        md = self._fetch("2301.12345", xml)
        assert "revised 2024" in md["filename"]

    def test_filename_no_revised_when_same_year(self):
        md = self._fetch("hep-th/0601001", ARXIV_RESPONSE_OLD)
        assert "revised" not in md["filename"]

    def test_title_length_limit(self):
        md = self._fetch("2301.12345", ARXIV_RESPONSE_NEW, title_length_limit=10)
        # Title is truncated to 10 chars + "..."
        assert "..." in md["filename"]

    def test_old_style_filename_slash_replaced(self):
        """Old-style IDs have / replaced in the filename."""
        md = self._fetch("hep-th/0601001", ARXIV_RESPONSE_OLD)
        # The filename itself should not contain a bare /
        assert "hep-th/" not in md["filename"]
        assert "hep-th_0601001" in md["filename"]

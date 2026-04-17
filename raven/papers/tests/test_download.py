"""Tests for the arXiv download tool — BibTeX ID extraction and deduplication."""

import textwrap
from unittest.mock import patch

import pytest

from raven.papers import download as download_module
from raven.papers.download import (
    download_papers,
    extract_ids_from_bib,
    format_filename,
    get_paper_metadata,
    parse_metadata_response,
)
from raven.papers.utils import deduplicate_arxiv_ids


def test_extracts_eprint_fields(tmp_path):
    bib = tmp_path / "test.bib"
    bib.write_text(textwrap.dedent("""\
        @article{Smith_2021_2103.12345,
            author = {Alice Smith},
            title = {A Paper},
            eprint = {2103.12345},
            archiveprefix = {arXiv},
        }
        @article{Jones_2022_2205.67890,
            author = {Bob Jones},
            title = {Another Paper},
            eprint = {2205.67890},
            archiveprefix = {arXiv},
        }
    """))
    ids = extract_ids_from_bib(str(bib))
    assert ids == ["2103.12345", "2205.67890"]


def test_skips_entries_without_eprint(tmp_path):
    bib = tmp_path / "test.bib"
    bib.write_text(textwrap.dedent("""\
        @article{nopreprint,
            author = {Carol},
            title = {No Preprint},
            journal = {Some Journal},
        }
        @article{haspreprint,
            author = {Dave},
            title = {Has Preprint},
            eprint = {2301.00001},
            archiveprefix = {arXiv},
        }
    """))
    ids = extract_ids_from_bib(str(bib))
    assert ids == ["2301.00001"]


def test_skips_non_arxiv_eprints(tmp_path):
    bib = tmp_path / "test.bib"
    bib.write_text(textwrap.dedent("""\
        @article{ssrn_paper,
            author = {Eve},
            title = {SSRN Paper},
            eprint = {12345},
            archiveprefix = {SSRN},
        }
        @article{arxiv_paper,
            author = {Frank},
            title = {arXiv Paper},
            eprint = {2401.99999},
            archiveprefix = {arXiv},
        }
    """))
    ids = extract_ids_from_bib(str(bib))
    assert ids == ["2401.99999"]


def test_accepts_missing_archiveprefix(tmp_path):
    """An eprint without archiveprefix is assumed to be arXiv."""
    bib = tmp_path / "test.bib"
    bib.write_text(textwrap.dedent("""\
        @article{test,
            author = {Grace},
            title = {Implicit arXiv},
            eprint = {2501.11111},
        }
    """))
    ids = extract_ids_from_bib(str(bib))
    assert ids == ["2501.11111"]


def test_empty_bib_returns_empty(tmp_path):
    bib = tmp_path / "test.bib"
    bib.write_text("")
    ids = extract_ids_from_bib(str(bib))
    assert ids == []


def test_deduplicates_versions_in_bib(tmp_path):
    """When a .bib has the same paper in multiple versions, keep the highest."""
    bib = tmp_path / "test.bib"
    bib.write_text(textwrap.dedent("""\
        @article{old,
            eprint = {2103.12345v1},
            archiveprefix = {arXiv},
        }
        @article{new,
            eprint = {2103.12345v3},
            archiveprefix = {arXiv},
        }
    """))
    ids = extract_ids_from_bib(str(bib))
    assert ids == ["2103.12345v3"]


# ---------------------------------------------------------------------------
# deduplicate_arxiv_ids (unit tests for the utility)
# ---------------------------------------------------------------------------

class TestDeduplicateArxivIds:
    def test_keeps_highest_version(self):
        assert deduplicate_arxiv_ids(["2103.12345v1", "2103.12345v3", "2103.12345v2"]) == ["2103.12345v3"]

    def test_bare_id_treated_as_v1(self):
        assert deduplicate_arxiv_ids(["2103.12345", "2103.12345v2"]) == ["2103.12345v2"]

    def test_preserves_order_of_first_occurrence(self):
        result = deduplicate_arxiv_ids(["2205.00001", "2103.12345v2", "2205.00001v3"])
        assert result == ["2205.00001v3", "2103.12345v2"]

    def test_no_duplicates(self):
        ids = ["2103.12345", "2205.67890", "2301.00001"]
        assert deduplicate_arxiv_ids(ids) == ids

    def test_empty(self):
        assert deduplicate_arxiv_ids([]) == []

    def test_single(self):
        assert deduplicate_arxiv_ids(["2103.12345v2"]) == ["2103.12345v2"]


# ---------------------------------------------------------------------------
# format_filename — pure filename construction
# ---------------------------------------------------------------------------

class TestFormatFilename:
    """Verify filename assembly from metadata fields."""

    def test_single_author(self):
        author_str, resolved_id, filename = format_filename(
            "2301.12345", ["Smith, Alice"], "2023", None, "A Paper", "v1"
        )
        assert author_str == "Smith, Alice"
        assert resolved_id == "2301.12345v1"
        assert filename == "Smith, Alice (2023) - A Paper - 2301.12345v1.pdf"

    def test_two_authors_joined_with_and(self):
        author_str, _, _ = format_filename(
            "2301.12345", ["Smith, Alice", "Jones, Bob"], "2023", None, "T", "v1"
        )
        assert author_str == "Smith, Alice and Jones, Bob"

    def test_three_or_more_authors_abbreviated(self):
        author_str, _, _ = format_filename(
            "2301.12345", ["A", "B", "C", "D"], "2023", None, "T", "v1"
        )
        assert author_str == "A and B et al."

    def test_no_authors_becomes_unknown(self):
        author_str, _, _ = format_filename(
            "2301.12345", [], "2023", None, "T", "v1"
        )
        assert author_str == "Unknown"

    def test_revision_year_appended_when_different(self):
        _, _, filename = format_filename(
            "2301.12345", ["X"], "2023", "2024", "T", "v2"
        )
        assert "(2023, revised 2024)" in filename

    def test_same_revision_year_omitted(self):
        _, _, filename = format_filename(
            "2301.12345", ["X"], "2023", "2023", "T", "v1"
        )
        assert ", revised" not in filename

    def test_title_colon_replaced(self):
        """Colons in titles are common and bad for filenames; replace with ' - '."""
        _, _, filename = format_filename(
            "2301.12345", ["X"], "2023", None, "Foo: A Subtitle", "v1"
        )
        assert "Foo - A Subtitle" in filename
        assert ":" not in filename

    def test_title_sanitized(self):
        """Unsafe characters are stripped."""
        _, _, filename = format_filename(
            "2301.12345", ["X"], "2023", None, "Paper/with*special?chars", "v1"
        )
        assert "/" not in filename
        assert "*" not in filename
        assert "?" not in filename

    def test_title_length_limited(self):
        long_title = "A" * 200
        _, _, filename = format_filename(
            "2301.12345", ["X"], "2023", None, long_title, "v1", title_length_limit=50
        )
        assert "..." in filename
        # Title-portion should be capped at limit + ellipsis
        assert filename.count("A") == 50

    def test_old_style_id_slash_replaced(self):
        """Old-style IDs (``hep-th/0601001``) have / replaced in the filename part."""
        _, resolved_id, filename = format_filename(
            "hep-th/0601001", ["X"], "2006", None, "T", "v1"
        )
        assert resolved_id == "hep-th/0601001v1"  # canonical resolved_id keeps /
        assert "hep-th_0601001v1" in filename     # filename is sanitized

    def test_version_overrides_input_id_version(self):
        """If arxiv_id has v2 but resolved version is v3, use v3."""
        _, resolved_id, _ = format_filename(
            "2301.12345v2", ["X"], "2023", None, "T", "v3"
        )
        assert resolved_id == "2301.12345v3"


# ---------------------------------------------------------------------------
# parse_metadata_response — Atom XML parsing
# ---------------------------------------------------------------------------

def _atom_response(arxiv_id="2301.12345",
                   version="1",
                   title="A Study of Widgets",
                   authors=("Smith, Alice",),
                   published="2023-05-15T00:00:00Z",
                   updated=None,
                   summary="Widgets are interesting.",
                   include_pdf_link=True) -> bytes:
    """Render a minimal arXiv Atom response for one paper."""
    if updated is None:
        updated = published
    authors_xml = "".join(f"<author><name>{a}</name></author>" for a in authors)
    pdf_link = (
        f'<link title="pdf" rel="related" href="http://arxiv.org/pdf/{arxiv_id}v{version}"/>'
        if include_pdf_link else ""
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom">'
        '<entry>'
        f'<id>http://arxiv.org/abs/{arxiv_id}v{version}</id>'
        f'<title>{title}</title>'
        f'<summary>{summary}</summary>'
        f'{authors_xml}'
        f'<published>{published}</published>'
        f'<updated>{updated}</updated>'
        f'{pdf_link}'
        '</entry>'
        '</feed>'
    ).encode("utf-8")


class TestParseMetadataResponse:
    """Verify XML → metadata-dict parsing."""

    def test_full_response(self):
        xml = _atom_response()
        md = parse_metadata_response(xml, "2301.12345")
        assert md["original_id"] == "2301.12345"
        assert md["resolved_id"] == "2301.12345v1"
        assert md["version"] == "v1"
        assert md["authors"] == "Smith, Alice"
        assert md["original_year"] == "2023"
        assert md["title"] == "A Study of Widgets"
        assert md["abstract"] == "Widgets are interesting."
        assert md["pdf_url"] == "http://arxiv.org/pdf/2301.12345v1"
        assert md["filename"].endswith(".pdf")

    def test_updated_different_year(self):
        """When updated year differs from published year, version_year is set."""
        xml = _atom_response(published="2023-01-01T00:00:00Z",
                             updated="2024-06-01T00:00:00Z")
        md = parse_metadata_response(xml, "2301.12345")
        assert md["original_year"] == "2023"
        assert md["version_year"] == "2024"
        assert "revised 2024" in md["filename"]

    def test_no_pdf_link(self):
        xml = _atom_response(include_pdf_link=False)
        md = parse_metadata_response(xml, "2301.12345")
        assert md["pdf_url"] is None

    def test_multiple_authors(self):
        xml = _atom_response(authors=("Smith, A.", "Jones, B.", "Zhang, C."))
        md = parse_metadata_response(xml, "2301.12345")
        assert md["authors"] == "Smith, A. and Jones, B. et al."

    def test_higher_version(self):
        xml = _atom_response(arxiv_id="2301.12345", version="3")
        md = parse_metadata_response(xml, "2301.12345")
        assert md["version"] == "v3"
        assert md["resolved_id"] == "2301.12345v3"


# ---------------------------------------------------------------------------
# get_paper_metadata — thin HTTP wrapper
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, content, status=200):
        self.content = content
        self._status = status

    def raise_for_status(self):
        if self._status >= 400:
            raise RuntimeError(f"HTTP {self._status}")


class TestGetPaperMetadata:
    """Exercise the thin HTTP wrapper."""

    def test_happy_path(self):
        xml = _atom_response()
        with patch.object(download_module.requests, "get", return_value=_FakeResponse(xml)) as mock_get:
            md = get_paper_metadata("2301.12345")
        assert md["title"] == "A Study of Widgets"
        # Built the expected API URL
        assert "id_list=2301.12345" in mock_get.call_args[0][0]

    def test_http_error_propagates(self):
        with patch.object(download_module.requests, "get", return_value=_FakeResponse(b"", status=500)):
            with pytest.raises(RuntimeError, match="HTTP 500"):
                get_paper_metadata("2301.12345")


# ---------------------------------------------------------------------------
# download_papers — end-to-end with mocked requests
# ---------------------------------------------------------------------------

class _NoWaitRateLimiter:
    """RateLimiter substitute that never sleeps."""
    def __init__(self, delay=3.0):
        pass

    def wait(self, show_progress=True):
        pass


def _mock_requests_get(metadata_responses, pdf_content=b"%PDF-fake-bytes"):
    """Return a ``requests.get`` stand-in that answers metadata+PDF calls.

    *metadata_responses* is a dict mapping arXiv ID → Atom XML bytes.  Any
    URL containing ``/pdf/`` returns *pdf_content*; any URL containing
    ``api/query?id_list=<id>`` returns the mapped XML.
    """
    def fake_get(url, *args, **kwargs):
        if "/pdf/" in url:
            return _FakeResponse(pdf_content)
        for arxiv_id, xml in metadata_responses.items():
            if f"id_list={arxiv_id}" in url:
                return _FakeResponse(xml)
        raise AssertionError(f"Unexpected URL in test: {url}")
    return fake_get


class TestDownloadPapers:
    """End-to-end download orchestration with mocked HTTP and filesystem."""

    def test_downloads_single_paper(self, tmp_path):
        xml = _atom_response()
        with patch.object(download_module, "RateLimiter", _NoWaitRateLimiter), \
             patch.object(download_module.requests, "get",
                          side_effect=_mock_requests_get({"2301.12345": xml})):
            download_papers(["2301.12345"], output_dir=str(tmp_path))
        pdfs = list(tmp_path.glob("*.pdf"))
        assert len(pdfs) == 1
        assert b"%PDF" in pdfs[0].read_bytes()
        assert "2301.12345v1" in pdfs[0].name

    def test_skips_paper_already_in_output_dir(self, tmp_path):
        """If a PDF with the same arXiv ID already exists, don't re-download."""
        # Pre-populate with a file whose filename contains the arXiv ID
        existing = tmp_path / "Existing (2023) - Old - 2301.12345v1.pdf"
        existing.write_bytes(b"old content, do not overwrite")
        xml = _atom_response()
        with patch.object(download_module, "RateLimiter", _NoWaitRateLimiter), \
             patch.object(download_module.requests, "get",
                          side_effect=_mock_requests_get({"2301.12345": xml})):
            download_papers(["2301.12345"], output_dir=str(tmp_path))
        # Still only one PDF, still original content
        pdfs = list(tmp_path.glob("*.pdf"))
        assert len(pdfs) == 1
        assert pdfs[0].read_bytes() == b"old content, do not overwrite"

    def test_no_pdf_url_no_file_written(self, tmp_path):
        """When the Atom entry has no PDF link, no file is created."""
        xml = _atom_response(include_pdf_link=False)
        with patch.object(download_module, "RateLimiter", _NoWaitRateLimiter), \
             patch.object(download_module.requests, "get",
                          side_effect=_mock_requests_get({"2301.12345": xml})):
            download_papers(["2301.12345"], output_dir=str(tmp_path))
        assert list(tmp_path.glob("*.pdf")) == []

    def test_duplicate_session_ids_fetched_once(self, tmp_path):
        """Listing the same ID twice downloads the PDF once."""
        xml = _atom_response()
        calls = []

        def tracking_get(url, *a, **kw):
            calls.append(url)
            return _mock_requests_get({"2301.12345": xml})(url, *a, **kw)

        with patch.object(download_module, "RateLimiter", _NoWaitRateLimiter), \
             patch.object(download_module.requests, "get", side_effect=tracking_get):
            download_papers(["2301.12345", "2301.12345"], output_dir=str(tmp_path))
        pdf_calls = [u for u in calls if "/pdf/" in u]
        assert len(pdf_calls) == 1  # PDF fetched once, even though ID repeated

    def test_exception_during_fetch_continues(self, tmp_path):
        """A failure on one ID doesn't abort the whole run."""
        xml_good = _atom_response(arxiv_id="2301.00002")

        def flaky_get(url, *a, **kw):
            if "id_list=2301.00001" in url:
                raise RuntimeError("simulated network blip")
            return _mock_requests_get({"2301.00002": xml_good})(url, *a, **kw)

        with patch.object(download_module, "RateLimiter", _NoWaitRateLimiter), \
             patch.object(download_module.requests, "get", side_effect=flaky_get):
            download_papers(["2301.00001", "2301.00002"], output_dir=str(tmp_path))
        # The good one still lands
        pdfs = list(tmp_path.glob("*.pdf"))
        assert len(pdfs) == 1
        assert "2301.00002" in pdfs[0].name

    def test_creates_output_dir(self, tmp_path):
        """Nonexistent output_dir is created."""
        out = tmp_path / "new_subdir"
        assert not out.exists()
        xml = _atom_response()
        with patch.object(download_module, "RateLimiter", _NoWaitRateLimiter), \
             patch.object(download_module.requests, "get",
                          side_effect=_mock_requests_get({"2301.12345": xml})):
            download_papers(["2301.12345"], output_dir=str(out))
        assert out.is_dir()


# ---------------------------------------------------------------------------
# extract_ids_from_bib — failed-blocks branch
# ---------------------------------------------------------------------------

def test_failed_blocks_warns_but_continues(tmp_path, capsys):
    """Malformed entries trigger the stderr warning; valid ones still come through."""
    bib = tmp_path / "broken.bib"
    bib.write_text(textwrap.dedent("""\
        @article{valid,
            eprint = {2301.00001},
            archiveprefix = {arXiv},
        }
        @article{broken_no_closing_brace,
            eprint = {2301.00002},
    """))
    ids = extract_ids_from_bib(str(bib))
    assert ids == ["2301.00001"]
    captured = capsys.readouterr()
    assert "failed to parse" in captured.err

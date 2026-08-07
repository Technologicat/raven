"""Tests for the arXiv download tool — BibTeX ID extraction and deduplication."""

import textwrap
from unittest.mock import patch

import bibtexparser
import pytest

from raven.papers import download as download_module
from raven.papers import httpfetch as httpfetch_module
from raven.papers.download import (
    ArxivMetadataError,
    download_papers,
    extract_ids_from_bib,
    format_filename,
    format_years,
    get_paper_metadata,
    parse_metadata_response,
    parse_metadata_responses,
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

    def test_title_question_mark_replaced(self):
        """A mid-title '? ' becomes ' - ' so the question doesn't run into the next clause."""
        _, _, filename = format_filename(
            "2301.12345", ["X"], "2023", None, "Is It Enough? A Study", "v1"
        )
        assert "Is It Enough - A Study" in filename
        assert "?" not in filename

    def test_title_trailing_question_mark_dropped(self):
        """A '?' with nothing after it is just stripped — no dangling separator inserted."""
        _, _, filename = format_filename(
            "2301.12345", ["X"], "2023", None, "Is It Enough?", "v1"
        )
        assert filename == "X (2023) - Is It Enough - 2301.12345v1.pdf"

    def test_title_exclamation_and_semicolon_replaced(self):
        """'! ' and '; ' clause boundaries also become ' - '."""
        _, _, bang = format_filename(
            "2301.12345", ["X"], "2023", None, "Surprise! A Method", "v1"
        )
        assert "Surprise - A Method" in bang
        _, _, semi = format_filename(
            "2301.12345", ["X"], "2023", None, "First Part; Second Part", "v1"
        )
        assert "First Part - Second Part" in semi

    def test_title_em_and_en_dash_normalized(self):
        """Em/en dashes become a plain hyphen instead of collapsing to a double space."""
        _, _, em = format_filename(
            "2301.12345", ["X"], "2023", None, "Attention — Revisited", "v1"
        )
        assert "Attention - Revisited" in em
        assert "  " not in em  # no double space left behind
        _, _, en = format_filename(
            "2301.12345", ["X"], "2023", None, "Results 2020–2023", "v1"
        )
        assert "Results 2020-2023" in en

    def test_title_sanitized(self):
        """Unsafe characters are stripped."""
        _, _, filename = format_filename(
            "2301.12345", ["X"], "2023", None, "Paper*with*special chars", "v1"
        )
        assert "*" not in filename

    def test_title_slash_becomes_hyphen(self):
        """'/' is reserved and has no single-word sense; render it as '-' so the sides stay distinct."""
        _, _, filename = format_filename(
            "2301.12345", ["X"], "2023", None, "Brain Rot on Twitter/X", "v1"
        )
        assert "Brain Rot on Twitter-X" in filename
        assert "/" not in filename

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
# format_years — year parenthetical
# ---------------------------------------------------------------------------

class TestFormatYears:
    """Verify the publication-year parenthetical rendering."""

    def test_single_year(self):
        assert format_years("2023", None) == "(2023)"

    def test_same_revision_year_collapsed(self):
        assert format_years("2023", "2023") == "(2023)"

    def test_different_revision_year(self):
        assert format_years("2023", "2024") == "(2023, revised 2024)"


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
        assert md["citation"] == "Smith, Alice (2023) - A Study of Widgets"

    def test_citation_uses_real_title_not_filename_safe(self):
        """The citation keeps punctuation the filename would strip (colon, '?')."""
        xml = _atom_response(title="Is One Layer Enough? A Study: Widgets")
        md = parse_metadata_response(xml, "2301.12345")
        assert md["citation"] == "Smith, Alice (2023) - Is One Layer Enough? A Study: Widgets"
        # ...while the filename sanitizes those same characters away.
        assert "?" not in md["filename"]
        assert ":" not in md["filename"]

    def test_citation_includes_revised_year(self):
        xml = _atom_response(published="2023-01-01T00:00:00Z",
                             updated="2024-06-01T00:00:00Z")
        md = parse_metadata_response(xml, "2301.12345")
        assert md["citation"] == "Smith, Alice (2023, revised 2024) - A Study of Widgets"

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

    def test_entryless_feed_raises_readable_error(self):
        """A nonexistent/malformed ID yields an entry-less feed; fail clearly."""
        empty_feed = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
        ).encode("utf-8")
        with pytest.raises(ArxivMetadataError, match="2614.19062"):
            parse_metadata_response(empty_feed, "2614.19062")
        # Still a ValueError subclass, so broad handlers keep catching it.
        assert issubclass(ArxivMetadataError, ValueError)


# ---------------------------------------------------------------------------
# get_paper_metadata — thin HTTP wrapper
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, content, status=200):
        self.content = content
        self.status_code = status
        self.headers: dict = {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class TestGetPaperMetadata:
    """Exercise the thin HTTP wrapper."""

    def test_happy_path(self):
        xml = _atom_response()
        with patch.object(httpfetch_module.requests, "get", return_value=_FakeResponse(xml)) as mock_get:
            md = get_paper_metadata("2301.12345")
        assert md["title"] == "A Study of Widgets"
        # Built the expected API URL
        assert "id_list=2301.12345" in mock_get.call_args[0][0]

    def test_http_error_propagates(self):
        with patch.object(httpfetch_module.requests, "get", return_value=_FakeResponse(b"", status=500)):
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


def _requested_ids(url, kwargs) -> list[str]:
    """The `id_list` of a metadata request, however it was spelled.

    `get_paper_metadata` builds the query into the URL; `get_papers_metadata` passes `params=`, which is
    the right way round for a batch (requests then encodes the commas, and the `/` in an old-style
    identifier). The mock has to answer both.
    """
    id_list = (kwargs.get("params") or {}).get("id_list")
    if id_list is None and "id_list=" in url:
        id_list = url.split("id_list=", 1)[1].split("&", 1)[0]
    return id_list.split(",") if id_list else []


def _atom_feed(*entry_xmls: bytes) -> bytes:
    """Splice one-entry feeds into a single multi-entry feed, as a batch request answers."""
    entries = b"".join(b"<entry>" + x.split(b"<entry>", 1)[1].rsplit(b"</entry>", 1)[0] + b"</entry>"
                       for x in entry_xmls if b"<entry>" in x)
    return (b'<?xml version="1.0" encoding="UTF-8"?>'
            b'<feed xmlns="http://www.w3.org/2005/Atom">' + entries + b'</feed>')


def _mock_requests_get(metadata_responses, pdf_content=b"%PDF-fake-bytes"):
    """Return a ``requests.get`` stand-in that answers metadata+PDF calls.

    *metadata_responses* is a dict mapping arXiv ID → Atom XML bytes.  Any URL containing ``/pdf/``
    returns *pdf_content*; a metadata request returns a feed holding an entry for every requested ID
    that the dict knows about.

    A batch naming an unknown ID is answered with the entries it *can* supply rather than raising, since
    that is what arXiv does and what `parse_metadata_responses` is written against — the caller detects
    the gap by diffing the request against the result.
    """
    def fake_get(url, *args, **kwargs):
        if "/pdf/" in url:
            return _FakeResponse(pdf_content)
        wanted = _requested_ids(url, kwargs)
        if not wanted:
            raise AssertionError(f"Unexpected URL in test: {url}")
        known = [metadata_responses[i] for i in wanted if i in metadata_responses]
        return _FakeResponse(_atom_feed(*known) if known else _atom_feed())
    return fake_get


class TestParseMetadataResponses:
    """Batched parsing: a multi-entry feed mapped back onto the identifiers that were requested."""

    def test_maps_each_requested_id_to_its_entry(self):
        feed = _atom_feed(_atom_response(arxiv_id="2301.00001", title="First"),
                          _atom_response(arxiv_id="2301.00002", title="Second"))
        got = parse_metadata_responses(feed, ["2301.00001", "2301.00002"])
        assert got["2301.00001"]["title"] == "First"
        assert got["2301.00002"]["title"] == "Second"

    def test_order_of_entries_does_not_matter(self):
        """Matching is by identifier, never by position — arXiv's ordering is not a contract."""
        feed = _atom_feed(_atom_response(arxiv_id="2301.00002", title="Second"),
                          _atom_response(arxiv_id="2301.00001", title="First"))
        got = parse_metadata_responses(feed, ["2301.00001", "2301.00002"])
        assert got["2301.00001"]["title"] == "First"
        assert got["2301.00002"]["title"] == "Second"

    def test_two_versions_of_one_paper_do_not_collide(self):
        """The regression that motivated exact-version matching.

        Both requests share a base identifier, so matching on the base alone hands them the same entry —
        and the loser silently receives the other version's `pdf_url` and filename: the wrong PDF, saved
        under a name asserting it is the right one.
        """
        feed = _atom_feed(_atom_response(arxiv_id="2301.12345", version="1", title="Original"),
                          _atom_response(arxiv_id="2301.12345", version="3", title="Revised"))
        got = parse_metadata_responses(feed, ["2301.12345v1", "2301.12345v3"])
        assert got["2301.12345v1"]["title"] == "Original"
        assert got["2301.12345v3"]["title"] == "Revised"
        assert got["2301.12345v1"]["pdf_url"] != got["2301.12345v3"]["pdf_url"]

    def test_unversioned_request_takes_the_highest_version(self):
        """An identifier with no version means "whatever is current", as it does to arXiv."""
        feed = _atom_feed(_atom_response(arxiv_id="2301.12345", version="1", title="Original"),
                          _atom_response(arxiv_id="2301.12345", version="3", title="Revised"))
        got = parse_metadata_responses(feed, ["2301.12345"])
        assert got["2301.12345"]["title"] == "Revised"

    def test_requested_but_unreturned_id_is_absent_not_an_error(self):
        """One unusable identifier must not cost the rest of its batch."""
        feed = _atom_feed(_atom_response(arxiv_id="2301.00002"))
        got = parse_metadata_responses(feed, ["2614.19062", "2301.00002"])
        assert "2614.19062" not in got
        assert "2301.00002" in got

    def test_version_named_but_not_returned_is_absent(self):
        """Asking for v2 and being handed v3 is a miss, not a near-enough match."""
        feed = _atom_feed(_atom_response(arxiv_id="2301.12345", version="3"))
        assert parse_metadata_responses(feed, ["2301.12345v2"]) == {}


class TestDownloadPapers:
    """End-to-end download orchestration with mocked HTTP and filesystem."""

    def test_metadata_is_fetched_in_one_request_per_batch(self, tmp_path):
        """The point of batching: N papers cost ceil(N / batch_size) metadata requests, not N."""
        responses = {f"2301.0000{i}": _atom_response(arxiv_id=f"2301.0000{i}") for i in range(1, 5)}
        metadata_calls = []

        def counting_get(url, *a, **kw):
            if "/pdf/" not in url:
                metadata_calls.append(kw.get("params", url))
            return _mock_requests_get(responses)(url, *a, **kw)

        with patch.object(download_module, "RateLimiter", _NoWaitRateLimiter), \
             patch.object(httpfetch_module.requests, "get", side_effect=counting_get):
            download_papers(list(responses), output_dir=str(tmp_path))
        assert len(metadata_calls) == 1
        assert len(list(tmp_path.glob("*.pdf"))) == 4

    def test_downloads_single_paper(self, tmp_path):
        xml = _atom_response()
        with patch.object(download_module, "RateLimiter", _NoWaitRateLimiter), \
             patch.object(httpfetch_module.requests, "get",
                          side_effect=_mock_requests_get({"2301.12345": xml})):
            download_papers(["2301.12345"], output_dir=str(tmp_path))
        pdfs = list(tmp_path.glob("*.pdf"))
        assert len(pdfs) == 1
        assert b"%PDF" in pdfs[0].read_bytes()
        assert "2301.12345v1" in pdfs[0].name

    def test_a_repeated_identifier_is_dropped_before_the_metadata_request(self, tmp_path):
        """`raven-arxiv2bib` drops exact repeats before fetching; this used to carry them to the download
        step, paying for a batch slot on the way."""
        xml = _atom_response()
        requested = []

        def counting_get(url, *a, **kw):
            if "/pdf/" not in url:
                requested.append(kw.get("params", {}).get("id_list", ""))
            return _mock_requests_get({"2301.12345": xml})(url, *a, **kw)

        with patch.object(download_module, "RateLimiter", _NoWaitRateLimiter), \
             patch.object(httpfetch_module.requests, "get", side_effect=counting_get):
            download_papers(["2301.12345", "2301.12345", "2301.12345"], output_dir=str(tmp_path))
        assert len(requested) == 1
        assert requested[0].split(",") == ["2301.12345"]

    def test_the_run_is_summarized_by_outcome(self, capsys, tmp_path):
        """A rerun does almost nothing, so a bare total says nothing. The counts have to be by outcome."""
        xml = _atom_response()
        with patch.object(download_module, "RateLimiter", _NoWaitRateLimiter), \
             patch.object(httpfetch_module.requests, "get",
                          side_effect=_mock_requests_get({"2301.12345": xml})):
            # The same identifier twice: one download, one recognized as a repeat within the run.
            download_papers(["2301.12345", "2301.12345"], output_dir=str(tmp_path))
        summary = capsys.readouterr().out.strip().splitlines()[-1]
        assert "2 identifiers processed" in summary
        assert "1 downloaded" in summary
        assert "1 duplicate identifier" in summary

    def test_the_summary_names_only_outcomes_that_happened(self, capsys, tmp_path):
        """A clean run should not have to say "0 failed" for the reader to see that nothing failed."""
        xml = _atom_response()
        with patch.object(download_module, "RateLimiter", _NoWaitRateLimiter), \
             patch.object(httpfetch_module.requests, "get",
                          side_effect=_mock_requests_get({"2301.12345": xml})):
            download_papers(["2301.12345"], output_dir=str(tmp_path))
        summary = capsys.readouterr().out.strip().splitlines()[-1]
        assert "1 downloaded" in summary
        assert "failed" not in summary and "duplicate" not in summary

    def test_skips_paper_already_in_output_dir(self, tmp_path):
        """If a PDF with the same arXiv ID already exists, don't re-download."""
        # Pre-populate with a file whose filename contains the arXiv ID
        existing = tmp_path / "Existing (2023) - Old - 2301.12345v1.pdf"
        existing.write_bytes(b"old content, do not overwrite")
        xml = _atom_response()
        with patch.object(download_module, "RateLimiter", _NoWaitRateLimiter), \
             patch.object(httpfetch_module.requests, "get",
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
             patch.object(httpfetch_module.requests, "get",
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
             patch.object(httpfetch_module.requests, "get", side_effect=tracking_get):
            download_papers(["2301.12345", "2301.12345"], output_dir=str(tmp_path))
        pdf_calls = [u for u in calls if "/pdf/" in u]
        assert len(pdf_calls) == 1  # PDF fetched once, even though ID repeated

    def test_save_bib_writes_parseable_bibtex_without_extra_requests(self, tmp_path):
        """`--save-bib` is made from metadata already fetched, so it costs no additional requests."""
        responses = {"2301.00001": _atom_response(arxiv_id="2301.00001", title="First"),
                     "2301.00002": _atom_response(arxiv_id="2301.00002", title="Second")}
        metadata_calls = []

        def counting_get(url, *a, **kw):
            if "/pdf/" not in url:
                metadata_calls.append(url)
            return _mock_requests_get(responses)(url, *a, **kw)

        bib_path = tmp_path / "out.bib"
        with patch.object(download_module, "RateLimiter", _NoWaitRateLimiter), \
             patch.object(httpfetch_module.requests, "get", side_effect=counting_get):
            download_papers(list(responses), output_dir=str(tmp_path), save_bib=str(bib_path))

        assert len(metadata_calls) == 1  # one batch, and nothing extra for the bibliography
        library = bibtexparser.parse_string(bib_path.read_text(encoding="utf-8"))
        assert not library.failed_blocks
        titles = {e.fields_dict["title"].value for e in library.entries}
        assert titles == {"First", "Second"}
        # Versions are kept: a download names a specific version, and the bibliography should say which.
        assert {e.fields_dict["eprint"].value for e in library.entries} == {"2301.00001v1",
                                                                           "2301.00002v1"}

    def test_save_bib_records_papers_already_present(self, tmp_path):
        """The bibliography describes the set that was asked for, not the subset that was missing."""
        (tmp_path / "Existing (2023) - Old - 2301.00001v1.pdf").write_bytes(b"already here")
        responses = {"2301.00001": _atom_response(arxiv_id="2301.00001", title="First")}
        bib_path = tmp_path / "out.bib"
        with patch.object(download_module, "RateLimiter", _NoWaitRateLimiter), \
             patch.object(httpfetch_module.requests, "get",
                          side_effect=_mock_requests_get(responses)):
            download_papers(["2301.00001"], output_dir=str(tmp_path), save_bib=str(bib_path))
        library = bibtexparser.parse_string(bib_path.read_text(encoding="utf-8"))
        assert len(library.entries) == 1

    def test_failed_metadata_batch_does_not_abort_the_run(self, tmp_path, capsys):
        """A network blip costs its batch, not the whole run, and keeps its traceback for debugging.

        Batch size is forced to 1 so the two identifiers land in separate requests — with the real batch
        size they would share one, and a single failure would legitimately take both. That is the trade
        batching makes, and the property worth pinning is the weaker one: whatever else was going to be
        fetched still gets fetched.
        """
        xml_good = _atom_response(arxiv_id="2301.00002")

        def flaky_get(url, *a, **kw):
            if "2301.00001" in str(kw.get("params", "")) or "id_list=2301.00001" in url:
                raise RuntimeError("simulated network blip")
            return _mock_requests_get({"2301.00002": xml_good})(url, *a, **kw)

        with patch.object(download_module, "RateLimiter", _NoWaitRateLimiter), \
             patch.object(httpfetch_module.requests, "get", side_effect=flaky_get):
            download_papers(["2301.00001", "2301.00002"], output_dir=str(tmp_path), batch_size=1)
        # The good one still lands
        pdfs = list(tmp_path.glob("*.pdf"))
        assert len(pdfs) == 1
        assert "2301.00002" in pdfs[0].name
        # An unexpected error keeps its traceback for debugging.
        assert "Traceback" in capsys.readouterr().err

    def test_bad_id_reports_cleanly_and_continues(self, tmp_path, capsys):
        """A malformed/nonexistent ID prints a one-line failure (no traceback) and the run continues."""
        empty_feed = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
        ).encode("utf-8")
        xml_good = _atom_response(arxiv_id="2301.00002")
        with patch.object(download_module, "RateLimiter", _NoWaitRateLimiter), \
             patch.object(httpfetch_module.requests, "get",
                          side_effect=_mock_requests_get({"2614.19062": empty_feed,
                                                          "2301.00002": xml_good})):
            download_papers(["2614.19062", "2301.00002"], output_dir=str(tmp_path))
        captured = capsys.readouterr()
        assert "2614.19062 failed" in captured.out
        assert "Traceback" not in captured.err  # expected error → no traceback
        # The good one still lands.
        pdfs = list(tmp_path.glob("*.pdf"))
        assert len(pdfs) == 1
        assert "2301.00002" in pdfs[0].name

    def test_creates_output_dir(self, tmp_path):
        """Nonexistent output_dir is created."""
        out = tmp_path / "new_subdir"
        assert not out.exists()
        xml = _atom_response()
        with patch.object(download_module, "RateLimiter", _NoWaitRateLimiter), \
             patch.object(httpfetch_module.requests, "get",
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

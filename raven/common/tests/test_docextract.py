"""Unit tests for raven.common.docextract.

Every binary format is generated on the fly from the shared test utilities rather than committed as a fixture
file. For PDF that means assembling the bytes by hand (`make_minimal_pdf` writes a correct xref table, so pypdf
reads it as a real born-digital PDF); for the office formats it means writing the file with the same library
family that reads it back. The office round-trips are therefore weaker evidence than the PDF ones — a shared
misunderstanding of the format would cancel out — so they are aimed at what Raven decides rather than at what
the library does: which blocks get collected, and in what order.
"""

import pytest

from raven.common import docextract
from raven.common.tests import make_docx, make_minimal_pdf, make_odp, make_odt, make_pptx, make_textless_pdf


# ---------------------------------------------------------------------------
# Capability queries
# ---------------------------------------------------------------------------

def test_supported_extensions_is_the_full_advertised_set():
    # Pinned as an exact set rather than spot-checked, because this list is the contract that several other
    # places are kept in sync with by hand: `raven.librarian.config.llm_docs_exts`, the attachment dialog's
    # filter, and the format list in the Librarian README. Spot-checking a few members would let any of the
    # others fall out unnoticed; pinning the set makes both directions a deliberate edit.
    #
    # `.bib` in particular has to stay: `raven-burstbib` exists to split one BibTeX file into per-entry `.bib`
    # files precisely so they can be dropped into the document database as individual documents. Dropping the
    # extension here would break that workflow at the far end of the constellation, where nothing points back.
    assert set(docextract.supported_extensions()) == {".txt", ".md", ".rst", ".org", ".bib", ".tex",
                                                      ".pdf",
                                                      ".docx", ".pptx", ".odt", ".odp"}


def test_supported_extensions_has_no_duplicates():
    # `supported_extensions` splices two independently maintained lists (the plaintext tuple and the parser
    # dispatch table), and the set comparison above would hide an extension that appears in both. It matters
    # because the dispatch table wins: a format listed in both would quietly stop being read as plain text.
    exts = docextract.supported_extensions()
    assert len(exts) == len(set(exts))


@pytest.mark.parametrize("name, expected", [
    ("notes.txt", True),
    ("README.MD", True),   # case-insensitive
    ("paper.pdf", True),
    ("report.docx", True),
    ("REPORT.DOCX", True),
    ("deck.pptx", True),
    ("notes.odt", True),
    ("deck.odp", True),
    ("sheet.xlsx", False),  # spreadsheets are a separate problem class; see TODO_DEFERRED.md
    ("legacy.doc", False),
    ("photo.png", False),
    ("archive.tar.gz", False),
    ("noext", False),
])
def test_is_supported(name, expected):
    assert docextract.is_supported(name) is expected


# ---------------------------------------------------------------------------
# Plain-text extraction
# ---------------------------------------------------------------------------

def test_plaintext_roundtrip(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("Hello Raven", encoding="utf-8")
    assert docextract.extract_text(p) == "Hello Raven"


def test_plaintext_is_stripped(tmp_path):
    p = tmp_path / "a.md"
    p.write_text("\n\n  # Title  \n\n", encoding="utf-8")
    assert docextract.extract_text(p) == "# Title"


def test_whitespace_only_returns_none(tmp_path):
    p = tmp_path / "blank.txt"
    p.write_text("   \n\t \n", encoding="utf-8")
    assert docextract.extract_text(p) is None


def test_accepts_str_path(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("string path ok", encoding="utf-8")
    assert docextract.extract_text(str(p)) == "string path ok"


# ---------------------------------------------------------------------------
# Error situations raise (never silently return None)
# ---------------------------------------------------------------------------

def test_missing_file_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        docextract.extract_text(tmp_path / "nope.txt")


def test_non_utf8_text_raises(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_bytes(b"\xff\xfe\x00garbage\x80\x81")
    with pytest.raises(docextract.DocumentExtractionError):
        docextract.extract_text(p)


def test_non_utf8_error_chains_cause(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_bytes(b"\x80\x81\x82")
    with pytest.raises(docextract.DocumentExtractionError) as excinfo:
        docextract.extract_text(p)
    assert isinstance(excinfo.value.__cause__, UnicodeDecodeError)


def test_not_a_pdf_raises(tmp_path):
    p = tmp_path / "fake.pdf"
    p.write_bytes(b"this is plainly not a PDF")
    with pytest.raises(docextract.DocumentExtractionError):
        docextract.extract_text(p)


def test_empty_file_with_pdf_extension_raises(tmp_path):
    p = tmp_path / "empty.pdf"
    p.write_bytes(b"")
    with pytest.raises(docextract.DocumentExtractionError):
        docextract.extract_text(p)


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def test_pdf_roundtrip(tmp_path):
    p = tmp_path / "sample.pdf"
    p.write_bytes(make_minimal_pdf("Hello Raven PDF extraction"))
    assert docextract.extract_text(p) == "Hello Raven PDF extraction"


def test_pdf_uppercase_extension(tmp_path):
    p = tmp_path / "SAMPLE.PDF"
    p.write_bytes(make_minimal_pdf("Case insensitive"))
    assert docextract.extract_text(p) == "Case insensitive"


def test_pdf_without_text_layer_returns_none(tmp_path):
    # A parseable PDF that has no text to extract (stand-in for a scanned/image-only page) is "empty", not an
    # error -> None, so the caller skips it rather than treating it as a failure.
    p = tmp_path / "scanned.pdf"
    p.write_bytes(make_textless_pdf())
    assert docextract.extract_text(p) is None


# ---------------------------------------------------------------------------
# Word documents (.docx)
# ---------------------------------------------------------------------------

def test_docx_paragraphs(tmp_path):
    p = tmp_path / "report.docx"
    p.write_bytes(make_docx(["First para", "Second para"]))
    assert docextract.extract_text(p) == "First para\nSecond para"


def test_docx_table_is_read_in_place(tmp_path):
    # The interesting property is placement, not presence: a table has to come back between the paragraphs that
    # surround it, because retrieval chunks on a sliding window and a chunk spanning the boundary is only
    # meaningful if the reading order survived.
    p = tmp_path / "report.docx"
    p.write_bytes(make_docx(["Before", [["Name", "Value"], ["alpha", "42"]], "After"]))
    assert docextract.extract_text(p) == "Before\nName\tValue\nalpha\t42\nAfter"


def test_docx_empty_returns_none(tmp_path):
    p = tmp_path / "blank.docx"
    p.write_bytes(make_docx([]))
    assert docextract.extract_text(p) is None


# ---------------------------------------------------------------------------
# PowerPoint presentations (.pptx)
# ---------------------------------------------------------------------------

def test_pptx_slides_in_order(tmp_path):
    p = tmp_path / "deck.pptx"
    p.write_bytes(make_pptx([{"text": ["Slide one"]}, {"text": ["Slide two"]}]))
    assert docextract.extract_text(p) == "Slide one\nSlide two"


def test_pptx_includes_presenter_notes(tmp_path):
    # Notes are part of what the deck says even though they are never projected — on a lecture deck they often
    # carry the argument the slide only asserts.
    p = tmp_path / "deck.pptx"
    p.write_bytes(make_pptx([{"text": ["Bullet"], "notes": "The reasoning behind the bullet"}]))
    assert docextract.extract_text(p) == "Bullet\nThe reasoning behind the bullet"


def test_pptx_reads_grouped_shapes(tmp_path):
    # Text inside a group is invisible to a flat pass over `slide.shapes`, and grouping is something a person
    # does for layout reasons without any expectation that it hides the words.
    p = tmp_path / "deck.pptx"
    p.write_bytes(make_pptx([{"text": ["Loose"], "group": ["Grouped"]}]))
    assert docextract.extract_text(p) == "Loose\nGrouped"


def test_pptx_table(tmp_path):
    p = tmp_path / "deck.pptx"
    p.write_bytes(make_pptx([{"table": [["k", "v"], ["x", "9"]]}]))
    assert docextract.extract_text(p) == "k\tv\nx\t9"


def test_pptx_empty_returns_none(tmp_path):
    p = tmp_path / "blank.pptx"
    p.write_bytes(make_pptx([]))
    assert docextract.extract_text(p) is None


# ---------------------------------------------------------------------------
# OpenDocument (.odt, .odp)
# ---------------------------------------------------------------------------

def test_odt_headings_and_paragraphs_in_order(tmp_path):
    # ODF marks a heading and a paragraph with different element names, so an extractor that collects them by
    # type rather than by position reads back every heading first and every paragraph after.
    p = tmp_path / "notes.odt"
    p.write_bytes(make_odt([("h", "Chapter one"), ("p", "Body of one"),
                            ("h", "Chapter two"), ("p", "Body of two")]))
    assert docextract.extract_text(p) == "Chapter one\nBody of one\nChapter two\nBody of two"


def test_odt_empty_returns_none(tmp_path):
    p = tmp_path / "blank.odt"
    p.write_bytes(make_odt([]))
    assert docextract.extract_text(p) is None


def test_odp_slides_in_order(tmp_path):
    p = tmp_path / "deck.odp"
    p.write_bytes(make_odp([["Slide one title", "point a"], ["Slide two title"]]))
    assert docextract.extract_text(p) == "Slide one title\npoint a\nSlide two title"


# ---------------------------------------------------------------------------
# Office formats: malformed input raises rather than returning empty
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["fake.docx", "fake.pptx", "fake.odt", "fake.odp"])
def test_office_file_that_is_not_a_zip_raises(tmp_path, name):
    # Every office format here is a zip container, so the cheapest wrong file is one that is not a zip at all.
    # It must raise: a file the user deliberately handed us and that we cannot read is an error situation, not
    # an empty document, and the two are what the raise-vs-None split exists to keep apart.
    p = tmp_path / name
    p.write_bytes(b"this is plainly not an office document")
    with pytest.raises(docextract.DocumentExtractionError):
        docextract.extract_text(p)


@pytest.mark.parametrize("name", ["empty.docx", "empty.pptx", "empty.odt", "empty.odp"])
def test_zero_byte_office_file_raises(tmp_path, name):
    p = tmp_path / name
    p.write_bytes(b"")
    with pytest.raises(docextract.DocumentExtractionError):
        docextract.extract_text(p)


def test_office_extraction_error_chains_cause(tmp_path):
    # The underlying exception is what says *why* the file could not be read, and an interactive attach site
    # shows that to the user; dropping the chain would leave only "could not be read".
    p = tmp_path / "fake.docx"
    p.write_bytes(b"not a zip")
    with pytest.raises(docextract.DocumentExtractionError) as excinfo:
        docextract.extract_text(p)
    assert excinfo.value.__cause__ is not None

"""Unit tests for raven.common.docextract.

The PDF cases build a minimal but valid single-page PDF on the fly (`make_minimal_pdf`, from the shared test
utilities) rather than committing a binary fixture — the generator writes a correct xref table, so pypdf reads
it as a real born-digital PDF.
"""

import pytest

from raven.common import docextract
from raven.common.tests import make_minimal_pdf, make_textless_pdf


# ---------------------------------------------------------------------------
# Capability queries
# ---------------------------------------------------------------------------

def test_supported_extensions_includes_text_and_pdf():
    exts = docextract.supported_extensions()
    assert ".txt" in exts
    assert ".md" in exts
    assert ".pdf" in exts


@pytest.mark.parametrize("name, expected", [
    ("notes.txt", True),
    ("README.MD", True),   # case-insensitive
    ("paper.pdf", True),
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

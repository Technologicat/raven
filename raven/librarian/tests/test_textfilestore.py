"""Unit tests for raven.librarian.textfilestore (document sidecar store, text resolution, GC mark phase)."""

import pytest

from raven.librarian import chattree, textfilestore
from raven.common.tests import make_minimal_pdf


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def datastore(tmp_path):
    """A fresh no-autosave PersistentForest with a temp-dir sidecar directory, GC-configured for file refs."""
    return chattree.PersistentForest(tmp_path / "chat.json", autosave=False,
                                     sidecar_extractor=textfilestore.sidecar_refs_in_payload)


@pytest.fixture(autouse=True)
def _isolate_extract_cache():
    """The extracted-text memo is module-level; clear it around each test so they don't couple through it."""
    textfilestore._extracted_text_cache.clear()
    yield
    textfilestore._extracted_text_cache.clear()


# ---------------------------------------------------------------------------
# store_file_as_sidecar
# ---------------------------------------------------------------------------

def test_store_text_file_roundtrip(datastore):
    raw = b"Hello Raven document."
    result = textfilestore.store_file_as_sidecar(datastore, raw,
                                             name="notes.txt",
                                             provenance_url="file:///tmp/notes.txt",
                                             provenance_source="user_attachment")
    assert result.part == {"type": "text_file",
                           "text_file": {"url": f"sidecar:{result.filename}", "name": "notes.txt"}}
    assert datastore.read_sidecar(result.filename) == raw  # stored byte-for-byte
    assert result.filename.endswith(".txt")
    md = result.sidecar_metadata
    assert md["url"] == "file:///tmp/notes.txt"
    assert md["source"] == "user_attachment"
    assert md["name"] == "notes.txt"
    assert md["content_type"] == "text/plain"
    assert md["size_bytes"] == len(raw)


def test_store_pdf_records_pdf_mime_and_extension(datastore):
    result = textfilestore.store_file_as_sidecar(datastore, make_minimal_pdf("x"),
                                             name="paper.pdf",
                                             provenance_url="file:///paper.pdf",
                                             provenance_source="user_attachment")
    assert result.filename.endswith(".pdf")
    assert result.sidecar_metadata["content_type"] == "application/pdf"


def test_store_from_path(datastore, tmp_path):
    p = tmp_path / "doc.md"
    p.write_text("# Title\n\nBody.", encoding="utf-8")
    result = textfilestore.store_file_as_sidecar(datastore, p,
                                             name="doc.md",
                                             provenance_url=p.as_uri(),
                                             provenance_source="user_attachment")
    assert datastore.read_sidecar(result.filename) == p.read_bytes()


# ---------------------------------------------------------------------------
# sidecar_to_text
# ---------------------------------------------------------------------------

def test_sidecar_to_text_plaintext(datastore):
    result = textfilestore.store_file_as_sidecar(datastore, b"plain content here",
                                             name="a.txt", provenance_url="file:///a.txt",
                                             provenance_source="user_attachment")
    assert textfilestore.sidecar_to_text(datastore, result.part["text_file"]["url"]) == "plain content here"


def test_sidecar_to_text_pdf(datastore):
    pdf = make_minimal_pdf("Extracted from a PDF attachment")
    result = textfilestore.store_file_as_sidecar(datastore, pdf,
                                             name="paper.pdf", provenance_url="file:///paper.pdf",
                                             provenance_source="user_attachment")
    assert textfilestore.sidecar_to_text(datastore, result.part["text_file"]["url"]) == "Extracted from a PDF attachment"


def test_sidecar_to_text_is_memoized_on_immutable_filename(datastore):
    result = textfilestore.store_file_as_sidecar(datastore, b"cache me",
                                             name="c.txt", provenance_url="file:///c.txt",
                                             provenance_source="user_attachment")
    url = result.part["text_file"]["url"]
    first = textfilestore.sidecar_to_text(datastore, url)
    # Corrupt the on-disk sidecar; because the memo keys on the (content-addressed) filename, a second read must
    # still return the original text without touching disk.
    datastore.sidecar_path(result.filename).write_bytes(b"different now")
    assert textfilestore.sidecar_to_text(datastore, url) == first == "cache me"


def test_sidecar_to_text_empty_document_placeholder(datastore):
    result = textfilestore.store_file_as_sidecar(datastore, b"   \n\t ",
                                             name="blank.txt", provenance_url="file:///blank.txt",
                                             provenance_source="user_attachment")
    # Whitespace-only extracts to None; the wire path must get a placeholder rather than an exception.
    assert textfilestore.sidecar_to_text(datastore, result.part["text_file"]["url"]) == "[no extractable text]"


def test_sidecar_to_text_bad_scheme_raises(datastore):
    with pytest.raises(ValueError):
        textfilestore.sidecar_to_text(datastore, "https://example.com/x.txt")


# ---------------------------------------------------------------------------
# sidecar_refs_in_payload (GC mark phase)
# ---------------------------------------------------------------------------

def test_sidecar_refs_collects_only_text_file_refs():
    payload = {"message": {"role": "user",
                           "content": [{"type": "text", "text": "hi"},
                                       {"type": "text_file", "text_file": {"url": "sidecar:abc.txt", "name": "abc.txt"}},
                                       {"type": "image_url", "image_url": {"url": "sidecar:img.png"}}]}}
    # Only the document ref — images are the imagestore extractor's job (the two are composed at GC time).
    assert textfilestore.sidecar_refs_in_payload(payload) == {"abc.txt"}


def test_sidecar_refs_legacy_string_content_is_empty():
    assert textfilestore.sidecar_refs_in_payload({"message": {"content": "bare pre-migration string"}}) == set()

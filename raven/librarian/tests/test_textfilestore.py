"""Unit tests for raven.librarian.textfilestore (document sidecar store, text resolution, GC mark phase)."""

import pytest

from raven.librarian import chattree, textfilestore
from raven.common.tests import make_docx, make_minimal_pdf


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
    # `provenance_source` lands on the part as well as in the sidecar metadata below: the wire builder sees
    # bare messages, so the part is the only copy it can reach.
    assert result.part == {"type": "text_file",
                           "text_file": {"url": f"sidecar:{result.filename}", "name": "notes.txt",
                                         "source": "user_attachment"}}
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


@pytest.mark.parametrize("name, expected_mime", [
    ("report.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    ("deck.pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
    ("notes.odt", "application/vnd.oasis.opendocument.text"),
    ("deck.odp", "application/vnd.oasis.opendocument.presentation"),
])
def test_store_office_document_records_its_own_mime(datastore, name, expected_mime):
    # The MIME is provenance only, never dispatch — but "text/plain" on a Word document is a visibly wrong
    # record, and it is what the fallback produces for any extension not in the table.
    result = textfilestore.store_file_as_sidecar(datastore, make_docx(["x"]),
                                             name=name,
                                             provenance_url=f"file:///{name}",
                                             provenance_source="user_attachment")
    assert result.sidecar_metadata["content_type"] == expected_mime


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


def test_sidecar_to_text_office_document(datastore):
    # The point of this one is the wiring, not the parsing: `docextract` is meant to be the single chokepoint, so
    # a format added there must reach the attachment path without any edit on this side. If that ever stops
    # being true, the two surfaces have started to drift and this is where it shows.
    result = textfilestore.store_file_as_sidecar(datastore, make_docx(["Extracted from a Word attachment"]),
                                             name="report.docx", provenance_url="file:///report.docx",
                                             provenance_source="user_attachment")
    text = textfilestore.sidecar_to_text(datastore, result.part["text_file"]["url"])
    assert text == "Extracted from a Word attachment"


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


# ---------------------------------------------------------------------------
# The stored file describes itself
# ---------------------------------------------------------------------------

class TestSidecarSelfDescription:
    def test_stored_document_carries_its_provenance_beside_the_file(self, datastore):
        # The payload copy of the provenance dies with its node. This copy is what a cleanup preview reads to
        # name an orphan, so it has to be written at store time -- there is no later chance.
        result = textfilestore.store_file_as_sidecar(datastore, b"some document text",
                                                 name="thesis.pdf",
                                                 provenance_url="file:///thesis.pdf",
                                                 provenance_source="user_attachment")
        stored = datastore.get_sidecar_metadata(result.filename)
        assert stored is not None
        assert stored["name"] == "thesis.pdf"
        assert stored["url"] == "file:///thesis.pdf"

    def test_description_matches_the_payload_provenance(self, datastore):
        # One dict, written to two places. If they can differ, the preview and the chat log can disagree about
        # what the same file is.
        result = textfilestore.store_file_as_sidecar(datastore, b"content",
                                                 name="notes.txt",
                                                 provenance_url="file:///notes.txt",
                                                 provenance_source="user_attachment")
        assert datastore.get_sidecar_metadata(result.filename) == result.sidecar_metadata

    def test_description_survives_deletion_of_the_referencing_node(self, datastore):
        # The whole point, stated as a test: delete the message, and the file can still say what it was.
        result = textfilestore.store_file_as_sidecar(datastore, b"orphan me",
                                                 name="orphaned.pdf",
                                                 provenance_url="file:///orphaned.pdf",
                                                 provenance_source="user_attachment")
        node_id = datastore.create_node({"message": {"role": "user", "content": [result.part]},
                                         "general_metadata": {"sidecars": {result.filename: result.sidecar_metadata}}},
                                        parent_id=None)
        datastore.delete_subtree(node_id)

        assert datastore.list_unreferenced_sidecars() == [result.filename]
        assert datastore.get_sidecar_metadata(result.filename)["name"] == "orphaned.pdf"

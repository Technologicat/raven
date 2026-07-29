"""Unit tests for raven.librarian.cleanup — the datastore maintenance operation.

Only the operation half is covered here; `DPGCleanupDialog` needs a DPG context and a live render loop, so
it is exercised by hand (see the live-GUI notes in the project CLAUDE.md).
"""

import pytest

from raven.librarian import appstate, cleanup
from raven.librarian.chattree import PersistentForest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def datastore(tmp_path):
    """A datastore wired with the real sidecar extractor both Librarian frontends use."""
    return PersistentForest(tmp_path / "chat.json",
                            autosave=False,
                            sidecar_extractor=appstate.sidecar_refs_in_payload)


def _payload_with_image(filename, *, name=None, url=None):
    """A node payload referencing an image sidecar, shaped as `imagestore` writes it."""
    entry = {"url": url or f"file:///home/someone/{name or filename}",
             "content_type": "image/png",
             "source": "user_attachment"}
    return {"message": {"role": "user",
                        "content": [{"type": "text", "text": "look at this"},
                                    {"type": "image_url", "image_url": {"url": f"sidecar:{filename}"}}]},
            "general_metadata": {"sidecars": {filename: entry}}}


def _payload_with_document(filename, name):
    """A node payload referencing a document sidecar, shaped as `textfilestore` writes it."""
    entry = {"url": f"file:///home/someone/{name}",
             "content_type": "application/pdf",
             "source": "user_attachment",
             "name": name}
    return {"message": {"role": "user",
                        "content": [{"type": "text", "text": "read this"},
                                    {"type": "text_file", "text_file": {"url": f"sidecar:{filename}",
                                                                        "name": name}}]},
            "general_metadata": {"sidecars": {filename: entry}}}


# ---------------------------------------------------------------------------
# describe_sidecar
# ---------------------------------------------------------------------------

class TestDescribeSidecar:
    def test_document_is_named_by_its_name_field(self, datastore):
        filename = datastore.store_sidecar(b"%PDF-1.4 whatever", "pdf",
                                           metadata={"name": "Attention Is All You Need.pdf",
                                                     "content_type": "application/pdf"})
        entry = cleanup.describe_sidecar(datastore, filename)
        assert entry.display_name == "Attention Is All You Need.pdf"
        assert not entry.is_image
        assert entry.size_bytes == len(b"%PDF-1.4 whatever")

    def test_image_is_named_by_its_provenance_url(self, datastore):
        """An image has no `name` field — the store may have re-encoded it — so the URL basename stands in."""
        filename = datastore.store_sidecar(b"fake png bytes", "png",
                                           metadata={"url": "file:///home/someone/my%20photo.png",
                                                     "content_type": "image/png"})
        entry = cleanup.describe_sidecar(datastore, filename)
        assert entry.display_name == "my photo.png"  # percent-escapes decoded
        assert entry.is_image

    def test_undescribed_sidecar_falls_back_to_its_hash(self, datastore):
        """Sidecars stored before descriptions existed still have to appear in the preview."""
        filename = datastore.store_sidecar(b"legacy", "png")  # no metadata
        entry = cleanup.describe_sidecar(datastore, filename)
        assert entry.display_name == filename
        assert entry.is_image  # decided by extension, since there is no content type to go on
        assert entry.metadata == {}

    def test_kind_falls_back_to_extension_without_content_type(self, datastore):
        pdf = datastore.store_sidecar(b"doc", "pdf")
        png = datastore.store_sidecar(b"pic", "png")
        assert not cleanup.describe_sidecar(datastore, pdf).is_image
        assert cleanup.describe_sidecar(datastore, png).is_image

    def test_content_type_wins_over_extension(self, datastore):
        """A `.dat` holding an image is an image; the store's own record beats guessing from the name."""
        filename = datastore.store_sidecar(b"pic", "dat", metadata={"content_type": "image/webp"})
        assert cleanup.describe_sidecar(datastore, filename).is_image

    def test_vanished_sidecar_degrades_instead_of_raising(self, datastore):
        """The sidecar directory is live storage; a preview must survive a file disappearing mid-scan."""
        filename = datastore.store_sidecar(b"here for now", "png")
        datastore.sidecar_path(filename).unlink()
        entry = cleanup.describe_sidecar(datastore, filename)
        assert entry.size_bytes == 0
        assert entry.filename == filename


# ---------------------------------------------------------------------------
# preview_cleanup
# ---------------------------------------------------------------------------

class TestPreviewCleanup:
    def test_empty_datastore_previews_as_nothing_to_do(self, datastore):
        root = datastore.create_node({"message": {"role": "system", "content": []}}, parent_id=None)
        preview = cleanup.preview_cleanup(datastore, root)
        assert preview.is_empty
        assert preview.total_bytes == 0

    def test_splits_images_from_documents_and_sorts_each(self, datastore):
        root = datastore.create_node({"message": {"role": "system", "content": []}}, parent_id=None)
        zebra = datastore.store_sidecar(b"z" * 10, "png", metadata={"url": "file:///x/zebra.png",
                                                                    "content_type": "image/png"})
        apple = datastore.store_sidecar(b"a" * 20, "png", metadata={"url": "file:///x/Apple.png",
                                                                    "content_type": "image/png"})
        paper = datastore.store_sidecar(b"p" * 30, "pdf", metadata={"name": "paper.pdf",
                                                                    "content_type": "application/pdf"})

        preview = cleanup.preview_cleanup(datastore, root)
        assert [entry.display_name for entry in preview.images] == ["Apple.png", "zebra.png"]  # case-insensitive
        assert [entry.display_name for entry in preview.documents] == ["paper.pdf"]
        assert preview.total_bytes == 60
        assert {entry.filename for entry in preview.sidecars} == {zebra, apple, paper}

    def test_counts_attachments_held_only_by_unreachable_nodes(self, datastore):
        """The whole point of the preview: what the *pair* of prunes would take, not what step one would."""
        root = datastore.create_node({"message": {"role": "system", "content": []}}, parent_id=None)
        doomed_file = datastore.store_sidecar(b"orphan-to-be", "png",
                                              metadata={"url": "file:///x/forgotten.png",
                                                        "content_type": "image/png"})
        unreachable = datastore.create_node(_payload_with_image(doomed_file, name="forgotten.png"),
                                            parent_id=None)

        preview = cleanup.preview_cleanup(datastore, root)
        assert preview.node_ids == [unreachable]
        assert [entry.display_name for entry in preview.images] == ["forgotten.png"]

    def test_preserved_original_is_folded_into_its_image(self, datastore):
        """A downsampled image and its kept original are two files but one attachment, as in the chat log.

        Both are referenced from the same payload, so both fall unreferenced together — listing them
        separately would show the same picture twice under the same name, and make the count wrong.
        """
        root = datastore.create_node({"message": {"role": "system", "content": []}}, parent_id=None)
        original = datastore.store_sidecar(b"o" * 900, "jpg",
                                           metadata={"url": "file:///x/holiday.jpg",
                                                     "content_type": "image/jpeg",
                                                     "role": "preserved_original"})
        primary = datastore.store_sidecar(b"d" * 100, "png",
                                          metadata={"url": "file:///x/holiday.jpg",
                                                    "content_type": "image/jpeg",
                                                    "original_sidecar": original})

        preview = cleanup.preview_cleanup(datastore, root)
        assert [entry.filename for entry in preview.images] == [primary]  # one tile, not two
        assert preview.images[0].companion_filenames == [original]
        assert preview.images[0].total_bytes == 1000  # both files, so the reclaim figure is honest
        assert preview.total_bytes == 1000

    def test_undownsampled_image_has_no_companion(self, datastore):
        """An image small enough to store verbatim is its own original; nothing to fold."""
        root = datastore.create_node({"message": {"role": "system", "content": []}}, parent_id=None)
        filename = datastore.store_sidecar(b"small", "png", metadata={"url": "file:///x/small.png",
                                                                      "content_type": "image/png"})
        entry = cleanup.preview_cleanup(datastore, root).images[0]
        assert entry.companion_filenames == []
        assert entry.archival_filename == filename  # opening it resolves to itself

    def test_leaves_reachable_attachments_alone(self, datastore):
        root = datastore.create_node({"message": {"role": "system", "content": []}}, parent_id=None)
        live = datastore.store_sidecar(b"in use", "pdf", metadata={"name": "live.pdf",
                                                                   "content_type": "application/pdf"})
        datastore.create_node(_payload_with_document(live, "live.pdf"), parent_id=root)
        assert cleanup.preview_cleanup(datastore, root).is_empty


# ---------------------------------------------------------------------------
# commit_cleanup
# ---------------------------------------------------------------------------

class TestCommitCleanup:
    def test_deletes_what_the_preview_promised_and_saves(self, datastore, tmp_path):
        root = datastore.create_node({"message": {"role": "system", "content": []}}, parent_id=None)
        live = datastore.store_sidecar(b"in use", "png", metadata={"url": "file:///x/live.png",
                                                                   "content_type": "image/png"})
        doomed = datastore.store_sidecar(b"orphaned", "png", metadata={"url": "file:///x/doomed.png",
                                                                       "content_type": "image/png"})
        datastore.create_node(_payload_with_image(live, name="live.png"), parent_id=root)
        unreachable = datastore.create_node(_payload_with_image(doomed, name="doomed.png"), parent_id=None)

        preview = cleanup.preview_cleanup(datastore, root)
        result = cleanup.commit_cleanup(datastore, root)

        assert result.deleted_node_ids == preview.node_ids == [unreachable]
        assert result.deleted_sidecars == [doomed]
        assert datastore.list_sidecar_files() == [live]
        assert (tmp_path / "chat.json").exists()  # "& save" is part of the operation, not the caller's job

    def test_takes_the_description_file_with_the_sidecar(self, datastore):
        """Otherwise the descriptions become their own slow leak — the very thing the sweep exists to stop."""
        root = datastore.create_node({"message": {"role": "system", "content": []}}, parent_id=None)
        doomed = datastore.store_sidecar(b"orphaned", "png", metadata={"url": "file:///x/doomed.png",
                                                                       "content_type": "image/png"})
        assert datastore.get_sidecar_metadata(doomed) is not None

        cleanup.commit_cleanup(datastore, root)
        assert datastore.get_sidecar_metadata(doomed) is None
        assert list(datastore.sidecar_dir.iterdir()) == []

    def test_is_idempotent(self, datastore):
        root = datastore.create_node({"message": {"role": "system", "content": []}}, parent_id=None)
        datastore.store_sidecar(b"orphaned", "png")
        cleanup.commit_cleanup(datastore, root)
        second = cleanup.commit_cleanup(datastore, root)
        assert second.deleted_node_ids == []
        assert second.deleted_sidecars == []


# ---------------------------------------------------------------------------
# rescue_to_staging
# ---------------------------------------------------------------------------

class TestRescueToStaging:
    def test_copies_under_the_human_readable_name(self, datastore, tmp_path):
        filename = datastore.store_sidecar(b"important", "pdf", metadata={"name": "thesis.pdf",
                                                                          "content_type": "application/pdf"})
        entry = cleanup.describe_sidecar(datastore, filename)
        staging = tmp_path / "staging"

        staged = cleanup.rescue_to_staging(datastore, entry, staging)
        assert staged == staging / "thesis.pdf"
        assert staged.read_bytes() == b"important"
        assert datastore.sidecar_path(filename).exists()  # copy, not move: cancelling the cleanup must be free

    def test_rescuing_twice_is_idempotent(self, datastore, tmp_path):
        filename = datastore.store_sidecar(b"important", "pdf", metadata={"name": "thesis.pdf"})
        entry = cleanup.describe_sidecar(datastore, filename)
        staging = tmp_path / "staging"

        first = cleanup.rescue_to_staging(datastore, entry, staging)
        second = cleanup.rescue_to_staging(datastore, entry, staging)
        assert first == second
        assert list(staging.iterdir()) == [first]  # same bytes under the same name -> the same rescue

    def test_different_content_under_the_same_name_does_not_overwrite(self, datastore, tmp_path):
        """Two attachments really can share a display name; losing one to the other is not acceptable."""
        staging = tmp_path / "staging"
        staging.mkdir()
        (staging / "notes.pdf").write_bytes(b"a different document entirely")

        filename = datastore.store_sidecar(b"my notes", "pdf", metadata={"name": "notes.pdf"})
        entry = cleanup.describe_sidecar(datastore, filename)

        staged = cleanup.rescue_to_staging(datastore, entry, staging)
        assert staged == staging / "notes (2).pdf"
        assert (staging / "notes.pdf").read_bytes() == b"a different document entirely"

    def test_rescues_the_preserved_original_not_the_downsample(self, datastore, tmp_path):
        """Handing the user the downsample would give them a copy strictly worse than one sitting beside it."""
        root = datastore.create_node({"message": {"role": "system", "content": []}}, parent_id=None)
        original = datastore.store_sidecar(b"full resolution bytes", "jpg",
                                           metadata={"url": "file:///x/holiday.jpg",
                                                     "content_type": "image/jpeg",
                                                     "role": "preserved_original"})
        datastore.store_sidecar(b"downsampled", "png", metadata={"url": "file:///x/holiday.jpg",
                                                                 "content_type": "image/jpeg",
                                                                 "original_sidecar": original})
        entry = cleanup.preview_cleanup(datastore, root).images[0]

        staged = cleanup.rescue_to_staging(datastore, entry, tmp_path / "staging")
        assert staged.name == "holiday.jpg"
        assert staged.read_bytes() == b"full resolution bytes"

    def test_hash_named_sidecar_keeps_its_extension(self, datastore, tmp_path):
        """With no description there is no better name, but the extension still has to survive the rescue."""
        filename = datastore.store_sidecar(b"legacy", "png")
        entry = cleanup.describe_sidecar(datastore, filename)
        staged = cleanup.rescue_to_staging(datastore, entry, tmp_path / "staging")
        assert staged.suffix == ".png"
        assert staged.read_bytes() == b"legacy"


# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------

class TestFormatting:
    @pytest.mark.parametrize("size_bytes, expected", [(0, "0 B"),
                                                      (999, "999 B"),
                                                      (1000, "1.0 kB"),
                                                      (150_000, "150 kB"),
                                                      (3_400_000, "3.4 MB"),
                                                      (2_000_000_000, "2.0 GB")])
    def test_format_size(self, size_bytes, expected):
        assert cleanup.format_size(size_bytes) == expected

    def test_ellipsize_keeps_both_ends(self):
        """The topic is at the front of a filename and the file type at the back; both have to survive."""
        result = cleanup._ellipsize("quarterly_report_2026_final.pdf", 22)
        assert len(result) == 22
        assert result.startswith("quarterly_r")
        assert result.endswith(".pdf")

    def test_ellipsize_leaves_short_names_alone(self):
        assert cleanup._ellipsize("short.pdf", 22) == "short.pdf"

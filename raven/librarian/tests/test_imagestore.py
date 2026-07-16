"""Unit tests for raven.librarian.imagestore (image sidecar store, wire resolution, GC mark phase)."""

import io

import numpy as np
from PIL import Image

import pytest

from raven.librarian import chattree, imagestore
from raven.librarian import config as librarian_config


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def datastore(tmp_path):
    """A fresh no-autosave PersistentForest with a temp-dir sidecar directory, GC-configured."""
    return chattree.PersistentForest(tmp_path / "chat.json", autosave=False,
                                     sidecar_extractor=imagestore.sidecar_refs_in_payload)


def _png_bytes(width, height, color=(200, 30, 30), mode="RGB"):
    image = Image.new(mode, (width, height), color if mode != "RGBA" else color + (255,))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _decode_mp(data):
    array = np.asarray(Image.open(io.BytesIO(data)))
    return (array.shape[0] * array.shape[1]) / 2 ** 20


# ---------------------------------------------------------------------------
# downsample_dims
# ---------------------------------------------------------------------------

def test_downsample_dims_hits_cap_and_preserves_aspect():
    # Brief's worked example: 4000x3000 (12 MP) at a 1 MP cap -> ~1183x887.
    new_h, new_w = imagestore.downsample_dims(4000, 3000, 1.0)
    assert abs(new_h - 1183) <= 1 and abs(new_w - 887) <= 1
    # within the cap, and aspect ratio preserved to sub-percent.
    assert new_h * new_w <= 1.02 * 2 ** 20
    assert abs((new_w / new_h) - (3000 / 4000)) < 0.01


def test_downsample_dims_square():
    new_h, new_w = imagestore.downsample_dims(2000, 2000, 1.0)
    assert new_h == new_w == 1024  # sqrt(2**20) == 1024 exactly


# ---------------------------------------------------------------------------
# store_image_as_sidecar — the three storage cases
# ---------------------------------------------------------------------------

def test_store_case1_under_cap_is_verbatim_primary(datastore):
    """Image within the cap: primary sidecar IS the byte-exact original (metadata preserved); no second file."""
    raw = _png_bytes(64, 48)
    result = imagestore.store_image_as_sidecar(datastore, raw,
                                               provenance_url="file:///tmp/x.png",
                                               provenance_source="user_attachment")
    assert result.part == {"type": "image_url", "image_url": {"url": f"sidecar:{result.filename}"}}
    assert datastore.read_sidecar(result.filename) == raw  # byte-for-byte
    md = result.sidecar_metadata
    assert md["url"] == "file:///tmp/x.png"
    assert md["source"] == "user_attachment"
    assert md["content_type"] == "image/png"
    assert "original_sidecar" not in md and "original_dimensions" not in md
    assert len(datastore.list_sidecar_files()) == 1


def test_store_dedup_identical_bytes(datastore):
    raw = _png_bytes(64, 48)
    r1 = imagestore.store_image_as_sidecar(datastore, raw, provenance_url="file:///a.png", provenance_source="user_attachment")
    r2 = imagestore.store_image_as_sidecar(datastore, raw, provenance_url="file:///b.png", provenance_source="user_attachment")
    assert r1.filename == r2.filename  # content-addressed
    assert len(datastore.list_sidecar_files()) == 1


def test_store_case2_over_cap_downsamples_and_keeps_original(datastore):
    """Over cap with store_original_image=True (default): downsampled primary + verbatim original sidecar."""
    raw = _png_bytes(2000, 2000, color=(10, 120, 240))  # 4 MP
    result = imagestore.store_image_as_sidecar(datastore, raw,
                                               provenance_url="file:///big.png",
                                               provenance_source="user_attachment")
    md = result.sidecar_metadata
    assert md["original_dimensions"] == [2000, 2000]
    assert md["original_size_bytes"] == len(raw)
    assert "original_sidecar" in md
    # primary is downsampled to within the cap; original is byte-exact.
    assert _decode_mp(datastore.read_sidecar(result.filename)) <= 1.02
    assert datastore.read_sidecar(md["original_sidecar"]) == raw
    assert result.filename != md["original_sidecar"]
    assert len(datastore.list_sidecar_files()) == 2


def test_store_case3_over_cap_discard_original(datastore, monkeypatch):
    monkeypatch.setattr(librarian_config, "store_original_image", False)
    raw = _png_bytes(2000, 2000, color=(10, 120, 240))
    result = imagestore.store_image_as_sidecar(datastore, raw,
                                               provenance_url="file:///big.png",
                                               provenance_source="user_attachment")
    md = result.sidecar_metadata
    assert md["original_dimensions"] == [2000, 2000]  # still recorded for provenance
    assert "original_sidecar" not in md  # but not kept on disk
    assert len(datastore.list_sidecar_files()) == 1


def test_store_cap_disabled_stores_verbatim(datastore, monkeypatch):
    monkeypatch.setattr(librarian_config, "image_store_max_megapixels", None)
    raw = _png_bytes(2000, 2000)
    result = imagestore.store_image_as_sidecar(datastore, raw, provenance_url="file:///big.png", provenance_source="user_attachment")
    assert datastore.read_sidecar(result.filename) == raw  # no downsampling at all
    assert "original_dimensions" not in result.sidecar_metadata


def test_store_palette_image_does_not_crash(datastore):
    """A palette-mode ('P') PNG over the cap must downsample cleanly rather than trip the tensor conversion."""
    palette_image = Image.new("P", (2000, 2000))
    palette_image.putpalette([i % 256 for _ in range(256) for i in (1, 2, 3)])
    buffer = io.BytesIO()
    palette_image.save(buffer, format="PNG")
    result = imagestore.store_image_as_sidecar(datastore, buffer.getvalue(),
                                               provenance_url="file:///pal.png", provenance_source="user_attachment")
    assert _decode_mp(datastore.read_sidecar(result.filename)) <= 1.02


# ---------------------------------------------------------------------------
# sidecar_url_to_data_url — wire substitution
# ---------------------------------------------------------------------------

def test_sidecar_url_to_data_url_roundtrip(datastore):
    raw = _png_bytes(32, 32)
    result = imagestore.store_image_as_sidecar(datastore, raw, provenance_url="file:///x.png", provenance_source="user_attachment")
    data_url = imagestore.sidecar_url_to_data_url(datastore, result.part["image_url"]["url"])
    assert data_url.startswith("data:image/png;base64,")
    # the base64 payload decodes back to the exact stored bytes.
    import base64
    payload = data_url.split(",", 1)[1]
    assert base64.b64decode(payload) == raw


def test_sidecar_url_to_data_url_rejects_non_sidecar(datastore):
    with pytest.raises(ValueError):
        imagestore.sidecar_url_to_data_url(datastore, "https://example.com/foo.png")


# ---------------------------------------------------------------------------
# sidecar_refs_in_payload (per-payload GC mark interpreter)
# ---------------------------------------------------------------------------

def test_sidecar_refs_in_payload_scans_both_sites():
    payload = {"message": {"role": "user",
                           "content": [{"type": "text", "text": "look:"},
                                       {"type": "image_url", "image_url": {"url": "sidecar:aaa.png"}}]},
               "general_metadata": {"sidecars": {"aaa.png": {"original_sidecar": "aaa.original.png"}}}}
    assert imagestore.sidecar_refs_in_payload(payload) == {"aaa.png", "aaa.original.png"}


def test_sidecar_refs_in_payload_ignores_legacy_string_content():
    """A pre-migration payload whose content is still a bare string must not crash the interpreter."""
    assert imagestore.sidecar_refs_in_payload({"message": {"role": "user", "content": "not yet migrated"}}) == set()


def test_sidecar_refs_in_payload_ignores_non_sidecar_urls():
    payload = {"message": {"content": [{"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}]}}
    assert imagestore.sidecar_refs_in_payload(payload) == set()


# ---------------------------------------------------------------------------
# GC integration: chattree drives the traversal, imagestore interprets payloads
# ---------------------------------------------------------------------------

def test_prune_keeps_referenced_sweeps_orphans(datastore):
    referenced_img = _png_bytes(2000, 2000, color=(1, 2, 3))    # over cap -> primary + original
    orphan_img = _png_bytes(64, 48, color=(9, 9, 9))            # under cap -> single file

    kept = imagestore.store_image_as_sidecar(datastore, referenced_img, provenance_url="file:///k.png", provenance_source="user_attachment")
    orphan = imagestore.store_image_as_sidecar(datastore, orphan_img, provenance_url="file:///o.png", provenance_source="user_attachment")

    payload = {"message": {"role": "user", "content": [kept.part]},
               "general_metadata": {"sidecars": {kept.filename: kept.sidecar_metadata}}}
    datastore.create_node(payload, parent_id=None)

    # dry-run first: reports exactly the orphan, deletes nothing.
    would_delete = datastore.unreferenced_sidecars()
    assert would_delete == [orphan.filename]
    assert orphan.filename in datastore.list_sidecar_files()  # untouched by the dry-run

    deleted = datastore.prune_unreferenced_sidecars()
    assert deleted == [orphan.filename]
    remaining = set(datastore.list_sidecar_files())
    assert kept.filename in remaining
    assert kept.sidecar_metadata["original_sidecar"] in remaining  # kept via metadata, not a content-part
    assert orphan.filename not in remaining


def test_prune_without_extractor_is_safe_noop(tmp_path):
    """A datastore with no configured extractor must never delete sidecars it can't prove are unreferenced."""
    ds = chattree.PersistentForest(tmp_path / "chat.json", autosave=False)  # no sidecar_extractor
    ds.store_sidecar(b"\x89PNG\r\n\x1a\n" + b"x" * 32, "png")
    assert ds.prune_unreferenced_sidecars() == []
    assert len(ds.list_sidecar_files()) == 1  # untouched

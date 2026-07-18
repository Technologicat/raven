"""Unit tests for raven.librarian.sidecarstore (shared attachment-sidecar foundation)."""

import pytest

from raven.librarian import sidecarstore


# ---------------------------------------------------------------------------
# read_source_bytes
# ---------------------------------------------------------------------------

def test_read_source_bytes_passthrough_copies():
    src = bytearray(b"mutate me")
    out = sidecarstore.read_source_bytes(src)
    assert out == b"mutate me"
    assert isinstance(out, bytes)
    src[0] = ord("X")  # a later mutation of the caller's bytearray must not reach what we returned
    assert out == b"mutate me"

def test_read_source_bytes_from_path(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_bytes(b"on disk")
    assert sidecarstore.read_source_bytes(p) == b"on disk"
    assert sidecarstore.read_source_bytes(str(p)) == b"on disk"  # str path too


# ---------------------------------------------------------------------------
# base_provenance
# ---------------------------------------------------------------------------

def test_base_provenance_has_the_four_common_keys():
    md = sidecarstore.base_provenance(url="file:///a", source="user_attachment",
                                      content_type="text/plain", fetched_at="2026-07-18 12:00:00")
    assert md == {"url": "file:///a",
                  "fetched_at": "2026-07-18 12:00:00",
                  "content_type": "text/plain",
                  "source": "user_attachment"}

def test_base_provenance_defaults_fetched_at_when_none():
    md = sidecarstore.base_provenance(url="file:///a", source="user_attachment",
                                      content_type="text/plain", fetched_at=None)
    # Filled with a formatted local-time string, not left as None.
    assert md["fetched_at"] and md["fetched_at"] != "None"
    assert len(md["fetched_at"]) == len("YYYY-MM-DD HH:MM:SS")

def test_base_provenance_returns_a_fresh_mutable_dict():
    md = sidecarstore.base_provenance(url="file:///a", source="s", content_type="t", fetched_at="x")
    md["name"] = "extra.txt"  # each store extends the returned dict with its own kind-specific fields
    md2 = sidecarstore.base_provenance(url="file:///a", source="s", content_type="t", fetched_at="x")
    assert "name" not in md2  # not a shared/aliased dict


# ---------------------------------------------------------------------------
# sidecar_filename_from_url
# ---------------------------------------------------------------------------

def test_sidecar_filename_from_url_strips_scheme():
    assert sidecarstore.sidecar_filename_from_url("sidecar:abc.png", caller="t") == "abc.png"

def test_sidecar_filename_from_url_rejects_non_sidecar():
    with pytest.raises(ValueError, match="expected a 'sidecar:' URL"):
        sidecarstore.sidecar_filename_from_url("https://example.com/x.png", caller="t")

def test_sidecar_filename_from_url_names_the_caller():
    with pytest.raises(ValueError, match="my_resolver:"):
        sidecarstore.sidecar_filename_from_url("data:...", caller="my_resolver")


# ---------------------------------------------------------------------------
# content_part_sidecar_refs
# ---------------------------------------------------------------------------

def _payload(*parts):
    return {"message": {"role": "user", "content": list(parts)}}

def test_content_part_sidecar_refs_selects_only_the_requested_type():
    payload = _payload({"type": "text", "text": "hi"},
                       {"type": "image_url", "image_url": {"url": "sidecar:img.png"}},
                       {"type": "text_file", "text_file": {"url": "sidecar:doc.txt", "name": "doc.txt"}})
    assert sidecarstore.content_part_sidecar_refs(payload, "image_url") == {"img.png"}
    assert sidecarstore.content_part_sidecar_refs(payload, "text_file") == {"doc.txt"}

def test_content_part_sidecar_refs_ignores_non_sidecar_urls():
    payload = _payload({"type": "image_url", "image_url": {"url": "https://example.com/x.png"}})
    assert sidecarstore.content_part_sidecar_refs(payload, "image_url") == set()

def test_content_part_sidecar_refs_legacy_string_content_is_empty():
    assert sidecarstore.content_part_sidecar_refs({"message": {"content": "pre-migration string"}}, "image_url") == set()

def test_content_part_sidecar_refs_missing_message_is_empty():
    assert sidecarstore.content_part_sidecar_refs({}, "text_file") == set()

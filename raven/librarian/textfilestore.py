"""Text/PDF document sidecar lifecycle for Librarian chat messages: store, resolve for the wire, and GC.

The file sibling of `imagestore`. When the user attaches a document (plain text or PDF) to a message, its bytes
are stored *verbatim* as a sidecar file next to the chat datastore JSON (in the same content-addressed sidecar
store `chattree` manages for images), and referenced from the message by a `text_file` content part carrying a
`sidecar:<filename>` URL and the original filename. No document text is written inline into the chat JSON, so
the datastore stays small even for a large PDF, and a saved chat reloads offline.

Unlike an image (which the model consumes natively as a `data:` URL), a document has no native wire form: its
plaintext is extracted on demand (`raven.common.docextract`) and folded into the message's text at wire-build
time by `llmclient.invoke`. So any model can use an attached document — no vision capability required.

The shared sidecar mechanics (URL scheme, provenance skeleton, byte ingestion, GC content-walk) live in
`sidecarstore`, the common foundation with `imagestore`. Three public operations, mirroring `imagestore`:

  - `store_file_as_sidecar`: store the document bytes verbatim, return the `text_file` content-part plus the
    provenance metadata entry.
  - `sidecar_to_text`: resolve a stored `sidecar:` URL to the document's extracted plaintext, memoized on
    the content-addressed filename (so a chat with an attached PDF re-extracts it at most once per process).
  - `sidecar_refs_in_payload`: the GC mark-phase interpreter for `text_file` parts. Compose it (set union) with
    `imagestore.sidecar_refs_in_payload` when configuring a datastore's `sidecar_extractor`, so both attached
    images and attached documents are seen by the mark phase.
"""

__all__ = ["store_file_as_sidecar",
           "sidecar_to_text",
           "sidecar_refs_in_payload"]

import logging
logger = logging.getLogger(__name__)

import pathlib

from unpythonic.env import env

from ..common import docextract

from . import chatutil
from . import chattree
from . import sidecarstore


# Extracted-text cache, keyed by the content-addressed sidecar filename (`<sha256>.<ext>`). A sidecar is
# immutable and content-addressed, so its extracted text is a pure function of the filename and can never go
# stale; the cache only avoids re-running the (potentially slow) PDF extractor on every wire-build of a chat
# that carries an attached document.
_extracted_text_cache: dict[str, str] = {}


def _mime_for_ext(ext: str) -> str:
    """A reasonable MIME type for a document extension (no leading dot). Informational provenance only."""
    ext = ext.lower().lstrip(".")
    if ext == "pdf":
        return "application/pdf"
    return "text/plain"


def store_file_as_sidecar(datastore: chattree.PersistentForest,
                          file_source: bytes | str | pathlib.Path,
                          *,
                          name: str,
                          provenance_url: str,
                          provenance_source: str,
                          content_type: str | None = None,
                          fetched_at: str | None = None) -> env:
    """Store an attached document as a datastore sidecar; return its content-part and provenance metadata.

    `datastore`: the `PersistentForest` whose sidecar directory receives the file.
    `file_source`: the document bytes, or a filesystem path (`str` / `pathlib.Path`) to read them from.
    `name`: the original filename (e.g. `"report.pdf"`), kept for display and for the wire header, and used to
            derive the sidecar's file extension so the extractor later dispatches by type.
    `provenance_url`: the `url` recorded in provenance — for a user-attached local file, `"file:///<abspath>"`.
    `provenance_source`: the categorical pathway — `"user_attachment"`, `"paste_url"`, or `"mcp:<server>"`.
    `content_type`: original MIME type; derived from the extension if `None`.
    `fetched_at`: materialization timestamp string; current local time if `None`.

    The document is stored byte-for-byte — no transformation, unlike an image, which may be downsampled.

    Returns an `env` with:
      `part`: the `text_file` content-part to append to the message content — its URL is `sidecar:<filename>`.
      `filename`: the sidecar's filename (the key under which `sidecar_metadata` should be recorded).
      `sidecar_metadata`: the provenance dict to store at `general_metadata["sidecars"][filename]`.
    """
    raw = sidecarstore.read_source_bytes(file_source)

    ext = pathlib.Path(name).suffix.lstrip(".").lower() or "txt"
    content_type = content_type or _mime_for_ext(ext)

    filename = datastore.store_sidecar(raw, ext)
    metadata = sidecarstore.base_provenance(url=provenance_url, source=provenance_source,
                                            content_type=content_type, fetched_at=fetched_at)
    metadata["name"] = name
    metadata["size_bytes"] = len(raw)
    return env(part=chatutil.text_file_content_part(f"{sidecarstore.SIDECAR_SCHEME}{filename}", name),
               filename=filename,
               sidecar_metadata=metadata)


def sidecar_to_text(datastore: chattree.PersistentForest, url: str) -> str:
    """Resolve a stored `sidecar:<filename>` document URL to its extracted plaintext, memoized by filename.

    Reads the sidecar file and extracts its text via `raven.common.docextract` (plain text verbatim; PDF text
    layer via pypdf). Used by `llmclient.invoke` to fold an attached document into the outgoing message text.
    An extraction failure or an empty document degrades to a short bracketed placeholder rather than raising, so
    a single unreadable attachment can never break the LLM call.
    """
    filename = sidecarstore.sidecar_filename_from_url(url, caller="sidecar_to_text")
    if filename in _extracted_text_cache:
        return _extracted_text_cache[filename]
    path = datastore.sidecar_path(filename)
    try:
        text = docextract.extract_text(path)
    except Exception as exc:  # noqa: BLE001 -- wire-build must never crash on one unreadable attachment
        logger.warning(f"sidecar_to_text: could not extract text from sidecar '{filename}': {type(exc)}: {exc}")
        text = None
    if not text:
        text = "[no extractable text]"
    _extracted_text_cache[filename] = text
    return text


def sidecar_refs_in_payload(payload: dict) -> set[str]:
    """Return the `text_file` sidecar filenames referenced by a single node `payload` — the GC mark interpreter.

    The file sibling of `imagestore.sidecar_refs_in_payload`; take the set union of the two when configuring a
    datastore's `sidecar_extractor`, so both attached images and attached documents are seen by the mark phase.
    Robust to a pre-migration bare-string `content` (returns no refs rather than iterating the string).
    """
    return sidecarstore.content_part_sidecar_refs(payload, "text_file")

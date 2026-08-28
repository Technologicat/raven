"""Text/PDF document sidecar lifecycle for Librarian chat messages: store, resolve for the wire, and GC.

The file sibling of `imagestore`. When a document (plain text, PDF, markdown, ...) becomes part of a message —
the user attached it, or a tool fetched one and `scaffold` stored it rather than dumping it into the chat log —
its bytes are stored *verbatim* as a sidecar file next to the chat datastore JSON (in the same content-addressed
sidecar store `chattree` manages for images), and referenced from the message by a `text_file` content part
carrying a `sidecar:<filename>` URL and the original filename. No document text is written inline into the chat
JSON, so the datastore stays small even for a large PDF, and a saved chat reloads offline.

Unlike an image (which the model consumes natively as a `data:` URL), a document has no native wire form: its
plaintext is extracted on demand (`raven.common.docextract`) and folded into the message's text at wire-build
time by `llmclient.invoke`. So any model can use an attached document — no vision capability required.

The shared sidecar mechanics (URL scheme, provenance skeleton, byte ingestion, GC content-walk) live in
`sidecarstore`, the common foundation with `imagestore`. Three public operations, mirroring `imagestore`:

  - `store_file_as_sidecar`: store the document bytes verbatim, return the `text_file` content-part plus the
    provenance metadata entry.
  - `sidecar_to_text`: resolve a stored `sidecar:` URL to the document's extracted plaintext, memoized on
    the content-addressed filename (so a chat with an attached PDF re-extracts it at most once per process).
    `sidecar_text_if_extracted` is the same question asked without paying it — the answer if the work is
    already done, `None` if it is not — for a caller that would rather do without than wait for pypdf.
  - `sidecar_refs_in_payload`: the GC mark-phase interpreter for `text_file` parts. Compose it (set union) with
    `imagestore.sidecar_refs_in_payload` when configuring a datastore's `sidecar_extractor`, so both attached
    images and attached documents are seen by the mark phase.
"""

__all__ = ["remember_extracted_text",
           "store_file_as_sidecar",
           "sidecar_to_text",
           "sidecar_text_if_extracted",
           "sidecar_refs_in_payload"]

import logging
logger = logging.getLogger(__name__)

import pathlib
from typing import Optional

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


# Spelled out rather than taken from `mimetypes.guess_type`, whose answers depend on the host's mime.types file
# — provenance recorded in a datastore should not differ between two machines that attached the same document.
_MIME_TYPES = {"pdf": "application/pdf",
               "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
               "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
               "odt": "application/vnd.oasis.opendocument.text",
               "odp": "application/vnd.oasis.opendocument.presentation"}


def _mime_for_ext(ext: str) -> str:
    """A reasonable MIME type for a document extension (no leading dot). Informational provenance only."""
    return _MIME_TYPES.get(ext.lower().lstrip("."), "text/plain")


def _ext_for_name(name: str) -> str:
    """The sidecar extension a document called `name` gets (no leading dot); "txt" when it has none.

    The extension is what selects the extractor later, so every path that computes one has to agree — which
    is why `remember_extracted_text` and `store_file_as_sidecar` both come here rather than each spelling it
    out.
    """
    return pathlib.Path(name).suffix.lstrip(".").lower() or "txt"


def remember_extracted_text(name: str, raw: bytes, text: str) -> None:
    """Record `text` as the extracted plaintext of the document `raw`, to be stored later under `name`.

    For a caller that has already extracted a document's text for its own reasons — validating an attachment
    at the moment the user picks it, say — and would otherwise throw the result away. Extraction is the
    expensive step (seconds for a large PDF) and it runs again at wire-build; this makes the second run free.

    Safe to call before the document is stored, and safe to call for one that never is. A sidecar's name is a
    hash of its bytes, so it is knowable as soon as the bytes are, and an entry for a document that is never
    attached costs one dictionary slot for the session.
    """
    _extracted_text_cache[chattree.sidecar_filename_for(raw, _ext_for_name(name))] = text


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
    `provenance_url`: the `url` recorded in provenance — for a user-attached local file, `"file:///<abspath>"`;
                      for a document a tool fetched, the URL it came from.
    `provenance_source`: the categorical pathway; see `sidecarstore.base_provenance` for the vocabulary.
                         `"user_attachment"` and `"tool_result"` are what currently emit.
    `content_type`: original MIME type; derived from the extension if `None`.
    `fetched_at`: materialization timestamp string; current local time if `None`.

    The document is stored byte-for-byte — no transformation, unlike an image, which may be downsampled.

    Returns an `env` with:
      `part`: the `text_file` content-part to append to the message content — its URL is `sidecar:<filename>`.
      `filename`: the sidecar's filename (the key under which `sidecar_metadata` should be recorded).
      `sidecar_metadata`: the provenance dict to store at `general_metadata["sidecars"][filename]`.
    """
    raw = sidecarstore.read_source_bytes(file_source)

    ext = _ext_for_name(name)
    content_type = content_type or _mime_for_ext(ext)

    metadata = sidecarstore.base_provenance(url=provenance_url, source=provenance_source,
                                            content_type=content_type, fetched_at=fetched_at)
    metadata["name"] = name
    metadata["size_bytes"] = len(raw)
    # The same provenance goes into the payload (as the caller's return value) and beside the file. The payload
    # copy is what the chat log reads; the file-side copy is what remains once the referencing node is deleted,
    # which is exactly when a cleanup preview needs to say what the orphan was.
    filename = datastore.store_sidecar(raw, ext, metadata=metadata)
    # `provenance_source` is recorded twice on purpose — beside the file, and on the content-part. The
    # part-side copy is the one the wire builder can reach; see `chatutil.text_file_content_part`.
    return env(part=chatutil.text_file_content_part(f"{sidecarstore.SIDECAR_SCHEME}{filename}", name,
                                                    provenance_source),
               filename=filename,
               sidecar_metadata=metadata)


def sidecar_to_text(datastore: chattree.PersistentForest, url: str) -> str:
    """Resolve a stored `sidecar:<filename>` document URL to its extracted plaintext, memoized by filename.

    Reads the sidecar's bytes and extracts its text via `raven.common.docextract`, which decides what each
    format means — plain text verbatim, a PDF's text layer, an office document's prose and tables, an HTML
    page's readable content. Used by `llmclient.invoke` to fold an attached document into the outgoing message text.
    An extraction failure or an empty document degrades to a short bracketed placeholder rather than raising, so
    a single unreadable attachment can never break the LLM call.
    """
    filename = sidecarstore.sidecar_filename_from_url(url, caller="sidecar_to_text")
    if filename in _extracted_text_cache:
        return _extracted_text_cache[filename]
    try:
        # By bytes rather than by path: the extension of the content-addressed name still selects the
        # reader, and asking the store for the bytes is the narrower requirement — a store that can hand
        # them over needs no filesystem, which is what an attachment to an in-memory chat would want.
        text = docextract.extract_text_from_bytes(datastore.read_sidecar(filename), filename)
    except Exception as exc:  # noqa: BLE001 -- wire-build must never crash on one unreadable attachment
        logger.warning(f"sidecar_to_text: could not extract text from sidecar '{filename}': {type(exc)}: {exc}")
        text = None
    if not text:
        text = "[no extractable text]"
    _extracted_text_cache[filename] = text
    return text


def sidecar_text_if_extracted(url: str) -> Optional[str]:
    """The extracted plaintext for `url`, but **only if extracting it has already happened**. Else `None`.

    For a caller that wants the text if it is free and would rather do without it than wait — the GUI's
    context-fill readout being the case this exists for. Extraction of a large PDF takes seconds, and that
    readout is refreshed on every HEAD change, from a DPG callback: paying for it there freezes the keyboard
    until it finishes, since DPG runs callbacks one at a time.

    `None` therefore means *not extracted yet*, never *no text* — an unreadable document is cached as a
    placeholder by `sidecar_to_text`, so a second call returns that rather than `None`.
    """
    filename = sidecarstore.sidecar_filename_from_url(url, caller="sidecar_text_if_extracted")
    return _extracted_text_cache.get(filename)


def sidecar_refs_in_payload(payload: dict) -> set[str]:
    """Return the `text_file` sidecar filenames referenced by a single node `payload` — the GC mark interpreter.

    The file sibling of `imagestore.sidecar_refs_in_payload`; take the set union of the two when configuring a
    datastore's `sidecar_extractor`, so both attached images and attached documents are seen by the mark phase.
    Robust to a pre-migration bare-string `content` (returns no refs rather than iterating the string).
    """
    return sidecarstore.content_part_sidecar_refs(payload, "text_file")

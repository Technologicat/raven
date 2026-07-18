"""Shared foundation for the Librarian's per-kind attachment sidecar stores (`imagestore`, `textfilestore`).

An attachment to a chat message — an image or a document — is stored as a *sidecar file* next to the chat
datastore JSON (content-addressed, in `<datastore>.images/`, managed by `chattree.PersistentForest`), and
referenced from the message by a Raven-internal `sidecar:<filename>` URL. Two kind-specific modules build on
this: `imagestore` (images, resolved to `data:` URLs for the wire) and `textfilestore` (documents, resolved to
extracted plaintext). They differ in transform, content-part shape, and wire resolution, but share the
mechanics beneath: the URL scheme, the provenance-metadata skeleton, byte ingestion from a bytes-or-path
source, the scheme-strip both resolvers need, and the GC mark-phase content-list walk. Those live here so the
two kind-specific modules have a single source of truth for them and can't drift apart under maintenance.

This module is deliberately dependency-light — stdlib only, no `chatutil` / `chattree` / `config` — so it can
sit beneath every attachment store. It knows the *sidecar URL scheme and provenance shape*; it does not know
any content-part schema (which part `type` a kind uses is passed in by the caller).
"""

__all__ = ["SIDECAR_SCHEME",
           "format_now",
           "read_source_bytes",
           "base_provenance",
           "sidecar_filename_from_url",
           "content_part_sidecar_refs"]

import datetime
import pathlib

# The Raven-internal URL scheme marking an attachment part as "resolve against the datastore's sidecar directory".
# A `sidecar:` URL never leaves the datastore: a saved chat reloads offline, survives the source going away, and
# never phones home when reopened. Both image and document parts use it.
SIDECAR_SCHEME = "sidecar:"


def format_now() -> str:
    """Current local time as `"YYYY-MM-DD HH:MM:SS"` — the format used for `general_metadata["datetime"]`."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_source_bytes(source: bytes | str | pathlib.Path) -> bytes:
    """Materialize an attachment source to `bytes`: raw bytes pass through (copied), a path is read from disk.

    `source` is either the attachment's bytes, or a filesystem path (`str` / `pathlib.Path`) to read them from.
    A `bytes` / `bytearray` input is returned as a fresh immutable `bytes` (so a caller's `bytearray` can't
    later mutate what we hand to the store).
    """
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    return pathlib.Path(source).read_bytes()


def base_provenance(*,
                    url: str,
                    source: str,
                    content_type: str,
                    fetched_at: str | None) -> dict:
    """The provenance fields common to every stored sidecar, as a fresh dict for the caller to extend.

    `general_metadata["sidecars"][filename]` always carries at least these four keys; each store then adds its
    own kind-specific fields (image dimensions, document name/size, ...). Returned mutable so the caller can
    `metadata[...] = ...` its extras onto it.

    `url`: where the attachment came from — for a user-attached local file, `"file:///<absolute_path>"`.
    `source`: the categorical pathway — `"user_attachment"`, `"paste_url"`, or `"mcp:<server>"`.
    `content_type`: original MIME type.
    `fetched_at`: materialization timestamp string; defaults to the current local time if `None`.
    """
    return {"url": url,
            "fetched_at": fetched_at or format_now(),
            "content_type": content_type,
            "source": source}


def sidecar_filename_from_url(url: str, *, caller: str) -> str:
    """Strip the `sidecar:` scheme from a stored attachment URL, returning the bare sidecar filename.

    Used by both stores' wire-resolution functions, which require a *stored* reference. Raises `ValueError`
    (naming `caller`, for a legible message) if `url` isn't a `sidecar:` URL — a live `https://` / `data:` URL
    is never a valid input here.
    """
    if not url.startswith(SIDECAR_SCHEME):
        raise ValueError(f"{caller}: expected a '{SIDECAR_SCHEME}' URL, got '{url[:32]}'.")
    return url[len(SIDECAR_SCHEME):]


def content_part_sidecar_refs(payload: dict, part_type: str) -> set[str]:
    """Sidecar filenames referenced by `part_type` content-parts in a node `payload` — the shared GC mark walk.

    Both kinds carry their live reference the same way: a content-part whose `"type"` is `part_type` and whose
    nested `part[part_type]["url"]` is a `sidecar:<filename>` URL (`image_url` parts nest under `"image_url"`,
    `text_file` parts under `"text_file"` — the part type and the nesting key coincide). This walks the parts
    list and returns the `sidecar:`-scheme filenames for parts of that type. Each store calls it with its own
    part type and unions in any extra references (e.g. image originals) itself.

    Robust to a pre-migration bare-string `content` (returns an empty set rather than iterating the string),
    though in practice GC only ever runs on post-migration data.
    """
    referenced = set()
    message = payload.get("message", {})
    content = message.get("content")
    if isinstance(content, list):  # post-migration content is always a parts list; guard legacy strings
        for part in content:
            if isinstance(part, dict) and part.get("type") == part_type:
                part_url = part.get(part_type, {}).get("url", "")
                if part_url.startswith(SIDECAR_SCHEME):
                    referenced.add(part_url[len(SIDECAR_SCHEME):])
    return referenced

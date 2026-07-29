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
           "provenance_filename_from_url",
           "sidecar_filename_from_url",
           "content_part_sidecar_refs",
           "provenance_entries_in_payload"]

import datetime
import pathlib
import urllib.parse

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
    `source`: the categorical pathway that produced this attachment. `"user_attachment"` — a file the user
              picked — is the only value anything currently emits. `"paste_url"` (materialized from a URL the
              user supplied) and `"mcp:<server>"` name pathways that do not exist yet; they are listed because
              this field is read back out of stored provenance, so a reader needs to know the vocabulary is
              open rather than assume the one value is the whole of it.
    `content_type`: original MIME type.
    `fetched_at`: materialization timestamp string; defaults to the current local time if `None`.
    """
    return {"url": url,
            "fetched_at": fetched_at or format_now(),
            "content_type": content_type,
            "source": source}


def provenance_filename_from_url(maybe_url: str | None) -> str | None:
    """Best-effort original filename from a provenance URL: the basename of a `file://` or `https://` URL.

    Returns `None` for an inline `data:` URL (carries no filename), an empty URL, or a URL whose path has no
    basename (e.g. a bare host). Percent-escapes are decoded, so `.../my%20photo.png` -> `my photo.png`.

    This is how an *image* sidecar recovers a human-readable name: unlike a document, it has no `name` field of
    its own, because the stored bytes may be a re-encoded downsample rather than the file the user picked.
    """
    if not maybe_url or maybe_url.startswith("data:"):
        return None
    path = urllib.parse.urlparse(maybe_url).path
    name = pathlib.Path(urllib.parse.unquote(path)).name
    return name or None


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


def provenance_entries_in_payload(payload: dict) -> dict[str, dict]:
    """Return `{sidecar filename: provenance entry}` recorded in a node `payload`, for every kind at once.

    The counterpart to `content_part_sidecar_refs`: that answers *which* sidecars a payload references, this
    answers *what is known about them*. Both attached images and attached documents write into the same
    `general_metadata["sidecars"]` mapping, so one reader covers both — the entries differ in their fields, not
    in where they live.

    Used to backfill sidecar descriptions for datastores predating them. The information is only recoverable
    while the referencing payload still exists, which is why it is worth copying beside the file: once a node is
    deleted, its sidecars are orphaned *and* anonymous, and the cleanup preview has nothing to show but a hash.

    Returns `{}` for a payload with no attachments, and skips malformed entries rather than raising — this runs
    over data written by older versions, where being lenient costs less than being right.
    """
    sidecars = payload.get("general_metadata", {}).get("sidecars", {})
    if not isinstance(sidecars, dict):
        return {}
    return {filename: entry for filename, entry in sidecars.items()
            if isinstance(filename, str) and isinstance(entry, dict)}

"""Image sidecar lifecycle for Librarian chat messages: store, resolve for the wire, and GC.

When a vision-capable model is loaded, the user can attach images to a message. An attached image is stored
as a *sidecar file* next to the chat datastore JSON (in `<datastore>.sidecars/`, managed by
`chattree.PersistentForest`), and referenced from the message by a Raven-internal `sidecar:<filename>` URL. No
`https://` URL ever lands in a stored datastore, so a saved chat reloads without network access, survives the
source going away (link rot), and never phones home when reopened.

This is the image-specific store; the shared sidecar mechanics (URL scheme, provenance skeleton, byte
ingestion, GC content-walk) live in `sidecarstore`, the common foundation with `textfilestore`. On top of those,
this module is the bridge between three lower layers — the image codec / Lanczos resampler
(`raven.common.image`), the sidecar file store (`chattree`), and the image-storage config knobs
(`raven.librarian.config`). It knows the image content-part and provenance-metadata shapes, so the storage
layer beneath it doesn't have to.

`supported_extensions` / `is_supported` declare which image formats the composer accepts — the image sibling
of `docextract`'s pair, so a caller offering both kinds of attachment can ask both the same question.

Three public operations:

  - `store_image_as_sidecar`: decode an attached image, downsample it if it exceeds the megapixel cap, write
    the sidecar file(s), and return the `image_url` content-part plus the provenance metadata entry.
  - `sidecar_url_to_data_url`: resolve a stored `sidecar:` URL to a real `data:` URL for wire-send (used by
    `llmclient.invoke` just before a request goes out).
  - `sidecar_refs_in_payload`: read the sidecar filenames one (chattree-opaque) node payload references. This
    is the per-payload interpreter that a `chattree.PersistentForest` is configured with at construction;
    `chattree` drives the revision traversal and calls this per revision for its mark-and-sweep GC. It lives here
    because reading a `sidecar:` reference out of a payload needs the message-content-part and
    `sidecars`-metadata schema knowledge, which chattree deliberately doesn't have.
"""

__all__ = ["supported_extensions", "is_supported",

           "downsample_dims",
           "store_image_as_sidecar",
           "sidecar_url_to_data_url",
           "sidecar_refs_in_payload"]

import logging
logger = logging.getLogger(__name__)

import base64
import io
import math
import pathlib

from unpythonic.env import env

from . import chatutil
from . import chattree
from . import config as librarian_config
from . import sidecarstore

# One megapixel, in pixels. The downsample target is expressed in megapixels (config `image_store_max_megapixels`).
_ONE_MEGAPIXEL = 2 ** 20

# The image formats offered for attachment. Unlike `docextract.supported_extensions`, this cannot be derived
# from a dispatch table: Pillow decodes far more than this, so the list is a deliberate choice of what to
# *offer* rather than a report of what the decoder would manage. It lives here, beside the rest of the
# image-attach knowledge, so that the picker, the routing check and anything that has to describe the feature
# to a human all read the same list instead of each keeping its own.
#
# `.qoi` is here because Raven's own avatar recorder writes it, and a constellation that emits a format but
# refuses to read it back is missing a piece of itself. Pillow has no QOI codec, so it is transcoded to PNG on
# ingest (see `store_image_as_sidecar`) rather than handled all the way down.
_SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff", ".qoi")


def supported_extensions() -> tuple[str, ...]:
    """Return the image file extensions that can be attached (lowercase, with the leading dot).

    Deliberately mirrors `raven.common.docextract.supported_extensions`, so that a caller offering both kinds
    of attachment — the attach picker, the composer's tooltip — can treat images and documents alike.
    """
    return _SUPPORTED_EXTENSIONS

def is_supported(path: str | pathlib.Path) -> bool:
    """Return whether `path`'s file extension names an attachable image format.

    The image sibling of `docextract.is_supported`, which see.
    """
    return pathlib.Path(path).suffix.lower() in _SUPPORTED_EXTENSIONS


def _mime_for_ext(ext: str) -> str:
    """MIME type for a sidecar filename extension (no leading dot), e.g. "png" -> "image/png"."""
    ext = ext.lower().lstrip(".")
    if ext in ("jpg", "jpeg", "mpo"):
        return "image/jpeg"
    return f"image/{ext}"

def _mime_for_pil_format(pil_format: str) -> str:
    """MIME type for a PIL format name, e.g. "JPEG" -> "image/jpeg", "PNG" -> "image/png"."""
    return _mime_for_ext(pil_format)

def downsample_dims(height: int, width: int, max_megapixels: float) -> tuple[int, int]:
    """Target `(height, width)` to fit `height * width` within `max_megapixels`, aspect ratio preserved.

    Solves `H * W = max_megapixels * 2**20` at fixed aspect `r = W / H`: `new_H = sqrt(cap / r)`,
    `new_W = sqrt(cap * r)`. E.g. a 4000x3000 (12 MP) image at a 1.0 MP cap -> ~1183x887 (~1.05 MP). Each
    dimension is clamped to at least 1 pixel.
    """
    cap = max_megapixels * _ONE_MEGAPIXEL
    aspect = width / height
    new_height = max(1, round(math.sqrt(cap / aspect)))
    new_width = max(1, round(math.sqrt(cap * aspect)))
    return new_height, new_width


def store_image_as_sidecar(datastore: chattree.PersistentForest,
                           image_source: bytes | str | pathlib.Path,
                           *,
                           provenance_url: str,
                           provenance_source: str,
                           content_type: str | None = None,
                           fetched_at: str | None = None) -> env:
    """Store an attached image as a datastore sidecar; return its content-part and provenance metadata.

    `datastore`: the `PersistentForest` whose sidecar directory receives the file(s).
    `image_source`: the image bytes, or a filesystem path (`str` / `pathlib.Path`) to read them from.
    `provenance_url`: the `url` recorded in the provenance entry — where the image came from. For a
                      user-attached local file, `"file:///<absolute_path>"`; for a future paste/fetch,
                      the `https://...` source. Recorded as provenance only; never used as a live reference.
    `provenance_source`: the categorical pathway; see `sidecarstore.base_provenance` for the vocabulary.
                         `"user_attachment"` is the only value anything currently emits.
    `content_type`: original MIME type; sniffed from the image if `None`. Recorded verbatim even when the
                    stored primary is a re-encoded downsample (it documents the *original*).
    `fetched_at`: materialization timestamp string; current local time if `None`.

    Storage follows three cases (aspect ratio always preserved on downsample):

      1. Image within the megapixel cap (or cap disabled): the primary sidecar IS the verbatim original —
         stored byte-for-byte, so embedded metadata (EXIF, ICC, AI-generation parameters) is preserved. No
         second file; no `original_*` provenance fields.
      2. Image over the cap, `store_original_image=True` (default): the primary sidecar is a downsampled
         re-encode; the verbatim original is kept as a second sidecar, recorded in `original_sidecar`.
      3. Image over the cap, `store_original_image=False`: the primary is the downsampled re-encode; the
         original is discarded. `original_dimensions` / `original_size_bytes` are still recorded, but there
         is no `original_sidecar`.

    Returns an `env` with:
      `part`: the `image_url` content-part to append to the message content — its URL is `sidecar:<filename>`.
      `filename`: the primary sidecar's filename (the key under which `sidecar_metadata` should be recorded).
      `sidecar_metadata`: the provenance dict to store at `general_metadata["sidecars"][filename]`.
    """
    from PIL import Image  # deferred: Pillow is heavy and only needed on an actual attach

    raw = sidecarstore.read_source_bytes(image_source)
    original_size_bytes = len(raw)  # of the bytes the user actually attached, before any transcode below

    # QOI is transcoded to PNG on the way in, and everything downstream sees a PNG. Two independent reasons,
    # either of which alone would force it: Pillow has no QOI codec (so the probe and decode below would fail),
    # and no VLM accepts `image/qoi` on the wire (so a verbatim-stored QOI would reach the model unreadable).
    # Nothing is lost by transcoding: both formats are lossless, and QOI carries no EXIF/ICC payload to strip —
    # only a colorspace flag. The provenance still records what was attached.
    if raw[:4] == b"qoif":  # the QOI magic; see https://qoiformat.org/qoi-specification.pdf
        from ..common.image import codec  # deferred: pulls turbojpeg / Pillow
        content_type = content_type or "image/qoi"
        raw = codec.encode(codec.decode(raw), "PNG")

    # Probe format + dimensions without decoding pixels (PIL is lazy; `.format` / `.size` need no full load).
    with Image.open(io.BytesIO(raw)) as probe:
        pil_format = (probe.format or "PNG").upper()
        width, height = probe.size  # PIL reports (width, height)
        has_alpha = probe.mode in ("RGBA", "LA", "PA") or ("transparency" in probe.info)

    content_type = content_type or _mime_for_pil_format(pil_format)

    max_megapixels = librarian_config.image_store_max_megapixels
    megapixels = (height * width) / _ONE_MEGAPIXEL
    needs_downsample = (max_megapixels is not None) and (megapixels > max_megapixels)

    if not needs_downsample:
        # Case 1: store the verbatim original as the primary — preserves embedded metadata, no re-encode.
        metadata = sidecarstore.base_provenance(url=provenance_url, source=provenance_source,
                                                content_type=content_type, fetched_at=fetched_at)
        metadata["stored_dimensions"] = [height, width]  # dims of the bytes actually on disk (= original here)
        # Stored beside the file as well as in the payload: the payload copy dies with its node, and an orphan
        # with no description is a hash in a cleanup preview.
        filename = datastore.store_sidecar(raw, pil_format.lower(), metadata=metadata)
        return env(part=chatutil.image_content_part(f"{sidecarstore.SIDECAR_SCHEME}{filename}"),
                   filename=filename,
                   sidecar_metadata=metadata)

    # Cases 2/3: downsample to the cap, re-encode, store as primary.
    from ..common.image import codec  # deferred: pulls turbojpeg / Pillow
    from ..common.image import utils as image_utils  # deferred: pulls torch
    from ..common.image import lanczos  # deferred: pulls torch
    import numpy as np  # deferred with the image stack

    new_height, new_width = downsample_dims(height, width, max_megapixels)

    # Decode via PIL with an explicit mode convert, so palette / grayscale / CMYK inputs become clean RGB(A)
    # instead of tripping the (H, W, C) assumption in the tensor conversion.
    with Image.open(io.BytesIO(raw)) as source_image:
        arr = np.array(source_image.convert("RGBA" if has_alpha else "RGB"))  # np.array (not asarray): writable copy for torch.from_numpy
    tensor = image_utils.np_to_tensor(arr, device="cpu")  # (1, C, H, W) float32 on CPU; a rare one-shot resize
    tensor = lanczos.resize(tensor, new_height, new_width)
    downsampled = image_utils.tensor_to_np(tensor)  # (new_H, new_W, C) uint8

    # Pick an output format that can represent the channels: alpha needs a lossless-alpha container; otherwise
    # keep the original format when it round-trips cleanly, else fall back to PNG.
    if has_alpha:
        out_format = pil_format if pil_format in ("PNG", "WEBP", "TIFF") else "PNG"
    else:
        out_format = pil_format if pil_format in ("JPEG", "PNG", "WEBP", "BMP", "TIFF") else "PNG"

    metadata = sidecarstore.base_provenance(url=provenance_url, source=provenance_source,
                                            content_type=content_type, fetched_at=fetched_at)
    metadata["stored_dimensions"] = [new_height, new_width]  # dims of the downsampled bytes actually on disk (= what goes on the wire)
    metadata["original_dimensions"] = [height, width]
    metadata["original_size_bytes"] = original_size_bytes
    if librarian_config.store_original_image:
        # Keep the full-resolution original verbatim (metadata intact) as a second sidecar. It gets a
        # description of its own rather than sharing the primary's: it is a separate file in the directory, and
        # one that no content-part ever points at, so without one it is the single most inexplicable thing in
        # there — a large hash-named image referenced by nothing.
        original_metadata = sidecarstore.base_provenance(url=provenance_url, source=provenance_source,
                                                         content_type=content_type, fetched_at=fetched_at)
        original_metadata["stored_dimensions"] = [height, width]  # this file *is* the original, at full size
        original_metadata["role"] = "preserved_original"
        original_filename = datastore.store_sidecar(raw, pil_format.lower(), metadata=original_metadata)
        metadata["original_sidecar"] = original_filename

    # Stored last, so `metadata` is complete — `original_sidecar` included — by the time it is written beside
    # the file. `store_sidecar`'s metadata is first-write-wins, so a later call could not fill it in afterwards.
    primary_filename = datastore.store_sidecar(codec.encode(downsampled, out_format), out_format.lower(),
                                               metadata=metadata)

    return env(part=chatutil.image_content_part(f"{sidecarstore.SIDECAR_SCHEME}{primary_filename}"),
               filename=primary_filename,
               sidecar_metadata=metadata)


def sidecar_url_to_data_url(datastore: chattree.PersistentForest, url: str) -> str:
    """Resolve a stored `sidecar:<filename>` URL to a `data:<mime>;base64,...` URL for wire-send.

    Reads the sidecar bytes and base64-encodes them. The MIME type is derived from the sidecar filename's
    extension, which reflects the *stored* bytes' actual format (a downsampled primary may have been re-encoded
    to a different format than the original — this returns the format actually on disk). Used by
    `llmclient.invoke` to substitute a real image reference into the outgoing message; the persisted message
    keeps its `sidecar:` URL.
    """
    filename = sidecarstore.sidecar_filename_from_url(url, caller="sidecar_url_to_data_url")
    data = datastore.read_sidecar(filename)
    ext = filename.rsplit(".", 1)[-1] if "." in filename else "png"
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{_mime_for_ext(ext)};base64,{encoded}"


def sidecar_refs_in_payload(payload: dict) -> set[str]:
    """Return the sidecar filenames referenced by a single node `payload` — the GC mark-phase interpreter.

    Configure a `chattree.PersistentForest` with this (`sidecar_extractor=`); chattree drives the traversal
    over its own revisions and calls this per revision, so this function reads exactly one payload and never
    touches `chattree`'s node structure. Two reference sites:

      - `sidecar:` URLs in `image_url` content-parts (the images shown / sent).
      - `original_sidecar` entries in `general_metadata["sidecars"]` (the preserved full-resolution originals,
        which have no content-part of their own — case 2 of `store_image_as_sidecar`).

    Robust to a pre-migration payload whose `content` is still a bare string (returns no refs rather than
    iterating the string), though in practice GC only ever runs on post-migration data.
    """
    referenced = sidecarstore.content_part_sidecar_refs(payload, "image_url")
    sidecars = payload.get("general_metadata", {}).get("sidecars", {})
    for entry in sidecars.values():
        if isinstance(entry, dict) and "original_sidecar" in entry:
            referenced.add(entry["original_sidecar"])
    return referenced

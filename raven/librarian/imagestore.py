"""Image sidecar lifecycle for Librarian chat messages: store, resolve for the wire, and GC.

When a vision-capable model is loaded, the user can attach images to a message. An attached image is stored
as a *sidecar file* next to the chat datastore JSON (in `<datastore>.images/`, managed by
`chattree.PersistentForest`), and referenced from the message by a Raven-internal `sidecar:<filename>` URL. No
`https://` URL ever lands in a stored datastore, so a saved chat reloads without network access, survives the
source going away (link rot), and never phones home when reopened.

This module is the bridge between three lower layers — the image codec / Lanczos resampler
(`raven.common.image`), the sidecar file store (`chattree`), and the image-storage config knobs
(`raven.librarian.config`). It knows the chat-message content-part and provenance-metadata shapes, so the
storage layer beneath it doesn't have to.

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

__all__ = ["SIDECAR_SCHEME",
           "downsample_dims",
           "store_image_as_sidecar",
           "sidecar_url_to_data_url",
           "sidecar_refs_in_payload"]

import logging
logger = logging.getLogger(__name__)

import base64
import datetime
import io
import math
import pathlib

from unpythonic.env import env

from . import chatutil
from . import chattree
from . import config as librarian_config

# The Raven-internal URL scheme marking an image part as "resolve against the datastore's sidecar directory".
SIDECAR_SCHEME = "sidecar:"

# One megapixel, in pixels. The downsample target is expressed in megapixels (config `image_store_max_megapixels`).
_ONE_MEGAPIXEL = 2 ** 20


def _format_now() -> str:
    """Current local time as `"YYYY-MM-DD HH:MM:SS"` — the format used for `general_metadata["datetime"]`."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
    `provenance_source`: the categorical pathway — `"user_attachment"`, `"paste_url"`, or `"mcp:<server>"`.
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

    raw = image_source if isinstance(image_source, (bytes, bytearray)) else pathlib.Path(image_source).read_bytes()
    raw = bytes(raw)
    original_size_bytes = len(raw)

    # Probe format + dimensions without decoding pixels (PIL is lazy; `.format` / `.size` need no full load).
    with Image.open(io.BytesIO(raw)) as probe:
        pil_format = (probe.format or "PNG").upper()
        width, height = probe.size  # PIL reports (width, height)
        has_alpha = probe.mode in ("RGBA", "LA", "PA") or ("transparency" in probe.info)

    content_type = content_type or _mime_for_pil_format(pil_format)
    fetched_at = fetched_at or _format_now()

    max_megapixels = librarian_config.image_store_max_megapixels
    megapixels = (height * width) / _ONE_MEGAPIXEL
    needs_downsample = (max_megapixels is not None) and (megapixels > max_megapixels)

    if not needs_downsample:
        # Case 1: store the verbatim original as the primary — preserves embedded metadata, no re-encode.
        filename = datastore.store_sidecar(raw, pil_format.lower())
        metadata = {"url": provenance_url,
                    "fetched_at": fetched_at,
                    "content_type": content_type,
                    "source": provenance_source}
        return env(part=chatutil.image_content_part(f"{SIDECAR_SCHEME}{filename}"),
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

    primary_filename = datastore.store_sidecar(codec.encode(downsampled, out_format), out_format.lower())
    metadata = {"url": provenance_url,
                "fetched_at": fetched_at,
                "content_type": content_type,
                "source": provenance_source,
                "original_dimensions": [height, width],
                "original_size_bytes": original_size_bytes}
    if librarian_config.store_original_image:
        # Keep the full-resolution original verbatim (metadata intact) as a second sidecar.
        original_filename = datastore.store_sidecar(raw, pil_format.lower())
        metadata["original_sidecar"] = original_filename

    return env(part=chatutil.image_content_part(f"{SIDECAR_SCHEME}{primary_filename}"),
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
    if not url.startswith(SIDECAR_SCHEME):
        raise ValueError(f"sidecar_url_to_data_url: expected a '{SIDECAR_SCHEME}' URL, got '{url[:32]}'.")
    filename = url[len(SIDECAR_SCHEME):]
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
    referenced = set()
    message = payload.get("message", {})
    content = message.get("content")
    if isinstance(content, list):  # post-migration content is always a parts list; guard legacy strings
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                part_url = part.get("image_url", {}).get("url", "")
                if part_url.startswith(SIDECAR_SCHEME):
                    referenced.add(part_url[len(SIDECAR_SCHEME):])
    sidecars = payload.get("general_metadata", {}).get("sidecars", {})
    for entry in sidecars.values():
        if isinstance(entry, dict) and "original_sidecar" in entry:
            referenced.add(entry["original_sidecar"])
    return referenced

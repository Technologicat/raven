"""Encoding and decoding of still-image files.

Format-agnostic decode (PNG, JPEG, QOI, BMP, TIFF, WebP, …) with an optional
turbojpeg fast path for JPEG scaled decode. Accepts bytes, binary streams,
or filesystem paths interchangeably.

Parallel to `raven.common.audio.codec` for still images. As there, the
natural output shape is whatever the underlying decoder gives; callers who
need a specific channel layout (e.g. RGBA) apply a normalization step
separately via `raven.common.image.utils.ensure_rgba`.
"""

__all__ = ["IMAGE_EXTENSIONS", "encode", "decode"]

import io
import logging
import pathlib
from typing import BinaryIO, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# File extensions for formats supported by this module's `decode`.
# Used by file-browser / directory-scan code to filter image candidates.
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".qoi",
                              ".bmp", ".tiff", ".tif", ".webp"})

# --------------------------------------------------------------------------------
# Optional fast JPEG decoder (turbojpeg)

_HAS_TURBOJPEG = False
_turbojpeg_instance = None

try:
    from turbojpeg import TurboJPEG, TJPF_RGB
    _turbojpeg_instance = TurboJPEG()
    _HAS_TURBOJPEG = True
    logger.info("codec: turbojpeg available — fast JPEG decode enabled")
except ImportError:
    logger.info("codec: turbojpeg not available — using Pillow for JPEG decode")


# --------------------------------------------------------------------------------
# Internal helpers

def _read_to_bytes(source: Union[bytes, BinaryIO, pathlib.Path, str]) -> bytes:
    """Coerce `source` to `bytes` regardless of which input flavor the caller supplied."""
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if isinstance(source, (pathlib.Path, str)):
        return pathlib.Path(source).read_bytes()
    # Assume binary file-like (BinaryIO-compatible).
    return source.read()

def _sniff_format(buf: bytes) -> Optional[str]:
    """Return `"jpeg"`, `"qoi"`, or `None` (PIL fallback), based on file magic.

    Only formats with a fast path need to be distinguished here; everything
    else falls through to Pillow, which handles its own format detection.
    """
    if buf.startswith(b"\xff\xd8\xff"):  # SOI marker + next byte (any of E0..EF, DB, E0)
        return "jpeg"
    if buf.startswith(b"qoif"):
        return "qoi"
    return None

def _turbojpeg_decode(jpeg_bytes: bytes, max_size: Optional[int]) -> np.ndarray:
    """Decode JPEG via turbojpeg, optionally at a reduced scale factor.

    `max_size`: if given, triggers scaled decode: the image is decoded at the
                smallest JPEG scale factor (1/1, 1/2, 1/4, 1/8) whose output
                is still ≥ `max_size` on both dimensions. Much faster than
                full decode + external resize for large JPEGs (thumbnails).

    Returns an RGB uint8 array of shape `(h, w, 3)`. Callers that need RGBA
    should apply `raven.common.image.utils.ensure_rgba`.
    """
    assert _turbojpeg_instance is not None  # guarded by `_HAS_TURBOJPEG` at call site
    if max_size is not None:
        # Query dimensions without decoding.
        width, height, _, _ = _turbojpeg_instance.decode_header(jpeg_bytes)
        # Pick the largest scale that still yields >= max_size on both edges.
        best_factor = (1, 1)
        for num, den in [(1, 1), (1, 2), (1, 4), (1, 8)]:
            scaled_w = (width * num + den - 1) // den
            scaled_h = (height * num + den - 1) // den
            if scaled_w >= max_size and scaled_h >= max_size:
                best_factor = (num, den)
        return _turbojpeg_instance.decode(jpeg_bytes,
                                          pixel_format=TJPF_RGB,
                                          scaling_factor=best_factor)
    return _turbojpeg_instance.decode(jpeg_bytes, pixel_format=TJPF_RGB)


# --------------------------------------------------------------------------------
# Public API

def decode(source: Union[bytes, BinaryIO, pathlib.Path, str],
           max_size: Optional[int] = None) -> np.ndarray:
    """Decode an image from bytes, a binary stream, or a filesystem path.

    `source`: one of:
        - `bytes` / `bytearray`: raw image file contents.
        - binary file-like (`BinaryIO`): e.g. `open(path, "rb")`, `io.BytesIO`,
          `flask.request.files[...].stream`. The entire stream is read into
          memory (needed for format sniffing and for rewindability).
        - `pathlib.Path` / `str`: filesystem path.

    `max_size`: if given, hint that the result will be downscaled to at most
                this many pixels on each edge. For JPEG with turbojpeg
                available, triggers hardware-accelerated scaled decode; for
                other formats, the hint is ignored.

    Returns an `np.ndarray` of dtype uint8, shape `(h, w, c)` where `c` is
    whatever the underlying decoder naturally produces:

        - JPEG: 3 channels (RGB). No alpha, whether via turbojpeg or Pillow.
        - QOI: 3 or 4 channels depending on the source file.
        - PNG / BMP / TIFF / WebP / others: whatever Pillow returns for that file.

    Callers that need a guaranteed 4-channel RGBA output should pass the
    result through `raven.common.image.utils.ensure_rgba`.
    """
    buf = _read_to_bytes(source)

    fmt = _sniff_format(buf)

    if fmt == "jpeg" and _HAS_TURBOJPEG:
        return _turbojpeg_decode(buf, max_size=max_size)

    if fmt == "qoi":
        import qoi as _qoi  # deferred; qoi is a small C extension
        return _qoi.decode(buf)

    # Pillow handles everything else: PNG, BMP, TIFF, WebP, and JPEG when
    # turbojpeg is not installed. `max_size` is not honored here — Pillow has
    # no native scaled-decode equivalent.
    from PIL import Image  # deferred to avoid import cost when not needed
    return np.asarray(Image.open(io.BytesIO(buf)))

def encode(image: np.ndarray, format: str) -> bytes:
    """Encode an image to `bytes` in the given file format.

    `image`: `np.ndarray` of shape `(h, w, 3)` or `(h, w, 4)`, dtype uint8.
             Three channels are treated as RGB; four as RGBA.

    `format`: output format name, case-insensitive. Supported:
              `"qoi"` (fast lossless C encoder), plus any format Pillow
              accepts (`"png"`, `"jpeg"`, `"bmp"`, `"tga"`, `"tiff"`,
              `"webp"`, …). PNG uses `compress_level=1` (fast), TGA uses
              `tga_rle` compression.

    Returns the encoded file contents as a `bytes` object.
    """
    fmt_upper = format.upper()

    if fmt_upper == "QOI":
        import qoi as _qoi  # deferred
        return _qoi.encode(image.copy(order="C"))

    from PIL import Image  # deferred
    pil_image = Image.fromarray(np.uint8(image[:, :, :3]))
    if image.shape[2] == 4:
        pil_image.putalpha(Image.fromarray(np.uint8(image[:, :, 3])))

    buffer = io.BytesIO()
    if fmt_upper == "PNG":
        kwargs = {"compress_level": 1}
    elif fmt_upper == "TGA":
        kwargs = {"compression": "tga_rle"}
    else:
        kwargs = {}
    pil_image.save(buffer, format=fmt_upper, **kwargs)
    return buffer.getvalue()

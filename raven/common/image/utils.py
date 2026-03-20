"""Image I/O and tensor conversion utilities.

Canonical conversions between numpy (HWC uint8), PyTorch (BCHW float32),
and DPG dynamic texture (flat float32 RGBA) formats.  Also provides
format-agnostic image decoding (PNG, JPEG, QOI, etc.) with optional
turbojpeg fast path for JPEG.
"""

__all__ = ["decode_image",
           "np_to_tensor", "tensor_to_np", "tensor_to_dpg_flat",
           "letterbox"]

import logging
import pathlib
from typing import Optional, Union

import numpy as np
import torch

from . import lanczos

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported image extensions
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".qoi",
                              ".bmp", ".tiff", ".tif", ".webp"})

# ---------------------------------------------------------------------------
# Optional fast JPEG decoder
# ---------------------------------------------------------------------------

_HAS_TURBOJPEG = False
_turbojpeg_instance = None

try:
    from turbojpeg import TurboJPEG, TJPF_RGBX  # noqa: F401
    _turbojpeg_instance = TurboJPEG()
    _HAS_TURBOJPEG = True
    logger.info("imageutil: turbojpeg available — fast JPEG decode enabled")
except ImportError:
    logger.info("imageutil: turbojpeg not available — using PIL for JPEG decode")


# ---------------------------------------------------------------------------
# Image decoding
# ---------------------------------------------------------------------------

def _turbojpeg_scaled_decode(path: pathlib.Path,
                             max_size: Optional[int]) -> Optional[np.ndarray]:
    """Attempt JPEG decode via turbojpeg with optional downscaling.

    *max_size*, if given, triggers scaled decode: the image is decoded at the
    smallest JPEG scale factor (1/1, 1/2, 1/4, 1/8) whose output is still
    ≥ *max_size* on both dimensions.  This is much faster than full decode +
    external resize for large JPEGs.

    Returns an RGBA uint8 numpy array, or ``None`` if turbojpeg is unavailable
    or the file isn't JPEG.
    """
    if not _HAS_TURBOJPEG or _turbojpeg_instance is None:
        return None
    if path.suffix.lower() not in (".jpg", ".jpeg"):
        return None

    jpeg_data = path.read_bytes()

    if max_size is not None:
        # Query dimensions without decoding.
        width, height, _, _ = _turbojpeg_instance.decode_header(jpeg_data)

        # Pick the largest scale that still yields >= max_size on both edges.
        best_factor = (1, 1)
        for num, den in [(1, 1), (1, 2), (1, 4), (1, 8)]:
            scaled_w = (width * num + den - 1) // den
            scaled_h = (height * num + den - 1) // den
            if scaled_w >= max_size and scaled_h >= max_size:
                best_factor = (num, den)

        rgb = _turbojpeg_instance.decode(jpeg_data,
                                         pixel_format=TJPF_RGBX,
                                         scaling_factor=best_factor)
    else:
        rgb = _turbojpeg_instance.decode(jpeg_data, pixel_format=TJPF_RGBX)

    # TJPF_RGBX gives (H, W, 4) with X = padding byte (not alpha).
    # Set alpha to 255.
    rgb[:, :, 3] = 255
    return rgb


def decode_image(path: Union[pathlib.Path, str],
                 max_size: Optional[int] = None) -> np.ndarray:
    """Decode an image file to an RGBA uint8 numpy array.

    *max_size*: if given, hint that the result will be downscaled to at most
    this many pixels on each edge.  For JPEG with turbojpeg available, this
    triggers hardware-accelerated scaled decode (much faster for large images).
    For other formats, the hint is ignored and the full image is returned.

    Supported formats: PNG, JPEG, QOI, and anything else PIL handles.
    """
    from PIL import Image  # deferred to avoid import cost when not needed

    path = pathlib.Path(path)
    suffix = path.suffix.lower()

    # Try turbojpeg first (fast path for JPEG).
    result = _turbojpeg_scaled_decode(path, max_size)
    if result is not None:
        return result

    # QOI — fast C decoder.
    if suffix == ".qoi":
        import qoi as _qoi
        return _qoi.decode(path.read_bytes())

    # PIL fallback — handles everything else.
    return np.array(Image.open(path).convert("RGBA"))


# ---------------------------------------------------------------------------
# Tensor conversions
# ---------------------------------------------------------------------------

def np_to_tensor(arr: np.ndarray,
                 device: Union[torch.device, str],
                 dtype: torch.dtype = torch.float32,
                 batch: bool = True) -> torch.Tensor:
    """Convert an ``(H, W, C)`` uint8 numpy image to a float tensor.

    Returns ``(1, C, H, W)`` when *batch* is True (default), ``(C, H, W)``
    when False.  Combines dtype conversion and device transfer in one
    ``.to()`` call to minimize intermediate copies.
    """
    t = torch.from_numpy(arr).permute(2, 0, 1)
    if batch:
        t = t.unsqueeze(0)
    return t.to(dtype=dtype, device=device) / 255.0


def tensor_to_np(tensor: torch.Tensor) -> np.ndarray:
    """Convert a float tensor to an ``(H, W, C)`` uint8 numpy image.

    Accepts both ``(1, C, H, W)`` (batched) and ``(C, H, W)`` (unbatched)
    input — auto-detected from ``tensor.ndim``.
    Clamps to [0, 1] before conversion (handles Lanczos ringing).
    """
    if tensor.ndim == 4:
        tensor = tensor[0]
    return (tensor
            .clamp(0.0, 1.0)
            .permute(1, 2, 0)
            .cpu()
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .numpy())


def tensor_to_dpg_flat(tensor: torch.Tensor) -> np.ndarray:
    """Convert a float tensor to a flat float32 array for DPG.

    Accepts both ``(1, C, H, W)`` (batched) and ``(C, H, W)`` (unbatched)
    input — auto-detected from ``tensor.ndim``.
    DPG dynamic textures expect a flat array of ``width × height × channels``
    floats in [0, 1].  Clamps to handle Lanczos ringing.
    """
    if tensor.ndim == 4:
        tensor = tensor[0]
    return (tensor
            .clamp(0.0, 1.0)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
            .ravel())


# ---------------------------------------------------------------------------
# Letterbox
# ---------------------------------------------------------------------------

def letterbox(tensor: torch.Tensor,
              tile_size: int,
              order: int = lanczos.DEFAULT_ORDER,
              bg_value: float = 0.3) -> torch.Tensor:
    """Resize *tensor* to fit within ``tile_size × tile_size``, letterbox the rest.

    Preserves aspect ratio.  Non-image area is filled with *bg_value* (0.3 =
    dark gray, looks reasonable in both light and dark mode).

    *tensor*: ``(1, C, H, W)`` float32 on any device.
    Returns:  ``(1, C, tile_size, tile_size)`` float32 on the same device.
    """
    _, C, H, W = tensor.shape
    scale = min(tile_size / H, tile_size / W)
    new_h = max(1, round(H * scale))
    new_w = max(1, round(W * scale))

    resized = lanczos.resize(tensor, new_h, new_w, order=order)

    result = torch.full((1, C, tile_size, tile_size), bg_value,
                        device=tensor.device, dtype=tensor.dtype)
    y_off = (tile_size - new_h) // 2
    x_off = (tile_size - new_w) // 2
    result[:, :, y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return result

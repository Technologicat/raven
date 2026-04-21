"""Image I/O and tensor conversion utilities.

Canonical conversions between numpy (HWC uint8), PyTorch (BCHW float32),
and DPG dynamic texture (flat float32 RGBA) formats. Also provides a small
RGBA-normalization helper for pipelines that need a guaranteed 4-channel
output (e.g. DPG dynamic textures, the server-side avatar postprocessor).

For image decoding / encoding, see `raven.common.image.codec`.
"""

__all__ = ["ensure_rgba",
           "np_to_tensor", "tensor_to_np", "tensor_to_dpg_flat",
           "letterbox"]

import logging
from typing import Union

import numpy as np
import torch

from . import lanczos

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Channel normalization
# ---------------------------------------------------------------------------

def ensure_rgba(image: np.ndarray) -> np.ndarray:
    """Return `image` with a guaranteed 4-channel RGBA layout.

    If `image` already has 4 channels, returned as-is (no copy). If it has
    3 channels (RGB), an opaque alpha channel (value 255 for uint8, 1.0 for
    floats) is appended.

    Useful at the boundary between `raven.common.image.codec.decode` — which
    returns the natural channel count produced by the underlying decoder
    (e.g. 3 for JPEG) — and pipelines that require RGBA (DPG dynamic
    textures, the server-side avatar postprocessor, the cherrypick mip
    pipeline).

    Raises `ValueError` if `image` doesn't look like an image (wrong rank
    or channel count).
    """
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"ensure_rgba: expected shape (h, w, 3|4), got {image.shape}")
    if image.shape[2] == 4:
        return image
    # Append an opaque alpha channel in the input's own dtype range.
    if np.issubdtype(image.dtype, np.integer):
        alpha_value = np.iinfo(image.dtype).max
    else:  # floating
        alpha_value = 1.0
    alpha = np.full(image.shape[:2] + (1,), alpha_value, dtype=image.dtype)
    return np.concatenate([image, alpha], axis=2)


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

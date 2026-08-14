"""Image I/O and tensor conversion utilities.

Canonical conversions between numpy (HWC uint8), PyTorch (BCHW float32),
and DPG dynamic texture (flat float32 RGBA) formats. Also provides a small
RGBA-normalization helper for pipelines that need a guaranteed 4-channel
output (e.g. DPG dynamic textures, the server-side avatar postprocessor).

For image decoding / encoding, see `raven.common.image.codec`.
"""

__all__ = ["ensure_rgba",
           "np_to_tensor", "tensor_to_np", "tensor_to_dpg_flat",

           "fit_contain", "fit_cover", "letterbox"]

import logging
from collections.abc import Sequence
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
        # This makes all code paths always return a writable array.
        return image if image.flags.writeable else image.copy()
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
    if arr.dtype != np.uint8:
        raise ValueError(f"np_to_tensor: expected uint8 array, got an array with dtype {arr.dtype}")
    t = torch.from_numpy(arr).permute(2, 0, 1)
    if batch:
        t = t.unsqueeze(0)
    # Normalize in float32 — universally supported across CPU/CUDA/MPS, and
    # avoids float16-on-CPU op gaps and float64-on-MPS (MPS has no float64).
    # Cast to the requested dtype last, after all arithmetic, so nothing runs
    # in a dtype the target device can't execute.
    t = t.to(device=device, dtype=torch.float32)
    return (t / 255.0).to(dtype)


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
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .cpu()           # transfer uint8, not float32 — 4x less PCIe/unified traffic
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
# Fitting an image to a target box
# ---------------------------------------------------------------------------
#
# The two ways to resize an image into a box of a different aspect ratio, both preserving aspect ratio; the
# same pair CSS calls ``object-fit: contain`` and ``object-fit: cover``.
# https://developer.mozilla.org/en-US/docs/Web/CSS/object-fit
#
#   - `fit_contain` scales until the image fits *inside* the box. All of the image survives; the box is not
#     filled, so the result is smaller than the box in one dimension. Right when the content is what matters
#     and the frame can give — a thumbnail, an inline illustration.
#   - `fit_cover` scales until the image *covers* the box, then crops the overflow. The box is filled exactly;
#     the edges of the image are lost. Right when the frame is what matters — a backdrop, a fixed-size tile.
#
# `letterbox` (below) is the third member in practice: `fit_contain` into a square, with the unfilled part
# padded rather than left to the caller. Kept separate because the padding value is a rendering decision.

def fit_contain(tensor: torch.Tensor,
                max_h: int,
                max_w: int,
                allow_upscale: bool = False,
                order: int = lanczos.DEFAULT_ORDER) -> torch.Tensor:
    """Scale *tensor* to fit inside ``max_h × max_w``, preserving aspect ratio. Nothing is cropped.

    *tensor*: ``(1, C, H, W)`` float32 on any device.
    *allow_upscale*: whether an image already smaller than the box may be enlarged to fill it. Default `False`,
                     which is what a viewer expects of a thumbnail — a small image shown at its native size,
                     not blown up and blurry.
    Returns:  ``(1, C, h, w)`` float32 on the same device, with ``h <= max_h`` and ``w <= max_w``. At least one
              of the two is met exactly, unless the no-upscale cap bound instead.
    """
    _, _, H, W = tensor.shape
    scale = min(max_h / H, max_w / W)
    if not allow_upscale:
        scale = min(scale, 1.0)
    return lanczos.resize(tensor, max(1, round(H * scale)), max(1, round(W * scale)), order=order)


def fit_cover(tensor: torch.Tensor,
              out_h: int,
              out_w: int,
              order: int = lanczos.DEFAULT_ORDER) -> torch.Tensor:
    """Scale *tensor* to cover ``out_h × out_w``, preserving aspect ratio, then crop the overflow.

    The output is exactly the requested size, with no empty area — at the cost of the parts of the image that
    fell outside the box. Cropping is anchored at the top left rather than centered, matching what the avatar
    backdrop does; a backdrop's interesting content is rarely dead center, and top-left is at least predictable.

    *tensor*: ``(1, C, H, W)`` float32 on any device.
    Returns:  ``(1, C, out_h, out_w)`` float32 on the same device.
    """
    _, _, H, W = tensor.shape
    scale = max(out_h / H, out_w / W)  # max, not min: overshoot in both dimensions, then cut back
    resized = lanczos.resize(tensor, max(out_h, round(H * scale)), max(out_w, round(W * scale)), order=order)
    return resized[:, :, :out_h, :out_w]


# ---------------------------------------------------------------------------
# Letterbox
# ---------------------------------------------------------------------------

def letterbox(tensor: torch.Tensor,
              tile_size: int,
              order: int = lanczos.DEFAULT_ORDER,
              bg_value: Union[float, Sequence[float]] = 0.3,
              allow_upscale: bool = True) -> torch.Tensor:
    """Resize *tensor* to fit within ``tile_size × tile_size``, letterbox the rest.

    `fit_contain` into a square, with the leftover area filled in rather than left to the caller — which is
    what makes a grid of these line up into a regular lattice however the individual images are shaped.

    Preserves aspect ratio.  Non-image area is filled with *bg_value* (0.3 =
    dark gray, looks reasonable in both light and dark mode).

    *bg_value* may instead be one value per channel, which is how an RGBA image gets *transparent* padding:
    a single 0.3 sets the alpha channel to 0.3 as well, so the bars come out as a translucent gray wash
    rather than as nothing. ``(0, 0, 0, 0)`` is what a tile drawn over the panel's own background wants.

    *tensor*: ``(1, C, H, W)`` float32 on any device.
    *allow_upscale*: as in `fit_contain`, but defaulting the other way: a tile that is mostly padding reads as
                     an error rather than as a small picture, so filling the tile is usually what a caller
                     wants here.  Pass `False` when fidelity matters more than a uniform apparent size — the
                     tile then keeps its shape, with a small image centered in it at 1:1.
    Returns:  ``(1, C, tile_size, tile_size)`` float32 on the same device.
    """
    _, C, _, _ = tensor.shape
    resized = fit_contain(tensor, tile_size, tile_size, allow_upscale=allow_upscale, order=order)
    new_h, new_w = int(resized.shape[2]), int(resized.shape[3])

    if isinstance(bg_value, (int, float)):
        result = torch.full((1, C, tile_size, tile_size), float(bg_value),
                            device=tensor.device, dtype=tensor.dtype)
    else:
        if len(bg_value) != C:
            raise ValueError(f"letterbox: expected one background value per channel ({C}), got {len(bg_value)}")
        result = (torch.as_tensor(bg_value, device=tensor.device, dtype=tensor.dtype)
                  .reshape(1, C, 1, 1)
                  .expand(1, C, tile_size, tile_size)
                  .clone())
    y_off = (tile_size - new_h) // 2
    x_off = (tile_size - new_w) // 2
    result[:, :, y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return result

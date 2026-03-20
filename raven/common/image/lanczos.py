"""GPU-accelerated Lanczos image resize using PyTorch.

Separable implementation: horizontal and vertical passes applied independently.
Multi-stage downsampling for large reduction ratios (>2×): repeatedly halve
with a Lanczos strided convolution, then a final general Lanczos pass for the
remaining ratio. This keeps the kernel compact while maintaining full quality.

The kernel order *a* (default 3) is configurable.  Higher orders give better
stopband attenuation (less aliasing) at the cost of slightly more computation
and ringing on sharp edges:

  - Lanczos-3: 12 taps per halving step.  Industry standard.
  - Lanczos-4: 16 taps.  Better stopband.
  - Lanczos-5: 20 taps.  Very good stopband.

All operations use standard PyTorch tensor ops.
"""

__all__ = ["resize", "mipchain"]

import math

import torch
import torch.nn.functional as F

# Cache for the 2× downsample kernel, keyed by (device, dtype, order).
_halving_kernel_cache: dict[tuple, torch.Tensor] = {}

DEFAULT_ORDER = 4  # for the public API


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

def _lanczos_kernel(x: torch.Tensor, a: int) -> torch.Tensor:
    """Lanczos-*a* window: ``sinc(x) · sinc(x/a)`` for ``|x| < a``, else 0.

    Uses ``torch.sinc`` (normalized sinc: ``sin(πx)/(πx)``).
    """
    out = torch.sinc(x) * torch.sinc(x / float(a))
    out = out * (x.abs() < a)
    return out


# ---------------------------------------------------------------------------
# 2× downsample — fast path via strided convolution
# ---------------------------------------------------------------------------

def _make_halving_kernel(device: torch.device, dtype: torch.dtype,
                         order: int) -> torch.Tensor:
    """Compute the 1-D Lanczos-*order* kernel for exact 2× downsampling.

    For ``scale = 0.5``, ``filterscale = 2.0``, ``support = 2 * order``.

    Output pixel *j* is centered at input coordinate ``2j + 0.5``.
    Contributing input pixels span ``4 * order`` taps.

    In a stride-2 convolution with this kernel, padding is **asymmetric**:
    ``pad_left = 2 * order - 1``, ``pad_right = 2 * order``.

    Returns a 1-D tensor of length ``4 * order``, normalized to sum to 1.
    """
    filterscale = 2.0
    K = 4 * order  # number of taps
    pad_left = 2 * order - 1

    # For kernel tap n (n = 0…K-1), the sampled input pixel is at position
    # 2j + n − pad_left.  Distance from center (2j + 0.5):
    #   d = n − pad_left − 0.5
    # Normalized Lanczos argument: d / filterscale.
    n = torch.arange(K, device=device, dtype=dtype)
    x = (n - pad_left - 0.5) / filterscale

    weights = _lanczos_kernel(x, order)
    weights /= weights.sum()
    return weights


def _get_halving_kernel(device: torch.device, dtype: torch.dtype,
                        order: int) -> torch.Tensor:
    """Get (or create and cache) the 2× downsample kernel."""
    key = (device, dtype, order)
    if key not in _halving_kernel_cache:
        _halving_kernel_cache[key] = _make_halving_kernel(device, dtype, order)
    return _halving_kernel_cache[key]


def _halve(tensor: torch.Tensor, dim: int, order: int) -> torch.Tensor:
    """Halve ``tensor`` along *dim* (2 = height, 3 = width) using Lanczos.

    Implemented as a strided depthwise convolution — constant-time regardless
    of channel count, and very VRAM-friendly.
    """
    kernel_1d = _get_halving_kernel(tensor.device, tensor.dtype, order)
    C = tensor.shape[1]
    in_size = tensor.shape[dim]
    out_size = in_size // 2

    K = kernel_1d.shape[0]
    pad_left = 2 * order - 1
    pad_right = K - 1 - pad_left  # = 2 * order

    if dim == 2:  # height
        tensor = F.pad(tensor, (0, 0, pad_left, pad_right), mode="replicate")
        kernel = kernel_1d.reshape(1, 1, K, 1).expand(C, -1, -1, -1)
        result = F.conv2d(tensor, kernel, stride=(2, 1), groups=C)
        return result[:, :, :out_size, :]
    else:  # width
        tensor = F.pad(tensor, (pad_left, pad_right, 0, 0), mode="replicate")
        kernel = kernel_1d.reshape(1, 1, 1, K).expand(C, -1, -1, -1)
        result = F.conv2d(tensor, kernel, stride=(1, 2), groups=C)
        return result[:, :, :, :out_size]


# ---------------------------------------------------------------------------
# General resize — any ratio, gather-based
# ---------------------------------------------------------------------------

def _resize_1d(tensor: torch.Tensor, out_size: int, dim: int,
               order: int) -> torch.Tensor:
    """Resize ``tensor`` along *dim* using Lanczos interpolation.

    Handles both upscaling and downscaling at any ratio.  Uses advanced
    indexing to gather contributing input pixels — fine for moderate sizes
    (call after multi-stage halving, so the tensor is at most 2× the target).
    """
    in_size = tensor.shape[dim]
    if in_size == out_size:
        return tensor

    device = tensor.device
    dtype = tensor.dtype

    scale = out_size / in_size
    filterscale = max(1.0, 1.0 / scale)
    support = math.ceil(order * filterscale)

    # Output pixel centers in input coordinates.
    out_pos = torch.arange(out_size, device=device, dtype=dtype)
    centers = (out_pos + 0.5) / scale - 0.5  # (out_size,)

    # Integer index of the leftmost contributing input pixel for each output pixel.
    left = (centers - support).floor().long()  # (out_size,)

    # All contributing integer indices: (out_size, K).
    K = 2 * support + 2  # generous; includes potential zero-weight boundary taps
    tap_offsets = torch.arange(K, device=device).long()
    idx = left.unsqueeze(1) + tap_offsets.unsqueeze(0)  # (out_size, K)

    # Exact distances from center, normalized for Lanczos.
    distances = idx.to(dtype) - centers.unsqueeze(1)  # (out_size, K)
    x = distances / filterscale
    weights = _lanczos_kernel(x, order)
    weights /= weights.sum(dim=1, keepdim=True).clamp(min=1e-8)

    # Clamp indices for boundary handling (replicate-edge).
    idx = idx.clamp(0, in_size - 1)

    # Gather and weighted sum.
    if dim == 2:  # height
        gathered = tensor[:, :, idx, :]          # (B, C, out_H, K, W)
        w = weights[None, None, :, :, None]      # (1, 1, out_H, K, 1)
        return (gathered * w).sum(dim=3)
    else:  # width
        gathered = tensor[:, :, :, idx]          # (B, C, H, out_W, K)
        w = weights[None, None, None, :, :]      # (1, 1, 1, out_W, K)
        return (gathered * w).sum(dim=4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resize(tensor: torch.Tensor,
           target_h: int,
           target_w: int,
           order: int = DEFAULT_ORDER) -> torch.Tensor:
    """Resize a ``(B, C, H, W)`` tensor using Lanczos interpolation.

    *order* controls the kernel size (default 3).  Higher values give sharper
    frequency cutoff at the cost of more computation and ringing:

      - 3: industry standard, 12 taps per halving step.
      - 4: better stopband attenuation, 16 taps.
      - 5: very good stopband, 20 taps.

    For downscaling ratios larger than 2×, automatically applies multi-stage
    halving (strided Lanczos convolution) before the final general resize.

    Works for both upscaling and downscaling.  Input and output share the
    same device and dtype.
    """
    _B, _C, H, W = tensor.shape

    # Multi-stage: interleave height/width halving to keep intermediates small.
    while H > target_h * 2 or W > target_w * 2:
        if H > target_h * 2:
            tensor = _halve(tensor, dim=2, order=order)
            H = tensor.shape[2]
        if W > target_w * 2:
            tensor = _halve(tensor, dim=3, order=order)
            W = tensor.shape[3]

    # Final resize for the remaining (≤ 2×) ratio.
    if H != target_h:
        tensor = _resize_1d(tensor, target_h, dim=2, order=order)
    if W != target_w:
        tensor = _resize_1d(tensor, target_w, dim=3, order=order)

    return tensor


def mipchain(tensor: torch.Tensor,
             min_size: int = 64,
             order: int = DEFAULT_ORDER) -> list[torch.Tensor]:
    """Generate a chain of Lanczos-downsampled mip levels.

    ``tensor`` is ``(B, C, H, W)``.  Returns ``[original, ½, ¼, …]``,
    halving both dimensions at each step, down to *min_size* on the short
    edge.

    Each halving step uses the fast strided-convolution path, so the whole
    chain is generated very quickly.
    """
    levels = [tensor]
    while True:
        _B, _C, H, W = levels[-1].shape
        if min(H, W) < min_size * 2:
            break
        level = _halve(_halve(levels[-1], dim=2, order=order), dim=3, order=order)
        levels.append(level)
    return levels

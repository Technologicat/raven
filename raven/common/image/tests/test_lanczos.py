"""Tests for raven.common.image.lanczos — GPU-accelerated Lanczos resize."""

import pytest
import numpy as np
import torch
from PIL import Image

from raven.common.image.lanczos import lanczos_resize, lanczos_mipchain


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pil_lanczos_resize(np_image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Reference Lanczos-3 resize via PIL.

    ``np_image`` is (H, W, C) uint8.  Returns (target_h, target_w, C) uint8.
    """
    pil = Image.fromarray(np_image)
    pil = pil.resize((target_w, target_h), Image.LANCZOS)
    return np.array(pil)


def _np_to_tensor(np_image: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """Convert (H, W, C) uint8 numpy image to (1, C, H, W) float32 tensor."""
    t = torch.from_numpy(np_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return t.to(device)


def _tensor_to_np(tensor: torch.Tensor) -> np.ndarray:
    """Convert (1, C, H, W) float32 tensor to (H, W, C) uint8 numpy image."""
    return (tensor[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


def _make_checkerboard(h: int, w: int, block: int = 8, channels: int = 3) -> np.ndarray:
    """Generate a checkerboard pattern (H, W, C) uint8."""
    y = np.arange(h) // block
    x = np.arange(w) // block
    grid = (y[:, None] + x[None, :]) % 2
    if channels == 1:
        return (grid * 255).astype(np.uint8)[:, :, None]
    return np.stack([grid * 255] * channels, axis=-1).astype(np.uint8)


def _make_gradient(h: int, w: int, channels: int = 3) -> np.ndarray:
    """Generate a horizontal gradient (H, W, C) uint8."""
    grad = np.linspace(0, 255, w, dtype=np.uint8)
    img = np.tile(grad[None, :], (h, 1))
    return np.stack([img] * channels, axis=-1)


# ---------------------------------------------------------------------------
# Devices to test
# ---------------------------------------------------------------------------

_devices = ["cpu"]
if torch.cuda.is_available():
    _devices.append("cuda")
if torch.backends.mps.is_available():
    _devices.append("mps")


# ---------------------------------------------------------------------------
# Tests: kernel correctness
# ---------------------------------------------------------------------------

class TestLanczosResize:
    """Correctness tests for lanczos_resize."""

    @pytest.mark.parametrize("device", _devices)
    def test_identity_resize(self, device):
        """Resizing to the same size should be (approximately) the identity."""
        img = _make_checkerboard(64, 96)
        t = _np_to_tensor(img, device)
        result = lanczos_resize(t, 64, 96)
        assert result.shape == (1, 3, 64, 96)
        # Should be very close to input (within FP rounding).
        diff = (result - t).abs().max().item()
        assert diff < 1e-5, f"Identity resize diff {diff} too large"

    @pytest.mark.parametrize("device", _devices)
    def test_downscale_vs_pil(self, device):
        """Moderate downscale should closely match PIL Lanczos."""
        img = _make_gradient(200, 300)
        target_h, target_w = 100, 150

        t = _np_to_tensor(img, device)
        ours = _tensor_to_np(lanczos_resize(t, target_h, target_w))
        ref = _pil_lanczos_resize(img, target_h, target_w)

        # Allow some tolerance — boundary handling and exact kernel differ slightly.
        mae = np.abs(ours.astype(float) - ref.astype(float)).mean()
        assert mae < 3.0, f"Mean absolute error vs PIL: {mae:.2f}"

    @pytest.mark.parametrize("device", _devices)
    def test_large_downscale_vs_pil(self, device):
        """Large downscale (multi-stage path) should still match PIL reasonably.

        Uses a gradient (smooth, representative of real images) rather than a
        checkerboard (adversarial for multi-stage vs single-pass comparison).
        """
        img = _make_gradient(512, 768)
        target_h, target_w = 64, 96

        t = _np_to_tensor(img, device)
        ours = _tensor_to_np(lanczos_resize(t, target_h, target_w))
        ref = _pil_lanczos_resize(img, target_h, target_w)

        mae = np.abs(ours.astype(float) - ref.astype(float)).mean()
        # Multi-stage introduces slightly more error than single-pass;
        # boundary handling also differs from PIL.
        assert mae < 4.0, f"Mean absolute error vs PIL (large downscale): {mae:.2f}"

    @pytest.mark.parametrize("device", _devices)
    def test_upscale_vs_pil(self, device):
        """Upscale should closely match PIL Lanczos."""
        img = _make_gradient(64, 64)
        target_h, target_w = 128, 192

        t = _np_to_tensor(img, device)
        ours = _tensor_to_np(lanczos_resize(t, target_h, target_w))
        ref = _pil_lanczos_resize(img, target_h, target_w)

        mae = np.abs(ours.astype(float) - ref.astype(float)).mean()
        assert mae < 3.0, f"Mean absolute error vs PIL (upscale): {mae:.2f}"

    @pytest.mark.parametrize("device", _devices)
    def test_output_shape(self, device):
        """Output shape should always match the requested target."""
        img = _make_checkerboard(100, 150)
        t = _np_to_tensor(img, device)

        for th, tw in [(50, 75), (33, 47), (200, 300), (1, 1), (100, 50)]:
            result = lanczos_resize(t, th, tw)
            assert result.shape == (1, 3, th, tw), f"Expected shape (1,3,{th},{tw}), got {result.shape}"

    @pytest.mark.parametrize("device", _devices)
    def test_odd_dimensions(self, device):
        """Odd input and target dimensions should work without error."""
        img = _make_checkerboard(101, 77)
        t = _np_to_tensor(img, device)
        result = lanczos_resize(t, 51, 39)
        assert result.shape == (1, 3, 51, 39)

    @pytest.mark.parametrize("device", _devices)
    def test_single_pixel_output(self, device):
        """Downscaling to 1×1 should produce the (weighted) mean."""
        img = _make_gradient(64, 64)
        t = _np_to_tensor(img, device)
        result = lanczos_resize(t, 1, 1)
        assert result.shape == (1, 3, 1, 1)
        # Just check it's a valid value — exact mean depends on kernel.
        val = result[0, 0, 0, 0].item()
        assert 0.0 <= val <= 1.0

    @pytest.mark.parametrize("device", _devices)
    def test_batch_dimension(self, device):
        """Batch of images should be resized independently."""
        imgs = np.stack([_make_checkerboard(64, 64), _make_gradient(64, 64)], axis=0)
        t = torch.from_numpy(imgs).permute(0, 3, 1, 2).float().to(device) / 255.0
        result = lanczos_resize(t, 32, 32)
        assert result.shape == (2, 3, 32, 32)

    @pytest.mark.parametrize("device", _devices)
    def test_rgba(self, device):
        """4-channel (RGBA) input should work."""
        img = _make_checkerboard(64, 96, channels=3)
        alpha = np.full((64, 96, 1), 200, dtype=np.uint8)
        rgba = np.concatenate([img, alpha], axis=-1)

        t = _np_to_tensor(rgba, device)
        result = lanczos_resize(t, 32, 48)
        assert result.shape == (1, 4, 32, 48)

    @pytest.mark.parametrize("device", _devices)
    def test_output_range(self, device):
        """Output values should stay close to [0, 1] (Lanczos can ring slightly).

        Lanczos-3 has negative lobes, so ringing on high-contrast edges
        (like a fine checkerboard) is expected.  ±0.3 is normal.
        """
        img = _make_checkerboard(128, 128, block=4)
        t = _np_to_tensor(img, device)
        result = lanczos_resize(t, 64, 64)
        assert result.min().item() > -0.35, f"Output min {result.min().item()}"
        assert result.max().item() < 1.35, f"Output max {result.max().item()}"


class TestDeviceConsistency:
    """CPU and GPU results should match."""

    _gpu_devices = [d for d in _devices if d != "cpu"]

    @pytest.mark.parametrize("gpu", _gpu_devices)
    def test_cpu_gpu_match(self, gpu):
        """Same input on CPU and GPU should produce (nearly) identical output."""
        img = _make_gradient(200, 300)
        t_cpu = _np_to_tensor(img, "cpu")
        t_gpu = _np_to_tensor(img, gpu)

        out_cpu = lanczos_resize(t_cpu, 100, 150)
        out_gpu = lanczos_resize(t_gpu, 100, 150)

        diff = (out_cpu - out_gpu.cpu()).abs().max().item()
        assert diff < 1e-4, f"CPU/{gpu} mismatch: max diff {diff}"


class TestMipchain:
    """Tests for lanczos_mipchain."""

    @pytest.mark.parametrize("device", _devices)
    def test_mipchain_sizes(self, device):
        """Each mip level should be half the previous, down to min_size."""
        img = _make_checkerboard(512, 256)
        t = _np_to_tensor(img, device)
        chain = lanczos_mipchain(t, min_size=64)

        assert chain[0] is t  # first level is the original
        for i in range(1, len(chain)):
            prev_h, prev_w = chain[i - 1].shape[2], chain[i - 1].shape[3]
            cur_h, cur_w = chain[i].shape[2], chain[i].shape[3]
            assert cur_h == prev_h // 2
            assert cur_w == prev_w // 2

    @pytest.mark.parametrize("device", _devices)
    def test_mipchain_stops_at_min_size(self, device):
        """Chain should stop when the short edge would drop below min_size."""
        img = _make_checkerboard(512, 256)
        t = _np_to_tensor(img, device)
        chain = lanczos_mipchain(t, min_size=64)

        last = chain[-1]
        short_edge = min(last.shape[2], last.shape[3])
        assert short_edge >= 64, f"Short edge {short_edge} dropped below min_size"

        # Verify that one more halving *would* drop below.
        assert min(last.shape[2] // 2, last.shape[3] // 2) < 64

    @pytest.mark.parametrize("device", _devices)
    def test_mipchain_small_input(self, device):
        """Input already at or below min_size should return just the original."""
        img = _make_checkerboard(32, 32)
        t = _np_to_tensor(img, device)
        chain = lanczos_mipchain(t, min_size=64)
        assert len(chain) == 1
        assert chain[0] is t

    @pytest.mark.parametrize("device", _devices)
    def test_mipchain_preserves_batch_and_channels(self, device):
        """Batch and channel dimensions should be preserved at every level."""
        img = _make_checkerboard(256, 256, channels=3)
        alpha = np.full((256, 256, 1), 128, dtype=np.uint8)
        rgba = np.concatenate([img, alpha], axis=-1)
        t = _np_to_tensor(rgba, device)

        chain = lanczos_mipchain(t, min_size=32)
        for level in chain:
            assert level.shape[0] == 1  # batch
            assert level.shape[1] == 4  # RGBA

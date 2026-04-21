"""Tests for `raven.common.image.utils` — tensor conversions and RGBA normalization.

Image decoding / encoding is tested separately; see `test_codec.py`.
"""

import numpy as np
import pytest
import torch

from raven.common.image.utils import (
    ensure_rgba, np_to_tensor, tensor_to_dpg_flat, letterbox,
)
from raven.common.image.lanczos import DEFAULT_ORDER


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def device():
    """Best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# ensure_rgba
# ---------------------------------------------------------------------------

class TestEnsureRgba:
    def test_passthrough_when_already_rgba(self):
        arr = np.zeros((4, 5, 4), dtype=np.uint8)
        arr[..., 3] = 128  # non-default alpha; verify it isn't clobbered
        result = ensure_rgba(arr)
        assert result is arr  # no copy when already 4-channel
        assert (result[..., 3] == 128).all()

    def test_appends_alpha_for_rgb_uint8(self):
        arr = np.zeros((4, 5, 3), dtype=np.uint8)
        result = ensure_rgba(arr)
        assert result.shape == (4, 5, 4)
        assert result.dtype == np.uint8
        assert (result[..., 3] == 255).all()  # fully opaque

    def test_appends_alpha_for_rgb_float(self):
        arr = np.zeros((4, 5, 3), dtype=np.float32)
        result = ensure_rgba(arr)
        assert result.shape == (4, 5, 4)
        assert result.dtype == np.float32
        assert (result[..., 3] == 1.0).all()

    def test_raises_on_wrong_rank(self):
        with pytest.raises(ValueError):
            ensure_rgba(np.zeros((4, 5), dtype=np.uint8))

    def test_raises_on_wrong_channel_count(self):
        with pytest.raises(ValueError):
            ensure_rgba(np.zeros((4, 5, 2), dtype=np.uint8))


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------

class TestTensorHelpers:
    def test_np_to_tensor_shape(self, device):
        arr = np.zeros((100, 200, 4), dtype=np.uint8)
        t = np_to_tensor(arr, device)
        assert t.shape == (1, 4, 100, 200)
        assert t.dtype == torch.float32
        assert t.device.type == device.type

    def test_np_to_tensor_range(self, device):
        arr = np.full((10, 10, 4), 255, dtype=np.uint8)
        t = np_to_tensor(arr, device)
        assert t.max().item() == pytest.approx(1.0, abs=1e-5)

        arr = np.zeros((10, 10, 4), dtype=np.uint8)
        t = np_to_tensor(arr, device)
        assert t.min().item() == pytest.approx(0.0, abs=1e-5)

    def test_tensor_to_dpg_flat(self, device):
        t = torch.rand(1, 4, 32, 32, device=device)
        flat = tensor_to_dpg_flat(t)
        assert flat.dtype == np.float32
        assert flat.shape == (32 * 32 * 4,)
        assert flat.min() >= 0.0
        assert flat.max() <= 1.0

    def test_tensor_to_dpg_flat_clamps(self, device):
        """Lanczos ringing can produce values outside [0, 1]; DPG flat must be clamped."""
        t = torch.tensor([[[[-0.5, 1.5]]]], device=device).expand(1, 4, 1, 2)
        flat = tensor_to_dpg_flat(t)
        assert flat.min() >= 0.0
        assert flat.max() <= 1.0


# ---------------------------------------------------------------------------
# Letterbox
# ---------------------------------------------------------------------------

class TestLetterbox:
    def test_square_input(self, device):
        t = torch.rand(1, 4, 100, 100, device=device)
        result = letterbox(t, 64, order=DEFAULT_ORDER)
        assert result.shape == (1, 4, 64, 64)

    def test_landscape_input(self, device):
        """Wide image: should have letterbox bars top/bottom."""
        t = torch.ones(1, 4, 50, 200, device=device)
        result = letterbox(t, 64, order=DEFAULT_ORDER, bg_value=0.0)
        assert result.shape == (1, 4, 64, 64)
        # Top-left corner should be background (0.0), center should have content.
        assert result[0, 0, 0, 0].item() == pytest.approx(0.0, abs=0.1)

    def test_portrait_input(self, device):
        """Tall image: should have letterbox bars left/right."""
        t = torch.ones(1, 4, 200, 50, device=device)
        result = letterbox(t, 64, order=DEFAULT_ORDER, bg_value=0.0)
        assert result.shape == (1, 4, 64, 64)
        assert result[0, 0, 0, 0].item() == pytest.approx(0.0, abs=0.1)

    def test_tiny_input(self, device):
        """Very small input should still produce the right output size."""
        t = torch.rand(1, 4, 3, 5, device=device)
        result = letterbox(t, 64, order=DEFAULT_ORDER)
        assert result.shape == (1, 4, 64, 64)

    def test_already_tile_size(self, device):
        """Input already at tile size should be ~identity."""
        t = torch.rand(1, 4, 64, 64, device=device)
        result = letterbox(t, 64, order=DEFAULT_ORDER)
        assert result.shape == (1, 4, 64, 64)

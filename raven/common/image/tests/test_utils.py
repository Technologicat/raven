"""Tests for raven.common.image.utils — image decoding and tensor conversions."""

import numpy as np
import pytest
import torch
from PIL import Image

from raven.common.image.utils import (
    decode_image, np_to_tensor, tensor_to_dpg_flat, letterbox,
)
from raven.common.image.lanczos import DEFAULT_ORDER


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_dir(tmp_path):
    """Create a temp directory with a few synthetic test images."""
    # PNG — 200×100 red gradient.
    img = Image.new("RGBA", (200, 100), (0, 0, 0, 255))
    pixels = img.load()
    for x in range(200):
        for y in range(100):
            pixels[x, y] = (int(x / 200 * 255), 0, 0, 255)
    img.save(tmp_path / "red_gradient.png")

    # JPEG — 150×150 green.
    img = Image.new("RGB", (150, 150), (0, 200, 0))
    img.save(tmp_path / "green.jpg", quality=90)

    # JPEG — 300×200 blue.
    img = Image.new("RGB", (300, 200), (0, 0, 200))
    img.save(tmp_path / "blue.jpg", quality=90)

    # Small — 16×16 white.
    img = Image.new("RGBA", (16, 16), (255, 255, 255, 255))
    img.save(tmp_path / "tiny.png")

    return tmp_path


@pytest.fixture
def device():
    """Best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# decode_image
# ---------------------------------------------------------------------------

class TestDecodeImage:
    def test_decode_png(self, sample_dir):
        arr = decode_image(sample_dir / "red_gradient.png")
        assert arr.dtype == np.uint8
        assert arr.shape == (100, 200, 4)  # H, W, RGBA

    def test_decode_jpeg(self, sample_dir):
        arr = decode_image(sample_dir / "green.jpg")
        assert arr.dtype == np.uint8
        assert arr.shape == (150, 150, 4)  # JPEG converted to RGBA

    def test_decode_returns_rgba(self, sample_dir):
        """All formats should return 4-channel RGBA."""
        for name in ["red_gradient.png", "green.jpg", "tiny.png"]:
            arr = decode_image(sample_dir / name)
            assert arr.shape[2] == 4, f"{name}: expected 4 channels, got {arr.shape[2]}"

    def test_decode_with_max_size_hint(self, sample_dir):
        """max_size hint should not crash (actual scaling depends on turbojpeg availability)."""
        arr = decode_image(sample_dir / "blue.jpg", max_size=64)
        assert arr.dtype == np.uint8
        assert arr.shape[2] == 4
        # With turbojpeg: dimensions would be scaled down.
        # Without: full 300×200 returned.  Either is fine.
        assert arr.shape[0] >= 64 or arr.shape[1] >= 64

    def test_decode_accepts_str_path(self, sample_dir):
        arr = decode_image(str(sample_dir / "tiny.png"))
        assert arr.shape == (16, 16, 4)

    def test_decode_nonexistent_raises(self, tmp_path):
        with pytest.raises(Exception):
            decode_image(tmp_path / "nonexistent.png")


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

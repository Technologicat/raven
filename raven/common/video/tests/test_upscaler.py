"""Tests for raven.common.video.upscaler — Anime4K-based image upscaler."""

import pytest
import torch

from raven.common.video.upscaler import Upscaler


# ---------------------------------------------------------------------------
# Tests: parameter validation
# ---------------------------------------------------------------------------

class TestUpscalerValidation:
    def test_invalid_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            Upscaler("cpu", torch.float32, 256, 256, preset="X")

    def test_invalid_quality_raises(self):
        with pytest.raises(ValueError, match="Unknown quality"):
            Upscaler("cpu", torch.float32, 256, 256, quality="ultra")

    def test_valid_presets(self):
        """All valid preset/quality combinations construct without error."""
        for preset in ("A", "B", "C"):
            for quality in ("low", "high"):
                u = Upscaler("cpu", torch.float32, 64, 64, preset=preset, quality=quality)
                assert u.preset == preset
                assert u.quality == quality


# ---------------------------------------------------------------------------
# Tests: shape contracts
# ---------------------------------------------------------------------------

class TestUpscalerShape:
    """Output shape must match the configured target dimensions."""

    @pytest.fixture(scope="class")
    def upscaler(self):
        """Shared upscaler instance (preset C/low is fastest)."""
        return Upscaler("cpu", torch.float32, 64, 48, preset="C", quality="low")

    def test_rgb_output_shape(self, upscaler):
        image = torch.rand(3, 16, 16)
        result = upscaler.upscale(image)
        assert result.shape == (3, 48, 64)

    def test_rgba_output_shape(self, upscaler):
        image = torch.rand(4, 16, 16)
        result = upscaler.upscale(image)
        assert result.shape == (4, 48, 64)

    def test_invalid_channel_count_raises(self, upscaler):
        image = torch.rand(2, 16, 16)
        with pytest.raises(ValueError, match="3.*or 4.*channels"):
            upscaler.upscale(image)

    def test_non_square_input(self, upscaler):
        """Non-square input produces the configured output shape."""
        image = torch.rand(3, 8, 24)
        result = upscaler.upscale(image)
        assert result.shape == (3, 48, 64)


# ---------------------------------------------------------------------------
# Tests: alpha channel handling
# ---------------------------------------------------------------------------

class TestUpscalerAlpha:
    @pytest.fixture(scope="class")
    def upscaler(self):
        return Upscaler("cpu", torch.float32, 64, 48, preset="C", quality="low")

    def test_alpha_preserved_opaque(self, upscaler):
        """Fully opaque alpha remains ~1.0 after bilinear upscale."""
        image = torch.rand(4, 16, 16)
        image[3] = 1.0
        result = upscaler.upscale(image)
        assert torch.allclose(result[3], torch.ones(48, 64), atol=1e-4)

    def test_alpha_preserved_transparent(self, upscaler):
        """Fully transparent alpha remains ~0.0 after bilinear upscale."""
        image = torch.rand(4, 16, 16)
        image[3] = 0.0
        result = upscaler.upscale(image)
        assert torch.allclose(result[3], torch.zeros(48, 64), atol=1e-4)


# ---------------------------------------------------------------------------
# Tests: value sanity
# ---------------------------------------------------------------------------

class TestUpscalerValues:
    @pytest.fixture(scope="class")
    def upscaler(self):
        return Upscaler("cpu", torch.float32, 64, 48, preset="C", quality="low")

    def test_output_in_plausible_range(self, upscaler):
        """Upscaled output should be in a plausible range (roughly [0, 1])."""
        image = torch.rand(3, 16, 16)
        result = upscaler.upscale(image)
        # Allow some overshoot from the neural network
        assert result.min() >= -0.5
        assert result.max() <= 1.5

    def test_all_presets_produce_output(self):
        """Smoke test: every preset/quality combo runs without error."""
        image = torch.rand(3, 8, 8)
        for preset in ("A", "B", "C"):
            for quality in ("low", "high"):
                u = Upscaler("cpu", torch.float32, 32, 32, preset=preset, quality=quality)
                result = u.upscale(image)
                assert result.shape == (3, 32, 32)

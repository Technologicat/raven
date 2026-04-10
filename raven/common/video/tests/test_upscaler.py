"""Tests for raven.common.video.upscaler — Anime4K and interpolation upscaling."""

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

    def test_bilinear_bypass_has_no_pipeline(self):
        u = Upscaler("cpu", torch.float32, 64, 64, quality="bilinear")
        assert u.pipeline is None

    def test_bicubic_bypass_has_no_pipeline(self):
        u = Upscaler("cpu", torch.float32, 64, 64, quality="bicubic")
        assert u.pipeline is None

    def test_anime4k_has_pipeline(self):
        u = Upscaler("cpu", torch.float32, 64, 64, preset="C", quality="low")
        assert u.pipeline is not None


# ---------------------------------------------------------------------------
# Tests: shape contracts — Anime4K
# ---------------------------------------------------------------------------

class TestUpscalerShapeAnime4K:
    """Output shape must match the configured target dimensions (Anime4K path)."""

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
# Tests: shape contracts — bilinear/bicubic bypass
# ---------------------------------------------------------------------------

class TestUpscalerShapeBypass:
    """Output shape for interpolation-only bypass modes."""

    def test_bilinear_rgb_shape(self):
        u = Upscaler("cpu", torch.float32, 64, 48, quality="bilinear")
        result = u.upscale(torch.rand(3, 16, 16))
        assert result.shape == (3, 48, 64)

    def test_bilinear_rgba_shape(self):
        u = Upscaler("cpu", torch.float32, 64, 48, quality="bilinear")
        result = u.upscale(torch.rand(4, 16, 16))
        assert result.shape == (4, 48, 64)

    def test_bicubic_rgb_shape(self):
        u = Upscaler("cpu", torch.float32, 64, 48, quality="bicubic")
        result = u.upscale(torch.rand(3, 16, 16))
        assert result.shape == (3, 48, 64)

    def test_bicubic_rgba_shape(self):
        u = Upscaler("cpu", torch.float32, 64, 48, quality="bicubic")
        result = u.upscale(torch.rand(4, 16, 16))
        assert result.shape == (4, 48, 64)

    def test_non_square_input(self):
        u = Upscaler("cpu", torch.float32, 64, 48, quality="bilinear")
        result = u.upscale(torch.rand(3, 8, 24))
        assert result.shape == (3, 48, 64)

    def test_invalid_channel_count_raises(self):
        u = Upscaler("cpu", torch.float32, 64, 48, quality="bilinear")
        with pytest.raises(ValueError, match="3.*or 4.*channels"):
            u.upscale(torch.rand(2, 16, 16))

    def test_downscale(self):
        """Upscaler can also downscale (target smaller than input)."""
        u = Upscaler("cpu", torch.float32, 16, 16, quality="bilinear")
        result = u.upscale(torch.rand(3, 64, 64))
        assert result.shape == (3, 16, 16)

    def test_identity_size(self):
        """Same input and output size should produce the target shape."""
        u = Upscaler("cpu", torch.float32, 32, 32, quality="bilinear")
        result = u.upscale(torch.rand(3, 32, 32))
        assert result.shape == (3, 32, 32)


# ---------------------------------------------------------------------------
# Tests: alpha channel handling
# ---------------------------------------------------------------------------

class TestUpscalerAlphaAnime4K:
    """Alpha is upscaled bilinearly (Anime4K is RGB-only)."""

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

    def test_alpha_gradient_monotonic(self, upscaler):
        """A monotonic alpha gradient should remain monotonic after upscale."""
        image = torch.rand(4, 16, 16)
        image[3, :, :] = torch.linspace(0.0, 1.0, 16).unsqueeze(0)
        result = upscaler.upscale(image)
        # Each column should be >= the previous (monotonic along width)
        diffs = result[3, :, 1:] - result[3, :, :-1]
        assert (diffs >= -1e-5).all()


class TestUpscalerAlphaBypass:
    """Alpha handling in bypass modes."""

    def test_bilinear_alpha_opaque(self):
        u = Upscaler("cpu", torch.float32, 64, 48, quality="bilinear")
        image = torch.rand(4, 16, 16)
        image[3] = 1.0
        result = u.upscale(image)
        assert torch.allclose(result[3], torch.ones(48, 64), atol=1e-4)

    def test_bilinear_alpha_transparent(self):
        u = Upscaler("cpu", torch.float32, 64, 48, quality="bilinear")
        image = torch.rand(4, 16, 16)
        image[3] = 0.0
        result = u.upscale(image)
        assert torch.allclose(result[3], torch.zeros(48, 64), atol=1e-4)

    def test_bicubic_alpha_uses_bilinear(self):
        """Bicubic mode uses bilinear for alpha to avoid ringing.

        Verify: opaque alpha stays clean (no overshoot from Gibbs phenomenon).
        """
        u = Upscaler("cpu", torch.float32, 64, 48, quality="bicubic")
        image = torch.rand(4, 16, 16)
        # Sharp alpha edge — bicubic would cause ringing, bilinear won't
        image[3, :, :] = 0.0
        image[3, :, 8:] = 1.0
        result = u.upscale(image)
        # Bilinear alpha should stay in [0, 1] — no Gibbs ringing
        assert result[3].min() >= -1e-5
        assert result[3].max() <= 1.0 + 1e-5

    def test_bicubic_rgb_can_overshoot(self):
        """Bicubic RGB may overshoot [0, 1] — that's expected (Gibbs phenomenon)."""
        u = Upscaler("cpu", torch.float32, 128, 128, quality="bicubic")
        # Sharp edge in RGB — bicubic will ring
        image = torch.zeros(3, 16, 16)
        image[:, :, 8:] = 1.0
        result = u.upscale(image)
        # We just verify it runs and produces the right shape;
        # slight overshoot/undershoot is the nature of bicubic
        assert result.shape == (3, 128, 128)


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

    def test_bilinear_preserves_uniform(self):
        """Bilinear upscale of a uniform image should stay uniform."""
        u = Upscaler("cpu", torch.float32, 64, 48, quality="bilinear")
        image = torch.full((3, 16, 16), 0.5)
        result = u.upscale(image)
        assert torch.allclose(result, torch.full((3, 48, 64), 0.5), atol=1e-5)

    def test_bicubic_preserves_uniform(self):
        """Bicubic upscale of a uniform image should stay uniform."""
        u = Upscaler("cpu", torch.float32, 64, 48, quality="bicubic")
        image = torch.full((3, 16, 16), 0.5)
        result = u.upscale(image)
        assert torch.allclose(result, torch.full((3, 48, 64), 0.5), atol=1e-5)


# ---------------------------------------------------------------------------
# Tests: bilinear vs bicubic — behavioral difference
# ---------------------------------------------------------------------------

class TestBypassModeDifferences:
    """Bilinear and bicubic should produce different results on non-trivial input."""

    def test_bilinear_vs_bicubic_differ(self):
        image = torch.rand(3, 16, 16)
        u_bilinear = Upscaler("cpu", torch.float32, 64, 48, quality="bilinear")
        u_bicubic = Upscaler("cpu", torch.float32, 64, 48, quality="bicubic")
        result_bilinear = u_bilinear.upscale(image)
        result_bicubic = u_bicubic.upscale(image)
        assert not torch.equal(result_bilinear, result_bicubic)

    def test_bilinear_smoother_than_bicubic(self):
        """Bicubic should have sharper edges (higher gradient magnitude) than bilinear."""
        image = torch.zeros(3, 16, 16)
        image[:, :, 8:] = 1.0
        u_bilinear = Upscaler("cpu", torch.float32, 64, 48, quality="bilinear")
        u_bicubic = Upscaler("cpu", torch.float32, 64, 48, quality="bicubic")
        result_bilinear = u_bilinear.upscale(image)
        result_bicubic = u_bicubic.upscale(image)
        # Horizontal gradient magnitude
        grad_bilinear = (result_bilinear[0, :, 1:] - result_bilinear[0, :, :-1]).abs().max()
        grad_bicubic = (result_bicubic[0, :, 1:] - result_bicubic[0, :, :-1]).abs().max()
        # Bicubic produces steeper transitions (sharper)
        assert grad_bicubic > grad_bilinear


# ---------------------------------------------------------------------------
# Tests: dtype propagation
# ---------------------------------------------------------------------------

class TestUpscalerDtype:
    def test_bilinear_preserves_dtype(self):
        u = Upscaler("cpu", torch.float32, 64, 48, quality="bilinear")
        image = torch.rand(3, 16, 16, dtype=torch.float32)
        result = u.upscale(image)
        assert result.dtype == torch.float32

    def test_bicubic_preserves_dtype(self):
        u = Upscaler("cpu", torch.float32, 64, 48, quality="bicubic")
        image = torch.rand(3, 16, 16, dtype=torch.float32)
        result = u.upscale(image)
        assert result.dtype == torch.float32

    def test_anime4k_preserves_dtype(self):
        u = Upscaler("cpu", torch.float32, 64, 48, preset="C", quality="low")
        image = torch.rand(3, 16, 16, dtype=torch.float32)
        result = u.upscale(image)
        assert result.dtype == torch.float32

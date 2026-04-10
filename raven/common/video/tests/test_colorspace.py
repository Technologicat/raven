"""Tests for raven.common.video.colorspace — RGB/YUV conversion and utilities."""

import torch

from raven.common.video.colorspace import rgb_to_yuv, yuv_to_rgb, luminance, hex_to_rgb


# ---------------------------------------------------------------------------
# Tests: round-trip accuracy
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """rgb_to_yuv -> yuv_to_rgb should be (near-)identity for valid RGB inputs."""

    def test_round_trip_uniform(self):
        """Round-trip through YUV preserves a uniform color field."""
        rgb = torch.rand(3, 16, 16)
        recovered = yuv_to_rgb(rgb_to_yuv(rgb), clamp=False)
        assert torch.allclose(rgb, recovered, atol=1e-5)

    def test_round_trip_black(self):
        rgb = torch.zeros(3, 4, 4)
        recovered = yuv_to_rgb(rgb_to_yuv(rgb), clamp=False)
        assert torch.allclose(rgb, recovered, atol=1e-5)

    def test_round_trip_white(self):
        rgb = torch.ones(3, 4, 4)
        recovered = yuv_to_rgb(rgb_to_yuv(rgb), clamp=False)
        assert torch.allclose(rgb, recovered, atol=1e-5)

    def test_round_trip_primaries(self):
        """Round-trip preserves pure R, G, B channels."""
        for c in range(3):
            rgb = torch.zeros(3, 4, 4)
            rgb[c] = 1.0
            recovered = yuv_to_rgb(rgb_to_yuv(rgb), clamp=False)
            assert torch.allclose(rgb, recovered, atol=1e-5), f"Failed for channel {c}"


# ---------------------------------------------------------------------------
# Tests: YUV value ranges
# ---------------------------------------------------------------------------

class TestYuvRanges:
    """For valid RGB in [0, 1], Y should be in [0, 1] and U, V in [-0.5, 0.5]."""

    def test_y_range(self):
        rgb = torch.rand(3, 32, 32)
        yuv = rgb_to_yuv(rgb)
        assert yuv[0].min() >= -1e-6
        assert yuv[0].max() <= 1.0 + 1e-6

    def test_uv_range(self):
        rgb = torch.rand(3, 32, 32)
        yuv = rgb_to_yuv(rgb)
        assert yuv[1].min() >= -0.5 - 1e-6
        assert yuv[1].max() <= 0.5 + 1e-6
        assert yuv[2].min() >= -0.5 - 1e-6
        assert yuv[2].max() <= 0.5 + 1e-6

    def test_neutral_gray_has_zero_chroma(self):
        """50% gray has Y=0.5, U=V=0."""
        rgb = torch.full((3, 4, 4), 0.5)
        yuv = rgb_to_yuv(rgb)
        assert torch.allclose(yuv[0], torch.full((4, 4), 0.5), atol=1e-5)
        assert torch.allclose(yuv[1], torch.zeros(4, 4), atol=1e-5)
        assert torch.allclose(yuv[2], torch.zeros(4, 4), atol=1e-5)


# ---------------------------------------------------------------------------
# Tests: shape contracts
# ---------------------------------------------------------------------------

class TestShapeContracts:
    def test_rgb_to_yuv_shape(self):
        rgb = torch.rand(3, 16, 32)
        yuv = rgb_to_yuv(rgb)
        assert yuv.shape == (3, 16, 32)

    def test_yuv_to_rgb_shape(self):
        yuv = torch.rand(3, 16, 32)
        rgb = yuv_to_rgb(yuv)
        assert rgb.shape == (3, 16, 32)

    def test_luminance_shape(self):
        """luminance returns [H, W], not [1, H, W]."""
        rgb = torch.rand(3, 16, 32)
        y = luminance(rgb)
        assert y.shape == (16, 32)


# ---------------------------------------------------------------------------
# Tests: luminance
# ---------------------------------------------------------------------------

class TestLuminance:
    def test_luminance_matches_y_channel(self):
        """luminance() should match the Y channel from rgb_to_yuv()."""
        rgb = torch.rand(3, 16, 16)
        y_from_luminance = luminance(rgb)
        y_from_yuv = rgb_to_yuv(rgb)[0]
        assert torch.allclose(y_from_luminance, y_from_yuv, atol=1e-6)

    def test_luminance_black_is_zero(self):
        rgb = torch.zeros(3, 4, 4)
        assert torch.allclose(luminance(rgb), torch.zeros(4, 4))

    def test_luminance_white_is_one(self):
        rgb = torch.ones(3, 4, 4)
        assert torch.allclose(luminance(rgb), torch.ones(4, 4), atol=1e-5)

    def test_green_brighter_than_red_brighter_than_blue(self):
        """BT.709: green has the highest luminance coefficient, blue the lowest."""
        r = torch.zeros(3, 1, 1)
        r[0] = 1.0
        g = torch.zeros(3, 1, 1)
        g[1] = 1.0
        b = torch.zeros(3, 1, 1)
        b[2] = 1.0
        assert luminance(g).item() > luminance(r).item() > luminance(b).item()


# ---------------------------------------------------------------------------
# Tests: hex_to_rgb
# ---------------------------------------------------------------------------

class TestHexToRgb:
    def test_black(self):
        assert hex_to_rgb("#000000") == (0, 0, 0)

    def test_white(self):
        assert hex_to_rgb("#ffffff") == (255, 255, 255)

    def test_red(self):
        assert hex_to_rgb("#ff0000") == (255, 0, 0)

    def test_with_alpha(self):
        assert hex_to_rgb("#ff000080") == (255, 0, 0, 128)

    def test_uppercase(self):
        assert hex_to_rgb("#FF8800") == (255, 136, 0)

    def test_without_hash(self):
        """Tolerates missing '#' prefix."""
        assert hex_to_rgb("ff0000") == (255, 0, 0)


# ---------------------------------------------------------------------------
# Tests: dtype propagation
# ---------------------------------------------------------------------------

class TestDtypePropagation:
    def test_rgb_to_yuv_preserves_dtype(self):
        for dt in (torch.float32, torch.float64):
            rgb = torch.rand(3, 4, 4, dtype=dt)
            yuv = rgb_to_yuv(rgb)
            assert yuv.dtype == dt

    def test_luminance_preserves_dtype(self):
        for dt in (torch.float32, torch.float64):
            rgb = torch.rand(3, 4, 4, dtype=dt)
            y = luminance(rgb)
            assert y.dtype == dt


# ---------------------------------------------------------------------------
# Tests: clamping behavior
# ---------------------------------------------------------------------------

class TestClamping:
    def test_yuv_to_rgb_clamp_on(self):
        """With clamp=True, out-of-gamut values are clamped to [0, 1]."""
        yuv = torch.zeros(3, 4, 4)
        yuv[1] = 0.5  # extreme chroma
        yuv[2] = 0.5
        rgb = yuv_to_rgb(yuv, clamp=True)
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0

    def test_yuv_to_rgb_clamp_off(self):
        """With clamp=False, out-of-gamut values are preserved."""
        yuv = torch.zeros(3, 4, 4)
        yuv[1] = 0.5  # this will produce negative RGB
        yuv[2] = 0.5
        rgb = yuv_to_rgb(yuv, clamp=False)
        # At least one channel should be negative or > 1 for this extreme chroma
        assert rgb.min() < 0.0 or rgb.max() > 1.0


# ---------------------------------------------------------------------------
# Tests: BT.709 coefficient verification
# ---------------------------------------------------------------------------

class TestBT709Coefficients:
    """Verify the conversion uses BT.709 (HDTV) color primaries."""

    def test_pure_red_luminance(self):
        """Pure red should have Y = 0.2126 (BT.709 red coefficient)."""
        rgb = torch.zeros(3, 1, 1)
        rgb[0] = 1.0
        assert torch.allclose(luminance(rgb), torch.tensor(0.2126), atol=1e-4)

    def test_pure_green_luminance(self):
        """Pure green should have Y = 0.7152 (BT.709 green coefficient)."""
        rgb = torch.zeros(3, 1, 1)
        rgb[1] = 1.0
        assert torch.allclose(luminance(rgb), torch.tensor(0.7152), atol=1e-4)

    def test_pure_blue_luminance(self):
        """Pure blue should have Y = 0.0722 (BT.709 blue coefficient)."""
        rgb = torch.zeros(3, 1, 1)
        rgb[2] = 1.0
        assert torch.allclose(luminance(rgb), torch.tensor(0.0722), atol=1e-4)


# ---------------------------------------------------------------------------
# Tests: additional coverage
# ---------------------------------------------------------------------------

class TestYuvToRgbDtype:
    def test_yuv_to_rgb_preserves_dtype(self):
        for dt in (torch.float32, torch.float64):
            yuv = torch.rand(3, 4, 4, dtype=dt)
            rgb = yuv_to_rgb(yuv)
            assert rgb.dtype == dt


class TestLuminanceRange:
    def test_luminance_in_unit_interval(self):
        """For valid RGB in [0, 1], luminance should be in [0, 1]."""
        torch.manual_seed(42)
        rgb = torch.rand(3, 64, 64)
        y = luminance(rgb)
        assert y.min() >= -1e-6
        assert y.max() <= 1.0 + 1e-6

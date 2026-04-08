"""Tests for raven.common.video.postprocessor — noise primitives and cache mechanics."""

import torch

from raven.common.video.postprocessor import vhs_noise, isotropic_noise, Postprocessor


# ---------------------------------------------------------------------------
# Tests: vhs_noise shape contract
# ---------------------------------------------------------------------------

class TestVhsNoise:
    """Shape and value-range contracts for the VHS noise primitive."""

    def test_pal_shape(self):
        """PAL mode returns [1, H, W]."""
        result = vhs_noise(64, 32, device="cpu", mode="PAL")
        assert result.shape == (1, 32, 64)

    def test_ntsc_shape(self):
        """NTSC mode returns [3, H, W]."""
        result = vhs_noise(64, 32, device="cpu", mode="NTSC")
        assert result.shape == (3, 32, 64)

    def test_pal_double_size_shape(self):
        """double_size produces the requested output dimensions, not half."""
        result = vhs_noise(64, 32, device="cpu", mode="PAL", double_size=True)
        assert result.shape == (1, 32, 64)

    def test_ntsc_double_size_shape(self):
        result = vhs_noise(64, 32, device="cpu", mode="NTSC", double_size=True)
        assert result.shape == (3, 32, 64)

    def test_pal_double_size_odd_dimensions(self):
        """Odd target sizes are handled correctly (ceil-div then trim)."""
        result = vhs_noise(63, 31, device="cpu", mode="PAL", double_size=True)
        assert result.shape == (1, 31, 63)

    def test_ntsc_double_size_odd_dimensions(self):
        result = vhs_noise(63, 31, device="cpu", mode="NTSC", double_size=True)
        assert result.shape == (3, 31, 63)

    def test_double_size_grain_structure(self):
        """With double_size, adjacent 2x2 pixel blocks should be identical."""
        # Use even dimensions so repeat_interleave doesn't need trimming
        result = vhs_noise(64, 32, device="cpu", mode="PAL", double_size=True)
        # Check that horizontally adjacent pairs are equal
        assert torch.equal(result[:, :, 0::2], result[:, :, 1::2])
        # Check that vertically adjacent pairs are equal
        assert torch.equal(result[:, 0::2, :], result[:, 1::2, :])

    def test_unknown_mode_raises(self):
        """Unknown mode raises ValueError."""
        try:
            vhs_noise(64, 32, device="cpu", mode="SECAM")
            assert False, "Expected ValueError"
        except ValueError:
            pass

    def test_pal_value_range(self):
        """PAL luma noise is in [0, 1] (blurred uniform)."""
        result = vhs_noise(128, 128, device="cpu", mode="PAL")
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_dtype_propagation(self):
        """Output dtype matches requested dtype."""
        for dt in (torch.float32, torch.float16):
            result = vhs_noise(32, 32, device="cpu", dtype=dt, mode="PAL")
            assert result.dtype == dt


# ---------------------------------------------------------------------------
# Tests: isotropic_noise shape contract
# ---------------------------------------------------------------------------

class TestIsotropicNoise:
    """Shape and value-range contracts for the isotropic noise primitive."""

    def test_shape(self):
        """Returns [H, W]."""
        result = isotropic_noise(64, 32, device="cpu")
        assert result.shape == (32, 64)

    def test_double_size_shape(self):
        """double_size produces the requested output dimensions."""
        result = isotropic_noise(64, 32, device="cpu", double_size=True)
        assert result.shape == (32, 64)

    def test_double_size_odd_dimensions(self):
        result = isotropic_noise(63, 31, device="cpu", double_size=True)
        assert result.shape == (31, 63)

    def test_double_size_grain_structure(self):
        """With double_size, adjacent 2x2 pixel blocks should be identical."""
        result = isotropic_noise(64, 32, device="cpu", double_size=True)
        assert torch.equal(result[0::2, :], result[1::2, :])
        assert torch.equal(result[:, 0::2], result[:, 1::2])

    def test_no_blur(self):
        """sigma=0 skips Gaussian blur (output is raw uniform noise)."""
        result = isotropic_noise(64, 32, device="cpu", sigma=0.0)
        assert result.shape == (32, 64)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_value_range(self):
        """Blurred noise is in [0, 1]."""
        result = isotropic_noise(128, 128, device="cpu", sigma=1.0)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_dtype_propagation(self):
        for dt in (torch.float32, torch.float16):
            result = isotropic_noise(32, 32, device="cpu", dtype=dt)
            assert result.dtype == dt


# ---------------------------------------------------------------------------
# Tests: Postprocessor noise cache mechanics
# ---------------------------------------------------------------------------

def _make_postprocessor():
    """Create a Postprocessor with an empty chain for cache testing."""
    pp = Postprocessor("cpu", torch.float32, chain=[])
    # Simulate render_into having seen at least one frame, so _prev_h/_prev_w are set.
    pp._prev_h = 64
    pp._prev_w = 128
    return pp


def _make_image(h=64, w=128, c=4):
    """Create a dummy RGBA image tensor."""
    return torch.rand(c, h, w, dtype=torch.float32, device="cpu")


class TestNoiseCacheHit:
    """The noise cache should return the same tensor within an integer frame boundary."""

    def test_noise_filter_cache_hit(self):
        """noise filter: same integer frame -> same tensor."""
        pp = _make_postprocessor()
        pp.frame_no = 1.0
        pp.last_frame_no = 0.0
        image = _make_image()
        pp.noise(image, strength=0.3, sigma=0.0, channel="Y", double_size=False)
        cached1 = pp.noise_last_image["noise0"]

        # Sub-frame advance (same integer frame)
        pp.last_frame_no = pp.frame_no
        pp.frame_no = 1.5
        image = _make_image()
        pp.noise(image, strength=0.3, sigma=0.0, channel="Y", double_size=False)
        cached2 = pp.noise_last_image["noise0"]

        assert cached1 is cached2

    def test_noise_filter_cache_miss_on_frame_boundary(self):
        """noise filter: crossing integer frame boundary -> new tensor."""
        pp = _make_postprocessor()
        pp.frame_no = 1.0
        pp.last_frame_no = 0.0
        image = _make_image()
        pp.noise(image, strength=0.3, sigma=0.0, channel="Y", double_size=False)
        cached1 = pp.noise_last_image["noise0"]

        # Advance past integer boundary
        pp.last_frame_no = pp.frame_no
        pp.frame_no = 2.0
        image = _make_image()
        pp.noise(image, strength=0.3, sigma=0.0, channel="Y", double_size=False)
        cached2 = pp.noise_last_image["noise0"]

        assert cached1 is not cached2

    def test_noise_filter_cache_miss_on_strength_change(self):
        """noise filter: strength change invalidates cache."""
        pp = _make_postprocessor()
        pp.frame_no = 1.0
        pp.last_frame_no = 0.0
        image = _make_image()
        pp.noise(image, strength=0.3, sigma=0.0, channel="Y", double_size=False)
        cached1 = pp.noise_last_image["noise0"]

        # Same frame, different strength
        pp.last_frame_no = pp.frame_no
        pp.frame_no = 1.5
        image = _make_image()
        pp.noise(image, strength=0.5, sigma=0.0, channel="Y", double_size=False)
        cached2 = pp.noise_last_image["noise0"]

        assert cached1 is not cached2

    def test_headswitching_cache_hit(self):
        """analog_vhs_headswitching: same integer frame -> same tensor."""
        pp = _make_postprocessor()
        pp.frame_no = 1.0
        pp.last_frame_no = 0.0
        # Need meshgrid for headswitching (it uses grid_sample)
        pp._meshy, pp._meshx = torch.meshgrid(
            torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 128), indexing="ij")
        image = _make_image()
        pp.analog_vhs_headswitching(image, noise_blend=0.5, double_size=False)
        cached1 = pp.vhs_headswitching_noise["analog_vhs_headswitching0"]

        pp.last_frame_no = pp.frame_no
        pp.frame_no = 1.5
        image = _make_image()
        pp.analog_vhs_headswitching(image, noise_blend=0.5, double_size=False)
        cached2 = pp.vhs_headswitching_noise["analog_vhs_headswitching0"]

        assert cached1 is cached2

    def test_headswitching_cache_miss_on_frame_boundary(self):
        """analog_vhs_headswitching: crossing integer frame boundary -> new tensor."""
        pp = _make_postprocessor()
        pp.frame_no = 1.0
        pp.last_frame_no = 0.0
        pp._meshy, pp._meshx = torch.meshgrid(
            torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 128), indexing="ij")
        image = _make_image()
        pp.analog_vhs_headswitching(image, noise_blend=0.5, double_size=False)
        cached1 = pp.vhs_headswitching_noise["analog_vhs_headswitching0"]

        pp.last_frame_no = pp.frame_no
        pp.frame_no = 2.0
        image = _make_image()
        pp.analog_vhs_headswitching(image, noise_blend=0.5, double_size=False)
        cached2 = pp.vhs_headswitching_noise["analog_vhs_headswitching0"]

        assert cached1 is not cached2

    def test_tracking_cache_hit(self):
        """analog_vhstracking: same integer frame -> same tensor."""
        pp = _make_postprocessor()
        pp.frame_no = 1.0
        pp.last_frame_no = 0.0
        pp._meshy, pp._meshx = torch.meshgrid(
            torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 128), indexing="ij")
        # Use large base_offset to ensure noise_pixels > 0
        image = _make_image()
        pp.analog_vhstracking(image, base_offset=0.1, max_dynamic_offset=0.0,
                              double_size=False)
        cached1 = pp.vhs_tracking_noise["analog_vhstracking0"]

        pp.last_frame_no = pp.frame_no
        pp.frame_no = 1.5
        image = _make_image()
        pp.analog_vhstracking(image, base_offset=0.1, max_dynamic_offset=0.0,
                              double_size=False)
        cached2 = pp.vhs_tracking_noise["analog_vhstracking0"]

        assert cached1 is cached2

    def test_tracking_cache_miss_on_frame_boundary(self):
        """analog_vhstracking: crossing integer frame boundary -> new tensor."""
        pp = _make_postprocessor()
        pp.frame_no = 1.0
        pp.last_frame_no = 0.0
        pp._meshy, pp._meshx = torch.meshgrid(
            torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 128), indexing="ij")
        image = _make_image()
        pp.analog_vhstracking(image, base_offset=0.1, max_dynamic_offset=0.0,
                              double_size=False)
        cached1 = pp.vhs_tracking_noise["analog_vhstracking0"]

        pp.last_frame_no = pp.frame_no
        pp.frame_no = 2.0
        image = _make_image()
        pp.analog_vhstracking(image, base_offset=0.1, max_dynamic_offset=0.0,
                              double_size=False)
        cached2 = pp.vhs_tracking_noise["analog_vhstracking0"]

        assert cached1 is not cached2


class TestNoiseCacheSizeInvalidation:
    """The noise cache should regenerate when image size changes."""

    def test_noise_filter_size_change(self):
        pp = _make_postprocessor()
        pp.frame_no = 1.0
        pp.last_frame_no = 0.0
        image = _make_image(h=64, w=128)
        pp.noise(image, strength=0.3, sigma=0.0, channel="Y", double_size=False)
        cached1 = pp.noise_last_image["noise0"]

        # Same frame, different size
        pp.last_frame_no = pp.frame_no
        pp.frame_no = 1.5
        image = _make_image(h=32, w=64)
        pp.noise(image, strength=0.3, sigma=0.0, channel="Y", double_size=False)
        cached2 = pp.noise_last_image["noise0"]

        assert cached1 is not cached2


class TestTrackingNoiseJitter:
    """The tracking filter handles noise_pixels height jitter within a frame."""

    def test_slice_when_cached_taller(self):
        """If cached noise is taller than needed, slice it (don't regenerate)."""
        pp = _make_postprocessor()
        pp.frame_no = 1.0
        pp.last_frame_no = 0.0
        pp._meshy, pp._meshx = torch.meshgrid(
            torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 128), indexing="ij")

        # First call generates the noise
        image = _make_image()
        pp.analog_vhstracking(image, base_offset=0.1, max_dynamic_offset=0.0,
                              double_size=False)
        cached = pp.vhs_tracking_noise["analog_vhstracking0"]

        # Manually shrink the offset to simulate jitter producing fewer noise_pixels.
        # The cached tensor should be sliced, not regenerated.
        pp.last_frame_no = pp.frame_no
        pp.frame_no = 1.5  # same integer frame
        image = _make_image()
        pp.analog_vhstracking(image, base_offset=0.08, max_dynamic_offset=0.0,
                              double_size=False)
        # The cache entry itself is not replaced on a slice
        still_cached = pp.vhs_tracking_noise["analog_vhstracking0"]
        assert still_cached is cached  # same object — sliced at use, not in cache


class TestNoiseCacheIndependentNames:
    """Different name= values get independent caches."""

    def test_independent_noise_caches(self):
        pp = _make_postprocessor()
        pp.frame_no = 1.0
        pp.last_frame_no = 0.0

        image = _make_image()
        pp.noise(image, strength=0.3, sigma=0.0, channel="Y", double_size=False,
                 name="a")
        cached_a = pp.noise_last_image["a"]

        image = _make_image()
        pp.noise(image, strength=0.5, sigma=0.0, channel="Y", double_size=False,
                 name="b")
        cached_b = pp.noise_last_image["b"]

        assert cached_a is not cached_b
        # "a" was not invalidated by calling with name="b"
        assert pp.noise_last_image["a"] is cached_a

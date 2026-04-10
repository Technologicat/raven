"""Tests for raven.common.video.postprocessor — noise primitives, cache mechanics, and filters."""

import torch

from raven.common.video.colorspace import rgb_to_yuv
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


# ---------------------------------------------------------------------------
# Tests: chroma_subsample filter
# ---------------------------------------------------------------------------

def _make_colorful_image(h=64, w=128):
    """Create a test RGBA image with strong chroma content.

    Horizontal red-to-blue gradient — maximizes chroma variation so that
    subsampling produces a measurable difference.
    """
    image = torch.zeros(4, h, w, dtype=torch.float32, device="cpu")
    ramp = torch.linspace(0.0, 1.0, w).unsqueeze(0).expand(h, -1)
    image[0, :, :] = 1.0 - ramp  # R: 1→0
    image[2, :, :] = ramp         # B: 0→1
    image[1, :, :] = 0.3          # G: constant
    image[3, :, :] = 1.0          # A: opaque
    return image


class TestChromaSubsampleShape:
    """Output shape is always identical to input (in-place mutation)."""

    def test_analog_preserves_shape(self):
        pp = _make_postprocessor()
        image = _make_colorful_image()
        original_shape = image.shape
        pp.chroma_subsample(image, mode="analog", sigma=2.0)
        assert image.shape == original_shape

    def test_digital_420_preserves_shape(self):
        pp = _make_postprocessor()
        image = _make_colorful_image()
        original_shape = image.shape
        pp.chroma_subsample(image, mode="digital", subsampling="4:2:0", upscale="nearest")
        assert image.shape == original_shape

    def test_digital_422_preserves_shape(self):
        pp = _make_postprocessor()
        image = _make_colorful_image()
        original_shape = image.shape
        pp.chroma_subsample(image, mode="digital", subsampling="4:2:2", upscale="bilinear")
        assert image.shape == original_shape

    def test_digital_double_size_preserves_shape(self):
        pp = _make_postprocessor()
        image = _make_colorful_image()
        original_shape = image.shape
        pp.chroma_subsample(image, mode="digital", subsampling="4:2:0",
                            upscale="nearest", double_size=True)
        assert image.shape == original_shape

    def test_odd_dimensions(self):
        """Odd image sizes don't crash or change shape."""
        pp = _make_postprocessor()
        image = _make_colorful_image(h=63, w=127)
        original_shape = image.shape
        pp.chroma_subsample(image, mode="digital", subsampling="4:2:0", upscale="nearest")
        assert image.shape == original_shape


class TestChromaSubsampleAlpha:
    """Alpha channel must pass through untouched."""

    def test_analog_preserves_alpha(self):
        pp = _make_postprocessor()
        image = _make_colorful_image()
        alpha_before = image[3].clone()
        pp.chroma_subsample(image, mode="analog", sigma=3.0)
        assert torch.equal(image[3], alpha_before)

    def test_digital_preserves_alpha(self):
        pp = _make_postprocessor()
        image = _make_colorful_image()
        alpha_before = image[3].clone()
        pp.chroma_subsample(image, mode="digital", subsampling="4:2:0", upscale="nearest")
        assert torch.equal(image[3], alpha_before)

    def test_varying_alpha_preserved(self):
        """Non-trivial alpha (gradient) survives the filter."""
        pp = _make_postprocessor()
        image = _make_colorful_image()
        image[3, :, :] = torch.linspace(0.0, 1.0, image.shape[2]).unsqueeze(0)
        alpha_before = image[3].clone()
        pp.chroma_subsample(image, mode="digital", subsampling="4:2:0",
                            upscale="bilinear", double_size=True)
        assert torch.equal(image[3], alpha_before)


class TestChromaSubsampleLuma:
    """Luminance (Y) should be preserved — only chroma is affected."""

    def test_analog_preserves_luma(self):
        pp = _make_postprocessor()
        image = _make_colorful_image()
        luma_before = rgb_to_yuv(image[:3])[0].clone()
        pp.chroma_subsample(image, mode="analog", sigma=3.0)
        luma_after = rgb_to_yuv(image[:3])[0]
        # Not bitwise equal (YUV round-trip + clamping), but very close
        assert torch.allclose(luma_after, luma_before, atol=1e-5)

    def test_digital_preserves_luma(self):
        pp = _make_postprocessor()
        image = _make_colorful_image()
        luma_before = rgb_to_yuv(image[:3])[0].clone()
        pp.chroma_subsample(image, mode="digital", subsampling="4:2:0", upscale="nearest")
        luma_after = rgb_to_yuv(image[:3])[0]
        # Slightly looser tolerance than analog — the RGB clamping in yuv_to_rgb
        # can shift luma when modified chroma pushes RGB channels out of [0, 1].
        assert torch.allclose(luma_after, luma_before, atol=1e-3)


class TestChromaSubsampleEffect:
    """The filter must actually change the image (on colorful input)."""

    def test_analog_modifies_image(self):
        pp = _make_postprocessor()
        image = _make_colorful_image()
        original = image.clone()
        pp.chroma_subsample(image, mode="analog", sigma=3.0)
        assert not torch.equal(image, original)

    def test_digital_420_nearest_modifies_image(self):
        pp = _make_postprocessor()
        image = _make_colorful_image()
        original = image.clone()
        pp.chroma_subsample(image, mode="digital", subsampling="4:2:0", upscale="nearest")
        assert not torch.equal(image, original)

    def test_digital_422_nearest_modifies_image(self):
        pp = _make_postprocessor()
        image = _make_colorful_image()
        original = image.clone()
        pp.chroma_subsample(image, mode="digital", subsampling="4:2:2", upscale="nearest")
        assert not torch.equal(image, original)

    def test_digital_420_bilinear_modifies_image(self):
        pp = _make_postprocessor()
        image = _make_colorful_image()
        original = image.clone()
        pp.chroma_subsample(image, mode="digital", subsampling="4:2:0", upscale="bilinear")
        assert not torch.equal(image, original)

    def test_double_size_stronger_than_normal(self):
        """double_size should produce a larger deviation from the original."""
        pp = _make_postprocessor()
        image1 = _make_colorful_image()
        image2 = image1.clone()
        pp.chroma_subsample(image1, mode="digital", subsampling="4:2:0",
                            upscale="nearest", double_size=False)
        pp.chroma_subsample(image2, mode="digital", subsampling="4:2:0",
                            upscale="nearest", double_size=True)
        original = _make_colorful_image()
        diff_normal = (image1 - original).abs().mean()
        diff_double = (image2 - original).abs().mean()
        assert diff_double > diff_normal


class TestChromaSubsampleDigitalBlockStructure:
    """Digital nearest-upsample should produce block structure in chroma."""

    def test_420_nearest_has_2x2_chroma_blocks(self):
        """4:2:0 nearest: adjacent 2×2 pixel blocks share the same chroma.

        Checked with allclose — RGB clamping in yuv_to_rgb introduces tiny
        differences (~0.0006) when the modified chroma pushes some RGB
        channels out of [0, 1].
        """
        pp = _make_postprocessor()
        image = _make_colorful_image(h=64, w=128)
        pp.chroma_subsample(image, mode="digital", subsampling="4:2:0",
                            upscale="nearest", double_size=False)
        yuv = rgb_to_yuv(image[:3])
        chroma = yuv[1:3]  # [2, h, w]
        # Vertically adjacent pairs share chroma
        assert torch.allclose(chroma[:, 0::2, :], chroma[:, 1::2, :], atol=1e-3)
        # Horizontally adjacent pairs share chroma
        assert torch.allclose(chroma[:, :, 0::2], chroma[:, :, 1::2], atol=1e-3)

    def test_422_nearest_has_1x2_chroma_blocks(self):
        """4:2:2 nearest: horizontally adjacent pairs share chroma, vertical differs."""
        pp = _make_postprocessor()
        image = _make_colorful_image(h=64, w=128)
        pp.chroma_subsample(image, mode="digital", subsampling="4:2:2",
                            upscale="nearest", double_size=False)
        yuv = rgb_to_yuv(image[:3])
        chroma = yuv[1:3]
        # Horizontally adjacent pairs share chroma
        assert torch.allclose(chroma[:, :, 0::2], chroma[:, :, 1::2], atol=1e-3)


class TestChromaSubsampleErrors:
    """Bad parameter values raise ValueError."""

    def test_unknown_mode(self):
        pp = _make_postprocessor()
        image = _make_colorful_image()
        try:
            pp.chroma_subsample(image, mode="SECAM")
            assert False, "Expected ValueError"
        except ValueError:
            pass

    def test_unknown_subsampling(self):
        pp = _make_postprocessor()
        image = _make_colorful_image()
        try:
            pp.chroma_subsample(image, mode="digital", subsampling="4:4:4")
            assert False, "Expected ValueError"
        except ValueError:
            pass

    def test_unknown_upscale(self):
        pp = _make_postprocessor()
        image = _make_colorful_image()
        try:
            pp.chroma_subsample(image, mode="digital", upscale="cubic")
            assert False, "Expected ValueError"
        except ValueError:
            pass

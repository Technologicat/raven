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

def _make_postprocessor(h=64, w=128):
    """Create a Postprocessor with an empty chain for testing.

    Sets up meshgrids and frame state as if `render_into` had been called once,
    so individual filters can be invoked directly.
    """
    pp = Postprocessor("cpu", torch.float32, chain=[])
    pp._setup_meshgrid(h, w)
    pp.frame_no = 0.0
    pp.last_frame_no = -1.0
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


# ---------------------------------------------------------------------------
# Tests: bloom filter
# ---------------------------------------------------------------------------

class TestBloomShape:
    def test_preserves_shape(self):
        pp = _make_postprocessor()
        image = _make_image()
        original_shape = image.shape
        pp.bloom(image)
        assert image.shape == original_shape

    def test_output_range(self):
        """Output must be clamped to [0, 1]."""
        pp = _make_postprocessor()
        image = _make_image()
        pp.bloom(image)
        assert image.min() >= 0.0
        assert image.max() <= 1.0


class TestBloomEffect:
    def test_modifies_image(self):
        pp = _make_postprocessor()
        image = _make_image()
        original = image.clone()
        pp.bloom(image)
        assert not torch.equal(image, original)

    def test_threshold_one_is_exposure_only(self):
        """threshold=1.0 disables bloom glow; only tonemapping/exposure remains."""
        pp = _make_postprocessor()
        image1 = _make_image()
        image2 = image1.clone()
        # With threshold=1.0, no pixels glow — bloom branch is skipped.
        pp.bloom(image1, threshold=1.0, exposure=1.0)
        # The filter should still modify the image (tonemapping).
        assert not torch.equal(image1, image2)

    def test_higher_exposure_brighter(self):
        """Higher exposure should produce a brighter image overall."""
        pp = _make_postprocessor()
        image_lo = _make_image()
        image_hi = image_lo.clone()
        pp.bloom(image_lo, threshold=1.0, exposure=0.5)
        pp.bloom(image_hi, threshold=1.0, exposure=2.0)
        assert image_hi[:3].mean() > image_lo[:3].mean()

    def test_alpha_max_combined(self):
        """Alpha should be max-combined with the bloom, not just passed through."""
        pp = _make_postprocessor()
        image = _make_image()
        # Make alpha partially transparent
        image[3, :, :] = 0.5
        # Make some pixels very bright so bloom kicks in
        image[:3, :10, :10] = 1.0
        alpha_before = image[3].clone()
        pp.bloom(image, threshold=0.3, exposure=1.0)
        # Alpha in the bright region should be >= what it was (max-combine)
        assert (image[3] >= alpha_before - 1e-6).all()


# ---------------------------------------------------------------------------
# Tests: chromatic_aberration filter
# ---------------------------------------------------------------------------

class TestChromaticAberrationShape:
    def test_preserves_shape(self):
        pp = _make_postprocessor()
        image = _make_image()
        original_shape = image.shape
        pp.chromatic_aberration(image)
        assert image.shape == original_shape


class TestChromaticAberrationEffect:
    def test_modifies_image(self):
        pp = _make_postprocessor()
        image = _make_colorful_image()
        original = image.clone()
        pp.chromatic_aberration(image, scale=0.01, sigma=1.0)
        assert not torch.equal(image, original)

    def test_green_channel_passthrough(self):
        """G channel is the lens reference wavelength — passed through unwarped.

        Note: G is still affected by transverse CA (blur), so we test with sigma=0
        to isolate the axial (geometric) component. With sigma=0, blur kernel size
        is 3 (the torchvision minimum), which causes slight blurring at edges. So
        we check the interior only.
        """
        pp = _make_postprocessor()
        image = _make_colorful_image()
        green_before = image[1].clone()
        pp.chromatic_aberration(image, scale=0.01, sigma=0.1)
        # Interior pixels should be very close (edge pixels blur from border padding)
        margin = 5
        interior = slice(margin, -margin), slice(margin, -margin)
        assert torch.allclose(image[1][interior], green_before[interior], atol=1e-2)

    def test_r_and_b_diverge(self):
        """R and B channels should be warped in opposite directions."""
        pp = _make_postprocessor()
        image = torch.ones(4, 64, 128, dtype=torch.float32)
        # Paint a centered bright patch — after CA, R and B copies will shift apart
        image[:3, :, :] = 0.2
        image[:3, 20:44, 40:88] = 1.0
        image[3, :, :] = 1.0
        original = image.clone()
        pp.chromatic_aberration(image, scale=0.02, sigma=0.1)
        r_diff = (image[0] - original[0]).abs().sum()
        b_diff = (image[2] - original[2]).abs().sum()
        # Both R and B should have changed
        assert r_diff > 0.1
        assert b_diff > 0.1


# ---------------------------------------------------------------------------
# Tests: vignetting filter
# ---------------------------------------------------------------------------

class TestVignettingShape:
    def test_preserves_shape(self):
        pp = _make_postprocessor()
        image = _make_image()
        original_shape = image.shape
        pp.vignetting(image)
        assert image.shape == original_shape


class TestVignettingAlpha:
    def test_alpha_untouched(self):
        pp = _make_postprocessor()
        image = _make_image()
        alpha_before = image[3].clone()
        pp.vignetting(image)
        assert torch.equal(image[3], alpha_before)


class TestVignettingEffect:
    def test_modifies_image(self):
        pp = _make_postprocessor()
        image = _make_image()
        original = image.clone()
        pp.vignetting(image, strength=0.3)
        assert not torch.equal(image, original)

    def test_center_brightest(self):
        """Center pixel should be the brightest after vignetting a uniform image."""
        pp = _make_postprocessor()
        image = torch.ones(4, 64, 128, dtype=torch.float32)
        pp.vignetting(image, strength=0.3)
        center_val = image[0, 32, 64]
        corner_val = image[0, 0, 0]
        assert center_val > corner_val

    def test_corners_darkest(self):
        """Corners should be darker than edge midpoints."""
        pp = _make_postprocessor()
        image = torch.ones(4, 64, 128, dtype=torch.float32)
        pp.vignetting(image, strength=0.3)
        corner_val = image[0, 0, 0]
        edge_mid_val = image[0, 32, 0]  # midpoint of left edge
        assert edge_mid_val > corner_val

    def test_only_darkens(self):
        """Vignetting is multiplicative — it can only darken, never brighten."""
        pp = _make_postprocessor()
        image = _make_image()
        original = image.clone()
        pp.vignetting(image, strength=0.3)
        assert (image[:3] <= original[:3] + 1e-6).all()

    def test_radially_symmetric(self):
        """Opposite corners should have the same brightness on a uniform image."""
        pp = _make_postprocessor(h=64, w=64)  # square for clean symmetry
        image = torch.ones(4, 64, 64, dtype=torch.float32)
        pp.vignetting(image, strength=0.3)
        assert torch.allclose(image[0, 0, 0], image[0, 0, -1], atol=1e-5)
        assert torch.allclose(image[0, 0, 0], image[0, -1, 0], atol=1e-5)
        assert torch.allclose(image[0, 0, 0], image[0, -1, -1], atol=1e-5)


# ---------------------------------------------------------------------------
# Tests: desaturate filter
# ---------------------------------------------------------------------------

class TestDesaturateAlpha:
    def test_alpha_untouched(self):
        pp = _make_postprocessor()
        image = _make_colorful_image()
        alpha_before = image[3].clone()
        pp.desaturate(image, strength=1.0)
        assert torch.equal(image[3], alpha_before)


class TestDesaturateEffect:
    def test_strength_zero_is_noop(self):
        pp = _make_postprocessor()
        image = _make_colorful_image()
        original = image.clone()
        pp.desaturate(image, strength=0.0)
        assert torch.allclose(image, original, atol=1e-6)

    def test_strength_one_produces_grayscale(self):
        """Full desaturation with white tint should produce a grayscale image."""
        pp = _make_postprocessor()
        image = _make_colorful_image()
        pp.desaturate(image, strength=1.0, tint_rgb=[1.0, 1.0, 1.0])
        # All RGB channels should be equal (grayscale)
        assert torch.allclose(image[0], image[1], atol=1e-5)
        assert torch.allclose(image[0], image[2], atol=1e-5)

    def test_partial_strength_blends(self):
        """Partial strength should produce something between original and grayscale."""
        pp = _make_postprocessor()
        image = _make_colorful_image()
        original = image.clone()
        pp.desaturate(image, strength=0.5, tint_rgb=[1.0, 1.0, 1.0])
        # Not the same as original (some desaturation happened)
        assert not torch.equal(image, original)
        # Not fully grayscale either
        assert not torch.allclose(image[0], image[1], atol=1e-3)

    def test_tint_colors_output(self):
        """A non-white tint should shift the color of the desaturated image."""
        pp = _make_postprocessor()
        image1 = _make_colorful_image()
        image2 = image1.clone()
        pp.desaturate(image1, strength=1.0, tint_rgb=[1.0, 1.0, 1.0])
        pp.desaturate(image2, strength=1.0, tint_rgb=[0.5, 1.0, 0.5])
        # Green-tinted result should differ from white-tinted
        assert not torch.equal(image1, image2)
        # Green channel should be brightest in the green-tinted version
        assert image2[1].mean() > image2[0].mean()
        assert image2[1].mean() > image2[2].mean()

    def test_hue_bandpass_preserves_reference_hue(self):
        """With hue bandpass, pixels near the reference hue should stay colorful."""
        pp = _make_postprocessor()
        # Pure red image
        image = torch.zeros(4, 64, 128, dtype=torch.float32)
        image[0, :, :] = 0.8  # strong red
        image[1, :, :] = 0.1
        image[2, :, :] = 0.1
        image[3, :, :] = 1.0
        # Bandpass centered on red — red pixels should survive
        pp.desaturate(image, strength=1.0, tint_rgb=[1.0, 1.0, 1.0],
                      bandpass_reference_rgb=[1.0, 0.0, 0.0], bandpass_q=0.5)
        # Red pixels should still have more red than green/blue
        assert image[0].mean() > image[1].mean()


# ---------------------------------------------------------------------------
# Tests: monochrome_display filter
# ---------------------------------------------------------------------------

class TestMonochromeDisplayAlpha:
    def test_alpha_untouched(self):
        pp = _make_postprocessor()
        image = _make_colorful_image()
        alpha_before = image[3].clone()
        pp.monochrome_display(image, strength=1.0)
        assert torch.equal(image[3], alpha_before)


class TestMonochromeDisplayEffect:
    def test_strength_zero_is_noop(self):
        pp = _make_postprocessor()
        image = _make_colorful_image()
        original = image.clone()
        pp.monochrome_display(image, strength=0.0)
        assert torch.allclose(image, original, atol=1e-6)

    def test_white_tint_produces_grayscale(self):
        """White tint should produce equal R=G=B (pure grayscale)."""
        pp = _make_postprocessor()
        image = _make_colorful_image()
        pp.monochrome_display(image, strength=1.0, tint_rgb=[1.0, 1.0, 1.0])
        assert torch.allclose(image[0], image[1], atol=1e-5)
        assert torch.allclose(image[0], image[2], atol=1e-5)

    def test_green_tint(self):
        """Green phosphor tint: G should be brightest, R and B attenuated."""
        pp = _make_postprocessor()
        image = _make_colorful_image()
        pp.monochrome_display(image, strength=1.0, tint_rgb=[0.5, 1.0, 0.5])
        assert image[1].mean() > image[0].mean()
        assert image[1].mean() > image[2].mean()

    def test_amber_tint(self):
        """Amber phosphor tint: R > G > B."""
        pp = _make_postprocessor()
        image = _make_colorful_image()
        pp.monochrome_display(image, strength=1.0, tint_rgb=[1.0, 0.5, 0.2])
        assert image[0].mean() > image[1].mean()
        assert image[1].mean() > image[2].mean()

    def test_tint_is_not_just_passthrough(self):
        """Tinted output should differ from untinted grayscale."""
        pp = _make_postprocessor()
        image1 = _make_colorful_image()
        image2 = image1.clone()
        pp.monochrome_display(image1, strength=1.0, tint_rgb=[1.0, 1.0, 1.0])
        pp.monochrome_display(image2, strength=1.0, tint_rgb=[0.5, 1.0, 0.5])
        assert not torch.equal(image1, image2)


# ---------------------------------------------------------------------------
# Tests: translucent_display filter
# ---------------------------------------------------------------------------

class TestTranslucentDisplay:
    def test_preserves_shape(self):
        pp = _make_postprocessor()
        image = _make_image()
        original_shape = image.shape
        pp.translucent_display(image)
        assert image.shape == original_shape

    def test_rgb_untouched(self):
        pp = _make_postprocessor()
        image = _make_image()
        rgb_before = image[:3].clone()
        pp.translucent_display(image, alpha=0.5)
        assert torch.equal(image[:3], rgb_before)

    def test_alpha_scaled(self):
        """Alpha should be multiplicatively scaled."""
        pp = _make_postprocessor()
        image = _make_image()
        image[3, :, :] = 1.0
        pp.translucent_display(image, alpha=0.7)
        assert torch.allclose(image[3], torch.full_like(image[3], 0.7), atol=1e-6)

    def test_alpha_scales_nonuniform(self):
        """Non-uniform alpha should scale proportionally."""
        pp = _make_postprocessor()
        image = _make_image()
        alpha_before = image[3].clone()
        pp.translucent_display(image, alpha=0.5)
        expected = alpha_before * 0.5
        assert torch.allclose(image[3], expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Tests: analog_lowres filter
# ---------------------------------------------------------------------------

class TestAnalogLowresShape:
    def test_preserves_shape(self):
        pp = _make_postprocessor()
        image = _make_image()
        original_shape = image.shape
        pp.analog_lowres(image)
        assert image.shape == original_shape


class TestAnalogLowresEffect:
    def test_modifies_image(self):
        pp = _make_postprocessor()
        image = _make_colorful_image()
        original = image.clone()
        pp.analog_lowres(image, sigma=2.0)
        assert not torch.equal(image, original)

    def test_blurs_alpha_too(self):
        """analog_lowres blurs all channels, including alpha."""
        pp = _make_postprocessor()
        image = _make_image()
        # Sharp alpha edge
        image[3, :, :] = 0.0
        image[3, :, 64:] = 1.0
        alpha_before = image[3].clone()
        pp.analog_lowres(image, sigma=2.0)
        # Alpha should have changed (edge blurred)
        assert not torch.equal(image[3], alpha_before)

    def test_reduces_high_frequency(self):
        """Blurring should reduce pixel-to-pixel variation."""
        pp = _make_postprocessor()
        # Checkerboard pattern — maximum high-frequency content
        image = torch.zeros(4, 64, 128, dtype=torch.float32)
        image[:3, 0::2, 0::2] = 1.0
        image[:3, 1::2, 1::2] = 1.0
        image[3, :, :] = 1.0
        # Measure variation before
        diff_before = (image[0, :, 1:] - image[0, :, :-1]).abs().mean()
        pp.analog_lowres(image, sigma=2.0)
        diff_after = (image[0, :, 1:] - image[0, :, :-1]).abs().mean()
        assert diff_after < diff_before


# ---------------------------------------------------------------------------
# Tests: scanlines filter
# ---------------------------------------------------------------------------

class TestScanlinesShape:
    def test_preserves_shape(self):
        pp = _make_postprocessor()
        image = _make_image()
        original_shape = image.shape
        pp.scanlines(image)
        assert image.shape == original_shape


class TestScanlinesEffect:
    def test_modifies_image(self):
        pp = _make_postprocessor()
        image = _make_image()
        original = image.clone()
        pp.scanlines(image, strength=0.3)
        assert not torch.equal(image, original)

    def test_alpha_mode_preserves_rgb(self):
        """In alpha mode, RGB channels should be untouched."""
        pp = _make_postprocessor()
        image = _make_image()
        rgb_before = image[:3].clone()
        pp.scanlines(image, channel="A", strength=0.3)
        assert torch.equal(image[:3], rgb_before)

    def test_alpha_mode_modifies_alpha(self):
        pp = _make_postprocessor()
        image = _make_image()
        alpha_before = image[3].clone()
        pp.scanlines(image, channel="A", strength=0.3)
        assert not torch.equal(image[3], alpha_before)

    def test_luma_mode_preserves_alpha(self):
        """In Y mode, alpha should be untouched."""
        pp = _make_postprocessor()
        image = _make_image()
        alpha_before = image[3].clone()
        pp.scanlines(image, channel="Y", strength=0.3)
        assert torch.equal(image[3], alpha_before)

    def test_alternating_lines_dimmed(self):
        """Every other line should be dimmer than its neighbor."""
        pp = _make_postprocessor()
        image = torch.ones(4, 64, 128, dtype=torch.float32)
        pp.scanlines(image, channel="A", strength=0.3, dynamic=False,
                     field=0, double_size=False)
        # Even lines (field=0) should be dimmed
        even_alpha = image[3, 0::2, :].mean()
        odd_alpha = image[3, 1::2, :].mean()
        assert even_alpha < odd_alpha

    def test_double_size_dims_pairs(self):
        """With double_size, two adjacent lines should be dimmed together."""
        pp = _make_postprocessor()
        image = torch.ones(4, 64, 128, dtype=torch.float32)
        pp.scanlines(image, channel="A", strength=0.3, dynamic=False,
                     field=0, double_size=True)
        # Lines 0,1 should be dimmed equally (both in the first double-line)
        assert torch.allclose(image[3, 0, :], image[3, 1, :], atol=1e-6)
        # Lines 2,3 should be bright (next double-line, undimmed)
        assert image[3, 2, :].mean() > image[3, 0, :].mean()

    def test_only_darkens(self):
        """Scanlines should only darken, never brighten."""
        pp = _make_postprocessor()
        image = _make_image()
        original = image.clone()
        pp.scanlines(image, channel="A", strength=0.3)
        assert (image[3] <= original[3] + 1e-6).all()


# ---------------------------------------------------------------------------
# Tests: zoom filter (low quality)
# ---------------------------------------------------------------------------

class TestZoomShape:
    def test_preserves_shape(self):
        pp = _make_postprocessor()
        image = _make_image()
        original_shape = image.shape
        pp.zoom(image, factor=2.0, quality="low")
        assert image.shape == original_shape


class TestZoomEffect:
    def test_factor_one_is_noop(self):
        """factor=1.0 should be an identity operation."""
        pp = _make_postprocessor()
        image = _make_image()
        original = image.clone()
        pp.zoom(image, factor=1.0)
        assert torch.equal(image, original)

    def test_modifies_image(self):
        pp = _make_postprocessor()
        image = _make_colorful_image()
        original = image.clone()
        pp.zoom(image, factor=2.0, quality="low")
        assert not torch.equal(image, original)

    def test_zoom_in_magnifies_center(self):
        """Zooming in should spread the center region across the whole image."""
        pp = _make_postprocessor()
        # Bright center dot on dark background
        image = torch.zeros(4, 64, 128, dtype=torch.float32)
        image[:3, 30:34, 62:66] = 1.0
        image[3, :, :] = 1.0
        pp.zoom(image, factor=2.0, center_x=0.0, center_y=0.0, quality="low")
        # The bright region should now be larger
        bright_count = (image[0] > 0.5).sum().item()
        assert bright_count > 4 * 4  # original was 4×4 pixels


# ---------------------------------------------------------------------------
# Tests: crt filter
# ---------------------------------------------------------------------------

def _crt_off():
    """Settings that switch every part of `crt` off, as a base for turning exactly one back on."""
    return dict(warp_x=0.0, warp_y=0.0, overscan=1.0,
                scanline_strength=0.0, mask_type="none", corner_falloff=0.0,
                beam_bleed=0.0, glow_strength=0.0, persistence_tau=0.0)


def _light(image):
    """What a viewer sees: the emitted light, `rgb * alpha`.

    The quantity to assert on for anything about brightness, because `crt` divides its modulation
    between the colour channels and alpha - which of the two carries the scanlines is what `alpha_mode`
    selects. A test that read the colour channels alone would report the default configuration as
    having no scanlines at all.
    """
    return image[:3] * image[3:4]


def _flat_grey(value=0.1, h=64, w=128):
    """An opaque flat grey, for measuring what the filter does to a level rather than to a picture."""
    image = torch.full((4, h, w), value)
    image[3] = 1.0
    return image


class TestCrtContract:
    def test_preserves_shape_dtype_and_device(self):
        pp = _make_postprocessor()
        image = _make_image()
        pp.crt(image)
        assert image.shape == (4, 64, 128)
        assert image.dtype == torch.float32
        assert image.device == torch.device("cpu")

    def test_mutates_in_place(self):
        pp = _make_postprocessor()
        image = _make_image()
        original = image.clone()
        pp.crt(image)
        assert not torch.equal(image, original)

    def test_output_stays_in_unit_range(self):
        """Brightness compensation drives the beam above 1.0 on purpose, so the clamp is load-bearing."""
        pp = _make_postprocessor()
        image = _make_image()
        pp.crt(image, brightness_compensation=1.0)
        assert image.min() >= 0.0
        assert image.max() <= 1.0

    def test_everything_off_is_bitwise_identity(self):
        """Not `allclose`: this is what exercises the skip of the warp resample, and a tolerance-based
        comparison would accept a stray bilinear pass that softens the raster this filter exists to draw."""
        pp = _make_postprocessor()
        image = _make_image()
        original = image.clone()
        pp.crt(image, **_crt_off())
        assert torch.equal(image, original)


class TestCrtBrightnessCompensation:
    """Scanlines and mask are both multiplicative and both below 1, so they darken the picture. Full
    compensation is supposed to hand back the original mean brightness."""

    def test_mean_level_is_preserved_across_a_parameter_sweep(self):
        # Dim enough that the compensated peaks stay under 1.0; the clamp would otherwise eat the
        # brightness this test is measuring, and the failure would look like a bad constant.
        for mask_type in ("aperture_grille", "slot", "shadow", "none"):
            for mask_pitch in (2, 3, 6):
                for scanline_period, scanline_weight in ((2, 2.0), (3, 4.0), (1, 0.5)):
                    pp = _make_postprocessor()
                    image = _flat_grey(0.1)
                    pp.crt(image, brightness_compensation=1.0,
                           mask_type=mask_type, mask_pitch=mask_pitch, mask_strength=0.35,
                           scanline_period=scanline_period, scanline_weight=scanline_weight,
                           scanline_strength=0.6, corner_falloff=0.0, beam_bleed=0.0,
                           glow_strength=0.0, persistence_tau=0.0)
                    mean = float(_light(image).mean())
                    # Tight on purpose. The compensation constant is the exact mean of the modulation,
                    # not an approximation of it, and the cheap approximation that treats the scanline
                    # and mask terms as independent is off by several percent for the staggered masks -
                    # their vertical structure lines up with the scanlines whenever the two pitches
                    # share a factor. A percent-level tolerance here would accept that silently.
                    assert abs(mean - 0.1) < 1e-5, (f"{mask_type}, pitch {mask_pitch}, period "
                                                    f"{scanline_period}: mean {mean:.5f}, wanted 0.1")

    def test_without_compensation_the_picture_is_darker(self):
        """The negative control for the test above: if this filter happened to darken nothing, full
        compensation would preserve the mean for a reason that has nothing to do with the constant."""
        pp = _make_postprocessor()
        image = _flat_grey(0.1)
        pp.crt(image, brightness_compensation=0.0, corner_falloff=0.0, beam_bleed=0.0,
               glow_strength=0.0, persistence_tau=0.0)
        assert float(_light(image).mean()) < 0.06


class TestCrtMask:
    """The mask is in output pixel space, and carries chroma structure only."""

    @staticmethod
    def _column_weights(pp, mask_type="aperture_grille", mask_pitch=3, w=128):
        """Per-column channel weights, read off a flat grey image with everything else switched off."""
        settings = _crt_off()
        settings.update(mask_type=mask_type, mask_pitch=mask_pitch, mask_strength=0.35,
                        brightness_compensation=0.0)
        image = torch.full((4, 64, w), 0.5)
        pp.crt(image, **settings)
        return image[:3] / 0.5  # [3, h, w]

    def test_aperture_grille_is_periodic_in_the_pitch(self):
        pp = _make_postprocessor()
        weights = self._column_weights(pp, mask_pitch=3)
        assert torch.allclose(weights[:, :, :-3], weights[:, :, 3:], atol=1e-6)

    def test_aperture_grille_is_not_periodic_in_less_than_the_pitch(self):
        """The negative control. A mask that dimmed every column equally would satisfy the test above
        for every shift, and this fixture could not tell that apart from a working stripe pattern."""
        pp = _make_postprocessor()
        weights = self._column_weights(pp, mask_pitch=3)
        assert not torch.allclose(weights[:, :, :-1], weights[:, :, 1:], atol=1e-6)

    def test_each_column_lights_exactly_one_channel(self):
        pp = _make_postprocessor()
        weights = self._column_weights(pp, mask_pitch=3)
        lit = (weights[:, 0, :] > 0.99).sum(dim=0)  # [w], channels at full strength in each column
        assert torch.equal(lit, torch.ones(128))

    def test_the_pitch_is_in_output_pixels_not_normalized_coordinates(self):
        """Rendering the same content wider must not stretch the mask with it. Asserted so that nobody
        later 'fixes' the pitch into normalized coordinates, which would make it resolution-independent
        and therefore invisible at 4K - the same problem `scanlines` solved with `double_size`."""
        narrow = self._column_weights(_make_postprocessor(64, 128), mask_pitch=3, w=128)
        wide = self._column_weights(_make_postprocessor(64, 256), mask_pitch=3, w=256)
        assert torch.allclose(narrow[:, 0, :12], wide[:, 0, :12], atol=1e-6)


class TestCrtScanlines:
    @staticmethod
    def _row_profile(pp, **overrides):
        """Per-row mean emitted light, on a flat grey image with everything but the scanlines off."""
        settings = _crt_off()
        settings.update(scanline_strength=0.8, scanline_weight=3.0, scanline_period=2,
                        brightness_compensation=0.0)
        settings.update(overrides)
        image = _flat_grey(0.5)
        pp.crt(image, **settings)
        return _light(image)[0].mean(dim=1)  # [h]

    def test_alternate_rows_are_dimmed(self):
        pp = _make_postprocessor()
        profile = self._row_profile(pp, dynamic_field=False, field=0)
        assert profile[0::2].mean() < profile[1::2].mean()

    def test_the_field_alternates_between_consecutive_frames(self):
        """Which set of rows is bright must swap, or `dynamic_field` is doing nothing."""
        pp = _make_postprocessor()
        pp.frame_no = 0.0
        even_first = self._row_profile(pp, dynamic_field=True)
        pp.frame_no = 1.0
        even_second = self._row_profile(pp, dynamic_field=True)
        first_bias = float(even_first[0::2].mean() - even_first[1::2].mean())
        second_bias = float(even_second[0::2].mean() - even_second[1::2].mean())
        assert first_bias * second_bias < 0, (f"the two frames have the same field: biases {first_bias:.4f} "
                                              f"and {second_bias:.4f}")

    def test_output_depends_on_the_value_of_frame_no_not_on_the_call_history(self):
        """`frame_no` is wall-clock time in disguise, so a filter that counted its own calls instead would
        drift out of step with everything else in the chain the moment a frame was dropped."""
        pp = _make_postprocessor()
        pp.frame_no = 4.0
        straight_there = self._row_profile(pp, dynamic_field=True)
        for intermediate in (1.0, 2.0, 3.0):
            pp.frame_no = intermediate
            self._row_profile(pp, dynamic_field=True)
        pp.frame_no = 4.0
        by_way_of_others = self._row_profile(pp, dynamic_field=True)
        assert torch.equal(straight_there, by_way_of_others)


class TestCrtAlpha:
    """Where the beam is not writing, a hologram emits no light, so the gaps between scanlines are
    transparent rather than dark. Alpha follows the scanline term and nothing else."""

    def test_luma_mode_leaves_alpha_untouched(self):
        pp = _make_postprocessor()
        image = _make_image()
        alpha_before = image[3].clone()
        pp.crt(image, alpha_mode="luma", beam_bleed=0.0, glow_strength=0.0, persistence_tau=0.0)
        assert torch.equal(image[3], alpha_before)

    def test_both_mode_modulates_alpha(self):
        pp = _make_postprocessor()
        image = _make_image()
        alpha_before = image[3].clone()
        pp.crt(image, alpha_mode="both", beam_bleed=0.0, glow_strength=0.0, persistence_tau=0.0)
        assert not torch.equal(image[3], alpha_before)

    def test_the_two_modes_emit_the_same_light(self):
        """`alpha_mode` picks where the scanlines land, not how many times they are applied.

        What a viewer sees is `rgb * alpha`, so a filter that put the scanline term into both channels
        would square it there - and the brightness compensation, which corrects for one factor of it,
        would leave the character washed out and half-transparent instead of rastered. That is what this
        filter did on its first run, and the still it produced is the only reason anyone noticed.
        """
        common = dict(mask_type="aperture_grille", mask_pitch=3, mask_strength=0.35,
                      scanline_strength=0.6, corner_falloff=0.1, brightness_compensation=0.0,
                      beam_bleed=0.0, glow_strength=0.0, persistence_tau=0.0)
        source = torch.rand(4, 64, 128) * 0.5  # dim enough that nothing meets the clamp

        luma = source.clone()
        _make_postprocessor().crt(luma, alpha_mode="luma", **common)
        both = source.clone()
        _make_postprocessor().crt(both, alpha_mode="both", **common)

        assert torch.allclose(luma[:3] * luma[3:4], both[:3] * both[3:4], atol=1e-6)
        # The negative control: the two modes must still be telling the fixture apart somewhere, or the
        # assertion above holds for the uninteresting reason that nothing happened in either of them.
        assert not torch.allclose(luma[3], both[3], atol=1e-3), "neither mode touched alpha"

    def test_alpha_carries_no_mask_pitch_structure(self):
        """The screen-door regression. Dimming one channel's emitters does not make that patch of the
        picture transparent, so alpha must not vary along a row."""
        pp = _make_postprocessor()
        image = torch.full((4, 64, 128), 0.5)
        pp.crt(image, alpha_mode="both", mask_type="aperture_grille", mask_pitch=3, mask_strength=0.9,
               beam_bleed=0.0, glow_strength=0.0, persistence_tau=0.0, corner_falloff=0.0)
        row_variation = (image[3].max(dim=1).values - image[3].min(dim=1).values).max()
        assert float(row_variation) < 1e-6, "alpha varies along a row, so the mask is punching holes in it"

    def test_a_transparent_pixel_never_becomes_opaque(self):
        """An RGBA frame can always be composited onto a backdrop later; writing an opaque background is
        irrecoverable, and would destroy both the client-side backdrop and any alpha work done upstream."""
        pp = _make_postprocessor()
        image = _make_image()
        image[3, :, :] = 0.0
        pp.crt(image, alpha_mode="both", glow_strength=0.5, beam_bleed=1.0)
        assert float(image[3].max()) == 0.0


class TestCrtPhosphorPersistence:
    """The accumulator is the one piece of state in this filter, and it is the reason the rest of the
    suite's determinism assumptions need pinning down."""

    def test_off_allocates_nothing(self):
        pp = _make_postprocessor()
        pp.crt(_make_image(), persistence_tau=0.0)
        assert "crt0" not in pp.crt_phosphor

    def test_off_is_repeatable_at_a_fixed_frame(self):
        pp = _make_postprocessor()
        source = _make_image()
        first = source.clone()
        pp.crt(first, persistence_tau=0.0)
        second = source.clone()
        pp.crt(second, persistence_tau=0.0)
        assert torch.equal(first, second)

    def test_on_is_not_repeatable_at_a_fixed_frame(self):
        """The negative control for the test above: with the accumulator engaged the second call sees the
        first one's trail, so a fixture where both runs agreed would prove nothing about statelessness."""
        pp = _make_postprocessor()
        source = torch.rand(4, 64, 128)
        first = source.clone()
        pp.crt(first, persistence_tau=0.2)
        pp.last_frame_no, pp.frame_no = 0.0, 1.0
        second = source.clone()
        pp.crt(second, persistence_tau=0.2)
        assert not torch.equal(first, second)

    def test_decay_is_per_second_not_per_frame(self):
        """Two steps of one frame must leave the same trail as one step of two. A per-frame decay constant
        would pass every other test here and then shorten the trail whenever the frame rate rose."""
        def trail(steps):
            pp = _make_postprocessor()
            image = torch.full((4, 64, 128), 0.8)
            pp.crt(image, persistence_tau=0.2)  # seeds the accumulator
            previous = 0.0
            for step in steps:
                pp.last_frame_no, pp.frame_no = previous, step
                pp.crt(torch.zeros(4, 64, 128), persistence_tau=0.2)
                previous = step
            return pp.crt_phosphor["crt0"]["acc"]
        # Not bitwise: exp(-a) * exp(-a) and exp(-2a) are the same number and different floats.
        assert torch.allclose(trail([1.0, 2.0]), trail([2.0]), atol=1e-6)

    def test_the_accumulator_is_dropped_when_the_frame_size_changes(self):
        """The crop can change mid-stream. A stale accumulator would be silently wrong rather than loud."""
        pp = _make_postprocessor(64, 128)
        pp.crt(torch.full((4, 64, 128), 0.9), persistence_tau=0.2)

        pp._setup_meshgrid(32, 64)
        pp.last_frame_no, pp.frame_no = 0.0, 1.0
        resized = torch.full((4, 32, 64), 0.1)
        pp.crt(resized, persistence_tau=0.2)
        assert pp.crt_phosphor["crt0"]["shape"] == torch.Size([4, 32, 64])

        stateless = torch.full((4, 32, 64), 0.1)
        pp2 = _make_postprocessor(32, 64)
        pp2.last_frame_no, pp2.frame_no = 0.0, 1.0
        pp2.crt(stateless, persistence_tau=0.0)
        assert torch.equal(resized, stateless), "the bright first frame bled into the resized one"


# ---------------------------------------------------------------------------
# Tests: the chain's own `enabled` parameter
# ---------------------------------------------------------------------------

class TestChainEnabledFlag:
    """`enabled` belongs to the chain rather than to any filter.

    It lets a filter be switched off while its settings stay in the chain, so switching it back on does not
    mean tuning it again. No filter declares such a parameter, so the engine has to consume it: one that
    leaked through would raise `TypeError` on every frame.
    """

    @staticmethod
    def _spy_on(pp, filter_name):
        """Replace one filter with a recorder, and return the list its keyword arguments land in.

        An instance attribute shadows the class's method, which is what the chain's `getattr` finds.
        """
        calls = []
        setattr(pp, filter_name, lambda image, **kwargs: calls.append(kwargs))
        return calls

    def test_a_chain_that_never_heard_of_the_key_is_unaffected(self):
        """The overwhelmingly common case, and the one a regression here would break everywhere at once."""
        pp = _make_postprocessor()
        calls = self._spy_on(pp, "zoom")
        pp.chain = [["zoom", {"factor": 2.0}]]
        pp.render_into(_make_image())
        assert calls == [{"factor": 2.0}]

    def test_an_enabled_filter_runs_and_is_not_handed_the_key(self):
        pp = _make_postprocessor()
        calls = self._spy_on(pp, "zoom")
        pp.chain = [["zoom", {"factor": 2.0, "enabled": True}]]
        pp.render_into(_make_image())
        assert calls == [{"factor": 2.0}], "`enabled` reached the filter, which declares no such parameter"

    def test_a_disabled_filter_is_skipped(self):
        pp = _make_postprocessor()
        calls = self._spy_on(pp, "zoom")
        pp.chain = [["zoom", {"factor": 2.0, "enabled": False}]]
        pp.render_into(_make_image())
        assert calls == []

    def test_the_chain_entry_is_left_as_it_was(self):
        """A chain is typically read from a settings file once and then used for every frame thereafter, so
        stripping the key in place would switch the filter on permanently after the first frame."""
        pp = _make_postprocessor()
        self._spy_on(pp, "zoom")
        chain = [["zoom", {"factor": 2.0, "enabled": True}]]
        pp.chain = chain
        pp.render_into(_make_image())
        assert chain == [["zoom", {"factor": 2.0, "enabled": True}]], f"the chain became {chain}"

    def test_a_disabled_filter_leaves_the_image_untouched(self):
        """The same thing again through the real filter, which is what catches a leaked `enabled`: the spy
        above accepts any keyword, and `zoom` does not."""
        pp = _make_postprocessor()
        image = _make_colorful_image()
        original = image.clone()

        pp.chain = [["zoom", {"factor": 2.0, "quality": "low", "enabled": False}]]
        pp.render_into(image)
        assert torch.equal(image, original), "a disabled filter changed the image"

        # The negative control. Without it, a fixture in which `zoom` happened to be a no-op — the wrong
        # factor, an image it cannot magnify — would satisfy the assertion above for the wrong reason, and
        # go on satisfying it forever.
        pp.chain = [["zoom", {"factor": 2.0, "quality": "low", "enabled": True}]]
        pp.render_into(image)
        assert not torch.equal(image, original), ("this filter does nothing to this image either way, so "
                                                  "the fixture cannot tell a skipped filter from a "
                                                  "running one")

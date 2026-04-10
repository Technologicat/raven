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

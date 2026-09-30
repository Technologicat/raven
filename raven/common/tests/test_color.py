"""Unit tests for raven.common.color."""

import colorsys
import subprocess
import sys

import pytest

from raven.common import color


class TestHexToRgb:
    def test_black(self):
        assert color.hex_to_rgb("#000000") == (0, 0, 0)

    def test_white(self):
        assert color.hex_to_rgb("#ffffff") == (255, 255, 255)

    def test_red(self):
        assert color.hex_to_rgb("#ff0000") == (255, 0, 0)

    def test_with_alpha(self):
        assert color.hex_to_rgb("#ff000080") == (255, 0, 0, 128)

    def test_uppercase(self):
        assert color.hex_to_rgb("#FF8800") == (255, 136, 0)

    def test_without_hash(self):
        assert color.hex_to_rgb("ff0000") == (255, 0, 0)

    def test_short_forms_double_each_digit(self):
        assert color.hex_to_rgb("#f80") == (255, 136, 0)
        assert color.hex_to_rgb("#f808") == (255, 136, 0, 136)

    @pytest.mark.parametrize("bad", ["#ff000", "#fffffff", "#gg0000", "red", ""])
    def test_anything_else_is_refused(self, bad):
        with pytest.raises(ValueError):
            color.hex_to_rgb(bad)


class TestParse:
    def test_hex(self):
        assert color.parse("#ff000080") == (255, 0, 0, 128)

    def test_an_all_digit_hex_is_a_colour_not_a_number(self):
        """Recognized by its shape first: evaluated as a literal, this would be the integer 123456."""
        assert color.parse("123456") == (0x12, 0x34, 0x56)

    def test_literal(self):
        assert color.parse("(255, 0, 0)") == (255, 0, 0)
        assert color.parse(" [10, 20, 30, 40] ") == (10, 20, 30, 40)

    def test_sequence_as_given(self):
        assert color.parse([0.5, 0.25, 1.0]) == (0.5, 0.25, 1.0)

    @pytest.mark.parametrize("bad", ["red", "(1, 2)", "(1, 2, 3, 4, 5)", "42", [1, 2], "(1, 2"])
    def test_anything_else_is_refused(self, bad):
        with pytest.raises(ValueError):
            color.parse(bad)


class TestMix:
    def test_endpoints_and_middle(self):
        a, b = (0.0, 0.0, 0.0, 1.0), (1.0, 0.5, 0.0, 0.0)
        assert color.mix(a, b, 0.0) == a
        assert color.mix(a, b, 1.0) == b
        assert color.mix(a, b, 0.5) == (0.5, 0.25, 0.0, 0.5)

    def test_stops_at_the_shorter_colour(self):
        assert color.mix((0, 0, 0), (100, 100, 100, 255), 0.5) == (50.0, 50.0, 50.0)


class TestLuminanceAndContrast:
    def test_srgb_to_linear_meets_itself_at_the_cutoff(self):
        below = color.SRGB_ENCODED_CUTOFF / color.SRGB_LINEAR_SLOPE
        above = ((color.SRGB_ENCODED_CUTOFF + color.SRGB_ALPHA) / (1.0 + color.SRGB_ALPHA)) ** color.SRGB_GAMMA
        assert abs(below - above) < 1e-6

    @pytest.mark.parametrize("value", [0.0, 0.002, 0.0031308, 0.01, 0.2, 0.5, 1.0])  # both sides of the joint
    def test_linear_to_srgb_undoes_srgb_to_linear(self, value):
        assert color.srgb_to_linear(color.linear_to_srgb(value)) == pytest.approx(value, abs=1e-9)

    def test_relative_luminance_ends(self):
        assert color.relative_luminance((0.0, 0.0, 0.0)) == 0.0
        assert color.relative_luminance((1.0, 1.0, 1.0, 0.5)) == pytest.approx(1.0)

    def test_luma_is_not_linearized(self):
        """Mid-grey: luma is the encoded value itself, relative luminance is far darker."""
        assert color.luma((0.5, 0.5, 0.5)) == pytest.approx(0.5)
        assert color.relative_luminance((0.5, 0.5, 0.5)) == pytest.approx(0.214, abs=1e-3)

    def test_contrast_ratio_range(self):
        assert color.contrast_ratio((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)) == pytest.approx(21.0)
        assert color.contrast_ratio((0.3, 0.6, 0.1), (0.3, 0.6, 0.1)) == pytest.approx(1.0)


class TestHlsAdjustments:
    def test_scale_saturation_keeps_hue_lightness_and_alpha(self):
        orange = (1.0, 0.5, 0.0, 0.7)
        pale = color.scale_saturation(orange, 0.3)
        h0, l0, s0 = colorsys.rgb_to_hls(*orange[:3])
        h1, l1, s1 = colorsys.rgb_to_hls(*pale[:3])
        assert (h1, l1) == pytest.approx((h0, l0))
        assert s1 == pytest.approx(0.3 * s0)
        assert pale[3] == 0.7

    def test_invert_lightness_maps_the_ends(self):
        assert color.invert_lightness(0.0, 0.9, 0.2) == pytest.approx(0.9)
        assert color.invert_lightness(1.0, 0.9, 0.2) == pytest.approx(0.2)

    @pytest.mark.parametrize("lightness", [0.0, 0.26, 0.5, 0.776, 1.0])
    def test_uninvert_undoes_invert(self, lightness):
        inverted = color.invert_lightness(lightness, 220 / 255, 45 / 255)
        assert color.uninvert_lightness(inverted, 220 / 255, 45 / 255) == pytest.approx(lightness)

    def test_legible_against_reaches_the_bar_and_keeps_the_hue(self):
        red, fill = (1.0, 0.0, 0.0, 1.0), (0.9, 0.5, 0.1, 1.0)
        assert color.contrast_ratio(red, fill) < 3.0, "red already reads on this fill, so this proves nothing"
        moved = color.legible_against(red, fill, min_ratio=3.0, lightness_range=(0.1, 0.9))
        assert color.contrast_ratio(moved, fill) >= 3.0
        assert colorsys.rgb_to_hls(*moved[:3])[0] == pytest.approx(colorsys.rgb_to_hls(*red[:3])[0])
        assert moved[3] == 1.0

    def test_legible_against_leaves_a_legible_colour_alone(self):
        black, white = (0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 1.0)
        assert color.legible_against(black, white, min_ratio=3.0, lightness_range=(0.1, 0.9)) == black


def test_imports_without_torch():
    """The reason this module exists apart from `raven.common.image.colorspace`: a config can use it for free."""
    code = "import sys\nimport raven.common.color\nprint('torch' in sys.modules)\n"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False", "importing raven.common.color imported Torch"

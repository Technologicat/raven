"""Operations on a single colour value: parsing, mixing, luminance, contrast, and HLS adjustments.

Standard library only, so that a config module can build its colours without importing Torch. The same
conversions on whole images, as tensors, are in `raven.common.image.colorspace`, which takes its constants
from here so that the two cannot disagree.

Two representations, and every function says which it takes:

  - **Parsed** colours (`hex_to_rgb`, `parse`) are tuples of integers in [0, 255], as DPG wants them.
  - **Everything else** works on tuples of floats in [0, 1], sRGB-encoded as a colour is written down,
    unless the function says it wants linear light.

RGB and RGBA are both accepted throughout. Alpha is a coverage fraction rather than a light intensity, so
the lightness and saturation operations carry it through untouched.

This module is licensed under the 2-clause BSD license, to facilitate integration anywhere.
"""

__all__ = ["BT709_WEIGHTS",
           "SRGB_LINEAR_SLOPE", "SRGB_ENCODED_CUTOFF", "SRGB_LINEAR_CUTOFF", "SRGB_ALPHA", "SRGB_GAMMA",

           "hex_to_rgb", "parse",

           "mix",

           "srgb_to_linear", "linear_to_srgb", "relative_luminance", "luma", "contrast_ratio",

           "scale_saturation",
           "invert_lightness", "uninvert_lightness",
           "legible_against"]

import ast
import colorsys
import re
from collections.abc import Sequence

# --------------------------------------------------------------------------------
# Constants shared with `raven.common.image.colorspace`

# ITU-R BT.709 luminance weights for R, G, B. Applied to *linear* light they give true relative luminance;
# applied to sRGB-encoded values they give luma, Y', which is the cheaper and rougher of the two.
#   https://www.itu.int/rec/R-REC-BT.709
#   https://en.wikipedia.org/wiki/Relative_luminance
#   https://en.wikipedia.org/wiki/Luma_(video)
BT709_WEIGHTS = (0.2126, 0.7152, 0.0722)

# The sRGB transfer function: linear near black and a power law above it, the joint chosen so that both
# value and slope match.
#   https://en.wikipedia.org/wiki/SRGB
#   https://www.color.org/chardata/rgb/srgb.xalter (IEC 61966-2-1)
SRGB_LINEAR_SLOPE = 12.92
SRGB_ENCODED_CUTOFF = 0.04045  # where the encoded side switches from the linear segment to the power law
# The linear-light cutoff is derived rather than quoted. The standard rounds it to 0.0031308, which puts a
# discontinuity of about 1e-8 in the curve; dividing keeps the two segments meeting exactly, and matches the
# constant every other implementation of this ends up with.
SRGB_LINEAR_CUTOFF = SRGB_ENCODED_CUTOFF / SRGB_LINEAR_SLOPE
SRGB_ALPHA = 0.055
SRGB_GAMMA = 2.4

# --------------------------------------------------------------------------------
# Parsing

_HEX = re.compile(r"#?(?:[0-9a-fA-F]{3,4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})")

def hex_to_rgb(hex: str) -> tuple[int, ...]:
    """HTML hex colour to a tuple of integers in [0, 255]: RGB, or RGBA where the hex carries an alpha.

    Accepts `'#rrggbb'` and `'#rrggbbaa'`, and CSS's short forms `'#rgb'` and `'#rgba'`, in which each digit
    stands for a doubled one. The `'#'` is optional. Raises `ValueError` for anything else.
    """
    if not _HEX.fullmatch(hex):
        raise ValueError(f"hex_to_rgb: '{hex}' is not a hex colour ('#rrggbb', '#rrggbbaa', '#rgb' or '#rgba').")
    digits = hex.removeprefix("#")
    if len(digits) <= 4:  # short form: "f80" means "ff8800"
        digits = "".join(digit * 2 for digit in digits)
    return tuple(int(digits[i:i + 2], 16) for i in range(0, len(digits), 2))

def parse(spec: str | Sequence) -> tuple:
    """A colour in any of the spellings Raven accepts, to a tuple: RGB, or RGBA where `spec` has an alpha.

    - A hex string, anything `hex_to_rgb` accepts. Recognized by its shape before anything else is tried,
      so an all-digit hex such as `'123456'` is a colour rather than a number.
    - A Python literal as a string, `'(255, 0, 0)'` or `'[255, 0, 0, 128]'`.
    - A sequence, taken as it is.

    The components are returned as given, so their scale is the caller's: integers in [0, 255] for a hex
    string. Raises `ValueError` for anything that is none of the above.
    """
    if isinstance(spec, str):
        text = spec.strip()
        if _HEX.fullmatch(text):
            return hex_to_rgb(text)
        try:
            maybe_sequence = ast.literal_eval(text)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"parse: '{spec}' is not a colour: {type(exc)}: {exc}") from exc
        spec = maybe_sequence
    if not isinstance(spec, Sequence) or isinstance(spec, str) or len(spec) not in (3, 4):
        raise ValueError(f"parse: {spec!r} is not a colour: expected three or four components.")
    return tuple(spec)

# --------------------------------------------------------------------------------
# Mixing

def mix(first: Sequence[float], second: Sequence[float], t: float) -> tuple[float, ...]:
    """Mix two colours: `first` at `t` = 0, `second` at `t` = 1, component by component.

    The formula is::

        out = (1 - t) * first  +  t * second

    which is Porter-Duff 'over' on an opaque background. Any number of components, alpha included.
    """
    return tuple((1.0 - t) * a + t * b for a, b in zip(first, second))

# --------------------------------------------------------------------------------
# Luminance and contrast

def srgb_to_linear(value: float) -> float:
    """One sRGB-encoded channel in [0, 1] to linear light. The scalar twin of `raven.common.image.colorspace`'s."""
    if value <= SRGB_ENCODED_CUTOFF:
        return value / SRGB_LINEAR_SLOPE
    return ((value + SRGB_ALPHA) / (1.0 + SRGB_ALPHA)) ** SRGB_GAMMA

def linear_to_srgb(value: float) -> float:
    """Inverse of `srgb_to_linear`, which see: one linear-light channel in [0, 1] to sRGB-encoded."""
    if value <= SRGB_LINEAR_CUTOFF:
        return value * SRGB_LINEAR_SLOPE
    return (1.0 + SRGB_ALPHA) * value ** (1.0 / SRGB_GAMMA) - SRGB_ALPHA

def relative_luminance(color: Sequence[float]) -> float:
    """Relative luminance of an sRGB-encoded colour, as WCAG defines it: linearized, then BT.709-weighted."""
    return sum(weight * srgb_to_linear(channel) for weight, channel in zip(BT709_WEIGHTS, color[:3]))

def luma(color: Sequence[float]) -> float:
    """Luma, Y', of an sRGB-encoded colour: the BT.709 weights applied without linearizing first.

    Cheaper than `relative_luminance`, and good enough to decide which side of mid-grey a colour is on.
    """
    return sum(weight * channel for weight, channel in zip(BT709_WEIGHTS, color[:3]))

def contrast_ratio(one: Sequence[float], other: Sequence[float]) -> float:
    """WCAG contrast ratio between two colours, from 1 (identical) to 21 (black against white)."""
    first, second = relative_luminance(one), relative_luminance(other)
    lighter, darker = max(first, second), min(first, second)
    return (lighter + 0.05) / (darker + 0.05)

# --------------------------------------------------------------------------------
# HLS adjustments

def scale_saturation(color: Sequence[float], factor: float) -> tuple[float, ...]:
    """Return `color` with its HLS saturation scaled by `factor`, its hue and lightness untouched."""
    hue, lightness, saturation = colorsys.rgb_to_hls(*color[:3])
    return (*colorsys.hls_to_rgb(hue, lightness, saturation * factor), *color[3:])

def invert_lightness(lightness: float, brightest: float, darkest: float) -> float:
    """Map an HLS lightness in [0, 1] linearly onto [`brightest`, `darkest`], reversed: 0 → `brightest`, 1 → `darkest`.

    A dark mode's remap. Endpoints short of white and black keep it from being harsh: black text becomes a
    light grey rather than white, and a white background a dark grey rather than black.
    """
    return brightest - lightness * (brightest - darkest)

def uninvert_lightness(lightness: float, brightest: float, darkest: float) -> float:
    """Inverse of `invert_lightness`, which see: the lightness that it maps to `lightness`.

    For authoring a colour by how it should look *after* the remap.
    """
    return (brightest - lightness) / (brightest - darkest)

def legible_against(color: Sequence[float], background: Sequence[float], *,
                    min_ratio: float, lightness_range: tuple[float, float]) -> tuple[float, ...]:
    """Return `color` with its lightness moved as little as is needed to reach `min_ratio` against `background`.

    Hue and saturation are left alone, so a colour chosen to *say* something goes on saying it — a red
    stays red — while lightness, the dimension that decides whether it can be read, is spent on that.

    `min_ratio`: the WCAG contrast ratio to reach; see `contrast_ratio`.
    `lightness_range`: `(low, high)`, the lightnesses the result may take.

    Where neither end of the range reaches `min_ratio`, the better end is returned.
    """
    if contrast_ratio(color, background) >= min_ratio:
        return tuple(color)

    hue, lightness, saturation = colorsys.rgb_to_hls(*color[:3])

    def at(value: float) -> tuple[float, ...]:
        return (*colorsys.hls_to_rgb(hue, value, saturation), *color[3:])

    # For a fixed hue and saturation, luminance rises with lightness, so one end of the range helps and the
    # other does not; the binary search then finds the least move that clears the bar.
    best_end = max(lightness_range, key=lambda value: contrast_ratio(at(value), background))
    if contrast_ratio(at(best_end), background) < min_ratio:
        return at(best_end)
    near, far = lightness, best_end
    for _ in range(16):
        middle = 0.5 * (near + far)
        if contrast_ratio(at(middle), background) >= min_ratio:
            far = middle
        else:
            near = middle
    return at(far)

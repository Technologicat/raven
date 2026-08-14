"""Simple RGB/YUV colorspace conversion for video postprocessing.

Input/output is Torch tensor in [c, h, w] format.

This module is licensed under the 2-clause BSD license.
"""

__all__ = ["rgb_to_yuv", "yuv_to_rgb", "luminance",
           "linear_to_srgb", "srgb_to_linear",
           "hex_to_rgb"]

from typing import Tuple

import torch

# --------------------------------------------------------------------------------
# Colorspace conversion
#
# Some postprocessor filters need a colorspace that separates the brightness information from the colors.
# For simplicity, and considering that the filters are meant to simulate TV technology, we use YUV.
# (Also, it's familiar to those who have worked on the technical side of digital video.)
#
# RGB<->YCbCr conversion based on:
#   https://github.com/TheZino/pytorch-color-conversions/
#   https://gist.github.com/yohhoy/dafa5a47dade85d8b40625261af3776a
#   https://www.itu.int/rec/R-REC-BT.601 (SDTV)
#   https://www.itu.int/rec/R-REC-BT.709 (HDTV)
#   https://www.itu.int/rec/R-REC-BT.2020 (UHDTV)
#   https://en.wikipedia.org/wiki/Relative_luminance
#   https://en.wikipedia.org/wiki/Luma_(video)
#   https://en.wikipedia.org/wiki/Chrominance
#   https://en.wikipedia.org/wiki/YUV

# # Original approach of https://github.com/TheZino/pytorch-color-conversions/
# _RGB_TO_YCBCR = torch.tensor([[0.257, 0.504, 0.098],
#                               [-0.148, -0.291, 0.439],
#                               [0.439, -0.368, -0.071]])  # BT.601 (SDTV)
# _YCBCR_TO_RGB = torch.inverse(_RGB_TO_YCBCR)
# _YCBCR_OFF = torch.tensor([0.063, 0.502, 0.502])  # zero point: limited range black Y = 16/255, zero chroma U = V = 128/255
# def _colorspace_convert_mul(coeffs: torch.tensor, image: torch.tensor) -> torch.tensor:
#     return torch.einsum("rc,cij->rij", (coeffs.to(image.dtype).to(image.device), image))
#
# def _rgb_to_yuv(image: torch.tensor) -> torch.tensor:
#     YUV_zero = _YCBCR_OFF.to(image.dtype).to(image.device)
#     image_yuv = _colorspace_convert_mul(_RGB_TO_YCBCR, image) + YUV_zero
#     return torch.clamp(image_yuv, 0.0, 1.0)
#
# def _yuv_to_rgb(image: torch.tensor) -> torch.tensor:
#     YUV_zero = _YCBCR_OFF.to(image.dtype).to(image.device)
#     image_rgb = _colorspace_convert_mul(_YCBCR_TO_RGB, image - YUV_zero)
#     return torch.clamp(image_rgb, 0.0, 1.0)

# Note that since we are working in *linear* (i.e. before gamma) RGB space,
# Y is the true relative luminance (it is NOT the luma Y').
#
# Y  = a * R + b * G + c * B
# Cb = (B - Y) / d
# Cr = (R - Y) / e
#
# Here the chroma coordinates Cb and Cr are in [-0.5, 0.5].
# The inverse conversion is:
#
# R = Y + e * Cr
# G = Y - (a * e / b) * Cr - (c * d / b) * Cb
# B = Y + d * Cb
#
# where
#
#    BT.601  BT.709  BT.2020
# a  0.299   0.2126  0.2627
# b  0.587   0.7152  0.6780
# c  0.114   0.0722  0.0593
# d  1.772   1.8556  1.8814
# e  1.402   1.5748  1.4746
#
# Let's fully solve the first system for YCbCr in terms of RGB:
#
# Y  = a * R + b * G + c * B
# Cb = (B - (a * R + b * G + c * B)) / d
# Cr = (R - (a * R + b * G + c * B)) / e
#
# Y  = a * R + b * G + c * B
# Cb = (- a * R - b * G + (1 - c) * B) / d
# Cr = ((1 - a) * R - b * G - c * B) / e
#
# so YCbCr = M * RGB, where the matrix M is:
#
a, b, c, d, e = [0.2126, 0.7152, 0.0722, 1.8556, 1.5748]  # ITU-R Rec. 709 (HDTV)
_RGB_TO_YCBCR = torch.tensor([[a, b, c],
                              [-a / d, -b / d, (1.0 - c) / d],
                              [(1.0 - a) / e, -b / e, -c / e]])
_YCBCR_TO_RGB = torch.inverse(_RGB_TO_YCBCR)
def _colorspace_convert_mul(coeffs: torch.tensor, image: torch.tensor) -> torch.tensor:
    return torch.einsum("rc,cij->rij", (coeffs.to(image.dtype).to(image.device), image))

def rgb_to_yuv(image: torch.tensor) -> torch.tensor:
    """RGB (linear) 0...1 -> YUV, where Y = 0...1, U, V = -0.5 ... +0.5"""
    return _colorspace_convert_mul(_RGB_TO_YCBCR, image)

def yuv_to_rgb(image: torch.tensor, clamp: bool = True) -> torch.tensor:
    """Inverse of `rgb_to_yuv`, which see, optionally clamping the resulting RGB image to [0, 1]."""
    image_rgb = _colorspace_convert_mul(_YCBCR_TO_RGB, image)
    if clamp:
        image_rgb = torch.clamp(image_rgb, 0.0, 1.0)
    return image_rgb

_RGB_TO_Y = _RGB_TO_YCBCR[0, :]
def luminance(image: torch.tensor) -> torch.tensor:
    """RGB (linear 0...1) -> Y (true relative luminance)"""
    return torch.einsum("c,cij->ij", (_RGB_TO_Y.to(image.dtype).to(image.device), image))

# --------------------------------------------------------------------------------
# sRGB transfer function
#
# The conversions above work in *linear* RGB, which is where light adds up the way physics says it does —
# so that is where filtering, blending and resampling belong. A display, and every image file feeding one,
# instead stores sRGB: the same values passed through a roughly-gamma-2.2 curve that spends more of the
# available precision on the dark end, where the eye is more sensitive.
#
# So anything generated or computed in linear light needs `linear_to_srgb` before it is shown, and anything
# read from an ordinary image file needs `srgb_to_linear` before it is computed with. Skipping either is not
# a subtle error: mid-gray comes out at the wrong brightness and gradients bunch up at one end.
#
# The curve is linear near black and a power law above it, the joint chosen so both value and slope match.
#   https://en.wikipedia.org/wiki/SRGB
#   https://www.color.org/chardata/rgb/srgb.xalter (IEC 61966-2-1)

_SRGB_LINEAR_SLOPE = 12.92
_SRGB_ENCODED_CUTOFF = 0.04045  # where the encoded side switches from the linear segment to the power law
# The linear-light cutoff is derived rather than quoted. The standard rounds it to 0.0031308, which puts a
# discontinuity of about 1e-8 in the curve; dividing keeps the two segments meeting exactly, and matches the
# constant every other implementation of this ends up with.
_SRGB_LINEAR_CUTOFF = _SRGB_ENCODED_CUTOFF / _SRGB_LINEAR_SLOPE
_SRGB_ALPHA = 0.055
_SRGB_GAMMA = 2.4

def linear_to_srgb(image: torch.tensor) -> torch.tensor:
    """Linear RGB in [0, 1] -> sRGB-encoded in [0, 1]. Apply before displaying computed-in-linear pixels.

    Shape-agnostic and elementwise, so it takes an `[c, h, w]` image, a single channel, or a whole batch.
    Feed it colour channels only: an alpha channel is a coverage fraction rather than a light intensity,
    and is not gamma-encoded.
    """
    image = torch.clamp(image, 0.0, 1.0)
    # `clamp(min=...)` before the fractional power: at exactly zero its derivative is infinite, so autograd
    # and some backends produce a NaN for a value the `where` is going to discard anyway.
    high = (1.0 + _SRGB_ALPHA) * torch.clamp(image, min=1e-8) ** (1.0 / _SRGB_GAMMA) - _SRGB_ALPHA
    return torch.where(image <= _SRGB_LINEAR_CUTOFF, image * _SRGB_LINEAR_SLOPE, high)

def srgb_to_linear(image: torch.tensor) -> torch.tensor:
    """Inverse of `linear_to_srgb`, which see. Apply to pixels read from an ordinary image file."""
    image = torch.clamp(image, 0.0, 1.0)
    high = ((image + _SRGB_ALPHA) / (1.0 + _SRGB_ALPHA)) ** _SRGB_GAMMA
    return torch.where(image <= _SRGB_ENCODED_CUTOFF, image / _SRGB_LINEAR_SLOPE, high)

def hex_to_rgb(hex: str) -> Tuple[int]:
    """HTML hex color '#rrggbb' or '#rrggbbaa' to tuple of integers in [0, 255]."""
    hex = hex.removeprefix('#')
    rgb = tuple(int(hex[i:i + 2], 16) for i in [*range(0, len(hex), 2)])
    return rgb

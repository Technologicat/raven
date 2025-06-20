"""Smoke and mirrors. Glitch artistry. Pixel-space postprocessing effects.

These effects work in linear intensity space, before gamma correction.

This module is licensed under the 2-clause BSD license.
"""

__all__ = ["Postprocessor"]

from collections import defaultdict
from functools import wraps
import inspect
import logging
import math
import time
from typing import Dict, List, Optional, Tuple, TypeVar, Union

from unpythonic import getfunc, memoize

import numpy as np

import torch
import torchvision

from .colorspace import rgb_to_yuv, yuv_to_rgb, luminance
from .upscaler import Upscaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar("T")
Atom = Union[str, bool, int, float]
MaybeContained = Union[T, List[T], Dict[str, T]]

VHS_GLITCH_BLANK = object()  # nonce value, see `analog_vhsglitches`

# --------------------------------------------------------------------------------
# Advanced tonemapper HDR -> LDR.

# Currently not used; slow, and the input data would need some preprocessing to eliminate blank pixels.

class HistogramEqualizer:
    def __init__(self,
                 device: torch.device,
                 dtype: torch.dtype,
                 nbins: int = 256):
        self.device = device
        self.dtype = dtype
        self.nbins = nbins

        self.display_range = np.log(255.0 / 1.0)  # max/min brightness, ~45 dB for 8 bits per channel

    def _compute_cdf(self,
                     x: torch.Tensor,
                     bin_edges: torch.Tensor,
                     discard_first_bin: bool = False,
                     discard_last_bin: bool = False):
        """Compute the cumulative distribution function of the data, given `bin_edges`.

        The result with have `n_bins + 2` entries, where `n_bins = len(bin_edges) - 1`;
        the first and last elements correspond to data less than the first bin edge,
        and data greater than the last bin edge, respectively.

        `discard_first_bin`: Optionally, drop the first bin. Useful for ignoring data less than the first bin edge.
        `discard_last_bin`: Optionally, drop the last bin. Useful for ignoring data greater than the last bin edge.

        Needed because `torch.histogram` doesn't support the CUDA backend:
            https://github.com/pytorch/pytorch/issues/69519
        """
        level = torch.bucketize(x, bin_edges, out_int32=True)  # sum(x > v for v in bin_edges)
        count = x.new_zeros(len(bin_edges) + 1, dtype=torch.float32)
        count.index_put_((level,), count.new_ones(1), True)

        if discard_first_bin:
            count = count[1:]
        if discard_last_bin:
            count = count[:-1]
        norm = x.numel()
        cdf = torch.cumsum(count, dim=0) / norm
        pdf = count / norm
        return pdf, cdf

    def _loghist(self, x: torch.Tensor, min_nonzero_x, max_x):
        # Histogramming linearly in logarithmic space gives us a logarithmically spaced histogram.
        log_bin_edges = torch.linspace(torch.log(min_nonzero_x), torch.log(max_x), self.nbins, dtype=self.dtype, device=self.device)
        bin_edges = torch.exp(log_bin_edges)
        # - Ignore the less-than-first-bin-edge slot to ignore black pixels (empty background). (TODO: should do this for *blank*, not *black*, but that would need some kind of mask support.)
        # - Ignore the greater-than-last-bin-edge slot, since we always put the last bin edge at max(x).
        pdf, cdf = self._compute_cdf(x, bin_edges, discard_first_bin=True, discard_last_bin=True)
        zero = torch.tensor([0.0], device=self.device, dtype=self.dtype)
        pdf = torch.cat([zero, pdf], dim=0)  # pretend there's no data below the first bin edge
        cdf = torch.cat([zero, cdf], dim=0)  # a CDF should start from zero
        return bin_edges, pdf, cdf

    def loghist(self, x: torch.Tensor):
        """Logarithmically spaced histogram.

        - `x` must be in a floating-point type
        - negative values are ignored (these get mapped to the first bin)
        """
        max_x = torch.max(x)
        # min_nonzero_x = torch.min(torch.where(x > 0.0, x, max_x))  # for general data
        min_nonzero_x = torch.tensor(0.01, device=self.device, dtype=self.dtype)  # for images
        return self._loghist(x, min_nonzero_x, max_x)

    def larson(self, x: torch.Tensor):
        """Data-adaptive histogram corrector of Ward-Larson, Rushmeier and Piatko (1997).

        Compresses dynamic range, but does not exaggerate contrast of sparsely populated parts of the histogram.
        This is done by limiting superlinear growth of the display intensity. We modify the measured histogram,
        and then histogram-equalize with the result.

        This typically looks better than simple log scaling, and brings out details in the data.
        """
        max_x = torch.max(x)
        # min_nonzero_x = torch.min(torch.where(x > 0.0, x, max_x))  # for general data
        min_nonzero_x = torch.tensor(0.01, device=self.device, dtype=self.dtype)  # for images
        bin_edges, pdf, cdf = self._loghist(x, min_nonzero_x, max_x)

        data_range = torch.log(max_x / min_nonzero_x).cpu()
        if data_range <= self.display_range:  # data is LDR, nothing to do
            return bin_edges, pdf, cdf
        orig_pdf = pdf
        orig_cdf = cdf

        # Tolerance of the iteration (an arbitrary small number).
        #
        # Smaller tolerances produce better conformance to the ceiling (i.e. better prevent superlinear growth).
        # The original article used the value 0.025.
        #
        tol = 0.005 * torch.max(pdf)

        delta_b = data_range / self.nbins  # width of one logarithmic bin

        trimmings = torch.tensor(np.inf, device=self.device, dtype=self.dtype)  # loop at least once
        while trimmings > tol:
            pdf_sum = torch.sum(pdf)
            if pdf_sum < tol:  # degenerate case: histogram has been zeroed out -> return original image
                logger.warning("HistogramEqualizer.tonemap: histogram zeroed out, returning original histogram")
                return bin_edges, orig_pdf, orig_cdf
            # Cut any peaks from PDF that are above the current ceiling; calculate how much we cut in total (to decide whether to iterate again).
            ceiling = pdf_sum * (delta_b / self.display_range)
            excess = pdf - ceiling
            trimmings = torch.sum(torch.where(excess > 0.0, excess, 0.0))
            pdf = torch.where(pdf > ceiling, ceiling, pdf)

        cdf = torch.cumsum(pdf, dim=0)
        cdfmax = torch.max(cdf)
        pdf /= cdfmax  # normalize sum of adjusted histogram to 1
        cdf /= cdfmax  # normalize maximum of adjusted cumulative distribution function to 1

        return bin_edges, pdf, cdf

    def equalize_by_cdf(self,
                        x: torch.Tensor,
                        bin_edges: torch.Tensor,
                        cdf: torch.Tensor):
        """Histogram-equalize data by a given CDF.

        See `loghist` and `larson` to get the inputs for this.
        """
        discretized = torch.bucketize(x, bin_edges)
        return cdf[discretized.view(-1)].reshape(discretized.shape)

# --------------------------------------------------------------------------------

# Gaussian blur kernel size for most filters that use Gaussian blur.
#
# The largest sensible value for `sigma` is such that
#
#     kernel_size = 2 * (2 * sigma) + 1,
#
# so that the kernel reaches its "2 sigma" (95% mass) point where the
# finitely sized kernel cuts the tail. If you want to be really hi-fi,
# you could go for the "3 sigma" (99.7% mass) point, but in practice
# this isn't necessary.
#
# Hence, the largest sensible value is `sigma = 3.0` (which is already
# really blurry).
#
_kernel_size = 13

# Convenient for GUI auto-population so that we can specify sensible ranges for parameters once, at the implementation site (not separately at each GUI use site).
def with_metadata(**metadata):
    def decorator(func):
        @wraps(func)
        def func_with_metadata(*args, **kwargs):
            return func(*args, **kwargs)
        func_with_metadata.metadata = metadata  # stash it on the function object
        return func_with_metadata
    return decorator

class Postprocessor:
    """
    `chain`: Postprocessor filter chain configuration.

             For an example, see `postprocessor_defaults` in `config.py`.

             Don't mind the complicated type signature; the format is just::

                 [(filter_name0, {param0: value0, ...}),
                  ...]

             The filter name must be a method of `Postprocessor`, taking in an image, and any number of named parameters.
             To use a filter's default parameter values, supply an empty dictionary for the parameters.

             The outer `Optional[List[Tuple[...]]]` just formalizes that `chain` may be omitted (to use the built-in
             default chain, for testing), and the top-level format that it's an ordered list of filters. The filters
             are applied in order, first to last.

             The auxiliary type definitions are::

                 MaybeContained = Union[T, List[T], Dict[str: T]]
                 Atom = Union[str, bool, int, float]

             The leaf value (atom) types are restricted so that filter chain configurations JSON easily.

             The leaf values may actually be contained inside arbitrarily nested lists and dicts (with str keys),
             which is currently not captured by the type signature (the definition should be recursive).

             The chain is stored as `self.chain`. Any modifications to that attribute modify the chain,
             taking effect immediately. It is recommended to update the chain atomically, by::

                 my_postprocessor.chain = my_new_chain

    In filter descriptions:
        [static] := depends only on input image, no explicit time dependence.
        [dynamic] := beside input image, also depends on time. In other words,
                     produces animation even for a stationary input image.
    """

    def __init__(self,
                 device: torch.device,
                 dtype: torch.dtype,
                 chain: List[Tuple[str, Dict[str, MaybeContained[Atom]]]] = None):
        # We intentionally keep very little state in this class, for a more FP/REST approach with less bugs.
        # Filters for static effects are stateless.
        #
        # We deviate from FP in that:
        #   - The filters MUTATE, i.e. they overwrite the image being processed.
        #     This is to allow optimizing their implementations for memory usage and speed.
        #   - The filter for a dynamic effect may store state, if needed for performing FPS correction.
        self.device = device
        self.dtype = dtype
        self.chain = chain

        self.histeq = HistogramEqualizer(self.device, self.dtype)

        # Meshgrid cache for geometric position of each pixel
        self._yy = None
        self._xx = None
        self._meshy = None
        self._meshx = None
        self._prev_h = None
        self._prev_w = None

        # FPS correction
        self.CALIBRATION_FPS = 25  # design FPS for dynamic effects (for automatic FPS correction)
        self.stream_start_timestamp = time.time_ns()  # for updating frame counter reliably (no accumulation)
        self.frame_no = -1  # float, frame counter for *normalized* frame number *at CALIBRATION_FPS*
        self.last_frame_no = -1

        # Caches for individual dynamic effects
        self.zoom_data = defaultdict(lambda: None)
        self.noise_last_image = defaultdict(lambda: None)
        self.vhs_glitch_interval = defaultdict(lambda: 0.0)
        self.vhs_glitch_last_frame_no = defaultdict(lambda: 0.0)
        self.vhs_glitch_last_image = defaultdict(lambda: None)
        self.vhs_glitch_last_mask = defaultdict(lambda: None)
        self.digital_glitches_interval = defaultdict(lambda: 0.0)
        self.digital_glitches_last_frame_no = defaultdict(lambda: 0.0)
        self.digital_glitches_grid = defaultdict(lambda: None)

    @classmethod
    @memoize
    def get_filters(cls):
        """Return a list of available postprocessing filters and their default configurations.

        This is convenient for dynamically populating a GUI.

        Return format is `[(name0: settings0), ...]`
        """
        filters = []
        for name in dir(cls):
            if name.startswith("_"):
                continue
            if name in ("get_filters", "render_into"):
                continue

            # The authoritative source for parameter defaults is the source code, so:
            meth = getattr(cls, name)  # get the method from the class
            if not callable(meth):  # some other kind of attribute? (e.g. cache data for filters)
                continue
            # is a method

            func, _ = getfunc(meth)  # get the underlying raw function, for use with `inspect.signature`
            sig = inspect.signature(func)
            # All of our filter settings have a default value.
            settings = {v.name: v.default for v in sig.parameters.values() if v.default is not inspect.Parameter.empty}
            # Parameter ranges are specified at definition site via our internal `with_metadata` mechanism.
            ranges = {name: meth.metadata[name] for name in settings}
            param_info = {"defaults": settings,
                          "ranges": ranges}
            filters.append((name, param_info))
        def rendering_priority(metadata_record):
            name, _ = metadata_record
            meth = getattr(cls, name)
            return meth.metadata["_priority"]
        return list(sorted(filters, key=rendering_priority))

    def render_into(self, image):
        """Apply current postprocess chain, modifying `image` in-place."""
        time_render_start = time.time_ns()

        chain = self.chain  # read just once; other threads might reassign it while we're rendering
        if not chain:
            return

        c, h, w = image.shape
        if h != self._prev_h or w != self._prev_w:
            logger.info(f"render_into: Computing pixel position tensors for image size {w}x{h}")
            # Compute base meshgrid for the geometric position of each pixel.
            # This is needed by filters that either vary by geometric position (e.g. `vignetting`),
            # or deform the image (e.g. `analog_rippling_hsync`).
            #
            # This postprocessor is typically applied to a video stream. As long as
            # the image dimensions stay constant, we can re-use the previous meshgrid.
            #
            # We don't strictly keep state here - we just cache. :P

            with torch.no_grad():
                self._yy = torch.linspace(-1.0, 1.0, h, dtype=self.dtype, device=self.device)
                self._xx = torch.linspace(-1.0, 1.0, w, dtype=self.dtype, device=self.device)
                self._meshy, self._meshx = torch.meshgrid((self._yy, self._xx), indexing="ij")
            logger.info("render_into: Pixel position tensors cached")

        # Update the frame counter.
        #
        # We consider the frame number to be a float, so that dynamic filters can decide what
        # to do at fractional frame positions. For continuously animated effects (e.g. banding)
        # it makes sense to interpolate continuously, whereas other effects (e.g. scanlines)
        # can make their decisions based on the integer part.
        #
        # As always with floats, we must be careful. Note that we operate in a mindset of robust
        # engineering. Since doing the Right Thing here does not cost significantly more engineering
        # effort than doing the intuitive but Wrong Thing, it is preferable to go for the proper solution,
        # regardless of whether it would take a centuries-long session to actually trigger a failure
        # in the less robust approach.
        #
        # So, floating point accuracy considerations? First, we note that accumulation invites
        # disaster in two ways:
        #
        #   - Accumulating the result accumulates also representation error and roundoff error.
        #   - When accumulating small positive numbers to a sum total, the update eventually
        #     becomes too small to add, causing the counter to get stuck. (For floats, `x + ϵ = x`
        #     for sufficiently small ϵ dependent on the magnitude of `x`.)
        #
        # Fortunately, frame number is a linear function of time, and time diffs can be measured
        # precisely. Thus, we can freshly compute the current frame number at each frame, completely
        # bypassing the need for accumulation:
        #
        seconds_since_stream_start = (time_render_start - self.stream_start_timestamp) / 10**9
        self.last_frame_no = self.frame_no
        self.frame_no = self.CALIBRATION_FPS * seconds_since_stream_start  # float!

        # That leaves just the questions of how accurate the calculation is, and for how long.
        # As to the first question:
        #
        #  - Timestamps are an integer number of nanoseconds, so they are exact.
        #  - Dividing by 10**9, we move the decimal point. But floats are base-2, so 0.1
        #    is not representable in IEEE-754. So there will be some small representation error,
        #    which for float64 likely appears in the ~15th significant digit.
        #  - Basic arithmetic, such as multiplication, is guaranteed by IEEE-754
        #    to be accurate to the ULP.
        #
        # Thus, as the result, we obtain the closest number that is representable in IEEE-754,
        # and the strategy works for the whole range of float64.
        #
        # As for the second question, floats are logarithmically spaced. So if this is left running
        # "for long enough" during the same session, accuracy will eventually suffer. Instead of the
        # counter getting stuck, however, this will manifest as the frame number updating by more
        # than `1.0` each time it updates (i.e. whenever the elapsed number of frames reaches the
        # next representable float).
        #
        # This could be fixed by resetting `stream_start_timestamp` once the frame number
        # becomes too large. But in practice, how long does it take for this issue to occur?
        # The ULP becomes 1.0 at ~5e15. To reach frame number 5e15, at the reference 25 FPS,
        # the time required is 2e14 seconds, i.e. 2.31e9 days, or 6.34 million years.
        # While I can almost imagine the eventual bug report, I think it's safe to ignore this.

        # Apply the current filter chain.
        with torch.no_grad():
            for filter_name, settings in chain:
                apply_filter = getattr(self, filter_name)
                apply_filter(image, **settings)

        # Remember last seen video stream size (for caches that store grids/images)
        self._prev_h = h
        self._prev_w = w

    # --------------------------------------------------------------------------------
    # Physical input signal

    @with_metadata(center_x=[-1.0, 1.0],
                   center_y=[-1.0, 1.0],
                   factor=[1.0, 4.0],
                   quality=["low", "high", "ultra"],
                   name=["!ignore"],
                   _priority=-1.0)
    def zoom(self, image: torch.tensor, *,
             center_x: float = 0.0,
             center_y: float = -0.867,
             factor: float = 2.0,
             quality: str = "low",
             name: str = "zoom0"):
        """[dynamic] Simulated optical zoom for anime video.

        The default settings zoom to the head/shoulders of most characters.

        `center_x`: Center position of zoom on x axis, where image is [-1, 1].
        `center_y`: Center position of zoom on y axis, where image is [-1, 1],
                    negative upward.
        `factor`: Zoom by this much. Values larger than 1.0 zoom in.
                  At exactly 1.0, the zoom filter is disabled.
        `quality`: One of:
                   "low": geometric distortion with bilinear interpolation (fast)
                   "high": crop, then low-quality Anime4K upscale (fast-ish)
                   "ultra": crop, then high-quality Anime4K upscale

        How much quality you need depends on what other filters are enabled,
        how large `factor` is, and how large the final size of the avatar is.
        If you upscale the avatar by 2x, without much other postprocessing
        than this zoom filter, and zoom in by `factor=4.0`, then the "ultra"
        quality may be necessary, and might still not look good. But if you
        use lots of other filters, and limit to at most `factor=2.0`, then
        even "low" might look acceptable.

        Dynamic only to save compute; we cache the distortion mesh and upscaler.
        """
        if factor == 1.0:
            return

        # Recompute mesh when the filter settings change, or the video stream size changes.
        do_setup = True
        c, h, w = image.shape
        size_changed = (h != self._prev_h or w != self._prev_w)
        if not size_changed and name in self.zoom_data:
            if (factor == self.zoom_data[name]["factor"] and
                    center_x == self.zoom_data[name]["center_x"] and
                    center_y == self.zoom_data[name]["center_y"] and
                    quality == self.zoom_data[name]["quality"]):
                do_setup = False
                grid = self.zoom_data[name]["grid"]
                upscaler = self.zoom_data[name]["upscaler"]
        if do_setup:
            meshx = center_x + (self._meshx - center_x) / factor  # x coordinate, [h, w]
            meshy = center_y + (self._meshy - center_y) / factor  # y coordinate, [h, w]
            grid = torch.stack((meshx, meshy), 2)  # [h, w, x/y]
            grid = grid.unsqueeze(0)  # batch of one
            if quality in ("high", "ultra"):  # need an Anime4K?
                old_upscaler = self.zoom_data[name]["upscaler"] if name in self.zoom_data else None
                old_quality = self.zoom_data[name]["quality"] if name in self.zoom_data else None
                if size_changed or old_upscaler is None or quality != old_quality:
                    upscaler_quality = "high" if quality == "ultra" else "low"
                    upscaler = Upscaler(device=self.device, dtype=self.dtype,
                                        upscaled_width=w, upscaled_height=h,
                                        preset="C", quality=upscaler_quality)
                else:
                    upscaler = old_upscaler  # only factor/center changed - save some compute by recycling the existing upscaler.
            else:
                upscaler = None
            self.zoom_data[name] = {"center_x": center_x,
                                    "center_y": center_y,
                                    "factor": factor,
                                    "quality": quality,
                                    "grid": grid,
                                    "upscaler": upscaler}

        if quality == "low":  # geometric distortion
            image_batch = image.unsqueeze(0)  # batch of one -> [1, c, h, w]
            warped = torch.nn.functional.grid_sample(image_batch, grid, mode="bilinear", padding_mode="border", align_corners=False)
            warped = warped.squeeze(0)  # [1, c, h, w] -> [c, h, w]
            image[:, :, :] = warped
        else:  # "high" or "ultra" - crop, then Anime4K upscale
            g = grid.squeeze(0)
            top_left_xy = g[0, 0]
            bottom_right_xy = g[-1, -1]
            x0 = int((top_left_xy[0] + 1.0) / 2 * w)
            y0 = int((top_left_xy[1] + 1.0) / 2 * h)
            x1 = int((bottom_right_xy[0] + 1.0) / 2 * w)
            y1 = int((bottom_right_xy[1] + 1.0) / 2 * h)
            # print(x0, y0, x1, y1)  # DEBUG
            cropped = torchvision.transforms.functional.crop(image, top=y0, left=x0, height=(y1 - y0), width=(x1 - x0))
            image[:, :, :] = upscaler.upscale(cropped)

    # Defaults chosen so that they look good for a handful of characters rendered in SD Forge, with the Wai 14.0 Illustrious-SDXL checkpoint.
    @with_metadata(threshold=[0.0, 1.0],
                   exposure=[0.1, 5.0],
                   _priority=0.0)
    def bloom(self, image: torch.tensor, *,
              threshold: float = 0.560,
              exposure: float = 0.842) -> None:
        """[static] Bloom effect (fake HDR). Makes the image look brighter. Popular in early 2000s anime.

        Can also be used as just a camera exposure adjustment by setting `threshold=1.0` to disable the glow.

        Makes bright parts of the image bleed light into their surroundings, enhancing perceived contrast.
        Only makes sense when the avatar is rendered on a dark-ish background.

        `threshold`: How bright is bright. 0.0 is full black (all pixels glow), 1.0 is full white (bloom disabled).
                     Technically, this is true relative luminance, not luma, since we work in linear RGB space.
        `exposure`: Overall brightness of the output. Like in photography, higher exposure means brighter image,
                    saturating toward white.
        """
        # There are online tutorials for how to create this effect, see e.g.:
        #   https://learnopengl.com/Advanced-Lighting/Bloom

        if threshold < 1.0:
            # Find the bright parts.
            # original_yuv = rgb_to_yuv(image[:3, :, :])
            # Y = original_yuv[0, :, :]
            Y = luminance(image[:3, :, :])
            mask = torch.ge(Y, threshold)  # [h, w]

            # Make a copy of the image with just the bright parts.
            mask = torch.unsqueeze(mask, 0)  # -> [1, h, w]
            brights = image * mask  # [c, h, w]

            # Blur the bright parts. Two-pass blur to save compute, since we need a very large blur kernel.
            # It seems that in Torch, one large 1D blur is faster than looping with a smaller one.
            #
            # Although everything else in Torch takes (height, width), kernel size is given as (size_x, size_y);
            # see `gaussian_blur_image` in https://pytorch.org/vision/main/_modules/torchvision/transforms/v2/functional/_misc.html
            # for a hint (the part where it computes the padding).
            brights = torchvision.transforms.GaussianBlur((21, 1), sigma=7.0)(brights)  # blur along x
            brights = torchvision.transforms.GaussianBlur((1, 21), sigma=7.0)(brights)  # blur along y

            # Additively blend the images. Note we are working in linear intensity space, and we will now go over 1.0 intensity.
            image.add_(brights)

        # We now have a fake HDR image. Tonemap it back to LDR.
        image[:3, :, :] = 1.0 - torch.exp(-image[:3, :, :] * exposure)  # RGB: tonemap

        # # TEST - apply Larson's adaptive histogram remapper to the Y (luminance) channel, and keep the color channels (U, V) as-is.
        # # Doesn't look that good, the classical method seems better for this use.
        # hdr_Y = luminance(image[:3, :, :])  # Eris have mercy on us, this function isn't designed for HDR data. Seems to work fine, though.
        # bin_edges, pdf, cdf = self.histeq.loghist(hdr_Y)  # or use `self.histeq.loghist` for classical histogram equalization
        # ldr_Y = self.histeq.equalize_by_cdf(hdr_Y, bin_edges, cdf)
        # image[:3, :, :] = yuv_to_rgb(torch.cat([ldr_Y.unsqueeze(0), original_yuv[1:]], dim=0))

        if threshold < 1.0:
            image[3, :, :] = torch.maximum(image[3, :, :], brights[3, :, :])  # alpha: max-combine

        torch.clamp_(image, min=0.0, max=1.0)

    # --------------------------------------------------------------------------------
    # Video camera

    @with_metadata(scale=[0.001, 0.05],
                   sigma=[0.1, 3.0],
                   _priority=1.0)
    def chromatic_aberration(self, image: torch.tensor, *,
                             scale: float = 0.005,
                             sigma: float = 1.0) -> None:
        """[static] Simulate the two types of chromatic aberration in a camera lens.

        Like everything else here, this is of course made of smoke and mirrors. We simulate the axial effect
        (index of refraction varying w.r.t. wavelength) by geometrically scaling the RGB channels individually,
        and the transverse effect (focal distance varying w.r.t. wavelength) by a gaussian blur.

        `scale`: Axial CA geometric distortion parameter.
        `sigma`: Transverse CA blur parameter.

        Note that in a real lens:
          - Axial CA is typical at long focal lengths (e.g. tele/zoom lens)
          - Axial CA increases at high F-stops (low depth of field, i.e. sharp focus at all distances)
          - Transverse CA is typical at short focal lengths (e.g. macro lens)

        However, in an RGB postproc effect, it is useful to apply both together, to help hide the clear-cut red/blue bands
        resulting from the different geometric scalings of just three wavelengths (instead of a continuous spectrum, like
        a scene lit with natural light would have).

        See:
            https://en.wikipedia.org/wiki/Chromatic_aberration
        """
        # Axial CA: Shrink R (deflected less), pass G through (lens reference wavelength), enlarge B (deflected more).
        grid_R = torch.stack((self._meshx * (1.0 + scale), self._meshy * (1.0 + scale)), 2)
        grid_R = grid_R.unsqueeze(0)
        grid_B = torch.stack((self._meshx * (1.0 - scale), self._meshy * (1.0 - scale)), 2)
        grid_B = grid_B.unsqueeze(0)

        image_batch_R = image[0, :, :].unsqueeze(0).unsqueeze(0)  # [h, w] -> [c, h, w] -> [n, c, h, w]
        warped_R = torch.nn.functional.grid_sample(image_batch_R, grid_R, mode="bilinear", padding_mode="border", align_corners=False)
        warped_R = warped_R.squeeze(0)  # [1, c, h, w] -> [c, h, w]
        image_batch_B = image[2, :, :].unsqueeze(0).unsqueeze(0)
        warped_B = torch.nn.functional.grid_sample(image_batch_B, grid_B, mode="bilinear", padding_mode="border", align_corners=False)
        warped_B = warped_B.squeeze(0)  # [1, c, h, w] -> [c, h, w]

        # Transverse CA (blur to simulate wrong focal distance for R and B)
        warped_R[:, :, :] = torchvision.transforms.GaussianBlur((_kernel_size, 1), sigma=sigma)(warped_R)  # blur along x
        warped_R[:, :, :] = torchvision.transforms.GaussianBlur((1, _kernel_size), sigma=sigma)(warped_R)  # blur along y
        warped_B[:, :, :] = torchvision.transforms.GaussianBlur((_kernel_size, 1), sigma=sigma)(warped_B)  # blur along x
        warped_B[:, :, :] = torchvision.transforms.GaussianBlur((1, _kernel_size), sigma=sigma)(warped_B)  # blur along y

        # Alpha channel: treat similarly to each of R,G,B and average the three resulting alpha channels
        image_batch_A = image[3, :, :].unsqueeze(0).unsqueeze(0)
        warped_A1 = torch.nn.functional.grid_sample(image_batch_A, grid_R, mode="bilinear", padding_mode="border", align_corners=False)
        warped_A1[:, :, :] = torchvision.transforms.GaussianBlur((_kernel_size, 1), sigma=sigma)(warped_A1)
        warped_A1[:, :, :] = torchvision.transforms.GaussianBlur((1, _kernel_size), sigma=sigma)(warped_A1)
        warped_A2 = torch.nn.functional.grid_sample(image_batch_A, grid_B, mode="bilinear", padding_mode="border", align_corners=False)
        warped_A2[:, :, :] = torchvision.transforms.GaussianBlur((_kernel_size, 1), sigma=sigma)(warped_A2)
        warped_A2[:, :, :] = torchvision.transforms.GaussianBlur((1, _kernel_size), sigma=sigma)(warped_A2)
        averaged_alpha = (warped_A1 + image[3, :, :] + warped_A2) / 3.0

        image[0, :, :] = warped_R
        # image[1, :, :] passed through as-is
        image[2, :, :] = warped_B
        image[3, :, :] = averaged_alpha

    @with_metadata(strength=[0.1, 0.5],
                   _priority=2.0)
    def vignetting(self, image: torch.tensor, *,
                   strength: float = 0.42) -> None:
        """[static] Simulate vignetting (less light hitting the corners of a film frame or CCD sensor).

        The profile used here is [cos(strength * d * pi)]**2, where `d` is the distance
        from the center, scaled such that `d = 1.0` is reached at the corners.
        Thus, at the midpoints of the frame edges, `d = 1 / sqrt(2) ~ 0.707`.
        """
        euclidean_distance_from_center = (self._meshy**2 + self._meshx**2)**0.5 / 2**0.5  # [h, w]
        brightness = torch.cos(strength * euclidean_distance_from_center * math.pi)**2  # [h, w]
        brightness = torch.unsqueeze(brightness, 0)  # -> [1, h, w]
        image[:3, :, :] *= brightness

    # --------------------------------------------------------------------------------
    # Scifi hologram

    @with_metadata(alpha=[0.1, 0.9],
                   _priority=3.0)
    def translucency(self, image: torch.tensor, *,
                     alpha: float = 0.9) -> None:
        """[static] A simple translucency filter for a hologram look.

        Multiplicatively adjusts the alpha channel.
        """
        image[3, :, :].mul_(alpha)

    # --------------------------------------------------------------------------------
    # General use

    @with_metadata(strength=[0.1, 1.0],
                   sigma=[0.1, 3.0],
                   channel=["Y", "A"],
                   name=["!ignore"],  # hint for GUI to ignore this parameter
                   _priority=4.0)
    def noise(self, image: torch.tensor, *,
              strength: float = 0.3,
              sigma: float = 1.0,
              channel: str = "Y",
              name: str = "noise0") -> None:
        """[dynamic] Add noise to the luminance or to the alpha channel.

        NOTE: At small values of `sigma`, this filter causes the video to use a lot of bandwidth
        during network transfer, because noise is impossible to compress.

        `strength`: Fraction of noise in the output's Y or A channel.

                    How much noise to apply. 0 is no noise, 1 replaces the input image with noise.

                    The formula is:

                        out = in * ((1 - magnitude) + magnitude * noise_texture)

                    The filter is multiplicative, so it never brightens the image, and
                    never makes visible pixels that are fully translucent (alpha = 0.0) in the input.

                    A larger `strength` makes the image darker/more translucent overall, because
                    then a larger portion of the luminance/alpha axis is reserved for the noise.

        `sigma`: If nonzero, apply a Gaussian blur to the noise, thus reducing its spatial frequency
                 (i.e. making larger and smoother "noise blobs").

        `channel`: One of:
                     "Y": modulate the luminance (converts to YUV and back; slower)
                     "A": modulate the alpha channel (fast; perhaps less effect on image file size
                          in compressed RGBA formats; makes the noise translucent)

        `name`: Optional name for this filter instance in the chain. Used as cache key to store the noise texture.
                If you have more than one `noise` filter in the chain, they should have different names so that
                each one gets its own noise texture.

                (Of course, to save some compute, you can give them the same name; then they'll use the same
                 noise texture. But be aware that then the sigma values should be the same, because sigma
                 affects the texture.)

        Suggested settings:
            Scifi hologram:   strength=0.1, sigma=0.0
            Analog VHS tape:  strength=0.2, sigma=2.0
        """
        # Re-randomize the noise texture whenever the normalized frame number changes, or the video stream size changes.
        c, h, w = image.shape
        size_changed = (h != self._prev_h or w != self._prev_w)
        if self.noise_last_image[name] is None or size_changed or int(self.frame_no) > int(self.last_frame_no):
            c, h, w = image.shape
            noise_image = torch.rand(h, w, device=self.device, dtype=image.dtype)
            if sigma > 0.0:
                noise_image = noise_image.unsqueeze(0)  # [h, w] -> [c, h, w] (where c=1)
                noise_image = torchvision.transforms.GaussianBlur((_kernel_size, 1), sigma=sigma)(noise_image)
                noise_image = torchvision.transforms.GaussianBlur((1, _kernel_size), sigma=sigma)(noise_image)
                noise_image = noise_image.squeeze(0)  # -> [h, w]
            self.noise_last_image[name] = noise_image
        else:
            noise_image = self.noise_last_image[name]
        base_multiplier = 1.0 - strength

        if channel == "A":  # alpha
            image[3, :, :].mul_(base_multiplier + strength * noise_image)
        else:  # "Y", luminance
            image_yuv = rgb_to_yuv(image[:3, :, :])
            image_yuv[0, :, :].mul_(base_multiplier + strength * noise_image)
            image_rgb = yuv_to_rgb(image_yuv)
            image[:3, :, :] = image_rgb

    # --------------------------------------------------------------------------------
    # Lo-fi analog video

    @with_metadata(sigma=[0.1, 3.0],
                   _priority=5.0)
    def analog_lowres(self, image: torch.tensor, *,
                      sigma: float = 1.5) -> None:
        """[static] Low-resolution analog video signal, simulated by blurring.

        `sigma`: standard deviation of the Gaussian blur kernel, in pixels.
        """
        image[:, :, :] = torchvision.transforms.GaussianBlur((_kernel_size, 1), sigma=sigma)(image)  # blur along x
        image[:, :, :] = torchvision.transforms.GaussianBlur((1, _kernel_size), sigma=sigma)(image)  # blur along y

    @with_metadata(speed=[1.0, 16.0],
                   amplitude1=[0.001, 0.01],
                   density1=[1.0, 100.0],
                   amplitude2=[0.001, 0.01],
                   density2=[1.0, 100.0],
                   amplitude3=[0.001, 0.01],
                   density3=[1.0, 100.0],
                   _priority=6.0)
    def analog_rippling_hsync(self, image: torch.tensor, *,
                              speed: float = 8.0,
                              amplitude1: float = 0.001, density1: float = 4.0,
                              amplitude2: Optional[float] = 0.001, density2: Optional[float] = 13.0,
                              amplitude3: Optional[float] = 0.001, density3: Optional[float] = 27.0) -> None:
        """[dynamic] Analog video signal with fluctuating hsync.

        In practice, this looks like a rippling effect added to the outline of the character.

        We superpose three waves with different densities (1 / cycle length)
        to make the pattern look more irregular.

        E.g. density of 2.0 means that two full waves fit into the image height.

        Amplitudes are given in units where the height and width of the image
        are both 2.0.

        `speed`: At speed 1.0, a wave of `density = 1.0` completes a full cycle every
                 `image_height` frames. So effectively the cycle position updates by
                 `speed * (1 / image_height)` at each frame.

        NOTE: "frame" here refers to the normalized frame number, at a reference of 25 FPS.
        """
        c, h, w = image.shape

        # Animation
        # FPS correction happens automatically, because `frame_no` is normalized to CALIBRATION_FPS.
        cycle_pos = (self.frame_no / h) * speed
        cycle_pos = cycle_pos - float(int(cycle_pos))  # fractional part
        cycle_pos = 1.0 - cycle_pos  # -> motion from top toward bottom
        cycle_pos *= 2.0  # full cycle = 2 units

        # Deformation
        meshy = self._meshy
        meshx = self._meshx + amplitude1 * torch.sin((density1 * (self._meshy + cycle_pos)) * math.pi)
        if amplitude2 and density2:
            meshx = meshx + amplitude2 * torch.sin((density2 * (self._meshy + cycle_pos)) * math.pi)
        if amplitude3 and density3:
            meshx = meshx + amplitude3 * torch.sin((density3 * (self._meshy + cycle_pos)) * math.pi)

        grid = torch.stack((meshx, meshy), 2)
        grid = grid.unsqueeze(0)  # batch of one
        image_batch = image.unsqueeze(0)  # batch of one -> [1, c, h, w]
        warped = torch.nn.functional.grid_sample(image_batch, grid, mode="bilinear", padding_mode="border", align_corners=False)
        warped = warped.squeeze(0)  # [1, c, h, w] -> [c, h, w]
        image[:, :, :] = warped

    @with_metadata(speed=[1.0, 16.0],
                   asymmetry=[0.0, 0.5],
                   ripple_amplitude=[0.001, 0.01],
                   ripple_density1=[1.0, 100.0],
                   ripple_density2=[1.0, 100.0],
                   ripple_density3=[1.0, 100.0],
                   placement=["bottom", "top"],
                   _priority=7.0)
    def analog_runaway_hsync(self, image: torch.tensor, *,
                             speed: float = 8.0,
                             asymmetry: float = 0.1,
                             ripple_amplitude: float = 0.05,
                             ripple_density1: float = 4.0,
                             ripple_density2: Optional[float] = 13.0,
                             ripple_density3: Optional[float] = 27.0,
                             placement: str = "bottom") -> None:
        """[dynamic] Analog video signal distorted by a runaway hsync near the top or bottom edge.

        A bad video cable connection can do this, e.g. when connecting a game console to a display
        with an analog YPbPr component cable 10m in length. In reality, when I ran into this phenomenon,
        the distortion only occurred for near-white images, but as glitch art, it looks better if it's
        always applied at full strength.

        `speed`: At speed 1.0, a full cycle of the rippling effect completes every `image_height` frames.
                 So effectively the cycle position updates by `speed * (1 / image_height)` at each frame.
        `asymmetry`: Base strength for maximum distortion at the edge of the image. The image will ripple
                     around this base strength.

                     In units where the height and width of the image are both 2.0.
                     Positive values shift toward the right.
        `ripple_amplitude`: Strength variation, added on top of `asymmetry`.
        `ripple_density1`: Like `density` in `analog_rippling_hsync`, but in time. How many cycles the first
                           component wave completes per one cycle of the ripple effect.
        `ripple_density2`: Like `ripple_density1`, but for the second component wave.
                           Set to `None` or to 0.0 to disable the second component wave.
        `ripple_density3`: Like `ripple_density1`, but for the third component wave.
                           Set to `None` or to 0.0 to disable the third component wave.
        `placement`: one of "top", "bottom". Near which edge of the image to apply the maximal distortion.
                     The distortion then decays to zero, with a quadratic profile, in 1/8 of the image height.

        NOTE: "frame" here refers to the normalized frame number, at a reference of 25 FPS.
        """
        c, h, w = image.shape

        # Animation
        # FPS correction happens automatically, because `frame_no` is normalized to CALIBRATION_FPS.
        cycle_pos = (self.frame_no / h) * speed
        cycle_pos = cycle_pos - float(int(cycle_pos))  # fractional part
        cycle_pos *= 2.0  # full cycle = 2 units

        # Deformation
        # The spatial distort profile is a quadratic curve [0, 1], for 1/8 of the image height.
        meshy = self._meshy
        if placement == "top":
            spatial_distort_profile = (torch.clamp(meshy + 0.75, max=0.0) * 4.0)**2  # distort near y = -1
        else:  # placement == "bottom":
            spatial_distort_profile = (torch.clamp(meshy - 0.75, min=0.0) * 4.0)**2  # distort near y = +1
        ripple = math.sin(ripple_density1 * cycle_pos * math.pi)
        if ripple_density2:
            ripple += math.sin(ripple_density2 * cycle_pos * math.pi)
        if ripple_density3:
            ripple += math.sin(ripple_density3 * cycle_pos * math.pi)
        instantaneous_strength = (1.0 - ripple_amplitude) * asymmetry + ripple_amplitude * ripple
        # The minus sign: with positive strength, read coordinates toward the left -> shift the image toward the right.
        meshx = self._meshx - instantaneous_strength * spatial_distort_profile

        # Then just the usual incantation for applying a geometric distortion in Torch:
        grid = torch.stack((meshx, meshy), 2)
        grid = grid.unsqueeze(0)  # batch of one
        image_batch = image.unsqueeze(0)  # batch of one -> [1, c, h, w]
        warped = torch.nn.functional.grid_sample(image_batch, grid, mode="bilinear", padding_mode="border", align_corners=False)
        warped = warped.squeeze(0)  # [1, c, h, w] -> [c, h, w]
        image[:, :, :] = warped

    def _vhs_noise(self, image: torch.tensor, *,
                   height: int) -> torch.tensor:
        """Generate a horizontal band of noise that looks as if it came from a blank VHS tape.

        `height`: desired height of noise band, in pixels.

        Output is a tensor of shape `[1, height, w]`, where `w` is the width of `image`.
        """
        c, h, w = image.shape
        # This looks best if we randomize the alpha channel, too.
        noise_image = torch.rand(height, w, device=self.device, dtype=image.dtype).unsqueeze(0)  # [1, h, w]
        # Real VHS noise has horizontal runs of the same color, and the transitions between black and white are smooth.
        noise_image = torchvision.transforms.GaussianBlur((5, 1), sigma=2.0)(noise_image)
        return noise_image

    @with_metadata(strength=[0.1, 0.9],
                   unboost=[0.1, 10.0],
                   max_glitches=[1, 10],
                   min_glitch_height=[1, 3],
                   max_glitch_height=[3, 10],
                   hold_min=[1, 3],
                   hold_max=[3, 6],
                   name=["!ignore"],
                   _priority=8.0)
    def analog_vhsglitches(self, image: torch.tensor, *,
                           strength: float = 0.1,
                           unboost: float = 4.0,
                           max_glitches: int = 3,
                           min_glitch_height: int = 3, max_glitch_height: int = 6,
                           hold_min: int = 1, hold_max: int = 3,
                           name: str = "analog_vhsglitches0") -> None:
        """[dynamic] Damaged 1980s VHS video tape, with transient (per-frame) glitching lines.

        This leaves the alpha channel alone, so the effect only affects parts that already show something.
        This is an artistic interpretation that makes the effect less distracting when used with RGBA data.

        `strength`: How much to blend in noise.
        `unboost`: Use this to adjust the probability profile for the appearance of glitches.
                   The higher `unboost` is, the less probable it is for glitches to appear at all,
                   and there will be fewer of them (in the same video frame) when they do appear.
        `max_glitches`: Maximum number of glitches in the video frame.
        `min_glitch_height`, `max_glitch_height`: in pixels. The height is randomized separately for each glitch.
        `hold_min`, `hold_max`: in frames (at a reference of 25 FPS). Limits for the random time that the
                                filter holds one glitch pattern before randomizing the next one.

        `name`: Optional name for this filter instance in the chain. Used as cache key.
                If you have more than one `analog_vhsglitches` in the chain, they should have
                different names so that each one gets its own cache.
        """
        # Re-randomize the glitch noise image whenever enough frames have elapsed after last randomization, or the video stream size changes.
        c, h, w = image.shape
        size_changed = (h != self._prev_h or w != self._prev_w)
        if self.vhs_glitch_last_image[name] is None or size_changed or (int(self.frame_no) - int(self.vhs_glitch_last_frame_no[name])) >= self.vhs_glitch_interval[name]:
            n_glitches = torch.rand(1, device="cpu")**unboost  # unboost: increase probability of having none or few glitching lines
            n_glitches = int(max_glitches * n_glitches[0])
            if not n_glitches:
                vhs_glitch_image = VHS_GLITCH_BLANK  # use a nonce value instead of None to distinguish between "uninitialized" and "no glitches during current glitch interval"
                vhs_glitch_mask = None
            else:
                c, h, w = image.shape
                vhs_glitch_image = torch.zeros(1, h, w, dtype=image.dtype, device=self.device)  # monochrome
                vhs_glitch_mask = torch.zeros(1, h, w, dtype=image.dtype, device=self.device)  # alpha only
                glitch_start_lines = torch.rand(n_glitches, device="cpu")
                glitch_start_lines = [int((h - (max_glitch_height - 1)) * x) for x in glitch_start_lines]
                for line in glitch_start_lines:
                    glitch_height = torch.rand(1, device="cpu")
                    glitch_height = int(min_glitch_height + (max_glitch_height - min_glitch_height) * glitch_height[0])
                    vhs_glitch_image[0, line:(line + glitch_height), :] = self._vhs_noise(image, height=glitch_height)  # [1, h, w]
                    vhs_glitch_mask[0, line:(line + glitch_height), :] = 1.0  # mark the glitching lines for blending
            self.vhs_glitch_last_image[name] = vhs_glitch_image
            self.vhs_glitch_last_mask[name] = vhs_glitch_mask
            # Randomize time until next change of glitch pattern
            self.vhs_glitch_interval[name] = round(hold_min + float(torch.rand(1, device="cpu")[0]) * (hold_max - hold_min))
            self.vhs_glitch_last_frame_no[name] = self.frame_no
        else:
            vhs_glitch_image = self.vhs_glitch_last_image[name]
            vhs_glitch_mask = self.vhs_glitch_last_mask[name]

        if vhs_glitch_image is not VHS_GLITCH_BLANK:
            # Apply glitch to RGB only, so fully transparent parts stay transparent (important to make the effect less distracting).
            strength_field = strength * vhs_glitch_mask  # "field" as in physics, NOT as in CRT TV
            image[:3, :, :] = (1.0 - strength_field) * image[:3, :, :] + strength_field * vhs_glitch_image

    @with_metadata(base_offset=[0.0, 1.0],
                   max_dynamic_offset=[0.0, 1.0],
                   speed=[1.0, 16.0],
                   _priority=9.0)
    def analog_vhstracking(self, image: torch.tensor, *,
                           base_offset: float = 0.03,
                           max_dynamic_offset: float = 0.01,
                           speed: float = 8.0) -> None:
        """[dynamic] 1980s VHS tape with bad tracking.

        Image floats up and down, and a band of black and white noise appears at the bottom.

        Units like in `analog_rippling_hsync`:

        Offsets are given in units where the height of the image is 2.0.

        `speed`: At speed 1.0, the floating motion completes a full cycle every
                 `image_height` frames. So effectively the cycle position updates by
                 `speed * (1 / image_height)` at each frame.

        NOTE: "frame" here refers to the normalized frame number, at a reference of 25 FPS.
        """
        c, h, w = image.shape

        # Animation
        # FPS correction happens automatically, because `frame_no` is normalized to CALIBRATION_FPS.
        cycle_pos = (self.frame_no / h) * speed
        cycle_pos = cycle_pos - float(int(cycle_pos))  # fractional part
        cycle_pos *= 2.0  # full cycle = 2 units

        # Deformation - move image up/down
        yoffs = max_dynamic_offset * math.sin(cycle_pos * math.pi)
        meshy = self._meshy + yoffs
        meshx = self._meshx

        grid = torch.stack((meshx, meshy), 2)
        grid = grid.unsqueeze(0)  # batch of one
        image_batch = image.unsqueeze(0)  # batch of one -> [1, c, h, w]
        warped = torch.nn.functional.grid_sample(image_batch, grid, mode="bilinear", padding_mode="border", align_corners=False)
        warped = warped.squeeze(0)  # [1, c, h, w] -> [c, h, w]
        image[:, :, :] = warped

        # Noise from bad VHS tracking at bottom
        yoffs_pixels = int((yoffs / 2.0) * h)
        base_offset_pixels = int((base_offset / 2.0) * h)
        noise_pixels = yoffs_pixels + base_offset_pixels
        if noise_pixels > 0:
            image[:, -noise_pixels:, :] = self._vhs_noise(image, height=noise_pixels)
            # # Fade out toward left/right, since the character does not take up the full width.
            # # Works, but fails at reaching the iconic VHS look.
            # xx = torch.linspace(0, math.pi, w, dtype=image.dtype, device=self.device)
            # fade = torch.sin(xx)**2  # [w]
            # fade = fade.unsqueeze(0)  # [1, w]
            # image[3, -noise_pixels:, :] = fade

    @with_metadata(strength=[-0.1, 0.1],
                   unboost=[0.1, 10.0],
                   max_glitches=[1, 10],
                   min_glitch_height=[1, 20],
                   max_glitch_height=[30, 100],
                   hold_min=[1, 3],
                   hold_max=[3, 6],
                   name=["!ignore"],
                   _priority=10.0)
    def digital_glitches(self, image: torch.tensor, *,
                         strength: float = 0.05,
                         unboost: float = 4.0,
                         max_glitches: int = 3,
                         min_glitch_height: int = 20, max_glitch_height: int = 30,
                         hold_min: int = 1, hold_max: int = 3,
                         name: str = "digital_glitches0") -> None:
        """[dynamic] Glitchy digital video transport, with transient (per-frame) blocks of lines shifted left or right.

        `strength`: Amount of the horizontal shift, in units where 2.0 is the width of the full image.
                    Positive values shift toward the right.
                    For shifting both left and right, use two copies of the filter in your chain,
                    one with `strength > 0` and one with `strength < 0`.
        `unboost`: Use this to adjust the probability profile for the appearance of glitches.
                   The higher `unboost` is, the less probable it is for glitches to appear at all,
                   and there will be fewer of them (in the same video frame) when they do appear.
        `max_glitches`: Maximum number of glitches in the video frame.
        `min_glitch_height`, `max_glitch_height`: in pixels. The height is randomized separately for each glitch.
        `hold_min`, `hold_max`: in frames (at a reference of 25 FPS). Limits for the random time that the
                                filter holds one glitch pattern before randomizing the next one.

        `name`: Optional name for this filter instance in the chain. Used as cache key.
                If you have more than one `digital_glitches` in the chain, they should have
                different names so that each one gets its own cache.
        """
        # Re-randomize the glitch pattern whenever enough frames have elapsed after last randomization, or the video stream size changes.
        c, h, w = image.shape
        size_changed = (h != self._prev_h or w != self._prev_w)
        if self.digital_glitches_grid[name] is None or size_changed or (int(self.frame_no) - int(self.digital_glitches_last_frame_no[name])) >= self.digital_glitches_interval[name]:
            n_glitches = torch.rand(1, device="cpu")**unboost  # unboost: increase probability of having none or few glitching lines
            n_glitches = int(max_glitches * n_glitches[0])
            meshy = self._meshy
            meshx = self._meshx.clone()  # don't modify the original; also, make sure each element has a unique memory address
            if n_glitches:
                c, h, w = image.shape
                glitch_start_lines = torch.rand(n_glitches, device="cpu")
                glitch_start_lines = [int((h - (max_glitch_height - 1)) * x) for x in glitch_start_lines]
                for line in glitch_start_lines:
                    glitch_height = torch.rand(1, device="cpu")
                    glitch_height = int(min_glitch_height + (max_glitch_height - min_glitch_height) * glitch_height[0])
                    meshx[line:(line + glitch_height), :] -= strength
            digital_glitches_grid = torch.stack((meshx, meshy), 2)
            digital_glitches_grid = digital_glitches_grid.unsqueeze(0)  # batch of one
            self.digital_glitches_grid[name] = digital_glitches_grid
            # Randomize time until next change of glitch pattern
            self.digital_glitches_interval[name] = round(hold_min + float(torch.rand(1, device="cpu")[0]) * (hold_max - hold_min))
            self.digital_glitches_last_frame_no[name] = self.frame_no
        else:
            digital_glitches_grid = self.digital_glitches_grid[name]

        image_batch = image.unsqueeze(0)  # batch of one -> [1, c, h, w]
        warped = torch.nn.functional.grid_sample(image_batch, digital_glitches_grid, mode="bilinear", padding_mode="border", align_corners=False)
        warped = warped.squeeze(0)  # [1, c, h, w] -> [c, h, w]
        image[:, :, :] = warped

    # --------------------------------------------------------------------------------
    # CRT TV output

    def _rgb_to_hue(self, rgb: List[float]) -> float:
        """Convert an RGB color to an HSL hue, for use as `bandpass_hue` in `desaturate`.

        This uses a cartesian-to-polar approximation of the HSL representation,
        which is fine for hue detection, but should not be taken as an authoritative
        H component of an accurate RGB->HSL conversion.
        """
        R, G, B = rgb
        alpha = 0.5 * (2.0 * R - G - B)
        beta = 3.0**0.5 / 2.0 * (G - B)
        hue = math.atan2(beta, alpha) / (2.0 * math.pi)  # note atan2(0, 0) := 0
        hue = hue + 0.5  # convert from `[-0.5, 0.5)` to `[0, 1)`
        return hue

    # This filter is adapted from an old GLSL code I made for Panda3D 1.8 back in 2014.
    @with_metadata(strength=[0.0, 1.0],
                   tint_rgb=["!RGB"],  # hint the GUI that this parameter needs an RGB color picker
                   bandpass_reference_rgb=["!RGB"],
                   bandpass_q=[0.0, 1.0],
                   _priority=11.0)
    def desaturate(self, image: torch.tensor, *,
                   strength: float = 1.0,
                   tint_rgb: List[float] = [1.0, 1.0, 1.0],
                   bandpass_reference_rgb: List[float] = [1.0, 0.0, 0.0], bandpass_q: float = 0.0) -> None:
        """[static] Desaturation with bells and whistles.

        Does not touch the alpha channel.

        `strength`: Overall blending strength of the filter (0 is off, 1 is fully applied).

        `tint_rgb`: Color to multiplicatively tint the image with. Applied after desaturation.

                    Some example tint values:
                        Green monochrome computer monitor: [0.5, 1.0, 0.5]
                        Amber monochrome computer monitor: [1.0, 0.5, 0.2]
                        Sepia effect:                      [0.8039, 0.6588, 0.5098]
                        No tint (off; default):            [1.0, 1.0, 1.0]

        `bandpass_reference_rgb`: Reference color for hue to let through the bandpass.
                                  Use this to let e.g. red things bypass the desaturation.
                                  The hue is extracted automatically from the given color.

        `bandpass_q`: Hue bandpass band half-width, in (0, 1]. Hues farther away from `bandpass_hue`
                      than `bandpass_q` will be fully desaturated. The opposite colors on the color
                      circle are defined as having the largest possible hue difference, 1.0.

                      The shape of the filter is a quadratic spike centered on the reference hue,
                      and smoothly decaying to zero at `bandpass_q` away from the center.

                      The special value 0 (default) switches the hue bandpass code off,
                      saving some compute.
        """
        R = image[0, :, :]
        G = image[1, :, :]
        B = image[2, :, :]
        if bandpass_q > 0.0:  # hue bandpass enabled?
            # Calculate hue of each pixel, using a cartesian-to-polar approximation of the HSL representation.
            # An approximation is fine here, because we only use this for a hue detector.
            # This is faster and requires less branching than the exact hexagonal representation.
            desat_alpha = 0.5 * (2.0 * R - G - B)
            desat_beta = 3.0**0.5 / 2.0 * (G - B)
            desat_hue = torch.atan2(desat_beta, desat_alpha) / (2.0 * math.pi)  # note atan2(0, 0) := 0
            desat_hue += 0.5  # convert from `[-0.5, 0.5)` to `[0, 1)`
            # -> [h, w]

            # Determine whether to keep this pixel or desaturate (and by how much).
            #
            # Calculate distance of each pixel from reference hue, accounting for wrap-around.
            bandpass_hue = self._rgb_to_hue(bandpass_reference_rgb)
            desat_temp1 = torch.abs(desat_hue - bandpass_hue)
            desat_temp2 = torch.abs((desat_hue + 1.0) - bandpass_hue)
            desat_temp3 = torch.abs(desat_hue - (bandpass_hue + 1.0))
            desat_hue_distance = 2.0 * torch.minimum(torch.minimum(desat_temp1, desat_temp2),
                                                     desat_temp3)  # [0, 0.5] -> [0, 1]
            # -> [h, w]

            # How to interpret the following factor:
            #    0: far away from reference hue, pixel should be desaturated
            #    1: at reference hue, pixel should be kept as-is
            #
            # - Pixels with their hue at least `bandpass_q` away from `bandpass_hue` are fully desaturated.
            # - As distance falls below `bandpass_q`, a blend starts very gradually.
            # - As the hue difference approaches zero, the pixel is fully passed through.
            # - The 1.0 - ... together with the square makes a sharp spike at the reference hue.
            desat_diff2 = (1.0 - torch.clamp(desat_hue_distance / bandpass_q, min=0.0, max=1.0))**2

            # Gray pixels should always be desaturated, so that they get the tint applied.
            # In HSL computations, gray pixels have an arbitrary hue, usually red, so we must filter them out separately.
            # We can do this in YUV space.
            YUV = rgb_to_yuv(image[:3, :, :])
            Y = YUV[0, :, :]  # -> [h, w]
            U = YUV[1, :, :]  # -> [h, w]
            V = YUV[2, :, :]  # -> [h, w]
            notgray_threshold = 0.05
            urel = torch.clamp(torch.abs(U) / notgray_threshold, min=0.0, max=1.0)
            vrel = torch.clamp(torch.abs(V) / notgray_threshold, min=0.0, max=1.0)
            notgray = (urel**2 + vrel**2)**0.5  # [h, w]; 0: completely gray; 1: has at least some color
            desat_diff2 *= notgray

            strength_field = strength * (1.0 - desat_diff2)  # [h, w]; "field" as in physics, NOT as in CRT TV
        else:
            Y = luminance(image[:3, :, :])  # save some compute since in this case we don't need U and V
            strength_field = strength  # just a scalar!

        # Desaturate, then apply tint
        Y = Y.unsqueeze(0)  # -> [1, h, w]
        tint_color = torch.tensor(tint_rgb, device=self.device, dtype=image.dtype).unsqueeze(1).unsqueeze(2)  # [c, 1, 1]
        tinted_desat_image = Y * tint_color  # -> [c, h, w]

        # Final blend
        image[:3, :, :] = (1.0 - strength_field) * image[:3, :, :] + strength_field * tinted_desat_image

    @with_metadata(strength=[0.1, 0.9],
                   density=[1.0, 4.0],
                   speed=[1.0, 32.0],
                   _priority=12.0)
    def banding(self, image: torch.tensor, *,
                strength: float = 0.4,
                density: float = 2.0,
                speed: float = 16.0) -> None:
        """[dynamic] Bad analog video signal, with traveling brighter and darker bands.

        This simulates a CRT display as it looks when filmed on video without syncing.

        `strength`: maximum brightness factor
        `density`: how many banding cycles per full image height
        `speed`: band movement, in pixels per frame

        NOTE: "frame" here refers to the normalized frame number, at a reference of 25 FPS.
        """
        c, h, w = image.shape
        yy = torch.linspace(0, math.pi, h, dtype=image.dtype, device=self.device)

        # Animation
        # FPS correction happens automatically, because `frame_no` is normalized to CALIBRATION_FPS.
        cycle_pos = (self.frame_no / h) * speed
        cycle_pos = cycle_pos - float(int(cycle_pos))  # fractional part
        cycle_pos = 1.0 - cycle_pos  # -> motion from top toward bottom

        band_effect = torch.sin(density * yy + cycle_pos * math.pi)**2  # [h]
        band_effect = torch.unsqueeze(band_effect, 0)  # -> [1, h] = [c, h]
        band_effect = torch.unsqueeze(band_effect, 2)  # -> [1, h, 1] = [c, h, w]
        image[:3, :, :].mul_(1.0 + strength * band_effect)
        torch.clamp_(image, min=0.0, max=1.0)

    @with_metadata(field=[0, 1],
                   dynamic=[False, True],
                   double_size=[False, True],
                   channel=["Y", "A"],
                   strength=[0.1, 0.9],
                   _priority=13.0)
    def scanlines(self, image: torch.tensor, *,
                  field: int = 0,
                  dynamic: bool = True,
                  double_size: bool = True,
                  channel: str = "Y",
                  strength: float = 0.1) -> None:
        """[dynamic] CRT TV like scanlines.

        `field`: Which CRT field is dimmed at the first frame. 0 = top, 1 = bottom.
        `dynamic`: If `True`, the dimmed field will alternate each frame (top, bottom, top, bottom, ...)
                   for a more authentic CRT look (like Phosphor deinterlacer in VLC).
        `double_size`: If `True`, each "scanline" consists of two actual lines.
                       Useful to obtain a lofi look on a high-resolution monitor.
        `channel`: One of:
                     "Y": darken the luminance (converts to YUV and back; slower)
                     "A": darken the alpha channel (fast; makes the darkened lines translucent)
        `strength`: E.g. 0.25 -> dim to 75% brightness/alpha.

        NOTE: "frame" here refers to the normalized frame number, at a reference of 25 FPS.
        """
        if dynamic:
            start = (field + int(self.frame_no)) % 2
        else:
            start = field
        dim = 1.0 - strength
        if channel == "A":  # alpha
            if double_size:
                image[3, start::4, :].mul_(dim)
                image[3, start + 1::4, :].mul_(dim)
            else:
                image[3, start::2, :].mul_(dim)
        else:  # "Y", luminance
            image_yuv = rgb_to_yuv(image[:3, :, :])
            if double_size:
                image_yuv[0, start::4, :].mul_(dim)
                image_yuv[0, start + 1::4, :].mul_(dim)
            else:
                image_yuv[0, start::2, :].mul_(dim)
            image_rgb = yuv_to_rgb(image_yuv)
            image[:3, :, :] = image_rgb

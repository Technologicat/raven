"""Smoke and mirrors. Glitch artistry. Pixel-space postprocessing effects.

These effects work in linear intensity space, before gamma correction.

This module was originally released under AGPL in SillyTavern-Extras.
This module is relicensed by its author (Juha Jeronen) under the
2-clause BSD license.
"""

__all__ = ["Postprocessor",
           "isotropic_noise",
           "vhs_noise", "vhs_noise_pool"]

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


# ---------------------------------------------------------------------------
# Video noise generator — public API
# ---------------------------------------------------------------------------

def isotropic_noise(width: int, height: int, *,
                    device: torch.device,
                    dtype: torch.dtype = torch.float32,
                    sigma: float = 1.0,
                    double_size: bool = False) -> torch.Tensor:
    """Generate isotropic Gaussian-blurred uniform noise.

    Returns a ``[height, width]`` tensor in [0, 1]. When ``sigma > 0``,
    the raw uniform noise is blurred with the same kernel size in both
    dimensions — contrast with `vhs_noise`, which uses horizontal-only
    blur to mimic helical-scan artifacts.

    `sigma`:       Standard deviation for Gaussian blur (0 = no blur).
                   Works up to 3.0.
    `double_size`: Generate at half resolution and upscale 2×.
                   `width` and `height` specify the final output size.
    """
    gen_w, gen_h = ((width + 1) // 2, (height + 1) // 2) if double_size else (width, height)
    noise = torch.rand(gen_h, gen_w, device=device, dtype=dtype)
    if sigma > 0.0:
        noise = noise.unsqueeze(0)  # [h, w] -> [1, h, w]
        ks = _blur_kernel_size(sigma)
        noise = torchvision.transforms.GaussianBlur((ks, 1), sigma=sigma)(noise)
        noise = torchvision.transforms.GaussianBlur((1, ks), sigma=sigma)(noise)
        noise = noise.squeeze(0)  # -> [h, w]
    if double_size:
        noise = noise.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)[:height, :width]
    return noise


def vhs_noise(width: int, height: int, *,
              device: torch.device,
              dtype: torch.dtype = torch.float32,
              mode: str = "PAL",
              ntsc_chroma: str = "nearest",
              double_size: bool = False) -> torch.Tensor:
    """Generate VHS noise.

    `mode`:
        ``"PAL"``:  Monochrome luma noise. Returns ``[1, H, W]`` in [0, 1].
                    PAL's phase-alternation cancels chroma errors, so tape
                    noise is overwhelmingly in luminance.

        ``"NTSC"``: Three independent noise planes (Y, U, V). Returns
                    ``[3, H, W]``, where the `Y` channel is in [0, 1],
                    and the `U` and `V` channels are in [-0.5, 0.5].
                    NTSC has no phase-alternation error correction,
                    so tape speed variation and magnetic noise cause
                    chroma speckling and hue wander — the
                    "Never The Same Color" phenomenon.

    `ntsc_chroma` (NTSC only): how the 4:2:0 chroma planes are
    upscaled to luma resolution.
        ``"nearest"``:  Nearest-neighbour — blocky 2×2 chroma texels.
                        Fast, compresses well. Retro digital look.
        ``"bilinear"``: Bilinear interpolation — smooth chroma transitions.
                        Slower, larger encoded frames. More analog look.

    `double_size`: Generate noise at half resolution and upscale 2×.
                   Produces chunkier grain. `width` and `height` specify
                   the final output size either way.

    Both modes use horizontal-only Gaussian blur to mimic helical-scan
    artifacts. NTSC U/V planes get a wider kernel (lower chroma bandwidth
    on VHS — ~500 kHz vs. ~3 MHz for luma) and are 4:2:0 subsampled
    (generated at half luma resolution). This matches human vision's low
    chroma spatial acuity and dramatically improves compressibility of
    the rendered image.

    This is the single source of truth for the VHS noise recipe used
    across Raven — the postprocessor's glitch/tracking filters and the
    cherrypick placeholder tiles all derive from this.
    """
    gen_w, gen_h = ((width + 1) // 2, (height + 1) // 2) if double_size else (width, height)
    # Y channel (luma): fine-grained horizontal runs.
    noise_y = torch.rand(gen_h, gen_w, device=device, dtype=dtype).unsqueeze(0)  # [1, H, W]
    noise_y = torchvision.transforms.GaussianBlur((5, 1), sigma=2.0)(noise_y)

    if mode == "PAL":
        result = noise_y
    elif mode == "NTSC":
        # NTSC: independent U/V planes with coarser horizontal blur
        # (lower chroma bandwidth → wider, smoother runs).
        #
        # 4:2:0 chroma subsampling: generate U/V at half luma resolution and
        # upscale. Human vision has low chroma spatial acuity, and the coarser
        # chroma blocks dramatically improve delta-based compression (QOI) —
        # random per-pixel chroma is the main entropy source.
        chroma_h = (gen_h + 1) // 2
        chroma_w = (gen_w + 1) // 2

        noise_u = torch.rand(chroma_h, chroma_w, device=device, dtype=dtype).unsqueeze(0)
        noise_u = torchvision.transforms.GaussianBlur((11, 1), sigma=4.0)(noise_u)
        noise_u -= 0.5

        noise_v = torch.rand(chroma_h, chroma_w, device=device, dtype=dtype).unsqueeze(0)
        noise_v = torchvision.transforms.GaussianBlur((11, 1), sigma=4.0)(noise_v)
        noise_v -= 0.5

        # Upscale chroma to luma resolution.
        chroma = torch.cat([noise_u, noise_v], dim=0)  # [2, chroma_h, chroma_w]
        if ntsc_chroma == "bilinear":  # analog aesthetic
            chroma = torch.nn.functional.interpolate(
                chroma.unsqueeze(0), size=(gen_h, gen_w), mode="bilinear", align_corners=False,
            ).squeeze(0)
        else:  # ntsc_chroma == "nearest":  # retro digital aesthetic
            chroma = (chroma.repeat_interleave(2, dim=-1)
                            .repeat_interleave(2, dim=-2)[:, :gen_h, :gen_w])
        noise_u = chroma[0:1]  # [1, H, W]
        noise_v = chroma[1:2]  # [1, H, W]

        result = torch.cat([noise_y, noise_u, noise_v], dim=0)  # [3, H, W]
    else:
        raise ValueError(f"vhs_noise: unknown mode {mode!r}; expected 'PAL' or 'NTSC'")

    if double_size:
        result = result.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)[..., :height, :width]
    return result


def vhs_noise_pool(n: int, width: int, height: int, *,
                   device: torch.device,
                   dtype: torch.dtype = torch.float32,
                   tint: Tuple[float, float, float] = (0.92, 0.92, 1.0),
                   brightness: Tuple[float, float] = (0.04, 0.40),
                   mode: str = "PAL") -> List[torch.Tensor]:
    """Generate *n* unique tinted VHS noise tiles.

    Each tile is a ``[4, height, width]`` RGBA tensor in [0, 1], suitable
    for conversion to DPG texture format at the call site.

    `tint`: per-channel multiplier (R, G, B) applied to the brightness.
            Default is a subtle cool blue-gray.
    `brightness`: (lo, hi) range that the raw [0, 1] noise is mapped to.
                  Lower values keep the placeholders dark and subdued.
    `mode`: ``"PAL"`` (monochrome luma × tint) or ``"NTSC"`` (per-tile
            color variation via independent chroma noise). See `vhs_noise`.
    """
    lo, hi = brightness
    tr, tg, tb = tint
    tiles: List[torch.Tensor] = []
    for _ in range(n):
        noise = vhs_noise(width, height, device=device, dtype=dtype, mode=mode)

        if mode == "PAL":
            luma = lo + (hi - lo) * noise.squeeze(0)  # [H, W]
            r = luma * tr
            g = luma * tg
            b = luma * tb
        else:
            # NTSC: Y plane → brightness-mapped luma, U/V → small chroma offsets.
            # Each tile gets its own random color cast — some warm, some cool.
            noise_y = noise[0]  # [H, W]
            noise_u = noise[1]  # [H, W], pre-scaled by 0.5
            noise_v = noise[2]

            y = lo + (hi - lo) * noise_y                    # [H, W]
            u = 0.5 * noise_u                               # centered, -0.25 ... +0.25
            v = 0.5 * noise_v
            yuv = torch.stack([y, u, v], dim=0)              # [3, H, W]
            rgb = yuv_to_rgb(yuv, clamp=True)                # [3, H, W]
            r = rgb[0] * tr
            g = rgb[1] * tg
            b = rgb[2] * tb

        a = torch.ones(height, width, device=device, dtype=dtype)
        rgba = torch.stack([r, g, b, a], dim=0).clamp(0.0, 1.0)  # [4, H, W]
        tiles.append(rgba)
    return tiles

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

def _blur_kernel_size(sigma: float) -> int:
    """Gaussian blur kernel size for a given sigma.

    Uses the 2-sigma rule: the kernel reaches the point where the Gaussian
    has decayed to ~1.8% of its peak. This captures ~95% of the mass.
    Sufficient for a visual effects postprocessor; 3-sigma (99.7%) is overkill.

    Result is always odd (required by torchvision GaussianBlur).
    Minimum kernel size is 3 (torchvision requirement).
    """
    return min(21, max(3, 2 * math.ceil(2 * sigma) + 1))

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
        self._meshgrid_prev_h = None
        self._meshgrid_prev_w = None

        # FPS correction
        self.CALIBRATION_FPS = 25  # design FPS for dynamic effects (for automatic FPS correction)
        self.stream_start_timestamp = time.monotonic_ns()  # for updating frame counter reliably (no accumulation)
        self.frame_no = -1  # float, frame counter for *normalized* frame number *at CALIBRATION_FPS*
        self.last_frame_no = -1

        # Caches for individual dynamic effects
        self.zoom_data = defaultdict(lambda: None)
        self.ca_grid_cache = defaultdict(lambda: None)  # name -> {"scale": float, "grid_R": Tensor, "grid_B": Tensor}
        self.noise_last_image = defaultdict(lambda: None)
        self.noise_last_strength = defaultdict(lambda: -1.0)
        self.vhs_glitch_interval = defaultdict(lambda: 0.0)
        self.vhs_glitch_last_frame_no = defaultdict(lambda: 0.0)
        self.vhs_glitch_last_image = defaultdict(lambda: None)
        self.vhs_glitch_last_mask = defaultdict(lambda: None)
        self.vhs_headswitching_noise = defaultdict(lambda: None)
        self.vhs_tracking_noise = defaultdict(lambda: None)
        self.digital_glitches_interval = defaultdict(lambda: 0.0)
        self.digital_glitches_last_frame_no = defaultdict(lambda: 0.0)
        self.digital_glitches_grid = defaultdict(lambda: None)

    def _setup_meshgrid(self, h: int, w: int) -> None:
        """Compute base meshgrid for the geometric position of each pixel.

        Needed by filters that vary by position (e.g. `vignetting`) or deform
        the image (e.g. `analog_rippling_hsync`). Called automatically by
        `render_into` when the image size changes; can also be called directly
        to prepare a `Postprocessor` for invoking individual filters outside
        the render loop (e.g. in tests).

        We cache the meshgrid, because this postprocessor is typically applied
        to a video stream. As long as the image dimensions stay constant, we can
        re-use the same meshgrid.
        """
        logger.info(f"_setup_meshgrid: Computing pixel position tensors for image size {w}x{h}")
        with torch.inference_mode():
            self._yy = torch.linspace(-1.0, 1.0, h, dtype=self.dtype, device=self.device)
            self._xx = torch.linspace(-1.0, 1.0, w, dtype=self.dtype, device=self.device)
            self._meshy, self._meshx = torch.meshgrid((self._yy, self._xx), indexing="ij")
        self._meshgrid_prev_h = h
        self._meshgrid_prev_w = w
        logger.info("_setup_meshgrid: Pixel position tensors cached")

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
            # Ship the docstring too, so GUI editors can render per-parameter help dynamically
            # without needing to import the filter implementation client-side.
            param_info = {"defaults": settings,
                          "ranges": ranges,
                          "docstring": inspect.getdoc(func) or ""}
            filters.append((name, param_info))
        def rendering_priority(metadata_record):
            name, _ = metadata_record
            meth = getattr(cls, name)
            return meth.metadata["_priority"]
        return list(sorted(filters, key=rendering_priority))

    def render_into(self, image):
        """Apply current postprocess chain, modifying `image` in-place."""
        time_render_start = time.monotonic_ns()

        chain = self.chain  # read just once; other threads might reassign it while we're rendering
        if not chain:
            return

        c, h, w = image.shape
        if h != self._meshgrid_prev_h or w != self._meshgrid_prev_w:
            self._setup_meshgrid(h, w)

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
        with torch.inference_mode():
            for filter_name, settings in chain:
                apply_filter = getattr(self, filter_name)
                apply_filter(image, **settings)

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
        cached_grid = self.zoom_data[name]["grid"] if name in self.zoom_data else None
        size_changed = cached_grid is None or cached_grid.shape[-3] != h or cached_grid.shape[-2] != w
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
                   sigma=[0.1, 7.0],
                   _priority=0.0)
    def bloom(self, image: torch.tensor, *,
              threshold: float = 0.560,
              exposure: float = 0.842,
              sigma: float = 7.0) -> None:
        """[static] Bloom effect (fake HDR). Makes the image look brighter. Popular in early 2000s anime.

        Can also be used as just a camera exposure adjustment by setting `threshold=1.0` to disable the glow.

        Makes bright parts of the image bleed light into their surroundings, enhancing perceived contrast.
        Only makes sense when the avatar is rendered on a dark-ish background.

        `threshold`: How bright is bright. 0.0 is full black (all pixels glow), 1.0 is full white (bloom disabled).
                     Technically, this is true relative luminance, not luma, since we work in linear RGB space.
        `exposure`: Overall brightness of the output. Like in photography, higher exposure means brighter image,
                    saturating toward white.
        `sigma`: Standard deviation for the bloom blur. Larger values produce a wider glow. Works up to 7.0.
                 Recommended values:
                     7.0 - wide, dreamy early-2000s anime bloom.
                     1.6 - tighter, more modern glow.
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
            ks = _blur_kernel_size(sigma)
            brights = torchvision.transforms.GaussianBlur((ks, 1), sigma=sigma)(brights)  # blur along x
            brights = torchvision.transforms.GaussianBlur((1, ks), sigma=sigma)(brights)  # blur along y

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
                   name=["!ignore"],
                   _priority=1.0)
    def chromatic_aberration(self, image: torch.tensor, *,
                             scale: float = 0.005,
                             sigma: float = 1.0,
                             name: str = "chromatic_aberration0") -> None:
        """[static] Simulate the two types of chromatic aberration in a camera lens.

        Like everything else here, this is of course made of smoke and mirrors. We simulate the axial effect
        (index of refraction varying w.r.t. wavelength) by geometrically scaling the RGB channels individually,
        and the transverse effect (focal distance varying w.r.t. wavelength) by a gaussian blur.

        `scale`: Axial CA geometric distortion parameter.
        `sigma`: Transverse CA blur parameter. Works up to 3.0.

        Note that in a real lens:

        - Axial CA is typical at long focal lengths (e.g. tele/zoom lens).
        - Axial CA increases at high F-stops (low depth of field, i.e. sharp focus at all distances).
        - Transverse CA is typical at short focal lengths (e.g. macro lens).

        However, in an RGB postproc effect, it is useful to apply both together, to help hide the clear-cut red/blue bands
        resulting from the different geometric scalings of just three wavelengths (instead of a continuous spectrum, like
        a scene lit with natural light would have).

        See:
            https://en.wikipedia.org/wiki/Chromatic_aberration
        """
        # Axial CA: Shrink R (deflected less), pass G through (lens reference wavelength), enlarge B (deflected more).
        # Cache grids — they depend only on `scale` + meshgrid (meshgrid invalidates on resolution change).
        c, h, w = image.shape
        cached = self.ca_grid_cache[name]
        size_changed = cached is not None and (cached["grid_R"].shape[-3] != h or cached["grid_R"].shape[-2] != w)
        if cached is None or size_changed or scale != cached["scale"]:
            grid_R = torch.stack((self._meshx * (1.0 + scale), self._meshy * (1.0 + scale)), 2).unsqueeze(0)
            grid_B = torch.stack((self._meshx * (1.0 - scale), self._meshy * (1.0 - scale)), 2).unsqueeze(0)
            self.ca_grid_cache[name] = {"scale": scale, "grid_R": grid_R, "grid_B": grid_B}
        else:
            grid_R, grid_B = cached["grid_R"], cached["grid_B"]

        # Batch R+A and B+A to halve grid_sample and GaussianBlur calls.
        # R and A-via-grid_R use the same warp grid; same for B and A-via-grid_B.
        # GaussianBlur processes all channels, so blurring a 2-channel tensor does both at once.
        ks = _blur_kernel_size(sigma)
        blur_x = torchvision.transforms.GaussianBlur((ks, 1), sigma=sigma)
        blur_y = torchvision.transforms.GaussianBlur((1, ks), sigma=sigma)

        # Warp R and A together with grid_R (advanced indexing → copy, safe before writes)
        warped_RA = torch.nn.functional.grid_sample(image[[0, 3], :, :].unsqueeze(0),
                                                    grid_R, mode="bilinear", padding_mode="border", align_corners=False)
        warped_RA = blur_y(blur_x(warped_RA))  # transverse CA

        # Warp B and A together with grid_B
        warped_BA = torch.nn.functional.grid_sample(image[[2, 3], :, :].unsqueeze(0),
                                                    grid_B, mode="bilinear", padding_mode="border", align_corners=False)
        warped_BA = blur_y(blur_x(warped_BA))  # transverse CA

        image[0, :, :] = warped_RA[0, 0, :, :]
        # image[1, :, :] passed through as-is (G is the lens reference wavelength)
        image[2, :, :] = warped_BA[0, 0, :, :]
        # Alpha: average the R-warped, G-original, and B-warped alpha values (in-place, no temporary)
        image[3, :, :].add_(warped_RA[0, 1, :, :]).add_(warped_BA[0, 1, :, :]).mul_(1.0 / 3.0)

    @with_metadata(strength=[0.1, 0.5],
                   _priority=2.0)
    def vignetting(self, image: torch.tensor, *,
                   strength: float = 0.42) -> None:
        """[static] Simulate vignetting (less light hitting the corners of a film frame or CCD sensor).

        The profile used here is [cos(strength * d * pi)]**2, where `d` is the distance
        from the center, scaled such that `d = 1.0` is reached at the corners.
        Thus, at the midpoints of the frame edges, `d = 1 / sqrt(2) ~ 0.707`.

        `strength`: Falloff rate of the cosine-squared profile. 0 leaves the image unchanged;
                    larger values darken the corners more aggressively. At `strength = 0.5`
                    the corners go fully black.
        """
        euclidean_distance_from_center = (self._meshy**2 + self._meshx**2)**0.5 / 2**0.5  # [h, w]
        brightness = torch.cos(strength * euclidean_distance_from_center * math.pi)**2  # [h, w]
        brightness = torch.unsqueeze(brightness, 0)  # -> [1, h, w]
        image[:3, :, :] *= brightness

    # --------------------------------------------------------------------------------
    # Retouching / color grading

    def _rgb_to_hue(self, rgb: List[float]) -> float:
        """Convert an RGB color to an HSL hue, for use as `bandpass_hue` in `_desaturate_impl`.

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

    def _desaturate_impl(self, image: torch.tensor, *,
                         strength: float = 1.0,
                         tint_rgb: List[float] = [1.0, 1.0, 1.0],
                         bandpass_reference_rgb: List[float] = [1.0, 0.0, 0.0],
                         bandpass_q: float = 0.0) -> None:
        """Shared implementation for `desaturate` and `monochrome_display`."""
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

    # This filter is adapted from an old GLSL code I made for Panda3D 1.8 back in 2014.
    @with_metadata(strength=[0.0, 1.0],
                   tint_rgb=["!RGB"],  # hint the GUI that this parameter needs an RGB color picker
                   bandpass_reference_rgb=["!RGB"],
                   bandpass_q=[0.0, 1.0],
                   _priority=3.5)
    def desaturate(self, image: torch.tensor, *,
                   strength: float = 1.0,
                   tint_rgb: List[float] = [1.0, 1.0, 1.0],
                   bandpass_reference_rgb: List[float] = [1.0, 0.0, 0.0],
                   bandpass_q: float = 0.0) -> None:
        """[static] Desaturate the image, with optional hue bandpass and tint.

        This is an image retouching / color grading effect. It runs early in the
        chain (before noise and analog degradation), so the bandpass sees clean
        color data. For monochrome display simulation, see `monochrome_display`.

        Does not touch the alpha channel.

        `strength`: Overall blending strength of the filter (0 is off, 1 is fully applied).

        `tint_rgb`: Color to multiplicatively tint the image with. Applied after desaturation.

                    Some example tint values:
                        Sepia effect:           [0.8039, 0.6588, 0.5098]
                        No tint (off; default): [1.0, 1.0, 1.0]

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
        self._desaturate_impl(image, strength=strength, tint_rgb=tint_rgb,
                              bandpass_reference_rgb=bandpass_reference_rgb, bandpass_q=bandpass_q)

    # --------------------------------------------------------------------------------
    # General use

    def _noise_impl(self, image: torch.tensor, *,
                    strength: float = 0.3,
                    sigma: float = 1.0,
                    channel: str = "Y",
                    double_size: bool = True,
                    ntsc_chroma: str = "nearest",
                    name: str = "noise0") -> None:
        """Shared implementation for `noise` and `analog_vhs_noise`."""
        # Re-randomize the noise texture whenever the normalized frame number changes, or the video stream size changes.
        c, h, w = image.shape
        cached = self.noise_last_image[name]
        size_changed = cached is not None and (cached.shape[-2] != h or cached.shape[-1] != w)
        strength_changed = (strength != self.noise_last_strength[name])
        if cached is None or size_changed or strength_changed or int(self.frame_no) > int(self.last_frame_no):
            c, h, w = image.shape
            if channel.startswith("VHS_"):
                vhs_mode = channel.removeprefix("VHS_")  # "PAL" or "NTSC"
                noise_image = vhs_noise(w, h, device=self.device, dtype=image.dtype,
                                        mode=vhs_mode, ntsc_chroma=ntsc_chroma, double_size=double_size)
                # PAL: [1, H, W]; NTSC: [3, H, W] — stored as-is, distinguished at application time.
                # NTSC chroma is already 4:2:0 subsampled inside vhs_noise.
            else:
                noise_image = isotropic_noise(w, h, device=self.device, dtype=image.dtype,
                                              sigma=sigma, double_size=double_size)
            noise_image.mul_(strength)  # bake in current strength to save a few full-image multiplications during rendering
            self.noise_last_image[name] = noise_image
            self.noise_last_strength[name] = strength
        else:
            noise_image = cached
        base_multiplier = 1.0 - strength

        if channel == "A":  # alpha
            image[3, :, :].mul_(base_multiplier + noise_image)
            return

        image_yuv = rgb_to_yuv(image[:3, :, :])
        if channel == "VHS_PAL":
            image_yuv[0, :, :].mul_(base_multiplier + noise_image.squeeze(0))
        elif channel == "VHS_NTSC":
            image_yuv = rgb_to_yuv(image[:3, :, :])
            # Luma: multiplicative (same as PAL/Y modes).
            image_yuv[0, :, :].mul_(base_multiplier + noise_image[0])
            # Chroma: additive — random hue/saturation shift.
            image_yuv[1, :, :].add_(noise_image[1])
            image_yuv[1].clamp_(-0.5, 0.5)
            image_yuv[2, :, :].add_(noise_image[2])
            image_yuv[2].clamp_(-0.5, 0.5)
        else:  # "Y", luminance
            image_yuv = rgb_to_yuv(image[:3, :, :])
            image_yuv[0, :, :].mul_(base_multiplier + noise_image)
        image_rgb = yuv_to_rgb(image_yuv)
        image[:3, :, :] = image_rgb

    @with_metadata(strength=[0.1, 1.0],
                   sigma=[0.1, 3.0],
                   channel=["Y", "A"],
                   double_size=[False, True],
                   name=["!ignore"],
                   _priority=1.5)
    def noise(self, image: torch.tensor, *,
              strength: float = 0.3,
              sigma: float = 1.0,
              channel: str = "Y",
              double_size: bool = True,
              name: str = "noise0") -> None:
        """[dynamic] Sensor / film grain noise.

        Isotropic noise applied at the camera/capture stage. Runs early in the
        chain, before analog transport degradation. For VHS tape noise, see
        `analog_vhs_noise`.

        NOTE: At small values of `sigma`, this filter causes the video to use a lot of bandwidth
        during network transfer, because noise is impossible to compress.

        `strength`: How much noise to apply. 0 is no noise, 1 replaces the input image with noise.

                    The formula is:

                        out = in * ((1 - strength) + strength * noise_texture)

                    The filter is multiplicative, so it never brightens the image, and
                    never makes visible those pixels that are already fully translucent
                    (alpha = 0.0) in the input.

                    A larger `strength` makes the image darker/more translucent overall, because
                    then a larger portion of the luminance/alpha axis is reserved for the noise.

        `sigma`: Gaussian blur sigma for the noise, reducing its spatial frequency
                 (i.e. making larger and smoother "noise blobs"). Works up to 3.0.
                 0 = no blur (raw uniform noise).

        `channel`: Operation mode:
                     ``"Y"``: Modulate luminance (converts to YUV and back).
                     ``"A"``: Modulate alpha (fast).

        `double_size`: If ``True``, each noise texel covers a 2×2 pixel block (generated at
                       half resolution, nearest-neighbour upscaled). Improves compressibility.

        `name`: Cache key for this filter instance. Use different names for multiple
                instances in the chain.

        Suggested settings:
            Scifi hologram:     strength=0.1, sigma=0.0
            Analog television:  strength=0.2, sigma=2.0
        """
        self._noise_impl(image, strength=strength, sigma=sigma, channel=channel,
                         double_size=double_size, name=name)

    # --------------------------------------------------------------------------------
    # Lo-fi analog video

    @with_metadata(sigma=[0.1, 3.0],
                   _priority=5.0)
    def analog_lowres(self, image: torch.tensor, *,
                      sigma: float = 1.5) -> None:
        """[static] Low-resolution analog video signal, simulated by blurring.

        `sigma`: standard deviation of the Gaussian blur kernel, in pixels. Works up to 3.0.
        """
        ks = _blur_kernel_size(sigma)
        image[:, :, :] = torchvision.transforms.GaussianBlur((ks, 1), sigma=sigma)(image)  # blur along x
        image[:, :, :] = torchvision.transforms.GaussianBlur((1, ks), sigma=sigma)(image)  # blur along y

    @with_metadata(mode=["analog", "digital"],
                   subsampling=["4:2:0", "4:2:2"],
                   upscale=["nearest", "bilinear"],
                   double_size=[False, True],
                   sigma=[0.1, 7.0],
                   _priority=5.5)
    def chroma_subsample(self, image: torch.tensor, *,
                         mode: str = "analog",
                         subsampling: str = "4:2:0",
                         upscale: str = "nearest",
                         double_size: bool = True,
                         sigma: float = 4.0) -> None:
        """[static] Chroma subsampling — lo-fi YUV color resolution reduction.

        Reduces chrominance (color) resolution while keeping luminance (brightness)
        at full resolution. This is done by all analog and digital video systems,
        each in their own way, thus improving compression by exploiting the human
        visual system's lower spatial acuity for color than for brightness.

        `mode`: Which chroma-reduction mechanism to model. The two options below cover the two main real-world mechanisms.

        ``"analog"``:  Bandwidth-limited chroma, as in analog broadcast (NTSC/PAL).
                       Applies a horizontal low-pass filter to the U and V channels,
                       simulating the limited chroma bandwidth of analog video
                       (~500 kHz chroma vs. ~3 MHz luma on VHS). The `subsampling`,
                       `upscale`, and `double_size` parameters are ignored in this mode.

        ``"digital"``: Block-subsampled chroma, as in digital codecs (DV, MPEG, H.264).
                       Downsamples the chroma planes by bilinear filtering, then
                       upsamples back to luma resolution. Produces the characteristic
                       blocky color fringing of early digital video.

        `subsampling` (digital mode only):
            ``"4:2:0"``: Half resolution in both dimensions (isotropic).
                         Standard in MPEG, H.264, most consumer digital video.
            ``"4:2:2"``: Half resolution horizontally only (anisotropic).
                         Used in DV, professional MPEG-2, broadcast contribution feeds.

        `upscale` (digital mode only): How subsampled chroma is reconstructed.
            ``"nearest"``:  Nearest-neighbour — blocky chroma edges. Retro digital look.
            ``"bilinear"``: Bilinear interpolation — smoother color transitions.
                            Closer to what a proper decoder does.

        `double_size` (digital mode only): If ``True``, each chroma block covers twice
                      the normal area — effectively 4:1:0 or 4:1:1 instead of 4:2:0 or
                      4:2:2. Exaggerates the blocky digital look for artistic effect.

        `sigma` (analog mode only): Standard deviation for the horizontal chroma blur,
                in pixels. Larger values simulate lower chroma bandwidth. Works up to 7.0
                (kernel size saturates around sigma 5, but the effect is already heavy).

        Note: uses BT.709 (HDTV) YCbCr, not BT.601 (SDTV). The visual difference
        is subtle for this kind of lo-fi effect.
        """
        c, h, w = image.shape
        image_yuv = rgb_to_yuv(image[:3, :, :])  # [3, h, w]

        if mode == "analog":
            ks = _blur_kernel_size(sigma)
            image_yuv[1:3, :, :] = torchvision.transforms.GaussianBlur((ks, 1), sigma=sigma)(image_yuv[1:3, :, :])
        elif mode == "digital":
            chroma = image_yuv[1:3, :, :]  # [2, h, w]

            if subsampling == "4:2:0":
                sub_h = (h + 1) // 2
                sub_w = (w + 1) // 2
            elif subsampling == "4:2:2":
                sub_h = h
                sub_w = (w + 1) // 2
            else:
                raise ValueError(f"chroma_subsample: unknown subsampling {subsampling!r}; expected '4:2:0' or '4:2:2'")

            if double_size:
                sub_h = (sub_h + 1) // 2
                sub_w = (sub_w + 1) // 2

            # Downsample chroma by strided slicing (pick every Nth sample).
            # Strided selection intentionally drops samples, introducing aliasing
            # that makes the subsampling visible.
            stride_h = (h + sub_h - 1) // sub_h  # 2 when subsampled, 1 when not
            stride_w = (w + sub_w - 1) // sub_w
            chroma_sub = chroma[:, ::stride_h, ::stride_w]  # [2, ~sub_h, ~sub_w]

            # # Downsample chroma bilinearly — real encoders filter, not drop.
            # # The quality of this approach is too good for the subsampling
            # # to be visible.
            # chroma_sub = torch.nn.functional.interpolate(
            #     chroma.unsqueeze(0), size=(sub_h, sub_w), mode="bilinear", align_corners=False,
            # ).squeeze(0)  # [2, sub_h, sub_w]

            # Reconstruct chroma to luma resolution
            if upscale == "bilinear":
                chroma_up = torch.nn.functional.interpolate(
                    chroma_sub.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False,
                ).squeeze(0)
            elif upscale == "nearest":
                # Compute per-axis repeat factor from the actual subsampled size
                repeat_h = (h + sub_h - 1) // sub_h
                repeat_w = (w + sub_w - 1) // sub_w
                chroma_up = (chroma_sub
                             .repeat_interleave(repeat_w, dim=-1)
                             .repeat_interleave(repeat_h, dim=-2)[:, :h, :w])
            else:
                raise ValueError(f"chroma_subsample: unknown upscale {upscale!r}; expected 'nearest' or 'bilinear'")

            image_yuv[1:3, :, :] = chroma_up
        else:
            raise ValueError(f"chroma_subsample: unknown mode {mode!r}; expected 'analog' or 'digital'")

        image[:3, :, :] = yuv_to_rgb(image_yuv)

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

        We superpose three sine waves (with per-wave `amplitudeN` / `densityN` parameters)
        to make the ripple pattern look irregular rather than periodic.

        `speed`: At speed 1.0, a wave of `density = 1.0` completes a full cycle every
                 `image_height` frames. So effectively the cycle position updates by
                 `speed * (1 / image_height)` at each frame. "Frame" here refers to the
                 normalized frame number, at a reference of 25 FPS.

        `amplitude1`, `amplitude2`, `amplitude3`: peak horizontal displacement of each
                wave component, in units where the image width (and height) is 2.0.
                Set a component's amplitude to 0 to disable that wave.

        `density1`, `density2`, `density3`: spatial frequency of each wave component
                along the vertical axis, in cycles per image height. E.g. density 2.0
                means two full waves fit into the image height. Larger values = finer
                ripples. Using three different density values gives the shimmering,
                non-repeating look.
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

    # VHS tape effects, in physical order:
    #   1. Tape medium noise — intrinsic magnetic noise, baked into the recorded signal.
    #   2. Glitches — localized tape surface damage / oxide dropout.
    #   3. Head switching — mechanical artifact from the rotary drum switching heads during playback.
    #   4. Tracking — playback speed/alignment error, tape doesn't track the helical scan correctly.

    @with_metadata(strength=[0.1, 1.0],
                   mode=["PAL", "NTSC"],
                   double_size=[False, True],
                   ntsc_chroma=["nearest", "bilinear"],
                   name=["!ignore"],
                   _priority=7.5)
    def analog_vhs_noise(self, image: torch.tensor, *,
                         strength: float = 0.3,
                         mode: str = "PAL",
                         double_size: bool = True,
                         ntsc_chroma: str = "nearest",
                         name: str = "analog_vhs_noise0") -> None:
        """[dynamic] VHS tape noise.

        Anisotropic noise characteristic of the VHS magnetic medium. Runs in the
        analog transport stage, after hsync artifacts but before glitches and
        head switching. For sensor / film grain noise, see `noise`.

        `strength`: How much noise to apply. 0 is no noise, 1 replaces the input with noise.

                    Luma is multiplicative (darkens). In NTSC mode, chroma is additive
                    (random hue/saturation shift).

        `mode`: VHS color system:
                  ``"PAL"``:  Luma noise only — horizontal runs, sharp vertical
                              transitions. Chroma untouched (PAL's phase alternation
                              cancels chroma errors).
                  ``"NTSC"``: Luma noise plus *additive* chroma noise on Cb/Cr — the
                              "Never The Same Color" phenomenon.

        `double_size`: If ``True``, each noise texel covers a 2×2 pixel block (generated at
                       half resolution, nearest-neighbour upscaled). Dramatically improves
                       compressibility. Particularly useful for ``"NTSC"``, where chroma noise
                       otherwise destroys delta-based compression.

        `ntsc_chroma` (``"NTSC"`` only): how the 4:2:0 chroma noise is upscaled.
                       ``"nearest"``:  Blocky 2×2 chroma texels. Fast, compresses well.
                                       Retro digital aesthetic.
                       ``"bilinear"``: Smooth chroma transitions. Slower, larger frames.
                                       More analog look — faithful to real tape noise.

        `name`: Cache key for this filter instance. Use different names for multiple
                instances in the chain.
        """
        self._noise_impl(image, strength=strength, channel=f"VHS_{mode}",
                         double_size=double_size, ntsc_chroma=ntsc_chroma, name=name)

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
        cached = self.vhs_glitch_last_image[name]
        size_changed = cached is not None and cached is not VHS_GLITCH_BLANK and (cached.shape[-2] != h or cached.shape[-1] != w)
        if cached is None or size_changed or (int(self.frame_no) - int(self.vhs_glitch_last_frame_no[name])) >= self.vhs_glitch_interval[name]:
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
                    vhs_glitch_image[0, line:(line + glitch_height), :] = vhs_noise(w, glitch_height, device=self.device, dtype=image.dtype)  # [1, h, w]
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

    @with_metadata(speed=[1.0, 32.0],
                   height=[0.01, 0.05],
                   max_displacement=[0.01, 0.1],
                   noise_blend=[0.0, 1.0],
                   double_size=[False, True],
                   name=["!ignore"],
                   _priority=8.5)
    def analog_vhs_headswitching(self, image: torch.Tensor, *,
                                 speed: float = 8.0,
                                 height: float = 0.035,
                                 max_displacement: float = 0.03,
                                 noise_blend: float = 0.075,
                                 double_size: bool = True,
                                 name: str = "analog_vhs_headswitching0") -> None:
        """[dynamic] VHS head switching noise at the bottom edge.

        Simulates the horizontal displacement and static visible at the
        bottom of a VHS frame during the rotary head switch. CRT televisions
        hid this behind overscan; it becomes visible on full-raster digital
        captures. The most immediately recognizable VHS artifact.

        Each scanline in the affected region is shifted by a different
        horizontal amount, increasing toward the bottom — producing the
        characteristic squiggly shear. VHS noise is blended in with an
        increasing envelope so the bottom rows dissolve into static.

        `height`: Height of the affected region as a fraction of image
                  height. At typical render sizes (512–1024), 0.015 ≈
                  8–15 pixels.
        `max_displacement`: Maximum horizontal shift amplitude. Units:
                            image width = 2.0 (same as `analog_rippling_hsync`).
                            Much larger than rippling hsync (~0.001) — this is
                            a clearly visible horizontal shear, not subtle edge
                            ripple.
        `noise_blend`: Maximum noise blend fraction at the very bottom of the
                       affected region. Ramps from 0 at the top of the band to
                       `noise_blend` at the bottom. 0.0 = pure displaced image
                       everywhere, 1.0 = pure noise at the bottom edge.
        `double_size`: Generate noise at half resolution and upscale 2×.
                       Produces chunkier, more authentic VHS grain.
        `speed`: Animation speed. Same convention as other analog filters
                 (cycle position updates by `speed / image_height` per frame).

        `name`: Optional name for this filter instance in the chain. Used as cache key.
                If you have more than one `analog_vhs_headswitching` in the chain, they should have
                different names so that each one gets its own cache.

        NOTE: "frame" refers to the normalized frame number, at 25 FPS.
        """
        c, h, w = image.shape
        n_rows = max(1, int(height * h))

        # Animation — continuous cycle position, no wrapping needed (feeds into sin).
        cycle_pos = (self.frame_no / h) * speed

        # Displacement profile: three sine waves at incommensurate frequencies,
        # modulated by an increasing envelope. The spatial variable is
        # normalized row position [0, 1] within the affected band, so the
        # wobble character is resolution-independent. The time terms at
        # different rates make adjacent frames correlated but not identical.
        row_indices = torch.arange(n_rows, dtype=self.dtype, device=self.device)
        envelope = row_indices / max(1, n_rows - 1)  # [0..1] ramp
        normalized_rows = envelope  # same thing — reuse for clarity

        wobble = (torch.sin((3.0 * normalized_rows + 2.0 * cycle_pos) * math.pi)
                  + 0.7 * torch.sin((7.0 * normalized_rows + 5.0 * cycle_pos) * math.pi)
                  + 0.4 * torch.sin((13.0 * normalized_rows + 11.0 * cycle_pos) * math.pi))
        displacement = envelope * max_displacement * wobble  # [n_rows]

        # Apply displacement to the bottom rows of meshx only.
        meshx = self._meshx.clone()
        meshy = self._meshy
        meshx[-n_rows:, :] += displacement.unsqueeze(1)  # broadcast across x

        grid = torch.stack((meshx, meshy), 2)
        grid = grid.unsqueeze(0)
        image_batch = image.unsqueeze(0)
        warped = torch.nn.functional.grid_sample(image_batch, grid, mode="bilinear", padding_mode="border", align_corners=False)
        warped = warped.squeeze(0)
        image[:, :, :] = warped

        # Blend VHS noise into the affected region, ramping from 0 at the
        # top of the band to `noise_blend` at the bottom — same envelope
        # shape as the displacement.
        if noise_blend > 0.0:
            # Head switching reads between tracks, not valid chroma data, so the noise is luma-only
            # on both PAL and NTSC systems. We get this by calling `vhs_noise` in its PAL mode.
            cached = self.vhs_headswitching_noise[name]
            size_changed = cached is not None and cached.shape[-1] != w
            if cached is None or size_changed or int(self.frame_no) > int(self.last_frame_no):
                noise = vhs_noise(w, n_rows, device=self.device, dtype=image.dtype, double_size=double_size)
                self.vhs_headswitching_noise[name] = noise
            else:
                noise = cached
            blend = (envelope * noise_blend).unsqueeze(1)  # [n_rows, 1] → broadcasts to [n_rows, w]
            blend = blend.unsqueeze(0)  # [1, n_rows, w] → broadcasts to [3, n_rows, w]
            image[:3, -n_rows:, :] = (1.0 - blend) * image[:3, -n_rows:, :] + blend * noise

    @with_metadata(base_offset=[0.0, 1.0],
                   max_dynamic_offset=[0.0, 1.0],
                   speed=[1.0, 16.0],
                   double_size=[False, True],
                   name=["!ignore"],
                   _priority=9.0)
    def analog_vhstracking(self, image: torch.tensor, *,
                           base_offset: float = 0.03,
                           max_dynamic_offset: float = 0.01,
                           speed: float = 8.0,
                           double_size: bool = True,
                           name: str = "analog_vhstracking0") -> None:
        """[dynamic] 1980s VHS tape with bad tracking.

        Image floats up and down, and a band of black and white noise appears at the bottom.

        Units like in `analog_rippling_hsync`:

        Offsets are given in units where the height of the image is 2.0.

        `base_offset`: Baseline height of the noise band, measured upward from the bottom
                       of the image. The band always hugs the bottom edge; this parameter
                       sets its nominal thickness, on top of which the floating animation
                       modulates it.

        `max_dynamic_offset`: Peak amplitude of the vertical floating motion (sine wave).
                              The image drifts up and down by ±this value over each cycle,
                              and the noise band grows and shrinks in sync (taller when the
                              image is displaced upward, shorter when displaced downward).

        `speed`: At speed 1.0, the floating motion completes a full cycle every
                 `image_height` frames. So effectively the cycle position updates by
                 `speed * (1 / image_height)` at each frame.
        `double_size`: Generate noise at half resolution and upscale 2×.
                       Produces chunkier, more authentic VHS grain.

        `name`: Optional name for this filter instance in the chain. Used as cache key.
                If you have more than one `analog_vhstracking` in the chain, they should have
                different names so that each one gets its own cache.

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
            cached = self.vhs_tracking_noise[name]
            size_changed = cached is not None and cached.shape[-1] != w
            if cached is None or size_changed or int(self.frame_no) > int(self.last_frame_no):
                # Generate at current noise_pixels height. Cached for sub-frame reuse.
                noise = vhs_noise(w, noise_pixels, device=self.device, dtype=image.dtype, double_size=double_size)
                self.vhs_tracking_noise[name] = noise
            else:
                noise = cached
                # noise_pixels may jitter ±1 within the same integer frame due to float frame_no.
                # Slice if the cached tensor is taller, regenerate if shorter.
                cached_h = noise.shape[1]
                if cached_h < noise_pixels:
                    noise = vhs_noise(w, noise_pixels, device=self.device, dtype=image.dtype, double_size=double_size)
                    self.vhs_tracking_noise[name] = noise
                elif cached_h > noise_pixels:
                    noise = noise[:, :noise_pixels, :]
            image[:, -noise_pixels:, :] = noise
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
        cached_grid = self.digital_glitches_grid[name]
        size_changed = cached_grid is not None and (cached_grid.shape[-3] != h or cached_grid.shape[-2] != w)
        if cached_grid is None or size_changed or (int(self.frame_no) - int(self.digital_glitches_last_frame_no[name])) >= self.digital_glitches_interval[name]:
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
    # Display output

    @with_metadata(alpha=[0.1, 0.9],
                   _priority=10.5)
    def translucent_display(self, image: torch.tensor, *,
                            alpha: float = 0.9) -> None:
        """[static] Translucent display (e.g. scifi hologram).

        Multiplicatively adjusts the alpha channel.

        `alpha`: Opacity multiplier in [0, 1]. 1.0 leaves the image unchanged;
                 lower values make the whole character progressively more see-through.
                 Only affects pixels that already have non-zero alpha, so the outline
                 of the character stays sharp.
        """
        image[3, :, :].mul_(alpha)

    @with_metadata(strength=[0.0, 1.0],
                   tint_rgb=["!RGB"],
                   _priority=11.0)
    def monochrome_display(self, image: torch.tensor, *,
                           strength: float = 1.0,
                           tint_rgb: List[float] = [1.0, 1.0, 1.0]) -> None:
        """[static] Monochrome display simulation.

        Desaturates the image as seen through a monochrome display. Runs late
        in the chain (after noise and analog degradation), so transport artifacts
        like NTSC chroma noise are correctly collapsed into luminance — as a
        real monochrome display would do. For color grading / bandpass
        desaturation, see `desaturate`.

        Does not touch the alpha channel.

        `strength`: Overall blending strength of the filter (0 is off, 1 is fully applied).

        `tint_rgb`: Color to multiplicatively tint the image with. Applied after desaturation.

                    Some example tint values:
                        Green monochrome CRT:              [0.5, 1.0, 0.5]
                        Amber monochrome CRT:              [1.0, 0.5, 0.2]
                        No tint (white phosphor; default): [1.0, 1.0, 1.0]
        """
        self._desaturate_impl(image, strength=strength, tint_rgb=tint_rgb)

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

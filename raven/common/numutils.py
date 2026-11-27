"""Numerical utilities.

This module is licensed under the 2-clause BSD license, to facilitate integration anywhere.
"""

__all__ = ["clamp", "nonanalytic_smooth_transition", "psi",
           "maximin_latin_square",
           "needs_histogram_adjustment", "linear_scale", "log_scale", "histogram_adjustment", "histogram_adjusted_heights",
           "decade_ticks", "spread_labels"]

import logging
logger = logging.getLogger(__name__)

import functools
import itertools
import math
from collections.abc import Callable
from typing import Union

import numpy as np

# --------------------------------------------------------------------------------
# Numerical utilities

def clamp(x: float, ell: float = 0.0, u: float = 1.0) -> float:  # not the manga studio
    """Clamp value `x` between `ell` and `u`. Return clamped value."""
    return min(max(ell, x), u)

def nonanalytic_smooth_transition(x: Union[float, np.array], m: float = 1.0) -> Union[float, np.array]:  # from `extrafeathers.pdes.numutil`
    """Non-analytic smooth transition from 0 to 1, on interval x ∈ [0, 1].

    Here "non-analytic" = "does not have a single global Taylor series".
    The function is smooth, as in "an element of the class C∞".

    The transition is reflection-symmetric through the point (1/2, 1/2).
    This is essentially a type of an S-curve.

    Outside the interval:
        s(x, m) = 0  for x < 0
        s(x, m) = 1  for x > 1

    The parameter `m` controls the steepness of the transition region.
    Larger `m` packs the transition closer to `x = 1/2`, making it
    more abrupt (although technically, still infinitely smooth).

    `m` is passed to `psi`, which see.
    """
    # Here `psi` is constructed so that we don't need bounds checking.
    p = psi(x, m)
    return p / (p + psi(1.0 - x, m))

def psi(x: Union[float, np.array], m: float = 1.0) -> Union[float, np.array]:  # from `extrafeathers.pdes.numutil`
    """Building block for non-analytic smooth functions.

        psi(x, m) := exp(-1 / x^m) χ(0, ∞)(x)

    where χ is the indicator function (1 if x is in the set, 0 otherwise).

    Suppresses divide by zero warnings and errors, so can be evaluated
    also at `x = 0`.

    This is the helper function used in the construction of the standard
    mollifier in PDE theory.
    """
    # Compute-then-mask: -1/0 = -inf, exp(-inf) = 0, then (x > 0.0)
    # zeroes the masked part.  NumPy still emits a RuntimeWarning for
    # the divide-by-zero during the computation even though the masked
    # result is correct, so we suppress it locally.
    #
    # Use `np.errstate`, not `warnings.filterwarnings`: the latter was
    # previously attempted with `module="__main__"` but never actually
    # matched in practice, because numpy emits the warning from its own
    # internal module (not from the caller's `__main__`), so the filter
    # silently failed to apply.  `np.errstate` is numpy's own mechanism
    # for toggling float-error handling within a dynamic extent and
    # doesn't depend on the warnings-module machinery at all.
    with np.errstate(divide="ignore", invalid="ignore"):
        try:
            return np.exp(-1.0 / x**m) * (x > 0.0)
        except (ZeroDivisionError, OverflowError):  # for scalar x
            return 0.0


# --------------------------------------------------------------------------------
# Layout

# Beyond this many points the square is constructed rather than searched for. `n!` is fine at 8 (40320,
# computed once and cached) and is not at 12.
MAX_SEARCHED_LATIN_SQUARE = 8


@functools.lru_cache(maxsize=None)
def maximin_latin_square(n: int, step_x: float = 1.0, step_y: float = 1.0) -> tuple[int, ...]:
    """Which column each of `n` rows takes in a maximin Latin square: one point per row and one per column.

    Of all the permutations, the one whose closest pair of points is as far apart as it can be, measured with
    columns `step_x` apart and rows `step_y` apart; ties are broken by the next-closest pair and so on down,
    and then lexicographically. So 2 points give `(0, 1)`, 3 give `(0, 2, 1)` and 4 give `(1, 3, 0, 2)`. The
    identity, a staircase, is a legal Latin square and is what the maximin criterion rules out: points placed
    this way read as scattered rather than stacked or gridded. It is the two-dimensional case of Latin
    hypercube sampling (McKay, Beckman and Conover, 1979): https://en.wikipedia.org/wiki/Latin_hypercube_sampling

    Above `MAX_SEARCHED_LATIN_SQUARE` points the search would be factorial, so the answer is constructed
    instead: odd columns first, then even. That is a permutation and spread out, but not the maximin optimum.
    """
    if n <= 1:
        return tuple(range(n))
    if n > MAX_SEARCHED_LATIN_SQUARE:
        return tuple(list(range(1, n, 2)) + list(range(0, n, 2)))

    def spread(permutation: tuple[int, ...]) -> list[float]:
        points = [(column * step_x, row * step_y) for row, column in enumerate(permutation)]
        return sorted(math.dist(a, b) for a, b in itertools.combinations(points, 2))

    return max(itertools.permutations(range(n)), key=spread)


def needs_histogram_adjustment(values, *, low: float, high: float) -> bool:
    """Whether `histogram_adjusted_heights` compresses `values` rather than scaling them linearly.

    It does when the largest value is more than `high / low` times the smallest positive one: linearly, that one
    would come out shorter than `low`, the shortest height there is to show it with.
    """
    positive = np.asarray(values, dtype=float)
    positive = positive[positive > 0]
    return bool(positive.size) and positive.max() / positive.min() > high / low


def _positive_range(values, caller: str) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    positive = values[values > 0]
    if not positive.size:
        raise ValueError(f"{caller}: `values` holds no positive value, so there is no scale to build.")
    return float(positive.min()), float(positive.max())


def linear_scale(values, *, high: float) -> tuple[Callable, Callable]:
    """A linear display scale for `values`, as `(forward, inverse)` functions: `value / max * high`.

    `forward` maps a value, or an array of them, to a display height, and `inverse` maps a height back; a value of zero
    or less maps to zero. The sibling of `log_scale` and `histogram_adjustment`, for switching between the three.

    `values` must hold a positive value.
    """
    _vmin, vmax = _positive_range(values, "linear_scale")

    def forward(x):
        return np.clip(np.asarray(x, dtype=float), 0.0, None) / vmax * high

    def inverse(h):
        return np.clip(np.asarray(h, dtype=float), 0.0, None) * vmax / high
    return forward, inverse


def log_scale(values, *, low: float, high: float) -> tuple[Callable, Callable]:
    """A logarithmic display scale for `values`, as `(forward, inverse)` functions.

    The smallest positive value maps to `low` and the largest to `high`, logarithmically in between. A value of zero or
    less maps to zero, and one between zero and the smallest positive value linearly to a height between zero and
    `low`, as `histogram_adjustment` has it. Where all the positive values are equal, the scale is `linear_scale`.

    `values` must hold a positive value. `0 < low < high`.
    """
    vmin, vmax = _positive_range(values, "log_scale")
    if vmax == vmin:
        return linear_scale(values, high=high)
    data_range = math.log(vmax / vmin)
    display_range = math.log(high / low)

    def forward(x):
        x = np.asarray(x, dtype=float)
        above = x >= vmin
        position = np.log(np.where(above, x, vmin) / vmin) / data_range
        return np.where(above, low * np.exp(display_range * position), np.where(x > 0, x / vmin * low, 0.0))

    def inverse(h):
        h = np.asarray(h, dtype=float)
        above = h >= low
        position = np.log(np.where(above, h, low) / low) / display_range
        return np.where(above, vmin * np.exp(data_range * position), np.where(h > 0, h / low * vmin, 0.0))
    return forward, inverse


def histogram_adjustment(values, *, low: float, high: float, nbins: int = 20) -> tuple[Callable, Callable]:
    """The display scale `histogram_adjusted_heights` places `values` on, as `(forward, inverse)` functions.

    `forward` maps a value, or an array of them, to a display height, and `inverse` maps a height back. The scale is
    built from `values`, and applies to any value: one of zero or less maps to zero, one between zero and the smallest
    positive value linearly to a height between zero and `low`, and one above the largest to `high`. Where the
    positive values span no more than `high / low`, the scale is linear, `value / max * high`.

    For an axis drawn on that scale, as matplotlib's `FuncScale` takes it; `decade_ticks` places its labels.

    `values`, `low`, `high`, `nbins`: as for `histogram_adjusted_heights`. `values` must hold a positive value.
    """
    vmin, _vmax = _positive_range(values, "histogram_adjustment")
    values = np.asarray(values, dtype=float)
    positive = values[values > 0]
    if not needs_histogram_adjustment(positive, low=low, high=high):  # fits as it is
        return linear_scale(positive, high=high)

    logs = np.log(positive)
    data_range = logs.max() - logs.min()
    display_range = math.log(high / low)

    counts, edges = np.histogram(logs, bins=nbins)
    pdf = counts.astype(float)
    # Trim each bin to the ceiling, in rounds, since trimming lowers the total and with it the ceiling. The ceiling is
    # the bin's share of the total at which equalizing would map it more steeply than a linear mapping does: a bin
    # holding more than that would have its contrast exaggerated. The tolerance is the paper's 0.025 of the total,
    # tightened to 0.005 of the largest bin, as the image version in `raven.common.video.postprocessor` has it.
    tol = 0.005 * pdf.max()
    delta_b = data_range / nbins
    while True:
        total = pdf.sum()
        if total < tol:  # trimmed away entirely: equalize the measured histogram instead
            pdf = counts.astype(float)
            break
        ceiling = total * delta_b / display_range
        trimmings = np.clip(pdf - ceiling, 0.0, None).sum()
        pdf = np.minimum(pdf, ceiling)
        if trimmings <= tol:
            break
    cdf = np.concatenate([[0.0], np.cumsum(pdf)])
    cdf /= cdf[-1]

    # Each value's position in the cumulative distribution, interpolated within its bin, sets its height on a
    # logarithmic display scale from `low` to `high`. Below the smallest value the scale continues linearly to zero.
    def forward(x):
        x = np.asarray(x, dtype=float)
        out = np.where(x > 0, x / vmin * low, 0.0)
        above = x >= vmin
        position = np.interp(np.log(np.where(above, x, vmin)), edges, cdf)
        return np.where(above, low * np.exp(display_range * position), out)

    # An empty bin is a flat stretch of `cdf`, where the inverse is not unique; `np.interp` picks a value from it.
    def inverse(h):
        h = np.asarray(h, dtype=float)
        out = np.where(h > 0, h / low * vmin, 0.0)
        above = h >= low
        position = np.log(np.where(above, h, low) / low) / display_range
        return np.where(above, np.exp(np.interp(position, cdf, edges)), out)
    return forward, inverse


def histogram_adjusted_heights(values, *, low: float, high: float, nbins: int = 20) -> np.ndarray:
    """Map positive `values` to display heights in `[low, high]`, compressing a wide range data-adaptively.

    The histogram adjustment of Ward Larson, Rushmeier and Piatko (1997, *A Visibility Matching Tone Reproduction
    Operator for High Dynamic Range Scenes*, https://doi.org/10.1109/2945.646233), for a list of numbers rather than
    an image's luminances: where the values span more than `high / low`, each is placed by where it falls in the
    distribution of their logarithms, so that crowded parts of the range get the contrast, while no stretch of it is
    steepened past a linear mapping. Where they span no more than that, the heights are linear, `value / max * high`.

    `values`: numbers, one per height. A value of zero or less maps to zero.
    `low`, `high`: the smallest and largest display height, `0 < low < high`; their ratio is the display range.
    `nbins`: the bins of the histogram of logarithms.

    Returns a float array shaped as `values`. The order of the values is kept: a larger value is never shorter.
    `histogram_adjustment` gives the same scale as functions, for drawing an axis on it.
    """
    values = np.asarray(values, dtype=float)
    if not (values > 0).any():
        return np.zeros_like(values)
    forward, _inverse = histogram_adjustment(values, low=low, high=high, nbins=nbins)
    return forward(values)


def decade_ticks(forward: Callable, low: float, high: float, *,
                 spacing: float, min_count: int = 16, include: tuple = (), resolution: float = 0.0) -> list[float]:
    """Tick values for an axis on a nonlinear scale, as many as fit without crowding.

    `forward`: maps a value to its position along the axis, as a fraction of its length.
    `low`, `high`: the positive range of values to place ticks in.
    `spacing`: the least distance between two ticks, as a fraction of the axis: a label's height, for a vertical axis.
    `min_count`: once this many ticks are placed, no further kind of tick is tried.
    `include`: values that get a tick first, such as the ends of the data.
    `resolution`: the finest step a tick may fall on; 1 for counts, which keeps ticks on whole numbers.

    Ticks are tried in rounds, at each power of ten, then at five times one, then two times, then three, four, six,
    seven, eight and nine times one; a tick is placed where it is at least `spacing` from every tick already placed.
    So a stretch the scale compresses gets fewer ticks, and a 1-2-5 sequence appears where there is room for it. Last,
    each decade is subdivided a step further, its fives and then its twos ("1500", then "1200", "1400", ...), for a
    stretch no round number falls in, as between 1000 and the end of data at 1344.

    Returns the tick values, sorted.
    """
    # A port of the colour bar labelling of `hdrplot.m`, Juha Jeronen's HDR plotter for the SAVU project (2010).
    placed: list[tuple[float, float]] = []  # (position, value)

    def scaled(n: int, k: int) -> float:  # n * 10**k, dividing for k < 0: 12 / 10 is 1.2 where 12 * 0.1 is not
        return n * 10.0 ** k if k >= 0 else n / 10.0 ** -k

    def try_place(value: float) -> None:
        y = float(forward(value))
        if all(abs(y - other) >= spacing for other, _value in placed):
            placed.append((y, value))

    for value in include:
        try_place(float(value))
    kmin, kmax = math.floor(math.log10(low)), math.ceil(math.log10(high))
    for multipliers in ((1,), (5,), (2,), (3, 4, 6, 7, 8, 9)):
        if len(placed) >= min_count:
            break
        for k in range(kmin, kmax + 1):
            for m in multipliers:
                value = scaled(m, k)
                if low <= value <= high:
                    try_place(value)
    # One level finer: n * 10**(k - 1) for n from 11 to 99, the multiples of five first, then the even ones. Stopping
    # there keeps a wide gap from filling with every step of the finer level.
    for keep in (lambda n: n % 5 == 0, lambda n: n % 2 == 0):
        if len(placed) >= min_count:
            break
        for k in range(kmin, kmax + 1):
            step = 10.0 ** (k - 1)
            if step < resolution:
                continue
            for n in range(11, 100):
                value = scaled(n, k - 1)
                if n % 10 and keep(n) and low <= value <= high:
                    try_place(value)
    return sorted(value for _y, value in placed)


def spread_labels(positions, *, spacing: float, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    """Positions for labels as near as possible to `positions`, at least `spacing` apart and within `[low, high]`.

    For labels along an axis that are allowed to move off their ticks, with a leader line back, rather than be dropped
    where they crowd. Minimizes the sum of squared moves, exactly.

    `positions`: the labels' wanted positions, sorted ascending. The result keeps their order.

    Raises `ValueError` if the labels cannot fit: `(len(positions) - 1) * spacing > high - low`.
    """
    # With q[i] = p[i] - i * spacing, the gaps become q nondecreasing, and the problem is isotonic regression of
    # positions[i] - i * spacing, which pool-adjacent-violators solves exactly in one pass. The bounds on the first and
    # last label bound every q to the same interval, and clipping an isotonic fit to an interval keeps it optimal.
    y = np.asarray(positions, dtype=float)
    n = len(y)
    if n == 0:
        return y.copy()
    if (n - 1) * spacing > high - low:
        raise ValueError(f"spread_labels: {n} labels at least {spacing} apart do not fit in [{low}, {high}].")
    target = y - spacing * np.arange(n)
    blocks: list[list[float]] = []  # [sum, count]
    for t in target:
        blocks.append([t, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            total, count = blocks.pop()
            blocks[-1][0] += total
            blocks[-1][1] += count
    q = np.concatenate([np.full(count, total / count) for total, count in blocks])
    q = np.clip(q, low, high - (n - 1) * spacing)
    return q + spacing * np.arange(n)

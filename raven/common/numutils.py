"""Numerical utilities.

This module is licensed under the 2-clause BSD license, to facilitate integration anywhere.
"""

__all__ = ["clamp", "nonanalytic_smooth_transition", "psi",
           "maximin_latin_square",
           "histogram_adjusted_heights"]

import logging
logger = logging.getLogger(__name__)

import functools
import itertools
import math
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
    """
    values = np.asarray(values, dtype=float)
    heights = np.zeros_like(values)
    positive = values > 0
    if not positive.any():
        return heights
    logs = np.log(values[positive])
    data_range = logs.max() - logs.min()
    display_range = math.log(high / low)
    if data_range <= display_range:  # fits as it is
        heights[positive] = values[positive] / values[positive].max() * high
        return heights

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
    # logarithmic display scale from `low` to `high`.
    position = np.interp(logs, edges, cdf)
    heights[positive] = low * np.exp(display_range * position)
    return heights

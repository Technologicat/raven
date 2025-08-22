"""Numerical utilities.

This module is licensed under the 2-clause BSD license, to facilitate integration anywhere.
"""

__all__ = ["clamp", "nonanalytic_smooth_transition", "psi"]

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import warnings

import numpy as np

# --------------------------------------------------------------------------------
# Numerical utilities

def clamp(x, ell=0.0, u=1.0):  # not the manga studio
    """Clamp value `x` between `ell` and `u`. Return clamped value."""
    return min(max(ell, x), u)

def nonanalytic_smooth_transition(x, m=1.0):  # from `extrafeathers.pdes.numutil`
    """Non-analytic smooth transition from 0 to 1, on interval x ∈ [0, 1].

    The transition is reflection-symmetric through the point (1/2, 1/2).

    Outside the interval:
        s(x, m) = 0  for x < 0
        s(x, m) = 1  for x > 1

    The parameter `m` controls the steepness of the transition region.
    Larger `m` packs the transition closer to `x = 1/2`, making it
    more abrupt (although technically, still infinitely smooth).

    `m` is passed to `psi`, which see.
    """
    p = psi(x, m)
    return p / (p + psi(1.0 - x, m))

def psi(x, m=1.0):  # from `extrafeathers.pdes.numutil`
    """Building block for non-analytic smooth functions.

        psi(x, m) := exp(-1 / x^m) χ(0, ∞)(x)

    where χ is the indicator function (1 if x is in the set, 0 otherwise).

    Suppresses divide by zero warnings and errors, so can be evaluated
    also at `x = 0`.

    This is the helper function used in the construction of the standard
    mollifier in PDE theory.
    """
    with warnings.catch_warnings():  # for NumPy arrays
        warnings.filterwarnings(action="ignore",
                                message="^divide by zero .*$",
                                category=RuntimeWarning,
                                module="__main__")
        try:
            return np.exp(-1.0 / x**m) * (x > 0.0)
        except ZeroDivisionError:  # for scalar x
            return 0.0

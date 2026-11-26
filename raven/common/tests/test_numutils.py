"""Unit tests for raven.common.numutils."""

import math

import numpy as np
import pytest

from raven.common import numutils
from raven.common.numutils import clamp, nonanalytic_smooth_transition, psi


# ---------------------------------------------------------------------------
# clamp
# ---------------------------------------------------------------------------

class TestClamp:
    def test_within_range_unchanged(self):
        assert clamp(0.5) == 0.5

    def test_below_lower_bound(self):
        assert clamp(-1.0) == 0.0

    def test_above_upper_bound(self):
        assert clamp(2.0) == 1.0

    def test_at_lower_bound(self):
        assert clamp(0.0) == 0.0

    def test_at_upper_bound(self):
        assert clamp(1.0) == 1.0

    def test_custom_bounds(self):
        assert clamp(5, ell=2, u=8) == 5
        assert clamp(1, ell=2, u=8) == 2
        assert clamp(10, ell=2, u=8) == 8

    def test_equal_bounds(self):
        assert clamp(0.0, ell=0.5, u=0.5) == 0.5
        assert clamp(1.0, ell=0.5, u=0.5) == 0.5

    def test_integer_inputs(self):
        assert clamp(5, ell=0, u=10) == 5
        assert clamp(-1, ell=0, u=10) == 0


# ---------------------------------------------------------------------------
# psi
# ---------------------------------------------------------------------------

class TestPsi:
    def test_zero_returns_zero(self):
        assert psi(0.0) == 0.0

    def test_negative_returns_zero(self):
        assert psi(-1.0) == 0.0
        assert psi(-100.0) == 0.0

    def test_one(self):
        assert psi(1.0) == pytest.approx(math.exp(-1.0))

    def test_large_x_approaches_one(self):
        # psi(x) = exp(-1/x) → 1 as x → ∞
        assert psi(1000.0) == pytest.approx(1.0, abs=1e-3)

    def test_numpy_array(self):
        x = np.array([0.0, 0.5, 1.0, 2.0])
        result = psi(x)
        assert result[0] == 0.0
        assert result[2] == pytest.approx(math.exp(-1.0))

    def test_custom_m(self):
        # psi(1, m) = exp(-1) for any m
        assert psi(1.0, m=2.0) == pytest.approx(math.exp(-1.0))
        # psi(0.5, m=1) = exp(-2)
        assert psi(0.5, m=1.0) == pytest.approx(math.exp(-2.0))

    def test_numpy_array_with_negatives(self):
        x = np.array([-1.0, 0.0, 1.0])
        result = psi(x)
        assert result[0] == 0.0
        assert result[1] == 0.0
        assert result[2] == pytest.approx(math.exp(-1.0))


# ---------------------------------------------------------------------------
# nonanalytic_smooth_transition
# ---------------------------------------------------------------------------

class TestNonanalyticSmoothTransition:
    def test_zero(self):
        assert nonanalytic_smooth_transition(0.0) == 0.0

    def test_one(self):
        assert nonanalytic_smooth_transition(1.0) == pytest.approx(1.0)

    def test_half_is_half(self):
        # Reflection symmetry through (1/2, 1/2)
        assert nonanalytic_smooth_transition(0.5) == pytest.approx(0.5)

    def test_negative_returns_zero(self):
        assert nonanalytic_smooth_transition(-0.5) == 0.0

    def test_above_one_returns_one(self):
        assert nonanalytic_smooth_transition(1.5) == pytest.approx(1.0)

    def test_monotonically_increasing(self):
        xs = np.linspace(0.01, 0.99, 50)
        values = [nonanalytic_smooth_transition(x) for x in xs]
        for i in range(len(values) - 1):
            assert values[i] < values[i + 1]

    def test_symmetry(self):
        # s(x) + s(1-x) = 1
        for x in [0.1, 0.2, 0.3, 0.4]:
            assert (nonanalytic_smooth_transition(x) +
                    nonanalytic_smooth_transition(1.0 - x)) == pytest.approx(1.0)

    def test_steeper_m(self):
        # Larger m → steeper transition. At x=0.25, larger m should give smaller value.
        val_m1 = nonanalytic_smooth_transition(0.25, m=1.0)
        val_m3 = nonanalytic_smooth_transition(0.25, m=3.0)
        assert val_m3 < val_m1


class TestMaximinLatinSquare:
    def test_the_small_answers_are_the_ones_drawn_by_hand(self):
        from raven.common.numutils import maximin_latin_square
        assert maximin_latin_square(2) == (0, 1)
        assert maximin_latin_square(3) == (0, 2, 1)
        assert maximin_latin_square(4) == (1, 3, 0, 2)

    @pytest.mark.parametrize("n", [0, 1, 5, 8, 9, 15])
    def test_every_answer_is_a_permutation(self, n):
        # One point per row and one per column, including above the size where the search gives way to a construction.
        from raven.common.numutils import maximin_latin_square
        assert sorted(maximin_latin_square(n)) == list(range(n))

    def test_the_answer_is_never_the_staircase(self):
        from raven.common.numutils import maximin_latin_square
        for n in range(3, 12):
            assert maximin_latin_square(n) != tuple(range(n))


class TestHistogramAdjustedHeights:
    def test_a_range_that_fits_is_linear(self):
        assert list(numutils.histogram_adjusted_heights([1, 2, 5], low=1, high=20)) == [4.0, 8.0, 20.0]

    def test_a_wide_range_is_compressed_in_order(self):
        values = [3000, 2500, 400, 300, 200, 100, 50, 20, 10, 5, 1]
        heights = numutils.histogram_adjusted_heights(values, low=1, high=21)
        assert heights[0] == pytest.approx(21) and heights[-1] == pytest.approx(1)
        assert all(a >= b for a, b in zip(heights, heights[1:])), "a larger value is never shorter"
        linear = [v / 3000 * 21 for v in values]
        assert sum(h >= 3 for h in heights) > sum(h >= 3 for h in linear), \
            "more of the values clear 3 px than linearly, which is what the compression is for"

    def test_zero_maps_to_zero(self):
        assert list(numutils.histogram_adjusted_heights([0, 0], low=1, high=10)) == [0.0, 0.0]
        assert numutils.histogram_adjusted_heights([0, 4, 2], low=1, high=10)[0] == 0.0

    def test_the_test_is_the_ratio_not_the_size(self):
        # Large counts that crowd nowhere fit a linear scale: the largest is five times the smallest.
        assert not numutils.needs_histogram_adjustment([0, 1000, 2000, 3000, 4000, 5000], low=1, high=6)
        assert numutils.needs_histogram_adjustment([1, 5000], low=1, high=21), "the 1 would vanish linearly"

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


class TestHistogramAdjustment:
    # Counts spanning four decades, crowded at the low end: compressed on a display range of 100.
    values = np.array([1, 2, 2, 3, 3, 3, 4, 5, 6, 8, 10, 15, 40, 300, 9000], dtype=float)

    def test_forward_is_the_heights(self):
        forward, _inverse = numutils.histogram_adjustment(self.values, low=1.0, high=100.0)
        assert np.allclose(forward(self.values), numutils.histogram_adjusted_heights(self.values, low=1.0, high=100.0))

    @pytest.mark.parametrize("values", [values, np.array([3.0, 5.0, 40.0])], ids=["compressed", "linear"])
    def test_inverse_undoes_forward(self, values):
        forward, inverse = numutils.histogram_adjustment(values, low=1.0, high=100.0)
        assert np.allclose(inverse(forward(values)), values)

    def test_the_scale_is_monotone_everywhere_and_zero_at_zero(self):
        forward, _inverse = numutils.histogram_adjustment(self.values, low=1.0, high=100.0)
        grid = np.concatenate([[0.0, 0.25, 0.5], np.geomspace(1.0, 20000.0, 200)])
        heights = forward(grid)
        assert heights[0] == 0.0
        assert np.all(np.diff(heights) >= 0.0)
        assert heights[1] == pytest.approx(0.25), "below the smallest value, linear down to zero"

    def test_no_positive_value_is_an_error(self):
        with pytest.raises(ValueError):
            numutils.histogram_adjustment([0.0, 0.0], low=1.0, high=100.0)


class TestDecadeTicks:
    @staticmethod
    def log_axis(x):  # 1 to 1000 on a log axis
        return np.log10(x) / 3.0

    def test_where_there_is_room_the_ticks_run_one_two_five(self):
        ticks = numutils.decade_ticks(self.log_axis, 1.0, 1000.0, spacing=0.04, min_count=10)
        assert ticks[:10] == [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]

    def test_where_there_is_not_only_the_decades(self):
        """Control for the test above: the same axis with labels too tall for the twos and fives."""
        assert numutils.decade_ticks(self.log_axis, 1.0, 1000.0, spacing=0.2) == [1.0, 10.0, 100.0, 1000.0]

    def test_a_compressed_stretch_gets_fewer_ticks(self):
        def axis(x):  # 1-10 squeezed into the bottom 5%
            x = np.asarray(x, dtype=float)
            return np.where(x <= 10, 0.05 * np.log10(x), 0.05 + 0.95 * np.log10(x / 10) / 2)
        ticks = numutils.decade_ticks(axis, 1.0, 1000.0, spacing=0.04)
        assert sum(1 for t in ticks if t < 10) <= 1
        assert sum(1 for t in ticks if t > 10) >= 6
        positions = sorted(float(axis(t)) for t in ticks)
        assert min(np.diff(positions)) >= 0.04

    def test_included_values_come_first(self):
        ticks = numutils.decade_ticks(self.log_axis, 1.0, 1000.0, spacing=0.2, include=(1.0, 700.0))
        assert 700.0 in ticks and 1000.0 not in ticks, "the included end wins over the decade crowding it"


class TestLinearAndLogScales:
    values = np.array([0.0, 2.0, 20.0, 2000.0])

    def test_log_scale_puts_the_ends_at_low_and_high(self):
        forward, inverse = numutils.log_scale(self.values, low=1.0, high=100.0)
        assert forward(2.0) == pytest.approx(1.0) and forward(2000.0) == pytest.approx(100.0)
        assert forward(np.sqrt(2.0 * 2000.0)) == pytest.approx(10.0), "the geometric mean of the ends sits halfway"
        assert forward(0.0) == 0.0 and forward(1.0) == pytest.approx(0.5), "linear below the smallest value"
        assert np.allclose(inverse(forward(self.values)), self.values)

    def test_linear_scale(self):
        forward, inverse = numutils.linear_scale(self.values, high=100.0)
        assert forward(1000.0) == pytest.approx(50.0)
        assert np.allclose(inverse(forward(self.values)), self.values)

    def test_equal_values_give_a_linear_log_scale(self):
        forward, _inverse = numutils.log_scale([5.0, 5.0], low=1.0, high=100.0)
        assert forward(5.0) == pytest.approx(100.0)


class TestDecadeTicksRefinement:
    @staticmethod
    def top_of_decade(v):  # 1000 to 1344, linear
        return (np.asarray(v, dtype=float) - 1000.0) / 344.0

    def test_a_stretch_no_round_number_falls_in_gets_a_finer_tick(self):
        assert numutils.decade_ticks(self.top_of_decade, 1000.0, 1344.0, spacing=0.2) == [1000.0, 1200.0]

    def test_resolution_keeps_ticks_on_whole_numbers(self):
        def axis(v):
            return (np.asarray(v, dtype=float) - 1.0) / 0.344
        assert numutils.decade_ticks(axis, 1.0, 1.344, spacing=0.2, resolution=1.0) == [1.0]
        assert numutils.decade_ticks(axis, 1.0, 1.344, spacing=0.2) == [1.0, 1.2], \
            "without a resolution the finer tick is placed, so this fixture can see the difference"


class TestSpreadLabels:
    def test_labels_with_room_stay_put(self):
        assert np.allclose(numutils.spread_labels([0.1, 0.5, 0.9], spacing=0.1), [0.1, 0.5, 0.9])

    def test_two_crowded_labels_part_evenly_around_where_they_were(self):
        assert np.allclose(numutils.spread_labels([0.48, 0.52], spacing=0.1), [0.45, 0.55])

    def test_labels_crowding_an_end_are_pushed_inward(self):
        assert np.allclose(numutils.spread_labels([0.95, 0.99, 1.0], spacing=0.1), [0.8, 0.9, 1.0])

    def test_too_many_labels_is_an_error(self):
        with pytest.raises(ValueError):
            numutils.spread_labels([0.1, 0.2, 0.3], spacing=0.6)

    def test_the_result_is_the_least_squares_layout(self):
        scipy_optimize = pytest.importorskip("scipy.optimize")
        rng = np.random.default_rng(1)
        wanted = np.sort(rng.uniform(0.0, 1.0, 12))
        spacing = 0.07
        got = numutils.spread_labels(wanted, spacing=spacing)
        constraints = [{"type": "ineq", "fun": (lambda p, i=i: p[i + 1] - p[i] - spacing)} for i in range(len(wanted) - 1)]
        reference = scipy_optimize.minimize(lambda p: np.sum((p - wanted) ** 2), wanted, constraints=constraints,
                                            bounds=[(0.0, 1.0)] * len(wanted), method="SLSQP", options={"ftol": 1e-12})
        assert np.all(np.diff(got) >= spacing - 1e-12)
        assert np.sum((got - wanted) ** 2) == pytest.approx(reference.fun, rel=1e-6)

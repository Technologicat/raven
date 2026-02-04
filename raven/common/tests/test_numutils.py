"""Unit tests for raven.common.numutils."""

import math

import numpy as np
import pytest

from raven.common.numutils import clamp, nonanalytic_smooth_transition, psi, si_prefix


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


# ---------------------------------------------------------------------------
# si_prefix
# ---------------------------------------------------------------------------

class TestSiPrefix:
    def test_small_number(self):
        assert si_prefix(42) == "42.00"

    def test_zero(self):
        assert si_prefix(0) == "0.00"

    def test_kilo(self):
        assert si_prefix(1000) == "1.00 K"

    def test_kilo_fractional(self):
        assert si_prefix(1500) == "1.50 K"

    def test_mega(self):
        assert si_prefix(1_500_000) == "1.50 M"

    def test_giga(self):
        assert si_prefix(1e9) == "1.00 G"

    def test_tera(self):
        assert si_prefix(1e12) == "1.00 T"

    def test_peta(self):
        assert si_prefix(1e15) == "1.00 P"

    def test_exa(self):
        # Beyond P, falls through to E
        assert si_prefix(1e18) == "1.00 E"

    def test_just_below_kilo(self):
        assert si_prefix(999) == "999.00"

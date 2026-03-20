"""Tests for raven.common.smoothvalue — framerate-independent animation."""

from raven.common.smoothvalue import SmoothValue, SmoothInt
from raven.common.tests import approx


# ---------------------------------------------------------------------------
# Tests: SmoothValue
# ---------------------------------------------------------------------------

class TestSmoothValue:
    """Test the SmoothValue animation class.

    The behavioral contract is:
    - Monotonic approach toward target (no overshoot).
    - Eventually reaches the target.
    - Higher rate converges faster.
    The tests don't assume any specific interpolation.
    """

    def test_initial_value(self):
        """Initial current and target are both set to the given value."""
        sv = SmoothValue(value=5.0)
        assert sv.current == 5.0
        assert sv.target == 5.0

    def test_set_target(self):
        """Setting target doesn't change current."""
        sv = SmoothValue(value=0.0)
        sv.target = 10.0
        assert sv.target == 10.0
        assert sv.current == 0.0

    def test_set_immediate(self):
        """set_immediate snaps both current and target."""
        sv = SmoothValue(value=0.0)
        sv.set_immediate(10.0)
        assert sv.current == 10.0
        assert sv.target == 10.0

    def test_is_animating(self):
        """is_animating is True when current differs from target."""
        sv = SmoothValue(value=0.0)
        assert not sv.is_animating()

        sv.target = 10.0
        assert sv.is_animating()

    def test_is_not_animating_after_immediate(self):
        """set_immediate should not leave the value animating."""
        sv = SmoothValue(value=0.0)
        sv.set_immediate(10.0)
        assert not sv.is_animating()

    def test_update_moves_toward_target(self):
        """A single update moves current toward target."""
        sv = SmoothValue(value=0.0, rate=0.5)
        sv.target = 10.0

        initial = sv.current
        sv.update(dt=0.1)

        assert sv.current > initial
        assert sv.current < sv.target

    def test_update_monotonic(self):
        """Successive updates move monotonically toward target (no overshoot)."""
        sv = SmoothValue(value=0.0, rate=0.9)
        sv.target = 10.0

        prev = sv.current
        for _ in range(50):
            sv.update(dt=0.05)
            assert sv.current >= prev  # monotonically increasing toward target
            assert sv.current <= sv.target  # no overshoot
            prev = sv.current

    def test_update_monotonic_decreasing(self):
        """Monotonic approach also works when target is below current."""
        sv = SmoothValue(value=10.0, rate=0.9)
        sv.target = 0.0

        prev = sv.current
        for _ in range(50):
            sv.update(dt=0.05)
            assert sv.current <= prev  # monotonically decreasing
            assert sv.current >= sv.target  # no overshoot
            prev = sv.current

    def test_update_reaches_target(self):
        """Repeated updates eventually reach the target."""
        sv = SmoothValue(value=0.0, rate=0.9)
        sv.target = 10.0

        for _ in range(100):
            sv.update(dt=0.05)

        assert approx(sv.current, sv.target)

    def test_higher_rate_converges_faster(self):
        """Higher rate reaches the target in fewer steps."""
        def steps_to_converge(rate, tol=0.01):
            sv = SmoothValue(value=0.0, rate=rate)
            sv.target = 10.0
            for i in range(1000):
                sv.update(dt=0.05)
                if approx(sv.current, sv.target, tol=tol):
                    return i
            return 1000  # didn't converge

        slow = steps_to_converge(rate=0.3)
        fast = steps_to_converge(rate=0.9)
        assert fast < slow


# ---------------------------------------------------------------------------
# Tests: SmoothInt
# ---------------------------------------------------------------------------

class TestSmoothInt:
    """Test the SmoothInt animation class.

    Same behavioral contract as SmoothValue, plus:
    - ``current`` returns an integer (truncated from internal float).
    - Subpixel tracking prevents quantization drift.
    """

    def test_initial_value(self):
        """Initial current and target are both set to the given value."""
        si = SmoothInt(value=5)
        assert si.current == 5
        assert si.target == 5

    def test_current_is_int(self):
        """current always returns an int."""
        si = SmoothInt(value=0, rate=0.5)
        si.target = 100
        si.update(dt=0.04)
        assert isinstance(si.current, int)

    def test_set_target(self):
        """Setting target doesn't change current."""
        si = SmoothInt(value=0)
        si.target = 100
        assert si.target == 100
        assert si.current == 0

    def test_set_immediate(self):
        """set_immediate snaps both current and target."""
        si = SmoothInt(value=0)
        si.set_immediate(100)
        assert si.current == 100
        assert si.target == 100

    def test_is_animating(self):
        """is_animating is True when current differs from target."""
        si = SmoothInt(value=0)
        assert not si.is_animating()

        si.target = 100
        assert si.is_animating()

    def test_update_moves_toward_target(self):
        """A single update moves current toward target."""
        si = SmoothInt(value=0, rate=0.5)
        si.target = 1000

        si.update(dt=0.04)
        assert si.current > 0
        assert si.current < si.target

    def test_update_monotonic(self):
        """Successive updates move monotonically toward target (no overshoot)."""
        si = SmoothInt(value=0, rate=0.9)
        si.target = 1000

        prev = si.current_exact
        for _ in range(50):
            si.update(dt=0.04)
            assert si.current_exact >= prev
            assert si.current_exact <= si.target
            prev = si.current_exact

    def test_update_monotonic_decreasing(self):
        """Monotonic approach also works when target is below current."""
        si = SmoothInt(value=1000, rate=0.9)
        si.target = 0

        prev = si.current_exact
        for _ in range(50):
            si.update(dt=0.04)
            assert si.current_exact <= prev
            assert si.current_exact >= si.target
            prev = si.current_exact

    def test_update_reaches_target(self):
        """Repeated updates eventually reach the target."""
        si = SmoothInt(value=0, rate=0.9)
        si.target = 500

        for _ in range(200):
            si.update(dt=0.04)

        assert si.current == si.target

    def test_subpixel_accumulation(self):
        """Small fractional deltas accumulate rather than being lost to truncation.

        Without subpixel tracking, integer truncation would stall
        the animation when the step is < 1 pixel.
        """
        si = SmoothInt(value=0, rate=0.1)  # slow rate
        si.target = 10

        # With only integer tracking, the first few updates might produce
        # step < 1 and the position would never change. With subpixel
        # tracking, the internal float accumulates and eventually ticks
        # the integer forward.
        for _ in range(200):
            si.update(dt=0.04)

        assert si.current == si.target

    def test_higher_rate_converges_faster(self):
        """Higher rate reaches the target in fewer steps."""
        def steps_to_converge(rate, tol=1):
            si = SmoothInt(value=0, rate=rate)
            si.target = 500
            for i in range(1000):
                si.update(dt=0.04)
                if abs(si.current - si.target) <= tol:
                    return i
            return 1000

        slow = steps_to_converge(rate=0.3)
        fast = steps_to_converge(rate=0.9)
        assert fast < slow

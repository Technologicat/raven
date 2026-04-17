"""Unit tests for raven.common.running_average."""

import pytest

from raven.common.running_average import RunningAverage


class TestRunningAverage:
    def test_empty_returns_zero(self):
        ra = RunningAverage()
        assert ra.average() == 0.0

    def test_single_datapoint(self):
        ra = RunningAverage()
        ra.add_datapoint(42.0)
        assert ra.average() == 42.0

    def test_simple_mean(self):
        ra = RunningAverage()
        for x in (1.0, 2.0, 3.0):
            ra.add_datapoint(x)
        assert ra.average() == pytest.approx(2.0)

    def test_window_caps_at_count(self):
        ra = RunningAverage()
        assert ra.count == 100  # documented default; the window-cap logic depends on this
        # Push 150 datapoints; only the last 100 should remain.
        for x in range(150):
            ra.add_datapoint(float(x))
        assert len(ra.data) == ra.count
        # Mean of integers [50..149] inclusive = 99.5.
        assert ra.average() == pytest.approx((50 + 149) / 2)

    def test_oldest_datapoints_drop_out(self):
        ra = RunningAverage()
        # Fill the window with zeros, then push a single large value; old entries vanish one by one.
        for _ in range(ra.count):
            ra.add_datapoint(0.0)
        assert ra.average() == 0.0
        ra.add_datapoint(100.0)
        # One slot displaced by the 100.0 → mean = 100 / count.
        assert ra.average() == pytest.approx(100.0 / ra.count)

    def test_mixed_sign_values(self):
        ra = RunningAverage()
        for x in (-1.0, 1.0, -2.0, 2.0):
            ra.add_datapoint(x)
        assert ra.average() == pytest.approx(0.0)

    def test_integer_inputs_accepted(self):
        # Duck-typing: `add_datapoint` type-hints float but sum()/len() work with ints too.
        ra = RunningAverage()
        for x in (1, 2, 3, 4):
            ra.add_datapoint(x)
        assert ra.average() == pytest.approx(2.5)

"""Tests for the API rate limiter."""

import threading
import time

from raven.papers.ratelimit import RateLimiter


class TestRateLimiter:
    """Verify rate limiter timing and thread safety."""

    def test_first_wait_immediate(self):
        """First call to wait should return immediately (no prior action)."""
        rl = RateLimiter(delay=10.0)
        t0 = time.monotonic()
        rl.wait(show_progress=False)
        elapsed = time.monotonic() - t0
        assert elapsed < 0.1

    def test_second_wait_delays(self):
        """Second call should wait approximately the configured delay."""
        rl = RateLimiter(delay=0.3)
        rl.wait(show_progress=False)
        t0 = time.monotonic()
        rl.wait(show_progress=False)
        elapsed = time.monotonic() - t0
        assert 0.25 <= elapsed <= 0.5

    def test_no_delay_if_enough_time_passed(self):
        """If enough wall time has passed, wait returns immediately."""
        rl = RateLimiter(delay=0.1)
        rl.wait(show_progress=False)
        time.sleep(0.15)
        t0 = time.monotonic()
        rl.wait(show_progress=False)
        elapsed = time.monotonic() - t0
        assert elapsed < 0.1

    def test_custom_delay(self):
        rl = RateLimiter(delay=0.2)
        rl.wait(show_progress=False)
        t0 = time.monotonic()
        rl.wait(show_progress=False)
        elapsed = time.monotonic() - t0
        assert 0.15 <= elapsed <= 0.35

    def test_thread_safety(self):
        """Multiple threads waiting should each respect the delay."""
        rl = RateLimiter(delay=0.15)
        timestamps = []
        lock = threading.Lock()

        def worker():
            rl.wait(show_progress=False)
            with lock:
                timestamps.append(time.monotonic())

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        # All three should have completed
        assert len(timestamps) == 3
        # Sort timestamps — consecutive actions should be ~0.15s apart
        timestamps.sort()
        for i in range(1, len(timestamps)):
            gap = timestamps[i] - timestamps[i - 1]
            assert gap >= 0.1, f"gap {i - 1}→{i} was {gap:.3f}s, expected >= 0.1s"

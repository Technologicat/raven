"""Rate limiter for API access.

The arXiv API terms of use require waiting a minimum of 3 seconds between
requests: https://info.arxiv.org/help/api/tou.html
"""

from __future__ import annotations

__all__ = ["RateLimiter"]

import math
import threading
import time

from tqdm import tqdm


class RateLimiter:
    """Rate-limit an action.

    Instantiate one `RateLimiter` per set-of-things-that-share-the-same-rate-limit.

    Thread-safe. Multiple threads resolve in the order in which they entered `wait`.
    Each time a thread finishes waiting, the time counter resets (before the next thread
    is allowed to start waiting), so that each thread waits for the correct amount of time.
    """

    def __init__(self, delay: float = 3.0) -> None:
        """`delay`: minimum required delay between actions, seconds."""
        self.delay = delay
        self.timestamp = 0  # can use any value that causes the first `wait()` to return immediately
        self.lock = threading.RLock()

    def wait(self, show_progress: bool = True) -> None:
        """Wait until `self.delay` seconds of wall time has elapsed since the last action.

        `show_progress`: If ``True``, show a `tqdm` progress bar while waiting.
                         If ``False``, wait silently.

        Call this just before performing the actual action to honor the rate limit for that action.

        When the `RateLimiter` starts up, the first `wait` returns immediately,
        so that no special handling is needed for the first action at the calling end.

        Further calls to `wait` measure time from when the previous call to `wait` finished.
        """
        with self.lock:
            t = time.monotonic_ns()
            delay_ns = self.delay * 10**9
            wait_duration_ns = delay_ns - (t - self.timestamp)
            if wait_duration_ns > 0:
                if show_progress:
                    # Segments of 0.1 seconds (last one may be shorter)
                    total_segments = math.ceil((wait_duration_ns / 10**9) * 10)
                    with tqdm(desc="Waiting for API rate limit", leave=False) as pbar:
                        pbar.total = total_segments
                        pbar.n = 0
                        while (time.monotonic_ns() - t) < wait_duration_ns:
                            time.sleep(min(0.1, wait_duration_ns / 10**9))
                            pbar.update()
                else:
                    time.sleep(wait_duration_ns / 10**9)
            self.timestamp = time.monotonic_ns()

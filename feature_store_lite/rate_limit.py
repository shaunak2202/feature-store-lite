"""A small in-memory fixed-window rate limiter.

Not distributed, not persisted across restarts, and not safe across multiple
worker processes -- a real deployment behind more than one uvicorn worker would
need Redis or a proxy-level limiter instead. This exists to make the concept
concrete and testable within the scope of a single-process portfolio demo, and is
deliberately kept small enough to read in one sitting.
"""

import time
from collections import defaultdict
from typing import Dict, Tuple


class FixedWindowRateLimiter:
    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # client_key -> (window_start_epoch_seconds, count_in_window)
        self._windows: Dict[str, Tuple[float, int]] = defaultdict(lambda: (0.0, 0))

    def allow(self, client_key: str) -> bool:
        now = time.time()
        window_start, count = self._windows[client_key]

        if now - window_start >= self.window_seconds:
            # Start a fresh window.
            self._windows[client_key] = (now, 1)
            return True

        if count < self.max_requests:
            self._windows[client_key] = (window_start, count + 1)
            return True

        return False

    def reset(self) -> None:
        self._windows.clear()

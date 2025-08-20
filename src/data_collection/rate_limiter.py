# src/data_collection/rate_limiter.py

import time
import random
import threading
from collections import deque
from datetime import datetime
import logging
from config.settings import RATE_LIMIT_CONFIG

logger = logging.getLogger(__name__)


class RateLimiter:
    def __init__(self):
        self.requests_per_minute = RATE_LIMIT_CONFIG["REQUESTS_PER_MINUTE"]
        self.requests_per_hour = RATE_LIMIT_CONFIG["REQUESTS_PER_HOUR"]
        self.backoff_factor = RATE_LIMIT_CONFIG["BACKOFF_FACTOR"]
        self.max_backoff = RATE_LIMIT_CONFIG["MAX_BACKOFF"]
        self.jitter = RATE_LIMIT_CONFIG["JITTER"]

        self._minute_requests = deque()
        self._hour_requests = deque()
        self._failure_count = 0
        self._lock = threading.Lock()
        self._stats = {
            "total_requests": 0,
            "blocked_requests": 0,
            "total_wait_time": 0.0,
        }

    def wait_if_needed(self) -> float:
        with self._lock:
            now = datetime.now()
            wait_time = 0.0

            # Drop entries outside their windows
            while (
                self._minute_requests
                and (now - self._minute_requests[0]).total_seconds() > 60
            ):
                self._minute_requests.popleft()
            while (
                self._hour_requests
                and (now - self._hour_requests[0]).total_seconds() > 3600
            ):
                self._hour_requests.popleft()

            # Compute required wait due to minute/hour caps
            if len(self._minute_requests) >= self.requests_per_minute:
                oldest_min = self._minute_requests[0]
                wait_time = max(wait_time, 60.0 - (now - oldest_min).total_seconds())
            if len(self._hour_requests) >= self.requests_per_hour:
                oldest_hr = self._hour_requests[0]
                wait_time = max(wait_time, 3600.0 - (now - oldest_hr).total_seconds())

            # Backoff on failures
            if self._failure_count > 0:
                backoff = min(
                    self.backoff_factor**self._failure_count, self.max_backoff
                )
                if self.jitter:
                    backoff *= random.uniform(0.5, 1.5)
                wait_time = max(wait_time, float(backoff))

            if wait_time > 0:
                self._stats["blocked_requests"] += 1
                self._stats["total_wait_time"] += wait_time
                logger.warning(f"Rate limit hit, waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)

            # Record this request
            self._minute_requests.append(now)
            self._hour_requests.append(now)
            self._stats["total_requests"] += 1
            return wait_time

    def record_success(self) -> None:
        with self._lock:
            self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            logger.warning(f"Request failed, failure count: {self._failure_count}")

    def get_stats(self) -> dict:
        with self._lock:
            return {
                **self._stats,
                "current_minute_requests": len(self._minute_requests),
                "current_hour_requests": len(self._hour_requests),
                "failure_count": self._failure_count,
            }

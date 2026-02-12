"""Stale data policy for graceful degradation."""

from datetime import datetime
from typing import Optional, Any
import structlog

logger = structlog.get_logger()


class StaleDataPolicy:
    """Manages stale cache usage and failure tracking."""

    MAX_STALE_AGE_SECONDS = 600  # 10 minutes
    CONSECUTIVE_FAILURE_ALERT = 3  # Alert after 3 failures

    def __init__(self):
        self._consecutive_failures = 0
        self._last_success_time: Optional[datetime] = None
        self._stale_cache: Optional[tuple[Any, datetime]] = None

    def record_success(self, data: Any):
        """Record successful fetch."""
        self._consecutive_failures = 0
        self._last_success_time = datetime.now()
        self._stale_cache = (data, datetime.now())

    def record_failure(self):
        """Record failed fetch."""
        self._consecutive_failures += 1

        if self._consecutive_failures >= self.CONSECUTIVE_FAILURE_ALERT:
            logger.error(
                "ALERT: Multiple consecutive fetch failures",
                consecutive_failures=self._consecutive_failures,
                last_success_age=self._get_time_since_success()
            )

    def can_use_stale_cache(self) -> bool:
        """Check if stale cache is acceptable."""
        if not self._stale_cache:
            return False

        _, cached_at = self._stale_cache
        age = (datetime.now() - cached_at).total_seconds()

        return age < self.MAX_STALE_AGE_SECONDS

    def _get_time_since_success(self) -> str:
        """Human-readable time since last success."""
        if not self._last_success_time:
            return "never"

        delta = datetime.now() - self._last_success_time
        minutes = int(delta.total_seconds() / 60)
        return f"{minutes} minutes ago"

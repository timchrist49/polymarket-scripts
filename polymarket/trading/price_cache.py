"""Price caching with intelligent TTL strategies."""

from datetime import datetime
from typing import Optional

from polymarket.models import PricePoint


class CandleCache:
    """Per-candle caching with age-based TTL."""

    def __init__(self):
        self._candles: dict[int, tuple[PricePoint, datetime]] = {}
        # Key: timestamp_minute, Value: (candle, cached_at)

    def get_ttl(self, candle_timestamp: datetime, current_time: Optional[datetime] = None) -> int:
        """
        Calculate TTL based on candle age.

        TTL Strategy:
        - >60 minutes old: 3600s (1 hour) - immutable historical data
        - 5-60 minutes old: 300s (5 minutes) - recent closed candles
        - <5 minutes old: 60s (1 minute) - current/active candles

        Args:
            candle_timestamp: Timestamp of the candle
            current_time: Current time for age calculation (defaults to now())

        Returns:
            TTL in seconds
        """
        if current_time is None:
            current_time = datetime.now()
        age_minutes = (current_time - candle_timestamp).total_seconds() / 60

        if age_minutes > 60:
            return 3600  # 1 hour TTL for old candles (immutable)
        elif age_minutes > 5:
            return 300   # 5 min TTL for recent closed candles
        else:
            return 60    # 1 min TTL for current/recent candles

    def put(self, timestamp: int, candle: PricePoint) -> None:
        """
        Cache a candle.

        Args:
            timestamp: Minute-level timestamp (Unix seconds)
            candle: Price data point
        """
        self._candles[timestamp] = (candle, datetime.now())

    def get(self, timestamp: int) -> Optional[PricePoint]:
        """
        Retrieve cached candle if valid.

        Args:
            timestamp: Minute-level timestamp (Unix seconds)

        Returns:
            Cached candle or None if invalid/missing
        """
        if not self.is_valid(timestamp):
            return None

        candle, _ = self._candles[timestamp]
        return candle

    def is_valid(self, timestamp: int) -> bool:
        """
        Check if cached candle is still valid.

        Args:
            timestamp: Minute-level timestamp (Unix seconds)

        Returns:
            True if cache is valid, False otherwise
        """
        if timestamp not in self._candles:
            return False

        candle, cached_at = self._candles[timestamp]
        age = (datetime.now() - cached_at).total_seconds()
        ttl = self.get_ttl(candle.timestamp)

        return age < ttl

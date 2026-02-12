"""Price caching with intelligent TTL strategies."""

from datetime import datetime
from typing import Optional

from polymarket.models import PricePoint


class CandleCache:
    """Per-candle caching with age-based TTL."""

    def __init__(self):
        self._candles: dict[int, tuple[PricePoint, datetime]] = {}
        # Key: timestamp_minute, Value: (candle, cached_at)

    def get_ttl(self, candle_timestamp: datetime) -> int:
        """
        Calculate TTL based on candle age.

        Returns:
            TTL in seconds
        """
        age_minutes = (datetime.now() - candle_timestamp).total_seconds() / 60

        if age_minutes > 60:
            return 3600  # 1 hour TTL for old candles (immutable)
        elif age_minutes > 5:
            return 300   # 5 min TTL for recent closed candles
        else:
            return 60    # 1 min TTL for current/recent candles

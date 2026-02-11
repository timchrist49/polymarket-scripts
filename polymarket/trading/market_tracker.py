"""
Market Tracker Service

Tracks market timing, calculates time remaining, and manages price-to-beat.
"""

from decimal import Decimal
from datetime import datetime, timezone
from typing import Optional, Dict
import structlog

from polymarket.config import Settings

logger = structlog.get_logger()


class MarketTracker:
    """Track market timing and price-to-beat for BTC 15-minute markets."""

    MARKET_DURATION_SECONDS = 15 * 60  # 15 minutes
    END_OF_MARKET_THRESHOLD = 3 * 60  # Last 3 minutes

    def __init__(self, settings: Settings):
        self.settings = settings
        self._price_to_beat: Dict[str, Decimal] = {}  # slug -> starting_price

    def parse_market_start(self, slug: str) -> Optional[datetime]:
        """
        Parse market start timestamp from slug.

        Slug format: btc-updown-15m-{epoch_timestamp}
        Example: btc-updown-15m-1739203200
        """
        try:
            parts = slug.split("-")
            if len(parts) < 4:
                logger.warning("Invalid market slug format", slug=slug)
                return None

            timestamp_str = parts[-1]  # Last part is epoch
            timestamp = int(timestamp_str)

            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except (ValueError, IndexError) as e:
            logger.error("Failed to parse market slug", slug=slug, error=str(e))
            return None

    def calculate_time_remaining(
        self,
        start_time: datetime,
        current_time: Optional[datetime] = None
    ) -> int:
        """
        Calculate seconds remaining in 15-minute market.

        Returns:
            Seconds remaining (0 if market expired)
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        elapsed = (current_time - start_time).total_seconds()
        remaining = self.MARKET_DURATION_SECONDS - elapsed

        return max(0, int(remaining))

    def is_end_of_market(
        self,
        start_time: datetime,
        current_time: Optional[datetime] = None
    ) -> bool:
        """
        Check if we're in the last 3 minutes of market.

        Returns:
            True if <= 3 minutes remaining
        """
        remaining = self.calculate_time_remaining(start_time, current_time)
        return remaining <= self.END_OF_MARKET_THRESHOLD

    def set_price_to_beat(self, slug: str, price: Decimal):
        """Store starting price for market."""
        self._price_to_beat[slug] = price
        logger.info(
            "Price-to-beat set",
            slug=slug,
            price=f"${price:,.2f}"
        )

    def get_price_to_beat(self, slug: str) -> Optional[Decimal]:
        """Get starting price for market."""
        return self._price_to_beat.get(slug)

    def calculate_price_difference(
        self,
        current_price: Decimal,
        price_to_beat: Decimal
    ) -> tuple[Decimal, float]:
        """
        Calculate price difference and percentage change.

        Returns:
            (difference_amount, percentage_change)
        """
        diff = current_price - price_to_beat
        pct = float(diff / price_to_beat * 100)
        return diff, pct

"""
Market Tracker Service

Tracks market timing, calculates time remaining, and manages price-to-beat.
"""

from decimal import Decimal
from datetime import datetime, timezone
from typing import Optional, Dict
from pathlib import Path
import json
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
        self._cache_file = Path("/root/polymarket-scripts/.cache/price_to_beat.json")
        self._load_from_file()

    def parse_market_start(self, slug: str) -> Optional[datetime]:
        """
        Parse market start timestamp from slug.

        Slug format: btc-updown-15m-{epoch_timestamp}
        Example: btc-updown-15m-1739203200

        IMPORTANT: The timestamp in the slug is the market CLOSE time.
        We subtract 15 minutes (900 seconds) to get the actual START time.
        """
        try:
            parts = slug.split("-")
            if len(parts) < 4:
                logger.warning("Invalid market slug format", slug=slug)
                return None

            timestamp_str = parts[-1]  # Last part is close timestamp
            close_timestamp = int(timestamp_str)

            # Calculate START time: close - 15 minutes (900 seconds)
            start_timestamp = close_timestamp - 900

            return datetime.fromtimestamp(start_timestamp, tz=timezone.utc)
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
        self._save_to_file()

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

    def _save_to_file(self):
        """Persist price-to-beat data to disk."""
        try:
            # Ensure cache directory exists
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)

            # Convert to JSON-serializable format
            data = {
                slug: {
                    "price": str(price),
                    "timestamp": datetime.now(timezone.utc).timestamp()
                }
                for slug, price in self._price_to_beat.items()
            }

            # Write atomically using temp file
            temp_file = self._cache_file.with_suffix('.tmp')
            with temp_file.open('w') as f:
                json.dump(data, f, indent=2)
            temp_file.replace(self._cache_file)

            logger.debug("Price-to-beat data saved", count=len(data))
        except Exception as e:
            logger.error("Failed to save price-to-beat data", error=str(e))

    def _load_from_file(self):
        """Load price-to-beat data from disk."""
        try:
            if not self._cache_file.exists():
                logger.debug("No price-to-beat cache file found, starting fresh")
                return

            with self._cache_file.open('r') as f:
                data = json.load(f)

            # Convert back to Decimal
            self._price_to_beat = {
                slug: Decimal(info["price"])
                for slug, info in data.items()
            }

            logger.info(
                "Price-to-beat data loaded",
                count=len(self._price_to_beat),
                markets=list(self._price_to_beat.keys())
            )
        except Exception as e:
            logger.error("Failed to load price-to-beat data", error=str(e))
            self._price_to_beat = {}

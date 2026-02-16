"""Market validation for real-time odds monitoring."""

from typing import Optional
from datetime import datetime, timezone
import structlog

logger = structlog.get_logger(__name__)


class MarketValidator:
    """Validates market activity and timing."""

    @staticmethod
    def parse_market_timestamp(slug: str) -> Optional[int]:
        """Parse Unix timestamp from market slug.

        Args:
            slug: Market slug like 'btc-updown-15m-1771270200'

        Returns:
            Unix timestamp as integer, or None if invalid format
        """
        if not slug:
            logger.error("Empty market slug provided")
            return None

        parts = slug.split('-')

        # Validate format: btc-updown-15m-{timestamp}
        if len(parts) >= 4 and parts[0] == 'btc' and parts[1] == 'updown':
            try:
                timestamp = int(parts[3])
                logger.debug(
                    "Parsed timestamp from slug",
                    slug=slug,
                    timestamp=timestamp
                )
                return timestamp
            except (ValueError, IndexError) as e:
                logger.error(
                    "Invalid timestamp in market slug",
                    slug=slug,
                    error=str(e)
                )
                return None

        logger.error(
            "Unexpected market slug format",
            slug=slug,
            expected_format="btc-updown-15m-{timestamp}"
        )
        return None

    @staticmethod
    def is_market_active(slug: str, tolerance_minutes: int = 2) -> bool:
        """Check if market is currently active (within tolerance window).

        Args:
            slug: Market slug like 'btc-updown-15m-1771270200'
            tolerance_minutes: Minutes of tolerance before/after market time (default: 2)

        Returns:
            True if market is active (within tolerance), False otherwise
        """
        # Parse timestamp from slug
        market_timestamp = MarketValidator.parse_market_timestamp(slug)
        if not market_timestamp:
            logger.error("Cannot validate market with invalid slug", slug=slug)
            return False

        # Get current time
        current_time = datetime.now(timezone.utc)
        market_time = datetime.fromtimestamp(market_timestamp, timezone.utc)

        # Calculate time difference in seconds
        time_diff_seconds = abs((current_time - market_time).total_seconds())
        tolerance_seconds = tolerance_minutes * 60

        is_active = time_diff_seconds <= tolerance_seconds

        logger.debug(
            "Checked market activity",
            slug=slug,
            market_time=market_time.isoformat(),
            current_time=current_time.isoformat(),
            time_diff_seconds=time_diff_seconds,
            tolerance_seconds=tolerance_seconds,
            is_active=is_active
        )

        return is_active

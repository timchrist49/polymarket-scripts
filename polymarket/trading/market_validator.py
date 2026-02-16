"""Market validation for real-time odds monitoring."""

from typing import Optional
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

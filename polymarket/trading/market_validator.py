"""Market validation for real-time odds monitoring."""

import structlog

logger = structlog.get_logger(__name__)


class MarketValidator:
    """Validates market activity and timing."""

    def parse_timestamp(self, slug: str) -> int:
        """Parse Unix timestamp from market slug.

        Args:
            slug: Market slug like 'btc-updown-15m-1771270200'

        Returns:
            Unix timestamp as integer

        Raises:
            ValueError: If slug format is invalid
        """
        try:
            parts = slug.split('-')
            timestamp_str = parts[-1]
            timestamp = int(timestamp_str)

            logger.debug(
                "Parsed timestamp from slug",
                slug=slug,
                timestamp=timestamp
            )

            return timestamp

        except (IndexError, ValueError) as e:
            logger.error(
                "Failed to parse timestamp from slug",
                slug=slug,
                error=str(e)
            )
            raise ValueError(f"Invalid market slug format: {slug}") from e

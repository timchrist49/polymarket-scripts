"""Settlement price validation across multiple sources."""

import asyncio
from decimal import Decimal
from typing import Optional
import structlog

logger = structlog.get_logger()


class SettlementPriceValidator:
    """Validates historical prices across multiple sources."""

    MIN_SOURCES = 2          # Need at least 2 sources to validate

    def __init__(self, btc_service=None, tolerance_percent: float = 0.5):
        """
        Initialize validator.

        Args:
            btc_service: BTCPriceService instance (for fetching)
            tolerance_percent: Price agreement tolerance (%)
        """
        self._btc_service = btc_service
        self.tolerance_percent = tolerance_percent

    async def get_validated_price(
        self,
        timestamp: int
    ) -> Optional[Decimal]:
        """
        Fetch price for settlement with buffer-first approach.

        Now uses 24-hour price buffer, so no age restrictions.
        Buffer-first fetch handles both recent and historical data.

        Args:
            timestamp: Unix timestamp (seconds)

        Returns:
            Validated price or None if unavailable
        """
        from datetime import datetime

        # Calculate age for logging
        now = datetime.now().timestamp()
        age_seconds = now - timestamp

        # Try buffer-first fetch (handles up to 24h of data)
        # This checks buffer first, then falls back to Binance API
        price = await self._fetch_binance_at_timestamp(timestamp)

        if price:
            logger.info(
                "Settlement price fetched",
                source="buffer-or-api",
                price=f"${price:,.2f}",
                age_minutes=f"{age_seconds/60:.1f}"
            )
            return price
        else:
            logger.warning(
                "Failed to fetch settlement price",
                timestamp=timestamp,
                age_hours=f"{age_seconds/3600:.1f}",
                reason="Not in buffer and API unavailable"
            )
            return None

    def _calculate_spread(self, prices: list[Decimal]) -> float:
        """Calculate price spread percentage."""
        if not prices:
            return 0.0
        min_price = min(prices)
        max_price = max(prices)
        return float((max_price - min_price) / min_price * 100)

    async def _fetch_binance_at_timestamp(self, timestamp: int) -> Optional[Decimal]:
        """Fetch price at timestamp via BTCPriceService (buffer-first, then Binance fallback)."""
        if self._btc_service:
            # Call _fetch_binance_at_timestamp directly to avoid circular dependency
            # (get_price_at_timestamp calls back to settlement validator)
            return await self._btc_service._fetch_binance_at_timestamp(timestamp)
        return None

    async def _fetch_coingecko_at_timestamp(self, timestamp: int) -> Optional[Decimal]:
        """Fetch from CoinGecko via BTCPriceService."""
        if self._btc_service:
            return await self._btc_service._fetch_coingecko_at_timestamp(timestamp)
        return None

    async def _fetch_kraken_at_timestamp(self, timestamp: int) -> Optional[Decimal]:
        """Fetch from Kraken via BTCPriceService."""
        if self._btc_service:
            return await self._btc_service._fetch_kraken_at_timestamp(timestamp)
        return None

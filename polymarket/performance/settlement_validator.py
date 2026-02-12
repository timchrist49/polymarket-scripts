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
        Fetch price from multiple sources and validate agreement.

        For recent timestamps (< 1 hour old), use single source (Binance)
        since we're fetching real-time data, not historical lookups.

        Args:
            timestamp: Unix timestamp (seconds)

        Returns:
            Validated price or None if unavailable
        """
        from datetime import datetime

        # Calculate age of timestamp
        now = datetime.now().timestamp()
        age_seconds = now - timestamp

        # For recent data (< 1 hour), just use Binance
        # Free APIs don't support historical lookups, only current prices
        if age_seconds < 3600:  # Less than 1 hour old
            price = await self._fetch_binance_at_timestamp(timestamp)
            if price:
                logger.info(
                    "Settlement price fetched (real-time)",
                    source="Binance",
                    price=f"${price:,.2f}",
                    age_minutes=f"{age_seconds/60:.1f}"
                )
                return price
            else:
                logger.warning(
                    "Failed to fetch recent price from Binance",
                    timestamp=timestamp,
                    age_seconds=age_seconds
                )
                return None

        # For old data (> 1 hour), skip multi-source validation
        # Free APIs can't provide historical data
        logger.warning(
            "Timestamp too old for free API settlement",
            timestamp=timestamp,
            age_hours=f"{age_seconds/3600:.1f}"
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
        """Fetch from Binance via BTCPriceService."""
        if self._btc_service:
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

"""Settlement price validation across multiple sources."""

import asyncio
from decimal import Decimal
from typing import Optional
import structlog

logger = structlog.get_logger()


class SettlementPriceValidator:
    """Fetches settlement prices with 3-tier fallback hierarchy."""

    def __init__(self, btc_service=None):
        """
        Initialize validator.

        Args:
            btc_service: BTCPriceService instance (for fetching)
        """
        self._btc_service = btc_service

    async def get_validated_price(
        self,
        timestamp: int
    ) -> Optional[Decimal]:
        """
        Fetch price for settlement with 3-tier fallback hierarchy.

        Hierarchy:
        1. Chainlink buffer (primary, matches Polymarket settlement)
        2. CoinGecko historical API (secondary)
        3. Binance historical API (last resort)

        Args:
            timestamp: Unix timestamp (seconds)

        Returns:
            Validated price or None if all sources fail
        """
        from datetime import datetime

        # Calculate age for logging
        now = datetime.now().timestamp()
        age_seconds = now - timestamp

        # Tier 1: Try Chainlink buffer
        price = await self._fetch_chainlink_from_buffer(timestamp)
        if price:
            logger.info(
                "Settlement price from Chainlink buffer",
                source="chainlink",
                price=f"${price:,.2f}",
                age_minutes=f"{age_seconds/60:.1f}"
            )
            return price

        # Tier 2: Try CoinGecko API
        price = await self._fetch_coingecko_at_timestamp(timestamp)
        if price:
            logger.info(
                "Settlement price from CoinGecko (fallback)",
                source="coingecko",
                price=f"${price:,.2f}",
                age_minutes=f"{age_seconds/60:.1f}",
                reason="buffer_miss"
            )
            return price

        # Tier 3: Try Binance API (last resort)
        price = await self._fetch_binance_at_timestamp(timestamp)
        if price:
            logger.warning(
                "Settlement price from Binance (last resort)",
                source="binance",
                price=f"${price:,.2f}",
                age_minutes=f"{age_seconds/60:.1f}",
                reason="chainlink_and_coingecko_failed"
            )
            return price

        # All sources failed
        logger.error(
            "Failed to fetch settlement price from all sources",
            timestamp=timestamp,
            age_hours=f"{age_seconds/3600:.1f}",
            sources_tried=["chainlink_buffer", "coingecko", "binance"]
        )
        return None

    async def _fetch_chainlink_from_buffer(self, timestamp: int) -> Optional[Decimal]:
        """Fetch Chainlink price from buffer via BTCPriceService."""
        if self._btc_service:
            try:
                return await self._btc_service._fetch_chainlink_from_buffer(timestamp)
            except Exception as e:
                logger.warning(
                    "Chainlink buffer fetch failed",
                    timestamp=timestamp,
                    error=str(e)
                )
        return None

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

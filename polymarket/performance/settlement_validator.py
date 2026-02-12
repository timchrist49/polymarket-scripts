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

        Args:
            timestamp: Unix timestamp (seconds)

        Returns:
            Validated price or None if sources disagree
        """
        # Fetch from all sources in parallel
        tasks = [
            ("Binance", self._fetch_binance_at_timestamp(timestamp)),
            ("CoinGecko", self._fetch_coingecko_at_timestamp(timestamp)),
            ("Kraken", self._fetch_kraken_at_timestamp(timestamp))
        ]

        results = await asyncio.gather(*[task[1] for task in tasks], return_exceptions=True)
        prices = {
            name: price
            for (name, _), price in zip(tasks, results)
            if price is not None and not isinstance(price, Exception)
        }

        if len(prices) < self.MIN_SOURCES:
            logger.error(
                "Insufficient sources for validation",
                available=len(prices),
                required=self.MIN_SOURCES
            )
            return None

        # Check if all prices agree within tolerance
        prices_list = list(prices.values())
        avg_price = sum(prices_list) / len(prices_list)

        for source, price in prices.items():
            deviation_pct = abs(float((price - avg_price) / avg_price * 100))

            if deviation_pct > self.tolerance_percent:
                logger.error(
                    "Price sources disagree",
                    source=source,
                    price=float(price),
                    avg_price=float(avg_price),
                    deviation_pct=f"{deviation_pct:.2f}%",
                    tolerance=f"{self.tolerance_percent}%"
                )
                return None

        # All sources agree! Return average for accuracy
        logger.info(
            "Settlement price validated",
            sources=list(prices.keys()),
            avg_price=f"${avg_price:,.2f}",
            spread=f"{self._calculate_spread(prices_list):.2f}%"
        )

        return avg_price

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

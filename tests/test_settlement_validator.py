import pytest
from decimal import Decimal
from unittest.mock import AsyncMock

from polymarket.performance.settlement_validator import SettlementPriceValidator


@pytest.mark.asyncio
async def test_validate_prices_agree():
    """Prices from multiple sources agree within tolerance."""
    validator = SettlementPriceValidator()

    # Mock the fetch methods
    validator._fetch_binance_at_timestamp = AsyncMock(return_value=Decimal("67123.45"))
    validator._fetch_coingecko_at_timestamp = AsyncMock(return_value=Decimal("67150.20"))
    validator._fetch_kraken_at_timestamp = AsyncMock(return_value=Decimal("67100.80"))

    result = await validator.get_validated_price(1707696000)

    assert result is not None
    # Should return average: (67123.45 + 67150.20 + 67100.80) / 3 = 67124.82
    assert abs(result - Decimal("67124.82")) < Decimal("1.0")


@pytest.mark.asyncio
async def test_validate_prices_disagree():
    """Prices disagree beyond tolerance."""
    validator = SettlementPriceValidator()

    # Mock the fetch methods with prices that differ by >0.5%
    # Average will be 67200, 67700 is 0.74% away which exceeds 0.5%
    validator._fetch_binance_at_timestamp = AsyncMock(return_value=Decimal("67000"))
    validator._fetch_coingecko_at_timestamp = AsyncMock(return_value=Decimal("67700"))
    validator._fetch_kraken_at_timestamp = AsyncMock(return_value=Decimal("67100"))

    result = await validator.get_validated_price(1707696000)

    assert result is None  # Should reject due to disagreement


@pytest.mark.asyncio
async def test_validate_insufficient_sources():
    """Less than 2 sources available."""
    validator = SettlementPriceValidator()

    # Only one source succeeds
    validator._fetch_binance_at_timestamp = AsyncMock(return_value=Decimal("67000"))
    validator._fetch_coingecko_at_timestamp = AsyncMock(return_value=None)
    validator._fetch_kraken_at_timestamp = AsyncMock(return_value=None)

    result = await validator.get_validated_price(1707696000)

    assert result is None  # Need at least 2 sources


def test_calculate_spread():
    """Calculate price spread percentage."""
    validator = SettlementPriceValidator()

    prices = [Decimal("67000"), Decimal("67100"), Decimal("67200")]
    spread = validator._calculate_spread(prices)

    # Spread = (67200 - 67000) / 67000 * 100 = 0.298%
    assert 0.29 < spread < 0.30

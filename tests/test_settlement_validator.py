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


@pytest.mark.asyncio
async def test_fetch_coingecko_at_timestamp_integration():
    """Integration test for CoinGecko timestamp fetch."""
    from polymarket.trading.btc_price import BTCPriceService
    from polymarket.config import Settings

    settings = Settings()
    btc_service = BTCPriceService(settings)
    validator = SettlementPriceValidator(btc_service)

    # Mock the session with proper context manager support
    class MockResponse:
        status = 200

        def raise_for_status(self):
            pass

        async def json(self):
            return {
                "market_data": {
                    "current_price": {"usd": 67123.45}
                }
            }

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    class MockSession:
        closed = False

        def get(self, *args, **kwargs):
            return MockResponse()

    async def mock_get_session():
        return MockSession()

    btc_service._get_session = mock_get_session

    result = await validator._fetch_coingecko_at_timestamp(1707696000)

    assert result == Decimal("67123.45")

    await btc_service.close()

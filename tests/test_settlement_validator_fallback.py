"""Test settlement validator 3-tier fallback hierarchy."""
import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock
from polymarket.performance.settlement_validator import SettlementPriceValidator


@pytest.mark.asyncio
async def test_validator_uses_chainlink_first():
    """Test validator tries Chainlink buffer first."""
    validator = SettlementPriceValidator(MagicMock())

    # Mock Chainlink success
    validator._fetch_chainlink_from_buffer = AsyncMock(
        return_value=Decimal("68598.02")
    )
    validator._fetch_coingecko_at_timestamp = AsyncMock()
    validator._fetch_binance_at_timestamp = AsyncMock()

    price = await validator.get_validated_price(1771178400)

    assert price == Decimal("68598.02")
    validator._fetch_chainlink_from_buffer.assert_called_once()
    validator._fetch_coingecko_at_timestamp.assert_not_called()
    validator._fetch_binance_at_timestamp.assert_not_called()


@pytest.mark.asyncio
async def test_validator_falls_back_to_coingecko():
    """Test validator falls back to CoinGecko if Chainlink fails."""
    validator = SettlementPriceValidator(MagicMock())

    # Mock Chainlink fail, CoinGecko success
    validator._fetch_chainlink_from_buffer = AsyncMock(return_value=None)
    validator._fetch_coingecko_at_timestamp = AsyncMock(
        return_value=Decimal("68600.00")
    )
    validator._fetch_binance_at_timestamp = AsyncMock()

    price = await validator.get_validated_price(1771178400)

    assert price == Decimal("68600.00")
    validator._fetch_chainlink_from_buffer.assert_called_once()
    validator._fetch_coingecko_at_timestamp.assert_called_once()
    validator._fetch_binance_at_timestamp.assert_not_called()


@pytest.mark.asyncio
async def test_validator_falls_back_to_binance():
    """Test validator falls back to Binance as last resort."""
    validator = SettlementPriceValidator(MagicMock())

    # Mock Chainlink fail, CoinGecko fail, Binance success
    validator._fetch_chainlink_from_buffer = AsyncMock(return_value=None)
    validator._fetch_coingecko_at_timestamp = AsyncMock(return_value=None)
    validator._fetch_binance_at_timestamp = AsyncMock(
        return_value=Decimal("68650.00")
    )

    price = await validator.get_validated_price(1771178400)

    assert price == Decimal("68650.00")
    validator._fetch_chainlink_from_buffer.assert_called_once()
    validator._fetch_coingecko_at_timestamp.assert_called_once()
    validator._fetch_binance_at_timestamp.assert_called_once()


@pytest.mark.asyncio
async def test_validator_returns_none_if_all_fail():
    """Test validator returns None if all sources fail."""
    validator = SettlementPriceValidator(MagicMock())

    # Mock all sources fail
    validator._fetch_chainlink_from_buffer = AsyncMock(return_value=None)
    validator._fetch_coingecko_at_timestamp = AsyncMock(return_value=None)
    validator._fetch_binance_at_timestamp = AsyncMock(return_value=None)

    price = await validator.get_validated_price(1771178400)

    assert price is None

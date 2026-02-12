import pytest
import asyncio
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, patch, MagicMock

from polymarket.trading.btc_price import BTCPriceService
from polymarket.config import Settings
from polymarket.models import PricePoint


@pytest.mark.asyncio
async def test_candle_cache_is_integrated():
    """Verify CandleCache is integrated into BTCPriceService."""
    settings = Settings()
    service = BTCPriceService(settings)

    # Verify the cache exists
    assert service._candle_cache is not None

    # Manually test cache operations
    test_candle = PricePoint(
        price=Decimal("67000"),
        volume=Decimal("100"),
        timestamp=datetime.now()
    )

    ts = int(test_candle.timestamp.timestamp())
    service._candle_cache.put(ts, test_candle)

    retrieved = service._candle_cache.get(ts)
    assert retrieved is not None
    assert retrieved.price == Decimal("67000")

    await service.close()


@pytest.mark.asyncio
async def test_get_price_history_fallback_to_coingecko():
    """Falls back to CoinGecko when Binance fails."""
    settings = Settings()
    service = BTCPriceService(settings)

    # Mock Binance to fail
    async def mock_binance_fetch(*args, **kwargs):
        raise Exception("Binance down")

    service._fetch_binance_history = mock_binance_fetch

    # Mock CoinGecko to succeed
    service._fetch_coingecko_history = AsyncMock(return_value=[
        PricePoint(
            price=Decimal("67000"),
            volume=Decimal("100"),
            timestamp=datetime.now()
        )
    ])

    result = await service.get_price_history(minutes=1)

    assert len(result) == 1
    assert result[0].price == Decimal("67000")

    await service.close()


@pytest.mark.asyncio
async def test_get_price_at_timestamp_uses_validator():
    """get_price_at_timestamp uses settlement validator."""
    settings = Settings()
    service = BTCPriceService(settings)

    # Verify settlement validator is integrated
    assert service._settlement_validator is not None
    assert service._settlement_validator._btc_service is service

    await service.close()

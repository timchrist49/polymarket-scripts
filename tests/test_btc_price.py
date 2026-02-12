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

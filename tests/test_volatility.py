"""Test volatility calculation."""
import pytest
from decimal import Decimal
from datetime import datetime
from polymarket.trading.btc_price import BTCPriceService
from polymarket.config import Settings
from polymarket.models import PricePoint


@pytest.mark.asyncio
async def test_volatility_calculation_from_buffer():
    """Should calculate actual volatility from price buffer."""
    service = BTCPriceService(Settings())
    await service.start()

    # This will fail initially because function isn't async
    vol = await service.calculate_15min_volatility()

    # Should be a reasonable value, not fixed 0.005
    assert isinstance(vol, float)
    assert 0.0001 <= vol <= 0.05  # Reasonable range for BTC

    await service.close()


@pytest.mark.asyncio
async def test_volatility_fallback_when_no_buffer():
    """Should fallback to 0.005 if buffer unavailable."""
    service = BTCPriceService(Settings())
    # Don't start stream - buffer unavailable

    vol = await service.calculate_15min_volatility()

    assert vol == 0.005  # Default fallback

    await service.close()

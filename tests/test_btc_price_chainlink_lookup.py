"""Tests for BTCPriceService Chainlink buffer lookup."""

import pytest
from decimal import Decimal
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from polymarket.trading.btc_price import BTCPriceService
from polymarket.models import BTCPriceData


@pytest.mark.asyncio
async def test_fetch_chainlink_from_buffer_success():
    """Test fetching Chainlink price from buffer."""
    service = BTCPriceService(MagicMock())

    # Mock stream with buffer
    mock_buffer = AsyncMock()
    mock_buffer.get_price_at = AsyncMock(return_value=BTCPriceData(
        price=Decimal("68598.02"),
        timestamp=datetime.fromtimestamp(1771178400),
        source="chainlink",
        volume_24h=Decimal("0")
    ))

    service._stream = MagicMock()
    service._stream.price_buffer = mock_buffer

    # Fetch price at timestamp
    price = await service._fetch_chainlink_from_buffer(1771178400)

    assert price == Decimal("68598.02")
    mock_buffer.get_price_at.assert_called_once_with(1771178400, tolerance=30)


@pytest.mark.asyncio
async def test_fetch_chainlink_from_buffer_miss():
    """Test buffer miss returns None."""
    service = BTCPriceService(MagicMock())

    # Mock buffer returning None (no data)
    mock_buffer = AsyncMock()
    mock_buffer.get_price_at = AsyncMock(return_value=None)

    service._stream = MagicMock()
    service._stream.price_buffer = mock_buffer

    # Fetch price at timestamp
    price = await service._fetch_chainlink_from_buffer(1771178400)

    assert price is None


@pytest.mark.asyncio
async def test_fetch_chainlink_from_buffer_wrong_source():
    """Test non-Chainlink source returns None."""
    service = BTCPriceService(MagicMock())

    # Mock buffer returning binance data
    mock_buffer = AsyncMock()
    mock_buffer.get_price_at = AsyncMock(return_value=BTCPriceData(
        price=Decimal("68500.00"),
        timestamp=datetime.fromtimestamp(1771178400),
        source="binance",
        volume_24h=Decimal("0")
    ))

    service._stream = MagicMock()
    service._stream.price_buffer = mock_buffer

    # Fetch price at timestamp
    price = await service._fetch_chainlink_from_buffer(1771178400)

    # Should return None for non-chainlink source
    assert price is None


@pytest.mark.asyncio
async def test_fetch_chainlink_from_buffer_no_stream():
    """Test returns None when stream not initialized."""
    service = BTCPriceService(MagicMock())
    service._stream = None

    # Fetch price at timestamp
    price = await service._fetch_chainlink_from_buffer(1771178400)

    assert price is None


@pytest.mark.asyncio
async def test_fetch_chainlink_from_buffer_no_buffer():
    """Test returns None when buffer not enabled."""
    service = BTCPriceService(MagicMock())

    service._stream = MagicMock()
    service._stream.price_buffer = None

    # Fetch price at timestamp
    price = await service._fetch_chainlink_from_buffer(1771178400)

    assert price is None


@pytest.mark.asyncio
async def test_fetch_chainlink_from_buffer_exception():
    """Test handles buffer exceptions gracefully."""
    service = BTCPriceService(MagicMock())

    # Mock buffer that raises exception
    mock_buffer = AsyncMock()
    mock_buffer.get_price_at = AsyncMock(side_effect=Exception("Buffer error"))

    service._stream = MagicMock()
    service._stream.price_buffer = mock_buffer

    # Fetch price at timestamp
    price = await service._fetch_chainlink_from_buffer(1771178400)

    # Should return None and not propagate exception
    assert price is None

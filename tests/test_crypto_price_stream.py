"""Tests for Polymarket crypto price WebSocket client."""

import pytest
import asyncio
from decimal import Decimal
from polymarket.trading.crypto_price_stream import CryptoPriceStream
from polymarket.config import Settings


@pytest.mark.asyncio
async def test_websocket_connect():
    """Test WebSocket connection and subscription."""
    settings = Settings()
    stream = CryptoPriceStream(settings)

    # Start stream
    asyncio.create_task(stream.start())
    await asyncio.sleep(1)  # Wait for connection

    assert stream.is_connected()
    await stream.stop()


@pytest.mark.asyncio
async def test_receive_btc_price():
    """Test receiving BTC price update."""
    settings = Settings()
    stream = CryptoPriceStream(settings)

    # Start stream
    asyncio.create_task(stream.start())
    await asyncio.sleep(5)  # Wait longer for price update

    price = await stream.get_current_price()
    assert price is not None
    assert price.price > Decimal("0")
    assert price.source == "polymarket"

    await stream.stop()


@pytest.mark.asyncio
async def test_btc_price_service_with_polymarket():
    """Test BTCPriceService uses Polymarket WebSocket."""
    from polymarket.trading.btc_price import BTCPriceService

    settings = Settings()
    service = BTCPriceService(settings)

    # Start service (initializes WebSocket)
    await service.start()
    await asyncio.sleep(2)  # Wait for price

    # Get current price
    price = await service.get_current_price()
    assert price.source == "polymarket"
    assert price.price > Decimal("0")

    await service.close()

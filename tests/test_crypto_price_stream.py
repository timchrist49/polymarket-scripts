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


@pytest.mark.asyncio
async def test_price_buffer_integration():
    """Test WebSocket price updates are appended to price buffer."""
    import tempfile
    import os

    settings = Settings()

    # Use temporary file for test
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Create stream with buffer enabled
        stream = CryptoPriceStream(settings, buffer_enabled=True, buffer_file=tmp_path)

        # Start stream
        asyncio.create_task(stream.start())
        await asyncio.sleep(5)  # Wait for price updates

        # Verify buffer has been initialized
        assert stream.price_buffer is not None
        assert stream.price_buffer.size() > 0, "Buffer should contain price updates"

        # Get current price from stream
        current_price = await stream.get_current_price()
        assert current_price is not None

        # Verify we can query the buffer for recent prices
        timestamp = int(current_price.timestamp.timestamp())
        buffer_price = await stream.price_buffer.get_price_at(timestamp, tolerance=30)
        assert buffer_price is not None, "Should find price in buffer"

        # Stop and verify save to disk
        await stream.stop()

        # Verify file was created
        assert os.path.exists(tmp_path), "Buffer should be saved to disk"

    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

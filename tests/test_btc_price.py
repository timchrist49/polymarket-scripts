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


@pytest.mark.asyncio
async def test_fetch_binance_at_timestamp_checks_buffer_first():
    """Test that buffer is queried before Binance API."""
    settings = Settings()
    service = BTCPriceService(settings)

    # Start service (enables buffer)
    await service.start()
    await asyncio.sleep(1.5)  # Wait for initial price from WebSocket

    # Get the current timestamp from the buffer's latest price
    if service._stream and service._stream.price_buffer:
        # Query a price that should be in the buffer (from WebSocket)
        buffer = service._stream.price_buffer
        if not buffer.is_empty():
            # Get the most recent entry in buffer
            recent_entry = buffer._buffer[-1]
            recent_timestamp = recent_entry.timestamp

            # Fetch price - should come from buffer, not Binance
            price = await service._fetch_binance_at_timestamp(recent_timestamp)

            # Verify we got a price (from buffer)
            assert price is not None
            assert price > Decimal("0")

            # The price should match what's in the buffer
            expected_price = recent_entry.price
            assert price == expected_price
        else:
            pytest.skip("No prices in buffer yet")
    else:
        pytest.skip("Buffer not available")

    await service.close()


@pytest.mark.asyncio
async def test_fetch_binance_at_timestamp_falls_back_to_binance():
    """Test fallback to Binance when buffer doesn't have data."""
    settings = Settings()
    service = BTCPriceService(settings)
    await service.start()
    await asyncio.sleep(0.5)

    # Request price not in buffer (very old timestamp)
    old_timestamp = int(datetime.now().timestamp()) - (48 * 3600)  # 48 hours ago

    # Mock Binance response
    with patch.object(service, '_get_session') as mock_session:
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value=[
            [old_timestamp * 1000, "66500.00", "66600", "66400", "66550", "100", None, None]
        ])
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock()

        mock_session_obj = MagicMock()
        mock_session_obj.get.return_value = mock_response
        mock_session.return_value = mock_session_obj

        # Should fall back to Binance
        price = await service._fetch_binance_at_timestamp(old_timestamp)

        # Price should be fetched from Binance (open price)
        assert price == Decimal("66500.00")

    await service.close()


@pytest.mark.asyncio
async def test_fetch_binance_at_timestamp_buffer_disabled():
    """Test service works when buffer is disabled."""
    settings = Settings()
    service = BTCPriceService(settings)
    # Don't start service - buffer will be None

    timestamp = int(datetime.now().timestamp()) - 3600

    # Mock Binance response
    with patch.object(service, '_get_session') as mock_session:
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value=[
            [timestamp * 1000, "68000.00", "68100", "67900", "68050", "100", None, None]
        ])
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock()

        mock_session_obj = MagicMock()
        mock_session_obj.get.return_value = mock_response
        mock_session.return_value = mock_session_obj

        # Should go directly to Binance (buffer not initialized)
        price = await service._fetch_binance_at_timestamp(timestamp)

        assert price == Decimal("68000.00")

    await service.close()


@pytest.mark.asyncio
async def test_calculate_15min_volatility():
    """Test 15-minute volatility calculation."""
    settings = Settings()
    service = BTCPriceService(settings)

    # Start service to initialize buffer
    await service.start()
    await asyncio.sleep(0.5)

    # Mock the price buffer with 4 sample prices
    # Prices: 67000, 67200, 67100, 67300
    # Returns: (67200-67000)/67000 = 0.00298..., (67100-67200)/67200 = -0.00149..., (67300-67100)/67100 = 0.00298...
    mock_prices = [
        PricePoint(price=Decimal("67000"), volume=Decimal("100"), timestamp=datetime.now()),
        PricePoint(price=Decimal("67200"), volume=Decimal("100"), timestamp=datetime.now()),
        PricePoint(price=Decimal("67100"), volume=Decimal("100"), timestamp=datetime.now()),
        PricePoint(price=Decimal("67300"), volume=Decimal("100"), timestamp=datetime.now()),
    ]

    # Mock the buffer's get_price_range method
    if service._stream and service._stream.price_buffer:
        # Use AsyncMock to return the list as a coroutine
        from unittest.mock import AsyncMock
        service._stream.price_buffer.get_price_range = AsyncMock(return_value=mock_prices)

        volatility = await service.calculate_15min_volatility()

        # Verify volatility is within reasonable range
        assert volatility > 0.0
        assert volatility < 0.10  # 10% is unreasonably high for 15min

        # Verify it's not the default value (indicates calculation worked)
        assert volatility != 0.005
    else:
        pytest.skip("Buffer not available")

    await service.close()

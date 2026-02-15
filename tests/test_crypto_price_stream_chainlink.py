"""Tests for Chainlink RTDS integration in CryptoPriceStream."""

import pytest
import json
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from polymarket.trading.crypto_price_stream import CryptoPriceStream
from polymarket.config import Settings


@pytest.mark.asyncio
async def test_chainlink_subscription_format():
    """Test that Chainlink subscription uses correct format."""
    settings = Settings()
    stream = CryptoPriceStream(settings, use_chainlink=True)

    # Mock websocket
    mock_ws = AsyncMock()

    with patch('websockets.connect', return_value=mock_ws):
        # Start connection (will fail since we're mocking)
        try:
            await stream._subscribe_to_feed(mock_ws)
        except:
            pass

    # Verify subscription message format
    assert mock_ws.send.called
    sent_msg = json.loads(mock_ws.send.call_args[0][0])

    # Chainlink format requirements
    assert sent_msg["action"] == "subscribe"
    assert sent_msg["subscriptions"][0]["topic"] == "crypto_prices_chainlink"
    assert sent_msg["subscriptions"][0]["type"] == "*"
    assert sent_msg["subscriptions"][0]["filters"] == '{"symbol":"btc/usd"}'


@pytest.mark.asyncio
async def test_binance_subscription_format():
    """Test that Binance subscription still works (backward compatibility)."""
    settings = Settings()
    stream = CryptoPriceStream(settings, use_chainlink=False)

    mock_ws = AsyncMock()

    with patch('websockets.connect', return_value=mock_ws):
        try:
            await stream._subscribe_to_feed(mock_ws)
        except:
            pass

    sent_msg = json.loads(mock_ws.send.call_args[0][0])

    # Binance format requirements
    assert sent_msg["subscriptions"][0]["topic"] == "crypto_prices"
    assert sent_msg["subscriptions"][0]["type"] == "update"
    assert sent_msg["subscriptions"][0]["filters"] == "btcusdt"


@pytest.mark.asyncio
async def test_parse_chainlink_initial_message():
    """Test parsing Chainlink initial data dump."""
    settings = Settings()
    stream = CryptoPriceStream(settings, use_chainlink=True)

    # Simulate initial Chainlink message
    chainlink_initial = json.dumps({
        "topic": "crypto_prices_chainlink",
        "type": "subscribe",
        "payload": {
            "data": [
                {"timestamp": 1771133368000, "value": 70314.50691332904},
                {"timestamp": 1771133372000, "value": 70312.32709805031}
            ]
        }
    })

    await stream._handle_message(chainlink_initial)

    # Should parse latest price
    current = await stream.get_current_price()
    assert current is not None
    assert current.price == pytest.approx(Decimal("70312.33"), abs=0.01)
    assert current.source == "chainlink"


@pytest.mark.asyncio
async def test_parse_chainlink_update_message():
    """Test parsing Chainlink real-time update."""
    settings = Settings()
    stream = CryptoPriceStream(settings, use_chainlink=True)

    # Simulate real-time update
    chainlink_update = json.dumps({
        "topic": "crypto_prices_chainlink",
        "type": "update",
        "timestamp": 1771133430140,
        "payload": {
            "symbol": "btc/usd",
            "value": 70283.97530686231,
            "full_accuracy_value": "70283975306862305000000",
            "timestamp": 1771133429000
        }
    })

    await stream._handle_message(chainlink_update)

    current = await stream.get_current_price()
    assert current is not None
    assert current.price == pytest.approx(Decimal("70283.98"), abs=0.01)
    assert current.source == "chainlink"

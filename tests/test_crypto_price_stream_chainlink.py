"""Tests for Chainlink RTDS integration in CryptoPriceStream."""

import pytest
import json
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

# tests/test_telegram_bot.py
import pytest
from unittest.mock import Mock, AsyncMock, patch
from polymarket.telegram.bot import TelegramBot
from polymarket.config import Settings

@pytest.fixture
def mock_settings():
    """Mock settings with Telegram config."""
    settings = Mock(spec=Settings)
    settings.telegram_bot_token = "test-token"
    settings.telegram_chat_id = "test-chat-id"
    settings.telegram_enabled = True
    return settings

@pytest.mark.asyncio
async def test_send_notification(mock_settings):
    """Test sending a notification."""
    bot = TelegramBot(mock_settings)

    with patch.object(bot, '_send_message', new_callable=AsyncMock) as mock_send:
        await bot.send_trade_alert(
            market_slug="btc-updown-15m-123",
            action="NO",
            confidence=1.0,
            position_size=5.0,
            price=0.52,
            reasoning="Test reasoning"
        )

        mock_send.assert_called_once()
        args = mock_send.call_args[0]
        message = args[0]

        assert "Trade Executed" in message
        assert "NO" in message
        assert "0.52" in message

@pytest.mark.asyncio
async def test_disabled_telegram(mock_settings):
    """Test that disabled telegram doesn't send."""
    mock_settings.telegram_enabled = False
    bot = TelegramBot(mock_settings)

    # Should not raise, just return silently
    await bot.send_trade_alert(
        market_slug="test",
        action="YES",
        confidence=0.8,
        position_size=5.0,
        price=0.50,
        reasoning="Test"
    )

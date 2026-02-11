# tests/test_telegram_bot.py
import pytest
import asyncio
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

@pytest.mark.asyncio
async def test_request_approval_approved(mock_settings):
    """Test approval request with immediate approval."""
    bot = TelegramBot(mock_settings)

    # Mock the bot's send_message and simulate approval
    with patch.object(bot, '_bot') as mock_bot:
        mock_message = Mock()
        mock_message.message_id = 12345
        mock_bot.send_message = AsyncMock(return_value=mock_message)

        # Simulate immediate approval
        async def approve_immediately():
            await asyncio.sleep(0.1)
            approval_key = f"bot_confidence_threshold_12345"
            if approval_key in bot._pending_approvals:
                bot._pending_approvals[approval_key]["approved"] = True

        approval_task = asyncio.create_task(approve_immediately())

        result = await bot.request_approval(
            parameter_name="bot_confidence_threshold",
            old_value=0.75,
            new_value=0.68,
            reason="Test",
            change_pct=-9.3,
            timeout_hours=4
        )

        await approval_task
        assert result is True

@pytest.mark.asyncio
async def test_request_approval_rejected(mock_settings):
    """Test approval request with immediate rejection."""
    bot = TelegramBot(mock_settings)

    with patch.object(bot, '_bot') as mock_bot:
        mock_message = Mock()
        mock_message.message_id = 12346
        mock_bot.send_message = AsyncMock(return_value=mock_message)

        # Simulate immediate rejection
        async def reject_immediately():
            await asyncio.sleep(0.1)
            approval_key = f"bot_max_position_dollars_12346"
            if approval_key in bot._pending_approvals:
                bot._pending_approvals[approval_key]["approved"] = False

        reject_task = asyncio.create_task(reject_immediately())

        result = await bot.request_approval(
            parameter_name="bot_max_position_dollars",
            old_value=10.0,
            new_value=12.0,
            reason="Test rejection",
            change_pct=20.0,
            timeout_hours=4
        )

        await reject_task
        assert result is False

@pytest.mark.asyncio
async def test_request_approval_timeout(mock_settings):
    """Test approval request timeout."""
    bot = TelegramBot(mock_settings)

    with patch.object(bot, '_bot') as mock_bot:
        mock_message = Mock()
        mock_message.message_id = 12347
        mock_bot.send_message = AsyncMock(return_value=mock_message)

        # Don't simulate any response - let it timeout
        # Use very short timeout for test
        result = await bot.request_approval(
            parameter_name="bot_max_exposure_percent",
            old_value=0.30,
            new_value=0.36,
            reason="Test timeout",
            change_pct=20.0,
            timeout_hours=0.0001  # ~0.36 seconds
        )

        assert result is False
        # Should have cleaned up pending approval
        assert len(bot._pending_approvals) == 0

@pytest.mark.asyncio
async def test_request_approval_no_bot(mock_settings):
    """Test approval request when bot is disabled."""
    mock_settings.telegram_enabled = False
    bot = TelegramBot(mock_settings)

    result = await bot.request_approval(
        parameter_name="bot_confidence_threshold",
        old_value=0.75,
        new_value=0.68,
        reason="Test",
        change_pct=-9.3,
        timeout_hours=4
    )

    assert result is False

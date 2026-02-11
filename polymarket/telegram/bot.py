# polymarket/telegram/bot.py
"""Telegram bot implementation."""

from typing import Optional
import structlog
from telegram import Bot
from telegram.constants import ParseMode

from polymarket.config import Settings

logger = structlog.get_logger()


class TelegramBot:
    """Telegram bot for notifications and interactive control."""

    def __init__(self, settings: Settings):
        """
        Initialize Telegram bot.

        Args:
            settings: Bot settings with Telegram config
        """
        self.settings = settings
        self._bot: Optional[Bot] = None

        if settings.telegram_enabled:
            if not settings.telegram_bot_token:
                raise ValueError("TELEGRAM_BOT_TOKEN not configured")
            if not settings.telegram_chat_id:
                raise ValueError("TELEGRAM_CHAT_ID not configured")

            self._bot = Bot(token=settings.telegram_bot_token)
            logger.info("Telegram bot initialized")
        else:
            logger.info("Telegram bot disabled")

    async def send_trade_alert(
        self,
        market_slug: str,
        action: str,
        confidence: float,
        position_size: float,
        price: float,
        reasoning: str
    ):
        """Send trade execution alert."""
        if not self._bot:
            return

        message = f"""🎯 **Trade Executed**

Market: `{market_slug}`
Action: **{action}** ({"UP" if action == "YES" else "DOWN"})
Confidence: {confidence*100:.0f}%
Position: ${position_size:.2f} @ {price:.2f}

Reasoning: {reasoning}

Expected profit: ~${position_size * (1/price - 1):.2f} if correct
"""

        await self._send_message(message)

    async def _send_message(self, text: str):
        """Send message to configured chat."""
        try:
            await self._bot.send_message(
                chat_id=self.settings.telegram_chat_id,
                text=text,
                parse_mode=ParseMode.MARKDOWN
            )
            logger.debug("Telegram message sent")
        except Exception as e:
            logger.error("Failed to send Telegram message", error=str(e))

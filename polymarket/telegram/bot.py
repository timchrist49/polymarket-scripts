# polymarket/telegram/bot.py
"""Telegram bot implementation."""

from typing import Optional, Dict
import asyncio
import structlog
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
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
        self._pending_approvals: Dict[str, Dict] = {}

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

    async def request_approval(
        self,
        parameter_name: str,
        old_value: float,
        new_value: float,
        reason: str,
        change_pct: float,
        timeout_hours: int = 4
    ) -> bool:
        """
        Request approval for parameter adjustment via Telegram.

        Args:
            parameter_name: Parameter to adjust
            old_value: Current value
            new_value: Proposed value
            reason: Reason for adjustment
            change_pct: Percentage change
            timeout_hours: Hours to wait for approval

        Returns:
            True if approved, False if rejected or timeout
        """
        if not self._bot:
            return False

        message = f"""⚠️ **Parameter Adjustment Approval Required** (Tier 2)

Parameter: `{parameter_name}`
Current: {old_value:.4f}
Proposed: {new_value:.4f}
Change: {change_pct:+.1f}%

Reason: {reason}

This change requires your approval.
Timeout: {timeout_hours} hours
"""

        # Create inline keyboard
        keyboard = [
            [
                InlineKeyboardButton("✅ Approve", callback_data=f"approve_{parameter_name}"),
                InlineKeyboardButton("❌ Reject", callback_data=f"reject_{parameter_name}")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        try:
            # Send message with buttons
            sent_message = await self._bot.send_message(
                chat_id=self.settings.telegram_chat_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=reply_markup
            )

            # Store pending approval
            approval_key = f"{parameter_name}_{sent_message.message_id}"
            self._pending_approvals[approval_key] = {
                "parameter_name": parameter_name,
                "old_value": old_value,
                "new_value": new_value,
                "reason": reason,
                "approved": None,  # None = pending, True = approved, False = rejected
                "message_id": sent_message.message_id
            }

            # Wait for response with timeout
            timeout_seconds = timeout_hours * 3600
            start_time = asyncio.get_event_loop().time()

            while asyncio.get_event_loop().time() - start_time < timeout_seconds:
                approval_data = self._pending_approvals.get(approval_key)
                if approval_data and approval_data["approved"] is not None:
                    # Decision made
                    result = approval_data["approved"]
                    del self._pending_approvals[approval_key]
                    return result

                await asyncio.sleep(1)  # Check every second

            # Timeout - reject by default
            logger.warning("Approval request timed out", parameter=parameter_name)
            if approval_key in self._pending_approvals:
                del self._pending_approvals[approval_key]

            await self._send_message(f"⏱️ Approval request timed out for `{parameter_name}`. Change rejected.")
            return False

        except Exception as e:
            logger.error("Failed to request approval", error=str(e))
            return False

    async def handle_callback(self, callback_query):
        """Handle button callback from Telegram."""
        data = callback_query.data
        message_id = callback_query.message.message_id

        if data.startswith("approve_") or data.startswith("reject_"):
            action, parameter_name = data.split("_", 1)
            approval_key = f"{parameter_name}_{message_id}"

            if approval_key in self._pending_approvals:
                approved = (action == "approve")
                self._pending_approvals[approval_key]["approved"] = approved

                # Update message
                result_text = "✅ APPROVED" if approved else "❌ REJECTED"
                await callback_query.edit_message_text(
                    text=f"{callback_query.message.text}\n\n**Decision: {result_text}**",
                    parse_mode=ParseMode.MARKDOWN
                )

                await callback_query.answer(f"Parameter adjustment {result_text.lower()}")

                logger.info(
                    "Approval decision made",
                    parameter=parameter_name,
                    approved=approved
                )

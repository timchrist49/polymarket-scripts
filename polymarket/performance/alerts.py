"""Alert system for order verification anomalies."""

import structlog
from typing import Optional

logger = structlog.get_logger()


class VerificationAlerts:
    """Send alerts for order verification issues."""

    def __init__(self, telegram_bot):
        """
        Initialize alert system.

        Args:
            telegram_bot: TelegramBot instance for sending alerts
        """
        self.telegram = telegram_bot

    async def alert_order_not_filled(self, trade_id: int, order_id: str, status: str):
        """
        Alert when order shows as unfilled in API.

        Args:
            trade_id: Database trade ID
            order_id: Polymarket order ID
            status: Order status from API
        """
        message = (
            f"🚨 Order Not Filled\n"
            f"Trade ID: {trade_id}\n"
            f"Order ID: {order_id[:8]}...\n"
            f"Status: {status}\n"
            f"Action: Check Polymarket UI"
        )

        try:
            await self.telegram.send_message(message)
            logger.info("Sent unfilled order alert", trade_id=trade_id)
        except Exception as e:
            logger.error("Failed to send alert", error=str(e))

    async def alert_price_mismatch(
        self,
        trade_id: int,
        estimated: float,
        actual: float,
        discrepancy_pct: float
    ):
        """
        Alert when fill price differs significantly from estimate.

        Args:
            trade_id: Database trade ID
            estimated: Estimated price at decision time
            actual: Actual fill price from API
            discrepancy_pct: Percentage difference
        """
        message = (
            f"⚠️ Price Mismatch\n"
            f"Trade ID: {trade_id}\n"
            f"Expected: ${estimated:.3f}\n"
            f"Actual: ${actual:.3f}\n"
            f"Discrepancy: {discrepancy_pct:+.2f}%\n"
            f"Impact: {'Favorable' if discrepancy_pct < 0 else 'Unfavorable'}"
        )

        try:
            await self.telegram.send_message(message)
            logger.info(
                "Sent price mismatch alert",
                trade_id=trade_id,
                discrepancy_pct=discrepancy_pct
            )
        except Exception as e:
            logger.error("Failed to send alert", error=str(e))

    async def alert_partial_fill(
        self,
        trade_id: int,
        expected: float,
        filled: float,
        fill_pct: float
    ):
        """
        Alert when order only partially fills.

        Args:
            trade_id: Database trade ID
            expected: Expected fill amount (shares)
            filled: Actual filled amount (shares)
            fill_pct: Percentage filled
        """
        message = (
            f"📊 Partial Fill\n"
            f"Trade ID: {trade_id}\n"
            f"Expected: {expected:.2f} shares\n"
            f"Filled: {filled:.2f} shares\n"
            f"Fill Rate: {fill_pct:.1f}%\n"
            f"Note: P&L calculated on filled amount only"
        )

        try:
            await self.telegram.send_message(message)
            logger.info(
                "Sent partial fill alert",
                trade_id=trade_id,
                fill_pct=fill_pct
            )
        except Exception as e:
            logger.error("Failed to send alert", error=str(e))

    async def alert_verification_failed(self, trade_id: int, error: str):
        """
        Alert when verification API call fails.

        Args:
            trade_id: Database trade ID
            error: Error message
        """
        message = (
            f"❌ Verification Failed\n"
            f"Trade ID: {trade_id}\n"
            f"Error: {error}\n"
            f"Fallback: Using estimated data"
        )

        try:
            await self.telegram.send_message(message)
            logger.info("Sent verification failure alert", trade_id=trade_id)
        except Exception as e:
            logger.error("Failed to send alert", error=str(e))

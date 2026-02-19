# polymarket/telegram/bot.py
"""Telegram bot implementation."""

from typing import Optional, Dict, TYPE_CHECKING
import asyncio
import structlog
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode

from polymarket.config import Settings

if TYPE_CHECKING:
    from polymarket.performance.tracker import TestModeMetrics

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
        reasoning: str,
        btc_current: Optional[float] = None,
        btc_price_to_beat: Optional[float] = None
    ):
        """Send trade execution alert with BTC price context."""
        if not self._bot:
            return

        # Build BTC price context if available
        btc_context = ""
        if btc_current and btc_price_to_beat:
            price_diff = btc_current - btc_price_to_beat
            price_diff_pct = (price_diff / btc_price_to_beat) * 100
            direction = "📈 UP" if price_diff > 0 else "📉 DOWN" if price_diff < 0 else "➡️ FLAT"

            btc_context = f"""
**BTC Price Context:**
Price to Beat: ${btc_price_to_beat:,.2f}
Current BTC: ${btc_current:,.2f}
Movement: {direction} {price_diff_pct:+.2f}% (${price_diff:+,.2f})
"""

        # Sanitize AI-generated text to avoid breaking Telegram Markdown parser
        safe_reasoning = self._escape_markdown(reasoning)

        message = f"""🎯 **Trade Executed**

Market: `{market_slug}`
Action: **{action}** ({"UP" if action == "YES" else "DOWN"})
Confidence: {confidence*100:.0f}%
Position: ${position_size:.2f} @ {price:.2f}
{btc_context}
Reasoning: {safe_reasoning}

Expected profit: ~${position_size * (1/price - 1):.2f} if correct
"""

        await self._send_message(message)

    @staticmethod
    def _escape_markdown(text: str) -> str:
        """Escape characters that break Telegram's legacy Markdown parser.

        Telegram Markdown v1 treats _, *, `, [ as special. AI-generated text
        (reasoning, insights) can contain any of these and cause 400 errors.
        """
        # Only escape inside non-formatting spans — simplest approach: escape all
        # occurrences that aren't part of our own **bold** / `code` markup.
        # Since we control our own formatting, the safest fix is to strip these
        # from dynamic/user-supplied fields before insertion (done at call site),
        # then rely on this fallback for anything we missed.
        for ch in ['_', '[', ']']:
            text = text.replace(ch, f'\\{ch}')
        return text

    async def _send_message(self, text: str):
        """Send message to configured chat, with plain-text fallback.

        Attempts Markdown first. If Telegram returns 400 (e.g. unescaped
        special chars from AI-generated text), retries as plain text so the
        alert always gets delivered.
        """
        try:
            await self._bot.send_message(
                chat_id=self.settings.telegram_chat_id,
                text=text,
                parse_mode=ParseMode.MARKDOWN
            )
            logger.debug("Telegram message sent")
        except Exception as e:
            error_str = str(e)
            logger.warning(
                "Telegram Markdown send failed, retrying as plain text",
                error=error_str
            )
            # Strip markdown formatting characters for plain-text fallback
            plain = text
            for ch in ['*', '`', '\\']:
                plain = plain.replace(ch, '')
            try:
                await self._bot.send_message(
                    chat_id=self.settings.telegram_chat_id,
                    text=plain
                    # no parse_mode = plain text
                )
                logger.debug("Telegram message sent (plain text fallback)")
            except Exception as e2:
                logger.error("Failed to send Telegram message", error=str(e2))

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

    async def send_emergency_alert(
        self,
        parameter_name: str,
        old_value: float,
        new_value: float,
        reason: str,
        change_pct: float
    ):
        """Send urgent emergency alert for dangerous adjustment."""
        if not self._bot:
            return

        message = f"""🚨 **EMERGENCY PAUSE TRIGGERED** 🚨

⚠️ **Dangerous parameter adjustment detected**

Parameter: `{parameter_name}`
Current: {old_value:.4f}
Proposed: {new_value:.4f}
Change: {change_pct:+.1f}%

Reason: {reason}

🛑 **BOT HAS BEEN PAUSED**

This change exceeds the ±20% safety threshold.
Manual review required before resuming trading.

To resume:
1. Review the recommendation carefully
2. Manually adjust parameters if needed
3. Set EMERGENCY_PAUSE_ENABLED=false
4. Restart the bot

**Do not resume without understanding why this was triggered.**
"""

        await self._send_message(message)

    async def send_reflection_summary(
        self,
        trigger_type: str,
        insights_text: str,
        trades_analyzed: int
    ):
        """Send reflection analysis summary."""
        if not self._bot:
            return

        # Format trigger type for display
        trigger_display = {
            "10_trades": "10 Trades Milestone",
            "3_losses": "3 Consecutive Losses",
            "end_of_day": "End of Day Review"
        }.get(trigger_type, trigger_type)

        safe_insights = self._escape_markdown(insights_text)

        message = f"""🤖 **Self-Reflection Analysis**

**Trigger:** {trigger_display}
**Trades Analyzed:** {trades_analyzed}

**Key Insights:**
{safe_insights}

The bot will now process recommendations and adjust parameters according to the tiered autonomy system.
"""

        await self._send_message(message)

    async def send_test_mode_report(self, metrics: 'TestModeMetrics', trade_range: str) -> None:
        """Send test mode performance report.

        Args:
            metrics: Aggregated test mode metrics
            trade_range: Description like "Trades 21-40"
        """
        if not self._bot:
            return

        # Format alignment stats
        alignment_lines = []
        for alignment, count in metrics.timeframe_alignment_stats.items():
            pct = (count / metrics.total_trades) * 100
            alignment_lines.append(f"• {alignment}: {count} trades ({pct:.0f}%)")

        alignment_text = "\n".join(alignment_lines)

        message = f"""🎯 TEST MODE REPORT ({trade_range})

📊 Performance:
• Win Rate: {metrics.wins}/{metrics.wins + metrics.losses} ({metrics.win_rate:.1%})
• Total P&L: ${metrics.total_pnl:+.2f}
• Execution Rate: {metrics.executed_trades}/{metrics.total_trades} ({metrics.execution_rate:.1%})

📈 Trade Quality:
• Avg Arbitrage Edge: {metrics.avg_arbitrage_edge:.2%}
• Avg Confidence: {metrics.avg_confidence:.1%}

🕐 Timeframe Analysis:
{alignment_text}

Next report after 20 more trades."""

        await self._send_message(message)
        logger.info("Test mode report sent", trade_range=trade_range)

    async def send_paper_trade_alert(
        self,
        market_slug: str,
        action: str,
        confidence: float,
        position_size: float,
        executed_price: float,
        time_remaining_seconds: int,
        technical_summary: str,
        sentiment_summary: str,
        odds_yes: float,
        odds_no: float,
        odds_qualified: bool,
        timeframe_summary: str,
        signal_lag_detected: bool,
        signal_lag_reason: str | None,
        conflict_severity: str,
        conflicts_list: list[str],
        ai_reasoning: str
    ):
        """
        Send detailed paper trade alert to Telegram.

        Args:
            market_slug: Market identifier
            action: 'YES' or 'NO'
            confidence: AI confidence (0.0-1.0)
            position_size: Position size in USDC
            executed_price: Simulated execution price
            time_remaining_seconds: Time remaining in market
            technical_summary: Technical indicators summary
            sentiment_summary: Sentiment analysis summary
            odds_yes: YES token odds
            odds_no: NO token odds
            odds_qualified: Whether chosen side met > 75% threshold
            timeframe_summary: Timeframe alignment summary
            signal_lag_detected: Whether signal lag was detected
            signal_lag_reason: Reason for signal lag
            conflict_severity: 'NONE', 'MINOR', 'MODERATE', 'SEVERE'
            conflicts_list: List of conflict descriptions
            ai_reasoning: AI's reasoning text
        """
        # Format direction
        direction_emoji = "📈" if action == "YES" else "📉"
        token_name = "YES (UP)" if action == "YES" else "NO (DOWN)"

        # Format time remaining
        minutes = time_remaining_seconds // 60
        seconds = time_remaining_seconds % 60
        time_str = f"{minutes}m {seconds}s"

        # Format odds check
        chosen_odds = odds_yes if action == "YES" else odds_no
        odds_status = "✅" if odds_qualified else "❌"
        odds_check = f"{odds_status} Odds Check: {action} = {chosen_odds:.0%} ({'PASS' if odds_qualified else 'FAIL'} > 75%)"

        # Format signal lag
        lag_status = "⚠️" if signal_lag_detected else "✅"
        lag_text = f"{lag_status} Signal Lag: {'DETECTED' if signal_lag_detected else 'NO LAG DETECTED'}"
        if signal_lag_detected and signal_lag_reason:
            lag_text += f"\n   {signal_lag_reason}"

        # Format conflicts
        if conflict_severity == "NONE":
            conflicts_text = "✅ Conflicts: NONE"
        else:
            conflict_emoji = {"MINOR": "⚠️", "MODERATE": "⚠️⚠️", "SEVERE": "🚫"}[conflict_severity]
            conflicts_text = f"{conflict_emoji} Conflicts: {conflict_severity} ({len(conflicts_list)} detected)"
            for conflict in conflicts_list:
                conflicts_text += f"\n   - {conflict}"

        safe_ai_reasoning = self._escape_markdown(ai_reasoning)

        message = f"""🧪 PAPER TRADE SIGNAL
━━━━━━━━━━━━━━━━━━━━
📊 Market: {market_slug}
{direction_emoji} Direction: {token_name}
💵 Position: ${position_size:.2f} @ {executed_price:.2f} odds
⏰ Time Remaining: {time_str}

🎯 SIGNAL ANALYSIS:
{technical_summary}
{sentiment_summary}
{odds_check}
{timeframe_summary}
{lag_text}

🤖 AI REASONING:
"{safe_ai_reasoning}"

📊 CONFIDENCE: {confidence:.2f}
{conflicts_text}

━━━━━━━━━━━━━━━━━━━━"""

        await self._send_message(message)

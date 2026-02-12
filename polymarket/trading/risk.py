"""
Risk Management Module

Handles position sizing, stop-loss evaluation, and portfolio safety.
Validates all trading decisions before execution.
"""

from decimal import Decimal
from datetime import datetime, timedelta
from typing import Optional
import structlog

from polymarket.models import (
    TradingDecision,
    ValidationResult
)
from polymarket.config import Settings

logger = structlog.get_logger()


class RiskManager:
    """Risk management and position sizing."""

    def __init__(self, settings: Settings):
        self.settings = settings

    async def validate_decision(
        self,
        decision: TradingDecision,
        portfolio_value: Decimal,
        market: dict,
        open_positions: Optional[list[dict]] = None
    ) -> ValidationResult:
        """Validate a trading decision against risk rules."""

        # Check 1: Confidence threshold
        if decision.confidence < self.settings.bot_confidence_threshold:
            return ValidationResult(
                approved=False,
                reason=f"Confidence {decision.confidence:.2f} below threshold {self.settings.bot_confidence_threshold}",
                adjusted_position=None
            )

        # Check 2: Not a HOLD action
        if decision.action == "HOLD":
            return ValidationResult(
                approved=False,
                reason="Action is HOLD",
                adjusted_position=None
            )

        # Check 3: Calculate position size
        max_position = portfolio_value * Decimal(str(self.settings.bot_max_position_percent))
        suggested_size = self._calculate_position_size(
            decision, portfolio_value, max_position
        )

        # Check 4: Total exposure
        open_exposure = Decimal("0")
        if open_positions:
            open_exposure = sum(
                Decimal(str(p.get("amount", 0))) for p in open_positions
            )

        max_exposure = portfolio_value * Decimal(str(self.settings.bot_max_exposure_percent))

        if open_exposure + suggested_size > max_exposure:
            return ValidationResult(
                approved=False,
                reason=f"Total exposure {open_exposure + suggested_size} would exceed max {max_exposure}",
                adjusted_position=None
            )

        # Check 5: Sufficient funds
        if suggested_size > portfolio_value:
            return ValidationResult(
                approved=False,
                reason=f"Position size {suggested_size} exceeds portfolio {portfolio_value}",
                adjusted_position=None
            )

        # Check 6: Not already positioned in this market
        if open_positions:
            for pos in open_positions:
                if pos.get("token_id") == decision.token_id:
                    return ValidationResult(
                        approved=False,
                        reason=f"Already positioned in market {decision.token_id}",
                        adjusted_position=None
                    )

        # Check 7: Market is accepting orders
        if not market.get("active", True):
            return ValidationResult(
                approved=False,
                reason="Market is not active",
                adjusted_position=None
            )

        # Approved
        return ValidationResult(
            approved=True,
            reason="All risk checks passed",
            adjusted_position=suggested_size
        )

    def _calculate_position_size(
        self,
        decision: TradingDecision,
        portfolio_value: Decimal,
        max_position: Decimal
    ) -> Decimal:
        """Calculate position size based on confidence."""
        base_size = portfolio_value * Decimal(str(self.settings.bot_max_position_percent))

        # Scale by confidence
        confidence = decision.confidence

        if 0.75 <= confidence < 0.80:
            multiplier = Decimal("0.5")
        elif 0.80 <= confidence < 0.90:
            multiplier = Decimal("0.75")
        elif confidence >= 0.90:
            multiplier = Decimal("1.0")
        else:
            multiplier = Decimal("0.0")

        calculated = base_size * multiplier

        # Apply absolute dollar cap
        dollar_cap = Decimal(str(self.settings.bot_max_position_dollars))
        calculated = min(calculated, dollar_cap)
        max_position = min(max_position, dollar_cap)

        # Use AI-suggested size if provided and reasonable
        if decision.position_size > 0:
            ai_size = min(decision.position_size, max_position)
            return min(ai_size, calculated)

        return min(calculated, max_position)

    def _calculate_odds_multiplier(self, odds: Decimal) -> Decimal:
        """
        Scale down position size for low-odds bets.

        Logic:
        - odds >= 0.50: No scaling (100% of position)
        - odds < 0.50:  Linear scale from 100% down to 50%
        - odds < 0.25:  Reject bet entirely (too risky)

        Examples:
        - 0.83 odds → 1.00x (no reduction)
        - 0.50 odds → 1.00x (breakeven)
        - 0.40 odds → 0.80x (20% reduction)
        - 0.31 odds → 0.62x (38% reduction)
        - 0.25 odds → 0.50x (50% reduction, minimum)
        - 0.20 odds → REJECT (below threshold)

        Args:
            odds: The odds for the side being bet (Decimal 0.0 to 1.0)

        Returns:
            Decimal: Multiplier between 0.0 (reject) and 1.0 (no scaling)
        """
        MINIMUM_ODDS = Decimal("0.25")
        SCALE_THRESHOLD = Decimal("0.50")

        if odds < MINIMUM_ODDS:
            logger.info(
                "Bet rejected - odds below minimum threshold",
                odds=float(odds),
                minimum=0.25
            )
            return Decimal("0")

        if odds >= SCALE_THRESHOLD:
            return Decimal("1.0")

        # Linear interpolation between 0.5x and 1.0x
        multiplier = Decimal("0.5") + (odds - MINIMUM_ODDS) / (SCALE_THRESHOLD - MINIMUM_ODDS) * Decimal("0.5")

        logger.debug(
            "Odds multiplier calculated",
            odds=float(odds),
            multiplier=float(multiplier)
        )

        return multiplier

    def _extract_odds_for_action(self, action: str, market: dict) -> Decimal:
        """
        Get the odds for the side being bet.

        Args:
            action: Trading action ("YES", "NO", "HOLD")
            market: Market data containing yes_price and no_price

        Returns:
            Decimal: Odds for the action (0.0 to 1.0), defaults to 0.50
        """
        if action == "YES":
            odds = market.get("yes_price") or 0.50  # Handle None and missing
        elif action == "NO":
            odds = market.get("no_price") or 0.50  # Handle None and missing
        else:
            # HOLD or invalid action
            odds = 0.50

        return Decimal(str(odds))

    async def evaluate_stop_loss(
        self,
        open_positions: list[dict],
        current_markets: dict[str, dict]  # token_id -> market data
    ) -> list[dict]:
        """Evaluate stop-loss conditions for open positions."""
        to_close = []

        for position in open_positions:
            token_id = position.get("token_id")
            if not token_id:
                continue

            market = current_markets.get(token_id)
            if not market:
                continue

            # Get position details
            side = position.get("side", "YES")  # YES or NO
            entry_odds = float(position.get("entry_odds", 0.50))

            # Current odds
            yes_price = float(market.get("yes_price", 0.50))
            no_price = float(market.get("no_price", 0.50))

            # Check stop-loss conditions
            should_close = False
            reason = ""

            # Condition 1: Odds for our position dropped below threshold
            if side == "YES" and yes_price < self.settings.stop_loss_odds_threshold:
                should_close = True
                reason = f"YES odds {yes_price:.2f} below threshold {self.settings.stop_loss_odds_threshold}"
            elif side == "NO" and no_price < self.settings.stop_loss_odds_threshold:
                should_close = True
                reason = f"NO odds {no_price:.2f} below threshold {self.settings.stop_loss_odds_threshold}"

            # Condition 2: Odds against our position rose above 0.70
            if not should_close:
                if side == "YES" and no_price > 0.70:
                    should_close = True
                    reason = f"NO odds {no_price:.2f} above 0.70 (bearish signal)"
                elif side == "NO" and yes_price > 0.70:
                    should_close = True
                    reason = f"YES odds {yes_price:.2f} above 0.70 (bullish signal)"

            # Condition 3: Time-based exit (5 minutes before expiry)
            if not should_close:
                end_time = market.get("end_time")
                if end_time:
                    if isinstance(end_time, str):
                        end_time = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
                    time_remaining = end_time - datetime.now()
                    if time_remaining.total_seconds() <= self.settings.stop_loss_force_exit_minutes * 60:
                        should_close = True
                        reason = f"Forced exit: {time_remaining.seconds // 60} minutes remaining"

            if should_close:
                to_close.append({
                    "token_id": token_id,
                    "side": side,
                    "reason": reason,
                    "current_yes": yes_price,
                    "current_no": no_price
                })

        return to_close

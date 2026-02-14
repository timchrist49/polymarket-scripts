"""
Arbitrage Detector

Detects mispriced markets by comparing actual probability (from ProbabilityCalculator)
against Polymarket odds. When edge >= 5%, generates trading recommendations.

Edge = |actual_probability - market_odds|
Action = BUY_YES if actual > yes_odds, BUY_NO if (1-actual) > no_odds, else HOLD
Confidence boost = edge * 2, capped at 0.20
Urgency = HIGH (15%+), MEDIUM (10-15%), LOW (<10%)
"""

import structlog
from polymarket.models import ArbitrageOpportunity

logger = structlog.get_logger()


class ArbitrageDetector:
    """Detect arbitrage opportunities from mispriced markets."""

    # Constants
    MIN_EDGE = 0.05  # Minimum 5% edge to trade
    HIGH_EDGE_THRESHOLD = 0.10  # 10%+ = MEDIUM urgency
    EXTREME_EDGE_THRESHOLD = 0.15  # 15%+ = HIGH urgency
    MAX_CONFIDENCE_BOOST = 0.20  # Cap boost at 20%

    def __init__(self):
        """Initialize arbitrage detector."""
        pass

    def _get_minimum_edge(self, probability: float) -> float:
        """
        Calculate minimum edge threshold based on prediction confidence.

        High confidence predictions can accept smaller edges.
        Low confidence predictions require larger edges for safety.

        Args:
            probability: Actual probability from ProbabilityCalculator (0.0 to 1.0)

        Returns:
            Minimum edge threshold (0.05 to 0.12)

        Examples:
            >>> detector._get_minimum_edge(0.70)  # High confidence
            0.05
            >>> detector._get_minimum_edge(0.65)  # Medium confidence
            0.08
            >>> detector._get_minimum_edge(0.55)  # Low confidence
            0.12
        """
        # Calculate confidence as distance from 50% (0.0 to 1.0)
        confidence = abs(probability - 0.5) * 2

        # Use slightly lower thresholds to handle floating point precision
        if confidence >= 0.39:  # Probability >= 70% or <= 30%
            return 0.05  # 5% edge sufficient for high confidence
        elif confidence >= 0.19:  # Probability 60-70% or 30-40%
            return 0.08  # 8% edge required for medium confidence
        else:  # Probability 50-60% or 40-50%
            return 0.12  # 12% edge required for low confidence (conservative)

    def detect_arbitrage(
        self,
        actual_probability: float,
        market_yes_odds: float,
        market_no_odds: float,
        market_id: str,
        time_remaining_seconds: int = 3600
    ) -> ArbitrageOpportunity:
        """
        Detect opportunities by following probability direction.

        CRITICAL LOGIC:
        - If probability >= 50%: We predict YES, only check YES edge
        - If probability < 50%: We predict NO, only check NO edge
        - Never bet against our own probability prediction

        Args:
            actual_probability: Calculated probability from ProbabilityCalculator
            market_yes_odds: Current YES odds on Polymarket
            market_no_odds: Current NO odds on Polymarket
            market_id: Polymarket market ID
            time_remaining_seconds: Seconds until market settlement

        Returns:
            ArbitrageOpportunity with action, confidence, urgency
        """
        # Calculate edges for both sides
        yes_edge = actual_probability - market_yes_odds
        no_edge = (1.0 - actual_probability) - market_no_odds

        # Get confidence-adjusted minimum edge threshold
        min_edge = self._get_minimum_edge(actual_probability)

        # CRITICAL: Only trade in probability direction
        if actual_probability >= 0.50:
            # We predict YES - only consider YES edge
            if yes_edge >= min_edge:
                action = "BUY_YES"
                edge = yes_edge
                expected_profit = ((1.0 - market_yes_odds) / market_yes_odds) if market_yes_odds > 0 else 0.0
            else:
                action = "HOLD"
                edge = yes_edge
                expected_profit = 0.0

            logger.info(
                "Probability direction: YES",
                actual_prob=f"{actual_probability:.2%}",
                yes_edge=f"{yes_edge:+.2%}",
                min_edge_required=f"{min_edge:.2%}",
                action=action
            )
        else:
            # We predict NO - only consider NO edge
            if no_edge >= min_edge:
                action = "BUY_NO"
                edge = no_edge
                expected_profit = ((1.0 - market_no_odds) / market_no_odds) if market_no_odds > 0 else 0.0
            else:
                action = "HOLD"
                edge = no_edge
                expected_profit = 0.0

            logger.info(
                "Probability direction: NO",
                actual_prob=f"{actual_probability:.2%}",
                no_edge=f"{no_edge:+.2%}",
                min_edge_required=f"{min_edge:.2%}",
                action=action
            )

        # Calculate confidence boost (only if trading)
        if action != "HOLD":
            confidence_boost = min(edge * 2, self.MAX_CONFIDENCE_BOOST)
        else:
            confidence_boost = 0.0

        # Determine urgency based on edge size
        if edge >= self.EXTREME_EDGE_THRESHOLD:
            urgency = "HIGH"
        elif edge >= self.HIGH_EDGE_THRESHOLD:
            urgency = "MEDIUM"
        else:
            urgency = "LOW"

        # Log detected opportunity
        if action != "HOLD":
            logger.info(
                "Arbitrage opportunity detected",
                market_id=market_id,
                action=action,
                edge_pct=f"{edge:.2%}",
                actual_prob=f"{actual_probability:.2%}",
                yes_odds=f"{market_yes_odds:.2%}",
                no_odds=f"{market_no_odds:.2%}",
                confidence_boost=f"{confidence_boost:.2%}",
                urgency=urgency,
                expected_profit_pct=f"{expected_profit:.2%}"
            )

        return ArbitrageOpportunity(
            market_id=market_id,
            actual_probability=actual_probability,
            polymarket_yes_odds=market_yes_odds,
            polymarket_no_odds=market_no_odds,
            edge_percentage=edge,
            recommended_action=action,
            confidence_boost=confidence_boost,
            urgency=urgency,
            expected_profit_pct=expected_profit
        )

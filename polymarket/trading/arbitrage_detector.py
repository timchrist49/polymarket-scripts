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

    def detect_arbitrage(
        self,
        actual_probability: float,
        market_yes_odds: float,
        market_no_odds: float,
        market_id: str,
        ai_base_confidence: float = 0.75
    ) -> ArbitrageOpportunity:
        """
        Detect arbitrage opportunities by comparing actual vs market odds.

        Args:
            actual_probability: Calculated probability from ProbabilityCalculator
            market_yes_odds: Current YES odds on Polymarket
            market_no_odds: Current NO odds on Polymarket
            market_id: Polymarket market ID
            ai_base_confidence: Base confidence for AI decision (default 0.75)

        Returns:
            ArbitrageOpportunity with action, confidence boost, urgency

        Example:
            >>> detector = ArbitrageDetector()
            >>> opp = detector.detect_arbitrage(
            ...     actual_probability=0.68,
            ...     market_yes_odds=0.55,
            ...     market_no_odds=0.45,
            ...     market_id="btc-market-1"
            ... )
            >>> print(f"Action: {opp.recommended_action}")
            Action: BUY_YES
            >>> print(f"Edge: {opp.edge_percentage:.2%}")
            Edge: 13.00%
            >>> print(f"Urgency: {opp.urgency}")
            Urgency: MEDIUM
        """

        # Calculate edges for YES and NO
        yes_edge = actual_probability - market_yes_odds
        no_edge = (1.0 - actual_probability) - market_no_odds

        # Determine which edge is larger (absolute value) and positive
        # We only trade if the edge is positive (mispricing in our favor)
        if yes_edge >= self.MIN_EDGE and yes_edge >= no_edge:
            action = "BUY_YES"
            edge = yes_edge
            # Expected profit = ROI on Polymarket (deterministic)
            # Buy YES at market_yes_odds, win pays $1.00
            expected_profit = ((1.0 - market_yes_odds) / market_yes_odds) if market_yes_odds > 0 else 0.0
        elif no_edge >= self.MIN_EDGE and no_edge > yes_edge:
            action = "BUY_NO"
            edge = no_edge
            # Expected profit = ROI on Polymarket (deterministic)
            # Buy NO at market_no_odds, win pays $1.00
            expected_profit = ((1.0 - market_no_odds) / market_no_odds) if market_no_odds > 0 else 0.0
        else:
            action = "HOLD"
            edge = max(yes_edge, no_edge)  # Store the larger edge for reporting
            expected_profit = 0.0

        # Calculate confidence boost (edge * 2, capped at 0.20)
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

"""
Signal Aggregator

Combines social sentiment and market microstructure with dynamic confidence.
"""

from datetime import datetime
import structlog

from polymarket.models import SocialSentiment, MarketSignals, AggregatedSentiment

logger = structlog.get_logger()


class SignalAggregator:
    """Aggregates social and market signals with agreement-based confidence."""

    # Weights for final score calculation
    SOCIAL_WEIGHT = 0.4
    MARKET_WEIGHT = 0.6

    # Agreement boost/penalty limits
    MAX_BOOST = 0.5    # Max 1.5x confidence boost
    MAX_PENALTY = 0.5  # Max 0.5x confidence penalty

    # Confidence penalty when one source missing
    MISSING_SOURCE_PENALTY = 0.7  # Use 70% of confidence

    def _calculate_agreement_score(self, score1: float, score2: float) -> float:
        """
        Calculate agreement multiplier based on signal alignment.

        Returns:
            0.5 (total conflict) to 1.5 (perfect agreement)
        """
        # Both in same direction (both positive or both negative)
        if score1 * score2 > 0:
            # How aligned are they? (0 to 1)
            alignment = 1 - abs(score1 - score2) / 2
            # Boost confidence (1.0 to 1.5x)
            return 1.0 + (alignment * self.MAX_BOOST)

        # Opposite directions (conflict)
        elif score1 * score2 < 0:
            # How conflicted? (0 to 1)
            conflict = abs(score1 - score2) / 2
            # Penalize confidence (0.5 to 1.0x)
            return 1.0 - (conflict * self.MAX_PENALTY)

        # One or both neutral
        else:
            return 1.0  # No boost or penalty

    def _classify_signal(self, score: float, confidence: float) -> str:
        """Classify signal strength and direction."""
        direction = "BULLISH" if score > 0.1 else "BEARISH" if score < -0.1 else "NEUTRAL"
        strength = "STRONG" if confidence >= 0.7 else "WEAK" if confidence >= 0.5 else "CONFLICTED"
        return f"{strength}_{direction}"

    def aggregate(
        self,
        social: SocialSentiment,
        market: MarketSignals
    ) -> AggregatedSentiment:
        """
        Aggregate social and market signals with dynamic confidence.

        Args:
            social: Social sentiment data
            market: Market microstructure data

        Returns:
            AggregatedSentiment with final score, confidence, and agreement info.
        """
        # Case 1: Both signals available
        if social.confidence > 0 and market.confidence > 0:
            # Calculate weighted final score
            final_score = (
                market.score * self.MARKET_WEIGHT +
                social.score * self.SOCIAL_WEIGHT
            )

            # Base confidence from individual confidences
            base_confidence = (social.confidence + market.confidence) / 2

            # Calculate agreement multiplier
            agreement_multiplier = self._calculate_agreement_score(social.score, market.score)

            # Apply agreement to confidence
            final_confidence = min(base_confidence * agreement_multiplier, 1.0)

            # Classify signal
            signal_type = self._classify_signal(final_score, final_confidence)

            logger.info(
                "Signals aggregated",
                social_score=f"{social.score:+.2f}",
                market_score=f"{market.score:+.2f}",
                final_score=f"{final_score:+.2f}",
                final_conf=f"{final_confidence:.2f}",
                agreement=f"{agreement_multiplier:.2f}x",
                signal=signal_type
            )

        # Case 2: Only market available
        elif market.confidence > 0:
            final_score = market.score
            base_confidence = market.confidence
            agreement_multiplier = 1.0
            final_confidence = base_confidence * self.MISSING_SOURCE_PENALTY
            signal_type = f"MARKET_ONLY_{market.signal_type}"

            logger.warning("Using market signals only (social unavailable)")

        # Case 3: Only social available
        elif social.confidence > 0:
            final_score = social.score
            base_confidence = social.confidence
            agreement_multiplier = 1.0
            final_confidence = base_confidence * self.MISSING_SOURCE_PENALTY
            signal_type = f"SOCIAL_ONLY_{social.signal_type}"

            logger.warning("Using social signals only (market unavailable)")

        # Case 4: Both unavailable
        else:
            final_score = 0.0
            base_confidence = 0.0
            agreement_multiplier = 0.0
            final_confidence = 0.0
            signal_type = "TECHNICAL_ONLY"

            logger.warning("All sentiment unavailable, falling back to technical indicators")

        return AggregatedSentiment(
            social=social,
            market=market,
            final_score=final_score,
            final_confidence=final_confidence,
            agreement_multiplier=agreement_multiplier,
            signal_type=signal_type,
            timestamp=datetime.now()
        )

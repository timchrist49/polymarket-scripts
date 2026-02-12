"""
Signal Aggregator

Combines social sentiment and market microstructure with dynamic confidence.
"""

from datetime import datetime
import structlog

from polymarket.models import SocialSentiment, MarketSignals, FundingRateSignal, BTCDominanceSignal, AggregatedSentiment

logger = structlog.get_logger()


class SignalAggregator:
    """Aggregates social, market, funding, and dominance signals with agreement-based confidence."""

    # Weights for final score calculation (total = 1.0)
    MARKET_WEIGHT = 0.40  # Market microstructure (order book, whales, volume, momentum)
    SOCIAL_WEIGHT = 0.20  # Social sentiment (fear/greed, trending, votes)
    FUNDING_WEIGHT = 0.20  # Funding rates (perpetual futures sentiment)
    DOMINANCE_WEIGHT = 0.15  # BTC dominance (capital flow)
    # Note: Order book is already part of MARKET_WEIGHT (5% internally)

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
        market: MarketSignals,
        funding: FundingRateSignal | None = None,
        dominance: BTCDominanceSignal | None = None
    ) -> AggregatedSentiment:
        """
        Aggregate social, market, funding, and dominance signals with dynamic confidence.

        Args:
            social: Social sentiment data
            market: Market microstructure data
            funding: Funding rate signals (optional)
            dominance: BTC dominance signals (optional)

        Returns:
            AggregatedSentiment with final score, confidence, and agreement info.
        """
        # Collect available signals with their weights
        available_signals = []
        total_weight = 0.0

        if market.confidence > 0:
            available_signals.append(("market", market.score, market.confidence, self.MARKET_WEIGHT))
            total_weight += self.MARKET_WEIGHT

        if social.confidence > 0:
            available_signals.append(("social", social.score, social.confidence, self.SOCIAL_WEIGHT))
            total_weight += self.SOCIAL_WEIGHT

        if funding and funding.confidence > 0:
            available_signals.append(("funding", funding.score, funding.confidence, self.FUNDING_WEIGHT))
            total_weight += self.FUNDING_WEIGHT

        if dominance and dominance.confidence > 0:
            available_signals.append(("dominance", dominance.score, dominance.confidence, self.DOMINANCE_WEIGHT))
            total_weight += self.DOMINANCE_WEIGHT

        # Case 1: Multiple signals available
        if len(available_signals) >= 2:
            # Calculate weighted final score (normalize weights to sum to 1.0)
            final_score = sum(
                score * (weight / total_weight)
                for _, score, _, weight in available_signals
            )

            # Base confidence from average of individual confidences
            base_confidence = sum(conf for _, _, conf, _ in available_signals) / len(available_signals)

            # Calculate agreement multiplier across all pairs
            agreement_scores = []
            for i in range(len(available_signals)):
                for j in range(i + 1, len(available_signals)):
                    score1 = available_signals[i][1]
                    score2 = available_signals[j][1]
                    agreement_scores.append(self._calculate_agreement_score(score1, score2))

            # Average agreement multiplier
            agreement_multiplier = sum(agreement_scores) / len(agreement_scores) if agreement_scores else 1.0

            # Apply agreement to confidence
            final_confidence = min(base_confidence * agreement_multiplier, 1.0)

            # Classify signal
            signal_type = self._classify_signal(final_score, final_confidence)

            logger.info(
                "Signals aggregated",
                social_score=f"{social.score:+.2f}" if social.confidence > 0 else "N/A",
                market_score=f"{market.score:+.2f}" if market.confidence > 0 else "N/A",
                funding_score=f"{funding.score:+.2f}" if funding and funding.confidence > 0 else "N/A",
                dominance_score=f"{dominance.score:+.2f}" if dominance and dominance.confidence > 0 else "N/A",
                final_score=f"{final_score:+.2f}",
                final_conf=f"{final_confidence:.2f}",
                agreement=f"{agreement_multiplier:.2f}x",
                signal=signal_type,
                num_signals=len(available_signals)
            )

        # Case 2: Only one signal available
        elif len(available_signals) == 1:
            source_name, final_score, base_confidence, _ = available_signals[0]
            agreement_multiplier = 1.0
            final_confidence = base_confidence * self.MISSING_SOURCE_PENALTY
            signal_type = f"{source_name.upper()}_ONLY"

            logger.warning(f"Using {source_name} signals only (others unavailable)")

        # Case 3: No signals available
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
            funding=funding,
            dominance=dominance,
            final_score=final_score,
            final_confidence=final_confidence,
            agreement_multiplier=agreement_multiplier,
            signal_type=signal_type,
            timestamp=datetime.now()
        )

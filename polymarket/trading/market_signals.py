"""Market signal processing for enhanced trading decisions.

This module provides signal processing capabilities for various market indicators
including funding rates, exchange premiums, and volume confirmation.
"""

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

logger = logging.getLogger(__name__)

# Signal thresholds
FUNDING_RATE_THRESHOLD = 0.0003  # 0.03% - overleveraged threshold
EXCHANGE_PREMIUM_THRESHOLD = 0.005  # 0.5% - significant premium
VOLUME_HIGH_PERCENTILE = 50  # 50th percentile for "high" volume
VOLUME_LOW_PERCENTILE = 25  # 25th percentile for "low" volume


@dataclass
class Signal:
    """Represents a processed market signal."""

    signal_type: str  # "funding_rate", "exchange_premium", "volume"
    direction: Literal["bullish", "bearish", "neutral"]
    confidence: float  # 0.0 to 1.0
    raw_value: float | None
    metadata: dict


@dataclass
class CompositeSignal:
    """Aggregated signal from multiple sources."""

    direction: Literal["bullish", "bearish", "neutral"]
    confidence: float  # 0.0 to 1.0
    contributing_signals: list[Signal]
    weights: dict[str, float]


class MarketSignalProcessor:
    """Processes and aggregates market signals for trading decisions."""

    def __init__(
        self,
        *,
        funding_threshold: float = FUNDING_RATE_THRESHOLD,
        premium_threshold: float = EXCHANGE_PREMIUM_THRESHOLD,
    ) -> None:
        """Initialize the signal processor.

        Args:
            funding_threshold: Funding rate threshold for signal detection (default 0.03%).
            premium_threshold: Exchange premium threshold for signal detection (default 0.5%).
        """
        self._funding_threshold = funding_threshold
        self._premium_threshold = premium_threshold

    def process_funding_rate(self, funding_rate: float | None) -> Signal:
        """Process funding rate into a directional signal.

        Funding rate interpretation:
        - Positive rate: Longs pay shorts (overleveraged longs → bearish)
        - Negative rate: Shorts pay longs (overleveraged shorts → bullish)

        Args:
            funding_rate: Funding rate as decimal (0.0001 = 0.01%).

        Returns:
            Signal with direction and confidence.
        """
        if funding_rate is None:
            logger.debug("Funding rate unavailable, returning neutral signal")
            return Signal(
                signal_type="funding_rate",
                direction="neutral",
                confidence=0.0,
                raw_value=None,
                metadata={"reason": "data_unavailable"},
            )

        # Determine direction and confidence
        if funding_rate > self._funding_threshold:
            # Overleveraged longs → bearish
            confidence = min(funding_rate / self._funding_threshold, 2.0) / 2.0
            direction = "bearish"
            reason = "overleveraged_longs"
        elif funding_rate < -self._funding_threshold:
            # Overleveraged shorts → bullish
            confidence = min(abs(funding_rate) / self._funding_threshold, 2.0) / 2.0
            direction = "bullish"
            reason = "overleveraged_shorts"
        else:
            # Balanced
            confidence = 0.0
            direction = "neutral"
            reason = "balanced"

        logger.info(
            f"Funding rate signal processed: {funding_rate * 100:.4f}% ({direction}) confidence={confidence:.2f}"
        )

        return Signal(
            signal_type="funding_rate",
            direction=direction,
            confidence=confidence,
            raw_value=funding_rate,
            metadata={
                "reason": reason,
                "threshold": self._funding_threshold,
                "rate_pct": funding_rate * 100,
            },
        )

    def process_exchange_premium(
        self,
        prices: dict[str, float] | None,
        base_exchange: str = "binance",
    ) -> Signal:
        """Process exchange price premium into a signal.

        Premium interpretation:
        - Positive premium (e.g., Coinbase > Binance): Retail buying (bullish)
        - Negative premium (e.g., Coinbase < Binance): Retail selling (bearish)

        Args:
            prices: Dictionary mapping exchange names to BTC prices.
            base_exchange: Base exchange for comparison (default "binance").

        Returns:
            Signal with direction and confidence.
        """
        if not prices or base_exchange not in prices:
            logger.debug("Exchange prices unavailable, returning neutral signal")
            return Signal(
                signal_type="exchange_premium",
                direction="neutral",
                confidence=0.0,
                raw_value=None,
                metadata={"reason": "data_unavailable"},
            )

        base_price = prices[base_exchange]
        premiums = {}

        # Calculate premium for each exchange
        for exchange, price in prices.items():
            if exchange != base_exchange:
                premium_pct = ((price - base_price) / base_price) * 100
                premiums[exchange] = premium_pct

        if not premiums:
            return Signal(
                signal_type="exchange_premium",
                direction="neutral",
                confidence=0.0,
                raw_value=None,
                metadata={"reason": "no_comparison_exchanges"},
            )

        # Use Coinbase premium if available (retail sentiment)
        # Otherwise use average premium
        if "coinbase" in premiums:
            primary_premium = premiums["coinbase"] / 100  # Convert to decimal
            primary_exchange = "coinbase"
        else:
            primary_premium = sum(premiums.values()) / len(premiums) / 100
            primary_exchange = "average"

        # Determine direction and confidence
        if primary_premium > self._premium_threshold:
            # Retail buying pressure → bullish
            confidence = min(primary_premium / self._premium_threshold, 2.0) / 2.0
            direction = "bullish"
            reason = "retail_buying"
        elif primary_premium < -self._premium_threshold:
            # Retail selling pressure → bearish
            confidence = min(abs(primary_premium) / self._premium_threshold, 2.0) / 2.0
            direction = "bearish"
            reason = "retail_selling"
        else:
            # Balanced
            confidence = 0.0
            direction = "neutral"
            reason = "balanced"

        logger.info(
            "Exchange premium signal processed",
            primary_exchange=primary_exchange,
            premium_pct=f"{primary_premium * 100:.2f}%",
            direction=direction,
            confidence=f"{confidence:.2f}",
        )

        return Signal(
            signal_type="exchange_premium",
            direction=direction,
            confidence=confidence,
            raw_value=primary_premium,
            metadata={
                "reason": reason,
                "primary_exchange": primary_exchange,
                "premiums": premiums,
                "threshold_pct": self._premium_threshold * 100,
            },
        )

    def process_volume_confirmation(
        self,
        current_volume: Decimal | None,
        recent_volumes: list[Decimal] | None,
        movement_usd: float,
    ) -> Signal:
        """Process volume data into a confirmation signal.

        Volume interpretation:
        - High volume + large movement: Strong signal (confirm trend)
        - Low volume + large movement: Weak signal (potential fake-out)
        - High volume + small movement: Accumulation phase

        Args:
            current_volume: Current period volume.
            recent_volumes: Historical volumes for percentile calculation.
            movement_usd: Recent BTC price movement in USD.

        Returns:
            Signal with direction and confidence.
        """
        if current_volume is None or not recent_volumes:
            logger.debug("Volume data unavailable, returning neutral signal")
            return Signal(
                signal_type="volume",
                direction="neutral",
                confidence=0.0,
                raw_value=None,
                metadata={"reason": "data_unavailable"},
            )

        # Calculate volume percentile
        sorted_volumes = sorted(recent_volumes)
        percentile = (
            sum(1 for v in sorted_volumes if v < current_volume) / len(sorted_volumes)
        ) * 100

        # Determine signal based on volume and movement
        if percentile >= VOLUME_HIGH_PERCENTILE and abs(movement_usd) > 100:
            # High volume + significant movement = strong confirmation
            confidence = 0.8
            direction = "bullish" if movement_usd > 0 else "bearish"
            reason = "high_volume_confirmation"
        elif percentile < VOLUME_LOW_PERCENTILE and abs(movement_usd) > 100:
            # Low volume + significant movement = weak signal (potential fake-out)
            confidence = 0.3
            direction = "neutral"
            reason = "low_volume_weak_signal"
        elif percentile >= VOLUME_HIGH_PERCENTILE and abs(movement_usd) <= 100:
            # High volume + small movement = accumulation
            confidence = 0.5
            direction = "neutral"
            reason = "accumulation_phase"
        else:
            # Normal conditions
            confidence = 0.0
            direction = "neutral"
            reason = "normal"

        logger.info(
            f"Volume confirmation signal processed: {current_volume:,.0f} ({direction}) "
            f"confidence={confidence:.2f} percentile={percentile:.1f}% movement=${movement_usd:.2f}"
        )

        return Signal(
            signal_type="volume",
            direction=direction,
            confidence=confidence,
            raw_value=float(current_volume),
            metadata={
                "reason": reason,
                "percentile": percentile,
                "movement_usd": movement_usd,
            },
        )

    def aggregate_signals(
        self,
        signals: list[Signal],
        weights: dict[str, float] | None = None,
    ) -> CompositeSignal:
        """Aggregate multiple signals into a composite signal.

        Args:
            signals: List of processed signals.
            weights: Optional weights for each signal type. If not provided,
                uses equal weighting for available signals.

        Returns:
            CompositeSignal with weighted direction and confidence.
        """
        if not signals:
            return CompositeSignal(
                direction="neutral",
                confidence=0.0,
                contributing_signals=[],
                weights={},
            )

        # Default equal weights if not provided
        if weights is None:
            weights = {signal.signal_type: 1.0 / len(signals) for signal in signals}

        # Normalize weights for available signals only
        available_types = {s.signal_type for s in signals}
        total_weight = sum(w for t, w in weights.items() if t in available_types)

        if total_weight == 0:
            normalized_weights = {t: 1.0 / len(signals) for t in available_types}
        else:
            normalized_weights = {
                t: w / total_weight for t, w in weights.items() if t in available_types
            }

        # Calculate weighted direction scores
        bullish_score = 0.0
        bearish_score = 0.0
        total_confidence = 0.0

        for signal in signals:
            weight = normalized_weights.get(signal.signal_type, 0.0)
            weighted_confidence = signal.confidence * weight

            if signal.direction == "bullish":
                bullish_score += weighted_confidence
            elif signal.direction == "bearish":
                bearish_score += weighted_confidence

            total_confidence += weighted_confidence

        # Determine final direction
        if bullish_score > bearish_score and bullish_score > 0.1:
            direction = "bullish"
            confidence = bullish_score
        elif bearish_score > bullish_score and bearish_score > 0.1:
            direction = "bearish"
            confidence = bearish_score
        else:
            direction = "neutral"
            confidence = total_confidence

        logger.info(
            "Signals aggregated",
            direction=direction,
            confidence=f"{confidence:.2f}",
            bullish_score=f"{bullish_score:.2f}",
            bearish_score=f"{bearish_score:.2f}",
            num_signals=len(signals),
        )

        return CompositeSignal(
            direction=direction,
            confidence=confidence,
            contributing_signals=signals,
            weights=normalized_weights,
        )

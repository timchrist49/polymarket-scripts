"""Market regime detection for adaptive strategy selection."""
import structlog
from datetime import datetime
from polymarket.models import MarketRegime

logger = structlog.get_logger()


class RegimeDetector:
    """Detect market regime: trending, ranging, or volatile."""

    def detect_regime(
        self,
        price_changes: list[float],  # Last N price changes (%)
        current_price: float,
        high_24h: float,
        low_24h: float
    ) -> MarketRegime:
        """
        Detect current market regime.

        Args:
            price_changes: List of recent % price changes
            current_price: Current BTC price
            high_24h: 24-hour high
            low_24h: 24-hour low

        Returns:
            MarketRegime classification
        """
        # Calculate volatility (24h range as % of price)
        price_range = high_24h - low_24h
        volatility = (price_range / current_price) * 100

        # Calculate trend strength (sum of directional moves)
        if len(price_changes) < 5:
            return MarketRegime(
                regime="UNCLEAR",
                volatility=volatility,
                is_trending=False,
                trend_direction="SIDEWAYS",
                confidence=0.3,
                timestamp=datetime.now()
            )

        positive_moves = sum(1 for change in price_changes if change > 0.5)
        negative_moves = sum(1 for change in price_changes if change < -0.5)
        neutral_moves = len(price_changes) - positive_moves - negative_moves

        # Determine regime
        if volatility > 5.0:
            regime = "VOLATILE"
            is_trending = False
            trend_direction = "SIDEWAYS"
            confidence = 0.4
        elif positive_moves >= len(price_changes) * 0.7:
            regime = "TRENDING"
            is_trending = True
            trend_direction = "UP"
            confidence = 0.8
        elif negative_moves >= len(price_changes) * 0.7:
            regime = "TRENDING"
            is_trending = True
            trend_direction = "DOWN"
            confidence = 0.8
        elif neutral_moves >= len(price_changes) * 0.6:
            regime = "RANGING"
            is_trending = False
            trend_direction = "SIDEWAYS"
            confidence = 0.7
        else:
            regime = "UNCLEAR"
            is_trending = False
            trend_direction = "SIDEWAYS"
            confidence = 0.5

        logger.info(
            "Market regime detected",
            regime=regime,
            volatility=f"{volatility:.2f}%",
            trend_direction=trend_direction,
            confidence=confidence
        )

        return MarketRegime(
            regime=regime,
            volatility=volatility,
            is_trending=is_trending,
            trend_direction=trend_direction,
            confidence=confidence,
            timestamp=datetime.now()
        )

"""Multi-timeframe analysis for trend confirmation."""
import structlog
from datetime import datetime
from polymarket.models import TimeframeAnalysis

logger = structlog.get_logger()


class TimeframeAnalyzer:
    """Analyze multiple timeframes for trend confirmation."""

    async def analyze_timeframes(
        self,
        current_price: float,
        price_4h_ago: float,
        price_24h_ago: float
    ) -> TimeframeAnalysis:
        """
        Analyze daily and 4-hour trends.

        Args:
            current_price: Current BTC price
            price_4h_ago: BTC price 4 hours ago
            price_24h_ago: BTC price 24 hours ago

        Returns:
            TimeframeAnalysis with trend direction and support/resistance
        """
        # Daily trend (24h)
        daily_change_pct = ((current_price - price_24h_ago) / price_24h_ago) * 100
        if daily_change_pct > 2.0:
            daily_trend = "BULLISH"
        elif daily_change_pct < -2.0:
            daily_trend = "BEARISH"
        else:
            daily_trend = "NEUTRAL"

        # 4-hour trend
        four_hour_change_pct = ((current_price - price_4h_ago) / price_4h_ago) * 100
        if four_hour_change_pct > 1.0:
            four_hour_trend = "BULLISH"
        elif four_hour_change_pct < -1.0:
            four_hour_trend = "BEARISH"
        else:
            four_hour_trend = "NEUTRAL"

        # Check alignment
        if daily_trend == four_hour_trend and daily_trend != "NEUTRAL":
            alignment = "ALIGNED"
            confidence = 0.9
        elif daily_trend == "NEUTRAL" or four_hour_trend == "NEUTRAL":
            alignment = "NEUTRAL"
            confidence = 0.5
        else:
            alignment = "CONFLICTING"
            confidence = 0.3

        # Simple support/resistance (last 24h low/high)
        daily_support = min(price_24h_ago, current_price) * 0.98
        daily_resistance = max(price_24h_ago, current_price) * 1.02

        logger.info(
            "Timeframe analysis",
            daily_trend=daily_trend,
            four_hour_trend=four_hour_trend,
            alignment=alignment,
            confidence=confidence
        )

        return TimeframeAnalysis(
            daily_trend=daily_trend,
            four_hour_trend=four_hour_trend,
            alignment=alignment,
            daily_support=daily_support,
            daily_resistance=daily_resistance,
            confidence=confidence,
            timestamp=datetime.now()
        )

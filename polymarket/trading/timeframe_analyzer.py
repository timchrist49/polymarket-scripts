"""Multi-timeframe trend analysis for improved trading decisions."""

from dataclasses import dataclass
from typing import Optional
from decimal import Decimal
import time
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TimeframeTrend:
    """Represents trend direction for a single timeframe."""

    timeframe: str  # "15m", "1h", "4h"
    direction: str  # "UP", "DOWN", "NEUTRAL"
    strength: float  # 0.0 to 1.0 (how strong the trend)
    price_change_pct: float  # Actual percentage change
    price_start: Decimal  # Starting price
    price_end: Decimal  # Ending price


@dataclass
class TimeframeAnalysis:
    """Complete multi-timeframe analysis result."""

    tf_15m: TimeframeTrend
    tf_1h: TimeframeTrend
    tf_4h: TimeframeTrend
    alignment_score: str  # "ALIGNED_BULLISH", "ALIGNED_BEARISH", "MIXED", "CONFLICTING"
    confidence_modifier: float  # +0.15, 0.0, or -0.15

    def __str__(self) -> str:
        return (
            f"15m: {self.tf_15m.direction} ({self.tf_15m.price_change_pct:+.2f}%), "
            f"1H: {self.tf_1h.direction} ({self.tf_1h.price_change_pct:+.2f}%), "
            f"4H: {self.tf_4h.direction} ({self.tf_4h.price_change_pct:+.2f}%) "
            f"| Alignment: {self.alignment_score} | Modifier: {self.confidence_modifier:+.2%}"
        )


class TimeframeAnalyzer:
    """Analyzes BTC price trends across multiple timeframes."""

    def __init__(self, price_buffer):
        """Initialize analyzer with price history buffer.

        Args:
            price_buffer: PriceHistoryBuffer instance for historical lookback
        """
        self.price_buffer = price_buffer
        self.direction_threshold_pct = 0.5  # 0.5% move to be directional

    async def _calculate_trend(
        self,
        timeframe: str,
        lookback_seconds: int
    ) -> Optional[TimeframeTrend]:
        """Calculate trend for a single timeframe.

        Args:
            timeframe: Human-readable name ("15m", "1h", "4h")
            lookback_seconds: How far back to look

        Returns:
            TimeframeTrend if data available, None otherwise
        """
        try:
            # Calculate Unix timestamps
            current_time = int(time.time())
            start_time = current_time - lookback_seconds

            # Get price from lookback_seconds ago
            price_start = await self.price_buffer.get_price_at(start_time)

            # Get current price
            price_end = await self.price_buffer.get_price_at(current_time)

            if not price_start or not price_end:
                logger.warning(
                    "Insufficient price data for timeframe",
                    timeframe=timeframe,
                    lookback_seconds=lookback_seconds
                )
                return None

            # Validate price_start
            if price_start <= 0:
                logger.error(
                    "Invalid start price",
                    timeframe=timeframe,
                    price_start=price_start
                )
                return None

            # Calculate percentage change
            price_change_pct = float(
                ((price_end - price_start) / price_start) * 100
            )

            # Determine direction
            if price_change_pct > self.direction_threshold_pct:
                direction = "UP"
                strength = min(abs(price_change_pct) / 2.0, 1.0)  # Cap at 1.0
            elif price_change_pct < -self.direction_threshold_pct:
                direction = "DOWN"
                strength = min(abs(price_change_pct) / 2.0, 1.0)
            else:
                direction = "NEUTRAL"
                strength = 0.0

            return TimeframeTrend(
                timeframe=timeframe,
                direction=direction,
                strength=strength,
                price_change_pct=price_change_pct,
                price_start=price_start,
                price_end=price_end
            )

        except Exception as e:
            logger.error(
                "Error calculating trend",
                timeframe=timeframe,
                error=str(e)
            )
            return None

    def _calculate_alignment(
        self,
        tf_15m: TimeframeTrend,
        tf_1h: TimeframeTrend,
        tf_4h: TimeframeTrend
    ) -> tuple[str, float]:
        """Calculate alignment score and confidence modifier.

        Returns:
            (alignment_score, confidence_modifier)
        """
        directions = [tf_15m.direction, tf_1h.direction, tf_4h.direction]

        # Count directional votes (ignore NEUTRAL)
        up_count = directions.count("UP")
        down_count = directions.count("DOWN")

        # All aligned in same direction
        if up_count == 3:
            return ("ALIGNED_BULLISH", 0.15)
        elif down_count == 3:
            return ("ALIGNED_BEARISH", 0.15)

        # 2 of 3 agree (mixed signals)
        elif up_count == 2 or down_count == 2:
            return ("MIXED", 0.0)

        # 15m contradicts both longer timeframes (conflicting)
        elif (tf_15m.direction == "UP" and tf_1h.direction == "DOWN" and tf_4h.direction == "DOWN") or \
             (tf_15m.direction == "DOWN" and tf_1h.direction == "UP" and tf_4h.direction == "UP"):
            return ("CONFLICTING", -0.15)

        # Default: Mixed signals
        return ("MIXED", 0.0)

    async def analyze(self) -> Optional[TimeframeAnalysis]:
        """Analyze trends across 15m, 1H, 4H timeframes.

        Returns:
            TimeframeAnalysis if sufficient data, None otherwise
        """
        # Calculate trends for each timeframe
        tf_15m = await self._calculate_trend("15m", 15 * 60)  # 15 minutes
        tf_1h = await self._calculate_trend("1h", 60 * 60)    # 1 hour
        tf_4h = await self._calculate_trend("4h", 4 * 60 * 60)  # 4 hours

        # Require all timeframes to have data
        if not all([tf_15m, tf_1h, tf_4h]):
            logger.warning(
                "Skipping timeframe analysis - insufficient historical data",
                tf_15m=bool(tf_15m),
                tf_1h=bool(tf_1h),
                tf_4h=bool(tf_4h)
            )
            return None

        # Calculate alignment and confidence modifier
        alignment_score, confidence_modifier = self._calculate_alignment(
            tf_15m, tf_1h, tf_4h
        )

        analysis = TimeframeAnalysis(
            tf_15m=tf_15m,
            tf_1h=tf_1h,
            tf_4h=tf_4h,
            alignment_score=alignment_score,
            confidence_modifier=confidence_modifier
        )

        logger.info(
            "Timeframe analysis completed",
            analysis=str(analysis)
        )

        return analysis

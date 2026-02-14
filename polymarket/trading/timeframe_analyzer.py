"""Multi-timeframe trend analysis for improved trading decisions."""

from dataclasses import dataclass
from typing import Optional
from decimal import Decimal
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

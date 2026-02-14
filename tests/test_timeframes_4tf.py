"""Test 4-timeframe analysis."""
import pytest
from decimal import Decimal
from time import time


class MockPriceBuffer:
    """Mock price buffer for testing with Unix timestamps."""

    def __init__(self, prices_by_offset: dict[int, Decimal]):
        """Initialize with prices keyed by seconds ago.

        Args:
            prices_by_offset: {seconds_ago: price}
                e.g., {0: Decimal("98000"), 900: Decimal("97000")}
        """
        self.current_time = int(time())
        self.prices = {}

        # Convert offset-based keys to actual Unix timestamps
        for offset_seconds, price in prices_by_offset.items():
            timestamp = self.current_time - offset_seconds
            self.prices[timestamp] = price

    async def get_price_at(self, timestamp: int) -> Decimal:
        """Return price at timestamp.

        Args:
            timestamp: Unix timestamp

        Returns:
            Price at that timestamp, or None if not found
        """
        return self.prices.get(timestamp)


@pytest.mark.asyncio
async def test_analyzes_4_timeframes():
    """Should analyze 1m, 5m, 15m, 30m timeframes."""
    from polymarket.trading.timeframe_analyzer import TimeframeAnalyzer

    # Setup: All timeframes showing upward movement
    buffer = MockPriceBuffer({
        0: Decimal("70000"),          # Current price
        60: Decimal("69500"),         # 1m ago (-0.7%)
        300: Decimal("69000"),        # 5m ago (-1.4%)
        900: Decimal("68000"),        # 15m ago (-2.9%)
        1800: Decimal("67000"),       # 30m ago (-4.5%)
    })

    analyzer = TimeframeAnalyzer(buffer)
    result = await analyzer.analyze()

    # Should have 4 timeframes
    assert result is not None
    assert hasattr(result, 'tf_1m')
    assert hasattr(result, 'tf_5m')
    assert hasattr(result, 'tf_15m')
    assert hasattr(result, 'tf_30m')
    assert result.tf_1m.timeframe == "1m"
    assert result.tf_5m.timeframe == "5m"
    assert result.tf_15m.timeframe == "15m"
    assert result.tf_30m.timeframe == "30m"


def test_alignment_all_4_bullish():
    """All 4 timeframes bullish should return ALIGNED_BULLISH."""
    from polymarket.trading.timeframe_analyzer import TimeframeAnalyzer, TimeframeTrend
    from decimal import Decimal

    analyzer = TimeframeAnalyzer(None)

    # Create 4 UP trends
    tf_1m = TimeframeTrend("1m", "UP", 0.8, 0.5, Decimal("70000"), Decimal("70350"))
    tf_5m = TimeframeTrend("5m", "UP", 0.9, 1.2, Decimal("69500"), Decimal("70350"))
    tf_15m = TimeframeTrend("15m", "UP", 1.0, 2.0, Decimal("68700"), Decimal("70350"))
    tf_30m = TimeframeTrend("30m", "UP", 0.9, 2.5, Decimal("68100"), Decimal("70350"))

    alignment, modifier = analyzer._calculate_alignment_4tf(tf_1m, tf_5m, tf_15m, tf_30m)

    assert alignment == "ALIGNED_BULLISH"
    assert modifier == 0.20


def test_alignment_3_of_4_bullish():
    """3 of 4 bullish should return STRONG_BULLISH."""
    from polymarket.trading.timeframe_analyzer import TimeframeAnalyzer, TimeframeTrend
    from decimal import Decimal

    analyzer = TimeframeAnalyzer(None)

    # 3 UP, 1 NEUTRAL
    tf_1m = TimeframeTrend("1m", "UP", 0.8, 0.5, Decimal("70000"), Decimal("70350"))
    tf_5m = TimeframeTrend("5m", "UP", 0.9, 1.2, Decimal("69500"), Decimal("70350"))
    tf_15m = TimeframeTrend("15m", "UP", 1.0, 2.0, Decimal("68700"), Decimal("70350"))
    tf_30m = TimeframeTrend("30m", "NEUTRAL", 0.0, 0.2, Decimal("70300"), Decimal("70350"))

    alignment, modifier = analyzer._calculate_alignment_4tf(tf_1m, tf_5m, tf_15m, tf_30m)

    assert alignment == "STRONG_BULLISH"
    assert modifier == 0.15

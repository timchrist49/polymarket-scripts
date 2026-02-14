"""Unit tests for TimeframeAnalyzer."""

import pytest
from decimal import Decimal
from time import time
from polymarket.trading.timeframe_analyzer import (
    TimeframeAnalyzer,
    TimeframeTrend,
    TimeframeAnalysis
)


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
async def test_aligned_bullish_trend():
    """Test all timeframes aligned bullish."""
    # Setup: All timeframes showing upward movement (> 0.5% threshold)
    buffer = MockPriceBuffer({
        0: Decimal("98000"),          # Current price
        60: Decimal("97450"),         # 1m ago (-0.56%)
        5 * 60: Decimal("97000"),     # 5m ago (-1.02%)
        15 * 60: Decimal("96000"),    # 15m ago (-2.04%)
        30 * 60: Decimal("94000"),    # 30m ago (-4.08%)
    })

    analyzer = TimeframeAnalyzer(buffer)
    result = await analyzer.analyze()

    assert result is not None
    assert result.tf_1m.direction == "UP"
    assert result.tf_5m.direction == "UP"
    assert result.tf_15m.direction == "UP"
    assert result.tf_30m.direction == "UP"
    assert result.alignment_score == "ALIGNED_BULLISH"
    assert result.confidence_modifier == 0.20  # All 4 aligned = +0.20


@pytest.mark.asyncio
async def test_aligned_bearish_trend():
    """Test all timeframes aligned bearish."""
    buffer = MockPriceBuffer({
        0: Decimal("94000"),          # Current price
        60: Decimal("94550"),         # 1m ago (+0.58%)
        5 * 60: Decimal("95000"),     # 5m ago (+1.06%)
        15 * 60: Decimal("96000"),    # 15m ago (+2.13%)
        30 * 60: Decimal("98000"),    # 30m ago (+4.26%)
    })

    analyzer = TimeframeAnalyzer(buffer)
    result = await analyzer.analyze()

    assert result is not None
    assert result.tf_1m.direction == "DOWN"
    assert result.tf_5m.direction == "DOWN"
    assert result.tf_15m.direction == "DOWN"
    assert result.tf_30m.direction == "DOWN"
    assert result.alignment_score == "ALIGNED_BEARISH"
    assert result.confidence_modifier == 0.20  # All 4 aligned = +0.20


@pytest.mark.asyncio
async def test_mixed_signals():
    """Test mixed timeframe signals (2 UP, 2 DOWN)."""
    buffer = MockPriceBuffer({
        0: Decimal("97000"),          # Current
        60: Decimal("96450"),         # 1m ago: UP (+0.57%)
        5 * 60: Decimal("97550"),     # 5m ago: DOWN (-0.56%)
        15 * 60: Decimal("96000"),    # 15m ago: UP (+1.04%)
        30 * 60: Decimal("98000"),    # 30m ago: DOWN (-1.02%)
    })

    analyzer = TimeframeAnalyzer(buffer)
    result = await analyzer.analyze()

    assert result is not None
    assert result.alignment_score == "MIXED"
    assert result.confidence_modifier == 0.0


@pytest.mark.asyncio
async def test_strong_bullish():
    """Test 3 of 4 timeframes bullish (STRONG_BULLISH)."""
    buffer = MockPriceBuffer({
        0: Decimal("97000"),           # Current
        60: Decimal("96450"),          # 1m ago: UP (+0.57%)
        5 * 60: Decimal("96500"),      # 5m ago: UP (+0.52%)
        15 * 60: Decimal("96000"),     # 15m ago: UP (+1.04%)
        30 * 60: Decimal("97550"),     # 30m ago: DOWN (-0.56%) (1 dissenter)
    })

    analyzer = TimeframeAnalyzer(buffer)
    result = await analyzer.analyze()

    assert result is not None
    assert result.tf_1m.direction == "UP"
    assert result.tf_5m.direction == "UP"
    assert result.tf_15m.direction == "UP"
    assert result.tf_30m.direction == "DOWN"
    assert result.alignment_score == "STRONG_BULLISH"  # 3 of 4 UP
    assert result.confidence_modifier == 0.15


@pytest.mark.asyncio
async def test_insufficient_data():
    """Test graceful degradation with missing data."""
    buffer = MockPriceBuffer({
        0: Decimal("97000"),
        60: Decimal("96900"),
        # Missing 5m, 15m, and 30m data
    })

    analyzer = TimeframeAnalyzer(buffer)
    result = await analyzer.analyze()

    assert result is None  # Should return None when data missing

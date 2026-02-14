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
    # Setup: All timeframes showing upward movement
    buffer = MockPriceBuffer({
        0: Decimal("98000"),          # Current price
        15 * 60: Decimal("97000"),    # 15m ago (-1%)
        60 * 60: Decimal("96000"),    # 1H ago (-2%)
        4 * 60 * 60: Decimal("94000")  # 4H ago (-4%)
    })

    analyzer = TimeframeAnalyzer(buffer)
    result = await analyzer.analyze()

    assert result is not None
    assert result.tf_15m.direction == "UP"
    assert result.tf_1h.direction == "UP"
    assert result.tf_4h.direction == "UP"
    assert result.alignment_score == "ALIGNED_BULLISH"
    assert result.confidence_modifier == 0.15


@pytest.mark.asyncio
async def test_aligned_bearish_trend():
    """Test all timeframes aligned bearish."""
    buffer = MockPriceBuffer({
        0: Decimal("94000"),          # Current price
        15 * 60: Decimal("95000"),    # 15m ago (+1%)
        60 * 60: Decimal("96000"),    # 1H ago (+2%)
        4 * 60 * 60: Decimal("98000")  # 4H ago (+4%)
    })

    analyzer = TimeframeAnalyzer(buffer)
    result = await analyzer.analyze()

    assert result is not None
    assert result.tf_15m.direction == "DOWN"
    assert result.tf_1h.direction == "DOWN"
    assert result.tf_4h.direction == "DOWN"
    assert result.alignment_score == "ALIGNED_BEARISH"
    assert result.confidence_modifier == 0.15


@pytest.mark.asyncio
async def test_mixed_signals():
    """Test mixed timeframe signals."""
    buffer = MockPriceBuffer({
        0: Decimal("97000"),          # Current
        15 * 60: Decimal("96000"),    # 15m ago: UP
        60 * 60: Decimal("98000"),    # 1H ago: DOWN
        4 * 60 * 60: Decimal("96500")  # 4H ago: UP
    })

    analyzer = TimeframeAnalyzer(buffer)
    result = await analyzer.analyze()

    assert result is not None
    assert result.alignment_score == "MIXED"
    assert result.confidence_modifier == 0.0


@pytest.mark.asyncio
async def test_conflicting_signals():
    """Test 15m contradicting longer timeframes (short-term bounce in downtrend).

    Note: This scenario (1 UP, 2 DOWN) returns MIXED, not CONFLICTING.
    The CONFLICTING case in the code is actually unreachable because
    it's checked after the "2 of 3 agree" condition.
    """
    buffer = MockPriceBuffer({
        0: Decimal("96000"),           # Current
        15 * 60: Decimal("95000"),     # 15m ago: UP (+1.05%) - short bounce
        60 * 60: Decimal("97000"),     # 1H ago: DOWN (-1.03%) - longer down
        4 * 60 * 60: Decimal("98000")  # 4H ago: DOWN (-2.04%) - longer down
    })

    analyzer = TimeframeAnalyzer(buffer)
    result = await analyzer.analyze()

    assert result is not None
    assert result.tf_15m.direction == "UP"    # Short-term bounce
    assert result.tf_1h.direction == "DOWN"   # Longer-term downtrend
    assert result.tf_4h.direction == "DOWN"   # Longer-term downtrend
    assert result.alignment_score == "MIXED"  # 2 agree (DOWN), so MIXED
    assert result.confidence_modifier == 0.0


@pytest.mark.asyncio
async def test_insufficient_data():
    """Test graceful degradation with missing data."""
    buffer = MockPriceBuffer({
        0: Decimal("97000"),
        15 * 60: Decimal("96000"),
        # Missing 1H and 4H data
    })

    analyzer = TimeframeAnalyzer(buffer)
    result = await analyzer.analyze()

    assert result is None  # Should return None when data missing

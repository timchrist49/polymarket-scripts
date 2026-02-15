"""Tests for price history buffer tolerance feature."""

import pytest
from decimal import Decimal
from polymarket.trading.price_history_buffer import PriceHistoryBuffer


@pytest.mark.asyncio
async def test_get_price_at_with_tolerance():
    """Test buffer finds price within tolerance window."""
    buffer = PriceHistoryBuffer(retention_hours=1)

    # Add price at 18:00:00
    exact_time = 1771178400
    await buffer.append(
        timestamp=exact_time,
        price=Decimal("68598.02"),
        source="chainlink"
    )

    # Query at 18:00:15 (15 seconds later) with ±30s tolerance
    result = await buffer.get_price_at(exact_time + 15, tolerance=30)

    assert result is not None
    assert result.price == Decimal("68598.02")
    assert result.source == "chainlink"


@pytest.mark.asyncio
async def test_get_price_at_outside_tolerance():
    """Test buffer returns None outside tolerance window."""
    buffer = PriceHistoryBuffer(retention_hours=1)

    # Add price at 18:00:00
    exact_time = 1771178400
    await buffer.append(
        timestamp=exact_time,
        price=Decimal("68598.02"),
        source="chainlink"
    )

    # Query at 18:01:00 (60 seconds later) with ±30s tolerance
    result = await buffer.get_price_at(exact_time + 60, tolerance=30)

    assert result is None


@pytest.mark.asyncio
async def test_get_price_at_finds_closest_within_tolerance():
    """Test buffer finds closest price when multiple entries exist."""
    buffer = PriceHistoryBuffer(retention_hours=1)

    # Add prices at 18:00:00, 18:00:30, 18:01:00
    base_time = 1771178400
    await buffer.append(
        timestamp=base_time,
        price=Decimal("68500.00"),
        source="chainlink"
    )
    await buffer.append(
        timestamp=base_time + 30,
        price=Decimal("68550.00"),
        source="chainlink"
    )
    await buffer.append(
        timestamp=base_time + 60,
        price=Decimal("68600.00"),
        source="chainlink"
    )

    # Query at 18:00:25 with ±30s tolerance
    # Should find 18:00:30 entry (5s away) not 18:00:00 (25s away)
    result = await buffer.get_price_at(base_time + 25, tolerance=30)

    assert result is not None
    assert result.price == Decimal("68550.00")
    assert result.source == "chainlink"


@pytest.mark.asyncio
async def test_get_price_at_exact_match_preferred():
    """Test buffer prefers exact match over tolerance match."""
    buffer = PriceHistoryBuffer(retention_hours=1)

    # Add prices at 18:00:00, 18:00:30
    base_time = 1771178400
    await buffer.append(
        timestamp=base_time,
        price=Decimal("68500.00"),
        source="chainlink"
    )
    await buffer.append(
        timestamp=base_time + 30,
        price=Decimal("68550.00"),
        source="chainlink"
    )

    # Query exact timestamp with tolerance
    result = await buffer.get_price_at(base_time + 30, tolerance=30)

    assert result is not None
    assert result.price == Decimal("68550.00")
    assert result.source == "chainlink"


@pytest.mark.asyncio
async def test_get_price_at_tolerance_boundary():
    """Test buffer respects exact tolerance boundary."""
    buffer = PriceHistoryBuffer(retention_hours=1)

    # Add price at 18:00:00
    exact_time = 1771178400
    await buffer.append(
        timestamp=exact_time,
        price=Decimal("68598.02"),
        source="chainlink"
    )

    # Query exactly 30 seconds away (should match with tolerance=30)
    result = await buffer.get_price_at(exact_time + 30, tolerance=30)
    assert result is not None
    assert result.price == Decimal("68598.02")
    assert result.source == "chainlink"

    # Query 31 seconds away (should NOT match with tolerance=30)
    result = await buffer.get_price_at(exact_time + 31, tolerance=30)
    assert result is None


@pytest.mark.asyncio
async def test_get_price_at_preserves_source_attribution():
    """Test that source attribution is preserved in returned BTCPriceData."""
    buffer = PriceHistoryBuffer(retention_hours=1)

    # Add prices from different sources
    base_time = 1771178400
    await buffer.append(
        timestamp=base_time,
        price=Decimal("68500.00"),
        source="chainlink"
    )
    await buffer.append(
        timestamp=base_time + 60,
        price=Decimal("68600.00"),
        source="binance"
    )

    # Verify chainlink source preserved
    result = await buffer.get_price_at(base_time, tolerance=5)
    assert result is not None
    assert result.source == "chainlink"

    # Verify binance source preserved
    result = await buffer.get_price_at(base_time + 60, tolerance=5)
    assert result is not None
    assert result.source == "binance"

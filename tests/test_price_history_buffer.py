"""Tests for price history buffer."""

import pytest
import asyncio
from decimal import Decimal
from datetime import datetime
from polymarket.trading.price_history_buffer import PriceHistoryBuffer


def test_buffer_initializes_empty():
    """Buffer should start empty."""
    buffer = PriceHistoryBuffer(retention_hours=24)

    assert buffer.size() == 0
    assert buffer.is_empty() == True


def test_buffer_has_correct_max_length():
    """Buffer should have maxlen = retention_hours * 120 (2x safety)."""
    buffer = PriceHistoryBuffer(retention_hours=24)

    # 24 hours × 120 entries per hour (2x for real-time density)
    assert buffer.max_size() == 2880


@pytest.mark.asyncio
async def test_append_adds_price_to_buffer():
    """Appending a price should add it to buffer."""
    buffer = PriceHistoryBuffer(retention_hours=24)

    await buffer.append(
        timestamp=1770875100,
        price=Decimal("67018.35"),
        source="polymarket"
    )

    assert buffer.size() == 1
    assert buffer.is_empty() == False


@pytest.mark.asyncio
async def test_append_rejects_out_of_order_timestamps():
    """Buffer should reject timestamps older than last entry."""
    buffer = PriceHistoryBuffer(retention_hours=24)

    await buffer.append(1770875100, Decimal("67018.35"), "polymarket")
    await buffer.append(1770875160, Decimal("67025.10"), "polymarket")

    # Try to append older timestamp (should be rejected)
    await buffer.append(1770875050, Decimal("67000.00"), "polymarket")

    # Should still have only 2 entries
    assert buffer.size() == 2


@pytest.mark.asyncio
async def test_append_marks_buffer_dirty():
    """Appending should mark buffer as dirty for saving."""
    buffer = PriceHistoryBuffer(retention_hours=24)

    assert buffer.is_dirty() == False

    await buffer.append(1770875100, Decimal("67018.35"), "polymarket")

    assert buffer.is_dirty() == True

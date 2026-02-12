"""Tests for price history buffer."""

import pytest
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

"""Tests for MarketValidator."""

import pytest
from polymarket.trading.market_validator import MarketValidator


def test_parse_timestamp_from_slug():
    """Test parsing Unix timestamp from market slug."""
    validator = MarketValidator()

    # Valid slug format
    slug = "btc-updown-15m-1771270200"
    timestamp = validator.parse_timestamp(slug)
    assert timestamp == 1771270200

    # Different timestamp
    slug2 = "btc-updown-15m-1234567890"
    timestamp2 = validator.parse_timestamp(slug2)
    assert timestamp2 == 1234567890

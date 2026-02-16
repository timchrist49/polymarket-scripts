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


def test_parse_timestamp_invalid_formats():
    """Test parsing invalid slug formats raises ValueError."""
    validator = MarketValidator()

    # Non-numeric timestamp
    with pytest.raises(ValueError, match="Invalid market slug format"):
        validator.parse_timestamp("btc-updown-15m-abc")

    # Empty string
    with pytest.raises(ValueError, match="Invalid market slug format"):
        validator.parse_timestamp("")

    # No hyphens
    with pytest.raises(ValueError, match="Invalid market slug format"):
        validator.parse_timestamp("invalidslug")

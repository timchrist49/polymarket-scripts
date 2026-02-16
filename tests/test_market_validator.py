"""Tests for MarketValidator."""

from polymarket.trading.market_validator import MarketValidator


def test_parse_market_timestamp_valid():
    """Test parsing Unix timestamp from valid market slugs."""
    # Standard format
    timestamp = MarketValidator.parse_market_timestamp("btc-updown-15m-1771270200")
    assert timestamp == 1771270200

    # Different timestamp
    timestamp2 = MarketValidator.parse_market_timestamp("btc-updown-15m-1234567890")
    assert timestamp2 == 1234567890


def test_parse_market_timestamp_invalid_format():
    """Test parsing invalid slug formats returns None."""
    # Empty string
    assert MarketValidator.parse_market_timestamp("") is None

    # Wrong asset
    assert MarketValidator.parse_market_timestamp("eth-updown-15m-1234567890") is None

    # Wrong market type
    assert MarketValidator.parse_market_timestamp("btc-will-15m-1234567890") is None

    # Missing timestamp
    assert MarketValidator.parse_market_timestamp("btc-updown-15m") is None

    # Non-numeric timestamp
    assert MarketValidator.parse_market_timestamp("btc-updown-15m-abc") is None

    # No hyphens
    assert MarketValidator.parse_market_timestamp("invalidslug") is None

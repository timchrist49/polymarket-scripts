"""Tests for MarketValidator."""

from datetime import datetime
from freezegun import freeze_time
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


@freeze_time("2026-02-16 14:30:00")  # Frozen at 14:30:00
def test_is_market_active_within_window():
    """Test market is active when timestamp matches current 15-min window."""
    # Market expires at 14:30:00 (frozen time)
    market_time = datetime(2026, 2, 16, 14, 30, 0)
    slug = f"btc-updown-15m-{int(market_time.timestamp())}"

    # Should be active (exact match)
    assert MarketValidator.is_market_active(slug, tolerance_minutes=2) is True

    # Should be active (within 2 min tolerance)
    frozen_at = datetime(2026, 2, 16, 14, 31, 30)  # 1.5 min after market time
    with freeze_time(frozen_at):
        assert MarketValidator.is_market_active(slug, tolerance_minutes=2) is True


@freeze_time("2026-02-16 14:30:00")
def test_is_market_active_outside_window():
    """Test market is inactive when timestamp is outside tolerance."""
    # Market expired at 14:15:00 (15 minutes ago)
    old_market_time = datetime(2026, 2, 16, 14, 15, 0)
    slug = f"btc-updown-15m-{int(old_market_time.timestamp())}"

    # Should be inactive (expired)
    assert MarketValidator.is_market_active(slug, tolerance_minutes=2) is False

    # Future market (14:45:00, 15 minutes from now)
    future_market_time = datetime(2026, 2, 16, 14, 45, 0)
    future_slug = f"btc-updown-15m-{int(future_market_time.timestamp())}"

    # Should be inactive (not yet active)
    assert MarketValidator.is_market_active(future_slug, tolerance_minutes=2) is False


@freeze_time("2026-02-16 14:30:00")
def test_is_market_active_invalid_slug():
    """Test invalid slugs return False."""
    assert MarketValidator.is_market_active("", tolerance_minutes=2) is False
    assert MarketValidator.is_market_active("invalid-slug", tolerance_minutes=2) is False

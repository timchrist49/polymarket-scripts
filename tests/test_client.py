"""Tests for client module."""

import pytest
from datetime import datetime, timezone
from polymarket.config import Settings, reset_settings
from polymarket.auth import reset_auth_manager
from polymarket.client import PolymarketClient, floor_to_15min_interval, generate_btc_15min_slug
from polymarket.exceptions import MarketDiscoveryError, ValidationError


def test_client_read_only_mode():
    """Test client can be initialized in read_only mode."""
    reset_settings()
    reset_auth_manager()
    import os
    os.environ["POLYMARKET_MODE"] = "read_only"
    client = PolymarketClient()
    assert client is not None
    assert client.mode == "read_only"


def test_floor_to_15min_interval():
    """Test 15-minute interval floor calculation."""

    # 10:09 should floor to 10:00
    dt = datetime(2025, 2, 9, 10, 9, 30, tzinfo=timezone.utc)
    floored = floor_to_15min_interval(dt)
    assert floored.hour == 10
    assert floored.minute == 0
    assert floored.second == 0

    # 10:00 should stay 10:00
    dt = datetime(2025, 2, 9, 10, 0, 0, tzinfo=timezone.utc)
    floored = floor_to_15min_interval(dt)
    assert floored.hour == 10
    assert floored.minute == 0

    # 10:15 should floor to 10:15
    dt = datetime(2025, 2, 9, 10, 15, 0, tzinfo=timezone.utc)
    floored = floor_to_15min_interval(dt)
    assert floored.hour == 10
    assert floored.minute == 15

    # 10:23 should floor to 10:15
    dt = datetime(2025, 2, 9, 10, 23, 45, tzinfo=timezone.utc)
    floored = floor_to_15min_interval(dt)
    assert floored.hour == 10
    assert floored.minute == 15


def test_generate_btc_slug():
    """Test BTC slug generation."""

    # Known time: 2025-02-09 10:00:00 UTC
    dt = datetime(2025, 2, 9, 10, 0, 0, tzinfo=timezone.utc)
    slug = generate_btc_15min_slug(dt)
    assert slug.startswith("btc-updown-15m-")
    # Extract timestamp
    timestamp_str = slug.split("-")[-1]
    timestamp = int(timestamp_str)
    # Should be close to the expected timestamp
    assert timestamp > 0


def test_generate_btc_slug_default_time():
    """Test BTC slug generation with default time (now)."""
    slug = generate_btc_15min_slug()
    assert slug.startswith("btc-updown-15m-")
    timestamp_str = slug.split("-")[-1]
    timestamp = int(timestamp_str)
    # Should be a recent timestamp
    assert timestamp > 1700000000  # Some time in 2023+


def test_client_trading_mode():
    """Test client can be initialized in trading mode."""
    reset_settings()
    reset_auth_manager()
    import os
    os.environ["POLYMARKET_MODE"] = "trading"
    os.environ["POLYMARKET_PRIVATE_KEY"] = "0x" + "a" * 64
    client = PolymarketClient()
    assert client is not None
    assert client.mode == "trading"


def test_client_clob_requires_trading():
    """Test that CLOB operations require trading mode."""
    reset_settings()
    reset_auth_manager()
    import os
    os.environ["POLYMARKET_MODE"] = "read_only"
    client = PolymarketClient()

    # _get_clob_client should fail in read_only mode
    with pytest.raises(ValidationError, match="TRADING mode"):
        client._get_clob_client()


def test_floor_boundary_cases():
    """Test floor_to_15min_interval at boundary cases."""

    # Test all 15-minute boundaries in an hour
    for minute in [0, 14, 15, 29, 30, 44, 45, 59]:
        dt = datetime(2025, 2, 9, 10, minute, 30, tzinfo=timezone.utc)
        floored = floor_to_15min_interval(dt)

        expected_minute = (minute // 15) * 15
        assert floored.minute == expected_minute
        assert floored.second == 0
        assert floored.microsecond == 0


def test_generate_slug_consistency():
    """Test that slug generation is consistent."""
    dt = datetime(2025, 2, 9, 10, 15, 0, tzinfo=timezone.utc)
    slug1 = generate_btc_15min_slug(dt)
    slug2 = generate_btc_15min_slug(dt)
    assert slug1 == slug2

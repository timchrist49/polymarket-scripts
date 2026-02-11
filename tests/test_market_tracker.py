"""Tests for market timing and price-to-beat tracker."""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from polymarket.trading.market_tracker import MarketTracker
from polymarket.config import Settings


def test_parse_market_slug():
    """Test parsing epoch timestamp from market slug."""
    settings = Settings()
    tracker = MarketTracker(settings)

    slug = "btc-updown-15m-1739203200"  # Example: 2026-02-11 00:00:00 UTC
    start_time = tracker.parse_market_start(slug)

    assert start_time == datetime.fromtimestamp(1739203200, tz=timezone.utc)


def test_calculate_time_remaining():
    """Test calculating time remaining in market."""
    settings = Settings()
    tracker = MarketTracker(settings)

    slug = "btc-updown-15m-1739203200"
    start_time = tracker.parse_market_start(slug)

    # Mock current time as 5 minutes after start
    current_time = datetime.fromtimestamp(1739203200 + 300, tz=timezone.utc)  # +5 min
    remaining = tracker.calculate_time_remaining(start_time, current_time)

    assert remaining == 600  # 10 minutes remaining (15 - 5)


def test_track_price_to_beat():
    """Test tracking starting price for market."""
    settings = Settings()
    tracker = MarketTracker(settings)

    slug = "btc-updown-15m-1739203200"
    starting_price = Decimal("95000.50")

    # Set starting price
    tracker.set_price_to_beat(slug, starting_price)

    # Retrieve it
    price = tracker.get_price_to_beat(slug)
    assert price == starting_price


@pytest.mark.asyncio
async def test_is_end_of_market():
    """Test detecting last 3 minutes of market."""
    settings = Settings()
    tracker = MarketTracker(settings)

    slug = "btc-updown-15m-1739203200"
    start_time = tracker.parse_market_start(slug)

    # 13 minutes elapsed (2 remaining) = END OF MARKET
    current_time = datetime.fromtimestamp(1739203200 + 780, tz=timezone.utc)  # +13 min
    is_end = tracker.is_end_of_market(start_time, current_time)
    assert is_end is True

    # 10 minutes elapsed (5 remaining) = NOT end
    current_time = datetime.fromtimestamp(1739203200 + 600, tz=timezone.utc)  # +10 min
    is_end = tracker.is_end_of_market(start_time, current_time)
    assert is_end is False

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from polymarket.trading.price_cache import CandleCache
from polymarket.models import PricePoint


def test_get_ttl_old_candle():
    """Old candles (>60 min) get 1 hour TTL."""
    cache = CandleCache()
    now = datetime.now()
    old_timestamp = now - timedelta(minutes=120)

    ttl = cache.get_ttl(old_timestamp, current_time=now)

    assert ttl == 3600  # 1 hour


def test_get_ttl_recent_candle():
    """Recent closed candles (5-60 min) get 5 min TTL."""
    cache = CandleCache()
    now = datetime.now()
    recent_timestamp = now - timedelta(minutes=30)

    ttl = cache.get_ttl(recent_timestamp, current_time=now)

    assert ttl == 300  # 5 minutes


def test_get_ttl_current_candle():
    """Current candles (<5 min) get 1 min TTL."""
    cache = CandleCache()
    now = datetime.now()
    current_timestamp = now - timedelta(minutes=2)

    ttl = cache.get_ttl(current_timestamp, current_time=now)

    assert ttl == 60  # 1 minute


def test_get_ttl_boundary_60_minutes():
    """Candle exactly 60 minutes old gets 5 min TTL (recent, not old)."""
    cache = CandleCache()
    now = datetime.now()
    exactly_60_min = now - timedelta(minutes=60)

    ttl = cache.get_ttl(exactly_60_min, current_time=now)

    assert ttl == 300  # 5 minutes (not 1 hour, because age is exactly 60, not > 60)


def test_get_ttl_boundary_5_minutes():
    """Candle exactly 5 minutes old gets 1 min TTL (current, not recent)."""
    cache = CandleCache()
    now = datetime.now()
    exactly_5_min = now - timedelta(minutes=5)

    ttl = cache.get_ttl(exactly_5_min, current_time=now)

    assert ttl == 60  # 1 minute (not 5 minutes, because age is exactly 5, not > 5)


def test_get_ttl_future_timestamp():
    """Future candle (negative age) gets current candle TTL."""
    cache = CandleCache()
    now = datetime.now()
    future_timestamp = now + timedelta(minutes=10)

    ttl = cache.get_ttl(future_timestamp, current_time=now)

    assert ttl == 60  # 1 minute (treat as current candle)

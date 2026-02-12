import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from polymarket.trading.price_cache import CandleCache
from polymarket.models import PricePoint


def test_get_ttl_old_candle():
    """Old candles (>60 min) get 1 hour TTL."""
    cache = CandleCache()
    old_timestamp = datetime.now() - timedelta(minutes=120)

    ttl = cache.get_ttl(old_timestamp)

    assert ttl == 3600  # 1 hour


def test_get_ttl_recent_candle():
    """Recent closed candles (5-60 min) get 5 min TTL."""
    cache = CandleCache()
    recent_timestamp = datetime.now() - timedelta(minutes=30)

    ttl = cache.get_ttl(recent_timestamp)

    assert ttl == 300  # 5 minutes


def test_get_ttl_current_candle():
    """Current candles (<5 min) get 1 min TTL."""
    cache = CandleCache()
    current_timestamp = datetime.now() - timedelta(minutes=2)

    ttl = cache.get_ttl(current_timestamp)

    assert ttl == 60  # 1 minute

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from polymarket.trading.stale_policy import StaleDataPolicy
from polymarket.models import BTCPriceData


def test_record_success_resets_failures():
    """Recording success resets consecutive failure count."""
    policy = StaleDataPolicy()
    policy._consecutive_failures = 5

    data = BTCPriceData(
        price=Decimal("67000"),
        timestamp=datetime.now(),
        source="test",
        volume_24h=Decimal("1000")
    )

    policy.record_success(data)

    assert policy._consecutive_failures == 0


def test_record_failure_increments_count():
    """Recording failure increments consecutive count."""
    policy = StaleDataPolicy()

    policy.record_failure()
    policy.record_failure()

    assert policy._consecutive_failures == 2


def test_can_use_stale_cache_fresh():
    """Fresh cache (< 10 min) is usable."""
    policy = StaleDataPolicy()
    data = BTCPriceData(
        price=Decimal("67000"),
        timestamp=datetime.now(),
        source="test",
        volume_24h=Decimal("1000")
    )
    policy.record_success(data)

    assert policy.can_use_stale_cache() is True


def test_can_use_stale_cache_too_old():
    """Old cache (> 10 min) is not usable."""
    policy = StaleDataPolicy()
    data = BTCPriceData(
        price=Decimal("67000"),
        timestamp=datetime.now(),
        source="test",
        volume_24h=Decimal("1000")
    )
    # Manually set old cache time
    policy._stale_cache = (data, datetime.now() - timedelta(minutes=11))

    assert policy.can_use_stale_cache() is False


def test_can_use_stale_cache_no_cache():
    """No cache means not usable."""
    policy = StaleDataPolicy()

    assert policy.can_use_stale_cache() is False

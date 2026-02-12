import pytest
import asyncio
from unittest.mock import AsyncMock

from polymarket.trading.parallel_fetch import fetch_with_fallbacks


@pytest.mark.asyncio
async def test_primary_succeeds():
    """Primary source succeeds, fallbacks not tried."""
    primary = AsyncMock(return_value={"price": 67000})
    fallback1 = AsyncMock(return_value={"price": 67100})
    fallback2 = AsyncMock(return_value={"price": 67200})

    result = await fetch_with_fallbacks(
        primary,
        [("Fallback1", fallback1), ("Fallback2", fallback2)]
    )

    assert result == {"price": 67000}
    assert primary.call_count >= 1  # May be retried
    assert fallback1.call_count == 0
    assert fallback2.call_count == 0


@pytest.mark.asyncio
async def test_primary_fails_fallback_succeeds():
    """Primary fails, first fallback succeeds."""
    primary = AsyncMock(return_value=None)  # Simulates fetch_with_retry returning None
    fallback1 = AsyncMock(return_value={"price": 67100})
    fallback2 = AsyncMock(return_value={"price": 67200})

    result = await fetch_with_fallbacks(
        primary,
        [("Fallback1", fallback1), ("Fallback2", fallback2)]
    )

    assert result == {"price": 67100}
    assert fallback1.call_count >= 1


@pytest.mark.asyncio
async def test_all_sources_fail():
    """All sources fail."""
    primary = AsyncMock(return_value=None)
    fallback1 = AsyncMock(return_value=None)
    fallback2 = AsyncMock(return_value=None)

    result = await fetch_with_fallbacks(
        primary,
        [("Fallback1", fallback1), ("Fallback2", fallback2)]
    )

    assert result is None


@pytest.mark.asyncio
async def test_fastest_fallback_wins():
    """When racing fallbacks, fastest wins."""
    primary = AsyncMock(return_value=None)

    async def slow_fallback():
        await asyncio.sleep(1)
        return {"price": 67100}

    async def fast_fallback():
        await asyncio.sleep(0.1)
        return {"price": 67200}

    # Note: We're testing the racing behavior here
    # The implementation should cancel slower tasks
    result = await fetch_with_fallbacks(
        primary,
        [("Slow", slow_fallback), ("Fast", fast_fallback)]
    )

    assert result == {"price": 67200}  # Fast one wins

import pytest
import asyncio
from unittest.mock import AsyncMock

from polymarket.trading.retry_logic import RetryConfig, fetch_with_retry


@pytest.mark.asyncio
async def test_retry_success_on_first_attempt():
    """Function succeeds on first attempt."""
    fetch_func = AsyncMock(return_value={"price": 67000})

    result = await fetch_with_retry(fetch_func, "TestAPI")

    assert result == {"price": 67000}
    assert fetch_func.call_count == 1


@pytest.mark.asyncio
async def test_retry_success_after_failure():
    """Function fails once then succeeds."""
    fetch_func = AsyncMock(side_effect=[
        Exception("Temporary error"),
        {"price": 67000}
    ])

    result = await fetch_with_retry(fetch_func, "TestAPI")

    assert result == {"price": 67000}
    assert fetch_func.call_count == 2


@pytest.mark.asyncio
async def test_retry_all_attempts_fail():
    """All retry attempts fail."""
    fetch_func = AsyncMock(side_effect=Exception("Persistent error"))

    result = await fetch_with_retry(fetch_func, "TestAPI")

    assert result is None
    assert fetch_func.call_count == 3  # 1 initial + 2 retries


@pytest.mark.asyncio
async def test_retry_respects_timeout():
    """Timeout is enforced per attempt."""
    async def slow_func():
        await asyncio.sleep(35)  # Longer than 30s timeout
        return {"price": 67000}

    config = RetryConfig(timeout=1, max_attempts=1)
    result = await fetch_with_retry(slow_func, "TestAPI", config)

    assert result is None

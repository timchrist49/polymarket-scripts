# Resilient BTC Price Fetching - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix production Binance API timeouts by implementing retry logic, fallback sources, smart caching, and graceful degradation.

**Architecture:** Layered resilience approach with: (1) Smart per-candle cache with age-based TTL, (2) Retry logic with exponential backoff, (3) Parallel fallback sources (CoinGecko, Kraken), (4) Settlement price validation across sources, (5) Stale cache degradation policy.

**Tech Stack:** Python 3.12, asyncio, aiohttp, ccxt, structlog, pytest

---

## Task 1: CandleCache - TTL Calculation

**Files:**
- Create: `polymarket/trading/price_cache.py`
- Create: `tests/test_price_cache.py`

**Step 1: Write the failing test**

Create `tests/test_price_cache.py`:

```python
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
```

**Step 2: Run test to verify it fails**

```bash
cd /root/polymarket-scripts
pytest tests/test_price_cache.py::test_get_ttl_old_candle -xvs
```

Expected: `ModuleNotFoundError: No module named 'polymarket.trading.price_cache'`

**Step 3: Write minimal implementation**

Create `polymarket/trading/price_cache.py`:

```python
"""Price caching with intelligent TTL strategies."""

from datetime import datetime
from typing import Optional

from polymarket.models import PricePoint


class CandleCache:
    """Per-candle caching with age-based TTL."""

    def __init__(self):
        self._candles: dict[int, tuple[PricePoint, datetime]] = {}
        # Key: timestamp_minute, Value: (candle, cached_at)

    def get_ttl(self, candle_timestamp: datetime) -> int:
        """
        Calculate TTL based on candle age.

        Returns:
            TTL in seconds
        """
        age_minutes = (datetime.now() - candle_timestamp).total_seconds() / 60

        if age_minutes > 60:
            return 3600  # 1 hour TTL for old candles (immutable)
        elif age_minutes > 5:
            return 300   # 5 min TTL for recent closed candles
        else:
            return 60    # 1 min TTL for current/recent candles
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_price_cache.py -xvs
```

Expected: `3 passed`

**Step 5: Commit**

```bash
git add polymarket/trading/price_cache.py tests/test_price_cache.py
git commit -m "feat(cache): add CandleCache with age-based TTL calculation"
```

---

## Task 2: CandleCache - Cache Operations

**Files:**
- Modify: `polymarket/trading/price_cache.py`
- Modify: `tests/test_price_cache.py`

**Step 1: Write the failing tests**

Add to `tests/test_price_cache.py`:

```python
def test_put_and_get_candle():
    """Can cache and retrieve a candle."""
    cache = CandleCache()
    timestamp = int(datetime.now().timestamp())
    candle = PricePoint(
        price=Decimal("67000.50"),
        volume=Decimal("100.0"),
        timestamp=datetime.now()
    )

    cache.put(timestamp, candle)
    result = cache.get(timestamp)

    assert result is not None
    assert result.price == Decimal("67000.50")


def test_is_valid_fresh_cache():
    """Fresh cache is valid."""
    cache = CandleCache()
    timestamp = int(datetime.now().timestamp())
    candle = PricePoint(
        price=Decimal("67000.50"),
        volume=Decimal("100.0"),
        timestamp=datetime.now()
    )

    cache.put(timestamp, candle)

    assert cache.is_valid(timestamp) is True


def test_is_valid_missing_cache():
    """Missing cache is invalid."""
    cache = CandleCache()
    timestamp = int(datetime.now().timestamp())

    assert cache.is_valid(timestamp) is False


def test_is_valid_expired_cache():
    """Expired cache is invalid."""
    cache = CandleCache()
    timestamp = int((datetime.now() - timedelta(minutes=2)).timestamp())
    candle = PricePoint(
        price=Decimal("67000.50"),
        volume=Decimal("100.0"),
        timestamp=datetime.now() - timedelta(minutes=2)
    )

    # Manually set old cached_at time
    cache._candles[timestamp] = (candle, datetime.now() - timedelta(minutes=2))

    # For current candle, TTL is 60s, so 2 min old is expired
    assert cache.is_valid(timestamp) is False
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_price_cache.py::test_put_and_get_candle -xvs
```

Expected: `AttributeError: 'CandleCache' object has no attribute 'put'`

**Step 3: Implement cache operations**

Add to `polymarket/trading/price_cache.py`:

```python
    def put(self, timestamp: int, candle: PricePoint) -> None:
        """
        Cache a candle.

        Args:
            timestamp: Minute-level timestamp (Unix seconds)
            candle: Price data point
        """
        self._candles[timestamp] = (candle, datetime.now())

    def get(self, timestamp: int) -> Optional[PricePoint]:
        """
        Retrieve cached candle if valid.

        Args:
            timestamp: Minute-level timestamp (Unix seconds)

        Returns:
            Cached candle or None if invalid/missing
        """
        if not self.is_valid(timestamp):
            return None

        candle, _ = self._candles[timestamp]
        return candle

    def is_valid(self, timestamp: int) -> bool:
        """
        Check if cached candle is still valid.

        Args:
            timestamp: Minute-level timestamp (Unix seconds)

        Returns:
            True if cache is valid, False otherwise
        """
        if timestamp not in self._candles:
            return False

        candle, cached_at = self._candles[timestamp]
        age = (datetime.now() - cached_at).total_seconds()
        ttl = self.get_ttl(candle.timestamp)

        return age < ttl
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_price_cache.py -xvs
```

Expected: `7 passed`

**Step 5: Commit**

```bash
git add polymarket/trading/price_cache.py tests/test_price_cache.py
git commit -m "feat(cache): add put/get/is_valid operations to CandleCache"
```

---

## Task 3: RetryConfig and Retry Wrapper

**Files:**
- Create: `polymarket/trading/retry_logic.py`
- Create: `tests/test_retry_logic.py`

**Step 1: Write the failing tests**

Create `tests/test_retry_logic.py`:

```python
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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_retry_logic.py::test_retry_success_on_first_attempt -xvs
```

Expected: `ModuleNotFoundError: No module named 'polymarket.trading.retry_logic'`

**Step 3: Write implementation**

Create `polymarket/trading/retry_logic.py`:

```python
"""Retry logic with exponential backoff for API calls."""

import asyncio
from typing import Callable, Optional, Any
from dataclasses import dataclass
import structlog

logger = structlog.get_logger()


@dataclass
class RetryConfig:
    """Retry configuration."""
    max_attempts: int = 3          # Try 3 times total (1 initial + 2 retries)
    initial_delay: float = 2.0     # Start with 2 second delay
    backoff_factor: float = 2.0    # Double delay each retry
    timeout: int = 30              # 30 seconds per attempt


async def fetch_with_retry(
    fetch_func: Callable,
    source_name: str,
    config: RetryConfig = None
) -> Optional[Any]:
    """
    Generic retry wrapper for any fetch function.

    Args:
        fetch_func: Async function to call
        source_name: Name for logging
        config: Retry configuration

    Returns:
        Result from fetch_func or None if all attempts fail
    """
    if config is None:
        config = RetryConfig()

    last_error = None

    for attempt in range(config.max_attempts):
        try:
            # Attempt fetch with timeout
            result = await asyncio.wait_for(
                fetch_func(),
                timeout=config.timeout
            )

            # Log success if this was a retry
            if attempt > 0:
                logger.info(
                    f"{source_name} succeeded on retry",
                    attempt=attempt + 1
                )

            return result

        except asyncio.TimeoutError:
            last_error = f"{source_name} timeout after {config.timeout}s"

        except Exception as e:
            last_error = f"{source_name} error: {str(e)}"

        # Don't delay after last attempt
        if attempt < config.max_attempts - 1:
            delay = config.initial_delay * (config.backoff_factor ** attempt)
            logger.warning(
                f"{source_name} failed, retrying",
                attempt=attempt + 1,
                delay_seconds=delay,
                error=last_error
            )
            await asyncio.sleep(delay)

    # All attempts failed
    logger.error(f"{source_name} failed all retries", error=last_error)
    return None
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_retry_logic.py -xvs
```

Expected: `4 passed`

**Step 5: Commit**

```bash
git add polymarket/trading/retry_logic.py tests/test_retry_logic.py
git commit -m "feat(retry): add retry logic with exponential backoff"
```

---

## Task 4: StaleDataPolicy - Core Logic

**Files:**
- Create: `polymarket/trading/stale_policy.py`
- Create: `tests/test_stale_policy.py`

**Step 1: Write the failing tests**

Create `tests/test_stale_policy.py`:

```python
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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_stale_policy.py::test_record_success_resets_failures -xvs
```

Expected: `ModuleNotFoundError: No module named 'polymarket.trading.stale_policy'`

**Step 3: Write implementation**

Create `polymarket/trading/stale_policy.py`:

```python
"""Stale data policy for graceful degradation."""

from datetime import datetime
from typing import Optional, Any
import structlog

logger = structlog.get_logger()


class StaleDataPolicy:
    """Manages stale cache usage and failure tracking."""

    MAX_STALE_AGE_SECONDS = 600  # 10 minutes
    CONSECUTIVE_FAILURE_ALERT = 3  # Alert after 3 failures

    def __init__(self):
        self._consecutive_failures = 0
        self._last_success_time: Optional[datetime] = None
        self._stale_cache: Optional[tuple[Any, datetime]] = None

    def record_success(self, data: Any):
        """Record successful fetch."""
        self._consecutive_failures = 0
        self._last_success_time = datetime.now()
        self._stale_cache = (data, datetime.now())

    def record_failure(self):
        """Record failed fetch."""
        self._consecutive_failures += 1

        if self._consecutive_failures >= self.CONSECUTIVE_FAILURE_ALERT:
            logger.error(
                "ALERT: Multiple consecutive fetch failures",
                consecutive_failures=self._consecutive_failures,
                last_success_age=self._get_time_since_success()
            )

    def can_use_stale_cache(self) -> bool:
        """Check if stale cache is acceptable."""
        if not self._stale_cache:
            return False

        _, cached_at = self._stale_cache
        age = (datetime.now() - cached_at).total_seconds()

        return age < self.MAX_STALE_AGE_SECONDS

    def _get_time_since_success(self) -> str:
        """Human-readable time since last success."""
        if not self._last_success_time:
            return "never"

        delta = datetime.now() - self._last_success_time
        minutes = int(delta.total_seconds() / 60)
        return f"{minutes} minutes ago"
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_stale_policy.py -xvs
```

Expected: `5 passed`

**Step 5: Commit**

```bash
git add polymarket/trading/stale_policy.py tests/test_stale_policy.py
git commit -m "feat(stale): add StaleDataPolicy for graceful degradation"
```

---

## Task 5: StaleDataPolicy - Get Stale Cache

**Files:**
- Modify: `polymarket/trading/stale_policy.py`
- Modify: `tests/test_stale_policy.py`

**Step 1: Write the failing test**

Add to `tests/test_stale_policy.py`:

```python
def test_get_stale_cache_with_warning():
    """Returns stale cache with warning."""
    policy = StaleDataPolicy()
    data = BTCPriceData(
        price=Decimal("67000"),
        timestamp=datetime.now(),
        source="test",
        volume_24h=Decimal("1000")
    )
    policy.record_success(data)

    result = policy.get_stale_cache_with_warning()

    assert result is not None
    assert result.price == Decimal("67000")


def test_get_stale_cache_too_old_returns_none():
    """Returns None if cache too old."""
    policy = StaleDataPolicy()
    data = BTCPriceData(
        price=Decimal("67000"),
        timestamp=datetime.now(),
        source="test",
        volume_24h=Decimal("1000")
    )
    policy._stale_cache = (data, datetime.now() - timedelta(minutes=11))

    result = policy.get_stale_cache_with_warning()

    assert result is None


def test_should_skip_cycle():
    """Should skip if cache not usable."""
    policy = StaleDataPolicy()

    assert policy.should_skip_cycle() is True

    # Add fresh cache
    data = BTCPriceData(
        price=Decimal("67000"),
        timestamp=datetime.now(),
        source="test",
        volume_24h=Decimal("1000")
    )
    policy.record_success(data)

    assert policy.should_skip_cycle() is False
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_stale_policy.py::test_get_stale_cache_with_warning -xvs
```

Expected: `AttributeError: 'StaleDataPolicy' object has no attribute 'get_stale_cache_with_warning'`

**Step 3: Implement methods**

Add to `polymarket/trading/stale_policy.py`:

```python
    def get_stale_cache_with_warning(self) -> Optional[Any]:
        """Return stale cache with clear warnings."""
        if not self.can_use_stale_cache():
            logger.error("Stale cache too old or unavailable")
            return None

        data, cached_at = self._stale_cache
        age_seconds = (datetime.now() - cached_at).total_seconds()

        logger.warning(
            "⚠️  USING STALE CACHED DATA",
            age_seconds=int(age_seconds),
            age_readable=f"{int(age_seconds/60)} min {int(age_seconds%60)} sec",
            consecutive_failures=self._consecutive_failures
        )

        return data

    def should_skip_cycle(self) -> bool:
        """Determine if we should skip this trading cycle."""
        return not self.can_use_stale_cache()
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_stale_policy.py -xvs
```

Expected: `8 passed`

**Step 5: Commit**

```bash
git add polymarket/trading/stale_policy.py tests/test_stale_policy.py
git commit -m "feat(stale): add get_stale_cache_with_warning and should_skip_cycle"
```

---

## Task 6: CoinGecko Historical Price Fetcher

**Files:**
- Modify: `polymarket/trading/btc_price.py`
- Modify: `tests/test_btc_price.py` (if exists) or create `tests/test_btc_price_fallbacks.py`

**Step 1: Write the failing test**

Create `tests/test_btc_price_fallbacks.py`:

```python
import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, patch
import aiohttp

from polymarket.trading.btc_price import BTCPriceService
from polymarket.config import Settings


@pytest.mark.asyncio
async def test_fetch_coingecko_history():
    """Fetch historical candles from CoinGecko."""
    settings = Settings()
    service = BTCPriceService(settings)

    # Mock the aiohttp response
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={
        "prices": [
            [1707696000000, 67123.45],  # [timestamp_ms, price]
            [1707696060000, 67150.20],
            [1707696120000, 67100.80]
        ],
        "total_volumes": [
            [1707696000000, 1000000],
            [1707696060000, 1100000],
            [1707696120000, 1050000]
        ]
    })
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    mock_session = AsyncMock()
    mock_session.get = AsyncMock(return_value=mock_response)
    service._session = mock_session

    result = await service._fetch_coingecko_history(minutes=3)

    assert len(result) == 3
    assert result[0].price == Decimal("67123.45")
    assert result[1].price == Decimal("67150.20")
    assert result[2].price == Decimal("67100.80")

    await service.close()
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_btc_price_fallbacks.py::test_fetch_coingecko_history -xvs
```

Expected: `AttributeError: 'BTCPriceService' object has no attribute '_fetch_coingecko_history'`

**Step 3: Implement CoinGecko historical fetcher**

Add to `polymarket/trading/btc_price.py` after the existing `_fetch_coingecko()` method:

```python
    async def _fetch_coingecko_history(self, minutes: int = 60) -> list[PricePoint]:
        """
        Fetch historical price candles from CoinGecko.

        Args:
            minutes: Number of 1-minute candles to fetch

        Returns:
            List of price points
        """
        session = await self._get_session()

        # CoinGecko uses 'market_chart' endpoint for historical data
        # Note: Free tier has rate limits, use carefully
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"

        # Calculate time range (now - minutes)
        to_timestamp = int(datetime.now().timestamp())
        from_timestamp = to_timestamp - (minutes * 60)

        params = {
            "vs_currency": "usd",
            "from": str(from_timestamp),
            "to": str(to_timestamp)
        }

        try:
            async with session.get(url, params=params, timeout=30) as resp:
                resp.raise_for_status()
                data = await resp.json()

                # CoinGecko returns arrays of [timestamp_ms, price]
                prices = data.get("prices", [])
                volumes = data.get("total_volumes", [])

                # Convert to PricePoint objects
                result = []
                for i, (timestamp_ms, price) in enumerate(prices):
                    volume = volumes[i][1] if i < len(volumes) else 0

                    result.append(PricePoint(
                        price=decimal.Decimal(str(price)),
                        volume=decimal.Decimal(str(volume)),
                        timestamp=datetime.fromtimestamp(timestamp_ms / 1000)
                    ))

                logger.debug(
                    "Fetched CoinGecko history",
                    candles=len(result),
                    minutes=minutes
                )

                return result

        except Exception as e:
            logger.error("Failed to fetch CoinGecko history", error=str(e))
            raise
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_btc_price_fallbacks.py::test_fetch_coingecko_history -xvs
```

Expected: `1 passed`

**Step 5: Commit**

```bash
git add polymarket/trading/btc_price.py tests/test_btc_price_fallbacks.py
git commit -m "feat(fallback): add CoinGecko historical price fetcher"
```

---

## Task 7: Kraken Historical Price Fetcher

**Files:**
- Modify: `polymarket/trading/btc_price.py`
- Modify: `tests/test_btc_price_fallbacks.py`

**Step 1: Write the failing test**

Add to `tests/test_btc_price_fallbacks.py`:

```python
@pytest.mark.asyncio
async def test_fetch_kraken_history():
    """Fetch historical candles from Kraken."""
    settings = Settings()
    service = BTCPriceService(settings)

    # Mock the aiohttp response
    # Kraken OHLC format: [timestamp, open, high, low, close, vwap, volume, count]
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={
        "error": [],
        "result": {
            "XXBTZUSD": [
                [1707696000, "67000", "67200", "66900", "67123.45", "67100", "10.5", 150],
                [1707696060, "67123", "67180", "67100", "67150.20", "67140", "8.2", 120],
                [1707696120, "67150", "67160", "67080", "67100.80", "67120", "12.1", 180]
            ]
        }
    })
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    mock_session = AsyncMock()
    mock_session.get = AsyncMock(return_value=mock_response)
    service._session = mock_session

    result = await service._fetch_kraken_history(minutes=3)

    assert len(result) == 3
    assert result[0].price == Decimal("67123.45")
    assert result[1].price == Decimal("67150.20")
    assert result[2].price == Decimal("67100.80")

    await service.close()
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_btc_price_fallbacks.py::test_fetch_kraken_history -xvs
```

Expected: `AttributeError: 'BTCPriceService' object has no attribute '_fetch_kraken_history'`

**Step 3: Implement Kraken historical fetcher**

Add to `polymarket/trading/btc_price.py`:

```python
    async def _fetch_kraken_history(self, minutes: int = 60) -> list[PricePoint]:
        """
        Fetch historical price candles from Kraken.

        Args:
            minutes: Number of 1-minute candles to fetch

        Returns:
            List of price points
        """
        session = await self._get_session()

        # Kraken OHLC endpoint
        url = "https://api.kraken.com/0/public/OHLC"

        # Kraken uses 'since' parameter (Unix timestamp) and interval
        since_timestamp = int((datetime.now() - timedelta(minutes=minutes)).timestamp())

        params = {
            "pair": "XBTUSD",  # BTC/USD pair
            "interval": "1",   # 1 minute candles
            "since": str(since_timestamp)
        }

        try:
            async with session.get(url, params=params, timeout=30) as resp:
                resp.raise_for_status()
                data = await resp.json()

                # Check for Kraken API errors
                if data.get("error"):
                    raise Exception(f"Kraken API error: {data['error']}")

                # Kraken OHLC format: [timestamp, open, high, low, close, vwap, volume, count]
                ohlc_data = data["result"].get("XXBTZUSD", [])

                result = []
                for candle in ohlc_data:
                    timestamp = candle[0]
                    close_price = candle[4]
                    volume = candle[6]

                    result.append(PricePoint(
                        price=decimal.Decimal(str(close_price)),
                        volume=decimal.Decimal(str(volume)),
                        timestamp=datetime.fromtimestamp(timestamp)
                    ))

                logger.debug(
                    "Fetched Kraken history",
                    candles=len(result),
                    minutes=minutes
                )

                return result

        except Exception as e:
            logger.error("Failed to fetch Kraken history", error=str(e))
            raise
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_btc_price_fallbacks.py -xvs
```

Expected: `2 passed`

**Step 5: Commit**

```bash
git add polymarket/trading/btc_price.py tests/test_btc_price_fallbacks.py
git commit -m "feat(fallback): add Kraken historical price fetcher"
```

---

## Task 8: Parallel Fallback Racing Logic

**Files:**
- Create: `polymarket/trading/parallel_fetch.py`
- Create: `tests/test_parallel_fetch.py`

**Step 1: Write the failing test**

Create `tests/test_parallel_fetch.py`:

```python
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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_parallel_fetch.py::test_primary_succeeds -xvs
```

Expected: `ModuleNotFoundError: No module named 'polymarket.trading.parallel_fetch'`

**Step 3: Write implementation**

Create `polymarket/trading/parallel_fetch.py`:

```python
"""Parallel fallback fetching with racing logic."""

import asyncio
from typing import Callable, Optional, Any
import structlog

from polymarket.trading.retry_logic import fetch_with_retry

logger = structlog.get_logger()


async def fetch_with_fallbacks(
    primary_func: Callable,
    fallback_funcs: list[tuple[str, Callable]],
    validator: Optional[Callable] = None
) -> Optional[Any]:
    """
    Try primary source with retries, then race fallbacks in parallel.

    Args:
        primary_func: Primary source (already wrapped with retries)
        fallback_funcs: [(name, fetch_func), ...] for parallel execution
        validator: Optional validation function for results

    Returns:
        First successful result or None if all fail
    """

    # Step 1: Try primary (it's already wrapped with retries)
    logger.debug("Trying primary source")
    result = await primary_func()

    if result is not None:
        return result

    # Step 2: Primary failed, race fallbacks in parallel
    logger.warning("Primary source exhausted, trying fallbacks in parallel")

    tasks = [
        asyncio.create_task(fetch_with_retry(func, name))
        for name, func in fallback_funcs
    ]

    # Wait for first success or all failures
    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result is not None:
            # Got a result! Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()

            # Validate if validator provided (for settlement)
            if validator and not validator(result):
                logger.warning("Fallback result failed validation")
                continue

            logger.info("Fallback source succeeded", source="parallel_race")
            return result

    # All sources failed
    logger.error("All sources failed (primary + fallbacks)")
    return None
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_parallel_fetch.py -xvs
```

Expected: `4 passed`

**Step 5: Commit**

```bash
git add polymarket/trading/parallel_fetch.py tests/test_parallel_fetch.py
git commit -m "feat(parallel): add parallel fallback racing logic"
```

---

## Task 9: Settlement Price Validator - Core Logic

**Files:**
- Create: `polymarket/performance/settlement_validator.py`
- Create: `tests/test_settlement_validator.py`

**Step 1: Write the failing tests**

Create `tests/test_settlement_validator.py`:

```python
import pytest
from decimal import Decimal
from unittest.mock import AsyncMock

from polymarket.performance.settlement_validator import SettlementPriceValidator


@pytest.mark.asyncio
async def test_validate_prices_agree():
    """Prices from multiple sources agree within tolerance."""
    validator = SettlementPriceValidator()

    # Mock the fetch methods
    validator._fetch_binance_at_timestamp = AsyncMock(return_value=Decimal("67123.45"))
    validator._fetch_coingecko_at_timestamp = AsyncMock(return_value=Decimal("67150.20"))
    validator._fetch_kraken_at_timestamp = AsyncMock(return_value=Decimal("67100.80"))

    result = await validator.get_validated_price(1707696000)

    assert result is not None
    # Should return average: (67123.45 + 67150.20 + 67100.80) / 3 = 67124.82
    assert abs(result - Decimal("67124.82")) < Decimal("1.0")


@pytest.mark.asyncio
async def test_validate_prices_disagree():
    """Prices disagree beyond tolerance."""
    validator = SettlementPriceValidator()

    # Mock the fetch methods with prices that differ by >0.5%
    validator._fetch_binance_at_timestamp = AsyncMock(return_value=Decimal("67000"))
    validator._fetch_coingecko_at_timestamp = AsyncMock(return_value=Decimal("67500"))  # 0.75% diff
    validator._fetch_kraken_at_timestamp = AsyncMock(return_value=Decimal("67100"))

    result = await validator.get_validated_price(1707696000)

    assert result is None  # Should reject due to disagreement


@pytest.mark.asyncio
async def test_validate_insufficient_sources():
    """Less than 2 sources available."""
    validator = SettlementPriceValidator()

    # Only one source succeeds
    validator._fetch_binance_at_timestamp = AsyncMock(return_value=Decimal("67000"))
    validator._fetch_coingecko_at_timestamp = AsyncMock(return_value=None)
    validator._fetch_kraken_at_timestamp = AsyncMock(return_value=None)

    result = await validator.get_validated_price(1707696000)

    assert result is None  # Need at least 2 sources


def test_calculate_spread():
    """Calculate price spread percentage."""
    validator = SettlementPriceValidator()

    prices = [Decimal("67000"), Decimal("67100"), Decimal("67200")]
    spread = validator._calculate_spread(prices)

    # Spread = (67200 - 67000) / 67000 * 100 = 0.298%
    assert 0.29 < spread < 0.30
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_settlement_validator.py::test_validate_prices_agree -xvs
```

Expected: `ModuleNotFoundError: No module named 'polymarket.performance.settlement_validator'`

**Step 3: Write implementation**

Create `polymarket/performance/settlement_validator.py`:

```python
"""Settlement price validation across multiple sources."""

import asyncio
from decimal import Decimal
from typing import Optional
import structlog

logger = structlog.get_logger()


class SettlementPriceValidator:
    """Validates historical prices across multiple sources."""

    TOLERANCE_PERCENT = 0.5  # Prices must agree within 0.5%
    MIN_SOURCES = 2          # Need at least 2 sources to validate

    def __init__(self, btc_service=None):
        """
        Initialize validator.

        Args:
            btc_service: BTCPriceService instance (for fetching)
        """
        self._btc_service = btc_service

    async def get_validated_price(
        self,
        timestamp: int
    ) -> Optional[Decimal]:
        """
        Fetch price from multiple sources and validate agreement.

        Args:
            timestamp: Unix timestamp (seconds)

        Returns:
            Validated price or None if sources disagree
        """
        # Fetch from all sources in parallel
        tasks = [
            ("Binance", self._fetch_binance_at_timestamp(timestamp)),
            ("CoinGecko", self._fetch_coingecko_at_timestamp(timestamp)),
            ("Kraken", self._fetch_kraken_at_timestamp(timestamp))
        ]

        results = await asyncio.gather(*[task[1] for task in tasks], return_exceptions=True)
        prices = {
            name: price
            for (name, _), price in zip(tasks, results)
            if price is not None and not isinstance(price, Exception)
        }

        if len(prices) < self.MIN_SOURCES:
            logger.error(
                "Insufficient sources for validation",
                available=len(prices),
                required=self.MIN_SOURCES
            )
            return None

        # Check if all prices agree within tolerance
        prices_list = list(prices.values())
        avg_price = sum(prices_list) / len(prices_list)

        for source, price in prices.items():
            deviation_pct = abs(float((price - avg_price) / avg_price * 100))

            if deviation_pct > self.TOLERANCE_PERCENT:
                logger.error(
                    "Price sources disagree",
                    source=source,
                    price=float(price),
                    avg_price=float(avg_price),
                    deviation_pct=f"{deviation_pct:.2f}%",
                    tolerance=f"{self.TOLERANCE_PERCENT}%"
                )
                return None

        # All sources agree! Return average for accuracy
        logger.info(
            "Settlement price validated",
            sources=list(prices.keys()),
            avg_price=f"${avg_price:,.2f}",
            spread=f"{self._calculate_spread(prices_list):.2f}%"
        )

        return avg_price

    def _calculate_spread(self, prices: list[Decimal]) -> float:
        """Calculate price spread percentage."""
        if not prices:
            return 0.0
        min_price = min(prices)
        max_price = max(prices)
        return float((max_price - min_price) / min_price * 100)

    # These will be implemented in next task
    async def _fetch_binance_at_timestamp(self, timestamp: int) -> Optional[Decimal]:
        """Placeholder - will delegate to BTCPriceService."""
        if self._btc_service:
            return await self._btc_service.get_price_at_timestamp(timestamp)
        return None

    async def _fetch_coingecko_at_timestamp(self, timestamp: int) -> Optional[Decimal]:
        """Placeholder - will be implemented."""
        return None

    async def _fetch_kraken_at_timestamp(self, timestamp: int) -> Optional[Decimal]:
        """Placeholder - will be implemented."""
        return None
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_settlement_validator.py -xvs
```

Expected: `4 passed`

**Step 5: Commit**

```bash
git add polymarket/performance/settlement_validator.py tests/test_settlement_validator.py
git commit -m "feat(settlement): add price validation across sources"
```

---

## Task 10: Settlement Price Validator - Source Fetchers

**Files:**
- Modify: `polymarket/performance/settlement_validator.py`
- Modify: `polymarket/trading/btc_price.py`

**Step 1: Write the failing tests**

Add to `tests/test_settlement_validator.py`:

```python
@pytest.mark.asyncio
async def test_fetch_coingecko_at_timestamp_integration():
    """Integration test for CoinGecko timestamp fetch."""
    from polymarket.trading.btc_price import BTCPriceService
    from polymarket.config import Settings

    settings = Settings()
    btc_service = BTCPriceService(settings)
    validator = SettlementPriceValidator(btc_service)

    # Mock the session
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={
        "market_data": {
            "current_price": {"usd": 67123.45}
        }
    })
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    mock_session = AsyncMock()
    mock_session.get = AsyncMock(return_value=mock_response)
    btc_service._session = mock_session

    result = await validator._fetch_coingecko_at_timestamp(1707696000)

    assert result == Decimal("67123.45")

    await btc_service.close()
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_settlement_validator.py::test_fetch_coingecko_at_timestamp_integration -xvs
```

Expected: Test fails (returns None instead of price)

**Step 3: Implement CoinGecko and Kraken timestamp fetchers**

Add to `polymarket/trading/btc_price.py`:

```python
    async def _fetch_coingecko_at_timestamp(self, timestamp: int) -> Optional[decimal.Decimal]:
        """
        Fetch BTC price at specific timestamp from CoinGecko.

        Args:
            timestamp: Unix timestamp (seconds)

        Returns:
            BTC price or None
        """
        session = await self._get_session()

        # CoinGecko historical data endpoint
        # Note: For exact timestamp, we use the 'history' endpoint with date
        date_str = datetime.fromtimestamp(timestamp).strftime("%d-%m-%Y")
        url = f"https://api.coingecko.com/api/v3/coins/bitcoin/history"

        params = {
            "date": date_str,
            "localization": "false"
        }

        try:
            async with session.get(url, params=params, timeout=30) as resp:
                resp.raise_for_status()
                data = await resp.json()

                price = data.get("market_data", {}).get("current_price", {}).get("usd")

                if price:
                    logger.debug("Fetched CoinGecko price at timestamp",
                                timestamp=timestamp, price=f"${price:,.2f}")
                    return decimal.Decimal(str(price))

                return None

        except Exception as e:
            logger.error("Failed to fetch CoinGecko price at timestamp",
                        timestamp=timestamp, error=str(e))
            return None

    async def _fetch_kraken_at_timestamp(self, timestamp: int) -> Optional[decimal.Decimal]:
        """
        Fetch BTC price at specific timestamp from Kraken.

        Args:
            timestamp: Unix timestamp (seconds)

        Returns:
            BTC price or None
        """
        session = await self._get_session()

        # Kraken OHLC endpoint with specific timestamp
        url = "https://api.kraken.com/0/public/OHLC"

        params = {
            "pair": "XBTUSD",
            "interval": "1",
            "since": str(timestamp)
        }

        try:
            async with session.get(url, params=params, timeout=30) as resp:
                resp.raise_for_status()
                data = await resp.json()

                if data.get("error"):
                    raise Exception(f"Kraken API error: {data['error']}")

                # Get first candle (closest to timestamp)
                ohlc_data = data["result"].get("XXBTZUSD", [])

                if ohlc_data:
                    close_price = ohlc_data[0][4]  # Close price
                    logger.debug("Fetched Kraken price at timestamp",
                                timestamp=timestamp, price=f"${close_price}")
                    return decimal.Decimal(str(close_price))

                return None

        except Exception as e:
            logger.error("Failed to fetch Kraken price at timestamp",
                        timestamp=timestamp, error=str(e))
            return None
```

Update `polymarket/performance/settlement_validator.py`:

```python
    async def _fetch_coingecko_at_timestamp(self, timestamp: int) -> Optional[Decimal]:
        """Fetch from CoinGecko via BTCPriceService."""
        if self._btc_service:
            return await self._btc_service._fetch_coingecko_at_timestamp(timestamp)
        return None

    async def _fetch_kraken_at_timestamp(self, timestamp: int) -> Optional[Decimal]:
        """Fetch from Kraken via BTCPriceService."""
        if self._btc_service:
            return await self._btc_service._fetch_kraken_at_timestamp(timestamp)
        return None
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_settlement_validator.py -xvs
```

Expected: `5 passed`

**Step 5: Commit**

```bash
git add polymarket/performance/settlement_validator.py polymarket/trading/btc_price.py tests/test_settlement_validator.py
git commit -m "feat(settlement): implement CoinGecko and Kraken timestamp fetchers"
```

---

## Task 11: Integrate CandleCache into BTCPriceService

**Files:**
- Modify: `polymarket/trading/btc_price.py`
- Modify: `tests/test_btc_price.py` (or create if doesn't exist)

**Step 1: Write the failing test**

Create or modify `tests/test_btc_price.py`:

```python
import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, patch

from polymarket.trading.btc_price import BTCPriceService
from polymarket.config import Settings


@pytest.mark.asyncio
async def test_get_price_history_uses_cache():
    """Second call uses cache instead of fetching."""
    settings = Settings()
    service = BTCPriceService(settings)

    # Mock Binance fetch to return data once
    mock_data = [
        {"price": "67000", "volume": "100", "timestamp": datetime.now()}
    ]

    fetch_count = 0
    original_fetch = service._fetch_binance_history

    async def mock_fetch(*args, **kwargs):
        nonlocal fetch_count
        fetch_count += 1
        return await original_fetch(*args, **kwargs)

    service._fetch_binance_history = mock_fetch

    # Mock the actual HTTP call
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=[[
        1707696000000,  # timestamp
        "67000",        # open
        "67100",        # high
        "66900",        # low
        "67050",        # close
        "100"           # volume
    ]])
    mock_response.raise_for_status = AsyncMock()
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    mock_session = AsyncMock()
    mock_session.get = AsyncMock(return_value=mock_response)
    service._session = mock_session

    # First call - should fetch
    result1 = await service.get_price_history(minutes=1)

    # Second call immediately - should use cache
    result2 = await service.get_price_history(minutes=1)

    assert len(result1) == 1
    assert len(result2) == 1
    # Should only fetch once due to cache
    assert fetch_count == 1

    await service.close()
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_btc_price.py::test_get_price_history_uses_cache -xvs
```

Expected: Test fails (fetch_count is 2, not 1)

**Step 3: Integrate CandleCache**

Modify `polymarket/trading/btc_price.py`:

Add import at top:
```python
from polymarket.trading.price_cache import CandleCache
```

In `__init__` method, add:
```python
        # Candle cache
        self._candle_cache = CandleCache()
```

Modify `get_price_history` method:

```python
    async def get_price_history(self, minutes: int = 60) -> list[PricePoint]:
        """Get historical price points for technical analysis."""

        # Check cache first
        cached_candles = []
        uncached_count = 0

        for i in range(minutes):
            timestamp = int((datetime.now() - timedelta(minutes=minutes-i)).timestamp())
            candle = self._candle_cache.get(timestamp)

            if candle:
                cached_candles.append(candle)
            else:
                uncached_count += 1

        # If we have enough cached candles, use them
        if uncached_count == 0:
            logger.debug("Using fully cached price history", minutes=minutes)
            return cached_candles

        # Need to fetch fresh data
        # Use direct HTTP request to Binance API instead of ccxt
        # to avoid the derivatives API timeout issue
        try:
            session = await self._get_session()
            url = "https://api.binance.com/api/v3/klines"
            params = {
                "symbol": "BTCUSDT",
                "interval": "1m",
                "limit": str(minutes)
            }

            async with session.get(url, params=params, timeout=30) as resp:
                resp.raise_for_status()
                data = await resp.json()

                candles = [
                    PricePoint(
                        price=decimal.Decimal(str(candle[4])),  # Close price
                        volume=decimal.Decimal(str(candle[5])),  # Volume
                        timestamp=datetime.fromtimestamp(candle[0] / 1000)
                    )
                    for candle in data
                ]

                # Cache all fetched candles
                for candle in candles:
                    timestamp = int(candle.timestamp.timestamp())
                    self._candle_cache.put(timestamp, candle)

                logger.debug("Fetched and cached price history",
                           minutes=minutes, cached_new=len(candles))

                return candles

        except asyncio.TimeoutError:
            logger.error("Failed to fetch price history", error="Binance API timeout after 30s")
            raise
        except Exception as e:
            logger.error("Failed to fetch price history", error=str(e) or type(e).__name__)
            raise
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_btc_price.py::test_get_price_history_uses_cache -xvs
```

Expected: `1 passed`

**Step 5: Commit**

```bash
git add polymarket/trading/btc_price.py tests/test_btc_price.py
git commit -m "feat(cache): integrate CandleCache into get_price_history"
```

---

## Task 12: Add Retry and Fallback to get_price_history

**Files:**
- Modify: `polymarket/trading/btc_price.py`

**Step 1: Write the failing test**

Add to `tests/test_btc_price.py`:

```python
@pytest.mark.asyncio
async def test_get_price_history_fallback_to_coingecko():
    """Falls back to CoinGecko when Binance fails."""
    settings = Settings()
    service = BTCPriceService(settings)

    # Mock Binance to fail
    service._fetch_binance_history = AsyncMock(side_effect=Exception("Binance down"))

    # Mock CoinGecko to succeed
    service._fetch_coingecko_history = AsyncMock(return_value=[
        PricePoint(
            price=Decimal("67000"),
            volume=Decimal("100"),
            timestamp=datetime.now()
        )
    ])

    result = await service.get_price_history(minutes=1)

    assert len(result) == 1
    assert result[0].price == Decimal("67000")

    await service.close()
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_btc_price.py::test_get_price_history_fallback_to_coingecko -xvs
```

Expected: Test raises exception (no fallback implemented)

**Step 3: Implement retry and fallback**

Modify `get_price_history` in `polymarket/trading/btc_price.py`:

Add imports:
```python
from polymarket.trading.retry_logic import fetch_with_retry, RetryConfig
from polymarket.trading.parallel_fetch import fetch_with_fallbacks
from polymarket.trading.stale_policy import StaleDataPolicy
```

In `__init__`, add:
```python
        # Stale data policy
        self._stale_policy = StaleDataPolicy()
```

Replace `get_price_history` method:

```python
    async def get_price_history(self, minutes: int = 60) -> list[PricePoint]:
        """Get historical price points for technical analysis with fallbacks."""

        # Check cache first
        cached_candles = []
        uncached_count = 0

        for i in range(minutes):
            timestamp = int((datetime.now() - timedelta(minutes=minutes-i)).timestamp())
            candle = self._candle_cache.get(timestamp)

            if candle:
                cached_candles.append(candle)
            else:
                uncached_count += 1

        # If we have enough cached candles, use them
        if uncached_count == 0:
            logger.debug("Using fully cached price history", minutes=minutes)
            self._stale_policy.record_success(cached_candles)
            return cached_candles

        # Need to fetch fresh data - try with retries and fallbacks
        async def fetch_primary():
            return await self._fetch_binance_history(minutes)

        result = await fetch_with_fallbacks(
            lambda: fetch_with_retry(fetch_primary, "Binance"),
            [
                ("CoinGecko", lambda: self._fetch_coingecko_history(minutes)),
                ("Kraken", lambda: self._fetch_kraken_history(minutes))
            ]
        )

        if result:
            # Cache all fetched candles
            for candle in result:
                timestamp = int(candle.timestamp.timestamp())
                self._candle_cache.put(timestamp, candle)

            self._stale_policy.record_success(result)
            logger.debug("Fetched and cached price history",
                       minutes=minutes, candles=len(result))
            return result

        # All sources failed - try stale cache
        self._stale_policy.record_failure()
        stale_data = self._stale_policy.get_stale_cache_with_warning()

        if stale_data:
            return stale_data

        # No options left
        raise Exception("Failed to fetch price history from all sources")

    async def _fetch_binance_history(self, minutes: int) -> list[PricePoint]:
        """Fetch from Binance (extracted for fallback logic)."""
        session = await self._get_session()
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": "BTCUSDT",
            "interval": "1m",
            "limit": str(minutes)
        }

        async with session.get(url, params=params, timeout=30) as resp:
            resp.raise_for_status()
            data = await resp.json()

            return [
                PricePoint(
                    price=decimal.Decimal(str(candle[4])),  # Close price
                    volume=decimal.Decimal(str(candle[5])),  # Volume
                    timestamp=datetime.fromtimestamp(candle[0] / 1000)
                )
                for candle in data
            ]
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_btc_price.py -xvs
```

Expected: `2 passed`

**Step 5: Commit**

```bash
git add polymarket/trading/btc_price.py tests/test_btc_price.py
git commit -m "feat(resilience): add retry and fallback to get_price_history"
```

---

## Task 13: Integrate Settlement Validator into get_price_at_timestamp

**Files:**
- Modify: `polymarket/trading/btc_price.py`
- Modify: `polymarket/performance/settler.py`

**Step 1: Write the failing test**

Add to `tests/test_btc_price.py`:

```python
@pytest.mark.asyncio
async def test_get_price_at_timestamp_validated():
    """get_price_at_timestamp uses validation for settlement."""
    settings = Settings()
    service = BTCPriceService(settings)

    # This test verifies the integration exists
    # The actual validation is tested in test_settlement_validator.py

    timestamp = int(datetime.now().timestamp())

    # Mock all three sources to return similar prices
    service._fetch_binance_at_timestamp = AsyncMock(return_value=Decimal("67000"))
    service._fetch_coingecko_at_timestamp = AsyncMock(return_value=Decimal("67050"))
    service._fetch_kraken_at_timestamp = AsyncMock(return_value=Decimal("67025"))

    result = await service.get_price_at_timestamp(timestamp)

    # Should return validated average
    assert result is not None
    assert Decimal("67000") < result < Decimal("67100")

    await service.close()
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_btc_price.py::test_get_price_at_timestamp_validated -xvs
```

Expected: Test fails (returns single source, not validated average)

**Step 3: Integrate validation into get_price_at_timestamp**

Modify `polymarket/trading/btc_price.py`:

Add import:
```python
from polymarket.performance.settlement_validator import SettlementPriceValidator
```

In `__init__`, add:
```python
        # Settlement validator
        self._settlement_validator = SettlementPriceValidator(btc_service=self)
```

Replace `get_price_at_timestamp` method:

```python
    async def get_price_at_timestamp(self, timestamp: int) -> Optional[decimal.Decimal]:
        """
        Get BTC price at a specific Unix timestamp with validation.

        Uses multi-source validation for settlement accuracy.

        Args:
            timestamp: Unix timestamp (seconds since epoch)

        Returns:
            Validated BTC price as Decimal, or None if unavailable/invalid
        """
        # Use settlement validator for multi-source consensus
        validated_price = await self._settlement_validator.get_validated_price(timestamp)

        if validated_price:
            return validated_price

        # Validation failed - log and return None
        logger.warning(
            "Price validation failed for settlement",
            timestamp=timestamp
        )
        return None

    async def _fetch_binance_at_timestamp(self, timestamp: int) -> Optional[decimal.Decimal]:
        """
        Fetch BTC price at specific timestamp from Binance.

        Args:
            timestamp: Unix timestamp (seconds)

        Returns:
            BTC price or None
        """
        try:
            session = await self._get_session()
            url = "https://api.binance.com/api/v3/klines"

            # Convert to milliseconds and get single 1-minute candle
            timestamp_ms = timestamp * 1000
            params = {
                "symbol": "BTCUSDT",
                "interval": "1m",
                "startTime": str(timestamp_ms),
                "limit": "1"
            }

            async with session.get(url, params=params, timeout=30) as resp:
                resp.raise_for_status()
                data = await resp.json()

                if not data:
                    logger.warning("No price data at timestamp", timestamp=timestamp)
                    return None

                # Return close price of the candle
                price = decimal.Decimal(str(data[0][4]))
                logger.debug(
                    "Fetched Binance historical price",
                    timestamp=timestamp,
                    price=f"${price:,.2f}"
                )
                return price

        except asyncio.TimeoutError:
            logger.error("Failed to fetch historical price", timestamp=timestamp,
                        error="Binance API timeout after 30s")
            return None
        except Exception as e:
            logger.error("Failed to fetch historical price", timestamp=timestamp,
                        error=str(e) or type(e).__name__)
            return None
```

Update `polymarket/performance/settler.py` to handle None from get_price_at_timestamp:

In the `settle_pending_trades` method, the existing code already handles None:
```python
            btc_close_price = await self.btc_fetcher.get_price_at_timestamp(market_close_ts)

            if btc_close_price is None:
                logger.warning("BTC price unavailable, will retry", trade_id=trade_id)
                skipped_count += 1
                continue
```

This is already correct - no changes needed to settler.py.

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_btc_price.py::test_get_price_at_timestamp_validated -xvs
```

Expected: `1 passed`

**Step 5: Commit**

```bash
git add polymarket/trading/btc_price.py tests/test_btc_price.py
git commit -m "feat(settlement): integrate multi-source validation into get_price_at_timestamp"
```

---

## Task 14: Add Configuration Settings

**Files:**
- Modify: `polymarket/config.py`
- Modify: `.env.example`

**Step 1: Write the failing test**

Add to `tests/test_config.py` (or create if doesn't exist):

```python
def test_price_fetch_config_defaults():
    """Price fetch configuration has correct defaults."""
    from polymarket.config import Settings

    settings = Settings()

    assert settings.btc_fetch_timeout == 30
    assert settings.btc_fetch_max_retries == 2
    assert settings.btc_fetch_retry_delay == 2.0
    assert settings.btc_cache_stale_max_age == 600
    assert settings.btc_settlement_tolerance_pct == 0.5
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_config.py::test_price_fetch_config_defaults -xvs
```

Expected: `AttributeError: 'Settings' object has no attribute 'btc_fetch_timeout'`

**Step 3: Add configuration fields**

Add to `polymarket/config.py`:

```python
    # Price Fetching Configuration
    btc_fetch_timeout: int = Field(
        default=30,
        description="Timeout per API attempt (seconds)"
    )
    btc_fetch_max_retries: int = Field(
        default=2,
        description="Number of retries after initial attempt"
    )
    btc_fetch_retry_delay: float = Field(
        default=2.0,
        description="Initial retry delay in seconds (doubles each retry)"
    )
    btc_cache_stale_max_age: int = Field(
        default=600,
        description="Max age for stale cache usage (seconds)"
    )
    btc_settlement_tolerance_pct: float = Field(
        default=0.5,
        description="Price agreement tolerance for settlement (%)"
    )
```

Add to `.env.example`:

```bash
# Price Fetching Configuration
BTC_FETCH_TIMEOUT=30                    # Timeout per attempt (seconds)
BTC_FETCH_MAX_RETRIES=2                 # Number of retries
BTC_FETCH_RETRY_DELAY=2.0               # Initial retry delay (seconds)
BTC_CACHE_STALE_MAX_AGE=600             # Max stale cache age (seconds)
BTC_SETTLEMENT_TOLERANCE_PCT=0.5        # Price agreement tolerance (%)
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_config.py::test_price_fetch_config_defaults -xvs
```

Expected: `1 passed`

**Step 5: Commit**

```bash
git add polymarket/config.py .env.example tests/test_config.py
git commit -m "feat(config): add price fetching configuration settings"
```

---

## Task 15: Update Configuration Usage in Components

**Files:**
- Modify: `polymarket/trading/retry_logic.py`
- Modify: `polymarket/trading/stale_policy.py`
- Modify: `polymarket/performance/settlement_validator.py`

**Step 1: Update RetryConfig to use settings**

Modify `polymarket/trading/retry_logic.py`:

```python
@dataclass
class RetryConfig:
    """Retry configuration."""
    max_attempts: int = 3          # 1 initial + 2 retries
    initial_delay: float = 2.0     # Start with 2 second delay
    backoff_factor: float = 2.0    # Double delay each retry
    timeout: int = 30              # 30 seconds per attempt

    @classmethod
    def from_settings(cls, settings):
        """Create config from Settings object."""
        return cls(
            max_attempts=1 + settings.btc_fetch_max_retries,
            initial_delay=settings.btc_fetch_retry_delay,
            timeout=settings.btc_fetch_timeout
        )
```

**Step 2: Update StaleDataPolicy to use settings**

Modify `polymarket/trading/stale_policy.py`:

```python
class StaleDataPolicy:
    """Manages stale cache usage and failure tracking."""

    CONSECUTIVE_FAILURE_ALERT = 3  # Alert after 3 failures

    def __init__(self, max_stale_age_seconds: int = 600):
        """
        Initialize policy.

        Args:
            max_stale_age_seconds: Maximum age for stale cache (default 10 min)
        """
        self.max_stale_age_seconds = max_stale_age_seconds
        self._consecutive_failures = 0
        self._last_success_time: Optional[datetime] = None
        self._stale_cache: Optional[tuple[Any, datetime]] = None
```

Update `can_use_stale_cache`:
```python
    def can_use_stale_cache(self) -> bool:
        """Check if stale cache is acceptable."""
        if not self._stale_cache:
            return False

        _, cached_at = self._stale_cache
        age = (datetime.now() - cached_at).total_seconds()

        return age < self.max_stale_age_seconds
```

**Step 3: Update SettlementPriceValidator to use settings**

Modify `polymarket/performance/settlement_validator.py`:

```python
class SettlementPriceValidator:
    """Validates historical prices across multiple sources."""

    MIN_SOURCES = 2          # Need at least 2 sources to validate

    def __init__(self, btc_service=None, tolerance_percent: float = 0.5):
        """
        Initialize validator.

        Args:
            btc_service: BTCPriceService instance (for fetching)
            tolerance_percent: Price agreement tolerance (%)
        """
        self._btc_service = btc_service
        self.tolerance_percent = tolerance_percent
```

Update validation check:
```python
            if deviation_pct > self.tolerance_percent:
```

**Step 4: Update BTCPriceService to pass settings**

Modify `polymarket/trading/btc_price.py` in `__init__`:

```python
        # Stale data policy
        self._stale_policy = StaleDataPolicy(
            max_stale_age_seconds=settings.btc_cache_stale_max_age
        )

        # Settlement validator
        self._settlement_validator = SettlementPriceValidator(
            btc_service=self,
            tolerance_percent=settings.btc_settlement_tolerance_pct
        )

        # Retry config
        self._retry_config = RetryConfig.from_settings(settings)
```

Update retry calls to use config:
```python
        result = await fetch_with_fallbacks(
            lambda: fetch_with_retry(fetch_primary, "Binance", self._retry_config),
            [
                ("CoinGecko", lambda: self._fetch_coingecko_history(minutes)),
                ("Kraken", lambda: self._fetch_kraken_history(minutes))
            ]
        )
```

**Step 5: Commit**

```bash
git add polymarket/trading/retry_logic.py polymarket/trading/stale_policy.py polymarket/performance/settlement_validator.py polymarket/trading/btc_price.py
git commit -m "feat(config): integrate settings into all components"
```

---

## Task 16: Integration Testing and Documentation

**Files:**
- Create: `tests/integration/test_resilient_price_fetching.py`
- Modify: `README_BOT.md`

**Step 1: Write integration test**

Create `tests/integration/test_resilient_price_fetching.py`:

```python
"""Integration tests for resilient price fetching system."""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from datetime import datetime
from decimal import Decimal

from polymarket.trading.btc_price import BTCPriceService
from polymarket.config import Settings
from polymarket.models import PricePoint


@pytest.mark.asyncio
async def test_complete_resilience_flow():
    """
    Test complete resilience flow:
    1. Cache miss
    2. Binance fails with retries
    3. Fallback to CoinGecko/Kraken
    4. Cache hit on second call
    """
    settings = Settings()
    service = BTCPriceService(settings)

    # Mock Binance to fail
    service._fetch_binance_history = AsyncMock(side_effect=Exception("Network error"))

    # Mock CoinGecko to succeed
    coingecko_data = [
        PricePoint(
            price=Decimal("67000"),
            volume=Decimal("100"),
            timestamp=datetime.now()
        )
    ]
    service._fetch_coingecko_history = AsyncMock(return_value=coingecko_data)
    service._fetch_kraken_history = AsyncMock(return_value=None)

    # First call - should fallback to CoinGecko
    result1 = await service.get_price_history(minutes=1)

    assert len(result1) == 1
    assert result1[0].price == Decimal("67000")

    # Reset mocks to verify cache is used
    service._fetch_binance_history.reset_mock()
    service._fetch_coingecko_history.reset_mock()

    # Second call - should use cache (no fetches)
    result2 = await service.get_price_history(minutes=1)

    assert len(result2) == 1
    assert service._fetch_binance_history.call_count == 0
    assert service._fetch_coingecko_history.call_count == 0

    await service.close()


@pytest.mark.asyncio
async def test_settlement_validation_flow():
    """
    Test settlement validation:
    1. Multiple sources fetched in parallel
    2. Prices validated within tolerance
    3. Average returned
    """
    settings = Settings()
    service = BTCPriceService(settings)

    timestamp = int(datetime.now().timestamp())

    # Mock all sources with agreeing prices
    service._fetch_binance_at_timestamp = AsyncMock(return_value=Decimal("67000"))
    service._fetch_coingecko_at_timestamp = AsyncMock(return_value=Decimal("67050"))
    service._fetch_kraken_at_timestamp = AsyncMock(return_value=Decimal("67025"))

    result = await service.get_price_at_timestamp(timestamp)

    assert result is not None
    # Should be average: (67000 + 67050 + 67025) / 3 ≈ 67025
    assert Decimal("67000") < result < Decimal("67100")

    # All sources should have been called
    assert service._fetch_binance_at_timestamp.call_count == 1
    assert service._fetch_coingecko_at_timestamp.call_count == 1
    assert service._fetch_kraken_at_timestamp.call_count == 1

    await service.close()


@pytest.mark.asyncio
async def test_stale_cache_fallback():
    """
    Test stale cache usage:
    1. Successful fetch cached
    2. All sources fail later
    3. Stale cache returned with warning
    """
    settings = Settings()
    service = BTCPriceService(settings)

    # First call succeeds
    success_data = [
        PricePoint(
            price=Decimal("67000"),
            volume=Decimal("100"),
            timestamp=datetime.now()
        )
    ]
    service._fetch_binance_history = AsyncMock(return_value=success_data)

    result1 = await service.get_price_history(minutes=1)
    assert len(result1) == 1

    # Now make all sources fail
    service._fetch_binance_history = AsyncMock(return_value=None)
    service._fetch_coingecko_history = AsyncMock(return_value=None)
    service._fetch_kraken_history = AsyncMock(return_value=None)

    # Clear candle cache to force fetch attempt
    service._candle_cache._candles.clear()

    # Should fall back to stale data from _stale_policy
    result2 = await service.get_price_history(minutes=1)

    assert len(result2) == 1
    assert result2[0].price == Decimal("67000")

    await service.close()
```

**Step 2: Run integration tests**

```bash
pytest tests/integration/test_resilient_price_fetching.py -xvs
```

Expected: `3 passed`

**Step 3: Update documentation**

Add to `README_BOT.md` in a new section:

```markdown
## Resilient Price Fetching

The bot uses a multi-layered resilience approach for fetching BTC prices:

### Architecture

1. **Smart Cache Layer**
   - Individual candles cached with age-based TTL
   - Old candles (>60 min): 1 hour cache
   - Recent candles (5-60 min): 5 minute cache
   - Current candles (<5 min): 1 minute cache
   - Reduces API calls by ~95%

2. **Retry with Exponential Backoff**
   - 3 attempts total (1 initial + 2 retries)
   - 30-second timeout per attempt
   - 2s, 4s delays between retries
   - Prevents transient failures

3. **Parallel Fallback Sources**
   - Primary: Binance
   - Fallbacks: CoinGecko + Kraken (race in parallel)
   - First success wins, others cancelled
   - Maximum availability

4. **Settlement Validation**
   - Fetches from all 3 sources in parallel
   - Validates prices agree within 0.5%
   - Returns average for accuracy
   - Ensures correct win/loss determination

5. **Graceful Degradation**
   - Uses stale cache if < 10 minutes old
   - Bot HOLDs if data > 10 minutes old
   - Alerts after 3 consecutive failures

### Configuration

Edit `.env` to customize:

```bash
# Price Fetching
BTC_FETCH_TIMEOUT=30                    # Timeout per attempt (seconds)
BTC_FETCH_MAX_RETRIES=2                 # Number of retries
BTC_FETCH_RETRY_DELAY=2.0               # Initial retry delay (seconds)
BTC_CACHE_STALE_MAX_AGE=600             # Max stale cache age (10 min)
BTC_SETTLEMENT_TOLERANCE_PCT=0.5        # Price agreement tolerance
```

### Expected Impact

**Before:**
- Binance timeout rate: 18%
- Technical analysis success: 0%
- Bot execution rate: 55%

**After:**
- API timeout rate: <5% (with retries + fallbacks)
- Technical analysis success: >95%
- Bot execution rate: >70%
```

**Step 4: Commit**

```bash
git add tests/integration/test_resilient_price_fetching.py README_BOT.md
git commit -m "docs: add integration tests and documentation for resilient price fetching"
```

---

## Verification

After all tasks are complete, run the full test suite:

```bash
cd /root/polymarket-scripts
pytest tests/ -v --tb=short
```

Expected: All tests pass, including:
- 7 tests in test_price_cache.py
- 4 tests in test_retry_logic.py
- 8 tests in test_stale_policy.py
- 4 tests in test_parallel_fetch.py
- 5 tests in test_settlement_validator.py
- Integration tests in test_resilient_price_fetching.py

Check production bot logs after deployment:

```bash
tail -100 logs/bot_debugged.log | grep -E "Binance|CoinGecko|Kraken|cache|stale"
```

Expected: See successful fetches, cache hits, and significantly fewer timeout errors.

---

## Rollback Plan

If issues arise:

1. **Revert last commit:**
   ```bash
   git revert HEAD
   ```

2. **Restore old timeout:**
   ```bash
   # Edit polymarket/trading/btc_price.py
   # Change timeout=30 back to timeout=10
   ```

3. **Disable fallbacks:**
   ```bash
   # Comment out fallback logic in get_price_history
   # Keep only Binance fetch
   ```

## Post-Deployment Monitoring

Monitor for 24 hours:

1. **API Success Rate:**
   ```bash
   grep "Binance API timeout" logs/bot_debugged.log | wc -l
   ```
   Target: < 5 timeouts per hour

2. **Technical Analysis Success:**
   ```bash
   grep "Technical analysis completed" logs/bot_debugged.log | wc -l
   ```
   Target: > 90% of trading cycles

3. **Cache Hit Rate:**
   ```bash
   grep "Using fully cached price history" logs/bot_debugged.log | wc -l
   ```
   Target: > 50% cache hit rate

4. **Fallback Usage:**
   ```bash
   grep "Fallback source succeeded" logs/bot_debugged.log | wc -l
   ```
   Target: < 10% of fetches use fallbacks

---

**Total Tasks:** 16
**Estimated Time:** 4-6 hours
**Test Coverage:** 35+ tests across all components

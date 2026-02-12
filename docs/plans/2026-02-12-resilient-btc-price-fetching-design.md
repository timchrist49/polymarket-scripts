# Resilient BTC Price Fetching System - Design Document

**Date:** 2026-02-12
**Author:** Claude Sonnet 4.5
**Status:** Approved

## Problem Statement

The production trading bot is experiencing frequent Binance API timeouts (22 failures in last 1000 logs), resulting in:
- Failed technical analysis (no RSI, MACD, momentum data)
- Bot falling back to "neutral defaults"
- Conservative behavior with excessive HOLD decisions
- Reduced trading frequency

**Root Cause:** 10-second timeout is too aggressive for Binance API, no retry logic, no fallback sources for historical data.

## Design Goals

1. **Increase timeout** from 10s → 30s per attempt
2. **Add retry logic** with exponential backoff (2 retries max)
3. **Implement fallback sources** (CoinGecko, Kraken) in parallel
4. **Add smart caching** for historical data (per-candle with TTL)
5. **Graceful degradation** with stale cache policy
6. **Settlement validation** to ensure price accuracy across sources

## Architecture Overview

### Layered Resilience Approach

```
┌─────────────────────────────────────────────────────────┐
│                    Price Request                         │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   Smart Cache Layer   │ ◄─── Check cached data first
         │  (per-candle TTL)     │
         └───────────┬───────────┘
                     │ Cache miss
                     ▼
         ┌───────────────────────┐
         │  Primary: Binance     │
         │  (2 retries, 30s TO)  │ ◄─── Retry with backoff
         └───────────┬───────────┘
                     │ All retries fail
                     ▼
         ┌─────────────────────────────┐
         │ Parallel Fallback Sources   │
         │  CoinGecko + Kraken         │ ◄─── Race them
         │  (first success wins)       │
         └───────────┬─────────────────┘
                     │ All sources fail
                     ▼
         ┌───────────────────────┐
         │  Stale Cache Policy   │
         │  < 10min: Use it      │
         │  > 10min: Return None │ ◄─── Graceful degradation
         └───────────────────────┘
```

**Key Changes:**
- 10s → 30s timeout per attempt
- 2 retries with exponential backoff (2s, 4s delays)
- Parallel fallback fetching (fastest wins)
- Smart per-candle caching
- Staleness-based degradation

## Component Design

### 1. Smart Cache Layer

**Challenge:** Historical data (60-minute candles) is requested every 30 seconds, but most candles don't change.

**Solution:** Cache individual candles with age-based TTL

```python
class CandleCache:
    """Per-candle caching with age-based TTL"""

    def __init__(self):
        self._candles: dict[int, tuple[PricePoint, datetime]] = {}
        # Key: timestamp_minute, Value: (candle, cached_at)

    def get_ttl(self, candle_timestamp: datetime) -> int:
        """Calculate TTL based on candle age"""
        age_minutes = (datetime.now() - candle_timestamp).total_seconds() / 60

        if age_minutes > 60:
            return 3600  # 1 hour TTL for old candles (immutable)
        elif age_minutes > 5:
            return 300   # 5 min TTL for recent closed candles
        else:
            return 60    # 1 min TTL for current/recent candles

    def is_valid(self, timestamp: int) -> bool:
        """Check if cached candle is still valid"""
        if timestamp not in self._candles:
            return False

        candle, cached_at = self._candles[timestamp]
        age = (datetime.now() - cached_at).total_seconds()
        ttl = self.get_ttl(candle.timestamp)

        return age < ttl
```

**Benefits:**
- Old candles (60+ min ago): Cached 1 hour (effectively permanent)
- Recent closed candles (5-60 min): Cached 5 minutes
- Current candle (<5 min): Cached 1 minute (stays fresh)
- Reduces API calls from ~120/hour to ~5/hour
- Memory usage: ~60 candles × 50 bytes = 3KB (negligible)

### 2. Retry Logic with Exponential Backoff

**Implementation:** Decorator pattern for clean retry handling

```python
class RetryConfig:
    """Retry configuration"""
    max_attempts: int = 3      # Try 3 times total (1 initial + 2 retries)
    initial_delay: float = 2.0 # Start with 2 second delay
    backoff_factor: float = 2.0 # Double delay each retry
    timeout: int = 30          # 30 seconds per attempt

async def fetch_with_retry(
    fetch_func: Callable,
    source_name: str,
    config: RetryConfig = RetryConfig()
) -> Optional[Any]:
    """Generic retry wrapper for any fetch function"""

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

**Retry Timeline:**
- Attempt 1: 0s (immediate)
- Attempt 2: 2s delay
- Attempt 3: 4s delay
- Total: ~6 seconds of retries + 3×30s timeout windows = max 96s

### 3. Parallel Fallback Sources

**Challenge:** When Binance fails after retries, try multiple backup sources simultaneously

**Implementation:** Async racing with first-success wins

```python
async def fetch_with_fallbacks(
    primary_func: Callable,
    fallback_funcs: list[tuple[str, Callable]],
    validator: Optional[Callable] = None
) -> Optional[Any]:
    """
    Try primary source with retries, then race fallbacks in parallel

    Args:
        primary_func: Primary source (Binance with retries)
        fallback_funcs: [(name, fetch_func), ...] for parallel execution
        validator: Optional validation function for results
    """

    # Step 1: Try primary with retries
    logger.debug("Trying primary source")
    result = await fetch_with_retry(primary_func, "Binance")

    if result is not None:
        return result

    # Step 2: Primary failed, race fallbacks in parallel
    logger.warning("Primary source exhausted, trying fallbacks in parallel")

    tasks = [
        asyncio.create_task(fetch_with_retry(func, name))
        for name, func in fallback_funcs
    ]

    # Wait for first success or all failures
    results = []
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

        results.append(result)

    # All sources failed
    logger.error("All sources failed (primary + fallbacks)")
    return None
```

**Fallback Sources:**
```python
fallback_funcs = [
    ("CoinGecko", self._fetch_coingecko_history),
    ("Kraken", self._fetch_kraken_history)
]
```

**Timing:** Whichever backup responds first wins (typically 1-3 seconds)

### 4. Settlement Price Validation

**Challenge:** Different APIs might return slightly different historical prices. Must ensure accuracy for win/loss determination.

**Solution:** Multi-source consensus with tolerance checking

```python
class SettlementPriceValidator:
    """Validates historical prices across multiple sources"""

    TOLERANCE_PERCENT = 0.5  # Prices must agree within 0.5%
    MIN_SOURCES = 2          # Need at least 2 sources to validate

    async def get_validated_price(
        self,
        timestamp: int
    ) -> Optional[Decimal]:
        """
        Fetch price from multiple sources and validate agreement

        Returns validated price or None if sources disagree
        """
        # Fetch from all sources in parallel
        tasks = [
            ("Binance", self._fetch_binance_at_timestamp(timestamp)),
            ("CoinGecko", self._fetch_coingecko_at_timestamp(timestamp)),
            ("Kraken", self._fetch_kraken_at_timestamp(timestamp))
        ]

        results = await asyncio.gather(*[task[1] for task in tasks])
        prices = {
            name: price
            for (name, _), price in zip(tasks, results)
            if price is not None
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
        """Calculate price spread percentage"""
        if not prices:
            return 0.0
        min_price = min(prices)
        max_price = max(prices)
        return float((max_price - min_price) / min_price * 100)
```

**Example:**
- Binance: $67,123.45
- CoinGecko: $67,150.20 (0.04% diff) ✓
- Kraken: $67,100.80 (0.03% diff) ✓
- Average: $67,124.82 (used for settlement)

**Safety:** If any source differs by >0.5%, settlement is skipped and retried next cycle.

### 5. Graceful Degradation & Stale Cache Policy

**Challenge:** When ALL sources fail, what data should the bot use?

**Solution:** Age-based stale cache policy with failure tracking

```python
class StaleDataPolicy:
    """Manages stale cache usage and failure tracking"""

    MAX_STALE_AGE_SECONDS = 600  # 10 minutes
    CONSECUTIVE_FAILURE_ALERT = 3 # Alert after 3 failures

    def __init__(self):
        self._consecutive_failures = 0
        self._last_success_time: Optional[datetime] = None
        self._stale_cache: Optional[tuple[Any, datetime]] = None

    def record_success(self, data: Any):
        """Record successful fetch"""
        self._consecutive_failures = 0
        self._last_success_time = datetime.now()
        self._stale_cache = (data, datetime.now())

    def record_failure(self):
        """Record failed fetch"""
        self._consecutive_failures += 1

        if self._consecutive_failures >= self.CONSECUTIVE_FAILURE_ALERT:
            logger.error(
                "ALERT: Multiple consecutive fetch failures",
                consecutive_failures=self._consecutive_failures,
                last_success_age=self._get_time_since_success()
            )

    def can_use_stale_cache(self) -> bool:
        """Check if stale cache is acceptable"""
        if not self._stale_cache:
            return False

        _, cached_at = self._stale_cache
        age = (datetime.now() - cached_at).total_seconds()

        return age < self.MAX_STALE_AGE_SECONDS

    def get_stale_cache_with_warning(self) -> Optional[Any]:
        """Return stale cache with clear warnings"""
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
        """Determine if we should skip this trading cycle"""
        return not self.can_use_stale_cache()

    def _get_time_since_success(self) -> str:
        """Human-readable time since last success"""
        if not self._last_success_time:
            return "never"

        delta = datetime.now() - self._last_success_time
        minutes = int(delta.total_seconds() / 60)
        return f"{minutes} minutes ago"
```

**Decision Flow:**
```
All sources fail
    │
    ├─ Cache < 10 min old?
    │   ├─ YES → Use stale data + ⚠️ warning
    │   └─ NO  → Return None (bot HOLDs)
    │
    └─ 3+ consecutive failures?
        └─ YES → Log ERROR alert
```

**Example Log Output:**
```
[WARNING] ⚠️ USING STALE CACHED DATA | age: 8 min 23 sec
[ERROR] ALERT: Multiple consecutive fetch failures | failures=3 | last_success=12 minutes ago
```

## Implementation Plan

### Phase 1: Core Infrastructure
1. Create `CandleCache` class
2. Create `RetryConfig` and `fetch_with_retry()` wrapper
3. Create `StaleDataPolicy` class
4. Add unit tests for each component

### Phase 2: Fallback Sources
1. Implement `_fetch_coingecko_history()` method
2. Implement `_fetch_kraken_history()` method
3. Implement `_fetch_coingecko_at_timestamp()` for settlement
4. Implement `_fetch_kraken_at_timestamp()` for settlement
5. Add parallel racing logic in `fetch_with_fallbacks()`

### Phase 3: Settlement Validation
1. Create `SettlementPriceValidator` class
2. Integrate into `get_price_at_timestamp()` method
3. Add validation tests with mock price sources

### Phase 4: Integration
1. Update `get_price_history()` to use cache + retries + fallbacks
2. Update `get_price_at_timestamp()` to use validation
3. Update `get_current_price()` to use enhanced fallback chain
4. Increase all timeouts from 10s → 30s

### Phase 5: Testing & Deployment
1. Run integration tests with real APIs
2. Test settlement with historical trades
3. Monitor logs for 24 hours on staging
4. Deploy to production with monitoring

## Success Metrics

**Before:**
- Binance API timeout rate: ~18% (22/120 attempts)
- Technical analysis success rate: 0%
- Bot execution rate: 55% (45% HOLD due to missing data)

**Target After:**
- API timeout rate: <5% (with retries + fallbacks)
- Technical analysis success rate: >95%
- Bot execution rate: >70% (reduced unnecessary HOLDs)
- Settlement success rate: >95% (validated prices)

## Risk Mitigation

**Risk 1:** Multiple API failures simultaneously
- **Mitigation:** Stale cache fallback (up to 10 minutes)
- **Impact:** Bot continues trading with slightly old data

**Risk 2:** Price sources disagree on historical data
- **Mitigation:** 0.5% tolerance validation for settlement
- **Impact:** Trade stays unsettled until next cycle

**Risk 3:** Increased API costs from fallback sources
- **Mitigation:** Smart caching reduces calls by 95%
- **Impact:** Minimal cost increase (<$5/month)

**Risk 4:** Longer response times (retries + fallbacks)
- **Mitigation:** Parallel fallback racing, typical <5s overhead
- **Impact:** Acceptable for 15-minute trading cycles

## Configuration

Add to `.env`:
```bash
# Price Fetching Configuration
BTC_FETCH_TIMEOUT=30                    # Timeout per attempt (seconds)
BTC_FETCH_MAX_RETRIES=2                 # Number of retries
BTC_FETCH_RETRY_DELAY=2.0               # Initial retry delay (seconds)
BTC_CACHE_STALE_MAX_AGE=600             # Max stale cache age (seconds)
BTC_SETTLEMENT_TOLERANCE_PCT=0.5        # Price agreement tolerance (%)
```

## Future Enhancements

1. **Adaptive timeout:** Increase timeout during known high-latency periods
2. **Circuit breaker:** Temporarily skip failing sources to reduce latency
3. **Metrics dashboard:** Real-time success/failure rates per source
4. **Price prediction:** Use ML to predict missing prices during outages

---

**Approved by:** User
**Next Steps:** Create implementation plan and execute

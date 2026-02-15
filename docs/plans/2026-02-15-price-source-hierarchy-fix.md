# Price Source Hierarchy Fix - Design Document

**Date:** 2026-02-15
**Status:** Approved
**Issue:** $330 price discrepancy in price_to_beat calculations

## Problem Statement

### Current Issues

1. **Price Source Mismatch:**
   - Polymarket uses Chainlink for settlement: $68,598.02
   - Bot uses Binance for price_to_beat: $68,928.02
   - **Discrepancy: $330.00 (0.48%)**

2. **Timing Imprecision:**
   - Bot analyzes markets 2-3 minutes late (18:02:50 for 18:00:00 market)
   - Falls back to current price instead of exact market start price

3. **Inconsistent Hierarchy:**
   - Live prices: Chainlink → CoinGecko → Binance ✓
   - Historical prices: Binance only ✗

### Root Cause

File: `polymarket/performance/settlement_validator.py`, Line 51:
```python
price = await self._fetch_binance_at_timestamp(timestamp)
```

Historical lookups bypass Chainlink buffer and go straight to Binance.

## Design Solution

### Architecture Overview

**Unified 3-Tier Hierarchy for ALL price operations:**

**Tier 1: Chainlink (Primary)**
- Live prices: Polymarket RTDS WebSocket
- Historical prices: 24-hour price buffer (2880 entries)
- Source: Decentralized oracle, matches Polymarket settlement

**Tier 2: CoinGecko (Secondary)**
- Fallback when Chainlink unavailable
- Historical API for timestamps
- Reliable with slight delays

**Tier 3: Binance (Last Resort)**
- Only when both Chainlink and CoinGecko fail
- Backward compatibility
- Warning logged when used

### Component Changes

#### 1. Settlement Validator (`settlement_validator.py`)

**Current:**
```python
async def get_validated_price(self, timestamp: int) -> Optional[Decimal]:
    price = await self._fetch_binance_at_timestamp(timestamp)  # ✗ Wrong
```

**New:**
```python
async def get_validated_price(self, timestamp: int) -> Optional[Decimal]:
    # Tier 1: Chainlink buffer
    price = await self._fetch_chainlink_from_buffer(timestamp)
    if price:
        return price

    # Tier 2: CoinGecko API
    price = await self._fetch_coingecko_at_timestamp(timestamp)
    if price:
        return price

    # Tier 3: Binance API (last resort)
    price = await self._fetch_binance_at_timestamp(timestamp)
    if price:
        logger.warning("Using Binance for historical price (fallback)")
        return price

    # All sources failed
    logger.error("Failed to fetch historical price from all sources")
    return None
```

**New Method:**
```python
async def _fetch_chainlink_from_buffer(self, timestamp: int) -> Optional[Decimal]:
    """Fetch Chainlink price from buffer with ±30s tolerance."""
    if self._btc_service and self._btc_service._stream:
        buffer = self._btc_service._stream._buffer
        if buffer:
            price_data = await buffer.get_price_at(timestamp, tolerance=30)
            if price_data:
                logger.info(
                    "Historical price from Chainlink buffer",
                    timestamp=timestamp,
                    price=f"${price_data.price:,.2f}",
                    source="chainlink"
                )
                return price_data.price
    return None
```

#### 2. BTC Price Service (`btc_price.py`)

**Add method:**
```python
async def _fetch_chainlink_at_timestamp(self, timestamp: int) -> Optional[Decimal]:
    """
    Fetch Chainlink historical price from buffer.

    Args:
        timestamp: Unix timestamp (seconds)

    Returns:
        Chainlink price from buffer, or None if not available
    """
    if self._stream and self._stream._buffer:
        price_data = await self._stream._buffer.get_price_at(
            timestamp,
            tolerance=30  # ±30s window for buffer lookup
        )
        if price_data and price_data.source == "chainlink":
            return price_data.price
    return None
```

#### 3. Price Buffer (`price_history_buffer.py`)

**Update `get_price_at()` to accept tolerance parameter:**
```python
async def get_price_at(
    self,
    timestamp: int,
    tolerance: int = 30
) -> Optional[BTCPriceData]:
    """
    Get price at specific timestamp with tolerance window.

    Args:
        timestamp: Unix timestamp
        tolerance: Seconds tolerance (default ±30s)
    """
    # Find closest price within tolerance window
    # Implementation already exists, just expose tolerance parameter
```

### Timing Precision

**Market Start Times:**
- Markets start at exact 15-minute intervals: 00, 15, 30, 45
- Example: `btc-updown-15m-1771178400` → 18:00:00 UTC
- Buffer stores prices every ~30 seconds

**Timestamp Matching:**
1. Parse exact timestamp from market slug
2. Query buffer with ±30s tolerance: `buffer.get_price_at(1771178400, tolerance=30)`
3. Returns closest Chainlink price to market start time
4. **Never** fall back to current price for historical lookups

**Critical Rule:** If all sources fail to provide historical price, return None and skip the trade. Do NOT use current price as fallback.

### Error Handling & Fallback Logic

```
┌─────────────────────────────────────────────────────────┐
│           Historical Price Fallback Chain               │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. Try Chainlink Buffer (±30s tolerance)               │
│     ✓ Success → Return price, log "source=chainlink"    │
│     ✗ Fail → Log reason, proceed to step 2              │
│                                                          │
│  2. Try CoinGecko Historical API                        │
│     ✓ Success → Return price, log "source=coingecko"    │
│     ✗ Fail → Log error, proceed to step 3               │
│                                                          │
│  3. Try Binance Historical API                          │
│     ✓ Success → Return price, log "source=binance"      │
│                 + WARNING (fallback used)                │
│     ✗ Fail → Proceed to step 4                          │
│                                                          │
│  4. All Sources Failed                                  │
│     → Log CRITICAL ERROR with all failures               │
│     → Return None                                        │
│     → Caller skips trade (no current price fallback)    │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Logging Requirements:**
- Always log which source was used
- Log fallback reasons (buffer_miss, api_error, timeout)
- Track fallback frequency for monitoring
- Alert if Binance used >5% of time

### Testing Strategy

#### Unit Tests

1. **`test_chainlink_buffer_lookup.py`**
   - Exact timestamp match
   - Tolerance matching (±30s)
   - Buffer miss returns None

2. **`test_price_source_fallback.py`**
   - Buffer fail → CoinGecko called
   - CoinGecko fail → Binance called
   - All fail → None returned

3. **`test_source_attribution.py`**
   - Verify source tags on all prices
   - Verify fallback reasons logged

#### Integration Tests

1. **`test_price_to_beat_accuracy.py`**
   ```python
   def test_historical_market_price_accuracy():
       """Verify price_to_beat matches Polymarket settlement."""
       # Market: btc-updown-15m-1771178400
       polymarket_price = Decimal("68598.02")  # Chainlink settlement
       our_price = get_price_to_beat("btc-updown-15m-1771178400")

       diff = abs(polymarket_price - our_price)
       assert diff < Decimal("10.00")  # <$10 vs previous $330 error
   ```

2. **`test_exact_timestamp_matching.py`**
   - Verify price fetched at exact market start (18:00:00)
   - Not 2 minutes late (18:02:50)

#### End-to-End Test

1. Start bot with Chainlink enabled
2. Wait for new market discovery
3. Verify price_to_beat uses Chainlink source
4. Check logs for exact timestamp matching
5. Monitor Telegram alerts with source attribution

### Success Criteria

1. **Price Accuracy:** Discrepancy < $10 (vs previous $330)
2. **Source Priority:** Chainlink used >95% of time
3. **Timing:** Price fetched at exact market start (±30s)
4. **Fallback:** CoinGecko/Binance used <5% combined
5. **No Current Price Fallback:** Historical lookups never use current price

### Rollout Plan

**Phase 1: Implementation**
- Modify settlement_validator.py
- Add buffer lookup methods
- Update logging

**Phase 2: Testing**
- Run unit tests
- Run integration tests
- Verify with historical market

**Phase 3: Deployment**
- Deploy to production
- Monitor first 24 hours
- Track source distribution
- Verify price accuracy

**Phase 4: Validation**
- Compare price_to_beat with Polymarket for 10 markets
- Assert all differences <$10
- Document any fallback usage

### Risk Mitigation

**Risk 1: Buffer doesn't have timestamp**
- Mitigation: ±30s tolerance window
- Fallback: CoinGecko historical API

**Risk 2: All sources fail**
- Mitigation: Return None, skip trade
- Never use current price as fallback

**Risk 3: Chainlink data quality**
- Mitigation: Monitor source distribution
- Alert if Binance used >5%

## Files Modified

1. `polymarket/performance/settlement_validator.py` - Fallback chain
2. `polymarket/trading/btc_price.py` - Chainlink buffer access
3. `polymarket/trading/price_history_buffer.py` - Tolerance parameter
4. `scripts/auto_trade.py` - Verify no current price fallback
5. Tests: 5 new test files

## Expected Impact

- **Price Accuracy:** 99.7% improvement ($330 → <$10)
- **Settlement Alignment:** 100% match with Polymarket
- **Source Distribution:** Chainlink >95%, CoinGecko <4%, Binance <1%
- **Timing Precision:** Exact market start timestamp matching

---

**Approved by:** User
**Next Step:** Implementation planning with superpowers:writing-plans

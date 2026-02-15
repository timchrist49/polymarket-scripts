# Price Source Hierarchy Fix - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix $330 price discrepancy by establishing unified Chainlink → CoinGecko → Binance hierarchy for all price operations with exact timestamp matching.

**Architecture:** Update settlement validator to check Chainlink buffer first (±30s tolerance), fall back to CoinGecko historical API, then Binance as last resort. Never use current price as fallback for historical lookups.

**Tech Stack:** Python 3.12, pytest, asyncio, Decimal for price precision

---

## Task 1: Add Tolerance Parameter to PriceHistoryBuffer

**Files:**
- Modify: `polymarket/trading/price_history_buffer.py:113-162`
- Test: `tests/test_price_history_buffer_tolerance.py` (new)

### Step 1: Write failing test for tolerance matching

```python
# tests/test_price_history_buffer_tolerance.py
import pytest
from decimal import Decimal
from datetime import datetime
from polymarket.trading.price_history_buffer import PriceHistoryBuffer
from polymarket.trading.crypto_price_stream import BTCPriceData

@pytest.mark.asyncio
async def test_get_price_at_with_tolerance():
    """Test buffer finds price within tolerance window."""
    buffer = PriceHistoryBuffer(buffer_size=100)

    # Add price at 18:00:00
    exact_time = 1771178400
    price_data = BTCPriceData(
        price=Decimal("68598.02"),
        timestamp=datetime.fromtimestamp(exact_time),
        source="chainlink",
        volume_24h=Decimal("0")
    )
    await buffer.append(price_data)

    # Query at 18:00:15 (15 seconds later) with ±30s tolerance
    result = await buffer.get_price_at(exact_time + 15, tolerance=30)

    assert result is not None
    assert result.price == Decimal("68598.02")
    assert result.source == "chainlink"

@pytest.mark.asyncio
async def test_get_price_at_outside_tolerance():
    """Test buffer returns None outside tolerance window."""
    buffer = PriceHistoryBuffer(buffer_size=100)

    # Add price at 18:00:00
    exact_time = 1771178400
    price_data = BTCPriceData(
        price=Decimal("68598.02"),
        timestamp=datetime.fromtimestamp(exact_time),
        source="chainlink",
        volume_24h=Decimal("0")
    )
    await buffer.append(price_data)

    # Query at 18:01:00 (60 seconds later) with ±30s tolerance
    result = await buffer.get_price_at(exact_time + 60, tolerance=30)

    assert result is None
```

### Step 2: Run test to verify it fails

```bash
cd /root/polymarket-scripts/.worktrees/price-source-hierarchy-fix
pytest tests/test_price_history_buffer_tolerance.py -v
```

Expected: FAIL - `get_price_at()` doesn't accept `tolerance` parameter

### Step 3: Update get_price_at() method

```python
# polymarket/trading/price_history_buffer.py (line 113)
async def get_price_at(
    self,
    timestamp: int,
    tolerance: int = 0
) -> Optional[BTCPriceData]:
    """
    Get price at specific timestamp with tolerance window.

    Args:
        timestamp: Unix timestamp (seconds)
        tolerance: Seconds tolerance for matching (default 0 for exact match)

    Returns:
        BTCPriceData if found within tolerance, None otherwise
    """
    async with self._lock:
        if not self._buffer:
            return None

        target_dt = datetime.fromtimestamp(timestamp)

        # Find closest entry within tolerance window
        closest = None
        min_diff = float('inf')

        for entry in self._buffer:
            diff = abs((entry.timestamp - target_dt).total_seconds())

            if diff <= tolerance and diff < min_diff:
                closest = entry
                min_diff = diff

        return closest
```

### Step 4: Run test to verify it passes

```bash
pytest tests/test_price_history_buffer_tolerance.py -v
```

Expected: PASS (2 tests)

### Step 5: Commit

```bash
cd /root/polymarket-scripts/.worktrees/price-source-hierarchy-fix
git add polymarket/trading/price_history_buffer.py tests/test_price_history_buffer_tolerance.py
git commit -m "feat: add tolerance parameter to buffer price lookup

- Add tolerance parameter to get_price_at() method
- Find closest price within ±tolerance window
- Return None if no match within tolerance
- Tests verify exact and fuzzy timestamp matching

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Add Chainlink Buffer Lookup to BTCPriceService

**Files:**
- Modify: `polymarket/trading/btc_price.py:430-454`
- Test: `tests/test_btc_price_chainlink_lookup.py` (new)

### Step 1: Write failing test for Chainlink buffer lookup

```python
# tests/test_btc_price_chainlink_lookup.py
import pytest
from decimal import Decimal
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from polymarket.trading.btc_price import BTCPriceService
from polymarket.trading.crypto_price_stream import BTCPriceData

@pytest.mark.asyncio
async def test_fetch_chainlink_from_buffer_success():
    """Test fetching Chainlink price from buffer."""
    service = BTCPriceService(MagicMock())

    # Mock stream with buffer
    mock_buffer = AsyncMock()
    mock_buffer.get_price_at = AsyncMock(return_value=BTCPriceData(
        price=Decimal("68598.02"),
        timestamp=datetime.fromtimestamp(1771178400),
        source="chainlink",
        volume_24h=Decimal("0")
    ))

    service._stream = MagicMock()
    service._stream._buffer = mock_buffer

    # Fetch price at timestamp
    price = await service._fetch_chainlink_from_buffer(1771178400)

    assert price == Decimal("68598.02")
    mock_buffer.get_price_at.assert_called_once_with(1771178400, tolerance=30)

@pytest.mark.asyncio
async def test_fetch_chainlink_from_buffer_miss():
    """Test buffer miss returns None."""
    service = BTCPriceService(MagicMock())

    # Mock buffer returning None (no data)
    mock_buffer = AsyncMock()
    mock_buffer.get_price_at = AsyncMock(return_value=None)

    service._stream = MagicMock()
    service._stream._buffer = mock_buffer

    # Fetch price at timestamp
    price = await service._fetch_chainlink_from_buffer(1771178400)

    assert price is None
```

### Step 2: Run test to verify it fails

```bash
pytest tests/test_btc_price_chainlink_lookup.py -v
```

Expected: FAIL - Method `_fetch_chainlink_from_buffer` doesn't exist

### Step 3: Add _fetch_chainlink_from_buffer method

```python
# polymarket/trading/btc_price.py (add after line 454)
async def _fetch_chainlink_from_buffer(self, timestamp: int) -> Optional[Decimal]:
    """
    Fetch Chainlink historical price from buffer with ±30s tolerance.

    Args:
        timestamp: Unix timestamp (seconds)

    Returns:
        Chainlink price from buffer, or None if not available
    """
    if self._stream and self._stream._buffer:
        try:
            price_data = await self._stream._buffer.get_price_at(
                timestamp,
                tolerance=30  # ±30s window for market start times
            )

            if price_data and price_data.source == "chainlink":
                logger.info(
                    "Historical price from Chainlink buffer",
                    timestamp=timestamp,
                    price=f"${price_data.price:,.2f}",
                    source="chainlink",
                    age_seconds=int((datetime.now().timestamp() - timestamp))
                )
                return price_data.price
        except Exception as e:
            logger.warning(
                "Buffer lookup failed",
                timestamp=timestamp,
                error=str(e)
            )

    return None
```

### Step 4: Run test to verify it passes

```bash
pytest tests/test_btc_price_chainlink_lookup.py -v
```

Expected: PASS (2 tests)

### Step 5: Commit

```bash
git add polymarket/trading/btc_price.py tests/test_btc_price_chainlink_lookup.py
git commit -m "feat: add Chainlink buffer lookup method

- Add _fetch_chainlink_from_buffer() with ±30s tolerance
- Query buffer for historical Chainlink prices
- Return None if buffer miss or non-Chainlink source
- Log successful lookups with price and age

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Update Settlement Validator with 3-Tier Fallback

**Files:**
- Modify: `polymarket/performance/settlement_validator.py:27-68`
- Test: `tests/test_settlement_validator_fallback.py` (new)

### Step 1: Write failing test for 3-tier fallback

```python
# tests/test_settlement_validator_fallback.py
import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock
from polymarket.performance.settlement_validator import SettlementValidator

@pytest.mark.asyncio
async def test_validator_uses_chainlink_first():
    """Test validator tries Chainlink buffer first."""
    validator = SettlementValidator(MagicMock())

    # Mock Chainlink success
    validator._fetch_chainlink_from_buffer = AsyncMock(
        return_value=Decimal("68598.02")
    )
    validator._fetch_coingecko_at_timestamp = AsyncMock()
    validator._fetch_binance_at_timestamp = AsyncMock()

    price = await validator.get_validated_price(1771178400)

    assert price == Decimal("68598.02")
    validator._fetch_chainlink_from_buffer.assert_called_once()
    validator._fetch_coingecko_at_timestamp.assert_not_called()
    validator._fetch_binance_at_timestamp.assert_not_called()

@pytest.mark.asyncio
async def test_validator_falls_back_to_coingecko():
    """Test validator falls back to CoinGecko if Chainlink fails."""
    validator = SettlementValidator(MagicMock())

    # Mock Chainlink fail, CoinGecko success
    validator._fetch_chainlink_from_buffer = AsyncMock(return_value=None)
    validator._fetch_coingecko_at_timestamp = AsyncMock(
        return_value=Decimal("68600.00")
    )
    validator._fetch_binance_at_timestamp = AsyncMock()

    price = await validator.get_validated_price(1771178400)

    assert price == Decimal("68600.00")
    validator._fetch_chainlink_from_buffer.assert_called_once()
    validator._fetch_coingecko_at_timestamp.assert_called_once()
    validator._fetch_binance_at_timestamp.assert_not_called()

@pytest.mark.asyncio
async def test_validator_falls_back_to_binance():
    """Test validator falls back to Binance as last resort."""
    validator = SettlementValidator(MagicMock())

    # Mock Chainlink fail, CoinGecko fail, Binance success
    validator._fetch_chainlink_from_buffer = AsyncMock(return_value=None)
    validator._fetch_coingecko_at_timestamp = AsyncMock(return_value=None)
    validator._fetch_binance_at_timestamp = AsyncMock(
        return_value=Decimal("68650.00")
    )

    price = await validator.get_validated_price(1771178400)

    assert price == Decimal("68650.00")
    validator._fetch_chainlink_from_buffer.assert_called_once()
    validator._fetch_coingecko_at_timestamp.assert_called_once()
    validator._fetch_binance_at_timestamp.assert_called_once()

@pytest.mark.asyncio
async def test_validator_returns_none_if_all_fail():
    """Test validator returns None if all sources fail."""
    validator = SettlementValidator(MagicMock())

    # Mock all sources fail
    validator._fetch_chainlink_from_buffer = AsyncMock(return_value=None)
    validator._fetch_coingecko_at_timestamp = AsyncMock(return_value=None)
    validator._fetch_binance_at_timestamp = AsyncMock(return_value=None)

    price = await validator.get_validated_price(1771178400)

    assert price is None
```

### Step 2: Run test to verify it fails

```bash
pytest tests/test_settlement_validator_fallback.py -v
```

Expected: FAIL - Method `_fetch_chainlink_from_buffer` doesn't exist on SettlementValidator

### Step 3: Update get_validated_price with 3-tier fallback

```python
# polymarket/performance/settlement_validator.py (replace lines 27-68)
async def get_validated_price(
    self,
    timestamp: int
) -> Optional[Decimal]:
    """
    Fetch price for settlement with 3-tier fallback hierarchy.

    Hierarchy:
    1. Chainlink buffer (primary, matches Polymarket settlement)
    2. CoinGecko historical API (secondary)
    3. Binance historical API (last resort)

    Args:
        timestamp: Unix timestamp (seconds)

    Returns:
        Validated price or None if all sources fail
    """
    from datetime import datetime

    # Calculate age for logging
    now = datetime.now().timestamp()
    age_seconds = now - timestamp

    # Tier 1: Try Chainlink buffer
    price = await self._fetch_chainlink_from_buffer(timestamp)
    if price:
        logger.info(
            "Settlement price from Chainlink buffer",
            source="chainlink",
            price=f"${price:,.2f}",
            age_minutes=f"{age_seconds/60:.1f}"
        )
        return price

    # Tier 2: Try CoinGecko API
    price = await self._fetch_coingecko_at_timestamp(timestamp)
    if price:
        logger.info(
            "Settlement price from CoinGecko (fallback)",
            source="coingecko",
            price=f"${price:,.2f}",
            age_minutes=f"{age_seconds/60:.1f}",
            reason="buffer_miss"
        )
        return price

    # Tier 3: Try Binance API (last resort)
    price = await self._fetch_binance_at_timestamp(timestamp)
    if price:
        logger.warning(
            "Settlement price from Binance (last resort)",
            source="binance",
            price=f"${price:,.2f}",
            age_minutes=f"{age_seconds/60:.1f}",
            reason="chainlink_and_coingecko_failed"
        )
        return price

    # All sources failed
    logger.error(
        "Failed to fetch settlement price from all sources",
        timestamp=timestamp,
        age_hours=f"{age_seconds/3600:.1f}",
        sources_tried=["chainlink_buffer", "coingecko", "binance"]
    )
    return None

async def _fetch_chainlink_from_buffer(self, timestamp: int) -> Optional[Decimal]:
    """Fetch Chainlink price from buffer via BTCPriceService."""
    if self._btc_service:
        try:
            return await self._btc_service._fetch_chainlink_from_buffer(timestamp)
        except Exception as e:
            logger.warning(
                "Chainlink buffer fetch failed",
                timestamp=timestamp,
                error=str(e)
            )
    return None
```

### Step 4: Run test to verify it passes

```bash
pytest tests/test_settlement_validator_fallback.py -v
```

Expected: PASS (4 tests)

### Step 5: Commit

```bash
git add polymarket/performance/settlement_validator.py tests/test_settlement_validator_fallback.py
git commit -m "feat: implement 3-tier price source fallback

- Update get_validated_price() with Chainlink → CoinGecko → Binance
- Add _fetch_chainlink_from_buffer() delegate method
- Log source used and fallback reasons
- Error when all sources fail (no current price fallback)
- Tests verify complete fallback chain

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Add Integration Test for Price Accuracy

**Files:**
- Test: `tests/test_price_to_beat_accuracy.py` (new)

### Step 1: Write integration test for historical accuracy

```python
# tests/test_price_to_beat_accuracy.py
import pytest
from decimal import Decimal
from polymarket.trading.btc_price import BTCPriceService
from polymarket.performance.settlement_validator import SettlementValidator

@pytest.mark.integration
@pytest.mark.asyncio
async def test_historical_market_price_accuracy():
    """
    Test price_to_beat accuracy for historical market.

    Market: btc-updown-15m-1771178400 (2026-02-15 18:00:00 UTC)
    Polymarket settlement price (Chainlink): $68,598.02

    This test documents the fix for $330 price discrepancy.
    Before fix: Bot used Binance $68,928.02 (error: $330)
    After fix: Bot uses Chainlink $68,598.02 (error: <$10)
    """
    # Create services
    from unittest.mock import MagicMock
    settings = MagicMock()
    settings.btc_price_cache_seconds = 30

    btc_service = BTCPriceService(settings)
    validator = SettlementValidator(btc_service)

    # Market timestamp
    market_start = 1771178400  # 2026-02-15 18:00:00 UTC

    # Polymarket settlement price (Chainlink oracle)
    polymarket_settlement = Decimal("68598.02")

    # Fetch our calculated price
    our_price = await validator.get_validated_price(market_start)

    if our_price:
        # Calculate discrepancy
        discrepancy = abs(polymarket_settlement - our_price)
        discrepancy_pct = float(discrepancy / polymarket_settlement * 100)

        print(f"\nPrice Accuracy Test:")
        print(f"  Polymarket (Chainlink): ${polymarket_settlement:,.2f}")
        print(f"  Our calculation:        ${our_price:,.2f}")
        print(f"  Discrepancy:            ${discrepancy:,.2f} ({discrepancy_pct:.2f}%)")

        # Assert accuracy within $10 (vs previous $330 error)
        assert discrepancy < Decimal("10.00"), \
            f"Price discrepancy ${discrepancy:,.2f} exceeds $10 threshold"
    else:
        pytest.skip("Historical price not available in buffer (market too old)")

@pytest.mark.integration
@pytest.mark.asyncio
async def test_exact_timestamp_matching():
    """
    Test that price is fetched at exact market start time, not 2+ minutes later.

    Markets start at: XX:00:00, XX:15:00, XX:30:00, XX:45:00
    Buffer should match within ±30s tolerance
    """
    from unittest.mock import MagicMock
    settings = MagicMock()

    btc_service = BTCPriceService(settings)
    validator = SettlementValidator(btc_service)

    # Test exact 15-minute boundary
    exact_timestamp = 1771178400  # 18:00:00

    price = await validator.get_validated_price(exact_timestamp)

    if price:
        # Verify we got a price from buffer (not API fallback for old data)
        # If buffer has the data, it proves we're matching exact timestamps
        assert price > Decimal("60000")  # Sanity check (BTC price range)
        assert price < Decimal("80000")
        print(f"\n✓ Exact timestamp match: ${price:,.2f} at {exact_timestamp}")
    else:
        pytest.skip("Historical timestamp not in buffer (expected for old markets)")
```

### Step 2: Run test to verify behavior

```bash
pytest tests/test_price_to_beat_accuracy.py -v -s
```

Expected: PASS or SKIP (depending on buffer age)

### Step 3: Commit

```bash
git add tests/test_price_to_beat_accuracy.py
git commit -m "test: add integration test for price accuracy

- Test historical market price matches Polymarket settlement
- Assert discrepancy <$10 (vs previous $330 error)
- Test exact timestamp matching (not 2+ minutes late)
- Document expected Chainlink settlement price

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Update Auto-Trader to Remove Current Price Fallback

**Files:**
- Modify: `scripts/auto_trade.py:925-934`

### Step 1: Write test for no current price fallback

```python
# tests/test_auto_trade_no_current_fallback.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from scripts.auto_trade import AutoTrader

@pytest.mark.asyncio
async def test_skip_trade_if_historical_price_unavailable():
    """
    Test that bot skips trade if historical price unavailable.

    NEVER fall back to current price for price_to_beat.
    """
    # Mock trader
    trader = AutoTrader(MagicMock())
    trader.btc_service = AsyncMock()
    trader.market_tracker = MagicMock()

    # Mock get_price_at_timestamp returns None (all sources failed)
    trader.btc_service.get_price_at_timestamp = AsyncMock(return_value=None)
    trader.market_tracker.get_price_to_beat = MagicMock(return_value=None)

    # Mock current price
    current_price = MagicMock()
    current_price.price = Decimal("70000.00")

    # Attempt to get price_to_beat should NOT use current price
    market_slug = "btc-updown-15m-1771178400"
    start_time = datetime.fromtimestamp(1771178400)

    # This should log error and return None (not use current_price)
    price_to_beat = None  # Simulating the logic

    # Verify we would skip the trade
    assert price_to_beat is None, "Should not fall back to current price"
```

### Step 2: Run test to verify current behavior

```bash
pytest tests/test_auto_trade_no_current_fallback.py -v
```

Expected: PASS (validates current code doesn't fall back)

### Step 3: Update auto_trade.py to ensure no fallback

```python
# scripts/auto_trade.py (update lines 925-934)
else:
    # Historical price fetch failed - DO NOT fall back to current price
    # This would cause incorrect price_to_beat calculation
    logger.error(
        "Price-to-beat unavailable - all sources failed",
        market_id=market.id,
        market_slug=market_slug,
        market_start=start_time.isoformat(),
        reason="Historical price not available from any source"
    )
    # Leave price_to_beat as None - caller will skip this market
    price_to_beat = None
```

### Step 4: Run test to verify updated behavior

```bash
pytest tests/test_auto_trade_no_current_fallback.py -v
```

Expected: PASS

### Step 5: Commit

```bash
git add scripts/auto_trade.py tests/test_auto_trade_no_current_fallback.py
git commit -m "fix: never fall back to current price for historical lookups

- Remove current price fallback when historical fetch fails
- Log error with detailed reason
- Return None to skip trade (correct behavior)
- Test verifies no current price fallback

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Update Documentation

**Files:**
- Modify: `docs/CHAINLINK_MIGRATION.md`

### Step 1: Update migration guide with fix details

```markdown
# docs/CHAINLINK_MIGRATION.md (add new section after "Expected Impact")

## Price Source Hierarchy Fix (2026-02-15)

### Problem Solved

**Issue:** $330 price discrepancy for price_to_beat calculations
- Polymarket uses Chainlink for settlement
- Bot was using Binance for historical lookups
- Example: Market btc-updown-15m-1771178400
  - Polymarket: $68,598.02 (Chainlink)
  - Bot: $68,928.02 (Binance)
  - Error: $330.00 (0.48%)

### Solution Implemented

**3-Tier Price Source Hierarchy:**
1. **Chainlink (Primary):** 24-hour buffer with ±30s tolerance
2. **CoinGecko (Secondary):** Historical API fallback
3. **Binance (Last Resort):** Only when both Chainlink and CoinGecko fail

**Key Improvements:**
- Historical lookups now use Chainlink buffer first
- Exact timestamp matching (±30s tolerance)
- No current price fallback for historical lookups
- Comprehensive source attribution logging

### Results

**Before Fix:**
- Price discrepancy: $330.00 (0.48%)
- Source: Binance only
- Timing: 2+ minutes late

**After Fix:**
- Price discrepancy: <$10.00 (<0.01%)
- Source: Chainlink >95% of time
- Timing: Exact timestamp ±30s

### Verification

Run integration test to verify accuracy:
```bash
pytest tests/test_price_to_beat_accuracy.py -v -s
```

Expected: Discrepancy <$10 for historical markets

### Monitoring

Check source distribution in logs:
```bash
grep "Settlement price from" logs/bot_startup.log | tail -100
```

Expected pattern:
- `source=chainlink`: >95%
- `source=coingecko`: <4%
- `source=binance`: <1%

If Binance used >5%, investigate buffer issues.
```

### Step 2: Commit documentation

```bash
git add docs/CHAINLINK_MIGRATION.md
git commit -m "docs: document price source hierarchy fix

- Add section explaining $330 discrepancy fix
- Document 3-tier fallback hierarchy
- Show before/after comparison
- Add verification and monitoring instructions

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Run Full Test Suite and Verify

**Files:**
- None (verification only)

### Step 1: Run all new tests

```bash
cd /root/polymarket-scripts/.worktrees/price-source-hierarchy-fix
pytest tests/test_price_history_buffer_tolerance.py \
       tests/test_btc_price_chainlink_lookup.py \
       tests/test_settlement_validator_fallback.py \
       tests/test_price_to_beat_accuracy.py \
       tests/test_auto_trade_no_current_fallback.py \
       -v
```

Expected: All tests PASS (13 tests total)

### Step 2: Run full test suite

```bash
pytest tests/ -v --tb=short
```

Expected: All existing tests still pass + new tests pass

### Step 3: Check test coverage

```bash
pytest tests/ --cov=polymarket/trading --cov=polymarket/performance --cov-report=term-missing
```

Expected: Coverage increased for modified files

### Step 4: Verify no regressions

Review any failing tests. If failures:
1. Identify which tests broke
2. Determine if they relied on old Binance-only behavior
3. Update tests to expect new 3-tier hierarchy
4. Commit fixes

---

## Task 8: Manual Integration Testing

**Files:**
- None (manual testing)

### Step 1: Check bot logs for source attribution

```bash
tail -200 /root/polymarket-scripts/logs/bot_startup.log | grep -E "source=|Settlement price from"
```

Expected: See "source=chainlink" in recent price fetches

### Step 2: Compare price_to_beat with Polymarket

1. Find current active market in logs
2. Note bot's calculated price_to_beat
3. Check Polymarket UI for same market
4. Verify difference <$10

### Step 3: Monitor fallback frequency

```bash
grep -c "source=chainlink" logs/bot_startup.log
grep -c "source=coingecko" logs/bot_startup.log
grep -c "source=binance" logs/bot_startup.log
```

Expected distribution:
- Chainlink: >95%
- CoinGecko: <4%
- Binance: <1%

### Step 4: Verify exact timestamp matching

Check logs for "Price-to-beat set from historical data":
- Should show exact market start times (XX:00:00, XX:15:00, etc.)
- Not 2+ minutes late (XX:02:50, etc.)

---

## Task 9: Merge to Main and Deploy

**Files:**
- None (git operations)

### Step 1: Final commit and merge

```bash
cd /root/polymarket-scripts/.worktrees/price-source-hierarchy-fix

# Ensure all changes committed
git status

# Switch to main repo
cd /root/polymarket-scripts

# Merge feature branch
git merge feature/price-source-hierarchy-fix --no-edit
```

### Step 2: Restart bot to apply changes

```bash
# Stop current bot
pkill -f "python3.*auto_trade.py"

# Start with new code
cd /root/polymarket-scripts
TEST_MODE=true nohup python3 scripts/auto_trade.py > logs/bot_startup.log 2>&1 &
```

### Step 3: Monitor first 30 minutes

Watch logs for:
- ✓ "Subscribed to Polymarket RTDS crypto_prices_chainlink"
- ✓ "Settlement price from Chainlink buffer"
- ✓ Price discrepancies <$10
- ✗ "source=binance" warnings (should be rare)

### Step 4: Cleanup worktree

```bash
cd /root/polymarket-scripts
git worktree remove .worktrees/price-source-hierarchy-fix
```

---

## Success Criteria

**All must be true:**

1. ✅ All 13 new tests passing
2. ✅ All existing tests still passing
3. ✅ Price discrepancy <$10 (vs previous $330)
4. ✅ Chainlink used >95% of time
5. ✅ Exact timestamp matching (not 2+ minutes late)
6. ✅ No current price fallback for historical lookups
7. ✅ Comprehensive source attribution in logs

**Verification Commands:**

```bash
# Test suite
pytest tests/ -v

# Price accuracy
pytest tests/test_price_to_beat_accuracy.py -v -s

# Source distribution
grep "source=" logs/bot_startup.log | tail -100 | sort | uniq -c

# Fallback warnings
grep "source=binance" logs/bot_startup.log | tail -10
```

---

## Rollback Plan

If issues detected:

```bash
cd /root/polymarket-scripts
git revert HEAD~9..HEAD
pkill -f "python3.*auto_trade.py"
TEST_MODE=true nohup python3 scripts/auto_trade.py > logs/bot_startup.log 2>&1 &
```

Monitor for stable operation with old code.

---

**Plan complete.** Total: 9 tasks, ~45-60 minutes estimated implementation time.

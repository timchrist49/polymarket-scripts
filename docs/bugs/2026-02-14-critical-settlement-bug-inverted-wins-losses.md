# CRITICAL BUG: Settlement Inverting Wins and Losses

**Date Discovered:** 2026-02-14
**Severity:** 🔴 CRITICAL - Bot losing real money
**Status:** ROOT CAUSE IDENTIFIED

---

## Symptom

- Bot logs show WINS
- Polymarket dashboard shows LOSSES
- User is losing real money
- Trades not properly settled

---

## Root Cause

**File:** `polymarket/trading/market_tracker.py:31-52`

The `parse_market_start()` method has a **CRITICAL error** in its documentation and logic:

```python
def parse_market_start(self, slug: str) -> Optional[datetime]:
    """
    Parse market start timestamp from slug.
    ...
    The timestamp represents the market START time (when trading opens).  # ❌ FALSE!
    """
    timestamp_str = parts[-1]  # Last part is epoch (market START time)  # ❌ FALSE!
    timestamp = int(timestamp_str)
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)  # ❌ Returns END time!
```

**The timestamp in the market slug is the CLOSE/END time, NOT the start time!**

###Example

Market slug: `btc-updown-15m-1771051500`
- Timestamp 1771051500 = 2026-02-14 06:45:00 (**END** of 15-min window)
- Actual START should be: 1771050600 = 2026-02-14 06:30:00 (15 min earlier)

---

## Data Flow Bug

### Current (BROKEN) Flow:

```
1. auto_trade.py calls parse_market_start("btc-updown-15m-1771051500")
   ↓
2. Returns datetime(06:45:00)  # END time, but code thinks it's START
   ↓
3. Fetches BTC price at 06:45:00 → $68,849.03
   ↓
4. Stores as price_to_beat: $68,849.03  # WRONG - This is the END price!
   ↓
5. Settlement:
   - Fetches btc_close_price at 06:45:00 → $68,849.03
   - Compares: $68,849.03 > $68,849.03 ? NO
   - Outcome: "NO" (DOWN won)
   ↓
6. ❌ WRONG OUTCOME! Actual BTC went UP from $68,847.89 to $68,849.03
```

### Correct Flow Should Be:

```
1. Parse market slug to get CLOSE time: 06:45:00
   ↓
2. Calculate START time: 06:45:00 - 15 min = 06:30:00
   ↓
3. Fetch BTC price at START (06:30:00) → $68,847.89
   ↓
4. Store as price_to_beat: $68,847.89  # CORRECT!
   ↓
5. Settlement:
   - Fetch btc_close_price at 06:45:00 → $68,849.03
   - Compare: $68,849.03 > $68,847.89 ? YES
   - Outcome: "YES" (UP won)
   ↓
6. ✓ CORRECT OUTCOME!
```

---

## Evidence

### Trade #228 Analysis

```
Market: btc-updown-15m-1771051500
Window: 06:30:00 to 06:45:00

Actual BTC Prices:
  START (06:30:00): $68,847.89
  END   (06:45:00): $68,849.03
  Movement: +$1.14 (UP ↑)

Database Values:
  price_to_beat: $68,849.03  # ❌ WRONG - This is END price!
  actual_outcome: NO         # ❌ WRONG - Should be YES!
  is_win: TRUE               # ❌ WRONG for a NO trade!

Reality:
  BTC went UP → YES wins
  Bot traded NO → Should LOSE
  But bot thinks it WON → Losing real money!
```

### Impact on Recent Trades

Out of 10 recent trades:
- 8 trades show as WINS in database
- But checking actual BTC movements:
  - **Most are INVERTED** (claiming NO won when YES actually won)
  - User is losing money on "winning" trades

---

## The Fix

### Option 1: Fix parse_market_start (RECOMMENDED)

Update `market_tracker.py` to correctly return START time:

```python
def parse_market_start(self, slug: str) -> Optional[datetime]:
    """
    Parse market start timestamp from slug.

    Slug format: btc-updown-15m-{close_timestamp}
    The timestamp represents the market CLOSE time.
    We subtract 15 minutes to get the START time.
    """
    try:
        parts = slug.split("-")
        if len(parts) < 4:
            logger.warning("Invalid market slug format", slug=slug)
            return None

        timestamp_str = parts[-1]  # Close timestamp
        close_timestamp = int(timestamp_str)

        # Market duration is 15 minutes (900 seconds)
        start_timestamp = close_timestamp - 900

        return datetime.fromtimestamp(start_timestamp, tz=timezone.utc)
    except (ValueError, IndexError) as e:
        logger.error("Failed to parse market slug", slug=slug, error=str(e))
        return None
```

### Option 2: Fix auto_trade.py

Alternative: Keep parse_market_start as-is (returning close time) and fix auto_trade.py:

```python
# In auto_trade.py around line 826-832
close_time = self.market_tracker.parse_market_start(market_slug)  # Returns close time
if close_time and market_slug:
    # Calculate START time (15 minutes before close)
    start_time = close_time - timedelta(minutes=15)
    start_timestamp = int(start_time.timestamp())
    historical_price = await self.btc_service.get_price_at_timestamp(start_timestamp)
    # ...
```

---

## Additional Issues Found

1. **Order Verification Not Catching This**
   - The recently implemented order verification system checks fill prices
   - But doesn't verify the settlement outcome is correct
   - Need to add outcome verification against actual market resolution

2. **Database Contains Wrong Historical Data**
   - All existing trades have wrong price_to_beat values
   - Need to recalculate outcomes for all settled trades
   - Migration script required

---

## Urgency

**CRITICAL - IMMEDIATE ACTION REQUIRED**

This bug is causing the bot to:
1. Think it's winning when it's losing
2. Lose real money on Polymarket
3. Have completely inverted settlement logic
4. Store wrong data for all historical trades

Every trade since the system started is affected by this bug.

---

## Next Steps

1. **STOP BOT** immediately to prevent further losses
2. Implement fix (Option 1 recommended)
3. Create migration script to fix historical data
4. Verify fix with test cases
5. Recalculate all existing trade outcomes
6. Restart bot with corrected logic

---

## Test Cases Required

```python
def test_market_slug_parsing():
    """Test that market slug correctly identifies START time."""
    tracker = MarketTracker(settings)

    # Market closes at 06:45:00
    slug = "btc-updown-15m-1771051500"
    start_time = tracker.parse_market_start(slug)

    # Should return START time (06:30:00), not close time
    expected = datetime(2026, 2, 14, 6, 30, 0, tzinfo=timezone.utc)
    assert start_time == expected, f"Expected {expected}, got {start_time}"

def test_settlement_outcome():
    """Test that settlement correctly determines outcome."""
    # BTC moves from $68,847.89 to $68,849.03 (UP)
    outcome = settler._determine_outcome(
        btc_close_price=68849.03,
        price_to_beat=68847.89  # START price
    )

    assert outcome == "YES", f"BTC went UP, YES should win, got {outcome}"
```

---

**STATUS:** Awaiting fix implementation
**ASSIGNED:** Immediate priority
**VERIFICATION:** Required before bot restart

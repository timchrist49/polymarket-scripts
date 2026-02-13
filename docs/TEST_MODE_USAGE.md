# Test Mode Usage Guide

## Overview

Test mode forces trading on **every 15-minute BTC market** on Polymarket with strict safety controls to validate:
- New CoinGecko Pro market signals (funding rates, exchange premium, volume)
- AI decision-making with signal integration
- Edge detection and confidence calibration

**Key Features:**
- ✅ **Real money** trading with **$1 strict limit** per bet
- ✅ **70% minimum confidence** threshold
- ✅ **No duplicate** bets on same market
- ✅ **All safety filters bypassed** but data sent to AI
- ✅ **Forced YES/NO decisions** (no HOLD allowed)
- ✅ **Tracked separately** in database for analysis

---

## Activation

### Environment Variable

```bash
export TEST_MODE=true
```

Or inline:

```bash
TEST_MODE=true python scripts/auto_trade.py
```

### Verification

When test mode is active, you'll see:

```
======================================================================
[WARNING] TEST MODE ENABLED
[WARNING] Trading with $1 bets, 70% min confidence, forcing decisions
[WARNING]   max_bet: 1.0
[WARNING]   min_confidence: 0.70
======================================================================
```

---

## How It Works

### 1. Safety Filter Bypasses

All safety checks are **bypassed** but data is **still collected** and sent to AI:

| Filter | Normal Behavior | Test Mode Behavior |
|--------|----------------|-------------------|
| **Movement threshold** | Skip if < $100 | ✅ Bypass, log `[TEST] Bypassing movement threshold` |
| **Spread check** | Skip if > 500 bps | ✅ Bypass, log `[TEST] Bypassing spread check` |
| **Volume confirmation** | Skip low-volume breakouts | ✅ Bypass, log `[TEST] Bypassing volume confirmation` |
| **Timeframe alignment** | Skip conflicting trends | ✅ Bypass, log `[TEST] Bypassing timeframe check` |
| **Market regime** | Skip unclear/volatile | ✅ Bypass, log `[TEST] Bypassing regime check` |

**Why bypass?** To test new signals in all market conditions, even those normally avoided.

### 2. Forced AI Decisions

The AI receives a **critical instruction**:

```
⚠️ TEST MODE ACTIVE - FORCED TRADING
CRITICAL: You MUST return either "YES" or "NO" - HOLD is NOT allowed.
```

**Fallback:** If AI returns HOLD anyway, the system forces direction based on sentiment:
- Positive sentiment → YES
- Negative sentiment → NO

### 3. Confidence Threshold

After forcing a decision, the system checks:

```python
if decision.confidence < 0.70:
    logger.info("[TEST] Skipping trade - confidence below threshold")
    return
```

**Result:** Only trades with ≥70% AI confidence are executed.

### 4. Position Size Override

```python
decision.position_size = Decimal("1.0")  # Always $1
```

**No exceptions.** Regardless of AI suggestions, every trade is exactly $1.

### 5. Duplicate Prevention

```python
if market.id in self.test_mode.traded_markets:
    logger.info("[TEST] Skipping market - already traded in this session")
    return
```

**Tracking:** In-memory set prevents multiple bets on same market until bot restart.

---

## Log Output Examples

### Startup Banner
```
======================================================================
[WARNING] TEST MODE ENABLED
[WARNING] Trading with $1 bets, 70% min confidence, forcing decisions
======================================================================
```

### Bypassed Safety Checks
```
[TEST] Bypassing movement threshold - data sent to AI
  market_id: 0x123abc
  movement: $45.20
  threshold: $100.00
  bypassed: True
```

### Duplicate Prevention
```
[TEST] Skipping market - already traded in this session
  market_id: 0x123abc
  market_question: Will BTC be above $95,500 at 3:45 PM UTC?
  traded_count: 5
```

### Confidence Threshold Check
```
[TEST] Skipping trade - confidence below threshold
  market_id: 0x123abc
  ai_confidence: 0.62
  min_required: 0.70
  action: YES
```

### Position Size Override
```
[TEST] Overriding position size
  market_id: 0x123abc
  ai_suggested: $50.00
  test_override: $1.00
```

### Market Marked as Traded
```
[TEST] Market marked as traded
  market_id: 0x123abc
  total_traded_markets: 6
```

---

## Database Tracking

All test trades are marked in the database:

```sql
-- View all test mode trades
SELECT * FROM trades WHERE is_test_mode = 1;

-- Count test trades
SELECT COUNT(*) FROM trades WHERE is_test_mode = 1;

-- Compare test vs production win rates
SELECT
    is_test_mode,
    COUNT(*) as total_trades,
    SUM(is_win) as wins,
    ROUND(AVG(is_win) * 100, 2) as win_rate_pct
FROM trades
WHERE execution_status = 'filled'
GROUP BY is_test_mode;

-- Test trades by confidence range
SELECT
    CASE
        WHEN confidence < 0.75 THEN '70-75%'
        WHEN confidence < 0.80 THEN '75-80%'
        ELSE '80%+'
    END as confidence_range,
    COUNT(*) as trades,
    SUM(is_win) as wins
FROM trades
WHERE is_test_mode = 1 AND execution_status = 'filled'
GROUP BY confidence_range;
```

---

## Safety Considerations

### ✅ Safe to Use

- **$1 bet limit** is hardcoded (no way to increase)
- **Duplicate prevention** ensures one bet per market
- **Confidence threshold** filters low-quality setups
- **Database tracking** allows performance analysis
- **In-memory tracking** resets on restart (fresh session)

### ⚠️ Important Notes

1. **Real Money:** Test mode uses real funds (not dry-run)
2. **No Undo:** Executed trades cannot be reversed
3. **Session-Scoped:** Duplicate prevention resets on bot restart
4. **Rate Limits:** CoinGecko Pro limits still apply (currently 6/250 calls per minute)

### 🚫 Do Not

- Run test mode and production mode simultaneously
- Manually modify `traded_markets` set during execution
- Expect high profitability (this is for learning, not profit optimization)
- Use test mode as primary trading strategy (it's for validation only)

---

## Monitoring

### Check Test Mode Status

```bash
# Verify environment variable
echo $TEST_MODE

# Check logs for test mode activation
grep "TEST MODE ENABLED" logs/*.log
```

### Monitor Test Trades

```bash
# Count test trades in current session
grep "\[TEST\] Market marked as traded" logs/*.log | wc -l

# View confidence rejections
grep "\[TEST\] Skipping trade - confidence below" logs/*.log

# View duplicates prevented
grep "\[TEST\] Skipping market - already traded" logs/*.log
```

### Performance Analysis

```bash
# Run test mode for 1 hour
TEST_MODE=true python scripts/auto_trade.py

# After completion, analyze results
python -c "
import sqlite3
conn = sqlite3.connect('data/performance.db')
cursor = conn.cursor()
cursor.execute('''
    SELECT
        COUNT(*) as trades,
        SUM(is_win) as wins,
        ROUND(AVG(confidence) * 100, 2) as avg_confidence,
        ROUND(SUM(position_size_usd), 2) as total_spent
    FROM trades
    WHERE is_test_mode = 1 AND execution_status = 'filled'
''')
print(cursor.fetchone())
"
```

---

## Expected Behavior

### Scenario 1: High-Quality Setup (Confidence ≥ 70%)

```
Market: Will BTC be above $95,500 at 3:45 PM UTC?
Movement: $45 (bypassed < $100 threshold)
Signals: BULLISH (funding=-0.02%, premium=+0.7%, volume=high)
Timeframe: Conflicting (bypassed)
AI Decision: YES with 0.78 confidence
Result: ✅ Trade executed with $1 bet
```

### Scenario 2: Low-Confidence Setup (< 70%)

```
Market: Will BTC be above $95,400 at 3:30 PM UTC?
Movement: $30 (bypassed)
Signals: NEUTRAL (confidence=0.35)
Regime: UNCLEAR (bypassed)
AI Decision: YES with 0.62 confidence
Result: ❌ Skipped (below 70% threshold)
```

### Scenario 3: Duplicate Market

```
Market: Will BTC be above $95,500 at 3:45 PM UTC?
Status: Already traded in this session
Result: ❌ Skipped (duplicate prevention)
```

---

## Deactivation

```bash
unset TEST_MODE
```

Or simply run without the variable:

```bash
python scripts/auto_trade.py
```

**Verification:** No test mode banner should appear on startup.

---

## Troubleshooting

### Issue: Test mode not activating

**Check:**
```bash
python3 -c "
import os
os.environ['TEST_MODE'] = 'true'
from scripts.auto_trade import TestModeConfig
config = TestModeConfig(enabled=os.getenv('TEST_MODE', '').lower() == 'true')
print(f'Enabled: {config.enabled}')
"
```

**Expected:** `Enabled: True`

### Issue: Database errors

**Solution:** Recreate database with updated schema:
```bash
rm -f data/performance.db
python3 -c "from polymarket.performance.database import PerformanceDatabase; PerformanceDatabase('data/performance.db')"
```

### Issue: No trades executing

**Check logs for:**
- `[TEST] Skipping trade - confidence below threshold` (confidence too low)
- `[TEST] Skipping market - already traded` (duplicate)
- `Skipping market - not a 15-minute market` (wrong market type)

---

## Success Metrics

After running test mode for 24 hours, evaluate:

1. **Signal Integration:** Do CoinGecko signals appear in AI reasoning?
2. **Decision Quality:** Is AI confidence calibrated (>70% setups profitable)?
3. **Edge Detection:** Are bypassed setups actually unprofitable?
4. **System Stability:** Any errors, timeouts, or unexpected behavior?

**Analysis Query:**
```sql
SELECT
    COUNT(*) as total_test_trades,
    SUM(is_win) as wins,
    ROUND(AVG(is_win) * 100, 2) as win_rate_pct,
    ROUND(AVG(confidence) * 100, 2) as avg_confidence_pct,
    ROUND(SUM(CASE WHEN is_win = 1 THEN profit_loss ELSE 0 END), 2) as total_profit,
    ROUND(SUM(CASE WHEN is_win = 0 THEN profit_loss ELSE 0 END), 2) as total_loss,
    ROUND(SUM(profit_loss), 2) as net_pnl
FROM trades
WHERE is_test_mode = 1 AND execution_status = 'filled';
```

---

## Next Steps

1. **Run Integration Tests:**
   ```bash
   ./test_test_mode.sh
   ```

2. **Start Test Mode:**
   ```bash
   TEST_MODE=true python scripts/auto_trade.py
   ```

3. **Monitor for 1 hour** (check logs and database)

4. **Analyze Results** (use SQL queries above)

5. **Tune If Needed:**
   - Adjust confidence threshold (currently 70%)
   - Modify signal weights (currently 35/35/30)
   - Update thresholds (funding rate, spread, volume)

---

**Created:** 2026-02-13
**Purpose:** Validate CoinGecko signals integration
**Status:** Ready for production testing 🚀

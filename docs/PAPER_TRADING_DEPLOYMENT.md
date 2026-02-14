# Paper Trading Mode - Deployment Guide

> **Version:** 2.0 - Post-Performance Fixes
> **Date:** February 14, 2026
> **Status:** Ready for 24-48 Hour Testing

## Overview

This guide covers deploying the trading bot with the new paper trading improvements:

1. **Signal Lag Detection** - Detects when market sentiment lags BTC price movement
2. **Conflict Detection** - Identifies conflicting signals and reduces confidence
3. **Odds Polling** - Background service checking market odds every 60s
4. **Paper Trading Mode** - No real money, full signal tracking, Telegram alerts
5. **Arbitrage Gate Removed** - No longer requires arbitrage edge (kept for AI context)

## Prerequisites

### Required Environment Variables

```bash
# Core Trading
TEST_MODE=true                    # Enable test mode (required)
PRIVATE_KEY=<your_private_key>    # Wallet private key
POLYGON_RPC_URL=<your_rpc_url>    # Polygon RPC endpoint

# Telegram (Required for Paper Trading Alerts)
TELEGRAM_BOT_TOKEN=<your_token>   # Bot token from @BotFather
TELEGRAM_CHAT_ID=<your_chat_id>   # Your chat ID

# API Keys
COINGECKO_API_KEY=<your_key>      # CoinGecko Pro API
OPENAI_API_KEY=<your_key>         # OpenAI API for AI decisions
```

### Database Migration

Before running, ensure database schema is up to date:

```bash
cd /root/polymarket-scripts
python3 scripts/migrations/add_paper_trading_support.py
```

This creates the `paper_trades` table and adds signal tracking columns.

## Deployment Steps

### Step 1: Run Paper Trading Mode (24-48 Hours)

```bash
cd /root/polymarket-scripts
TEST_MODE=true python3 scripts/auto_trade.py
```

**What Happens:**
- Bot polls market odds every 60 seconds
- Evaluates markets when they open
- Logs simulated trades to `paper_trades` table
- Sends detailed Telegram alerts for each paper trade
- NO real money is spent (stops before order placement)

### Step 2: Monitor Telegram Alerts

Each paper trade alert includes:

```
🧪 PAPER TRADE SIGNAL
━━━━━━━━━━━━━━━━━━━━
📊 Market: btc-updown-15m-1771234500
📈 Direction: YES (UP)
💵 Position: $5.00 @ 0.82 odds
⏰ Time Remaining: 14m 30s

🎯 SIGNAL ANALYSIS:
📊 Technical: BULLISH (RSI: 68.5, EMA: UP)
💬 Sentiment: BULLISH (0.85 confidence)
✅ Odds Check: YES = 82% (PASS > 75%)
📈 Timeframe: ALIGNED (1m/5m/15m/30m all bullish)
✅ Signal Lag: NO LAG DETECTED

🤖 AI REASONING:
"Strong upward momentum with bullish sentiment..."

📊 CONFIDENCE: 0.85
✅ Conflicts: NONE

━━━━━━━━━━━━━━━━━━━━
```

**Key Indicators to Watch:**
- ⚠️ Signal lag warnings
- 🚫 SEVERE conflicts (auto-HOLD)
- ❌ Odds check failures
- 📉 Confidence penalties

### Step 3: Analyze Paper Trading Results

Query the database after 24-48 hours:

```sql
-- Overall paper trading metrics
SELECT
    COUNT(*) as total_paper_trades,
    AVG(confidence) as avg_confidence,
    SUM(CASE WHEN signal_lag_detected = 1 THEN 1 ELSE 0 END) as signal_lag_count,
    SUM(CASE WHEN conflict_severity = 'SEVERE' THEN 1 ELSE 0 END) as severe_conflicts,
    SUM(CASE WHEN conflict_severity = 'MODERATE' THEN 1 ELSE 0 END) as moderate_conflicts,
    SUM(CASE WHEN odds_qualified = 1 THEN 1 ELSE 0 END) as odds_qualified_count
FROM paper_trades;

-- Paper trades by action
SELECT
    action,
    COUNT(*) as count,
    AVG(confidence) as avg_confidence,
    AVG(executed_price) as avg_odds
FROM paper_trades
GROUP BY action;

-- Signal lag distribution
SELECT
    signal_lag_reason,
    COUNT(*) as occurrences
FROM paper_trades
WHERE signal_lag_detected = 1
GROUP BY signal_lag_reason
ORDER BY occurrences DESC;

-- Conflict severity distribution
SELECT
    conflict_severity,
    COUNT(*) as count,
    AVG(confidence) as avg_confidence_after_penalty
FROM paper_trades
WHERE conflict_severity IS NOT NULL
GROUP BY conflict_severity;

-- Odds qualification rate
SELECT
    CASE
        WHEN odds_qualified = 1 THEN 'Qualified (>75%)'
        ELSE 'Not Qualified'
    END as odds_status,
    COUNT(*) as count
FROM paper_trades
GROUP BY odds_qualified;
```

### Step 4: Review Key Metrics

**Expected Results (Good):**
- ✅ Signal lag detected rate: < 10%
- ✅ SEVERE conflict rate: < 5%
- ✅ Odds qualified rate: > 80%
- ✅ Average confidence: > 0.75
- ✅ NO crashes or exceptions

**Red Flags (Bad):**
- 🚫 Signal lag > 20% (market sentiment severely lagging)
- 🚫 SEVERE conflicts > 15% (conflicting signals common)
- 🚫 Odds qualified < 50% (markets too uncertain)
- 🚫 Frequent crashes or database errors

### Step 5: Production Deployment (If Results Good)

If paper trading results look good:

1. **Disable Test Mode:**
   ```bash
   # Set in .env or environment
   TEST_MODE=false
   ```

2. **Set Production Position Sizing:**
   ```bash
   # Example: $10-50 bets instead of $5-10
   # (This is controlled by Kelly sizing + risk manager)
   ```

3. **Run Production Bot:**
   ```bash
   cd /root/polymarket-scripts
   python3 scripts/auto_trade.py
   ```

4. **Monitor First 10 Trades Closely:**
   - Check Telegram alerts for each trade
   - Verify actual execution prices match expectations
   - Monitor for signal lag/conflict warnings
   - Watch for auto-HOLD triggers

## New Features Explained

### 1. Signal Lag Detection

**What it Does:**
- Compares BTC actual direction vs market sentiment direction
- Only triggers on high-confidence contradictions (>0.6)
- Auto-HOLD if lag detected (in production mode)

**Example:**
```
BTC: UP (+$300)
Market Sentiment: BEARISH (0.75 confidence)
→ SIGNAL LAG DETECTED → AUTO-HOLD (don't trade against reality)
```

### 2. Conflict Detection

**Severity Levels:**
- **NONE:** All signals aligned → No penalty
- **MINOR:** 1 conflict → -0.10 confidence penalty
- **MODERATE:** 2 conflicts → -0.20 confidence penalty
- **SEVERE:** 3+ conflicts → AUTO-HOLD (too uncertain)

**Example:**
```
BTC: UP
Technical: DOWN (conflict 1)
Sentiment: BEARISH (conflict 2)
Timeframes: CONFLICTING (conflict 3)
→ SEVERE conflict → AUTO-HOLD
```

### 3. Odds Polling

**What it Does:**
- Background service polls Polymarket API every 60 seconds
- Caches odds for current BTC 15-min market
- Enables early filtering: skip markets where neither side > 75% odds

**Benefits:**
- Avoids expensive AI calls for uncertain markets
- Ensures we only trade high-conviction markets
- Reduces noise trades

### 4. Arbitrage Gate Removed

**What Changed:**
- Old: Required 2% arbitrage edge to trade
- New: No arbitrage requirement (but still calculated for AI context)

**Rationale:**
- Bot had 10.6% win rate with arbitrage gate
- Signal conflict detection + odds validation are more effective filters
- Arbitrage data still passed to AI for decision-making

## Troubleshooting

### Issue: No Paper Trades Being Logged

**Possible Causes:**
1. No markets meet >75% odds threshold
2. Signal lag auto-HOLD triggered
3. SEVERE conflicts auto-HOLD triggered
4. TEST_MODE not set correctly

**Solution:**
```bash
# Check logs for skip reasons
tail -f logs/polymarket_bot.log | grep "Skipping"

# Verify TEST_MODE enabled
echo $TEST_MODE  # Should be "true"
```

### Issue: Database Errors

**Error:** `no such table: paper_trades`

**Solution:**
```bash
# Run migration
python3 scripts/migrations/add_paper_trading_support.py
```

### Issue: Telegram Alerts Not Sending

**Possible Causes:**
1. TELEGRAM_BOT_TOKEN not set
2. TELEGRAM_CHAT_ID incorrect
3. Bot not started with @BotFather

**Solution:**
```bash
# Verify environment variables
echo $TELEGRAM_BOT_TOKEN
echo $TELEGRAM_CHAT_ID

# Test Telegram connection
python3 -c "
from polymarket.telegram.bot import TelegramBot
from polymarket.config import Settings
settings = Settings()
bot = TelegramBot(settings)
import asyncio
asyncio.run(bot._send_message('Test message'))
"
```

## Rollback Plan

If issues arise in production:

1. **Immediate:** Set `TEST_MODE=true` to stop real trades
2. **Investigate:** Check logs and database for errors
3. **Revert:** Check out previous git commit if needed:
   ```bash
   cd /root/polymarket-scripts
   git log --oneline  # Find last good commit
   git checkout <commit-hash>
   ```

## Success Criteria

**Paper Trading (24-48 hours):**
- ✅ 20+ paper trades logged
- ✅ Signal lag < 10%
- ✅ SEVERE conflicts < 5%
- ✅ No crashes or exceptions
- ✅ Telegram alerts working

**Production (First 10 Trades):**
- ✅ Trades execute at expected prices
- ✅ No unexpected losses
- ✅ Auto-HOLD triggers working as designed
- ✅ Confidence penalties applied correctly

## Support

For issues or questions:
1. Check logs: `tail -f logs/polymarket_bot.log`
2. Query database: `sqlite3 polymarket_trades.db`
3. Review Telegram alerts for specific trade details

---

**Last Updated:** February 14, 2026
**Implementation Commits:** 11 commits (Tasks 1-11)
**Ready for:** 24-48 hour paper trading validation

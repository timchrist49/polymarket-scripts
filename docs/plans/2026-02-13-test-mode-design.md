# Test Mode Design - Force Trading with $1 Bets

**Date:** 2026-02-13
**Status:** Design Approved - Ready for Implementation
**Purpose:** Test new CoinGecko signals with forced trading on every 15-min BTC market

---

## Overview

Test mode allows forced trading on every Polymarket 15-minute BTC market with strict $1 bet limits and 70% minimum AI confidence. All safety filters are bypassed, but data is still collected and sent to the AI for informed decision-making.

**Key Constraints:**
- ✅ Real money trading with $1 strict limit per bet
- ✅ Minimum 70% AI confidence required
- ✅ One bet per market (no duplicates)
- ✅ All CoinGecko signals included in AI analysis
- ✅ Safety filter data sent to AI (spread, volume, regime)
- ✅ Loud logging for visibility
- ✅ Database tracking for AI learning

---

## Activation

**Environment Variable:**
```bash
TEST_MODE=true python scripts/auto_trade.py
```

**Why Environment Variable:**
- Explicit activation (can't accidentally leave on)
- No code changes between test and production
- Easy to see in process list
- Clear in logs

---

## Architecture

### Test Mode Configuration

**New class in `auto_trade.py`:**

```python
@dataclass
class TestModeConfig:
    """Test mode configuration."""
    enabled: bool = False
    max_bet_amount: Decimal = Decimal("1.0")
    min_confidence: float = 0.70
    traded_markets: set[str] = field(default_factory=set)
```

### Execution Flow

```
┌─────────────────────────────────────────────────────────┐
│                     TEST MODE FLOW                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. Detect Market                                        │
│     └─► Check if already traded this market             │
│         ├─► YES: Skip with [TEST] log                   │
│         └─► NO: Continue                                 │
│                                                          │
│  2. Collect All Data (bypass filters but gather info)   │
│     ├─► Movement: $23 < $100 [TEST BYPASS]              │
│     ├─► Spread: 8.2% > 5% [TEST BYPASS]                 │
│     ├─► Volume: Low [TEST BYPASS]                       │
│     ├─► Regime: VOLATILE [TEST BYPASS]                  │
│     └─► CoinGecko Signals: Funding + Premium + Volume   │
│                                                          │
│  3. Send ALL data to AI                                 │
│     └─► AI gets complete context including bypassed     │
│         conditions for informed decision                 │
│                                                          │
│  4. Force AI Decision                                   │
│     ├─► If HOLD: Force YES or NO based on sentiment     │
│     └─► Log: [TEST] Forced direction                    │
│                                                          │
│  5. Check Confidence                                    │
│     ├─► >= 70%: Continue                                │
│     └─► < 70%: Skip with [TEST] log                     │
│                                                          │
│  6. Override Position Size                              │
│     └─► Force $1.00 (ignore AI suggestion)              │
│                                                          │
│  7. Execute Trade                                       │
│     ├─► Real money, real Polymarket                     │
│     ├─► Mark market as traded                           │
│     └─► Log: [TEST] TRADE EXECUTED                      │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Implementation Details

### 1. Initialization

**Location:** `scripts/auto_trade.py` - `AutoTrader.__init__()`

```python
import os
from dataclasses import dataclass, field

@dataclass
class TestModeConfig:
    enabled: bool = False
    max_bet_amount: Decimal = Decimal("1.0")
    min_confidence: float = 0.70
    traded_markets: set[str] = field(default_factory=set)

class AutoTrader:
    def __init__(self, settings: Settings):
        self.settings = settings

        # Test mode configuration
        self.test_mode = TestModeConfig(
            enabled=os.getenv("TEST_MODE", "").lower() == "true",
            max_bet_amount=Decimal("1.0"),
            min_confidence=0.70,
            traded_markets=set()
        )

        if self.test_mode.enabled:
            logger.warning("=" * 60)
            logger.warning("🧪 TEST MODE ACTIVE - SAFETY FILTERS BYPASSED")
            logger.warning("=" * 60)
            logger.warning(f"Max bet per trade: ${self.test_mode.max_bet_amount}")
            logger.warning(f"Min AI confidence: {self.test_mode.min_confidence:.0%}")
            logger.warning(f"One bet per market: ENABLED")
            logger.warning(f"Safety filters: BYPASSED (data sent to AI)")
            logger.warning(f"CoinGecko signals: INCLUDED")
            logger.warning("=" * 60)
```

### 2. Safety Filter Bypass

**Location:** Trading cycle in `auto_trade.py` (lines 792-860)

**Movement Threshold:**
```python
MIN_MOVEMENT_THRESHOLD = 100
abs_diff = abs(diff)

if abs_diff < MIN_MOVEMENT_THRESHOLD:
    if not self.test_mode.enabled:
        logger.info("Skipping market - insufficient movement")
        return
    else:
        logger.info(
            "[TEST] Bypassing movement threshold",
            movement=f"${abs_diff:.2f}",
            threshold=f"${MIN_MOVEMENT_THRESHOLD}",
            reason="Test mode - info sent to AI"
        )
        # Continue...
```

**Spread Check:**
```python
if orderbook_analysis.spread_bps > 500:
    if not self.test_mode.enabled:
        logger.info("Skipping trade - spread too wide")
        return
    else:
        logger.info(
            "[TEST] Bypassing spread check",
            spread_bps=f"{orderbook_analysis.spread_bps:.1f}",
            threshold="500 bps",
            reason="Test mode - info sent to AI"
        )
```

**Volume Check:**
```python
if abs_diff > 200 and volume_data:
    if not volume_data.is_high_volume:
        if not self.test_mode.enabled:
            logger.info("Skipping large move without volume")
            return
        else:
            logger.info(
                "[TEST] Bypassing volume check",
                volume_ratio=f"{volume_data.volume_ratio:.2f}x",
                reason="Test mode - info sent to AI"
            )
```

**Timeframe Alignment:**
```python
if timeframe_analysis and timeframe_analysis.alignment == "CONFLICTING":
    if not self.test_mode.enabled:
        logger.info("Skipping trade - conflicting timeframes")
        return
    else:
        logger.info(
            "[TEST] Bypassing timeframe check",
            alignment=timeframe_analysis.alignment,
            reason="Test mode - info sent to AI"
        )
```

**Market Regime:**
```python
if regime and regime.regime in ["UNCLEAR", "VOLATILE"]:
    if not self.test_mode.enabled:
        logger.info("Skipping trade - unfavorable regime")
        return
    else:
        logger.info(
            "[TEST] Bypassing regime check",
            regime=regime.regime,
            reason="Test mode - info sent to AI"
        )
```

### 3. Market Duplicate Prevention

**Location:** Before AI decision call (around line 950)

```python
# Test mode: Check if already traded this market
if self.test_mode.enabled:
    market_id = market.id

    if market_id in self.test_mode.traded_markets:
        logger.info(
            "[TEST] Skipping - already traded this market",
            market_id=market_id,
            market_question=market.question,
            total_traded=len(self.test_mode.traded_markets),
            reason="One bet per market rule"
        )
        return

    logger.info(
        "[TEST] New market - proceeding",
        market_id=market_id,
        total_traded=len(self.test_mode.traded_markets)
    )
```

### 4. AI Decision Forcing

**Location:** After AI decision call (around line 994)

```python
# Make AI decision with force_trade flag
decision = await self.ai_service.make_decision(
    btc_price=btc_data,
    technical_indicators=indicators,
    aggregated_sentiment=aggregated_sentiment,
    market_data=market_dict,
    portfolio_value=portfolio_value,
    orderbook_data=orderbook_analysis,
    volume_data=volume_data,
    timeframe_analysis=timeframe_analysis,
    regime=regime,
    arbitrage_opportunity=arbitrage_opportunity,
    market_signals=market_signals,  # ← CoinGecko signals included
    force_trade=self.test_mode.enabled  # ← NEW PARAMETER
)

# Test mode: Force YES/NO and check confidence
if self.test_mode.enabled:
    # Force direction if HOLD
    if decision.action == "HOLD":
        logger.warning(
            "[TEST] AI returned HOLD - forcing direction",
            confidence=f"{decision.confidence:.2f}",
            sentiment_score=f"{aggregated_sentiment.final_score:+.2f}"
        )

        # Pick direction based on aggregated sentiment
        if aggregated_sentiment.final_score > 0:
            decision.action = "YES"
        else:
            decision.action = "NO"

        logger.info(
            "[TEST] Forced direction",
            action=decision.action,
            based_on=f"sentiment_score_{aggregated_sentiment.final_score:+.2f}"
        )

    # Check confidence threshold
    if decision.confidence < self.test_mode.min_confidence:
        logger.info(
            "[TEST] Skipping - confidence below threshold",
            confidence=f"{decision.confidence:.2f}",
            threshold=f"{self.test_mode.min_confidence:.2f}",
            action=decision.action
        )
        return

    # Override position size to $1
    original_size = decision.position_size
    decision.position_size = self.test_mode.max_bet_amount

    logger.info(
        "[TEST] Position size overridden",
        original=f"${original_size:.2f}",
        test_mode=f"${self.test_mode.max_bet_amount:.2f}"
    )
```

### 5. Trade Execution & Tracking

**Location:** After trade execution (around line 1100)

```python
# Execute trade
result = await self.execute_trade(...)

# Test mode: Mark market as traded
if self.test_mode.enabled and result.success:
    self.test_mode.traded_markets.add(market.id)

    logger.info(
        "[TEST] ✓ TRADE EXECUTED",
        market_id=market.id,
        action=decision.action,
        amount=f"${decision.position_size:.2f}",
        confidence=f"{decision.confidence:.2f}",
        total_test_trades=len(self.test_mode.traded_markets)
    )
```

### 6. Database Schema Update

**Location:** `polymarket/performance/database.py`

**Add column to trades table:**

```sql
-- Migration
ALTER TABLE trades ADD COLUMN is_test_mode BOOLEAN DEFAULT FALSE;

-- Index for querying test trades
CREATE INDEX idx_trades_test_mode ON trades(is_test_mode);
```

**Update log_decision in performance tracker:**

```python
# In PerformanceTracker.log_decision()
async def log_decision(
    self,
    market: Market,
    decision: TradingDecision,
    is_test_mode: bool = False,  # NEW PARAMETER
    ...
) -> int:
    """Log trading decision to database."""

    cursor.execute(
        """
        INSERT INTO trades (
            ...,
            is_test_mode
        ) VALUES (?, ?, ?, ?, ...)
        """,
        (
            ...,
            is_test_mode
        )
    )
```

**Update caller in auto_trade.py:**

```python
trade_id = await self.performance_tracker.log_decision(
    market=market,
    decision=decision,
    btc_data=btc_data,
    ...,
    is_test_mode=self.test_mode.enabled  # ← NEW PARAMETER
)
```

### 7. AI Service Modification

**Location:** `polymarket/trading/ai_decision.py`

**Update method signature:**

```python
async def make_decision(
    self,
    btc_price: BTCPriceData,
    technical_indicators: TechnicalIndicators,
    aggregated_sentiment: AggregatedSentiment,
    market_data: dict,
    portfolio_value: Decimal = Decimal("1000"),
    orderbook_data: "OrderbookData | None" = None,
    volume_data: VolumeData | None = None,
    timeframe_analysis: TimeframeAnalysis | None = None,
    regime: MarketRegime | None = None,
    arbitrage_opportunity: "ArbitrageOpportunity | None" = None,
    market_signals: "Any | None" = None,
    force_trade: bool = False  # ← NEW PARAMETER
) -> TradingDecision:
```

**Update prompt building:**

```python
def _build_prompt(self, ..., force_trade: bool = False) -> str:
    """Build AI prompt with optional forced trading instruction."""

    # Add test mode instruction
    if force_trade:
        force_trade_instruction = """
═══════════════════════════════════════════════════════════════════
⚠️ TEST MODE ACTIVE - FORCED TRADING
═══════════════════════════════════════════════════════════════════

CRITICAL: You MUST return either "YES" or "NO" - HOLD is NOT allowed.

Even if conditions are uncertain, you must pick the side you believe
has the highest probability of winning based on ALL available data:

DATA SOURCES (use all of these):
1. CoinGecko Market Signals (funding rates, exchange premium, volume)
2. Technical Indicators (RSI, MACD, trend)
3. Social Sentiment (fear/greed, community votes)
4. Market Microstructure (orderbook, whale activity)
5. Timeframe Analysis (daily, 4h trends)
6. Market Regime (trending, ranging, volatile)

NOTE: Safety filters were bypassed in test mode, but you still received
all the data (spread, volume, regime). Use this information to make
an informed decision, but you MUST pick YES or NO.

DECISION PRIORITY (if truly uncertain):
1. Strongest signal direction (market signals > technical > sentiment)
2. Break ties with aggregated sentiment score
3. Consider risk: wide spreads and low volume reduce confidence

Your confidence score should reflect your actual conviction, and you
must still pick YES or NO even if confidence is low.
═══════════════════════════════════════════════════════════════════
"""
    else:
        force_trade_instruction = ""

    # Insert into prompt before DECISION FORMAT
    return f"""...

{force_trade_instruction}

DECISION FORMAT:
...
"""
```

---

## Logging Strategy

### Startup Banner

```
==============================================================
🧪 TEST MODE ACTIVE - SAFETY FILTERS BYPASSED
==============================================================
Max bet per trade: $1.0
Min AI confidence: 70%
One bet per market: ENABLED
Safety filters: BYPASSED (data sent to AI)
CoinGecko signals: INCLUDED
==============================================================
```

### Per-Cycle Logging

```
[TEST] Market: btc-updown-15m-1770981300
[TEST] Checking if already traded... NO (0 markets traded so far)
[TEST] Bypassing movement threshold: $23.00 < $100.00
[TEST] Bypassing spread check: 8.2% > 5.0%
[TEST] Bypassing volume check: Low volume (0.8x average)
[TEST] Bypassing regime check: VOLATILE
[TEST] CoinGecko signals collected: BULLISH (0.76 confidence)
[TEST] Forcing AI decision: HOLD not allowed
[TEST] AI Decision: YES, confidence 0.78
[TEST] Position size overridden: $8.50 → $1.00
[TEST] ✓ TRADE EXECUTED
[TEST] Total test trades this session: 1
```

### Skip Logging

```
[TEST] Skipping - already traded this market | market_id=1369055
[TEST] Skipping - confidence below threshold | confidence=0.65 < 0.70
```

---

## Safety Considerations

### Built-In Protections

1. **$1 Hard Limit**
   - Position size always overridden to $1.00
   - Even if AI suggests $10, only $1 is traded

2. **70% Confidence Minimum**
   - Skip trades where AI confidence < 70%
   - Protects from completely uncertain bets

3. **One Bet Per Market**
   - In-memory tracking: `traded_markets` set
   - Database tracking: `is_test_mode` flag
   - Prevents duplicate bets on same market

4. **Environment Variable Activation**
   - Must explicitly set `TEST_MODE=true`
   - Can't accidentally leave on
   - Clear in process list

5. **Loud Logging**
   - `[TEST]` prefix on every log
   - Startup banner impossible to miss
   - Clear audit trail

6. **Database Separation**
   - `is_test_mode` flag for analysis
   - Easy to query: `SELECT * FROM trades WHERE is_test_mode = TRUE`
   - Won't affect production metrics

### Maximum Loss Calculation

```
Markets per hour: ~4 (one every 15 minutes)
Max loss per trade: $1.00
Confidence filter: ~50% of markets will be < 70% (skipped)

Conservative estimate:
- Trading opportunity: 2 markets per hour
- Worst case loss: $2/hour
- Daily worst case: $48
- Weekly worst case: $336
```

### Recommended Testing Approach

**Phase 1: Initial Test (2 hours)**
- Markets: ~8 opportunities
- Max loss: $8
- Monitor: All logs, database entries
- Verify: CoinGecko signals in AI reasoning

**Phase 2: Extended Test (24 hours)**
- Markets: ~96 opportunities
- Max loss: $96 (if every trade loses)
- Analyze: Win rate, signal accuracy
- Compare: Test vs production performance

**Phase 3: Scale Decision**
- If win rate > 50%: Consider production deployment
- If win rate < 50%: Analyze failures, tune signals

---

## Analysis Queries

### View All Test Trades

```sql
SELECT
    id,
    market_id,
    action,
    confidence,
    amount,
    win,
    profit_loss,
    timestamp
FROM trades
WHERE is_test_mode = TRUE
ORDER BY timestamp DESC;
```

### Test Mode Performance

```sql
SELECT
    COUNT(*) as total_trades,
    SUM(CASE WHEN win = 1 THEN 1 ELSE 0 END) as wins,
    SUM(CASE WHEN win = 0 THEN 1 ELSE 0 END) as losses,
    ROUND(AVG(CASE WHEN win = 1 THEN 1.0 ELSE 0.0 END) * 100, 2) as win_rate,
    ROUND(SUM(profit_loss), 2) as total_pl
FROM trades
WHERE is_test_mode = TRUE;
```

### Compare Test vs Production

```sql
SELECT
    is_test_mode,
    COUNT(*) as trades,
    ROUND(AVG(confidence), 3) as avg_confidence,
    ROUND(AVG(CASE WHEN win = 1 THEN 1.0 ELSE 0.0 END) * 100, 2) as win_rate,
    ROUND(SUM(profit_loss), 2) as total_pl
FROM trades
GROUP BY is_test_mode;
```

---

## Testing Checklist

**Before Starting:**
- [ ] Understand max loss: $2/hour, $48/day
- [ ] Database schema updated with `is_test_mode` column
- [ ] Code deployed with test mode implementation
- [ ] Bot stopped (no running instances)

**Starting Test:**
- [ ] Set environment variable: `TEST_MODE=true`
- [ ] Start bot: `nohup python scripts/auto_trade.py > test-mode.log 2>&1 &`
- [ ] Verify startup banner in logs
- [ ] Watch for first `[TEST]` trade

**Monitoring (every 30 minutes):**
- [ ] Check logs: `tail -f test-mode.log | grep TEST`
- [ ] Count trades: `SELECT COUNT(*) FROM trades WHERE is_test_mode = TRUE`
- [ ] Check win rate: Run performance query
- [ ] Verify CoinGecko signals in logs

**After Test:**
- [ ] Stop bot
- [ ] Run analysis queries
- [ ] Review win rate and P&L
- [ ] Decide: Continue, tune, or stop

---

## Success Criteria

**Test Mode is Working If:**
1. ✅ Startup banner shows `TEST MODE ACTIVE`
2. ✅ Every log has `[TEST]` prefix
3. ✅ Trades execute with $1.00 amount
4. ✅ Database shows `is_test_mode = TRUE`
5. ✅ Markets traded only once (no duplicates)
6. ✅ Confidence filter works (skips < 70%)
7. ✅ CoinGecko signals appear in logs

**Test Mode is Successful If:**
1. Win rate > 50% (better than coin flip)
2. Trades execute on markets with AI confidence ≥ 70%
3. No duplicate bets on same market
4. Total loss within expected range
5. CoinGecko signals improve decision quality

---

## Rollback Plan

**If Test Mode Causes Issues:**

1. **Stop Bot Immediately**
   ```bash
   pkill -f auto_trade.py
   ```

2. **Disable Test Mode**
   ```bash
   unset TEST_MODE
   python scripts/auto_trade.py  # Normal mode
   ```

3. **Analyze Test Trades**
   ```sql
   SELECT * FROM trades WHERE is_test_mode = TRUE ORDER BY timestamp DESC LIMIT 10;
   ```

4. **Resume Normal Operation**
   - Bot continues with production logic
   - Test trades remain in database for analysis
   - No impact on production performance tracking

---

## Future Enhancements

**Potential Improvements:**
1. **Configurable Thresholds**
   - `TEST_MODE_MAX_BET=1.0`
   - `TEST_MODE_MIN_CONFIDENCE=0.70`

2. **Test Duration Limit**
   - `TEST_MODE_MAX_TRADES=20`
   - Auto-stop after N trades

3. **Dry-Run Test Mode**
   - `TEST_MODE=true POLYMARKET_MODE=read_only`
   - Log decisions without real trades

4. **Signal Weight Testing**
   - Vary CoinGecko signal weights
   - A/B test different configurations

---

## Implementation Status

- [ ] Database schema updated
- [ ] Test mode initialization
- [ ] Safety filter bypass logic
- [ ] Duplicate market prevention
- [ ] AI decision forcing
- [ ] Position size override
- [ ] Database tracking integration
- [ ] AI prompt modification
- [ ] Comprehensive logging
- [ ] Testing and verification

---

**Design Approved:** 2026-02-13
**Next Step:** Create implementation plan with /superpowers:write-plan

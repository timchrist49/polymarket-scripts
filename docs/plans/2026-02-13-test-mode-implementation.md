# Test Mode Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement test mode that forces trading on every 15-min BTC market with $1 bets and 70% min confidence

**Architecture:** Environment variable activation, bypass safety filters while collecting data, force AI YES/NO decisions, track trades in database with `is_test_mode` flag

**Tech Stack:** Python, SQLite, structlog, existing trading bot infrastructure

---

## Task 1: Update Database Schema

**Files:**
- Modify: `polymarket/performance/database.py`
- Modify: `polymarket/performance/tracker.py`

**Step 1: Add is_test_mode column to schema**

Location: `polymarket/performance/database.py` - Update `_create_schema()` method

```python
# In the CREATE TABLE trades section, add the new column:
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT NOT NULL,
    action TEXT NOT NULL,
    confidence REAL NOT NULL,
    amount REAL NOT NULL,
    timestamp INTEGER NOT NULL,
    btc_price REAL,
    technical_rsi REAL,
    technical_macd REAL,
    technical_trend TEXT,
    sentiment_score REAL,
    sentiment_confidence REAL,
    win INTEGER,
    profit_loss REAL,
    settlement_price REAL,
    settlement_timestamp INTEGER,
    price_to_beat REAL,
    time_remaining_seconds INTEGER,
    is_end_phase INTEGER DEFAULT 0,
    actual_probability REAL,
    arbitrage_edge REAL,
    arbitrage_urgency TEXT,
    is_test_mode INTEGER DEFAULT 0  -- NEW COLUMN
);
```

**Step 2: Add index for test mode queries**

Location: `polymarket/performance/database.py` - Add after trades table creation

```python
# After CREATE TABLE trades, add:
cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_trades_test_mode
    ON trades(is_test_mode)
""")
```

**Step 3: Update log_decision method signature**

Location: `polymarket/performance/tracker.py` - Update `log_decision()`

```python
async def log_decision(
    self,
    market: Market,
    decision: TradingDecision,
    btc_data: BTCPriceData,
    technical: TechnicalIndicators,
    aggregated: AggregatedSentiment,
    price_to_beat: Decimal | None = None,
    time_remaining_seconds: int | None = None,
    is_end_phase: bool = False,
    actual_probability: float | None = None,
    arbitrage_edge: float | None = None,
    arbitrage_urgency: str | None = None,
    is_test_mode: bool = False  # NEW PARAMETER
) -> int:
```

**Step 4: Update INSERT statement**

Location: `polymarket/performance/tracker.py` - In `log_decision()` method

```python
# Update the INSERT query to include is_test_mode
cursor.execute(
    """
    INSERT INTO trades (
        market_id, action, confidence, amount, timestamp,
        btc_price, technical_rsi, technical_macd, technical_trend,
        sentiment_score, sentiment_confidence,
        price_to_beat, time_remaining_seconds, is_end_phase,
        actual_probability, arbitrage_edge, arbitrage_urgency,
        is_test_mode  -- NEW COLUMN
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
    (
        market.id,
        decision.action,
        decision.confidence,
        float(decision.position_size),
        int(btc_data.timestamp.timestamp()),
        float(btc_data.price),
        technical.rsi,
        technical.macd_value,
        technical.trend,
        aggregated.final_score,
        aggregated.final_confidence,
        float(price_to_beat) if price_to_beat else None,
        time_remaining_seconds,
        1 if is_end_phase else 0,
        actual_probability,
        arbitrage_edge,
        arbitrage_urgency,
        1 if is_test_mode else 0  # NEW VALUE
    )
)
```

**Step 5: Test database migration**

Run: `python -c "from polymarket.performance.database import PerformanceDatabase; from polymarket.config import Settings; db = PerformanceDatabase(Settings()); print('Database initialized successfully')"`

Expected: "Database initialized successfully" (no errors)

**Step 6: Commit database changes**

```bash
git add polymarket/performance/database.py polymarket/performance/tracker.py
git commit -m "feat: add is_test_mode column to trades table

- Add is_test_mode BOOLEAN column to track test trades
- Add index for efficient test mode queries
- Update log_decision signature with is_test_mode parameter
- Update INSERT statement to include test mode flag

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Add TestModeConfig Class

**Files:**
- Modify: `scripts/auto_trade.py` (top of file, after imports)

**Step 1: Add dataclass import**

Location: `scripts/auto_trade.py` - In the import section (around line 20)

```python
from dataclasses import dataclass, field
```

**Step 2: Create TestModeConfig class**

Location: `scripts/auto_trade.py` - After imports, before AutoTrader class (around line 50)

```python
@dataclass
class TestModeConfig:
    """Test mode configuration for forced trading with safety bypasses.

    Test mode allows testing new signals by forcing trades on every market
    with strict bet limits. All safety filters are bypassed but data is
    still collected and sent to the AI.

    Attributes:
        enabled: Whether test mode is active (from TEST_MODE env var)
        max_bet_amount: Maximum bet per trade (default $1.00)
        min_confidence: Minimum AI confidence required (default 0.70)
        traded_markets: Set of market IDs already traded this session
    """
    enabled: bool = False
    max_bet_amount: Decimal = Decimal("1.0")
    min_confidence: float = 0.70
    traded_markets: set[str] = field(default_factory=set)
```

**Step 3: Initialize test mode in AutoTrader.__init__**

Location: `scripts/auto_trade.py` - In `AutoTrader.__init__()` method

```python
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
        logger.warning(
            "Test mode configuration",
            max_bet=f"${self.test_mode.max_bet_amount}",
            min_confidence=f"{self.test_mode.min_confidence:.0%}",
            one_bet_per_market=True,
            safety_filters="BYPASSED (data sent to AI)",
            coingecko_signals="INCLUDED"
        )
        logger.warning("=" * 60)
```

**Step 4: Test initialization**

Run: `TEST_MODE=true python scripts/auto_trade.py --once 2>&1 | head -20`

Expected: See startup banner with "🧪 TEST MODE ACTIVE"

**Step 5: Commit test mode config**

```bash
git add scripts/auto_trade.py
git commit -m "feat: add TestModeConfig class and initialization

- Create TestModeConfig dataclass with enabled, limits, tracking
- Read TEST_MODE environment variable on startup
- Display loud warning banner when test mode active
- Track traded markets in-memory to prevent duplicates

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Bypass Movement Threshold

**Files:**
- Modify: `scripts/auto_trade.py` (around line 792-802)

**Step 1: Locate movement threshold check**

Find this code block (around line 792):

```python
MIN_MOVEMENT_THRESHOLD = 100  # $100 minimum BTC movement
abs_diff = abs(diff)
if abs_diff < MIN_MOVEMENT_THRESHOLD:
    logger.info(
        "Skipping market - insufficient BTC movement",
        market_id=market.id,
        movement=f"${abs_diff:.2f}",
        threshold=f"${MIN_MOVEMENT_THRESHOLD}",
        reason="Wait for clearer directional signal"
    )
    return  # Skip this market, no trade
```

**Step 2: Add test mode bypass**

Replace the above code with:

```python
MIN_MOVEMENT_THRESHOLD = 100  # $100 minimum BTC movement
abs_diff = abs(diff)
if abs_diff < MIN_MOVEMENT_THRESHOLD:
    if not self.test_mode.enabled:
        logger.info(
            "Skipping market - insufficient BTC movement",
            market_id=market.id,
            movement=f"${abs_diff:.2f}",
            threshold=f"${MIN_MOVEMENT_THRESHOLD}",
            reason="Wait for clearer directional signal"
        )
        return  # Skip this market, no trade
    else:
        logger.info(
            "[TEST] Bypassing movement threshold",
            market_id=market.id,
            movement=f"${abs_diff:.2f}",
            threshold=f"${MIN_MOVEMENT_THRESHOLD}",
            reason="Test mode - data sent to AI for analysis"
        )
        # Continue to next check...
```

**Step 3: Test movement bypass**

Run: `TEST_MODE=true python scripts/auto_trade.py --once 2>&1 | grep -A2 "Bypassing movement"`

Expected: See "[TEST] Bypassing movement threshold" log

**Step 4: Commit movement bypass**

```bash
git add scripts/auto_trade.py
git commit -m "feat: bypass movement threshold in test mode

- Check test_mode.enabled before skipping on low movement
- Log [TEST] prefix when bypassing
- Data still collected and sent to AI for analysis

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Bypass Spread Check

**Files:**
- Modify: `scripts/auto_trade.py` (around line 852-860)

**Step 1: Locate spread check**

Find this code block (around line 852):

```python
if orderbook_analysis.spread_bps > 500:  # 5% spread
    logger.info(
        "Skipping trade - spread too wide",
        market_id=market.id,
        spread_bps=f"{orderbook_analysis.spread_bps:.1f}",
        liquidity_score=f"{orderbook_analysis.liquidity_score:.2f}",
        reason="Wide spread = poor execution quality"
    )
    return
```

**Step 2: Add test mode bypass**

Replace with:

```python
if orderbook_analysis.spread_bps > 500:  # 5% spread
    if not self.test_mode.enabled:
        logger.info(
            "Skipping trade - spread too wide",
            market_id=market.id,
            spread_bps=f"{orderbook_analysis.spread_bps:.1f}",
            liquidity_score=f"{orderbook_analysis.liquidity_score:.2f}",
            reason="Wide spread = poor execution quality"
        )
        return
    else:
        logger.info(
            "[TEST] Bypassing spread check",
            market_id=market.id,
            spread_bps=f"{orderbook_analysis.spread_bps:.1f}",
            threshold="500 bps",
            reason="Test mode - spread data sent to AI"
        )
```

**Step 3: Commit spread bypass**

```bash
git add scripts/auto_trade.py
git commit -m "feat: bypass spread check in test mode

- Check test_mode.enabled before skipping on wide spreads
- Log [TEST] prefix with spread percentage
- Spread data still sent to AI for informed decision

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Bypass Volume and Other Checks

**Files:**
- Modify: `scripts/auto_trade.py`

**Step 1: Locate volume confirmation check**

Find this code block (around line 805-814):

```python
# Volume confirmation for large moves (breakout detection)
if abs_diff > 200 and volume_data:  # $200+ move = potential breakout
    if not volume_data.is_high_volume:
        logger.info(
            "Skipping large move without volume confirmation",
            market_id=market.id,
            movement=f"${diff:+,.2f}",
            volume_ratio=f"{volume_data.volume_ratio:.2f}x",
            reason="Breakouts require volume > 1.5x average"
        )
        return  # Skip low-volume breakouts
```

**Step 2: Add test mode bypass for volume**

Replace with:

```python
# Volume confirmation for large moves (breakout detection)
if abs_diff > 200 and volume_data:  # $200+ move = potential breakout
    if not volume_data.is_high_volume:
        if not self.test_mode.enabled:
            logger.info(
                "Skipping large move without volume confirmation",
                market_id=market.id,
                movement=f"${diff:+,.2f}",
                volume_ratio=f"{volume_data.volume_ratio:.2f}x",
                reason="Breakouts require volume > 1.5x average"
            )
            return  # Skip low-volume breakouts
        else:
            logger.info(
                "[TEST] Bypassing volume check",
                market_id=market.id,
                movement=f"${diff:+,.2f}",
                volume_ratio=f"{volume_data.volume_ratio:.2f}x",
                reason="Test mode - volume data sent to AI"
            )
```

**Step 3: Locate timeframe alignment check**

Find this code block (around line 817-825):

```python
# Timeframe alignment check - don't trade against larger trend
if timeframe_analysis and timeframe_analysis.alignment == "CONFLICTING":
    logger.info(
        "Skipping trade - conflicting timeframes",
        market_id=market.id,
        daily_trend=timeframe_analysis.daily_trend,
        four_hour_trend=timeframe_analysis.four_hour_trend,
        reason="Don't trade against larger timeframe trend"
    )
    return
```

**Step 4: Add test mode bypass for timeframe**

Replace with:

```python
# Timeframe alignment check - don't trade against larger trend
if timeframe_analysis and timeframe_analysis.alignment == "CONFLICTING":
    if not self.test_mode.enabled:
        logger.info(
            "Skipping trade - conflicting timeframes",
            market_id=market.id,
            daily_trend=timeframe_analysis.daily_trend,
            four_hour_trend=timeframe_analysis.four_hour_trend,
            reason="Don't trade against larger timeframe trend"
        )
        return
    else:
        logger.info(
            "[TEST] Bypassing timeframe check",
            market_id=market.id,
            alignment=timeframe_analysis.alignment,
            daily_trend=timeframe_analysis.daily_trend,
            four_hour_trend=timeframe_analysis.four_hour_trend,
            reason="Test mode - timeframe data sent to AI"
        )
```

**Step 5: Locate market regime check**

Find this code block (around line 828-837):

```python
# Market regime check - skip unclear/volatile markets
if regime and regime.regime in ["UNCLEAR", "VOLATILE"]:
    logger.info(
        "Skipping trade - unfavorable market regime",
        market_id=market.id,
        regime=regime.regime,
        volatility=f"{regime.volatility:.2f}%",
        confidence=f"{regime.confidence:.2f}",
        reason="Only trade in trending or ranging markets"
    )
    return
```

**Step 6: Add test mode bypass for regime**

Replace with:

```python
# Market regime check - skip unclear/volatile markets
if regime and regime.regime in ["UNCLEAR", "VOLATILE"]:
    if not self.test_mode.enabled:
        logger.info(
            "Skipping trade - unfavorable market regime",
            market_id=market.id,
            regime=regime.regime,
            volatility=f"{regime.volatility:.2f}%",
            confidence=f"{regime.confidence:.2f}",
            reason="Only trade in trending or ranging markets"
        )
        return
    else:
        logger.info(
            "[TEST] Bypassing regime check",
            market_id=market.id,
            regime=regime.regime,
            volatility=f"{regime.volatility:.2f}%",
            reason="Test mode - regime data sent to AI"
        )
```

**Step 7: Commit all remaining bypasses**

```bash
git add scripts/auto_trade.py
git commit -m "feat: bypass volume, timeframe, and regime checks in test mode

- Volume: bypass low-volume breakout skip
- Timeframe: bypass conflicting timeframe skip
- Regime: bypass UNCLEAR/VOLATILE regime skip
- All data still collected and sent to AI
- Consistent [TEST] logging prefix

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Add Duplicate Market Prevention

**Files:**
- Modify: `scripts/auto_trade.py` (around line 870, before AI decision call)

**Step 1: Add duplicate check before AI decision**

Location: After orderbook analysis, before the market_dict creation (around line 870)

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
                    "[TEST] New market - proceeding with analysis",
                    market_id=market_id,
                    total_test_trades=len(self.test_mode.traded_markets)
                )

            # Build market data dict with ALL context
            # (existing code continues...)
```

**Step 2: Test duplicate prevention**

Create a simple test script:

```python
# test_duplicate_prevention.py
from scripts.auto_trade import TestModeConfig
from decimal import Decimal

config = TestModeConfig(enabled=True)
market_id = "test-market-123"

# First check - should allow
assert market_id not in config.traded_markets
config.traded_markets.add(market_id)

# Second check - should skip
assert market_id in config.traded_markets
print("✓ Duplicate prevention works")
```

Run: `python test_duplicate_prevention.py`

Expected: "✓ Duplicate prevention works"

**Step 3: Commit duplicate prevention**

```bash
git add scripts/auto_trade.py
git commit -m "feat: prevent duplicate trades in test mode

- Check traded_markets set before AI decision
- Skip with [TEST] log if market already traded
- Track total test trades for visibility

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Force AI Decision and Check Confidence

**Files:**
- Modify: `scripts/auto_trade.py` (around line 994, after AI decision call)
- Modify: `polymarket/trading/ai_decision.py`

**Step 1: Update AI service method signature**

Location: `polymarket/trading/ai_decision.py` - `make_decision()` method

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
    force_trade: bool = False  # NEW PARAMETER
) -> TradingDecision:
    """Generate trading decision using AI with optional forced trading."""
    try:
        client = self._get_client()

        # Build the prompt
        prompt = self._build_prompt(
            btc_price, technical_indicators, aggregated_sentiment,
            market_data, portfolio_value, orderbook_data,
            volume_data, timeframe_analysis, regime,
            arbitrage_opportunity, market_signals,
            force_trade  # Pass through to prompt builder
        )
```

**Step 2: Update _build_prompt signature**

Location: `polymarket/trading/ai_decision.py` - `_build_prompt()` method

```python
def _build_prompt(
    self,
    btc_price: BTCPriceData,
    technical: TechnicalIndicators,
    aggregated: AggregatedSentiment,
    market: dict,
    portfolio_value: Decimal,
    orderbook_data: "OrderbookData | None" = None,
    volume_data: VolumeData | None = None,
    timeframe_analysis: TimeframeAnalysis | None = None,
    regime: MarketRegime | None = None,
    arbitrage_opportunity: "ArbitrageOpportunity | None" = None,
    market_signals: "Any | None" = None,
    force_trade: bool = False  # NEW PARAMETER
) -> str:
    """Build the AI prompt with optional forced trading instruction."""
```

**Step 3: Add test mode instruction to prompt**

Location: `polymarket/trading/ai_decision.py` - In `_build_prompt()` method, after market_signals_context

```python
        # NEW: Test mode forced trading instruction
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
```

**Step 4: Insert force_trade_instruction into prompt**

Location: In the return statement of `_build_prompt()`, add after arbitrage_context

```python
        return f"""You are an autonomous trading bot for Polymarket BTC 15-minute up/down markets.
Use your reasoning tokens to carefully analyze all signals before making a decision.

{price_context}

{validation_rules}

{timing_context}

{momentum_context}

{regime_context}

{volume_context}

{timeframe_context}

{arbitrage_context}

{market_signals_context}

{force_trade_instruction}

CURRENT MARKET DATA:
...
"""
```

**Step 5: Update AI decision call in auto_trade.py**

Location: `scripts/auto_trade.py` - Around line 994

```python
            # Step 1: AI Decision - pass force_trade flag
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
                market_signals=market_signals,
                force_trade=self.test_mode.enabled  # NEW PARAMETER
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

**Step 6: Commit AI forcing logic**

```bash
git add polymarket/trading/ai_decision.py scripts/auto_trade.py
git commit -m "feat: force AI YES/NO decision in test mode

- Add force_trade parameter to make_decision and _build_prompt
- Add test mode instruction to AI prompt (no HOLD allowed)
- Force YES/NO based on sentiment if AI returns HOLD
- Check 70% confidence threshold before trading
- Override position size to $1.00

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Mark Trades and Update Database Tracking

**Files:**
- Modify: `scripts/auto_trade.py` (around line 986 and 1100)

**Step 1: Update performance log_decision call**

Location: `scripts/auto_trade.py` - Around line 986

```python
            # NOW log decision to performance tracker (only if validation passed)
            trade_id = -1
            try:
                trade_id = await self.performance_tracker.log_decision(
                    market=market,
                    decision=decision,
                    btc_data=btc_data,
                    technical=indicators,
                    aggregated=aggregated_sentiment,
                    price_to_beat=price_to_beat,
                    time_remaining_seconds=time_remaining,
                    is_end_phase=is_end_of_market,
                    actual_probability=arbitrage_opportunity.actual_probability if arbitrage_opportunity else None,
                    arbitrage_edge=arbitrage_opportunity.edge_percentage if arbitrage_opportunity else None,
                    arbitrage_urgency=arbitrage_opportunity.urgency if arbitrage_opportunity else None,
                    is_test_mode=self.test_mode.enabled  # NEW PARAMETER
                )
            except Exception as e:
                logger.error("Performance logging failed", error=str(e))
                # Continue trading - don't block on logging failures
                trade_id = -1
```

**Step 2: Mark market as traded after successful execution**

Location: `scripts/auto_trade.py` - After trade execution (around line 1100)

Find the section after trade execution and add:

```python
            # After successful trade execution
            if result and result.get("success"):
                # Test mode: Mark market as traded
                if self.test_mode.enabled:
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

**Step 3: Commit database tracking**

```bash
git add scripts/auto_trade.py
git commit -m "feat: track test trades in database and memory

- Pass is_test_mode flag to performance tracker
- Mark market in traded_markets set after execution
- Log test trade success with running count

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 9: Integration Testing

**Files:**
- Create: `test_test_mode.sh`

**Step 1: Create test script**

```bash
#!/bin/bash
# Test mode integration test

set -e

echo "Testing test mode integration..."

# Test 1: Verify startup banner
echo "Test 1: Startup banner"
TEST_MODE=true timeout 10s python scripts/auto_trade.py --once 2>&1 | grep "TEST MODE ACTIVE" || {
    echo "❌ FAIL: No test mode banner"
    exit 1
}
echo "✓ Test mode banner displayed"

# Test 2: Verify database schema
echo "Test 2: Database schema"
python -c "
import sqlite3
conn = sqlite3.connect('data/performance.db')
cursor = conn.cursor()
cursor.execute('PRAGMA table_info(trades)')
columns = [col[1] for col in cursor.fetchall()]
assert 'is_test_mode' in columns, 'is_test_mode column missing'
print('✓ is_test_mode column exists')
"

# Test 3: Verify test mode detection
echo "Test 3: Test mode detection"
python -c "
import os
os.environ['TEST_MODE'] = 'true'
from scripts.auto_trade import TestModeConfig
config = TestModeConfig(enabled=os.getenv('TEST_MODE', '').lower() == 'true')
assert config.enabled == True, 'Test mode not enabled'
assert config.max_bet_amount == 1.0, 'Max bet not $1'
assert config.min_confidence == 0.70, 'Min confidence not 70%'
print('✓ Test mode config correct')
"

echo ""
echo "All tests passed! ✓"
```

**Step 2: Make executable and run**

Run: `chmod +x test_test_mode.sh && ./test_test_mode.sh`

Expected: "All tests passed! ✓"

**Step 3: Commit test script**

```bash
git add test_test_mode.sh
git commit -m "test: add integration test for test mode

- Verify startup banner displays
- Check database schema has is_test_mode column
- Validate test mode configuration

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 10: Documentation and Usage Guide

**Files:**
- Create: `docs/TEST_MODE_USAGE.md`

**Step 1: Create usage guide**

```markdown
# Test Mode Usage Guide

## Overview

Test mode forces trading on every 15-minute BTC market with strict $1 bet limits to test new signals and AI improvements.

## Activation

```bash
TEST_MODE=true python scripts/auto_trade.py
```

## What Test Mode Does

**Bypasses:**
- ✅ Movement threshold ($100 minimum)
- ✅ Spread check (5% maximum)
- ✅ Volume confirmation
- ✅ Timeframe alignment
- ✅ Market regime filter

**Enforces:**
- ✅ $1.00 maximum bet per trade
- ✅ 70% minimum AI confidence
- ✅ One bet per market (no duplicates)
- ✅ AI must pick YES or NO (no HOLD)

**Includes:**
- ✅ All CoinGecko signals (funding, premium, volume)
- ✅ All bypassed data sent to AI for analysis
- ✅ Database tracking with `is_test_mode = TRUE`

## Monitoring

### Watch Logs

```bash
tail -f trading-bot.log | grep TEST
```

### Query Test Trades

```sql
-- View all test trades
SELECT * FROM trades WHERE is_test_mode = 1 ORDER BY timestamp DESC;

-- Test mode performance
SELECT
    COUNT(*) as trades,
    SUM(CASE WHEN win = 1 THEN 1 ELSE 0 END) as wins,
    ROUND(AVG(CASE WHEN win = 1 THEN 1.0 ELSE 0.0 END) * 100, 2) as win_rate,
    ROUND(SUM(profit_loss), 2) as total_pl
FROM trades
WHERE is_test_mode = 1;

-- Compare test vs production
SELECT
    is_test_mode,
    COUNT(*) as trades,
    ROUND(AVG(confidence), 3) as avg_confidence,
    ROUND(AVG(CASE WHEN win = 1 THEN 1.0 ELSE 0.0 END) * 100, 2) as win_rate
FROM trades
GROUP BY is_test_mode;
```

## Risk Management

**Maximum Loss:**
- Per trade: $1.00
- Per hour: ~$8 (4 markets × 50% trade rate)
- Per day: ~$48
- Per week: ~$336

**Start Small:**
1. Run for 2 hours (max $8 loss)
2. Check win rate and analysis
3. Extend if results promising

## Expected Logs

```
==============================================================
🧪 TEST MODE ACTIVE - SAFETY FILTERS BYPASSED
==============================================================
Max bet per trade: $1.0
Min AI confidence: 70%
One bet per market: ENABLED
==============================================================

[TEST] New market - proceeding with analysis
[TEST] Bypassing movement threshold: $23.00 < $100.00
[TEST] Bypassing spread check: 8.2% > 5.0%
[TEST] AI Decision: YES, confidence 0.78
[TEST] Position size overridden: $8.50 → $1.00
[TEST] ✓ TRADE EXECUTED
[TEST] Total test trades this session: 1
```

## Stopping Test Mode

```bash
# Stop bot
pkill -f auto_trade.py

# Resume normal mode (remove TEST_MODE)
python scripts/auto_trade.py
```

## Analysis

After testing period:

1. **Check Win Rate**: Should be > 50%
2. **Analyze Signals**: Which CoinGecko signals correlated with wins?
3. **Review Confidence**: Were 70%+ trades actually better?
4. **Compare P&L**: Test vs production performance

## Troubleshooting

**Issue: No trades executing**
- Check logs for confidence threshold (< 70%)
- Verify AI is returning YES/NO (not HOLD)

**Issue: Duplicate trades**
- Check `traded_markets` set is working
- Verify database has `is_test_mode` flag

**Issue: Trades > $1**
- Check position_size override logic
- Verify test_mode.max_bet_amount = 1.0
```

**Step 2: Commit usage guide**

```bash
git add docs/TEST_MODE_USAGE.md
git commit -m "docs: add test mode usage guide

- Activation instructions
- Monitoring commands
- Risk management guidelines
- Expected log output
- Analysis queries

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 11: Final Verification

**Files:**
- N/A (verification only)

**Step 1: Verify all components**

Run comprehensive check:

```bash
# 1. Database schema
python -c "
from polymarket.performance.database import PerformanceDatabase
from polymarket.config import Settings
db = PerformanceDatabase(Settings())
print('✓ Database schema updated')
"

# 2. Test mode config
TEST_MODE=true python -c "
import os
from scripts.auto_trade import TestModeConfig
config = TestModeConfig(enabled=os.getenv('TEST_MODE', '').lower() == 'true')
assert config.enabled
print('✓ Test mode config works')
"

# 3. Run integration tests
./test_test_mode.sh

# 4. Check git status
git status
```

**Step 2: Review implementation checklist**

From design document:

- [x] Database schema updated
- [x] Test mode initialization
- [x] Safety filter bypass logic
- [x] Duplicate market prevention
- [x] AI decision forcing
- [x] Position size override
- [x] Database tracking integration
- [x] AI prompt modification
- [x] Comprehensive logging
- [x] Testing and verification

**Step 3: Create final summary commit**

```bash
git commit --allow-empty -m "feat: test mode implementation complete

Test mode now fully functional with:
- Environment variable activation (TEST_MODE=true)
- All safety filters bypassed with data collection
- Forced AI YES/NO decisions (no HOLD)
- $1 strict bet limit with 70% min confidence
- One bet per market tracking
- Database tracking with is_test_mode flag
- CoinGecko signals included in analysis
- Comprehensive [TEST] logging

Ready for production testing.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Implementation Complete!

**Total Tasks:** 11
**Estimated Time:** 2-3 hours
**Commits:** 12 atomic commits

**Next Steps:**

1. Review implementation thoroughly
2. Start bot with `TEST_MODE=true`
3. Monitor for 2 hours (max $8 loss)
4. Analyze results
5. Scale up or tune based on performance

**Key Files Modified:**
- `polymarket/performance/database.py` - Database schema
- `polymarket/performance/tracker.py` - Tracking integration
- `scripts/auto_trade.py` - Main trading logic
- `polymarket/trading/ai_decision.py` - AI prompt modification

**Key Files Created:**
- `docs/TEST_MODE_USAGE.md` - Usage guide
- `test_test_mode.sh` - Integration tests

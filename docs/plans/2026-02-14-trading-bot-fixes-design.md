# Trading Bot Performance Fixes - Design Document

**Date:** 2026-02-14
**Status:** Approved
**Problem:** Bot has 10.6% win rate (90% loss rate) with -$74.88 P/L over 155 trades

---

## Executive Summary

This design addresses the catastrophic 10.6% win rate by implementing 5 interconnected improvements:

1. **Paper Trading Mode** - TEST_MODE stops before real money execution
2. **Signal Lag Detection** - Auto-HOLD when market sentiment contradicts BTC movement
3. **Odds Polling System** - Only trade when one side > 75% odds
4. **Conflict-Based Confidence Reduction** - Reduce confidence or HOLD on signal conflicts
5. **Remove Arbitrage Gate** - Keep calculation but don't block trades

**Expected Outcome:** Win rate improvement from 10.6% to > 50% (validated through paper trading)

---

## Architecture Overview

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    TRADING CYCLE (60s)                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Market Discovery                                         │
│     └─► Filter markets < 5 min remaining                     │
│                                                              │
│  2. Background: Odds Poller (every 60s)                      │
│     └─► Check cached odds: Either side > 75%?               │
│         ├─► NO: Skip market early                           │
│         └─► YES: Continue                                    │
│                                                              │
│  3. Data Collection (parallel)                               │
│     ├─► BTC price (Polymarket → CoinGecko → Binance)        │
│     ├─► Social sentiment (Fear/Greed, trending, votes)      │
│     ├─► Market microstructure (orderbook, whales)           │
│     ├─► Funding rates                                       │
│     ├─► BTC dominance                                       │
│     └─► Volume data                                         │
│                                                              │
│  4. Technical Analysis                                       │
│     └─► Calculate RSI, MACD, EMA, volume, velocity          │
│                                                              │
│  5. ⚠️ NEW: Signal Lag Detection                            │
│     └─► Compare BTC direction vs market sentiment           │
│         ├─► Contradiction + high confidence: HOLD           │
│         └─► Aligned: Continue                               │
│                                                              │
│  6. Signal Aggregation                                       │
│     └─► Combine all signals with weights                    │
│                                                              │
│  7. AI Decision (GPT-5-Nano with reasoning)                  │
│     ├─► Input: All signals + arbitrage edge (no gate)       │
│     └─► Output: YES/NO/HOLD + confidence + reasoning        │
│                                                              │
│  8. ⚠️ NEW: Conflict Detection                              │
│     └─► Analyze signal conflicts                            │
│         ├─► SEVERE (3+ conflicts): AUTO-HOLD                │
│         ├─► MODERATE (2 conflicts): -0.20 confidence        │
│         └─► MINOR (1 conflict): -0.10 confidence            │
│                                                              │
│  9. ⚠️ NEW: JIT Odds Validation                             │
│     └─► Fetch fresh odds before execution                   │
│         └─► Chosen side > 75%? NO: HOLD                     │
│                                                              │
│ 10. Risk Validation                                          │
│     └─► Check portfolio limits, position sizing             │
│                                                              │
│ 11. ⚠️ NEW: Paper Trading Fork                              │
│     ├─► TEST_MODE + paper_trading = true:                   │
│     │   ├─► STOP before order placement                     │
│     │   ├─► Log paper trade to database                     │
│     │   └─► Send detailed Telegram alert                    │
│     │                                                        │
│     └─► PRODUCTION:                                          │
│         ├─► Execute real order                              │
│         ├─► Verify execution                                │
│         └─► Send Telegram notification                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Feature 1: Paper Trading Mode

### Purpose
Enable risk-free testing of trading logic without spending real money.

### Implementation

**Configuration:**
```python
# auto_trade.py TestModeConfig
class TestModeConfig:
    enabled: bool = False
    paper_trading: bool = True  # NEW: No real money
    min_confidence: float = 0.70
    min_odds_threshold: float = 0.75  # NEW: Odds requirement
    traded_markets: set[str] = field(default_factory=set)
```

**Execution Flow:**
```python
# auto_trade.py:1491-1799 _execute_trade()
async def _execute_trade(self, ...):
    # ... (all analysis and validation runs normally)

    # NEW: Paper trading fork
    if self.test_mode.enabled and self.test_mode.paper_trading:
        # STOP HERE - don't place real order
        await self._execute_paper_trade(
            market, decision, amount, token_name,
            btc_current, btc_price_to_beat,
            conflict_analysis, lag_detected, odds_snapshot
        )
        return  # Exit before real order placement

    # Real trading continues here...
```

**Paper Trade Logging:**
- New table: `paper_trades` (mirrors `trades` schema)
- Simulated fill: Uses current market `best_bid/best_ask`
- Tracks all signal analysis for performance review
- Can be "settled" later using real market outcomes

**Telegram Alert Format:**
```
🧪 PAPER TRADE SIGNAL
━━━━━━━━━━━━━━━━━━━━
📊 Market: btc-updown-15m-1771234500
📈 Direction: UP (YES token)
💵 Position: $8.50 @ 0.82 odds
⏰ Time Remaining: 12m 30s

🎯 SIGNAL ANALYSIS:
✅ Technical: BULLISH (RSI: 42, MACD: +0.15)
✅ Sentiment: BULLISH (score: +0.45, conf: 0.82)
✅ Odds Check: YES = 82% (PASS > 75%)
⚠️ Timeframes: 3/4 ALIGNED (minor conflict)
✅ Signal Lag: NO LAG DETECTED

🤖 AI REASONING:
"Technical indicators show bullish momentum with RSI
recovering from oversold. Market sentiment aligned.
BTC moved +$150 from price-to-beat, momentum confirms
UP direction. Moderate confidence due to 15m timeframe
showing slight divergence."

📊 CONFIDENCE: 0.78
⚠️ CONFLICTS: MINOR (1 detected)
   - Timeframe 15m: NEUTRAL vs overall UP

━━━━━━━━━━━━━━━━━━━━
```

---

## Feature 2: Signal Lag Detection

### Purpose
Catch when market sentiment lags behind actual BTC price movement, preventing bets on stale data.

### Root Cause
- Polymarket odds update slower than BTC price
- Bot collects sentiment over 2-5 minutes
- BTC can move significantly during collection window
- Example: BTC bounces +$200 but sentiment still bearish from 5 min ago

### Implementation

**Detector Logic:**
```python
# polymarket/trading/signal_lag_detector.py
def detect_signal_lag(
    btc_actual_direction: str,  # "UP" or "DOWN" (from price-to-beat)
    market_sentiment_direction: str,  # "BULLISH" or "BEARISH"
    sentiment_confidence: float
) -> tuple[bool, str]:
    """
    Detect when market sentiment lags behind actual BTC movement.

    Returns: (is_lagging, reason)
    """
    # Map sentiment to direction
    sentiment_dir = "UP" if market_sentiment_direction == "BULLISH" else "DOWN"

    # Check for contradiction
    if btc_actual_direction != sentiment_dir:
        # Only flag if sentiment is confident (> 0.6)
        if sentiment_confidence > 0.6:
            reason = (
                f"SIGNAL LAG DETECTED: BTC moving {btc_actual_direction} "
                f"but market sentiment is {market_sentiment_direction} "
                f"(confidence: {sentiment_confidence:.2f}). "
                f"Market odds lagging behind reality."
            )
            return True, reason

    return False, "No lag detected"
```

**Integration Point:**
```python
# auto_trade.py _process_market()
# After price-to-beat calculation (line ~906)

if price_to_beat:
    btc_direction = "UP" if btc_data.price > price_to_beat else "DOWN"
    sentiment_direction = "BULLISH" if aggregated_sentiment.final_score > 0 else "BEARISH"

    is_lagging, lag_reason = detect_signal_lag(
        btc_direction,
        sentiment_direction,
        aggregated_sentiment.final_confidence
    )

    if is_lagging:
        if not self.test_mode.enabled:
            logger.warning("Skipping trade due to signal lag", reason=lag_reason)
            return  # HOLD - don't trade contradictions
        else:
            logger.info("[TEST] Signal lag detected - sending to AI anyway", reason=lag_reason)
```

**Why This Works:**
- Catches obvious contradictions causing 90% loss rate
- Simple, deterministic, easy to debug
- Only flags confident contradictions (> 0.6 confidence)
- In TEST mode: Logs warning but continues (for analysis)
- In PRODUCTION: Hard HOLD (prevents trade)

---

## Feature 3: Odds Polling System

### Purpose
Only trade when market odds strongly favor one side (> 75%), reducing noise trades on coin-flip markets.

### Architecture: Hybrid Approach

**Background Polling (Early Filter):**
- Runs every 60 seconds
- Eliminates bad markets before AI analysis
- Saves OpenAI API costs on non-qualifying markets

**JIT Validation (Stale Data Prevention):**
- Fetches fresh odds immediately before execution
- Ensures odds haven't shifted since background poll
- Prevents trading on stale data

### Implementation

**Service: `MarketOddsPoller`**
```python
# polymarket/trading/odds_poller.py
class MarketOddsPoller:
    """
    Background service that polls Polymarket API for current market odds.
    Runs every 60 seconds, stores odds in shared state.
    """

    def __init__(self, client: PolymarketClient):
        self.client = client
        self.current_odds: dict[str, OddsSnapshot] = {}  # market_id -> odds
        self._lock = asyncio.Lock()

    async def start_polling(self):
        """Run polling loop every 60 seconds."""
        while True:
            try:
                await self._poll_current_market()
                await asyncio.sleep(60)
            except Exception as e:
                logger.error("Odds polling failed", error=str(e))

    async def _poll_current_market(self):
        """Fetch odds for current active market."""
        # Discover current BTC 15-min market
        market = self.client.discover_btc_15min_market()

        # Fetch fresh market data with odds
        fresh_market = self.client.get_market_by_slug(market.slug)

        # Extract odds
        yes_odds = fresh_market.best_bid  # YES/UP token odds
        no_odds = 1.0 - yes_odds  # NO/DOWN = complement

        # Store snapshot
        async with self._lock:
            self.current_odds[market.id] = OddsSnapshot(
                market_id=market.id,
                market_slug=market.slug,
                yes_odds=yes_odds,
                no_odds=no_odds,
                timestamp=datetime.now(),
                yes_qualifies=(yes_odds > 0.75),
                no_qualifies=(no_odds > 0.75)
            )

    async def get_odds(self, market_id: str) -> OddsSnapshot | None:
        """Get cached odds for market."""
        async with self._lock:
            return self.current_odds.get(market_id)
```

**Data Model:**
```python
@dataclass
class OddsSnapshot:
    market_id: str
    market_slug: str
    yes_odds: float  # 0.0-1.0 (best_bid)
    no_odds: float   # 0.0-1.0 (1 - best_bid)
    timestamp: datetime
    yes_qualifies: bool  # > 0.75
    no_qualifies: bool   # > 0.75
```

**Integration:**
```python
# auto_trade.py __init__
self.odds_poller = MarketOddsPoller(self.client)

# auto_trade.py initialize()
odds_task = asyncio.create_task(self.odds_poller.start_polling())
self.background_tasks.append(odds_task)

# auto_trade.py _process_market()
# 1. Early filter (after market discovery)
cached_odds = await self.odds_poller.get_odds(market.id)
if cached_odds and not (cached_odds.yes_qualifies or cached_odds.no_qualifies):
    logger.info("Skipping market - neither side > 75%",
                yes=cached_odds.yes_odds, no=cached_odds.no_odds)
    return

# 2. JIT validation (after AI decision)
fresh_market = await self._get_fresh_market_data(market.id)
yes_odds_fresh = fresh_market.best_bid
no_odds_fresh = 1.0 - yes_odds_fresh

if decision.action == "YES" and yes_odds_fresh <= 0.75:
    logger.info("Skipping trade - YES odds below threshold", odds=yes_odds_fresh)
    return
elif decision.action == "NO" and no_odds_fresh <= 0.75:
    logger.info("Skipping trade - NO odds below threshold", odds=no_odds_fresh)
    return
```

**Why 75% Threshold:**
- Market already favors one side (not a coin flip)
- Higher probability = higher confidence in outcome
- Reduces noise trades on unclear markets
- Still allows contrarian opportunities when odds are wrong

---

## Feature 4: Conflict-Based Confidence Reduction

### Purpose
Prevent overconfident trades (72-76% confidence) when signals contradict each other.

### Severity Classification

| Severity | Triggers | Penalty | Action |
|----------|----------|---------|--------|
| **SEVERE** | 3+ conflicts OR timeframes CONFLICTING | None | AUTO-HOLD |
| **MODERATE** | 2 conflicts | -0.20 | Reduce confidence |
| **MINOR** | 1 conflict | -0.10 | Reduce confidence |
| **NONE** | 0 conflicts | 0.00 | No change |

### Implementation

**Detector Service:**
```python
# polymarket/trading/conflict_detector.py
from enum import Enum
from dataclasses import dataclass

class ConflictSeverity(Enum):
    NONE = "NONE"
    MINOR = "MINOR"
    MODERATE = "MODERATE"
    SEVERE = "SEVERE"

@dataclass
class ConflictAnalysis:
    severity: ConflictSeverity
    confidence_penalty: float
    should_hold: bool
    conflicts_detected: list[str]

class SignalConflictDetector:
    """Detects and classifies conflicts between trading signals."""

    def analyze_conflicts(
        self,
        btc_direction: str,  # "UP" or "DOWN"
        technical_trend: str,
        sentiment_direction: str,
        regime_trend: str | None,
        timeframe_alignment: str | None,
        market_signals_direction: str | None,
        market_signals_confidence: float | None
    ) -> ConflictAnalysis:
        """Analyze all signals for conflicts and classify severity."""
        conflicts = []

        # Map directions to UP/DOWN
        technical_dir = self._map_to_direction(technical_trend)
        sentiment_dir = self._map_to_direction(sentiment_direction)
        regime_dir = self._map_to_direction(regime_trend) if regime_trend else None
        market_dir = self._map_to_direction(market_signals_direction) if market_signals_direction else None

        # Check conflicts
        if technical_dir and technical_dir != btc_direction:
            conflicts.append(f"Technical ({technical_trend}) vs BTC actual ({btc_direction})")

        if sentiment_dir and sentiment_dir != btc_direction:
            conflicts.append(f"Sentiment ({sentiment_direction}) vs BTC actual ({btc_direction})")

        if regime_dir and regime_dir != btc_direction:
            conflicts.append(f"Regime ({regime_trend}) vs BTC actual ({btc_direction})")

        if market_dir and market_signals_confidence and market_signals_confidence > 0.6:
            if market_dir != btc_direction:
                conflicts.append(
                    f"Market Signals ({market_signals_direction}, {market_signals_confidence:.2f}) "
                    f"vs BTC actual ({btc_direction})"
                )

        if timeframe_alignment == "CONFLICTING":
            conflicts.append("Timeframes CONFLICTING (don't trade against larger trend)")

        # Classify severity
        severity = self._classify_severity(len(conflicts), timeframe_alignment)

        # Determine action
        if severity == ConflictSeverity.SEVERE:
            return ConflictAnalysis(
                severity=severity,
                confidence_penalty=0.0,
                should_hold=True,
                conflicts_detected=conflicts
            )
        elif severity == ConflictSeverity.MODERATE:
            return ConflictAnalysis(
                severity=severity,
                confidence_penalty=-0.20,
                should_hold=False,
                conflicts_detected=conflicts
            )
        elif severity == ConflictSeverity.MINOR:
            return ConflictAnalysis(
                severity=severity,
                confidence_penalty=-0.10,
                should_hold=False,
                conflicts_detected=conflicts
            )
        else:
            return ConflictAnalysis(
                severity=ConflictSeverity.NONE,
                confidence_penalty=0.0,
                should_hold=False,
                conflicts_detected=[]
            )

    def _classify_severity(self, num_conflicts: int, timeframe_alignment: str | None) -> ConflictSeverity:
        """Classify conflict severity based on number and type."""
        if num_conflicts >= 3 or timeframe_alignment == "CONFLICTING":
            return ConflictSeverity.SEVERE
        elif num_conflicts == 2:
            return ConflictSeverity.MODERATE
        elif num_conflicts == 1:
            return ConflictSeverity.MINOR
        else:
            return ConflictSeverity.NONE
```

**Integration:**
```python
# auto_trade.py _process_market()
# After AI decision, before risk validation

conflict_detector = SignalConflictDetector()
conflict_analysis = conflict_detector.analyze_conflicts(
    btc_direction="UP" if btc_data.price > price_to_beat else "DOWN",
    technical_trend=indicators.trend,
    sentiment_direction="BULLISH" if aggregated_sentiment.final_score > 0 else "BEARISH",
    regime_trend=regime.trend_direction if regime else None,
    timeframe_alignment=timeframe_analysis.alignment_score if timeframe_analysis else None,
    market_signals_direction=market_signals.direction if market_signals else None,
    market_signals_confidence=market_signals.confidence if market_signals else None
)

# Apply conflict analysis
if conflict_analysis.should_hold:
    logger.warning(
        "AUTO-HOLD due to SEVERE signal conflicts",
        severity=conflict_analysis.severity.value,
        conflicts=conflict_analysis.conflicts_detected
    )
    return  # Don't trade

# Apply confidence penalty
if conflict_analysis.confidence_penalty != 0.0:
    original_confidence = decision.confidence
    decision.confidence += conflict_analysis.confidence_penalty
    decision.confidence = max(0.0, min(1.0, decision.confidence))

    logger.info(
        "Applied conflict penalty",
        original=f"{original_confidence:.2f}",
        penalty=f"{conflict_analysis.confidence_penalty:+.2f}",
        final=f"{decision.confidence:.2f}",
        conflicts=conflict_analysis.conflicts_detected
    )
```

---

## Feature 5: Remove Arbitrage Gate

### Purpose
Arbitrage edge is useful context but too restrictive as a hard requirement.

### Changes

**Remove Gate in `auto_trade.py`:**
```python
# REMOVE these lines (1624-1634):
if self.test_mode.enabled and arbitrage_opportunity:
    arb_edge = arbitrage_opportunity.edge_percentage
    if arb_edge < self.test_mode.min_arbitrage_edge:
        logger.info("Skipping trade - arbitrage edge below minimum")
        return
```

**Keep Calculation:**
- Still calculate arbitrage edge
- Still pass to AI in prompt
- AI can factor it into confidence naturally
- But don't block trades solely on edge

---

## Database Schema Changes

### New Table: `paper_trades`

```sql
CREATE TABLE paper_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    market_id TEXT NOT NULL,
    market_slug TEXT NOT NULL,
    question TEXT,
    action TEXT NOT NULL,  -- 'YES' or 'NO'
    confidence REAL NOT NULL,
    reasoning TEXT,

    -- Execution details
    executed_price REAL NOT NULL,
    position_size REAL NOT NULL,
    simulated_shares REAL NOT NULL,

    -- Market context
    btc_price_current REAL,
    btc_price_to_beat REAL,
    time_remaining_seconds INTEGER,

    -- Signal analysis (NEW)
    signal_lag_detected BOOLEAN DEFAULT 0,
    signal_lag_reason TEXT,
    conflict_severity TEXT,  -- 'NONE', 'MINOR', 'MODERATE', 'SEVERE'
    conflicts_list TEXT,  -- JSON array of conflict descriptions
    odds_yes REAL,
    odds_no REAL,
    odds_qualified BOOLEAN,  -- Did chosen side meet > 75% threshold?

    -- Outcome (filled during settlement)
    actual_outcome TEXT,  -- 'YES' or 'NO'
    is_win BOOLEAN,
    profit_loss REAL,
    settled_at TEXT
);

CREATE INDEX idx_paper_trades_timestamp ON paper_trades(timestamp);
CREATE INDEX idx_paper_trades_market ON paper_trades(market_slug);
```

### Modify Existing `trades` Table

```sql
-- Add new tracking columns
ALTER TABLE trades ADD COLUMN signal_lag_detected BOOLEAN DEFAULT 0;
ALTER TABLE trades ADD COLUMN signal_lag_reason TEXT;
ALTER TABLE trades ADD COLUMN conflict_severity TEXT;
ALTER TABLE trades ADD COLUMN conflicts_list TEXT;
ALTER TABLE trades ADD COLUMN odds_yes REAL;
ALTER TABLE trades ADD COLUMN odds_no REAL;
```

---

## Testing Strategy

### Phase 1: Unit Tests

```python
# tests/test_signal_lag_detector.py
def test_lag_detection_contradiction():
    """Test lag detector catches BTC UP + BEARISH sentiment."""
    is_lagging, reason = detect_signal_lag("UP", "BEARISH", 0.75)
    assert is_lagging == True
    assert "SIGNAL LAG DETECTED" in reason

def test_lag_detection_aligned():
    """Test no lag when signals align."""
    is_lagging, reason = detect_signal_lag("UP", "BULLISH", 0.75)
    assert is_lagging == False

# tests/test_conflict_detector.py
def test_severe_conflict_classification():
    """Test 3+ conflicts trigger SEVERE severity."""
    analysis = detector.analyze_conflicts(
        btc_direction="UP",
        technical_trend="BEARISH",
        sentiment_direction="BEARISH",
        regime_trend="TRENDING DOWN",
        timeframe_alignment="NEUTRAL",
        market_signals_direction="bearish",
        market_signals_confidence=0.75
    )
    assert analysis.severity == ConflictSeverity.SEVERE
    assert analysis.should_hold == True
    assert len(analysis.conflicts_detected) >= 3

def test_moderate_conflict_penalty():
    """Test 2 conflicts apply -0.20 penalty."""
    analysis = detector.analyze_conflicts(
        btc_direction="UP",
        technical_trend="BEARISH",
        sentiment_direction="BEARISH",
        regime_trend=None,
        timeframe_alignment="ALIGNED",
        market_signals_direction=None,
        market_signals_confidence=None
    )
    assert analysis.severity == ConflictSeverity.MODERATE
    assert analysis.confidence_penalty == -0.20
    assert analysis.should_hold == False

# tests/test_odds_poller.py
async def test_odds_polling_threshold():
    """Test odds poller correctly identifies qualifying markets."""
    # Mock market with 82% YES odds
    mock_market = create_mock_market(best_bid=0.82)

    poller = MarketOddsPoller(mock_client)
    await poller._poll_current_market()

    snapshot = await poller.get_odds(mock_market.id)
    assert snapshot.yes_odds == 0.82
    assert snapshot.no_odds == 0.18
    assert snapshot.yes_qualifies == True  # > 0.75
    assert snapshot.no_qualifies == False  # < 0.75
```

### Phase 2: Integration Tests

```python
# tests/test_paper_trading_flow.py
async def test_paper_trade_execution():
    """Test full paper trading flow with Telegram alert."""
    # Setup
    settings = Settings()
    trader = AutoTrader(settings, interval=60)
    trader.test_mode.enabled = True
    trader.test_mode.paper_trading = True

    # Mock dependencies
    with patch.object(trader.client, 'place_order') as mock_order:
        with patch.object(trader.telegram_bot, 'send_message') as mock_telegram:
            # Run single cycle
            await trader.run_once()

            # Verify no real orders placed
            assert mock_order.call_count == 0

            # Verify paper trade logged
            db = trader.performance_tracker.db
            paper_trades = db.conn.execute("SELECT * FROM paper_trades").fetchall()
            assert len(paper_trades) > 0

            # Verify Telegram alert sent
            assert mock_telegram.call_count >= 1
            alert = mock_telegram.call_args[0][0]
            assert "🧪 PAPER TRADE SIGNAL" in alert
            assert "SIGNAL ANALYSIS" in alert
            assert "AI REASONING" in alert
```

### Phase 3: Live Paper Trading (Manual)

**Duration:** 24-48 hours
**Environment:** Production with `TEST_MODE=true` and `paper_trading=true`

**Metrics to Track:**
- Total potential trades (paper trades logged)
- Signal lag detection rate (% of cycles flagged)
- Conflict detection rate (MINOR/MODERATE/SEVERE distribution)
- Odds qualification rate (% of markets passing 75% threshold)
- Paper win rate (after settlement)

**Success Criteria:**
- ✅ Zero real money spent
- ✅ All paper trades logged with complete signal analysis
- ✅ Telegram alerts sent for every potential trade
- ✅ Signal lag detector prevents obvious contradictions
- ✅ Conflict detector reduces confidence or HOLDs appropriately
- ✅ Odds poller only allows trades when > 75% threshold met
- ✅ **Win rate improves from 10.6% baseline (target: > 50%)**

### Phase 4: Gradual Rollout

1. **Week 1:** Paper trading (48 hours minimum)
2. **Week 2:** If win rate > 50%, enable real trading with $5 min bet
3. **Week 3:** If profitable, increase to $5-$15 range
4. **Week 4:** Scale up to Kelly Criterion sizing

---

## Risk Mitigation

### Risk: Paper trading has bugs that allow real trades
**Mitigation:**
- Hard check: `if paper_trading: return` before order placement
- Unit tests verify `place_order` never called in paper mode
- Manual verification of first 10 paper trades

### Risk: Odds polling fails, causing all trades to be skipped
**Mitigation:**
- Graceful fallback: If polling fails, log error but continue cycle
- JIT validation still runs even if background poll fails
- Monitor polling health via Telegram alerts

### Risk: Signal lag detector too aggressive, blocks good trades
**Mitigation:**
- Only flags contradictions with > 0.6 confidence
- Track lag detection rate in paper trading phase
- Adjust threshold if > 30% of cycles flagged

### Risk: Conflict detector too strict, never trades
**Mitigation:**
- SEVERE only triggers on 3+ conflicts or timeframe CONFLICTING
- MODERATE/MINOR apply penalties, don't block
- Track conflict distribution in paper trading

---

## Implementation Plan

**Files to Create:**
- `polymarket/trading/signal_lag_detector.py`
- `polymarket/trading/conflict_detector.py`
- `polymarket/trading/odds_poller.py`
- `tests/test_signal_lag_detector.py`
- `tests/test_conflict_detector.py`
- `tests/test_odds_poller.py`
- `tests/test_paper_trading_flow.py`

**Files to Modify:**
- `scripts/auto_trade.py` - Integrate all features
- `polymarket/performance/tracker.py` - Add paper trade logging
- `polymarket/telegram/bot.py` - Add paper trade alert format
- `polymarket/config.py` - Add new config options
- `data/schema.sql` - Add paper_trades table, modify trades table

**Migration Script:**
- `scripts/migrations/add_paper_trading_support.py`

---

## Rollout Checklist

- [ ] Implement all 5 features
- [ ] Write and pass all unit tests
- [ ] Write and pass integration tests
- [ ] Run paper trading for 48 hours
- [ ] Analyze paper trade performance
- [ ] If win rate > 50%: Enable real trading with $5 bets
- [ ] Monitor for 7 days
- [ ] If profitable: Scale up position sizing

---

## Expected Outcomes

**Immediate (Paper Trading):**
- Zero risk of losing real money
- Full visibility into what trades would be placed
- Complete signal analysis for every potential trade
- Data to validate win rate improvement

**Short-term (Week 1-2):**
- Win rate improvement from 10.6% to > 50%
- Signal lag detection prevents obvious contradiction trades
- Conflict detection reduces overconfident bets
- Odds polling eliminates coin-flip markets

**Long-term (Week 3-4):**
- Profitable trading with real money
- Scalable to larger position sizes
- Clear understanding of which signal combinations work best

---

## Design Approval

**Status:** ✅ APPROVED
**Date:** 2026-02-14
**Approved By:** User

**Next Steps:**
1. Create detailed implementation plan using `/superpowers:write-plan`
2. Execute implementation using `/superpowers:execute-plan`
3. Run paper trading validation
4. Deploy to production

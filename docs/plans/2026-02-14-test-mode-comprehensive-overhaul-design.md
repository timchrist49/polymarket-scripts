# Test Mode Comprehensive Overhaul - Design Document

**Date:** 2026-02-14
**Status:** Approved
**Scope:** Fix execution blockers, add multi-timeframe analysis, enhance metrics tracking

---

## Overview

The current test mode has a critical execution blocker: position sizing produces orders below Polymarket's $5 minimum, resulting in 0% execution rate (165 trades attempted, 0 filled). Additionally, the AI lacks multi-timeframe context for better decision-making.

This overhaul focuses on:
1. **Execution Fixes**: Enforce minimum bet size, filter low-edge trades
2. **Multi-Timeframe Analysis**: Add 15m/1H/4H trend context to improve decisions
3. **Metrics Tracking**: Monitor performance with Telegram reports every 20 trades
4. **AI Enhancement**: Provide timeframe context without artificial constraints

**Philosophy:** Trust the bot's directional judgment. Don't force 50/50 YES/NO balance or cap confidence. Instead, give it better information (multi-timeframe trends) and let it make informed calls.

---

## Architecture Overview

### 1. Enhanced TestModeConfig

**Core Configuration with Execution Guarantees:**

```python
@dataclass
class TestModeConfig:
    enabled: bool = False
    max_bet_amount: Decimal = Decimal("10.0")  # Allow Kelly sizing room
    min_bet_amount: Decimal = Decimal("5.0")   # Enforce Polymarket minimum
    min_arbitrage_edge: float = 0.02           # Require 2% edge minimum
    min_confidence: float = 0.70
    traded_markets: set[str] = field(default_factory=set)
```

**Position Sizing Formula:**
```python
final_size = max(kelly_size, min_bet_amount)
```

This ensures all orders meet Polymarket's $5 minimum while allowing Kelly criterion to reduce position when appropriate (up to the floor).

**Edge Filtering:**
- Only execute trades with `arbitrage_edge >= 2%`
- Filters out noise (0.1-0.3% edge trades observed in logs)
- Focuses on meaningful mispricings

---

### 2. Multi-Timeframe Analyzer

**New Service:** `polymarket/trading/timeframe_analyzer.py`

**Core Data Structures:**

```python
@dataclass
class TimeframeTrend:
    """Represents trend direction for a single timeframe."""
    timeframe: str           # "15m", "1h", "4h"
    direction: str           # "UP", "DOWN", "NEUTRAL"
    strength: float          # 0.0 to 1.0 (how strong the trend)
    price_change_pct: float  # Actual percentage change

@dataclass
class TimeframeAnalysis:
    """Complete multi-timeframe analysis result."""
    tf_15m: TimeframeTrend
    tf_1h: TimeframeTrend
    tf_4h: TimeframeTrend
    alignment_score: str     # "ALIGNED_BULLISH", "ALIGNED_BEARISH", "MIXED", "CONFLICTING"
    confidence_modifier: float  # +0.15, 0.0, or -0.15
```

**Trend Calculation Logic:**

- **15-minute trend**: BTC price change over last 15 minutes
- **1-hour trend**: BTC price change over last 60 minutes
- **4-hour trend**: BTC price change over last 240 minutes

**Direction Thresholds:**
- UP if price change > +0.5%
- DOWN if price change < -0.5%
- NEUTRAL otherwise

**Alignment Scoring:**

| Scenario | Alignment Score | Confidence Modifier |
|----------|----------------|---------------------|
| All 3 trends same direction (all UP or all DOWN) | ALIGNED_BULLISH / ALIGNED_BEARISH | +15% |
| 2 of 3 agree | MIXED | 0% |
| 15m contradicts both 1H and 4H | CONFLICTING | -15% |

**Data Source:**
- Uses existing `PriceHistoryBuffer` for historical price lookback
- No additional API calls needed

**Example Output:**
```
15m: UP (+0.8%), 1H: UP (+1.2%), 4H: UP (+2.5%)
Alignment: ALIGNED_BULLISH
Confidence Modifier: +15%
```

---

### 3. Metrics Tracker & Telegram Reporting

**Enhanced Database Schema:**

Add columns to `trades` table:
```sql
ALTER TABLE trades ADD COLUMN timeframe_15m_direction TEXT;
ALTER TABLE trades ADD COLUMN timeframe_1h_direction TEXT;
ALTER TABLE trades ADD COLUMN timeframe_4h_direction TEXT;
ALTER TABLE trades ADD COLUMN timeframe_alignment TEXT;
ALTER TABLE trades ADD COLUMN confidence_modifier REAL;
```

**Metrics Aggregation** (calculated every 20 trades):

```python
@dataclass
class TestModeMetrics:
    total_trades: int
    executed_trades: int           # Actually filled
    execution_rate: float          # executed / total
    wins: int
    losses: int
    win_rate: float
    total_pnl: Decimal
    avg_arbitrage_edge: float
    avg_confidence: float
    timeframe_alignment_stats: dict  # How often aligned vs mixed vs conflicting
```

**Telegram Report Format** (sent every 20 trades):

```
🎯 TEST MODE REPORT (Trades 21-40)

📊 Performance:
• Win Rate: 12/20 (60%)
• Total P&L: +$23.45
• Execution Rate: 18/20 (90%)

📈 Trade Quality:
• Avg Arbitrage Edge: 6.2%
• Avg Confidence: 78%

🕐 Timeframe Analysis:
• Aligned: 14 trades (70%)
• Mixed: 4 trades (20%)
• Conflicting: 2 trades (10%)

Next report after 20 more trades.
```

**What We DON'T Track:**
- ❌ YES/NO bias alerts (bot decides direction naturally)
- ❌ Confidence calibration warnings (trust AI judgment)
- ❌ Forced balance metrics (90% NO is fine if trend is bearish)

**Focus:** Performance outcomes only - execution rate, win rate, P&L, edge quality.

---

### 4. AI Integration & Enhanced Prompts

**AI Prompt Enhancement:**

Add new section to AI decision service prompt:

```
TIMEFRAME CONTEXT:
- 15-min trend: {tf_15m.direction} ({tf_15m.price_change_pct:+.2f}%)
- 1-hour trend: {tf_1h.direction} ({tf_1h.price_change_pct:+.2f}%)
- 4-hour trend: {tf_4h.direction} ({tf_4h.price_change_pct:+.2f}%)
- Alignment: {alignment_score}

Consider multi-timeframe alignment when forming conviction:
- If all timeframes aligned in same direction: Strong directional signal
- If timeframes mixed: Exercise caution, look for strong arbitrage edge
- If 15m contradicts longer timeframes: Short-term move against trend (mean reversion risk)

Your confidence will be automatically adjusted based on alignment:
- Aligned timeframes: +15% confidence boost
- Mixed signals: No adjustment
- Conflicting signals: -15% confidence reduction
```

**Key Principle:**
- The AI receives timeframe context as additional information
- No artificial constraints on confidence or direction
- The bot can still be 95% confident or 90% NO if signals support it
- Timeframe alignment modifies final confidence but doesn't override AI judgment

---

## Complete Trade Flow

**Step-by-Step Execution:**

1. **Signal Gathering**: Bot collects existing signals (social sentiment, technical indicators, arbitrage detector)

2. **Timeframe Analysis**: New `TimeframeAnalyzer` runs:
   ```python
   tf_analysis = await timeframe_analyzer.analyze()
   # Returns: TimeframeAnalysis with 15m/1H/4H trends + alignment + modifier
   ```

3. **AI Decision**: AI considers all signals + timeframe context:
   ```python
   decision = await ai_service.make_decision(
       signals=all_signals,
       timeframe_analysis=tf_analysis,
       market=market
   )
   # Returns: TradingDecision with base_confidence
   ```

4. **Confidence Adjustment**: Apply timeframe modifier:
   ```python
   final_confidence = min(decision.confidence + tf_analysis.confidence_modifier, 1.0)
   ```

5. **Edge Validation**: Check minimum edge requirement:
   ```python
   if decision.arbitrage_edge < self.test_mode.min_arbitrage_edge:
       logger.info("[TEST] Skipping trade - edge below 2% minimum")
       return
   ```

6. **Position Sizing**: Kelly calculation with floor enforcement:
   ```python
   kelly_size = calculate_kelly_size(odds, confidence, max_bet)
   final_size = max(kelly_size, self.test_mode.min_bet_amount)
   logger.info(
       "[TEST] Position sizing",
       kelly=f"${kelly_size:.2f}",
       final=f"${final_size:.2f}",
       enforced_minimum=kelly_size < self.test_mode.min_bet_amount
   )
   ```

7. **Order Execution**: Place order with guaranteed minimum size:
   ```python
   result = await smart_order_executor.execute(
       token_id=decision.token_id,
       side=decision.action,
       amount=final_size,
       target_price=execution_price
   )
   ```

8. **Metrics Logging**: Store trade with complete timeframe data:
   ```python
   await performance_tracker.log_decision(
       decision=decision,
       timeframe_15m=tf_analysis.tf_15m.direction,
       timeframe_1h=tf_analysis.tf_1h.direction,
       timeframe_4h=tf_analysis.tf_4h.direction,
       timeframe_alignment=tf_analysis.alignment_score,
       confidence_modifier=tf_analysis.confidence_modifier
   )
   ```

9. **Periodic Reporting**: Every 20 trades:
   ```python
   if self.total_trades % 20 == 0:
       metrics = await calculate_test_mode_metrics(last_20_trades)
       await telegram_bot.send_test_mode_report(metrics)
   ```

---

## Implementation Phases

### Phase 1: Core Execution Fixes (Priority 0)
**Goal:** Unblock trading - orders actually execute

**Changes:**
- Add `min_bet_amount`, `min_arbitrage_edge` to `TestModeConfig`
- Modify position sizing to enforce minimum
- Add edge validation before order placement
- Update initialization to use new config

**Expected Outcome:** Execution rate jumps from 0% to >80%

---

### Phase 2: Multi-Timeframe Analysis (Priority 1)
**Goal:** Better decision context

**New Files:**
- `polymarket/trading/timeframe_analyzer.py` - Analyzer service
- `tests/test_timeframe_analyzer.py` - Unit tests

**Changes to Existing:**
- Integrate analyzer into `auto_trade.py` trading cycle
- Pass timeframe analysis to AI decision service
- Apply confidence modifiers

**Expected Outcome:** AI has 15m/1H/4H context for decisions

---

### Phase 3: Metrics & Reporting (Priority 2)
**Goal:** Visibility into test mode performance

**Changes:**
- Database schema updates (add timeframe columns)
- Metrics aggregation logic
- Telegram report formatting
- Trigger reports every 20 trades

**Expected Outcome:** Actionable performance summaries every 20 trades

---

### Phase 4: AI Prompt Enhancement (Priority 3)
**Goal:** Teach AI to use timeframe context

**Changes:**
- Update AI decision service prompt
- Add timeframe interpretation guidance
- No constraints on confidence or direction

**Expected Outcome:** AI incorporates multi-timeframe alignment in reasoning

---

## Success Criteria

After implementation, test mode should achieve:

### Execution Metrics:
- ✅ Execution rate > 80% (orders actually fill)
- ✅ Zero "Size lower than minimum" errors

### Decision Quality:
- ✅ Average arbitrage edge > 3% (filter working)
- ✅ Trades only taken with edge >= 2%

### Timeframe Integration:
- ✅ All trades logged with 15m/1H/4H trend data
- ✅ Confidence modifiers applied based on alignment

### Reporting:
- ✅ Telegram reports sent every 20 trades
- ✅ Metrics include: win rate, P&L, execution rate, edge distribution, alignment stats

### Philosophy:
- ✅ Bot can be 90% NO if trend is bearish (no forced balance)
- ✅ Bot can be 95% confident if signals align (no calibration caps)
- ✅ Focus on outcomes (win rate, P&L), not process (bias alerts)

---

## Files to Modify

### New Files:
1. `polymarket/trading/timeframe_analyzer.py` - Multi-timeframe analysis service
2. `tests/test_timeframe_analyzer.py` - Unit tests for analyzer
3. `docs/plans/2026-02-14-test-mode-comprehensive-overhaul.md` - Implementation plan

### Modified Files:
1. `scripts/auto_trade.py`:
   - Update `TestModeConfig` dataclass
   - Add timeframe analyzer integration
   - Modify position sizing logic
   - Add edge validation
   - Integrate metrics reporting

2. `polymarket/trading/ai_decision.py`:
   - Add timeframe_analysis parameter
   - Enhance prompt with timeframe context
   - Apply confidence modifiers

3. `polymarket/performance/tracker.py`:
   - Add timeframe fields to decision logging
   - Add test mode metrics aggregation
   - Add Telegram report formatting

4. `polymarket/performance/database.py`:
   - Database schema migration for timeframe columns

5. `polymarket/telegram/bot.py`:
   - Add test mode report sending method

---

## Risk Mitigation

**Risk 1: Timeframe data unavailable (buffer empty)**
- Mitigation: Gracefully degrade - if no historical data, skip timeframe analysis
- Log warning, proceed with base AI confidence (no modifier)

**Risk 2: Confidence modifier pushes confidence >100%**
- Mitigation: Cap at 100%: `final_confidence = min(base + modifier, 1.0)`

**Risk 3: Min bet enforcement reduces expected value**
- Mitigation: Acceptable tradeoff - can't trade below Polymarket minimum anyway
- Kelly sizing still optimizes within constraints

**Risk 4: 2% edge filter too restrictive**
- Mitigation: User chose this threshold deliberately
- Can be adjusted if needed after observing results

---

## Testing Strategy

### Unit Tests:
- `TimeframeAnalyzer`: Test trend calculation, alignment scoring, confidence modifiers
- Position sizing: Verify min/max enforcement
- Edge validation: Verify filtering logic
- Metrics aggregation: Verify calculations

### Integration Tests:
- Full trade flow with timeframe analysis
- Database logging of timeframe data
- Telegram report generation

### Manual Testing:
- Run bot in test mode for 20 trades
- Verify execution rate >80%
- Confirm Telegram report received
- Validate metrics accuracy

---

## Rollout Plan

1. **Implement Phase 1** (Execution Fixes) → Deploy → Verify execution rate
2. **Implement Phase 2** (Timeframe Analysis) → Deploy → Monitor for 20 trades
3. **Implement Phase 3** (Metrics) → Deploy → Verify Telegram reports
4. **Implement Phase 4** (AI Enhancement) → Deploy → Final validation

**Rollback Strategy:** Each phase is independently deployable. If issues occur, roll back to previous phase.

---

## Conclusion

This comprehensive overhaul transforms test mode from "unusable" (0% execution) to "production-grade testing environment." The bot will:
- Actually execute trades (min bet enforcement)
- Make smarter decisions (multi-timeframe context)
- Provide actionable feedback (Telegram reports)
- Respect its own judgment (no artificial constraints)

**Expected Timeline:** 6-8 hours implementation + testing across 4 phases.
